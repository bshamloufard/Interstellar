#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets>=3.0"]
# ///
"""Normalize agent traces into the canonical trace.json schema.

Deterministic. No model in the loop — same input always yields byte-identical
output, which is what makes it safe to diff and cheap to re-run.

    uv run normalize.py --row 0 --out ../traces/row0.json
    uv run normalize.py --limit 50 --out-dir ../traces/
    uv run normalize.py --harness claude_code --benchmark swebench --limit 20 --out-dir ../traces/

Why precompute metrics here instead of letting the analyzer do it: counting 955
tool definitions and diffing them against 258 call sites is arithmetic over a
long list, which is the single thing LLMs are least reliable at and most
expensive at. The model should be spending its tokens on judgement — "this
skill is dead weight" — not on tallying. Every number in `metrics` is computed
here, exactly, and the prompt is told to cite rather than recompute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from schema import (  # noqa: E402
    DUR_DERIVED,
    DUR_DERIVED_SHARED,
    DUR_MEASURED,
    DUR_UNKNOWN,
    KIND_LLM,
    KIND_MCP,
    KIND_TOOL,
    make_span,
    make_trace,
)

DATASET = "Exgentic/agent-llm-traces"

# Rough bytes-per-token. Only used for the context-tax estimate, and every
# field derived from it is suffixed `_est` so nothing downstream mistakes it
# for a measurement.
BYTES_PER_TOKEN = 4


# --------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------

def _ts(value):
    """Parse an ISO timestamp, forcing tz-aware UTC. The dataset mixes naive
    and aware stamps; subtracting the two raises, so normalize on the way in."""
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _json(value, default):
    """Attributes are JSON-in-a-string and are sometimes null outright
    (1305 null tool.definitions and 634 null output.messages in the first 400
    rows), so every read goes through here."""
    if not value:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _parts(messages):
    for msg in messages or []:
        for part in msg.get("parts") or []:
            yield part


def classify_tool(name):
    """Split a tool name into (kind, server, bare_name).

    MCP tools arrive as `mcp__<server>__<path...>`; in this dataset that's
    `mcp__environment__spotify__show_song`, so the server is `environment` and
    everything after is the tool path.
    """
    if name and name.startswith("mcp__"):
        bits = name.split("__")
        if len(bits) >= 3:
            return KIND_MCP, bits[1], "__".join(bits[2:])
        return KIND_MCP, "unknown", name
    return KIND_TOOL, None, name


def _args_hash(args):
    """Stable identity for a call's arguments, so exact-duplicate work is
    detectable without storing every payload."""
    blob = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12], len(blob)


def _preview(value, limit):
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:limit], len(text)


# --------------------------------------------------------------------------
# core normalization
# --------------------------------------------------------------------------

def normalize_row(row, *, payload_chars=400):
    """One dataset row (= one session) -> one trace dict."""
    warnings = []
    spans_in = row.get("spans") or []
    if not spans_in:
        warnings.append("row has no spans")
        return None

    # --- session envelope -------------------------------------------------
    starts = [_ts(s.get("start_time")) for s in spans_in]
    ends = [_ts(s.get("end_time")) for s in spans_in]
    starts = [s for s in starts if s]
    ends = [e for e in ends if e]
    if not starts:
        warnings.append("no parseable timestamps")
        return None
    t0, t1 = min(starts), max(ends or starts)
    rel = lambda dt: int((dt - t0).total_seconds() * 1000)  # noqa: E731

    # --- tool definitions (the context tax) -------------------------------
    # Definitions are resent on every call; take the richest one seen and note
    # if they ever change mid-session.
    definitions, def_sizes, def_shapes = {}, {}, set()
    for s in spans_in:
        defs = _json((s.get("attributes") or {}).get("gen_ai.tool.definitions"), [])
        if defs:
            def_shapes.add(len(defs))
            for d in defs:
                name = d.get("name")
                if name and name not in definitions:
                    definitions[name] = d
                    def_sizes[name] = len(json.dumps(d, default=str))
    if len(def_shapes) > 1:
        warnings.append(
            f"tool definition count varied mid-session: {sorted(def_shapes)}"
        )
    if not definitions:
        warnings.append("no tool definitions present; inventory findings unavailable")

    # --- pass 1: collect every tool result, keyed by call id --------------
    # A result lands in a *later* span's input.messages, so this cannot be
    # done in a single forward pass over calls.
    results = {}
    for s in spans_in:
        attrs = s.get("attributes") or {}
        for part in _parts(_json(attrs.get("gen_ai.input.messages"), [])):
            if part.get("type") == "tool_call_response":
                cid = part.get("id")
                if cid and cid not in results:
                    results[cid] = part.get("result")

    # --- pass 2: build spans ---------------------------------------------
    spans, call_rows = [], []
    total_in = total_out = 0
    negative_gaps = 0

    for i, s in enumerate(spans_in):
        attrs = s.get("attributes") or {}
        s_start, s_end = _ts(s.get("start_time")), _ts(s.get("end_time"))
        if not s_start:
            continue
        llm_id = s.get("span_id") or f"llm{i}"
        tok_in = int(attrs.get("gen_ai.usage.input_tokens") or 0)
        tok_out = int(attrs.get("gen_ai.usage.output_tokens") or 0)
        total_in += tok_in
        total_out += tok_out
        finish = attrs.get("gen_ai.response.finish_reasons") or []
        if isinstance(finish, str):
            finish = _json(finish, [finish])

        spans.append(
            make_span(
                llm_id,
                KIND_LLM,
                f"llm {attrs.get('gen_ai.request.model', '?')}",
                parent_id=None,
                t_start_ms=rel(s_start),
                duration_ms=int(((s_end or s_start) - s_start).total_seconds() * 1000),
                duration_source=DUR_MEASURED,  # the model call IS timed
                status="error" if s.get("status") not in (None, "", "ok", "OK") else "ok",
                tokens={"input": tok_in, "output": tok_out},
                attrs={
                    "model": attrs.get("gen_ai.response.model"),
                    "finish_reasons": finish,
                    "turn_index": i,
                },
            )
        )

        calls = [
            p for p in _parts(_json(attrs.get("gen_ai.output.messages"), []))
            if p.get("type") == "tool_call"
        ]
        if not calls:
            continue

        # Tool execution occupies the gap between this call ending and the
        # next one starting. With one call the attribution is unambiguous;
        # with several we cannot tell how the gap divided, so say so.
        gap_ms, source = 0, DUR_UNKNOWN
        if i + 1 < len(spans_in) and s_end:
            nxt = _ts(spans_in[i + 1].get("start_time"))
            if nxt:
                delta = (nxt - s_end).total_seconds() * 1000
                if delta < 0:
                    negative_gaps += 1
                else:
                    gap_ms = int(delta)
                    source = DUR_DERIVED if len(calls) == 1 else DUR_DERIVED_SHARED

        for j, call in enumerate(calls):
            name = call.get("name") or "?"
            kind, server, bare = classify_tool(name)
            cid = call.get("id") or f"{llm_id}-c{j}"
            ahash, abytes = _args_hash(call.get("arguments"))
            res = results.get(cid)
            rprev, rbytes = _preview(res, payload_chars) if res is not None else ("", 0)
            aprev, _ = _preview(call.get("arguments"), payload_chars)

            spans.append(
                make_span(
                    cid,
                    kind,
                    bare,
                    parent_id=llm_id,
                    t_start_ms=rel(s_end or s_start),
                    duration_ms=gap_ms // len(calls) if source == DUR_DERIVED_SHARED else gap_ms,
                    duration_source=source,
                    status="ok" if res is not None else "no_result",
                    attrs={
                        "full_name": name,
                        "server": server,
                        "args_hash": ahash,
                        "args_bytes": abytes,
                        "args_preview": aprev,
                        "result_bytes": rbytes,
                        "result_preview": rprev,
                        "concurrent_calls": len(calls),
                    },
                )
            )
            call_rows.append(
                {"name": name, "kind": kind, "server": server, "hash": ahash,
                 "turn": i, "result_bytes": rbytes, "gap_ms": gap_ms}
            )

    if negative_gaps:
        warnings.append(
            f"{negative_gaps} negative inter-span gaps (overlapping spans); "
            "those tool durations are unknown"
        )
    if any(s["duration_source"] in (DUR_DERIVED, DUR_DERIVED_SHARED) for s in spans):
        warnings.append(
            "tool durations are DERIVED from gaps between model calls, not measured. "
            "They include harness overhead and rate-limit backoff. Do not raise "
            "latency findings from them."
        )

    session = {
        "session_id": row.get("session_id"),
        "source": f"hf:{DATASET}",
        "harness": row.get("harness"),
        "benchmark": row.get("benchmark"),
        "models": row.get("models") or [],
        "started_at": t0.isoformat(),
        "duration_ms": int((t1 - t0).total_seconds() * 1000),
        "collected_at": row.get("collected_at"),
    }

    inventory = build_inventory(definitions, def_sizes, call_rows)
    metrics = build_metrics(spans, call_rows, inventory, row)
    return make_trace(session, spans, inventory, metrics, warnings)


# --------------------------------------------------------------------------
# inventory + metrics — every number the analyzer is told to cite
# --------------------------------------------------------------------------

def build_inventory(definitions, def_sizes, call_rows):
    counts = Counter(c["name"] for c in call_rows)
    tools = []
    for name, d in sorted(definitions.items()):
        kind, server, bare = classify_tool(name)
        nbytes = def_sizes.get(name, 0)
        tools.append({
            "name": name,
            "bare_name": bare,
            "kind": kind,
            "server": server,
            "definition_bytes": nbytes,
            "definition_tokens_est": nbytes // BYTES_PER_TOKEN,
            "call_count": counts.get(name, 0),
            "description_chars": len(d.get("description") or ""),
        })

    # Tools called but never declared: real in this dataset, because some
    # harnesses strip the mcp__environment__ prefix at the call site.
    undeclared = sorted(set(counts) - set(definitions))

    by_server = defaultdict(lambda: {"tools": 0, "calls": 0, "definition_bytes": 0})
    for t in tools:
        if t["kind"] == KIND_MCP:
            e = by_server[t["server"]]
            e["tools"] += 1
            e["calls"] += t["call_count"]
            e["definition_bytes"] += t["definition_bytes"]

    total_bytes = sum(t["definition_bytes"] for t in tools)
    unused = [t for t in tools if t["call_count"] == 0]
    return {
        "tools": tools,
        "undeclared_but_called": undeclared,
        "mcp_servers": dict(by_server),
        "totals": {
            "defined": len(tools),
            "called": len([t for t in tools if t["call_count"]]),
            "never_called": len(unused),
            "definition_bytes": total_bytes,
            "definition_tokens_est": total_bytes // BYTES_PER_TOKEN,
            "wasted_definition_bytes": sum(t["definition_bytes"] for t in unused),
            "wasted_definition_tokens_est":
                sum(t["definition_bytes"] for t in unused) // BYTES_PER_TOKEN,
        },
    }


def build_metrics(spans, call_rows, inventory, row):
    llm = [s for s in spans if s["kind"] == KIND_LLM]
    tool_spans = [s for s in spans if s["kind"] in (KIND_TOOL, KIND_MCP)]

    # Exact repeats: same tool, byte-identical arguments, more than once.
    # Strong signal for a redundant call the agent should have cached.
    seen = Counter((c["name"], c["hash"]) for c in call_rows)
    repeats = sorted(
        ({"tool": n, "args_hash": h, "times": k} for (n, h), k in seen.items() if k > 1),
        key=lambda r: -r["times"],
    )

    # Back-to-back identical tool names: thrash / retry loops.
    runs, run = [], None
    for c in call_rows:
        if run and run["tool"] == c["name"]:
            run["length"] += 1
        else:
            if run and run["length"] >= 3:
                runs.append(run)
            run = {"tool": c["name"], "start_turn": c["turn"], "length": 1}
    if run and run["length"] >= 3:
        runs.append(run)

    # Context growth: input tokens per model call. A steep monotone climb is
    # the fingerprint of an unpruned transcript.
    ctx = [s["tokens"].get("input", 0) for s in llm]

    # Biggest tool outputs — the truncation candidates.
    fat = sorted(
        ({"tool": s["name"], "result_bytes": s["attrs"]["result_bytes"], "span": s["id"]}
         for s in tool_spans if s["attrs"].get("result_bytes")),
        key=lambda r: -r["result_bytes"],
    )[:15]

    per_tool_bytes = defaultdict(int)
    for s in tool_spans:
        per_tool_bytes[s["name"]] += s["attrs"].get("result_bytes", 0)

    return {
        "counts": {
            "llm_calls": len(llm),
            "tool_calls": len(tool_spans),
            "mcp_calls": len([s for s in tool_spans if s["kind"] == KIND_MCP]),
            "distinct_tools_used": len({c["name"] for c in call_rows}),
            # Zero in this dataset — recorded so the gap is visible, not silent.
            "skill_invocations": len([s for s in spans if s["kind"] == "skill"]),
            "subagent_spawns": len([s for s in spans if s["kind"] == "subagent"]),
        },
        "tokens": {
            "input_total": sum(s["tokens"].get("input", 0) for s in llm),
            "output_total": sum(s["tokens"].get("output", 0) for s in llm),
            "reported_total": row.get("total_tokens"),
            "peak_context": max(ctx) if ctx else 0,
            "context_series": ctx,
        },
        "tool_frequency": Counter(c["name"] for c in call_rows).most_common(20),
        "duplicate_calls": repeats[:20],
        "duplicate_call_total": sum(r["times"] - 1 for r in repeats),
        "repeat_runs": runs,
        "largest_results": fat,
        "result_bytes_by_tool":
            sorted(per_tool_bytes.items(), key=lambda kv: -kv[1])[:15],
        "never_called_tools":
            [t["name"] for t in inventory["tools"] if t["call_count"] == 0],
        # Present but explicitly untrusted; the prompt refuses latency findings
        # unless duration_source is `measured`.
        "derived_gap_ms": {
            "note": "DERIVED, not measured — includes backoff/overhead",
            "p50": _pct([c["gap_ms"] for c in call_rows], 50),
            "p95": _pct([c["gap_ms"] for c in call_rows], 95),
        },
    }


def _pct(values, p):
    vals = sorted(v for v in values if v)
    if not vals:
        return 0
    return vals[min(int(len(vals) * p / 100), len(vals) - 1)]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--row", type=int, help="normalize a single row index")
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--harness", help="filter, e.g. claude_code")
    ap.add_argument("--benchmark", help="filter, e.g. swebench")
    ap.add_argument("--out", type=Path, help="single output file (with --row)")
    ap.add_argument("--out-dir", type=Path, help="write one file per session")
    ap.add_argument("--payload-chars", type=int, default=400,
                    help="chars of args/result preview to keep (default 400)")
    ap.add_argument("--summary", action="store_true",
                    help="print a human summary instead of JSON")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.dataset)["train"]

    if args.harness:
        ds = ds.filter(lambda r: r["harness"] == args.harness)
    if args.benchmark:
        ds = ds.filter(lambda r: r["benchmark"] == args.benchmark)

    idxs = [args.row] if args.row is not None else range(min(args.limit, len(ds)))
    written = 0
    for i in idxs:
        trace = normalize_row(ds[int(i)], payload_chars=args.payload_chars)
        if trace is None:
            print(f"row {i}: skipped (empty)", file=sys.stderr)
            continue
        blob = json.dumps(trace, indent=2, default=str)

        if args.summary:
            print_summary(trace)
        elif args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            sid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(trace["session"]["session_id"]))
            path = args.out_dir / f"{sid}.json"
            path.write_text(blob)
            print(f"wrote {path}  ({len(blob):,} bytes)", file=sys.stderr)
        elif args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(blob)
            print(f"wrote {args.out}  ({len(blob):,} bytes)", file=sys.stderr)
        else:
            print(blob)
        written += 1
    print(f"normalized {written} session(s)", file=sys.stderr)


def print_summary(t):
    s, m, inv = t["session"], t["metrics"], t["inventory"]
    tot = inv["totals"]
    print(f"session   {s['session_id']}  [{s['harness']} / {s['benchmark']}]")
    print(f"model     {', '.join(s['models'])}")
    print(f"duration  {s['duration_ms'] / 1000:.1f}s   spans {len(t['spans'])}")
    print(f"calls     {m['counts']['llm_calls']} llm, {m['counts']['tool_calls']} tool "
          f"({m['counts']['mcp_calls']} mcp)")
    print(f"tokens    in {m['tokens']['input_total']:,} / out "
          f"{m['tokens']['output_total']:,} / peak ctx {m['tokens']['peak_context']:,}")
    print()
    print(f"TOOL INVENTORY  {tot['defined']} defined, {tot['called']} called, "
          f"{tot['never_called']} never called")
    print(f"  definitions cost ~{tot['definition_tokens_est']:,} tok/call; "
          f"~{tot['wasted_definition_tokens_est']:,} of that is never-called tools")
    if m["duplicate_call_total"]:
        print(f"REDUNDANT       {m['duplicate_call_total']} duplicate calls "
              f"(identical tool + args)")
        for r in m["duplicate_calls"][:3]:
            print(f"  {r['times']}x  {r['tool']}")
    if m["repeat_runs"]:
        print(f"THRASH          {len(m['repeat_runs'])} run(s) of >=3 same-tool calls")
        for r in m["repeat_runs"][:3]:
            print(f"  {r['length']}x  {r['tool']}  from turn {r['start_turn']}")
    if m["largest_results"]:
        print("BIGGEST RESULTS (truncation candidates)")
        for r in m["largest_results"][:3]:
            print(f"  {r['result_bytes']:>8,}B  {r['tool']}")
    print()
    print(f"COVERAGE        skills {m['counts']['skill_invocations']}, "
          f"subagents {m['counts']['subagent_spawns']}")
    for w in t["warnings"]:
        print(f"  ! {w}")


if __name__ == "__main__":
    main()
