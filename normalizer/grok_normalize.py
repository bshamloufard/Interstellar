#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Normalize a real grok-build session directory into the canonical trace.json.

This is the mechanism the HF dataset could not provide. grok already records
MEASURED timings in `events.jsonl` — `tool_completed.duration_ms`,
`permission_resolved.wait_ms`, `mcp_server_connected.duration_ms` — so unlike
the OTel path nothing here is inferred from gaps.

Three span kinds are DERIVED rather than read, because grok has no native
event for them (verified against a live build: the registered tool list
contains no `skill` tool at all):

  skill     a `read_file` whose path matches  .../skills/<name>/SKILL.md
  subagent  a `spawn_subagent` tool call
  mcp       a `use_tool` call, whose `tool_name` argument names the server

Deriving them is not a workaround for the analyzer's benefit — it is how a
skill load is actually observable in this harness. The derivation is recorded
per span in `attrs.derived_from` so nothing pretends to be a native event.

The section-utilization pass is what powers line-level skill recommendations:
for every `## ` section of a loaded SKILL.md we measure whether its distinctive
vocabulary shows up anywhere downstream. A section that entered the context and
was never echoed is a truncation candidate, cited by line range.

    uv run grok_normalize.py <session_dir> --summary
    uv run grok_normalize.py --all --out-dir ../traces/grok/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from schema import (  # noqa: E402
    DUR_MEASURED,
    DUR_UNKNOWN,
    KIND_LLM,
    KIND_MCP,
    KIND_SKILL,
    KIND_SUBAGENT,
    KIND_TOOL,
    make_span,
    make_trace,
)

SESSIONS_ROOT = Path.home() / ".grok" / "sessions"
BYTES_PER_TOKEN = 4
SKILL_PATH = re.compile(r"/skills/([^/]+)/SKILL\.md$")

# Words too common to prove a section was actually used.
STOP = set("""the a an and or of to in is are be for with on at by it this that as
from not you your if then when do does use used using should must can may will
each any all one two some other more most into out over under file files line
lines code do not""".split())


def _ts(v):
    return datetime.fromisoformat(v.replace("Z", "+00:00")) if v else None


def _read_jsonl(path):
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


# --------------------------------------------------------------------------
# skill section utilization
# --------------------------------------------------------------------------

def split_sections(body):
    """Split a SKILL.md into (title, start_line, end_line, text) by `## `."""
    lines = body.splitlines()
    marks = [i for i, l in enumerate(lines) if l.startswith("## ")]
    if not marks:
        return [("(whole body)", 1, len(lines), body)]
    out = []
    if marks[0] > 0:
        out.append(("(preamble)", 1, marks[0], "\n".join(lines[: marks[0]])))
    for k, start in enumerate(marks):
        end = marks[k + 1] if k + 1 < len(marks) else len(lines)
        out.append((lines[start][3:].strip(), start + 1, end,
                    "\n".join(lines[start:end])))
    return out


def _vocab(text):
    return {w for w in re.findall(r"[a-z][a-z0-9_-]{4,}", text.lower())
            if w not in STOP}


def utilization(body, downstream):
    """Which sections of a loaded skill actually show up in later output.

    Distinctive vocabulary only: a section's words minus words that appear
    elsewhere in the same skill, so shared boilerplate cannot make a dead
    section look used.
    """
    secs = split_sections(body)
    vocabs = [_vocab(t) for _, _, _, t in secs]
    later = _vocab(downstream)

    # A section repeated verbatim would cancel its own twin's distinctive
    # vocabulary and end up unjudgeable, so compare against DISTINCT bodies
    # only. The repetition is itself worth reporting, so record it.
    first_seen = {}
    dup_of = [None] * len(secs)
    for i, (_, _, _, text) in enumerate(secs):
        key = " ".join(text.split())
        if key in first_seen:
            dup_of[i] = first_seen[key]
        else:
            first_seen[key] = i

    rows = []
    for i, (title, s, e, text) in enumerate(secs):
        canon = dup_of[i] if dup_of[i] is not None else i
        others = set()
        for j, v in enumerate(vocabs):
            if (dup_of[j] if dup_of[j] is not None else j) != canon:
                others |= v
        distinctive = vocabs[i] - others
        hits = distinctive & later
        rows.append({
            "section": title,
            "start_line": s,
            "end_line": e,
            "lines": e - s + 1,
            "bytes": len(text),
            "tokens_est": len(text) // BYTES_PER_TOKEN,
            "distinctive_terms": len(distinctive),
            "terms_referenced": len(hits),
            # No distinctive vocabulary at all => cannot judge; don't claim unused.
            "referenced": bool(hits) if distinctive else None,
            "sample_hits": sorted(hits)[:5],
            # Verbatim repeat of an earlier section: always removable.
            "duplicate_of_section": secs[dup_of[i]][0] if dup_of[i] is not None else None,
        })
    return rows


# --------------------------------------------------------------------------
# main normalization
# --------------------------------------------------------------------------

def normalize_session(session_dir, *, payload_chars=400):
    d = Path(session_dir)
    events = _read_jsonl(d / "events.jsonl")
    history = _read_jsonl(d / "chat_history.jsonl")
    summary = {}
    if (d / "summary.json").is_file():
        try:
            summary = json.loads((d / "summary.json").read_text())
        except json.JSONDecodeError:
            pass
    if not events:
        return None

    warnings = []
    t0 = _ts(events[0].get("ts"))
    rel = lambda ts: int((_ts(ts) - t0).total_seconds() * 1000) if ts else 0  # noqa: E731

    # ---- index chat history by tool_call_id -----------------------------
    calls, results, assistant_text = {}, {}, []
    for msg in history:
        typ = msg.get("type")
        if typ == "assistant":
            if msg.get("content"):
                assistant_text.append(str(msg["content"]))
            for tc in msg.get("tool_calls") or []:
                args = tc.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                calls[tc.get("id")] = {"name": tc.get("name"), "args": args or {},
                                       "model": msg.get("model_id")}
        elif typ == "tool_result":
            results[msg.get("tool_call_id")] = str(msg.get("content") or "")
        elif typ == "reasoning" and msg.get("content"):
            assistant_text.append(str(msg["content"]))
    downstream_blob = "\n".join(assistant_text)

    # ---- walk events -----------------------------------------------------
    spans, warn_seen = [], set()
    started_at, perm_wait = {}, defaultdict(int)
    turn_id, turn_no = None, -1
    mcp_servers, tool_stats = {}, Counter()
    failed_mcp = []
    skills_loaded, subagents = [], []

    # ---- native skill events (upstream a959d1a) -------------------------
    # `skill_activated` is authoritative: it names the skill, how it was
    # triggered, and — for skill_md_read / skill_tool — the tool_call_id to
    # join on. Before this commit the only way to see a skill was to notice a
    # read_file of SKILL.md, which is still the fallback for older sessions.
    native_skills = [e for e in events if e.get("type") == "skill_activated"]
    native_by_call = {e["related_tool_call_id"]: e for e in native_skills
                      if e.get("related_tool_call_id")}
    if native_skills:
        warnings.append(
            f"{len(native_skills)} native skill_activated event(s); "
            "skill spans are read, not derived")

    for ev in events:
        typ, ts = ev.get("type"), ev.get("ts")

        if typ == "mcp_server_connected":
            name = ev["server_name"]
            mcp_servers[name] = {
                "transport": ev.get("transport"),
                "tool_count": ev.get("tool_count", 0),
                "connect_ms": ev.get("duration_ms", 0),
                "tools": ev.get("tools") or [],
                "calls": 0,
            }
            spans.append(make_span(
                f"mcpconn-{name}", KIND_MCP, f"connect {name}",
                t_start_ms=rel(ts), duration_ms=ev.get("duration_ms", 0),
                duration_source=DUR_MEASURED, status="ok",
                attrs={"event": "mcp_server_connected", "server": name,
                       "tool_count": ev.get("tool_count"),
                       "transport": ev.get("transport"),
                       "phase": "startup"},
            ))
        elif typ == "mcp_server_failed":
            failed_mcp.append({
                "server": ev.get("server_name"),
                "error_type": ev.get("error_type"),
                "error": str(ev.get("error_message"))[:200],
                "wasted_ms": ev.get("duration_ms") or 0,
            })
            spans.append(make_span(
                f"mcpfail-{ev.get('server_name')}", KIND_MCP,
                f"connect {ev.get('server_name')} FAILED",
                t_start_ms=rel(ts), duration_ms=ev.get("duration_ms") or 0,
                duration_source=DUR_MEASURED, status="error",
                attrs={"event": "mcp_server_failed",
                       "error_type": ev.get("error_type"),
                       "error": ev.get("error_message")},
            ))
        elif typ == "turn_started":
            turn_no += 1
            turn_id = f"turn-{turn_no}"
            spans.append(make_span(
                turn_id, KIND_LLM, f"turn {turn_no}", t_start_ms=rel(ts),
                duration_source=DUR_UNKNOWN,
                attrs={"model": ev.get("model_id"), "turn_number": ev.get("turn_number"),
                       "yolo": ev.get("yolo_mode"),
                       "messages_at_start": ev.get("conversation_message_count")},
            ))
        elif typ == "turn_ended" and turn_id:
            for s in spans:
                if s["id"] == turn_id:
                    s["duration_ms"] = rel(ts) - s["t_start_ms"]
                    s["duration_source"] = DUR_MEASURED
                    s["status"] = "ok" if ev.get("outcome") == "completed" else "error"
                    s["attrs"]["outcome"] = ev.get("outcome")
        elif typ == "tool_started":
            started_at.setdefault(ev["tool_name"], []).append(rel(ts))
        elif typ == "permission_resolved":
            perm_wait[ev["tool_name"]] += ev.get("wait_ms", 0)
        elif typ == "tool_completed":
            cid = ev.get("tool_call_id") or f"tool-{len(spans)}"
            name = ev["tool_name"]
            dur = ev.get("duration_ms", 0)
            queue = started_at.get(name) or []
            t_start = queue.pop(0) if queue else rel(ts) - dur
            call = calls.get(cid, {})
            args = call.get("args", {})
            result = results.get(cid, "")
            tool_stats[name] += 1

            kind, attrs = KIND_TOOL, {
                "tool": name,
                "outcome": ev.get("outcome"),
                "permission_wait_ms": perm_wait.pop(name, 0),
                "args_bytes": len(json.dumps(args, default=str)),
                "result_bytes": len(result),
                "result_tokens_est": len(result) // BYTES_PER_TOKEN,
                "args_preview": json.dumps(args, default=str)[:payload_chars],
                "result_preview": result[:payload_chars],
            }

            # --- derived: MCP dispatch ---------------------------------
            if name == "use_tool":
                qualified = str(args.get("tool_name", ""))
                server = qualified.split("__")[0] if "__" in qualified else qualified
                kind = KIND_MCP
                attrs.update({"derived_from": "use_tool call", "server": server,
                              "mcp_tool": qualified})
                if server in mcp_servers:
                    mcp_servers[server]["calls"] += 1

            # --- derived: subagent -------------------------------------
            elif name == "spawn_subagent":
                kind = KIND_SUBAGENT
                attrs.update({"derived_from": "spawn_subagent call",
                              "agent": args.get("agent") or args.get("subagent_type"),
                              "task": str(args.get("prompt") or args.get("task") or "")[:payload_chars]})
                subagents.append({"agent": attrs["agent"], "duration_ms": dur,
                                  "result_bytes": len(result)})

            # --- derived: skill load -----------------------------------
            elif name == "read_file" or cid in native_by_call:
                target = str(args.get("target_file") or args.get("path") or "")
                native = native_by_call.get(cid)
                m = SKILL_PATH.search(target)
                if native or m:
                    kind = KIND_SKILL
                    sname = native["skill_name"] if native else m.group(1)
                    body = re.sub(r"^\s*\d+→", "", result, flags=re.M)  # strip cat -n
                    # main records the injected body size directly as of the
                    # skill-cost patch. Trust it over the joined tool_result,
                    # which is absent entirely for slash_command activations.
                    native_bytes = (native or {}).get("skill_bytes")
                    native_lines = (native or {}).get("skill_lines")
                    secs = utilization(body, downstream_blob)
                    dead = [s for s in secs if s["referenced"] is False]
                    repeated = [s for s in secs if s["duplicate_of_section"]]
                    attrs.update({
                        "source": "native_event" if native else "derived",
                        "derived_from": None if native else "read_file of SKILL.md",
                        "trigger": (native or {}).get("trigger", "skill_md_read"),
                        "plugin_source": (native or {}).get("plugin_source"),
                        "skill": sname, "path": target,
                        "skill_bytes": native_bytes if native_bytes is not None else len(body),
                        "skill_tokens_est": (native_bytes if native_bytes is not None
                                             else len(body)) // BYTES_PER_TOKEN,
                        "skill_lines": (native_lines if native_lines is not None
                                        else body.count("\n") + 1),
                        "size_source": "native_event" if native_bytes is not None else "tool_result",
                        "sections": secs,
                        "unreferenced_sections": len(dead),
                        "unreferenced_tokens_est": sum(s["tokens_est"] for s in dead),
                        "unreferenced_line_ranges":
                            [[s["start_line"], s["end_line"]] for s in dead],
                        "repeated_sections": len(repeated),
                        "repeated_tokens_est": sum(s["tokens_est"] for s in repeated),
                        "repeated_line_ranges":
                            [[s["start_line"], s["end_line"]] for s in repeated],
                    })
                    skills_loaded.append({
                        "skill": sname,
                        "tokens_est": attrs["skill_tokens_est"],
                        "unreferenced_tokens_est": attrs["unreferenced_tokens_est"],
                        "sections": len(secs), "dead_sections": len(dead),
                        "repeated_sections": len(repeated),
                        "repeated_tokens_est": attrs["repeated_tokens_est"],
                    })

            spans.append(make_span(
                cid, kind, name, parent_id=turn_id, t_start_ms=t_start,
                duration_ms=dur, duration_source=DUR_MEASURED,
                status="ok" if ev.get("outcome") == "success" else "error",
                attrs=attrs,
            ))

    # ---- slash activations have no tool call, so the walk above never made
    # a span for them. Emit one from the native event. Size comes from the
    # event; the body (for section utilization) can only come from disk, which
    # may have changed since the run — recorded as `body_source`.
    seen_skills = {sp["attrs"].get("skill") for sp in spans if sp["kind"] == KIND_SKILL}
    for i, ev in enumerate(native_skills):
        if ev.get("related_tool_call_id") or ev.get("skill_name") in seen_skills:
            continue
        sname = ev.get("skill_name")
        disk = Path.home() / ".grok" / "skills" / str(sname) / "SKILL.md"
        secs, body_source = [], "unavailable"
        if disk.is_file():
            body = disk.read_text(errors="replace")
            secs = utilization(body, downstream_blob)
            body_source = "disk_at_analysis_time"
        dead = [x for x in secs if x["referenced"] is False]
        repeated = [x for x in secs if x["duplicate_of_section"]]
        nb, nl = ev.get("skill_bytes"), ev.get("skill_lines")
        skills_loaded.append({
            "skill": sname,
            "tokens_est": (nb or 0) // BYTES_PER_TOKEN,
            "unreferenced_tokens_est": sum(x["tokens_est"] for x in dead),
            "sections": len(secs), "dead_sections": len(dead),
            "repeated_sections": len(repeated),
            "repeated_tokens_est": sum(x["tokens_est"] for x in repeated),
        })
        spans.append(make_span(
            f"skill-slash-{i}", KIND_SKILL, f"/{sname}",
            parent_id=turn_id, t_start_ms=rel(ev.get("ts")),
            duration_ms=0, duration_source=DUR_UNKNOWN, status="ok",
            attrs={
                "source": "native_event", "derived_from": None,
                "trigger": ev.get("trigger"), "skill": sname,
                "plugin_source": ev.get("plugin_source"),
                "skill_bytes": nb, "skill_lines": nl,
                "skill_tokens_est": (nb or 0) // BYTES_PER_TOKEN,
                "size_source": "native_event" if nb is not None else "unmeasured",
                "body_source": body_source,
                "sections": secs,
                "unreferenced_sections": len(dead),
                "unreferenced_tokens_est": sum(x["tokens_est"] for x in dead),
                "unreferenced_line_ranges": [[x["start_line"], x["end_line"]] for x in dead],
                "repeated_sections": len(repeated),
                "repeated_tokens_est": sum(x["tokens_est"] for x in repeated),
                "repeated_line_ranges": [[x["start_line"], x["end_line"]] for x in repeated],
                "result_bytes": 0, "result_tokens_est": 0, "permission_wait_ms": 0,
            },
        ))
        if nb is None:
            warnings.append(
                f"slash activation of '{sname}' has no skill_bytes; "
                "binary predates the skill-cost patch")

    if not history:
        warnings.append("chat_history.jsonl missing; payload sizes unavailable")
    for k in warn_seen:
        warnings.append(k)

    info = summary.get("info") or {}
    session = {
        "session_id": info.get("id") or d.name,
        "source": "grok-build",
        "cwd": info.get("cwd"),
        "title": summary.get("generated_title") or summary.get("session_summary"),
        "model": summary.get("current_model_id"),
        "started_at": summary.get("created_at") or (t0.isoformat() if t0 else None),
        "duration_ms": rel(events[-1].get("ts")),
        "session_dir": str(d),
    }
    inventory = {
        "mcp_servers": mcp_servers,
        "skills_loaded": skills_loaded,
        "tools_used": dict(tool_stats),
        "totals": {
            "mcp_servers": len(mcp_servers),
            "mcp_tools_exposed": sum(v["tool_count"] for v in mcp_servers.values()),
            "mcp_connect_ms": sum(v["connect_ms"] for v in mcp_servers.values()),
            "skills_loaded": len(skills_loaded),
            "skill_tokens_est": sum(s["tokens_est"] for s in skills_loaded),
            "skill_wasted_tokens_est": sum(s["unreferenced_tokens_est"] for s in skills_loaded),
        },
    }
    metrics = build_metrics(spans, mcp_servers, skills_loaded, subagents, failed_mcp)
    return make_trace(session, spans, inventory, metrics, warnings)


def build_metrics(spans, mcp_servers, skills, subagents, failed_mcp=None):
    tools = [s for s in spans if s["kind"] in (KIND_TOOL, KIND_MCP, KIND_SKILL, KIND_SUBAGENT)
             and s["attrs"].get("event") is None]

    # exact-duplicate work: same tool, byte-identical arguments
    sig = Counter((s["name"], s["attrs"].get("args_preview")) for s in tools)
    dupes = [{"tool": n, "times": c, "args": (a or "")[:160]}
             for (n, a), c in sig.items() if c > 1]

    slow = sorted(tools, key=lambda s: -s["duration_ms"])[:10]
    fat = sorted(tools, key=lambda s: -s["attrs"].get("result_bytes", 0))[:10]

    # MCP servers that cost context but were never called
    idle_mcp = [{"server": k, "tools_exposed": v["tool_count"],
                 "connect_ms": v["connect_ms"]}
                for k, v in mcp_servers.items() if v["calls"] == 0]

    # A skill that under-specifies something shows up as discovery work
    # immediately after it loads: the agent reads the skill, then has to go
    # find out where/what before it can act. That is the observable signature
    # of a MISSING instruction, and the evidence for skill_add_lines.
    DISCOVERY = {"list_dir", "grep", "glob", "search_tool", "codebase_search"}
    ordered = sorted(tools, key=lambda s: s["t_start_ms"])
    post_skill = []
    for i, s in enumerate(ordered):
        if s["kind"] != KIND_SKILL:
            continue
        following = [x for x in ordered[i + 1:] if x["name"] in DISCOVERY]
        if following:
            post_skill.append({
                "skill": s["attrs"].get("skill"),
                "discovery_calls": [
                    {"tool": x["name"], "ms": x["duration_ms"],
                     "args": (x["attrs"].get("args_preview") or "")[:120]}
                    for x in following[:6]
                ],
                "count": len(following),
                "total_ms": sum(x["duration_ms"] for x in following),
            })

    return {
        "counts": {
            "turns": len([s for s in spans if s["kind"] == KIND_LLM]),
            "tool_calls": len(tools),
            "skill_loads": len(skills),
            "subagent_spawns": len(subagents),
            "mcp_calls": len([s for s in tools if s["kind"] == KIND_MCP]),
        },
        "timing": {
            "total_tool_ms": sum(s["duration_ms"] for s in tools),
            "total_permission_wait_ms":
                sum(s["attrs"].get("permission_wait_ms", 0) for s in tools),
            "mcp_startup_ms": sum(v["connect_ms"] for v in mcp_servers.values()),
            "slowest_mcp_connect": sorted(
                ({"server": k, "ms": v["connect_ms"], "tools": v["tool_count"],
                  "calls": v["calls"]} for k, v in mcp_servers.items()),
                key=lambda r: -r["ms"])[:6],
            "slowest": [{"tool": s["name"], "ms": s["duration_ms"],
                         "kind": s["kind"], "span": s["id"]} for s in slow if s["duration_ms"]],
        },
        "context_cost": {
            "tool_result_tokens_est":
                sum(s["attrs"].get("result_tokens_est", 0) for s in tools),
            "skill_tokens_est": sum(s["tokens_est"] for s in skills),
            "skill_wasted_tokens_est": sum(s["unreferenced_tokens_est"] for s in skills),
            "largest_results": [{"tool": s["name"], "bytes": s["attrs"].get("result_bytes", 0),
                                 "span": s["id"]} for s in fat
                                if s["attrs"].get("result_bytes")],
        },
        "duplicate_calls": sorted(dupes, key=lambda r: -r["times"]),
        "discovery_after_skill_load": post_skill,
        "idle_mcp_servers": idle_mcp,
        "failed_mcp_servers": failed_mcp or [],
        "skill_detail": skills,
        "subagents": subagents,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def discover():
    return sorted((p for p in SESSIONS_ROOT.glob("*/*")
                   if p.is_dir() and (p / "events.jsonl").is_file()),
                  key=lambda p: p.stat().st_mtime)


def print_summary(t):
    s, m, inv = t["session"], t["metrics"], t["inventory"]
    print(f"session  {s['session_id']}  [{s['model']}]")
    print(f"title    {s['title']}")
    print(f"wall     {s['duration_ms'] / 1000:.1f}s   spans {len(t['spans'])}")
    c = m["counts"]
    print(f"counts   {c['turns']} turns, {c['tool_calls']} tools, "
          f"{c['skill_loads']} skills, {c['subagent_spawns']} subagents, {c['mcp_calls']} mcp")
    tm = m["timing"]
    print(f"timing   tool {tm['total_tool_ms']}ms | perm wait "
          f"{tm['total_permission_wait_ms']}ms | mcp startup {tm['mcp_startup_ms']}ms")
    cc = m["context_cost"]
    print(f"context  tool results ~{cc['tool_result_tokens_est']:,} tok | "
          f"skills ~{cc['skill_tokens_est']:,} tok "
          f"({cc['skill_wasted_tokens_est']:,} unreferenced)")
    for sk in m["skill_detail"]:
        print(f"  SKILL {sk['skill']}: {sk['tokens_est']} tok, "
              f"{sk['dead_sections']}/{sk['sections']} sections unreferenced "
              f"({sk['unreferenced_tokens_est']} tok)")
    for d in m["duplicate_calls"][:3]:
        print(f"  DUPLICATE {d['times']}x {d['tool']}  {d['args'][:70]}")
    for i in m["idle_mcp_servers"]:
        print(f"  IDLE MCP {i['server']}: {i['tools_exposed']} tools exposed, "
              f"{i['connect_ms']}ms connect, 0 calls")
    for w in t["warnings"]:
        print(f"  ! {w}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()

    dirs = discover() if a.all else [Path(a.session_dir)]
    if not dirs:
        raise SystemExit("no grok sessions found")
    for d in dirs:
        t = normalize_session(d)
        if t is None:
            print(f"skip {d} (no events)", file=sys.stderr)
            continue
        if a.out_dir:
            a.out_dir.mkdir(parents=True, exist_ok=True)
            p = a.out_dir / f"{t['session']['session_id']}.json"
            p.write_text(json.dumps(t, indent=2, default=str))
            print(f"wrote {p} ({p.stat().st_size:,}B)", file=sys.stderr)
        if a.summary or not a.out_dir:
            print_summary(t)
            print()


if __name__ == "__main__":
    main()
