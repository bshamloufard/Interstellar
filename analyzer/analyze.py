#!/usr/bin/env python3
"""Run the analyzer agent over a normalized trace, using the same grok build.

Two jobs, kept apart on purpose:

  digest()  - deterministic. Shrinks a trace to the facts worth reasoning about
              and STRIPS ground_truth, so the model is never graded open-book.
  analyze() - the only model call. Structured output via --json-schema, so the
              result is parseable without any cleanup pass.

    python3 analyze.py ../traces/corpus/skill_bloat_*.json
    python3 analyze.py --all ../traces/corpus --out results/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
GROK = Path.home() / ".local" / "bin" / "grok-dev"

SCHEMA = {
    "type": "object",
    "properties": {
        "session_verdict": {
            "type": "string",
            "enum": ["clean", "minor_waste", "significant_waste"],
        },
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": [
                        "skill_truncate", "skill_remove", "skill_add_lines",
                        "skill_keep", "tool_redundant", "tool_output_truncate",
                        "mcp_remove", "mcp_latency", "mcp_fix_broken",
                        "subagent_inline", "permission_friction"]},
                    "target": {"type": "string",
                               "description": "skill name, tool name, or server name"},
                    "action": {"type": "string",
                               "description": "one imperative sentence"},
                    "evidence": {"type": "string",
                                 "description": "numbers quoted from the digest"},
                    "line_ranges": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "integer"}},
                        "description": "required for skill_truncate",
                    },
                    "tokens_saved_est": {"type": "integer"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["type", "target", "action", "evidence", "confidence"],
            },
        },
    },
    "required": ["session_verdict", "recommendations"],
}


def digest(trace, *, max_sections=24):
    """Deterministic shrink. Ground truth is removed here, not later."""
    m, inv, s = trace["metrics"], trace["inventory"], trace["session"]

    skills = []
    for span in trace["spans"]:
        if span["kind"] != "skill":
            continue
        a = span["attrs"]
        skills.append({
            "skill": a["skill"],
            "tokens_est": a["skill_tokens_est"],
            "lines": a["skill_lines"],
            "unreferenced_tokens_est": a["unreferenced_tokens_est"],
            "unreferenced_line_ranges": a["unreferenced_line_ranges"],
            "repeated_sections": a.get("repeated_sections", 0),
            "repeated_line_ranges": a.get("repeated_line_ranges", []),
            "sections": [
                {k: sec[k] for k in
                 ("section", "start_line", "end_line", "tokens_est", "referenced")}
                for sec in a["sections"][:max_sections]
            ],
        })

    # Only surface timing the analyzer is allowed to act on.
    measured = [x for x in m["timing"]["slowest"] if x["ms"] > 0]
    return {
        "session": {"id": s["session_id"], "title": s["title"],
                    "model": s["model"], "wall_ms": s["duration_ms"]},
        "counts": m["counts"],
        "timing": {
            "note": "all durations below are duration_source=measured",
            "total_tool_ms": m["timing"]["total_tool_ms"],
            "permission_wait_ms": m["timing"]["total_permission_wait_ms"],
            "mcp_startup_ms": m["timing"]["mcp_startup_ms"],
            "slowest_measured": measured[:6],
        },
        "context_cost": {
            "tool_result_tokens_est": m["context_cost"]["tool_result_tokens_est"],
            "skill_tokens_est": m["context_cost"]["skill_tokens_est"],
            "skill_wasted_tokens_est": m["context_cost"]["skill_wasted_tokens_est"],
            "largest_results": m["context_cost"]["largest_results"][:5],
        },
        "skills_loaded": skills,
        "installed_skills_never_loaded": trace.get("installed_not_loaded", []),
        "discovery_after_skill_load": m.get("discovery_after_skill_load", []),
        "duplicate_calls": m["duplicate_calls"][:8],
        "idle_mcp_servers": m["idle_mcp_servers"],
        "failed_mcp_servers": m.get("failed_mcp_servers", []),
        "slowest_mcp_connect": m["timing"].get("slowest_mcp_connect", []),
        "mcp_servers": {k: {kk: vv for kk, vv in v.items() if kk != "tools"}
                        for k, v in inv["mcp_servers"].items()},
        "subagents": m["subagents"],
        "tools_used": inv["tools_used"],
        "warnings": trace["warnings"],
    }


def analyze(dig, *, model=None, timeout=600):
    prompt = (
        (HERE / "prompt.md").read_text()
        + "\n\n## DIGEST\n\n```json\n"
        + json.dumps(dig, indent=1, default=str)
        + "\n```\n"
    )
    cmd = [str(GROK), "-p", prompt,
           "--json-schema", json.dumps(SCHEMA),
           "--permission-mode", "bypassPermissions",
           "--disallowed-tools", "run_terminal_command,read_file,write,search_replace"]
    if model:
        cmd += ["--model", model]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None, {"error": "unparseable", "stdout": proc.stdout[-500:],
                      "stderr": proc.stderr[-500:]}
    text = out.get("text") or ""
    try:
        return json.loads(text), {"cost_usd": out.get("total_cost_usd"),
                                  "usage": out.get("usage")}
    except json.JSONDecodeError:
        return None, {"error": "model output not JSON", "text": text[:500]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="*", type=Path)
    ap.add_argument("--all", type=Path, help="directory of traces")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--model")
    ap.add_argument("--digest-only", action="store_true")
    a = ap.parse_args()

    files = list(a.traces)
    if a.all:
        files += sorted(p for p in a.all.glob("*.json") if p.name != "index.json")
    if not files:
        raise SystemExit("no traces given")

    a.out.mkdir(parents=True, exist_ok=True)
    cost = 0.0
    for f in files:
        trace = json.loads(f.read_text())
        dig = digest(trace)
        if a.digest_only:
            print(json.dumps(dig, indent=2, default=str))
            continue

        print(f"{f.name:44}", end="", flush=True)
        recs, meta = analyze(dig, model=a.model)
        cost += (meta or {}).get("cost_usd") or 0
        if recs is None:
            print(f"  FAILED {meta}")
            continue
        payload = {
            "trace_file": f.name,
            "digest": dig,
            "result": recs,
            "meta": meta,
            # kept for the scorer only; never fed to the model
            "ground_truth": trace.get("ground_truth"),
        }
        (a.out / f"{f.stem}.result.json").write_text(
            json.dumps(payload, indent=2, default=str))
        rs = recs.get("recommendations", [])
        print(f"  {recs.get('session_verdict'):18} {len(rs)} rec(s): "
              f"{', '.join(sorted({r['type'] for r in rs})) or '-'}")
    if not a.digest_only:
        print(f"\nanalyzer cost ${cost:.4f} -> {a.out}")


if __name__ == "__main__":
    main()
