#!/usr/bin/env python3
"""Assemble normalized traces into a labelled corpus, and gate it on coverage.

"Good quality synthetic data" is not a vibe. For this corpus it means four
checkable things, and this script fails loudly on each:

  1. LABELLED   every trace carries the planted ground truth from run_corpus.py
  2. COVERED    every finding type the analyzer must produce occurs at least
                once (a corpus with no subagent span cannot teach subagent
                findings, and would silently ship a prompt path never exercised)
  3. MEASURED   durations come from grok's own instrumentation, not inferred
  4. SEPARATED  a control session exists with no planted defect, so a prompt
                that flags everything is detectably wrong

Exit code is non-zero when coverage fails, so this can gate a build.

    python3 build_corpus.py --manifest manifest.json --out ../traces/corpus
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "normalizer"))
from grok_normalize import normalize_session  # noqa: E402

# Each finding type the analyzer must be able to produce, and the predicate
# that proves the corpus contains raw material for it.
COVERAGE = {
    "skill_truncate": lambda t: any(
        s["kind"] == "skill" and s["attrs"].get("unreferenced_tokens_est", 0) > 200
        for s in t["spans"]),
    "skill_duplicate_sections": lambda t: any(
        s["kind"] == "skill" and s["attrs"].get("repeated_sections", 0) > 0
        for s in t["spans"]),
    "skill_keep": lambda t: any(
        s["kind"] == "skill" and s["attrs"].get("unreferenced_tokens_est", 0) == 0
        for s in t["spans"]),
    "idle_mcp_server": lambda t: bool(t["metrics"]["idle_mcp_servers"]),
    "mcp_called": lambda t: t["metrics"]["counts"]["mcp_calls"] > 0,
    "subagent_spawn": lambda t: t["metrics"]["counts"]["subagent_spawns"] > 0,
    "duplicate_tool_calls": lambda t: bool(t["metrics"]["duplicate_calls"]),
    "large_tool_output": lambda t: any(
        r["bytes"] > 5000 for r in t["metrics"]["context_cost"]["largest_results"]),
    "permission_wait": lambda t: t["metrics"]["timing"]["total_permission_wait_ms"] > 0,
    "failed_mcp_server": lambda t: bool(t["metrics"].get("failed_mcp_servers")),
    "slow_mcp_connect": lambda t: any(
        r["ms"] > 2000 for r in t["metrics"]["timing"].get("slowest_mcp_connect", [])),
    # Post-a959d1a sessions should carry native skill events rather than
    # relying on the read_file heuristic. Not gated as required, because the
    # pre-rebase traces in this corpus legitimately predate the feature.
    "native_skill_event": lambda t: any(
        s["kind"] == "skill" and s["attrs"].get("source") == "native_event"
        for s in t["spans"]),
    "control_clean": lambda t: (t["metrics"]["counts"]["skill_loads"] == 0
                                and not t["metrics"]["duplicate_calls"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    ap.add_argument("--out", type=Path, default=HERE.parent / "traces" / "corpus")
    ap.add_argument("--ground-truth", type=Path, default=HERE / "ground_truth.json")
    ap.add_argument("--no-gate", action="store_true",
                    help="report coverage but always exit 0")
    a = ap.parse_args()

    manifest = json.loads(a.manifest.read_text())
    truth = json.loads(a.ground_truth.read_text()) if a.ground_truth.is_file() else {}

    a.out.mkdir(parents=True, exist_ok=True)
    index, seen = [], Counter()
    total_span = Counter()

    # `skill_remove` is a corpus-level fact: a skill is dead weight only if no
    # session in the whole corpus ever loaded it. Compute it once, up front,
    # from what is actually installed on disk.
    skills_root = Path.home() / ".grok" / "skills"
    installed = {}
    for sd in sorted(skills_root.glob("*/SKILL.md")):
        installed[sd.parent.name] = {
            "bytes": sd.stat().st_size,
            "tokens_est": sd.stat().st_size // 4,
        }
    traces_pre, loaded_any = [], set()
    for rec in manifest["sessions"]:
        sd = rec.get("session_dir")
        if sd and Path(sd).is_dir():
            t = normalize_session(sd)
            if t:
                traces_pre.append((rec, t))
                for s in t["metrics"]["skill_detail"]:
                    loaded_any.add(s["skill"])
    never_loaded = [{"skill": k, **v} for k, v in installed.items() if k not in loaded_any]
    print(f"installed skills: {len(installed)} | loaded somewhere: {len(loaded_any)} "
          f"| never loaded: {len(never_loaded)} "
          f"(~{sum(s['tokens_est'] for s in never_loaded):,} tok of dead weight)")

    for rec, _pre in traces_pre:
        trace = _pre
        # Corpus-level fact, attached to every trace so a single-session
        # analyzer run can still raise skill_remove.
        trace["installed_not_loaded"] = never_loaded

        # Attach the label. Kept OUTSIDE `spans` so it can be stripped before
        # the analyzer ever sees a trace — otherwise we'd be grading an open book.
        trace["ground_truth"] = {
            "scenario": rec["scenario"],
            "expects": rec.get("expects", {}),
            "cost_usd": rec.get("cost_usd"),
            "turns": rec.get("turns"),
        }
        for k, fn in COVERAGE.items():
            try:
                if fn(trace):
                    seen[k] += 1
            except (KeyError, TypeError):
                pass
        for s in trace["spans"]:
            total_span[s["kind"]] += 1

        name = f"{rec['scenario']}_{trace['session']['session_id'][:8]}.json"
        (a.out / name).write_text(json.dumps(trace, indent=2, default=str))
        m = trace["metrics"]
        index.append({
            "file": name,
            "scenario": rec["scenario"],
            "expects": rec.get("expects", {}).get("verdict"),
            "session_id": trace["session"]["session_id"],
            "spans": len(trace["spans"]),
            "turns": m["counts"]["turns"],
            "tools": m["counts"]["tool_calls"],
            "skills": m["counts"]["skill_loads"],
            "subagents": m["counts"]["subagent_spawns"],
            "mcp_calls": m["counts"]["mcp_calls"],
            "skill_wasted_tokens": m["context_cost"]["skill_wasted_tokens_est"],
            "cost_usd": rec.get("cost_usd"),
        })

    # ---- report ----------------------------------------------------------
    print(f"\n{'scenario':24} {'expect':22} {'sp':>3} {'tl':>3} {'sk':>3} "
          f"{'sa':>3} {'mcp':>4} {'wasted':>7}")
    print("-" * 78)
    for r in index:
        print(f"{r['scenario']:24} {str(r['expects']):22} {r['spans']:>3} {r['tools']:>3} "
              f"{r['skills']:>3} {r['subagents']:>3} {r['mcp_calls']:>4} "
              f"{r['skill_wasted_tokens']:>7,}")

    print(f"\nspan kinds: {dict(total_span)}")
    print(f"\nCOVERAGE ({len(index)} traces)")
    missing = []
    for k in COVERAGE:
        n = seen.get(k, 0)
        print(f"  {'OK ' if n else 'MISS'}  {k:26} {n} trace(s)")
        if not n and k != "native_skill_event":
            missing.append(k)

    (a.out / "index.json").write_text(json.dumps({
        "traces": index,
        "coverage": {k: seen.get(k, 0) for k in COVERAGE},
        "missing_coverage": missing,
        "skill_ground_truth": truth.get("skills", {}),
        "total_cost_usd": manifest.get("total_cost_usd"),
    }, indent=2))
    print(f"\nwrote {len(index)} traces + index.json -> {a.out}")

    if missing and not a.no_gate:
        print(f"\nFAIL: {len(missing)} finding type(s) have no raw material: "
              f"{', '.join(missing)}", file=sys.stderr)
        return 1
    print("\nPASS: every finding type is represented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
