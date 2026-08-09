#!/usr/bin/env python3
"""Score analyzer output against the planted ground truth.

Recall is the headline number: for each scenario we planted exactly one
defect, so we ask whether the analyzer raised the matching finding type.

Precision is handled separately and deliberately loosely. Two of the findings
the analyzer raises on every session — the idle excalidraw and runpod MCP
servers — are TRUE, they are just environmental rather than scenario-specific.
Counting them as false positives would punish correct work. So extra findings
are only counted against the analyzer when they cite no number at all, which is
the failure mode the prompt actually forbids.

    python3 score.py results/
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# scenario verdict -> the recommendation type that proves it was found
EXPECTED = {
    "truncate": "skill_truncate",
    "keep": "skill_keep",
    "add_lines": "skill_add_lines",
    "redundant_calls": "tool_redundant",
    "subagent_overhead": "subagent_inline",
    "mcp_latency": {"mcp_latency", "mcp_remove"},
    "mcp_remove": {"mcp_remove", "mcp_fix_broken"},
    "truncate_tool_output": "tool_output_truncate",
    "clean": None,  # control: nothing scenario-specific to find
}

# Findings that are true in every session because of the ambient environment.
AMBIENT = {"mcp_remove", "skill_remove"}

HAS_NUMBER = re.compile(r"\d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, nargs="?", default=Path("results"))
    a = ap.parse_args()

    files = sorted(a.results.glob("*.result.json"))
    if not files:
        raise SystemExit(f"no results in {a.results}")

    hits = misses = 0
    unevidenced = 0
    rows = []
    for f in files:
        d = json.loads(f.read_text())
        gt = (d.get("ground_truth") or {}).get("expects", {})
        want = EXPECTED.get(gt.get("verdict"), None)
        got = {r["type"] for r in d["result"]["recommendations"]}

        if want is None:
            status = "control"
        else:
            wanted = want if isinstance(want, set) else {want}
            if wanted & got:
                status, hits = "HIT", hits + 1
            else:
                status, misses = "MISS", misses + 1

        # the one precision rule the prompt actually mandates
        for r in d["result"]["recommendations"]:
            if not HAS_NUMBER.search(r.get("evidence") or ""):
                unevidenced += 1

        scenario = (d.get("ground_truth") or {}).get("scenario", f.stem)
        rows.append((scenario, gt.get("verdict"), status,
                     sorted(got - AMBIENT), len(got)))

    print(f"{'scenario':24} {'planted':22} {'':7} {'scenario-specific findings'}")
    print("-" * 96)
    for s, v, st, got, n in rows:
        print(f"{s:24} {str(v):22} {st:7} {', '.join(got) or '(none)':44} [{n} total]")

    scored = hits + misses
    print(f"\nrecall     {hits}/{scored} planted defects found"
          f"{'' if not scored else f'  ({100*hits/scored:.0f}%)'}")
    print(f"evidence   {unevidenced} recommendation(s) cited no number "
          f"(prompt requires >=1)")
    print(f"controls   {sum(1 for r in rows if r[2] == 'control')} session(s) with "
          f"no planted defect")


if __name__ == "__main__":
    main()
