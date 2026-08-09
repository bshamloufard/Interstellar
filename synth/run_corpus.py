#!/usr/bin/env python3
"""Drive grok-build headlessly to produce a labelled corpus of real sessions.

Each scenario is written to provoke a specific, known defect so the analyzer
has something findable and we have something to score against. The scenario's
`expects` field is the label; it is carried through to the manifest and never
shown to the model.

    python3 run_corpus.py --list
    python3 run_corpus.py --only skill_bloat
    python3 run_corpus.py --all --out manifest.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

GROK = Path.home() / ".local" / "bin" / "grok-dev"
HERE = Path(__file__).parent
WORKSPACE = HERE / "workspace"

# --------------------------------------------------------------------------
# Scenarios. `expects` is ground truth for scoring; the model never sees it.
# --------------------------------------------------------------------------
SCENARIOS = [
    {
        "id": "skill_bloat",
        "prompt": "Read the strict-audit skill at ~/.grok/skills/strict-audit/SKILL.md "
                  "and apply its procedure to app.py.",
        "expects": {
            "skill": "strict-audit",
            "verdict": "truncate",
            "note": "loads 5.8KB but only the 6-line Procedure section drives behaviour",
        },
    },
    {
        "id": "skill_lean",
        "prompt": "Read the py-import-hygiene skill at ~/.grok/skills/py-import-hygiene/SKILL.md "
                  "and apply it to utils.py.",
        "expects": {"skill": "py-import-hygiene", "verdict": "keep",
                    "note": "small body, every line actionable"},
    },
    {
        "id": "skill_underspecified",
        "prompt": "Read the changelog-entry skill at ~/.grok/skills/changelog-entry/SKILL.md "
                  "and follow it for the change 'added config parser'.",
        "expects": {"skill": "changelog-entry", "verdict": "add_lines",
                    "note": "skill never names the target file, so the agent "
                            "burns exploratory tool calls finding one"},
    },
    {
        "id": "tool_thrash",
        "prompt": "Check whether app.py has a syntax error. Read it, then run python3 -m py_compile "
                  "on it, then read it again to double check, then compile it again to be sure.",
        "expects": {"verdict": "redundant_calls",
                    "note": "same file read twice, same compile run twice, identical args"},
    },
    {
        "id": "subagent_spawn",
        "prompt": "Use a subagent to summarize what utils.py does, then report its answer.",
        "expects": {"verdict": "subagent_overhead",
                    "note": "delegates a task small enough to do inline"},
    },
    {
        "id": "mcp_slow",
        "prompt": "Use the runpod MCP tools to list the available GPU types. "
                  "Then tell me how many there were.",
        "expects": {"verdict": "mcp_latency",
                    "note": "runpod is an npx stdio server; connect + call are slow"},
    },
    # --- scenarios against REAL public skills (sourced from the Anthropic /
    # superpowers skill sets) and REAL MCP servers, so the corpus is not
    # entirely made of hand-built fixtures.
    {
        "id": "real_skill_bloat",
        "prompt": "Read the code-review skill at ~/.grok/skills/code-review/SKILL.md "
                  "and apply it to app.py.",
        "expects": {"skill": "code-review", "verdict": "truncate",
                    "note": "211-line real skill; history/philosophy sections "
                            "never drive behaviour"},
    },
    {
        "id": "real_skill_lean",
        "prompt": "Read the verification-before-completion skill at "
                  "~/.grok/skills/verification-before-completion/SKILL.md and "
                  "apply it to verify utils.py is correct.",
        "expects": {"skill": "verification-before-completion", "verdict": "keep",
                    "note": "37-line real skill, all actionable"},
    },
    {
        "id": "real_skill_debug",
        "prompt": "Read the systematic-debugging skill at "
                  "~/.grok/skills/systematic-debugging/SKILL.md and use it to find "
                  "why parse_config crashes on a line with no equals sign.",
        "expects": {"skill": "systematic-debugging", "verdict": "truncate",
                    "note": "152-line real skill with long rationalization prose"},
    },
    {
        "id": "mcp_filesystem",
        "prompt": "Use the filesystem MCP server to list the directory "
                  "/Users/sentorcc/Interstellar/synth/workspace and read utils.py "
                  "through it. Report what you found.",
        "expects": {"verdict": "mcp_latency",
                    "note": "npx stdio server; multi-second connect for a task "
                            "native tools already cover"},
    },
    {
        "id": "mcp_unused_fleet",
        "prompt": "What is 17 * 23? Answer with just the number.",
        "expects": {"verdict": "mcp_remove",
                    "note": "trivial task pays full MCP startup for 96 tools, "
                            "calls none of them"},
    },
    {
        # The slash path is the one main could not price before the skill-cost
        # patch: no backing tool call, so nothing recorded what it cost.
        "id": "skill_slash_invoke",
        "prompt": "/strict-audit app.py",
        "expects": {"skill": "strict-audit", "verdict": "truncate",
                    "note": "slash activation; cost comes from skill_bytes on "
                            "the native event, not a tool_result join"},
    },
    {
        "id": "baseline_clean",
        "prompt": "In one sentence, what does slugify in utils.py do?",
        "expects": {"verdict": "clean", "note": "minimal well-scoped session; control group"},
    },
    {
        "id": "big_output",
        # Must force the bytes THROUGH the context. An earlier version asked for
        # a count, so the agent piped to `wc -l`, pulled back 17 bytes, and there
        # was correctly nothing to flag.
        "prompt": "Run 'cat /Users/sentorcc/Interstellar/Cargo.toml' and then name the "
                  "three workspace members whose paths sort last alphabetically. "
                  "You must read the actual file contents, do not use grep, sort, "
                  "tail or any other filter to narrow it down first.",
        "expects": {"verdict": "truncate_tool_output",
                    "note": "large terminal result pulled wholesale into context"},
    },
]


def session_ids_before():
    root = Path.home() / ".grok" / "sessions"
    return {p.name for p in root.glob("*/*") if p.is_dir()}


def run(scn, *, model=None, timeout=600):
    """Run one scenario, return a manifest record."""
    before = session_ids_before()
    cmd = [str(GROK), "-p", scn["prompt"],
           "--output-format", "json",
           "--permission-mode", "bypassPermissions"]
    if model:
        cmd += ["--model", model]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=WORKSPACE, capture_output=True,
                          text=True, timeout=timeout)
    wall = time.time() - t0

    rec = {"scenario": scn["id"], "expects": scn["expects"],
           "prompt": scn["prompt"], "wall_s": round(wall, 1)}

    try:
        out = json.loads(proc.stdout)
        rec.update({
            "session_id": out.get("sessionId"),
            "turns": out.get("num_turns"),
            "cost_usd": out.get("total_cost_usd"),
            "usage": out.get("usage"),
            "stop_reason": out.get("stopReason"),
            "ok": True,
        })
    except (json.JSONDecodeError, TypeError):
        # Fall back to diffing the sessions dir — a session almost always
        # lands on disk even when stdout is unusable.
        new = session_ids_before() - before
        rec.update({"ok": False,
                    "session_id": next(iter(new), None),
                    "stderr": (proc.stderr or "")[-400:]})
    return rec


def locate(session_id):
    if not session_id:
        return None
    root = Path.home() / ".grok" / "sessions"
    for p in root.glob(f"*/{session_id}"):
        return str(p)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", action="append", help="scenario id (repeatable)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run each scenario N times for variance")
    ap.add_argument("--out", type=Path, default=HERE / "manifest.json")
    a = ap.parse_args()

    if a.list:
        for s in SCENARIOS:
            print(f"  {s['id']:22} -> {s['expects']['verdict']}")
        return

    picked = [s for s in SCENARIOS if not a.only or s["id"] in a.only]
    if not picked:
        print("no scenarios matched")
        return
    if not GROK.exists():
        raise SystemExit(f"grok binary not found at {GROK}")

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    records, cost = [], 0.0
    for rep in range(a.repeat):
        for s in picked:
            print(f"[{len(records)+1}/{len(picked)*a.repeat}] {s['id']} ...",
                  end="", flush=True)
            try:
                r = run(s, model=a.model)
            except subprocess.TimeoutExpired:
                r = {"scenario": s["id"], "ok": False, "error": "timeout"}
            r["repeat"] = rep
            r["session_dir"] = locate(r.get("session_id"))
            records.append(r)
            cost += r.get("cost_usd") or 0
            status = "ok" if r.get("ok") else "FAIL"
            print(f" {status}  {r.get('turns','?')} turns  "
                  f"${r.get('cost_usd') or 0:.4f}  {r.get('wall_s','?')}s")

    manifest = {
        "generated_by": "run_corpus.py",
        "grok_version": subprocess.run([str(GROK), "--version"],
                                       capture_output=True, text=True).stdout.split("\n")[0],
        "workspace": str(WORKSPACE),
        "total_cost_usd": round(cost, 4),
        "sessions": records,
    }
    a.out.write_text(json.dumps(manifest, indent=2))
    ok = sum(1 for r in records if r.get("ok"))
    print(f"\n{ok}/{len(records)} sessions ok | ${cost:.4f} | manifest -> {a.out}")


if __name__ == "__main__":
    main()
