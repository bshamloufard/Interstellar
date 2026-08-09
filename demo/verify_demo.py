#!/usr/bin/env python3
"""Verify the fabricated demo session, for free, then with one real analyzer call.

    python3 demo/verify_demo.py            # structural + normalizer checks only
    python3 demo/verify_demo.py --dry-run  # + a real (cheap) analyzer pass
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.parse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_HOME = Path.home() / ".grok-interstellar-demo"
REPO_DIR = Path.home() / "interstellar-demo" / "checkout"
SESSION_ID = "019fe500-a100-7000-8000-000000000001"


def session_dir() -> Path:
    encoded = urllib.parse.quote(str(REPO_DIR), safe="")
    return DEMO_HOME / "sessions" / encoded / SESSION_ID


def check(label, cond):
    status = "OK  " if cond else "FAIL"
    print(f"[{status}] {label}")
    return cond


def main():
    ok = True
    d = session_dir()
    ok &= check(f"session dir exists ({d})", d.is_dir())
    for name in ("summary.json", "events.jsonl", "chat_history.jsonl",
                 "updates.jsonl", "signals.json", "prompt_context.json",
                 "system_prompt.txt"):
        ok &= check(f"{name} present", (d / name).is_file())

    # every line of every .jsonl parses
    for name in ("events.jsonl", "chat_history.jsonl", "updates.jsonl"):
        try:
            for line in (d / name).read_text().splitlines():
                if line.strip():
                    json.loads(line)
            ok &= check(f"{name} is valid JSONL", True)
        except json.JSONDecodeError as e:
            ok &= check(f"{name} is valid JSONL ({e})", False)

    # tool_call_ids referenced in chat_history all have a matching tool_result
    chat = [json.loads(l) for l in (d / "chat_history.jsonl").read_text().splitlines() if l.strip()]
    call_ids = {tc["id"] for m in chat if m.get("type") == "assistant"
                for tc in m.get("tool_calls") or []}
    result_ids = {m["tool_call_id"] for m in chat if m.get("type") == "tool_result"}
    ok &= check("every tool_call has a matching tool_result", call_ids == result_ids)

    sys.path.insert(0, str(REPO_ROOT / "normalizer"))
    import grok_normalize  # noqa: E402

    trace = grok_normalize.normalize_session(d)
    ok &= check("normalizer produced a trace", trace is not None)
    if trace is None:
        sys.exit(1 if not ok else 0)

    m = trace["metrics"]
    skill = next((s for s in m["skill_detail"] if s["skill"] == "checkout-debugging"), None)
    ok &= check("skill checkout-debugging was loaded", skill is not None)
    if skill:
        ok &= check(f"skill has >=150 unreferenced tokens (got {skill['unreferenced_tokens_est']})",
                     skill["unreferenced_tokens_est"] >= 150)
        ok &= check(f"skill has a repeated (duplicate) section (got {skill['repeated_sections']})",
                     skill["repeated_sections"] >= 1)

    failed = {f["server"] for f in m["failed_mcp_servers"]}
    ok &= check(f"github is a failed MCP server (got {failed})", "github" in failed)

    idle = {i["server"] for i in m["idle_mcp_servers"]}
    ok &= check(f"filesystem/memory/sequential-thinking/time are idle (got {idle})",
                {"filesystem", "memory", "sequential-thinking", "time"} <= idle)

    slow = [s for s in m["timing"]["slowest"] if s["ms"] >= 2000]
    ok &= check(f"a >=2000ms measured call exists (got {[s['ms'] for s in slow]})", bool(slow))

    dupes = {d_["tool"]: d_["times"] for d_ in m["duplicate_calls"]}
    ok &= check(f"read_file has a duplicate-call finding (got {dupes.get('read_file')})",
                dupes.get("read_file", 0) >= 3)

    out_dir = REPO_ROOT / "demo" / "_verify_out"
    out_dir.mkdir(exist_ok=True)
    trace_path = out_dir / f"{SESSION_ID}.json"
    trace_path.write_text(json.dumps(trace, indent=2, default=str))
    print(f"trace written to {trace_path}")

    if "--dry-run" in sys.argv:
        print("\n== running the real analyzer (--dry-run, no sandbox spend) ==")
        result = subprocess.run(
            [sys.executable, "-m", "interstellar", "review", str(trace_path),
             "--dry-run", "--grok-home", str(DEMO_HOME)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
        ok &= check("interstellar review --dry-run exited 0", result.returncode == 0)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
