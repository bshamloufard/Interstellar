#!/usr/bin/env python3
"""Fabricate the 'before' grok-build session for the Interstellar demo.

Writes a real grok-build session directory — the exact five-file raw schema —
from one ordered list of narrative "beats". Every field below was reverse-
engineered from a real local grok-build session; see
docs/superpowers/specs/2026-08-08-interstellar-demo-design.md.

Run only after install.sh has materialized the repo + harness fixtures — this
script reads their on-disk content to compute accurate byte/line counts, and
NEVER writes to the checkout repo (only reads it).
"""
from __future__ import annotations

import json
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEMO_HOME = Path.home() / ".grok-interstellar-demo"
REPO_DIR = Path.home() / "interstellar-demo" / "checkout"
SKILL_PATH = DEMO_HOME / "skills" / "checkout-debugging" / "SKILL.md"

SESSION_ID = "019fe500-a100-7000-8000-000000000001"
MODEL_ID = "grok-4.5"
MODEL_FINGERPRINT = "fp_demo000000001"
AGENT_NAME = "grok-build"
START = datetime(2026, 8, 9, 15, 0, 0, tzinfo=timezone.utc)

TASK_PROMPT = (
    "Checkout totals are wrong on orders with a discount code — tax looks "
    "like it's being charged on the pre-discount amount. Find the bug and fix "
    "it. pytest should pass."
)


def ts_at(ms: int) -> str:
    t = START + timedelta(milliseconds=ms)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def numbered(text: str) -> str:
    """Match grok-build's read_file convention: '<n>→<line>' per line."""
    return "\n".join(f"{i}→{line}" for i, line in enumerate(text.splitlines(), 1))


def read_numbered(path: Path) -> str:
    return numbered(path.read_text())


# ---------------------------------------------------------------------------
# Fixed-vs-buggy pricing.py text: read the real (buggy) file off disk, derive
# the fixed variant by a targeted, self-verifying string replace so the two
# versions can never silently drift apart.
# ---------------------------------------------------------------------------
BUGGY_PRICING = (REPO_DIR / "pricing.py").read_text()
_OLD_BODY = (
    "    discount = discount_amount_cents(subtotal_cents, discount_pct)\n"
    "    tax = tax_cents(subtotal_cents)  # BUG: taxes the pre-discount subtotal\n"
    "    return subtotal_cents - discount + tax"
)
_NEW_BODY = (
    "    discount = discount_amount_cents(subtotal_cents, discount_pct)\n"
    "    discounted_subtotal = subtotal_cents - discount\n"
    "    tax = tax_cents(discounted_subtotal)\n"
    "    return discounted_subtotal + tax"
)
assert BUGGY_PRICING.count(_OLD_BODY) == 1, (
    "pricing.py fixture text has drifted from Task 1 Step 2 — "
    "update _OLD_BODY/_NEW_BODY to match before regenerating the session"
)
FIXED_PRICING = BUGGY_PRICING.replace(_OLD_BODY, _NEW_BODY)

_call_i = [0]


def call_id() -> str:
    _call_i[0] += 1
    return f"call-demo0000-0000-4000-8000-{_call_i[0]:012d}"


# ---------------------------------------------------------------------------
# The narrative. Each beat is one tool interaction. `result` is a thunk so it
# can reference on-disk content read above without re-reading it per-beat.
# ---------------------------------------------------------------------------
BEATS = [
    dict(
        gap_ms=1800, tool="read_file",
        args={"target_file": str(SKILL_PATH)},
        result=lambda: read_numbered(SKILL_PATH),
        duration_ms=6,
        skill=dict(name="checkout-debugging", trigger="skill_md_read",
                   plugin_source=None,
                   bytes=len(SKILL_PATH.read_bytes()),
                   lines=SKILL_PATH.read_text().count("\n") + 1),
        thought=("There's a checkout-debugging skill installed for this repo "
                  "— reading it before touching pricing code."),
    ),
    dict(
        gap_ms=900, tool="read_file",
        args={"target_file": str(REPO_DIR / "locale.py")},
        result=lambda: f"Error: file not found: {REPO_DIR / 'locale.py'}",
        duration_ms=4, outcome="error",
        thought=("The skill says formatting/locale is the usual cause of a "
                  "wrong checkout total — checking locale.py first."),
    ),
    dict(
        gap_ms=400, tool="run_terminal_command",
        args={"command": f"grep -ril locale {REPO_DIR}"},
        result=lambda: "",
        duration_ms=85,
    ),
    dict(
        gap_ms=500, tool="list_dir",
        args={"target_directory": str(REPO_DIR)},
        result=lambda: "- pricing.py\n- cart.py\n- test_pricing.py\n- README.md\n- .git/",
        duration_ms=12,
        message_after=("No locale/formatting module in this repo — "
                        "moving to the pricing module directly."),
    ),
    dict(
        gap_ms=600, tool="read_file",
        args={"target_file": str(REPO_DIR / "pricing.py")},
        result=lambda: numbered(BUGGY_PRICING),
        duration_ms=5,
    ),
    dict(
        gap_ms=300, tool="read_file",
        args={"target_file": str(REPO_DIR / "cart.py")},
        result=lambda: read_numbered(REPO_DIR / "cart.py"),
        duration_ms=4,
    ),
    dict(
        gap_ms=300, tool="read_file",
        args={"target_file": str(REPO_DIR / "test_pricing.py")},
        result=lambda: read_numbered(REPO_DIR / "test_pricing.py"),
        duration_ms=4,
        thought=("checkout_total taxes subtotal_cents directly, before the "
                  "discount is subtracted, instead of taxing the discounted "
                  "subtotal — that matches the report exactly."),
    ),
    dict(
        gap_ms=700, tool="use_tool",
        args={"tool_name": "everything__trigger-long-running-operation",
              "duration_seconds": 20},
        result=lambda: "Long-running operation completed after 20000ms.",
        duration_ms=20000,
        thought="Per the skill, verifying environment timing before making the change.",
    ),
    dict(
        gap_ms=500, tool="edit_file",
        args={"target_file": str(REPO_DIR / "pricing.py"),
              "instructions": "Tax the post-discount subtotal instead of the raw subtotal"},
        result=lambda: "Applied 1 edit to pricing.py",
        duration_ms=40,
    ),
    dict(
        gap_ms=300, tool="read_file",
        args={"target_file": str(REPO_DIR / "pricing.py")},
        result=lambda: numbered(FIXED_PRICING),
        duration_ms=5,
    ),
    dict(
        gap_ms=300, tool="read_file",
        args={"target_file": str(REPO_DIR / "pricing.py")},
        result=lambda: numbered(FIXED_PRICING),
        duration_ms=5,
    ),
    dict(
        gap_ms=400, tool="run_terminal_command",
        args={"command": f"cd {REPO_DIR} && pytest -q"},
        result=lambda: "2 passed in 0.04s",
        duration_ms=340,
        message_after=("Fixed — tax is now computed on the discounted "
                        "subtotal, and both tests pass."),
    ),
]

FINAL_MESSAGE = (
    "Fixed the checkout total bug: `checkout_total` was taxing the pre-discount "
    "subtotal instead of the post-discount amount. Updated `pricing.py` to "
    "compute tax on the discounted subtotal, and `pytest` is green (2 passed)."
)


def _update(kind: str, payload: dict, clock_ms: int) -> dict:
    return {
        "timestamp": int(START.timestamp()) + clock_ms // 1000,
        "method": "session/update",
        "params": {"sessionId": SESSION_ID,
                   "update": {"sessionUpdate": kind, **payload}},
    }


def build():
    events, chat, updates = [], [], []
    clock = 0

    servers = [
        ("everything", "stdio", 420, 19),
        ("filesystem", "stdio", 480, 14),
        ("memory", "stdio", 540, 9),
        ("sequential-thinking", "stdio", 580, 1),
        ("time", "stdio", 610, 2),
    ]
    events.append({"ts": ts_at(0), "type": "mcp_config_resolved",
                    "servers": [{"name": n, "transport": t, "source": "local"}
                                for n, t, _, _ in servers]
                               + [{"name": "github", "transport": "http", "source": "local"}],
                    "disabled": []})
    for n, t, _, _ in servers + [("github", "http", 0, 0)]:
        events.append({"ts": ts_at(5), "type": "mcp_server_starting",
                        "server_name": n, "transport": t, "target": n,
                        "timeout_sec": 120})
    for n, t, at_ms, count in servers:
        events.append({"ts": ts_at(at_ms), "type": "mcp_server_connected",
                        "server_name": n, "transport": t, "tool_count": count,
                        "duration_ms": at_ms,
                        "tools": [f"{n}_tool_{i}" for i in range(count)]})
    events.append({"ts": ts_at(9210), "type": "mcp_server_failed",
                    "server_name": "github", "transport": "http",
                    "target": "https://api.githubcopilot.com/mcp/",
                    "error_type": "handshake_failed",
                    "error_message": "MCP server 'github' handshake failed: HTTP 401 on initialize",
                    "duration_ms": 9200, "timeout_sec": 10})
    events.append({"ts": ts_at(9215), "type": "mcp_init_completed",
                    "total_servers": 6, "succeeded": 5, "failed": 1,
                    "auth_required": 0, "total_tools": 45, "duration_ms": 9210,
                    "is_reinit": False, "failed_servers": ["github"]})
    clock = 9215

    events.append({"ts": ts_at(clock), "type": "turn_started",
                    "session_id": SESSION_ID, "turn_number": 0,
                    "model_id": MODEL_ID, "yolo_mode": False,
                    "conversation_message_count": 1,
                    "session_relationship": "primary", "schema_version": "1.0"})
    events.append({"ts": ts_at(clock + 2), "type": "loop_started", "loop_index": 0})
    clock += 1200
    events.append({"ts": ts_at(clock), "type": "first_token"})

    chat.append({"type": "system",
                 "content": ("You are Grok 4.5 released by xAI. You are an "
                             "interactive CLI tool that helps users with "
                             "software engineering tasks.")})
    chat.append({"type": "user", "content": [{"type": "text", "text": (
        f"<user_info>\nOS Version: macos\nShell: /bin/zsh\nWorkspace Path: "
        f"{REPO_DIR}\nToday's date: {START.date().isoformat()}\n</user_info>")}]})
    chat.append({"type": "user", "synthetic_reason": "skills_available",
                 "content": [{"type": "text", "text": (
                     "<system-reminder>\nThe following skills are available "
                     "for use:\n\n- checkout-debugging: Use when a checkout, "
                     "cart, or pricing calculation produces an incorrect "
                     "total.\n</system-reminder>")}]})
    chat.append({"type": "user", "synthetic_reason": "mcp_connected",
                 "content": [{"type": "text", "text": (
                     "<system-reminder>\nMCP servers connected: everything, "
                     "filesystem, memory, sequential-thinking, time\n"
                     "MCP servers failed: github\n</system-reminder>")}]})
    chat.append({"type": "user", "prompt_index": 0, "content": [
        {"type": "text", "text": f"<user_query>\n{TASK_PROMPT}\n</user_query>"}]})
    updates.append(_update("user_message_chunk",
                            {"content": {"type": "text", "text": TASK_PROMPT}}, clock))

    for beat in BEATS:
        clock += beat["gap_ms"]
        cid = call_id()
        tool, args = beat["tool"], beat["args"]
        result_text = beat["result"]()
        outcome = beat.get("outcome", "success")
        dur = beat["duration_ms"]

        if beat.get("thought"):
            chat.append({"type": "reasoning", "id": f"rs_{cid}",
                         "summary": [{"type": "summary_text", "text": beat["thought"]}]})
            updates.append(_update("agent_thought_chunk",
                                    {"content": {"type": "text", "text": beat["thought"]}}, clock))

        chat.append({"type": "assistant", "content": "", "tool_calls": [
            {"id": cid, "name": tool, "arguments": json.dumps(args)}],
            "model_id": f"{MODEL_ID}-build", "model_fingerprint": MODEL_FINGERPRINT,
            "reasoning_effort": "high"})
        updates.append(_update("tool_call",
                                {"toolCallId": cid, "title": tool, "rawInput": args}, clock))
        updates.append(_update("tool_call_update",
                                {"toolCallId": cid, "kind": "other", "title": tool}, clock))

        events.append({"ts": ts_at(clock), "type": "tool_started", "tool_name": tool})
        events.append({"ts": ts_at(clock), "type": "permission_requested", "tool_name": tool})
        events.append({"ts": ts_at(clock), "type": "permission_resolved",
                        "tool_name": tool, "decision": "allow", "wait_ms": 0})
        if beat.get("skill"):
            sk = beat["skill"]
            events.append({"ts": ts_at(clock), "type": "skill_activated",
                            "skill_name": sk["name"], "trigger": sk["trigger"],
                            "plugin_source": sk["plugin_source"],
                            "related_tool_call_id": cid,
                            "skill_bytes": sk["bytes"], "skill_lines": sk["lines"]})

        clock += dur
        events.append({"ts": ts_at(clock), "type": "tool_completed",
                        "tool_name": tool, "duration_ms": dur,
                        "outcome": outcome, "tool_call_id": cid})
        chat.append({"type": "tool_result", "tool_call_id": cid, "content": result_text})
        updates.append(_update("tool_call_update", {
            "toolCallId": cid, "status": "completed",
            "content": [{"type": "content",
                        "content": {"type": "text", "text": result_text[:2000]}}]}, clock))

        if beat.get("message_after"):
            clock += 400
            msg = beat["message_after"]
            chat.append({"type": "assistant", "content": msg,
                         "model_id": f"{MODEL_ID}-build",
                         "model_fingerprint": MODEL_FINGERPRINT,
                         "reasoning_effort": "high"})
            updates.append(_update("agent_message_chunk",
                                    {"content": {"type": "text", "text": msg}}, clock))

    clock += 700
    chat.append({"type": "assistant", "content": FINAL_MESSAGE,
                 "model_id": f"{MODEL_ID}-build", "model_fingerprint": MODEL_FINGERPRINT,
                 "reasoning_effort": "high"})
    updates.append(_update("agent_message_chunk",
                            {"content": {"type": "text", "text": FINAL_MESSAGE}}, clock))
    events.append({"ts": ts_at(clock), "type": "turn_ended", "outcome": "completed"})

    return events, chat, updates, clock


def write_all():
    events, chat, updates, final_clock = build()

    encoded_cwd = urllib.parse.quote(str(REPO_DIR), safe="")
    session_dir = DEMO_HOME / "sessions" / encoded_cwd / SESSION_ID
    session_dir.mkdir(parents=True, exist_ok=True)

    def jsonl(name, rows):
        (session_dir / name).write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")

    jsonl("events.jsonl", events)
    jsonl("chat_history.jsonl", chat)
    jsonl("updates.jsonl", updates)

    summary = {
        "info": {"id": SESSION_ID, "cwd": str(REPO_DIR)},
        "session_summary": "Checkout total wrong with discount code",
        "created_at": ts_at(0),
        "updated_at": ts_at(final_clock),
        "num_messages": len(updates),
        "num_chat_messages": len(chat),
        "current_model_id": MODEL_ID,
        "next_trace_turn": 1,
        "chat_format_version": 1,
        "git_root_dir": str(REPO_DIR) + "/",
        "git_remotes": [],
        "head_commit": "0" * 40,
        "head_branch": "main",
        "request_id": "demo0000-0000-0000-0000-000000000000",
        "grok_home": str(DEMO_HOME),
        "last_active_at": ts_at(final_clock),
        "generated_title": "Checkout total wrong with discount code",
        "agent_name": AGENT_NAME,
        "sandbox_profile": "off",
        "reasoning_effort": "high",
        "last_turn_summary": "Fixed the discount/tax ordering bug in checkout_total; pytest green.",
        "last_turn_summary_prompt_id": "demo0000-0000-0000-0000-000000000001",
    }
    (session_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    signals = {
        "turnCount": 1, "userMessageCount": 1, "assistantMessageCount":
            len([c for c in chat if c.get("type") == "assistant"]),
        "errorCount": 1, "toolFailureCount": 1, "cancellationCount": 0,
        "toolCallCount": len(BEATS), "toolsUsed": sorted({b["tool"] for b in BEATS}),
        "skillCallCount": 1, "skillsUsed": ["checkout-debugging"],
        "modelsUsed": [MODEL_ID], "primaryModelId": MODEL_ID,
        "sessionDurationSeconds": final_clock // 1000,
        "contextWindowTokens": 500000, "contextTokensUsed": 18000,
        "contextWindowUsage": 0.036,
        "gitCommitCount": 0, "positiveRatings": 0, "negativeRatings": 0,
    }
    (session_dir / "signals.json").write_text(json.dumps(signals))

    prompt_context = {
        "version": 1, "prompt_mode": "extend", "audience": "primary",
        "agents_md_files": [], "persona_summaries": [],
        "build_timestamp_utc": ts_at(final_clock),
        "memory_enabled": False, "os_name": "macos", "shell_path": "/bin/zsh",
        "working_directory": str(REPO_DIR),
        "current_date": START.date().isoformat(),
        "is_non_interactive": False, "system_prompt_label": "Grok 4.5",
    }
    (session_dir / "prompt_context.json").write_text(json.dumps(prompt_context, indent=2))

    (session_dir / "system_prompt.txt").write_text(
        "You are Grok 4.5 released by xAI. You are an interactive CLI tool "
        "that helps users with software engineering tasks.\n")

    print(f"wrote session {SESSION_ID} to {session_dir}")
    return session_dir


if __name__ == "__main__":
    write_all()
