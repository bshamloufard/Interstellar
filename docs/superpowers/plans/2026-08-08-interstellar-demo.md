# Interstellar product demo — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real, fully-loadable-in-product "before" grok-build session (a
checkout-bug task wrecked by a bloated/misdirecting skill and bad MCP config),
verify it end-to-end with free/cheap checks, and hand the presenter a shot list —
so their live recording (`/dashboard` → open session → `/interstellar` →
prompt the agent to run the review-loop) is a clean repeat of a known-good setup.

**Architecture:** Two fixture bundles versioned in this repo (`demo/repo_fixture/`,
a buggy checkout library; `demo/harness_fixture/`, a bloated/misdirecting skill +
MCP config) get materialized by `demo/install.sh` into `~/interstellar-demo/checkout/`
and an isolated `~/.grok-interstellar-demo/` GROK_HOME. A generator
(`demo/build_session.py`) then fabricates a real grok-build session — five raw
files in grok-build's exact on-disk schema — under that GROK_HOME's `sessions/`
tree, driven by one ordered list of narrative "beats." `demo/verify_demo.py` runs
the real normalizer and one real (cheap) analyzer `--dry-run` pass to confirm the
scenario actually produces the intended findings before handoff.

**Tech Stack:** Python 3 stdlib only (matches this repo's `interstellar/`/`normalizer/`
convention — no pip installs). Bash for orchestration (`install.sh`), matching
`synth/`'s existing install/clean pattern.

## Global Constraints

- Never write to `~/interstellar-demo/checkout/` from the generator — only read
  from it. The live review-loop later copies this repo per-arm; it must stay in
  its buggy, unedited state at rest, forever, or the later live "control" runs
  are meaningless.
- Never touch the presenter's real `~/.grok`. Everything lives under
  `~/.grok-interstellar-demo` (a distinct `GROK_HOME`), including the one-time
  `auth.json` copy.
- The fabricated session's raw files must match grok-build's real on-disk schema
  exactly enough that `normalizer/grok_normalize.py` (unmodified, this repo's real
  code) parses it without error and produces sane metrics. This is the hard
  correctness bar for this plan — schema field names below are taken directly
  from a real local session, not guessed.
- Every dollar spent making live model calls happens in `demo/verify_demo.py`'s
  one `--dry-run` step (~$0.05) — nothing else in this plan spends money.
- `x-synthetic-fixture: true` goes in the skill's frontmatter, matching
  `synth/make_skills.py`'s/`synth/real_skills.py`'s existing convention, so it's
  identifiable and never mistaken for a real user skill.

---

## File Structure

```
demo/
  README.md                                 overview + rebuild/reset instructions
  repo_fixture/
    pricing.py                              buggy checkout math (the bug)
    cart.py                                 small dependency of pricing.py
    test_pricing.py                         one passing test, one failing test
    README.md                               one-liner project readme
  harness_fixture/
    skills/checkout-debugging/SKILL.md      bloated + misdirecting skill
    config.toml                             MCP servers: 1 broken, 1 slow, 4 idle
  install.sh                                materializes both fixtures + auth.json
  build_session.py                          fabricates the "before" session
  verify_demo.py                            structural + real-normalizer + dry-run checks
  SHOT_LIST.md                              recording script + live trigger prompt
```

---

### Task 1: Demo repo fixture — the buggy checkout library

**Files:**
- Create: `demo/repo_fixture/pricing.py`
- Create: `demo/repo_fixture/cart.py`
- Create: `demo/repo_fixture/test_pricing.py`
- Create: `demo/repo_fixture/README.md`

**Interfaces:**
- Produces: a `checkout_total(subtotal_cents, discount_pct=0)` function in
  `pricing.py` with a real, demonstrable bug (tax computed on the pre-discount
  subtotal instead of the post-discount one) and a `pytest` suite that pins it.
  Task 3 reads this file's exact text (buggy, and a known-fix variant computed by
  string replacement) to build tool-result content, so the function body below is
  the frozen source of truth — don't reformat it later without re-checking Task 3.

- [ ] **Step 1: Write `cart.py`**

```python
"""Cart line items."""

from dataclasses import dataclass, field


@dataclass
class LineItem:
    name: str
    unit_price_cents: int
    quantity: int = 1


@dataclass
class Cart:
    items: list = field(default_factory=list)
    discount_pct: int = 0

    def add(self, name, unit_price_cents, quantity=1):
        self.items.append(LineItem(name, unit_price_cents, quantity))

    def subtotal_cents(self):
        return sum(item.unit_price_cents * item.quantity for item in self.items)
```

- [ ] **Step 2: Write `pricing.py` (the bug)**

```python
"""Pricing calculations for checkout."""

TAX_RATE = 0.08  # 8% sales tax


def discount_amount_cents(subtotal_cents, discount_pct):
    """Dollar amount (in cents) knocked off by a percentage discount."""
    return round(subtotal_cents * discount_pct / 100)


def tax_cents(amount_cents):
    """Sales tax owed on an amount, in cents."""
    return round(amount_cents * TAX_RATE)


def checkout_total(subtotal_cents, discount_pct=0):
    """Final charge for a cart, in cents: subtotal, minus discount, plus tax."""
    discount = discount_amount_cents(subtotal_cents, discount_pct)
    tax = tax_cents(subtotal_cents)  # BUG: taxes the pre-discount subtotal
    return subtotal_cents - discount + tax
```

- [ ] **Step 3: Write `test_pricing.py`**

```python
from cart import Cart
from pricing import checkout_total


def test_checkout_total_no_discount():
    cart = Cart()
    cart.add("Widget", 5000, quantity=2)  # $100.00 subtotal
    assert checkout_total(cart.subtotal_cents()) == 10800  # +8% tax


def test_checkout_total_with_discount():
    cart = Cart()
    cart.add("Widget", 5000, quantity=2)  # $100.00 subtotal
    cart.discount_pct = 10
    total = checkout_total(cart.subtotal_cents(), cart.discount_pct)
    assert total == 9720  # $90 discounted subtotal + 8% tax on $90 = $97.20
```

- [ ] **Step 4: Write `README.md`**

```markdown
# checkout

Small internal pricing/checkout library.

## Run tests

    pytest
```

- [ ] **Step 5: Verify the bug is real (RED) — free, local, no network**

```bash
cd /Users/sentorcc/Interstellar/demo/repo_fixture
python3 -m pytest -q
```

Expected: `test_checkout_total_no_discount` PASSES, `test_checkout_total_with_discount`
**FAILS** with `assert 9800 == 9720` (buggy code taxes the full $100 subtotal —
$8.00 tax — instead of the $90 discounted subtotal — $7.20 tax — giving $98.00
instead of $97.20).

- [ ] **Step 6: Verify the intended fix is real (GREEN), in a scratch copy — never edit `repo_fixture/` itself**

```bash
cp -r /Users/sentorcc/Interstellar/demo/repo_fixture /tmp/checkout-fix-check
cd /tmp/checkout-fix-check
python3 - <<'EOF'
import pathlib
p = pathlib.Path("pricing.py")
old = (
    "    discount = discount_amount_cents(subtotal_cents, discount_pct)\n"
    "    tax = tax_cents(subtotal_cents)  # BUG: taxes the pre-discount subtotal\n"
    "    return subtotal_cents - discount + tax"
)
new = (
    "    discount = discount_amount_cents(subtotal_cents, discount_pct)\n"
    "    discounted_subtotal = subtotal_cents - discount\n"
    "    tax = tax_cents(discounted_subtotal)\n"
    "    return discounted_subtotal + tax"
)
text = p.read_text()
assert text.count(old) == 1, "fixture drifted from Task 1 Step 2 — update `old` to match"
p.write_text(text.replace(old, new))
EOF
python3 -m pytest -q
rm -rf /tmp/checkout-fix-check
```

Expected: both tests PASS. This confirms the fix Task 3 will narrate is a real,
minimal, correct fix — not an assumption.

- [ ] **Step 7: Commit**

```bash
cd /Users/sentorcc/Interstellar
git add demo/repo_fixture/
git commit -m "Add demo repo fixture: buggy checkout pricing library"
```

---

### Task 2: Harness fixture — the bloated/misdirecting skill and MCP config

**Files:**
- Create: `demo/harness_fixture/skills/checkout-debugging/SKILL.md`
- Create: `demo/harness_fixture/config.toml`

**Interfaces:**
- Produces: a skill file whose frontmatter `name` is `checkout-debugging`, and a
  `config.toml` defining exactly these `[mcp_servers.*]` tables:
  `github` (broken — no credential), `everything` (has a slow tool, gets called
  once), `filesystem`, `memory`, `sequential-thinking`, `time` (all idle — never
  called by Task 3's beats). Task 3 reads `SKILL.md`'s real bytes/line count off
  disk at generation time, so don't rename the skill or move the file without
  updating Task 3's `SKILL_PATH`.

- [ ] **Step 1: Write `SKILL.md`**

The two `## Glossary` sections below must stay byte-for-byte identical — the
real normalizer's duplicate-section detector keys on exact text match including
the heading, and this is what produces a `repeated_line_ranges` finding.

```markdown
---
name: checkout-debugging
description: Use when a checkout, cart, or pricing calculation produces an incorrect total.
x-synthetic-fixture: true
---

# Checkout Debugging

## Historical Context

This skill was written after the Q2 pricing incident, when a rounding change in
the tax service caused checkout totals to drift by fractions of a cent across
millions of orders before anyone noticed. The team's post-mortem concluded that
`checkout_total` had grown fragile because tax, discount, and currency
formatting all lived close together conceptually but were debugged as if they
were unrelated. This skill exists to make sure whoever picks up a checkout bug
starts from the same mental model the pricing team converged on, rather than
re-deriving it turn by turn every time a report comes in.

## Rationale and Philosophy

Checkout bugs are deceptively narrow-looking. A user reports "the total is
wrong," and it is tempting to jump straight into `pricing.py` and start reading
arithmetic. In practice, the pricing team's experience is that **the majority
of checkout total bugs reported by users trace back to currency and locale
formatting**, not the arithmetic itself — a total that is off by a fraction of
a cent, or that displays with the wrong number of decimal places, gets
reported by users as "the total is wrong" indistinguishably from a genuine
arithmetic bug. Treat formatting as the prior, not the arithmetic.

## Glossary

- **Subtotal** — the sum of line-item prices before discount or tax.
- **Discount** — a percentage or fixed-amount reduction applied to a
  subtotal.
- **Tax** — a percentage added on top of a (discounted or undiscounted)
  amount, depending on jurisdiction rules.
- **Total** — the final charge presented to the customer.
- **Line item** — a single product/quantity/price triple within a cart.

## Procedure

1. Start with `locale.py` at the project root — check the currency formatting
   table it maintains and confirm it is current for the locales named in the
   bug report. A stale formatting table is the most common root cause seen in
   practice, so rule it out before touching arithmetic.
2. If `locale.py` looks fine or is not present in this checkout, move to the
   pricing module and read through the discount/tax computation in order.
3. After making any change, re-read the file you changed to confirm the edit
   landed the way you intended, then re-read it again immediately before
   running the test suite — pricing edits are easy to fat-finger and a second
   read catches most of those before a test run wastes a cycle.
4. Before making any change, verify environment timing is sane using the
   diagnostics tool — a slow sandbox can make an unrelated test flake look
   like a pricing bug, and it's cheap to rule out first.
5. Run the project's test suite and confirm green before considering the bug
   fixed.

## Team Conventions (deprecated)

*The conventions in this section predate the 2025 style guide and are kept
for historical reference only; the current style guide supersedes them.*
Money values were previously passed as floats rather than integer cents in
some older modules. Function names previously used a `calc_` prefix rather
than a verb-first name. Test files previously lived alongside the module
under test rather than in a top-level `tests/` directory.

## Further Reading

- The pricing team's Q2 incident retro (internal wiki, pricing-incidents
  space).
- The currency formatting RFC that introduced `locale.py` in the original
  checkout service.
- The original checkout service design doc.

## Rationale and Philosophy, continued

It bears repeating: formatting-first is the correct default triage order for
this class of bug. Engineers who skip straight to the arithmetic tend to
either miss the actual bug because it was cosmetic, or spend time convincing
themselves the arithmetic is correct (it usually is) before eventually
checking formatting anyway. Save that detour by checking formatting first,
every time, regardless of how the bug report is phrased.

## Glossary

- **Subtotal** — the sum of line-item prices before discount or tax.
- **Discount** — a percentage or fixed-amount reduction applied to a
  subtotal.
- **Tax** — a percentage added on top of a (discounted or undiscounted)
  amount, depending on jurisdiction rules.
- **Total** — the final charge presented to the customer.
- **Line item** — a single product/quantity/price triple within a cart.
```

- [ ] **Step 2: Write `config.toml`** (server bodies adapted verbatim from the
  already-verified `synth/mcp_servers.toml`, repointed at the demo repo)

```toml
[mcp_servers.filesystem]
command = "npx"
args = [
  "-y",
  "@modelcontextprotocol/server-filesystem",
  "/Users/sentorcc/interstellar-demo/checkout",
]
startup_timeout_sec = 120
tool_timeout_sec = 60

[mcp_servers.memory]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-memory"]
env = { MEMORY_FILE_PATH = "/Users/sentorcc/.grok-interstellar-demo/memory.jsonl" }
startup_timeout_sec = 120
tool_timeout_sec = 60

[mcp_servers.sequential-thinking]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-sequential-thinking"]
startup_timeout_sec = 120
tool_timeout_sec = 60

[mcp_servers.everything]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-everything"]
startup_timeout_sec = 120
tool_timeout_sec = 180
tool_timeouts = { "trigger-long-running-operation" = 300 }

[mcp_servers.time]
command = "uvx"
args = ["mcp-server-time", "--local-timezone=America/Los_Angeles"]
startup_timeout_sec = 120
tool_timeout_sec = 30

# ############################################################################
# ## INTENTIONAL FAILURE — THIS SERVER WILL NOT CONNECT.                    ##
# ## GitHub's official remote MCP server requires OAuth or a PAT. With no   ##
# ## credential it returns HTTP 401 on initialize. Short startup timeout so ##
# ## it fails fast instead of stalling session start.                      ##
# ############################################################################
[mcp_servers.github]
url = "https://api.githubcopilot.com/mcp/"
enabled = true
startup_timeout_sec = 10
tool_timeout_sec = 30
```

- [ ] **Step 3: Commit**

```bash
cd /Users/sentorcc/Interstellar
git add demo/harness_fixture/
git commit -m "Add demo harness fixture: bloated skill + broken/slow/idle MCP config"
```

---

### Task 3: Session generator — fabricate the "before" session

**Files:**
- Create: `demo/build_session.py`

**Interfaces:**
- Consumes: `demo/harness_fixture/skills/checkout-debugging/SKILL.md` and
  `demo/repo_fixture/{pricing,cart,test_pricing}.py`, but reads them from their
  **materialized** locations (`~/.grok-interstellar-demo/skills/...`,
  `~/interstellar-demo/checkout/...`) — i.e. this script must run *after* Task 4's
  `install.sh` has copied the fixtures into place, so byte/line counts match
  exactly what the real harness snapshot will later see.
- Produces: `summary.json`, `events.jsonl`, `chat_history.jsonl`, `updates.jsonl`,
  `signals.json`, `prompt_context.json`, `system_prompt.txt` written to
  `~/.grok-interstellar-demo/sessions/<url-encoded-checkout-path>/<SESSION_ID>/`.
  `SESSION_ID = "019fe500-a100-7000-8000-000000000001"` — Task 5 and `SHOT_LIST.md`
  reference this exact literal.

- [ ] **Step 1: Write `demo/build_session.py`**

```python
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
```

- [ ] **Step 2: Run it** (only after Task 4's `install.sh` has run at least once)

```bash
python3 /Users/sentorcc/Interstellar/demo/build_session.py
```

Expected: `wrote session 019fe500-a100-7000-8000-000000000001 to
/Users/sentorcc/.grok-interstellar-demo/sessions/%2FUsers%2Fsentorcc%2Finterstellar-demo%2Fcheckout/019fe500-a100-7000-8000-000000000001`,
and that directory contains all 7 files.

- [ ] **Step 3: Commit**

```bash
cd /Users/sentorcc/Interstellar
git add demo/build_session.py
git commit -m "Add session generator: fabricates the before-session from repo+harness fixtures"
```

---

### Task 4: Install script — materialize both fixtures + auth

**Files:**
- Create: `demo/install.sh`

**Interfaces:**
- Consumes: `demo/repo_fixture/`, `demo/harness_fixture/`, `demo/build_session.py`
  (Tasks 1-3), and the presenter's real `~/.grok/auth.json`.
- Produces: `~/interstellar-demo/checkout/` (git-initialized) and
  `~/.grok-interstellar-demo/` (skills, config.toml, auth.json, sessions/) fully
  materialized and ready to launch grok-build against.

- [ ] **Step 1: Write `demo/install.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$HOME/interstellar-demo/checkout"
DEMO_HOME="$HOME/.grok-interstellar-demo"

echo "== materializing checkout repo at $REPO_DIR =="
rm -rf "$REPO_DIR"
mkdir -p "$REPO_DIR"
cp "$HERE/repo_fixture/"*.py "$HERE/repo_fixture/README.md" "$REPO_DIR/"
(
  cd "$REPO_DIR"
  git init -q
  git config user.email "demo@example.invalid"
  git config user.name "Demo Setup"
  git add -A
  git commit -q -m "Initial checkout library"
)

echo "== materializing demo GROK_HOME at $DEMO_HOME =="
mkdir -p "$DEMO_HOME/skills" "$DEMO_HOME/sessions"
rm -rf "$DEMO_HOME/skills/checkout-debugging"
cp -r "$HERE/harness_fixture/skills/checkout-debugging" "$DEMO_HOME/skills/"
cp "$HERE/harness_fixture/config.toml" "$DEMO_HOME/config.toml"

if [ ! -f "$HOME/.grok/auth.json" ]; then
  echo "error: $HOME/.grok/auth.json not found — log in with the real grok CLI first" >&2
  exit 1
fi
cp "$HOME/.grok/auth.json" "$DEMO_HOME/auth.json"
chmod 600 "$DEMO_HOME/auth.json"

echo "== fabricating the before-session =="
python3 "$HERE/build_session.py"

echo "== done =="
echo "Launch with:  GROK_HOME=$DEMO_HOME grok   (cwd: $REPO_DIR)"
```

- [ ] **Step 2: Make it executable and run it**

```bash
chmod +x /Users/sentorcc/Interstellar/demo/install.sh
/Users/sentorcc/Interstellar/demo/install.sh
```

Expected: both directories exist, ends with `wrote session ... ` from Task 3 and
the `Launch with: ...` line. Verify manually:

```bash
test -f "$HOME/interstellar-demo/checkout/pricing.py" && echo "repo OK"
test -f "$HOME/.grok-interstellar-demo/config.toml" && echo "harness OK"
test -f "$HOME/.grok-interstellar-demo/auth.json" && echo "auth OK"
```

- [ ] **Step 3: Commit**

```bash
cd /Users/sentorcc/Interstellar
git add demo/install.sh
git commit -m "Add install script: materializes the demo repo, harness, and session"
```

---

### Task 5: Verification — real normalizer + real (cheap) analyzer dry-run

**Files:**
- Create: `demo/verify_demo.py`

**Interfaces:**
- Consumes: the materialized session dir from Task 3/4 (path computed the same
  way `build_session.py` computes it), this repo's real
  `normalizer/grok_normalize.py`.
- Produces: a pass/fail console report. This is the last gate before declaring
  the demo "good to go" — it spends the one real dollar this plan spends.

- [ ] **Step 1: Write `demo/verify_demo.py`**

```python
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
```

- [ ] **Step 2: Run the free checks first**

```bash
python3 /Users/sentorcc/Interstellar/demo/verify_demo.py
```

Expected: every line `[OK  ]`, ending `ALL CHECKS PASSED`. If anything is `[FAIL]`,
fix the generator (Task 3) or fixtures (Tasks 1-2) and re-run `install.sh` +
this check before proceeding — do not spend money on a broken digest.

- [ ] **Step 3: Run the one real, paid check**

```bash
python3 /Users/sentorcc/Interstellar/demo/verify_demo.py --dry-run
```

Expected: all the free checks still pass, plus the analyzer's `--dry-run` output
printed — read it and confirm it selected patches covering the bloated/duplicate
skill, the broken `github` server, the slow `everything` call, the idle fleet,
and the redundant `read_file` calls, per `analyzer/prompt.md`'s recommendation
types (`skill_truncate`, `mcp_fix_broken`, `mcp_latency`, `mcp_remove`,
`tool_redundant`). If a finding type is missing, that's a materiality-floor or
narrative-strength issue in Tasks 1-3, not a bug in this script — go back and
strengthen the relevant beat (e.g. a slower MCP call, a bigger unreferenced
skill section) and regenerate.

- [ ] **Step 4: Commit**

```bash
cd /Users/sentorcc/Interstellar
echo "demo/_verify_out/" >> .gitignore
git add demo/verify_demo.py .gitignore
git commit -m "Add demo verification: structural checks + real normalizer + dry-run analyzer"
```

---

### Task 6: Shot list — the live recording script and trigger prompt

**Files:**
- Create: `demo/SHOT_LIST.md`

**Interfaces:**
- Consumes: the fully verified demo from Tasks 1-5 (this task only writes
  documentation — no code).

- [ ] **Step 1: Write `demo/SHOT_LIST.md`**

```markdown
# Interstellar demo — recording shot list

Prereq: `demo/install.sh` has been run, and `python3 demo/verify_demo.py --dry-run`
passed. `install.sh` is safe to re-run any time for a clean take — `SESSION_ID`
is a fixed literal in `build_session.py`, so it overwrites the same session
directory rather than creating a duplicate.

## Beat 1 — open the "before" session

```bash
cd ~/interstellar-demo/checkout
GROK_HOME=~/.grok-interstellar-demo grok
```

In the TUI: `/dashboard` (or `/resume`) — the "Checkout total wrong with discount
code" session is listed like any other past conversation. Open it. Scroll: the
skill load, the `locale.py` dead end, the 20-second diagnostics call, the
re-reads, the eventual fix, tests passing.

## Beat 2 — visualize it

In the same or a fresh session in this workspace: `/interstellar`. Real
visualizer, real timeline, opens in the browser. Point at: the MCP startup
block (github failing, 4 servers connected-but-idle), the 20s span, the
duplicate `read_file` calls, the skill's unreferenced/duplicate sections.

## Beat 3 — let the agent fix it

Prompt the agent (same product, new turn) with this exact text so it reaches
for the right command without guessing:

> Run our Interstellar review-loop against this session
> (`019fe500-a100-7000-8000-000000000001`) and show me what it finds and fixes.
> The trace normalizes from this session directory; the review-loop CLI lives
> at `/Users/sentorcc/Interstellar` — run
> `python3 -m interstellar review <trace.json> --grok-home ~/.grok-interstellar-demo
> --out ~/interstellar-demo/runs/checkout --k 5 --serve`
> from there (normalize the session to a trace.json first if one doesn't exist
> yet), and open the report when it's done.

This is a real, live, several-minute, several-dollar sandboxed cycle: live
grok-dev re-runs the original prompt k=5 times against the real bad harness
(control) and a real patched harness (treatment), grades pairwise, and renders
`index.html`. Narrate while it runs — this is the actual "agent finds the
problem and proves the fix" moment.

## Beat 4 — the scorecard

Open the served report. Point at: the badge (ACCEPTED/DIRECTIONAL), the
efficiency deltas (tokens, wall time), the patches applied (skill truncated,
github removed, everything's slow call flagged, idle servers removed, the
redundant-read rule), and that quality held (no regression in the paired
judge).

## Reset between takes

```bash
~/Interstellar/demo/install.sh   # idempotent: repo, harness, and session all rebuild clean
```
```

- [ ] **Step 2: Write `demo/README.md`**

```markdown
# Interstellar demo kit

Everything needed to reproduce the "before" session used in the product demo.

- `repo_fixture/` — the buggy checkout library the fabricated session works on.
- `harness_fixture/` — the bloated/misdirecting skill and MCP config that made
  the "before" session slow.
- `install.sh` — materializes both into `~/interstellar-demo/checkout` and
  `~/.grok-interstellar-demo`, then generates the session.
- `build_session.py` — the generator; the narrative lives in its `BEATS` list.
- `verify_demo.py` — structural checks (free) + one real analyzer `--dry-run`
  pass (~$0.05). Run with `--dry-run` before any recording session.
- `SHOT_LIST.md` — the recording script.

Rebuild from scratch any time: `./install.sh && python3 verify_demo.py --dry-run`.

Design rationale: `docs/superpowers/specs/2026-08-08-interstellar-demo-design.md`.
```

- [ ] **Step 3: Commit**

```bash
cd /Users/sentorcc/Interstellar
git add demo/SHOT_LIST.md demo/README.md
git commit -m "Add demo shot list and README"
```
