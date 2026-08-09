# Interstellar product demo — design spec

Date: 2026-08-08

## Goal

Produce a video-ready, fully-loadable-in-product demo of the Interstellar review-loop:
a "before" agent session that was quietly wrecked by realistic-but-exaggerated bad
skills/MCP config, followed by the real Interstellar pipeline (visualizer → analyzer →
sandboxed replay → graded report) finding and fixing it, with measured before/after
numbers. The presenter drives the whole thing live inside grok-build — open the
fabricated session from `/dashboard`, run `/interstellar` to visualize it, then prompt
the agent to run the review-loop, which spends real API calls doing real work.

**My job before handoff:** build every piece for real — demo repo, harness, and the
fabricated "already-completed" session — and verify it with the free/cheap checks
(local `pytest`, session load, normalize, one `--dry-run` analyzer pass). I do **not**
run the full live `--k 5` sandbox cycle myself; that costs real spend and is the
moment the presenter triggers live, for real, during the recording.

**Non-goal:** I do not capture video. The deliverable is the environment plus a shot
list; the presenter records their own screen.

## Honesty boundary — what's fabricated vs. real

Only one thing is fabricated: the raw session files for the "before" run
(`summary.json`, `events.jsonl`, `chat_history.jsonl`, `updates.jsonl`, `signals.json`,
`prompt_context.json`, `system_prompt.txt`), written directly in grok-build's on-disk
session format. Everything downstream that touches them — `/dashboard`, `/resume`, the
welcome screen, the `/interstellar` visualizer, `normalizer/grok_normalize.py`,
`analyzer/analyze.py`, and the full `interstellar review` sandboxed replay/grade/report
cycle — is real code doing real work, including live model calls at every stage after
the fabrication. The presenter never has to caveat any of this on camera; it's a
genuine past conversation as far as the product is concerned, and everything it
triggers from there is a genuine live computation.

## Components

### 1. Demo repo — `~/interstellar-demo/checkout/`

A small, self-contained, git-initialized Python project (outside this repo, so it
can't collide with review-loop `--out` nesting and doesn't pollute product history).
Not the thin 11-line `synth/workspace` — a slightly richer fixture:

- `pricing.py` — `apply_discount`, `apply_tax`, `checkout_total` — the bug: discount
  is applied *after* tax instead of before, so discounted orders overcharge tax.
- `cart.py` — small `Cart`/line-item model `pricing.py` depends on.
- `test_pricing.py` — one passing test (no discount), one failing test (with a
  discount code) that pins the bug.
- `README.md` — a couple of lines, enough to look like a real tiny project.

Task prompt (used both in the fabricated session and as the real prompt the
review-loop replays): *"Checkout totals are wrong on orders with a discount code —
tax looks like it's being charged on the pre-discount amount. Find the bug and fix it.
`pytest` should pass."* Short, names no fixture explicitly (unlike the corpus
scenarios), so the misdirection in the skill has room to actually mislead.

Minimal correct path: read `pricing.py`, spot the order-of-operations bug, fix it, run
`pytest` — roughly 6-8 tool calls, well under a minute.

### 2. Isolated demo harness — `GROK_HOME=~/.grok-interstellar-demo`

Used for the *entire* recording (TUI session, `/dashboard`, `/interstellar`, and the
review-loop's `--grok-home`). Never touches the presenter's real `~/.grok`.

- `skills/checkout-debugging/SKILL.md` — bloated (repeated filler sections, same rot
  pattern `synth/real_skills.py`/`make_skills.py` already use for `strict-audit`) *and*
  misdirecting: opens by confidently pointing at currency/locale formatting as the
  usual cause of checkout total bugs and instructing a detour through a `locale.py`
  that doesn't exist, before ever mentioning discount/tax order. Also instructs
  "re-read a file immediately after every edit, then again before running tests" —
  the redundant-tool-call driver. It still leads to the correct fix eventually
  (per the brief: wrong path first, right answer, just slow).
- `config.toml` — MCP servers modeled on `synth/mcp_servers.toml`'s existing verified
  roles: `github` (intentional auth-fail → broken MCP, wasted handshake every
  startup), `everything` (has a genuinely slow tool → latency finding),
  `filesystem`/`memory`/`sequential-thinking`/`time` connected but never called during
  this task → idle fleet.
- `auth.json` — one-time copy of the presenter's real `~/.grok/auth.json`, the same
  pattern `harness.py`'s `materialize()` already uses, so real grok-dev calls
  authenticate.

Stacking these four (misdirecting/bloated skill, broken MCP, slow MCP, idle fleet,
plus the redundant-read instruction) is designed to produce analyzer findings across
`skill_truncate`, `mcp_fix_broken`, `mcp_latency`, `mcp_remove`, and `tool_redundant` —
most of the recommendation surface in one report.

### 3. Fabricated "before" session

A generator script (single source-of-truth list of narrative "beats": turn text, tool
calls with names/args/results/timings, skill loads, MCP connect/fail events) emits the
five raw session files to
`~/.grok-interstellar-demo/sessions/<url-encoded-checkout-repo-path>/<uuid>/`, matching
the exact schema reverse-engineered from a real local session (events.jsonl event
types, chat_history.jsonl message shapes, updates.jsonl ACP stream, summary.json /
signals.json / prompt_context.json fields). Internal consistency (matching
`tool_call_id`s across all three conversational files, plausible monotonic timestamps)
is enforced by generating all files from the one beat list rather than hand-writing
each file separately.

Target shape of the fabricated session: agent loads the bloated skill, follows the
locale red herring for a few tool calls (reads/searches for the nonexistent file,
comes up empty), pays MCP startup for the broken + idle servers, calls the slow
`everything` tool once (contrived "verify timing" justification from the skill),
re-reads files redundantly after edits per the skill's instruction, eventually finds
and fixes the real bug in `pricing.py`, and finishes with `pytest` green. Roughly
35-50 tool-call-level events, several minutes of fabricated wall time — a dramatic
(~6-8x) contrast against the minimal correct path.

### 4. Pre-handoff verification (cheap/free only — I execute this)

1. Normalize the fabricated session with the real normalizer → canonical `trace.json`.
2. Sanity-check it: `python3 -m interstellar review trace.json --dry-run` and eyeball
   the digest for the expected findings (this is the one step that spends real money —
   a single cheap analyzer call, roughly $0.05, no sandbox replay).
3. Confirm the `--dry-run` output selects patches covering the intended finding types
   (`skill_truncate` on the bloated/misdirecting skill, `mcp_fix_broken` on `github`,
   `mcp_latency` on `everything`, `mcp_remove` on the idle fleet, `tool_redundant` on
   the re-read pattern) at reasonable confidence, without inventing anything the digest
   doesn't back.
4. Minimal-path sanity, entirely local/free: confirm the *correct* fix to `pricing.py`
   and that `pytest` goes green in a clean copy of the demo repo, so the "ideal"
   comparison point is real, not assumed.

**What I deliberately do not run:** the full `python3 -m interstellar review ... --k 5
--serve` sandboxed cycle. That's a real, several-minutes, ~$1-3 live replay — it's the
"agent runs again" moment, and it belongs to the recording, triggered by the presenter
prompting the agent live. Everything I build should make that live run land well the
first time, without me having spent the money to pre-confirm the final scorecard.

### 5. Live trigger mechanism (what the presenter does)

- Launch grok-build with `GROK_HOME=~/.grok-interstellar-demo`, cwd =
  `~/interstellar-demo/checkout/`.
- `/dashboard` (or `/resume`) → the fabricated session appears as a real past
  conversation. Open it, scroll the waste.
- `/interstellar` → real visualizer opens on it.
- Prompt the (same or a fresh) agent in natural language to run the review-loop. I'll
  hand the presenter an exact prompt string that names the trace path and the
  `interstellar` package location explicitly, so the live agent doesn't have to guess
  and reliably reaches for `run_terminal_command` with the right invocation. The agent
  turn that follows is genuinely doing the analysis + sandbox replay live.

### 6. Shot list

A short beat-by-beat recording script (what to open, what to type, what to point at,
in what order, with rough timings) delivered alongside the built environment.

## Verification plan

- Raw session parses: `grok sessions list` (with the demo `GROK_HOME`/cwd) shows the
  fabricated session; `interstellar-normalize <id>` succeeds with no errors.
- Digest sanity: normalized trace's `metrics` show the expected shape (skill
  unreferenced/wasted tokens, `failed_mcp_servers` includes `github`,
  `idle_mcp_servers` includes the unused fleet, `slowest_measured` surfaces the
  `everything` call, `duplicate_calls` shows the redundant re-reads).
- `--dry-run` analyzer pass produces recommendations covering the intended finding
  types, at `high`/`medium` confidence, without inventing anything not backed by the
  digest (matches `analyzer/prompt.md`'s evidence rules).
- One full `--k 5` live cycle completes, `index.html` renders, and the badge/verdict is
  favorable and not `NO DATA`.
- Minimal-path sanity: manually confirm the *correct* fix to `pricing.py` and that
  `pytest` goes green in a clean copy of the demo repo (so the "ideal" comparison
  point is real, not assumed).

## Risks

- **Live model variance in the control arm.** The real replay's control runs may not
  reproduce the fabricated session's exact wastefulness — mitigated by recommending
  `k=5` in the shot list (not the CLI default 3, so the gate can reach a real
  `ACCEPTED`/`REJECTED` instead of only `DIRECTIONAL`) and by making the harness's
  badness structural (broken/idle MCPs and skill bloat cost the same regardless of
  model choices), not solely behavioral.
- **I never see the full live cycle before handoff, so I can't empirically confirm the
  final scorecard.** Mitigated by erring generous on severity/margin in the harness
  (each issue costs clearly more than the analyzer's materiality floors in
  `analyzer/prompt.md` — e.g. skill waste well above the 150-token floor, MCP latency
  well above 2000ms) and by the `--dry-run` digest check, which does run for real and
  catches a digest that's too thin before any live spend happens.
- **Agent doesn't reach for the right CLI invocation unprompted.** Mitigated by
  handing the presenter an exact, explicit trigger prompt rather than relying on
  discovery.

## Deliverables checklist

1. `~/interstellar-demo/checkout/` demo repo, bug verified real, correct fix verified.
2. `~/.grok-interstellar-demo/` harness (bloated/misdirecting skill, MCP config,
   auth.json).
3. Session-generator script + the fabricated session it produces, verified to load
   and normalize cleanly, with a `--dry-run` analyzer pass confirming the intended
   findings surface.
4. Shot list + the exact live trigger prompt for the presenter (the full `--k 5`
   review-loop run happens live, during the recording, not pre-run by me).
