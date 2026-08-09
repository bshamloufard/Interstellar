# Interstellar demo — recording shot list

Prereq: `demo/install.sh` has been run, and `python3 demo/verify_demo.py --dry-run`
passed. `install.sh` is safe to re-run any time for a clean take — `SESSION_ID`
is a fixed literal in `build_session.py`, so it overwrites the same session
directory rather than creating a duplicate.

## Beat 1 — open the "before" session

```bash
cd ~/interstellar-demo/checkout
GROK_HOME=~/.grok-interstellar-demo grok-dev
```

In the TUI: **`/resume`** — the session picker for past conversations on disk,
scoped to the current workspace. (Not `/dashboard`: that lists *live* agent
sessions in this pager process's active roster, not saved history — a
fabricated session was never loaded by a live process, so it won't appear
there.) The "Checkout total wrong with discount code" session is listed like
any other past conversation. Open it. Scroll: the skill load, the `locale.py`
dead end, the 20-second diagnostics call, the re-reads, the eventual fix,
tests passing.

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
problem and proves the fix" moment. Verified live with a `--dry-run` pass: the
analyzer selects `skill.truncate` on the bloated skill (377 wasted tokens),
`mcp.remove` on the broken `github` server (9200ms wasted), and a
`rules.append` targeting the redundant re-reads — cost estimate ~$1.75 at k=5.
The idle-fleet servers (`time`/`sequential-thinking`/`memory`/`filesystem`)
show up too but rank below the default `--max-patches=3` budget; add
`--max-patches 6` to the command above if you want those in the report as well.

## Beat 4 — the scorecard

Open the served report. Point at: the badge (ACCEPTED/DIRECTIONAL), the
efficiency deltas (tokens, wall time), the patches applied (skill truncated,
github removed, the redundant-read rule), and that quality held (no
regression in the paired judge).

## Reset between takes

```bash
~/Interstellar/demo/install.sh   # idempotent: repo, harness, and session all rebuild clean
```
