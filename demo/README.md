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
