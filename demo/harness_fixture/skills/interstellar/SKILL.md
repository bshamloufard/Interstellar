---
name: interstellar
description: Normalize the current (or given) Grok session and open the session-analysis visualizer.
---

# Interstellar session analysis

## When to use

User runs `/interstellar`, `/interstellar <session-id>`, or asks to analyze/visualize a session.

## Setup

CLIs live in the Interstellar repo, not in this workspace:

```text
/Users/sentorcc/Interstellar/hackathon/session-analysis/bin/interstellar
/Users/sentorcc/Interstellar/hackathon/session-analysis/bin/interstellar-normalize
/Users/sentorcc/Interstellar/hackathon/session-analysis/bin/interstellar-visualize
```

Put them on `PATH` — this workspace is not the Interstellar repo itself, so
`git rev-parse --show-toplevel` here would resolve to the wrong git root:

```bash
REPO_ROOT="/Users/sentorcc/Interstellar"
export PATH="$REPO_ROOT/hackathon/session-analysis/bin:$PATH"
export INTERSTELLAR_SA_ROOT="$REPO_ROOT/hackathon/session-analysis"
export GROK_HOME="${GROK_HOME:-$HOME/.grok-hackathon}"
command -v interstellar
```

## Commands

| Command | Behavior |
|---------|----------|
| `interstellar` | Normalize **current** session + open UI |
| `interstellar <session-id>` | Find by UUID, normalize + open UI |
| `interstellar-normalize [session-id]` | Package only |
| `interstellar-visualize [session-id]` | Normalize first, then UI |

Session id = UUID under `$GROK_HOME/sessions`. Default = current workspace session, else newest.

## What you should do

1. Put CLIs on `PATH` as above if needed.
2. Run **in the background** (long-lived server):
   ```bash
   interstellar --port 8765
   # or:
   interstellar <session-id> --port 8765
   ```
3. Tell the user:
   - **session-id**
   - **package dir**: `/Users/sentorcc/Interstellar/hackathon/session-analysis/packages/<session-id>/` (or `$INTERSTELLAR_PACKAGES_DIR`)
   - **URL**: `http://127.0.0.1:8765/`
   - Re-run: `interstellar-visualize <session-id>`

## Notes

- This copy is installed into the demo's isolated `GROK_HOME` as a user-level
  skill (rather than relying on the project-scoped copy at
  `/Users/sentorcc/Interstellar/.grok/skills/interstellar/`) because this
  workspace is a different repo than Interstellar itself.
- No extra pip deps (Python 3 stdlib).
