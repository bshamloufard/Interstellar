---
name: interstellar
description: Normalize the current (or given) Grok session and open the session-analysis visualizer.
---

# Interstellar session analysis

## When to use

User runs `/interstellar`, `/interstellar <session-id>`, or asks to analyze/visualize a session.

## Setup (automatic if launched via repo `grok-hack`)

CLIs live in this repo:

```text
hackathon/session-analysis/bin/interstellar
hackathon/session-analysis/bin/interstellar-normalize
hackathon/session-analysis/bin/interstellar-visualize
```

Ensure they are on `PATH` (the repo `grok-hack` launcher does this):

```bash
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
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
   - **package dir**: `hackathon/session-analysis/packages/<session-id>/` (or `$INTERSTELLAR_PACKAGES_DIR`)
   - **URL**: `http://127.0.0.1:8765/`
   - Re-run: `interstellar-visualize <session-id>`

## Notes

- Skill is shipped in-repo at `.grok/skills/interstellar/` — available after pull when this project is the workspace (or on the skill search path).
- No extra pip deps (Python 3 stdlib).
