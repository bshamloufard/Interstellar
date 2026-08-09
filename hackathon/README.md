# Hackathon extras (Interstellar fork)

## Session analysis (`/interstellar`)

After `git pull`, this tree includes:

- **Skill** (auto-loaded as a **project skill** when your workspace is this repo):  
  `.grok/skills/interstellar/SKILL.md`
- **CLI + normalizer + visualizer**:  
  `hackathon/session-analysis/`

### Use

```bash
# from repo root (or via sibling grok-hack launcher)
export PATH="$PWD/hackathon/session-analysis/bin:$PATH"
export GROK_HOME="${GROK_HOME:-$HOME/.grok-hackathon}"

interstellar                 # current/latest session + UI
interstellar <session-id>    # by UUID
```

In Grok (workspace = this repo): **`/interstellar`** or **`/interstellar <session-id>`**.

Restart the Grok session after pull so the skill is picked up.
