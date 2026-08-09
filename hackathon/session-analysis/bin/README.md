# interstellar CLI

| Command | Behavior |
|---------|----------|
| **`interstellar`** | Normalize **current** (or given) session **and** open the visualizer |
| **`interstellar <session-id>`** | Same, for that session UUID |
| **`interstellar-normalize [session-id]`** | Package only; prints path |
| **`interstellar-visualize [session-id]`** | **Always normalizes first** (for a session id), then serves UI |

## Session id (not paths)

Pass the session **UUID** (folder name under `$GROK_HOME/sessions/.../<id>/`), e.g.:

```text
019fe3d9-f0be-7650-bc85-2c24366ffb86
```

**Default** when omitted:

1. `$GROK_SESSION_ID` if set  
2. Newest session for the **current workspace cwd**  
3. Newest session under `$GROK_HOME/sessions`

Full session directory paths still work if you pass one.

## Examples

```bash
export PATH="…/grokathon/session-analysis/bin:$PATH"
export GROK_HOME=~/.grok-hackathon   # grok-hack default

interstellar
interstellar 019fe3d9-f0be-7650-bc85-2c24366ffb86

interstellar-normalize
interstellar-visualize 019fe3d9-f0be-7650-bc85-2c24366ffb86 --port 8765
```

`grok-hack` puts this `bin/` on `PATH` automatically.

In-session: **`/interstellar`** or **`/interstellar <session-id>`**.
