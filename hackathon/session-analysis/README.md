# session-analysis

Normalize Grok Build sessions → `session_analysis/1.0` packages + local HTML visualizer.

## Commands (preferred)

On `PATH` via **`grok-hack`**, or:

```bash
export PATH="$HOME/Desktop/Coding projects/grokathon/session-analysis/bin:$PATH"
export GROK_HOME=~/.grok-hackathon
```

| Command | What it does |
|---------|----------------|
| `interstellar` | Normalize **current** session + open UI |
| `interstellar <session-id>` | Same for that UUID |
| `interstellar-normalize [session-id]` | Write package only |
| `interstellar-visualize [session-id]` | Normalize first, then UI |

**Session id** = UUID directory name (not a full path).  
**Default** = current workspace session → else newest under `$GROK_HOME`.

In grok-hack chat: **`/interstellar`** or **`/interstellar <session-id>`**.

## Package output

`session-analysis/packages/<session-id>/`

## Low-level

```bash
python -m sa normalize <session_dir> -o packages/<id>
python -m sa serve packages/<id>
```

Schemas: `../session-analysis-schema/`.
