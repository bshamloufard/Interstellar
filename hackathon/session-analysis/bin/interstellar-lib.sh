#!/usr/bin/env bash
# Shared helpers for interstellar CLI family.
# shellcheck disable=SC2034

_interstellar_root() {
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$here/.." && pwd
}

INTERSTELLAR_SA_ROOT="${INTERSTELLAR_SA_ROOT:-$(_interstellar_root)}"
INTERSTELLAR_PACKAGES_DIR="${INTERSTELLAR_PACKAGES_DIR:-$INTERSTELLAR_SA_ROOT/packages}"
GROK_HOME="${GROK_HOME:-$HOME/.grok-hackathon}"

_interstellar_python() {
  if [[ -n "${INTERSTELLAR_PYTHON:-}" && -x "${INTERSTELLAR_PYTHON}" ]]; then
    printf '%s\n' "$INTERSTELLAR_PYTHON"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  echo "error: python3 not found" >&2
  return 1
}

_interstellar_run_sa() {
  local py
  py="$(_interstellar_python)" || return 1
  PYTHONPATH="$INTERSTELLAR_SA_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$py" -m sa "$@"
}

# Resolve session_id / path / empty → absolute session directory.
# Priority when arg empty:
#   1) $GROK_SESSION_ID
#   2) newest session for $PWD under GROK_HOME/sessions
#   3) newest session anywhere under GROK_HOME/sessions
_interstellar_resolve_session() {
  local arg="${1:-}"
  local py
  py="$(_interstellar_python)" || return 1

  GROK_HOME="$GROK_HOME" \
  GROK_SESSION_ID="${GROK_SESSION_ID:-}" \
  PWD_NOW="$(pwd -P 2>/dev/null || pwd)" \
  "$py" - "$arg" <<'PY'
import os, sys
from pathlib import Path
from urllib.parse import quote

arg = sys.argv[1] if len(sys.argv) > 1 else ""
grok_home = Path(os.environ.get("GROK_HOME", Path.home() / ".grok-hackathon"))
sessions = grok_home / "sessions"
env_sid = (os.environ.get("GROK_SESSION_ID") or "").strip()
cwd = os.environ.get("PWD_NOW") or os.getcwd()


def is_session_dir(p: Path) -> bool:
    return p.is_dir() and ((p / "summary.json").is_file() or (p / "events.jsonl").is_file())


def newest_under(root: Path):
    best, best_m = None, -1.0
    if not root.is_dir():
        return None
    for p in root.rglob("summary.json"):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m >= best_m:
            best_m = m
            best = p.parent
    # also allow events-only sessions
    if best is None:
        for p in root.rglob("events.jsonl"):
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if m >= best_m:
                best_m = m
                best = p.parent
    return best


def find_by_session_id(sid: str):
    """Match directory basename == sid (UUIDv7), anywhere under sessions/."""
    sid = sid.strip().rstrip("/")
    if not sid:
        return None
    # Direct path still allowed
    p = Path(sid).expanduser()
    if is_session_dir(p):
        return p.resolve()
    if not sessions.is_dir():
        return None
    # Exact basename match
    hits = []
    for p in sessions.rglob("*"):
        if p.is_dir() and p.name == sid and is_session_dir(p):
            try:
                m = (p / "summary.json").stat().st_mtime if (p / "summary.json").is_file() else p.stat().st_mtime
            except OSError:
                m = 0
            hits.append((m, p))
    if not hits:
        # prefix match if unique
        for p in sessions.rglob("*"):
            if p.is_dir() and p.name.startswith(sid) and is_session_dir(p):
                try:
                    m = (p / "summary.json").stat().st_mtime if (p / "summary.json").is_file() else p.stat().st_mtime
                except OSError:
                    m = 0
                hits.append((m, p))
    if not hits:
        return None
    hits.sort(key=lambda x: x[0], reverse=True)
    if len(sid) < 36 and len(hits) > 1:
        # ambiguous short id
        names = ", ".join(h[1].name for h in hits[:5])
        print(f"error: session id prefix {sid!r} is ambiguous: {names}", file=sys.stderr)
        sys.exit(2)
    return hits[0][1].resolve()


def current_workspace_session():
    """Newest session for this cwd (Grok encodes cwd in the parent folder name)."""
    if not sessions.is_dir():
        return None
    enc = quote(str(Path(cwd).resolve()), safe="")
    # exact encoded folder
    bucket = sessions / enc
    if bucket.is_dir():
        hit = newest_under(bucket)
        if hit:
            return hit
    # fuzzy: folder name contains encoded path or raw path fragments
    candidates = []
    for child in sessions.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name == enc or cwd.replace("/", "%2F") in name or enc in name:
            hit = newest_under(child)
            if hit:
                candidates.append(hit)
    if not candidates:
        return None
    # newest among matches
    best, best_m = None, -1.0
    for p in candidates:
        try:
            m = (p / "summary.json").stat().st_mtime if (p / "summary.json").is_file() else p.stat().st_mtime
        except OSError:
            m = 0
        if m >= best_m:
            best_m, best = m, p
    return best


# --- resolve ---
found = None
if arg:
    found = find_by_session_id(arg)
    if found is None:
        print(f"error: could not find session for {arg!r}", file=sys.stderr)
        print(f"  looked under {sessions}", file=sys.stderr)
        print("  pass a session id (folder name / UUID) or a full session directory path", file=sys.stderr)
        sys.exit(1)
else:
    if env_sid:
        found = find_by_session_id(env_sid)
    if found is None:
        found = current_workspace_session()
    if found is None:
        found = newest_under(sessions)
    if found is None:
        print(f"error: no session found under {sessions}", file=sys.stderr)
        print("  run a grok-hack session first, or pass a session id", file=sys.stderr)
        sys.exit(1)

print(found)
PY
}

_interstellar_session_id() {
  basename "$1"
}

_interstellar_default_package_out() {
  local session_dir="$1"
  local sid
  sid="$(_interstellar_session_id "$session_dir")"
  mkdir -p "$INTERSTELLAR_PACKAGES_DIR"
  printf '%s\n' "$INTERSTELLAR_PACKAGES_DIR/$sid"
}

_interstellar_is_package() {
  [[ -d "${1:-}" && -f "${1}/manifest.json" ]]
}

# Normalize session_dir → package_dir (prints package path on stdout last line via echo to stderr for logs)
_interstellar_normalize_to() {
  local session_dir="$1"
  local out_dir="$2"
  echo "normalizing:" >&2
  echo "  session  $session_dir" >&2
  echo "  package  $out_dir" >&2
  _interstellar_run_sa normalize "$session_dir" -o "$out_dir" >&2
  printf '%s\n' "$out_dir"
}
