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
echo "Launch with:  GROK_HOME=$DEMO_HOME grok-dev   (cwd: $REPO_DIR)"
