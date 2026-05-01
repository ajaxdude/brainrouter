#!/usr/bin/env bash
# Install git hooks from scripts/hooks/ into .git/hooks/.
# Run once after cloning: bash scripts/install-hooks.sh

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_SRC="$REPO_ROOT/scripts/hooks"
HOOKS_DST="$REPO_ROOT/.git/hooks"

if [[ ! -d "$HOOKS_SRC" ]]; then
    echo "install-hooks: $HOOKS_SRC not found" >&2
    exit 1
fi

installed=0
for hook in "$HOOKS_SRC"/*; do
    name="$(basename "$hook")"
    dst="$HOOKS_DST/$name"
    if [[ -f "$dst" && ! -L "$dst" ]]; then
        echo "install-hooks: backing up existing $name → $name.bak"
        mv "$dst" "$dst.bak"
    fi
    ln -sf "$hook" "$dst"
    chmod +x "$hook"
    echo "install-hooks: installed $name"
    (( installed++ )) || true
done

echo "install-hooks: $installed hook(s) installed"
