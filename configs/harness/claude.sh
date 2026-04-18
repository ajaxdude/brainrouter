#!/usr/bin/env bash
set -euo pipefail
claude mcp add-json brainrouter '{"type":"stdio","command":"/path/to/brainrouter","args":["mcp","--socket","/run/user/$UID/brainrouter.sock"]}' --scope user
if [ "${1:-}" = "--shell-rc" ]; then
  RC="${HOME}/.zshrc"
  grep -qxF 'export ANTHROPIC_BASE_URL=http://127.0.0.1:9099' "$RC" || echo 'export ANTHROPIC_BASE_URL=http://127.0.0.1:9099' >> "$RC"
  grep -qxF 'export ANTHROPIC_AUTH_TOKEN=not-used' "$RC" || echo 'export ANTHROPIC_AUTH_TOKEN=not-used' >> "$RC"
  echo "Shell rc updated: $RC"
fi
echo "Done. Restart Claude Code for changes to take effect."
