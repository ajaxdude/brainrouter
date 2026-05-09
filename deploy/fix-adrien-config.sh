#!/usr/bin/env bash
# deploy/fix-adrien-config.sh — fix adrien's brainrouter config and MCP settings
#
# Run as:  sudo bash deploy/fix-adrien-config.sh
#
# What it fixes:
#   1.  brainrouter.yaml: omp_path → /home/adrien/.bun/bin/omp
#   2.  brainrouter.yaml: work_dir → /home/adrien
#   3.  brainrouter.yaml: fallback_model → qwen3.6-35b-a3b (Qwen3.6 35B A3B Q6_K)
#   4.  brainrouter.yaml: bridge.signal disabled (adrien has no Signal account)
#   5.  brainrouter.yaml: add models: section (shared /mnt/models, read-only)
#   6.  mcp.json: timeout → 2100000 (35 min, matches papa's MCP fix)
#   7.  /usr/local/bin/brainrouter: update to current build
#   8.  Restart adrien's brainrouter service
#   9.  Add adrien to aistack group (for future /opt/models write access)

set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "Run with sudo: sudo bash $0"; exit 1; }

TARGET=adrien
TARGET_HOME=$(getent passwd $TARGET | cut -d: -f6)
TARGET_UID=$(id -u $TARGET)
TARGET_GID=$(id -g $TARGET)
BR_SRC="/home/papa/ai/projects/brainrouter"

log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok\033[0m %s\n' "$*"; }
skip() { printf '\033[1;36m --\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m !!\033[0m %s\n' "$*" >&2; }

# ── 1–5. brainrouter.yaml ──────────────────────────────────────────────────────
log "1-5. Rewrite $TARGET_HOME/.config/brainrouter/brainrouter.yaml"

CONFIG="$TARGET_HOME/.config/brainrouter/brainrouter.yaml"
install -d -o $TARGET -g $TARGET_GID -m 750 "$TARGET_HOME/.config/brainrouter"

cat > "$CONFIG" <<YAML
# brainrouter.yaml — $TARGET
#
# Flow:
#   model=auto  → Bonsai classifies → Manifest (cloud) OR llama-swap (local)
#   model=local → skip Bonsai, rewrite system prompt, go to llama-swap
#   model=cloud → skip Bonsai, go straight to Manifest
#   On Manifest fail → fall back to llama-swap with fallback_model

manifest:
  base_url: "http://localhost:3001/v1"
  api_key_env: MANIFEST_API_KEY

llama_swap:
  # Shared llama-swap instance (papa's, running on :8081)
  base_url: "http://localhost:8081/v1"
  # Default local model: Qwen3.6 35B A3B Q6_K — best quality/VRAM for coding
  fallback_model: "qwen3.6-35b-a3b"

bonsai:
  # Shared model — /mnt/models is traverse-accessible by all users
  model_path: "/mnt/models/prism/prism-ml_Bonsai-8B-unpacked-Q6_K_L.gguf"

review:
  # Route code reviews to cloud — avoid spinning up local model for reviews
  forced_mode: cloud

models:
  # Shared read-only model store mounted at /mnt/models (papa owns, 0701 traverse)
  path: "/mnt/models"
  shared_write: false

bridge:
  omp_path: "/home/$TARGET/.bun/bin/omp"
  work_dir: "/home/$TARGET"
  discord:
    enabled: true
    # Token copied from papa's brainrouter.yaml at setup time
    token: "$(python3 -c "import yaml,sys; d=yaml.safe_load(open('/home/papa/ai/projects/brainrouter/brainrouter.yaml')); print(d.get('bridge',{}).get('discord',{}).get('token','MISSING'))" 2>/dev/null || echo 'MISSING')"
  # signal: disabled — adrien has no Signal account
YAML

chown $TARGET:$TARGET_GID "$CONFIG"
chmod 640 "$CONFIG"
ok "brainrouter.yaml written"

# ── 6. mcp.json timeout ────────────────────────────────────────────────────────
log "6. mcp.json timeout (300000 → 2100000)"

MCP_JSON="$TARGET_HOME/.omp/agent/mcp.json"
if [[ -f "$MCP_JSON" ]]; then
    python3 -c "
import json, sys
data = json.load(open('$MCP_JSON'))
br = data.get('mcpServers', {}).get('brainrouter', {})
old = br.get('timeout', 0)
br['timeout'] = 2100000
open('$MCP_JSON', 'w').write(json.dumps(data, indent=2) + '\n')
print(f'  timeout: {old} → 2100000')
"
    chown $TARGET:$TARGET_GID "$MCP_JSON"
    ok "mcp.json timeout updated"
else
    warn "mcp.json not found at $MCP_JSON — skipping"
fi

# ── 7. Update brainrouter binary ───────────────────────────────────────────────
log "7. Update /usr/local/bin/brainrouter"

NEW_BIN="$BR_SRC/target/release/brainrouter"
INSTALLED="/usr/local/bin/brainrouter"

if [[ ! -f "$NEW_BIN" ]]; then
    warn "Release binary not found at $NEW_BIN — build with: cargo build --release"
else
    NEW_MTIME=$(stat -c %Y "$NEW_BIN")
    INST_MTIME=$(stat -c %Y "$INSTALLED" 2>/dev/null || echo 0)
    if [[ "$NEW_MTIME" -gt "$INST_MTIME" ]]; then
        install -m 755 "$NEW_BIN" "$INSTALLED"
        ok "brainrouter binary updated ($(date -d @$NEW_MTIME '+%Y-%m-%d %H:%M'))"
    else
        skip "installed binary is already current"
    fi
fi

# ── 8. Restart adrien's brainrouter ───────────────────────────────────────────
log "8. Restart brainrouter for $TARGET"

systemctl --machine=${TARGET}@ --user daemon-reload 2>/dev/null || true
if systemctl --machine=${TARGET}@ --user restart brainrouter 2>/dev/null; then
    ok "brainrouter restarted"
else
    warn "systemctl restart failed — trying kill+wait approach"
    # Find adrien's brainrouter PID and send SIGTERM
    BR_PID=$(pgrep -u $TARGET -x brainrouter 2>/dev/null | head -1 || true)
    if [[ -n "$BR_PID" ]]; then
        kill "$BR_PID"
        sleep 2
        ok "Sent SIGTERM to brainrouter PID $BR_PID — systemd will restart it"
    else
        warn "No running brainrouter for $TARGET found"
    fi
fi

# ── 9. aistack group ──────────────────────────────────────────────────────────
log "9. aistack group membership"

if getent group aistack &>/dev/null; then
    if id -nG $TARGET | grep -qw aistack; then
        skip "$TARGET already in aistack group"
    else
        usermod -aG aistack $TARGET
        ok "$TARGET added to aistack group"
    fi
else
    skip "aistack group does not exist yet (created by install.sh when models: is configured)"
fi

# ── Verify ────────────────────────────────────────────────────────────────────
log "Verify"
sleep 3

HEALTH=$(curl -sf http://127.0.0.1:9100/health 2>&1 || echo "FAIL")
echo ""
echo "  brainrouter health (:9100)   : $HEALTH"
echo "  /usr/local/bin/brainrouter   : $(stat -c '%y' /usr/local/bin/brainrouter | cut -d. -f1)"
echo "  $TARGET groups               : $(id -nG $TARGET)"
echo "  config omp_path              : $(grep omp_path $CONFIG || echo 'not found')"
echo "  config fallback_model        : $(grep fallback_model $CONFIG || echo 'not found')"
echo "  config models.path           : $(grep 'path:' $CONFIG | head -1 || echo 'not found')"
echo ""
