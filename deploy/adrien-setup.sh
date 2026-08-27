#!/usr/bin/env bash
# deploy/adrien-setup.sh — set up brainrouter + full AI stack for user adrien
#
# Run as:  sudo bash deploy/adrien-setup.sh
#
# What it does (idempotent — safe to re-run):
#   1.  Ensure /run/user/<uid> exists with correct ownership
#   2.  Install bun to /usr/local/bin if missing
#   3.  Install brainrouter binary to /usr/local/bin if missing
#   4.  Install llama-server-toolbox wrapper to /usr/local/bin if missing
#   5.  Install brainrouter-cleanup helper to /usr/local/bin if missing
#   6.  Install oh-my-pi for adrien via bun
#   7.  Create llama-vulkan-radv toolbox container for adrien
#   8.  Write ~/.config/brainrouter/brainrouter.yaml (copy from papa)
#   9.  Write ~/.config/brainrouter/.env with Manifest API key
#   10. Write ~/.config/systemd/user/brainrouter.service
#   11. Enable linger (brainrouter starts at boot without login)
#   12. daemon-reload, enable, start brainrouter
#   13. Write ~/.omp/agent/{models.yml,mcp.json,APPEND_SYSTEM.md,config.yml}

set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "Run with sudo: sudo bash $0"; exit 1; }

TARGET=adrien
TARGET_HOME=$(getent passwd $TARGET | cut -d: -f6)
TARGET_UID=$(id -u $TARGET)
TARGET_GID=$(id -g $TARGET)

PAPA_HOME=/home/papa
BR_SRC="$PAPA_HOME/ai/projects/brainrouter"
BR_RELEASE="$BR_SRC/target/release/brainrouter"
MANIFEST_API_KEY=""  # sourced at .env-write time; never stored in this script
BRAINROUTER_PORT=9100  # papa uses 9099; each user needs a unique TCP port

log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok\033[0m %s\n' "$*"; }
skip() { printf '\033[1;36m --\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m !!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERR\033[0m %s\n' "$*" >&2; exit 1; }

run_as() {
    sudo -u $TARGET HOME="$TARGET_HOME" "$@"
}
run_as_env() {
    sudo -u $TARGET \
        HOME="$TARGET_HOME" \
        XDG_RUNTIME_DIR="/run/user/$TARGET_UID" \
        DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$TARGET_UID/bus" \
        "$@"
}

log "Setting up AI stack for $TARGET (uid=$TARGET_UID, home=$TARGET_HOME)"

# ── 1. XDG runtime dir ────────────────────────────────────────────────────────
log "1. XDG runtime dir"
if [[ ! -d /run/user/$TARGET_UID ]]; then
    mkdir -p /run/user/$TARGET_UID
    chown $TARGET:$TARGET /run/user/$TARGET_UID
    chmod 700 /run/user/$TARGET_UID
    ok "/run/user/$TARGET_UID created"
else
    skip "/run/user/$TARGET_UID already exists"
fi

# ── 2. bun ────────────────────────────────────────────────────────────────────
log "2. bun (system-wide)"
if [[ ! -x /usr/local/bin/bun ]]; then
    BUN_TMP=$(mktemp -d)
    curl -fsSL "https://bun.sh/install" | BUN_INSTALL="$BUN_TMP" bash
    install -m 755 "$BUN_TMP/bin/bun" /usr/local/bin/bun
    rm -rf "$BUN_TMP"
    ok "bun $(/usr/local/bin/bun --version) installed to /usr/local/bin/bun"
else
    skip "bun $(/usr/local/bin/bun --version) already at /usr/local/bin/bun"
fi

# ── 3. brainrouter binary ─────────────────────────────────────────────────────
log "3. brainrouter binary"
[[ -f "$BR_RELEASE" ]] || die "Built binary not found at $BR_RELEASE — run: cd $BR_SRC && cargo build --release"
if [[ ! -x /usr/local/bin/brainrouter ]] \
   || ! diff -q "$BR_RELEASE" /usr/local/bin/brainrouter &>/dev/null; then
    install -m 755 "$BR_RELEASE" /usr/local/bin/brainrouter
    ok "brainrouter installed/updated at /usr/local/bin/brainrouter"
else
    skip "brainrouter already up-to-date at /usr/local/bin/brainrouter"
fi

# ── 4. llama-server-toolbox ───────────────────────────────────────────────────
log "4. llama-server-toolbox"
TOOLBOX_WRAPPER="$PAPA_HOME/.local/bin/llama-server-toolbox"
[[ -x "$TOOLBOX_WRAPPER" ]] || die "$TOOLBOX_WRAPPER not found"
if [[ ! -x /usr/local/bin/llama-server-toolbox ]]; then
    install -m 755 "$TOOLBOX_WRAPPER" /usr/local/bin/llama-server-toolbox
    ok "llama-server-toolbox installed"
else
    skip "llama-server-toolbox already at /usr/local/bin"
fi

# ── 5. brainrouter-cleanup ────────────────────────────────────────────────────
log "5. brainrouter-cleanup"
CLEANUP_WRAPPER="$PAPA_HOME/.local/bin/brainrouter-cleanup"
[[ -x "$CLEANUP_WRAPPER" ]] || die "$CLEANUP_WRAPPER not found"
if [[ ! -x /usr/local/bin/brainrouter-cleanup ]]; then
    install -m 755 "$CLEANUP_WRAPPER" /usr/local/bin/brainrouter-cleanup
    ok "brainrouter-cleanup installed"
else
    skip "brainrouter-cleanup already at /usr/local/bin"
fi

# ── 6. oh-my-pi ───────────────────────────────────────────────────────────────
log "6. oh-my-pi"
if ! run_as /usr/local/bin/bun pm ls -g 2>/dev/null | grep -q '@oh-my-pi/pi-coding-agent'; then
    run_as /usr/local/bin/bun install -g @oh-my-pi/pi-coding-agent 2>&1 | tail -5
    ok "oh-my-pi installed for $TARGET"
else
    skip "oh-my-pi already installed for $TARGET"
fi
# Always chown ~/.bun/ to ensure ownership is correct even if a prior root-run
# corrupted it (bun writes dirs as the calling user, but sudo sometimes creates
# intermediate dirs as root, leaving pi_natives .node files unreadable/unwritable
# by adrien → errno EACCES when OMP tries to extract or load the native module).
if [[ -d "$TARGET_HOME/.bun" ]]; then
    chown -R $TARGET:$TARGET_GID "$TARGET_HOME/.bun"
    ok ".bun ownership fixed for $TARGET"
fi

# ── 7. toolbox container ──────────────────────────────────────────────────────
log "7. toolbox container (llama-vulkan-radv)"
if ! run_as_env toolbox list 2>/dev/null | grep -q 'llama-vulkan-radv'; then
    if run_as_env toolbox create \
        --image docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
        llama-vulkan-radv 2>&1; then
        ok "llama-vulkan-radv container created for $TARGET"
    else
        warn "toolbox create failed — run manually as $TARGET:"
        warn "  toolbox create --image docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv llama-vulkan-radv"
    fi
else
    skip "llama-vulkan-radv already exists for $TARGET"
fi

# ── 8. brainrouter config ────────────────────────────────────────────────────
log "8. brainrouter config"
install -d -o $TARGET -g $TARGET_GID -m 750 "$TARGET_HOME/.config/brainrouter"
CONFIG="$TARGET_HOME/.config/brainrouter/brainrouter.yaml"
if [[ ! -f "$CONFIG" ]]; then
    cat > "$CONFIG" <<BRCONFIG
# brainrouter.yaml — $TARGET (generated)

manifest:
  base_url: "http://127.0.0.1:3001/v1"
  api_key_env: MANIFEST_API_KEY

llama_swap:
  base_url: "http://127.0.0.1:8081/v1"
  fallback_model: "qwen3.6-35b-a3b"

bonsai:
  # Off by default — enable only after confirming the model file exists.
  # Requires the PrismML fork (fork_path); stock llama.cpp cannot load it.
  enabled: false
  model_path: "/mnt/models/prism/Bonsai-27B-Q1_0.gguf"
  fork_path: "/home/papa/.local/share/brainrouter/llama-prism/llama-server"
  server_port: 9200

models:
  path: "/mnt/models"
  shared_write: false

bridge:
  omp_path: "${TARGET_HOME}/.bun/bin/omp"
  work_dir: "${TARGET_HOME}"
  discord:
    enabled: false
  signal:
    enabled: false
BRCONFIG
    chown $TARGET:$TARGET_GID "$CONFIG"
    chmod 640 "$CONFIG"
    ok "brainrouter.yaml written"
else
    skip "brainrouter.yaml already exists"
fi

# ── 9. .env ───────────────────────────────────────────────────────────────────
log "9. .env (Manifest API key)"
ENV_FILE="$TARGET_HOME/.config/brainrouter/.env"
# Never store the key in this script: pull from an existing .env on the
# machine (papa's first), else prompt the admin.
_find_key() {
    local f k
    for f in /home/papa/.config/brainrouter/.env /etc/brainrouter/env; do
        [[ -r "$f" ]] || continue
        k=$(grep -m1 '^MANIFEST_API_KEY=mnfst_' "$f" 2>/dev/null | cut -d= -f2-)
        [[ -n "$k" ]] && { echo "$k"; return 0; }
    done
    return 1
}
if [[ -f "$ENV_FILE" ]] && grep -q '^MANIFEST_API_KEY=mnfst_' "$ENV_FILE"; then
    skip ".env already has a valid Manifest key"
else
    if MANIFEST_API_KEY=$(_find_key); then
        log "Manifest API key pulled from existing .env"
    else
        read -rsp "Paste MANIFEST_API_KEY (mnfst_*): " MANIFEST_API_KEY
        echo
        [[ "$MANIFEST_API_KEY" == mnfst_* ]] || { echo "ERR: key must start with mnfst_"; exit 1; }
    fi
    cat > "$ENV_FILE" <<ENV
MANIFEST_API_KEY=$MANIFEST_API_KEY
PATH=/usr/local/bin:/usr/bin:/bin:$TARGET_HOME/.bun/bin
ENV
    chown $TARGET:$TARGET_GID "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    ok ".env written with Manifest API key"
fi

# ── 10. systemd service ───────────────────────────────────────────────────────
log "10. systemd user service"
install -d -o $TARGET -g $TARGET_GID -m 755 "$TARGET_HOME/.config/systemd/user"
SERVICE="$TARGET_HOME/.config/systemd/user/brainrouter.service"
if [[ ! -f "$SERVICE" ]]; then
    # Note: unquoted heredoc so $BRAINROUTER_PORT and $TARGET_UID expand
    cat > "$SERVICE" <<SERVICE_CONTENT
[Unit]
Description=brainrouter — Bonsai-routed LLM proxy
After=network-online.target

[Service]
Type=simple
EnvironmentFile=%h/.config/brainrouter/.env
ExecStartPre=%h/.local/bin/brainrouter-cleanup
ExecStart=/usr/local/bin/brainrouter serve \
    --config %h/.config/brainrouter/brainrouter.yaml \
    --socket /run/user/%U/brainrouter.sock \
    --tcp-addr 127.0.0.1:$BRAINROUTER_PORT
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
SERVICE_CONTENT
    chown $TARGET:$TARGET_GID "$SERVICE"
    ok "brainrouter.service written"
else
    skip "brainrouter.service already exists"
fi

# ── 5b. brainrouter-cleanup for adrien (port-aware) ─────────────────────────
# The shared /usr/local/bin/brainrouter-cleanup hardcodes port 9099 (papa).
# Write a user-local override that uses adrien's port.
log "5b. brainrouter-cleanup (port $BRAINROUTER_PORT)"
install -d -o $TARGET -g $TARGET_GID -m 755 "$TARGET_HOME/.local/bin"
CLEANUP="$TARGET_HOME/.local/bin/brainrouter-cleanup"
if [[ ! -f "$CLEANUP" ]]; then
    cat > "$CLEANUP" <<CLEANUP_SCRIPT
#!/bin/bash
# Per-user brainrouter cleanup for $TARGET (port $BRAINROUTER_PORT)
set -uo pipefail
SOCKET="/run/user/\$(id -u)/brainrouter.sock"
TCP_PORT=$BRAINROUTER_PORT
[[ -S "\$SOCKET" ]] && rm -f "\$SOCKET" && echo "Removed stale socket \$SOCKET"
for pid in \$(pgrep -u "\$(id -u)" -x brainrouter 2>/dev/null || true); do
    port=\$(ss -tlnp 2>/dev/null | grep "pid=\$pid," | grep -oP ':\K[0-9]+(?=\s)' || true)
    if [[ -n "\$port" && "\$port" == "\$TCP_PORT" ]]; then
        echo "Killing stale brainrouter pid=\$pid on port=\$port"
        kill "\$pid" 2>/dev/null || true; sleep 1
        kill -0 "\$pid" 2>/dev/null && kill -9 "\$pid" 2>/dev/null || true
    fi
done
echo "Cleanup done"
CLEANUP_SCRIPT
    chown $TARGET:$TARGET_GID "$CLEANUP"
    chmod 755 "$CLEANUP"
    ok "brainrouter-cleanup written for $TARGET (port $BRAINROUTER_PORT)"
else
    skip "brainrouter-cleanup already exists for $TARGET"
fi

# ── 11. linger ────────────────────────────────────────────────────────────────
log "11. loginctl linger"
loginctl enable-linger $TARGET
ok "linger enabled (brainrouter starts at boot)"

# ── 12. Enable + start ────────────────────────────────────────────────────────
log "12. Enable and start brainrouter"
# sudo -u doesn't connect to the user's D-Bus; use systemctl --machine= instead
systemctl --machine=${TARGET}@ --user daemon-reload
systemctl --machine=${TARGET}@ --user enable brainrouter
systemctl --machine=${TARGET}@ --user start brainrouter \
    && ok "brainrouter started on :$BRAINROUTER_PORT" \
    || warn "start failed — check: journalctl --machine=${TARGET}@ --user-unit=brainrouter -n 20"

# ── 13. OMP agent config ──────────────────────────────────────────────────────
log "13. OMP agent config"
OMP_AGENT="$TARGET_HOME/.omp/agent"
install -d -o $TARGET -g $TARGET_GID -m 750 "$OMP_AGENT"

# models.yml — copy papa's then patch port 9099 → adrien's port
if [[ ! -f "$OMP_AGENT/models.yml" ]]; then
    sed "s|http://127.0.0.1:9099|http://127.0.0.1:$BRAINROUTER_PORT|g" \
        "$PAPA_HOME/.omp/agent/models.yml" \
        > "$OMP_AGENT/models.yml"
    chown $TARGET:$TARGET_GID "$OMP_AGENT/models.yml"
    chmod 644 "$OMP_AGENT/models.yml"
    ok "models.yml written (port $BRAINROUTER_PORT)"
else
    skip "models.yml already exists"
fi

# mcp.json — brainrouter MCP server, socket path for adrien's UID
if [[ ! -f "$OMP_AGENT/mcp.json" ]]; then
    cat > "$OMP_AGENT/mcp.json" <<MCP
{
  "mcpServers": {
    "brainrouter": {
      "type": "stdio",
      "command": "/usr/local/bin/brainrouter",
      "args": ["mcp", "--socket", "/run/user/$TARGET_UID/brainrouter.sock"],
      "timeout": 300000
    }
  }
}
MCP
    chown $TARGET:$TARGET_GID "$OMP_AGENT/mcp.json"
    ok "mcp.json written (socket for uid=$TARGET_UID)"
else
    skip "mcp.json already exists"
fi

# APPEND_SYSTEM.md — review loop instructions
if [[ ! -f "$OMP_AGENT/APPEND_SYSTEM.md" ]]; then
    install -o $TARGET -g $TARGET_GID -m 644 \
        "$PAPA_HOME/.omp/agent/APPEND_SYSTEM.md" "$OMP_AGENT/APPEND_SYSTEM.md"
    ok "APPEND_SYSTEM.md copied"
else
    skip "APPEND_SYSTEM.md already exists"
fi

# config.yml — OMP agent settings
if [[ ! -f "$OMP_AGENT/config.yml" ]]; then
    install -o $TARGET -g $TARGET_GID -m 644 \
        "$PAPA_HOME/.omp/agent/config.yml" "$OMP_AGENT/config.yml"
    ok "config.yml copied"
else
    skip "config.yml already exists"
fi

# ── Verify ────────────────────────────────────────────────────────────────────
log "Verifying..."
sleep 4
if sudo -u $TARGET \
       XDG_RUNTIME_DIR=/run/user/$TARGET_UID \
       curl -sf http://127.0.0.1:$BRAINROUTER_PORT/health &>/dev/null; then
    ok "brainrouter health check passed for $TARGET on :$BRAINROUTER_PORT"
else
    warn "health check failed — check: sudo -u $TARGET journalctl --user -u brainrouter -n 30"
fi

echo ""
echo "┌─────────────────────────────────────────────────────────────────┐"
echo "│  adrien setup complete                                          │"
echo "├─────────────────────────────────────────────────────────────────┤"
echo "│  Port:       :9100  (papa uses :9099)                          │"
echo "│  Health:     curl http://127.0.0.1:9100/health                 │"
echo "│  Dashboard:  http://127.0.0.1:9100/dashboard                   │"
echo "│  Logs:       sudo -u adrien journalctl --user -u brainrouter   │"
echo "└─────────────────────────────────────────────────────────────────┘"
echo ""
