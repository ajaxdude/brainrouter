#!/usr/bin/env bash
# uninstall.sh — remove the brainrouter AI stack
#
# Usage:
#   bash uninstall.sh                    # remove per-user components only
#   bash uninstall.sh --system           # also remove system services (requires sudo)
#   bash uninstall.sh --remove-models    # also delete downloaded model files
#   bash uninstall.sh --yes              # non-interactive mode
#
# This script is safe to re-run.

set -euo pipefail

SYSTEM=0
REMOVE_MODELS=0
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system)         SYSTEM=1; shift ;;
    --remove-models)  REMOVE_MODELS=1; shift ;;
    --yes|-y)         ASSUME_YES=1; shift ;;
    -h|--help)        sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                echo "Unknown flag: $1"; exit 1 ;;
  esac
done

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*" >&2; }
skip() { printf '\033[1;36m--\033[0m  %s (not found)\n' "$*"; }

confirm() {
    [[ "$ASSUME_YES" -eq 1 ]] && return 0
    read -rp "$1 [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]]
}

# ── 1. Stop brainrouter ───────────────────────────────────────────────
log "Stopping brainrouter"
if systemctl --user is-active brainrouter &>/dev/null; then
    systemctl --user stop brainrouter
    systemctl --user disable brainrouter 2>/dev/null
    ok "brainrouter stopped"
else
    skip "brainrouter service"
fi
rm -f "$HOME/.config/systemd/user/brainrouter.service"
systemctl --user daemon-reload 2>/dev/null || true
rm -f "/run/user/$(id -u)/brainrouter.sock" 2>/dev/null || true

# ── 2. Remove brainrouter config ──────────────────────────────────────
log "Removing brainrouter config"
if [[ -d "$HOME/.config/brainrouter" ]]; then
    rm -rf "$HOME/.config/brainrouter"
    ok "Config removed"
else
    skip "brainrouter config"
fi

# ── 3. Remove harness integrations ────────────────────────────────────
log "Removing harness integrations"

# Claude Code
if command -v claude &>/dev/null; then
    claude mcp remove brainrouter --scope user 2>/dev/null && ok "Claude MCP removed" || true
fi
sed -i '/ANTHROPIC_BASE_URL.*127.0.0.1:9099/d' "$HOME/.zshrc" "$HOME/.bashrc" 2>/dev/null || true
sed -i '/ANTHROPIC_AUTH_TOKEN.*not-used/d' "$HOME/.zshrc" "$HOME/.bashrc" 2>/dev/null || true

# OMP mcp.json
if [[ -f "$HOME/.omp/agent/mcp.json" ]]; then
    python3 -c "
import json
with open('$HOME/.omp/agent/mcp.json', 'r') as f:
    data = json.load(f)
if 'mcpServers' in data and 'brainrouter' in data['mcpServers']:
    del data['mcpServers']['brainrouter']
    with open('$HOME/.omp/agent/mcp.json', 'w') as f:
        json.dump(data, f, indent=2)
    print('OMP MCP entry removed')
" 2>/dev/null && ok "OMP mcp.json cleaned" || true
fi

# OpenCode
if [[ -f "$HOME/.config/opencode/config.json" ]]; then
    python3 -c "
import json
with open('$HOME/.config/opencode/config.json', 'r') as f:
    data = json.load(f)
changed = False
for key in ['provider', 'mcp']:
    if key in data and 'brainrouter' in data[key]:
        del data[key]['brainrouter']
        changed = True
if changed:
    with open('$HOME/.config/opencode/config.json', 'w') as f:
        json.dump(data, f, indent=2)
" 2>/dev/null && ok "OpenCode config cleaned" || true
fi

# Droid
if [[ -f "$HOME/.factory/mcp.json" ]]; then
    python3 -c "
import json
with open('$HOME/.factory/mcp.json', 'r') as f:
    data = json.load(f)
if 'mcpServers' in data and 'brainrouter' in data['mcpServers']:
    del data['mcpServers']['brainrouter']
    with open('$HOME/.factory/mcp.json', 'w') as f:
        json.dump(data, f, indent=2)
" 2>/dev/null && ok "Droid mcp.json cleaned" || true
fi

# ── 4. System cleanup ─────────────────────────────────────────────────
if [[ $SYSTEM -eq 1 ]]; then
    log "Removing system services"

    for svc in llama-swap manifest; do
        if systemctl is-active "$svc" &>/dev/null; then
            sudo systemctl stop "$svc"
            sudo systemctl disable "$svc" 2>/dev/null
            ok "$svc stopped"
        fi
        sudo rm -f "/etc/systemd/system/$svc.service"
    done
    sudo systemctl daemon-reload

    # Docker containers
    for dir in /opt/ai/llama-swap /opt/ai/manifest; do
        if [[ -f "$dir/docker-compose.yml" ]]; then
            sudo docker compose -f "$dir/docker-compose.yml" down -v 2>/dev/null || true
            ok "Docker containers in $dir removed"
        fi
    done

    # brainrouter binary
    if [[ -f /usr/local/bin/brainrouter ]]; then
        sudo rm -f /usr/local/bin/brainrouter
        ok "brainrouter binary removed"
    fi

    # /opt/ai
    if [[ -d /opt/ai ]] && confirm "Remove /opt/ai directory?"; then
        sudo rm -rf /opt/ai
        ok "/opt/ai removed"
    fi

    # /etc/skel templates
    sudo rm -f /etc/skel/.config/brainrouter/brainrouter.yaml 2>/dev/null
    sudo rm -f /etc/skel/.config/brainrouter/.env 2>/dev/null
    sudo rm -f /etc/skel/.config/systemd/user/brainrouter.service 2>/dev/null
    sudo rm -f /etc/skel/.local/bin/llama-server-toolbox 2>/dev/null
    sudo rm -f /etc/profile.d/ai-stack.sh 2>/dev/null

    # aistack group
    if getent group aistack &>/dev/null && confirm "Remove aistack group?"; then
        sudo groupdel aistack 2>/dev/null
        ok "aistack group removed"
    fi
fi

# ── 5. Models ─────────────────────────────────────────────────────────
if [[ $REMOVE_MODELS -eq 1 ]]; then
    log "Removing model files"
    if [[ -d "$HOME/models/bonsai" ]] && confirm "Remove ~/models/bonsai?"; then
        rm -rf "$HOME/models/bonsai"
        ok "User models removed"
    fi
    if [[ $SYSTEM -eq 1 && -d /opt/models ]] && confirm "Remove /opt/models? (shared models for all users)"; then
        sudo rm -rf /opt/models
        ok "/opt/models removed"
    fi
fi

# ── Verify ───────────────────────────────────────────────────────────
echo ""
log "Removal complete. Verification:"
curl -sf http://127.0.0.1:9099/health &>/dev/null && warn "brainrouter still responding on :9099" || ok "brainrouter: removed"
curl -sf http://localhost:8081/v1/models &>/dev/null && warn "llama-swap still responding on :8081" || ok "llama-swap: removed"
curl -sf http://localhost:2099/api/v1/health &>/dev/null && warn "Manifest still responding on :2099" || ok "Manifest: removed"
echo ""
echo "Note: Docker, Rust, system packages, and huggingface-cli are preserved."
