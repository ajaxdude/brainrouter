#!/usr/bin/env bash
# user-setup.sh — per-user brainrouter activation (run after system deploy)
#
# Each user runs this to activate brainrouter and connect their harnesses.
# Designed for multi-user deployments where the admin has already run deploy.sh --multi-user.
#
# Usage: bash user-setup.sh [--yes]

set -euo pipefail

ASSUME_YES=0
[[ "${1:-}" == "--yes" || "${1:-}" == "-y" ]] && ASSUME_YES=1

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m  %s\n' "$*" >&2; exit 1; }

log "Setting up brainrouter for $USER"

# Check prerequisites
command -v brainrouter &>/dev/null || die "brainrouter not found in PATH. Ask admin to install it."
curl -sf http://localhost:8081/v1/models &>/dev/null || warn "llama-swap not responding on :8081"
curl -sf http://localhost:2099/api/v1/health &>/dev/null || die "Manifest not running on :2099. Ask admin to start it."

# Check config exists
CONFIG_DIR="$HOME/.config/brainrouter"
if [[ ! -f "$CONFIG_DIR/brainrouter.yaml" ]]; then
    die "Config not found at $CONFIG_DIR/brainrouter.yaml. Ask admin to run deploy.sh --multi-user."
fi

# Check/set Manifest API key
if ! grep -q 'mnfst_[a-zA-Z0-9]' "$CONFIG_DIR/.env" 2>/dev/null; then
    echo ""
    echo "You need a Manifest API key."
    echo "Open http://localhost:2099 -> Settings -> API Keys -> Create key"
    echo ""
    if [[ $ASSUME_YES -eq 0 ]]; then
        read -rp "Paste your MANIFEST_API_KEY: " key
        if [[ -n "$key" ]]; then
            echo "MANIFEST_API_KEY=$key" > "$CONFIG_DIR/.env"
            chmod 600 "$CONFIG_DIR/.env"
            ok "API key saved"
        else
            warn "No key entered. Edit $CONFIG_DIR/.env manually."
        fi
    else
        warn "API key not set. Edit $CONFIG_DIR/.env before starting brainrouter."
    fi
fi

# Enable brainrouter service
log "Enabling brainrouter service"
systemctl --user daemon-reload
systemctl --user enable --now brainrouter

# Wait for it
sleep 3
if curl -sf http://127.0.0.1:9099/health &>/dev/null; then
    ok "brainrouter is running"
else
    warn "brainrouter not responding yet. Check: journalctl --user -u brainrouter -n 20"
    warn "Common fixes:"
    warn "  - Ensure MANIFEST_API_KEY is set in $CONFIG_DIR/.env"
    warn "  - Ensure fallback_model in brainrouter.yaml matches a llama-swap model"
    warn "  - Ensure bonsai model_path exists"
    exit 1
fi

# Install harness integrations
log "Connecting harnesses"

brainrouter install claude --shell-rc --yes 2>/dev/null && ok "claude configured" || warn "claude not available"

if command -v omp &>/dev/null; then
    brainrouter install omp --yes 2>/dev/null && ok "omp configured"
fi

echo ""
echo "Done! Reload your shell to pick up environment changes."
echo "Test: curl http://127.0.0.1:9099/health"
echo "Dashboard: http://127.0.0.1:9099/dashboard"
echo ""
echo "Other harnesses (run any that apply):"
echo "  brainrouter install opencode"
echo "  brainrouter install vibe"
echo "  brainrouter install codex"
echo "  brainrouter install droid"
