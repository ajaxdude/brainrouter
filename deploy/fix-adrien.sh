#!/usr/bin/env bash
# deploy/fix-adrien.sh — fix "omp: command not found" for adrien in toolbox
#
# Run as:  sudo bash deploy/fix-adrien.sh
#
# What it does:
#   1. Writes /etc/profile.d/ai-stack.sh  (PATH + ANTHROPIC_BASE_URL for all users)
#   2. Writes /usr/local/bin/omp          (system-wide wrapper, works inside toolbox)
#   3. Installs oh-my-pi for adrien via bun
#   4. Runs deploy/adrien-setup.sh        (brainrouter service, config, OMP agent config)

set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "Run with sudo: sudo bash $0"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok\033[0m %s\n' "$*"; }
skip() { printf '\033[1;36m --\033[0m %s\n' "$*"; }

# ── 1. /etc/profile.d/ai-stack.sh ────────────────────────────────────────────

log "1. /etc/profile.d/ai-stack.sh"
if [[ -f /etc/profile.d/ai-stack.sh ]]; then
    skip "/etc/profile.d/ai-stack.sh already exists"
else
    cat > /etc/profile.d/ai-stack.sh <<'PROFILE'
# AI stack — added by brainrouter deploy/fix-adrien.sh
# Puts /usr/local/bin and ~/.bun/bin on PATH for every login shell.
export BUN_INSTALL="$HOME/.bun"
export PATH="/usr/local/bin:$BUN_INSTALL/bin:$PATH"
export ANTHROPIC_BASE_URL="http://127.0.0.1:9099"
export OPENAI_BASE_URL="http://127.0.0.1:9099/v1"
PROFILE
    chmod 644 /etc/profile.d/ai-stack.sh
    ok "/etc/profile.d/ai-stack.sh written"
fi

# ── 2. /usr/local/bin/omp wrapper ────────────────────────────────────────────
#
# Delegates to the per-user bun-installed omp binary.
# Putting a real file here (not just profile.d) means it works inside toolbox
# containers too, which inherit /usr/local/bin via the host bind-mount.

log "2. /usr/local/bin/omp"
if [[ -x /usr/local/bin/omp ]]; then
    skip "/usr/local/bin/omp already exists"
else
    cat > /usr/local/bin/omp <<'OMP'
#!/usr/bin/env bash
# System-wide omp launcher.
# Delegates to the per-user ~/.bun/bin/omp installed by bun.
BUN_OMP="$HOME/.bun/bin/omp"
if [[ -x "$BUN_OMP" ]]; then
    exec "$BUN_OMP" "$@"
fi
echo "omp not found. Install it first:" >&2
echo "  bun install -g @oh-my-pi/pi-coding-agent" >&2
exit 1
OMP
    chmod 755 /usr/local/bin/omp
    ok "/usr/local/bin/omp wrapper written"
fi

# ── 3. Install oh-my-pi for adrien ───────────────────────────────────────────

log "3. oh-my-pi for adrien"
if sudo -u adrien /usr/local/bin/bun pm ls -g 2>/dev/null | grep -q '@oh-my-pi/pi-coding-agent'; then
    skip "oh-my-pi already installed for adrien"
else
    sudo -u adrien HOME=/home/adrien /usr/local/bin/bun install -g @oh-my-pi/pi-coding-agent 2>&1 | tail -5
    ok "oh-my-pi installed for adrien"
fi

# ── 4. Full adrien brainrouter setup ─────────────────────────────────────────

log "4. adrien-setup.sh"
bash "$SCRIPT_DIR/adrien-setup.sh"

# ── Verify ────────────────────────────────────────────────────────────────────

log "Verify"
ADRIEN_UID=$(id -u adrien)

echo ""
echo "  /etc/profile.d/ai-stack.sh : $(test -f /etc/profile.d/ai-stack.sh && echo OK || echo MISSING)"
echo "  /usr/local/bin/omp          : $(test -x /usr/local/bin/omp && echo OK || echo MISSING)"
echo "  adrien ~/.bun/bin/omp       : $(test -x /home/adrien/.bun/bin/omp && echo OK || echo MISSING)"
echo "  brainrouter health (:9100)  : $(curl -sf http://127.0.0.1:9100/health && echo OK || echo NOT RUNNING)"
echo ""
echo "  adrien can now open a new shell (or toolbox) and run: omp"
echo ""
