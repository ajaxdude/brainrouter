#!/usr/bin/env bash
# Install omp + brainrouter service for all human users missing the installation.
# Run as: sudo bash /tmp/fix-omp-users.sh
set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "Run with sudo: sudo bash $0"; exit 1; }

SYSTEM_BUN=/usr/local/bin/bun
[[ -x "$SYSTEM_BUN" ]] || { echo "ERROR: $SYSTEM_BUN not found"; exit 1; }

human_users() {
    getent passwd | awk -F: '$3 >= 1000 && $3 < 65534 && $7 !~ /nologin|false/ {print $1}'
}

while IFS= read -r user; do
    uhome=$(getent passwd "$user" | cut -d: -f6)
    uid=$(id -u "$user")
    br_port=$((8099 + uid))
    [[ -d "$uhome" ]] || continue

    echo ""
    echo "=== $user (uid=$uid, port=$br_port) ==="

    # ── skel files ──────────────────────────────────────────────────────────
    for f in .bashrc .bash_profile .bash_logout; do
        [[ -f "$uhome/$f" ]] && continue
        [[ -f "/etc/skel/$f" ]] || continue
        install -o "$user" -g "$user" -m 644 "/etc/skel/$f" "$uhome/$f"
        echo "  created $f from skel"
    done

    # ── brainrouter env vars in .bashrc ─────────────────────────────────────
    if [[ -f "$uhome/.bashrc" ]] && ! grep -q "ANTHROPIC_BASE_URL" "$uhome/.bashrc"; then
        printf '\n# brainrouter — per-user AI proxy port\nexport ANTHROPIC_BASE_URL=http://127.0.0.1:%s\nexport OPENAI_BASE_URL=http://127.0.0.1:%s/v1\n' \
            "$br_port" "$br_port" >> "$uhome/.bashrc"
        chown "$user:$user" "$uhome/.bashrc"
        echo "  added ANTHROPIC_BASE_URL=$br_port to .bashrc"
    fi

    # ── omp global install ──────────────────────────────────────────────────
    if ! sudo -u "$user" HOME="$uhome" "$SYSTEM_BUN" pm ls -g 2>/dev/null | grep -q "pi-coding-agent"; then
        echo "  installing @oh-my-pi/pi-coding-agent..."
        sudo -u "$user" HOME="$uhome" "$SYSTEM_BUN" install -g @oh-my-pi/pi-coding-agent 2>&1 | tail -3
        echo "  omp installed: $(ls "$uhome/.bun/bin/omp" 2>/dev/null && echo yes || echo MISSING)"
    else
        echo "  omp already installed"
    fi

    # ── brainrouter config dir ──────────────────────────────────────────────
    install -d -o "$user" -g "$user" -m 750 "$uhome/.config/brainrouter"
    if [[ ! -f "$uhome/.config/brainrouter/brainrouter.yaml" && -f /etc/brainrouter/brainrouter.yaml ]]; then
        install -o "$user" -g "$user" -m 644 \
            /etc/brainrouter/brainrouter.yaml \
            "$uhome/.config/brainrouter/brainrouter.yaml"
        echo "  copied brainrouter.yaml"
    fi

    # ── brainrouter systemd user service ───────────────────────────────────
    install -d -o "$user" -g "$user" -m 755 "$uhome/.config/systemd/user"
    cat > "$uhome/.config/systemd/user/brainrouter.service" << SERVICE
[Unit]
Description=brainrouter — Bonsai-routed LLM proxy
After=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/brainrouter/env
ExecStart=/usr/local/bin/brainrouter serve \\
  --config /etc/brainrouter/brainrouter.yaml \\
  --socket /run/user/${uid}/brainrouter.sock \\
  --tcp-addr 127.0.0.1:${br_port}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
SERVICE
    chown "$user:$user" "$uhome/.config/systemd/user/brainrouter.service"
    echo "  wrote brainrouter.service (port $br_port)"

    # ── linger ─────────────────────────────────────────────────────────────
    if ! loginctl show-user "$user" 2>/dev/null | grep -q "Linger=yes"; then
        loginctl enable-linger "$user" && echo "  linger enabled"
    fi

    # ── reload + enable service ─────────────────────────────────────────────
    DBUS="unix:path=/run/user/${uid}/bus"
    XDG="/run/user/${uid}"
    sudo -u "$user" DBUS_SESSION_BUS_ADDRESS="$DBUS" XDG_RUNTIME_DIR="$XDG" \
        systemctl --user daemon-reload 2>/dev/null || true
    sudo -u "$user" DBUS_SESSION_BUS_ADDRESS="$DBUS" XDG_RUNTIME_DIR="$XDG" \
        systemctl --user enable --now brainrouter 2>/dev/null \
        && echo "  brainrouter.service enabled+started" \
        || echo "  brainrouter.service enabled (starts after reboot or API key set)"

done < <(human_users)

# ── Also install omp harness config for each user ──────────────────────────
echo ""
echo "=== Running brainrouter install omp for all users ==="
while IFS= read -r user; do
    uhome=$(getent passwd "$user" | cut -d: -f6)
    [[ -d "$uhome" && -x /usr/local/bin/brainrouter ]] || continue
    sudo -u "$user" HOME="$uhome" /usr/local/bin/brainrouter install omp --yes 2>/dev/null \
        && echo "  $user: OMP MCP configured" \
        || echo "  $user: OMP MCP config skipped (omp not configured)"
done < <(human_users)

echo ""
echo "=== Done. All users should open a new terminal to get updated PATH. ==="

# ── Fix the system-wide omp wrapper to use the system bun ──────────────────
echo "=== Updating /usr/local/bin/omp to use system bun ==="
# Find the pi-coding-agent script from papa's install (it's the most up-to-date)
# If it exists, update the wrapper to use /usr/local/bin/bun instead of papa's bun
PAPA_SCRIPT=$(ls /home/papa/node_modules/@oh-my-pi/pi-coding-agent/src/cli.ts 2>/dev/null || true)
if [[ -n "$PAPA_SCRIPT" ]]; then
    cat > /usr/local/bin/omp << WRAPPER
#!/bin/bash
# System-wide omp (oh-my-pi) wrapper — updated by fix-omp-users.sh
# Prefers user's own ~/.bun/bin/omp; falls back to papa's install via system bun.
# To install omp per-user: bun install -g @oh-my-pi/pi-coding-agent
if [[ -x "\$HOME/.bun/bin/omp" ]]; then
    exec "\$HOME/.bun/bin/omp" "\$@"
fi
exec /usr/local/bin/bun "$PAPA_SCRIPT" "\$@"
WRAPPER
    chmod 755 /usr/local/bin/omp
    echo "  wrapper updated (user-first, fallback to system bun + papa's install)"
else
    echo "  papa's pi-coding-agent not found, wrapper unchanged"
fi
