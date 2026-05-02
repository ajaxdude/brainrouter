#!/usr/bin/env bash
# install.sh — one-shot brainrouter AI stack installer for Fedora Linux
#
# Run as:  sudo bash install.sh
#
# What this does (all idempotent — safe to re-run):
#
#   SYSTEM (runs as root):
#     1.  Install Fedora packages (git, golang, toolbox, docker, etc.)
#     2.  Install bun system-wide to /usr/local/bin
#     3.  Create aistack group; add all human users
#     4.  Create /opt/models/bonsai (shared model storage)
#     5.  Download Bonsai Q4_K_M classifier to /opt/models/bonsai
#     6.  Set up Manifest (docker compose) at /opt/ai/manifest as a system service
#     7.  Set up llama-swap (docker compose) at /opt/ai/llama-swap as a system service
#     8.  Build brainrouter and install to /usr/local/bin
#     9.  Install llama-server-toolbox wrapper to /usr/local/bin
#     10. Create toolbox container llama-vulkan-radv (shared via /var/lib/toolbox image pull)
#     11. Write /etc/brainrouter/env (shared MANIFEST_API_KEY — root:aistack, mode 640)
#     12. Write /etc/brainrouter/brainrouter.yaml (shared base config)
#     13. Seed /etc/skel/.config/systemd/user/brainrouter.service
#     14. Write /etc/profile.d/ai-stack.sh (PATH for all login shells)
#
#   PER-USER (runs under each human user with loginctl enable-linger):
#     15. Install oh-my-pi globally via bun for each user
#     16. Seed ~/.config/brainrouter/ (symlink to /etc/brainrouter/env, copy yaml)
#     17. Install ~/.config/systemd/user/brainrouter.service
#     18. loginctl enable-linger (so services start at boot, not just login)
#     19. systemctl --user daemon-reload && enable --now brainrouter
#
#   POST-INSTALL:
#     Prints the one manual step: paste Manifest API key into /etc/brainrouter/env
#     then run: sudo systemctl restart brainrouter@USER for each user (or reboot).
#
# Assumptions:
#   - Fedora Linux with systemd
#   - AMD GPU with RADV Vulkan (llama-server-toolbox defaults to llama-vulkan-radv)
#   - Internet access for downloads
#   - brainrouter source is at ~/ai/projects/brainrouter relative to the sudo caller
#     OR will be cloned from GitHub
#   - llama-swap model: edit /opt/ai/llama-swap/config.yaml after install to point
#     at your actual GGUF model in /opt/models/

set -euo pipefail

# ── Constants ────────────────────────────────────────────────────────────────

BRAINROUTER_REPO="https://github.com/ajaxdude/brainrouter"
BONSAI_HF_REPO="bartowski/prism-ml_Bonsai-8B-unpacked-GGUF"
BONSAI_FILE="prism-ml_Bonsai-8B-unpacked-Q4_K_M.gguf"
BONSAI_PATH="/opt/models/bonsai/${BONSAI_FILE}"
MANIFEST_PORT=3001
LLAMA_SWAP_PORT=8081
BRAINROUTER_PORT=9099
TOOLBOX_IMAGE="docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv"
TOOLBOX_NAME="llama-vulkan-radv"

# The user who invoked sudo — this is who we clone/build brainrouter as.
SUDO_USER_HOME=$(getent passwd "${SUDO_USER:-root}" | cut -d: -f6)
BR_SRC="${SUDO_USER_HOME}/ai/projects/brainrouter"

# ── Colour helpers ────────────────────────────────────────────────────────────

log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m !!\033[0m %s\n' "$*" >&2; }
skip() { printf '\033[1;36m --\033[0m %s (already done)\n' "$*"; }
die()  { printf '\033[1;31mERR\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\033[1;35m >>>\033[0m %s\n' "$*"; }

has()      { command -v "$1" &>/dev/null; }
port_up()  { ss -ltn 2>/dev/null | grep -qE "[: ]${1}\\b"; }
human_users() {
    getent passwd | awk -F: '$3 >= 1000 && $3 < 65534 && $7 !~ /nologin|false/ {print $1}'
}

# ── Preflight ─────────────────────────────────────────────────────────────────

[[ $EUID -eq 0 ]] || die "Run with sudo: sudo bash install.sh"
[[ -n "${SUDO_USER:-}" ]] || die "Do not run as root directly. Run: sudo bash install.sh"

log "brainrouter AI stack installer"
echo "  Installing as: root (for system components)"
echo "  Caller user:   $SUDO_USER ($SUDO_USER_HOME)"
echo "  Human users:   $(human_users | tr '\n' ' ')"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────

log "Step 1/19: System packages"

PKGS=(git golang toolbox podman docker docker-compose-plugin
      gcc-c++ cmake vulkan-headers vulkan-loader-devel libshaderc
      python3-pip curl wget unzip)

missing=()
for pkg in "${PKGS[@]}"; do
    rpm -q "$pkg" &>/dev/null || missing+=("$pkg")
done

if [[ ${#missing[@]} -gt 0 ]]; then
    step "Installing: ${missing[*]}"
    dnf install -y "${missing[@]}"
else
    skip "System packages"
fi

# Docker needs to be running for Manifest and llama-swap
if ! systemctl is-active --quiet docker; then
    step "Starting docker"
    systemctl enable --now docker
fi
ok "Docker running"

# ── 2. bun system-wide ────────────────────────────────────────────────────────

log "Step 2/19: bun (system-wide)"

if [[ -x /usr/local/bin/bun ]]; then
    skip "bun $(/usr/local/bin/bun --version)"
else
    step "Installing bun to /usr/local/bin"
    # Download the official bun binary and install system-wide
    BUN_TMP=$(mktemp -d)
    curl -fsSL "https://bun.sh/install" | BUN_INSTALL="$BUN_TMP" bash
    install -m 755 "$BUN_TMP/bin/bun" /usr/local/bin/bun
    rm -rf "$BUN_TMP"
    ok "bun $(/usr/local/bin/bun --version) installed"
fi

# ── 3. aistack group ──────────────────────────────────────────────────────────

log "Step 3/19: aistack group"

groupadd -f aistack
while IFS= read -r user; do
    if id "$user" &>/dev/null && ! id -nG "$user" | grep -qw aistack; then
        usermod -aG aistack "$user"
        ok "  $user → aistack"
    else
        skip "  $user already in aistack"
    fi
done < <(human_users)

# ── 4. Shared directories ─────────────────────────────────────────────────────

log "Step 4/19: Shared directories"

mkdir -p /opt/models/bonsai /opt/ai/{llama-swap,manifest,bin} /etc/brainrouter
chown -R root:aistack /opt/models /opt/ai /etc/brainrouter
find /opt/models -type d -exec chmod 2775 {} \;
find /opt/models -type f -exec chmod 664 {} \; 2>/dev/null || true
chmod -R 775 /opt/ai
chmod 750 /etc/brainrouter
ok "Shared directories ready"

# ── 5. Bonsai classifier model ────────────────────────────────────────────────

log "Step 5/19: Bonsai classifier model"

if [[ -f "$BONSAI_PATH" ]]; then
    skip "Bonsai at $BONSAI_PATH"
else
    step "Downloading Bonsai Q4_K_M (~5.2 GB) — this will take a few minutes"
    if ! has huggingface-cli; then
        pip3 install -q "huggingface_hub[cli]"
    fi
    huggingface-cli download "$BONSAI_HF_REPO" \
        --include "$BONSAI_FILE" \
        --local-dir /opt/models/bonsai
    chown root:aistack "$BONSAI_PATH"
    chmod 664 "$BONSAI_PATH"
    ok "Bonsai downloaded"
fi

# ── 6. Manifest (cloud LLM router) ────────────────────────────────────────────

log "Step 6/19: Manifest (cloud LLM router)"

MANIFEST_DIR="/opt/ai/manifest"

if [[ ! -f "${MANIFEST_DIR}/docker-compose.yml" ]]; then
    step "Fetching Manifest docker-compose"
    curl -fsSL \
        "https://raw.githubusercontent.com/mnfst/manifest/main/docker-compose.yml" \
        -o "${MANIFEST_DIR}/docker-compose.yml"

    # Generate required secret and write .env
    AUTH_SECRET=$(openssl rand -hex 32)
    cat > "${MANIFEST_DIR}/.env" <<ENV
# Generated by brainrouter install.sh
BETTER_AUTH_SECRET=${AUTH_SECRET}
BETTER_AUTH_URL=http://localhost:${MANIFEST_PORT}
ENV
    chmod 640 "${MANIFEST_DIR}/.env"
    chown root:aistack "${MANIFEST_DIR}/.env"
    ok "Manifest docker-compose installed"
else
    skip "Manifest docker-compose exists"
fi

chown -R root:aistack "${MANIFEST_DIR}"

# Systemd system service for Manifest
if [[ ! -f /etc/systemd/system/manifest.service ]]; then
    step "Installing manifest.service"
    cat > /etc/systemd/system/manifest.service <<'SERVICE'
[Unit]
Description=Manifest — cloud LLM router
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/ai/manifest
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down --timeout 30
TimeoutStartSec=180
Restart=on-failure

[Install]
WantedBy=multi-user.target
SERVICE
    systemctl daemon-reload
fi

if ! systemctl is-active --quiet manifest; then
    step "Starting manifest"
    systemctl enable --now manifest
fi
ok "Manifest running (port $MANIFEST_PORT)"

# ── 7. llama-swap (local model runner) ────────────────────────────────────────

log "Step 7/19: llama-swap"

LSWAP_DIR="/opt/ai/llama-swap"

if [[ ! -f "${LSWAP_DIR}/docker-compose.yml" ]]; then
    step "Writing llama-swap docker-compose"
    cat > "${LSWAP_DIR}/docker-compose.yml" <<'YAML'
services:
  llama-swap:
    image: ghcr.io/mostlygeek/llama-swap:latest
    ports:
      - "127.0.0.1:8081:8080"
    volumes:
      - ./config.yaml:/config.yaml:ro
      - /opt/models:/models:ro
      - /dev/dri:/dev/dri
    devices:
      - /dev/dri:/dev/dri
    group_add:
      - video
      - render
    environment:
      - AMD_VULKAN_ICD=RADV
    restart: unless-stopped
    command: ["--config", "/config.yaml", "--listen", "0.0.0.0:8080"]
YAML
    ok "llama-swap docker-compose written"
else
    skip "llama-swap docker-compose exists"
fi

if [[ ! -f "${LSWAP_DIR}/config.yaml" ]]; then
    step "Writing llama-swap config.yaml (placeholder — edit model path after install)"
    cat > "${LSWAP_DIR}/config.yaml" <<'YAML'
# llama-swap config — edit "your-local-model" and model path to match your GGUF
# Models live in /opt/models/ on this machine.
# After editing, run: sudo systemctl restart llama-swap

startPort: 5800
healthCheckTimeout: 300
globalTTL: 180
logLevel: info
logToStdout: both
sendLoadingState: true

macros:
  "common": >-
    --no-webui --jinja
    -t 8 -tb 16 --parallel 1
    -ngl 999 --no-mmap -fa on
    --host 0.0.0.0
    -c 32768

models:
  "your-local-model":
    name: "Your Local Model"
    cmd: >
      llama-server --port ${PORT} ${common}
        --model /models/path/to/your-model.gguf
YAML
    chown root:aistack "${LSWAP_DIR}/config.yaml"
    chmod 664 "${LSWAP_DIR}/config.yaml"
    warn "EDIT REQUIRED: /opt/ai/llama-swap/config.yaml — set your model path"
else
    skip "llama-swap config.yaml exists"
fi

chown -R root:aistack "${LSWAP_DIR}"

if [[ ! -f /etc/systemd/system/llama-swap.service ]]; then
    step "Installing llama-swap.service"
    cat > /etc/systemd/system/llama-swap.service <<'SERVICE'
[Unit]
Description=llama-swap — on-demand local model router
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/ai/llama-swap
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down --timeout 30
TimeoutStartSec=120
Restart=on-failure

[Install]
WantedBy=multi-user.target
SERVICE
    systemctl daemon-reload
fi

if ! systemctl is-active --quiet llama-swap; then
    step "Starting llama-swap"
    systemctl enable --now llama-swap
fi
ok "llama-swap running (port $LLAMA_SWAP_PORT)"

# ── 8. brainrouter binary ─────────────────────────────────────────────────────

log "Step 8/19: brainrouter binary"

if has brainrouter && brainrouter --version &>/dev/null; then
    skip "brainrouter ($(brainrouter --version 2>&1 | head -1))"
else
    # Ensure Rust toolchain for the calling user
    if ! sudo -u "$SUDO_USER" bash -lc 'command -v cargo' &>/dev/null; then
        step "Installing Rust toolchain for $SUDO_USER"
        sudo -u "$SUDO_USER" bash -c \
            'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path'
    fi

    # Clone if needed
    if [[ ! -d "$BR_SRC" ]]; then
        step "Cloning brainrouter"
        sudo -u "$SUDO_USER" bash -c "mkdir -p $(dirname "$BR_SRC") && git clone $BRAINROUTER_REPO $BR_SRC"
    fi

    step "Building brainrouter (first build ~10-15 min)"
    sudo -u "$SUDO_USER" bash -lc \
        "cd $BR_SRC && source \$HOME/.cargo/env && cargo build --release"

    install -m 755 "${BR_SRC}/target/release/brainrouter" /usr/local/bin/brainrouter
    ok "brainrouter installed to /usr/local/bin"
fi

# ── 9. llama-server-toolbox wrapper ──────────────────────────────────────────

log "Step 9/19: llama-server-toolbox wrapper"

if [[ -x /usr/local/bin/llama-server-toolbox ]]; then
    skip "llama-server-toolbox"
else
    step "Installing /usr/local/bin/llama-server-toolbox"
    cat > /usr/local/bin/llama-server-toolbox <<'SCRIPT'
#!/bin/bash
# Wrapper: run llama-server inside a toolbox container with AMD Vulkan.
# LLAMA_CONTAINER defaults to llama-vulkan-radv (RADV driver, recommended).
CONTAINER=${LLAMA_CONTAINER:-llama-vulkan-radv}
ICD=${LLAMA_ICD:-RADV}
exec toolbox run --container "$CONTAINER" env AMD_VULKAN_ICD="$ICD" llama-server "$@"
SCRIPT
    chmod 755 /usr/local/bin/llama-server-toolbox
    ok "llama-server-toolbox installed"
fi

# ── 10. toolbox container ─────────────────────────────────────────────────────

log "Step 10/19: toolbox container ($TOOLBOX_NAME)"

# toolbox containers are per-user. We create it for SUDO_USER; other users can
# create their own with: toolbox create --image <IMAGE> llama-vulkan-radv
if sudo -u "$SUDO_USER" toolbox list 2>/dev/null | grep -q "$TOOLBOX_NAME"; then
    skip "toolbox $TOOLBOX_NAME (for $SUDO_USER)"
else
    step "Creating toolbox $TOOLBOX_NAME for $SUDO_USER"
    sudo -u "$SUDO_USER" toolbox create --image "$TOOLBOX_IMAGE" "$TOOLBOX_NAME"
    ok "Toolbox created"
fi

# ── 11. Shared /etc/brainrouter/env (Manifest API key) ───────────────────────

log "Step 11/19: /etc/brainrouter/env (shared Manifest API key)"

if [[ ! -f /etc/brainrouter/env ]]; then
    cat > /etc/brainrouter/env <<'ENV'
# Manifest API key — shared for all users on this machine.
# After completing the Manifest setup wizard at http://localhost:3001:
#   Settings → API Keys → Create key
# Paste the key (mnfst_xxxxx) below, then run:
#   sudo systemctl restart brainrouter@<username>   (for each user)
# OR simply reboot — services auto-start at boot via loginctl linger.
MANIFEST_API_KEY=mnfst_REPLACE_WITH_YOUR_KEY
ENV
    chown root:aistack /etc/brainrouter/env
    chmod 640 /etc/brainrouter/env
    ok "/etc/brainrouter/env created (needs key after Manifest wizard)"
else
    skip "/etc/brainrouter/env exists"
fi

# ── 12. Shared /etc/brainrouter/brainrouter.yaml ─────────────────────────────

log "Step 12/19: /etc/brainrouter/brainrouter.yaml"

if [[ ! -f /etc/brainrouter/brainrouter.yaml ]]; then
    cat > /etc/brainrouter/brainrouter.yaml <<YAML
# brainrouter system config — shared for all users.
# Edit fallback_model to match a model key in /opt/ai/llama-swap/config.yaml.
# After editing, run: sudo systemctl restart brainrouter@<username>

manifest:
  base_url: "http://localhost:${MANIFEST_PORT}/v1"
  api_key_env: MANIFEST_API_KEY

llama_swap:
  base_url: "http://localhost:${LLAMA_SWAP_PORT}/v1"
  fallback_model: "your-local-model"

bonsai:
  model_path: "${BONSAI_PATH}"
YAML
    chown root:aistack /etc/brainrouter/brainrouter.yaml
    chmod 644 /etc/brainrouter/brainrouter.yaml
    ok "/etc/brainrouter/brainrouter.yaml written"
else
    skip "/etc/brainrouter/brainrouter.yaml exists"
fi

# ── 13. /etc/skel — template brainrouter.service for new users ───────────────

log "Step 13/19: /etc/skel brainrouter.service"

install -Dm644 /dev/stdin /etc/skel/.config/systemd/user/brainrouter.service <<'SERVICE'
[Unit]
Description=brainrouter — Bonsai-routed LLM proxy
After=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/brainrouter/env
ExecStart=/usr/local/bin/brainrouter serve \
  --config /etc/brainrouter/brainrouter.yaml \
  --socket /run/user/%U/brainrouter.sock \
  --tcp-addr 127.0.0.1:9099
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
SERVICE
ok "/etc/skel brainrouter.service seeded"

# ── 14. /etc/profile.d/ai-stack.sh ───────────────────────────────────────────

log "Step 14/19: /etc/profile.d/ai-stack.sh"

cat > /etc/profile.d/ai-stack.sh <<'PROFILE'
# AI stack — added by brainrouter install.sh
# Makes bun, oh-my-pi (omp), brainrouter available in all login shells.
export BUN_INSTALL="$HOME/.bun"
export PATH="$BUN_INSTALL/bin:/usr/local/bin:$PATH"

# ANTHROPIC_BASE_URL and OPENAI_BASE_URL point to brainrouter so all
# harnesses (claude, omp, opencode, vibe, codex, droid) route through it.
export ANTHROPIC_BASE_URL="http://127.0.0.1:9099"
export OPENAI_BASE_URL="http://127.0.0.1:9099/v1"
PROFILE
chmod 644 /etc/profile.d/ai-stack.sh
ok "/etc/profile.d/ai-stack.sh written"

# ── 15-19. Per-user setup ─────────────────────────────────────────────────────

log "Steps 15-19: Per-user setup (oh-my-pi, brainrouter service, linger)"

while IFS= read -r user; do
    user_home=$(getent passwd "$user" | cut -d: -f6)
    [[ -d "$user_home" ]] || continue

    step "--- $user ($user_home) ---"

    # 15. oh-my-pi via bun (per-user global install)
    if sudo -u "$user" /usr/local/bin/bun pm ls -g 2>/dev/null | grep -q '@oh-my-pi/pi-coding-agent'; then
        skip "  oh-my-pi already installed for $user"
    else
        step "  Installing oh-my-pi for $user"
        sudo -u "$user" /usr/local/bin/bun install -g @oh-my-pi/pi-coding-agent 2>&1 | tail -3
        ok "  oh-my-pi installed for $user"
    fi

    # 16. ~/.config/brainrouter/ — point to shared config and env
    install -d -o "$user" -g "$user" -m 750 "${user_home}/.config/brainrouter"

    # Config: copy shared config (users can override locally)
    if [[ ! -f "${user_home}/.config/brainrouter/brainrouter.yaml" ]]; then
        install -o "$user" -g "$user" -m 644 \
            /etc/brainrouter/brainrouter.yaml \
            "${user_home}/.config/brainrouter/brainrouter.yaml"
        ok "  Seeded brainrouter.yaml for $user"
    else
        skip "  ${user_home}/.config/brainrouter/brainrouter.yaml"
    fi

    # 17. brainrouter.service (reads from /etc/brainrouter/env)
    install -d -o "$user" -g "$user" -m 755 \
        "${user_home}/.config/systemd/user"

    if [[ ! -f "${user_home}/.config/systemd/user/brainrouter.service" ]]; then
        install -o "$user" -g "$user" -m 644 \
            /etc/skel/.config/systemd/user/brainrouter.service \
            "${user_home}/.config/systemd/user/brainrouter.service"
        ok "  brainrouter.service installed for $user"
    else
        skip "  brainrouter.service for $user"
    fi

    # 18. loginctl enable-linger — user services start at boot without login
    if loginctl show-user "$user" 2>/dev/null | grep -q 'Linger=yes'; then
        skip "  linger already enabled for $user"
    else
        loginctl enable-linger "$user"
        ok "  linger enabled for $user (services boot without login)"
    fi

    # 19. Enable and start brainrouter service
    # Use machinectl/systemd-run to operate the user's systemd session as root.
    # loginctl enable-linger ensures the user@.service is running; we can then
    # use XDG_RUNTIME_DIR to reach the user's dbus/systemd.
    uid=$(id -u "$user")
    export XDG_RUNTIME_DIR="/run/user/${uid}"

    # Make sure the user session is started (linger starts it at boot;
    # we may need to kick it now if the machine hasn't rebooted yet).
    if ! systemctl --user --machine="${user}@.host" is-active --quiet default.target 2>/dev/null; then
        # Bring up the user session via systemd-run in user slice
        systemd-run --uid="$uid" --gid="$(id -g "$user")" \
            --unit="user-session-bootstrap-${user}" \
            --slice="user-${uid}.slice" \
            --property="PAMName=login" \
            true 2>/dev/null || true
        sleep 1
    fi

    sudo -u "$user" \
        DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/${uid}/bus" \
        XDG_RUNTIME_DIR="/run/user/${uid}" \
        systemctl --user daemon-reload 2>/dev/null || true

    sudo -u "$user" \
        DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/${uid}/bus" \
        XDG_RUNTIME_DIR="/run/user/${uid}" \
        systemctl --user enable brainrouter 2>/dev/null || true

    sudo -u "$user" \
        DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/${uid}/bus" \
        XDG_RUNTIME_DIR="/run/user/${uid}" \
        systemctl --user start brainrouter 2>/dev/null \
        && ok "  brainrouter started for $user" \
        || warn "  brainrouter not started yet for $user — will auto-start after reboot/login once API key is set"

done < <(human_users)

# ── Done ──────────────────────────────────────────────────────────────────────

# Check what's running
BR_OK=0
LSWAP_OK=0
MANIFEST_OK=0
BONSAI_OK=0

[[ -x /usr/local/bin/brainrouter ]] && BR_OK=1
port_up $LLAMA_SWAP_PORT && LSWAP_OK=1
curl -sf "http://localhost:${MANIFEST_PORT}/api/v1/health" &>/dev/null && MANIFEST_OK=1
[[ -f "$BONSAI_PATH" ]] && BONSAI_OK=1

echo ""
echo "┌─────────────────────────────────────────────────────────────────────┐"
echo "│  brainrouter AI stack — install complete                            │"
echo "├─────────────────────────────────────────────────────────────────────┤"
printf "│  brainrouter binary  %-48s│\n" "$( [[ $BR_OK -eq 1 ]] && echo 'installed (/usr/local/bin/brainrouter)' || echo 'NOT FOUND — check build output above')"
printf "│  llama-swap          %-48s│\n" "$( [[ $LSWAP_OK -eq 1 ]] && echo "running on :${LLAMA_SWAP_PORT}" || echo 'enabled (starts at boot)')"
printf "│  Manifest            %-48s│\n" "$( [[ $MANIFEST_OK -eq 1 ]] && echo "running on :${MANIFEST_PORT}" || echo 'enabled (starts at boot)')"
printf "│  Bonsai model        %-48s│\n" "$( [[ $BONSAI_OK -eq 1 ]] && echo "$BONSAI_PATH" || echo 'NOT FOUND — see step 5 above')"
echo "├─────────────────────────────────────────────────────────────────────┤"
echo "│                                                                     │"
echo "│  *** ONE MANUAL STEP REQUIRED — do this now: ***                   │"
echo "│                                                                     │"
echo "│  1. Open http://localhost:3001 in a browser                        │"
echo "│     Complete the setup wizard (create admin account,               │"
echo "│     add your cloud API keys: Anthropic, OpenAI, etc.)             │"
echo "│                                                                     │"
echo "│  2. Settings → API Keys → Create key                               │"
echo "│     Copy the key (looks like: mnfst_xxxxxxxxxx)                    │"
echo "│                                                                     │"
echo "│  3. Paste the key into the shared env file:                        │"
echo "│       sudo nano /etc/brainrouter/env                               │"
echo "│     Replace the placeholder line:                                  │"
echo "│       MANIFEST_API_KEY=mnfst_REPLACE_WITH_YOUR_KEY                 │"
echo "│     with your actual key. Save and close.                          │"
echo "│                                                                     │"
echo "│  4. Restart brainrouter for all users (or just reboot):            │"
echo "│       sudo reboot                                                   │"
echo "│     OR, without rebooting, for each user:                          │"
echo "│       sudo -u USERNAME systemctl --user restart brainrouter        │"
echo "│       (set XDG_RUNTIME_DIR=/run/user/$(id -u USERNAME) first)      │"
echo "│                                                                     │"
echo "│  5. (Optional) Edit model path in llama-swap config:               │"
echo "│       sudo nano /opt/ai/llama-swap/config.yaml                     │"
echo "│     Then: sudo systemctl restart llama-swap                        │"
echo "│                                                                     │"
echo "├─────────────────────────────────────────────────────────────────────┤"
echo "│  After reboot, every user on this machine will have:               │"
echo "│    brainrouter running at http://127.0.0.1:9099                    │"
echo "│    omp (oh-my-pi) available in PATH                                │"
echo "│    All AI harnesses auto-routing via brainrouter                   │"
echo "│                                                                     │"
echo "│  Dashboard:    http://127.0.0.1:9099                               │"
echo "│  Health check: curl http://127.0.0.1:9099/health                  │"
echo "└─────────────────────────────────────────────────────────────────────┘"
echo ""
