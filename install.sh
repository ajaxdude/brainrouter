#!/usr/bin/env bash
# install.sh — one-shot brainrouter AI stack installer for Fedora Linux
#
# Run as:  sudo bash install.sh
#
# Each installation step prompts: [R]ecommended/install, [S]kip, or [A]bort.
# Steps that detect the component is already present are silently skipped.
# Harness installation is a menu at the end: pick one, several, or all.

set -euo pipefail

BRAINROUTER_REPO="https://github.com/ajaxdude/brainrouter"
BONSAI_HF_REPO="bartowski/prism-ml_Bonsai-8B-unpacked-GGUF"
BONSAI_FILE="prism-ml_Bonsai-8B-unpacked-Q4_K_M.gguf"
BONSAI_PATH="/opt/models/bonsai/${BONSAI_FILE}"
MANIFEST_PORT=3001
LLAMA_SWAP_PORT=8081
BRAINROUTER_PORT=9099
TOOLBOX_IMAGE="docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv"
TOOLBOX_NAME="llama-vulkan-radv"

SUDO_USER_HOME=$(getent passwd "${SUDO_USER:-root}" | cut -d: -f6)
BR_SRC="${SUDO_USER_HOME}/ai/projects/brainrouter"

# ── Colour helpers ─────────────────────────────────────────────────────────────

log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m !!\033[0m %s\n' "$*" >&2; }
skip() { printf '\033[1;36m --\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mERR\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\033[1;35m >>>\033[0m %s\n' "$*"; }
info() { printf '     %s\n' "$*"; }

has()     { command -v "$1" &>/dev/null; }
port_up() { ss -ltn 2>/dev/null | grep -qE "[: ]${1}\\b"; }
human_users() {
    getent passwd | awk -F: '$3 >= 1000 && $3 < 65534 && $7 !~ /nologin|false/ {print $1}'
}

# confirm_step TITLE DESCRIPTION
#   Prints the step title + description and asks the user to install, skip, or abort.
#   Returns 0 (proceed) or 1 (skip). Exits on abort.
#   If the TTY is gone (piped), defaults to recommended (install).
confirm_step() {
    local title="$1"
    local desc="$2"
    printf '\n\033[1;33m[?]\033[0m \033[1m%s\033[0m\n' "$title"
    [[ -n "$desc" ]] && printf '    %s\n' "$desc"
    printf '    \033[1m[I]\033[0minstall (recommended)  \033[1m[S]\033[0mskip  \033[1m[A]\033[0mbort\n'
    if [[ ! -t 0 ]]; then
        printf '    (non-interactive — proceeding with install)\n'
        return 0
    fi
    while true; do
        printf '    Choice [I/s/a]: '
        local reply
        IFS= read -r reply </dev/tty
        reply="${reply,,}"   # lowercase
        case "${reply:-i}" in
            i|install|'') return 0 ;;
            s|skip)        skip "Skipped: $title"; return 1 ;;
            a|abort)       die "Aborted by user." ;;
        esac
        printf '    Please enter i, s, or a.\n'
    done
}

# ── Preflight ──────────────────────────────────────────────────────────────────

[[ $EUID -eq 0 ]] || die "Run with sudo: sudo bash install.sh"
[[ -n "${SUDO_USER:-}" ]] || die "Do not run as root directly. Run: sudo bash install.sh"

printf '\n\033[1;34m%s\033[0m\n' "brainrouter AI stack installer"
printf '  Caller user : %s (%s)\n' "$SUDO_USER" "$SUDO_USER_HOME"
printf '  Human users : %s\n' "$(human_users | tr '\n' ' ')"
printf '\n  Each step will ask before doing anything.\n'
printf '  Press Enter to accept the recommended action.\n'

# ── Step 1: System packages ────────────────────────────────────────────────────

PKGS=(git golang toolbox podman docker docker-compose-plugin
      gcc-c++ cmake vulkan-headers vulkan-loader-devel libshaderc
      python3-pip curl wget unzip)

missing=()
for pkg in "${PKGS[@]}"; do
    rpm -q "$pkg" &>/dev/null || missing+=("$pkg")
done

if [[ ${#missing[@]} -eq 0 ]]; then
    skip "Step 1/14 — System packages (all already installed)"
elif confirm_step \
    "Step 1/14 — System packages" \
    "Will install via dnf: ${missing[*]}"; then
    dnf install -y "${missing[@]}"
    ok "System packages installed"
fi

if ! systemctl is-active --quiet docker; then
    if confirm_step \
        "Step 1b — Start Docker" \
        "Docker is installed but not running. Will: systemctl enable --now docker"; then
        systemctl enable --now docker
        ok "Docker running"
    fi
else
    skip "Step 1b — Docker (already running)"
fi

# ── Step 2: bun (system-wide) ─────────────────────────────────────────────────

if [[ -x /usr/local/bin/bun ]]; then
    skip "Step 2/14 — bun $(/usr/local/bin/bun --version) (already installed)"
elif confirm_step \
    "Step 2/14 — bun (JavaScript runtime, system-wide)" \
    "Required for oh-my-pi. Will download from bun.sh and install to /usr/local/bin/bun."; then
    BUN_TMP=$(mktemp -d)
    curl -fsSL "https://bun.sh/install" | BUN_INSTALL="$BUN_TMP" bash
    install -m 755 "$BUN_TMP/bin/bun" /usr/local/bin/bun
    rm -rf "$BUN_TMP"
    ok "bun $(/usr/local/bin/bun --version) installed"
fi

# ── Step 3: aistack group ─────────────────────────────────────────────────────

_users_to_add=()
while IFS= read -r u; do
    id "$u" &>/dev/null && ! id -nG "$u" | grep -qw aistack && _users_to_add+=("$u")
done < <(human_users)

if [[ ${#_users_to_add[@]} -eq 0 ]]; then
    skip "Step 3/14 — aistack group (all users already in group)"
elif confirm_step \
    "Step 3/14 — aistack group" \
    "Will: groupadd -f aistack; add users to it: ${_users_to_add[*]}"; then
    groupadd -f aistack
    for u in "${_users_to_add[@]}"; do
        usermod -aG aistack "$u"
        ok "  $u → aistack"
    done
fi

# ── Step 4: Shared directories ────────────────────────────────────────────────

if [[ -d /opt/models/bonsai && -d /opt/ai/llama-swap && -d /etc/brainrouter ]]; then
    skip "Step 4/14 — Shared directories (already exist)"
elif confirm_step \
    "Step 4/14 — Shared directories" \
    "Will create /opt/models/bonsai, /opt/ai/{llama-swap,manifest,bin}, /etc/brainrouter with aistack group permissions."; then
    mkdir -p /opt/models/bonsai /opt/ai/{llama-swap,manifest,bin} /etc/brainrouter
    chown -R root:aistack /opt/models /opt/ai /etc/brainrouter
    find /opt/models -type d -exec chmod 2775 {} \;
    find /opt/models -type f -exec chmod 664 {} \; 2>/dev/null || true
    chmod -R 775 /opt/ai
    chmod 750 /etc/brainrouter
    ok "Shared directories ready"
fi

# ── Step 5: Bonsai model ──────────────────────────────────────────────────────

if [[ -f "$BONSAI_PATH" ]]; then
    skip "Step 5/14 — Bonsai model (already at $BONSAI_PATH)"
elif confirm_step \
    "Step 5/14 — Bonsai classifier model (~5.2 GB download)" \
    "Will download prism-ml/Bonsai-8B Q4_K_M from Hugging Face to /opt/models/bonsai/.
    This is the local LLM that decides cloud vs local routing in <200ms."; then
    if ! has huggingface-cli; then
        pip3 install -q "huggingface_hub[cli]"
    fi
    huggingface-cli download "$BONSAI_HF_REPO" \
        --include "$BONSAI_FILE" \
        --local-dir /opt/models/bonsai
    chown root:aistack "$BONSAI_PATH"
    chmod 664 "$BONSAI_PATH"
    ok "Bonsai model downloaded"
fi

# ── Step 6: Manifest ──────────────────────────────────────────────────────────

MANIFEST_DIR="/opt/ai/manifest"

if [[ -f "${MANIFEST_DIR}/docker-compose.yml" ]] && systemctl is-active --quiet manifest 2>/dev/null; then
    skip "Step 6/14 — Manifest (already installed and running on :${MANIFEST_PORT})"
else
    _manifest_desc="Will:
    - Fetch docker-compose.yml from github.com/mnfst/manifest
    - Generate BETTER_AUTH_SECRET and write /opt/ai/manifest/.env
    - Install /etc/systemd/system/manifest.service (system Docker service)
    - systemctl enable --now manifest
    Manifest is the cloud LLM router (Anthropic, OpenAI, Google, etc.) on port ${MANIFEST_PORT}."
    if confirm_step "Step 6/14 — Manifest (cloud LLM router)" "$_manifest_desc"; then
        if [[ ! -f "${MANIFEST_DIR}/docker-compose.yml" ]]; then
            mkdir -p "$MANIFEST_DIR"
            curl -fsSL \
                "https://raw.githubusercontent.com/mnfst/manifest/main/docker-compose.yml" \
                -o "${MANIFEST_DIR}/docker-compose.yml"
            AUTH_SECRET=$(openssl rand -hex 32)
            cat > "${MANIFEST_DIR}/.env" <<ENV
# Generated by brainrouter install.sh
BETTER_AUTH_SECRET=${AUTH_SECRET}
BETTER_AUTH_URL=http://localhost:${MANIFEST_PORT}
ENV
            chmod 640 "${MANIFEST_DIR}/.env"
            chown root:aistack "${MANIFEST_DIR}/.env"
        fi
        chown -R root:aistack "${MANIFEST_DIR}"
        if [[ ! -f /etc/systemd/system/manifest.service ]]; then
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
        systemctl enable --now manifest
        ok "Manifest installed and started"
    fi
fi

# ── Step 7: llama-swap ────────────────────────────────────────────────────────

LSWAP_DIR="/opt/ai/llama-swap"

if [[ -f "${LSWAP_DIR}/docker-compose.yml" ]] && systemctl is-active --quiet llama-swap 2>/dev/null; then
    skip "Step 7/14 — llama-swap (already installed and running on :${LLAMA_SWAP_PORT})"
else
    _lswap_desc="Will:
    - Write /opt/ai/llama-swap/docker-compose.yml (Docker, AMD RADV Vulkan, port ${LLAMA_SWAP_PORT})
    - Write /opt/ai/llama-swap/config.yaml (placeholder — you edit model path after install)
    - Install /etc/systemd/system/llama-swap.service
    - systemctl enable --now llama-swap
    llama-swap runs local GGUF models on demand and serves them at port ${LLAMA_SWAP_PORT}."
    if confirm_step "Step 7/14 — llama-swap (local model runner)" "$_lswap_desc"; then
        if [[ ! -f "${LSWAP_DIR}/docker-compose.yml" ]]; then
            mkdir -p "$LSWAP_DIR"
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
        fi
        if [[ ! -f "${LSWAP_DIR}/config.yaml" ]]; then
            cat > "${LSWAP_DIR}/config.yaml" <<'YAML'
# llama-swap config — edit model key and path to match your GGUF file.
# Models live in /opt/models/ on this machine.
# After editing: sudo systemctl restart llama-swap

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
            warn "EDIT REQUIRED after install: /opt/ai/llama-swap/config.yaml — set your model path"
        fi
        chown -R root:aistack "${LSWAP_DIR}"
        if [[ ! -f /etc/systemd/system/llama-swap.service ]]; then
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
        systemctl enable --now llama-swap
        ok "llama-swap installed and started"
    fi
fi

# ── Step 8: brainrouter binary ────────────────────────────────────────────────

if [[ -x /usr/local/bin/brainrouter ]]; then
    skip "Step 8/14 — brainrouter binary (already at /usr/local/bin/brainrouter)"
else
    _br_desc="Will:
    - Ensure Rust toolchain for $SUDO_USER (installs via rustup if missing)
    - Clone brainrouter from GitHub if not already at $BR_SRC
    - cargo build --release (~10-15 min on first build — compiles bundled llama.cpp)
    - Install binary to /usr/local/bin/brainrouter"
    if confirm_step "Step 8/14 — Build and install brainrouter" "$_br_desc"; then
        if ! sudo -u "$SUDO_USER" bash -lc 'command -v cargo' &>/dev/null; then
            step "Installing Rust toolchain for $SUDO_USER"
            sudo -u "$SUDO_USER" bash -c \
                'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path'
        fi
        if [[ ! -d "$BR_SRC" ]]; then
            step "Cloning brainrouter"
            sudo -u "$SUDO_USER" bash -c \
                "mkdir -p $(dirname "$BR_SRC") && git clone $BRAINROUTER_REPO $BR_SRC"
        fi
        step "Building brainrouter (first build ~10-15 min)"
        sudo -u "$SUDO_USER" bash -lc \
            "cd $BR_SRC && source \$HOME/.cargo/env && cargo build --release"
        install -m 755 "${BR_SRC}/target/release/brainrouter" /usr/local/bin/brainrouter
        ok "brainrouter installed to /usr/local/bin"
    fi
fi

# ── Step 9: llama-server-toolbox wrapper ──────────────────────────────────────

if [[ -x /usr/local/bin/llama-server-toolbox ]]; then
    skip "Step 9/14 — llama-server-toolbox wrapper (already installed)"
elif confirm_step \
    "Step 9/14 — llama-server-toolbox wrapper" \
    "Will install /usr/local/bin/llama-server-toolbox — runs llama-server inside the
    llama-vulkan-radv toolbox container with AMD RADV Vulkan.
    Required by llama-swap when using toolbox mode instead of Docker."; then
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

# ── Step 10: toolbox container ────────────────────────────────────────────────

if sudo -u "$SUDO_USER" toolbox list 2>/dev/null | grep -q "$TOOLBOX_NAME"; then
    skip "Step 10/14 — toolbox container $TOOLBOX_NAME (already exists for $SUDO_USER)"
elif confirm_step \
    "Step 10/14 — toolbox container ($TOOLBOX_NAME)" \
    "Will: toolbox create --image $TOOLBOX_IMAGE $TOOLBOX_NAME
    This pulls the AMD Strix Halo RADV Vulkan image (~1 GB) for $SUDO_USER.
    Other users can create their own container by running the same command."; then
    sudo -u "$SUDO_USER" toolbox create --image "$TOOLBOX_IMAGE" "$TOOLBOX_NAME"
    ok "Toolbox $TOOLBOX_NAME created for $SUDO_USER"
fi

# ── Step 11: /etc/brainrouter/env ────────────────────────────────────────────

if [[ -f /etc/brainrouter/env ]]; then
    skip "Step 11/14 — /etc/brainrouter/env (already exists)"
elif confirm_step \
    "Step 11/14 — Shared Manifest API key file (/etc/brainrouter/env)" \
    "Will create /etc/brainrouter/env (root:aistack mode 640) with a placeholder key.
    After install you will paste your real Manifest API key into this one file —
    all users on this machine share it automatically."; then
    cat > /etc/brainrouter/env <<'ENV'
# Manifest API key — shared for all users on this machine.
# After completing the Manifest setup wizard at http://localhost:3001:
#   Settings → API Keys → Create key
# Replace the value below with your real mnfst_* key, then reboot
# (or run: sudo -u USERNAME XDG_RUNTIME_DIR=/run/user/UID systemctl --user restart brainrouter)
MANIFEST_API_KEY=mnfst_REPLACE_WITH_YOUR_KEY
ENV
    chown root:aistack /etc/brainrouter/env
    chmod 640 /etc/brainrouter/env
    ok "/etc/brainrouter/env created"
fi

# ── Step 12: /etc/brainrouter/brainrouter.yaml ───────────────────────────────

if [[ -f /etc/brainrouter/brainrouter.yaml ]]; then
    skip "Step 12/14 — /etc/brainrouter/brainrouter.yaml (already exists)"
elif confirm_step \
    "Step 12/14 — Shared brainrouter config (/etc/brainrouter/brainrouter.yaml)" \
    "Will write the shared system config pointing at Manifest (:${MANIFEST_PORT}),
    llama-swap (:${LLAMA_SWAP_PORT}), and the Bonsai model at ${BONSAI_PATH}.
    Edit fallback_model to match a key in /opt/ai/llama-swap/config.yaml."; then
    cat > /etc/brainrouter/brainrouter.yaml <<YAML
# brainrouter system config — shared for all users.
# Edit fallback_model to match a model key in /opt/ai/llama-swap/config.yaml.
# After editing: sudo -u USER XDG_RUNTIME_DIR=/run/user/UID systemctl --user restart brainrouter

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
fi

# ── Step 13: /etc/skel + per-user brainrouter service ────────────────────────

_skel_needed=0
[[ ! -f /etc/skel/.config/systemd/user/brainrouter.service ]] && _skel_needed=1
while IFS= read -r u; do
    uh=$(getent passwd "$u" | cut -d: -f6)
    [[ ! -f "${uh}/.config/systemd/user/brainrouter.service" ]] && _skel_needed=1
done < <(human_users)

if [[ $_skel_needed -eq 0 ]]; then
    skip "Step 13/14 — brainrouter.service (already seeded for all users)"
elif confirm_step \
    "Step 13/14 — brainrouter systemd user services" \
    "Will write brainrouter.service to /etc/skel and to each user's
    ~/.config/systemd/user/. The service reads from /etc/brainrouter/env
    and /etc/brainrouter/brainrouter.yaml (shared — no per-user config needed).
    loginctl enable-linger is set per user so services start at boot."; then

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

    while IFS= read -r user; do
        user_home=$(getent passwd "$user" | cut -d: -f6)
        [[ -d "$user_home" ]] || continue
        step "  Configuring $user"

        # brainrouter.yaml copy
        install -d -o "$user" -g "$user" -m 750 "${user_home}/.config/brainrouter"
        if [[ ! -f "${user_home}/.config/brainrouter/brainrouter.yaml" ]]; then
            install -o "$user" -g "$user" -m 644 \
                /etc/brainrouter/brainrouter.yaml \
                "${user_home}/.config/brainrouter/brainrouter.yaml"
        fi

        # service file
        install -d -o "$user" -g "$user" -m 755 "${user_home}/.config/systemd/user"
        if [[ ! -f "${user_home}/.config/systemd/user/brainrouter.service" ]]; then
            install -o "$user" -g "$user" -m 644 \
                /etc/skel/.config/systemd/user/brainrouter.service \
                "${user_home}/.config/systemd/user/brainrouter.service"
        fi

        # linger
        if ! loginctl show-user "$user" 2>/dev/null | grep -q 'Linger=yes'; then
            loginctl enable-linger "$user"
            ok "    linger enabled for $user"
        fi

        # enable + start
        uid=$(id -u "$user")
        if ! systemctl --user --machine="${user}@.host" is-active --quiet default.target 2>/dev/null; then
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
            && ok "    brainrouter started for $user" \
            || warn "    brainrouter not yet running for $user (starts after reboot + API key set)"
    done < <(human_users)
fi

# ── Step 14: /etc/profile.d/ai-stack.sh ──────────────────────────────────────

if [[ -f /etc/profile.d/ai-stack.sh ]]; then
    skip "Step 14/14 — /etc/profile.d/ai-stack.sh (already installed)"
elif confirm_step \
    "Step 14/14 — Shell environment (/etc/profile.d/ai-stack.sh)" \
    "Will write /etc/profile.d/ai-stack.sh which:
    - Adds ~/.bun/bin and /usr/local/bin to PATH for all users
    - Sets ANTHROPIC_BASE_URL=http://127.0.0.1:9099
    - Sets OPENAI_BASE_URL=http://127.0.0.1:9099/v1
    This makes omp, brainrouter, and all harness env vars available in every login shell."; then
    cat > /etc/profile.d/ai-stack.sh <<'PROFILE'
# AI stack — added by brainrouter install.sh
export BUN_INSTALL="$HOME/.bun"
export PATH="$BUN_INSTALL/bin:/usr/local/bin:$PATH"
export ANTHROPIC_BASE_URL="http://127.0.0.1:9099"
export OPENAI_BASE_URL="http://127.0.0.1:9099/v1"
PROFILE
    chmod 644 /etc/profile.d/ai-stack.sh
    ok "/etc/profile.d/ai-stack.sh written"
fi

# ── oh-my-pi: per-user ────────────────────────────────────────────────────────

_omp_users_missing=()
while IFS= read -r u; do
    sudo -u "$u" /usr/local/bin/bun pm ls -g 2>/dev/null \
        | grep -q '@oh-my-pi/pi-coding-agent' || _omp_users_missing+=("$u")
done < <(human_users)

if [[ ${#_omp_users_missing[@]} -eq 0 ]]; then
    skip "oh-my-pi — already installed for all users"
elif confirm_step \
    "oh-my-pi (AI coding agent, all users)" \
    "Will run: bun install -g @oh-my-pi/pi-coding-agent
    for each of: ${_omp_users_missing[*]}"; then
    for u in "${_omp_users_missing[@]}"; do
        step "  Installing oh-my-pi for $u"
        sudo -u "$u" /usr/local/bin/bun install -g @oh-my-pi/pi-coding-agent 2>&1 | tail -3
        ok "  oh-my-pi installed for $u"
    done
fi

# ── Harness installation ───────────────────────────────────────────────────────
#
# brainrouter install <harness> patches harness config files to point at
# brainrouter on 127.0.0.1:9099. Run as the individual user (not root).
# We ask the admin which harnesses to install, then run them for every human user.

log "Harness installation"
printf '
  brainrouter can wire up AI coding harnesses so they all route through it.
  Select which harnesses to install for each user on this machine.

  Harnesses:
    1) omp          — oh-my-pi (recommended)
    2) claude       — Claude Code  (sets ANTHROPIC_BASE_URL)
    3) vibe         — Vibe
    4) opencode     — OpenCode
    5) codex        — Codex
    6) droid        — Droid  (Anthropic protocol)
    7) all          — Install all of the above

  Enter numbers separated by spaces (e.g. "1 2"), "7" for all, or "s" to skip.
  Press Enter for the recommended default: omp only (1).

'

HARNESSES_TO_INSTALL=()

if [[ ! -t 0 ]]; then
    printf '  (non-interactive — installing omp by default)\n'
    HARNESSES_TO_INSTALL=(omp)
else
    printf '  Choice [1-7 / s]: '
    IFS= read -r harness_reply </dev/tty
    harness_reply="${harness_reply:-1}"  # default: 1 (omp)

    case "${harness_reply,,}" in
        s|skip) printf '  Skipping harness installation.\n' ;;
        7|all|all\ *)
            HARNESSES_TO_INSTALL=(omp claude vibe opencode codex droid) ;;
        *)
            for tok in $harness_reply; do
                case "$tok" in
                    1) HARNESSES_TO_INSTALL+=(omp) ;;
                    2) HARNESSES_TO_INSTALL+=(claude) ;;
                    3) HARNESSES_TO_INSTALL+=(vibe) ;;
                    4) HARNESSES_TO_INSTALL+=(opencode) ;;
                    5) HARNESSES_TO_INSTALL+=(codex) ;;
                    6) HARNESSES_TO_INSTALL+=(droid) ;;
                    *) warn "  Unknown selection '$tok' — ignoring" ;;
                esac
            done
            ;;
    esac
fi

if [[ ${#HARNESSES_TO_INSTALL[@]} -gt 0 ]]; then
    # Deduplicate
    readarray -t HARNESSES_TO_INSTALL < <(printf '%s\n' "${HARNESSES_TO_INSTALL[@]}" | sort -u)

    printf '  Installing harnesses: %s\n' "${HARNESSES_TO_INSTALL[*]}"

    while IFS= read -r user; do
        user_home=$(getent passwd "$user" | cut -d: -f6)
        [[ -d "$user_home" ]] || continue
        [[ -x /usr/local/bin/brainrouter ]] || { warn "  brainrouter binary not found — skipping harness install for $user"; continue; }
        step "  Harnesses for $user"
        for h in "${HARNESSES_TO_INSTALL[@]}"; do
            _args=(install "$h" --yes)
            # claude needs --shell-rc to inject ANTHROPIC_BASE_URL into the user's shell RC
            [[ "$h" == "claude" ]] && _args+=(--shell-rc "${user_home}/.bashrc")
            sudo -u "$user" \
                HOME="$user_home" \
                /usr/local/bin/brainrouter "${_args[@]}" 2>&1 \
                && ok "    $h configured for $user" \
                || warn "    $h install failed for $user (harness may not be installed)"
        done
    done < <(human_users)
fi

# ── Summary ───────────────────────────────────────────────────────────────────

BR_OK=0;   [[ -x /usr/local/bin/brainrouter ]] && BR_OK=1
LSWAP_OK=0; port_up $LLAMA_SWAP_PORT && LSWAP_OK=1
MFT_OK=0;  curl -sf "http://localhost:${MANIFEST_PORT}/api/v1/health" &>/dev/null && MFT_OK=1
BSI_OK=0;  [[ -f "$BONSAI_PATH" ]] && BSI_OK=1

echo ""
echo "┌─────────────────────────────────────────────────────────────────────┐"
echo "│  brainrouter AI stack — install complete                            │"
echo "├─────────────────────────────────────────────────────────────────────┤"
printf "│  brainrouter  %-54s│\n" "$([[ $BR_OK   -eq 1 ]] && echo 'installed (/usr/local/bin/brainrouter)' || echo 'NOT installed — was step 8 skipped?')"
printf "│  llama-swap   %-54s│\n" "$([[ $LSWAP_OK -eq 1 ]] && echo "running on :${LLAMA_SWAP_PORT}" || echo 'enabled — starts at boot')"
printf "│  Manifest     %-54s│\n" "$([[ $MFT_OK  -eq 1 ]] && echo "running on :${MANIFEST_PORT}"  || echo 'enabled — starts at boot')"
printf "│  Bonsai       %-54s│\n" "$([[ $BSI_OK  -eq 1 ]] && echo "$BONSAI_PATH" || echo 'NOT found — was step 5 skipped?')"
echo "├─────────────────────────────────────────────────────────────────────┤"
echo "│                                                                     │"
echo "│  *** ONE MANUAL STEP REQUIRED: ***                                  │"
echo "│                                                                     │"
echo "│  1. Open http://localhost:3001 in your browser.                    │"
echo "│     Complete the setup wizard (create admin account, add your      │"
echo "│     cloud API keys: Anthropic, OpenAI, Google, etc.)               │"
echo "│                                                                     │"
echo "│  2. Settings → API Keys → Create key                               │"
echo "│     Copy the key (looks like: mnfst_xxxxxxxxxx)                    │"
echo "│                                                                     │"
echo "│  3. Paste into the shared key file:                                │"
echo "│       sudo nano /etc/brainrouter/env                               │"
echo "│     Replace:  MANIFEST_API_KEY=mnfst_REPLACE_WITH_YOUR_KEY         │"
echo "│     with your real key. Save and close.                            │"
echo "│                                                                     │"
echo "│  4. Reboot — every user comes up with brainrouter running.         │"
echo "│     Or restart per-user without rebooting:                         │"
echo "│       sudo -u USERNAME \\                                           │"
echo "│         XDG_RUNTIME_DIR=/run/user/\$(id -u USERNAME) \\             │"
echo "│         systemctl --user restart brainrouter                       │"
echo "│                                                                     │"
echo "│  5. Edit local model path (optional):                              │"
echo "│       sudo nano /opt/ai/llama-swap/config.yaml                     │"
echo "│       sudo systemctl restart llama-swap                            │"
echo "│                                                                     │"
echo "├─────────────────────────────────────────────────────────────────────┤"
echo "│  After reboot every user has:                                       │"
echo "│    brainrouter  →  http://127.0.0.1:9099                           │"
echo "│    Dashboard    →  http://127.0.0.1:9099                           │"
echo "│    Health       →  curl http://127.0.0.1:9099/health               │"
echo "└─────────────────────────────────────────────────────────────────────┘"
echo ""
