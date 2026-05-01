#!/usr/bin/env bash
# deploy.sh — deploy the brainrouter AI stack
#
# Usage:
#   bash deploy.sh                    # interactive single-user install
#   bash deploy.sh --multi-user       # multi-user install (requires sudo)
#   bash deploy.sh --skip-manifest    # skip Manifest install
#   bash deploy.sh --skip-llama-swap  # skip llama-swap install
#   bash deploy.sh --skip-bonsai      # skip Bonsai model download
#   bash deploy.sh --yes              # non-interactive mode
#
# This script is idempotent: it detects existing components and skips them.

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────
MANIFEST_PORT=2099
LLAMA_SWAP_PORT=8081
BRAINROUTER_PORT=9099
BONSAI_REPO="bartowski/prism-ml_Bonsai-8B-unpacked-GGUF"
BONSAI_FILE="prism-ml_Bonsai-8B-unpacked-Q4_K_M.gguf"
BRAINROUTER_REPO="https://github.com/ajaxdude/brainrouter"

# ── Flags ─────────────────────────────────────────────────────────────
MULTI_USER=0
SKIP_MANIFEST=0
SKIP_LLAMA_SWAP=0
SKIP_BONSAI=0
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --multi-user)      MULTI_USER=1; shift ;;
    --skip-manifest)   SKIP_MANIFEST=1; shift ;;
    --skip-llama-swap) SKIP_LLAMA_SWAP=1; shift ;;
    --skip-bonsai)     SKIP_BONSAI=1; shift ;;
    --yes|-y)          ASSUME_YES=1; shift ;;
    -h|--help)         sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                 echo "Unknown flag: $1"; exit 1 ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────
log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*" >&2; }
skip() { printf '\033[1;36m--\033[0m  %s (already present)\n' "$*"; }
die()  { printf '\033[1;31mxx\033[0m  %s\n' "$*" >&2; exit 1; }

confirm() {
    [[ "$ASSUME_YES" -eq 1 ]] && return 0
    read -rp "$1 [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]]
}

has_cmd()  { command -v "$1" &>/dev/null; }
is_port()  { ss -ltn 2>/dev/null | grep -qE "[: ]$1\\b"; }

MODEL_DIR="$HOME/models"
CONFIG_DIR="$HOME/.config/brainrouter"
if [[ $MULTI_USER -eq 1 ]]; then
    MODEL_DIR="/opt/models"
fi

# ── 1. System packages ────────────────────────────────────────────────
log "Checking system packages"

install_fedora_pkgs() {
    local missing=()
    for pkg in gcc-c++ vulkan-headers vulkan-loader-devel libshaderc cmake git; do
        rpm -q "$pkg" &>/dev/null || missing+=("$pkg")
    done
    # glslc is provided by libshaderc or glslc package
    has_cmd glslc || missing+=(glslc)

    if [[ ${#missing[@]} -gt 0 ]]; then
        log "Installing: ${missing[*]}"
        sudo dnf install -y "${missing[@]}"
    else
        skip "All system packages"
    fi
}

install_debian_pkgs() {
    local missing=()
    for pkg in build-essential libvulkan-dev glslang-tools cmake git; do
        dpkg -l "$pkg" &>/dev/null 2>&1 || missing+=("$pkg")
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        log "Installing: ${missing[*]}"
        sudo apt install -y "${missing[@]}"
    else
        skip "All system packages"
    fi
}

if has_cmd dnf; then
    install_fedora_pkgs
elif has_cmd apt; then
    install_debian_pkgs
else
    warn "Unknown package manager. Ensure gcc, cmake, vulkan headers, glslc are installed."
fi

# ── 2. Rust toolchain ─────────────────────────────────────────────────
log "Checking Rust toolchain"
if has_cmd cargo; then
    skip "Rust ($(cargo --version))"
else
    log "Installing Rust toolchain"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    ok "Rust installed"
fi

# ── 3. huggingface-cli ────────────────────────────────────────────────
log "Checking huggingface-cli"
if has_cmd huggingface-cli; then
    skip "huggingface-cli"
else
    log "Installing huggingface-cli"
    pip install -U "huggingface_hub[cli]" || pip3 install -U "huggingface_hub[cli]"
    ok "huggingface-cli installed"
fi

# ── 4. Docker ─────────────────────────────────────────────────────────
log "Checking Docker"
if has_cmd docker && docker compose version &>/dev/null; then
    skip "Docker ($(docker --version | head -1))"
else
    die "Docker with Compose v2 is required. Install: https://docs.docker.com/get-docker/"
fi

# ── 5. Bonsai model ───────────────────────────────────────────────────
BONSAI_PATH="$MODEL_DIR/bonsai/$BONSAI_FILE"

if [[ $SKIP_BONSAI -eq 0 ]]; then
    log "Checking Bonsai classifier model"
    if [[ -f "$BONSAI_PATH" ]]; then
        skip "Bonsai model at $BONSAI_PATH"
    else
        log "Downloading Bonsai classifier (~5.2 GB)"
        if [[ $MULTI_USER -eq 1 ]]; then
            sudo mkdir -p "$MODEL_DIR/bonsai"
            sudo huggingface-cli download "$BONSAI_REPO" \
                --include "$BONSAI_FILE" \
                --local-dir "$MODEL_DIR/bonsai"
            sudo chown root:aistack "$BONSAI_PATH"
            sudo chmod 664 "$BONSAI_PATH"
        else
            mkdir -p "$MODEL_DIR/bonsai"
            huggingface-cli download "$BONSAI_REPO" \
                --include "$BONSAI_FILE" \
                --local-dir "$MODEL_DIR/bonsai"
        fi
        ok "Bonsai model downloaded"
    fi
else
    skip "Bonsai model (--skip-bonsai)"
fi

# ── 6. brainrouter build ──────────────────────────────────────────────
log "Checking brainrouter"
BR_SRC="$HOME/ai/projects/brainrouter"
BR_BIN="$BR_SRC/target/release/brainrouter"

if has_cmd brainrouter; then
    skip "brainrouter binary ($(which brainrouter))"
elif [[ -f "$BR_BIN" ]]; then
    skip "brainrouter binary at $BR_BIN"
    log "Installing to /usr/local/bin"
    sudo cp "$BR_BIN" /usr/local/bin/brainrouter
    sudo chmod 755 /usr/local/bin/brainrouter
    ok "brainrouter installed"
else
    if [[ ! -d "$BR_SRC" ]]; then
        log "Cloning brainrouter"
        mkdir -p "$(dirname "$BR_SRC")"
        git clone "$BRAINROUTER_REPO" "$BR_SRC"
    fi
    log "Building brainrouter (this takes 10-15 minutes on first build)"
    cd "$BR_SRC"
    cargo build --release
    sudo cp "$BR_BIN" /usr/local/bin/brainrouter
    sudo chmod 755 /usr/local/bin/brainrouter
    ok "brainrouter built and installed"
fi

# ── 7. Multi-user setup ───────────────────────────────────────────────
if [[ $MULTI_USER -eq 1 ]]; then
    log "Setting up multi-user infrastructure"

    # aistack group
    sudo groupadd -f aistack
    for user in $(getent passwd | awk -F: '$3 >= 1000 && $3 < 65534 {print $1}'); do
        if ! id -nG "$user" | grep -qw aistack; then
            sudo usermod -aG aistack "$user"
            ok "$user added to aistack"
        fi
    done

    # Shared directories
    sudo mkdir -p /opt/ai/{llama-swap,manifest,bin}
    sudo chown -R root:aistack /opt/ai /opt/models 2>/dev/null || true
    find /opt/models -type d -exec sudo chmod 2775 {} \; 2>/dev/null || true
    find /opt/models -type f -exec sudo chmod 664 {} \; 2>/dev/null || true
    sudo chmod -R 775 /opt/ai
fi

# ── 8. llama-swap ─────────────────────────────────────────────────────
if [[ $SKIP_LLAMA_SWAP -eq 0 ]]; then
    log "Checking llama-swap"
    if is_port $LLAMA_SWAP_PORT; then
        skip "llama-swap (port $LLAMA_SWAP_PORT active)"
    elif [[ $MULTI_USER -eq 1 ]]; then
        warn "llama-swap docker-compose.yml and config.yaml need to be created at /opt/ai/llama-swap/"
        warn "See deploy/brainrouter_ecosystem.md Step 6 for the Docker setup."
        warn "After creating configs: sudo systemctl enable --now llama-swap"
    else
        if has_cmd llama-swap; then
            skip "llama-swap binary ($(which llama-swap))"
        else
            warn "llama-swap not found. Install options:"
            warn "  Go binary: go install github.com/mostlygeek/llama-swap@latest"
            warn "  Docker: see deploy/brainrouter_ecosystem.md Step 6"
        fi
    fi
else
    skip "llama-swap (--skip-llama-swap)"
fi

# ── 9. Manifest ───────────────────────────────────────────────────────
if [[ $SKIP_MANIFEST -eq 0 ]]; then
    log "Checking Manifest"
    if curl -sf http://localhost:$MANIFEST_PORT/api/v1/health &>/dev/null; then
        skip "Manifest (port $MANIFEST_PORT healthy)"
    else
        MANIFEST_DIR="$HOME/ai/stack/manifest"
        [[ $MULTI_USER -eq 1 ]] && MANIFEST_DIR="/opt/ai/manifest"

        if [[ -f "$MANIFEST_DIR/docker-compose.yml" ]]; then
            skip "Manifest directory exists at $MANIFEST_DIR"
            log "Starting Manifest..."
            if [[ $MULTI_USER -eq 1 ]]; then
                sudo docker compose -f "$MANIFEST_DIR/docker-compose.yml" up -d
            else
                docker compose -f "$MANIFEST_DIR/docker-compose.yml" up -d
            fi
        else
            log "Installing Manifest to $MANIFEST_DIR"
            if [[ $MULTI_USER -eq 1 ]]; then
                sudo bash <(curl -sSL https://raw.githubusercontent.com/mnfst/manifest/main/docker/install.sh) \
                    --dir "$MANIFEST_DIR" --yes
                sudo chown -R root:aistack "$MANIFEST_DIR"
                sudo chmod 750 "$MANIFEST_DIR/.env" 2>/dev/null || true
            else
                bash <(curl -sSL https://raw.githubusercontent.com/mnfst/manifest/main/docker/install.sh) \
                    --dir "$MANIFEST_DIR" --yes
            fi
            ok "Manifest installed"
        fi

        echo ""
        echo "============================================================"
        echo "  MANUAL STEP REQUIRED"
        echo "============================================================"
        echo "  1. Open http://localhost:$MANIFEST_PORT in your browser"
        echo "  2. Complete the setup wizard (create admin account)"
        echo "  3. Add cloud provider API keys"
        echo "  4. Settings -> API Keys -> Create key"
        echo "  5. Copy the mnfst_* key for the next step"
        echo "============================================================"
        echo ""
    fi
else
    skip "Manifest (--skip-manifest)"
fi

# ── 10. brainrouter config ────────────────────────────────────────────
log "Checking brainrouter config"

if [[ -f "$CONFIG_DIR/brainrouter.yaml" ]]; then
    skip "brainrouter config at $CONFIG_DIR/brainrouter.yaml"
else
    log "Creating brainrouter config"
    mkdir -p "$CONFIG_DIR"

    cat > "$CONFIG_DIR/brainrouter.yaml" << YAML
manifest:
  base_url: "http://localhost:$MANIFEST_PORT/v1"
  api_key_env: MANIFEST_API_KEY

llama_swap:
  base_url: "http://localhost:$LLAMA_SWAP_PORT/v1"
  fallback_model: "your-local-model"

bonsai:
  model_path: "$BONSAI_PATH"
YAML
    ok "Config created at $CONFIG_DIR/brainrouter.yaml"
    warn "Edit fallback_model to match your llama-swap model key"
fi

if [[ ! -f "$CONFIG_DIR/.env" ]]; then
    cat > "$CONFIG_DIR/.env" << 'ENV'
MANIFEST_API_KEY=mnfst_your_key_here
ENV
    chmod 600 "$CONFIG_DIR/.env"
    warn "Edit $CONFIG_DIR/.env and set your Manifest API key"
elif ! grep -q 'mnfst_[a-zA-Z0-9]' "$CONFIG_DIR/.env" 2>/dev/null; then
    warn "$CONFIG_DIR/.env still has placeholder API key. Edit it with your real key."
fi

# ── 11. brainrouter systemd service ───────────────────────────────────
log "Checking brainrouter systemd service"
SERVICE_FILE="$HOME/.config/systemd/user/brainrouter.service"

if [[ -f "$SERVICE_FILE" ]]; then
    skip "brainrouter.service exists"
else
    mkdir -p "$(dirname "$SERVICE_FILE")"
    cat > "$SERVICE_FILE" << 'SERVICE'
[Unit]
Description=brainrouter - Bonsai-routed LLM proxy
After=network-online.target

[Service]
Type=simple
EnvironmentFile=%h/.config/brainrouter/.env
Environment=AMD_VULKAN_ICD=RADV
ExecStart=/usr/local/bin/brainrouter serve \
  --config %h/.config/brainrouter/brainrouter.yaml \
  --socket /run/user/%U/brainrouter.sock \
  --tcp-addr 127.0.0.1:9099
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
SERVICE
    ok "brainrouter.service created"
fi

systemctl --user daemon-reload

# ── 12. Multi-user system services ────────────────────────────────────
if [[ $MULTI_USER -eq 1 ]]; then
    if [[ ! -f /etc/systemd/system/llama-swap.service ]] && [[ $SKIP_LLAMA_SWAP -eq 0 ]]; then
        sudo tee /etc/systemd/system/llama-swap.service > /dev/null << 'SERVICE'
[Unit]
Description=llama-swap - on-demand shared LLM model router
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/ai/llama-swap
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down --timeout 30
TimeoutStartSec=120
User=root

[Install]
WantedBy=multi-user.target
SERVICE
        sudo systemctl daemon-reload
        sudo systemctl enable llama-swap
        ok "llama-swap.service installed"
    fi

    if [[ ! -f /etc/systemd/system/manifest.service ]] && [[ $SKIP_MANIFEST -eq 0 ]]; then
        sudo tee /etc/systemd/system/manifest.service > /dev/null << 'SERVICE'
[Unit]
Description=Manifest - cloud LLM router
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/ai/manifest
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down --timeout 30
TimeoutStartSec=180
User=root

[Install]
WantedBy=multi-user.target
SERVICE
        sudo systemctl daemon-reload
        sudo systemctl enable manifest
        ok "manifest.service installed"
    fi

    # Seed /etc/skel
    log "Seeding /etc/skel"
    sudo install -Dm644 "$CONFIG_DIR/brainrouter.yaml" /etc/skel/.config/brainrouter/brainrouter.yaml 2>/dev/null || true
    sudo install -Dm600 "$CONFIG_DIR/.env" /etc/skel/.config/brainrouter/.env 2>/dev/null || true
    sudo install -Dm644 "$SERVICE_FILE" /etc/skel/.config/systemd/user/brainrouter.service 2>/dev/null || true
fi

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Deployment summary"
echo "============================================================"
echo ""

has_cmd brainrouter && echo "  brainrouter: installed" || echo "  brainrouter: NOT found"
is_port $LLAMA_SWAP_PORT && echo "  llama-swap:  running on :$LLAMA_SWAP_PORT" || echo "  llama-swap:  NOT running"
is_port $MANIFEST_PORT && echo "  Manifest:    running on :$MANIFEST_PORT" || echo "  Manifest:    NOT running"
[[ -f "$BONSAI_PATH" ]] && echo "  Bonsai:      $BONSAI_PATH" || echo "  Bonsai:      NOT found"

echo ""
echo "  Next steps:"
echo "    1. Edit $CONFIG_DIR/.env (set MANIFEST_API_KEY)"
echo "    2. Edit $CONFIG_DIR/brainrouter.yaml (set fallback_model)"
echo "    3. systemctl --user enable --now brainrouter"
echo "    4. curl http://127.0.0.1:$BRAINROUTER_PORT/health"
echo "    5. brainrouter install claude --shell-rc"
echo "============================================================"
