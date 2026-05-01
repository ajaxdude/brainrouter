# brainrouter Ecosystem Deployment Guide

This document is designed for both human operators and AI agents deploying the brainrouter AI stack on AMD GPU Linux systems. It covers single-user and multi-user deployment, is idempotent (safe to re-run), and detects components that are already installed.

## Stack Overview

```
coding harness (omp / claude / vibe / opencode / codex / droid)
       |
       v
 brainrouter :9099          <- per-user systemd user service
   |  Bonsai-8B classifies each query (<200ms)
   |--cloud--> Manifest :2099    <- cloud LLM router (Docker)
   +--local--> llama-swap :8081  <- local model runner (Docker or Go binary)
                   |
                   v
            llama-server         <- runs inside Docker with AMD RADV Vulkan
                   |
                   v
             /opt/models         <- shared GGUF model storage (multi-user)
             ~/models            <- user GGUF model storage (single-user)
```

brainrouter embeds Bonsai via the llama-cpp-2 Rust crate (not via llama-swap). llama-swap is only for the non-classifier models the user wants to run locally.

## Port Map

| Service | Port | Protocol |
|---------|------|----------|
| llama-swap | 8081 | OpenAI-compatible REST |
| Manifest | 2099 | OpenAI-compatible REST |
| brainrouter | 9099 | OpenAI + Anthropic REST |
| brainrouter UDS | /run/user/$UID/brainrouter.sock | Unix socket (MCP) |

## Hardware Requirements

- AMD GPU with RADV Vulkan driver (RDNA 2+, Strix Halo integrated GPU recommended)
- /dev/dri/renderD128 must be readable by the user running brainrouter
- Linux with systemd (Fedora 40+ recommended)
- 16 GB+ GPU memory minimum; 64 GB+ for multiple simultaneous models
- For Strix Halo / unified memory systems, add kernel parameters:
  ```
  iommu=pt amdgpu.gttsize=126976 ttm.pages_limit=32505856
  ```
  Add to /etc/kernel/cmdline or grub config to unlock up to 124 GB of unified memory for GPU use.

## Prerequisites

### Software
- Docker with Compose v2 plugin
- Rust toolchain (rustup + cargo)
- git, cmake, gcc-c++
- Vulkan development packages
- huggingface-cli (for model download)

### Detection Commands (for agents)

Before installing any component, check if it already exists:

```bash
# Docker
command -v docker && docker compose version

# Rust toolchain
command -v cargo && cargo --version

# Vulkan dev packages (Fedora)
rpm -q vulkan-headers vulkan-loader-devel libshaderc glslc cmake gcc-c++

# Vulkan dev packages (Ubuntu/Debian)
dpkg -l libvulkan-dev glslang-tools cmake g++

# huggingface-cli
command -v huggingface-cli

# brainrouter binary
command -v brainrouter || test -f /usr/local/bin/brainrouter

# llama-swap (Docker mode)
docker ps --filter name=llama-swap --format '{{.Names}}' 2>/dev/null

# llama-swap (binary mode)
command -v llama-swap

# Manifest
docker ps --filter name=manifest --format '{{.Names}}' 2>/dev/null
curl -sf http://localhost:2099/api/v1/health 2>/dev/null

# Bonsai model
test -f /opt/models/bonsai/prism-ml_Bonsai-8B-unpacked-Q4_K_M.gguf ||
  find ~/models/bonsai -name '*.gguf' 2>/dev/null

# brainrouter service
systemctl --user is-active brainrouter 2>/dev/null
```

---

## Deployment Mode: Single-User

Use this when one person owns the machine. Everything runs as systemd user services. No root required except for system package installation.

### Step 1: Install System Packages

```bash
# Fedora
sudo dnf install -y gcc-c++ vulkan-headers vulkan-loader-devel libshaderc glslc cmake git

# Ubuntu/Debian
sudo apt install -y build-essential libvulkan-dev glslang-tools cmake git
```

### Step 2: Install Rust Toolchain

Skip if `cargo --version` succeeds.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

### Step 3: Install huggingface-cli

Skip if `command -v huggingface-cli` succeeds.

```bash
pip install -U "huggingface_hub[cli]"
```

### Step 4: Download Bonsai Classifier Model

Skip if a Bonsai GGUF file already exists at the target path.

```bash
mkdir -p ~/models/bonsai
huggingface-cli download bartowski/prism-ml_Bonsai-8B-unpacked-GGUF \
  --include "prism-ml_Bonsai-8B-unpacked-Q4_K_M.gguf" \
  --local-dir ~/models/bonsai

# Verify
ls -lh ~/models/bonsai/prism-ml_Bonsai-8B-unpacked-Q4_K_M.gguf
```

The Q4_K_M quant (~5.2 GB) is the recommended balance of speed and size. Use Q6_K_L (~7.3 GB) for higher accuracy if VRAM allows.

### Step 5: Build and Install brainrouter

Skip build if `brainrouter --help` already works and you do not need to update.

```bash
git clone https://github.com/ajaxdude/brainrouter ~/ai/projects/brainrouter
cd ~/ai/projects/brainrouter
cargo build --release    # Takes 10-15 min first time (compiles bundled llama.cpp)

# Option A: Run from build directory (no install needed)
# Option B: Install system-wide
sudo cp target/release/brainrouter /usr/local/bin/brainrouter
sudo chmod 755 /usr/local/bin/brainrouter
```

### Step 6: Install llama-swap (Binary Mode)

Skip if llama-swap is already running or installed.

**Option A: Go binary** (simpler for single-user)
```bash
# Requires Go >= 1.22
go install github.com/mostlygeek/llama-swap@latest
cp ~/go/bin/llama-swap ~/.local/bin/llama-swap
```

**Option B: Docker** (better GPU isolation)
```bash
# Create docker-compose.yml for llama-swap
mkdir -p ~/ai/stack/llama-swap
cat > ~/ai/stack/llama-swap/docker-compose.yml << 'YAML'
services:
  llama-swap:
    image: ghcr.io/mostlygeek/llama-swap:latest
    ports:
      - "127.0.0.1:8081:8080"
    volumes:
      - ./config.yaml:/config.yaml:ro
      - ~/models:/models:ro
      - /dev/dri:/dev/dri
    devices:
      - /dev/dri:/dev/dri
    environment:
      - AMD_VULKAN_ICD=RADV
    restart: unless-stopped
    command: ["--config", "/config.yaml", "--listen", "0.0.0.0:8080"]
YAML
```

Create a minimal llama-swap config:

```bash
cat > ~/ai/stack/llama-swap/config.yaml << 'YAML'
startPort: 5800
healthCheckTimeout: 300
globalTTL: 180

macros:
  "common": >-
    --no-webui --jinja
    -t 8 -tb 16 --parallel 1
    -ngl 999 --no-mmap -fa on
    --host 0.0.0.0

models:
  "your-local-model":
    name: "Your Local Model"
    cmd: >
      llama-server --port ${PORT} ${common}
        --model /models/path/to/your-model.gguf
YAML
```

**Important:** Replace `your-local-model` and the model path with your actual local model. The model key is what you reference as `fallback_model` in brainrouter.yaml.

### Step 7: Create llama-swap Toolbox (Binary Mode Only)

Skip if using Docker mode for llama-swap. Skip if the toolbox already exists.

```bash
# Check if toolbox exists
toolbox list 2>/dev/null | grep -q llama-vulkan-radv && echo "Already exists" && exit 0

# Create toolbox with AMD Vulkan support
toolbox create --image docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv llama-vulkan-radv

# Verify llama-server is available
toolbox run --container llama-vulkan-radv llama-server --version

# Create wrapper script
mkdir -p ~/.local/bin
cat > ~/.local/bin/llama-server-toolbox << 'SCRIPT'
#!/bin/bash
CONTAINER=${LLAMA_CONTAINER:-llama-vulkan-radv}
ICD=${LLAMA_ICD:-RADV}
exec toolbox run --container "$CONTAINER" env AMD_VULKAN_ICD="$ICD" llama-server "$@"
SCRIPT
chmod +x ~/.local/bin/llama-server-toolbox
```

Update your llama-swap config.yaml macros to use the wrapper:
```yaml
macros:
  "ls": "/home/$USER/.local/bin/llama-server-toolbox"
```

### Step 8: Install Manifest

Skip if `curl -sf http://localhost:2099/api/v1/health` succeeds.

```bash
mkdir -p ~/ai/stack/manifest
bash <(curl -sSL https://raw.githubusercontent.com/mnfst/manifest/main/docker/install.sh) \
    --dir ~/ai/stack/manifest --yes
```

Wait for Manifest to become healthy, then complete the setup wizard:

**MANUAL STEP (cannot be automated):**
1. Open http://localhost:2099 in a browser
2. Complete the setup wizard (create admin account)
3. Add cloud provider API keys (Anthropic, OpenAI, Google, etc.)
4. Go to Settings -> API Keys -> Create key
5. Copy the key (looks like `mnfst_xxxxxxxx`)

### Step 9: Configure brainrouter

```bash
mkdir -p ~/.config/brainrouter

# Create config file
cat > ~/.config/brainrouter/brainrouter.yaml << 'YAML'
manifest:
  base_url: "http://localhost:2099/v1"
  api_key_env: MANIFEST_API_KEY

llama_swap:
  base_url: "http://localhost:8081/v1"
  fallback_model: "your-local-model"  # Must match a key in your llama-swap config

bonsai:
  model_path: "/home/$USER/models/bonsai/prism-ml_Bonsai-8B-unpacked-Q4_K_M.gguf"
YAML

# IMPORTANT: Replace $USER with your actual username in model_path above
sed -i "s|/home/\$USER|/home/$USER|g" ~/.config/brainrouter/brainrouter.yaml

# Create env file with Manifest API key
cat > ~/.config/brainrouter/.env << 'ENV'
MANIFEST_API_KEY=mnfst_your_key_here
ENV
chmod 600 ~/.config/brainrouter/.env
```

**MANUAL STEP:** Edit ~/.config/brainrouter/.env and paste your actual Manifest API key.

### Step 10: Create brainrouter systemd User Service

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/brainrouter.service << 'SERVICE'
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

systemctl --user daemon-reload
systemctl --user enable --now brainrouter
```

### Step 11: Verify

```bash
# Check brainrouter is running
curl -sf http://127.0.0.1:9099/health
# Expected: {"status":"ok"}

# Check llama-swap
curl -sf http://localhost:8081/v1/models

# Check Manifest
curl -sf http://localhost:2099/api/v1/health
```

### Step 12: Connect Harnesses

```bash
# Connect harnesses (idempotent - run for each harness you use)
brainrouter install claude --shell-rc
brainrouter install omp
brainrouter install opencode
brainrouter install vibe
brainrouter install codex
brainrouter install droid
```

Reload your shell or run `source ~/.zshrc` to pick up environment variable changes.

---

## Deployment Mode: Multi-User

Use this when multiple users share a machine. Manifest and llama-swap run as system services (root). Each user runs their own brainrouter instance.

Requires root (sudo) access for the system-level setup.

### Step 1: System Packages

Same as single-user Step 1.

### Step 2: Create aistack Group

```bash
sudo groupadd -f aistack

# Add all human users who should have access
for user in $(getent passwd | awk -F: '$3 >= 1000 && $3 < 65534 {print $1}'); do
    sudo usermod -aG aistack "$user"
    echo "Added $user to aistack"
done

# Users must log out and back in for group changes to take effect
```

### Step 3: Create Shared Directory Structure

```bash
sudo mkdir -p /opt/models/bonsai
sudo mkdir -p /opt/ai/{llama-swap,manifest,bin}
sudo chown -R root:aistack /opt/models /opt/ai
sudo chmod -R 2775 /opt/models   # setgid so new files inherit group
sudo chmod -R 775 /opt/ai
```

### Step 4: Download Bonsai Model to Shared Location

Skip if `/opt/models/bonsai/prism-ml_Bonsai-8B-unpacked-Q4_K_M.gguf` exists.

```bash
sudo huggingface-cli download bartowski/prism-ml_Bonsai-8B-unpacked-GGUF \
  --include "prism-ml_Bonsai-8B-unpacked-Q4_K_M.gguf" \
  --local-dir /opt/models/bonsai

sudo chown root:aistack /opt/models/bonsai/prism-ml_Bonsai-8B-unpacked-Q4_K_M.gguf
sudo chmod 664 /opt/models/bonsai/prism-ml_Bonsai-8B-unpacked-Q4_K_M.gguf
```

### Step 5: Build and Install brainrouter

```bash
# Build (as any user with Rust toolchain)
cd ~/ai/projects/brainrouter   # or wherever you cloned it
cargo build --release

# Install system-wide
sudo cp target/release/brainrouter /usr/local/bin/brainrouter
sudo chmod 755 /usr/local/bin/brainrouter
```

### Step 6: Deploy llama-swap as System Service

Skip if `systemctl is-active llama-swap` succeeds.

```bash
# Create llama-swap config
sudo tee /opt/ai/llama-swap/docker-compose.yml > /dev/null << 'YAML'
services:
  llama-swap:
    image: ghcr.io/mostlygeek/llama-swap:latest
    ports:
      - "8081:8080"
    volumes:
      - ./config.yaml:/config.yaml:ro
      - /opt/models:/models:ro
      - /dev/dri:/dev/dri
    devices:
      - /dev/dri:/dev/dri
    environment:
      - AMD_VULKAN_ICD=RADV
    restart: unless-stopped
    command: ["--config", "/config.yaml", "--listen", "0.0.0.0:8080"]
YAML

sudo tee /opt/ai/llama-swap/config.yaml > /dev/null << 'YAML'
startPort: 5800
healthCheckTimeout: 300
globalTTL: 180

macros:
  "common": >-
    --no-webui --jinja
    -t 8 -tb 16 --parallel 1
    -ngl 999 --no-mmap -fa on
    --host 0.0.0.0

models:
  "your-local-model":
    name: "Your Local Model"
    cmd: >
      llama-server --port ${PORT} ${common}
        --model /models/path/to/your-model.gguf
YAML

sudo chown -R root:aistack /opt/ai/llama-swap
sudo chmod 664 /opt/ai/llama-swap/{config.yaml,docker-compose.yml}

# Create systemd service
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
sudo systemctl enable --now llama-swap
```

### Step 7: Deploy Manifest as System Service

Skip if `curl -sf http://localhost:2099/api/v1/health` succeeds.

```bash
# Install Manifest
if [ ! -f /opt/ai/manifest/docker-compose.yml ]; then
    bash <(curl -sSL https://raw.githubusercontent.com/mnfst/manifest/main/docker/install.sh) \
        --dir /opt/ai/manifest --yes
fi

sudo chown -R root:aistack /opt/ai/manifest
sudo chmod 750 /opt/ai/manifest/.env 2>/dev/null || true

# Create systemd service
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
sudo systemctl enable --now manifest
```

**MANUAL STEP:** Open http://localhost:2099 and complete setup wizard. See single-user Step 8.

### Step 8: Seed Per-User Config Templates

```bash
# Install into /etc/skel for new users
sudo install -Dm644 /dev/stdin /etc/skel/.config/brainrouter/brainrouter.yaml << 'YAML'
manifest:
  base_url: "http://localhost:2099/v1"
  api_key_env: MANIFEST_API_KEY

llama_swap:
  base_url: "http://localhost:8081/v1"
  fallback_model: "your-local-model"

bonsai:
  model_path: "/opt/models/bonsai/prism-ml_Bonsai-8B-unpacked-Q4_K_M.gguf"
YAML

sudo install -Dm600 /dev/stdin /etc/skel/.config/brainrouter/.env << 'ENV'
MANIFEST_API_KEY=mnfst_your_key_here
ENV

sudo install -Dm644 /dev/stdin /etc/skel/.config/systemd/user/brainrouter.service << 'SERVICE'
[Unit]
Description=brainrouter
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
```

### Step 9: Apply Config to Existing Users

```bash
for user in $(getent group aistack | cut -d: -f4 | tr ',' ' '); do
    home=$(getent passwd "$user" | cut -d: -f6)
    [ -z "$home" ] && continue

    # brainrouter config (skip if exists)
    if [ ! -f "$home/.config/brainrouter/brainrouter.yaml" ]; then
        sudo install -o "$user" -g "$user" -m755 -d "$home/.config/brainrouter"
        sudo install -o "$user" -g "$user" -m644 \
            /etc/skel/.config/brainrouter/brainrouter.yaml \
            "$home/.config/brainrouter/brainrouter.yaml"
        sudo install -o "$user" -g "$user" -m600 \
            /etc/skel/.config/brainrouter/.env \
            "$home/.config/brainrouter/.env"
        echo "Seeded brainrouter config for $user"
    fi

    # systemd user service (skip if exists)
    if [ ! -f "$home/.config/systemd/user/brainrouter.service" ]; then
        sudo install -o "$user" -g "$user" -m755 -d "$home/.config/systemd/user"
        sudo install -o "$user" -g "$user" -m644 \
            /etc/skel/.config/systemd/user/brainrouter.service \
            "$home/.config/systemd/user/brainrouter.service"
        echo "Seeded brainrouter service for $user"
    fi
done
```

### Step 10: Per-User Activation

Each user runs this themselves (cannot be done by root):

```bash
# 1. Edit Manifest API key
$EDITOR ~/.config/brainrouter/.env
# Paste: MANIFEST_API_KEY=mnfst_your_actual_key

# 2. Enable brainrouter
systemctl --user daemon-reload
systemctl --user enable --now brainrouter

# 3. Verify
curl -sf http://127.0.0.1:9099/health
# Expected: {"status":"ok"}

# 4. Connect harnesses
brainrouter install claude --shell-rc
brainrouter install omp
# ... other harnesses as needed

# 5. (Optional) Enable linger for boot-time startup without login
sudo loginctl enable-linger $USER
```

---

## Adding Models to llama-swap

1. Download a GGUF model:
   ```bash
   # Single-user: ~/models/
   huggingface-cli download <repo> --include "<filename>.gguf" --local-dir ~/models/<category>

   # Multi-user: /opt/models/
   sudo huggingface-cli download <repo> --include "<filename>.gguf" --local-dir /opt/models/<category>
   sudo chown root:aistack /opt/models/<category>/<filename>.gguf
   sudo chmod 664 /opt/models/<category>/<filename>.gguf
   ```

2. Add entry to llama-swap config.yaml:
   ```yaml
   models:
     "model-key":
       name: "Human-readable name"
       cmd: >
         llama-server --port ${PORT} ${common}
           --model /models/<category>/<filename>.gguf
   ```

3. Restart llama-swap:
   ```bash
   # System service
   sudo systemctl restart llama-swap

   # Or Docker directly
   docker compose -f /opt/ai/llama-swap/docker-compose.yml restart
   ```

4. Optionally update `fallback_model` in `~/.config/brainrouter/brainrouter.yaml`

---

## Manual Steps That Cannot Be Automated

These steps require human interaction and cannot be performed by an agent:

1. **Manifest Setup Wizard**: Open http://localhost:2099 in a browser after first install. Create admin account and add cloud provider API keys (Anthropic, OpenAI, Google, etc.).

2. **Manifest API Key**: After setup, go to Settings -> API Keys -> Create key. Copy the `mnfst_*` key and paste into `~/.config/brainrouter/.env`.

3. **Kernel Parameters** (Strix Halo only): Editing `/etc/kernel/cmdline` or grub config requires understanding the boot loader setup and a reboot.

4. **Group Membership**: After adding users to `aistack` group, they must log out and back in for changes to take effect.

5. **Cloud Provider API Keys**: Each cloud provider (Anthropic, OpenAI, etc.) requires its own API key obtained from the provider's website and entered into Manifest's UI.

---

## Troubleshooting

### brainrouter won't start
```bash
# Check logs
journalctl --user -u brainrouter -n 50

# Common issues:
# - Bonsai model file not found: check model_path in brainrouter.yaml
# - Vulkan not available: ensure AMD_VULKAN_ICD=RADV is set and /dev/dri is accessible
# - Port already in use: another brainrouter or service on :9099
```

### llama-swap not responding
```bash
# System service mode
sudo systemctl status llama-swap
sudo docker compose -f /opt/ai/llama-swap/docker-compose.yml logs

# Check GPU access
ls -la /dev/dri/renderD128
```

### Manifest not healthy
```bash
sudo systemctl status manifest
sudo docker compose -f /opt/ai/manifest/docker-compose.yml logs
curl -v http://localhost:2099/api/v1/health
```

### Vulkan errors during brainrouter build
```bash
# Verify Vulkan packages
vulkaninfo 2>/dev/null | head -5

# On Fedora, ensure all packages:
sudo dnf install vulkan-headers vulkan-loader-devel libshaderc glslc
```

---

## Agent Deployment Checklist

For AI agents performing deployment, follow this decision tree:

1. **Detect deployment mode**: Ask the user if this is single-user or multi-user. If multiple human users exist on the system (`getent passwd | awk -F: '$3 >= 1000 && $3 < 65534' | wc -l` > 1), suggest multi-user.

2. **For each component**, run the detection command first. Only install if not detected.

3. **Order of operations**:
   a. System packages (sudo)
   b. Rust toolchain
   c. huggingface-cli
   d. Bonsai model download
   e. brainrouter build and install
   f. llama-swap (Docker or binary)
   g. Manifest (Docker)
   h. brainrouter config + systemd service
   i. Verify all services healthy
   j. Connect harnesses

4. **Pause for manual steps**: After installing Manifest, you MUST tell the user to complete the setup wizard and create an API key. Do not proceed with brainrouter verification until the user confirms they have set the API key.

5. **Verification**: After deployment, run:
   ```bash
   curl -sf http://127.0.0.1:9099/health && echo "brainrouter: OK"
   curl -sf http://localhost:8081/v1/models && echo "llama-swap: OK"
   curl -sf http://localhost:2099/api/v1/health && echo "manifest: OK"
   ```

6. **Report**: Tell the user what was installed, what was skipped (already present), and what manual steps remain.
