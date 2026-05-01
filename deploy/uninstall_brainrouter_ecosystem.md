# brainrouter Ecosystem Uninstall Guide

This document covers complete removal of the brainrouter AI stack. It is designed for both human operators and AI agents. Each step is independent and can be skipped if that component was never installed.

## Important Notes

- Uninstall in reverse order: brainrouter first, then llama-swap, then Manifest.
- Model files are preserved by default (they are large downloads). Pass `--remove-models` or follow the explicit model removal steps if you want them deleted.
- Multi-user uninstall requires both per-user cleanup AND system-level cleanup.
- All commands are idempotent (safe to re-run).

---

## Single-User Uninstall

### Step 1: Stop and Remove brainrouter Service

```bash
# Stop and disable the service
systemctl --user stop brainrouter 2>/dev/null
systemctl --user disable brainrouter 2>/dev/null

# Remove service file
rm -f ~/.config/systemd/user/brainrouter.service
systemctl --user daemon-reload

# Remove UDS socket
rm -f /run/user/$(id -u)/brainrouter.sock
```

### Step 2: Remove brainrouter Config

```bash
rm -rf ~/.config/brainrouter/
```

### Step 3: Remove brainrouter Binary

```bash
# If installed system-wide
sudo rm -f /usr/local/bin/brainrouter

# If running from build directory, just delete the clone:
# rm -rf ~/ai/projects/brainrouter
```

### Step 4: Remove Harness Integrations

brainrouter install patches harness configs. Undo each one:

```bash
# Claude Code: remove MCP server
claude mcp remove brainrouter --scope user 2>/dev/null

# Remove ANTHROPIC_BASE_URL from shell rc
sed -i '/ANTHROPIC_BASE_URL.*127.0.0.1:9099/d' ~/.zshrc ~/.bashrc 2>/dev/null
sed -i '/ANTHROPIC_AUTH_TOKEN.*not-used/d' ~/.zshrc ~/.bashrc 2>/dev/null

# OMP: remove brainrouter from mcp.json
if [ -f ~/.omp/agent/mcp.json ]; then
    # Remove the brainrouter key from mcpServers
    python3 -c "
import json, sys
with open('$HOME/.omp/agent/mcp.json', 'r') as f:
    data = json.load(f)
if 'mcpServers' in data and 'brainrouter' in data['mcpServers']:
    del data['mcpServers']['brainrouter']
with open('$HOME/.omp/agent/mcp.json', 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null
fi

# OMP: remove brainrouter from models.yml (manual)
# Edit ~/.omp/agent/models.yml and remove the brainrouter provider section

# OpenCode: remove brainrouter from config.json
if [ -f ~/.config/opencode/config.json ]; then
    python3 -c "
import json
with open('$HOME/.config/opencode/config.json', 'r') as f:
    data = json.load(f)
for key in ['provider', 'mcp']:
    if key in data and 'brainrouter' in data[key]:
        del data[key]['brainrouter']
with open('$HOME/.config/opencode/config.json', 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null
fi

# Droid: remove brainrouter from mcp.json
if [ -f ~/.factory/mcp.json ]; then
    python3 -c "
import json
with open('$HOME/.factory/mcp.json', 'r') as f:
    data = json.load(f)
if 'mcpServers' in data and 'brainrouter' in data['mcpServers']:
    del data['mcpServers']['brainrouter']
with open('$HOME/.factory/mcp.json', 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null
fi
```

### Step 5: Stop and Remove llama-swap

```bash
# Binary mode: stop service and remove binary
systemctl --user stop llama-swap 2>/dev/null
systemctl --user disable llama-swap 2>/dev/null
rm -f ~/.config/systemd/user/llama-swap.service
systemctl --user daemon-reload
rm -f ~/.local/bin/llama-swap
rm -f ~/.local/bin/llama-server-toolbox

# Docker mode: stop and remove container
if [ -f ~/ai/stack/llama-swap/docker-compose.yml ]; then
    docker compose -f ~/ai/stack/llama-swap/docker-compose.yml down -v
fi
rm -rf ~/ai/stack/llama-swap

# Remove config
rm -rf ~/.config/llama-swap
```

### Step 6: Remove Toolbox Container (if used)

```bash
toolbox rm -f llama-vulkan-radv 2>/dev/null
```

### Step 7: Stop and Remove Manifest

```bash
if [ -f ~/ai/stack/manifest/docker-compose.yml ]; then
    cd ~/ai/stack/manifest
    docker compose down -v   # -v removes volumes including database
fi
rm -rf ~/ai/stack/manifest
```

### Step 8: Remove Models (Optional)

Models are large downloads. Only remove if you are sure you do not need them.

```bash
# Remove Bonsai classifier
rm -f ~/models/bonsai/prism-ml_Bonsai-8B-unpacked-Q4_K_M.gguf

# Remove all models
# rm -rf ~/models/
```

### Step 9: Remove Rust Toolchain (Optional)

Only if it was installed solely for brainrouter.

```bash
rustup self uninstall
```

---

## Multi-User Uninstall

Multi-user uninstall has two phases: per-user cleanup (each user runs this) and system cleanup (admin runs once).

### Phase 1: Per-User Cleanup (each user runs)

Each user should run Steps 1-4 from the single-user uninstall above:

```bash
# Stop brainrouter
systemctl --user stop brainrouter 2>/dev/null
systemctl --user disable brainrouter 2>/dev/null
rm -f ~/.config/systemd/user/brainrouter.service
systemctl --user daemon-reload
rm -f /run/user/$(id -u)/brainrouter.sock

# Remove config
rm -rf ~/.config/brainrouter/

# Remove harness integrations (see Step 4 above)
claude mcp remove brainrouter --scope user 2>/dev/null
sed -i '/ANTHROPIC_BASE_URL.*127.0.0.1:9099/d' ~/.zshrc ~/.bashrc 2>/dev/null
sed -i '/ANTHROPIC_AUTH_TOKEN.*not-used/d' ~/.zshrc ~/.bashrc 2>/dev/null
```

### Phase 2: System Cleanup (admin runs once)

```bash
# 1. Stop and remove system services
sudo systemctl stop llama-swap 2>/dev/null
sudo systemctl disable llama-swap 2>/dev/null
sudo rm -f /etc/systemd/system/llama-swap.service

sudo systemctl stop manifest 2>/dev/null
sudo systemctl disable manifest 2>/dev/null
sudo rm -f /etc/systemd/system/manifest.service

sudo systemctl daemon-reload

# 2. Remove Docker containers and volumes
if [ -f /opt/ai/llama-swap/docker-compose.yml ]; then
    sudo docker compose -f /opt/ai/llama-swap/docker-compose.yml down -v
fi

if [ -f /opt/ai/manifest/docker-compose.yml ]; then
    sudo docker compose -f /opt/ai/manifest/docker-compose.yml down -v
fi

# 3. Remove brainrouter binary
sudo rm -f /usr/local/bin/brainrouter

# 4. Remove /opt/ai directory
sudo rm -rf /opt/ai

# 5. Remove /etc/skel templates
sudo rm -f /etc/skel/.config/brainrouter/brainrouter.yaml
sudo rm -f /etc/skel/.config/brainrouter/.env
sudo rm -f /etc/skel/.config/systemd/user/brainrouter.service
sudo rm -f /etc/skel/.local/bin/llama-server-toolbox

# 6. Remove profile.d script (if installed)
sudo rm -f /etc/profile.d/ai-stack.sh

# 7. Remove aistack group (optional)
sudo groupdel aistack 2>/dev/null

# 8. Remove shared models (optional - these are large)
# sudo rm -rf /opt/models

# 9. Clean up per-user config for all users (if they haven't done it)
for user in $(getent passwd | awk -F: '$3 >= 1000 && $3 < 65534 {print $1}'); do
    home=$(getent passwd "$user" | cut -d: -f6)
    [ -z "$home" ] && continue
    rm -rf "$home/.config/brainrouter" 2>/dev/null
    rm -f "$home/.config/systemd/user/brainrouter.service" 2>/dev/null
    rm -f "$home/.local/bin/llama-server-toolbox" 2>/dev/null
done
```

---

## Agent Uninstall Checklist

For AI agents performing uninstall:

1. **Determine deployment mode**: Check for system services (`systemctl is-active llama-swap` and `systemctl is-active manifest`) to distinguish multi-user from single-user.

2. **Ask before removing models**: Models are large downloads (5-50+ GB). Always confirm before deleting `/opt/models` or `~/models`.

3. **Order of operations**: brainrouter -> harness integrations -> llama-swap -> Manifest -> models (optional)

4. **Detection commands** (to check what is installed):
   ```bash
   systemctl --user is-active brainrouter 2>/dev/null
   systemctl is-active llama-swap 2>/dev/null
   systemctl is-active manifest 2>/dev/null
   test -f /usr/local/bin/brainrouter && echo "binary installed"
   test -d /opt/ai && echo "multi-user layout exists"
   test -d ~/.config/brainrouter && echo "user config exists"
   ```

5. **Verify removal**:
   ```bash
   curl -sf http://127.0.0.1:9099/health 2>/dev/null && echo "WARN: brainrouter still responding" || echo "brainrouter: removed"
   curl -sf http://localhost:8081/v1/models 2>/dev/null && echo "WARN: llama-swap still responding" || echo "llama-swap: removed"
   curl -sf http://localhost:2099/api/v1/health 2>/dev/null && echo "WARN: manifest still responding" || echo "manifest: removed"
   ```

---

## What Is NOT Removed

The following are intentionally preserved unless explicitly requested:

- **Docker itself**: Docker may be used by other services
- **Rust toolchain**: May be used for other projects
- **System packages** (vulkan-headers, cmake, etc.): May be used by other software
- **huggingface-cli**: May be used for other model downloads
- **Model files**: Large downloads that take time to re-acquire
- **Docker images**: Run `docker image prune` to clean up unused images
