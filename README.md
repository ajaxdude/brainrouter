# brainrouter

<p align="center">
  <img src="assets/brainrouter-logo.svg" alt="brainrouter logo" width="600">
</p>

A speed-first Rust proxy that sits between your AI coding harness and your LLMs. A local 8B classifier (Bonsai) decides in under 200 ms whether each request goes to cloud (via Manifest) or local inference (via llama-swap). Automatic fallback, system-prompt rewriting for local models, and an MCP-triggered iterative code-review loop that runs entirely on your own hardware.

```
coding harness (omp / claude / vibe / opencode / codex / droid)
        │
        ▼
  brainrouter :9099
        │
  ┌─────┼──────────────────────────────────────────────────┐
  │     ├─ model=auto  → Bonsai classifies query           │
  │     │    Cloud ──── Manifest :3001                     │
  │     │    Local ──── llama-swap :8081                   │
  │     ├─ model=local → rewrite prompt → llama-swap       │
  │     └─ model=cloud → Manifest (direct)                 │
  └─────────────────────────────────────────────────────────┘
        │
  on Manifest fail ───── llama-swap fallback_model
        │
  on task complete ────── review loop → local LLM → dashboard
```

- **One endpoint, all harnesses.** OpenAI-compatible on `POST /v1/chat/completions`. Anthropic-compatible on `POST /v1/messages`. Every harness connects to the same `:9099`.
- **Three routing modes.** `auto` uses Bonsai classification (<200 ms). `local` rewrites the system prompt and goes straight to llama-swap. `cloud` goes straight to Manifest.
- **Local prompt rewriting.** OMP's 15–20 K token system prompt overwhelms small local models. Local mode replaces it with a lean ~500 token prompt with anti-loop directives.
- **Manifest handles cloud failover.** Manifest runs locally in Docker and picks the right cloud provider (Anthropic, OpenAI, Copilot, Google, Mistral, DeepSeek, etc.) with its own automatic fallbacks.
- **MCP code review.** `mcp_brainrouter_request_review` triggers an iterative review loop (up to 5 rounds by default). The review LLM reads your PRD, git diff, and task summary, then either approves or gives actionable feedback.
- **Dashboard.** Live routing feed, review session list, version display, one-click upgrades and service restarts — all at `http://127.0.0.1:9099`.

---

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Install dependencies](#install-dependencies)
   - [1. llama.cpp inside a toolbox container](#1-llamacpp-inside-a-toolbox-container)
   - [2. llama-swap](#2-llama-swap)
   - [3. Manifest](#3-manifest)
   - [4. Bonsai classifier model](#4-bonsai-classifier-model)
3. [Install brainrouter](#install-brainrouter)
4. [Configure](#configure)
5. [Connect your harness](#connect-your-harness)
6. [Dashboard guide](#dashboard-guide)
7. [MCP code review guide](#mcp-code-review-guide)
8. [Reference](#reference)

---

## Prerequisites

- Linux (systemd user services, `/run/user/$UID`)
- Rust toolchain: `curl https://sh.rustup.rs | sh`
- Go ≥ 1.22 (for llama-swap): `sudo dnf install golang` or `sudo apt install golang`
- Docker / Podman (for Manifest)
- [toolbox](https://containertoolbx.org/) (`sudo dnf install toolbox`)
- A GPU with Vulkan support (AMD RDNA or NVIDIA; the llama-server toolbox ships Vulkan drivers)

---

## Install dependencies

### 1. llama.cpp inside a toolbox container

llama-swap launches `llama-server` for inference. brainrouter wraps it in a toolbox container so the GPU drivers stay isolated.

**a. Create the toolbox container**

For AMD RDNA (RADV driver — recommended for RX 7000 / RX 8000 / Strix Halo):
```bash
toolbox create --image docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv llama-vulkan-radv
```

For AMD with the AMDVLK driver:
```bash
toolbox create --image docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-amdvlk llama-vulkan-amdvlk
```

Verify `llama-server` is available inside:
```bash
toolbox run --container llama-vulkan-radv llama-server --version
```

**b. Create the wrapper script**

llama-swap calls this script instead of `llama-server` directly so it can route to the right container:

```bash
cat > ~/.local/bin/llama-server-toolbox << 'EOF'
#!/bin/bash
# Wrapper: run llama-server inside a toolbox container.
# LLAMA_CONTAINER defaults to llama-vulkan-radv.
# LLAMA_ICD defaults to RADV.
CONTAINER=${LLAMA_CONTAINER:-llama-vulkan-radv}
ICD=${LLAMA_ICD:-RADV}
exec toolbox run --container "$CONTAINER" env AMD_VULKAN_ICD="$ICD" llama-server "$@"
EOF
chmod +x ~/.local/bin/llama-server-toolbox
```

---

### 2. llama-swap

llama-swap loads models on demand, unloads the previous one, and presents a single OpenAI-compatible endpoint.

**a. Install**
```bash
go install github.com/mostlygeek/llama-swap@latest
cp ~/go/bin/llama-swap ~/.local/bin/llama-swap
```

**b. Configure**

Create `~/.config/llama-swap/config.yaml`. Minimal example:

```yaml
# ~/.config/llama-swap/config.yaml
startPort: 5800
healthCheckTimeout: 300
globalTTL: 180

macros:
  "ls": "/home/$USER/.local/bin/llama-server-toolbox"
  "ctx": "-c 32768"
  "common": >-
    --no-webui --jinja
    -t 8 -tb 16 --parallel 1
    -ngl 999 --no-mmap -fa on
    --host 0.0.0.0
    ${ctx}

models:
  "my-model":
    name: "My Model"
    cmd: |
      ${ls}
      --port ${PORT}
      ${common}
      --model /path/to/model.gguf
```

Replace `/path/to/model.gguf` with the path to a GGUF file on your system. The model key (`"my-model"`) is what you reference in `brainrouter.yaml` as `fallback_model`.

**c. Systemd user service**

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/llama-swap.service << 'EOF'
[Unit]
Description=llama-swap — on-demand local model router

[Service]
Type=simple
ExecStart=%h/.local/bin/llama-swap \
    --config %h/.config/llama-swap/config.yaml \
    --listen 0.0.0.0:8081
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now llama-swap
systemctl --user status llama-swap
```

Test it:
```bash
curl http://localhost:8081/v1/models
```

---

### 3. Manifest

Manifest is a self-hosted cloud LLM router. It runs in Docker and handles provider selection (Anthropic, OpenAI, Google, Mistral, DeepSeek, Copilot, etc.) with its own fallback logic. brainrouter delegates all cloud requests to it.

**a. Get the docker-compose stack**

```bash
mkdir -p ~/ai/stack/manifest
cd ~/ai/stack/manifest
curl -fsSL https://raw.githubusercontent.com/mnfst/manifest/main/docker-compose.yml -o docker-compose.yml
```

**b. Configure**

```bash
cp .env.example .env 2>/dev/null || touch .env
# Set a secret — required:
echo "BETTER_AUTH_SECRET=$(openssl rand -hex 32)" >> .env
```

**c. Start**

```bash
cd ~/ai/stack/manifest
docker compose up -d
```

Visit `http://localhost:3001` and complete the setup wizard (create admin account, add cloud provider API keys).

**d. Get your Manifest API key**

After setup: Settings → API Keys → Create key. It looks like `mnfst_xxxxxxxx`.

```bash
# Add to your brainrouter env file (created in the next step):
echo "MANIFEST_API_KEY=mnfst_your_key_here" >> ~/ai/projects/brainrouter/.env
```

---

### 4. Bonsai classifier model

Bonsai 8B is an 8-billion-parameter GGUF model that classifies each query as "cloud" or "local" in under 200 ms. Download it from Hugging Face:

```bash
# Install huggingface-cli if needed:
pip install huggingface_hub

# Download only the recommended Q6_K_L quant (~6 GB):
# Using --include with the exact filename avoids pulling all quants (can be 50+ GB).
huggingface-cli download prism-ml/Bonsai-8B \
  --include "*Q6_K_L*.gguf" \
  --local-dir ~/models/bonsai

# Verify the file is present and note the exact path for brainrouter.yaml:
ls ~/models/bonsai/*.gguf
```

Any Q4 or Q6 quant works. Q6_K_L is the recommended balance of speed and accuracy.
Use Q4_K_M if you are short on VRAM.


---

## Install brainrouter

```bash
# 1. Clone
git clone https://github.com/yourusername/brainrouter ~/ai/projects/brainrouter
cd ~/ai/projects/brainrouter

# 2. Build
cargo build --release

# 3. Create env file
cp .env.example .env
# Edit .env and set MANIFEST_API_KEY=mnfst_your_key_here
```

**Systemd user service**

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/brainrouter.service << 'EOF'
[Unit]
Description=brainrouter — Bonsai-routed LLM proxy
After=llama-swap.service network-online.target
Wants=llama-swap.service

[Service]
Type=simple
WorkingDirectory=%h/ai/projects/brainrouter
EnvironmentFile=%h/ai/projects/brainrouter/.env
ExecStart=%h/ai/projects/brainrouter/target/release/brainrouter serve \
    --config %h/ai/projects/brainrouter/brainrouter.yaml \
    --socket /run/user/%U/brainrouter.sock \
    --tcp-addr 127.0.0.1:9099
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now brainrouter
systemctl --user status brainrouter
```

Verify:
```bash
curl http://127.0.0.1:9099/health
# → {"status":"ok"}
```

---

## Configure

Copy the example config and fill in your paths:

```bash
cp brainrouter.example.yaml brainrouter.yaml
```

```yaml
# brainrouter.yaml

manifest:
  base_url: "http://localhost:3001/v1"
  api_key_env: MANIFEST_API_KEY   # env var name — value goes in .env

llama_swap:
  base_url: "http://localhost:8081/v1"
  # Must match a model key in ~/.config/llama-swap/config.yaml
  fallback_model: "my-model"

bonsai:
  # Absolute path to the Bonsai GGUF file you downloaded
  model_path: "/home/yourname/models/bonsai/Bonsai-8B-Q6_K_L.gguf"

# Optional — all fields have sensible defaults:
# review:
#   max_iterations: 5
#   context:
#     prd_paths: [docs/PRD.md, PRD.md, README.md]
#     include_git_diff: true
```

After editing, restart:
```bash
systemctl --user restart brainrouter
```

---

## Connect your harness

brainrouter includes an `install` subcommand that patches your harness config automatically.

```bash
# Auto-install (patches config files in place, asks for confirmation):
./target/release/brainrouter install omp
./target/release/brainrouter install claude --shell-rc
./target/release/brainrouter install vibe
./target/release/brainrouter install opencode
./target/release/brainrouter install codex
./target/release/brainrouter install droid

# Skip confirmation prompt:
./target/release/brainrouter install omp --yes
```

### Manual snippets

#### omp

```yaml
# ~/.omp/agent/models.yml — add under providers:
providers:
  brainrouter:
    baseUrl: http://127.0.0.1:9099/v1
    api: openai-completions
    auth: none
    models:
      - id: auto
        name: Brainrouter (auto)
      - id: local
        name: Brainrouter (local)
      - id: cloud
        name: Brainrouter (cloud)
```

```json
// ~/.omp/agent/mcp.json
{
  "mcpServers": {
    "brainrouter": {
      "type": "stdio",
      "command": "/home/yourname/ai/projects/brainrouter/target/release/brainrouter",
      "args": ["mcp", "--socket", "/run/user/$(id -u)/brainrouter.sock"],
      "timeout": 300000
    }
  }
}
```

#### Claude Code

```bash
# Register MCP tool:
brainrouter install claude --shell-rc

# Or manually:
claude mcp add-json brainrouter '{
  "type": "stdio",
  "command": "/path/to/brainrouter",
  "args": ["mcp", "--socket", "/run/user/$(id -u)/brainrouter.sock"]
}' --scope user

# Route Claude Code through brainrouter (add to ~/.zshrc):
export ANTHROPIC_BASE_URL=http://127.0.0.1:9099
export ANTHROPIC_AUTH_TOKEN=not-used
```

#### vibe

```toml
# Append to ~/.vibe/config.toml
[[providers]]
name = "brainrouter"
api_base = "http://127.0.0.1:9099/v1"
api_style = "openai"
backend = "generic"

[[models]]
name = "brainrouter-auto"
provider = "brainrouter"
alias = "auto"

mcp_servers = [
  { name = "brainrouter", command = "/path/to/brainrouter", args = ["mcp", "--socket", "/run/user/$(id -u)/brainrouter.sock"] },
]
```

#### opencode

```json
// Merge into ~/.config/opencode/config.json
{
  "provider": {
    "brainrouter": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Brainrouter",
      "options": { "baseURL": "http://127.0.0.1:9099/v1" },
      "models": { "auto": { "model": "auto", "name": "Brainrouter (auto)" } }
    }
  },
  "mcp": {
    "brainrouter": {
      "type": "local",
      "command": ["/path/to/brainrouter", "mcp", "--socket", "/run/user/$(id -u)/brainrouter.sock"]
    }
  }
}
```

#### codex

```toml
# ~/.codex/config.toml
model = "auto"
model_provider = "brainrouter"

[model_providers.brainrouter]
name = "Brainrouter"
base_url = "http://127.0.0.1:9099/v1"

[mcp_servers.brainrouter]
command = "/path/to/brainrouter"
args = ["mcp", "--socket", "/run/user/$(id -u)/brainrouter.sock"]
```

#### droid (factory.ai)

```json
// ~/.factory/mcp.json
{
  "custom_models": [{
    "model": "brainrouter-auto",
    "base_url": "http://127.0.0.1:9099/v1",
    "api_key": "not-used",
    "provider": "anthropic"
  }],
  "mcpServers": {
    "brainrouter": {
      "type": "stdio",
      "command": "/path/to/brainrouter",
      "args": ["mcp", "--socket", "/run/user/$(id -u)/brainrouter.sock"]
    }
  }
}
```

> **Note:** `provider: "anthropic"` is required for droid. Droid's `openai` mode posts to `/responses` (not served here). `anthropic` mode posts to `/v1/messages`, which brainrouter handles.


---

## Dashboard guide

Open **`http://127.0.0.1:9099`** in a browser. The dashboard auto-refreshes every 3 seconds.

### Live routing flow

The top panel shows the most recent request as it moves through the pipeline:

```
harness → Bonsai classify → [Cloud: Manifest] or [Local: llama-swap] → response
```

Each stage shows:
- **Bonsai decision** — `cloud` or `local` badge
- **Provider** — which upstream handled it
- **Model** — the model key that was used
- **Latency** — end-to-end time in ms
- **Fallback indicator** ↩ — appears when Manifest failed and llama-swap handled it instead

### Routing events feed

The table below the flow panel shows the last 50 routing events, deduplicated:

- Identical requests within a 30-second window are collapsed into a single row with a `×N` badge and cumulative latency.
- Review iterations within the same session collapse into one row with an `iter N` badge.
- Hover the **Prompt** cell to see the full prompt excerpt.
- The **Folder** badge shows which project directory the request came from.

### Version header and upgrades

The header row shows current installed versions of:

- **llama-swap** — the local model router binary
- **llama.cpp** — the llama-server build inside the toolbox container
- **Manifest** — the running Docker container (image date · short hash)
- **toolbox** — the OCI image version label

When a newer version is available (checked against GitHub / Docker Hub on each poll), an orange **`component → new-version`** button appears. Click it to upgrade. Each button is labelled so you know exactly what will be updated.

### Service controls (nav bar)

Four restart buttons in the top nav:

| Button | What it does |
|---|---|
| **Restart llama-swap** | `systemctl --user restart llama-swap` |
| **Restart llama.cpp** | Refreshes the toolbox container (runs configured restart script) |
| **Restart Manifest** | `docker compose restart manifest` |
| **Restart brainrouter** | `systemctl --user restart brainrouter` — page reloads after 3 s |

### Review sessions tab

Click **Review Sessions** in the nav to see the session list:

- Each row shows task ID, status (`pending` / `approved` / `needs_revision` / `escalated`), iteration count, reviewer type (LLM or human), and timestamps.
- Click a row to open the session detail view with the full conversation history.
- If a session is `escalated` (LLM couldn't resolve it after max iterations), a **Resolve** panel appears — type your feedback and submit to close the loop.

### Review config panel

A collapsible panel on the dashboard lets you control how code reviews run:

| Setting | Options | Effect |
|---|---|---|
| **Review mode** | Auto / Force Cloud / Force Local | Auto lets Bonsai decide; Force overrides for all reviews in this session |
| **Local model** | dropdown of llama-swap models | When forcing local, which specific model to use |

Changes take effect immediately for new review requests. The setting persists across daemon restarts.

---

## MCP code review guide

The review tool is exposed over MCP so any harness can call it after completing a task.

### How it works

1. Your harness calls `mcp_brainrouter_request_review` with a task ID and summary.
2. brainrouter gathers context: your project's PRD (auto-detected from `docs/PRD.md`, `PRD.md`, or `README.md`), the current `git diff HEAD`, and any `AGENTS.md`.
3. A review prompt is assembled and sent to the configured LLM (cloud or local, depending on review mode).
4. The LLM responds with `STATUS: approved` or `STATUS: needs_revision` plus feedback.
5. If `needs_revision`, the harness implements the feedback and calls `mcp_brainrouter_request_review` again. Up to 5 iterations.
6. After 5 failed iterations (or an LLM error), the session escalates to human review at `http://127.0.0.1:9099/review/`.

### Tool parameters

| Parameter | Required | Description |
|---|---|---|
| `taskId` | yes | Unique ID for this task, e.g. `feature-20260424-001` |
| `summary` | yes | 2–3 sentences: what changed, why, and any assumptions |
| `details` | no | Additional technical context |
| `conversationHistory` | no | Array of strings — recent conversation for context |
| `cwd` | no (strongly recommended) | Absolute path to the project directory — required for accurate git diff; falls back to peer-cred-resolved cwd if omitted |

### Calling the review tool (agent instruction)

If you are an LLM agent completing a task in a project, add this to your workflow:

```
After completing all work, call mcp_brainrouter_request_review with:
  taskId:  "<type>-<YYYYMMDD>-<seq>"  (e.g. feature-20260424-001)
  summary: "<2–3 sentences: what changed, why, assumptions>"
  cwd:     "<absolute path to the project root>"
  details: "<optional extra context, changed files, security notes>"

If the response status is "needs_revision", read the feedback, fix the issues,
then call mcp_brainrouter_request_review again. Repeat until "approved".
Do not consider the task complete until you receive status: "approved".
```

### MCP tools reference

| Tool | Parameters | Description |
|---|---|---|
| `request_review` | `taskId`, `summary`, `cwd?`, `details?`, `conversationHistory?` | Start or continue a review |
| `get_session_list` | — | List all review sessions |
| `get_session_details` | `sessionId` | Full detail for one session |
| `resolve_session` | `sessionId`, `feedback` | Human resolves: "lgtm"/"ok"/"approved" → approved; any other text → needs_revision |

---

## Reference

### brainrouter.yaml — full options

```yaml
manifest:
  base_url: "http://localhost:3001/v1"   # required
  api_key_env: MANIFEST_API_KEY          # optional — name of env var holding mnfst_* key

llama_swap:
  base_url: "http://localhost:8081/v1"   # required
  fallback_model: "my-model"             # required — must match a key in llama-swap config
  local_system_prompt: "/path/to/prompt.md"   # optional — override built-in lean prompt
  llama_cpp_restart_script: "/path/to/script.sh"  # optional — used by Restart llama.cpp button

bonsai:
  model_path: "/path/to/Bonsai-8B-Q6_K_L.gguf"  # required — absolute path

review:
  max_iterations: 5           # LLM review rounds before escalating to human
  retry_on_llm_error: true    # retry if the LLM returns a malformed response
  context:
    prd_paths: [docs/PRD.md, PRD.md, README.md]  # searched in order; first found wins
    include_git_diff: true
    git_diff_max_bytes: 102400
```

### Subcommands

| Command | Description |
|---|---|
| `brainrouter serve` | HTTP proxy daemon. Listens on TCP `:9099` and UDS `/run/user/$UID/brainrouter.sock` |
| `brainrouter mcp` | MCP stdio server. Spawned by harnesses; forwards tool calls to the daemon over UDS |
| `brainrouter install <harness>` | Idempotently patches harness config. Harnesses: `omp`, `vibe`, `opencode`, `codex`, `droid`, `claude`, `pi` |

### HTTP API

All on `http://127.0.0.1:9099`.

#### Proxy

| Method | Path | Protocol | Notes |
|---|---|---|---|
| `GET` | `/health` | — | `{"status":"ok"}` |
| `GET` | `/v1/models` | OpenAI | Returns `auto`, `local`, `cloud` |
| `POST` | `/v1/chat/completions` | OpenAI | Main routing endpoint |
| `POST` | `/v1/messages` | Anthropic | For Claude Code and droid |

#### Management (localhost-only, CSRF-protected)

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/versions` | Installed versions + latest available |
| `GET` | `/api/inference-status` | Current inference state (for progress bar) |
| `GET` | `/api/review-config` | Current review mode and forced model |
| `POST` | `/api/review-config` | Update review mode / forced model |
| `GET` | `/api/models/llama-swap` | Model list from llama-swap |
| `POST` | `/api/upgrade/llama-swap` | Build and install latest llama-swap binary |
| `POST` | `/api/upgrade/manifest` | Pull latest Manifest image and recreate container |
| `POST` | `/api/upgrade/toolbox` | Pull latest toolbox image and recreate container |
| `POST` | `/api/restart/:service` | Restart `llama-swap`, `manifest`, `llama-cpp`, or `brainrouter` |

#### Review

| Method | Path | Notes |
|---|---|---|
| `GET` | `/review/` | Session dashboard |
| `GET` | `/review/session/:id` | Session detail |
| `GET` | `/review/api/sessions` | JSON session list |
| `GET` | `/review/api/sessions/:id` | JSON session detail |
| `POST` | `/review/api/request` | Start a review. Body: `{taskId, summary, details?}` |
| `POST` | `/review/session/:id/resolve` | Human resolve. Body: `{feedback: "lgtm"}` |

### Architecture

```
src/
  main.rs            — clap dispatcher (serve | mcp | install)
  daemon.rs          — startup: loads Bonsai, wires state, starts server
  server.rs          — hyper HTTP router
  classifier.rs      — Bonsai 8B classifier (Cloud/Local decision)
  router.rs          — routes to Manifest or llama-swap; circuit breaker; fallback
  prompt_rewriter.rs — system prompt rewriter for local mode
  anthropic.rs       — Anthropic ↔ OpenAI protocol translation
  mcp_server.rs      — JSON-RPC stdio, forwards to daemon over UDS
  install.rs         — idempotent harness config merger
  session.rs         — in-memory review session store
  review/
    mod.rs           — ReviewService
    review_loop.rs   — iterative LLM review loop
    context.rs       — gathers PRD, git diff, AGENTS.md
    prompt.rs        — review prompt template
  escalation/
    mod.rs           — /review/* HTTP handlers + ReviewRequest parsing
    templates/       — embedded HTML: dashboard + session detail
  provider/
    mod.rs           — Provider trait + SseStream type
    openai.rs        — OpenAI-compatible HTTP adapter
  health.rs          — circuit breaker (3 failures → open; 60 s cooldown)
  stream.rs          — TimeoutStream: chunk stall detection
  config.rs          — YAML config parsing and validation
  types.rs           — OpenAI-compatible request/response types
```

### External services

| Service | Purpose | Default URL |
|---|---|---|
| **Manifest** | Cloud LLM router — provider selection, failover, cost tracking | `http://localhost:3001` |
| **llama-swap** | Local model runner — spawns llama-server on demand | `http://localhost:8081` |
| **Bonsai** | In-process classifier — no HTTP hop | loaded from `model_path` |

### Tests

```bash
cargo test
```

38 tests across 6 suites: circuit breaker, Anthropic↔OpenAI translation, idempotent config merging, review session lifecycle, classifier parse logic, request translation.

