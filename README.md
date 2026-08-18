# brainrouter

<p align="center">
  <img src="assets/brainrouter-logo.svg" alt="brainrouter logo" width="600">
</p>

A speed-first Rust proxy that sits between your AI coding harness and your LLMs. An external Bonsai 27B classifier decides in under 500 ms whether each request goes to cloud (via Manifest) or local inference (via llama-swap). Automatic fallback, system-prompt rewriting for local models, and an MCP-triggered iterative code-review loop that runs entirely on your own hardware.

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
- **Three routing modes.** `auto` uses Bonsai classification (<500 ms). `local` rewrites the system prompt and goes straight to llama-swap. `cloud` goes straight to Manifest.
- **Local prompt rewriting.** OMP's 15–20 K token system prompt overwhelms small local models. Local mode replaces it with a lean ~500 token prompt with anti-loop directives.
- **Manifest handles cloud failover.** Manifest runs locally in Docker and picks the right cloud provider (Anthropic, OpenAI, Copilot, Google, Mistral, DeepSeek, etc.) with its own automatic fallbacks.
- **MCP code review.** `mcp_brainrouter_request_review` triggers an iterative review loop (up to 5 rounds by default). The review LLM reads your PRD, git diff, and task summary, then either approves or gives actionable feedback.
- **Dashboard.** Live routing feed, review session list, version display, one-click upgrades and service restarts — all at `http://127.0.0.1:9099`.
- **VRAM control.** The dashboard can stop/start the Bonsai classifier and flush every model loaded in llama-swap — reclaim GPU memory without a reboot or a terminal.

---

## Table of contents

1. [Install (one script)](#install)
2. [Configure](#configure)
3. [Connect your harness](#connect-your-harness)
4. [Dashboard guide](#dashboard-guide)
5. [MCP code review guide](#mcp-code-review-guide)
6. [Bridge: Discord and Signal](#bridge-discord-and-signal)
7. [Reference](#reference)

---

## Install

For Fedora Linux with multiple users, one script installs and configures everything.
Run it as a user with sudo access:

```bash
git clone https://github.com/ajaxdude/brainrouter ~/ai/projects/brainrouter
cd ~/ai/projects/brainrouter
sudo bash install.sh
```

The script installs (idempotent — safe to re-run):

- **System packages** — git, golang, toolbox, docker, vulkan headers
- **bun** — JavaScript runtime for oh-my-pi, installed system-wide
- **oh-my-pi** — installed for every human user via bun
- **Bonsai 27B BF16** — downloaded to `/mnt/models/prism/` (~7.3 GB), runs as an external llama-server process on port 9200
- **Manifest** — cloud LLM router running as a system Docker service on port 3001
- **llama-swap** — local model runner as a system Docker service on port 8081
- **brainrouter** — compiled and installed to `/usr/local/bin/brainrouter`
- **llama-server-toolbox** — wrapper at `/usr/local/bin/llama-server-toolbox`
- **toolbox container** `llama-vulkan-radv` — AMD RADV Vulkan environment
- **Shared config** — `/etc/brainrouter/brainrouter.yaml` and `/etc/brainrouter/env`
- **Per-user systemd services** — brainrouter enabled for every user, auto-starts at boot via `loginctl linger`
- **Shell environment** — `/etc/profile.d/ai-stack.sh` sets PATH and harness env vars for all users

### After the script finishes — one manual step

The Manifest API key cannot be automated (you create it in the browser wizard):

1. Open **http://localhost:3001**, complete the setup wizard, add your cloud API keys
2. Go to **Settings → API Keys → Create key** — copy the `mnfst_…` key
3. Paste it into the shared env file:
   ```bash
   sudo nano /etc/brainrouter/env
   # Replace: MANIFEST_API_KEY=mnfst_REPLACE_WITH_YOUR_KEY
   ```
4. Reboot — all users come up with brainrouter running automatically.
   Or without rebooting, for each user:
   ```bash
   sudo -u USERNAME XDG_RUNTIME_DIR=/run/user/$(id -u USERNAME) \
     systemctl --user restart brainrouter
   ```

### Multi-user notes

- The Manifest API key lives in `/etc/brainrouter/env` (owned `root:aistack`, mode `640`).
  All users in the `aistack` group can read it. The script adds every human user to this group.
- `loginctl enable-linger` is set for each user so brainrouter starts at boot without anyone
  needing to log in.
- New users added after install: their service file comes from `/etc/skel`; run
  `sudo bash install.sh` again (idempotent) to complete their setup.
- To edit which local model llama-swap serves:
  ```bash
  sudo nano /opt/ai/llama-swap/config.yaml
  sudo systemctl restart llama-swap
  ```


## Configure

After `install.sh` runs, the system config is already in place at `/etc/brainrouter/brainrouter.yaml`.
Each user also gets a copy seeded to `~/.config/brainrouter/brainrouter.yaml` at install time.

The only value you need to change post-install is `fallback_model` — set it to match a model key
in `/opt/ai/llama-swap/config.yaml`:

```bash
sudo nano /etc/brainrouter/brainrouter.yaml
sudo nano /opt/ai/llama-swap/config.yaml  # define the model
sudo systemctl restart llama-swap
```

```yaml
# /etc/brainrouter/brainrouter.yaml (shared for all users)

manifest:
  base_url: "http://localhost:3001/v1"
  api_key_env: MANIFEST_API_KEY  # key lives in /etc/brainrouter/env

llama_swap:
  base_url: "http://localhost:8081/v1"
  fallback_model: "your-local-model"  # must match a key in /opt/ai/llama-swap/config.yaml

bonsai:
  model_path: "/mnt/models/prism/Bonsai-27B-dspark-bf16.gguf"  # GGUF model for classification
  server_port: 9200  # port for the external llama-server process
  fork_path: "/home/papa/.local/share/brainrouter/llama-prism/llama-server"  # PrismML fork binary

The Manifest API key lives in `/etc/brainrouter/env` (readable by the `aistack` group — all human
users are added to it by `install.sh`):

```bash
sudo nano /etc/brainrouter/env
# MANIFEST_API_KEY=mnfst_your_key_here
```

After any config change, restart brainrouter for your user:
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



## Multi-user deployment

For shared machines with multiple users, see [`deploy/brainrouter_ecosystem.md`](deploy/brainrouter_ecosystem.md) which covers:

- **System-level services**: llama-swap and Manifest run as system Docker services, shared by all users.
- **Per-user brainrouter**: Each user runs their own brainrouter instance as a systemd user service.
- **Shared model storage**: GGUFs in `/opt/models` with `aistack` group permissions.
- **Automated scripts**: `deploy/deploy.sh --multi-user` for the admin, `deploy/user-setup.sh` for each user.
- **Uninstall**: `deploy/uninstall.sh` and `deploy/uninstall_brainrouter_ecosystem.md`.


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
- **bonsai** — classifier server state: `on`, `off`, or `down`

Click the **bonsai** row to stop or start the classifier's llama-server. Stopping frees its GPU memory; while stopped, `auto` routing defaults to cloud until it is started again.

When a newer version is available (checked against GitHub / Docker Hub on each poll), an orange **`component → new-version`** button appears. Click it to upgrade. Each button is labelled so you know exactly what will be updated.

### Service controls (nav bar)

Four restart buttons in the top nav:

| Button | What it does |
|---|---|
| **Restart llama-swap** | `systemctl --user restart llama-swap` |
| **Restart llama.cpp** | Refreshes the toolbox container (runs configured restart script) |
| **Restart Manifest** | `docker compose restart manifest` |
| **Restart brainrouter** | `systemctl --user restart brainrouter` — page reloads after 3 s |
| **⏏ Flush Models** | Unloads every model from llama-swap memory (frees VRAM) without restarting; models reload on next request |

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

## Bridge: Discord and Signal

brainrouter includes bridge transports that connect Discord and Signal to OMP. Each bridge runs as part of the brainrouter daemon and shells out to the `omp` CLI to handle queries. Enable them in the `bridge` section of `brainrouter.yaml`.

### Discord bot commands

| Command | Description |
|---|---|
| `!br ping` | Health check |
| `!br reset` | Clear conversation session |
| `!br status` | Show current model |
| `!br auto` / `local` / `cloud` | Set routing mode |
| `!br <model-name>` | Set specific llama-swap model (names containing `-` or `.`) |
| `!br list` | List all models (routing + llama-swap) |
| `!br model <name> <query>` | One-off model override for a single query |
| `!br ls` | List files in current working directory |
| `!br cd <dir>` | Change working directory |
| `!br ..` | Go up one directory |
| `!br mkdir <name>` | Create a directory |
| `!br review` | Show review mode |
| `!br review auto\|local\|cloud` | Set review mode |
| `!br help` / `!br ?` | Show command help |
| bare text | Send query directly (no prefix needed) |

### Signal bot commands

| Command | Description |
|---|---|
| `!br ping` | Health check |
| `!br reset` | Clear conversation session |
| `!br status` | Show current model |
| `!br auto` / `local` / `cloud` | Set routing mode |
| `!br <model-name>` | Set specific llama-swap model (names containing `-` or `.`) |
| `!br model <name>` | Set model (legacy form) |
| `!br list` | List models |
| `!br review` | Show current review mode |
| `!br review auto\|local\|cloud` | Set review mode |
| `!br help` / `!br ?` | Show command help |
| bare text or `!br <query>` | Send query |

### Model aliases

Specific llama-swap models can be set directly via `!br <model-name>` (e.g. `!br gemma-4-26b-a4b`). Model names are detected by containing `-` or `.`. Use `!br list` to see all available models.

### Session management

Each channel (Discord) or conversation (Signal) maintains its own session. Sessions track conversation history, current working directory, and selected model. Use `!br reset` to clear a session.

### Persistence paths

| Data | Path |
|---|---|
| Discord sessions | `~/.local/share/omp-bridge/discord-sessions.json` |
| Discord channel models | `~/.local/share/omp-bridge/discord-channel-models.json` |
| Discord work dirs | `~/.local/share/omp-bridge/discord-work-dirs.json` |
| Signal sessions | `~/.local/share/omp-bridge/signal-sessions.json` |
| Signal channel models | `~/.local/share/omp-bridge/signal-channel-models.json` |
| Signal work dirs | `~/.local/share/omp-bridge/signal-work-dirs.json` |

Long responses are automatically chunked (1500 chars for Discord, 4000 chars for Signal).

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
  local_models: ["my-model", "other"]    # optional — model keys that bypass Bonsai when used directly
  local_system_prompt: "/path/to/prompt.md"  # optional — override built-in lean prompt

bonsai:
  model_path: "/mnt/models/prism/Bonsai-27B-dspark-bf16.gguf"  # required — absolute path
  server_port: 9200                                    # external llama-server port (default)
  fork_path: "/path/to/llama-server"                   # PrismML fork binary (default ~/.local/share/brainrouter/llama-prism/llama-server)

models:
  path: /opt/models                                    # shared GGUF dir; ${models_path} expands to this
  shared_write: false                                  # true = all aistack members can add/delete models

review:
  max_iterations: 5           # LLM review rounds before escalating to human
  forced_mode: "auto"         # "auto" | "cloud" | "local" — default "auto"
  forced_model: "my-model"    # only used when forced_mode = "local"

bridge:
  omp_path: "omp"                              # path to omp CLI binary
  work_dir: "/home/you"                         # default working directory
  aliases_config: "~/.config/omp-bridge/config.yaml"  # model alias definitions
  timeout_secs: 600                              # per-query timeout
  default_model: "brainrouter/auto"              # model for new sessions
  discord:
    enabled: false                               # set true + provide token to activate
    token: "Bot ..."                              # Discord bot token — required when enabled
    prefix: "!"                                   # command prefix
  signal:
    enabled: false                               # set true + provide account to activate
    account: "+15551234567"                       # E.164 phone — required when enabled
    group_id: "base64..."                         # restrict to one Signal group
    prefix: "!"                                   # command prefix
    storage_path: "/path/to/signal-cli/data"      # signal-cli storage
    llama_swap_url: "http://localhost:8081"        # for llama-list command
```

### Environment variables

| Variable | Description |
|---|---|
| `RUST_LOG` | Log level filter (default `info`). Overrides `--log-level`. Example: `RUST_LOG=debug` |
| `HOME` | User home directory. Used for default paths |
| `XDG_RUNTIME_DIR` | Runtime directory for UDS socket (default `/run/user/$UID`) |
| `XDG_CONFIG_HOME` | Config directory for persisted review state (default `~/.config`) |
| `BRAINROUTER_MANIFEST_DIR` | Override Manifest docker-compose directory for restart/upgrade |
| `<manifest.api_key_env>` | Dynamic: whatever env var name is set in `manifest.api_key_env` (e.g. `MANIFEST_API_KEY`) holds the Manifest API key |

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
| `GET` | `/api/routing-events` | Live routing events feed |
| `GET` | `/api/routing-stats` | Routing statistics |
| `GET` | `/api/service-health` | Service health status per provider |
| `GET` | `/api/bridge-status` | Bridge transport status (Discord / Signal) |
| `GET` | `/api/inference-status` | Current inference state (for progress bar) |
| `GET` | `/api/review-config` | Current review mode and forced model |
| `POST` | `/api/review-config` | Update review mode / forced model |
| `GET` | `/api/models/llama-swap` | Model list from llama-swap |
| `POST` | `/api/upgrade/llama-swap` | Build and install latest llama-swap binary |
| `POST` | `/api/upgrade/manifest` | Pull latest Manifest image and recreate container |
| `POST` | `/api/upgrade/toolbox` | Pull latest toolbox image and recreate container |
| `POST` | `/api/restart/:service` | Restart `llama-swap`, `manifest`, `llama-cpp`, or `brainrouter` |
| `GET` | `/api/bonsai` | Bonsai classifier server state (`enabled`, `healthy`) |
| `POST` | `/api/bonsai/toggle` | Stop/start the Bonsai classifier to free/reclaim VRAM |
| `POST` | `/api/models/flush` | Unload all models from llama-swap memory (no restart) |

#### Review

| Method | Path | Notes |
|---|---|---|
| `GET` | `/review/` | Session dashboard |
| `GET` | `/review/session/:id` | Session detail |
| `GET` | `/review/api/sessions` | JSON session list |
| `GET` | `/review/api/sessions/:id` | JSON session detail |
| `POST` | `/review/api/request` | Start a review. Body: `{taskId, summary, details?}` |
| `POST` | `/review/api/resolve` | Resolve a review session. Body: `{sessionId, feedback}` |
| `POST` | `/review/api/continue` | Continue a review iteration |
| `POST` | `/review/api/lgtm` | Quick-approve a review session |
| `POST` | `/review/session/:id/resolve` | Human resolve. Body: `{feedback: "lgtm"}` |

### Architecture

```
src/
  main.rs            	-- clap dispatcher (serve | mcp | install)
  daemon.rs          	-- startup: loads Bonsai, wires state, starts server
  classifier.rs      	-- Cloud/Local decision via external Bonsai server
  bonsai_server.rs    	-- Bonsai llama-server lifecycle (spawn, health, dashboard toggle)
  router.rs          	-- routes to Manifest or llama-swap; circuit breaker; fallback
  prompt_rewriter.rs 	-- system prompt rewriter for local mode
  anthropic.rs       	-- Anthropic <> OpenAI protocol translation
  mcp_server.rs      	-- JSON-RPC stdio, forwards to daemon over UDS
  install.rs         	-- idempotent harness config merger
  session.rs         	-- in-memory review session store
  config.rs          	-- YAML config parsing and validation
  types.rs           	-- OpenAI-compatible request/response types
  lib.rs             	-- library root
  peer_cwd.rs        	-- peer CWD resolution via /proc
  routing_events.rs  	-- routing event store (last 500 events)
  inference_state.rs 	-- inference state tracking
  review/
    mod.rs           	-- ReviewService
    review_loop.rs   	-- iterative LLM review loop
    context.rs       	-- gathers PRD, git diff, AGENTS.md
    prompt.rs        	-- review prompt template
  escalation/
    mod.rs           	-- /review/* HTTP handlers + ReviewRequest parsing
    templates/       	-- embedded HTML: dashboard + session detail
  provider/
    mod.rs           	-- Provider trait + SseStream type
    openai.rs        	-- OpenAI-compatible HTTP adapter
  health.rs          	-- circuit breaker (3 failures -> open; 60 s cooldown)
  stream.rs          	-- TimeoutStream: chunk stall detection
  bridge/
    mod.rs           	-- bridge feature flags and init
    core.rs          	-- shared bridge logic (session, OMP dispatch, chunking)
    persist.rs       	-- JSON persistence for sessions, models, work dirs
    discord/
      mod.rs         	-- Discord bot (serenity) with command handler
    signal/
      mod.rs         	-- Signal bot with polling loop
```

### External services

| Service | Purpose | Default URL |
|---|---|---|
| **Manifest** | Cloud LLM router — provider selection, failover, cost tracking | `http://localhost:3001` |
| **llama-swap** | Local model runner — spawns llama-server on demand | `http://localhost:8081` |
| **Bonsai** | External classifier — llama-server on port 9200 (PrismML fork) | `http://127.0.0.1:9200` |

### Tests

```bash
cargo test
```

89 tests across the codebase: circuit breaker, Anthropic protocol translation, idempotent config merging, review session lifecycle, classifier parse logic, request translation, failover, install, review loop, and bridge persistence.

