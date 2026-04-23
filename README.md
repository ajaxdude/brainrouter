# brainrouter

A speed-first Rust daemon that sits between your coding harness and your LLMs. Three routing modes:

- **auto** — Bonsai 8B classifies every query and routes to Manifest (cloud) or llama-swap (local)
- **local** — Bypasses Bonsai, rewrites the system prompt for local LLMs (prevents tool loops), routes to llama-swap
- **cloud** — Bypasses Bonsai, routes directly to Manifest

If Manifest stalls, it falls back automatically. When you're done with a task, brainrouter runs the code review locally so you don't burn cloud tokens.

## What it does

```
coding harness (omp / vibe / claude / opencode / codex / droid)
         │
         ▼
   brainrouter :9099
         │
   ┌─────┼─────────────────────────────────────────────┐
   │     ├─ model=auto  → Bonsai classifies query      │
   │     │    Cloud ──── Manifest :3001                 │
   │     │    Local ──── llama-swap :8081                │
   │     ├─ model=local → rewrite prompt → llama-swap   │
   │     └─ model=cloud → Manifest (direct)             │
   └─────────────────────────────────────────────────────┘
         │
   on Manifest fail ───── llama-swap fallback_model
         │
   on task complete ────── review loop → local LLM → dashboard
```

- **One socket for all harnesses.** OpenAI-compatible on `POST /v1/chat/completions`. Anthropic-compatible on `POST /v1/messages`. Claude Code, droid, vibe, opencode, codex, and omp all connect to the same `:9099`.
- **Three routing modes.** `auto` uses Bonsai classification (<200ms). `local` rewrites the system prompt and goes straight to llama-swap. `cloud` goes straight to Manifest.
- **Local mode rewrites system prompts.** OMP's 15-20K token system prompt overwhelms small local models. Local mode replaces it with a lean ~500 token prompt with anti-loop directives.
- **Manifest handles cloud failover.** Manifest runs locally in Docker and does its own provider selection (Anthropic, OpenAI, Copilot, Google, Mistral, DeepSeek, etc.) with automatic fallbacks.
- **llama-swap handles local model management.** brainrouter points at it. Bonsai picks the right model for simple tasks.
  - **Review loop is local.** `mcp_brainrouter_request_review` triggers an iterative code review against a local LLM. No cloud tokens consumed. Escalates to a human web UI if the LLM can't reach a verdict.
- **System monitoring and management.** Dashboard tracks `llama-swap` and `llama.cpp` versions, checks for updates, and allows one-click upgrades and service restarts.
- **Security-hardened.** Localhost-only guards for destructive operations, CSRF protection, and sanitized working directory tracking.
- **Robust protocol handling.** Industrial-grade SSE adapter guarantees protocol compliance for Anthropic clients even on empty or interrupted streams.

## Quick start

### Prerequisites

- Manifest running at `localhost:3001` (`docker compose up` in `~/ai/stack/manifest/`)
- llama-swap running at `localhost:8081`
- Bonsai model at `/mnt/models/prism/prism-ml_Bonsai-8B-unpacked-Q6_K_L.gguf`
- `MANIFEST_API_KEY` set in `~/ai/projects/brainrouter/.env`

### Build and run

```bash
cargo build --release

# Start the daemon
./target/release/brainrouter serve

# Or via systemd (starts at login)
systemctl --user start brainrouter
systemctl --user enable brainrouter
```

### Configure your harness

Install snippets live in `configs/harness/`. Either copy the snippet for your harness or use the install command:

```bash
brainrouter install vibe
brainrouter install opencode
brainrouter install codex
brainrouter install droid
brainrouter install claude        # also sets ANTHROPIC_BASE_URL
brainrouter install claude --shell-rc  # also appends to ~/.zshrc
brainrouter install omp           # updates models.yml + mcp.json
```

## Subcommands

| Command | What it does |
|---|---|
| `brainrouter serve` | HTTP proxy daemon. Listens on TCP `:9099` and UDS `/run/user/$UID/brainrouter.sock` |
| `brainrouter mcp` | MCP stdio server. Spawned by coding harnesses; forwards tool calls to the daemon over UDS |
| `brainrouter install <harness>` | Idempotently merges brainrouter config into the harness config file |

## HTTP API

All on `http://127.0.0.1:9099`.

### Proxy endpoints

| Method | Path | Protocol | Notes |
|---|---|---|---|
| `GET` | `/health` | — | `{"status":"ok"}` |
| `GET` | `/v1/models` | OpenAI | Returns `auto`, `local`, `cloud` models |
| `POST` | `/v1/chat/completions` | OpenAI | Main routing endpoint. `model: "auto"`, `"local"`, or `"cloud"` |
| `POST` | `/v1/messages` | Anthropic | For Claude Code and droid. Same routing, different wire format |

### Management API

Destructive endpoints are restricted to `127.0.0.1` / `::1` or UDS connections with CSRF protection.

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/versions` | Returns local `llama-swap` and toolbox versions |
| `GET` | `/api/inference-status` | High-frequency polling for progress bars |
| `POST` | `/api/upgrade/llama-swap` | Pulls and builds latest binary + restarts service |
| `POST` | `/api/restart/:service` | Restarts `llama-swap`, `manifest`, or `brainrouter` |
| `POST` | `/api/restart/llama-cpp` | Refreshes the `llama.cpp` toolbox container |

### Review endpoints

| Method | Path | Notes |
|---|---|---|
| `GET` | `/review/` | Session dashboard (auto-refreshes every 5s) |
| `GET` | `/review/session/:id` | Session detail view |
| `GET` | `/review/api/sessions` | JSON list of all sessions |
| `GET` | `/review/api/sessions/:id` | JSON session detail |
| `POST` | `/review/api/request` | Start a review. Body: `{taskId, summary, details?}` |
| `POST` | `/review/session/:id/resolve` | Human resolves a session. Body: `{feedback: "lgtm"}` |

## MCP tools

Four tools exposed via MCP:

| Tool | What it does |
|---|---|
| `request_review` | Start a code review. Required: `taskId`, `summary`. Optional: `details`, `conversationHistory` |
| `get_session_list` | List all review sessions |
| `get_session_details` | Get one session by `sessionId` |
| `resolve_session` | Human resolves a session with `feedback`. "lgtm"/"ok"/"approved" → approved; anything else → needs_revision |

## Configuration

`brainrouter.yaml`:

```yaml
manifest:
  base_url: "http://localhost:3001"
  api_key_env: MANIFEST_API_KEY   # env var holding your mnfst_* key

llama_swap:
  base_url: "http://localhost:8081"
  fallback_model: "gemma-4-26b-a4b"   # used when Manifest fails or Bonsai picks local
  # local_system_prompt: "/path/to/custom-prompt.md"  # optional: custom prompt for model=local
  # llama_cpp_restart_script: "/path/to/refresh-toolboxes.sh" # optional: for toolbox restart button

bonsai:
  model_path: "/mnt/models/prism/prism-ml_Bonsai-8B-unpacked-Q6_K_L.gguf"

# Optional review configuration (all fields have defaults)
review:
  max_iterations: 5
  retry_on_llm_error: true
  escalation:
    enabled: true
    port: 9099          # review UI runs under /review/ on the main port
  context:
    prd_paths: [docs/PRD.md, PRD.md, README.md]
    include_git_diff: true
    git_diff_max_bytes: 102400
```

API key goes in `.env` in the project root (loaded by the systemd service):

```
MANIFEST_API_KEY=mnfst_your_key_here
```

## Harness integration reference

### omp

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
      "command": "/home/papa/ai/projects/brainrouter/target/release/brainrouter",
      "args": ["mcp", "--socket", "/run/user/1000/brainrouter.sock"],
      "timeout": 300000
    }
  }
}
```

### vibe

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
  { name = "brainrouter", command = "/path/to/brainrouter", args = ["mcp", "--socket", "/run/user/1000/brainrouter.sock"] },
]
```

### opencode

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
      "command": ["/path/to/brainrouter", "mcp", "--socket", "/run/user/1000/brainrouter.sock"]
    }
  }
}
```

### codex

```toml
# ~/.codex/config.toml
model = "auto"
model_provider = "brainrouter"

[model_providers.brainrouter]
name = "Brainrouter"
base_url = "http://127.0.0.1:9099/v1"

[mcp_servers.brainrouter]
command = "/path/to/brainrouter"
args = ["mcp", "--socket", "/run/user/1000/brainrouter.sock"]
```

### droid (factory.ai)

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
      "args": ["mcp", "--socket", "/run/user/1000/brainrouter.sock"]
    }
  }
}
```

> **Note:** `provider: "anthropic"` is required. Droid's `openai` mode posts to `/responses` (not served). `anthropic` mode posts to `/v1/messages`, which brainrouter handles.

### claude

```bash
# Register MCP tool
claude mcp add-json brainrouter '{
  "type": "stdio",
  "command": "/path/to/brainrouter",
  "args": ["mcp", "--socket", "/run/user/1000/brainrouter.sock"]
}' --scope user

# Route Claude Code through brainrouter (add to ~/.zshrc)
export ANTHROPIC_BASE_URL=http://127.0.0.1:9099
export ANTHROPIC_AUTH_TOKEN=not-used
```

## Architecture

```
src/
  main.rs            — clap dispatcher (serve | mcp | install)
  daemon.rs          — startup: loads Bonsai, wires all state, starts server
  server.rs          — hyper HTTP router; /v1/* and /review/* routes
  classifier.rs      — Bonsai 8B complexity classifier (Cloud/Local decision)
  router.rs          — routes to Manifest or llama-swap; circuit breaker; fallback
  prompt_rewriter.rs — system prompt rewriter for local mode (strips OMP bloat)
  anthropic.rs       — Anthropic ↔ OpenAI protocol translation
  mcp_server.rs      — JSON-RPC stdio, forwards to daemon over UDS
  install.rs         — idempotent harness config merger
  session.rs         — in-memory review session store
  review/
    mod.rs           — ReviewService: start_review, resolve_session
    review_loop.rs   — iterative LLM review (up to max_iterations)
    context.rs       — gathers PRD, git diff, AGENTS.md
    prompt.rs        — review prompt template
  escalation/
    mod.rs           — /review/* HTTP handlers
    templates/       — askama HTML: dashboard + session detail
  provider/
    mod.rs           — Provider trait + SseStream type
    openai.rs        — OpenAI-compatible HTTP adapter (Manifest + llama-swap)
  health.rs          — circuit breaker (3 failures → open; 60s cooldown)
  stream.rs          — TimeoutStream: 15s chunk stall detection
  config.rs          — YAML config parsing and validation
  types.rs           — OpenAI-compatible request/response types
```

## External dependencies

| Service | What it does | How to run |
|---|---|---|
| **Manifest** | Cloud LLM router; handles provider selection, failover, cost tracking | `docker compose up` in `~/ai/stack/manifest/` |
| **llama-swap** | Local model runner; spawns llama-server on demand | systemd user service `llama-swap.service` |

brainrouter talks to both over HTTP. Swapping either out means changing one URL in `brainrouter.yaml`.

## Systemd services

All run as user services (no root). Start order: `llama-swap` → `manifest` → `brainrouter`.

```bash
systemctl --user status brainrouter manifest llama-swap
```

Service file: `~/.config/systemd/user/brainrouter.service`

## Tests

```bash
cargo test
```

38 tests across 6 suites:

- `failover_test.rs` — circuit breaker state machine
- `anthropic_shim_test.rs` — Anthropic ↔ OpenAI translation
- `install_test.rs` — idempotent config merging
- `review_test.rs` — session lifecycle
- lib unit tests — classifier parse logic, Anthropic request translation
