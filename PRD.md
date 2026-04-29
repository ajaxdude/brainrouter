# brainrouter PRD

**Status:** Implemented (V1)
**Language:** Rust
**Binary:** `target/release/brainrouter`
**Config:** `brainrouter.yaml` + `.env`
**Repository:** https://github.com/ajaxdude/brainrouter
**Tests:** 74 tests across the codebase

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Architecture Decisions](#architecture-decisions)
- [Component Map](#component-map)
- [Configuration Reference](#configuration-reference)
- [Routing Flow](#routing-flow)
- [Review Flow](#review-flow)
- [Bridge Architecture](#bridge-architecture)
- [HTTP API Reference](#http-api-reference)
- [MCP Tools Reference](#mcp-tools-reference)
- [Harness Compatibility](#harness-compatibility)
- [Failure Modes and Mitigations](#failure-modes-and-mitigations)
- [Security Model](#security-model)
- [What This Is Not](#what-this-is-not)
- [V2 Ideas](#v2-ideas)

---

## Problem Statement

A developer running multiple LLM subscriptions (Anthropic, OpenAI/Copilot, Google, Mistral) and local models constantly hits:

1. **Rate limits and quota exhaustion** -- one provider fails mid-session, work stops.
2. **Model selection overhead** -- manually deciding cloud vs local before every query burns time.
3. **Review token waste** -- every code review cycle through a cloud LLM costs premium quota.
4. **Harness fragmentation** -- each tool (omp, vibe, claude, opencode, codex, droid) has its own provider config; keeping them aligned is manual work.

---

## Solution

A single Rust daemon that:

1. **Routes in three modes:**
   - `auto` -- Bonsai 8B classifies every query in <200ms and routes to cloud or local.
   - `local` -- Bypasses Bonsai, rewrites the system prompt (strips OMP's 15-20K token bloat down to ~500 tokens with anti-loop directives), routes to llama-swap.
   - `cloud` -- Bypasses Bonsai, routes directly to Manifest.
2. **Falls back automatically** when Manifest stalls or fails -- no manual intervention.
3. **Reviews code locally (by default)** using the same routing infrastructure, exposing an MCP tool that every harness can call. Users can explicitly override Bonsai's routing via the dashboard to force local or cloud review.
4. **Manages system state** via the dashboard, allowing one-click upgrades of llama-swap and resets of the llama.cpp toolbox environment.
5. **Explicit review control.** The dashboard provides a "Code Review Mode" selector. In `auto` mode, Bonsai 8B decides the best model for the review. Users can force `cloud` (always Manifest) or `local` (always llama-swap, with a specific model dropdown).
6. **Presents a single OpenAI-compatible endpoint** to all harnesses, plus an Anthropic-compatible endpoint for harnesses (Claude Code, droid) that speak Anthropic's protocol natively.
7. **Bridges chat platforms** -- Discord and Signal transports shell out to `omp` CLI, bringing LLM access to messaging apps with session management, model selection, and working directory tracking. Commands use the `!br` prefix.
8. **Installs itself into harnesses** via an idempotent `install` subcommand that configures 7 harnesses (omp, vibe, opencode, codex, droid, claude, pi) in a single command.

---

## Architecture Decisions

### Bonsai as classifier, not a responder

Bonsai 8B is embedded in-process via `llama-cpp-2` on Vulkan. It runs synchronously on a blocking thread so it doesn't hold up the async server. It outputs exactly one word: `cloud` or `local`. This classification happens on every request before any network call goes out. Cold classification cost: <200ms on GPU (AMD Strix Halo, Radeon 8060S, 6.3 GB VRAM).

Bonsai was chosen specifically because it is purpose-trained to understand task complexity and model capability -- not as a general assistant. Using it only for routing preserves 113 GB of VRAM for llama-swap models.

### Manifest handles all cloud routing

brainrouter does not know about individual cloud providers (Anthropic, OpenAI, Google, etc.). It sends `model: "auto"` to Manifest and Manifest does provider selection, token accounting, and fallback. This keeps brainrouter's cloud integration down to one HTTP endpoint and one API key.

The coupling is minimal: if Manifest is replaced with LiteLLM or OpenRouter, one URL in `brainrouter.yaml` changes.

### Two wire protocols on one port

- `POST /v1/chat/completions` -- OpenAI format. Covers vibe, opencode, codex, omp, pi (via extension).
- `POST /v1/messages` -- Anthropic Messages API format. Covers Claude Code (via `ANTHROPIC_BASE_URL`) and droid (via `provider: "anthropic"` in custom_models).

Internally the request is immediately translated to OpenAI format and routed through the same Bonsai / Manifest / llama-swap pipeline. Response is translated back to Anthropic SSE events at the edge.

### MCP as thin client

The `brainrouter mcp` process does not load Bonsai and does not run the review loop. It is a ~250-line JSON-RPC stdio server that maps four tool calls to HTTP requests against the daemon's UDS. This keeps harness cold-start fast (no model load) and keeps all state -- sessions, health tracker, circuit breakers -- in one place.

### Review loop uses the same router

When an agent calls `request_review`, the review loop sends its LLM prompts through `Router::route()`. Bonsai classifies the review prompt exactly the same way it classifies chat queries. This means:

- Review calls respect the same cloud/local decision.
- If Manifest is down, review calls fall back to llama-swap automatically.
- No separate LLM configuration for reviews.

### Circuit breaker

Two independent circuit breakers: one for `manifest`, one for `llama-swap`. Three failures within a window open the breaker; 60-second cooldown before retry. This prevents hammering a degraded provider while allowing automatic recovery.

### Robust Anthropic SSE adapter

The SSE adapter follows a strict state machine to guarantee protocol compliance. It ensures that mandatory frames (`message_start`, `content_block_start`, `content_block_stop`, `message_delta`, `message_stop`) are always emitted in the correct sequence, even for empty or interrupted upstream responses. It handles partial line buffering and flushes on EOF to prevent client hangs.

### Security: localhost-only with CSRF protection

Destructive operations (upgrade, restart) are restricted to loopback interfaces (127.0.0.1, ::1) or Unix Domain Sockets. Browser-originated requests to management endpoints are validated against `Origin`/`Referer` headers. Working directory tracking for sessions includes absolute-path enforcement, null-byte rejection, and path-traversal component blocking.

### Prompt rewriting for local models

When routing to `local` mode, the prompt rewriter strips OMP's 15-20K system prompt down to ~500 tokens. This is necessary because local models have limited context windows and perform poorly with massive system prompts designed for cloud-tier models. The rewriter injects concise anti-loop directives tuned for local model behavior.

### Bridge as OMP subprocess

Discord and Signal transports do not call LLMs directly. They shell out to the `omp` CLI as a subprocess, which connects back to brainrouter through the normal proxy path. This means bridge conversations get the same routing, fallback, and review capabilities as direct harness usage. Session state, working directories, and model aliases are persisted to disk per-transport.

### Peer CWD resolution

The MCP server needs to know the caller's working directory. On Linux, brainrouter resolves this by mapping the peer's socket connection (TCP or UDS) back to a PID via `/proc`, then reading `/proc/<pid>/cwd`. This is a Linux-native approach that avoids requiring the caller to pass CWD explicitly, though callers may override it.

---

## Component Map

| Component | File | Responsibility |
|---|---|---|
| Entry point | `src/main.rs` | Clap subcommand dispatch |
| Daemon startup | `src/daemon.rs` | Load config, create all services, validate environment, start server |
| HTTP server | `src/server.rs` | Route `/v1/*`, `/review/*`, `/api/*`, and `/dashboard` |
| Classifier | `src/classifier.rs` | Bonsai-based cloud/local decision |
| Router | `src/router.rs` | Dispatch to Manifest or llama-swap; fallback; timeout |
| Prompt rewriter | `src/prompt_rewriter.rs` | System prompt rewriting for local mode (strips OMP bloat, injects anti-loop prompt) |
| Anthropic shim | `src/anthropic.rs` | `/v1/messages` to `/v1/chat/completions` translation; strict SSE state machine |
| MCP server | `src/mcp_server.rs` | JSON-RPC stdio; forwards to daemon over UDS |
| Installer | `src/install.rs` | Idempotent harness config merger for 7 harnesses |
| Session store | `src/session.rs` | In-memory `HashMap<id, Session>` behind `Mutex` |
| Config | `src/config.rs` | YAML parsing + validation |
| Types | `src/types.rs` | OpenAI request/response structs |
| Health tracker | `src/health.rs` | Per-provider circuit breaker |
| Stream wrapper | `src/stream.rs` | `TimeoutStream`: 180s chunk stall detection; `SafeStream`: error-to-SSE converter |
| Routing events | `src/routing_events.rs` | In-memory event buffer for dashboard live feed |
| Inference state | `src/inference_state.rs` | Track active inference status per provider |
| Peer CWD | `src/peer_cwd.rs` | Linux-native PID/inode mapping for directory discovery (IPv4/IPv6/UDS) |
| Lib | `src/lib.rs` | Crate-level re-exports |
| Review service | `src/review/mod.rs` | `start_review`, `resolve_session` |
| Review loop | `src/review/review_loop.rs` | Iterative LLM review, robust JSON parsing |
| Context gatherer | `src/review/context.rs` | PRD auto-detect, `git diff HEAD`, safe UTF-8 truncation |
| Prompt builder | `src/review/prompt.rs` | Review prompt template |
| Escalation UI | `src/escalation/mod.rs` | `/review/*` HTTP handlers + embedded HTML templates; CWD sanitization |
| Escalation templates | `src/escalation/templates/` | Embedded HTML for review session UI |
| Provider adapter | `src/provider/mod.rs` | Provider trait and common types |
| OpenAI provider | `src/provider/openai.rs` | OpenAI-compatible HTTP client with fault-aware circuit breaking (429/5xx) |
| Bridge core | `src/bridge/core.rs` | Shared transport logic: OMP subprocess management, message chunking, aliases |
| Bridge persistence | `src/bridge/persist.rs` | Session, model alias, and working directory persistence to JSON files |
| Bridge module | `src/bridge/mod.rs` | Bridge initialization and transport dispatch |
| Discord transport | `src/bridge/discord/mod.rs` | Serenity-based Discord bot with channel-scoped sessions |
| Signal transport | `src/bridge/signal/mod.rs` | Signal CLI polling-based transport with group support |
| Integration tests | `tests/anthropic_shim_test.rs` | Anthropic protocol compliance tests |
| Integration tests | `tests/failover_test.rs` | Circuit breaker and fallback behavior tests |
| Integration tests | `tests/install_test.rs` | Harness installer tests |
| Integration tests | `tests/review_test.rs` | Review loop and session lifecycle tests |

---

## Configuration Reference

### YAML Configuration (`brainrouter.yaml`)

#### manifest.*

| Field | Type | Default | Description |
|---|---|---|---|
| `base_url` | `String` | *(required)* | URL of the Manifest instance. Validated: must start with `http://` or `https://` |
| `api_key_env` | `String?` | `None` | Name of the environment variable holding the `mnfst_*` API key |

#### llama_swap.*

| Field | Type | Default | Description |
|---|---|---|---|
| `base_url` | `String` | *(required)* | URL of the llama-swap instance |
| `fallback_model` | `String` | *(required)* | Model key to use when Manifest fails or Bonsai picks local. Must match a key in llama-swap config |
| `local_system_prompt` | `String?` | `None` | Path to a custom system prompt file for `model=local` mode. Built-in lean prompt used if absent |

#### bonsai.*

| Field | Type | Default | Description |
|---|---|---|---|
| `model_path` | `PathBuf` | *(required)* | Path to the Bonsai GGUF file. Validated: must exist at startup |

#### review.*

| Field | Type | Default | Description |
|---|---|---|---|
| `max_iterations` | `u32` | `5` | Maximum LLM review iterations before escalation to human |
| `forced_mode` | `String` | `"auto"` | Routing mode override for reviews: `auto`, `cloud`, or `local`. Persisted to `$XDG_CONFIG_HOME/brainrouter/review_state.json` |
| `forced_model` | `String?` | `None` | For `forced_mode=local`, which llama-swap model to use. Persisted to `$XDG_CONFIG_HOME/brainrouter/review_state.json` |

#### bridge.*

| Field | Type | Default | Description |
|---|---|---|---|
| `omp_path` | `String?` | `"omp"` | Path to the `omp` CLI binary |
| `work_dir` | `String?` | `$HOME` | Default working directory for bridge sessions |
| `aliases_config` | `String?` | `~/.config/omp-bridge/config.yaml` | Path to model alias configuration |
| `timeout_secs` | `u64?` | `600` | OMP subprocess timeout in seconds |
| `default_model` | `String?` | `"brainrouter/auto"` | Default model string passed to OMP |

#### bridge.discord.*

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool?` | `false` | Enable the Discord transport |
| `token` | `String?` | `None` | Discord bot token. Required when `enabled=true` |
| `prefix` | `String?` | `"!"` | Command prefix for bot commands |
| `channel_id` | `String?` | `None` | Reserved, currently unused |

#### bridge.signal.*

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool?` | `false` | Enable the Signal transport |
| `account` | `String?` | `None` | E.164 phone number. Required when `enabled=true` |
| `group_id` | `String?` | `None` | Signal group ID to listen on |
| `prefix` | `String?` | `"!"` | Command prefix for bot commands |
| `storage_path` | `String?` | `None` | Path for Signal CLI data storage |
| `llama_swap_url` | `String?` | `"http://localhost:8081"` | llama-swap URL for model listing |

### CLI Arguments

#### `brainrouter serve`

| Argument | Default | Description |
|---|---|---|
| `--config` | `brainrouter.yaml` | Path to YAML configuration file |
| `--tcp-addr` | `127.0.0.1:9099` | TCP bind address |
| `--socket` | *(none)* | Unix domain socket path |

#### `brainrouter mcp`

| Argument | Default | Description |
|---|---|---|
| `--socket` | *(none)* | UDS path to connect to the daemon |

#### `brainrouter install <harness>`

| Argument | Default | Description |
|---|---|---|
| `<harness>` | *(required)* | One of: `omp`, `vibe`, `opencode`, `codex`, `droid`, `claude`, `pi` |
| `--yes` | `false` | Skip confirmation prompts |
| `--shell-rc` | *(none)* | Path to shell RC file for env var injection |
| `--bin` | *(none)* | Path to brainrouter binary |

#### Global

| Argument | Default | Description |
|---|---|---|
| `--log-level` | `info` | Log verbosity. Overridden by `RUST_LOG` env var |

### Environment Variables

| Variable | Description |
|---|---|
| `RUST_LOG` | Log filter directive. Overrides `--log-level` |
| `HOME` | User home directory. Used for default paths |
| `XDG_RUNTIME_DIR` | Runtime directory for UDS (e.g., `/run/user/$UID`) |
| `XDG_CONFIG_HOME` | Config directory for review state persistence |
| `BRAINROUTER_MANIFEST_DIR` | Override directory for Manifest configuration |
| *(dynamic)* | Whatever `manifest.api_key_env` names (e.g., `MANIFEST_API_KEY`) |

### Hardcoded Constants

| Constant | Value | Description |
|---|---|---|
| `FAILURE_THRESHOLD` | `3` | Failures before circuit breaker opens |
| `COOLDOWN_PERIOD` | `60s` | Circuit breaker cooldown before retry |
| `STREAM_STALL_TIMEOUT` | `180s` | Maximum time between SSE chunks before timeout |
| `MAX_EVENTS` | `500` | Maximum routing events kept in memory for dashboard feed |
| `CLASSIFY_MAX_TOKENS` | `10` | Maximum tokens for Bonsai classification response |
| `USER_MSG_TRUNCATE` | `800 chars` | Truncation limit for user message sent to classifier |
| `CLASSIFIER_CTX_SIZE` | `2048` | Context window size for Bonsai classifier |
| `Discord chunk size` | `1500 chars` | Maximum message length per Discord message |
| `Signal chunk size` | `4000 chars` | Maximum message length per Signal message |
| `Signal poll interval` | `3s` | Polling interval for Signal CLI message retrieval |

### Cargo Features

| Feature | Dependencies | Description |
|---|---|---|
| `default` | `bridge-discord`, `bridge-signal` | Both bridge transports enabled by default |
| `bridge-discord` | `serenity`, `async-trait`, `serde_yml` | Discord bot transport |
| `bridge-signal` | `serde_yml` | Signal CLI transport |

---

## Routing Flow

```
Incoming request (OpenAI or Anthropic format)
  |
  v server.rs: deserialize, translate if Anthropic
  |
  v router.rs: match on model field
  |
  +-- model="auto"  --> classifier.rs: Bonsai inference
  |     +-- Cloud --> manifest (if healthy) --> llama-swap fallback
  |     +-- Local --> llama-swap (Bonsai-chosen model)
  |
  +-- model="local" --> prompt_rewriter.rs: rewrite system msgs
  |                  --> llama-swap (fallback_model)
  |
  +-- model="cloud" --> manifest (direct) --> llama-swap fallback
  |
  +-- model="brainrouter/<model>" --> prompt_rewriter.rs: rewrite system msgs
  |                               --> llama-swap (specific model)
  |
  v provider/openai.rs: HTTP request to chosen backend
  |
  v stream.rs: TimeoutStream wraps SSE (180s stall detection)
  |
  v health.rs: record success/failure for circuit breaker
  |
  v If Anthropic client: anthropic.rs SSE state machine translates back
  |
  v Response streamed to caller
```

---

## Review Flow

```
Agent calls mcp_brainrouter_request_review
  |
  v mcp_server.rs: POST /review/api/request over UDS
  |
  v escalation/mod.rs: parse request, call ReviewService::start_review
  |
  v review/context.rs: gather context
  |   - Auto-detect PRD in project
  |   - git diff HEAD
  |   - Read AGENTS.md if present
  |   - Safe UTF-8 truncation
  |
  v review/review_loop.rs:
      for i in 1..max_iterations (default 5):
        review/prompt.rs: build review prompt
        router.route() --> cloud or local LLM (same routing as chat)
        parse JSON response (STATUS: approved | needs_revision)
        update session in session store
        if approved --> return success
      if max_iterations reached --> escalate to human UI
  |
  v response: {status, feedback, sessionId, iterationCount}
  |
  v Dashboard: GET /review/ --> live session list (5s auto-refresh)
  |
  v Human resolve: POST /review/session/:id/resolve
```

---

## Bridge Architecture

### Overview

Discord and Signal transports bring LLM access to messaging apps. They do not call LLMs directly -- they shell out to the `omp` CLI as a subprocess, which connects back to brainrouter through the normal proxy path. This means bridge conversations get the same routing, fallback, and review capabilities as direct harness usage.

### OMP Subprocess Model

Each user message triggers an `omp` subprocess invocation:

1. Transport receives a message (Discord event / Signal CLI poll)
2. Bridge core resolves the session context: working directory, active model, conversation history
3. `omp` CLI is spawned with the resolved context, pointed at brainrouter as its backend
4. Response is chunked to fit platform limits (Discord: 1500 chars, Signal: 4000 chars)
5. Session state is updated and persisted

### Persistence

All bridge state is persisted to `~/.local/share/omp-bridge/`:

| File | Contents |
|---|---|
| `discord-sessions.json` | Per-channel Discord session state |
| `discord-channel-models.json` | Per-channel model alias mappings |
| `discord-work-dirs.json` | Per-channel working directory |
| `signal-sessions.json` | Per-user/group Signal session state |
| `signal-channel-models.json` | Per-user/group model alias mappings |
| `signal-work-dirs.json` | Per-user/group working directory |

### Bot Commands

#### Discord

| Command | Description |
|---|---|
| `!br ping` | Health check |
| `!br reset` | Clear the current channel's session |
| `!br status` | Show current model |
| `!br auto` / `local` / `cloud` | Set routing mode |
| `!br <model-name>` | Set a specific llama-swap model (names containing `-` or `.`) |
| `!br list` | List all models (routing modes + llama-swap) |
| `!br review` | Show current review mode |
| `!br review auto\|local\|cloud` | Set review mode |
| `!br model <name> <query>` | One-off query with a specific model |
| `!br ls` | List files in current working directory |
| `!br cd <dir>` | Change working directory |
| `!br ..` | Move up one directory |
| `!br mkdir <name>` | Create a directory |
| `!br help` / `!br ?` | Show command help |
| `@bot <query>` | Send a query via mention |
| bare text | Send a query to the LLM (no prefix needed) |

#### Signal

| Command | Description |
|---|---|
| `!br ping` | Health check |
| `!br reset` | Clear the current session |
| `!br status` | Show current model |
| `!br model <name>` | Set model |
| `!br list` | List available models |
| `!br help` / `!br ?` | Show command help |
| `!br <query>` | Send a query to the LLM |
| bare text | Send a query (no prefix needed) |

### Model Selection and Session Management

Each transport maintains per-channel (Discord) or per-user/group (Signal) state:

- **Model selection:** Users can set a specific llama-swap model with `!br <model-name>` (Discord) or `!br model <name>` (Signal), or switch routing modes with `!br auto`, `!br local`, `!br cloud`. The selection persists across messages until explicitly changed.
- **Working directories:** Tracked per-channel/user. Commands like `!br cd`, `!br ..`, and `!br mkdir` manipulate the working directory, which is passed to `omp` as the CWD for file-aware operations.
- **Sessions:** Conversation context is maintained per-channel/user and can be reset with `!br reset`. Bare text (without any prefix) is treated as a query.

---

## HTTP API Reference

### Proxy Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check. Returns daemon status |
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completion proxy |
| `POST` | `/v1/messages` | Anthropic Messages API compatible proxy |

### Dashboard

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Redirect to `/dashboard` |
| `GET` | `/dashboard` | Embedded HTML dashboard with live routing feed, version display, one-click upgrades, review sessions |
| `GET` | *(favicon/logo assets)* | Static assets for dashboard UI |

### Dashboard API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/routing-events` | Live routing event feed (SSE or polling) |
| `GET` | `/api/routing-stats` | Aggregate routing statistics |
| `GET` | `/api/inference-status` | Current inference state per provider |
| `GET` | `/api/service-health` | Circuit breaker status for all providers |
| `GET` | `/api/versions` | Version information for brainrouter and dependencies |
| `GET` | `/api/review-config` | Current review configuration (mode, model) |
| `POST` | `/api/review-config` | Update review configuration |
| `GET` | `/api/models/llama-swap` | List models available in llama-swap |
| `GET` | `/api/bridge-status` | Bridge transport status (Discord/Signal) |

### Destructive Management Endpoints (localhost + CSRF only)

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/restart/llama-swap` | Restart llama-swap service |
| `POST` | `/api/restart/llama-cpp` | Restart llama.cpp toolbox |
| `POST` | `/api/restart/manifest` | Restart Manifest service |
| `POST` | `/api/restart/brainrouter` | Restart brainrouter daemon |
| `POST` | `/api/upgrade/llama-swap` | One-click upgrade llama-swap |
| `POST` | `/api/upgrade/manifest` | One-click upgrade Manifest |
| `POST` | `/api/upgrade/toolbox` | One-click upgrade llama.cpp toolbox |

### Review Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/review/` | Redirect to `/dashboard` |
| `GET` | `/review/session/:id` | View a specific review session (HTML) |
| `POST` | `/review/session/:id/resolve` | Resolve a review session (HTML form) |
| `GET` | `/review/api/sessions` | List all review sessions (JSON) |
| `GET` | `/review/api/sessions/:id` | Get review session details (JSON) |
| `POST` | `/review/api/request` | Start a new review (JSON) |
| `POST` | `/review/api/resolve` | Resolve a session programmatically (JSON) |
| `POST` | `/review/api/continue` | Continue a review iteration (JSON) |
| `POST` | `/review/api/lgtm` | Mark a session as approved (JSON) |

---

## MCP Tools Reference

The MCP server exposes 4 tools via JSON-RPC stdio, forwarded to the daemon over UDS.

### request_review

Start a new code review session.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `taskId` | `String` | Yes | Identifier for the task being reviewed |
| `summary` | `String` | Yes | Brief summary of the changes |
| `cwd` | `String` | No | Working directory for context gathering. Falls back to peer CWD resolution |
| `details` | `String` | No | Additional context about the changes |
| `conversationHistory` | `String[]` | No | Conversation history for context |

### get_session_list

List all review sessions. No parameters.

### get_session_details

Get details for a specific review session.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `sessionId` | `String` | Yes | The session ID to retrieve |

### resolve_session

Resolve a review session with feedback.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `sessionId` | `String` | Yes | The session ID to resolve |
| `feedback` | `String` | Yes | Resolution feedback |

---

## Harness Compatibility

| Harness | Protocol | MCP | Install Command | Notes |
|---|---|---|---|---|
| **omp** | OpenAI | stdio | `brainrouter install omp` | Updates models.yml and mcp.json |
| **vibe** | OpenAI | stdio | `brainrouter install vibe` | Appends to config.toml |
| **opencode** | OpenAI | local | `brainrouter install opencode` | Merges into config.json |
| **codex** | OpenAI | stdio | `brainrouter install codex` | Writes ~/.codex/config.toml entries |
| **droid** | Anthropic | stdio | `brainrouter install droid` | `provider: "anthropic"` required; hits `/v1/messages` |
| **claude** | Anthropic | stdio | `brainrouter install claude` | Sets `ANTHROPIC_BASE_URL=http://127.0.0.1:9099` |
| **pi** | HTTP (extension) | N/A | `brainrouter install pi` | Calls `/review/api/*` directly from a pi extension |

---

## Failure Modes and Mitigations

| Failure | Detection | Mitigation |
|---|---|---|
| Manifest returns 429 / 5xx | HTTP status code | Report failure to health tracker; try llama-swap fallback |
| Manifest stream stalls | `TimeoutStream` (180s per chunk) | Error propagates; health failure recorded |
| Manifest circuit open | Health tracker (3 failures) | Skip Manifest, route directly to llama-swap |
| llama-swap returns error / 404 | HTTP status code | Error returned to caller; next request may load a different model |
| llama-swap circuit open | Health tracker (3 failures) | Error: no backend available. 60s cooldown before retry |
| Both circuits open | Health tracker | Error returned to caller; both providers on 60s cooldown |
| Bonsai inference fails | Rust error from llama-cpp-2 | Default to `Cloud` (safe default -- preserves quality) |
| Bonsai OOM / crash | `spawn_blocking` panic caught | Default to `Cloud` |
| Daemon not running when MCP connects | UDS connect error | `brainrouter mcp` exits with a clear error message |
| Review LLM returns unparseable JSON | JSON parse failure in review_loop | Retry within iteration; count as iteration attempt |
| Review hits max iterations | Iteration counter | Escalate to human via dashboard UI |
| Peer CWD resolution fails | `/proc` read failure | Fall back to caller-supplied CWD or home directory |
| Config validation failure at startup | Missing required fields, invalid URLs, missing GGUF | Daemon refuses to start with descriptive error |
| Bridge OMP subprocess timeout | Configurable timeout (default 600s) | Kill subprocess, return timeout error to chat |
| Signal CLI unavailable | Subprocess spawn failure | Signal transport disabled; logged |
| Discord token invalid | Serenity connection failure | Discord transport disabled; logged |

---

## Security Model

### Localhost-only access

brainrouter binds to `127.0.0.1:9099` by default. It also listens on a Unix Domain Socket at `/run/user/$UID/brainrouter.sock`. No authentication is performed on the proxy boundary -- the assumption is that only local processes connect. All destructive management endpoints (restart, upgrade) are additionally restricted to loopback interfaces and UDS.

### CSRF protection

Browser-originated requests to management endpoints are validated against `Origin` and `Referer` headers. This prevents a malicious page loaded in the user's browser from triggering restarts or upgrades via cross-origin requests.

### Path sanitization

Working directory tracking (for bridge sessions and review context gathering) enforces:

- Absolute path requirement
- Null-byte rejection
- Path-traversal component blocking (no `..` escapes in session-tracked paths)

### Startup validation

Required configuration is validated at daemon startup:

- `manifest.base_url` must start with `http://` or `https://`
- `bonsai.model_path` must exist on disk
- `llama_swap.fallback_model` is required

The daemon refuses to start if validation fails, with descriptive error messages.

---

## What This Is Not

- **Not a provider adapter.** brainrouter does not implement Anthropic, OpenAI, Google, or any cloud provider API. Manifest handles that.
- **Not a model runner.** brainrouter does not spawn llama-server processes. llama-swap handles that.
- **Not an auth layer.** brainrouter is localhost-only with no authentication on the proxy boundary. Credentials live in provider configs (Manifest dashboard, llama-swap config).
- **Not a conversation store.** Chat history is managed by the harness. brainrouter is stateless for proxy calls; review sessions are in-memory and lost on daemon restart.
- **Not a chat application.** The Discord/Signal bridges are thin wrappers around the `omp` CLI, not standalone chat bots with their own reasoning.

---

## V2 Ideas (not planned)

- Bonsai as a Strategic Context Expert -- real-time synthesis of cloud agent output
- Interrupt-and-redirect: user types a correction mid-stream; brainrouter cancels and re-prompts
- Persistent review sessions (SQLite) so they survive daemon restarts
- Pi extension shipping alongside the main binary
- Token usage tracking and routing cost dashboard
