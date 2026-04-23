# brainrouter PRD

**Status:** Implemented (V1)
**Language:** Rust
**Binary:** `target/release/brainrouter`
**Config:** `brainrouter.yaml` + `.env`

---

## Problem statement

A developer running multiple LLM subscriptions (Anthropic, OpenAI/Copilot, Google, Mistral) and local models constantly hits:

1. **Rate limits and quota exhaustion** — one provider fails mid-session, work stops.
2. **Model selection overhead** — manually deciding cloud vs local before every query burns time.
3. **Review token waste** — every code review cycle through a cloud LLM costs premium quota.
4. **Harness fragmentation** — each tool (omp, vibe, claude, opencode, codex, droid) has its own provider config; keeping them aligned is manual work.

## Solution

A single Rust daemon that:

1. **Routes in three modes:**
   - `auto` — Bonsai 8B classifies every query in <200ms and routes to cloud or local.
   - `local` — Bypasses Bonsai, rewrites the system prompt (strips OMP's 15-20K token bloat down to ~500 tokens with anti-loop directives), routes to llama-swap.
   - `cloud` — Bypasses Bonsai, routes directly to Manifest.
2. **Falls back** automatically when Manifest stalls or fails — no manual intervention.
- **Reviews code** locally using the same routing infrastructure, exposing an MCP tool that every harness can call.
- **Manages system state** via the dashboard, allowing one-click upgrades of `llama-swap` and resets of the `llama.cpp` toolbox environment.
- **Presents a single OpenAI-compatible endpoint** to all harnesses, plus an Anthropic-compatible endpoint for harnesses (Claude Code, droid) that speak Anthropic's protocol natively.

## Architecture decisions

### Bonsai as classifier, not a responder

Bonsai 8B is embedded in-process via `llama-cpp-2` on Vulkan. It runs synchronously on a blocking thread so it doesn't hold up the async server. It outputs exactly one word: `cloud` or `local`. This classification happens on every request before any network call goes out. Cold classification cost: <200ms on GPU (AMD Strix Halo, Radeon 8060S, 6.3 GB VRAM).

Bonsai was chosen specifically because it is purpose-trained to understand task complexity and model capability — not as a general assistant. Using it only for routing preserves 113 GB of VRAM for llama-swap models.

### Manifest handles all cloud routing

brainrouter does not know about individual cloud providers (Anthropic, OpenAI, Google, etc.). It sends `model: "auto"` to Manifest and Manifest does provider selection, token accounting, and fallback. This keeps brainrouter's cloud integration down to one HTTP endpoint and one API key.

The coupling is minimal: if Manifest is replaced with LiteLLM or OpenRouter, one URL in `brainrouter.yaml` changes.

### Two wire protocols on one port

- `POST /v1/chat/completions` — OpenAI format. Covers vibe, opencode, codex, omp, pi (via extension).
- `POST /v1/messages` — Anthropic Messages API format. Covers Claude Code (via `ANTHROPIC_BASE_URL`) and droid (via `provider: "anthropic"` in custom_models).

Internally the request is immediately translated to OpenAI format and routed through the same Bonsai → Manifest/llama-swap pipeline. Response is translated back to Anthropic SSE events at the edge.

### MCP as thin client

The `brainrouter mcp` process does not load Bonsai and does not run the review loop. It is a ~250-line JSON-RPC stdio server that maps four tool calls to HTTP requests against the daemon's UDS. This keeps harness cold-start fast (no model load) and keeps all state — sessions, health tracker, circuit breakers — in one place.

### Review loop uses the same router

When an agent calls `request_review`, the review loop sends its LLM prompts through `Router::route()`. Bonsai classifies the review prompt exactly the same way it classifies chat queries. This means:

- Review calls respect the same cloud/local decision.
- If Manifest is down, review calls fall back to llama-swap automatically.
- No separate LLM configuration for reviews.

### Circuit breaker

Two independent circuit breakers: one for `manifest`, one for `llama-swap`. Three failures within a window open the breaker; 60-second cooldown before retry. This prevents hammering a degraded provider while allowing automatic recovery.

### Robust Anthropic SSE Adapter

The SSE adapter follows a strict state machine to guarantee protocol compliance. It ensures that mandatory frames (`message_start`, `content_block_start`, `content_block_stop`, `message_delta`, `message_stop`) are always emitted in the correct sequence, even for empty or interrupted upstream responses. It handles partial line buffering and flushes on EOF to prevent client hangs.

### Security and Management

- **Localhost-Only Access**: Destructive operations (upgrade, restart) are restricted to loopback interfaces (127.0.0.1, ::1) or Unix Domain Sockets.
- **CSRF Protection**: Browser-originated requests to management endpoints are validated against `Origin`/`Referer` headers.
- **Path Sanitization**: Working directory tracking for sessions includes absolute-path enforcement, null-byte rejection, and path-traversal component blocking.
- **Startup Validation**: Optional management script paths are validated at startup for existence and execution permissions.

## Component map

| Component | File | Responsibility |
|---|---|---|
| Entry point | `src/main.rs` | Clap subcommand dispatch |
| Daemon startup | `src/daemon.rs` | Load config, create all services, validate environment, start server |
| HTTP server | `src/server.rs` | Route `/v1/*`, `/review/*`, and `/api/*` management |
| Classifier | `src/classifier.rs` | Bonsai-based cloud/local decision |
| Router | `src/router.rs` | Dispatch to Manifest or llama-swap; fallback; timeout |
| Prompt rewriter | `src/prompt_rewriter.rs` | System prompt rewriting for local mode (strips OMP bloat, injects anti-loop prompt) |
| Anthropic shim | `src/anthropic.rs` | `/v1/messages` ↔ `/v1/chat/completions` translation; robust SSE state machine |
| MCP server | `src/mcp_server.rs` | JSON-RPC stdio; forwards to daemon over UDS |
| Installer | `src/install.rs` | Idempotent harness config merger |
| Session store | `src/session.rs` | In-memory `HashMap<id, Session>` behind `Mutex` |
| Review service | `src/review/mod.rs` | `start_review`, `resolve_session` |
| Review loop | `src/review/review_loop.rs` | Iterative LLM review, robust JSON parsing |
| Context gatherer | `src/review/context.rs` | PRD auto-detect, `git diff HEAD`, safe UTF-8 truncation |
| Prompt builder | `src/review/prompt.rs` | Review prompt template |
| Escalation UI | `src/escalation/mod.rs` | `/review/*` HTTP handlers + askama templates; CWD sanitization |
| Provider adapter | `src/provider/openai.rs` | OpenAI-compatible HTTP client with fault-aware circuit breaking (429/5xx) |
| Health tracker | `src/health.rs` | Per-provider circuit breaker |
| Stream wrapper | `src/stream.rs` | `TimeoutStream`: 180-second chunk stall detection; `SafeStream`: Error-to-SSE converter |
| Types | `src/types.rs` | OpenAI request/response structs |
| Config | `src/config.rs` | YAML parsing + validation |
| Peer CWD | `src/peer_cwd.rs` | Linux-native PID/Inode mapping for directory discovery (IPv4/IPv6/UDS) |

## Routing flow (request)

```
Incoming request (OpenAI or Anthropic format)
  │
  ▼ server.rs: deserialize, translate if Anthropic
  │
  ▼ router.rs: match on model field
  │
  ├─ model="auto"  → classifier.rs: Bonsai inference
  │    ├─ Cloud → manifest (if healthy) → llama-swap fallback
  │    └─ Local → llama-swap (Bonsai-chosen model)
  │
  ├─ model="local" → prompt_rewriter.rs: rewrite system msgs
  │                  → llama-swap (fallback_model)
  │
  └─ model="cloud" → manifest (direct) → llama-swap fallback
```

## Review flow

```
Agent calls mcp_brainrouter_request_review
  │
  ▼ mcp_server.rs: POST /review/api/request over UDS
  │
  ▼ escalation/mod.rs: parse, call ReviewService::start_review
  │
  ▼ review/review_loop.rs:
      gather context (PRD, git diff, AGENTS.md)
      for i in 1..max_iterations:
        build prompt
        router.route() → cloud or local LLM
        parse JSON response (STATUS: approved | needs_revision)
        update session
        if approved → return
      if max_iterations reached → escalate to human UI
  │
  ▼ response: {status, feedback, sessionId, iterationCount}
  │
  ▼ dashboard: GET /review/ → live session list (5s auto-refresh)
  │
  ▼ human resolve: POST /review/session/:id/resolve
```

## Configuration reference

```yaml
manifest:
  base_url: string        # URL of your Manifest instance
  api_key_env: string?    # env var name holding the mnfst_* key

llama_swap:
  base_url: string        # URL of your llama-swap instance
  fallback_model: string  # model key to use when Manifest fails or Bonsai picks local
  local_system_prompt: path?  # optional custom prompt for model=local; built-in lean prompt used if absent

bonsai:
  model_path: path        # path to the Bonsai GGUF file

review:
  max_iterations: int     # default 5. LLM iterations before escalation
  retry_on_llm_error: bool  # default true
  escalation:
    enabled: bool         # default true
    port: int             # review UI port (default: same as main, under /review/)
  context:
    prd_paths: [string]   # where to look for the project PRD
    include_git_diff: bool  # default true
    git_diff_max_bytes: int # default 102400
```

## Harness compatibility

| Harness | Protocol | MCP | Notes |
|---|---|---|---|
| **omp** | OpenAI | stdio | `brainrouter install omp` updates models.yml and mcp.json |
| **vibe** | OpenAI | stdio | `brainrouter install vibe` appends to config.toml |
| **opencode** | OpenAI | local | `brainrouter install opencode` merges into config.json |
| **codex** | OpenAI | stdio | `brainrouter install codex` writes ~/.codex/config.toml entries |
| **droid** | Anthropic | stdio | `provider: "anthropic"` required; droid hits `/v1/messages` |
| **claude** | Anthropic | stdio | `ANTHROPIC_BASE_URL=http://127.0.0.1:9099`; `brainrouter install claude` |
| **pi** | HTTP (extension) | N/A | Call `/review/api/*` directly from a pi extension |

## Failure modes and mitigations

| Failure | Detection | Mitigation |
|---|---|---|
| Manifest returns 429 / 5xx | HTTP status | Report failure to health tracker; try llama-swap fallback |
| Manifest stream stalls | `TimeoutStream` (15s per chunk) | Error propagates; health failure recorded |
| Manifest circuit open | Health tracker | Skip Manifest, go directly to llama-swap |
| llama-swap 404 (no model loaded) | HTTP status | Error returned to caller; next request may load a different model |
| llama-swap circuit open | Health tracker | Error: no backend available |
| Bonsai inference fails | Rust error | Default to `Cloud` (safe default — preserves quality) |
| Bonsai OOM / crash | spawn_blocking panic caught | Default to `Cloud` |
| Daemon not running when MCP connects | UDS connect error | `brainrouter mcp` exits with a clear error message |

## What this is not

- **Not a provider adapter.** brainrouter does not implement Anthropic, OpenAI, Google, or any cloud provider API. Manifest handles that.
- **Not a model runner.** brainrouter does not spawn llama-server processes. llama-swap handles that.
- **Not an auth layer.** brainrouter is localhost-only with no authentication on the proxy boundary. Credentials live in provider configs (Manifest dashboard, llama-swap config).
- **Not a conversation store.** Chat history is managed by the harness. brainrouter is stateless for proxy calls; review sessions are in-memory and lost on daemon restart.

## V2 ideas (not planned)

- Bonsai as a Strategic Context Expert — real-time synthesis of cloud agent output
- Interrupt-and-redirect: user types a correction mid-stream; brainrouter cancels and re-prompts
- Persistent review sessions (SQLite) so they survive daemon restarts
- Pi extension shipping alongside the main binary
- Token usage tracking and routing cost dashboard
