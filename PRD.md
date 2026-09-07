# brainrouter PRD

**Status:** Shipped proxy/control plane, benchmark explorer, and model observability; future work is explicitly separated below
**Language:** Rust
**Binary:** `target/release/brainrouter`
**Implementation baseline:** `3691dcc9ee7dee98f8355328d7e2e75f99faafdb` (2026-09-07)
**Config:** `$XDG_CONFIG_HOME/brainrouter/brainrouter.yaml` or `~/.config/brainrouter/brainrouter.yaml`; `serve --config` overrides it. Service environment files supply credentials; the daemon resolves configured environment-variable names.
**Repository:** https://github.com/ajaxdude/brainrouter
**Validation:** 192 Rust tests and 31 Node browser-logic tests passed at this baseline; see [Validation](#validation)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Architecture Decisions](#architecture-decisions)
- [Component Map](#component-map)
- [Configuration Reference](#configuration-reference)
- [Routing Flow](#routing-flow)
- [Review Flow](#review-flow)
- [Benchmark Explorer](#benchmark-explorer)
- [Model Observability and Regression Alerts](#model-observability-and-regression-alerts)
- [Bridge Architecture](#bridge-architecture)
- [HTTP API Reference](#http-api-reference)
- [MCP Tools Reference](#mcp-tools-reference)
- [Harness Compatibility](#harness-compatibility)
- [Failure Modes and Mitigations](#failure-modes-and-mitigations)
- [Security Model](#security-model)
- [What This Is Not](#what-this-is-not)
- [Validation](#validation)
- [Planned and Unshipped Work](#planned-and-unshipped-work)

---

## Problem Statement

A developer running multiple LLM subscriptions (Anthropic, OpenAI/Copilot, Google, Mistral) and local models constantly hits:

1. **Rate limits and quota exhaustion** -- one provider fails mid-session, work stops.
2. **Model selection overhead** -- manually deciding cloud vs local before every query burns time.
3. **Review token waste** -- every code review cycle through a cloud LLM costs premium quota.
4. **Harness fragmentation** -- each tool (omp, vibe, claude, opencode, codex, droid) has its own provider config; keeping them aligned is manual work.
5. **Role confusion** -- the main author, reviewer, and local subagent pool need independent choices without silently overriding explicitly selected models.
6. **Unreliable comparisons** -- performance numbers lose meaning without artifact, runtime, hardware, workload, configuration, and measurement provenance.
7. **Operational blind spots** -- current request activity, completed-stream measurements, and historical benchmarks need a common inspection surface without treating missing data as zero or adding load to inference.

---

## Solution

A single Rust daemon that:

1. **Separates routing roles.** The persisted Main choice handles `auto`, `brainrouter/auto`, and empty model selections; Reviewer has its own local/cloud/default choice; only `subs` or `brainrouter/subs` uses the separately selected local pool. Named presets configure compatible combinations without merging these roles.
2. **Preserves explicit choices and fallback semantics.** Bare model IDs and `brainrouter/<id>` select that local model with no rewrite or model substitution on failure. `cloud/<id>` sends the exact ID to Manifest. Explicit `local`/`cloud` aliases select backend defaults, not the Main override. Cloud requests retain the existing local fallback policy on disabled/unavailable Manifest; errors after streaming begins surface to the client rather than replaying output.
3. **Reviews code locally by default**, with explicit local or cloud reviewer IDs, requested-versus-actual reporting, and a reviewer snapshot retained for the lifetime of each review and its continuations.
4. **Manages system state** via the dashboard: one-click upgrades of llama-swap, resets of the llama.cpp toolbox, start/stop of the Bonsai classifier, and flushing of loaded models — full VRAM control without a terminal.
5. **Keeps opt-ins explicit.** Bonsai and Manifest are disabled by default. A role set to Auto consults the classifier only when enabled; otherwise it selects local. Applying a cloud or hybrid preset does not enable Manifest.
6. **Presents a single OpenAI-compatible endpoint** to all harnesses, plus an Anthropic-compatible endpoint for harnesses (Claude Code, droid) that speak Anthropic's protocol natively.
7. **Bridges chat platforms** -- Discord and Signal transports shell out to `omp` CLI, bringing LLM access to messaging apps with session management, model selection, and working directory tracking. Commands use the `!br` prefix.
8. **Installs itself into harnesses** via an idempotent `install` subcommand that configures 7 harnesses (omp, vibe, opencode, codex, droid, claude, pi) in a single command.
9. **Tracks in-flight requests live.** A registry records every request at handler entry (before routing, so the dashboard sees it during model load) and holds it until the stream ends or the user cancels it. The dashboard shows one row per active request — elapsed time, model, user agent, address, session, bytes received, PP progress, and an activity label (tool calling / reasoning / asking a multiple choice question / generating) — with a per-row Cancel button.
10. **Measures completed requests by unique event ID.** Actual first-output timing and provider-reported token counts support bounded recent measurements across both wire protocols. Concurrent turns in the same conversation cannot overwrite each other's samples. Missing prompt-processing speed is not inferred from TTFT.
11. **Explores imported benchmarks.** A normalized SQLite registry, strict Rust boundary models, immutable identities, transactional previews/imports, deterministic matrix expansion, structured inspector, telemetry plots, and bounded CSV/JSONL exports are exposed at `/benchmarks`.
12. **Observes model activity without running benchmarks.** `/models` combines exact local model keys, read-only backend metadata, active requests, recent completed measurements, indicative regression warnings, and explicit benchmark references.
13. **Isolates optional work.** Unavailable benchmark storage disables only explorer routes. Bounded blocking workers and response budgets keep benchmark work off the async routing executor. No benchmark runner, scheduler, automatic model switching, or outbound alerts are part of these additions.

---

## Architecture Decisions

### Bonsai as classifier, not a responder

The configured Bonsai model runs as an external llama-server process (PrismML fork of llama.cpp) on a dedicated port (default 9200). Brainrouter launches it at startup only when `bonsai.enabled: true`, waits for `/health`, then classifies through `/v1/chat/completions`. Decisions include cloud, local/light, and local/deep. Timing depends on hardware and workload; there is no guaranteed classification-latency SLA. `BonsaiControl` owns start/stop. An Auto role skips classification and chooses local when the classifier is disabled; enabling Bonsai does not change a Main role already pinned to local or cloud.

Bonsai was chosen specifically because it is purpose-trained to understand task complexity and model capability -- not as a general assistant. Using it only for routing preserves VRAM for llama-swap models.

### Manifest handles all cloud routing

Brainrouter connects to one OpenAI-compatible Manifest endpoint and one configured API-key environment variable. Default cloud selections send `model: "auto"`; explicit cloud role IDs and `cloud/<id>` pass the provider ID unchanged. Manifest selects the downstream cloud provider. The actual model is taken from reported response metadata, not assumed from the requested name. **Manifest is off by default**; cloud-disabled requests preserve local fallback behavior and expose that outcome.

The coupling is minimal: if Manifest is replaced with LiteLLM or OpenRouter, one URL in `brainrouter.yaml` changes.

### Two wire protocols on one port

- `POST /v1/chat/completions` -- OpenAI format. Covers vibe, opencode, codex, omp, pi (via extension).
- `POST /v1/messages` -- Anthropic Messages API format. Covers Claude Code (via `ANTHROPIC_BASE_URL`) and droid (via `provider: "anthropic"` in custom_models).

Internally the request is immediately translated to OpenAI format and routed through the same Bonsai / Manifest / llama-swap pipeline. Response is translated back to Anthropic SSE events at the edge.
Proxy responses are streaming SSE; non-streaming provider completion APIs and
OpenAI Responses endpoints are not implemented.

### MCP as thin client

The `brainrouter mcp` process does not load Bonsai and does not run the review loop. It is a JSON-RPC stdio server that maps four tool calls to HTTP requests against the daemon's UDS. This keeps harness cold-start fast (no model load) and keeps all state -- sessions, health tracker, circuit breakers -- in one place. Both `mcp` and the CLI share one `DaemonClient` (src/daemon_client.rs) so there is exactly one HTTP client path into the daemon.

### Headless control plane

`brainrouter cli` uses `DaemonClient` over UDS by default or TCP with `--url`. It covers core management, routing profiles, catalogs, review configuration, and review lifecycle. It does not start a second inference service. Review requests poll every five seconds with a 30-minute cap, or return immediately with `--async`. Benchmark and observability workflows use the documented HTTP API or their pages; dedicated CLI subcommands for those workflows are not shipped.

### Review loop uses the same router

When an agent requests a review, the service snapshots its Reviewer choice and iteration settings. The loop uses `Router::route_with_choice`, independently of the Main and subagent choices. This means:

- Explicit local/cloud reviewer IDs are honored; an Auto reviewer follows classifier behavior, not the current Main override.
- If Manifest is down, review calls fall back to llama-swap automatically.
- The same provider clients, circuit breakers, and streaming protections are reused.
- Changing global preferences affects new reviews, not the reviewer of an existing session or continuation.

### Circuit breaker

Two independent circuit breakers: one for `manifest`, one for `llama-swap`. Three failures within a window open the breaker; 60-second cooldown before retry. This prevents hammering a degraded provider while allowing automatic recovery.

### Robust Anthropic SSE adapter

The adapter translates content/tool events and protocol termination, handles split lines, and reports stream failures. It drains final usage, `[DONE]`, and clean EOF before final Anthropic terminal events so measurement capture is not abandoned at `finish_reason`. Tail waiting is bounded to two seconds from the first finish/DONE marker, with additional 256 KiB total, 64 KiB line, and 1,024 subsequent-chunk budgets. Content continues streaming; invalid, cancelled, errored, or incomplete tails do not produce successful performance samples.

### Security: localhost-only with CSRF protection

Destructive operations (upgrade, restart) are restricted to loopback interfaces (127.0.0.1, ::1) or Unix Domain Sockets. Browser-originated requests to management endpoints are validated against `Origin`/`Referer` headers. Working directory tracking for sessions includes absolute-path enforcement, null-byte rejection, and path-traversal component blocking.

### Prompt rewriting for local models (coupled to Bonsai)

When enabled on a managed local route, the prompt rewriter replaces verbose harness instructions with a lean local system prompt and anti-loop directives. It is off at daemon startup. Direct local IDs, including explicit local role models, bypass rewriting; not every `auto` alias necessarily reaches a managed local route.

The rewrite is coupled to the classifier **server-side**, not just in the UI: turning Bonsai off via the toggle/CLI forces the rewrite flag off with it (the invariant "rewrite on ⇒ Bonsai on" holds in the daemon), and `POST /api/prompt-rewrite` refuses to enable while Bonsai is down. Direct model picks are never rewritten — only managed local routes.

### Bridge as OMP subprocess

Discord and Signal transports do not call LLMs directly. They shell out to the `omp` CLI as a subprocess, which connects back to brainrouter through the normal proxy path. This means bridge conversations get the same routing, fallback, and review capabilities as direct harness usage. Session state, working directories, and model aliases are persisted to disk per-transport.

### Peer CWD resolution

The MCP server needs to know the caller's working directory. On Linux, brainrouter resolves this by mapping the peer's socket connection (TCP or UDS) back to a PID via `/proc`, then reading `/proc/<pid>/cwd`. This is a Linux-native approach that avoids requiring the caller to pass CWD explicitly, though callers may override it.

### In-flight request registry (src/inflight.rs)

The registry gives the dashboard a live view of every in-flight request — the "monitor LLM activity without opening llama-swap" goal. Each request registers an entry at handler entry, before `route_tagged`, so the row is visible during model load (when the registry is otherwise idle). The entry's identity (`InflightHandle`) is an `Arc` carried by the spawned routing task and, once routing resolves, by a `SniffStream` that wraps the provider's SSE stream. The row is removed when the last handle drops — i.e. at stream end, on a routing error (no stream is created, so the task-side handle is the last one), or on an explicit dashboard Cancel (a `watch` channel flips a flag; the `SniffStream` ends the stream immediately and the handles drop). Stale entries (older than a grace window) are swept on read.

The activity label is derived by sniffing the head window of each SSE chunk for provider-specific markers (`tool_calls` / `toolUse` → tool calling, `reasoning_content` / `thinking` → reasoning, `multiple choice` → asking, otherwise generating). The sniff is **sticky with a priority order** — tool calling > reasoning > asking > generating — so a strong signal is never downgraded by a later weaker one.

`model` is an interior-mutable `Mutex<String>`: it starts as the requested model and is updated to the resolved llama-swap model key after routing.

### Per-request token throughput (src/router.rs, src/routing_events.rs)

A stream wrapper records successful measurements only after `[DONE]` and clean EOF, keyed to the unique ID returned by `RoutingEvents::emit`. It never assigns data to the latest event merely because it shares a conversation. Measured TTFT ends at the first nonempty content/reasoning/tool-argument frame; role-only frames and heartbeats do not count. Generation TPS uses `(completion_tokens - 1) / (last output time - first output time)` only when reported usage and distinct observations support it. It is an estimate affected by buffering/backpressure, not engine decode instrumentation. PP throughput is not fabricated. See [Model Observability and Regression Alerts](#model-observability-and-regression-alerts) for retention, attribution, and comparability requirements.
---

## Component Map

| Component | File | Responsibility |
|---|---|---|
| Entry point | `src/main.rs` | Clap subcommand dispatch |
| Daemon startup | `src/daemon.rs` | Load config, create all services, validate environment, start server |
| HTTP server | `src/server.rs` | Route `/v1/*`, `/review/*`, `/api/*`, and `/dashboard` |
| Classifier | `src/classifier.rs` | Bonsai-based cloud/local decision |
| Bonsai server lifecycle | `src/bonsai_server.rs` | `BonsaiServer`: spawn, 60s health poll, SIGTERM→SIGKILL stop. `BonsaiControl`: runtime start/stop from the dashboard, shared enabled flag for the classifier |
| Router | `src/router.rs` | Dispatch to Manifest or llama-swap; fallback; timeout |
| Routing preferences | `src/routing_profile.rs` | Typed role choices/presets, strict writes, atomic per-user state, legacy migration |
| Prompt rewriter | `src/prompt_rewriter.rs` | System prompt rewriting for local mode (strips OMP bloat, injects anti-loop prompt) |
| Anthropic shim | `src/anthropic.rs` | `/v1/messages` to `/v1/chat/completions` translation; strict SSE state machine |
| MCP server | `src/mcp_server.rs` | JSON-RPC stdio; forwards to daemon over UDS |
| Installer | `src/install.rs` | Idempotent harness config merger for 7 harnesses |
| Session store | `src/session.rs` | In-memory `HashMap<id, Session>` behind `Mutex` |
| Config | `src/config.rs` | YAML parsing + validation |
| Types | `src/types.rs` | OpenAI request/response structs |
| Health tracker | `src/health.rs` | Per-provider circuit breaker |
| Stream wrapper | `src/stream.rs` | `TimeoutStream`: lazy-armed 180s inter-chunk stall detection; `SafeStream`: error-to-SSE converter |
| Routing events | `src/routing_events.rs` | In-memory event buffer for dashboard live feed |
| Benchmark store/domain | `src/benchmark.rs` | Validation, canonical identities, SQLite ingestion/query/detail/export, shared admission |
| Benchmark HTTP/imports | `src/benchmark/http.rs`, `src/benchmark/imports.rs` | Bounded workers/bodies/responses, rollback previews, reusable templates and examples |
| Benchmark schema | `migrations/0001_benchmark_explorer.sql` | Versioned normalized STRICT tables, indexes, summary view |
| Model observability | `src/observability.rs` | Read-only polling, completed measurements, rolling alerts, explicit reference settings |
| Explorer pages | `src/escalation/templates/benchmarks.html`, `src/escalation/templates/model_observability.html` | Embedded benchmark workflows and local model activity |
| Inference state | `src/inference_state.rs` | Track active inference status per provider |
| Peer CWD | `src/peer_cwd.rs` | Linux-native PID/inode mapping for directory discovery (IPv4/IPv6/UDS) |
| Lib | `src/lib.rs` | Crate-level re-exports |
| Review service | `src/review/mod.rs` | `start_review`, `resolve_session` |
| Review loop | `src/review/review_loop.rs` | Iterative LLM review, robust JSON parsing |
| Context gatherer | `src/review/context.rs` | PRD auto-detect, `git diff HEAD`, safe UTF-8 truncation |
| Prompt builder | `src/review/prompt.rs` | Review prompt template |
| Escalation UI | `src/escalation/mod.rs` | `/review/*` HTTP handlers + embedded HTML templates; CWD sanitization |
| Escalation templates | `src/escalation/templates/` | Embedded HTML for review session UI |
| In-flight registry | `src/inflight.rs` | `InflightRegistry` + `InflightHandle` + `SniffStream`: per-request live rows (elapsed, model, UA, address, session, bytes, PP, activity) with dashboard cancel; sticky head-window activity sniff |
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
| Role/upgrade tests | `tests/routing_profiles_test.rs`, `tests/daemon_availability_test.rs` | Exact roles, protocols, persistence, legacy upgrades, failure isolation, complete import/reference workflow |
| Browser-logic tests | `scripts/test-benchmark-ui.cjs` | Synthetic DOM/import/inspector/precision and asynchronous-state regressions |
| System installer | `install.sh` | One-script Fedora system-wide installer: packages, bun, oh-my-pi, Bonsai, Manifest, llama-swap, brainrouter, toolbox container, shared `/etc/brainrouter/` config, per-user services with linger |

---

## Configuration Reference

### System paths (multi-user install)

When installed via `install.sh` on a shared Fedora machine:

| Path | Purpose |
|---|---|
| `/usr/local/bin/brainrouter` | System-wide brainrouter binary |
| `/usr/local/bin/llama-server-toolbox` | Wrapper: runs `llama-server` inside the `llama-vulkan-radv` toolbox container |
| `/etc/brainrouter/brainrouter.yaml` | Shared base config (readable by all `aistack` group members) |
| `/etc/brainrouter/env` | Shared env file holding `MANIFEST_API_KEY` (`root:aistack`, mode `640`) |
| `/etc/skel/.config/systemd/user/brainrouter.service` | Template brainrouter service for new user accounts |
| `/etc/profile.d/ai-stack.sh` | Sets `PATH`, `ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL` for all login shells |
| `/opt/models/bonsai/` | Shared GGUF model storage (`root:aistack`, setgid `2775`) |
| `/opt/ai/llama-swap/` | llama-swap Docker compose stack |
| `/opt/ai/manifest/` | Manifest Docker compose stack |
| `/etc/systemd/system/llama-swap.service` | System service: starts llama-swap Docker stack at boot |
| `/etc/systemd/system/manifest.service` | System service: starts Manifest Docker stack at boot |

### YAML Configuration (`brainrouter.yaml`)

#### manifest.*

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` | Cloud routing is **off by default**. When `false`, `route_cloud` skips Manifest and falls back to llama-swap |
| `base_url` | `String` | *(required)* | URL of the Manifest instance. Validated: must start with `http://` or `https://` |
| `api_key_env` | `String?` | `None` | Name of the environment variable holding the `mnfst_*` API key (NOT the key itself) |

#### llama_swap.*

| Field | Type | Default | Description |
|---|---|---|---|
| `base_url` | `String` | *(required)* | URL of the llama-swap instance |
| `fallback_model` | `String` | *(required)* | Model key to use when Manifest fails or Bonsai picks local. Must match a key in llama-swap config |
| `local_models` | `Vec<String>?` | `[]` | Explicit llama-swap model keys; a request using one of them as `model=` routes straight to llama-swap, bypassing Bonsai |
| `subs_model` | `String?` | `None` | Subs-pool model key; `model=subs` / `brainrouter/subs` routes here, bypassing Bonsai. Absent → `subs` falls back to auto |
| `local_system_prompt` | `String?` | `None` | Path to a custom system prompt file for `model=local` mode. Built-in lean prompt used if absent |
| `nudge.enabled` | `bool` | `false` | Thinking-budget nudge: inject `reasoning_budget_tokens` into local routes when the client didn't set one |
| `nudge.model_key` | `String?` | `None` | Local model key that receives the nudge; when `None`, nudge applies to the routing target |
| `nudge.budgets.light` | `u32` | `10240` | Budget injected when the tier is `light`. Legacy configs may use `local` (accepted via serde alias) |
| `nudge.budgets.deep` | `u32` | `12288` | Budget injected when the tier is `deep` |

#### bonsai.*

| Field | Type | Required | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` | Classifier is **off by default**. When `false`, `auto` routing goes straight to local (no cloud hop) |
| `model_path` | `PathBuf?` | *(required when enabled)* | Configured Bonsai GGUF path; `${models_path}` is expanded. Existence is checked only when enabled, so missing classifier assets do not prevent a local-only startup |
| `fork_path` | `PathBuf` | `$HOME/.local/share/brainrouter/llama-prism/llama-server` | Configured PrismML fork binary; do not assume a particular host's build or quantization |
| `server_port` | `u16` | `9200` | Port for the external Bonsai llama-server process |
#### models.*

| Field | Type | Default | Description |
|---|---|---|---|
| `path` | `PathBuf` | `/opt/models` | Shared GGUF model directory; `${models_path}` inside `bonsai.model_path` expands to this value |
| `shared_write` | `bool` | `false` | When true, all `aistack` group members may add/delete models (dir mode `770`); when false, only the owner can write (mode `750`) |

#### review.*

| Field | Type | Default | Description |
|---|---|---|---|
| `max_iterations` | `u32` | `5` | Positive maximum LLM review iterations before escalation to human |
| `forced_mode` | `String` | `"local"` | Legacy YAML/default reviewer choice: `auto`, `cloud`, or `local`. Role profiles and saved preferences can override it |
| `forced_model` | `String?` | `None` | Exact local OR cloud provider ID; null selects that backend's default. New auto+model combinations are invalid |

#### routing.*

Optional typed YAML defaults: `{preset, main, reviewer, subagent_model}`.
Main/Reviewer are `{backend: auto}` or `{backend: local|cloud, model: null|"<explicit-id>"}`.
Subagent is a local model ID or null. Presets are `auto`, `cloud`,
`local_main_sub`, `local_custom`, `cloud_main_local_review`,
`local_main_cloud_review`, and `custom`; role backends must match the preset.
`local_custom` requires an explicit local Main model. Applying a preset preserves
the independently selected subagent pool.

Without `routing`, Main defaults to local, Reviewer derives from `review`,
and the pool derives from `llama_swap.subs_model`. Per-user
`$XDG_CONFIG_HOME/brainrouter/routing_state.json` (otherwise under `~/.config`)
takes precedence over YAML role defaults. Writes validate before committing
runtime state and use atomic owner-only files. `review.max_iterations` is
YAML-owned across restarts.

When the new state file is absent, valid legacy `review_state.json` overrides
migrate and persist once; the legacy file is not deleted. File reads alone
normalize the old, ignored `forced_mode: auto` + `forced_model` combination,
warning with the source path and removal guidance. New write APIs remain strict;
local/cloud choices and other errors are not silently normalized. Migration
errors identify their source/destination and cause.
Resetting to YAML requires moving aside both new routing state and any retained
legacy state while the daemon is stopped; removing only the new file can
trigger legacy migration again.

#### benchmarks.*

`database_path` is optional and defaults to
`$XDG_DATA_HOME/brainrouter/benchmarks.sqlite3`, or
`~/.local/share/brainrouter/benchmarks.sqlite3`. Use an explicit absolute path for
an override; no shell `~` expansion is performed on a configured path. Initialization
failure produces a logged explorer-only HTTP 503, not a core daemon startup failure.
Repair the path/database and restart to retry; newer schemas are never downgraded.

#### Observability settings (not YAML fields)

Alert policy and explicit baseline mappings live beside the selected config:
`brainrouter.yaml` -> `brainrouter.observability.json` (`Path::with_extension`).
The page/API performs revision-checked writes; copy the latest revision and full
policy from `GET /api/observability/settings` before changing it. Corrupt/unreadable
settings expose errors and disable saves until repaired/restarted, but leave
core routing and live observations available. Revision counters reset on restart;
clients must reload settings before writing.

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
| `--config` | `$XDG_CONFIG_HOME/brainrouter/brainrouter.yaml`, otherwise `~/.config/brainrouter/brainrouter.yaml` | Path to YAML configuration file |
| `--tcp-addr` | `127.0.0.1:9099` | TCP bind address |
| `--socket` | `$XDG_RUNTIME_DIR/brainrouter.sock`, otherwise `/run/brainrouter.sock` | Unix domain socket path |

#### `brainrouter mcp`

| Argument | Default | Description |
|---|---|---|
| `--socket` | Same runtime-directory default as `serve` | UDS path to connect to the daemon |

#### `brainrouter cli <command>`

Thin client for core management, roles, and reviews. `--socket <path>` selects the daemon UDS, `--url http://…` talks TCP instead. Output is JSON except raw YAML config commands. Explorer/observability workflows additionally use HTTP directly.

| Command | Description |
|---|---|
| `status` | Overall health: llama-swap, manifest, llama.cpp, toolbox, cloud-fallback |
| `versions` | Installed versions for all components + latest available |
| `inference` | Live inference status: loaded model, state, slot progress |
| `events` | Recent routing events feed |
| `stats` | Aggregated routing statistics |
| `models [--llama-swap]` | Model list (`/v1/models` view, or raw llama-swap keys with `--llama-swap`) |
| `bonsai status` / `enable` / `disable` / `toggle` | Classifier server control (idempotent: `enable`/`disable` are no-ops when already in that state) |
| `nudge status` / `enable` / `disable` / `toggle` / `tier <auto\|light\|deep>` | Thinking-budget nudge control (`light` = tight budget; legacy API spelling `local` still accepted) |
| `prompt-rewrite status` / `enable` / `disable` | Local prompt-rewrite toggle |
| `context status` / `set <tokens\|auto>` | Legacy CLI only: validates 2048..262144 (auto = 131072), but `/api/context` has no server handler at this baseline and returns 404 |
| `routing-mode status` / `set <auto\|cloud\|local>` | Routing-mode override |
| `routing-profile status` / `models` / `preset` / `choose` / `pool` / `set` | Inspect/replace typed preferences, choose exact Main/Reviewer models, and configure the independent local pool |
| `bridges status` / `enable` / `disable` / `toggle <discord\|signal>` | Bridge control (toggle flips the current state) |
| `toolboxes` | List llama-* toolbox containers |
| `restart <llama-swap\|llama-cpp\|manifest\|brainrouter>` | Restart a service. `llama-cpp` is the dashboard's `llama.cpp` row; `toolbox` is the `toolboxes` image |
| `upgrade <llama-swap\|manifest\|toolbox>` | Upgrade a component |
| `flush-models` | Unload every model from llama-swap (frees VRAM) |
| `sync-omp` | Push llama-swap models into OMP's models.yml (one-way) |
| `config-files` | List config files the daemon manages |
| `brainrouter-config show` / `set <path\|->` | Read brainrouter.yaml; write one from a file or stdin (`-`). Alias: `config` |
| `llama-swap-config show` / `set <path\|->` | Same for llama-swap's config.yaml |
| `review-config status` / `update [--max-iterations N] [--forced-mode auto\|cloud\|local] [--forced-model KEY\|--clear-model]` | Current/new-session reviewer configuration; continuations keep the initial snapshot |
| `review list` | All review sessions |
| `review get <sessionId>` | One session's details |
| `review request <taskId> <summary> [--details TEXT] [--cwd DIR] [--async]` | Request a review; blocks with progress (polls every 5 s, 30-min cap) unless `--async` |
| `review continue <sessionId>` | Additional LLM review rounds (seeds from the persisted turn history) |
| `review lgtm <sessionId>` | Quick-approve a session |
| `review resolve <sessionId> <feedback>` | Resolve a session with feedback |

`review get/continue/lgtm/resolve` validate `sessionId` (alphanumeric + `-`/`_`) before URL use — path-traversal guard.

#### `brainrouter install <harness>`

| Argument | Default | Description |
|---|---|---|
| `<harness>` | *(required)* | One of: `omp`, `vibe`, `opencode`, `codex`, `droid`, `claude`, `pi` |
| `--yes` | `false` | Skip confirmation prompts |
| `--shell-rc` | `false` | Boolean flag to append Claude environment variables to the shell RC |
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
| `XDG_CONFIG_HOME` | User YAML/config directory and `routing_state.json` location |
| `XDG_DATA_HOME` | Default benchmark database base directory; falls back to `~/.local/share` |
| `BRAINROUTER_MANIFEST_DIR` | Override directory for Manifest configuration |
| *(dynamic)* | Whatever `manifest.api_key_env` names (e.g., `MANIFEST_API_KEY`) |

### Hardcoded Constants

| Constant | Value | Description |
|---|---|---|
| `FAILURE_THRESHOLD` | `3` | Failures before circuit breaker opens |
| `COOLDOWN_PERIOD` | `60s` | Circuit breaker cooldown before retry |
| `STREAM_STALL_TIMEOUT` | `180s` | Maximum time between SSE chunks before timeout |
| `TTFT_TIMEOUT` / deferred routing wait | `600s` | Provider/routing wait budget; not a promised first-output latency |
| `MAX_EVENTS` | `500` | Maximum routing events kept in memory for dashboard feed |
| `CLASSIFY_MAX_TOKENS` | `10` | Maximum tokens for Bonsai classification response |
| `USER_MSG_TRUNCATE` | `800 chars` | Truncation limit for user message sent to classifier |
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
  v server.rs: deserialize, translate if Anthropic; extract_session_id()
  |           reads a client conversation header (x-omp-session / x-session-id /
  |           x-conv-id / x-conversation-id / x-client-session / x-request-conv)
  |           so the dashboard can group events per conversation
  |
  v router.rs: default/auto aliases select the Main profile first
  |           review calls instead use the session's Reviewer snapshot
  |
  v match on the resolved selector — only managed routing tokens get
  |           Bonsai/nudge/subs treatment; every other key is a DIRECT model
  |           selection and is authoritative (bypasses Bonsai entirely)
  |
  +-- resolved Auto role --> classifier.rs (skipped if Bonsai disabled)
  |     +-- Cloud --> manifest (if enabled + healthy) --> llama-swap fallback
  |     +-- Local --> prompt_rewriter.rs (when rewrite on) --> llama-swap (Bonsai-chosen model)
  |
  +-- model="local" --> prompt_rewriter.rs only when rewriting is enabled
  |                  --> llama-swap (fallback_model)
  |
  +-- model="cloud" --> manifest (if enabled + healthy) --> llama-swap fallback
  +-- model="cloud/<id>" --> exact Manifest model --> same cloud fallback policy
  |
  +-- model="subs" / "brainrouter/subs" --> current local pool, bypassing Bonsai;
  |     unconfigured -> classifier/auto, NOT the Main profile override
  |
  +-- model="brainrouter/<key>" or any bare key --> llama-swap (that key), no rewrite,
  |     no classifier hop; works even when Bonsai is off
  |
  +-- nudge (if enabled): inject reasoning_budget_tokens on local routes
  |     unless the client already supplied one
  |
  v provider/openai.rs: HTTP request to chosen backend
  |
  v stream.rs: timeout/keepalive protection; event-correlated measurement wrapper
  |
  v health.rs: record success/failure for circuit breaker
  |
  v If Anthropic client: anthropic.rs SSE state machine translates back
  |
  v Response streamed to caller
```

### Fallback semantics and hop chains

`route_local` takes an `allow_fallback` gate:

- **Managed routing** (`auto`/`local` tokens) passes `allow_fallback: true` — if the chosen local model fails, the router retries once with `fallback_model`.
- **Direct model picks** (any named/bare key, `subs`, specific-key routes) pass `allow_fallback: false` — a failed explicit selection surfaces the error to the client instead of silently switching models. This is the "memory compaction" fix: a failing `ds4-*` pick used to jump to the subs pool behind the agent's back; direct picks are authoritative.

Every failed hop is recorded in `RouteInfo.failed_attempts` (stage, provider, model_key, error) and the router emits one `RouteEvent` **per failed hop** plus one for the winner. The dashboard therefore renders multi-hop routes as a hop chain (e.g. `manifest ✗ → local ✓`) inside the card body; the Sankey ROUTING column embeds the chain in the routing node key itself.

Both Bonsai and Manifest disabled (the default): `auto` → local, `cloud` → local — every request is a single hop to llama-swap.

---

## Review Flow

```
Agent calls mcp_brainrouter_request_review (or: brainrouter cli review request <taskId> <summary>)
  |
  v mcp_server.rs / cli.rs: POST /review/api/request-async over UDS
  |
  v escalation/mod.rs: parse request (invalid cwd → HTTP 400, no silent fallback),
    call ReviewService::start_review_async
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
        router.route_with_choice(snapshot) --> selected local/cloud/default reviewer
        parse JSON response (STATUS: approved | needs_revision)
        update session + llm_turns in the in-memory session store
        if approved --> return success
      if max_iterations reached --> escalate to human UI
  |
  v response: {sessionId} (async) — CLI/MCP poll GET /review/api/sessions/{id} every 5s
  |
  v Dashboard: GET /review/ redirects to /dashboard; JSON/session detail APIs remain available
  |
  v Human resolve: POST /review/api/resolve (or cli review resolve / lgtm)
  |
  v Continue: POST /review/api/continue — registers a notifier and waits for
    human resolution if the continuation escalates; seeds from llm_turns
```

`/review/api/*` POST routes pass the same `is_destructive` gate as the other
management endpoints (loopback-only + Origin check), closing the browser-CSRF
hole. The legacy blocking `POST /review/api/request` endpoint is still routed
but no client uses it; CLI/MCP use `request-async` + polling.

---

## Benchmark Explorer

### Shipped scope

This is an import/planning/inspection subsystem inside the existing Rust/Hyper
daemon, with embedded HTML/JavaScript and SVG charts. It is not a Python/FastAPI
sidecar, a model runner, or a benchmark scheduler. Opening pages, discovering
metadata, planning matrices, and previewing imports must not start inference.

### Registry, identity, and provenance

| Entity / table | Shipped responsibility |
|---|---|
| `models` | Family, architecture, checkpoint/revision, tokenizer, parameter counts, native context, metadata |
| `artifacts` | Model reference, format/quantization, disk size/BPW, declared SHA-256/source and conversion provenance |
| `runtimes` | Repository/fork/commit, compiler/backend, build flags, executable/container identities, feature capabilities |
| `hardware_profiles` | CPU/cores/RAM, per-GPU definitions, unified-memory flag, OS/kernel/drivers, capture time |
| `workloads` | Versioned manifest identity, workload kind, input/output/corpus counts and license |
| `experiment_specs`, `experiments` | Canonical configuration and its SHA-256 identity; registry references, context/token/batch/thread/optimization/sampling settings |
| `runs` | Zero-based repetition, status, exact command, timestamps, seed/warmup, environment, paths, raw result and failure reason |
| `performance_metrics`, `speculative_metrics` | Optional one-to-one measurements per run |
| `quality_results`, `telemetry_samples` | Per-task quality and ordered host/per-GPU observations |
| `entity_fingerprints` | Reject payload changes under reused registry/experiment identities |
| `schema_migrations`, indexes, `run_summary` | Schema/version bookkeeping and joined query support |
| `exclusions` | Reserved schema table; the current planner returns exclusions in its response rather than persisting them |

The initial migration creates normalized SQLite **STRICT** tables and initializes
WAL. Each connection enables foreign keys and a five-second busy timeout.
Startup checks the migration identity/version and required tables. A future,
corrupt, or unusable database disables only the explorer; it is not reset or
downgraded. This is a new dedicated database, not a conversion of ephemeral review
sessions or the live routing buffer.

Rust boundary models reject unknown structural fields and invalid enums,
negative/out-of-range metrics, malformed lowercase SHA-256 values, invalid
timestamp order, invalid parameter/test/proposal relationships, and integers
too large for normalized SQLite integer columns. Backend-specific metadata remains JSON. Feature
states distinguish `unsupported`, `disabled`, `enabled`, and
`requested_unavailable`; speculation kinds are recorded, not executed.

An experiment hash covers the canonical serialized registry IDs and complete
experiment settings but excludes its application ID. Registry fingerprints
protect those referenced identities from later mutation. Artifact, executable,
manifest, and build metadata are **declarations supplied by the importer**; the
daemon does not hash/download files or attest what a runtime actually loaded.
Declared draft-artifact references must exist.

Runs are uniquely identified by ID and `(experiment_id, repetition)`. Those
dimensions cannot move. A succeeded run is immutable; a correction requires a
new repetition or experiment. Other statuses (`planned`, `running`, `failed`,
`oom`, `timeout`, `cancelled`, `skipped`) may receive a replacement full bundle
under the same attempt identity. This is not an append-only live telemetry feed.

### Import and matrix requirements

1. A complete bundle includes `model`, `artifact`, `runtime`, `hardware`,
   `workload`, `experiment`, and `run`, with optional performance, speculation,
   quality, and telemetry sections. Cross-references must agree.
2. Validation/prepare endpoints exercise transactional ingestion checks and
   roll back. Only explicit ingestion commits; previews neither reserve IDs nor
   guarantee a later concurrent import will succeed.
3. Reusable templates contain registry definitions plus an experiment, not a
   run payload. `/prepare` combines a template, explicit repetition/status/command,
   optional timestamps/llama-bench output, and optional selected plan candidate.
   Existing matching custom IDs are reused.
4. The llama-bench adapter accepts one object, an array, or a `results` array.
   It rejects ambiguous/repeated/contradictory phases, invalid metric types,
   conflicting bundle metrics, and incompatible reported token counts.
   Generic `avg_ts`/`tokens_per_second`/`tps` must resolve to one phase; combined
   tests require separate prompt and generation rates. Raw uploaded output is
   retained under `run.raw_result.llama_bench`.
5. JSON/YAML matrices deterministically expand registry IDs, contexts, token
   counts, and optimization combinations. Duplicate dimensions/configurations
   are invalid; prompts exceeding context are returned as exclusions. Planning
   does not validate artifact existence on disk, runtime feasibility, or actual
   capability support. Repetition/randomization/telemetry fields are execution
   hints for external tooling, not actions performed by this daemon.
6. The UI supports file upload, validation preview, explicit confirmation,
   success links, opt-in browser-local template reuse, and plan-candidate
   selection. Supplied examples are clearly labelled **synthetic**, with matching
   IDs and 8K/32K/128K contexts; examples never auto-populate the database.

### Queries, inspection, and export

`GET /api/benchmarks/runs` returns
`{items,page,per_page,total,total_pages}`. Defaults are page 1, 25 rows, and
`sort=started_at&order=desc`. `page` must be positive; `per_page` is 1..100.
Exact filters are `status`, `family`, `backend`, `workload`, `quant_name`, and
`speculator_type`. `q` is literal substring search across model/runtime/workload/run
labels with SQLite's ASCII case-insensitive LIKE behavior; `%`/`_` are escaped.
Blank filters are omitted. Repeated parameters take their last value; unknown
parameters, sort fields, and orders are rejected. Unknown exact filter values
simply match no rows.

Sort fields are `started_at`, `family`, `workload`, `context_tokens`,
`prompt_tps`, `generation_tps`, and `ttft_ms`, with `asc`/`desc`. Timestamp sorting
uses creation time when start time is absent. Missing numeric metrics sort last;
ties use ascending run ID. A page beyond the result count is empty.

The inspector presents all configuration/provenance, performance/latency,
speculation and task-quality data, with ordered host/per-GPU sparklines that keep
null gaps and never sum duplicated host readings. Commands, URLs, and paths are
inert text. Raw JSON view/download preserves original response text; the browser
warns on unsafe JavaScript numeric precision and rejects unsafe import rewrites.
`quality_score` in summaries is the mean of `pass@1` results only, not an average
of heterogeneous metrics.

The page's scatter/Pareto, context, runtime, quantization, and pass@1-per-GiB
views use the **current page only (at most 25 runs)**. The total matching count is
separate. Changing page, order, or status selection changes the displayed sample;
these views are not controlled cross-workload experiments.

CSV/JSONL export ignores UI pagination and uses one consistent SQLite snapshot.
It returns all matching **summaries** within the documented budgets or an
explicit limit error, not partial data. CSV escapes delimiters/newlines and
neutralizes spreadsheet formula prefixes. It is not a full database backup,
bundle archive, telemetry export, or Parquet implementation.

### Reliability and resource budgets

| Resource | Contract |
|---|---|
| Shared admission | Two slots across store clones and `run_blocking`; saturation returns 503 with `Retry-After: 1`, no waiting queue |
| Benchmark POST body | 16 MiB; 15-second collection deadline (413 / 408) |
| Benchmark URI | 8,192 bytes |
| HTTP SQLite row / JSON expression | 4 MiB |
| Serialized response / export / expanded plan | 8 MiB; explicit 413 on excess |
| Run detail | 10,000 telemetry rows and 1,000 quality rows, additionally byte-limited |
| Filter choices | 1,000 distinct values per field |
| Matrix | 10,000 candidate combinations, additionally response-budgeted |
| Filtered export | 10,000 summaries, additionally byte-limited |
| SQLite lock waiting | Five seconds; contention returns 503 |

HTTP parsing/validation/planning/SQLite/serialization executes on bounded
blocking workers. Admission survives cancellation until actual blocking work
exits, and survives Hyper body drop while response `Bytes` remain queued; output
frames are capped at 16 KiB. A disconnected ingestion may already have committed:
inspect its run ID before retrying. Larger records remain in SQLite for offline
or synchronous consumers; HTTP does not silently sample/truncate them. Async
internal consumers must reuse `BenchmarkStore::run_blocking`.

## Model Observability and Regression Alerts

### Shipped monitoring surface

`/models` (including `?model=<encoded-key>`) combines exact **local** model keys
from backend status and observed routes with active requests, reported metadata,
completed measurements, and explicit historical references. Read-only polls use
`/running`, `/v1/models`, and `/props` for at most four already-ready proxies.
No poll loads a model. Backend requests have two-second deadlines, 1 MiB bodies,
and a 256-model catalog limit; there is a five-second pause between poll cycles.
Unavailable metadata is represented explicitly. Runtime/build/quantization
values are shown only when reported; disk size is never used as resident memory.
Unattributed/cloud activity is not silently attached to a local model.

### Measurement and alert requirements

| Measurement | Meaning / exclusion |
|---|---|
| Routing latency | Time until the routing body returns a provider result; not full response latency or TTFT |
| Measured TTFT | Routing-body entry to first complete nonempty content/reasoning/tool-argument SSE frame; includes classification/retries, excludes outer role selection and client-side timing |
| Generation TPS | `(reported completion_tokens - 1) / observed first-to-last output interval`; requires >1 token and distinct output times; buffering/hidden tokens/backpressure limit precision |
| Usage | Provider-reported prompt/completion counts; missing/conflicting counts stay unknown |
| Completion validity | `[DONE]` plus clean EOF; errors, cancellation, malformed/oversized frames, dropped/incomplete streams do not become successful measurements |
| Live TPS / RAM / VRAM | Unavailable unless a reliable live source reports them; no inference from elapsed time or benchmark peaks |

The global in-memory limits are 500 route events and 500 completed samples;
per-model detail exposes up to 25 recent samples. Restart clears history and
alert latches. High traffic can evict a comparison window early.

Alerts compare two disjoint windows from the same local provider/model and
measurement source. They are **indicative live trends**, not statistically
controlled benchmark regressions or causal thermal-throttling diagnoses.

| Policy field | Default | Allowed range |
|---|---|---|
| `window_seconds` | 300 | 30..3600 |
| `min_samples` per window/metric | 5 | 3..100 |
| `generation_drop_percent` | 20 | 5..95 |
| `ttft_rise_percent` | 30 | 5..500 |
| `rate_increase_points` | 20 | 1..100 percentage points |
| `repeated_count` | 3 | 2..100 |
| `debounce_samples`, `recovery_samples` | 2 each | 1..10 |
| `baseline_max_age_days` | 30 | 1..3650 |

Performance comparisons use medians; route-error/fallback warnings use rates and
minimum occurrence counts. OOM/timeout labels require corresponding recorded
error text. New evidence is required to advance warning/recovery debounce;
repeated polls and insufficient data are not recovery. Late/out-of-order
completed event IDs remain deduplicated. Router errors are not an exhaustive
count of stream failures. Policy/mapping changes reset latches.

### Explicit reference and settings lifecycle

Operators preview a succeeded run with a completion timestamp and positive
generation/TTFT measurement, then select it for a known exact local model key
with the expected experiment hash and a nonempty rationale. Family names are not
matched automatically. Configuration/artifact/runtime/hardware/workload
provenance is retained independently of live observations.

Reference status is `not_selected`, `unavailable`, `stale`, `mismatched`, or
`not_comparable`. Even a valid reference remains not comparable until per-sample
artifact/build/hardware/context/workload/concurrency identity and metric
equivalence can be established. Benchmark failure affects reference lookup, not
the live page or core routing; mappings can be cleared while that database is down.

At most 32 baseline mappings are stored, alongside policy, in the config-adjacent
observability JSON file. Writes require the latest revision, 128 KiB maximum
body/settings size, and a five-second body deadline. Stale revision/hash
confirmation returns 409; validation, read, and write errors are visible.
Unreadable settings disable saving until repair/restart, not daemon startup.
No request history, automated remediation, scheduling, or outbound notifications
are persisted or performed by this subsystem.

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
| `!br auto` / `local` / `cloud` | Set routing mode |
| `!br <model-name>` | Set specific llama-swap model (names containing `-` or `.`) |
| `!br model <name>` | Set model (legacy form) |
| `!br list` | List available models |
| `!br review` | Show current review mode |
| `!br review auto\|local\|cloud` | Set review mode |
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
| `GET` | `/dashboard` | Embedded HTML dashboard: live routing feed grouped **per conversation** (client session header, or a stable hash of the conversation prefix when no header is present — each conversation renders as one card), hop-chain rendering for multi-hop fallbacks, in-flight request tracker (with cancel), a four-column Sankey flow diagram — **HARNESS** (Pi / OMP / Opencode / Claude / Droid, from User-Agent) → **SESSION** (OMP session title) → **ROUTING** (auto / cloud / local / fallback chain) → **MODEL** (resolved llama-swap key) — where clicking any node highlights its start-to-end path and clicking a card highlights the matching Sankey path, version display, one-click upgrades, review sessions |
| `GET` | `/benchmarks` | Imported benchmark explorer, inspector and import/planning workflows; 503 when storage initialization failed |
| `GET` | `/models` | Read-only local model activity, measurements, rolling alerts and reference controls |
| `GET` | *(favicon/logo assets)* | Static assets for dashboard UI |

### Dashboard API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/routing-events` | JSON polling feed, not an SSE event subscription |
| `GET` | `/api/routing-stats` | Aggregate routing statistics |
| `GET` | `/api/inference-status` | Current inference state, model, elapsed time, token progress. `n_tokens` = total tokens decoded on the llama-server slot (monotonic within a request) — the progress bar is `n_tokens / max_tokens`. Slot data comes from the llama-server `/slots` endpoint (served as a bare JSON array on this build; both shapes are tolerated), with the live per-request count read from `next_token.n_decoded` (object or single-element array) |
| `GET` | `/api/service-health` | Service reachability probes and fallback overview, not an exhaustive stream/circuit history |
| `GET` | `/api/versions` | Version information for brainrouter and dependencies |
| `GET` | `/api/review-config` | Current review configuration (mode, model) |
| `POST` | `/api/review-config` | Update review configuration |
| `GET/POST` | `/api/routing-profile` | Read/replace `{preset,main,reviewer,subagent_model}`; strict role/preset validation, 16 KiB JSON body limit |
| `GET` | `/api/routing-models` | Authenticated local/cloud metadata catalogs, visible per-provider errors; disabled cloud skips discovery |
| `GET` | `/api/models/llama-swap` | List models available in llama-swap |
| `GET` | `/api/bridge-status` | Bridge transport status (Discord/Signal) |
| `GET` | `/api/bonsai` | Bonsai classifier server state (`enabled`, `healthy`, `url`) |

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
| `POST` | `/api/bonsai/toggle` | Stop/start the Bonsai classifier llama-server to free or reclaim VRAM; while stopped, `auto` routing defaults to Local. Toggling Bonsai **off also turns prompt rewrite off** (server-side coupling — the invariant "rewrite on ⇒ Bonsai on" is enforced in the daemon, not just the UI) |
| `POST` | `/api/models/flush` | Unload all models from llama-swap memory (frees VRAM) without restarting the service |

### Benchmark API

All POST routes require the loopback/CSRF gate. Bodies and result limits apply to
previews as well as writes. Responses are JSON unless downloading examples,
CSV/JSONL, or HTML.

| Method | Path | Contract |
|---|---|---|
| `GET` | `/api/benchmarks/runs` | Paginated summaries and counts; filter/order rules above |
| `GET` | `/api/benchmarks/runs/:id` | `{run,configuration,run_record,performance_metrics,speculative_metrics,quality_results,telemetry_samples}` |
| `GET` | `/api/benchmarks/filters` | Sorted observed status/family/backend/workload/quant/speculator values |
| `GET` | `/api/benchmarks/export?format=csv` | Full filtered summaries within budgets; `jsonl` is also supported and is the default |
| `GET` | `/api/benchmarks/examples/:name` | `bundle.json`, `template.json`, `matrix.yaml`, `matrix.json`, `llama-bench.json`; labelled synthetic downloads |
| `POST` | `/api/benchmarks/prepare` | Template + run fields + optional selected experiment/llama output -> `{valid:true,persisted:false,bundle,warnings}` |
| `POST` | `/api/benchmarks/validate` | Complete bundle -> transactional rollback preview |
| `POST` | `/api/benchmarks/validate/llama-bench` | `{bundle,llama_bench}` -> adapter + rollback preview |
| `POST` | `/api/benchmarks/ingest` | Complete bundle -> 201 `{run_id}` |
| `POST` | `/api/benchmarks/ingest/llama-bench` | `{bundle,llama_bench}` -> adapter + committed ingestion |
| `POST` | `/api/benchmarks/plan` | Matrix JSON, or YAML with YAML content type -> `{name,experiments,exclusions,repetitions,run_count,randomize_order,capture_telemetry}` |

### Observability API

| Method | Path | Contract |
|---|---|---|
| `GET` | `/api/observability/models` | Cached local models, active/recent data, policy/retention/measurement definitions and visible backend/attribution errors |
| `GET` | `/api/observability/settings` | `{settings:{policy,baselines},revision,read_error,write_error,path}` |
| `POST` | `/api/observability/settings` | `{revision,policy}` using the complete validated policy |
| `GET` | `/api/observability/reference?run_id=...` | Validated successful reference projection; errors are not empty references |
| `GET` | `/api/observability/baseline?model_key=...` | Selected-reference/comparability status, including lookup failure reason |
| `POST` | `/api/observability/baseline` | `{revision,model_key,run_id,expected_experiment_hash,note}`; null run ID clears without a benchmark read |

### In-flight Tracking API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/inflight` | Active in-flight requests: `{requests:[{id, elapsed_ms, method_path, model, user_agent, peer_addr, conv_id, session_id, bytes_received, activity, pp_progress}]}`. The panel is hidden in the dashboard when this is empty |
| `POST` | `/api/inflight/cancel` | Cancel one in-flight request. Body: `{id}`. Returns `{"cancelled":true}` on success, HTTP 404 for an unknown id. Loopback-only + CSRF-protected (in the `is_destructive` gate list) |
| `GET` | `/api/omp-sessions` | OMP session titles for the Sankey "SESSION" column: `{home, sessions:[{slug, title, updated_ms}]}` — each session is a directory under `~/.omp/agent/sessions/<slug>` whose first JSONL line carries `{"type":"title",...}` |

### Review Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/review/` | Redirect to `/dashboard` |
| `GET` | `/review/session/:id` | View a specific review session (HTML) |
| `POST` | `/review/session/:id/resolve` | Resolve a review session (HTML form) |
| `GET` | `/review/api/sessions` | List all review sessions (JSON) |
| `GET` | `/review/api/sessions/:id` | Get review session details (JSON) |
| `POST` | `/review/api/request` | Start a new review (JSON) |
| `POST` | `/review/api/request-async` | Start and return `{sessionId,status}` immediately; does execute the configured reviewer asynchronously |
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
| Manifest stream stalls after output begins | `TimeoutStream` (180s per chunk) | Error surfaces in SSE; no output replay or successful completed measurement |
| Manifest circuit open | Health tracker (3 failures) | Skip Manifest, route directly to llama-swap |
| llama-swap returns error / 404 | HTTP status code | Error returned to caller; next request may load a different model |
| llama-swap circuit open | Health tracker (3 failures) | Error: no backend available. 60s cooldown before retry |
| Both circuits open | Health tracker | Error returned to caller; both providers on 60s cooldown |
| Enabled Bonsai classification fails | HTTP/parse error or timeout | Auto roles choose the cloud path, still respecting Manifest's enabled flag and local fallback policy |
| Bonsai server OOM / crash | Process exits; next health probe fails | Default to `Cloud` until the server is started again from the dashboard |
| Bonsai server stopped from dashboard | Shared `enabled` flag cleared before kill | Auto roles skip classification and choose local; pinned roles remain pinned |
| Daemon not running when MCP connects | UDS connect error | `brainrouter mcp` exits with a clear error message |
| Review LLM returns unparseable JSON | JSON parse failure in review_loop | Retry within iteration; count as iteration attempt |
| Review hits max iterations | Iteration counter | Escalate to human via dashboard UI |
| Peer CWD resolution fails | `/proc` read failure | Use a valid explicit review CWD; invalid/absent resolved review directories are rejected, not silently substituted |
| Config validation failure at startup | Missing required fields, invalid URLs, missing GGUF | Daemon refuses to start with descriptive error |
| Legacy auto reviewer retains an ignored model | YAML/legacy-state read normalization | Discard only that ignored field, warn with source/action, persist state migration once; keep new API writes strict |
| Invalid routing migration | Validation/read/write failure | Report source/destination and cause; do not silently change local/cloud semantics |
| Corrupt/unwritable/future benchmark database | Store initialization | Log cause; benchmark routes 503; core routing/listeners/Bonsai continue |
| Benchmark workers saturated or SQLite busy | Shared admission/lock timeout | 503 and Retry-After; no unbounded job queue |
| Benchmark result/export exceeds budget | Row/count/serialization checks | Explicit 413; narrow request or read offline, never silent truncation |
| Missing measurements or insufficient retained history | Source validation and sample minima | Unknown/insufficient-data state, no fabricated metric or recovery |
| Observability settings/reference unavailable | File read/CAS/reference validation | Visible error or reference status; core routing/live page remain available |
| Bridge OMP subprocess timeout | Configurable timeout (default 600s) | Kill subprocess, return timeout error to chat |
| Signal CLI unavailable | Subprocess spawn failure | Signal transport disabled; logged |
| Discord token invalid | Serenity connection failure | Discord transport disabled; logged |

---

## Security Model

### Localhost-only access

Main, Reviewer and Subagent are model-selection roles, not authorization roles.

Brainrouter binds to `127.0.0.1:9099` by default. Its Unix socket defaults to
`$XDG_RUNTIME_DIR/brainrouter.sock`, falling back to `/run/brainrouter.sock`.
There is no general proxy/read-API authentication layer. Do not expose a
non-loopback listener without a separate trusted access-control layer.
Protected mutations include routing profiles, `/review/api/*`, benchmark and
observability POSTs, and the listed management operations. They require a
loopback peer or the local Unix socket.

### CSRF protection

The protected gate checks Origin first, otherwise Referer. HTTP localhost,
IPv4 loopback, and IPv6 loopback hosts are accepted on any port to support local
tunnel/container port remapping. `null`, non-loopback, malformed, and
credential-bearing origins are rejected. Headerless local CLI clients are
allowed. The peer restriction still applies; a public HTTPS/Tailscale hostname
is not implicitly trusted by this loopback policy. This is not a configurable
general reverse-proxy origin allowlist.

The legacy `/review/session/:id/resolve` form route is not in the
`/review/api/*` prefix gate. Keep the listener local and use the protected JSON
review APIs for automation; this gate is not general authentication for every route.

Imported commands/paths/URLs are never executed or fetched. Browser rendering
uses inert text, CSV exports neutralize spreadsheet formula prefixes, and
benchmark/observability bodies and workloads are explicitly bounded. Raw imported
environment/result metadata can still be sensitive; operators must redact it
before importing or sharing exports.

### Path sanitization

Working directory tracking (for bridge sessions and review context gathering) enforces:

- Absolute path requirement
- Null-byte rejection
- Path-traversal component blocking (no `..` escapes in session-tracked paths)

### Startup validation

Required configuration is validated at daemon startup:

- `manifest.base_url` must start with `http://` or `https://`
- `bonsai.model_path` must exist on disk — **only when `bonsai.enabled: true`**; a missing file with Bonsai off does not crash startup (Bonsai can be enabled later at runtime once the model is in place)
- `llama_swap.fallback_model` is required

### No secrets in the repo (multi-user install)

The Manifest API key (`mnfst_*`) is never committed: it lives in `/etc/brainrouter/env` (shared) or each user's `~/.config/brainrouter/.env`, and configs reference it by **variable name** via `manifest.api_key_env`. Per-user repair scripts in `deploy/*.sh` are machine-local and **gitignored** — they never embed the key; they source it from an existing `.env` on the machine or prompt the admin for it at deploy time.

### Shared credentials (multi-user install)

`/etc/brainrouter/env` is owned `root:aistack` with mode `640`. Only the root user and members of
the `aistack` group can read it. `install.sh` adds every human system user to `aistack` automatically.
Individual users never have write access to this file -- only root can update the API key.
The file is not world-readable; a user not in `aistack` cannot extract the Manifest API key.

Required core configuration/routing-state failures remain explicit startup
errors. Optional benchmark storage and observability-settings failures are
isolated as described above; legacy auto-mode compatibility is a read-only
normalization, not weakened validation for new writes.

---

## What This Is Not

- **Not a multi-provider cloud backend.** Brainrouter implements OpenAI-compatible proxying and an Anthropic wire shim; Manifest handles downstream cloud providers and pricing.
- **Not a model runner.** brainrouter does not serve chat models -- it only spawns the single classifier llama-server for Bonsai. llama-swap handles local model serving.
- **Not an auth layer.** Loopback is the default trust boundary, not user authentication. Credentials live in provider configuration/environment.
- **Not a conversation store.** Chat history is managed by the harness. brainrouter is stateless for proxy calls; review sessions are in-memory and lost on daemon restart.
- **Not a chat application.** The Discord/Signal bridges are thin wrappers around the `omp` CLI, not standalone chat bots with their own reasoning.
- **Not a benchmark executor or runtime attestation service.** Registered capabilities/hashes and imported metrics do not prove artifact integrity, runtime feature support, or controlled comparability.
- **Not a persistent live-traffic monitor.** Samples and alert latches reset on restart; only role preferences, benchmark records and explicit observation settings persist.

---

## Validation

The implementation baseline was verified with 192 Rust tests and 31 Node
browser-logic tests using fixtures/stub providers, without real inference.
Coverage includes both wire protocols, event-ID/terminal-frame measurements,
hybrid roles and continuation snapshots, legacy migrations/strict write APIs,
actual daemon availability with invalid benchmark storage, bounded database/
response backpressure, synthetic import-to-reference flows, and alert
threshold/dedup/recovery behavior. HTML script syntax and scoped Rust formatting
were checked; a debug binary built successfully. This is not a performance
benchmark, a real-browser/accessibility certification, or proof of deployment
on a particular host.

```bash
cargo test --locked
cargo clippy --locked --all-targets
node --test scripts/test-benchmark-ui.cjs
bash scripts/check-html-js.sh
cargo build --locked --bin brainrouter
```

Existing unrelated Clippy warnings remain; strict CI must distinguish baseline
warnings from new diagnostics. Transferring source/Git commits does not rebuild,
install, or restart a running binary. Plan any runtime rollout separately.

## Planned and Unshipped Work

These are ideas requiring separate approval, not acceptance claims for this release.

- Bonsai as a Strategic Context Expert -- real-time synthesis of cloud agent output
- Interrupt-and-redirect: user types a correction mid-stream; brainrouter cancels and re-prompts
- Persistent review sessions (SQLite) so they survive daemon restarts
- Persistent request history, per-request token costs and budget enforcement
- Per-project profiles, optional strict cloud fallback consent, reviewer diversity rules, and escalation ladders
- **Quant Lab:** actual llama-perplexity/PPL/KLD execution, compatible same-family reference checking, queued/sequential runs with cancellation/progress, per-token distributions and GGUF parsing
- **Model Compare:** complete common-suite execution across unrelated families, token-space compatibility checks, and blind human evaluation
- Automated benchmark scheduling, controlled regression campaigns, outbound notifications, or automatic model switching
- Parquet export and an integrated analytical warehouse
- Runtime-loaded component attestation (PLE/n-gram tables, adapters, caches, retrieval assets), beyond declared registry provenance
- Runtime context-selection controls: the retained `cli context` command has no implemented `/api/context` server route

The notes motivating Quant Lab/Model Compare are not fully implemented by the
current explorer. Ember/Flash environment, cache-building, or provisioning work
is not a shipped Brainrouter feature. Existing tables, chart labels, and imported
measurements must not be presented as an execution pipeline or attestation.
