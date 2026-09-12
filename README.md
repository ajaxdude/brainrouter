# brainrouter

<p align="center">
  <img src="assets/brainrouter-logo.svg" alt="brainrouter logo" width="600">
</p>

A Rust proxy and local control plane between coding harnesses and LLMs. Choose independent Main, Reviewer, and local Subagent roles, inspect current model activity and completed-stream measurements, and explore imported benchmarks without running benchmark jobs. Bonsai classification and Manifest cloud access are opt-in; the defaults stay local. Core management and reviews have a headless CLI, while the benchmark and model-observability pages also expose HTTP APIs.

```
coding harness -> brainrouter :9099 (OpenAI or Anthropic wire format)
                      |
                      +-- auto/default -> Main choice (local by default)
                      +-- local/cloud  -> explicit backend default
                      +-- model ID     -> exact local, or cloud/<id>
                      +-- subs         -> separate local subagent pool
                      +-- review       -> snapshotted Reviewer choice
                      |
                      +-- local -> llama-swap
                      +-- cloud -> Manifest when enabled -> local fallback
                      +-- Auto role -> Bonsai when enabled, otherwise local

/dashboard  -> routing/profile/review/operations controls
/models     -> read-only activity, measured completions, indicative alerts
/benchmarks -> stored results, import previews, plan previews, inspector
```

- **One endpoint, all harnesses.** OpenAI-compatible on `POST /v1/chat/completions`. Anthropic-compatible on `POST /v1/messages`. Every harness connects to the same `:9099`.
- **Independent routing roles.** Choose Main, Reviewer (explicit local or cloud ID), and a separate local Subagent pool, or apply a named preset. Legacy `auto`, `local`, `cloud`, and `subs` aliases remain supported. Cloud and Bonsai stay off by default.
- **Off by default, opt in.** Fresh installs run fully local with a single hop — no Bonsai model download, no Manifest stack, no cloud API key required.

- **Optional local prompt rewriting.** Managed local routes can use a lean system prompt. Rewriting is off at startup, requires a running healthy Bonsai to enable through the control API, and never rewrites explicit local model selections.
- **Subs pool routing.** `subs` / `brainrouter/subs` selects the separate local pool (initially `llama_swap.subs_model`). The pool is independent of Main and Reviewer; clearing it restores classifier/auto behavior, not the Main override.
- **Manifest cloud failover.** When Manifest is enabled, `cloud` traffic goes through it (runs locally in Docker, picks the cloud provider) and falls back to llama-swap's `fallback_model` on failure.
- **MCP code review.** `mcp_brainrouter_request_review` triggers an iterative review loop (up to 5 rounds by default). The review LLM reads your PRD, git diff, and task summary, then either approves or gives actionable feedback.
- **Dashboard.** Live routing feed, review session list, version display, one-click upgrades and service restarts — all at `http://127.0.0.1:9099`.
- **Benchmark explorer.** Import reproducible model/runtime/hardware results, compare throughput, memory, context scaling, quantizations, runtimes, speculation, and quality, inspect configurations, and export filtered CSV or JSONL at `http://127.0.0.1:9099/benchmarks`.
- **Headless CLI.** `brainrouter cli` covers status, operations, routing profiles, and reviews. Use HTTP for benchmark and observability workflows without a dedicated CLI command. See [Headless CLI](#headless-cli-brainrouter-cli).
- **Model observability.** `/models` shows exact local model activity, available metadata, uniquely correlated completed measurements, indicative rolling warnings, and explicit benchmark references. Missing measurements and unverified comparability are visible, not fabricated.
- **VRAM control.** The dashboard or CLI can stop/start the Bonsai classifier and flush every model loaded in llama-swap — reclaim GPU memory without a reboot or a terminal.
- **In-flight request tracker.** The dashboard shows every active request as it runs — elapsed, model, user agent, address, session, bytes received, PP progress, and a live activity label (tool calling / reasoning / asking a multiple choice question / generating) — with a per-row Cancel button. No need to open the llama-swap UI.
- **Per-request throughput.** Conversation cards surface generation-tokens/s (tg) estimates from uniquely correlated completed streams. Missing measurements remain hidden; prompt-processing throughput is not inferred from first-output latency.
- **Four-column Sankey.** The flow diagram is HARNESS (Pi / OMP / Opencode / Claude / Droid) → SESSION (OMP session title) → ROUTING (auto / cloud / local / fallback chain) → MODEL (resolved key). Click any node to highlight its start-to-end path; click a conversation card to highlight the matching Sankey path.

---

## Table of contents

1. [Install (one script)](#install)
2. [Configure](#configure)
3. [Connect your harness](#connect-your-harness)
4. [Dashboard guide](#dashboard-guide)
5. [Model observability and regression alerts](#model-observability-and-regression-alerts)
6. [Benchmark explorer](#benchmark-explorer)
7. [Headless CLI](#headless-cli-brainrouter-cli)
8. [MCP code review guide](#mcp-code-review-guide)
9. [Bridge: Discord and Signal](#bridge-discord-and-signal)
10. [Reference](#reference)
11. [Planned, not shipped](#planned-not-shipped)

This documents the implemented source through `3691dcc` (2026-09-07).
Source transfer or a Git update is not a binary rollout: building, installing,
and restarting a running service are separate operations. See [PRD.md](PRD.md)
for requirements and the explicit boundary between shipped and future work.

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
- **Bonsai provisioning** — the script's prompted model-download step currently uses Bonsai-8B Q4_K_M under the shared models directory; classifier activation remains a separate runtime opt-in
- **Manifest** — cloud LLM router running as a system Docker service on port 3001
- **llama-swap** — local model runner as a system Docker service on port 8081
- **brainrouter** — compiled and installed to `/usr/local/bin/brainrouter`
- **llama-server-toolbox** — wrapper at `/usr/local/bin/llama-server-toolbox`
- **toolbox container** `llama-vulkan-radv` — AMD RADV Vulkan environment
- **Shared config** — `/etc/brainrouter/brainrouter.yaml` and `/etc/brainrouter/env`
- **Per-user systemd services** — brainrouter enabled for every user, auto-starts at boot via `loginctl linger`
- **Shell environment** — `/etc/profile.d/ai-stack.sh` sets PATH and harness env vars for all users

### Optional cloud setup

For cloud use, create the Manifest API key in its browser wizard and enable
`manifest.enabled` in the selected Brainrouter YAML. A cloud profile does not
enable the backend. Local-only operation needs no cloud key.

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

Verify backend URLs and set `fallback_model` to an existing llama-swap key.
The daemon reads the per-user config path by default; use `serve --config` when
your service uses the shared file. Keep each user's saved role preferences in
mind: those override YAML role defaults after restart.

```bash
sudo nano /etc/brainrouter/brainrouter.yaml
sudo nano /opt/ai/llama-swap/config.yaml  # define the model
sudo systemctl restart llama-swap
```

```yaml
# /etc/brainrouter/brainrouter.yaml (shared for all users)

manifest:
  # Cloud routing is OFF by default. Set enabled: true to route cloud traffic
  # to your Manifest instance.
  enabled: false
  base_url: "http://localhost:3001/v1"
  api_key_env: MANIFEST_API_KEY  # name of the env var, key lives in /etc/brainrouter/env

llama_swap:
  base_url: "http://localhost:8081/v1"
  fallback_model: "your-local-model"  # must match a key in /opt/ai/llama-swap/config.yaml
  # Optional: thinking-budget nudge for local routes (off by default)
  # nudge:
  #   enabled: true
  #   budgets:
  #     light: 10240
  #     deep: 12288

bonsai:
  # Classifier is OFF by default. Set enabled: true and point model_path at a
  # GGUF to run the external classifier server; auto-routing then hops
  # Bonsai → cloud/local. While off, auto goes straight to local.
  enabled: false
  model_path: "/mnt/models/prism/Bonsai-27B-dspark-bf16.gguf"  # GGUF model for classification
  server_port: 9200  # port for the external llama-server process
  fork_path: "/home/papa/.local/share/brainrouter/llama-prism/llama-server"  # PrismML fork binary
```

The service can load the Manifest key from `/etc/brainrouter/env` (readable by
the configured `aistack` group). A standalone `serve` process reads the named
environment variable; it does not automatically parse `.env` files:

```bash
sudo nano /etc/brainrouter/env
# MANIFEST_API_KEY=mnfst_your_key_here
```

After a YAML or service-environment change, restart brainrouter for your user:
```bash
systemctl --user restart brainrouter
```

Role/profile changes through the dashboard or CLI take effect without restart
for new requests/reviews. Existing reviews retain their initial Reviewer.
Benchmark database-path changes and repair of unavailable storage/settings need
a restart; see the persistence and availability sections below.

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

These examples require a harness mode that sends OpenAI chat-completion or
Anthropic Messages requests. Brainrouter does not serve the OpenAI Responses
API (`/responses` or `/v1/responses`). Check the wire protocol/model ID when
using a different harness version or an older installer-generated template.
The proxy returns streaming SSE; it is not a full implementation of non-streaming
provider APIs.

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

MCP registration in `~/.omp/agent/mcp.json`:

```json
{
  "mcpServers": {
    "brainrouter": {
      "type": "stdio",
      "command": "/home/yourname/ai/projects/brainrouter/target/release/brainrouter",
      "args": ["mcp", "--socket", "/run/user/1000/brainrouter.sock"],
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
  "args": ["mcp", "--socket", "/run/user/1000/brainrouter.sock"]
}' --scope user

# Route Claude Code through brainrouter (add to ~/.zshrc):
export ANTHROPIC_BASE_URL=http://127.0.0.1:9099
export ANTHROPIC_AUTH_TOKEN=not-used
```

#### vibe

```toml
# Merge into ~/.vibe/config.toml
# Keep root settings before TOML table declarations.
mcp_servers = [
  { name = "brainrouter", command = "/path/to/brainrouter", args = ["mcp", "--socket", "/run/user/1000/brainrouter.sock"] },
]

[[providers]]
name = "brainrouter"
api_base = "http://127.0.0.1:9099/v1"
api_style = "openai"
backend = "generic"

[[models]]
name = "auto"
provider = "brainrouter"
alias = "auto"
```

#### opencode

Merge into `~/.config/opencode/config.json`:

```json
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
args = ["mcp", "--socket", "/run/user/1000/brainrouter.sock"]
```

#### droid (factory.ai)

MCP registration in `~/.factory/mcp.json`:

```json
{
  "mcpServers": {
    "brainrouter": {
      "type": "stdio",
      "command": "/path/to/brainrouter",
      "args": ["mcp", "--socket", "/run/user/1000/brainrouter.sock"]
    }
  }
}
```

> **Note:** `provider: "anthropic"` is required for droid. Droid's `openai` mode posts to `/responses` (not served here). `anthropic` mode posts to `/v1/messages`, which brainrouter handles.

Configure the custom model separately in Droid's provider settings, using its
Anthropic-compatible mode, Brainrouter's localhost endpoint, and model ID
`auto` to follow Main (or an explicit selector). Provider settings are not MCP
server registration and their exact file/schema depends on the harness version.

The examples use UID `1000`; replace it with your user's numeric UID or omit
`--socket` to use `$XDG_RUNTIME_DIR/brainrouter.sock`. JSON/TOML launcher argument
strings do not expand shell expressions such as `$(id -u)`. Use `model: auto`
to follow the Main profile, not a made-up alias such as `brainrouter-auto`.
Older templates may use that name for display; it is not a reserved server-side
routing selector if sent as the actual model ID.



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
harness -> Main profile or explicit selection -> optional classifier -> backend
```

Each stage shows:
- **Bonsai decision** — `cloud` or `local` badge
- **Provider** — which upstream handled it
- **Model** — the model key that was used
- **Routing latency** — time until the router obtains its result, not complete-stream latency or measured TTFT
- **Fallback indicator** ↩ — appears when Manifest failed and llama-swap handled it instead

### Routing events feed

The feed groups the bounded 500-event buffer into conversations using client
session headers or a stable prompt-prefix fingerprint, with turn and fallback
details. Review events carry their review session ID. The displayed prompt is an
excerpt, not a complete conversation archive; restart clears this history.

### In-flight request tracker

While a request is running, the dashboard shows it live — no need to open the llama-swap UI. The panel lists every active request with:

- **Cancel** — a per-row button that aborts the in-flight request (the row is removed and the response stream ends)
- **Elapsed** — how long the request has been running (updates every second)
- **Model** — the resolved llama-swap model (updated after routing resolves; the requested model until then)
- **Request** — the endpoint path (`POST /v1/chat/completions` or `POST /v1/messages`)
- **Activity** — a live label derived from the response bytes: `tool calling` / `reasoning` / `asking a multiple choice question` / `generating`
- **PP** — prefill progress (0–100%) when the llama-server build exposes slot progress
- **Address / User-Agent / Session ID / Bytes received** — peer address, client UA, OMP session (or `#conv` hash), and bytes streamed so far

The panel is hidden while no requests are in flight. Card progress bars show the same PP value; once a card settles it shows "Done for now".

### Throughput chips on cards

Each conversation card shows averaged **tg** (generation tokens/s) estimates from the OpenAI SSE `usage` count and observed output-frame interval of each successfully completed request. Builds without valid usage counts, or whose output is observed in a single frame, leave the chip hidden. **pp** is not inferred: time to first output includes more than prompt processing. See [Model observability and regression alerts](#model-observability-and-regression-alerts) for measurement definitions and limits.

### Version header and upgrades

The header row shows current installed versions of:

- **llama-swap** — the local model router binary
- **llama.cpp** — the llama-server build inside the toolbox container
- **Manifest** — the running Docker container (image date · short hash)
- **toolbox** — the OCI image version label
- **bonsai** — classifier server state: `on`, `off`, or `down`

Click the **bonsai** row to stop or start the classifier. Auto roles choose local
while it is disabled; roles explicitly pinned to a backend remain pinned.
Stopping Bonsai also turns prompt rewriting off.

Version metadata is cached and refreshed in the background (normally every
30 minutes), not re-fetched from upstream for each dashboard poll. Upgrade
controls are explicit operations; they are not triggered by opening model or
benchmark pages.

### Service controls (nav bar)

The sidebar includes four restart controls and a model-flush action:

| Button | What it does |
|---|---|
| **Restart llama-swap** | `systemctl --user restart llama-swap` |
| **Restart llama.cpp** | Refreshes the toolbox container (runs configured restart script) |
| **Restart Manifest** | `docker compose restart manifest` |
| **Restart brainrouter** | `systemctl --user restart brainrouter` — page reloads after 3 s |
| **⏏ Flush Models** | Unloads every model from llama-swap memory (frees VRAM) without restarting; models reload on next request |

### Review sessions

Review sessions remain available through the review API/CLI and linked session pages:

- Each row shows task ID, status (`pending` / `approved` / `needs_revision` / `escalated`), iteration count, reviewer type (LLM or human), and timestamps.
- Open a session link for feedback, status, and requested/actual reviewer information.
- If a session is `escalated` (LLM couldn't resolve it after max iterations), a **Resolve** panel appears — type your feedback and submit to close the loop.

### Independent routing controls

The dashboard exposes Main, Reviewer, and local Subagent choices plus compatible presets:

| Setting | Options | Effect |
|---|---|---|
| **Main** | Auto / Local / Cloud, optional explicit ID | Handles default/auto client requests only |
| **Reviewer** | Auto / Local / Cloud, optional explicit local/cloud ID | Snapshotted for each new review; independent of Main |
| **Subagent pool** | Local ID or unset | Used only by subs aliases; presets do not silently overwrite it |
| **Preset** | Six named combinations or Custom | Sets compatible Main/Reviewer roles; cloud activation remains a separate opt-in |

Preferences persist across restart. Existing reviews and their continuations
retain the original reviewer; see [Routing controls](#routing-controls) for
precedence, migration, commands, and exact fallback behavior.

## Model observability and regression alerts

Open **`http://127.0.0.1:9099/models`** or select **Model activity** in the dashboard. An exact local model key connects read-only backend status, active registry requests, completed stream measurements, recent routing results, and an explicitly selected historical benchmark reference. `/models?model=<URL-encoded-key>` links to a model. `/running` and `/v1/models` are polled with two-second timeouts and 1 MiB response limits; `/props` is queried only for up to four already-ready proxies. Polls run independently of routing, with a five-second pause between polls. Before the first poll completes, the API reports an awaiting-observation state; an empty initial list is not proof that no models exist. Missing quantization, runtime identity, RAM/VRAM, live token rates, or prefill progress remain unavailable, never inferred from model names, file sizes, elapsed time, or benchmark peaks.

Performance measurements are keyed by the unique routing event ID, not the conversation. Only `[DONE]` followed by clean EOF records a completed sample; stream errors, cancellation, oversized/malformed frames, and incomplete streams are excluded. **Measured TTFT** is router-observed time from routing-body entry, including classification/retries but excluding outer profile selection, to the first complete SSE frame containing nonempty content, reasoning, or tool-call arguments. Role-only frames and heartbeats do not count. It is not client end-to-end latency. **Generation TPS** estimates `(provider completion_tokens - 1) / (last output frame time - first output frame time)`, requiring more than one reported token and distinct output times; usage/DONE tail time is excluded. Batched deltas, hidden reasoning tokens, buffering and backpressure limit precision. Missing or conflicting usage stays unknown; prompt-processing TPS is not fabricated. Route latency still means time until the router returns its result, not TTFT or completion latency.

The Anthropic adapter keeps content streaming while consuming final usage and confirming EOF before its terminal events. Waiting after the first finish marker is limited to two seconds; ready data/EOF wins over the timer so client backpressure does not discard a completed response. Continuously ready tails are separately capped at 256 KiB total, 64 KiB per line, and 1,024 subsequent chunks. Tail errors, timeout, or budget exhaustion do not produce completed measurements.

The default policy compares medians in the latest **five-minute window** with the preceding, disjoint five minutes. Each metric needs **five samples in each window**. Indicative warnings require a **20% generation drop**, a **30% TTFT rise**, or a **20-percentage-point error/fallback rate increase with at least three occurrences**. Two evaluations with new sample evidence activate a warning; two with new non-regressing evidence recover it. Repeated polls cannot advance debounce, and insufficient data is not recovery. The page exposes all thresholds and permits bounded operator changes. OOM/timeout categories require explicit recorded error text; fallback stages mean fallback-served routes, not that the fallback model caused a failure. Unattributed/cloud routing errors are shown separately. Router success does not guarantee stream success, so router-error counts are not an exhaustive stream-error rate.

**These are indicative live trends, not controlled benchmark regressions.** Selecting a successful reference preserves its full artifact/runtime/hardware/workload identity and requires a mapping rationale plus confirmation of the inspected experiment hash. No model-family matching is performed. References older than the policy limit (30 days by default), changed references, and unavailable benchmark storage are surfaced explicitly. Even a valid reference is **not comparable** until per-sample artifact SHA, build, hardware, context/workload, concurrency and equivalent metric definitions can be established. Benchmark reference failure does not disable live observations or routing.

Retention is bounded to **500 routing events and 500 completed stream samples globally**, in memory only. High traffic can evict a reference window early. Restart clears samples and alert state; policy or mapping changes reset debounce. Only explicit mappings and alert policy persist in the config-adjacent `*.observability.json` file (for example `brainrouter.observability.json`). Writes use a size-bounded atomic replacement and revision checks. Errors remain visible; unreadable/corrupt settings disable saves until repaired and the daemon restarted, but do not stop routing. There is no traffic-history database, automatic benchmarking, model switching, scheduling, or outbound notification.

Read APIs are `GET /api/observability/models`, `/settings`, `/reference?run_id=...`, and `/baseline?model_key=...` under `/api/observability`. `POST /api/observability/settings` accepts `{ "revision": 0, "policy": { ... } }`; `POST /api/observability/baseline` accepts `{ "revision": 0, "model_key": "...", "run_id": "...", "expected_experiment_hash": "...", "note": "..." }`. Set `run_id` to `null` to clear a mapping, including while benchmark storage is unavailable. Writes use the existing localhost/CSRF guard, 128 KiB body limit, five-second body deadline, and conflict responses for stale revisions.

Reference selection requires a known exact local key, a succeeded run with a
non-future completion timestamp, and a positive generation-TPS or TTFT
measurement; a metadata-only run is not a performance baseline. Up to 32 explicit
mappings are supported. Backend catalogs/running lists are limited to 256 entries.
The full policy shape and allowed ranges are documented in
[the PRD](PRD.md#model-observability-and-regression-alerts).
Revisions are process-local and reset on restart: always read the latest settings
before writing. A settings error may report that rename committed but crash
durability is uncertain; reload state before retrying rather than assuming
nothing changed.

```bash
API=http://127.0.0.1:9099
curl --fail-with-body -sS "$API/api/observability/models"
curl --fail-with-body -sS "$API/api/observability/settings"
# Inspect a real imported run before selecting it in the model page:
curl --fail-with-body -sS --get "$API/api/observability/reference" \
  --data-urlencode "run_id=your-imported-run-id"
```

## Benchmark explorer

Open **`http://127.0.0.1:9099/benchmarks`** or select **Benchmarks** in the dashboard navigation. Brainrouter initializes a normalized SQLite database at `~/.local/share/brainrouter/benchmarks.sqlite3` by default. The explorer only imports and analyzes completed or externally managed benchmark runs; it never starts a model or benchmark process.

Benchmark storage is optional for daemon availability. If initialization fails (for example, a corrupt database, an unwritable path, or a schema newer than this binary supports), core routing, listeners, and Bonsai startup continue normally. The daemon logs the cause, and benchmark pages/data APIs return HTTP **503** with recovery guidance instead of empty results. Repair the database or update `benchmarks.database_path`, then restart Brainrouter to retry; unsupported schemas are not downgraded.

The explorer provides deterministic filtering and pagination, throughput-vs-memory/Pareto visualization, 8K/32K/128K context comparisons, runtime and quantization comparisons, pass@1-per-GiB analysis, and filtered CSV or JSONL downloads. **Charts and extrema cover the current page only (at most 25 runs)**, across its selected statuses, excluding missing measurements. Page/order/filter changes alter the sample. The total matching-run count is separate; charts do not imply full-registry aggregates or controlled comparisons between different workloads.

### Schema and identity

`migrations/0001_benchmark_explorer.sql` creates a dedicated normalized SQLite
STRICT schema and initializes WAL; every connection enables foreign keys.
`schema_migrations` records the supported version/identity, with indexes and
`run_summary` for joined queries. This does not migrate live routing or review
sessions into a database.

| Data | Tables / meaning |
|---|---|
| Registry | `models`, `artifacts`, `runtimes`, `hardware_profiles`, `workloads`: declared model/tokenizer/quant/build/hardware/corpus identities |
| Configuration | `experiment_specs`, `experiments`: canonical registry references plus context, token, batch, thread, optimization and sampling settings |
| Attempts | `runs`: unique ID and `(experiment_id,repetition)`, status, timestamps, command/environment/log pointers and raw result |
| Native jobs | `benchmark_jobs`: queued/running/terminal Riddllr and Plumebench job state, progress, request identity, workspace and resulting run |
| Measurements | One-to-one `performance_metrics`/`speculative_metrics`; one-to-many `quality_results`/`telemetry_samples` |
| Immutability | `entity_fingerprints` rejects changed registry/config payloads under reused IDs |
| Exclusions | Schema reserves an `exclusions` table; current planning returns exclusions without persisting them |

The experiment SHA-256 is computed from the canonical Rust-serialized
configuration excluding its application ID. Referenced registry identities are
immutable too. Checkpoint revisions, artifact/executable/manifest hashes,
compiler/build flags, optimization feature states, and capture timestamps stay
separate. `unsupported`, `disabled`, `enabled`, and `requested_unavailable` are
data states, not proof that a backend implements or executed a feature.
Draft-artifact references must exist.

### Native Benchmark Lab

The opt-in Benchmark Lab runs two integrated suites from `/benchmarks`:

- **Riddllr** discovers matched `prompts/<case>.txt` and
  `solutions/<case>-solution.txt` files, routes the prompt through Brainrouter,
  grades cardinal-assignment or ordered-line answers deterministically, and
  stores `pass@1`, duration, output, route provenance and the declared suite
  manifest.
- **Plumebench** discovers complete `tasks/<case>/` directories. Brainrouter
  takes one immutable private snapshot of the task and grader, copies only
  `starter/` plus the prompt into a per-job workspace, and invokes OMP through
  the `brainrouter/<model>` provider inside a Bubblewrap filesystem sandbox.
  The sandbox receives the starter workspace, a generated credential-free OMP
  profile, the OMP executable, and read-only system files; it cannot see the
  suite root, hidden tests, daemon configuration, user home, or sibling job
  directories. Only after OMP exits does Brainrouter copy `tests_hidden/` from
  the snapshot into a separate grader stage. Pytest and the snapshotted
  `elegance.py` run in a second network-isolated Bubblewrap sandbox, with only
  the grader stage writable. Brainrouter records test counts, pass@1, and raw
  static metrics when the hidden gate passes.

Execution is localhost-only, explicit, and limited to one heavy job. Job
identity includes suite, case, model, repetition, source manifest and relevant
runner settings. Duplicate active identities are rejected; successful run
identities require a new repetition. Cancellation terminates the spawned Unix
process group, captured output is bounded, source symlinks and path escapes are
rejected, and queued/running jobs become `interrupted` after daemon restart.
Source suite directories are never modified, and the configured workspace may
not overlap either suite root. OMP requires a CLI exposing `--model`, `--mode`,
`--max-time`, `--thinking`, `--auto-approve`, `--no-session`, and `--cwd`
plus the `--no-extensions`, `--no-skills`, and `--no-rules` isolation switches
(OMP 18.1 or newer is recommended). Brainrouter enforces the configured turn
limit from OMP `turn_start` events rather than passing an unsupported CLI flag.
The OMP sandbox has a private network namespace. A short-lived proxy exposes
only `/v1/models` and `/v1/chat/completions` through a filtered Unix socket, so
generated code cannot reach Brainrouter's dashboard or administrative APIs.

The lab remains disabled unless `benchmarks.lab.enabled` is true. A missing
suite root disables that suite in the UI without disabling routing or imported
benchmark exploration. A missing OMP/Python executable fails only the affected
job. Completed, timed-out and cancelled executions use the same validated
registry ingestion path as uploads.

### Query and export semantics

`GET /api/benchmarks/runs` returns `{items,page,per_page,total,total_pages}`.
Defaults: `page=1`, `per_page=25`, `sort=started_at`, `order=desc`.
Pages are positive; page size is 1..100; out-of-range pages are empty.
Filters `status`, `family`, `backend`, `workload`, `quant_name`, and
`speculator_type` are exact matches. Blank filters are ignored; repeated
parameters take their last value. `q` is a literal substring search over
family/architecture/quant/runtime/workload/run-ID labels, using SQLite's
ASCII case-insensitive LIKE semantics with `%` and `_` escaped.

Sort by `started_at`, `family`, `workload`, `context_tokens`, `prompt_tps`,
`generation_tps`, or `ttft_ms`, with `asc`/`desc`. Timestamp sorting falls back to
creation time for runs without a start. Missing numeric metrics sort last;
ties use ascending run ID. Unknown parameters/sorts/orders return 400; an
unknown exact filter value just matches nothing. Summary `quality_score` averages
only the metric named `pass@1`, not unrelated quality measures.

**Run inspector.** Open a row with the keyboard or mouse, or link directly to `/benchmarks?run_id=<URL-encoded-ID>`. Separate sections show model, artifact, runtime, hardware, workload, optimization and sampling definitions; commands and provenance; prompt/generation timing, TTFT, inter-token p50/p95/p99, memory, disk and energy; speculative metrics; and per-task compile/pass/test results. Telemetry is ordered and grouped by reported GPU index, with absent/null readings displayed as gaps, not zero or interpolated measurements. Host readings are not summed across GPUs. Raw JSON remains available for viewing/download. Commands, file paths and source URLs are inert text: the inspector never runs commands, opens server logs, or fetches artifacts.

The raw inspector view/download preserves the original response text, including 64-bit integers. Rich browser values outside JavaScript's safe numeric range carry a precision warning. Browser import/plan previews reject unsafe numbers rather than silently rounding metadata or changing fingerprints; use the direct API for those numeric fields (or strings for free-form metadata).

**Import and reuse.** The visible import panel supports complete bundle JSON and llama-bench JSON. Choose files, preview validation, review the prepared bundle, and explicitly confirm the import. Preview and planning never persist records or run workloads; a preview does not reserve IDs, and confirmation revalidates against concurrent changes. The confirmation result links to the imported run.

The complete bundle format contains `model`, `artifact`, `runtime`, `hardware`, `workload`, `experiment`, and `run`, plus optional `performance_metrics`, `speculative_metrics`, `quality_results`, and `telemetry_samples`. `POST /api/benchmarks/validate` performs the same transactional ingestion checks and rolls back; `POST /api/benchmarks/ingest` commits. Cross-reference IDs must agree. SHA-256 fields require lowercase 64-character hex, metrics enforce bounds, and unknown fields are rejected. **Hashes and metadata are user-supplied declarations, not verified against artifact files.**

For repeated imports, edit a reusable template containing only the five registry definitions and `experiment`; save/load it explicitly in this browser or download it as JSON. Saving a template does not insert registry data into SQLite. It excludes run commands/results/timestamps, but registry metadata itself can contain sensitive information: do not save sensitive definitions in a shared browser. `/api/benchmarks/prepare` accepts `{ "template": {...}, "repetition": 0, "status": "succeeded", "exact_command": "...", "llama_bench": {...} }`, plus optional `started_at`, `ended_at`, and a selected plan `experiment`. It wires references from the template IDs, derives stable experiment/run IDs, and previews a complete bundle. A selected plan experiment must match the supplied registry definitions. Existing custom experiment IDs and repetition/run IDs are reused; changing only the repetition never rewrites registry fingerprints or capture timestamps. Download the prepared bundle to add other metrics/provenance, then validate it as a complete bundle.

Successful run payloads are immutable. To preserve reproducibility, corrections must use a new repetition or experiment. Other statuses (`planned`, `running`, `failed`, `oom`, `timeout`, `cancelled`, `skipped`) can receive a replacement full bundle until succeeded; run-to-experiment and repetition identity cannot change. Optional measurement sections replace that attempt's stored sections, not append a live telemetry stream. Registry rows and experiment configurations are insert-once.

**llama-bench adapter.** `/api/benchmarks/validate/llama-bench` and `/api/benchmarks/ingest/llama-bench` accept `{ "bundle": {...}, "llama_bench": ... }`. The output may be one object, an array, or an object with a `results` array. Every row must supply unambiguous metrics. Use one throughput alias (`avg_ts`, `tokens_per_second`, or `tps`) with `ppN` / `tgN` or positive `n_prompt` / `n_gen`; reported token counts must match the experiment. Combined rows require distinct `prompt_tps` and `generation_tps`, not a combined rate passed off as both. Repeated phase/TTFT rows, contradictory labels/counts, wrong metric types, conflicting existing bundle metrics, and unrecognized rows are rejected without persisting anything. Split multi-configuration results into separate imports; do not average rows from different configurations. Original uploaded output is retained under `run.raw_result.llama_bench`.

**Planning and examples.** The YAML/JSON matrix preview expands deterministically into experiments and exclusions, never execution. Select a candidate and combine it with a matching template to prepare an import; repetition is explicit and zero-based. Planning checks structure and configuration invariants, not the existence of artifact files, hardware feasibility, or runtime capabilities. Examples in [`examples/benchmarks/`](examples/benchmarks/) are **synthetic fixtures**, not measurements: [complete bundle](examples/benchmarks/synthetic-bundle.json), [matrix](examples/benchmarks/matrix.yaml), [llama-bench output](examples/benchmarks/llama-bench.json). Downloadable `/api/benchmarks/examples/{bundle.json,template.json,matrix.yaml,matrix.json,llama-bench.json}` versions use matching registry IDs and 8K/32K/128K contexts. Loading an example into an editor never inserts it; replacing synthetic metadata and placeholder hashes is the user's responsibility.

**Isolation and resource limits.** HTTP body parsing, validation, planning, SQLite calls and response serialization run on blocking workers, not Tokio async workers. Store clones and `BenchmarkStore::run_blocking` share **two admission slots**, with no waiting queue. Saturation or SQLite lock contention returns **503** with `Retry-After: 1`; core routing stays available. Each admitted upload has a 16 MiB body limit and 15-second body deadline. Cancellation releases admission only when the actual blocking task exits. Response allocations and emitted `Bytes` slices retain admission even after Hyper drops the body; 16 KiB frames also bound transport-side copying. SQLite lock waiting is at most five seconds; cancellation does not roll back an ingestion already executing, so check the run ID before retrying a disconnected import.

| Resource | Limit and recovery |
|---|---|
| HTTP SQLite row / JSON expression | 4 MiB; larger records remain in storage but require an offline reader or synchronous detail API |
| JSON response / CSV or JSONL export | 8 MiB, with an explicit **413**, never a partial success |
| HTTP run detail | 10,000 telemetry samples and 1,000 quality rows, plus byte limits; over-limit detail returns **413**, not a sampled or truncated timeline |
| Run query | 1-100 runs per page, plus byte limits; narrow filters or reduce `per_page` if necessary |
| Filter dropdowns | 1,000 distinct values per field, plus byte limits; use explicit run-query filters if exceeded |
| Matrix plan | 10,000 candidates and 8 MiB expansion/response budget |
| Filtered export | 10,000 runs and 8 MiB; narrow filters or export offline from a consistent SQLite snapshot |

CSV/JSONL exports contain **all matching run summaries or an explicit limit error**, independently of UI pagination; they are not full bundle or telemetry exports. Each export reads a single stable SQLite transaction and writes bounded pages directly into a capped buffer, never an unlimited list. Ordering is deterministic with run-ID ties. CSV quotes commas, quotes and newlines and neutralizes spreadsheet formula prefixes. Use the raw run JSON download for full detail within HTTP limits; larger history remains accessible through the synchronous store detail API or offline SQLite tools. The public synchronous API stays available for tests/offline consumers; async callers must use `store.run_blocking(move |store| store.run_detail(&id)).await` (or `query_runs`) to share admission and HTTP read budgets. Schema migration 2 adds only the native job lifecycle table and indexes. It
does not remove old records or turn optional benchmark storage/execution into a
daemon-startup dependency.

### API walkthrough: synthetic fixtures, no inference

From the repository root, with an existing daemon running, these first two
operations only validate/plan; they do not execute the command text or commit
benchmark records:

```bash
API=http://127.0.0.1:9099
curl --fail-with-body -sS -H 'Content-Type: application/json' \
  --data-binary @examples/benchmarks/synthetic-bundle.json \
  "$API/api/benchmarks/validate"
curl --fail-with-body -sS -H 'Content-Type: application/yaml' \
  --data-binary @examples/benchmarks/matrix.yaml \
  "$API/api/benchmarks/plan"
```

Only if you explicitly want a labelled synthetic record in the selected database:

```bash
curl --fail-with-body -sS -H 'Content-Type: application/json' \
  --data-binary @examples/benchmarks/synthetic-bundle.json \
  "$API/api/benchmarks/ingest"
curl --fail-with-body -sS \
  "$API/api/benchmarks/runs/synthetic-run-8k-0"
curl --fail-with-body -sS --get "$API/api/benchmarks/runs" \
  --data-urlencode 'status=succeeded' --data-urlencode 'sort=generation_tps' \
  --data-urlencode 'order=desc' --data-urlencode 'per_page=25'
curl --fail-with-body -sS \
  "$API/api/benchmarks/export?format=jsonl&status=succeeded" -o benchmarks.jsonl
curl --fail-with-body -sS \
  "$API/api/benchmarks/export?format=csv&status=succeeded" -o benchmarks.csv
```

Re-importing that succeeded run returns 409; previews are not idempotent upserts
for successful data. Use the template/plan preparation UI for a new repetition
with the same registry identity. Imports do not verify file hashes or execute
llama-bench/llama-perplexity. JSONL/CSV summaries are not a database backup;
back up the SQLite database consistently (including active WAL state).

---

## Headless CLI (`brainrouter cli`)

`brainrouter cli` is a thin client for core management, routing profiles and
reviews, using the daemon's Unix socket or TCP (`--url`). It does not load Bonsai
or run reviews in a second process; review commands ask the daemon to do the work
and may wait for it. Benchmark/observability workflows currently use their pages
or the HTTP API rather than dedicated CLI subcommands.

```bash
# Point at the daemon (only needed when the socket isn't the default)
brainrouter cli --socket /run/user/1000/brainrouter.sock status
brainrouter cli --url http://127.0.0.1:9099 status
```

All output is pretty-printed JSON. `brainrouter-config show` / `set` (alias `config`) print and write raw YAML instead.

### Health and telemetry

```bash
brainrouter cli status          # llama-swap, manifest, llama.cpp, toolbox + cloud-fallback
brainrouter cli versions         # installed + latest for every component
brainrouter cli inference        # loaded model, state, slot progress
brainrouter cli events           # recent routing decisions
brainrouter cli stats            # aggregated routing stats
brainrouter cli models           # proxy model list (auto/local/cloud + keys)
brainrouter cli models --llama-swap   # raw llama-swap model list
```

### Routing controls

```bash
brainrouter cli bonsai status            # classifier state (enabled/healthy)
brainrouter cli bonsai enable            # start classifier; affects roles set to Auto
brainrouter cli bonsai disable           # stop it, free VRAM — auto routing goes local
brainrouter cli bonsai toggle

brainrouter cli nudge status             # thinking-budget nudge state
brainrouter cli nudge enable             # alternatives: nudge disable / nudge toggle
brainrouter cli nudge tier auto          # Bonsai picks light/deep per request
brainrouter cli nudge tier light         # inject the tight budget
brainrouter cli nudge tier deep          # inject the full budget

brainrouter cli prompt-rewrite disable   # enable requires healthy Bonsai; explicit IDs bypass rewriting
brainrouter cli routing-mode status
brainrouter cli routing-mode set local   # auto | cloud | local
brainrouter cli routing-profile status
brainrouter cli routing-profile models   # metadata only; never starts a model
brainrouter cli routing-profile preset local_custom --main-model my-local-model
brainrouter cli routing-profile choose reviewer cloud --model vendor/reviewer-model
brainrouter cli routing-profile pool my-local-subagent-pool
brainrouter cli routing-profile pool     # clear pool; restore legacy auto behavior for subs
brainrouter cli routing-profile set ./profile.json  # same body as POST /api/routing-profile
brainrouter cli review-config update --forced-mode auto --clear-model

brainrouter cli bridges status
brainrouter cli bridges enable discord
brainrouter cli bridges disable signal
brainrouter cli bridges toggle signal
```

The dashboard has independent Main and Reviewer backend/model selectors and a separate
local Subagent pool selector. Model lists come from the configured providers' `/models`
endpoints using their existing authentication. Discovery errors remain visible; manual
explicit IDs are accepted even when discovery is unavailable. The provider validates
model availability when called. Choosing or saving a profile does not perform inference.

The older `brainrouter cli context` command still exists, but its `/api/context`
server handler is absent at this baseline (404). Automatic/manual runtime context
selection is therefore not a shipped control; benchmark matrix context values
are experiment metadata, not instructions to resize a running model.

| Preset | Main | Reviewer |
|---|---|---|
| `auto` | Classifier (local if off) | Classifier (local if off) |
| `cloud` | Manifest auto | Manifest auto |
| `local_main_sub` | Local default | Local default |
| `local_custom` | Required explicit local model | Local default |
| `cloud_main_local_review` | Manifest auto | Local default |
| `local_main_cloud_review` | Local default | Manifest auto |
| `custom` | Independent choice | Independent choice |

Presets preserve the independently selected subagent pool. Only `subs` /
`brainrouter/subs` requests use that pool; an unconfigured pool retains the legacy
classifier/auto behavior, not the main override. Presets can use explicit model IDs
within the indicated backends.

Main applies only to `auto`, `brainrouter/auto`, or empty/default model requests.
Explicit client IDs remain authoritative: bare names and `brainrouter/<id>` select
local models, while `cloud/<id>` selects an exact Manifest model. Explicit `local`
and `cloud` aliases still select their backend defaults. An explicit local failure
returns an error rather than switching models. Cloud failures and disabled cloud
retain the existing local fallback policy; requested and actual routes are shown
in the dashboard and review metadata. Cloud traffic requires `manifest.enabled: true`
in YAML; a profile never changes that opt-in.

Preferences are saved atomically, owner-readable/writable only, to
`$XDG_CONFIG_HOME/brainrouter/routing_state.json` (otherwise
`~/.config/brainrouter/routing_state.json`). Saved profiles override YAML role
defaults after restart. To reset to YAML, stop the daemon and back up/move aside
both routing state and any legacy review state; deleting only the new file can
trigger legacy migration again. Valid
legacy `review_state.json` overrides migrate once when no new state exists and
are persisted immediately to `routing_state.json`; the legacy file is left unchanged.
For upgrade compatibility only, reading YAML or legacy review state with
`forced_mode: auto` discards a leftover `forced_model`, which older versions
ignored. A warning names the source file and asks you to remove that field or
choose local/cloud explicitly. This normalization does not apply to new writes:
the config, review-config, and routing-profile APIs still reject auto with an
explicit model. Local/cloud model choices and other validation remain strict;
read/migration failures report the source path and cause.
Review sessions snapshot their reviewer at creation; continuations retain
that choice. As before, `review.max_iterations` reloads from YAML after restart.
The legacy routing-mode and review-config APIs/CLI update the same preferences.

Example `profile.json` (IDs are declarations, not model downloads):

```json
{
  "preset": "local_main_cloud_review",
  "main": {"backend": "local", "model": "my-local-model"},
  "reviewer": {"backend": "cloud", "model": "vendor/reviewer-model"},
  "subagent_model": "my-local-subagent-pool"
}
```

Use raw provider IDs in `model`, not `brainrouter/` or `cloud/` selector prefixes.
For Auto, use `{"backend":"auto"}` with no explicit model. Saving does not verify
that the provider can serve the ID and never silently enables cloud access.
`POST /api/routing-profile` replaces the profile. The legacy
`POST /api/review-config` also replaces its configuration, not a partial merge;
the CLI reads/merges current fields before posting. The legacy
`POST /api/routing-mode` changes Main only. These JSON control bodies are capped
at 16 KiB and retain strict validation.

### Operations

```bash
brainrouter cli toolboxes                # list llama-* toolbox containers
brainrouter cli restart llama-swap       # llama-swap | llama-cpp | manifest | brainrouter
brainrouter cli upgrade llama-swap       # llama-swap | manifest | toolbox
brainrouter cli flush-models             # unload every model (frees VRAM)
brainrouter cli sync-omp                 # push llama-swap models → OMP models.yml (one-way)
brainrouter cli config-files             # config files the daemon manages
brainrouter cli brainrouter-config show # current brainrouter.yaml (raw YAML); `config` alias works
brainrouter cli brainrouter-config set ./new.yaml  # replace it; "-" reads stdin
brainrouter cli llama-swap-config show   # llama-swap's config.yaml
brainrouter cli llama-swap-config set ./config.yaml
```

CLI spelling vs dashboard: `llama-cpp` (CLI value) is the `llama.cpp` row in the dashboard sidebar — same component. `toolbox` (upgrade value) is the `toolboxes` container image.

### Reviews

```bash
brainrouter cli review list                     # all sessions
SESSION_ID="paste-returned-session-id"
brainrouter cli review get "$SESSION_ID"        # one session
# Blocking request — prints progress, polls every 5 s, 30-minute cap:
brainrouter cli review request feature-20260819-001 "Added role profile controls" \
    --cwd /home/papa/ai/projects/brainrouter
# Fire-and-forget, then poll yourself:
brainrouter cli review request feature-20260819-001 "…" --cwd /path/to/project --async
brainrouter cli review continue "$SESSION_ID"   # extra LLM rounds (keeps context)
brainrouter cli review lgtm "$SESSION_ID"       # quick-approve
brainrouter cli review resolve "$SESSION_ID" "feedback text"
```

Session IDs are validated before use in URLs (alphanumeric + `-`/`_`), so `review get` can't path-traverse.

> **Headless workflow:** run `brainrouter serve` via systemd, then use the CLI for
> core operations and HTTP for explorer/observability APIs. A review request or
> continuation does perform inference; metadata discovery and benchmark previews
> do not. Schedule restarts/upgrades separately from source/Git transfer.

---

## MCP code review guide

The review tool is exposed over MCP so any harness can call it after completing a task.

### How it works

1. Your harness calls `mcp_brainrouter_request_review` with a task ID and summary.
2. brainrouter gathers context: your project's PRD (auto-detected from `docs/PRD.md`, `PRD.md`, or `README.md`), the current `git diff HEAD`, and any `AGENTS.md`.
3. The new session snapshots its Reviewer choice and iteration settings. Main and Subagent changes do not change that reviewer.
4. The loop requests JSON `{status, feedback}` from the selected reviewer, with bounded rounds and parsing/error handling.
5. Agents can act on feedback, request a new review, or continue the existing session through the CLI/API. Continuation reuses its initial reviewer and prior in-memory turns.
6. Exhausted iterations or LLM errors escalate to human review. `/review/` redirects to the dashboard; session pages and JSON/CLI session listing remain available. Sessions/turns do not survive daemon restart.

### Tool parameters

| Parameter | Required | Description |
|---|---|---|
| `taskId` | yes | Unique ID for this task, e.g. `feature-20260424-001` |
| `summary` | yes | 2–3 sentences: what changed, why, and any assumptions |
| `details` | no | Additional technical context |
| `conversationHistory` | no | Array of strings — recent conversation for context |
| `cwd` | yes (CLI defaults to its own directory) | Absolute path to the project directory — required for accurate git diff. An invalid cwd (relative, `..`, empty) is rejected with HTTP 400, never silently replaced |

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
| `request_review` | `taskId`, `summary`, `cwd?`, `details?`, `conversationHistory?` | Start a new review; existing-session continuation uses the CLI/API |
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
  enabled: false                        # cloud routing off by default
  base_url: "http://localhost:3001/v1"   # required
  api_key_env: MANIFEST_API_KEY          # optional — name of env var holding mnfst_* key

llama_swap:
  base_url: "http://localhost:8081/v1"   # required
  fallback_model: "my-model"             # required — must match a key in llama-swap config
  local_models: ["my-model", "other"]    # optional — model keys that bypass Bonsai when used directly
  subs_model: "my-subs-pool"             # optional — model=subs / brainrouter/subs routes here, bypassing Bonsai
  local_system_prompt: "/path/to/prompt.md"  # optional — override built-in lean prompt
  nudge:
    enabled: false                       # thinking-budget injection off by default
    model_key: "my-model"                # optional — which local model receives the budget
    budgets: { light: 10240, deep: 12288 }

bonsai:
  enabled: false                         # classifier off by default; auto → local while off
  model_path: "/mnt/models/prism/Bonsai-27B-dspark-bf16.gguf"  # required when enabled
  server_port: 9200                                    # external llama-server port (default)
  fork_path: "/path/to/llama-server"                   # PrismML fork binary (default ~/.local/share/brainrouter/llama-prism/llama-server)

models:
  path: /opt/models                                    # shared GGUF dir; ${models_path} expands to this
  shared_write: false                                  # true = all aistack members can add/delete models

benchmarks:
  database_path: "/home/you/.local/share/brainrouter/benchmarks.sqlite3"
  lab:
    enabled: false                       # opt in to native Riddllr/Plumebench jobs
    riddllr_root: "/home/you/ai/projects/riddllr"
    plumebench_root: "/home/you/ai/projects/plumebench"
    workspace_path: "/home/you/.local/share/brainrouter/benchmark-lab"
    omp_bin: "/home/you/.bun/bin/omp"
    plumebench_sandbox_bin: "/usr/bin/bwrap"
    python_bin: "python3"
    max_job_seconds: 900                 # whole suite job deadline, 1..86400
    riddllr_max_tokens: 4096
    plumebench_max_turns: 40
    plumebench_thinking: "low"           # off|minimal|low|medium|high|xhigh|max|auto

review:
  max_iterations: 5           # LLM review rounds before escalating to human
  forced_mode: "local"        # "auto" | "cloud" | "local" — default "local"
  forced_model: "my-model"    # explicit local OR cloud ID; null for backend default

# Optional YAML role defaults; saved dashboard/CLI preferences take precedence.
routing:
  preset: local_main_cloud_review
  main: { backend: local, model: null }
  reviewer: { backend: cloud, model: "vendor/reviewer-model" }
  subagent_model: "my-local-subagent-pool"
# This example selects a cloud reviewer but leaves cloud disabled above.
# Set manifest.enabled: true explicitly before expecting cloud execution.

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

### State and restart behavior

| State | Location / behavior |
|---|---|
| Selected YAML | `serve --config`, else `$XDG_CONFIG_HOME/brainrouter/brainrouter.yaml` or `~/.config/brainrouter/brainrouter.yaml` |
| Role preferences | Per-user `routing_state.json` in the default config directory, even with a different `--config`; overrides YAML role defaults |
| Legacy review overrides | Sibling `review_state.json`, read once if no new routing state exists, then normalized/persisted; source left unchanged |
| Benchmark records | `benchmarks.database_path`, default `$XDG_DATA_HOME/brainrouter/benchmarks.sqlite3` or `~/.local/share/brainrouter/benchmarks.sqlite3` |
| Observation policy/mappings | Selected YAML path with extension replaced by `.observability.json`; no live request history |
| Review sessions, route/sample buffers, alert latches, in-flight state | Process memory only; cleared on restart |

Role writes use owner-only atomic files and validate before changing runtime
preferences. Observation writes add revision checks and explicit I/O diagnostics.
Raw benchmark/environment/result data can be sensitive; use appropriate filesystem
permissions and redact imports/exports. No environment file is automatically
loaded by the daemon: the service or shell must supply the configured credentials.

### Environment variables

| Variable | Description |
|---|---|
| `RUST_LOG` | Log level filter (default `info`). Overrides `--log-level`. Example: `RUST_LOG=debug` |
| `HOME` | User home directory. Used for default paths |
| `XDG_RUNTIME_DIR` | Socket base; daemon uses `/run/brainrouter.sock` if absent (login systems commonly set `/run/user/$UID`) |
| `XDG_CONFIG_HOME` | User YAML/role-state directory base (default `~/.config`) |
| `XDG_DATA_HOME` | Data directory containing the default benchmark SQLite database (default `~/.local/share`) |
| `BRAINROUTER_MANIFEST_DIR` | Override Manifest docker-compose directory for restart/upgrade |
| `<manifest.api_key_env>` | Dynamic: whatever env var name is set in `manifest.api_key_env` (e.g. `MANIFEST_API_KEY`) holds the Manifest API key |

### Subcommands

| Command | Description |
|---|---|
| `brainrouter serve` | HTTP proxy daemon. Listens on TCP `:9099` and UDS `/run/user/$UID/brainrouter.sock` |
| `brainrouter cli` | Core management, role profiles and reviews over the daemon API. See [Headless CLI](#headless-cli-brainrouter-cli) |
| `brainrouter mcp` | MCP stdio server. Spawned by harnesses; forwards tool calls to the daemon over UDS |
| `brainrouter install <harness>` | Idempotently patches harness config. Harnesses: `omp`, `vibe`, `opencode`, `codex`, `droid`, `claude`, `pi` |

### HTTP API

All on `http://127.0.0.1:9099`.

#### Proxy

| Method | Path | Protocol | Notes |
|---|---|---|---|
| `GET` | `/health` | — | `{"status":"ok"}` |
| `GET` | `/v1/models` | OpenAI | Managed aliases plus available llama-swap model IDs |
| `POST` | `/v1/chat/completions` | OpenAI | Main routing endpoint |
| `POST` | `/v1/messages` | Anthropic | For Claude Code and droid |

#### Management (localhost-only, CSRF-protected)

Protected mutations require a loopback peer or the Unix socket. Browser `Origin`/`Referer` URLs may use HTTP `localhost`, IPv4 loopback, or IPv6 loopback on any port, including a port remapped by a local tunnel or proxy. `Origin: null`, non-loopback origins, and credential-bearing URLs remain forbidden. Non-loopback proxy frontends need a separate trusted-origin policy; port forwarding does not remove the peer restriction. There is no general proxy/read-API authentication: keep the daemon loopback-bound unless a separate access-control layer is in place.

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/versions` | Installed versions + latest available |
| `GET` | `/api/routing-events` | Live routing events feed |
| `GET` | `/api/inflight` | Active in-flight requests (for the dashboard tracker) |
| `POST` | `/api/inflight/cancel` | Cancel one in-flight request — body `{id}`; 404 for an unknown id |
| `GET` | `/api/omp-sessions` | OMP session titles for the Sankey SESSION column |
| `GET` | `/api/routing-stats` | Routing statistics |
| `GET` | `/api/service-health` | Service health status per provider |
| `GET` | `/api/bridge-status` | Bridge transport status (Discord / Signal) |
| `GET` | `/api/inference-status` | Current inference state (for progress bar) |
| `GET` | `/api/review-config` | Current review mode and forced model |
| `POST` | `/api/review-config` | Replace current/new-review configuration; existing session snapshots stay unchanged |
| `GET` | `/api/models/llama-swap` | Model list from llama-swap |
| `POST` | `/api/upgrade/llama-swap` | Build and install latest llama-swap binary |
| `POST` | `/api/upgrade/manifest` | Pull latest Manifest image and recreate container |
| `POST` | `/api/upgrade/toolbox` | Pull latest toolbox image and recreate container |
| `POST` | `/api/restart/:service` | Restart `llama-swap`, `manifest`, `llama-cpp`, or `brainrouter` |
| `GET` | `/api/bonsai` | Bonsai classifier server state (`enabled`, `healthy`) |
| `POST` | `/api/bonsai/toggle` | Stop/start the Bonsai classifier to free/reclaim VRAM |
| `GET/POST` | `/api/nudge` | Thinking-budget nudge state / update `{enabled, tier}` |
| `GET/POST` | `/api/prompt-rewrite` | Prompt-rewrite state / update `{enabled}` |
| `GET/POST` | `/api/routing-mode` | Routing override / set `{mode}` |
| `GET/POST` | `/api/routing-profile` | Independent role profile; POST `{preset,main,reviewer,subagent_model}` |
| `GET` | `/api/routing-models` | Local/cloud catalog with explicit per-provider discovery errors; no inference |
| `POST` | `/api/bridges/toggle` | Toggle `{bridge, enabled}`; read state from `/api/bridge-status` |
| `GET` | `/api/toolboxes` | List llama-* toolbox containers |
| `POST` | `/api/models/flush` | Unload all models from llama-swap memory (no restart) |
| `POST` | `/api/models/sync-omp` | Sync llama-swap models into OMP's models.yml |
| `GET` | `/api/config-files` | Config files the daemon manages |
| `GET/POST` | `/api/config` | Read raw YAML / validate and replace YAML; source changes require restart |
| `GET/POST` | `/api/llama-swap-config` | Read/write llama-swap YAML |

#### Model observability

| Method | Path | Notes |
|---|---|---|
| `GET` | `/models` | Local model activity/measurement/alert/reference page; `?model=` selects a key |
| `GET` | `/api/observability/models` | Cached observations, active/recent data, attribution/errors, policy/retention/measurement definitions |
| `GET` | `/api/observability/settings` | Current policy/mappings, revision, source path, read/write errors |
| `POST` | `/api/observability/settings` | `{revision,policy}` with all policy fields; revision checked |
| `GET` | `/api/observability/reference?run_id=...` | Validate/project a successful measured benchmark reference |
| `GET` | `/api/observability/baseline?model_key=...` | Explicit reference state/reason; lookup failures do not disable live observations |
| `POST` | `/api/observability/baseline` | `{revision,model_key,run_id,expected_experiment_hash,note}`; null run ID clears |

#### Benchmarks

| Method | Path | Notes |
|---|---|---|
| `GET` | `/benchmarks` | Native Benchmark Lab plus interactive benchmark explorer/import workflow |
| `GET` | `/api/benchmarks/lab/suites` | Configured Riddllr/Plumebench availability, cases and source manifest hashes |
| `GET` | `/api/benchmarks/lab/jobs` | Recent persisted native jobs; optional `limit` 1-100 |
| `GET` | `/api/benchmarks/lab/jobs/:id` | Current or persisted job detail |
| `POST` | `/api/benchmarks/lab/jobs` | Queue `{suite,case_id,model,repetition}`; localhost-only, one heavy execution slot |
| `POST` | `/api/benchmarks/lab/jobs/:id/cancel` | Cancel a queued/running job and terminate its process group |
| `GET` | `/api/benchmarks/runs` | Filtered page; supports `page`, `per_page` (1-100), `q`, `status`, `family`, `backend`, `workload`, `quant_name`, `speculator_type`, `sort`, and `order` |
| `GET` | `/api/benchmarks/runs/:id` | Full run detail within row/sample/byte limits; explicit 413 otherwise |
| `GET` | `/api/benchmarks/filters` | Deterministically ordered filter values |
| `GET` | `/api/benchmarks/export` | `format=csv` or `format=jsonl` (default); all matching summaries or explicit 413; 10,000-run / 8 MiB cap |
| `GET` | `/api/benchmarks/examples/:name` | Synthetic bundle, template, matrix YAML/JSON, or llama-bench JSON download; no inserts |
| `POST` | `/api/benchmarks/prepare` | Reusable template plus run fields / optional selected experiment / llama-bench output -> validated bundle preview |
| `POST` | `/api/benchmarks/validate` | Transactionally validate a complete bundle then roll back; no persistence |
| `POST` | `/api/benchmarks/validate/llama-bench` | Validate an adapter payload and return a complete bundle preview |
| `POST` | `/api/benchmarks/ingest` | Validate and transactionally import one result bundle (localhost-only) |
| `POST` | `/api/benchmarks/ingest/llama-bench` | Import a bundle plus llama-bench JSON output (localhost-only) |
| `POST` | `/api/benchmarks/plan` | Deterministically expand a JSON/YAML experiment matrix without running it (localhost-only) |

#### Review

| Method | Path | Notes |
|---|---|---|
| `GET` | `/review/` | Redirect to `/dashboard` |
| `GET` | `/review/session/:id` | Session detail |
| `GET` | `/review/api/sessions` | JSON session list |
| `GET` | `/review/api/sessions/:id` | JSON session detail |
| `POST` | `/review/api/request-async` | Start a review, return `sessionId` immediately. Body: `{taskId, summary, details?, cwd?}` |
| `POST` | `/review/api/request` | Legacy blocking variant — still routed, no client uses it |
| `POST` | `/review/api/resolve` | Resolve a review session. Body: `{sessionId, feedback}` |
| `POST` | `/review/api/continue` | Continue a review iteration (seeds prior turns; waits on human if it escalates) |
| `POST` | `/review/api/lgtm` | Quick-approve a review session |
| `POST` | `/review/session/:id/resolve` | Human resolve. Body: `{feedback: "lgtm"}` |

All `/review/api/*` POSTs go through the same loopback-only + Origin-check gate as the other management endpoints.
The legacy `/review/session/:id/resolve` form path is not covered by that prefix
gate; prefer the protected JSON APIs and do not expose the listener publicly.

### Architecture

```
src/
  main.rs            	-- clap dispatcher (serve | cli | mcp | install)
  cli.rs             	-- core management, profiles and reviews over the daemon API
  daemon_client.rs   	-- shared thin HTTP client (UDS + TCP) used by cli and mcp
  daemon.rs          	-- core startup; optional benchmark storage; enabled Bonsai only
  server.rs          	-- HTTP dispatch and protected API access gates
  classifier.rs      	-- Cloud/Local decision via external Bonsai server
  bonsai_server.rs    	-- Bonsai llama-server lifecycle (spawn, health, dashboard toggle)
  router.rs          	-- routes to Manifest or llama-swap; circuit breaker; fallback
  routing_profile.rs 	-- typed Main/Reviewer/Subagent choices, atomic preferences, migration
  prompt_rewriter.rs 	-- system prompt rewriter for local mode
  anthropic.rs       	-- Anthropic <> OpenAI protocol translation
  mcp_server.rs      	-- JSON-RPC stdio, forwards to daemon over UDS
  install.rs         	-- idempotent harness config merger
  session.rs         	-- in-memory review session store
  config.rs          	-- YAML config parsing and validation
  types.rs           	-- OpenAI-compatible request/response types
  lib.rs             	-- library root
  peer_cwd.rs        	-- peer CWD resolution via /proc
  routing_events.rs  	-- bounded events and event-correlated completed measurements
  benchmark.rs       	-- validated registry/domain, SQLite store, queries and budgets
  benchmark/
    http.rs          	-- admission, bounded bodies/responses, blocking HTTP work
    imports.rs       	-- template preparation and synthetic example downloads
    http_tests.rs    	-- synthetic response/backpressure/API coverage
  observability.rs   	-- read-only polling, rolling alerts and explicit reference settings
  inference_state.rs 	-- inference state tracking
  review/
    mod.rs           	-- ReviewService
    review_loop.rs   	-- iterative LLM review loop
    context.rs       	-- gathers PRD, git diff, AGENTS.md
    prompt.rs        	-- review prompt template
  escalation/
    mod.rs           	-- /review/* HTTP handlers + ReviewRequest parsing
    templates/       	-- dashboard, session, benchmark inspector and model activity HTML
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

The initial benchmark schema is in `migrations/0001_benchmark_explorer.sql`;
synthetic import/planning fixtures are in `examples/benchmarks/`.

### External services

| Service | Purpose | Default URL |
|---|---|---|
| **Manifest** | Cloud LLM router — provider selection, failover, cost tracking | `http://localhost:3001` |
| **llama-swap** | Local model runner — spawns llama-server on demand | `http://localhost:8081` |
| **Bonsai** | External classifier — llama-server on port 9200 (PrismML fork) | `http://127.0.0.1:9200` |

### Tests

```bash
cargo test --locked
cargo clippy --locked --all-targets
node --test scripts/test-benchmark-ui.cjs
bash scripts/check-html-js.sh
cargo build --locked --bin brainrouter
```

The repository test suite covers core routing/failover, both protocols, exact
role choices, review snapshots, event-correlated measurements, real-daemon
startup under bad benchmark storage and legacy auto state, strict write APIs,
transactional imports/previews, Benchmark Lab discovery/grading/lifecycle
helpers, backpressure/limits, inspector workflows and alert rules. Automated
tests use fixtures and mock processes rather than consuming a real model slot.
Node DOM tests are not a real-browser/accessibility certification. Existing
unrelated Clippy warnings remain; do not confuse baseline warnings with new
diagnostics.

## Planned, not shipped

The explorer/observability foundation does **not** implement the complete feature
plans found in the separate Quant Lab / Model Compare notes:

- **Quant Lab:** actual llama-perplexity/PPL/KLD execution, same-family reference
  compatibility, queued/sequential jobs with cancellation/live progress, GGUF
  parsing, and per-token distributions.
- **Model Compare:** complete common-suite execution for unrelated model
  families, compatible token-space/KLD checks, and blind human evaluation.
- Persistent request/review history, cloud cost/budget enforcement, per-project
  profiles, optional strict cloud fallback consent, reviewer-diversity policies,
  or review escalation ladders.
- Automatic benchmark scheduling, controlled regression campaigns, model
  switching, outbound alerts, Parquet export, or runtime-loaded-asset attestation.
- A working `/api/context` control/automatic runtime context selection; the
  retained legacy CLI command alone does not implement it.

Ember/Flash environment provisioning and cache-building work are not shipped
Brainrouter features. Imported metrics, declared hashes and existing charts do
not imply that these execution or verification pipelines exist.
