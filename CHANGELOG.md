# Changelog

All notable changes to brainrouter are documented here. Phase tags and per-leg
revert contract per `docs/design/v2-streamlining.md` ("Rollout, migration, and
rollback").

## v2 L4 — Swap supervisor + platform probe (branch `v2-p4-l4`, pre-merge)

brainrouter now owns engine processes and model lifecycle behind one
abstraction, with the owner's icon-script invariants inherited: single-instance
per-GGUF lock, start-ownership, health, and a served-model stem assertion. The
Residency view is the input side of every decision — **Residency before
budget**: a model already serving anywhere (llama-swap, a native child, a
server-mode container, or the L5 external slot) is attached with no spawn and
no budget gate. Shim-mode routing stays byte-identical.

### Added

- **Platform probe** (`src/platform.rs`): machine-id identity, GPU arch from
  KFD topology (sysfs PCI fallback), `MemTotal`/`MemAvailable`, GTT from
  `ttm.pages_limit`/`amdgpu.gttsize`, install markers, a vendored feed
  (`strix-halo`, `r9700`, `gb10`, `intel-b70`) plus an optional remote feed
  URL cached under `$XDG_STATE_HOME/brainrouter/platform_feed.json` with a
  24 h staleness bound. `id = feed platform id` when the detected arch
  matches, else `generic-<gpu-arch>` with an explicit unknown-platform banner
  (fedora1 = `generic-gfx1100`, fail-open: every engine/toolbox offered). The
  engine compatibility matrix is data (`ENGINE_MATRIX`: strixite = gfx1151 +
  GTT ≥ 100 GiB, gufo = gfx1151, hipfire = discrete gfx9/10/11/12,
  llama.cpp = universal) and drives offers + reasons-not. The dashboard's
  hardcoded `CURRENT_PLATFORM_ID` is replaced by the probe's id at render
  time; `GET /api/platform` exposes the probe.
- **Swap catalog** (`src/swap_catalog.rs`): `swap_catalog.json` (0600,
  atomic) with entries `{id, engine, files, ctx, quant, family, tools,
  vram_estimate_gib, launch_spec, aliases, groups, provenance,
  last_stream_completed, ewma_load_ms, fingerprints}`, sized **pins**, and
  llama-swap **groups** as metadata. Entry fingerprints (mtime+size) make a
  changed GGUF a `spec_stale` signal.
- **llama-swap import** (inverse of `llama_swap_register.rs`): parses
  `config.yaml`, resolves the macro chain recursively (top-level `macros:`
  shadowed by per-model `macros:`, so `${common}` → `${ctx}` → concrete
  flags) while preserving runtime placeholders (`${PORT}`), maps `--model`/
  `-c`/`--alias`/`--jinja`, records `env`, `aliases`, groups, and llama-swap
  settings. Idempotent merge (CLI-owned entries are never overwritten);
  `--dry-run` is stable and inert.
- **Residency view** (`src/residency.rs`): shim (`/running`, every entry, not
  just `running[0]`), native children, server-mode container identities, and
  the external-engine registry slot (L5 feeds it). `ready_for` matches
  catalog id / canonical GGUF / served alias.
- **Swap supervisor** (`src/swap_scheduler.rs`): lifecycle
  `Stopped|Loading|Ready|Draining|Failed`; `ensure` = Residency → queue →
  budget admission → LRU evict-to-fit → spawn; in-memory queue with cap +
  cancel; pins counted once as `max(vram_estimate, reserved_gib)`; append-only
  `budget.json` ledger; in-flight-aware LRU keyed on last **stream
  completion**; child-held per-GGUF `flock` (inherited fd, CLOEXEC cleared);
  adoption markers (`adoption/<model>.json` with pid/start_time/spec_hash)
  written before spawn and scanned at boot — live+match ⇒ adopt, mismatch ⇒
  adopt + `spec_stale` (never killed), dead ⇒ clean; bounded restart.
  `FakeAdapter` is the scripted acceptance substrate (load ms, stem set,
  crash, progress) with a real child process.
- **Native llama.cpp adapter** (`src/engine_native.rs`): spawns through
  `systemd-run --user --scope --quiet --collect -p KillMode=control-group`
  (cgroup escape — adoption needs the child outside the daemon's cgroup),
  plain `setsid` fallback where systemd is absent (documented best-effort
  adoption); SIGTERM → grace → KILL plus scope stop; `/health` + `/v1/models`
  status; load progress indeterminate + elapsed (EWMA ETA from the registry).
- **Shared llama-server shaping**: `sanitize_for_llama_family` (developer→
  system, single system block, assistant content fix) is applied to the shim
  and native paths identically; a golden-body test pins byte equality.
- **Engines tab + API**: `GET /api/engines` (models/queue/budget/adoption),
  `/api/engines/{ensure,evict,pin,unpin,import/preview,import/apply,logs,
  catalog/add,catalog/rm}`, `GET /api/platform`; loopback+CSRF gated
  mutations; Engines tab (platform card, offers+reasons, lifecycle,
  queue+cancel, budget gauge by source + pins, import panel) and a Models-tab
  "serve via" column.
- **CLI**: `brainrouter swap status|ensure|evict|catalog list|
  import-llamaswap [--dry-run]|pin|unpin|add|rm` and
  `brainrouter platform show [--json]`. Reads are direct; mutations go
  through the daemon and refuse cleanly when it is down.
- **Migration `v2.0-p4`**: first boot imports the live llama-swap config into
  `swap_catalog.json` (one stable snapshot directory per phase, atomic
  writes, CLI-owned entries preserved, one-shot marker);
  `--dry-run` previews and `migration restore v2.0-p4` round-trips the
  pre-migration catalog + budget ledger.
- **Events + in-flight (additive)**: `RouteEvent.swap_state` /
  `swap_source`; in-flight rows gain `catalog_id` (stamped at route time)
  and a clean-stream-completion hook that writes the LRU clock.

### Revert

`git revert -m 1 <merge>` for the phase; `brainrouter migration restore
v2.0-p4` restores the pre-import catalog/budget, then `v2.0-p3`/`v2.0-p2`
restore the pre-L3/L2 state. `v1-classic` remains the global rollback.

## v2 L3 — Role cards + decision slot (branch `v2-p3-roles`, pre-merge)

One model config for everything: role cards replace the routing profile, the
per-project pins store, presets, the reviewer `forced_*` keys, the Bonsai
config special case, and the sync-omp/`models.yml` machinery. Every proxied
request resolves through a single grammar — role (header, `subs`, review
loop) → card (project > global > default `auto`) → selector
(`auto | local | cloud | cloud/<id> | <bare-key>`), with v1's per-selector
strictness preserved verbatim. Explicit selectors bypass cards and are never
policy-clamped.

### Added

- **Role cards** (`src/roles.rs`): `role_cards.json` (0600, atomic) with
  `{scope, role, assignment}` records, project→global→default precedence,
  guardrail **policy** records (privacy `cloud-ok|local-only`, quality
  `fast|balanced|maximum`), and the review card's surviving extras
  (`max_iterations`, `hankndory_integration`). Migration table is exact
  (`Auto→auto, Local{None}→local, Local{Some}→<bare-key>, Cloud{None}→cloud,
  Cloud{Some}→cloud/<id>`); the `v2.0-p3` snapshot phase covers config +
  credentials + pins + routing state and `restore v2.0-p3` byte-round-trips
  the v1 YAML. `--dry-run` previews both phases.
- **Decision slot** (`src/decision.rs`, `decision_model:` config): features
  (`(bytes+3)/4` tokens, role, privacy, msg_count, has_tools, code_density,
  project) → `{route, tier, confidence}`. Kinds: `heuristic` (v1
  selector-matrix parity), `openai-compatible` (Bonsai migrates to exactly
  this, `prompt_profile: bonsai`, v1 prompt/grammar verbatim, v1-safe
  failure rows), `structured-schema` (`schema_version: 1` question, strict
  JSON), `webhook` (features JSON, no prompt excerpt), `plugin`
  (server-initiated `decision.classify`, plugin RPC **protocol v2** with
  request-id correlation + `timeout_ms`, negotiated at `initialize`).
  Guardrail clamps (privacy ⇒ local; quality raises tier) apply to slot
  decisions only. `auto` selects from the pool (llama-swap models +
  server-mode identities; `parameters.ctx` ranking, known-ctx only) and
  chains on connect/headline failure — never mid-stream.
- **Events** are additive: `role`, `decision{route,tier,confidence}`,
  `rationale`, `decision_fallback`, `chosen_model`.
- **Endpoints/CLI**: `GET/POST /api/roles*`, `/api/policy*`,
  `/api/credentials*` (masked list, add/update/remove/test + verdicts,
  grants, audit; loopback+CSRF-gated, plaintext never leaves the process),
  `GET /api/decision`, `POST /api/decision/test`; CLI verbs `roles
  get|set|remove`, `decision show|test`, `credential list|add|remove|test`
  (mutations via daemon; reads direct-state when down).
- **Dashboard**: Credentials tab (masked list/add/edit/delete/Test/
  grants/audit), role-card strip with per-project scope, policy chips,
  read-only Decisions panel, and a Sankey ROUTING column reading
  role → route/tier.
- Install templates are one entry (`auto`) per harness.

### Removed

- `routing_profile` store/presets/CLI/dashboard widgets, `project_pins`
  store, `review.forced_mode/forced_model`, the `bonsai:` config block
  (migrated; `bonsai_server.rs` stays as the launch backend under
  `decision_model.launch`), and `sync-omp` at all four points (CLI,
  route, daemon bootstrap, CSRF; templates print one `auto` entry).
- The proxy fast-fails: a routing failure landing within 15 s answers with
  a real HTTP status (deferred-200 streaming is preserved for slow model
  loads).

### Migrations

`v2.0-p3`: v1 `routing:` blocks, persisted profiles, project pins,
`review.forced_*`, and `bonsai.*` → role cards + `decision_model:` (exact
table, snapshot + `.bak`, `--dry-run`, `restore v2.0-p3`).

## v2 L2 — Plugins + bridge pilot + credentials store (branch `v2-p2-plugins`, pre-merge)

Sidecar plugins replace the compiled-in bridge (design R3,
`docs/design/v2-streamlining.L2.md`). Discord/Signal moved out of the core
binary into the `plugins/bridge/` plugin (`brainrouter-bridge`), spoken to
over a per-plugin Unix-domain socket using newline-delimited JSON-RPC 2.0.
The minimal credentials store (`src/credentials.rs`) ships with its first
consumer: the v1 plaintext `bridge.discord.token` migrates into
`credentials.json` (0600, atomic) at boot and reaches the plugin only
through grant-scoped `secrets.get`.

### Added

- **Plugin core** (`src/plugins/`): `plugin.toml` manifest
  (id/name/version/binary/min_daemon_version/grants), supervisor (boot
  spawn, exponential backoff capped at 60 s, 5 consecutive crashes →
  terminal `failed` until explicit enable, SIGTERM-grace stop with explicit
  PID kill), per-plugin UDS server (`$XDG_RUNTIME_DIR/brainrouter/plugins/
  <id>.sock`, dir 0700 / socket 0600, stale-socket unlink, peer-uid check),
  `initialize` handshake (constant-time token compare, grants + protocol
  version returned; auth failure closes+logs and never counts as a crash),
  per-plugin log capture, plugin state dir (`$XDG_STATE_HOME/brainrouter/
  plugins/<id>/`, 0700).
- **RPC v1 surface** (JSON-RPC 2.0, additive-only): `chat.completions`
  (full router path; routing events carry the additive
  `source: "plugin <id>"` tag), `models.list`, `routing.events.tail`,
  `inflight.list`, `config.get/set` + `kv.get/set` (namespaced per plugin,
  0600), `secrets.get` (grant-enforced), `notify`, `set_status`;
  daemon→plugin `config.changed` push; 1 MiB body cap; per-plugin in-flight
  4 + queue 16, excess → JSON-RPC limit error −32005; unknown method
  −32601. No system ops.
- **Credentials store** (`src/credentials.rs`): `credentials.json` (0600,
  atomic owner-only writes); kinds `store`/`env`/`file`; `SecretString`
  masks in Debug/Display/serialization; `resolve()` single implementation,
  never panics; grants live on the credential (removing a plugin drops
  grants naming it, logged).
- **Migration framework** (`src/migration.rs`): boot-time
  `bridge.discord.token` → credential `discord_token` granted to `bridge`
  (snapshot first to `$XDG_STATE_HOME/brainrouter/migrations/v2.0-p2/
  <UTC timestamp>/`, `serve --dry-run` preview, logged, YAML key removed
  with `.bak`); named restore `brainrouter migration restore v2.0-p2`.
- **CLI**: `brainrouter plugin install <dir>|list|enable|disable|remove|
  status|config set <id> k=v` (also `brainrouter cli plugin …`);
  `brainrouter migration restore <phase>`.
- **Bridge plugin** (`plugins/bridge/`, standalone workspace → binary
  `brainrouter-bridge`): the v1 transports moved as-is (behavior
  unchanged; Signal keeps its direct llama-swap listing via
  `llama_swap_url`); token via `secrets.get` only; per-transport pause
  keys move into plugin config; on first start the six v1 persistence
  JSON files are copied into the plugin state dir (v1 originals kept).
- **Dashboard**: Plugins tab (list with state/status line, enable/disable,
  config view/edit, HTML-escaped plugin logs, notification toasts).
- `GET /api/plugins*` endpoints; `cargo tree -e normal` core no longer
  contains serenity.

### Removed

- Cargo features `bridge-discord`/`bridge-signal` and deps
  serenity/async-trait/serde_yml from the core.
- `src/bridge/` (moved to the plugin crate), daemon bridge startup,
  `BridgeManager` from `AppState`, `bridge.*` YAML keys (the migration
  owns them), the `bridges` CLI verb, `GET /api/bridge-status`,
  `POST /api/bridges/toggle`, and the dashboard bridge config panel.

### Step 9 review pass (pre-merge fixes)

Restore no longer touches an unreadable-shaped credential set (map/object
form handled; corrupt shapes bail instead of wiping); every `/api/plugins*`
and plugin-state entry point validates the plugin id before touching the
filesystem; the daemon-less `plugin enable` persists state only (no
unsupervised child); corrupt `plugins.json`/`credentials.json` boot
quarantined (`.corrupt-<ts>`, daemon stays up); CI covers the
`plugins/bridge` workspace; plugin log/status text is scrubbed of control
characters; snapshot files and the migration snapshot set are written 0600.

### v2 L3 Step 9 (roles review pass)

Migration: a v1 config without a `bonsai:` block now receives the
parity-anchor `decision_model:` section, making the `v2.0-p3` migration
one-shot (previously it re-ran on EVERY boot over its own output: cards
reset to the stale v1 profile, per-boot snapshots). Snapshots are now one
directory per phase (stable path; a one-shot migration has one pre-state;
the legacy timestamped layout stays restorable), and the p3 snapshot
captures `role_cards.json`, so restore rolls the cards back. The migrated
YAML writes atomically; `restore v2.0-p3` byte-round-trips the pristine v1
YAML. The peer cwd used for project scope is re-read per REQUEST (a CLI
peer that chdir'd after connecting resolves its CURRENT project), and the
`/proc/net/tcp` scan identifies the client socket by (local port, remote
daemon port) — a bare port match can land on a foreign LISTEN socket under
ephemeral-port churn (the confirmed cause of the scope flakes). Proxy
fast-fail contract: routing failures within 15 s of the request return a
real HTTP status; the deferred-200 streaming contract for slow resolutions
(model loads) is unchanged. The prompt-rewrite gate now requires the
decision slot (any kind), probing classifier health only for the bonsai
profile. Bridge `review` commands set/read the review role card
(`/api/roles/set`; `/api/review-config` is reduced). Dashboard: scope
dropdown + model datalist escape option text; the session review label
shows the card snapshot (`review_selector`); Decisions panel gained the
slot summary + feature probe (`/api/decision/features`); the Credentials
tab gained the usage audit (`/api/credentials/audit`: masked records +
grant consumers). Boot-window hardening: listeners bind before the boot
work (UDS first), the CLI tolerates the startup window, shutdown sweeps
plugin children spawned mid-boot (no orphans), and a deleted state dir is
a clean startup error.

### Restoring v2.0-p2

`brainrouter migration restore v2.0-p2` restores the newest snapshot
(config YAML incl. the v1 `bridge.discord.token` key + credentials store)
— reverting code alone is never the whole restore.

## v2 L1 — P0-Purge (branch `v2-p1-purge`, pre-merge)

Removed the v1 halogen/vllm/r9v serving backends and moved the benchmark
explorer behind a non-default `benchmarks` cargo feature (design R1 + R2).
Kept backends (llama_cpp/ds4/gufo + Manifest/llama-swap routing) behave
identically; `src/router.rs` and the keeper modules
(`src/inflight.rs`, `src/routing_events.rs`, `src/observability.rs`) are
untouched.

### Removed

- **halogen / vllm / r9v server mode**: builders, request types, container
  name constants, and their `/api/server-mode/{halogen,vllm,r9v}/*` route
  registrations (incl. `vllm/cache-paths`, `r9v/paths`) — those paths answer
  404.
- **halogen / r9v download payloads**, the `PreparePle` job kind
  (`/api/model-downloads/r9v/prepare-ple` → 404), its `running` job status,
  `ModelPresence.ple_ready`, and the R9V sha256-specific verification copy.
- **`SupportedServingBackend::{Halogen, Vllm, R9v}`** + `ModelPayload::{…}`
  variants and `HalogenModel`/`VllmModel`/`R9vModel`/`R9vPleFile` typed
  payloads. The dashboard's server-mode panels, download-tab targets, and
  Prepare-PLE actions are gone.
- Dead `From<ApplyError>` conversion in `src/server_mode.rs`.

### Kept (feed-vocabulary contract)

- `CatalogBackendId` keeps parsing `halogen`/`vllm`/`r9v` (lossless feed
  strings) and `schema_validate::KNOWN_BACKEND_IDS` is unchanged, so the
  vendored cockpit feed parses exactly as before; their entries are
  browse-only (`supported_backend() == None`, `payload == None`).
- Applied migrations (`migrations/0003_toolbox_serving_dimension.sql` is a
  frozen DO-NOT-EDIT artifact) and `assets/cockpit-catalog/` are untouched;
  historical `toolbox_backend` rows stay (text column, no migration).
- Pre-existing gap recorded, not patched: the frozen serving-runtime CHECK
  list omits `gufo`; no gufo serving-runtime rows are written by current
  code.

### Added

- `benchmarks` cargo feature (NOT in `default`). Without it the daemon does
  not compile the explorer routes/lab and does not open the benchmark store;
  `/benchmarks` + `/api/benchmarks/*` answer the same documented
  503 envelope as the unavailable-storage path
  (`"Benchmark explorer unavailable: … Core routing is still available."`).
  The dashboard hides its Benchmarks nav link via a server-injected flag.
  CI (`.github/workflows/ci.yml`, dual matrix) builds and tests both
  configurations.
- Acceptance suite `tests/v2_l1_acceptance.rs` (grep gate, browse-only feed
  parse, 404-on-removed-routes, keeper-route registration, feature
  declaration, degradation contract).

### Upgrade notes (pre-merge checkpoint)

1. `podman ps -a --filter label=io.brainrouter.server_backend` — enumerate
   brainrouter-managed serving containers; `podman ps -a --filter label=io.brainrouter.managed` — enumerate dev-shell toolbox containers too (post-purge they are invisible in the dashboard and keep holding VRAM/ports).
2. Stop and remove any `halogen`/`vllm`/`r9v` server **and dev-shell**
   container (their routes are gone; leftovers keep occupying VRAM/ports
   outside the dashboard).
3. Finish or quit any active Prepare-PLE job (`GET /api/model-downloads`).
4. Deploy. `git revert -m 1 <merge>` is the phase rollback; `v1-classic` is
   the global rollback. No L1 state-file migrations.
