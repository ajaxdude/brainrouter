# Changelog

All notable changes to brainrouter are documented here. Phase tags and per-leg
revert contract per `docs/design/v2-streamlining.md` ("Rollout, migration, and
rollback").

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
