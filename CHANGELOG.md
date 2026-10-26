# Changelog

All notable changes to brainrouter are documented here. Phase tags and per-leg
revert contract per `docs/design/v2-streamlining.md` ("Rollout, migration, and
rollback").

## v2 L7: llama-swap absorbed — the flip (branch `v2-p3-roles`, pre-merge)

The local serving layer gains a mode switch and the whole `:8081` surface is
retargeted in native mode. `swap:` replaces `llama_swap:` (v2.0-p7, keys
verbatim, `base_url` removed with a note; snapshot + dry-run + restore), and
`swap.mode: native` makes the native supervisor the only local engine. Native
boot REFUSES while llama-swap still reports loaded models (flush-first), the
shim Residency source retires at the flip, and rollback (native → shim) leaves
no children behind. Shim mode is the byte-identical default floor.

### Added

- **`swap:` config section** (`src/config.rs`: `SwapConfig`, `SwapMode`) with
  `mode` (default `shim`), `config_path` (catalog import source; default
  `~/.config/llama-swap/config.yaml`, fedora hosts point at
  `/opt/ai/llama-swap/config.yaml`) and the kept v1 keys. The pre-migration
  `llama_swap:` spelling still parses (serde alias) so frozen v1 fixtures and
  un-migrated files boot.
- **Flip gate** (`daemon::enforce_native_flip_gate`): native boot refuses with
  the shim's loaded-model list; an unreachable shim proceeds. Native boot
  imports the configured source when the catalog is empty.
- **`v2.0-p7` migration** (`migration::run_v2_p7`): section rename, `base_url`
  drop with a "no longer used" note, snapshot + `.bak`, dry-run preview,
  `migration restore v2.0-p7`, idempotent; merges into an existing `swap:`
  section (existing values win).
- **Native-boot catalog import** from `swap.config_path` when the catalog is
  empty (fedora reality; the p4 one-shot may have run against the default path).

### Changed

- **Consumer retargets in native mode** (shim mode untouched): router local
  routing (catalog-only `ensure`; no shim fall-through), model listing
  (`/v1/models`, `/api/models/llama-swap`), inference-status + in-flight PP feed
  (supervisor per-model `/slots`), observability backend snapshot
  (registry/catalog, same state vocabulary), flush (`evict` all + optional
  warm-load), restart endpoint (per-engine stop; respawn on next request),
  GGUF registration (catalog add; the shim file is never written),
  `/api/llama-swap-config` (GET = swap-catalog view, POST refused with a
  pointer), config-files panel row (swap catalog), versions/upgrades rows
  (supervisor row; no shim binary probe or GitHub check), Engines tab
  (mode/catalog-keys/groups badge).
- **Provider literal** stays `"llama-swap"` on routed events through the flip
  and the soak (supervisor routes translate `native` at event emission); the
  deletion pass renames it to `"local"`.
- **Residency**: `shim_source_enabled` in the scheduler config; native mode
  never polls `/running` and lists no shim residents.
- Repo YAMLs (`brainrouter.yaml`, `brainrouter-headless.yaml`,
  `brainrouter.example.yaml`) use the `swap:` section with `mode: shim`.

### Notes

- `base_url` is still honored when present (parse compatibility for fixtures
  and non-default shim ports) but the migration deletes it from operator files.
- install.sh's llama-swap install/service step, the prompt-rewrite text and the
  shim adapter itself remain until the nod-gated deletion pass.

## v2 L6: providers, Manifest absorbed, LAN federation (branch `v2-p3-roles`, pre-merge)

Cloud is brainrouter's own decision. The hardcoded Manifest backend is
replaced by a provider registry (`providers.json`, 0600 atomic) that binds
credentials from the vault, pools enabled non-peer providers for the `cloud`
route, and speaks both OpenAI-compatible and Anthropic wires. v1 configs boot
unchanged: `manifest.*` migrates one-shot into a `manifest`-kind provider
(snapshot + `.bak` + `migration restore v2.0-p6`) and the v1 `cloud` /
`cloud/<id>` behavior is byte-identical (parity corpus replayed against mock
gateways). LAN federation adds a SECOND, opt-in listener with a bearer token,
per-token rate limit, expose double-enforcement, and role-header forwarding —
the distributed-role mechanism.

### Added

- **Provider registry** (`src/providers.rs`): record `{id, kind:
  openai-compatible|anthropic|manifest, base_url, credential_id?, auto,
  models: [{id, alias?, ctx, tools, tier, price?}], lan_peer, enabled,
  monthly_budget_usd?, budget_mode}`. CRUD API (`GET/POST /api/providers*`,
  `PUT|DELETE /api/providers/{id}`, `POST /api/providers/{id}/test`) and CLI
  `provider add|list|remove|test`. Two enabled `auto: true` manifest providers
  is refused at add time (rule (b) uniqueness).
- **Selector resolution matrix** (L6 spec (a)-(d)):
  `cloud` → enabled manifest-auto provider, else the auto pool's top candidate;
  `cloud/<id>` → unique id/alias match, else the single manifest-auto
  passthrough (verbatim v1 parity), multi-match = explicit 4xx naming the
  providers; `cloud/<provider>/<model>` is taken only when the first segment
  names an ENABLED provider (otherwise the whole remainder cascades to rule
  (a)/(b)). A `cloud/` prefix can never resolve to a local llama-swap key.
- **Anthropic outbound adapter** (`src/provider/anthropic.rs`):
  canonical OpenAI request → Messages body (`system` extraction, tools,
  stop_sequences), auth via `x-api-key` + `anthropic-version` (never Bearer),
  SSE translation back to OpenAI chunks (text/thinking/tool deltas,
  `message_stop` → `[DONE]`).
- **Cloud auto pool** (`route=cloud`): enabled non-peer providers' models,
  ctx/tools filtered, tier-fit ranked, deterministic tie-break = provider
  record order then model list order. `lan_peer` models never join the cloud
  pool; `privacy: local-only` treats peers as local-trusted.
- **Per-provider breakers**: `HealthTracker::state()` (`healthy`/`degraded`/
  `open`), keyed by provider id (the migrated Manifest provider keeps the v1
  literal `manifest`). 429 is a backend fault; 3 faults open, 60 s cooldown
  heals.
- **v2.0-p6 migration** (`src/migration.rs`): exact v1 keys
  `manifest.{enabled,base_url,api_key_env}` → provider `{id: "manifest",
  kind: manifest, base_url, credential: env:<api_key_env>, auto: true,
  models: []}`; env-var indirection kept (values never enter the record);
  `manifest.*` YAML removed after `.bak` + snapshot; monthly budget ledger
  created; `serve --dry-run` previews; `brainrouter migration restore v2.0-p6`
  restores config + removes providers/ledger. Idempotent (no manifest block =
  no writes).
- **LAN federation** (`src/lan.rs`, config `lan: {enabled, bind,
  token_credential, expose, rate:{burst,per_sec}}`): a second listener whose
  route table is ONLY `/v1/chat/completions`, `/v1/messages`, `/v1/models`
  (expose-filtered), `/api/peer-info`. Bearer compared constant-time against
  the resolved vault credential (rotation immediate), per-token bucket
  (defaults 20 burst / 4 per second) with 429 on excess, expose
  double-enforcement against the router's dispatch preview — RAW plus every
  dispatchable value (managed `local`⇒fallback_model, `subs`⇒subs_model,
  `auto`⇒chain/pool, `cloud`⇒target+fallback, and post-alias ids), so a
  managed token cannot escape into unexposed spend; admin/dashboard 404 at
  listener level, plaintext-LAN risk logged at boot.
  `peer-info` = `{host_id, version, models:[{key,ctx,tools,tier,state}],
  serving}` read-only.
- **Role-header forwarding**: only `lan_peer` providers receive the request's
  effective `x-brainrouter-role` verbatim (distributed roles: the peer applies
  its OWN role cards). Non-peer providers never see it.
- **Budget guard** (`src/provider_budget.rs`, `provider_budget.json`, 0600):
  per-provider monthly ledger keyed `YYYY-MM` (calendar rollover, survives
  restart), cost = price × usage tokens or the `(bytes+3)/4` estimator flagged
  `estimated`; `warn` records + badges, `stop` refuses as a PRIMARY failure
  (managed fallback fires; review escalates) and excludes the provider from
  the auto pool. Badges and the route event's `budget` field are backfilled
  with POST-accrual spend, and a stream that dies mid-flight reports a breaker
  failure (the header-time breaker cannot see it).
- **Credential env-line export**: `brainrouter credential export --env-line
  <file> [--force]` (0600 atomic, refuse-if-exists) — `store` credentials as
  `NAME=VALUE`, `env`/`file` references as comments, masked lists unchanged.
- **Events (additive)**: `provider` (`{host, endpoint}` hop on non-loopback
  provider routes), and `budget` (`{provider, mode, state, month, spent_usd,
  monthly_budget_usd}`). Sankey ROUTING renders the LAN hop.
- **Dashboard**: Providers section on the Credentials page (add/update, Test,
  Remove, live breaker state, budget badge, declared models) and per-provider
  model lists (grouped, with qualified selector hints) on the Models page.
- **Admission**: `cloud_available` (≥1 enabled non-peer provider) replaces
  `manifest_enabled` everywhere; `AdmittedBackend::Peer` / `BackendDecision::
  Peer` — a qualified `cloud/<lan-peer>/...` review card is peer-admitted with
  cloud-parity strictness and no local memory gate.

### Removed

- `config.manifest` as a required block (the migrated provider owns the URL;
  the struct stays parse-compatible and optional); `AdmissionInputs.
  manifest_enabled`; the Manifest-only router special case
  (`Router.manifest`/`manifest_enabled`); manifest-only breaker naming.

## v2 L5: strixite + hipfire adapters (branch `v2-p3-roles`, pre-merge)

The icon fleets become supervisor-managed engines. strixite (gfx1151) and
hipfire (gfx1100) are L4 `EngineAdapter` implementations: generated confs own
the port, readiness is a `/v1/models` stem assertion, and the cutover is
pid-scoped (`SIGTERM` → pid-gone → settle → `MemAvailable` gate → spawn →
stem poll). Icon-launched servers are now **visible** through a read-only
`/proc` observer and are attach-only until `swap adopt`. Routed streams
(proxy in-flight rows *and* router in-process streams the proxy never sees)
protect their endpoint through an Active-Consumer Set with an idle grace, so a
live session is never mid-swap.

### Added

- **Active-Consumer Set** (`src/active_consumers.rs`): per-endpoint/catalog-id
  spans entered by the router around in-process streams and by proxy rows,
  left on completion with a configurable idle grace
  (`active_consumer_grace_s`, default 120 s, per-host fleet profile). The
  scheduler consults it before every eviction; a protected model is skipped,
  the block is recorded (`budget.json` kind `blocked`, reason
  `blocked:active-consumer`), and `/api/engines/consumers` +
  `/api/engines/events` expose the span. One-action bypass:
  `swap ensure <model> --override` / dashboard **Swap anyway**, audit-logged.
- **strixite adapter** (`src/engine_strixite.rs`): per-model generated confs
  (profile template + supervisor-allocated port, `${PORT}` required,
  `--print-config` validation opt-in), engine-validated launch args (conf-owned
  flags rejected at spec build), port allocation that avoids every Residency
  endpoint, imported explicit `--port` values and llama-swap's dynamic
  `${PORT}` range, `/slots`-equivalent progress when exposed (indeterminate +
  EWMA otherwise), exclusive device class, and `strix_server.service` as the
  takeover unit.
- **hipfire adapter** (`src/engine_hipfire.rs`): card registry (`*.mq4` + `.mtp`
  sidecar, missing sidecar = warning), **version-keyed flag schema** re-derived
  when the probed engine version changes (schema-drift badge; `--mtp` rejected
  by construction), single-owner `:8080` with squat detection (foreign stems ⇒
  Failed + banner, nothing signalled), stop = SIGTERM + port-free gate, context
  math (`max_tokens` defaults to the engine's remaining context; a prompt that
  does not fit is a typed shortfall the proxy surfaces as 400), and
  `hipfire-serve.service` as the takeover unit.
- **External-engine Residency observer** (`src/engine_observer.rs`): read-only
  `/proc` scan for `strix_server`/`hipfire`, endpoint from argv `--port`
  (hipfire default 8080), `/v1/models` stems resolved against the catalog.
  Classification precedence: pids in the native registry/markers are never
  double-listed as external. External residents are Ready-at-external
  (attach-not-spawn), budget-counted, and a cutover never signals them.
- **pid-scoped unit takeover** (`src/engine_systemd.rs`): unit state from
  `systemctl --user show` (`ActiveState`, `MainPID`, `ControlGroup`), pids =
  `MainPID` + control-group members only; `swap takeover <model>` attaches when
  the unit already serves the model, adopts its pid, and otherwise requires the
  single-action `--override` (`blocked:unit-owned` without it). `systemctl` is
  replaceable (`BRAINROUTER_SYSTEMCTL`) and the runtime is injectable.
- **`swap adopt <model> [--pid N]` / `swap adopt --endpoint <url>`**: registers
  an observed icon-launched pid as a supervisor-owned resident with a
  **port-less** adopted spec hash (no permanent `spec_stale` badge); only then
  may a cutover signal it. The endpoint form resolves the model from the
  observer (gguf or served stems) and names the missing catalog entry instead
  of guessing.
- **Unregistered external occupant refusal**: a spawning ensure on an
  exclusive device whose matching external process resolves to no catalog
  entry refuses with `blocked:external-occupant` (503, busy code, ledger
  event) until the operator adopts or stops it; `--override` is the single
  audited bypass. The pid is never signalled.
- **Audit actor**: `budget.json` entries gain an `actor` field (ts = when,
  model = target). API mutations carry `admin@<origin>` (or a sanitized
  CLI-supplied `cli:uid=<uid>`); takeover reports surface the actor.
- **Engine probe wiring**: adapters are probed at daemon registration and once
  at the first ensure; the hipfire schema is derived from
  `<program> --version` at construction, re-derived on every spawn/probe, and
  refreshed (30 s TTL) for the Engines tab, so the version-keyed flag set is
  live rather than aspirational.
- **`min_mem_gib` gate**: after drain + settle the scheduler re-reads
  `MemAvailable` (injectable) against the adapter floor; below the floor the
  model is visible as `Waiting(gate)` with threshold + current value, fails
  after the timeout with a named reason, and `--force-gate` bypasses it
  (audit-logged).
- **APIs**: `GET /api/engines` gains additive `external`, `units`,
  `active_consumers`, `adapter_detail`; model rows gain `gate`, `port`,
  `conf_path`, `active_consumer`. New `GET /api/engines/external`,
  `GET /api/engines/consumers`, `GET /api/engines/events`,
  `POST /api/engines/adopt`, `POST /api/engines/takeover` (loopback + CSRF
  gated like the other engine mutations). Ensure/evict accept `override`,
  `force_gate`, `attach_only`.
- **Engines tab**: external residents with Adopt/Attach, takeover units,
  Active-Consumer spans, `Waiting(gate)` and **Swap anyway** / **Takeover**
  controls (escHtml, additive blocks).
- **CLI**: `swap ensure --override --force-gate --attach-only`,
  `swap evict --override`, `swap takeover --unit --override --force-gate`,
  `swap adopt --pid`.
- **Docs**: `docs/UPGRADE.md` host procedures (strix-halo live strixite pilot,
  fedora1 hipfire unit migration incl. Q4 default user-service takeover and
  rollback), plus `docs/design/v2-streamlining.L5.log.md`.

### Changed

- `EngineAdapter` gains defaulted L5 hooks (`min_mem_gib`, `exclusive_key`,
  `allocate_port`, `takeover_unit`, `conf_path_for`, `describe`); existing
  adapters (`fake`, `llama_cpp`, shim, external) are unchanged.
- `Resident` gains an additive `engine` field (external residents).
- The scheduler's admission order is now exclusive cutover → ACS-aware
  budget eviction → memory gate → spawn, serialized by a per-device lock.
- `swap_catalog::validate_entry` accepts the `strixite`/`hipfire` engines.

### Host tier (owner-nod-gated, not executed by this build)

Live strixite A→B swap on strix-halo and the fedora1 hipfire unit migration
are pilots: procedures in `docs/UPGRADE.md`, no host action taken here.

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
