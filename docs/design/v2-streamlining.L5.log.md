# v2 L5 implementation log: strixite + hipfire adapters (L5-v3)

Design: `docs/design/v2-streamlining.L5.md` (L5-v3, design gates closed);
Map: `docs/design/v2-streamlining.md`; charter:
`docs/design/v2-streamlining.charter.md`. Branch `v2-p3-roles`, base
`af1b01f` (L4 integrated: swap supervisor + platform probe). No commit made
(builder contract).

## Build order landed

1. **`src/engine_process.rs`**: one copy of the L5 spawn/stop plumbing:
   `systemd-run --user --scope` argv (cgroup escape, `--unit`, `KillMode=
   control-group`), `setsid` fallback, inherited GGUF flock, SIGTERM →
   grace → SIGKILL, `wait_pid_gone`, atomic 0600 writes, `/v1/models` stem
   probe.
2. **`src/engine_strixite.rs`** (adapter, spec step 1):
   - `StrixiteProfile` from the fleet profile/env (`strix_server`, conf dir,
     port range 5940–5999, `min_mem_gib` 100, device class gfx1151, unit
     `strix_server.service`, `--print-config` opt-in).
   - `render_conf`: `${PORT}`/`${WEIGHTS}`/`${MODEL}` only; unknown
     placeholder or a template without `${PORT}` is a spec-build error.
   - `validate_launch_args`: conf-owned flags (`--config/--weights/--model/
     -m/--port`) and `--mtp` rejected.
   - `PortAllocator`: registry-managed range + reserved (imported explicit
     `--port`, llama-swap dynamic `${PORT}` range from `settings.startPort`)
     + per-spawn avoid set (Residency endpoints), bind probe injectable.
   - Spawn writes the conf atomically, then
     `strix_server --config <conf> --weights <weights>`; health = `/v1/models`
     (stem asserted by the scheduler); progress = `/slots` fraction when the
     build exposes it, indeterminate otherwise; exclusive device class;
     takeover unit exposed.
3. **`src/engine_hipfire.rs`** (adapter, spec step 2):
   - `scan_cards`: `*.mq4` + `.mtp` sidecar; missing sidecar = warning.
   - `HipfireSchema::for_version`: version-keyed flag set (v1 adds
     `--flash-attn`); `validate_args` rejects unknown flags and `--mtp` by
     construction; `refresh_schema` re-derives on a probed version change and
     sets the drift badge; `set_schema_version` is the tool/fixture seam.
   - `squat_reason`: foreign stem set at the endpoint ⇒ named Failed reason;
     pre-spawn probe + bind check, never signals the squatter.
   - Stop = SIGTERM + port-free gate (`/health` must fail); single-owner
     `:8080` (profile/env override) via `allocate_port`; `min_mem_gib` 16;
     unit `hipfire-serve.service`.
   - Context math: `remaining_context`, `default_max_tokens`,
     `shape_max_tokens` (absent ⇒ remaining; overshoot clamps; no room ⇒
     typed `ContextShortfall`).
4. **`src/engine_systemd.rs`** (takeover cutover plumbing, spec step 3):
   `Systemctl` boundary (`BRAINROUTER_SYSTEMCTL`), `parse_show` of
   `ActiveState/MainPID/ControlGroup`, pids = `MainPID` + control-group
   members (pid-scoped, never a name pattern), `UnitRuntime` trait with the
   real CLI impl and a scripted `MockUnitRuntime`.
5. **`src/engine_observer.rs`** (external Residency observer, spec step 3):
   `/proc` pid source seam; classify `strix_server`/`hipfire` with argv
   `--port`/`--weights`/`--model`; marker-pid precedence (native never
   double-listed); catalog id from weights path or served stems; emits
   `Resident { source: external, engine }` + `ObservedEngine` detail rows.
6. **`src/active_consumers.rs`** (ACS, spec step 4): span registry keyed by
   catalog id and endpoint; `enter`/`leave`/`note_completion`; `protected`
   and `span_remaining`; `snapshot` for the Engines tab; `ActiveConsumerStream`
   guard releases on drop (EOF/error/abort). Router wiring: `try_local_once`
   enters the span **before** the upstream call for a catalogued native
   route (review-loop streams create no proxy row) and releases on error or
   stream drop; the proxy completion hook opens the grace window.
7. **`src/swap_scheduler.rs`** (rails, spec steps 3–4):
   - `EngineAdapter` defaulted hooks: `min_mem_gib`, `exclusive_key`,
     `allocate_port`, `takeover_unit`, `conf_path_for`, `describe`.
   - Admission order: Residency attach → per-device lock → (would spawn)
     exclusive cutover → ACS-aware budget eviction → `min_mem_gib` gate →
     spawn → stem assert.
   - Cutover: predecessors = Ready models sharing the device class;
     SIGTERM-point ACS re-check with one bounded wait
     (`BRAINROUTER_CUTOVER_RECHECK_MS`) ⇒ `blocked:active-consumer`;
     `--override` bypass (ledger `override`); `SIGTERM` only runtime handles
     (supervisor-spawned or adopted markers); pid-gone wait; injectable
     settle (default 2 s).
   - Gate: injectable `MemAvailable` (`BRAINROUTER_MEM_AVAILABLE_GIB` or
     `with_mem_source`); below floor ⇒ `Waiting(gate)` with threshold +
     current value, blocked event, timeout ⇒ `Failed` +
     `blocked:min-mem-gate`; `--force-gate` audit row.
   - Eviction: protected candidates skipped and recorded once
     (`kind: blocked`, reason `blocked:active-consumer: …`); explicit evict
     refuses a protected model unless overridden; override audit
     (`kind: override`).
   - `takeover(model, opts)`: attach-not-spawn when resident; unit active →
     attach/adopt when it serves the model, else `blocked:unit-owned`
     without `--override`, else pid-scoped SIGTERM + `systemctl --user stop`
     + normal admission.
   - `adopt_external`: marker with `adopted_spec_hash` (port-less) so adopted
     residents never wear `spec_stale`; `scan_adoption` accepts either hash.
   - Crash restart (maintenance) is serialized like an admission: per-device
     lock then the model spawn lock, state re-read under both, so a cutover
     that drained the model while the health probe ran cannot be resurrected
     (co-residency on an exclusive device). The Loading-recovery write is
     conditioned on still being Loading for the same reason.
   - `refresh_external` feeds the external Residency slot + detail rows;
     `status()` carries external/units/ACS/adapters/gate/port/conf.
   - Avoided-port set for generic allocation = catalog explicit + dynamic
     range + Residency endpoints.
8. **`src/fleet_profile.rs`** (spec step 4 fleet profiles): per-host engines
   from the platform offers (`strix-halo`/`fedora2` → strixite; fedora1 /
   unknown gfx11 → hipfire; `BRAINROUTER_FLEET_ENGINES` override), grace,
   gate timing, settle/re-check, unit names.
9. **API + CLI + dashboard** (spec step 5): see CHANGELOG. All engine
   mutations remain loopback + CSRF gated; GETs are additive; existing L4
   keys (`adapters` = name list) unchanged.
10. **Docs** (spec step 7): `docs/UPGRADE.md` host procedures + rollback,
    CHANGELOG L5 entry, this log.

## Fixture-tier evidence

`tests/v2_l5_smoke.rs` (builder smoke, 6 rows, real `FakeAdapter` children):
takeover attach-not-spawn from an observed external (spawn count 0) + adopt
marker; `blocked:min-mem-gate` refusal + `--force-gate` admission; in-process
ACS live pin blocks evict-to-fit with the `blocked:active-consumer` ledger row
and the override bypass; grace span bridges then expires; exclusive cutover
drain-before-spawn with no co-residency; SIGTERM-point re-check refuses a
newly protected predecessor without killing it. Unit tier: conf rendering and
unknown-var rejection, conf-owned flag rejection, port allocator
reserved/avoid/busy, llama-swap dynamic-range avoidance, card/sidecar scan,
version-keyed schema drift + `--mtp` rejection, context math, squat
classification, `systemctl show` parsing, marker-precedence observation,
catalog-id resolution, ACS nesting/grace/snapshot.

## Deliberate boundaries

- Adapter `probe()` is not called on every ensure (L4 behavior); a missing
  `strix_server` surfaces as a spawn failure with the program name, and
  `platform show`/the Engines tab offer matrix carry the install marker.
- `/slots`-equivalent progress for strixite is best-effort (builds that do not
  expose it stay indeterminate + EWMA), matching the spec's
  "readiness poll only" clause.
- The external observer runs in the existing 2 s maintenance loop (read-only
  `/proc` + loopback probe); no new poller.
- Host-tier pilots (live strixite A→B swap, fedora1 unit migration) are
  owner-nod-gated and NOT executed by this build; procedures: `docs/UPGRADE.md`.

## Findings round (acceptance probe, commit d309bed)

1. **hipfire version-keyed schema was dead code**: `HipfireAdapter::new` now
   derives the schema from `<program> --version` at construction;
   `probe()`/`spawn()` re-derive (an upgrade cannot poison a stored schema),
   and `describe()` re-probes on a 30 s TTL for the drift badge. The daemon
   probes every registered adapter at boot and the scheduler probes each
   adapter once at the first spawn (probe failure logged, never fatal).
2. **Audit actor**: `BudgetEntry.actor` (ts = when, model = target). API
   mutations send `admin@<origin>` unless the CLI supplies `cli:uid=<uid>`;
   actor is recorded on override / force-gate / blocked / adopt / takeover
   entries and surfaced in `TakeoverReport`.
3. **Unregistered external occupant**: `blocked:external-occupant` (503 +
   busy code + ledger event) when a spawning ensure on an exclusive device
   finds a matching engine process that resolves to **no catalog entry** and
   is not adopted. `--override` is the audited bypass; the pid is never
   signalled. (An external that resolves to a catalog entry stays a known
   resident and does not block: ensure attaches when it is the target, and the
   operator adopts or stops it through the catalog vocabulary. This keeps the
   acceptance row `l5_external_observer_ready_at_endpoint_attach_adopt_and_ownership`
   semantics intact.)
4. **Dashboard adopt**: the Adopt button carries the observer endpoint when the
   model could not be resolved; the server accepts `{model}` or `{endpoint}`
   (with `{pid}`) and returns a named error when the endpoint has no catalog
   entry. Attach is only offered for resolved rows.

## Verification run by the builder

- `cargo check --all-targets` and `--features benchmarks`: clean.
- `cargo clippy --all-targets` (both feature modes): the new/changed L5 files
  introduce zero warnings; the repo's pre-existing warnings are unchanged.
- `cargo test --lib`: 464 passed, 0 failed (repeated 3x for the pre-existing
  migration-restore parallel race, see below).
- `cargo test --test v2_l5_acceptance` (Ray, 11 rows): 11 passed.
- `cargo test --test v2_l5_smoke`: 9 passed (6 fixture rows + occupant
  refusal/override, adopt-by-endpoint, actor audit).
- `cargo test` pre-existing suites: L1 4/4, L2 14/14 + fixpass 3/3,
  L3 plugin-decision 6/6, L4 17/17, daemon_availability, stream,
  anthropic_shim, failover, install, routing_profiles, subs_routing: green.
  Node browser-logic: 38/38. `scripts/check-html-js.sh`: all templates OK.
- Live daemon smokes (isolated HOME/XDG, real child processes):
  1. external observer sees a fake `strix_server`, `POST /api/engines/takeover`
     attaches (`spawned: false`), `POST /api/engines/adopt` writes the marker,
     `ensure` attaches, `/api/engines/events` shows the adopt row, dashboard
     renders the L5 blocks;
  2. `min_mem` gate: ensure returns 503 `blocked:min-mem-gate`, the model row
     fails with threshold/current in the reason, ledger has `blocked` +
     `gate-fail`; a `force_gate` ensure spawns the real generated conf on
     port 15960, stem-asserts, and reports `ready` with `port` + `conf_path`;
  3. ACS: a live SSE stream through the proxy (in-flight row stamped with the
     catalog id) makes a same-device ensure return 503
     `blocked:active-consumer` while the streaming model stays `ready`; after
     the stream ends the cutover logs `SIGTERM` → `pid gone`, evicts to
     `stopped`, and the second model becomes `ready`.

### Known pre-existing failure (not caused by L5)

`v2_l3_acceptance::l3_migration_no_routing_review_survivors_restore_dryrun`
fails on this machine because its restore CLI invocation does not carry the
test environment, so it resolves the real
`~/.local/state/brainrouter/migrations/v2.0-p3` snapshot (the owner's headless
YAML) and restores that over the fixture config. The test's own daemon run is
isolated; only the restore subprocess is not. Unrelated to L5 (no L5 file is
touched by that path).
