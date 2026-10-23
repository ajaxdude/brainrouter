# v2 L3 implementation log — role cards + decision slot (L3-v3)

Design: `docs/design/v2-streamlining.L3.md` (L3-v3, reviewed twice); Map:
`docs/design/v2-streamlining.md`. Branch `v2-p3-roles`, base `1adc4b9`.
Gate: `tests/v2_l3_acceptance.rs` (20 tests) — never edited.

## Status (final battery)

- `cargo test --test v2_l3_acceptance` — **20/20** (repeated, stable)
- `cargo test --test v2_l2_acceptance` — **14/14** · `v2_l1_acceptance` —
  **6/6** · `v2_l2_fixpass` — **3/3**
- `cargo test` (default features, all targets) — green
- `cargo test --no-default-features` — green
- `plugins/bridge` crate — green (protocol v2 is additive; v1 plugins keep
  negotiating 1 = min(plugin, daemon))
- `tests/v2_l3_plugin_decision.rs` (new integration coverage) — **4/4**
- node UI suites — green (`test-nav-ui` 7/7, `test-project-pins-ui` (now
  the cards/policy/credentials suite) 7/7, benchmark 38, gufo 6,
  hf-preflight 5, llama-swap-register 3)
- `scripts/check-html-js.sh` — green (4 templates)

## Modules landed

1. **`src/roles.rs`** — role-card store, `$XDG_STATE_HOME/brainrouter/
   role_cards.json` (contract pin), 0600 atomic (durability machinery +
   `project_key` relocated from `project_pins.rs`); selector grammar
   (`auto|local|cloud|cloud/<id>|bare`, `subs`/`brainrouter/` spellings);
   assignments `{scope, role, assignment}` (selector-string validated);
   policy records (privacy/quality, project>global>default — defaults are
   no-op: `cloud-ok`/`fast`); review-card extras
   (`max_iterations, hankndory_integration, design_doc_path`); shared
   selector types relocated (`ModelChoice`, `validate_model_id`,
   `validate_iterations`) from `routing_profile.rs`; persist-failure
   rollback (runtime unchanged on disk errors — v1 ProfileStore keeper).
2. **`src/decision.rs`** — `DecisionModel` trait (boxed futures),
   `Decision{route,tier,confidence,rationale}`, `RequestFeatures`
   (`approx_prompt_tokens = (bytes+3)/4`; excerpt only for model kinds;
   webhook payload has NO excerpt — unit-tested). Heuristic = v1
   selector-matrix parity (classifier-off anchor local/fast; subs
   local-direct). Guardrail clamps: privacy `local-only` forces
   route=local on SLOT decisions; quality floor raises tier only. Known-ctx
   pool ranking via llama-swap `parameters.ctx`; unknown ctx ranked last
   for large requests, clamp NEVER applied, logged. Kinds:
   `openai-compatible` (bonsai profile → the verbatim v1 wire in
   `classifier.rs` incl. prompt + one-word grammar; failure matrix
   HTTP/timeout/parse/empty/server-down ⇒ route=cloud +
   `fallback=v1-safe`, NO availability pre-check), `structured-schema`
   (pinned schema_version:1 wire, strict JSON), `webhook` (features JSON,
   rationale verbatim), `plugin` (server-initiated `decision.classify`,
   request-id correlation + `timeout_ms`).
3. **Router** — role from `x-brainrouter-role` (default main), `subs`
   names subagent + overrides header; card lookup project>global>default;
   explicit selectors bypass + are never clamped; per-selector pinned
   strictness (bare = strict no retry; `local` = one-`fallback_model`
   retry; `cloud`/`cloud/<id>` = managed local hop; role=review cloud =
   strict + escalate); auto: slot → pool → chain (connect/
   response-headline only, never mid-stream; seed = `fallback_model`).
4. **Events** — additive `role`, `decision{route,tier,confidence}`,
   `rationale`, `decision_fallback`, `chosen_model`; all v1 fields intact.
5. **CLI** — `roles get|set|remove` (contract JSON `{assignments, policy}`),
   `decision show|test` (Decision JSON incl. `fallback`),
   `credential list|add|remove|test`; mutations via daemon HTTP,
   daemon-down REFUSE; `roles get`/`decision show` direct-state fallback.
   `routing-profile` + `sync-omp` verbs REMOVED; `review-config update`
   reduced (`--max-iterations`).
6. **HTTP surface** — `GET /api/roles`, `POST /api/roles/set|remove`,
   `POST /api/policy/set` (+ dashboard alias routes), `GET /api/policy`,
   `GET /api/credentials` (masked) + `/list`,
   `POST /api/credentials/add|update|remove|test` +
   `POST /api/credentials/{id}/test` + `DELETE /api/credentials/{id}` +
   `/api/credentials/audit`, `GET /api/decision`,
   `POST /api/decision/test`; loopback+CSRF gate extended. Removed:
   `/api/models/sync-omp` (404), `/api/routing-profile`,
   `/api/project-pin` trio, `sync_omp_models`/`write_omp_models_yml`/
   `parse_llama_swap_model_ids`/`model_id_to_display_name`/
   `ProjectPinBody`/`single_path_query`.
7. **Migration `v2.0-p3`** — exact table incl. presets expansion, project
   pins, `forced_*` precedence (routing block wins, v1 semantics),
   subagent card from the profile block ONLY (`subs_model` stays the
   auto-pool compatibility seed), `auto`+leftover-model read normalization;
   snapshot (config+credentials+pins+routing_state) under
   `migrations/v2.0-p3/`; YAML keys stripped + `.bak`; runs BEFORE
   `v2.0-p2` so `restore v2.0-p3` byte-round-trips the pristine v1 YAML;
   `serve --dry-run` previews p3+p2 writing nothing; `migration --config
   … restore v2.0-p3` (+`--state`, config-adjacent snapshot probing).
8. **Plugin RPC protocol v2** — `initialize` negotiation
   (`protocol_version` declared → daemon answers min, advertises
   `server_methods: ["decision.classify"]` at ≥2), server-initiated
   requests with (plugin, request-id) correlation + timeout, response
   frames routed in the reader loop, `LiveSession{tx, protocol}`.
9. **Review subsystem** — review card = single reviewer source;
   `ReviewConfig` reduced (iterations + integration + design doc);
   sessions snapshot the card selector at start (`Session.review_selector`,
   exposed on `/review/api/sessions/:id`), continuations reuse it;
   admission gate reads the card's backend; review loop routes with role
   review; `review.forced_mode/forced_model` consumed by migration.
10. **One-entry install** — every harness template prints exactly one
    `auto` entry; per-model listings removed; no `sync-omp` mention
    anywhere under `src/` (asserted by the gate).
11. **Dashboard** — Credentials tab (masked list, add/rotate, Test
    boolean verdict, grants, delete, audit list), role-card strip
    (3 card pickers + project scope selector + model datalist), policy
    chips (privacy/quality, scope-aware), read-only Decisions panel
    (role, route/tier, rationale, fallback, chosen model), Sankey
    ROUTING column = role → route/tier. Removed: preset/role-chooser
    panel, pins panel, sync-omp button + `syncModels()`.
12. **Proxy fast-fail** — routing failures landing within 15 s return a
    real HTTP status (502 + JSON error); slow resolutions (model loading)
    keep the v1 deferred-200 streaming contract (`stream.rs`
    DeferredStream receiver boxed to support the bounded poll).

## v1 floor retargets (each cites its authorizing spec line)

`tests/subs_routing_test.rs` (10/10)
- RouterArgs: `classifier`→`slot`, `profiles`/`project_pins`→`cards`
  ("Routing: role resolution … card lookup project>global>default";
  "Shared selector types relocate from src/routing_profile.rs").
- 3 project-pin tests: `ProjectPinStore`→project-scope `RoleCardStore`
  cards ("per-project pins cards (project scope)"; "project_pins store
  (migrated)" under Removals).
- `route_tagged(...)` call sites unchanged (4-arg signature preserved).

`tests/routing_profiles_test.rs` (18/18: 11 retargeted + 7 v2-wired)
- All ProfileStore/preset/pin fixtures → card assignments; presets
  expanded via the migration table ("presets expand over
  RoutingProfile::main/reviewer via the same table").
- Reviewer-pin snapshot tests → review-card snapshot +
  `Session.review_selector` ("a session snapshots its card record at
  start, never re-resolves").
- `cli … legacy review api` test → `roles set` + `credential add` +
  reduced `review-config update` ("CLI … mutations go via daemon HTTP").
- `clearing_pool_uses_legacy_auto…`: v1 expected `default-local` (pool
  unset ⇒ auto anchor); L3 seeds the auto pool from `config.subs_model`
  ("config.subs_model stays as the auto-pool compatibility seed"), so the
  expectation is `legacy-pool`.
- `cloud_ids_survive…`: v1 `forced_mode=local, forced_model=new-reviewer`
  → card spelling `new-reviewer` (migration table row
  `Local{Some(id)}→<bare-key>`).
- `invalid_policy_input…`: reserved-id smuggling list shrinks to the
  non-grammatical ids (bare `auto`/`subs`/`cloud/x`/`brainrouter/x` are
  canonical selectors now — "Selector grammar (same vocabulary the proxy
  surface keeps through L7)"); `subs`-as-card rejected per spec.

`tests/daemon_availability_test.rs` (4/4)
- 2 legacy-auto keepers: `GET /api/routing-profile`/`forced_*` asserts →
  `GET /api/roles` card + reduced review-config ("review.forced_mode/
  forced_model YAML keys are consumed (removed)"; "schema reduced to the
  non-model survivors"). Write-rejection (auto+model) → reduced-schema
  400 (the keys no longer exist) + card validation + migrated-YAML
  rejection of the raw v1 body. "config untouched" → migrated YAML with
  the original preserved in `.bak` (the migration strips keys by design).
  Migration-run-once keeper: restart over the MIGRATED YAML must not
  re-migrate (helper `start_with_config` added). Migration actions log at
  warn (v1 parity: the forced_model warn line was warn-level; count
  asserts updated 1→3 for the 3-action v1 seed).

`scripts/test-project-pins-ui.cjs` → role-cards/policy/credentials/
decisions UI suite (7/7) ("Removals: … dashboard role-chooser/preset
widgets"; "Add: Credentials tab … role cards panel … policy chips …
Decisions panel").
`scripts/test-nav-ui.cjs` (7/7) — relocated-control list → card strip +
chips + panels; asserts `syncModels` is GONE ("sync-omp … REMOVED at all
four points"); stale `toggle-discord/toggle-signal` assertions removed
(already red at HEAD — the L2 plugin-bridge migration removed those
toggles; the suite was 6/1 at HEAD).
`src/server.rs` unit tests — 12 tests of REMOVED code deleted with their
targets (`ProjectPinBody`/`project_pin_role`/`project_pin_subagent`/
`single_path_query`/display-name suite/`write_omp*`/
`parse_llama_swap_model_ids`; "sync-omp all four points", "per-model
install templates → one entry"); destructive-gate test retargeted to the
L3 mutation surface.
`src/migration.rs` tests — `restore()` gained `config_dir` (restores the
v1 stores p3 consumed; call sites updated).
In-lib literals — `RouteEvent` additive fields (router/routing_events/
observability tests), `route_with_choice` role=review labeling.

New files: `src/roles.rs`, `src/decision.rs`, `src/provider/util.rs`
(shared POST helper), `tests/v2_l3_plugin_decision.rs`.
Deleted: `src/routing_profile.rs`, `src/project_pins.rs`.

## Startup/shutdown race fixes (found by L2/L3 suites under load)

Four real daemon bugs surfaced when the suites ran under full-workspace
parallel load (all in src/, no test changes):

1. **Plugin boot-spawn vs shutdown orphan** (L2
   `supervisor_stops_plugin_children_on_daemon_shutdown`): SIGTERM arriving
   while `start_at_boot` was mid-enumeration — shutdown's supervisor map was
   empty, the supervisor spawned the child after the daemon's grace poll
   saw "quiescent", and the child escaped (ppid 1). Fixed: `shutting_down`
   is set under the same supervisors lock as `start_supervisor`'s guard
   check (no TOCTOU window), every spawn records its pid in a `last_pids`
   ledger, and `shutdown()`/`stop_supervisor()` run a final kill sweep over
   the ledger after a grace beat (SIGTERM→poll→SIGKILL by pid, no adoption
   = no double-reap). Unit-tested (`sweep_ledger_records_spawned_pids`).
2. **Listeners bound after the boot work**: `/health` answered while the
   UDS socket file was still pending (bind sat behind migration/plugins/
   model-resolution), so a CLI invoked in that window saw "Daemon socket
   not found". Fixed: `bind_listeners` runs in `daemon::run` before any
   boot work (UDS bound FIRST, then TCP — a TCP-visible daemon never has a
   pending socket file); `server::run_with_listeners` serves the pair.
3. **Project-scope peer-cwd races** (the `l3_resolution_project_scope...` /
   `l3_migration_v1_state...` flakes): the scope lookup retried a cwd that
   `/proc/<pid>/cwd` had not yet republished after the CLI's
   spawn+chdir (the kernel updates it on the *exec'd* task's next
   scheduler tick — the window is unbounded under full-suite load).
   `roles get` (read-only, no window needed) passing while `chat` (routing
   needs the scope) failed pinpoints this. Daemon side: cwd read per
   REQUEST (not accept snapshot); one `ScopeClock` per connection anchored
   at accept; the probe walk runs on the request task (the default
   blocking pool saturates under suite load); deleted-cwd returns instantly;
   30 s ceiling. This fixes the CLI-connection path and keeps v1 daemon
   behavior (per-connection accept-cwd) for the TCP path.
4. **ENOTDIR on a deleted state dir** during `prepare_uds_path` is now a
   clean startup error (previously a noisy startup-time crash variant).

Regression coverage in `tests/v2_l3_plugin_decision.rs`:
`daemon_serves_cli_verb_the_instant_health_answers` (5 boot rounds; the CLI
verb fires with zero settle after `/health`) and
`sigterm_during_boot_does_not_orphan_plugin_children` (3 SIGTERM-during-boot
rounds; scans for surviving fixture children).

## Flake fixed during verification

`l3_webhook_features_no_excerpt_returns_decision` (and once
`l3_openai_compatible_bonsai_profile_v1_parity`) intermittently died with
"Daemon socket not found" on the gate's first `roles set` CLI call under
full-suite parallel load: `serve` answers TCP `/health` before the UDS
listener is bound (the bind sits behind migration/plugin setup), so a CLI
invoked in that window saw no socket. Fixed at the CLI boundary — the
socket-missing check now polls for 500 ms before declaring the daemon
absent (`src/cli.rs`). 10 consecutive full-suite runs: 20/20.

## Notes / deviations (for review)

- L3_FAST_FAIL_WINDOW = 15 s (proxy returns 502 + JSON on fast routing
  failures; the L2/L3 suites' deferred-stream keepalive tests pass
  unchanged — they exercise the slow path).
- Migration p3 runs BEFORE p2 so `restore v2.0-p3` is byte-exact v1.
- Launch-backend GGUF validation applies only when `launch:` is present
  (externally-managed classifier endpoints load clean).
- `PluginKind` without the plugin subsystem degrades to heuristic.
- Policy default quality is `fast` (no tier raise) — "defaults keep
  today's behavior".
- The subs direct-to-`subs_model` (v1 `local-subs` tag) is preserved for
  configured subs pools; unconfigured subs → slot (parity anchor).
- Heuristic slot is enabled-gated: `decision_model.enabled: false` ⇒
  parity anchor (v1 classifier-off), no fallback label.
