# Leg L2 implementation log — Plugins + bridge pilot + minimal credentials store

Spec: `docs/design/v2-streamlining.L2.md` (post-review, v4). Build branch
`v2-p2-plugins`, base `64ca696`. Every spec contract mapped to its
implementation site and evidence. Hank's leg rulings (framing = NDJSON,
credential id `discord_token`, manifest/method param shapes, top-level CLI
verbs + snapshot layout, peer-uid code-only, limit code −32005) folded in.

## Component map

| Spec component | Implementation |
|---|---|
| Plugin core `src/plugins/` | `manifest.rs` (plugin.toml: id/name/version/binary/min_daemon_version/grants/config-schema path, grant whitelist), `registry.rs` (persisted state incl. terminal `failed`), `supervisor.rs`, `rpc.rs` (UDS + dispatcher), `store.rs` (paths/atomic 0600/0700, constant-time compare), `mod.rs` (`PluginManager`, install/enable/disable/remove, config/kv, toasts, sockets, shutdown) |
| Supervisor contract | `supervisor.rs`: boot spawn for enabled non-failed plugins; exponential backoff `min(2^(n-1) s, 60 s)`; `CRASH_LIMIT = 5` consecutive crashes → `PluginState::Failed` persisted, supervision stops until `enable`; clean exit → `stopped`, no restart; stop = SIGTERM → 5 s grace → SIGKILL by PID (explicit, never relying on tokio `Child` drop); 60 s uptime resets the counter; spawn-failure counts like a crash (a missing binary is a permanent one) |
| Per-plugin UDS | `mod.rs::bind_plugin_socket` + `store.rs`: `$XDG_RUNTIME_DIR/brainrouter/plugins/<id>.sock`, dir 0700, socket 0600, stale-socket probe+unlink (precedent `prepare_uds_path`, `src/server.rs`); peer uid checked on connect (`rpc.rs`), foreign-uid → close + log |
| `initialize` handshake | `rpc.rs::handle_initialize`: plugin id must own the socket, token constant-time compared against `plugins/<id>/token` (0600, minted at install), manifest `min_daemon_version` compared to `CARGO_PKG_VERSION`; result carries grants + `protocol_version` + daemon version. Auth failure: connection closed, logged (`plugin initialize rejected`), NOT counted as a crash (the supervisor never sees it) |
| RPC v1 surface | `rpc.rs::dispatch` — `chat.completions` (full router path via `Router::route_plugin`, source tag; SSE/plain-JSON aggregation to a final `chat.completion`), `models.list` (`server::model_list_json` = `/v1/models` payload verbatim), `routing.events.tail`, `inflight.list`, `config.get/set` + `kv.get/set` (namespaced files under the plugin state dir, 0600, dotted keys), `secrets.get` (method grant AND credential-grant axis), `notify` (dashboard toast), `set_status` (registry status line); daemon→plugin push `config.changed` (registered live sessions only) |
| Caps | 1 MiB body cap (`MAX_BODY_BYTES`, over-cap line → JSON-RPC error, connection and daemon survive); per-plugin in-flight 4 + queue 16 (`PluginRuntime` semaphores; excess → −32005 naming the cap, Hank #7); unknown method → −32601 |
| Routing events | `src/routing_events.rs::RouteEvent.source: Option<String>` (additive, `skip_serializing_if`); `src/router.rs::route_plugin` → `route_resolved(..., source)`; plugin events carry `source: "plugin <id>"` + `user_agent "plugin <id>"` → Sankey HARNESS bucket shows plugins |
| Credentials store | `src/credentials.rs`: `credentials.json` (0600 atomic temp+rename), kinds `store`/`env`/`file`, `SecretString` (Debug/Display/Serialize mask; plaintext only via `expose()` and the vault file itself), `resolve() -> Result<SecretString>` single implementation, never panics (poisoned lock = error), masked list/show APIs, `drop_grants_for_plugin` |
| Migration | `src/migration.rs` `run_v2_p2` (boot, before validation, `src/daemon.rs`): snapshot-first to `$XDG_STATE_HOME/brainrouter/migrations/v2.0-p2/<UTC ts>/snapshot-config.yaml|snapshot-credentials.json` (0700 tree; the `snapshot-` file names make the snapshot searchable), credential `discord_token` (kind store) granted to `bridge`, YAML key removed with `<config>.yaml.bak`, action logged (value masked). `serve --dry-run` → `preview_v2_p2`, exit 0, nothing written (no store, no plugin dirs, no socket bind). Restore: `migration restore v2.0-p2` (`src/cli.rs::run_migration_standalone`) restores the newest snapshot's files — config YAML (the v1 key returns) + credentials store (pre-migration grants) |
| Bridge plugin | `plugins/bridge/` standalone workspace (own Cargo.lock; root lock contains no `serenity`), binary `brainrouter-bridge`. `src/bridge/{core,persist,mod,discord,signal}` moved as-is (unit tests moved with them: 26 pass). `main.rs` startup: env → UDS `initialize` → `config.get *` + `secrets.get discord_token` (fatal-at-boot on resolve failure: log + `set_status` + nonzero exit) → first-start copy of the six v1 JSON files (originals kept) → v1 `start()` transports → 30 s `set_status` pushes → `config.changed` watcher re-applies transport enable flags (v1 `BridgeManager` pause semantics). Signal keeps its DIRECT llama-swap listing via the live `llama_swap_url` key (`!br list` payload unchanged); LLM path unchanged (omp subprocess → proxy). Config schema = v1 `bridge.*` minus `discord.token` |
| CLI | top-level `brainrouter plugin install/list/enable/disable/remove/status/config set|show` (Hank #4: also `brainrouter cli plugin …`; flags are `global = true` so the bare trailing `--socket/--config` form parses). Thin client over the daemon; daemon-less fallback operates on the plugin state directly. Install copies the dir to the state dir + mints the token; reinstall preserves token + state; remove cleans state, drops credential grants naming the plugin (logged) |
| Core removals | Cargo features `bridge-discord`/`bridge-signal` + serenity/async-trait/serde_yml (no other consumer — verified: only `src/bridge/` referenced them; `cargo tree -e normal` and `Cargo.lock` clean); `src/bridge/` deleted; `bridge.*` YAML keys; `BridgeManager` from `AppState`; daemon bridge startup; `bridges` CLI verb; `GET /api/bridge-status`; `POST /api/bridges/toggle`; dashboard bridge panel |
| Shutdown | `server.rs::run` selects on SIGTERM/Ctrl-C → `PluginManager::shutdown()` (every supervised child SIGTERM→kill by PID) → sockets unlinked → daemon UDS cleaned |
| Plugins tab | `main_dashboard.html`: nav item + `plugins-view`; per-plugin card (id/version/state/pid/status line, grants), Enable/Disable buttons, config JSON view + `key=value`/JSON edit (posts `/api/plugins/config`), log fetch rendered via `textContent` only (untrusted chat text; never `innerHTML` on plugin data), notification toasts. `scripts/check-html-js.sh` green |

## Verification

- `cargo test --lib`: 388 passed (adds credentials/migration/manifest/registry/store/supervisor/framing tests).
- `cargo test --workspace`: all v1 integration suites green (daemon_availability, routing_profiles, subs_routing, anthropic_shim, v2_l1_acceptance, failover, install, review, stream, toolbox) — no regressions from removals.
- `cargo tree -e normal | grep -c serenity` → 0; `grep -c 'name = "serenity"' Cargo.lock` → 0.
- `cargo test --test v2_l2_acceptance` (Ray): see status table in the leg report.
- Node browser-logic suite: 38/38 (dashboard edits included).
- Bridge plugin crate: `cargo test --manifest-path plugins/bridge/Cargo.toml` 26/26 (the moved v1 unit tests = R3 command-matrix coverage: alias resolution, omp NDJSON parsing, chunking, sandbox paths, Signal RPC/pairing).
- Plugin fixture smoke (real daemon, real bridge binary): install → token 0600 → enable → supervisor spawn → `initialize` accepted (grants 11) → `config.get`/`secrets.get` round-trip with the MIGRATED v1 token ("discord token resolved from credential store") → Discord transport start; token absent from plugin state files, daemon log, and the YAML (`.bak` + snapshot keep the pre-migration state); `plugin config set` → config.json + `config.changed` push; daemon SIGTERM → child reaped, plugin sockets unlinked.
- Dry-run: `serve --dry-run` on v1 YAML → exit 0, preview lines, no credentials.json, no plugin dirs, no socket bind.
- Crash loop: fixture plugin reaches terminal `failed` after 5 crashes, spawn count frozen, `plugin status` shows failed, `/health` green throughout.

## Known gaps / handed upstream

- `tests/v2_l2_acceptance.rs` `TestEnv::borrow()` returns an owning `TestEnv`;
  the shadowed boot-1 `Daemon` drop deletes the shared test root under the
  live boot-2 daemon (evidence in the leg report). Harness-side fix owned by
  Ray; the daemon flow passes an exact mirror of the same scenario.
- `chat.completions` streaming to plugins: the RPC surface is
  request/response; SSE deltas are aggregated into the final
  `chat.completion`. Streaming RPC is an additive L-future method.
- macOS peer-cred: Linux `SO_PEERCRED` only (L8 confirm item, per spec).
