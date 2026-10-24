# v2 L4 implementation log — swap supervisor + platform probe (L4-v3)

Design: `docs/design/v2-streamlining.L4.md` (L4-v3, design gates closed);
Map: `docs/design/v2-streamlining.md`; charter:
`docs/design/v2-streamlining.charter.md`. Branch `v2-p4-l4`, base
`f3a3d6a` (rebased from `d01be5d` per Main's L3-fix-round notice; the
worktree carried the parent's stable-snapshot migration shape).

## Build order landed

1. **`src/platform.rs`** — machine-id identity; KFD/sysfs GPU detection;
   `MemTotal`/`MemAvailable`; GTT (`ttm.pages_limit` / `amdgpu.gttsize`);
   install markers; vendored + remote/cached feed with a 24 h staleness
   bound; `id = feed platform id` or `generic-<gpu-arch>` with banner;
   `ENGINE_MATRIX` offers + reasons; `budget_gib` (GTT → VRAM → 75 % RAM);
   `current_platform_id()` and the dashboard's `__CURRENT_PLATFORM_ID__`
   render token.
2. **`src/swap_catalog.rs`** — `swap_catalog.json` (0600, atomic), entries
   with resolved `LaunchSpec`, sized pins, groups metadata, settings,
   migration markers, file fingerprints, stream clock, EWMA load; llama-swap
   import (both `models:` shapes) with recursive macro expansion, per-model
   macro shadowing, `${PORT}` preservation, alias/env/group capture,
   idempotent merge, stable dry-run, `launch_spec_hash`, entry validation.
3. **`src/residency.rs`** — shim `/running` observer (every entry; canonical
   GGUF + key through the catalog), native/container inputs, the L5 external
   registry slot, `ready_for` (id/GGUF/served alias), `compose`.
4. **`src/swap_scheduler.rs`** — lifecycle, `ensure` (Residency first →
   queue → budget admission → LRU evict-to-fit → spawn), queue cap/cancel,
   sized pins counted once, `budget.json` append-only ledger, in-flight LRU
   (catalog-id stamp), child-held GGUF flock, adoption markers + boot scan
   (`spec_stale` on hash mismatch, never kill), bounded restart, logs,
   status snapshot, `FakeAdapter` (real child; scripted load/stems/crash/
   progress via `FAKE_*` env).
5. **`src/engine_native.rs`** — `systemd-run --user --scope --quiet
   --collect -p KillMode=control-group` spawn (cgroup escape), `setsid`
   fallback, SIGTERM→grace→KILL + `systemctl --user stop <scope>`,
   `/health` + `/v1/models` health/stem set, indeterminate progress,
   minimal synchronous HTTP client, `build_command_line` for tests.
6. **Router integration** — `install_swap`; catalogued models in
   `try_local_once` resolve through `ensure` (native/external/container →
   direct provider with the shared sanitizer; shim → the untouched
   llama-swap path); `swap_state`/`swap_source` event fields;
   `sanitize_for_llama_family` extracted and shared (golden-body test).
   Uncatalogued selectors/keys keep the v1 path byte-identically
   (no scheduler installed = old behavior).
7. **API + CLI + dashboard** — `/api/platform`, `/api/engines*` (list,
   queue, ensure, evict, cancel, pin, unpin, import preview/apply, logs,
   catalog add/rm), loopback+CSRF gate, `-32005` 503 mapping; CLI
   `swap status|ensure|evict|catalog list|import-llamaswap|pin|unpin|add|rm`
   and `platform show [--json]` (reads direct; mutations refuse when the
   daemon is down); Engines tab (banner, platform/offers, budget gauge,
   lifecycle + actions, queue, import panel) and a Models-tab "serve via"
   column, all escaped.
8. **Migration `v2.0-p4`** — stable snapshot directory (catalog + budget),
   one-shot marker, idempotent import, dry-run preview, `restore_v2_p4`,
   special-cased in `migration::restore` so it never touches config/
   credentials/cards.
9. **Additive keeper changes** — `InflightRow.catalog_id`,
   `InflightHandle::set_catalog_id`/`mark_stream_completed`, registry
   completion hook (LRU clock written on clean EOF), loading/queued rows
   stamped at route time; `RouteEvent.swap_state`/`swap_source`.

## Evidence (final battery)

- `cargo test` (default features, all targets) — **exit 0**: lib 438,
  main 6, anthropic 12, daemon_availability 4, failover 3, install 10,
  review 9, routing_profiles 18, stream 4, subs_routing 10, v2_l1 6,
  v2_l2 14, v2_l2_fixpass 3, v2_l3 21, v2_l3_plugin_decision 7.
- `cargo test --features benchmarks` — **exit 0** (lib 455 + same suites).
- `cargo clippy --all-targets` and `--features benchmarks` — **exit 0**;
  zero warnings in the new modules (the remaining 57 warnings are baseline,
  chiefly `plugins::Inner` visibility; pre-existing `private_interfaces`).
- `plugins/bridge` crate — 26/26.
- Node UI suites (nav 7/7, roles/credentials 7/7, benchmark 38,
  gufo 6, hf-preflight 5, llama-swap-register 3) — green; `check-html-js.sh`
  green across 4 templates.
- Live llama-swap config import (owner's real
  `~/.config/llama-swap/config.yaml`): `swap import --dry-run` imports 12
  models / 2 groups; the macro chain is fully expanded (zero `${ls}` /
  `${common}` left), per-model macro shadowing wins (`ctx=196608` from the
  model's `ctx` override), 57 resolved args on the q8-vision entry, group
  metadata `dirk-dual` (swap=false, exclusive=false) / `non-dual`
  (swap=true, exclusive=true); two dry-runs are byte-identical.
- Isolated live smoke (`HOME`/`XDG_*` temp, port 19100/19101): p4 boot
  import wrote `swap_catalog.json` + `migrations/v2.0-p4` and recorded the
  `pair` group; `platform` resolved `strix-halo`, known, 124 GiB budget,
  offers strixite/gufo/llama.cpp (hipfire reason: APU); `ensure` (fake
  adapter) spawned at :5900 → second ensure attached (`spawned:false`);
  pin 6 GiB counted ONCE (used 6.0, not estimate+pin); evict stopped it;
  dashboard `CURRENT_PLATFORM_ID` rendered `strix-halo`; Engines tab
  rendered platform/offers/lifecycle/queue/budget/pins with working
  action buttons (browser-verified, no page errors); CLI
  `swap status|catalog list|pin|import --dry-run` work, daemon-down
  mutation exits 1, direct status falls back to state files; explicit
  import path outside `$HOME`/state refused.
- Flake notes: (1) the old `start_spawn` pre-check acquired the flock and
  then the adapter acquired it again — removed; (2) the pre-existing
  `stale_uds_is_removed_before_binding` unit test raced a concurrent
  fork's inherited (pre-exec) listener fd under the full parallel suite —
  the test now waits for the kernel to report the socket dead before
  calling `prepare_uds_path`. Six consecutive full-lib runs green.

## Conflicts / risks

- Worktree rebased from `d01be5d` onto `f3a3d6a` before any edit; migration
  P4 builds on the stable-snapshot shape.
- Dashboard diff is one additive block (nav item, view container, switchView
  wiring, JS block, `CURRENT_PLATFORM_ID` token, Models-tab subtitle) for
  clean transplant onto the L3-fix-round HTML.
- Native spawn is untested on hardware here (fixture tier only); the fake
  adapter exercises real child/flock/adoption semantics.
