# v2-streamlining leg L1 (P0-Purge) — implementation log

Branch `v2-p1-purge`, built from design `docs/design/v2-streamlining.md` v3.1
(approved @ `fa6ddf4`) with Ray's acceptance suite at `14b55dc`
(`tests/v2_l1_acceptance.rs`, builder never edited it).

## Design-section mapping

| Design contract | Implementation |
|---|---|
| R1 (P0-Purge): removing halogen/vllm/r9v leaves a daemon that builds, passes tests, routes kept backends identically | `src/toolbox_catalog/{types,models}.rs`, `src/server_mode.rs`, `src/server.rs`, `src/model_downloads.rs`, `src/serving_identity.rs`, `src/escalation/templates/main_dashboard.html` purged; kept backends' code paths untouched (ds4/gufo builders byte-identical; ds4/gufo routes, downloads, dashboard panels unchanged) |
| R1: catalog feed entries for removed backends parse as browse-only | `CatalogBackendId` keeps the `Halogen`/`Vllm`/`R9v` feed-vocabulary variants + string parsing (`types.rs`, grep-gate exception); `SupportedServingBackend` = {LlamaCpp, Ds4, Gufo}, `ALL` length 3; `TryFrom` maps the three ids to `UnsupportedBackend` → `ToolboxDefinition::supported_backend()` = `None`, `parse_payload` returns `None` (entries keep `id`/`name`/`raw`) |
| R2 (Benchmarks gate) | `Cargo.toml` `[features] benchmarks = []`, NOT in `default`. Explorer gated: `benchmark::http::handle_request` + its dispatch/body helpers + `EXPLORER_HTML` + `MAX_INGEST_BYTES` behind `cfg(feature)`; `benchmark_lab` module, `AppState.benchmark_lab`, daemon lab construction, and the `/api/benchmarks/lab` dispatch behind `cfg(feature)`. Core `benchmark` module (store schema, ingestion types, `unavailable_response`) stays compiled: it is the degradation contract + `observability` dependency. OFF build: daemon never opens the store (`benchmark_store = Err("the \`benchmarks\` cargo feature is not compiled into this build …")`), so `/benchmarks` and `/api/benchmarks/*` answer the same documented 503 envelope as the unavailable-storage path (Ray test 6 contract). Dashboard: server injects `window.BENCHMARKS_FEATURE_ENABLED` (token replace in `main_dashboard_html()`); the nav Benchmarks link hides when false |
| L1 Components: `src/server_mode.rs` | halogen/vllm/r9v builders, request types, container-name consts, cache/paths savers, and their 431-token-line unit tests deleted (module census confirmed); module docs rewritten as history note (file is not a grep-gate exception, so it carries no backend names); dead `From<ApplyError>` conversion removed (its only consumers were the purged savers); shared helpers (`clean_engine_args_for_server`, `upgrade_groups_for_podman`, `resolve_toolbox_for_server`, `compute_api_for_runtime_profile`) untouched |
| L1 Components: `src/server.rs` | 12 route registrations deleted (`{halogen,vllm,r9v}/{status,start,stop}` + `vllm/cache-paths` + `r9v/paths` + `model-downloads/r9v/prepare-ple`) → dispatch falls through to the generic 404 JSON `{"error": …}` envelope (Ray test 3 asserts 11 of the 12; `…/prepare-ple` 404 verified code-level by review — the `…/cancel` prefix arm cannot capture it, suffix requirement); handler fns `server_mode_{halogen,vllm,r9v}_*` + `model_downloads_prepare_ple_response` deleted; llama-swap register error message retargeted to ds4/gufo |
| L1 Components: `src/model_downloads.rs` | halogen/r9v `build_download` arms, `ResolvedHalogenBundle`, `R9vPaths`/`effective_r9v_paths`/`r9v_ple_ready`, `ResolvedR9vPackage`, `resolve_r9v_toolbox_image`, `StartPreparePleRequest`, `JobKind::PreparePle`, `DownloadStatus::Running`, `start_prepare_ple`, `execute_prepare_ple`, and `ModelPresence::ple_ready` removed; `JobKind`/`DownloadStatus` wire vocabulary documented as history (single-kind enum kept — serialized job records unchanged); `vllm` special-case in `resolve_catalog_entry` deleted; llama_cpp/ds4/gufo download paths byte-identical |
| L1 Components: `src/serving_identity.rs` | three backend arms removed from `openai_compatible_for_backend`; doc comments retargeted; all vllm/r9v test fixtures retargeted to gufo/ds4/llama_cpp (registry mechanics — exact-match lookup, overwrite, dedup, incompatibility exclusion — preserved) |
| L1 Components: `main_dashboard.html` | halogen/vllm/r9v server-mode panels + cache/paths panels, server-mode state/fetch/render wiring, halogen/vllm/r9v JS function blocks, downloads-tab PLE status (`ple_ready` badge), the r9v-only Verify/Prepare-PLE row actions (+ now-dead `verifyModelChecksums`), backend label/color entries, `SERVER_MODE_CAPABLE_BACKENDS`/`DOWNLOAD_CAPABLE_BACKENDS` lists, and the `running` job-status vocabulary removed; Sankey/cards/other tabs untouched; `scripts/check-html-js.sh` passes (6 blocks after the nav-flag script) |
| L1 Components: fixtures | `src/benchmark.rs` serving-runtime test rows: unrelated-filter `vllm`→`gufo`; missing-id row → gufo (asserts the same Validation gate); bad-snapshot row → ds4 (preserves the Conflict/Database schema mechanics — gufo rows are validation-rejected pre-SQL, so ds4 carries the FK test); `src/cockpit_config.rs` halogen round-trip cases → gufo/ds4; `scripts/test-hf-preflight-ui.cjs` halogen-bundle wiring assertions → ds4/gufo equivalents |
| Contracts kept: `CatalogBackendId` + `schema_validate::KNOWN_BACKEND_IDS` | parsing + the feed-vocabulary constant kept verbatim (`schema_validate.rs` untouched) so the vendored feed parses with no new warnings; `toolbox_backend` historical DB rows untouched (text column, no migration); `migrations/` and `assets/cockpit-catalog/` NOT edited (DO-NOT-EDIT) |
| Contracts kept: `/api/server-mode/{ds4,gufo}/*` | route registrations + handlers + panels unchanged (Ray test 4 keeper assertions) |
| Contracts kept: router.rs zero involvement | `src/router.rs` not modified; `src/inflight.rs`, `src/routing_events.rs`, `src/observability.rs` not modified (doc-comment exception files keep their history notes) |
| Error behavior | removed routes 404 via the existing unknown-route fallthrough (verified by Ray test 3); feature-off benchmark routes answer `benchmark::unavailable_response` (503, "Benchmark explorer unavailable: … Core routing is still available.") |
| Grep gate (L1 Acceptance) | no `halogen|vllm|r9v` (case-insensitive) under `src/`, `scripts/`, `tests/`, `templates/` outside the design's exception paths (`toolbox_catalog/{types,models,schema_validate}.rs`, `toolbox_catalog.rs`, `routing_events.rs`, the acceptance test file itself) |

## Known pre-existing gap recorded (out of v2 patch scope, per design)

The frozen `migrations/0003_toolbox_serving_dimension.sql` serving-runtime CHECK
list omits `gufo`; no serving-runtime rows for gufo are written by current code
(`gufo_serving_runtime_is_rejected_before_sql` documents the validation gate).
Recorded in PRD notes; DO-NOT-EDIT migration artifact — not touched here.

## Benchmarks feature gate — CI matrix

- `cargo check`/`cargo test` default (feature OFF): explorer routes 503-envelope,
  explorer tests compiled out (`#[cfg(all(test, feature = "benchmarks"))]` for
  `src/benchmark/http_tests.rs`), suite green.
- `cargo test --features benchmarks`: full explorer suite green.

## UPGRADE notes (pre-merge checkpoint, design "Build order" §Pre-merge)

Before deploying the purge on any host:

1. Enumerate brainrouter-managed serving containers:
   `podman ps -a --filter label=io.brainrouter.server_backend`
   **and** dev-shell toolbox containers: `podman ps -a --filter label=io.brainrouter.managed`
2. Stop and remove any serving container labeled `…server_backend=halogen|vllm|r9v`
   AND any dev-shell container whose model/backend stem is halogen/vllm/r9v —
   after the purge their routes 404 and the dashboard hides them, so a leftover
   keeps occupying VRAM/ports invisibly (review finding 3: the server_backend
   filter alone misses dev-shell leaks).
3. Finish or quit any active Prepare-PLE download job
   (Dashboard → Downloads, or `GET /api/model-downloads` to list; the
   `prepare-ple` route and job kind are removed; running jobs are lost on
   daemon restart like any download job).
4. Then deploy: `git revert -m 1 <merge>` remains the per-phase rollback,
   `v1-classic` the global rollback; no state file migrations ship in L1.

## Verification results (builder, final tree)

- `cargo test --test v2_l1_acceptance` — 6/6 green (grep gate, browse-only
  feed, 404 on the 11 routes the suite asserts, keeper routes, feature
  declared non-default, degradation contract).
- `cargo test --workspace` (default, features OFF) — green: 390 lib tests +
  all integration targets (477 Rust tests total, includes acceptance).
- `cargo test --workspace --features benchmarks` — green: 407 lib tests
  (explorer/http_tests included) + all integration targets (497 total).
- `cargo check --workspace --all-targets` — 0 errors / 0 warnings both with
  and without `--features benchmarks`.
- `bash scripts/check-html-js.sh` — OK (main dashboard 6 script blocks after
  the flag script; token replacement parses as JS both true/false).
- `node --test scripts/test-*.cjs` — 66/66 green (hf-preflight suite
  retargeted from the halogen-bundle wiring to ds4/gufo equivalents).
- Explorer-dependent daemon suites (`tests/daemon_availability_test.rs`:
  storage-degradation, ingest, native-lab, and the runs-tail of the CSRF
  matrix + their helpers) gated behind `#[cfg(feature = "benchmarks")]`;
  `benchmark_previews_check_csrf_before_storage_availability` stays ungated
  and passes in default builds, proving the degradation envelope from the
  daemon side.

## Step 9 review disposition (Bruce @ dfad853: GO, 0 blocking)

- F1 (important): CHANGELOG claimed a dual-config CI that didn't exist →
  `.github/workflows/ci.yml` added (dual matrix: check+test OFF/ON, catalog
  check, JS gates); claim now true.
- F2 (important): L1 "browse-only caption" display not implemented at L1 —
  entries drop from actionable listings via the pre-existing
  UnsupportedBackend filter; the visible caption is the L4 matrix's scope.
  Design Error-behavior sentence tightened to record this reading; code
  unchanged.
- F3 (important): upgrade checkpoint missed `io.brainrouter.managed`
  dev-shell containers (invisible post-purge, hold VRAM) → checkpoint step
  1–2 extended (above).
- F4 (nit): route-count wording 11 vs 12 clarified (above).
- F5 (nits): block count 5→6 (above); PRD "six backends" → "the feed's six
  backends".
