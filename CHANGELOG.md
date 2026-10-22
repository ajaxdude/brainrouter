# Changelog

All notable changes to brainrouter are documented here. Phase tags and per-leg
revert contract per `docs/design/v2-streamlining.md` ("Rollout, migration, and
rollback").

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
  CI builds/tests both configurations.
- Acceptance suite `tests/v2_l1_acceptance.rs` (grep gate, browse-only feed
  parse, 404-on-removed-routes, keeper-route registration, feature
  declaration, degradation contract).

### Upgrade notes (pre-merge checkpoint)

1. `podman ps -a --filter label=io.brainrouter.server_backend` — enumerate
   brainrouter-managed serving containers.
2. Stop and remove any `halogen`/`vllm`/`r9v` server (their routes are gone;
   leftovers keep occupying VRAM/ports outside the dashboard).
3. Finish or quit any active Prepare-PLE job (`GET /api/model-downloads`).
4. Deploy. `git revert -m 1 <merge>` is the phase rollback; `v1-classic` is
   the global rollback. No L1 state-file migrations.
