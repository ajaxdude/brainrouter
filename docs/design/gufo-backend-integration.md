# Gufo — 6th inference backend integration

> Source-of-truth design document (HankNDory). Authored in plan mode in the session
> folder; on approval it is committed to the repo at `docs/design/gufo-backend-integration.md`
> (its permanent home) as the first implementation step.

## Status
- Workflow state: `complete`. **All Dory gates cleared** (comprehension/clarity PASS; critic 4 isolated rounds → blocking 3→4→1→0; readiness READY), **approved**, and **Step 9 mean review → no issues**. Pre-code PC-1/PC-2 PASSED on strix (D10/D15). Local gate GREEN (`cargo test --test-threads=1`, `cargo clippy` 0 errors / no new warnings, `check-html-js.sh`, `test-gufo-ui.cjs` 6/6). **Deployed to strix master `40757b7` + GitHub; T-7 live-verified (see Dory validation record).**
- Change classification: **standard**, medium/high blast radius. New public serving backend across the catalog type system, catalog validation, podman server-mode lifecycle, `hf download` registry, serving-identity, HTTP API, benchmark-ingest guard, and dashboard UI. Full HankNDory method.
- Human review: requested after readiness passes. Reviewer = creator session `Brainrouter dory review` (`1c5cd7ef-0b4a-4672-8d5a-872e380a58a7`).
- Revisions:
  - v1 — 2026-09-24 — initial draft.
  - v2 — 2026-09-24 — Dory round-1 fixes (exact pins/sizes; single require-draft contract; server-only create honesty; overlay validation; serve-plan invariant; benchmark guard; corrected blast-radius; dashboard status wiring; server-stopping rollback; dflash2-only; typed defaults).
  - v3 — 2026-09-24 — Dory round-2 + readiness fixes: `ResolvedGufoModel { models_dir, plan }` (B1); transactional + fully-validated + fail-closed `load_effective_typed_catalog()` returning `Result`, `schema_validate::validate_catalog` on the merged whole, `gufo`∈`KNOWN_BACKEND_IDS` (B2); pinned gufo source commit `9cad13974cf6da0cd3674b4e0a88b14b7e4a2908` + `:latest`/`--pull=always` + **pre-code `gufo serve --help` probe** (B3); server-only toolbox decision + concrete pull-only contingency + **pre-code halogen-flash create check** (B4); `e.payload.*` dropdown paths (I1); pure `plan_from_entry` helper (I2); guard in `ServingRuntimeDefinition::validate` (I3); provenance note (N1); closed `GufoSpeculativeMode` enum (N2). **Appendix A** (exact Rust types/signatures) + **Appendix B** (full overlay JSON) added.
  - v4 — 2026-09-24 — Dory round-3 fixes: PC-2 now probes the **actual gufo image**'s `toolbox create` support (not halogen's) with a fully-specced pull-only fallback (B1); the compiler-enforced exhaustive matches (`parse_payload`, `openai_compatible_for_backend`) land atomically with the enum variant (I2); `StartGufoServerRequest` derives `Deserialize` (I3); normative `validate_merged` invariant list + tests (I4); AR (`speculative:None`) dashboard fixture (I5); `docs/SERVER.md`/`docs/CLI.md` paths (N6).
  - v5 — 2026-09-24 — Dory round-4 (no-blocking) hardening: pull-only response contract (`image_presence` map keyed by `toolbox_id` + server-side rejection of `toolbox_compatible:false` in `POST /api/toolbox-containers`); `validate_merged` now also requires the gufo section/profile/toolbox/default to exist exactly once, ≥1 `role=main`, and non-empty `id/name/repo/revision`; **any** duplicate gufo id is an error. Critic gate cleared (no blocking); readiness READY.

## Problem
brainrouter is a native Rust web UI over ai-toolbox-cockpit's local inference backends; today it wires five serving backends (`llama_cpp`, `ds4`, `halogen`, `vllm`, `r9v`). The user runs brainrouter on an AMD Strix Halo box (gfx1151, 128 GiB unified memory) and wants **gufo** — a vertical Strix-Halo inference engine (github.com/gufo-org/gufo) shipped as the podman image `ghcr.io/gufo-org/toolboxes/gufo-runtime:latest` — as a first-class 6th backend. gufo runs an OpenAI-compatible server (`gufo serve … llm`) with speculative decoding (Qwen3.8-27B + a DFlash2 draft). Because gufo is **not** in the upstream cockpit catalog, brainrouter must add it to its type system, inject a validated brainrouter-owned catalog overlay, add podman start/stop/status + `hf download` support, and make it appear and be fully interactive in the dashboard exactly like ds4/halogen/vllm/r9v.

## Goals and non-goals
**Goals (must-have):**
1. `gufo` recognized in the type system (`SupportedServingBackend::Gufo`, `CatalogBackendId::Gufo`), losslessly round-tripping.
2. A **validated brainrouter-owned gufo catalog overlay** injecting a gufo toolbox, runtime profile, `strix-halo` attachment, and model entries — merged into the same catalog every consumer reads, validated as a merged whole with the **full** structural validator + gufo semantics, **fail-closed**, vendored files byte-for-byte unchanged.
3. Server Mode: detached `podman run … gufo serve … llm` start/stop/status mirroring ds4/halogen, with Strix-Halo device flags + `--speculative dflash2 --dflash-model` handling, `LABEL_SERVER_MODEL`, serving-identity; `openai_compatible_for_backend(Gufo)==true`.
4. Downloads: `hf download` for gufo main + draft GGUFs via the existing registry + `hf` preflight.
5. Dashboard first-class citizenship: gufo appears + is interactive **everywhere** the others are — a tab in **both** Downloads and Server Mode, a `sm-gufo-*` panel (populated dropdowns, ctx/host/port, Start/Stop, live status, empty-model hint, `hf` banner), gufo model rows with Download (incl. the DFlash2 draft), and a listed gufo toolbox in the toolboxes panel, grouped into `strix-halo`. Visually indistinguishable from ds4/r9v.
6. Tests: a UI test (mirroring `scripts/test-hf-preflight-ui.cjs`) with static + behavioral fixtures; Rust unit + integration tests for the overlay+validation, conversions, download+presence, serve plan/command, and the benchmark guard.

**Non-goals (v1):** routing to a running gufo server (parity: bookkeeping only, `Router` untouched); gufo benchmarking (guarded out — DB CHECK permits only 5 backends); host/platform enforcement (brainrouter has no host detection today; v1 is a Strix deployment build); non-LLM gufo modalities + vision (`--mmproj`); auto-start/swap/idle-unload; editing the vendored catalog; speculative modes other than `dflash2`.

**Future:** DeepSeek-V4-Flash (DSpark) + Qwen3.8-Flash-Next (MTP, sharded) models; gufo vision; gufo as a routable upstream (Feature B); gufo benchmarks (needs a CHECK-widening migration + effective-catalog snapshot); pinning the image to an OCI digest.

## Current system
Verified against the cited files.

### Types — `src/toolbox_catalog/types.rs`
`CatalogBackendId` (33, open, `Other` fallback, `as_str` 50/`from_str` 67, hand-written serde). `SupportedServingBackend` (109, closed; `as_str` 118; `ALL:[_;5]` 130; `TryFrom<&CatalogBackendId>` 163; `From` 180; hand-written `Deserialize` 211). `ToolboxDefinition` (~270, `supported_backend()` 316). `RuntimeProfile` `{id, engine_args}` — **`id` comes from the `runtime_profiles` map key, not the value** (`types.rs:379-397`). `Platform` `{id,name,description,toolbox_ids,defaults}`. `ToolboxCatalog` (~354) `{schema_version, runtime_profiles: BTreeMap, toolboxes: Vec, platforms: Vec}` + `toolbox_by_id`/`platform_by_id`/`platform_id_for_toolbox`.

### Model catalog — `src/toolbox_catalog/models.rs`
`CatalogModelFile` (49) `{path, size_bytes, role: Option, sha256: Option}`. `ModelPayload` (`#[serde(tag="backend")]`, ~158). **`ModelCatalog::parse` (219-269) selects the backend from the `backends` map KEY; `parse_payload` (304-326) picks the variant from that key and deserializes the raw entry — entries carry no `backend` field** (verified). The typed API entry is `CatalogModelEntry { id, name, backend, payload, raw }` (172-183) — **per-backend fields live under `payload`**.

### Vendored catalog + validation — `src/toolbox_catalog.rs`, `schema_validate.rs`
`VENDORED_*` (37-39). `load_vendored_catalog()` (153-187) → parse to `Value`, `schema_validate::validate_catalog`, `VendoredCatalog { toolboxes_json: Option<Value>, models_json: Option<Value>, report }`. **`VendoredCatalog::typed()` (129-144) ignores `report`.** `vendored_catalog_revision()` (41-57) hashes only the two vendored constants; the create path records it as the `io.brainrouter.catalog_revision` label for **every** toolbox (`server.rs:2693-2699`). **No merge/overlay exists.**
`schema_validate.rs`: `validate_catalog(&Value,&Value) -> ValidationReport` (478) = `validate_toolboxes_json` (129) + `validate_models_json` (386). Checks: required strings, unique ids/container_names, valid `runtime_profile`/platform refs, closed channel/maturity/feature enums, **`storage.config_key` AND `storage.default` required** (419-430), duplicate model ids (457), platform `defaults` backend-agreement (343-350), unassigned-toolbox (360). `KNOWN_BACKEND_IDS = [llama_cpp,ds4,halogen,vllm,r9v,comfyui]` (30) — unknown id ⇒ **warning** (167-245). Run only on the vendored feed (daemon startup + sync tool).

### Runtime read chokepoints
`server.rs:2405` `load_typed_toolbox_catalog()` (→ `/api/toolbox-catalog` 2564, `/api/toolbox-containers` 2610, create/update/adopt 2819+; returns `catalog.platforms` wholesale; **warns-and-returns on report errors, 2405-2417**). `server.rs:2586` `/api/toolbox-models`. `server_mode.rs:353` `load_typed_toolbox_catalog()` (→ ds4/halogen/vllm/r9v resolvers via `resolve_toolbox_for_server` 372) + vllm direct read (~1047). `model_downloads.rs:275` `resolve_catalog_entry`, `:580` `local_presence_snapshot` (**`Err(_) => continue` swallows `build_download` failures**), `:820` r9v image. `daemon.rs:~197` startup validation (log-only, vendored). Three tests iterate `SupportedServingBackend::ALL` over the vendored catalog: `toolbox_catalog.rs:211`, `models.rs:344`, `types.rs:~529`.

### Server mode — `src/server_mode.rs`
ds4 = single-model precedent. `build_ds4_server_command` (288): `run -d --name … <upgrade_groups_for_podman(clean_engine_args_for_server(profile.engine_args))> --ipc=host --cap-add SYS_PTRACE --env DS4_ROCM_* --security-opt label=disable --userns=keep-id --label … -p host:port:port -v <models_dir>:/models:ro <image> ds4-server -m /models/<file> --ctx --host 0.0.0.0 --port`. **`ResolvedDs4Model` carries `models_dir` from the resolver into the builder** (`model_downloads.rs:625-654` → `server_mode.rs:288-332`); `effective_models_dir` is **private** to `model_downloads.rs` (256-269). Device flags come from `runtime_profile.engine_args`. `clean_engine_args_for_server` (204), `upgrade_groups_for_podman` (232), `compute_api_for_runtime_profile` (416, `/dev/kfd`→rocm). vLLM `serve` precedent (~1220) `<image> vllm serve <repo> …` — **no backend passes a second model file on the serve line**. `container_server_status` (549), `container_running` (532); start = `podman rm -f` then `run` (idempotent), stop = graceful stop + `rm -f`. `resolve_toolbox_for_server` accepts `features.server != Unavailable`.

### Serving identity, downloads, create, benchmark, dashboard
- `serving_identity.rs`: `openai_compatible_for_backend` (LlamaCpp+Vllm true; others false); `register_serving_identity` (`server.rs:3548`) at start, deregister at stop; **no routing**.
- `model_downloads.rs`: `build_download` (429; arms end with **`_ => Err`** — a new enum variant compiles without an arm); `build_multifile_download` (501) `hf download <repo> <file…> --revision <rev> --local-dir <dir>` (one repo+revision); `execute` (1062) one child; `check_completeness` (536) all files present + **exact `expected_size_bytes`**; `effective_models_dir` (263) config→`storage.default`→`~/models`, flat; `resolve_downloaded_ds4_model` (625) gates `Complete` and returns `{models_dir, filename}`.
- `server.rs`: `create_toolbox_container` (2819) → `recreate_toolbox_container(pull=true)` (2795) → `pull_toolbox_image` then **`toolbox_create` runs `toolbox create --image <image> <container>` UNCONDITIONALLY** (2723-2744), ignoring `toolbox_compatible`; records a sidecar ownership entry. Generic UI renders `+ create` for any missing toolbox (`main_dashboard.html:3137-3151`).
- `benchmark.rs`: `ServingRuntimeDefinition` (440) `toolbox_backend: SupportedServingBackend` (442); **`ServingRuntimeDefinition::validate` (451-467)** invoked before ingest (1717-1724, 2452-2455); SQL `INSERT OR IGNORE INTO serving_runtimes(…toolbox_backend…)` using `.as_str()` (2611-2628). `migrations/0003_toolbox_serving_dimension.sql:30-39`: `CHECK(toolbox_backend IN ('llama_cpp','ds4','halogen','vllm','r9v'))` — closed.
- `main_dashboard.html`: `TOOLBOX_BACKEND_LABELS` (3047); `DOWNLOAD_CAPABLE_BACKENDS` (3193); `SERVER_MODE_CAPABLE_BACKENDS` (3383); `CURRENT_PLATFORM_ID='strix-halo'` (3064, hardcoded, no host detection 3055-3062). Per Server-Mode backend: a status var (3384-3388), `sm-panel-<b>`+`sm-<b>-*` (ds4 630, halogen 674), a `Promise.all` fetch of `/api/server-mode/<b>/status` (3421) with destructure+assign, a show/hide line in `setServerModeBackend` (3400-3412), render fns (3442-3453). `renderToolboxModelRow` (3248, generic, reads `entry.payload`); toolboxes `renderToolboxes`/`renderToolboxRow` generic. `scripts/test-hf-preflight-ui.cjs` runs marker-delimited helper snippets in a Node `vm` + static asserts.

### gufo facts (verified — github.com/gufo-org/gufo @ pinned commit + HF API)
- **Gufo source pin:** `9cad13974cf6da0cd3674b4e0a88b14b7e4a2908` (`main`, 2026-09-24). CLI/endpoint contract from that commit's top-level `README.md`, `docs/models/qwen3.8-27b/README.md`, `docs/SERVER.md`, `docs/CLI.md`.
- Image `ghcr.io/gufo-org/toolboxes/gufo-runtime:latest`. Run: `podman run --rm --userns=keep-id:uid=1000,gid=1000 --device /dev/kfd --device /dev/dri --group-add keep-groups --ulimit memlock=-1 -p 8080:8080 -v ./models:/models:ro <image> gufo serve --host 0.0.0.0 --port 8080 llm --model /models/<main> --speculative dflash2 --dflash-model /models/<draft>`. OpenAI `/v1/chat/completions`,`/v1/completions`,`/v1/responses`,`/v1/models`,`/health`. ROCm gfx1151, Linux x86-64.
- **v1 model — Qwen3.8-27B**, pinned exactly (HF tree API):
  - Main `unsloth/Qwen3.8-27B-GGUF`@`4ca720788d1e01f1bff70c033e0d0028fd02e502` (gufo README pin): `Qwen3.8-27B-UD-Q4_K_XL.gguf` = **17559178144** B; `Qwen3.8-27B-UD-Q8_K_XL.gguf` = **31457991680** B.
  - Draft `z-lab/Qwen3.8-27B-DFlash2-GGUF`@`2d9571f8ce46e151f61c6499c99dee6079e1d610` (gufo README pin): `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` = **1143006816** B.
  - Serve: `--speculative dflash2 --dflash-model <draft> --context 32768 --sessions 2`.
- **Image reproducibility (B3):** the overlay uses `:latest`; the gufo server run command includes `--pull=always` (matching halogen-strix-halo's own `--pull=always` policy, `assets/cockpit-catalog/toolboxes.json:76-91`). The exact pulled digest is recorded in the verification notes at deploy. Pinning an OCI digest is future hardening.

## Requirements and acceptance criteria
- **R1 — Types.** gufo ↔ both enums, round-trips; other unknown ids still `Other`. (DI-1, T-1)
- **R2 — Validated overlay, fail-closed.** gufo toolbox + runtime profile + `strix-halo` attachment + model section merge into the effective catalog every consumer reads; the **merged whole** passes `schema_validate::validate_catalog` **plus** gufo semantics; a merge/validation failure yields a checked `Err` so consumers **fail closed** (never expose partial/invalid data) and never a silent partial merge; vendored files unchanged. AC: `/api/toolbox-catalog`+`/api/toolbox-models` list gufo; `git diff assets/cockpit-catalog/` empty; T-2. (DI-2/DI-3)
- **R3 — Server Mode + serving contract.** `POST /api/server-mode/gufo/start|stop`, `GET …/status`. A gufo model whose entry declares a DFlash2 draft is **servable only when both main and draft are downloaded** (dropdown lists it only then; start otherwise returns a specific validation error naming the draft); a `speculative: None` model serves AR. Start builds the detached container with correct flags + `--speculative dflash2 --dflash-model` iff the plan is `Dflash2`; labels + serving-identity; `openai_compatible_for_backend(Gufo)==true`. (DI-5/DI-6/DI-7, T-3, T-7)
- **R4 — Downloads + presence.** gufo main + draft download (exact pinned revision + exact-size completeness) via the existing registry + preflight; `build_download(Gufo,…)` returns a real command (not the `_ => Err`); `local_presence_snapshot` + `/api/model-downloads/status` include gufo. (DI-4/DI-6, T-4)
- **R5 — First-class UI.** gufo tab in **both** views; `sm-gufo-*` panel with populated dropdowns (draft-aware), ctx/host/port, Start/Stop, **live status**, empty-model hint, `hf` banner; gufo model rows (main + draft) Downloadable; gufo toolbox listed with the same create/pull affordance as halogen-flash; strix grouping includes gufo. (DI-7, T-5, T-7)
- **R6 — No regression.** Existing backends, catalog loading, the vendored-feed contract, and benchmark ingest unchanged; vendored files byte-for-byte; gufo cannot enter `serving_runtimes`. (T-6, T-8, DI-8)

## Technical plan
Eight slices (each mirrors a proven pattern): DI-1 types; DI-2 overlay JSON (Appendix B); DI-3 transactional+validated+fail-closed effective catalog; DI-4 `GufoModel`+payload; DI-5 server mode; DI-6 downloads+resolve→`GufoServePlan`; DI-7 serving-identity+HTTP+dashboard; DI-8 benchmark guard. Exact Rust in **Appendix A**.

```
 vendored (pure) ─ load_vendored_catalog() ─► validation · provenance · vendored tests
        │                                          ▲
 assets/gufo-catalog/{toolboxes,models}.json       │ merge (transactional) + validate_catalog(merged) + validate_merged
        └─► load_effective_typed_catalog() : Result<(ToolboxCatalog,ModelCatalog), EffectiveCatalogError>  ── fail-closed ──►
              /api/toolbox-catalog · /api/toolbox-models · /api/toolbox-containers · server_mode resolvers · model_downloads
                                     └──────► dashboard (gufo tabs + sm-gufo panel)
```

## Architecture and flows
- **Catalog load (B2, fail-closed):** `gufo_overlay::merge` builds **new** merged `Value`s from clones of the vendored ones (transactional — nothing is published unless the whole merge succeeds) and returns `Result<(Value,Value), String>` (Err if a base structure or the `strix-halo` platform is missing, or **any** gufo id already exists in the vendored base — every duplicate id is an error, matching T-2). `load_effective_typed_catalog() -> Result<(ToolboxCatalog, ModelCatalog), EffectiveCatalogError>` then runs `schema_validate::validate_catalog(&merged_t,&merged_m)` (the **full** structural validator — so gufo gets the same checks as vendored: storage fields, unique container names, features, platform refs, defaults) **and** `gufo_overlay::validate_merged` (gufo semantics), and returns `Err(EffectiveCatalogError)` if either reports errors, otherwise the typed pair. Every runtime consumer calls this and **fails closed** (maps `Err` to its own error type — HTTP 500 / `ServerModeError` / `DownloadError`), so a bad overlay never exposes partial/invalid data. `"gufo"` is added to `KNOWN_BACKEND_IDS` so `validate_catalog` on the merged doc does not warn "brainrouter can't act on gufo". Daemon startup calls it and logs errors prominently (non-fatal, consistent with today's vendored startup validation; runtime endpoints fail closed regardless). `load_vendored_catalog`/`vendored_catalog_revision`/`vendored_catalog_snapshot` stay pure.
- **Create gufo toolbox (B1/B4 — decided empirically on the gufo image):** gufo's toolbox is modeled identically to halogen-flash (`toolbox_compatible:false`, "Server only"). The create path (`create_toolbox_container` → unconditional `toolbox create --image <image>`) is **image-specific**, so gufo createability is **not** inferable from halogen (different images). **PC-2** probes the **actual gufo image** (`toolbox create --image <gufo image> <probe>` → `podman inspect` → `rm -f`). **If PC-2 succeeds** → keep the existing `+ create` path (parity; R5 met via `+ create`, no UI/API change). **If PC-2 fails** → use the fully-specced pull-only path for every `toolbox_compatible:false` toolbox (gufo **and** halogen): `/api/toolbox-containers`'s response keeps the existing `containers` array unchanged and **adds a sibling `image_presence` map** keyed by `toolbox_id` → `{image_present: bool, image_id?: string}` (computed via a new `podman image exists <image>` / `podman image inspect` subprocess for each `toolbox_compatible:false` catalog toolbox with no live container); the dashboard keeps `container_exists = !!byName[tb.container_name]` (unchanged) and reads image state from `image_presence[toolbox_id]`; **and `POST /api/toolbox-containers` rejects a `toolbox_compatible:false` toolbox server-side** so the known-invalid `toolbox create` path is unreachable even by a direct API caller. `renderToolboxRow` renders a **Pull image** button (→ a **new** route `POST /api/toolbox-containers/pull` accepting `{"toolbox_id":"<catalog id>"}`, distinct from the existing `POST /api/toolbox-containers` and `POST /api/toolbox-containers/{name}/…` routes, which calls `pull_toolbox_image(container_name, image)` (`server.rs:2703`) only, writes **no** managed-toolboxes sidecar entry, and returns `{status, image_present:true}`) when `!toolbox_compatible && !container_exists`, then "Image present — start in Server Mode" once `image_present`; `+ create` is rendered only for `toolbox_compatible:true`. Both branches are implementation-ready.
- **Download:** `+ Download` → `POST /api/model-downloads {backend:"gufo", model_id}` → `resolve_catalog_entry(Gufo)` → `build_download(Gufo)` = single-repo `hf download <repo> <file> --revision <pin> --local-dir <models_dir>` → `execute` + `check_completeness` (exact size). Main and draft are separate gufo entries → separate rows/jobs; both land flat in `~/models/gufo` (distinct filenames).
- **Serve plan (B1) + Start:** `resolve_downloaded_gufo_model(model_id) -> Result<ResolvedGufoModel, DownloadError>` where `ResolvedGufoModel { models_dir: PathBuf, plan: GufoServePlan }`. Its core logic is a pure helper `plan_from_entry(main_entry, lookup, presence)` (I2) so all three cases are unit-testable without the filesystem: main must be `role=Main` + `Complete`; if it declares a speculative draft, resolve that entry (`role=Draft`), require `Complete`, → `Dflash2 { main_filename, draft_filename }`, else `Err(Validation("gufo model <id> needs its DFlash2 draft <draft_id> downloaded first"))`; no speculative → `Autoregressive { main_filename }`. `build_gufo_server_command(image, profile, models_dir, plan, req)` mounts `-v <models_dir>:/models:ro` and appends `--speculative dflash2 --dflash-model /models/<draft>` iff `Dflash2`. Start → `resolve_gufo_toolbox` (effective catalog) + resolve → `podman rm -f` → `run` → `register_serving_identity(Gufo,…)`. Stop → stop + `rm -f` + deregister. Status → `container_server_status("gufo", …)`.
- **Serve command contract (option order + value sources):**
  `run -d --name brainrouter-gufo-server <profile engine_args> --security-opt label=disable --userns=keep-id:uid=1000,gid=1000 --pull=always --label io.brainrouter.managed=true --label io.brainrouter.server_backend=gufo --label io.brainrouter.server_model=<model_id> -p <bind>:<port>:<port> -v <models_dir>:/models:ro <image> gufo serve --host 0.0.0.0 --port <req.port> --sessions <req.sessions|2> llm --model /models/<main> --served-model-name <model_id> [--speculative dflash2 --dflash-model /models/<draft>] --context <req.ctx> [<shlex(req.custom_args)>]` — `ctx`/`sessions`/`port`/`host`/`custom_args` come from `StartGufoServerRequest`; the dashboard form pre-fills `ctx`/`sessions` from the model entry's `ctx_default`/`sessions_default` (fallback 32768 / 2). `custom_args` are appended last. `--served-model-name` is set to the brainrouter `model_id` — PC-1 found gufo otherwise defaults the served name to the GGUF filename, so setting it makes the served name deterministic and equal to `LABEL_SERVER_MODEL`. (PC-1: `--host`/`--port`/`--sessions` are `gufo serve` server options placed before the `llm` subcommand; `--model`/`--served-model-name`/`--context`/`--speculative`/`--dflash-model` follow `llm`.)

## Alternatives considered
- Edit vendored catalog — rejected (breaks vendored invariant; sync clobbers).
- Merge inside `load_vendored_catalog()` — rejected (pollutes pure-upstream semantics); distinct `load_effective_typed_catalog()` chosen.
- One entry / one job with two `hf` commands — rejected v1 (no precedent; two entries reuse the tested single-repo path).
- AR fallback for a missing declared draft — rejected (was the v1 hybrid, B2/round-1); one contract: `speculative` present ⇒ draft required (D7).
- Only append overlay errors to the report (v2) — rejected (round-2 B2): consumers ignore the report → could expose partial data; v3 returns a checked `Result` and fails closed.
- "Attempt create, fix at T-7" (v2) — rejected (round-2 B4): v3 decides parity-primary + a concrete pull-only contingency + a pre-code create check.
- `:latest` with no pull policy — rejected (round-2 B3): add `--pull=always` (halogen-consistent) + pinned source commit + pre-code `--help` probe.
- Support gufo benchmarks / effective snapshot now — deferred (CHECK-widening migration needed); guard instead.
- Route gufo / ship DeepSeek+Flash-Next now — deferred.

## Detailed implementation
Exact type/function shapes are in **Appendix A**; the full overlay JSON is **Appendix B**. Paths verified unless "(new)".

**Pre-code validation gates (run after approval, before the dependent code; model-free except pulling the image):**
- **PC-1 (B3):** on strix, pull the gufo image and confirm the CLI surface — `podman run --rm --pull=always ghcr.io/gufo-org/toolboxes/gufo-runtime:latest gufo --version`, `… gufo serve --help`, `… gufo serve llm --help` — asserting `--host/--port/llm/--model/--speculative dflash2/--dflash-model/--context/--sessions`; record the pulled digest. Run before writing `build_gufo_server_command`; if a flag differs, update the command contract + Appendix A first. (Pulling a runtime image loads no model — permitted.)
- **PC-2 (B1/B4):** on strix, probe whether the **actual gufo image** supports the create path: `toolbox create --image ghcr.io/gufo-org/toolboxes/gufo-runtime:latest gufo-create-probe`, then `podman inspect gufo-create-probe`, then `podman rm -f gufo-create-probe`. **Success → keep the parity `+ create` path (no create-path/UI change). Failure → implement the pull-only path** (see "Create gufo toolbox") for every `toolbox_compatible:false` toolbox. Do **not** infer gufo createability from halogen.

**DI-1 — `src/toolbox_catalog/types.rs`:** add `Gufo` to both enums + `as_str`/`from_str`/`TryFrom`/`From`; `ALL`→6. Tests: gufo round-trip; change the vendored `ALL`-toolbox test (~529) to the upstream five. **Compile order (I2):** the other two compiler-enforced exhaustive matches — `models.rs::parse_payload`'s `Gufo` arm (needs `GufoModel`/`ModelPayload::Gufo` from DI-4) and `serving_identity.rs::openai_compatible_for_backend(Gufo)=>true` (from DI-7) — have **no** wildcard, so they must land **in the same first buildable commit** as the enum variant; the tree does not compile otherwise (see the ordered sequence).

**DI-2 — `assets/gufo-catalog/{toolboxes,models}.json` (new):** exactly Appendix B (embedded via `include_str!`). Note: `storage.config_key:"models_dir"`, `storage.default:"~/models/gufo"`; **no per-entry `backend`**; toolboxes carry an `attach_platform:"strix-halo"` consumed by the merge (not emitted).

**DI-3 — `src/toolbox_catalog/gufo_overlay.rs` (new) + `src/toolbox_catalog.rs`:** `merge` + `validate_merged` + `load_effective_typed_catalog` per Appendix A + the Architecture contract. Add `pub mod gufo_overlay;`. Add `"gufo"` to `schema_validate::KNOWN_BACKEND_IDS` (30) with a comment (upstream ids + brainrouter's gufo overlay). Reroute consumers to `load_effective_typed_catalog()` (fail closed): `server.rs:2405`+`2586`; `server_mode.rs:353`+vllm read (~1047); `model_downloads.rs:275`+`580`+`820`. `daemon.rs` startup: keep vendored validation + add a log-only call of `load_effective_typed_catalog()` reporting any `EffectiveCatalogError`. Tests: change `toolbox_catalog.rs:211`+`models.rs:344` `ALL`-iteration to the upstream five; add effective-catalog tests. **`validate_merged` enforces (normative — all tested, I4):** the gufo backend model section, the `strix-halo-gufo-rocm` runtime profile, the `strix-halo-gufo-runtime` toolbox, and the `strix-halo` `defaults.gufo` each exist **exactly once**; at least one `role=main` gufo entry exists; every gufo entry has non-empty `id`/`name`/`repo`/`revision` and **exactly one** `files[]` entry with a non-empty `path` and positive `size_bytes`; `speculative` appears only on a `role=main` entry; a referenced `draft_model_id` exists, differs from the main, has `role=draft`, and itself declares **no** `speculative`; `mode` is `dflash2`; `ctx_default`/`sessions_default`, when present, are positive; the gufo toolbox is attached to `strix-halo`. It rejects a missing/empty section, a bad/self/missing draft ref, a non-`dflash2` mode, a zero-file or multi-file entry, and a missing platform. `merge` errors on a missing base structure or **any** duplicate id; `load_effective_typed_catalog` returns `Err` for any merge/validation error and every consumer fails closed.

**DI-4 — `src/toolbox_catalog/models.rs`:** add `GufoRole`, `GufoSpeculativeMode` (closed, `dflash2` only), `GufoSpeculative`, `GufoModel` (Appendix A); `ModelPayload::Gufo` + the `parse_payload` `Gufo` arm; export from `toolbox_catalog.rs`. `speculative: null` ≡ omitted (both → `None`). **These types + the `parse_payload` arm land in DI-1's first buildable commit** (I2 — `parse_payload` is a wildcard-free exhaustive match). Tests: parse the overlay entries; a `mode:"mtp"` entry fails typed parse (surfaced via `load_effective_typed_catalog` `Err`).

**DI-5 — `src/server_mode.rs`:** `GUFO_SERVER_CONTAINER_NAME`, `StartGufoServerRequest`, `build_gufo_server_command(image, profile, models_dir, plan, req)`, `resolve_gufo_toolbox`, `start/stop/status` (Appendix A) — mirror ds4 minus ds4-only flags; add `--pull=always`, `--userns=keep-id:uid=1000,gid=1000`, `--security-opt label=disable`. `GufoServePlan` is defined in `model_downloads.rs` and imported here.

**DI-6 — `src/model_downloads.rs`:** `build_download` `Gufo` arm = `build_multifile_download(&m.repo,&m.revision,&m.files,models_dir)` (note: not compiler-enforced due to `_ => Err`; covered by T-4). Reroute `resolve_catalog_entry` to `load_effective_typed_catalog`. Add `ResolvedGufoModel`, `GufoServePlan`, the pure `plan_from_entry`, and `resolve_downloaded_gufo_model` (Appendix A). `local_presence_snapshot` already includes gufo once `build_download` supports it.

**DI-7 — HTTP + serving identity + dashboard:**
- `server.rs`: routes `GET/POST /api/server-mode/gufo/{status,start,stop}` (mirror ds4 ~1120-1151); handlers `server_mode_gufo_{status,start,stop}_response` (mirror ds4 3577-3630) — start resolves toolbox+plan, `start_gufo_server`, `register_serving_identity(Gufo, GUFO_SERVER_CONTAINER_NAME, req.toolbox_id, &profile, serving_identity_endpoint(host,port))`; stop deregisters. Reroute 2405/2586.
- `serving_identity.rs`: `openai_compatible_for_backend` add `Gufo => true` + test.
- `src/escalation/templates/main_dashboard.html` (explicit): `TOOLBOX_BACKEND_LABELS` add `gufo:'Gufo'`; add `'gufo'` to `DOWNLOAD_CAPABLE_BACKENDS` + `SERVER_MODE_CAPABLE_BACKENDS`; add `let serverModeGufoStatus = null;`; add `sm-panel-gufo` + `sm-gufo-{toolbox,model,ctx,sessions,host,port,custom-args,status}` + `sm-gufo-status` (mirror ds4; port 8080, ctx 32768, sessions 2); in `loadServerMode` add `safeFetch('/api/server-mode/gufo/status')` to `Promise.all`, destructure `gufoStatusRes`, assign `serverModeGufoStatus`, call the gufo render fns; in `setServerModeBackend` add gufo panel+status show/hide. Render fns: `renderServerModeGufoToolboxOptions` (filter `tb.backend==='gufo'` + platform ids); `renderServerModeGufoModelOptions` (**read `e.payload.role==='main'`, `e.payload.speculative && e.payload.speculative.draft_model_id`; list a main only when it is presence-complete AND, if it declares a draft, that draft entry is also complete; empty-state message: main-present-draft-missing vs no-models** — I1); `renderServerModeGufoStatus`; `startGufoServer`/`stopGufoServer`. Reuse `server-mode-hf-banner` + `emptyModelOption('gufo models', …)`. Downloads/toolboxes rows are generic (no per-backend markup).
- `scripts/test-gufo-ui.cjs` (new): static asserts (arrays, label, `sm-panel-gufo`+`sm-gufo-*`, the gufo status fetch+destructure+assign, render-fn calls, `emptyModelOption('gufo models', …)`) + behavioral fixtures using the **real `/api/toolbox-models` `entry.payload` shape** exercising `renderServerModeGufoModelOptions` (main incomplete → hidden; main complete/draft missing → hidden + draft-missing message; main+draft complete → listed) and `renderServerModeGufoStatus` (running/stopped).

**DI-8 — `src/benchmark.rs` (benchmark guard, I3):** in **`ServingRuntimeDefinition::validate` (451-467)**, reject `toolbox_backend == SupportedServingBackend::Gufo` with the crate's validation error (e.g. `BenchmarkError::Validation("gufo serving-runtime benchmarking is not supported in v1")`) **before** any ingest work, so the `serving_runtimes` CHECK (migration :30-39) is never reached. Test: an end-to-end `BenchmarkStore::ingest` with a `gufo` serving runtime returns `Err` and writes no `serving_runtimes` row (mirror `serving_runtime_referencing_unknown_catalog_snapshot_is_rejected`, 4728). Provenance note (N1): a created gufo toolbox records `vendored_catalog_revision()` as its catalog-revision label (the upstream pin), which does not change when the gufo overlay changes; this is an accepted v1 limitation — a dedicated `effective_catalog_revision()` for overlay-owned toolbox provenance is future work.

### Ordered implementation sequence
PC-1/PC-2 (pre-code, on strix). 1. **One buildable commit** = DI-1 (`Gufo` enum variant + conversions) + DI-4 (`GufoModel`+`ModelPayload::Gufo`+`parse_payload` arm) + `serving_identity.rs::openai_compatible_for_backend(Gufo)=>true` — all three compiler-enforced exhaustive matches together, so the tree compiles (I2). 2. DI-2 (Appendix B). 3. DI-3 (merge/validate/`load_effective_typed_catalog`/reroute/`KNOWN_BACKEND_IDS`/tests) → `cargo test toolbox_catalog::`; hit `/api/toolbox-catalog|models`. 4. DI-6 → T-4. 5. DI-8 → T-8. 6. DI-5 + DI-7 HTTP/handlers → T-3. 7. DI-7 dashboard + `scripts/test-gufo-ui.cjs` → `check-html-js.sh` + `node --test`. 8. Full local gate (T-6), deploy-to-strix, live verify (T-7).

## Testing and evaluation
- **T-1 (R1):** types round-trip/conversions; unknown-id fallback intact.
- **T-2 (R2):** effective catalog contains gufo (toolbox in `strix-halo` `toolbox_ids`+`defaults`, runtime profile resolvable, model section); `validate_catalog(merged)` passes with `gufo`∈`KNOWN_BACKEND_IDS`; `validate_merged` rejects bad draft ref / non-`dflash2` / missing platform; `merge` errors on a missing base / dup id; `load_effective_typed_catalog` returns `Err` for a broken overlay and a consumer surfaces it (fail-closed); vendored-feed tests still pass.
- **T-3 (R3):** `build_gufo_server_command` argv (AR vs Dflash2; flags; `-v <models_dir>:/models:ro`; labels/name; `--pull=always`); `plan_from_entry` unit tests — `speculative:None`→AR, complete-dflash2→Dflash2, missing/incomplete draft→specific `Err`; `resolve_downloaded_gufo_model` returns the configured `models_dir` (non-default too).
- **T-4 (R4):** `build_download` gufo argv (exact pinned revision); `check_completeness` matches the exact sizes; **integration**: `build_download(Gufo,…)` Ok (not the wildcard Err), `local_presence_snapshot` includes gufo, `/api/model-downloads/status` contains gufo, an unsupported (backend,payload) pair still errors.
- **T-5 (R5):** `scripts/test-gufo-ui.cjs` static + behavioral (payload-shaped fixtures) covering `renderServerModeGufoModelOptions`: main incomplete → hidden; main complete/draft missing → hidden + draft-missing message; main+draft complete → listed; **a complete AR main (both omitted `speculative` and explicit `speculative:null`) → listed without a draft** (I5); and `renderServerModeGufoStatus` running/stopped.
- **T-6 (R6):** `cargo test --locked -- --test-threads=1` (single-threaded per runbook; the pre-existing `unnecessary_map_or` clippy warning at `src/daemon.rs:134` is not ours), `cargo clippy --locked --all-targets` (0 errors), `bash scripts/check-html-js.sh`, existing `node --test scripts/*.cjs`.
- **T-7 (live, strix 127.0.0.1:9099):** gufo tab in Downloads **and** Server Mode; `sm-gufo-*` panel populated; the gufo toolbox is listed and its image is obtainable from the UI (`+ create` if PC-2 passed, else **Pull image**); a gufo GGUF download **starts** (main + draft rows); the modest **Q4_K_XL** model serves and answers `/v1/models` + a tiny `/v1/chat/completions`; record the pulled digest. Report the dashboard state.
- **T-8 (R6):** `ServingRuntimeDefinition::validate` rejects `gufo`; `BenchmarkStore::ingest` writes no `serving_runtimes` row for gufo; existing benchmark tests pass.

## Security, privacy, reliability, and operations
- **Privileges:** same ROCm device posture as ds4/halogen (`/dev/kfd`,`/dev/dri`,`keep-groups`,`memlock=-1`,`userns=keep-id`,`label=disable`); no new privilege. Loopback bind unless host `0.0.0.0`.
- **No host gate (v1):** brainrouter has no host detection (`main_dashboard.html:3055-3062`; `server_mode.rs:372-398`); gufo, like every backend, is offered wherever the dashboard runs, and start attempts `/dev/kfd`/`/dev/dri` regardless of hardware. v1 is a Strix-Halo deployment build and adds no host gate; a detected-platform check is future work. "strix grouping" = catalog membership only.
- **Memory/OOM:** 128 GiB shared. v1 = Q4_K_XL (16.35) / Q8_K_XL (29.30); live test uses Q4. DeepSeek/Flash-Next deferred. Downloads NVMe-bound (`~/models/gufo`).
- **Fail-closed catalog:** a bad overlay never exposes partial/invalid data (checked `Result` at every consumer); daemon logs it loudly.
- **Observability:** reuses download status + `container_server_status` + serving-identity snapshot; no new persistence.

## Rollout, migration, and rollback
- No DB/schema/config migration.
- Deploy per runbook (`docs/design/hankndory-brainrouter-integration.md`; inlined): commit on this worktree (`git commit -F /tmp/m.txt` to preserve the `Co-authored-by` trailer's angle brackets); Mac worktree can't push to GitHub (403) → push to strix (`git push ssh://papa@192.168.1.252/home/papa/ai/projects/brainrouter HEAD:refs/heads/<branch>`); on strix back up `~/.local/share/brainrouter/benchmarks.sqlite3`, then `git checkout master && git merge --ff-only <branch> && ~/.cargo/bin/cargo build --release --locked && systemctl --user restart brainrouter && curl -s 127.0.0.1:9099/health`; then `git push origin master` + the branch from strix. Keep Mac worktree = strix master = GitHub in sync.
- **Rollback (server-stopping, ordered):** (1) `POST /api/server-mode/gufo/stop` (or `podman stop -t 10 brainrouter-gufo-server && podman rm -f brainrouter-gufo-server`) + confirm the port is free (`ss -ltnp`); (2) revert the merge commit, rebuild, restart; (3) intentionally **retain** `~/models/gufo`, the pulled image, and any `gufo-runtime` container + sidecar entry unless a full cleanup is wanted (`podman rm -f gufo-runtime`; remove the sidecar entry; `rm -rf ~/models/gufo`; `podman rmi …gufo-runtime:latest`). If the code was reverted before stopping, stop the server directly with the podman commands (the daemon has no gufo stop route after revert).

## Risks and mitigations
- **R-a (med): overlay/image drift** — exact pinned revisions+sizes (HF API) + `--pull=always` + PC-1 `--help` probe + recorded digest; `validate_catalog(merged)`+`validate_merged` fail-closed.
- **R-b (med): non-compiler-enforced download/presence (I4)** — grep checklist below + T-4 integration tests.
- **R-c (med): server-only create (B1/B4)** — PC-2 probes the **actual gufo image**; both `+ create` parity and the fully-specced pull-only fallback (with `image_present` API + Pull-image UI) are implementation-ready for gufo+halogen.
- **R-d (low-med): partial/invalid catalog exposure** — transactional merge + checked `Result` + fail-closed consumers + loud startup log.
- **R-e (low): benchmark CHECK violation** — DI-8 guard in `validate` + T-8.
- **R-f (low): rollback leaves a running server/port** — ordered stop-first rollback.
- **Grep checklist (I4):** compiler-enforced closed matches — `types.rs` (`as_str`/`TryFrom`/`From`), `models.rs::parse_payload` (311), `serving_identity.rs::openai_compatible_for_backend` (71). NOT enforced — `model_downloads.rs::build_download` (`_ => Err`) + `local_presence_snapshot` (`Err(_) => continue`). No `SupportedServingBackend`/`ModelPayload` match in `router.rs`/`observability.rs`/`cli.rs`. `benchmark.rs` uses the enum as a field only (DI-8 guard in `validate`). Three `ALL`-vendored tests → upstream-5.

## Open questions
None material. The only empirical unknowns are resolved by pre-code gates: PC-1 (gufo `serve --help` flag surface) and PC-2 (`toolbox create` on the **actual gufo image**), each with a defined outcome path and a fully-specced fallback.

## Decision log
- **D1** Overlay, not vendored edit — Adopted.
- **D2** `load_effective_typed_catalog()` distinct from `load_vendored_catalog()` — Adopted.
- **D3** Two catalog entries (main + draft), reuse `build_multifile_download` — Adopted.
- **D4** — Superseded by **D7**.
- **D5** `openai_compatible=true`, no routing — Adopted.
- **D6** v1 = Qwen3.8-27B (Q4_K_XL + Q8_K_XL) + shared DFlash2 draft; DeepSeek/Flash-Next deferred — Adopted.
- **D7** Single serving contract: `speculative` present ⇒ draft required (specific error); only `speculative:None` serves AR; dropdown draft-aware — Adopted.
- **D8** gufo excluded from benchmark ingest (guard in `ServingRuntimeDefinition::validate` before the CHECK-constrained INSERT) — Adopted.
- **D9** v1 adds no host gate; Strix deployment build; "strix grouping" = catalog grouping — Adopted.
- **D10** gufo toolbox = halogen-flash shape; the create path is image-specific, so gufo createability is decided by **PC-2 probing the actual gufo image** (not inferred from halogen); success → keep `+ create` parity, failure → the fully-specced pull-only path (`image_present` API + Pull-image UI) for both server-only toolboxes — Adopted (resolves round-3 B1 / B4). **PC-2 (2026-09-24) PASSED** on strix: `toolbox create --image ghcr.io/gufo-org/toolboxes/gufo-runtime:latest gufo-create-probe` → RC=0, container created (then removed); the parity `+ create` path is used — **no pull-only fallback needed for v1**.
- **D11** `~/models/gufo` storage default (`config_key:"models_dir"`), config-overridable — Adopted.
- **D12** Effective catalog is transactional + `schema_validate::validate_catalog`(merged) + `validate_merged`, returned as a checked `Result`; consumers fail closed; daemon logs — Adopted (resolves round-2 B2).
- **D13** Add `"gufo"` to `schema_validate::KNOWN_BACKEND_IDS` so the merged-catalog validation does not warn on gufo — Adopted (round-2 B2).
- **D14** `GufoSpeculativeMode` is a closed serde enum (v1: `Dflash2` only); a non-`dflash2` overlay fails typed parse and is surfaced via `load_effective_typed_catalog` `Err` — Adopted (round-2 N2).
- **D15** Image `:latest` + `--pull=always` (halogen-consistent) + pinned gufo source commit `9cad139…` + PC-1 probe; OCI-digest pinning is future — Adopted (round-2 B3). **PC-1 (2026-09-24) PASSED** on strix: `gufo version 9cad139` (matches the pinned commit); pulled image digest `sha256:989ab52a190244f08511a3ad0fd46546f2144220c37e6c808ad8008a934ebab5`; argv flags confirmed (`--host/--port/--sessions`, `llm --model/--served-model-name/--context`, `--speculative dflash2 --dflash-model`).

## Referenced files
- `src/toolbox_catalog/types.rs`; `src/toolbox_catalog/models.rs` (`parse_payload` KEY-selected; `CatalogModelEntry.payload`); `src/toolbox_catalog.rs` (`load_vendored_catalog`/`typed`/`VendoredCatalog`, `vendored_catalog_revision`; add `load_effective_typed_catalog`+`gufo_overlay`); `src/toolbox_catalog/schema_validate.rs` (`validate_catalog`, `KNOWN_BACKEND_IDS`, `storage.config_key`+`default`); `assets/cockpit-catalog/{toolboxes,models}.json`+`SOURCE` (halogen-flash shape + `--pull=always` profile).
- `src/server_mode.rs` (ds4 builder + helpers, `resolve_toolbox_for_server`, `ResolvedDs4Model.models_dir`, container-status, vllm serve); `src/model_downloads.rs` (`build_download`/`build_multifile_download`/`execute`/`check_completeness`/`effective_models_dir`/`resolve_downloaded_ds4_model`/`local_presence_snapshot`).
- `src/serving_identity.rs`; `src/server.rs` (routes/handlers, `register_serving_identity`, `create_toolbox_container`/`recreate_toolbox_container`/`toolbox_create`, `load_typed_toolbox_catalog`); `src/benchmark.rs` (`ServingRuntimeDefinition::validate` 451-467, ingest 2452-2455, insert 2611-2628) + `migrations/0003_toolbox_serving_dimension.sql` (closed CHECK).
- `src/escalation/templates/main_dashboard.html`; `scripts/test-hf-preflight-ui.cjs`.
- **External (pinned):** github.com/gufo-org/gufo @ `9cad13974cf6da0cd3674b4e0a88b14b7e4a2908` — `README.md`, `docs/models/qwen3.8-27b/README.md`, `docs/SERVER.md`, `docs/CLI.md`; HF trees `unsloth/Qwen3.8-27B-GGUF@4ca7207…`, `z-lab/Qwen3.8-27B-DFlash2-GGUF@2d9571f…`.
- `docs/design/{ai-toolbox-cockpit-integration,toolbox-create-label-sidecar-ownership,feature-b-servermode-routing,hf-preflight-download-diagnostics,hankndory-brainrouter-integration}.md`.

## Dory validation record
- **R1 comprehension+clarity (explore, isolated), v1.** Comprehension FAIL (self-containment nits only; code claims verified), Clarity PASS. Fixed in v2 (runbook named; exact pins/sizes; wording).
- **R1 critic (rubber-duck, isolated), v1.** 3 blocking / 7 important / 2 nits — all addressed in v2.
- **R2 critic (rubber-duck, isolated), v2.** 4 blocking (B1 models_dir; B2 fail-closed+full-validate; B3 source/image pin; B4 server-only create decision) / 3 important (I1 `payload` paths; I2 AR resolver seam; I3 guard location) / 2 nits (N1 provenance; N2 closed mode) — **all addressed in v3**; the critic separately verified overlay parse shape, download consistency, blast-radius, require-draft coherence, and dashboard wiring as correct. Self-audit: used `benchmark.rs`+`migrations/0003…` (now referenced).
- **R2 readiness (explore, isolated), v2 — NOT READY.** 7 concreteness gaps (source pin; exact `GufoModel`/`GufoRole`/`GufoServePlan`; full overlay JSON; command ordering/value sources; effective-catalog error contract; benchmark guard function). All resolved in v3 via Appendix A + Appendix B + DI-3/DI-8 + the command contract. Verified all named existing symbols exist.
- **R3 readiness (explore, isolated), v3 — READY.** All 7 dimensions PASS; every requirement maps to a DI element + test; all named symbols verified to exist; interfaces precise (Appendix A/B); pre-code gates defined; rollout/rollback actionable; ACs objectively testable.
- **R3 critic (rubber-duck, isolated), v3.** 1 blocking (B1: PC-1 inferred gufo createability from halogen's image) / 4 important (I2 compile order of exhaustive matches; I3 request `Deserialize` derive; I4 normative `validate_merged` list; I5 AR dashboard fixture) / 1 nit (N6 `docs/SERVER.md`/`docs/CLI.md`). The critic independently re-derived the merged `validate_catalog` result (no errors/warnings once `gufo`∈`KNOWN_BACKEND_IDS`) and verified serde shapes, argv, benchmark-guard location, dashboard payload, and Appendix B JSON as correct. **All addressed in v4.**
- **R4 critic (rubber-duck, isolated), v4 — NO BLOCKING DEFECTS** (0 blocking / 2 important / 1 nit). The critic independently re-verified (re-fetching the pinned gufo `README.md`/model guide + HF headers) the argv, exact pins/sizes, serde shapes, benchmark-guard location, dashboard `payload` paths, compile order, and the full effective-catalog consumer list as correct, and named the exact three `ALL`-iterating tests (`toolbox_catalog.rs:206-213`, `models.rs:338-353`, `types.rs:509-532`). The 2 important (pull-only response contract; `validate_merged` existence/non-empty invariants) + 1 nit (duplicate-id policy) are non-material hardening — **folded into v5**.
- **R4 readiness (explore, isolated), v4 — READY** after two doc-hygiene fixes (full `src/escalation/templates/main_dashboard.html` path; explicit pull-route contract), both applied.
- **Gate status: Step 6 critic PASSED (no blocking) + Step 7 readiness READY.** Isolation: each Dory phase ran as a separate zero-context sub-agent (no access to the Hank conversation), reading only the design doc + its referenced files — satisfying the amnesia requirement in lieu of a fully separate app session.
- **Step 9 mean code review (isolated), post-implementation — NO significant issues.** The reviewer independently built + ran the full suite (424 lib tests + 6 UI tests), clippy, and the HTML/JS gate (all pass) and verified all 8 focus areas (fail-closed catalog + rerouted consumers; two-file serve/download; server-mode + handlers incl. the intact `server_mode_ds4_stop_response`; dashboard `payload` wiring; benchmark guard; exact overlay pins; live-deploy robustness). Two items examined were confirmed non-defects.
- **T-7 live verification on strix (127.0.0.1:9099), 2026-09-24 — PASS.** `/api/toolbox-catalog` lists the gufo toolbox + `strix-halo` `defaults.gufo`; `/api/toolbox-models` lists all three gufo models (typed); `/api/server-mode/gufo/status` responds; `POST /api/toolbox-containers` created the `gufo-runtime` toolbox (parity with halogen-flash); the main + draft downloads ran the exact pinned `hf download` commands and completed with **byte-exact** sizes (17559178144 / 1143006816). The gufo server started, loaded in ~13 s (`speculative=dflash2 sessions=2 context=32768`, GPU ~31 GiB / 127 GiB), served `/v1/models` = `['gufo-qwen38-27b-q4kxl']` and `/v1/chat/completions` → HTTP 200 with active DFlash2 speculation (`draft_accepted=11/18`); serving-identity registered (`rocm`, `openai_compatible=true`); stop freed memory. Image digest `sha256:989ab52a190244f08511a3ad0fd46546f2144220c37e6c808ad8008a934ebab5`.

## Human approval
- **Approved for implementation — 2026-09-24** via plan-mode approval (autopilot). The creator session `Brainrouter dory review` (`1c5cd7ef-0b4a-4672-8d5a-872e380a58a7`) was sent the design for review in parallel. Human review was required (reviewer available) and was satisfied by that approval; any material implementation discovery returns to design (Step 10).

---

## Appendix A — Exact Rust type & function definitions (implementable shapes)
*(Illustrative signatures/shapes, not a substitute for implementation; field names/serde are normative.)*

```rust
// src/toolbox_catalog/models.rs
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GufoRole { Main, Draft }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GufoSpeculativeMode { Dflash2 } // v1: closed; a non-dflash2 mode fails typed parse

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GufoSpeculative { pub mode: GufoSpeculativeMode, pub draft_model_id: String }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GufoModel {
    pub id: String,
    pub name: String,
    pub repo: String,
    pub revision: String,
    pub role: GufoRole,
    pub files: Vec<CatalogModelFile>,          // v1 invariant: exactly one file per entry
    #[serde(default)] pub recommended: bool,
    #[serde(default)] pub ctx_default: Option<u32>,
    #[serde(default)] pub sessions_default: Option<u32>,
    #[serde(default)] pub speculative: Option<GufoSpeculative>, // null ≡ omitted ≡ None (AR)
    #[serde(flatten)] pub extra: serde_json::Map<String, serde_json::Value>,
}
// ModelPayload::Gufo(GufoModel); parse_payload adds the SupportedServingBackend::Gufo arm.

// src/toolbox_catalog.rs
pub struct EffectiveCatalogError { pub errors: Vec<String> }
pub fn load_effective_typed_catalog()
    -> Result<(ToolboxCatalog, ModelCatalog), EffectiveCatalogError>;
// = load_vendored_catalog() raw Values → gufo_overlay::merge (transactional, Result)
//   → schema_validate::validate_catalog(&merged_t,&merged_m) (errors ⇒ Err)
//   → gufo_overlay::validate_merged (errors ⇒ Err) → typed() → Ok

// src/toolbox_catalog/gufo_overlay.rs
pub fn merge(toolboxes: &Value, models: &Value) -> Result<(Value, Value), String>; // returns NEW merged values
pub fn validate_merged(toolboxes: &Value, models: &Value) -> Vec<String>;

// src/model_downloads.rs
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GufoServePlan {
    Autoregressive { main_filename: String },
    Dflash2 { main_filename: String, draft_filename: String },
}
pub struct ResolvedGufoModel { pub models_dir: PathBuf, pub plan: GufoServePlan }
// Pure, filesystem-free core (I2) — unit-testable for all three cases:
fn plan_from_entry(
    main: &GufoModel,
    lookup_draft: impl Fn(&str) -> Option<GufoModel>,
    is_complete: impl Fn(&GufoModel) -> bool,
) -> Result<GufoServePlan, DownloadError>;
pub fn resolve_downloaded_gufo_model(model_id: &str) -> Result<ResolvedGufoModel, DownloadError>;
// build_download arm: (Gufo, ModelPayload::Gufo(m)) => Ok(build_multifile_download(&m.repo,&m.revision,&m.files,models_dir))

// src/server_mode.rs
pub const GUFO_SERVER_CONTAINER_NAME: &str = "brainrouter-gufo-server";
#[derive(Debug, Clone, Deserialize)]
pub struct StartGufoServerRequest {
    pub toolbox_id: String, pub model_id: String, pub ctx: u32,
    #[serde(default)] pub sessions: Option<u32>,
    pub host: String, pub port: u16, pub custom_args: Option<String>,
}
pub fn build_gufo_server_command(
    toolbox_image: &str, runtime_profile: &RuntimeProfile,
    models_dir: &std::path::Path, plan: &GufoServePlan, req: &StartGufoServerRequest,
) -> Result<Vec<String>, ServerModeError>;
pub fn resolve_gufo_toolbox(toolbox_id: &str) -> Result<(ToolboxDefinition, RuntimeProfile), ServerModeError>;
pub async fn start_gufo_server(req: &StartGufoServerRequest) -> Result<(), ServerModeError>;
pub async fn stop_gufo_server() -> Result<(), ServerModeError>;
pub async fn gufo_server_status() -> ServerStatus;

// src/serving_identity.rs: openai_compatible_for_backend(SupportedServingBackend::Gufo) => true
// src/benchmark.rs: in ServingRuntimeDefinition::validate — reject toolbox_backend == Gufo (BenchmarkError::Validation)
```

## Appendix B — Complete overlay JSON (`assets/gufo-catalog/`)
`toolboxes.json`:
```json
{
  "runtime_profiles": {
    "strix-halo-gufo-rocm": {
      "engine_args": ["--device","/dev/kfd","--device","/dev/dri","--group-add","keep-groups","--ulimit","memlock=-1"]
    }
  },
  "toolboxes": [
    {
      "id": "strix-halo-gufo-runtime",
      "backend": "gufo",
      "name": "Gufo Runtime (Strix Halo)",
      "container_name": "gufo-runtime",
      "group": "Server only",
      "image": "ghcr.io/gufo-org/toolboxes/gufo-runtime:latest",
      "channel": "stable",
      "maturity": "experimental",
      "description": "gufo OpenAI-compatible inference server for Strix Halo (gfx1151). Pull the image here; start it in Server Mode. Not a Toolbx/Distrobox image.",
      "runtime_profile": "strix-halo-gufo-rocm",
      "toolbox_compatible": false,
      "features": { "interactive": "unavailable", "models": "supported", "server": "experimental" }
    }
  ],
  "attach_platform": "strix-halo"
}
```
`models.json`:
```json
{
  "backends": {
    "gufo": {
      "kind": "gguf",
      "storage": { "config_key": "models_dir", "default": "~/models/gufo" },
      "models": [
        {
          "id": "gufo-qwen38-27b-q4kxl",
          "name": "Qwen3.8-27B UD-Q4_K_XL (DFlash2)",
          "repo": "unsloth/Qwen3.8-27B-GGUF",
          "revision": "4ca720788d1e01f1bff70c033e0d0028fd02e502",
          "role": "main",
          "recommended": true,
          "ctx_default": 32768,
          "sessions_default": 2,
          "speculative": { "mode": "dflash2", "draft_model_id": "gufo-qwen38-27b-dflash2-draft" },
          "files": [ { "path": "Qwen3.8-27B-UD-Q4_K_XL.gguf", "size_bytes": 17559178144 } ]
        },
        {
          "id": "gufo-qwen38-27b-q8kxl",
          "name": "Qwen3.8-27B UD-Q8_K_XL (DFlash2)",
          "repo": "unsloth/Qwen3.8-27B-GGUF",
          "revision": "4ca720788d1e01f1bff70c033e0d0028fd02e502",
          "role": "main",
          "ctx_default": 32768,
          "sessions_default": 2,
          "speculative": { "mode": "dflash2", "draft_model_id": "gufo-qwen38-27b-dflash2-draft" },
          "files": [ { "path": "Qwen3.8-27B-UD-Q8_K_XL.gguf", "size_bytes": 31457991680 } ]
        },
        {
          "id": "gufo-qwen38-27b-dflash2-draft",
          "name": "Qwen3.8-27B DFlash2 draft (Q4_K_M)",
          "repo": "z-lab/Qwen3.8-27B-DFlash2-GGUF",
          "revision": "2d9571f8ce46e151f61c6499c99dee6079e1d610",
          "role": "draft",
          "files": [ { "path": "Qwen3.8-27B-DFlash2-Q4_K_M.gguf", "size_bytes": 1143006816 } ]
        }
      ]
    }
  }
}
```
*(Merge attaches `strix-halo-gufo-runtime` to the `strix-halo` platform's `toolbox_ids` and sets `defaults["gufo"]="strix-halo-gufo-runtime"`; `attach_platform` is consumed by the merge, not emitted into the served catalog.)*
