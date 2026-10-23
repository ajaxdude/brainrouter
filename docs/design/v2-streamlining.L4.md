This file is part of the design: the Map lives in `v2-streamlining.md`; this is leg L4's full spec (drafted while L3 builds; leg-scoped review pending).

# Leg L4 in full — Swap supervisor + platform probe

Goal (R5, R6-prep): brainrouter itself owns engine processes and model lifecycle — the API-driven generalization of the owner's icon scripts — with the invariants those scripts already debugged. L4 lands the abstraction + llama.cpp-family adapters + the shim; strixite/hipfire land in L5 on top of it.

## Platform probe (`src/platform.rs`)

- Detection (Linux): `lspci`/sysfs PCI enumeration (vendor/device ids), GPU arch via `rocminfo`/kfd topology where present, memory: `MemTotal`/`MemAvailable`, AMDGTT size from kernel params (`ttm.pages_limit`, `amdgpu.gttsize` read from `/sys/module` or cmdline), presence markers (toolbox runtime image, `hipfire` install roots, `strixite` build root config). macOS (L8): `sysctl hw.model`/Metal — interface stub now.
- Output: `PlatformInfo {id: "strix-halo"|"gfx1100"|"nvidia"|"generic-linux"|..., gpus: [{vendor, arch, vram, is_igpu}], gtt_budget, ram}`.
- Engine compatibility matrix: per-engine requirement records (data, not code): strixite = gfx1151 + GTT ≥ ~100 GiB; gufo = gfx1151; hipfire = ROCm gfx9/10/11 discrete (gfx1100, gfx1151); llama.cpp = universal (Vulkan/ROCm/CUDA/CPU). Feeds: Engines tab offers/why-not list (replaces hardcoded `CURRENT_PLATFORM_ID = 'strix-halo'` + toolbox image default in `src/server.rs`), toolbox catalog platform filter, install-time engine recommendations.

## Model registry + launch specs (`src/swap/`)

- Catalog entries (native, `swap_catalog.json` in state dir): `{id, engine, files (GGUF/card paths), ctx, quant, family, tools, vram_estimate, launch_spec {args template, env, port policy}}`. Seeds: (1) IMPORT of the live llama-swap `config.yaml` keys (inverse of `llama_swap_register.rs` — the existing import surface), so every model v1 served is a native catalog entry from day one; (2) toolbox/server-mode catalog (gufo/ds4 entries already modeled); (3) operator additions via CLI.
- Launch spec validation: engine-validated flags only (adapter-owned arg schema — the "flag the engine doesn't have kills the server" invariant); ctx/max-seq math documented per engine.
- Lifecycle: `ModelState = Stopped | Loading (load queued/in-progress, progress % where source gives it) | Ready (stem-asserted) | Draining (in-flight finishing) | Failed`. Scheduler: `ensure(model_id) -> Endpoint` — spawn if Stopped (respecting budget), queue behind loads (cap, visible), proxy when Ready. Health: pid liveness + periodic `/v1/models` + `/health` probes; crash detection → state + auto-restart policy (bounded, like plugin supervisor).
- Boot invariants (from the icon era, binding): single-instance lock per model (flock-equivalent); served-model STEM ASSERTION after health (expected stem in `/v1/models` id — mismatch = Failed, port squatting made impossible); stop-only-what-you-started (start-ownership tracked; daemon restart adopts by ownership marker, never kills what it didn't spawn); SIGTERM→grace→kill by PID.
- VRAM/GTT budget: per-platform budget (probe: GTT for strix-halo, VRAM+RAM policy elsewhere) minus pinned residents (Bonsai/decision model, classifier, brainrouter's own provider when self-hosted — pin list); eviction: in-flight-aware LRU (last-used, never evict with active streams; drain-then-evict); over-budget request → queue + dashboard ETA, evict lowest-LRU to fit, explicit "no fit" error after candidates exhausted.

## Engine adapters (`swap::EngineAdapter` trait)

- Surface: `probe() / spawn(LaunchSpec) / stop(handle) / health(endpoint) -> stem / vram_estimate(entry) / capabilities(entry)`; endpoint registration feeds `serving_identity.rs` (registry reuse; Served-model namespace contract: catalog id canonical, `served_model` string resolvable — migration window keeps both).
- L4 adapters: `llama_swap_shim` (today's behavior: ensure = no-op pass-through, llama-swap owns lifecycle — ZERO routing change for existing models), `llama_cpp` native (spawn llama-server directly: GGUF path, args from spec, Vulkan/ROCm toolbox wrapper precedent from server-mode; this is the L7 path, proven here in parallel operation), `fake` (scripted test engine: configurable load time, stem, crashes, slots progress — the acceptance substrate).

## Routing + dashboard integration

- Router: auto-path candidate pool and pinned-card routes resolve through the swap registry (`ensure` before proxy; Loading → request queued with the load, visible in in-flight tracker as "loading <model> p%" using existing pp-progress machinery; failure → chain per L3 rules). Selector semantics unchanged (Selector-vocabulary contract through L7: `local` keeps today's path primary; the swap layer is additive to it).
- Events: additive `swap_state` on route events (where the swap layer resolved).
- Engines tab: platform card (detected id, GPUs, budgets), engine offers + reasons-not, model lifecycle table (state, last-used, evictions today), swap queue, budget gauge; Models tab entries gain "serve via" (adapter) column.

## CLI

- `brainrouter swap status|ensure <model>|evict <model>|catalog list|import-llamaswap|add <spec.json>|rm <id>`; `brainrouter platform show`. Mutations via daemon (single-writer), reads direct.

## Contracts kept

v1 routing byte-identical while `llama_swap_shim` owns models (the shim default = today); selector vocabulary; served-model namespace dual-resolvable; events additive; configs boot (no v1 key changes in L4; catalog import is additive, `--dry-run` preview per migration contract, snapshot before first import).

## Build order

platform probe + matrix (+ replace hardcoded platform ids) → registry + specs + llama-swap import → state machine + scheduler (fake adapter) → budget + LRU/eviction → llama_cpp native adapter (parallel operation mode) → router ensure-queue integration → Engines tab + CLI → docs.

## Acceptance (Ray)

(1) Fake-adapter lifecycle suite: ensure spawns, stem assertion passes/fails→Failed, crash→bounded restart, queue cap + ETA, LRU evicts least-recent-idle, in-flight never evicted (drain-then-evict observed), start-ownership (foreign stem on port = no kill; own process = cleaned). (2) Budget: over-budget ensure evicts to fit (sequence asserted), no-fit explicit error, pinned residents never evicted. (3) Platform probe on this host = strix-halo/gfx1151 (real), matrix fixtures: gfx1100 probe offers hipfire+llama.cpp, NOT strixite/gufo, with reasons; Engines tab data endpoint reflects. (4) Shim parity: with shim sole owner, v1 selector/reviewer test floors byte-identical; import lists every live llama-swap key as catalog entry; dual-mode: same model via shim and via native adapter resolve to distinct endpoints, no state bleed. (5) Events additive swap_state; v1 parse. (6) CLI surface incl. daemon-down refuse for mutations. (7) All prior suites green (workspace both modes, L1/L2/L3 acceptance, node, html-js).
