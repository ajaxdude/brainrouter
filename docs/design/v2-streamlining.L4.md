This file is part of the design: the Map lives in `v2-streamlining.md`; this is leg L4's full spec. Version L4-v2 (2026-10-22): leg-scoped round 1 folded — Residency view (B1), adoption markers, LRU clocks, per-adapter progress, shaping contract, probe vocabulary, sized pins, import fidelity, measured tokens, concurrency tests.

# Leg L4 in full — Swap supervisor + platform probe

Goal (R5, R6-prep): brainrouter owns engine processes and model lifecycle — the API-driven generalization of the owner's icon scripts, inheriting their debugged invariants. L4 lands the abstraction + llama.cpp-family adapters + the shim; strixite/hipfire land in L5.

## Residency view (the input side of every budget/eviction/spawn decision)

`Resident = {catalog_id, source: shim | native | container | external, endpoint, gguf_path?, started_at, last_stream_completed}` assembled continuously by one observer:
- **Shim residents**: llama-swap `/running` polled per-model state (extends today's `fetch_slots` poller past its `running[0]` single-model assumption). Canonical GGUF path + key resolved through the imported catalog.
- **Native residents**: the swap registry's own live children (pid + health).
- **Container residents**: server-mode identities (existing `serving_identity.rs` registry).
- **External residents (pins)**: operator-set `pins: [{id, reserved_gib, reason}]` — the self-provider strixite model, the decision model, anything the supervisor must never evict and cannot observe natively. A pin whose id appears in any observed resident set is marked present; a pin absent at probe time logs a warning (under-reservation visible). Pins reserve `reserved_gib` at admission — an absent pin still subtracts budget (it may be loading out-of-band).
Budget input = sum(residents.vram_estimate) + sum(pins.reserved_gib) + headroom. **The scheduler's `ensure(X)`: when X's canonical GGUF (or catalog id) is resident at the shim, X is Ready-at-shim → the shim endpoint is returned, no native spawn.** The per-model flock (keyed on canonical GGUF) guards native-vs-native races only; llama-swap and strixite are not lock participants, so the Residency view is the cross-mechanism guard. Acceptance (4) asserts SINGLE residency: native ensure of a shim-loaded model attaches (no new process, no second `/v1/models` listing of the same file); the budget gauge reports source=shim|native|container|pin.

## Platform probe (`src/platform.rs`)

- Detection (Linux): PCI enumeration (lspci/sysfs), `rocminfo`/kfd arch where present, `MemTotal`/`MemAvailable`, GTT from kernel params (`ttm.pages_limit`, `amdgpu.gttsize`), engine install markers (toolbox runtime, hipfire roots, strixite build root). macOS (L8): sysctl/Metal stub now.
- `PlatformInfo {id, gpus:[{vendor, arch, vram, is_igpu}], gtt_budget, ram}`. **`id` = the catalog platform id when detected hardware matches a feed platform (`strix-halo` for gfx1151; feed id for gfx1100-class hosts), else `generic-<arch>`.** Consumers: dashboard `CURRENT_PLATFORM_ID` (replaces the hardcoded literal; an unknown id shows ALL toolboxes with an explicit "unknown platform" banner, never a silent fall-open), toolbox image default, cockpit config `active_platform` (stays operator-set; the probe offers a one-click "set as active").
- Engine compatibility matrix (data, not code): strixite = gfx1151 + GTT ≥ ~100 GiB; gufo = gfx1151; hipfire = ROCm gfx9/10/11 discrete; llama.cpp = universal. Drives Engines-tab offers + reasons-not + install recommendations.

## Model registry + launch specs (`src/swap/`)

- `swap_catalog.json` (state dir): `{id, engine, files, ctx, quant, family, tools, vram_estimate, launch_spec {resolved args, env, port policy, served_model_alias = catalog id}, last_stream_completed}`. Seeds: (1) **IMPORT of the live llama-swap `config.yaml`** — inverse of `llama_swap_register.rs`, EXPANDING the macro chain (`${ls}`, `${common}`) against llama-swap's own settings so every entry records fully-resolved args; groups recorded as catalog metadata (L7 slot-sharing input); idempotent merge, `--dry-run` stable; (2) toolbox/server-mode catalog; (3) CLI additions.
- Launch spec validation: adapter-owned arg schema; engine-validated flags only; ctx/max-seq math documented per engine; the native llama-server launch MUST pass the served-model-alias flag set to the catalog id (precedent: gufo `--served-model-name`).
- Lifecycle: `Stopped | Loading | Ready (stem-asserted) | Draining | Failed`. `ensure(model_id) -> Endpoint`: consult Residency view (attach if resident anywhere); Stopped → budget check → evict-to-fit (LRU) → spawn (flock; concurrent ensure(X) = exactly one spawn, all attach); Loading → queue (cap, per-request position); Ready → proxy. Crash → bounded restart (plugin-supervisor pattern). Queue: in-memory, lost on daemon restart (clients retry; adopted running loads per below), queued requests cancellable via the in-flight row.
- Boot invariants (binding): single-instance flock per canonical GGUF (held for child lifetime; auto-released on crash); stem assertion — for SPAWNING adapters (health returns the stem SET for shared endpoints like the shim; assertion scope = spawn validation, not per-request probing); start-ownership with a defined **adoption marker for native children**: state-dir file per model `{pid, gguf, start_time, spec_hash}`, written atomically before spawn, cleared on stop; native children spawn detached (setsid); daemon boot scans markers — live pid + matching gguf+hash ⇒ ADOPT (register as native resident, no kill, no respawn), dead pid/stale marker ⇒ cleanup. (systemd user-unit note: unit keeps `KillMode=control-group`; adopted children from a previous detached generation survive restarts by design, children of THIS unit's generation die with it — documented, visible in Engines tab.)
- LRU: key = catalog id; clock = last STREAM COMPLETION (monotonic, recorded on SniffStream drop into the registry); in-flight check matches the ADDITIVE catalog-id stamp on in-flight rows (stamped at route time; alias-window strings unchanged), never evict with active streams; drain-then-evict; over-budget → queue + ETA; explicit no-fit after candidates exhausted (pins included in the arithmetic).

## Engine adapters (`swap::EngineAdapter`)

`probe() / spawn(LaunchSpec) / stop(handle) / health(endpoint) -> stem set / vram_estimate(entry) / capabilities(entry)`.
- L4 adapters: `llama_swap_shim` (ensure = Residency-view pass-through; ZERO routing change; circuit breaker = existing `LLAMA_SWAP_KEY`), `llama_cpp` native (direct llama-server, toolbox wrapper precedent; L7's path, proven in parallel here), `fake` (scripted: load time, stem set, crashes, load progress — the acceptance substrate).
- Load progress per source: shim = llama-swap `/running` state (poller extended per model); native = llama-server offers no load-progress endpoint ⇒ indeterminate + elapsed, ETA from EWMA of measured past loads persisted in the registry; fake = scripted. Matching per-ENDPOINT.
- Request shaping: the local llama-server-family sanitization (system-message merge, developer→system, assistant content/tool_calls fix) moves from `try_llama_swap` into a SHARED router-side contract applied to ALL llama-server-family endpoints (shim + native). Golden-body test: same request, shim vs native, byte-identical sanitized body.

## Routing + dashboard + events

- Router: pinned-card + auto routes resolve through `ensure` (Loading queues with the load; in-flight row shows "loading <model>" per adapter progress source; failure → chain per L3). Selector semantics unchanged; shim sole-owner mode = byte-identical v1.
- Events additive: `swap_state`; `swap_source` (shim|native|container); measured `prompt_tokens` from the EXISTING usage-chunk backfill land per model in the registry (absent usage ⇒ estimator continues) — feeds guardrails, LRU, Engines tab. `swap_state` carried, not rendered, until the Engines tab consumes it.
- Engines tab: platform card (detected id, GPUs, budgets), offers + reasons, lifecycle table (state, last-used = registry clock, evictions today = in-memory counter), queue, budget gauge (residents by source + pins). Models tab gains "serve via" column.

## CLI

`brainrouter swap status|ensure <model>|evict <model>|catalog list|import-llamaswap|add <spec.json>|rm <id>`; `brainrouter platform show`; `brainrouter pin add <id> --reserved-gib N|rm <id>`. Mutations via daemon; reads direct.

## Contracts kept

Shim-mode routing byte-identical (v1 floors: all selector/reviewer tests); selector vocabulary (L7 range); served-model namespace dual-resolvable; events additive; configs boot (catalog import additive, snapshot + dry-run per migration contract).

## Build order

platform probe + matrix (replace hardcoded ids) → registry + specs + llama-swap import (macro expansion + fidelity test) → Residency view (shim observer) → state machine + scheduler + flock (fake adapter) → adoption markers + restart scan → budget + pins + LRU → llama_cpp native adapter (shared shaping + parallel mode) → router ensure-queue → progress/ETA plumbing → Engines tab + CLI → docs.

## Acceptance (Ray)

(1) Fake-adapter lifecycle: ensure spawns; stem assert fail→Failed; crash→bounded restart; **N concurrent ensure(X) → one spawn, all attach, flock released on crash**; queue cap + cancel via inflight; LRU evicts least-recent-STREAM-COMPLETED, in-flight never evicted (catalog-id stamp match); start-ownership: foreign stem on port untouched; own child cleaned. (2) Budget: shim-resident model's native ensure ATTACHES (single residency, no second process); sized pins subtract at admission (fake over-budget with pinned reserved_gib → no-fit explicit); eviction sequence asserted; pins-over-budget ⇒ every ensure = explicit no-fit. (3) Adoption: daemon restart while native model serving (fake + llama_cpp) ⇒ adopt (no kill, no respawn), stale marker cleaned. (4) Shim parity + dual mode: v1 floors byte-identical; import lists every live llama-swap key (resolved args, same stem+ctx as shim serves — fidelity assertions; idempotent re-import, dry-run stable); shim-resident ⇒ attach-not-spawn; golden-body shim==native sanitization. (5) Platform probe on this host = strix-halo (real); fixture probes: gfx1100 ⇒ offers hipfire+llama.cpp, NOT strixite/gufo, reasons shown; unknown id ⇒ banner + all-toolboxes. (6) Progress: native load = indeterminate + elapsed + EWMA ETA persisted; fake scripted progress renders. (7) Events additive swap_state/swap_source/measured tokens; v1 parse intact; registry last-used feeds LRU (asserted via eviction order). (8) CLI surface, daemon-down refuse. (9) All prior suites green (workspace both modes, L1/L2/L3 acceptance, node, html-js).
