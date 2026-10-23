This file is part of the design: the Map lives in `v2-streamlining.md`; this is leg L5's full spec (drafted while L3 completes; leg-scoped review pending).

# Leg L5 in full — strixite + hipfire adapters

Goal (R6): the icon fleets die. strixite (gfx1151) and hipfire (gfx1100) become engines the swap supervisor launches, swaps, and healths through brainrouter — same flow on every machine. L4 rails (adapters, launch specs, Residency view, adoption, budget, queue, stem assertion) carry the weight; L5 adds the two adapters + the cutover mechanics + pilots.

## strixite adapter

- Registry: strixw model dirs `~/models/<name>/` — required files per engine version (weights.strixw, vision.strixw, ngram table, tokenizer/config/generation_config/preprocessor_config, sha256.txt); file-presence validation at launch-spec build (v0.4.0 vision-required; missing file = Failed pre-spawn, not engine crash). Conf registry: named profiles (base/swift × capacity: `strix-server.conf` 256k desktop, `-headless` 512k dedicated — capacity metadata from the conf, exposed in the catalog).
- Launch: `strix_server --config <conf> --weights <weights>` (validated via `--print-config` at spec-build time — no GPU boot needed to validate), spawn detached via systemd scope (L4 adoption), endpoint `:5300`-family (port per model, registry-allocated; the single-server engine means ONE strixite resident at a time — the adapter declares `exclusive: true`, scheduler treats it as full-device class: any ensure of a second strixite model = swap, evict-then-spawn, never co-resident).
- Cutover state machine (codified from validated practice, binding): SIGTERM → poll `pgrep -f strix_server` gone → sleep 2 → `MemAvailable` ≥ threshold (from conf capacity class, ~90 GiB for 70 GB weights) → spawn → poll `/v1/models` for the expected stem (~30 s load) → Ready. Hung-start recovery: failed spawn that keeps holding GTT → SIGTERM, escalate to kill -9 after grace, re-run drain gate. Serialized per-device swap lock (one cutover at a time per GPU).
- **Self-provider pin (the brainrouter-lives-on-strixite rule):** the model currently serving brainrouter's own harness traffic (from in-flight rows on the proxy endpoints whose endpoint = strixite) is auto-pinned: scheduler NEVER swaps it as an eviction victim; explicit `swap ensure` of another strixite model returns `blocked:self-provider` with the dashboard showing WHY + an explicit override (owner confirmation gesture: `pin rm` then ensure). Pin list is the L4 sized-pins mechanism with `reserved_gib` auto-filled from the conf class.
- Load progress: strixite exposes `/v1/models` readiness only (no PP endpoint guarantee) → indeterminate + EWMA ETA (L4 native path).

## hipfire adapter

- Registry: card files `~/.hipfire/models/` (`*.mq4` + auto-loaded `.mtp` sidecar by stem — spec records card path, sidecar presence validated); launch = `hipfire serve` with engine-validated flag schema (MTP auto-loads — NO speculative `--mtp` flag; `--spec auto` exists; flag schema owned by the adapter, `serve --help`-derived at first registration, stored in spec).
- Port ownership: single-owner rule — the supervisor is the ONLY owner of `:8080` on fedora1; the pre-existing `hipfire-serve.service` (system, unit-scope) is resolved at install: Q4 default = migrate to a brainrouter-owned user-scope service (supervisor spawns/stops it as a resident, marked external-pinned so it survives daemon restarts; `systemctl is-active hipfire-serve` detected at boot → Residency view shows it as external resident, attach-not-spawn; the "system engine squats the port" contamination mode becomes structurally impossible: squat detection = Residency view stem mismatch at expected endpoint → Failed + banner).
- Context math: `prompt_tokens + max_tokens ≤ max-seq` validated pre-forward (measured tokens from L3 handoff; estimator fallback), 400 surfaced with the shortfall (the skill-burned "HTTP 400 on all engines" mode, made explicit).
- Cutover: stop = SIGTERM + port-free gate (`/health` must FAIL before next spawn — the bench-hygiene `port_free()` invariant); start → poll `/v1/models` for expected stem. Same exclusivity class as strixite (one resident per device).

## Fleet wiring

- Per-host profiles (config files, per-machine): strix-halo = strixite + gufo + llama_swap_shim + bonsai pins; fedora1 = hipfire + llama_cpp; fedora2 = strixite (prod) + deepinfra provider (L6). Engines tab shows adapter offers from the platform probe (strixite/gufo offered ONLY on gfx1151; hipfire on gfx9/10/11).
- Retire: strixite GUI icons + fedora1 `omp-launch.sh`/hipfire icon — UPGRADE notes give the one-time steps (stop icons, register models, set pins). The OMP-window-before-load behavior (harness opens while model loads) is L4's queue UX: the harness request queues at "loading", the window opens first, first token when Ready.

## Contracts kept

L4 rails unchanged (adapters are data + a trait impl; scheduler semantics identical); shim/gufo/ds4/llama_cpp byte-identical; strixite/hipfire models enter the Residency view (external/strixite source), budget counts them (conf-class reserved_gib), stem assertion scopes to spawn; icon-era invariants listed in the Map (stem assertion, port-free, MemAvailable gate, start-ownership, engine-validated flags) are adapter code, testable via scripted fake per adapter schema.

## Build order

strixite adapter (registry, conf profiles, cutover machine, exclusivity) → self-provider pin → hipfire adapter (cards, flag schema, service migration path, port-free) → fleet profiles (strix-halo first, fedora1 second) → Engines tab adapter states + swap UI (ensure/evict with blocked:self-provider + override) → pilots (live swaps on both hosts) → docs/UPGRADE.

## Acceptance (Ray)

Fixture tier (fake-binary adapters — scripted processes emulating startup/stem/GTT/port behavior): (1) strixite cutover sequence: evict-then-spawn ordering, drain gate (spawn blocked while process present), MemAvailable gate (fixture gate source; below-threshold = blocked, above = spawn), stem assert, hung-start kill -9 recovery, serialized swap lock (concurrent ensures = one cutover). (2) exclusivity: ensure(second strixite model) = swap, never co-resident. (3) self-provider: model serving active in-flight = pinned; ensure-other = blocked:self-provider; override path works; in-flight rows show the ETA. (4) hipfire: card + sidecar validation (missing sidecar warns, spec records), flag schema rejects unknown flags at spec build, port-free gate (fixture: /health answering = no spawn), context math 400 with shortfall. (5) Residency: external strixite/hipfire residents (fixture) show source=external, attach-not-spawn, squat detection (stem mismatch at expected endpoint → Failed + banner). Host tier (owner machines, Bailey cards): (6) real strixite swap on the Strix Halo: dashboard-driven model A→B cutover, drain gate timings logged, self-provider pin protects the live session; (7) fedora1: supervisor starts/stop/serves the Swift card via :8080 with the system unit migrated, harness request queues during load. (8) All prior suites green (L1–L4 acceptance, workspace both modes, node, html-js).
