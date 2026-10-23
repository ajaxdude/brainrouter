# brainrouter v2: from proxy with model selection to the model-operations app

## Status

State: L1 DONE (`v2.0-p1`), L2 DONE (`v2.0-p2`). L3 building (spec L3-v3 ratified, Ray 20-test suite red-committed). L4 spec v3 — leg rounds 2/2 used; round-2 blocking folded (cgroup-escape spawn via systemd-run scope, Residency-before-budget, spec_stale adoption, keeper additive authorization); builds after L3. Classification: **standard** (public API removals, config migrations, auth/credential handling, LAN security boundary — "Always standard"). Pace: **careful** (charter names no faster pace; auth/migration work). Charter: `docs/design/v2-streamlining.charter.md` v0. Target approval date: 2026-10-24. Review budget: 6 h (spent ~14 h — over, owner continued at each digest; standing read: proceed). Critic rounds: 2 of 3 (Map); per-leg 2/2 each, in history.
Revisions:
v1 — 2026-10-21 — first Map: contracts, legs L0–L8, leg L1 (Purge) in full.
v2 — 2026-10-21 — B1 credentials ordering (store lands L2, page L3); B2 state-snapshot rollback contract; I1 grep-gate migration exception; I2 L1 live-state checkpoint; I3 Referenced files; I4 LAN token scope; I5 heuristic contract; N1/N2 citation + contract-range fixes; 8-layer enumeration; test baseline verified green at a6e39f7.
v3 — 2026-10-21 — round-2 residuals: R7 page-timing contradiction (F1); grep-gate keeper/parent doc-comment exceptions (F2); `src/daemon.rs` referenced (F3); credentials-staging Decision log entry (F4); F5–F8 nits (pointer, duplication, baseline citation, date/type).
v3.1 — 2026-10-21 — diff-check residual: grep-gate exception extended to feed-vocabulary CODE constants (`schema_validate::KNOWN_BACKEND_IDS`, kept so the vendored feed parses); dedup nit in layer enumeration.
v4 — 2026-10-22 — L1 marked complete; L2 leg section added in full (plugins + bridge pilot + minimal credentials store); L1 full spec moved to history (record lives in the L1 log + git).
v5 — 2026-10-22 — L2 marked done; L3 leg spec file added (`v2-streamlining.L3.md`) with key contracts in the leg entry.
v6 — 2026-10-22 — L4-v3 keeper-additive authorization applied to Referenced files (`src/inflight.rs` named-additive changes); L4 spec pointer added.
v7 — 2026-10-23 — L5 design closed (READY); L6 spec drafted + selector-vocabulary contract amended with the provider-qualified federation form.

## Problem

The owner runs a fleet (Strix Halo gfx1151, fedora1 RX 7900 XT, fedora2, a mac) and today model work is scattered: GUI icons launch strixite per model on the Strix Halo, a hand-written icon script launches hipfire on fedora1, harnesses list and select models themselves, and API keys are hunted across YAML files, env vars, and chat (three messages burned locating one DeepInfra token). brainrouter v1 proxies everything but wraps model choice in eight overlapping selection layers (enumerated in Current system), hardcodes the strix-halo platform, and has no credential surface. The owner wants brainrouter to become the single app: it launches engines, swaps models inside them, keeps credentials, assigns role models (Main/Reviewer/Subagent), recommends engines per hardware, and lets any role run on any LAN machine.

## Goals and non-goals

Goals (owner-stated):
1. Harness configured once → brainrouter; the app owns model selection, swapping, subagent models, reviewer models.
2. Absorb llama-swap, then Manifest: engine adapters + cloud provider adapters.
3. Replace the strixite and hipfire icon fleets: API-driven launch and swap.
4. Pluggable decision model (≤2GB resident budget; Clef-contract schema; Bonsai becomes one preset).
5. Distributed roles: plan/dev/review bound to model endpoints on any LAN machine (hipfire, strixite, ds4, cloud).
6. Hardware-aware engine recommendations; never offer strixite/gufo on a 7900 XT.
7. Credentials UI (Manifest-style): paste, store, bind, test; nothing embedded in YAML/code.
8. v1 fully revertible at every point (`v1-classic` tag, `v1` branch pushed).

Non-goals: cross-host federation beyond bearer-token LAN endpoints; OS-sandboxed plugins; WASM in-process hooks (deferred); benchmark execution automation; multi-user daemon (one user per machine, as today).

## Current system

- Single Rust daemon (`src/main.rs`→`src/daemon.rs`), OpenAI + Anthropic wire on `:9099` (`src/server.rs`), routing in `src/router.rs` (Manifest | llama-swap | server-mode identities; `route()` dispatch at the `served_model` string match, `router.rs` `find_server_mode_identity`).
- Model selection is eight layers: (1) role profiles Main/Reviewer/Subagent (`src/routing_profile.rs`); (2) per-role backend choices auto|local|cloud; (3) explicit local/cloud model-id choices; (4) the presets (`src/routing_profile.rs` `RoutingPreset`: six + `Custom` sentinel); (5) per-project pins (`src/project_pins.rs`); (6) proxy selector strings (`auto/local/cloud/cloud/<id>/subs/bare`); (7) dashboard command-strip model controls; (8) CLI `routing-profile` plus harness model lists (`src/cli.rs`, `src/install.rs` templates, `sync-omp`).
- Classifier: Bonsai external llama-server (`src/classifier.rs`, `src/bonsai_server.rs`), YAML `bonsai.*`, invariant "rewrite ⇒ bonsai" enforced at the `POST /api/prompt-rewrite` handler (`src/server.rs`; refuses enable while Bonsai is down), with the bootstrap default in `src/daemon.rs`.
- Toolbox/server-mode management: `src/server_mode.rs` (ds4/halogen/vllm/r9v/gufo detached podman servers; ds4+gufo are the kept set), `src/serving_identity.rs` registry, `src/model_downloads.rs` (halogen/r9v download + Prepare-PLE jobs), `src/toolbox_catalog/*` (vendored cockpit feed + gufo overlay; `SupportedServingBackend` enum, `CatalogBackendId` feed vocabulary), `src/llama_swap_register.rs` (GGUF→llama-swap config).
- Platform hardcoding: `src/server.rs` toolbox image default `strix-halo`; dashboard `CURRENT_PLATFORM_ID = 'strix-halo'` (`src/escalation/templates/main_dashboard.html`); no hardware detection.
- Credentials today: env-var names in YAML (`manifest.api_key_env`), `/etc/brainrouter/env`, plaintext `bridge.discord.token` (`src/config.rs`).
- Bridges: `src/bridge/*` (Discord/Signal cargo features); LLM path is `omp` subprocess → proxy; no in-process daemon coupling.
- Keepers: Sankey + conversation cards (`main_dashboard.html`), in-flight registry (`src/inflight.rs`), routing events (`src/routing_events.rs`), observability (`src/observability.rs`), review loop (`src/review/*`), MCP (`src/mcp_server.rs`).
- Tests: 391 Rust tests + 38 Node browser-logic tests (`scripts/test-benchmark-ui.cjs`); baseline verified green at `a6e39f7` (full workspace suite, 2026-10-21).

## Requirements and acceptance criteria

R1 (P0-Purge). Removing halogen/vllm/r9v leaves a daemon that builds, passes all tests, routes llama-swap/Manifest/gufo/ds4 traffic identically; catalog feed entries for halogen/vllm/r9v parse as browse-only. Test: full `cargo test` + `toolbox_catalog_check` + dashboard loads.
R2 (Benchmarks gate). Default build excludes the explorer behind a `benchmarks` cargo feature (non-default, in `Cargo.toml`); explorer routes return 503-when-off as today's unavailable-storage path. Test: build without feature, request `/benchmarks` → documented degradation; with feature, all explorer tests pass.
R3 (P1-Plugins). Sidecar plugins over UDS JSON-RPC with manifest, supervisor, namespaced config/kv, grant-scoped secrets; bridge plugin (Discord+Signal) moves out of the core binary, behavior unchanged. Test: bridge plugin round-trip against a live daemon; `cargo tree` core shows no serenity. The minimal credentials store ships in L2 (see the Credentials ownership contract): the Discord token migrates out of v1 YAML into the store via the migration framework (Signal has no v1 YAML secret — signal-cli state stays in `bridge.signal.storage_path`) and reaches the plugin ONLY through grant-scoped `secrets.get`; acceptance includes a token round-trip store→plugin with the value absent from plugin config files.
R4 (P2-Roles). One harness model entry (`auto`) + `x-brainrouter-role: main|subagent|review` headers; role cards (Main/Reviewer/Subagent, global + per-project, Auto|pinned) replace presets/choosers/role-grammar; decision-model slot with kinds `heuristic` (default), `openai-compatible`, `structured-schema`, `webhook`, `plugin`. The `heuristic` kind is fully specified: inputs = resolved selector string, role header, project scope, request feature flags (tool definitions present, message count, prompt token estimate, code-block presence); outputs = the shared {route, tier} vocabulary; parity anchor = today's classifier-off routing (`route_auto` with Bonsai disabled, `src/router.rs`/`src/daemon.rs`) for existing selectors. Acceptance adds: heuristic outputs pinned for the full v1 selector matrix (auto/local/cloud/cloud-id/subs/bare). ≤2GB local budget; `bonsai.*` config auto-migrates. Test: role header routing matrix; migration boots v1 config with logged notice; sanitizer: explicit model ID preserved; heuristic matrix table.
R5 (P3-Swap). `src/swap/` supervisor: model lifecycle states, launch specs, VRAM budget, in-flight-aware LRU, load queue, stem assertion, start-ownership; `EngineAdapter` trait; llama-swap remains an adapter (shim) during migration. Test: scripted fake-engine adapter suite (spawn/health/evict/queue).
R6 (P4-Engines). strixite adapter (conf registry, drain-gated cutover, self-provider pin) + hipfire adapter (card files, MTP stem, single port owner) on gfx1151/gfx1100; platform probe feeds the engine-compatibility matrix; icon fleets retired. Test: live swap on both hosts with stem assertion; negative probe test (no strixite offered on gfx1100).
R7 (P5-Credentials+Cloud). (store exists since L2; the page UI shipped in L3 — this leg adds provider binding, rotation, and the cloud adapters) Credentials store (`credentials.json` 0600, masked APIs) + Credentials page (add/test/bind/audit); provider adapters (OpenAI-compat, Anthropic) with per-provider breakers; manifest→deepinfra(deepseek) reviewer binding as reference deployment; federation-lite: LAN provider endpoints + bearer token + `GET /api/peer-info`; role flow plan→fedora1/hipfire, dev→strixite, review→cloud verified end-to-end. Test: masked-value audit (no secret in any API/log/config-show); cross-host review session with Sankey LAN hop; store values exportable on owner command to env-line form so a reverted v1 can consume them (ops escape, 0600 file).
R8 (P6-P7). llama-swap removed at parity; mac (aarch64-apple-darwin) daemon + ds4-on-mac evaluated (Metal verified or llama.cpp-Metal fallback documented); distributed-role acceptance suite green.
R9 (Continuity). At all times: `v1-classic` tag + `v1` branch on origin; v1 YAML boots every v2 phase via auto-migration with `--dry-run` preview; CHANGELOG + per-phase PRD/README updates.

## Technical plan

Five subsystems, mapped to the existing tree: (1) **swap supervisor** (`src/swap/`) owns engine processes — the generalized, API-driven version of the owner's icon scripts, with their debugged invariants (single-instance lock, health, served-model stem assertion, stop-only-what-you-started, engine-validated launch args). (2) **Decision slot** replaces the Bonsai special-case: a trait with heuristic default and pluggable kinds; selection = features → decision (route+tier) → catalog candidate pool → ranked pick with chain fallback. (3) **Role cards** are the only human-facing model UI: Main/Reviewer/Subagent, each Auto (decision slot) or pinned, global + per-project. (4) **Credentials store** (+ page) is the only place a key value exists; providers/plugins/roles bind by id; LAN peers are just providers with bearer tokens. The store (values + resolve) is born in L2 beside its first consumer; the page in L3; providers bind in L6. (5) **Plugin system** (sidecar + UDS RPC) hosts non-core capabilities, bridges first. `install` shrinks to one model entry per harness; `sync-omp` dies.

## Architecture and flows

```
harness(auto, role header) → :9099 proxy → role resolve (global|project)
   ├─ pinned → endpoint (engine|provider|LAN peer)
   └─ Auto → decision slot → candidate pool → pick → endpoint
endpoint = EngineAdapter-managed process (strixite/hipfire/llama-server/gufo/ds4)
         | provider adapter (cloud, LAN peer) with credential-id → store lookup
Sankey/cards read the same events (role, decision, rationale, endpoint, hop).
```
State: role cards + pins (migrated store), `credentials.json`, catalog + per-host platform profile, engine launch specs, plugin manifests. All writes: atomic owner-only files (precedent `src/project_pins.rs`). Errors: selection chain falls to next candidate; decision parse-fail → heuristic; plugin crash → supervisor backoff, proxy unaffected; engine start fail → kill-hang recovery (codified from validated strixite practice).

## Alternatives considered

- **Keep roles/presets/pins, add app UI on top** — cheaper; rejected: the 8-layer grammar is the bloat the owner named; one grammar (role card → Auto|pinned) replaces all.
- **Cargo features as the plugin mechanism** (current bridges) — compile-time rebuild per optional; rejected for capabilities (kept temporarily for R2 benchmarks gate as a migration step).
- **libloading/WASM plugins v1** — ABI lockstep / long-running socket bots don't fit; WASM deferred for in-process hooks.
- **Full federation/control plane** — rejected for v2 scope: bearer-token LAN providers give the distributed-role use case with the existing provider machinery.
- **Absorb Manifest first** — its cloud routing is useful until P5; provider adapters replace it; order keeps owner working through the migration.
- **Rewrite in a fresh repo** — rejected: 391 green tests, migration machinery, and v1-restore requirements all favor in-place phased evolution.

## Detailed implementation

Cross-leg contracts (do not break without a design revision):
- **Selector vocabulary**: `auto/local/cloud/cloud/<id>/cloud/<provider>/<id>/subs/bare-id` keep routing meaning through L7 — `cloud`/`cloud/<id>` rebind to provider adapters at L6 (qualified form = federation), `local`/`subs`/bare-ids to the swap supervisor at L7; user-facing only as role cards from L3.
- **Served-model namespace**: catalog id is canonical from L4; `served_model` string match stays resolvable (migration).
- **Config migration framework**: one mechanism (`--dry-run`, logged, v1 boots); **every leg that migrates or creates a state file first snapshots the pre-migration state (owner-only, versioned, same atomic-write machinery); restore is a named command per leg** — reverse path includes credentials export to env-line form so reverted v1 can read them.
- **Credentials interface**: `resolve(credential_id) -> SecretString`, single implementation; plugins get `secrets.get` grant-scoped.
- **Credentials ownership**: the minimal store (values, 0600, `resolve()`, `secrets.get`) lands in L2 with its first consumer (bridge plugin); the Credentials page lands in L3; provider binding/rotation UI in L6.
- **Event schema** (routing events → Sankey/cards/observability): additive fields only (role, decision, rationale, hop).
- **Rewrite invariant** restated: prompt-rewrite ⇒ decision slot enabled (any kind).

Legs (door type; demo):
- L0 v1 snapshot — done (tag `v1-classic` + branch `v1` pushed). Reversible.
- L1 Purge — **done** (merged `e90bffe`, tag `v2.0-p1`; Ray 6/6, Bruce GO 0 blocking, net −3,579 lines).
- L2 Plugins + bridge pilot + credentials store — **done** (merged `b9319d2`, tag `v2.0-p2`; Ray 14/14 load-verified, Bruce NO-GO→fixed→GO, 7 P3 follow-ups in history).
- L3 One entry + role cards + decision slot + Credentials page — one-way (UI/config contracts). Demo: omp with single model, role header matrix, credentials page managing DeepInfra.
- L4 Swap supervisor + platform probe — one-way. Demo: scripted fake engine swapped by API; probe output on all three Linux hosts.
- L5 strixite + hipfire adapters — one-way. Demo: swap strixite models from dashboard, no GUI icons; hipfire launch on fedora1 with stem assertion.
- L6 Providers + federation-lite + Manifest absorbed — one-way. Demo: reviewer = deepinfra/DeepSeek-V4-flash from Credentials UI.
- L7 llama-swap removed — one-way (owner sign-off). Demo: native llama.cpp serves v1's model keys.
- L8 mac surface + distributed-role acceptance — one-way. Demo: plan→fedora1/hipfire, dev→strixite, review→cloud, one session.

**Leg L1 — complete** (full spec + review disposition: `docs/design/v2-streamlining.L1.log.md`; merged `e90bffe`, tag `v2.0-p1`).

**Leg L2 in full** — spec (post-review, I1–I7 folded): `docs/design/v2-streamlining.L2.md` (per-leg spec files; the Map keeps contracts + leg list). Key contracts: `initialize` handshake (token/peer-uid/grants), crash-loop terminal `failed` state, credentials migration = Discord token only (Signal has no YAML secret), bridge config schema MINUS the token key, Signal keeps v1 direct llama-swap listing (parity), plugin state dir under `$XDG_STATE_HOME`, per-transport pause survives as plugin config, `bridges` verb + bridge endpoints removed.

**Leg L3 in full** — spec: `docs/design/v2-streamlining.L3.md` (leg-scoped review in progress). Key contracts: role resolution (explicit selectors bypass cards; project>global>default; header + `subs` alias), decision slot with guardrail clamps + low-confidence fallback, Bonsai auto-migrates to `openai-compatible` with behavior continuity at the migration boundary, credentials page on the unchanged L2 store, one-entry install templates, events additive with decision fields.

**Leg L4 in full** — spec: `docs/design/v2-streamlining.L4.md` (v3). Key contracts: Residency view (shim `/running` observer + native registry + containers + sized pins; attach-not-spawn, single residency, each model counted once, Residency BEFORE budget), platform probe in catalog vocabulary (feed lacks gfx1100 → generic id + visible banner), child-held flock, systemd-scope cgroup escape for adoption, engine-validated launch specs, llama-swap config import with macro expansion.

**Leg L5 in full** — spec: `docs/design/v2-streamlining.L5.md` (v3; Step 6 closed, READY). Key contracts: Active-Consumer Set (proxy rows + in-process router streams + grace span), single-action override, pid-scoped cutover with external-engine Residency observer (icon squat structurally closed), conf-generated ports + `min_mem_gib` gate data, version-keyed flag schema, operator-gated icon retirement.

**Leg L6 in full** — spec: `docs/design/v2-streamlining.L6.md` (v4, leg rounds exhausted + diff-check fold). Key contracts: providers.json (credential-id bindings, per-provider breakers, `auto` manifest kind), cloud/<id> resolution with v1-parity passthrough, parity corpus (7 v1 behaviors, v1-classic vs mocks), v2.0-p6 migration, LAN second listener (auth/rate/expose/budget), role-header forwarding = distributed roles, hop events.

## Testing and evaluation

Per leg: existing suite (`cargo test`, Node browser-logic) + leg-specific tests named in R1–R9; Ray writes acceptance tests from R# criteria before code, builder never edits them. Live checks (L4–L8) run on owner machines (strix-halo local, fedora1, fedora2, mac) with Bailey cards for spikes (ds4-Metal probe; LFM2.5-GGUF load test).

## Security, privacy, reliability, and operations

Credentials: 0600 file, masked in every API/log/preview; write paths loopback+CSRF-gated (existing pattern). LAN: **second listener only** (primary loopback surface byte-unchanged), opt-in bind, bearer token (store-managed, rotate + revoke), constant-time compare, per-token rate limit, expose list double-enforced (raw + resolved), per-provider monthly budget guard (warn/stop); peer-info is read-only model+health+tier, enumerated to any token holder (accepted). **Role resolution over LAN:** a LAN-authenticated inbound request resolves the PEER's role cards — the distributed-role mechanism (R7), authorized by the peer's bearer token; role ADMIN APIs (writes) and all dashboard/admin endpoints stay loopback-only, never registered on the LAN listener. Bearer-over-HTTP-on-LAN is an explicitly accepted risk (trusted home network), stated so the owner-reserved LAN-enable nod carries it. Plugins: user-level, no system ops, documented unsandboxed. Decision slot: prompt content sent to classifiers = feature summary, never full prompt to `webhook`/cloud kinds. Reliability: v1-restore via tag; migrations logged and previewable; supervisors backoff; engine cutover serialized.

## Rollout, migration, and rollback

Leg order per the leg list in "Detailed implementation"; each leg = one PR to `master` via `v2-p<N>-<name>` branch, brainrouter review gate (Step 9) before merge, phase tag `v2.0-pN`; `git revert -m 1 <merge>` is the per-phase rollback; `v1-classic` is the global rollback. Config auto-migration per touching leg with `--dry-run`. Per-user systemd service stays the deployment unit. Per-leg state restore: `brainrouter migration restore <phase-tag>` (proposed command; snapshot per contract) — reverting code alone is never the whole restore.

## Risks and mitigations

- Sub-2GB decision models misroute → schema + few-shot prompts, heuristic guardrail clamps, confidence-gated fallback, replay-eval loop over routing events; cloud/webhook kind for full fidelity.
- strixite swap kills brainrouter's own provider (this harness runs on it) → self-provider pin + drain + explicit swap policy; no auto-swap of pinned models.
- VRAM contention (classifier + swaps) → supervisor budget counts resident decision model; pinned above LRU.
- llama-swap parity gap (load progress, registration) → adapter shim kept until checklist R8 passes; migration keeps import.
- Plugin RPC churn → minimal v1 surface, additive-only, `min_daemon_version`.
- ds4 on macOS unverified → spike (Bailey) before L8 commit; llama.cpp-Metal documented fallback.

## Open questions

Q1 (owner): benchmarks disposition confirmed — feature-gate (assumed) vs `brainrouter-lab` binary? Q2: default decision model — heuristic + 1-click LFM2.5-230M (assumed) vs 1.2B/2.6B-Q4? Q3: LAN auth bearer (assumed) vs mTLS? Q4: fedora1 hipfire unit — user-service takeover (assumed) vs sudo-system-unit? (Assumed = Marlin takes the default when the question's wait expires; reserved items in the charter.)

## Decision log

- Plugins = sidecar processes, not cargo/libloading/WASM — approved (owner asked plugin feature; isolation + language-agnostic) — 2026-10-21.
- Decision-model slot ≤2GB local, Clef-contract schema, Bonsai = preset — approved (owner budget statement) — 2026-10-21.
- Server-mode backends = ds4+gufo only; halogen/vllm/r9v deleted — approved (owner directive) — 2026-10-21.
- Role cards keep Reviewer/Subagent assignments (v2 keeps human role assignment; Auto is the smart default) — approved after owner clarified "app" framing (earlier "delete all selection UI" reading superseded) — 2026-10-21.
- Federation-lite (bearer LAN providers), no control plane — approved (owner distributed-role use case; scope control) — 2026-10-21.
- v2 in-place on master with v1 tag, phase-per-PR — approved (owner revert requirement) — 2026-10-21.
- Credentials UI-first, `env/file` refs as ops escape — approved (owner: Manifest-style UI, nothing in code) — 2026-10-21.
- Credentials delivery staged store L2 / page L3 / binding+rotation L6 — from batch-1 critic B1 (store ships with its first consumer, the bridge plugin) — 2026-10-21.
- Per-provider monthly budget guard (price × usage tokens; warn/stop; stop fires the managed-hop fallback): owner cost-safety for direct cloud + LAN tokens — 2026-10-23.
- LAN inbound requests resolve the PEER's role cards (distributed roles); supersedes the plain reading of "role resolution never exposed over LAN" — admin APIs stay loopback-only — 2026-10-23.

## Referenced files

- `docs/design/v2-streamlining.charter.md` — charter: pace, delegation, budget.
- `PRD.md` — v1 shipped-boundary claims (component map, planned-not-shipped).
- `src/router.rs` — `route()`/`route_tagged` dispatch, `find_server_mode_identity` (server-mode match contract).
- `src/server.rs` — route table (server-mode/`api/project-pin`/toolbox), hardcoded `strix-halo` toolbox default.
- `src/server_mode.rs` — per-backend builders; ds4/gufo kept, halogen/vllm/r9v removed in L1.
- `src/model_downloads.rs`, `src/serving_identity.rs`, `src/toolbox_catalog/types.rs` (+ `gufo_overlay.rs`) — L1 removal surface; feed-vocabulary split.
- `src/routing_profile.rs`, `src/project_pins.rs` — role/pin store shapes migrated in L3.
- `src/classifier.rs`, `src/bonsai_server.rs`, `src/prompt_rewriter.rs` — decision-slot replacement target; rewrite invariant.
- `src/daemon.rs` — bonsai bootstrap default; classifier-off routing context for the R4 heuristic parity anchor.
- `src/install.rs`, `src/config.rs` — harness templates (one entry), credential keys.
- `src/bridge/*` — L2 extraction subject.
- `src/escalation/templates/main_dashboard.html` — Sankey/cards (keep), `CURRENT_PLATFORM_ID` (L4), server-mode panels (L1).
- `scripts/test-benchmark-ui.cjs`, `tests/` — suite gates.
- `src/benchmark.rs` (`RunQuery`/serving-runtime fixtures), `src/cockpit_config.rs` (halogen round-trip cases) — L1 fixture retargets.
- `Cargo.toml` — bridge features now, `benchmarks` feature L1.
- `src/bin/toolbox_catalog_check.rs` — catalog structural gate (R1 test).
- `migrations/0003_toolbox_serving_dimension.sql` — frozen schema vocabulary; grep-gate exception, DO NOT EDIT.
- `src/cli.rs` — `routing-profile` CLI surface (L1 route removals, L3 replacement).
- `src/llama_swap_register.rs` — GGUF→llama-swap import; L7 catalog source.
- `docs/design/v2-streamlining.L7.md` — leg L7 spec (llama-swap absorbed; native-mode flip, parity checklist, deletion).
- `docs/design/v2-streamlining.L8.md` — leg L8 spec (mac surface, ds4-Metal spike gate, fleet-tier distributed roles).
- `docs/design/v2-streamlining.L6.md` — leg L6 full spec (providers, Manifest absorbed, LAN federation).
- `docs/design/v2-streamlining.L3.md` — leg L3 full spec (roles, decision slot, credentials page, one-entry).
- `docs/design/v2-streamlining.L5.md` — leg L5 full spec (strixite + hipfire adapters, Active-Consumer protection).
- `docs/design/v2-streamlining.L4.md` — leg L4 full spec (swap supervisor, platform probe, residency view).
- `docs/design/v2-streamlining.L2.md` — leg L2 full spec (per-leg detail files keep this Map under the length limit).
- keeper modules: `src/routing_events.rs`, `src/observability.rs` read-only for regression; `src/inflight.rs` — additive changes authorized where a leg spec names them (L4: catalog-id stamp, loading rows, registry clock hook).

## Dory validation record

comprehension (Step 5 PASS / Step 5b PASS) — v1 @ a6e39f7 — batch 1
critic — v1 @ a6e39f7 — 2 blocking, 5 important, 2 nits — revisions required (v2)
critic (scoped round 2) — v2 @ 20de00e — 0 blocking, 4 important, 4 nits — Step 6 closed; all fixed in v3
readiness (Step 7) — v3 @ fa6ddf4 — **READY**, zero blockers; grep census confirms acceptance-gate completeness; builder notes in history
diff check v3 — 20de00e..fa6ddf4 — all 8 findings fixed; 1 important residual (code-constant gate gap, fixed v3.1), 1 nit (fixed)

## Human approval

Approved by the owner (chat, 2026-10-21): design v3 @ `fa6ddf4`, all Dory gates passed (comprehension PASS/PASS; critic round 1 blocking fixed in v2; scoped round 2 closed Step 6, 0 blocking; readiness READY; diff check v3 clean after v3.1). Build L1 authorized on `v2-p1-purge`; merges to `master` remain owner-reserved per charter v0.
