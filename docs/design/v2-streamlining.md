# brainrouter v2: from proxy with model selection to the model-operations app

## Status

State: `drafting` → submitted for Dory batch 1. Classification: **standard** (public API removals, config migrations, auth/credential handling, LAN security boundary — "Always standard"). Pace: **careful** (charter names no faster pace; auth/migration work). Charter: `docs/design/v2-streamlining.charter.md` v0. Target approval date: 2026-10-24. Review budget: 6 h. Time spent: ~1.5 h (Hank phases 1–2). Critic rounds used: 0.
Revisions:
v1 — 2026-10-21 — first Map: contracts, legs L0–L8, leg L1 (Purge) in full.

## Problem

The owner runs a fleet (Strix Halo gfx1151, fedora1 RX 7900 XT, fedora2, a mac) and today model work is scattered: GUI icons launch strixite per model on the Strix Halo, a hand-written icon script launches hipfire on fedora1, harnesses list and select models themselves, and API keys are hunted across YAML files, env vars, and chat messages (three messages burned in one session just locating a DeepInfra token). brainrouter v1 already proxies everything but wraps model choice in eight overlapping selection layers, hardcodes the strix-halo platform, and has no credential surface. The owner wants brainrouter to become the single app: it launches engines, swaps models inside them, keeps credentials, assigns role models (Main/Reviewer/Subagent), recommends engines per hardware, and lets any role run on any machine on the LAN.

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
- Model selection layers: role profiles (`src/routing_profile.rs`), presets, per-project pins (`src/project_pins.rs`), proxy selectors (`auto/local/cloud/cloud/<id>/subs/bare`), command-strip UI, CLI `routing-profile` (`src/cli.rs`), harness model lists (`src/install.rs` templates, `sync-omp`).
- Classifier: Bonsai external llama-server (`src/classifier.rs`, `src/bonsai_server.rs`), YAML `bonsai.*`, invariant "rewrite ⇒ bonsai" in `src/prompt_rewriter.rs` coupling.
- Toolbox/server-mode management: `src/server_mode.rs` (ds4/halogen/vllm/r9v/gufo detached podman servers; ds4+gufo are the kept set), `src/serving_identity.rs` registry, `src/model_downloads.rs` (halogen/r9v download + Prepare-PLE jobs), `src/toolbox_catalog/*` (vendored cockpit feed + gufo overlay; `SupportedServingBackend` enum, `CatalogBackendId` feed vocabulary), `src/llama_swap_register.rs` (GGUF→llama-swap config).
- Platform hardcoding: `src/server.rs` toolbox image default `strix-halo`; dashboard `CURRENT_PLATFORM_ID = 'strix-halo'` (`src/escalation/templates/main_dashboard.html`); no hardware detection.
- Credentials today: env-var names in YAML (`manifest.api_key_env`), `/etc/brainrouter/env`, plaintext `bridge.discord.token` (`src/config.rs`).
- Bridges: `src/bridge/*` (Discord/Signal cargo features); LLM path is `omp` subprocess → proxy; no in-process daemon coupling.
- Keepers: Sankey + conversation cards (`main_dashboard.html`), in-flight registry (`src/inflight.rs`), routing events (`src/routing_events.rs`), observability (`src/observability.rs`), review loop (`src/review/*`), MCP (`src/mcp_server.rs`).
- Tests: 391 Rust tests + 38 Node browser-logic tests (`scripts/test-benchmark-ui.cjs`); baseline green at `790fd9b`.

## Requirements and acceptance criteria

R1 (P0-Purge). Removing halogen/vllm/r9v leaves a daemon that builds, passes all tests, routes llama-swap/Manifest/gufo/ds4 traffic identically; catalog feed entries for halogen/vllm/r9v parse as browse-only. Test: full `cargo test` + `toolbox_catalog_check` + dashboard loads.
R2 (Benchmarks gate). Default build excludes the explorer behind a `benchmarks` cargo feature; explorer routes return 503-when-off as today's unavailable-storage path. Test: build without feature, request `/benchmarks` → documented degradation; with feature, all explorer tests pass.
R3 (P1-Plugins). Sidecar plugins over UDS JSON-RPC with manifest, supervisor, namespaced config/kv, grant-scoped secrets; bridge plugin (Discord+Signal) moves out of the core binary, behavior unchanged. Test: bridge plugin round-trip against a live daemon; `cargo tree` core shows no serenity.
R4 (P2-Roles). One harness model entry (`auto`) + `x-brainrouter-role: main|subagent|review` headers; role cards (Main/Reviewer/Subagent, global + per-project, Auto|pinned) replace presets/choosers/role-grammar; decision-model slot with kinds `heuristic` (default), `openai-compatible`, `structured-schema`, `webhook`, `plugin`; ≤2GB local budget; `bonsai.*` config auto-migrates. Test: role header routing matrix; migration boots v1 config with logged notice; sanitizer: explicit model ID preserved.
R5 (P3-Swap). `src/swap/` supervisor: model lifecycle states, launch specs, VRAM budget, in-flight-aware LRU, load queue, stem assertion, start-ownership; `EngineAdapter` trait; llama-swap remains an adapter (shim) during migration. Test: scripted fake-engine adapter suite (spawn/health/evict/queue).
R6 (P4-Engines). strixite adapter (conf registry, drain-gated cutover, self-provider pin) + hipfire adapter (card files, MTP stem, single port owner) on gfx1151/gfx1100; platform probe feeds the engine-compatibility matrix; icon fleets retired. Test: live swap on both hosts with stem assertion; negative probe test (no strixite offered on gfx1100).
R7 (P5-Credentials+Cloud). Credentials store (`credentials.json` 0600, masked APIs) + Credentials page (add/test/bind/audit); provider adapters (OpenAI-compat, Anthropic) with per-provider breakers; manifest→deepinfra(deepseek) reviewer binding as reference deployment; federation-lite: LAN provider endpoints + bearer token + `GET /api/peer-info`; role flow plan→fedora1/hipfire, dev→strixite, review→cloud verified end-to-end. Test: masked-value audit (no secret in any API/log/config-show); cross-host review session with Sankey LAN hop.
R8 (P6-P7). llama-swap removed at parity; mac (aarch64-apple-darwin) daemon + ds4-on-mac evaluated (Metal verified or llama.cpp-Metal fallback documented); distributed-role acceptance suite green.
R9 (Continuity). At all times: `v1-classic` tag + `v1` branch on origin; v1 YAML boots every v2 phase via auto-migration with `--dry-run` preview; CHANGELOG + per-phase PRD/README updates.

## Technical plan

Five subsystems, mapped to the existing tree: (1) **swap supervisor** (`src/swap/`) owns engine processes — the generalized, API-driven version of the owner's icon scripts, with their debugged invariants (single-instance lock, health, served-model stem assertion, stop-only-what-you-started, engine-validated launch args). (2) **Decision slot** replaces the Bonsai special-case: a trait with heuristic default and pluggable kinds; selection = features → decision (route+tier) → catalog candidate pool → ranked pick with chain fallback. (3) **Role cards** are the only human-facing model UI: Main/Reviewer/Subagent, each Auto (decision slot) or pinned, global + per-project. (4) **Credentials store** (+ page) is the only place a key value exists; providers/plugins/roles bind by id; LAN peers are just providers with bearer tokens. (5) **Plugin system** (sidecar + UDS RPC) hosts non-core capabilities, bridges first. `install` shrinks to one model entry per harness; `sync-omp` dies.

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
- **Selector vocabulary**: `auto/local/cloud/cloud/<id>/subs/bare-id` keep routing meaning through L4; user-facing only as role cards thereafter.
- **Served-model namespace**: catalog id is canonical from L4; `served_model` string match stays resolvable (migration).
- **Config migration framework**: one mechanism (`--dry-run`, logged, v1 boots), reused by every touching leg.
- **Credentials interface**: `resolve(credential_id) -> SecretString`, single implementation; plugins get `secrets.get` grant-scoped.
- **Event schema** (routing events → Sankey/cards/observability): additive fields only (role, decision, rationale, hop).
- **Rewrite invariant** restated: prompt-rewrite ⇒ decision slot enabled (any kind).

Legs (door type; demo):
- L0 v1 snapshot — done (tag `v1-classic` + branch `v1` pushed). Reversible.
- L1 Purge (halogen/vllm/r9v; benchmarks feature gate R2 pending owner answer; assumed feature-gate) — two-way (git-revert of one PR). Demo: daemon builds, tests green, dashboard shows ds4+gufo server mode only.
- L2 Plugins + bridge pilot — one-way (core build change). Demo: Discord + Signal via plugin binary, same commands.
- L3 One entry + role cards + decision slot + Credentials page — one-way (UI/config contracts). Demo: omp with single model, role header matrix, credentials page managing DeepInfra.
- L4 Swap supervisor + platform probe — one-way. Demo: scripted fake engine swapped by API; probe output on all three Linux hosts.
- L5 strixite + hipfire adapters — one-way. Demo: swap strixite models from dashboard, no GUI icons; hipfire launch on fedora1 with stem assertion.
- L6 Providers + federation-lite + Manifest absorbed — one-way. Demo: reviewer = deepinfra/DeepSeek-V4-flash from Credentials UI.
- L7 llama-swap removed — one-way (owner sign-off). Demo: native llama.cpp serves v1's model keys.
- L8 mac surface + distributed-role acceptance — one-way. Demo: plan→fedora1/hipfire, dev→strixite, review→cloud, one session.

**Leg L1 in full (P0-Purge).**
Components: `src/server_mode.rs` (halogen/vllm/r9v server builders/requests/containers removed), `src/server.rs` (their route blocks removed), `src/model_downloads.rs` (halogen/r9v payloads + Prepare-PLE + sha256 verification removed), `src/toolbox_catalog/*` (`SupportedServingBackend::{Halogen,Vllm,R9v}` + payloads/types removed; `CatalogBackendId` keeps parsing those strings — feed vocabulary), `src/serving_identity.rs` (three arms removed), `main_dashboard.html` (server-mode panels + download tabs), fixtures (`src/benchmark.rs` vllm serving-runtime rows, `src/cockpit_config.rs` halogen cases retargeted to ds4/gufo), tests suite-wide, PRD/README/design-record notes.
Contracts kept: R1 (byte-identical behavior for kept backends — ds4/gufo/llama_cpp/Manifest paths untouched), `toolbox_backend` historical rows stay (text column, no migration), vendored catalog feed parses unchanged, `/api/server-mode/{ds4,gufo}/*` unchanged.
Error behavior: requests to removed routes → 404 (they already 404 for unknown backend paths); catalog entries for removed backends show browse-only "not supported by this host" (matrix, L4 generalizes).
Build order: catalog types → server_mode → routes/downloads → dashboard → fixtures/tests → docs. Checkpoint: `cargo test` green between steps 1–2 is not required (single PR, Step 9 reviews whole).
Acceptance: R1 + R2 tests; grep gate: no `halogen|vllm|r9v` outside `CatalogBackendId` parsing, historical docs, and the feed asset.

## Testing and evaluation

Per leg: existing suite (`cargo test`, Node browser-logic) + leg-specific tests named in R1–R9; Ray writes acceptance tests from R# criteria before code, builder never edits them. Live checks (L4–L8) run on owner machines (strix-halo local, fedora1, fedora2, mac) with Bailey cards for spikes (ds4-Metal probe; LFM2.5-GGUF load test).

## Security, privacy, reliability, and operations

Credentials: 0600 file, masked in every API/log/preview; write paths loopback+CSRF-gated (existing pattern). LAN: opt-in bind, bearer token, constant-time compare, per-provider tokens; peer-info is read-only model+health. Plugins: user-level, no system ops, documented unsandboxed. Decision slot: prompt content sent to classifiers = feature summary, never full prompt to `webhook`/cloud kinds. Reliability: v1-restore via tag; migrations logged and previewable; supervisors backoff; engine cutover serialized.

## Rollout, migration, and rollback

Leg order per the leg list in "Detailed implementation"; each leg = one PR to `master` via `v2-p<N>-<name>` branch, brainrouter review gate (Step 9) before merge, phase tag `v2.0-pN`; `git revert -m 1 <merge>` is the per-phase rollback; `v1-classic` is the global rollback. Config auto-migration per touching leg with `--dry-run`. Per-user systemd service stays the deployment unit.

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

## Referenced files

- `docs/design/v2-streamlining.charter.md` — charter: pace, delegation, budget.
- `docs/design/v2-streamlining.history.md` — record, not a rule (gate history; do not read for verdicts).
- `PRD.md` — v1 shipped-boundary claims (component map, planned-not-shipped).
- `src/router.rs` — `route()`/`route_tagged` dispatch, `find_server_mode_identity` (server-mode match contract).
- `src/server.rs` — route table (server-mode/`api/project-pin`/toolbox), hardcoded `strix-halo` toolbox default.
- `src/server_mode.rs` — per-backend builders; ds4/gufo kept, halogen/vllm/r9v removed in L1.
- `src/model_downloads.rs`, `src/serving_identity.rs`, `src/toolbox_catalog/types.rs` (+ `gufo_overlay.rs`) — L1 removal surface; feed-vocabulary split.
- `src/routing_profile.rs`, `src/project_pins.rs` — role/pin store shapes migrated in L3.
- `src/classifier.rs`, `src/bonsai_server.rs`, `src/prompt_rewriter.rs` — decision-slot replacement target; rewrite invariant.
- `src/install.rs`, `src/config.rs` — harness templates (one entry), credential keys.
- `src/bridge/*` — L2 extraction subject.
- `src/escalation/templates/main_dashboard.html` — Sankey/cards (keep), `CURRENT_PLATFORM_ID` (L4), server-mode panels (L1).
- `scripts/test-benchmark-ui.cjs`, `tests/` — suite gates.
- `~/ai/projects/brainrouter/phase0-findings.md` — baseline measurements (if present).

## Dory validation record

(to be filled by batch 1)

## Human approval

(pending)
