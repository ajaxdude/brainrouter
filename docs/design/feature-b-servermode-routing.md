# Feature B — route brainrouter requests to running Server-Mode backends

## Status
- Workflow state: `critic-revisions-required` — Dory round 1 **FAIL (7 blocking + 5 important)**. Feature B is a large multi-part feature, not a single routing tweak. Blockers: **B1** served-model name differs per backend (vLLM=repo; r9v=`R9V_SERVED_MODEL_NAME`/`qwen3.8-flash-next`; ds4/halogen set none) so `model_id` matching is wrong for 3/4; **B2** only vLLM is `openai_compatible=true` today (ds4/halogen/r9v unverified) → routing them needs per-backend compat verification; **B3** needs a lifecycle state machine (Starting/Ready/Stale + readiness probe + crash reconciliation; registry is in-memory, `snapshot()` shells `podman inspect` — unsafe on the hot path); **B4** "not running" error needs a persistent alias catalog / reserved namespace; **B5** `route_local` insertion affects auto/subs/reviewer + cloud-fallback bypasses it → needs an explicit routing/collision matrix + reserved-name rejection; **B6** provider construction/URL/**auth** unresolved (vLLM/r9v API keys must NOT be serialized into public `/api/serving-identities`); **B7** `Router` holds concrete `Arc<OpenAiProvider>` (no injectable trait) → tests need a provider abstraction; errors flow via deferred SSE after HTTP 200. Recommended narrowing (critic B2-opt-1): **vLLM-first**, other backends feature-gated pending per-backend served-name + compat contracts. Full verification needs a running backend server (no OOM-risky halogen serve).
- Change classification: **standard**, high blast radius — modifies the request routing hot path (a bug can break all local inference). Full rigor.
- Human review: user delegated ("do all of them", "work autonomously"). Isolated Dory critic + mean review still run; deploy is incremental with the strix build gate + live checks.

## Problem
Server-Mode backends (ds4/halogen/vllm/r9v) run as their own detached OpenAI servers (registered as serving-identities with an `endpoint`), but brainrouter can't *route* inference to them — a local `/v1/chat/completions` always goes to llama-swap (`route_local` → `self.llama_swap`). So a started halogen/ds4/r9v server is unreachable through brainrouter, unlike llama-swap models. This is the core of "brainrouter = llama-swap for these backends": once a backend server is running, a request for its model should be proxied to that server's endpoint.

## Goals / non-goals
Goals:
- When a request's `model` matches a **currently running** Server-Mode server, proxy the request to that server's `endpoint` (OpenAI-compatible) instead of llama-swap.
- Surface running Server-Mode models in `GET /v1/models` (and the routing model dropdowns) so clients can select them.
- Clear error when a Server-Mode model is requested but its server isn't running.

Non-goals:
- **On-demand auto-start / swap / idle-unload** (that's the separate "vision" stretch). Feature B requires the server already started via Server Mode.
- Starting any server here (esp. the OOM-risky ~124 GB halogen — do NOT start it as part of this work).
- ds4-via-llama-swap-wrapper (separate).

## Current system (verified)
- `src/router.rs`: `Router { llama_swap: Arc<OpenAiProvider>, … }`; `route_local` (~602) → `try_llama_swap` (~650) → `self.llama_swap.chat_completion(request)` (~691). `route_auto`/`route_cloud`/`route_resolved`/`route_with_choice`/`route_tagged`. `local_models()` (~190).
- `src/serving_identity.rs`: `ServingIdentity { toolbox_backend, compute_api, runtime_profile_id, endpoint, openai_compatible, registered_at }` — **no served model name today**. `ServingIdentityRegistry { register(container_name, identity), deregister(container_name), snapshot() }`. `openai_compatible_for_backend(backend)`.
- `src/server_mode.rs`: server start records the served model as the `LABEL_SERVER_MODEL` podman label = `req.model_id` (~318); status reads it back (~562). `-v <models_dir>:/models:ro`. `register_serving_identity` (`server.rs:3547`) builds+registers the identity at start — **has `req.model_id` available**.
- `src/server.rs`: `AppState.serving_identities: Arc<ServingIdentityRegistry>` (~179); `GET /v1/models` (~358) proxies **only** llama-swap `/v1/models` (`state.llama_swap_url`); `GET /api/serving-identities` (~1207).
- `OpenAiProvider` fronts one OpenAI base URL (the llama-swap provider). Need to confirm its constructor to build/cache one per Server-Mode endpoint.

## Requirements and acceptance criteria
R1. **Registry carries the served model name.** Add `served_model: String` (= the start request's `model_id`) to `ServingIdentity` (or a parallel map in the registry), populated in `register_serving_identity`. `deregister` on stop. — AC: after starting a server, the registry maps its `served_model → endpoint`; after stop, gone.

R2. **Routing.** A request whose resolved local model equals a running serving-identity's `served_model` (exact match) is proxied to that identity's `endpoint` via an OpenAI provider for that base URL (constructed on demand and cached by endpoint). The upstream request's `model` field is set to what that server expects (default: pass the same `served_model`; if servers need a fixed name, document it). Only identities with `openai_compatible == true` and a live/running container are eligible. — AC: a request for a running Server-Mode model reaches its endpoint (unit-tested with a stub provider/registry); a normal llama-swap model is unaffected (still → llama_swap).

R3. **Precedence + fallthrough.** Server-Mode match takes precedence over llama-swap for that exact model id; everything else routes exactly as today (auto/cloud/subs/llama-swap unchanged). A Server-Mode-looking model with **no running server** returns a clear error ("model X is a Server-Mode backend; start its server first"), NOT a silent llama-swap 404. — AC: not-running → explicit error; unrelated routing paths unchanged (existing router tests still pass).

R4. **Model discovery.** `GET /v1/models` unions llama-swap's models with running serving-identities' `served_model` ids (deduped). The dashboard routing dropdowns (fed by `/api/routing-models`) likewise include them so a user can pick a Server-Mode model as main/reviewer. — AC: `/v1/models` lists a running Server-Mode model.

R5. **Streaming + errors.** Proxying preserves streaming (SSE passthrough) and surfaces upstream errors like the llama-swap path (reuse the existing provider/forwarding machinery; do not hand-roll a second SSE pipeline). — AC: a streamed completion through a Server-Mode endpoint streams end-to-end.

R6. **No regression / safety.** The routing change is additive and guarded: if the registry is empty or no match, behavior is byte-for-byte the current path. No new panics on the hot path. — AC: full existing router/failover/stream test suites pass.

## Technical plan
- `serving_identity.rs`: add `served_model` to `ServingIdentity`; add `registry.find_by_served_model(model) -> Option<ServingIdentity>` (running + openai_compatible only, or let the caller check running).
- `server.rs::register_serving_identity`: set `served_model = req.model_id`; ensure deregister on stop.
- Router: add an optional `serving_identities` handle + an endpoint→provider cache (e.g. `Mutex<HashMap<String, Arc<OpenAiProvider>>>`). In the local-routing decision (before `try_llama_swap`), consult `find_by_served_model(model)`; if matched+running, forward via the cached provider; else current behavior. Keep the change localized to the local route; do not touch cloud/subs.
- `/v1/models` handler: after fetching llama-swap models, append running serving-identities' `served_model`s (dedup) before responding. Same for `/api/routing-models` local list.
- Provider construction: reuse `OpenAiProvider::new`-equivalent with the identity endpoint as base URL (confirm the exact constructor + whether it needs `/v1` suffix, matching `llama_swap_url` usage).

## Alternatives considered
- **Per-request `podman inspect` for the served model** — rejected: hot-path latency; the registry already knows it at start.
- **Rewriting llama-swap config to add Server-Mode models** — rejected: they're not llama.cpp; llama-swap can't run them.
- **On-demand start on cache miss** — deferred to the "vision" stretch (start + wait + route). B is route-to-already-running only.

## Detailed implementation
Files: `src/serving_identity.rs` (served_model field + finder + tests), `src/server.rs` (register set served_model, `/v1/models` union, `/api/routing-models` union, router wiring), `src/router.rs` (endpoint provider cache + local-route match + forward; unit tests with a stub registry+provider). No schema/migration. Order: registry field+finder+tests → router match+forward (stub-tested) → register wiring → model-listing union → dashboard dropdown. Build-gate on strix after router changes.

## Testing and evaluation
- Rust unit: `find_by_served_model` (match/no-match/not-running/not-openai); router local-route selects Server-Mode provider on match, llama_swap otherwise, and errors on not-running (inject a stub registry + stub providers — no network). `/v1/models` union dedup.
- Full existing suites (router/failover/stream/subs) must pass unchanged (R6).
- `cargo test --locked -- --test-threads=1`, clippy, `cargo build --release` on strix, `check-html-js.sh`.
- **Live** (strix): unit + listing checks are safe. End-to-end routing to a real endpoint requires a **running** Server-Mode server; the only downloaded model is the ~124 GB halogen bundle whose serve is OOM-gated — so DO NOT start it here. Instead verify: `/v1/models` unions correctly when a (hypothetical/stub or lightweight) identity is registered, and a request for a not-running Server-Mode model returns the explicit error. Full live inference routing is verified when the user starts a backend server.

## Security, privacy, reliability, and operations
- Endpoints are localhost backend servers brainrouter itself started. No new external exposure. Provider cache bounded by number of backends. Fail-safe: any lookup miss → existing llama-swap path.

## Rollout / rollback
- Additive; revert = single commit. Deploy via the runbook to strix master + GitHub; back up `benchmarks.sqlite3` first.

## Open questions
- Exact `OpenAiProvider` constructor + whether the endpoint needs a `/v1` suffix (confirm at implementation).
- Whether any Server-Mode server needs an explicit `--served-model-name` set = `model_id` at start so the upstream accepts that model name (may add a small server_mode.rs tweak). Confirm per backend during implementation; if needed, set it.

## Referenced files
`src/router.rs`, `src/serving_identity.rs`, `src/server_mode.rs` (start/register/label), `src/server.rs` (`/v1/models` ~358, `register_serving_identity` ~3547, `AppState` ~179, `/api/routing-models`), `src/config.rs` (`llama_swap.base_url`).

## Dory validation record
(pending)

## Human approval
Delegated ("do all of them"); Dory critic + mean review still performed; deploy incremental with live checks. Full end-to-end routing verification deferred to when a Server-Mode server is running (no OOM-risky serve as part of this).
