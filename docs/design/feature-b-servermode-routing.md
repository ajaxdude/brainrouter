# Feature B — route brainrouter requests to running Server-Mode backends

## Status
- Workflow state: `approved-for-implementation` — coordinator resolved Dory blockers B1–B7 into the MVP below and explicitly approved implementation. v2 — 2026-09-27 — narrowed Feature B to additive routing only for running OpenAI-compatible Server-Mode identities, with no-match fallthrough to llama-swap unchanged.
- Change classification: **standard**, high blast radius — modifies the request routing hot path (a bug can break all local inference). Full rigor.
- Human review: coordinator approved MVP implementation scope on 2026-09-27 after resolving Dory blockers.

## Problem
Server-Mode backends run as their own detached servers and are registered with a serving identity, but brainrouter's local `/v1/chat/completions` path currently always falls through to llama-swap. For the subset that is verified OpenAI-compatible (gufo and vLLM today), a request for the model name that backend actually serves should be proxied to that running backend. If no running compatible identity exactly matches, the local routing path must remain byte-for-byte equivalent to today's llama-swap behavior.

## Goals / non-goals
Goals:
- When a local chat request's `model` exactly matches a running serving identity whose `openai_compatible == true`, proxy to that identity's OpenAI-compatible endpoint instead of llama-swap.
- Surface running OpenAI-compatible Server-Mode served model names in `GET /v1/models` so clients can select them.
- Preserve zero-regression behavior: no identity, no exact match, or non-OpenAI-compatible backend falls through to llama-swap unchanged.

Non-goals:
- **On-demand auto-start / swap / idle-unload** (that's the separate "vision" stretch). Feature B requires the server already started via Server Mode.
- Starting any server here (esp. the OOM-risky ~124 GB halogen — do NOT start it as part of this work).
- ds4-via-llama-swap-wrapper (separate).
- A "not-running Server-Mode model" error, reserved namespace, collision matrix, persistent alias catalog, on-demand start, readiness state machine, and non-compatible backend routing are deferred follow-ups (resolves B3/B4/B5 by preserving fallthrough and avoiding `snapshot()` on the hot path).

## Current system (verified)
- `src/router.rs`: `Router { llama_swap: Arc<OpenAiProvider>, … }`; `route_local` (~602) → `try_llama_swap` (~650) → `self.llama_swap.chat_completion(request)` (~691). `route_auto`/`route_cloud`/`route_resolved`/`route_with_choice`/`route_tagged`. `local_models()` (~190).
- `src/serving_identity.rs`: `ServingIdentity { toolbox_backend, compute_api, runtime_profile_id, endpoint, openai_compatible, registered_at }` — **no served model name today**. `ServingIdentityRegistry { register(container_name, identity), deregister(container_name), snapshot() }`. `snapshot()` shells out to podman and must not be used on the routing hot path. `openai_compatible_for_backend(backend)` returns true for llama_cpp, vLLM, and gufo; false for ds4, halogen, and r9v.
- `src/server_mode.rs`: server start records the served model as the `LABEL_SERVER_MODEL` podman label = `req.model_id` (~318); status reads it back (~562). `-v <models_dir>:/models:ro`. `register_serving_identity` (`server.rs:3547`) builds+registers the identity at start — **has `req.model_id` available**.
- `src/server.rs`: `AppState.serving_identities: Arc<ServingIdentityRegistry>` (~179); `GET /v1/models` (~358) proxies **only** llama-swap `/v1/models` (`state.llama_swap_url`); `GET /api/serving-identities` (~1207).
- `OpenAiProvider` fronts one OpenAI base URL (the llama-swap provider). Need to confirm its constructor to build/cache one per Server-Mode endpoint.

## Requirements and acceptance criteria
R1. **Registry carries the served model name.** Add `served_model: String` to `ServingIdentity`, populated in `register_serving_identity`. Backend values: gufo ⇒ `request.model_id`; vLLM ⇒ resolved HF repo (`custom_repo` if set, else catalog entry `.repo`); ds4 ⇒ `request.model_id`; halogen ⇒ `request.bundle_id` (bookkeeping-only equivalent because the request shape has no `model_id`); r9v ⇒ `request.package_id` (bookkeeping-only equivalent because the request shape has no `model_id`). `deregister` on stop. — AC: after starting a server, the registry maps its served model to the endpoint; after stop, gone.

R2. **Hot-path finder.** Add `ServingIdentityRegistry::find_by_served_model(&self, model: &str) -> Option<ServingIdentity>` using only the in-memory map, matching only `openai_compatible == true` and exact `served_model == model`. Add `openai_compatible_served_models(&self) -> Vec<String>` for discovery, deduped and in-memory only. — AC: unit tests cover match, no-match, non-compatible exclusion, exact-only matching, and deduped served-model listing; no `snapshot()`/podman call on routing hot path.

R3. **Precedence + fallthrough.** In `route_local` only, before `try_llama_swap`, exact Server-Mode matches take precedence over llama-swap. Everything else, including a Server-Mode model that is not currently registered/running, falls through to llama-swap exactly as today. Do not add a not-running error or reserved-name behavior in MVP. — AC: unrelated routing paths unchanged; existing router/failover/stream tests still pass.

R4. **Model discovery.** `GET /v1/models` unions llama-swap's models with `openai_compatible_served_models()` (deduped). `/api/routing-models` can remain unchanged unless cheap to extend. — AC: `/v1/models` lists a running compatible Server-Mode served model.

R5. **Streaming + errors.** Proxying preserves streaming (SSE passthrough) and surfaces upstream errors like the llama-swap path (reuse the existing provider/forwarding machinery; do not hand-roll a second SSE pipeline). — AC: a streamed completion through a Server-Mode endpoint streams end-to-end.

R6. **No regression / safety.** The routing change is additive and guarded: if the registry is empty or no match, behavior is byte-for-byte the current path. No new panics on the hot path. — AC: full existing router/failover/stream test suites pass.

## Technical plan
- `serving_identity.rs`: add `served_model` to `ServingIdentity`; add in-memory `find_by_served_model(model) -> Option<ServingIdentity>` and `openai_compatible_served_models() -> Vec<String>`.
- `server.rs::register_serving_identity`: accept `served_model: String`; populate from each backend's resolved served-name contract; ensure deregister on stop remains unchanged.
- Router: add an optional `serving_identities` handle + an endpoint→provider cache (`tokio::sync::Mutex<HashMap<String, Arc<OpenAiProvider>>>`). In `route_local` only, consult `find_by_served_model(model)` before `try_llama_swap`; if matched, forward via cached `OpenAiProvider::new(format!("server-mode:{backend}"), format!("{}/v1", endpoint), None)`; else current behavior exactly.
- `/v1/models` handler: after fetching llama-swap models, append compatible serving identities' `served_model`s with dedup.
- Provider construction: reuse `OpenAiProvider` for streaming/errors; identity endpoint is stored without `/v1`, so append `/v1` when constructing the provider. No API key is stored or serialized in `ServingIdentity`.

## Alternatives considered
- **Per-request `podman inspect` for the served model** — rejected: hot-path latency; the registry already knows it at start.
- **Rewriting llama-swap config to add Server-Mode models** — rejected: they're not llama.cpp; llama-swap can't run them.
- **On-demand start on cache miss** — deferred to the "vision" stretch (start + wait + route).
- **Not-running errors / reserved namespace / collision matrix** — deferred to preserve Feature B's fail-safe MVP: no match means llama-swap unchanged.
- **Routing ds4/halogen/r9v** — rejected for MVP because their `openai_compatible` classification is false today; they still record `served_model` for bookkeeping, but `find_by_served_model` excludes them naturally. B is route-to-already-running only.

## Detailed implementation
Files: `src/serving_identity.rs` (served_model field + finder/list helpers + tests), `src/server.rs` (register signature/call sites, `/v1/models` union), `src/server_mode.rs` (make the existing vLLM served-repo resolver callable from `server.rs`), `src/router.rs` (optional registry handle + endpoint provider cache + local-route match/forward + pure helper test), `src/daemon.rs` (wire `AppState.serving_identities` into `RouterArgs`). No schema/migration. Order: design update → registry helpers/tests → vLLM resolver visibility → registration call sites → router routing/cache → `/v1/models` union → validation.

## Testing and evaluation
- Rust unit: `find_by_served_model` (match/no-match/not-openai/exact-only) and `openai_compatible_served_models` dedup; pure router helper confirms provider name/base URL construction with `/v1`.
- Full existing suites (router/failover/stream/subs) must pass unchanged (R6).
- `cargo test --locked -- --test-threads=1`, `cargo clippy --all-targets`, `bash scripts/check-html-js.sh`.
- Model-free only: do not start any Server-Mode backend or load any model. End-to-end live inference routing is verified later when a backend is already running.

## Security, privacy, reliability, and operations
- Endpoints are localhost backend servers brainrouter itself started. No new external exposure. Provider cache bounded by number of distinct endpoints. Fail-safe: any lookup miss → existing llama-swap path. `ServingIdentity` does not carry or serialize API keys; routing constructs providers with `api_key = None`.
- **Known MVP limitation:** because the routing provider uses `api_key = None`, a vLLM (or other) Server-Mode server started **with** `--api-key` will reject routed requests with HTTP 401 (surfaced as an error, no silent llama-swap fallback). Start routable Server-Mode backends **without** an API key for now; secure auth pass-through (keys held outside the serialized identity) is a deferred follow-up.

## Rollout / rollback
- Additive; revert = single commit. Deploy via the runbook to strix master + GitHub; back up `benchmarks.sqlite3` first.

## Open questions
- Deferred follow-ups: not-running errors, reserved namespace/collision policy, readiness/crash reconciliation beyond start/stop registration, and expanding routing eligibility if ds4/halogen/r9v are later proven OpenAI-compatible.

## Referenced files
`src/router.rs`, `src/serving_identity.rs`, `src/server_mode.rs` (start/register/label and vLLM served repo resolution), `src/server.rs` (`/v1/models`, `register_serving_identity`, `AppState`), `src/daemon.rs` (Router construction), `src/provider/openai.rs` (`OpenAiProvider::new` base URL contract).

## Dory validation record
- Readiness round 1 reviewed v1 and failed with B1–B7. Coordinator resolved blockers into the v2 MVP on 2026-09-27: eligibility is `openai_compatible == true`; exact served-model matching is in-memory; no-match falls through to llama-swap; secrets stay out of identities; tests target selection logic without trait refactor. Verdict after resolution: approved for implementation.

## Human approval
Coordinator explicitly approved v2 MVP implementation on 2026-09-27. Full end-to-end routing verification deferred to when a Server-Mode server is already running; this implementation remains model-free.
