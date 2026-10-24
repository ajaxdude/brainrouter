//! Request router — role cards, the decision slot, and the v1 selector engine.
//!
//! v2 L3 (design `docs/design/v2-streamlining.L3.md`): every proxied request
//! resolves through ONE grammar.
//!
//!   1. Role: the `x-brainrouter-role` header (default `main`); the `subs`
//!      selector NAMES the subagent role and overrides the header.
//!   2. Cards: a non-explicit selector consults the role card
//!      (project scope > global > default `auto`). Explicit selectors
//!      (`local`/`cloud`/`cloud/<id>`/bare-id) bypass cards and are NEVER
//!      policy-clamped — explicit = owner intent.
//!   3. The card assignment compiles through the SAME selector path: `auto`
//!      runs the decision slot (features → {route, tier} with guardrail
//!      clamps, then a candidate pool + chain), every other selector routes
//!      through the v1 route for that selector.
//!   4. Pinned-card strictness = v1 semantics PER SELECTOR: bare-key cards
//!      are direct picks (strict, no fallback, the error surfaces); `local`
//!      cards keep the managed one-`fallback_model`-retry hop;
//!      `cloud`/`cloud/<id>` cards keep the managed local-fallback hop EXCEPT
//!      for role `review`, where v1's role-specific strict rule (failure
//!      surfaces → the review loop escalates) is preserved. Chain fallback
//!      across pool candidates exists ONLY on the `auto` path, on
//!      connect/response-headline failure — never mid-stream.
//!
//! On success the resulting stream is wrapped in `TimeoutStream` so a stalled
//! provider surfaces as an error instead of hanging the client. KeepaliveStream
//! is applied in server.rs where the wire format (OpenAI/Anthropic) is known.

use crate::{
    classifier::BudgetTier,
    config::NudgeBudgets,
    decision::{DecisionSlot, RequestFeatures, Route as SlotRoute, Tier as SlotTier},
    health::HealthTracker,
    inference_state::{InferenceTracker, Phase},
    prompt_rewriter,
    provider::{openai::OpenAiProvider, Provider, ProviderResponse},
    roles::{parse_selector, project_root_of, Role, RoleCardStore, Selector},
    routing_events::{CompletedStreamMeasurement, RouteEvent, RoutingEvents, Stage},
    serving_identity::{ServingIdentity, ServingIdentityRegistry},
    stream::TimeoutStream,
    types::{ChatCompletionRequest, ChatMessage},
};
use anyhow::{anyhow, Result};
use futures_util::{stream as fstream, StreamExt};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::{sync::Arc, time::{Duration, Instant}};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::server::ScopeClock;

/// Maximum time to wait for the first SSE byte from a provider.
/// Long prompts (37k+ tokens) can spend 7+ minutes in prefill at ~85 tok/s;
/// this must comfortably exceed that worst case.
const TTFT_TIMEOUT: Duration = Duration::from_secs(600);

/// Maximum silence between consecutive SSE chunks once generation has started.
/// 180 s is generous but still catches genuinely hung connections.
const STREAM_STALL_TIMEOUT: Duration = Duration::from_secs(180);
/// Provider health-tracker keys. Used for circuit breaking.
const MANIFEST_KEY: &str = "manifest";
const LLAMA_SWAP_KEY: &str = "llama-swap";
/// Requests above this approximate token count get the known-ctx pool ranking
/// (L3 "pool-by-ctx applies … for large requests"). Below it every candidate
/// is presumed to fit and the configured order stands.
const LARGE_REQUEST_TOKENS: u64 = 4096;

/// The additive event record of the decision slot for one request (L3:
/// `decision {route, tier, confidence}` + `rationale` + `decision_fallback`).
#[derive(Debug, Clone, Serialize)]
pub struct SlotEvent {
    pub decision: DecisionEvent,
    pub rationale: String,
    pub decision_fallback: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecisionEvent {
    pub route: &'static str,
    pub tier: &'static str,
    pub confidence: f64,
}

/// Summary of what actually happened during a route call.
/// Returned alongside the response so callers (e.g. review loop) can
/// record which model handled the request.
pub struct RouteInfo {
    /// "cloud" or "local" — the Bonsai classification (kept verbatim for
    /// v1 consumers).
    pub bonsai_decision: &'static str,
    /// The backend that served the response, or None on error.
    pub effective_provider: Option<String>,
    /// Model actually used. For Manifest routes this is extracted from the first SSE
    /// response chunk (e.g. "claude-3-7-sonnet-20250219"); falls back to empty string
    /// if the chunk cannot be parsed. For llama-swap this is the model key.
    pub model_key: String,
    /// Failed hops in the fallback chain (e.g. Manifest down → local).
    /// The dashboard emits one RouteEvent per entry so multi-hop routes
    /// render as a hop chain instead of a single collapsed card.
    pub failed_attempts: Vec<FailedAttempt>,
}

/// One failed hop in a fallback chain.
#[derive(Clone)]
pub struct FailedAttempt {
    pub stage: Stage,
    pub provider: String,
    pub model_key: String,
    pub error: String,
}

impl RouteInfo {
    /// Human-readable description for the dashboard / session review_model field.
    pub fn display(&self) -> String {
        match self.effective_provider.as_deref() {
            Some("manifest") => {
                let model = if self.model_key.is_empty() { "auto" } else { &self.model_key };
                format!("{} → manifest ({})", self.bonsai_decision, model)
            }
            Some("llama-swap") => format!("{} → llama-swap ({})", self.bonsai_decision, self.model_key),
            Some(other) => format!("{} → {}", self.bonsai_decision, other),
            None => format!("{} → error", self.bonsai_decision),
        }
    }
}

/// Inputs of one resolved route (everything the dispatch and events need).
struct Resolution {
    /// The header role (the event label; `subs` overrides to `subagent`).
    role: Role,
    /// The selector the request will compile through (card assignment for
    /// managed selectors; the request selector for explicit ones).
    target: Selector,
    /// Canonical project scope (the peer-cwd git root), when resolvable.
    project: Option<String>,
    /// True when the REQUEST selector was `subs` (names the subagent role and
    /// keeps the v1 subs-pool behavior: the pool is `subs_model`).
    /// Read in `route_resolved`'s Auto arm; the explicit-bypass bookkeeping
    /// lives in `target` (the request selector itself).
    via_subs: bool,
    /// True only on the PROXY path with role=review (card cloud selector +
    /// review role ⇒ the pinned v1 role-specific strict rule). Review-loop
    /// calls carry the strictness in `allow_local_fallback` (v1 H3) and must
    /// not be double-strict.
    proxy_review_strict: bool,
}

tokio::task_local! {
    /// The UDS request's grace clock (`None` for TCP/Router-level callers,
    /// whose grace window starts at the request itself).
    static REQUEST_SCOPE_CLOCK: Option<Arc<ScopeClock>>;
}

pub struct Router {
    slot: Arc<DecisionSlot>,
    cards: Arc<RoleCardStore>,
    manifest: Arc<OpenAiProvider>,
    manifest_enabled: bool,
    llama_swap: Arc<OpenAiProvider>,
    fallback_model: String,
    /// Model keys known to belong to llama-swap. Requests for these bypass the slot.
    local_models: Vec<String>,
    /// Subs-pool model key — the `auto`-path compatibility seed for the
    /// subagent pool (v1 `llama_swap.subs_model`; the card assignment wins).
    subs_model: Option<String>,
    health: Arc<HealthTracker>,
    routing_events: Arc<RoutingEvents>,
    /// Last project-scope resolution latency (ms): 0 = instant (resolved on
    /// the first probe — the normal case). Diagnostics for the peer-cwd
    /// scope race; exposed on `/api/decision`.
    scope_lookup_ms: std::sync::atomic::AtomicU64,
    /// Optional custom system prompt for local routing mode.
    local_system_prompt: Option<String>,
    /// Dashboard inference progress tracker.
    pub inference_tracker: Arc<InferenceTracker>,
    /// Reasoning budgets per tier (nudge). Injected as
    /// `reasoning_budget_tokens` on slot-routed local requests when nudge is on.
    nudge_budgets: NudgeBudgets,
    /// Runtime nudge master switch (shared with dashboard).
    nudge_enabled: Arc<AtomicBool>,
    /// Runtime nudge tier override: 0 = auto (slot), 1 = light, 2 = deep.
    nudge_tier: Arc<AtomicU8>,
    /// llama-swap model key that runs the nudge fork (chain seed when the
    /// nudge switch is on — v1 `Classifier::default_local_model` parity).
    nudge_model_key: Option<String>,
    /// (v2 L3 Step 9 I4: the per-request grace clock travels as a tokio
    /// task-local — the Router is shared by every connection, and a slot
    /// on it let a concurrent request clear/overwrite the window of the
    /// request being resolved.)
    /// Runtime prompt-rewrite toggle. When off, local routes forward the
    /// incoming messages untouched (no system-prompt rewrite).
    prompt_rewrite: Arc<AtomicBool>,
    /// llama-swap's own config file, when known — the `parameters.ctx` source
    /// for pool ranking (L3: "known-ctx pool ranking via llama-swap config").
    llama_swap_config_path: Option<std::path::PathBuf>,
    /// Optional in-memory registry of running OpenAI-compatible Server-Mode
    /// backends. `None` preserves the pre-Feature-B llama-swap-only path.
    serving_identities: Option<Arc<ServingIdentityRegistry>>,
    serving_providers: Mutex<HashMap<String, Arc<OpenAiProvider>>>,
    /// v2 L4: the swap supervisor. `None` preserves the v1 llama-swap path
    /// byte-identically (integration tests and shim-only deployments).
    swap: Option<Arc<crate::swap_scheduler::SwapScheduler>>,
    /// Direct providers for native/external residents (outside llama-swap).
    native_providers: Mutex<HashMap<String, Arc<OpenAiProvider>>>,
}

pub struct RouterArgs {
    pub slot: Arc<DecisionSlot>,
    pub cards: Arc<RoleCardStore>,
    pub manifest: Arc<OpenAiProvider>,
    /// Whether the cloud backend (Manifest) is enabled. Off → cloud requests
    /// skip Manifest and fall straight back to llama-swap.
    pub manifest_enabled: bool,
    pub llama_swap: Arc<OpenAiProvider>,
    pub fallback_model: String,
    /// Model keys known to belong to llama-swap. See `LlamaSwapConfig::local_models`.
    pub local_models: Vec<String>,
    /// Subs-pool model key. See `LlamaSwapConfig::subs_model`.
    pub subs_model: Option<String>,
    pub health: Arc<HealthTracker>,
    pub routing_events: Arc<RoutingEvents>,
    pub local_system_prompt: Option<String>,
    pub inference_tracker: Arc<InferenceTracker>,
    pub nudge_budgets: NudgeBudgets,
    pub nudge_enabled: Arc<AtomicBool>,
    pub nudge_tier: Arc<AtomicU8>,
    pub nudge_model_key: Option<String>,
    pub prompt_rewrite: Arc<AtomicBool>,
    pub llama_swap_config_path: Option<std::path::PathBuf>,
    pub serving_identities: Option<Arc<ServingIdentityRegistry>>,
}

impl Router {
    pub fn new(args: RouterArgs) -> Self {
        Self {
            slot: args.slot,
            cards: args.cards,
            manifest: args.manifest,
            manifest_enabled: args.manifest_enabled,
            llama_swap: args.llama_swap,
            fallback_model: args.fallback_model,
            local_models: args.local_models,
            subs_model: args.subs_model,
            health: args.health,
            routing_events: args.routing_events,
            scope_lookup_ms: std::sync::atomic::AtomicU64::new(0),
            local_system_prompt: args.local_system_prompt,
            inference_tracker: args.inference_tracker,
            nudge_budgets: args.nudge_budgets,
            nudge_enabled: args.nudge_enabled,
            nudge_tier: args.nudge_tier,
            nudge_model_key: args.nudge_model_key,
            prompt_rewrite: args.prompt_rewrite,
            llama_swap_config_path: args.llama_swap_config_path,
            serving_identities: args.serving_identities,
            serving_providers: Mutex::new(HashMap::new()),
            swap: None,
            native_providers: Mutex::new(HashMap::new()),
        }
    }

    /// v2 L4: install the swap supervisor. Without it every local route is
    /// the untouched v1 path.
    pub fn install_swap(&mut self, swap: Arc<crate::swap_scheduler::SwapScheduler>) {
        self.swap = Some(swap);
    }

    pub fn swap(&self) -> Option<&Arc<crate::swap_scheduler::SwapScheduler>> {
        self.swap.as_ref()
    }

    pub fn cards(&self) -> &Arc<RoleCardStore> {
        &self.cards
    }

    /// Test/bench construction: swap the card store (integration tests seed
    /// project-scope cards directly).
    pub fn with_cards(mut self, cards: Arc<RoleCardStore>) -> Self {
        self.cards = cards;
        self
    }

    pub fn slot(&self) -> &Arc<DecisionSlot> {
        &self.slot
    }

    /// Catalog discovery is metadata-only; never invokes inference or starts a model.
    pub async fn model_catalog(&self) -> serde_json::Value {
        let local = self.llama_swap.list_models().await;
        let cloud = if self.manifest_enabled {
            self.manifest.list_models().await
        } else {
            Err(anyhow!("Cloud is disabled; discovery was not attempted"))
        };
        fn entry(result: Result<Vec<String>>) -> serde_json::Value {
            match result {
                Ok(models) => serde_json::json!({ "models": models, "error": null }),
                Err(error) => serde_json::json!({ "models": [], "error": error.to_string() }),
            }
        }
        serde_json::json!({
            "local": entry(local), "cloud": entry(cloud),
            "cloud_enabled": self.manifest_enabled,
            "local_default": self.fallback_model,
            "unknown_model_policy": "Explicit IDs are accepted without discovery; the provider validates availability. Explicit local failures are errors. Cloud failure or disabled cloud uses the existing local fallback.",
        })
    }

    /// Model keys known to belong to llama-swap (from config). Exposed so the
    /// flush-and-reload path can re-establish the local working set.
    pub fn local_models(&self) -> &Vec<String> {
        &self.local_models
    }

    /// Rewrite the system prompt for local routing, unless the dashboard
    /// prompt-rewrite toggle is off (pass-through mode).
    fn maybe_rewrite_local(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        if !self.prompt_rewrite.load(Ordering::Relaxed) {
            return messages;
        }
        prompt_rewriter::rewrite_for_local(messages, self.local_system_prompt.as_deref())
    }

    /// Inject the per-request reasoning budget for slot-routed local requests
    /// when nudge is enabled. The field flows through the request's flattened
    /// `extra` map into llama-swap and on to the nudge fork's
    /// `oaicompat_chat_params_parse`. A client-supplied value is never
    /// overridden, and direct/explicit model requests never reach this path.
    fn inject_nudge_budget(&self, request: &mut ChatCompletionRequest, tier: SlotTier) {
        if !self.nudge_enabled.load(Ordering::Relaxed) {
            return;
        }
        let tier = match self.nudge_tier.load(Ordering::Relaxed) {
            1 => BudgetTier::Light,
            2 => BudgetTier::Deep,
            _ => tier.budget_tier(),
        };
        let budget = match tier {
            BudgetTier::Light => self.nudge_budgets.light,
            BudgetTier::Deep => self.nudge_budgets.deep,
        };
        if !request.extra.is_object() {
            request.extra = serde_json::json!({});
        }
        if let serde_json::Value::Object(map) = &mut request.extra {
            // A client-supplied value is authoritative — never override it.
            if !map.contains_key("reasoning_budget_tokens") {
                map.insert(
                    "reasoning_budget_tokens".to_string(),
                    serde_json::json!(budget),
                );
                debug!(budget, ?tier, "Injected per-request reasoning budget");
            }
        }
    }

    /// Resolve a request's project scope: the peer cwd normalized to its
    /// canonical git root (the relocated `project_key`). Gated: zero
    /// filesystem work when the store has no scoped cards, and always cheap
    /// (one canonicalize + walk).
    /// Run one request's handling with its UDS grace clock installed (the
    /// server sets this at request arrival on the connection's task; the
    /// task-local propagates to the routing future, so concurrent requests
    /// on the shared Router can never clear each other's window).
    pub fn scope_clock_scope<F: std::future::Future>(
        clock: Option<Arc<ScopeClock>>,
        f: F,
    ) -> F {
        REQUEST_SCOPE_CLOCK.sync_scope(clock, || f)
    }

    async fn resolve_project(&self, cwd: &str) -> Option<String> {
        // A UDS request owns its grace window (the CLI peer is alive for
        // the request; the clock was stamped at arrival); Router-level
        // callers (tests, the review loop) start the window at the request.
        let window_start = REQUEST_SCOPE_CLOCK
            .try_with(|c| c.as_ref().and_then(|c| c.elapsed()))
            .ok()
            .flatten();
        self.resolve_project_with(cwd, std::time::Duration::from_secs(120), window_start)
            .await
    }

    /// Retry budget for the canonicalize: the UDS peer's cwd (/proc/<pid>/cwd)
    /// can change between connection accept and this resolution (a test
    /// process that spawned the daemon then chdir'd), and `project_key`
    /// returns None for a cwd it cannot canonicalize. Under parallel load
    /// that race used to drop the project scope for one request. The window
    /// is measured from the connection's ACCEPT time (`accept_elapsed`,
    /// stamped by the UDS accept loop — a connection accepted at t has a
    /// live peer, so a resolvable cwd cannot vanish mid-request; queueing
    /// between accept and request handling does not burn the grace).
    /// Unresolvable cwds (deleted dirs, tmpdirs) return at once.
    async fn resolve_project_with(
        &self,
        cwd: &str,
        grace: std::time::Duration,
        window_start: Option<std::time::Duration>,
    ) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        if cwd.is_empty() {
            return None;
        }
        // Synchronous probes on THIS task: a saturated default blocking
        // pool (full-suite load: dozens of mocks + daemons per process)
        // used to starve `spawn_blocking` probes for seconds each, burning
        // the entire retry window before the first probe ran. The walk is
        // 2-3 syscalls on a hot dentry (canonicalize + two .git checks).
        // The grace window is absolute from UDS request arrival; the
        // caller-side queue time before this resolution began is charged
        // too, so the total stays bounded by `grace`.
        let mut used = window_start.unwrap_or_default();
        loop {
            let probe_started = std::time::Instant::now();
            let (key, probe_ok) = match std::fs::canonicalize(cwd) {
                Ok(canonical) => {
                    let key = project_root_of(&canonical).map(|p| p.display().to_string());
                    (key, true)
                }
                Err(_) => (None, false),
            };
            used += probe_started.elapsed();
            if key.is_some() {
                self.scope_lookup_ms.store(used.as_millis() as u64, Relaxed);
                return key;
            }
            if !probe_ok {
                // The peer's cwd is gone from the filesystem (tmpdir cleanup
                // of a finished test, a deleted checkout): no retry can make
                // it a git root. Global scope.
                return None;
            }
            if used >= grace {
                // The peer's cwd exists and is a live process's directory,
                // yet canonicalize has not seen it for the whole grace
                // window (a CLI mid chdir-publish under full-suite load).
                // The project-scope answer is not knowable: route GLOBAL —
                // stalling the request further helps no one, and a
                // never-resolving walk must not fail the request outright.
                self.scope_lookup_ms.store(used.as_millis() as u64, Relaxed);
                return None;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            used += std::time::Duration::from_millis(25);
        }
    }

    /// Route a request, returning the response and metadata about the routing decision.
    /// `session_id` tags the emitted RouteEvent (used by the review loop so events are
    /// linkable to review sessions). `role` is the `x-brainrouter-role` header
    /// value (default `main`; the `subs` selector overrides it to `subagent`).
    pub async fn route_tagged(
        &self,
        request: ChatCompletionRequest,
        session_id: Option<String>,
        cwd: String,
        user_agent: String,
    ) -> Result<(ProviderResponse, RouteInfo)> {
        self.route_tagged_role(request, session_id, cwd, user_agent, "main").await
    }

    /// The L3 entry point: same as [`Self::route_tagged`] plus the role
    /// header (L3 spec "Role signal").
    pub async fn route_tagged_role(
        &self,
        request: ChatCompletionRequest,
        session_id: Option<String>,
        cwd: String,
        user_agent: String,
        role_header: &str,
    ) -> Result<(ProviderResponse, RouteInfo)> {
        let header_role = Role::parse(role_header).unwrap_or(Role::Main);
        let selector = parse_selector(&request.model);
        // `subs` NAMES the subagent role and overrides the header (L3 spec).
        let role = if selector == Selector::Subs { Role::Subagent } else { header_role };

        let explicit = selector.is_explicit();
        let via_subs = selector == Selector::Subs;
        let project = if explicit || !self.cards.has_scoped_cards() {
            // No scoped cards exist: the git-root walk cannot change the
            // card answer (every card is global), so skip it entirely.
            None
        } else {
            self.resolve_project(&cwd).await
        };
        // Managed selectors (auto/subs) compile through the role card:
        // project > global > default `auto`.
        let target = if explicit {
            selector
        } else {
            let assignment = self.cards.assignment(role, project.as_deref());
            let compiled = parse_selector(&assignment);
            if selector == Selector::Subs && compiled == Selector::Subs {
                // A card must never carry `subs`; treat as auto (pool).
                Selector::Auto
            } else {
                compiled
            }
        };
        let resolution = Resolution {
            role,
            target,
            project,
            via_subs,
            proxy_review_strict: role == Role::Review,
        };
        self.route_resolved(request, session_id, cwd, user_agent, resolution, true, true, None).await
    }

    /// Review calls use their session snapshot (the review CARD record taken
    /// at session start), never the live cards. L3: the loop calls the router
    /// with role `review` — the event label, and the cloud card's
    /// role-specific strict rule.
    pub async fn route_with_choice(
        &self,
        mut request: ChatCompletionRequest,
        choice: &crate::roles::ModelChoice,
        session_id: Option<String>,
        cwd: String,
        user_agent: String,
        allow_local_fallback: bool,
    ) -> Result<(ProviderResponse, RouteInfo)> {
        choice.validate()?;
        request.model = choice.selector();
        // Reviewer path: never rewrite the local prompt (apply_local_rewrite=false)
        // so the review criteria survive; never fall back to local on a cloud
        // failure (allow_local_fallback threaded from the caller). The review
        // card's project scope was applied upstream into the per-run snapshot.
        self.route_card_choice(request, session_id, cwd, user_agent, choice, allow_local_fallback).await
    }

    async fn route_card_choice(
        &self,
        request: ChatCompletionRequest,
        session_id: Option<String>,
        cwd: String,
        user_agent: String,
        choice: &crate::roles::ModelChoice,
        allow_local_fallback: bool,
    ) -> Result<(ProviderResponse, RouteInfo)> {
        let target = parse_selector(&choice.selector());
        let resolution = Resolution {
            role: Role::Review,
            target,
            // The reviewer choice is the owner's card snapshot — explicit.
            project: None,
            via_subs: false,
            proxy_review_strict: false,
        };
        // The reviewer's managed hop honors the caller's strictness flag
        // (v1: `route_with_choice(..., allow_local_fallback=false)`).
        self.route_resolved(request, session_id, cwd, user_agent, resolution, allow_local_fallback, false, None)
            .await
    }

    /// v2 L2 plugin RPC entry point. Full router path (slot, fallbacks,
    /// health) identical to the proxy; events carry the additive
    /// `source: "plugin <id>"` tag so the dashboard attributes the hop.
    pub async fn route_plugin(
        &self,
        request: ChatCompletionRequest,
        source: String,
    ) -> Result<(ProviderResponse, RouteInfo)> {
        let ua = source.clone();
        self.route_tagged(request, None, String::new(), ua).await
    }

    #[allow(clippy::too_many_arguments)]
    async fn route_resolved(
        &self,
        mut request: ChatCompletionRequest,
        session_id: Option<String>,
        cwd: String,
        user_agent: String,
        resolution: Resolution,
        allow_local_fallback: bool,
        apply_local_rewrite: bool,
        source: Option<String>,
    ) -> Result<(ProviderResponse, RouteInfo)> {
        let start = Instant::now();
        let requested_model = request.model.clone();
        let prompt_excerpt = extract_prompt_excerpt(&request);
        // Conversation id for dashboard grouping: computed from the ORIGINAL
        // messages (before any local rewrite) so every turn of one harness
        // conversation hashes identically.
        let conv_id = conversation_fingerprint(&request);
        let role_str = resolution.role.as_str();

        let tracker = &self.inference_tracker;
        let max_tokens = request.max_tokens;
        // (bonsai_decision, routing_class, slot record, result)
        let (bonsai_decision, routing_class, slot, result) = match resolution.target {
            // Managed local token: route to the local/fallback model, rewrite prompt.
            Selector::Local => {
                info!("Direct local mode — rewriting system prompt");
                tracker.set(Phase::LocalWaiting, Some(self.fallback_model.clone()), Some("llama-swap".into()), max_tokens);
                // Reviewer calls (design H9/FR-B) pass apply_local_rewrite=false so
                // the review criteria system prompt survives a local route instead
                // of being replaced by the lean coding prompt.
                if apply_local_rewrite {
                    request.messages = self.maybe_rewrite_local(request.messages);
                }
                request.model = self.fallback_model.clone();
                ("local-direct", "local", None, self.route_local(request, true).await)
            }
            // Direct cloud: skip the slot, go straight to Manifest.
            Selector::Cloud => {
                info!("Direct cloud mode — routing to Manifest");
                tracker.set(Phase::CloudWaiting, None, Some("Manifest".into()), max_tokens);
                request.model = "auto".into();
                // v1's role-specific strict rule: a `review` cloud selector
                // never hops to the local fallback (the failure surfaces and
                // the review loop maps it to escalated).
                (
                    "cloud-direct",
                    "cloud",
                    None,
                    self.route_cloud(request, allow_local_fallback && !resolution.proxy_review_strict).await,
                )
            }
            Selector::CloudId(model) => {
                crate::roles::validate_model_id(&model)?;
                request.model = model.clone();
                tracker.set(Phase::CloudWaiting, Some(model), Some("Manifest".into()), max_tokens);
                (
                    "cloud-direct",
                    "cloud",
                    None,
                    self.route_cloud(request, allow_local_fallback && !resolution.proxy_review_strict).await,
                )
            }
            Selector::Subs => {
                // An unmanaged `subs` (only reachable when a card carried it,
                // normalized above) — defensive: pool seed, strict local.
                let subs = self.subs_model.clone().unwrap_or_else(|| self.fallback_model.clone());
                info!(model = %subs, "Subs pool routing — direct to llama-swap");
                tracker.set(Phase::LocalWaiting, Some(subs.clone()), Some("llama-swap".into()), max_tokens);
                request.model = subs;
                ("local-subs", "local · subs", None, self.route_local(request, false).await)
            }
            // The `auto` path: the decision slot, then pool + chain.
            Selector::Auto => {
                // `subs` + the auto card + a configured subs pool keeps v1
                // subs-pool semantics: the POOL IS subs_model (direct,
                // strict, `local-subs` tag) — v1 `route_resolved`'s subs arm.
                if resolution.via_subs && self.subs_model.is_some() {
                    let subs = self.subs_model.clone().unwrap();
                    info!(model = %subs, "Subs pool routing — direct to llama-swap");
                    tracker.set(Phase::LocalWaiting, Some(subs.clone()), Some("llama-swap".into()), max_tokens);
                    request.model = subs;
                    (
                        "local-subs",
                        "local · subs",
                        None,
                        self.route_local(request, false).await,
                    )
                } else {
                    let (tag, class, slot, result) =
                        self.route_slot(request, tracker, max_tokens, &resolution, allow_local_fallback).await;
                    (tag, class, slot, result)
                }
            }
            // A bare key (card assignment or direct request). v1: direct picks
            // are authoritative — strict, no fallback.
            Selector::Bare(key) => {
                if self.local_models.contains(&key) {
                    info!(model = %key, "Known local model — routing to local backend (llama-swap unless a running Server-Mode server serves this exact model)");
                } else {
                    info!(model = %key, "Named model — routing to local backend (llama-swap unless a running Server-Mode server serves this exact model)");
                }
                tracker.set(Phase::LocalWaiting, Some(key.clone()), Some("llama-swap".into()), max_tokens);
                request.model = key;
                ("local-specific", "local · selected model", None, self.route_local(request, false).await)
            }
        };

        let latency_ms = start.elapsed().as_millis() as u64;

        let (response, info) = match result {
            Ok((resp, mut info)) => {
                info.bonsai_decision = bonsai_decision;
                // Transition tracker to streaming phase
                let streaming_phase = if info.effective_provider.as_deref() == Some("manifest") {
                    Phase::CloudStreaming
                } else {
                    Phase::LocalStreaming
                };
                tracker.set(streaming_phase, Some(info.model_key.clone()), None, max_tokens);
                // One event per failed hop so the dashboard shows the chain
                // (e.g. manifest ✗ → local ✓) instead of a single card.
                let winner_stage = provider_to_stage(&info.effective_provider, bonsai_decision);
                for f in info.failed_attempts.iter().filter(|f| f.stage != winner_stage) {
                    let (toolbox_backend, compute_api) =
                        crate::routing_events::derive_serving_identity(Some(f.provider.as_str()));
                    self.routing_events.emit(RouteEvent {
                        id: 0,
                        timestamp: String::new(),
                        prompt_excerpt: prompt_excerpt.clone(),
                        requested_model: requested_model.clone(),
                        effective_provider: Some(f.provider.clone()),
                        model_key: f.model_key.clone(),
                        latency_ms: 0,
                        stage: f.stage,
                        success: false,
                        error: f.error.clone(),
                        bonsai_decision,
                        routing_class,
                        cwd: cwd.clone(),
                        session_id: session_id.clone(),
                        user_agent: user_agent.clone(),
                        source: source.clone(),
                        conv_id: conv_id.clone(),
                        pp_tps: 0.0,
                        tg_tps: 0.0,
                        toolbox_backend,
                        compute_api,
                        role: role_str.to_string(),
                        decision: slot.as_ref().map(|s| s.decision.clone()),
                        rationale: slot.as_ref().map(|s| s.rationale.clone()),
                        decision_fallback: slot.as_ref().and_then(|s| s.decision_fallback),
                        chosen_model: Some(f.model_key.clone()),
                        swap_state: None,
                        swap_source: None,
                    });
                }
                let (toolbox_backend, compute_api) =
                    crate::routing_events::derive_serving_identity(info.effective_provider.as_deref());
                let (swap_state, swap_source) = self.swap_event_fields(&info);
                let event_id = self.routing_events.emit(RouteEvent {
                    id: 0, // overwritten by emit()
                    timestamp: String::new(), // overwritten by emit()
                    prompt_excerpt,
                    requested_model,
                    effective_provider: info.effective_provider.clone(),
                    model_key: info.model_key.clone(),
                    latency_ms,
                    stage: provider_to_stage(&info.effective_provider, bonsai_decision),
                    success: true,
                    error: String::new(),
                    bonsai_decision,
                    routing_class,
                    cwd: cwd.clone(),
                    session_id: session_id.clone(),
                    user_agent: user_agent.clone(),
                    source: source.clone(),
                    conv_id: conv_id.clone(),
                    pp_tps: 0.0,
                    tg_tps: 0.0,
                    toolbox_backend,
                    compute_api,
                    role: role_str.to_string(),
                    decision: slot.as_ref().map(|s| s.decision.clone()),
                    rationale: slot.as_ref().map(|s| s.rationale.clone()),
                    decision_fallback: slot.as_ref().and_then(|s| s.decision_fallback),
                    chosen_model: Some(info.model_key.clone()),
                    swap_state,
                    swap_source,
                });
                // Wrap the stream to clear the tracker when it completes
                let tracker_for_stream = Arc::clone(&self.inference_tracker);
                let events = Arc::clone(&self.routing_events);
                let resp = wrap_with_tracker_clear(resp, tracker_for_stream);
                let resp = wrap_with_tps_capture(
                    resp, events, event_id, info.model_key.clone(),
                    info.effective_provider.clone(), start,
                );
                (resp, info)
            }
            Err(e) => {
                tracker.clear();
                let (toolbox_backend, compute_api) = crate::routing_events::derive_serving_identity(None);
                self.routing_events.emit(RouteEvent {
                    id: 0,
                    timestamp: String::new(),
                    prompt_excerpt,
                    requested_model,
                    effective_provider: None,
                    model_key: String::new(),
                    latency_ms,
                    stage: if bonsai_decision.starts_with("cloud") { Stage::CloudPrimary } else { Stage::LocalPrimary },
                    success: false,
                    error: e.to_string(),
                    bonsai_decision,
                    routing_class,
                    cwd,
                    session_id: session_id.clone(),
                    user_agent,
                    source,
                    conv_id,
                    pp_tps: 0.0,
                    tg_tps: 0.0,
                    toolbox_backend,
                    compute_api,
                    role: role_str.to_string(),
                    decision: slot.as_ref().map(|s| s.decision.clone()),
                    rationale: slot.as_ref().map(|s| s.rationale.clone()),
                    decision_fallback: slot.as_ref().and_then(|s| s.decision_fallback),
                    chosen_model: None,
                    swap_state: None,
                    swap_source: None,
                });
                return Err(e);
            }
        };

        Ok((response, info))
    }

    /// The `auto` path: features → slot decision (guardrail-clamped) → pool
    /// + chain (local) / Manifest with managed hop (cloud).
    async fn route_slot(
        &self,
        mut request: ChatCompletionRequest,
        tracker: &Arc<InferenceTracker>,
        max_tokens: Option<u32>,
        resolution: &Resolution,
        allow_local_fallback: bool,
    ) -> (
        &'static str,
        &'static str,
        Option<SlotEvent>,
        Result<(ProviderResponse, RouteInfo)>,
    ) {
        tracker.set(Phase::Classifying, None, None, max_tokens);
        let policy = self.cards.policy_for(resolution.project.as_deref());
        let raw_bytes = serde_json::to_vec(&request).map(|v| v.len()).unwrap_or(0);
        let features = RequestFeatures::extract(
            &request,
            raw_bytes,
            resolution.role.as_str(),
            "auto",
            policy.privacy,
            policy.quality,
            resolution.project.as_deref().unwrap_or_default(),
        );
        let mut outcome = self.slot.decide(&features).await;
        DecisionSlot::clamp(&mut outcome, policy.privacy, policy.quality);
        let decision = outcome.decision.clone();
        let slot_event = SlotEvent {
            decision: DecisionEvent {
                route: decision.route.as_str(),
                tier: decision.tier.as_str(),
                confidence: decision.confidence,
            },
            rationale: decision.rationale.clone(),
            decision_fallback: outcome.fallback.map(|f| f.as_str()),
        };
        let used_model_kind = self.slot.kind() != "heuristic" && self.slot.config().enabled;
        let routing_local = if used_model_kind { "slot → local" } else { "auto → local" };
        let routing_cloud = if used_model_kind { "slot → cloud" } else { "auto → cloud" };
        match decision.route {
            SlotRoute::Cloud => {
                tracker.set(Phase::CloudWaiting, None, Some("Manifest".into()), max_tokens);
                request.model = "auto".into();
                // Managed auto path keeps the PRD cloud→local fallback
                // guarantee (the review-role strictness is card-scoped, not
                // slot-scoped).
                let _ = allow_local_fallback;
                (
                    "cloud",
                    routing_cloud,
                    Some(slot_event),
                    self.route_cloud(request, true).await,
                )
            }
            SlotRoute::Local => {
                request.messages = self.maybe_rewrite_local(request.messages);
                self.inject_nudge_budget(&mut request, decision.tier);
                let candidates = self.chain_candidates(features.approx_prompt_tokens).await;
                info!(?candidates, tier = decision.tier.as_str(), "Slot local chain");
                let (class, tag) = (routing_local, "local");
                tracker.set(
                    Phase::LocalWaiting,
                    Some(candidates.first().cloned().unwrap_or_else(|| self.fallback_model.clone())),
                    Some("llama-swap".into()),
                    max_tokens,
                );
                let result = self.route_chain(request, candidates).await;
                (tag, class, Some(slot_event), result)
            }
        }
    }

    /// The auto-path candidate chain (L3 "Auto-path selection"): pool =
    /// llama-swap models (config keys + `/v1/models`) + server-mode
    /// identities; the chain seed is `fallback_model` (v1's compat seed;
    /// logged deprecated in favor of the L4 catalog). Ranking: the seed, then
    /// known-ctx fits (llama-swap `parameters.ctx` where readable, unknown
    /// ctx ranked last for large requests — logged), then the rest in
    /// configured order.
    async fn chain_candidates(&self, approx_tokens: u64) -> Vec<String> {
        let mut seed = None;
        if self.nudge_enabled.load(Ordering::Relaxed) {
            seed = self.nudge_model_key.clone();
        }
        let seed = seed.unwrap_or_else(|| self.fallback_model.clone());

        let mut pool: Vec<String> = Vec::new();
        pool.push(seed.clone());
        for key in &self.local_models {
            if key != &seed && !pool.contains(key) {
                pool.push(key.clone());
            }
        }
        if let Ok(models) = self.llama_swap.list_models().await {
            for key in models {
                if key != seed && !pool.contains(&key) {
                    pool.push(key);
                }
            }
        }
        if let Some(registry) = &self.serving_identities {
            for key in registry.openai_compatible_served_models().await {
                if key != seed && !pool.contains(&key) {
                    pool.push(key);
                }
            }
        }
        if pool.len() > 1 {
            self.rank_by_ctx(&mut pool, &seed, approx_tokens);
        }
        pool
    }

    /// Pool ranking by KNOWN ctx only (L3 spec): "unknown ctx = ranked last
    /// for large requests, clamp not applied, logged". The ctx source is
    /// llama-swap's own config (`parameters.ctx`); when the file is
    /// unreadable every ctx is unknown and the configured order stands.
    fn rank_by_ctx(&self, pool: &mut Vec<String>, seed: &str, approx_tokens: u64) {
        let Some(path) = &self.llama_swap_config_path else {
            debug!("llama-swap config path unknown; ctx ranking skipped");
            return;
        };
        let ctx = match read_llama_swap_ctx(path) {
            Ok(ctx) => ctx,
            Err(e) => {
                debug!(error = %e, "llama-swap config unreadable; ctx ranking skipped");
                return;
            }
        };
        if ctx.is_empty() {
            return;
        }
        let known = |key: &str| ctx.get(key).copied();
        if approx_tokens < LARGE_REQUEST_TOKENS {
            // Small requests fit anywhere: keep the configured order.
            return;
        }
        let mut ranked: Vec<String> = Vec::with_capacity(pool.len());
        ranked.push(seed.to_string());
        let mut fits: Vec<String> = pool
            .iter()
            .filter(|k| k.as_str() != seed && known(k).is_some_and(|c| c >= approx_tokens))
            .cloned()
            .collect();
        let known_small: Vec<String> = pool
            .iter()
            .filter(|k| k.as_str() != seed && known(k).is_some_and(|c| c < approx_tokens))
            .cloned()
            .collect();
        let unknown: Vec<String> = pool
            .iter()
            .filter(|k| k.as_str() != seed && known(k).is_none())
            .cloned()
            .collect();
        if !unknown.is_empty() {
            info!(models = ?unknown, "Large request: unknown-ctx models ranked last (ctx clamp not applied)");
        }
        ranked.append(&mut fits);
        ranked.extend(unknown);
        ranked.extend(known_small);
        *pool = ranked;
    }

    /// Chain over local candidates: connect/response-headline failure moves
    /// to the NEXT candidate (never mid-stream — once a stream is returned it
    /// is the caller's). The seed stage is LocalPrimary; retries LocalFallback
    /// (v1 tags).
    async fn route_chain(
        &self,
        mut request: ChatCompletionRequest,
        candidates: Vec<String>,
    ) -> Result<(ProviderResponse, RouteInfo)> {
        let candidates = if candidates.is_empty() {
            vec![self.fallback_model.clone()]
        } else {
            candidates
        };
        let mut failed: Vec<FailedAttempt> = Vec::new();
        let mut last_error = None;
        for (index, key) in candidates.iter().enumerate() {
            let stage = if index == 0 { Stage::LocalPrimary } else { Stage::LocalFallback };
            request.model = key.clone();
            match self.try_local_once(request.clone(), stage).await {
                Ok((resp, provider)) => {
                    return Ok((
                        resp,
                        RouteInfo {
                            bonsai_decision: "local",
                            effective_provider: Some(provider),
                            model_key: key.clone(),
                            failed_attempts: failed,
                        },
                    ));
                }
                Err(e) => {
                    warn!(model = %key, error = %e, "Chain candidate failed; trying the next one");
                    failed.push(FailedAttempt {
                        stage,
                        provider: "llama-swap".to_string(),
                        model_key: key.clone(),
                        error: e.to_string(),
                    });
                    last_error = Some(e);
                }
            }
        }
        Err(last_error.unwrap_or_else(|| anyhow!("no local candidates")))
    }

    /// One local attempt: server-mode identity wins, then llama-swap.
    async fn try_local_once(
        &self,
        mut request: ChatCompletionRequest,
        stage: Stage,
    ) -> Result<(ProviderResponse, String)> {
        let requested = request.model.clone();
        if let Some(identity) = self.find_server_mode_identity(&requested).await {
            let provider_name = server_mode_provider_name(&identity);
            let provider = self.server_mode_provider(&identity).await;
            let resp = provider.chat_completion(request).await?;
            return Ok((wrap_with_timeout_stream(resp)?, provider_name));
        }
        // v2 L4: catalogued models resolve through the swap supervisor's
        // `ensure` (Residency-first attach / budget-admitted spawn / queue).
        // Catalogued models only — every v1 selector/key keeps the old path.
        if let Some(swap) = &self.swap {
            if let Some(resolved) = swap.catalog().resolve_alias(&requested) {
                use crate::residency::ResidentSource;
                let outcome = swap.ensure(&resolved).await?;
                let source = match &outcome {
                    crate::swap_scheduler::EnsureOutcome::Attach { source, .. } => *source,
                };
                if matches!(
                    source,
                    ResidentSource::Native | ResidentSource::External | ResidentSource::Container
                ) {
                    let endpoint = outcome.endpoint().to_string();
                    let entry = swap.catalog().entry(&resolved);
                    let served = entry
                        .as_ref()
                        .map(|e| e.served_alias())
                        .unwrap_or_else(|| resolved.clone());
                    let provider = self.native_provider(&endpoint).await;
                    sanitize_for_llama_family(&mut request);
                    request.model = served;
                    let resp = provider.chat_completion(request).await?;
                    return Ok((wrap_with_timeout_stream(resp)?, "native".to_string()));
                }
                // Shim/other resident: the existing llama-swap/identity path
                // below attaches to the same endpoint (single residency).
            }
        }
        let (resp, model_key) = self.try_llama_swap(request, stage).await?;
        let _ = model_key;
        Ok((resp, "llama-swap".to_string()))
    }

    /// Direct provider for a native/external resident endpoint.
    async fn native_provider(&self, endpoint: &str) -> Arc<OpenAiProvider> {
        let mut providers = self.native_providers.lock().await;
        if let Some(provider) = providers.get(endpoint) {
            return Arc::clone(provider);
        }
        let provider = Arc::new(OpenAiProvider::new(
            "native".to_string(),
            format!("{}/v1", endpoint.trim_end_matches('/')),
            None,
        ));
        providers.insert(endpoint.to_string(), Arc::clone(&provider));
        provider
    }

    /// v2 L4 event fields: the swap lifecycle/source for a routed model, when
    /// the catalog owns it. Additive; `None` for every non-swap route.
    fn swap_event_fields(&self, info: &RouteInfo) -> (Option<String>, Option<String>) {
        let Some(swap) = &self.swap else {
            return (None, None);
        };
        if swap.catalog().resolve_alias(&info.model_key).is_none() {
            return (None, None);
        }
        let source = match info.effective_provider.as_deref() {
            Some("native") => "native",
            Some("llama-swap") => "shim",
            Some("manifest") => return (None, None),
            _ => "external",
        };
        (Some("ready".to_string()), Some(source.to_string()))
    }

    /// Cloud path: try Manifest first. On error/circuit-open, fall back to
    /// llama-swap with the configured fallback model.
    async fn route_cloud(
        &self,
        mut request: ChatCompletionRequest,
        allow_local_fallback: bool,
    ) -> Result<(ProviderResponse, RouteInfo)> {
        // Callers normalize managed aliases to auto; exact cloud IDs pass through.
        let requested_cloud_model = request.model.clone();

        if !self.manifest_enabled {
            warn!(
                provider = MANIFEST_KEY,
                "Manifest is disabled (manifest.enabled: false) — cloud request falls back to llama-swap"
            );
        } else if self.health.is_healthy(MANIFEST_KEY) {
            info!(provider = MANIFEST_KEY, "Attempting Manifest");
            match self.manifest.chat_completion(request.clone()).await {
                Ok(ProviderResponse::Stream(stream)) => {
                    match peek_manifest_model(stream).await {
                        ManifestPeek::Accepted { stream, model } => {
                            let model_key = model.unwrap_or_else(|| requested_cloud_model.clone());
                            self.health.report_success(MANIFEST_KEY);
                            info!(provider = MANIFEST_KEY, model = %model_key, "Manifest accepted request");
                            return Ok((
                                wrap_with_timeout(stream),
                                RouteInfo {
                                    bonsai_decision: "cloud",
                                    effective_provider: Some("manifest".to_string()),
                                    model_key,
                                    failed_attempts: Vec::new(),
                                },
                            ));
                        }
                        ManifestPeek::PseudoError => {
                        warn!(provider = MANIFEST_KEY, "Manifest returned pseudo-success (model=manifest) — likely credits exhausted or auth error, falling back");
                        // Don't report health failure — Manifest is reachable, just can't fulfill.
                        }
                        ManifestPeek::Failed(error) => {
                            warn!(provider = MANIFEST_KEY, error = %error, "Manifest stream failed before metadata, falling back");
                            self.health.report_failure(MANIFEST_KEY);
                        }
                    }
                }
                Err(e) => {
                    warn!(provider = MANIFEST_KEY, error = %e, "Manifest failed, falling back to llama-swap");
                    if e.is_backend_fault {
                        self.health.report_failure(MANIFEST_KEY);
                    }
                    // Always fall through to llama-swap — PRD guarantees automatic fallback.
                }
            }
        } else {
            warn!(provider = MANIFEST_KEY, "Manifest circuit open, skipping");
        }

        // Cloud fallback → llama-swap with fallback_model.
        // Reviewer calls (design H3) pass allow_local_fallback=false so a cloud
        // reviewer never silently executes on a local model (which would use
        // the wrong rubric and could load an unadmitted model / OOM). The
        // failure surfaces and the review loop maps it to `escalated`.
        if !allow_local_fallback {
            anyhow::bail!(
                "cloud reviewer route: Manifest unavailable (disabled, circuit open, or error) and local fallback is disabled"
            );
        }
        // Cloud fallback → llama-swap with fallback_model
        request.model = self.fallback_model.clone();
        let model_key = self.fallback_model.clone();
        let (resp, _) = self.try_local_once(request, Stage::CloudFallback).await?;
        Ok((
            resp,
            RouteInfo {
                bonsai_decision: "cloud",
                effective_provider: Some("llama-swap".to_string()),
                model_key,
                failed_attempts: vec![FailedAttempt {
                    stage: Stage::CloudPrimary,
                    provider: "manifest".to_string(),
                    model_key: requested_cloud_model,
                    error: "manifest unavailable (disabled, circuit open, or error)".to_string(),
                }],
            },
        ))
    }

    /// Managed local path (the `local` selector): v1 semantics — route to
    /// `fallback_model` with the one-`fallback_model`-retry hop
    /// (`allow_fallback`, short-circuited when requested == fallback).
    async fn route_local(
        &self,
        request: ChatCompletionRequest,
        allow_fallback: bool,
    ) -> Result<(ProviderResponse, RouteInfo)> {
        let requested = request.model.clone();
        match self.try_local_once(request.clone(), Stage::LocalPrimary).await {
            Ok((resp, provider)) => Ok((
                resp,
                RouteInfo {
                    bonsai_decision: "local",
                    effective_provider: Some(provider),
                    model_key: requested,
                    failed_attempts: Vec::new(),
                },
            )),
            Err(e) => {
                if !allow_fallback || requested == self.fallback_model {
                    return Err(e);
                }
                warn!(
                    requested = %requested,
                    fallback = %self.fallback_model,
                    error = %e,
                    "Local model failed, retrying with fallback"
                );
                let mut fallback_req = request;
                fallback_req.model = self.fallback_model.clone();
                let (resp, provider) =
                    self.try_local_once(fallback_req, Stage::LocalFallback).await?;
                Ok((
                    resp,
                    RouteInfo {
                        bonsai_decision: "local",
                        effective_provider: Some(provider),
                        model_key: self.fallback_model.clone(),
                        failed_attempts: vec![FailedAttempt {
                            stage: Stage::LocalPrimary,
                            provider: "llama-swap".to_string(),
                            model_key: requested,
                            error: e.to_string(),
                        }],
                    },
                ))
            }
        }
    }

    async fn find_server_mode_identity(&self, model: &str) -> Option<ServingIdentity> {
        self.serving_identities.as_ref()?.find_by_served_model(model).await
    }

    async fn server_mode_provider(&self, identity: &ServingIdentity) -> Arc<OpenAiProvider> {
        let mut providers = self.serving_providers.lock().await;
        if let Some(provider) = providers.get(&identity.endpoint) {
            return Arc::clone(provider);
        }
        let provider = Arc::new(OpenAiProvider::new(
            server_mode_provider_name(identity),
            server_mode_provider_base_url(identity),
            None,
        ));
        providers.insert(identity.endpoint.clone(), Arc::clone(&provider));
        provider
    }

    /// Attempt a llama-swap call. Returns the response and the model key used.
    async fn try_llama_swap(
        &self,
        mut request: ChatCompletionRequest,
        stage: Stage,
    ) -> Result<(ProviderResponse, String)> {
        if !self.health.is_healthy(LLAMA_SWAP_KEY) {
            return Err(anyhow!(
                "llama-swap circuit open, no backend available (stage={:?})",
                stage
            ));
        }
        // Shared llama-server-family shaping (shim + native): the exact same
        // sanitized body must reach both endpoint kinds.
        sanitize_for_llama_family(&mut request);
        let model_key = request.model.clone();
        debug!(
            provider = LLAMA_SWAP_KEY,
            ?stage,
            model = %model_key,
            msg_count = request.messages.len(),
            roles = %request.messages.iter().map(|m| m.role.as_str()).collect::<Vec<_>>().join(","),
            "Messages being sent to llama-swap"
        );
        info!(provider = LLAMA_SWAP_KEY, ?stage, model = %model_key, "Attempting llama-swap");
        match self.llama_swap.chat_completion(request).await {
            Ok(ProviderResponse::Stream(stream)) => {
                self.health.report_success(LLAMA_SWAP_KEY);
                info!(provider = LLAMA_SWAP_KEY, ?stage, "llama-swap accepted request");
                Ok((wrap_with_timeout(stream), model_key))
            }
            Err(e) => {
                warn!(provider = LLAMA_SWAP_KEY, ?stage, error = %e, "llama-swap failed");
                // Only trip the circuit for backend faults (connection errors, server
                // crashes). Application errors (bad request format, wrong message
                // order) mean the backend is healthy — don't penalise it.
                if e.is_backend_fault {
                    self.health.report_failure(LLAMA_SWAP_KEY);
                }
                Err(e.into())
            }
        }
    }
}

/// `parameters.ctx` per model id from llama-swap's config (best-effort:
/// file YAML `models: [{name/id, parameters: {ctx}}]`). Missing file /
/// unparseable = empty map (everything unknown, ranking skipped).
pub fn read_llama_swap_ctx(path: &std::path::Path) -> Result<HashMap<String, u64>> {
    let text = std::fs::read_to_string(path)?;
    let root: serde_yaml::Value = serde_yaml::from_str(&text)?;
    let mut out = HashMap::new();
    if let Some(models) = root.get("models").and_then(|m| m.as_sequence()) {
        for model in models {
            let id = model
                .get("name")
                .or_else(|| model.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if id.is_empty() {
                continue;
            }
            if let Some(ctx) = model
                .get("parameters")
                .and_then(|p| p.get("ctx"))
                .and_then(|c| c.as_u64())
            {
                out.insert(id, ctx);
            }
        }
    }
    Ok(out)
}

fn wrap_with_timeout_stream(resp: ProviderResponse) -> Result<ProviderResponse> {
    match resp {
        ProviderResponse::Stream(stream) => Ok(wrap_with_timeout(stream)),
    }
}

fn server_mode_provider_name(identity: &ServingIdentity) -> String {
    format!("server-mode:{}", identity.toolbox_backend)
}

fn server_mode_provider_base_url(identity: &ServingIdentity) -> String {
    format!("{}/v1", identity.endpoint.trim_end_matches('/'))
}

/// The shared llama-server-family request shape (v2 L4): applied to BOTH the
/// llama-swap shim and native llama-server endpoints so one request produces a
/// byte-identical sanitized body on either path (golden-body contract).
///
/// 1. Normalize 'developer' role → 'system' (OpenAI-style role that Qwen3
///    templates don't recognize).
/// 2. Merge system messages into one (Qwen3 requires a single system msg at
///    position 0).
/// 3. Ensure assistant messages have content (llama-server rejects assistant
///    messages with neither content nor tool_calls).
pub fn sanitize_for_llama_family(request: &mut ChatCompletionRequest) {
    for msg in &mut request.messages {
        if msg.role == "developer" {
            msg.role = "system".to_string();
        }
    }
    let (sys, rest): (Vec<_>, Vec<_>) = request
        .messages
        .drain(..)
        .partition(|m| m.role == "system");
    request.messages = merge_system_messages(sys)
        .into_iter()
        .chain(rest)
        .collect();
    sanitize_assistant_messages(&mut request.messages);
}

/// Collapse multiple system messages into a single one by concatenating their
/// text content with double newlines. Returns an empty vec if the input is
/// empty, or a single-element vec with the merged message.
///
/// Non-string content (arrays, objects) is serialized to its JSON form so
/// tool-schema payloads embedded in system messages are not silently dropped.
fn merge_system_messages(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
    if messages.len() <= 1 {
        return messages;
    }

    let mut parts: Vec<String> = Vec::with_capacity(messages.len());
    for msg in &messages {
        match &msg.content {
            Some(serde_json::Value::String(s)) => parts.push(s.clone()),
            Some(other) => parts.push(other.to_string()),
            None => {}
        }
    }

    vec![ChatMessage {
        role: "system".to_string(),
        content: Some(serde_json::Value::String(parts.join("\n\n"))),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }]
}

/// Ensure every assistant message has `content` set. llama-server rejects
/// assistant messages with neither `content` nor `tool_calls`. OMP and other
/// coding harnesses send tool-only assistant turns where content is null.
fn sanitize_assistant_messages(messages: &mut [ChatMessage]) {
    for msg in messages.iter_mut() {
        if msg.role == "assistant" && msg.content.is_none() {
            msg.content = Some(serde_json::Value::String(String::new()));
        }
    }
}

/// Consume a bounded prefix of a Manifest SSE stream, extract the `model`
/// field from complete frames, then reassemble the stream so no bytes are lost.
///
/// Manifest's first SSE frame looks like:
///   `data: {"id":"...","model":"claude-3-7-sonnet-20250219","choices":[...]}\n\n`
///
/// Missing metadata is not an error: fragmented or non-standard events are
/// forwarded unchanged. Only an explicit Manifest error payload triggers the
/// pseudo-success fallback path.
enum ManifestPeek {
    Accepted {
        stream: crate::provider::SseStream,
        model: Option<String>,
    },
    PseudoError,
    Failed(anyhow::Error),
}

async fn peek_manifest_model(
    mut stream: crate::provider::SseStream,
) -> ManifestPeek {
    const MAX_PEEK_BYTES: usize = 64 * 1024;

    let deadline = tokio::time::Instant::now() + TTFT_TIMEOUT;
    let mut chunks = Vec::new();
    let mut buffered = Vec::new();
    loop {
        let chunk = match tokio::time::timeout_at(deadline, stream.next()).await {
            Ok(Some(Ok(chunk))) => chunk,
            Ok(Some(Err(error))) => return ManifestPeek::Failed(error),
            Ok(None) if chunks.is_empty() => {
                return ManifestPeek::Failed(anyhow!("Manifest returned an empty stream"));
            }
            Ok(None) => break,
            Err(_) => {
                return ManifestPeek::Failed(anyhow!(
                    "Manifest stream stalled before metadata ({}s timeout)",
                    TTFT_TIMEOUT.as_secs()
                ));
            }
        };

        buffered.extend_from_slice(&chunk);
        chunks.push(chunk);
        let complete_frame = buffered.windows(2).any(|window| window == b"\n\n")
            || buffered.windows(4).any(|window| window == b"\r\n\r\n");
        let metadata = extract_manifest_metadata(&buffered);
        if complete_frame
            && (metadata.explicit_error || metadata.model.as_deref() == Some("manifest"))
        {
            return ManifestPeek::PseudoError;
        }
        if complete_frame || buffered.len() >= MAX_PEEK_BYTES {
            let reassembled: crate::provider::SseStream =
                Box::pin(fstream::iter(chunks.into_iter().map(Ok)).chain(stream));
            return ManifestPeek::Accepted {
                stream: reassembled,
                model: metadata.model,
            };
        }
    }

    let metadata = extract_manifest_metadata(&buffered);
    let reassembled: crate::provider::SseStream =
        Box::pin(fstream::iter(chunks.into_iter().map(Ok)).chain(stream));
    ManifestPeek::Accepted {
        stream: reassembled,
        model: metadata.model,
    }
}

#[derive(Default)]
struct ManifestMetadata {
    model: Option<String>,
    explicit_error: bool,
    done: bool,
}

fn extract_manifest_metadata(bytes: &[u8]) -> ManifestMetadata {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return ManifestMetadata::default();
    };
    let mut metadata = ManifestMetadata::default();
    for line in text.lines() {
        let payload = match line.strip_prefix("data:") {
            Some(value) => value.trim(),
            _ => continue,
        };
        if payload == "[DONE]" {
            metadata.done = true;
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        metadata.explicit_error |= value.get("error").is_some()
            || value.get("type").and_then(|value| value.as_str()) == Some("error");
        if metadata.model.is_none() {
            metadata.model = value
                .get("model")
                .and_then(|value| value.as_str())
                .filter(|value| !value.is_empty())
                .map(ToString::to_string);
        }
    }
    metadata
}


fn wrap_with_timeout(
    stream: crate::provider::SseStream,
) -> ProviderResponse {
    let timeout_stream = TimeoutStream::new(stream, TTFT_TIMEOUT, STREAM_STALL_TIMEOUT);
    ProviderResponse::Stream(Box::pin(timeout_stream))
}

/// Wrap a ProviderResponse stream so the inference tracker is cleared when
/// the stream is dropped (completes, errors, or client disconnects).
fn wrap_with_tracker_clear(
    resp: ProviderResponse,
    tracker: Arc<InferenceTracker>,
) -> ProviderResponse {
    match resp {
        ProviderResponse::Stream(stream) => {
            ProviderResponse::Stream(Box::pin(TrackerClearStream { stream, tracker }))
        }
    }
}

/// Stream wrapper that clears the inference tracker on drop.
struct TrackerClearStream {
    stream: crate::provider::SseStream,
    tracker: Arc<InferenceTracker>,
}

impl futures_util::Stream for TrackerClearStream {
    type Item = <crate::provider::SseStream as futures_util::Stream>::Item;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.stream.as_mut().poll_next(cx)
    }
}

impl Drop for TrackerClearStream {
    fn drop(&mut self) {
        self.tracker.clear();
    }
}

/// Observe the exact successful attempt emitted by the routing body. Its start
/// precedes in-body classification, provider connection, and any Manifest peek,
/// but excludes outer profile resolution. Output times are local SSE observations.
fn wrap_with_tps_capture(
    resp: ProviderResponse,
    routing_events: Arc<RoutingEvents>,
    event_id: u64,
    model_key: String,
    effective_provider: Option<String>,
    started: Instant,
) -> ProviderResponse {
    match resp {
        ProviderResponse::Stream(stream) => ProviderResponse::Stream(Box::pin(TpsCaptureStream {
            stream,
            routing_events,
            event_id,
            model_key,
            effective_provider,
            capture: SseMeasurementCapture::new(started),
            finished: false,
        })),
    }
}

/// Pass-through observer: only [DONE] followed by clean EOF records a sample.
/// Drop intentionally does nothing; even a client that stops polling at [DONE]
/// has not supplied clean EOF, so it cannot contribute a completed measurement.
struct TpsCaptureStream {
    stream: crate::provider::SseStream,
    routing_events: Arc<RoutingEvents>,
    event_id: u64,
    model_key: String,
    effective_provider: Option<String>,
    capture: SseMeasurementCapture,
    finished: bool,
}

impl futures_util::Stream for TpsCaptureStream {
    type Item = <crate::provider::SseStream as futures_util::Stream>::Item;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let item = self.stream.as_mut().poll_next(cx);
        if !self.finished {
            match &item {
                std::task::Poll::Ready(Some(Ok(bytes))) => {
                    self.capture.observe_bytes(bytes, Instant::now());
                }
                std::task::Poll::Ready(Some(Err(_))) => self.capture.invalid = true,
                std::task::Poll::Ready(None) => {
                    self.finished = true;
                    if let Some(sample) = self.capture.measurement(
                        self.event_id, self.model_key.clone(), self.effective_provider.clone(),
                    ) {
                        self.routing_events.record_measurement(sample);
                    }
                }
                std::task::Poll::Pending => {}
            }
        }
        item
    }
}

/// Hard cap on a whole SSE frame (including ignored fields/comments). Exceeding
/// it disables measurement, not forwarding, and releases the decoder buffers.
const MAX_MEASUREMENT_FRAME_BYTES: usize = 256 * 1024;

#[derive(Default)]
struct UsageTokenCount {
    value: Option<u64>,
    invalid: bool,
}

impl UsageTokenCount {
    fn observe(&mut self, value: Option<&serde_json::Value>) {
        let Some(value) = value else { return };
        let count = value.as_u64();
        if self.invalid || count.is_none() || self.value.is_some_and(|old| Some(old) != count) {
            self.invalid = true;
            self.value = None;
            return;
        }
        self.value = count;
    }
}

struct SseMeasurementCapture {
    started: Instant,
    first_output: Option<Instant>,
    last_output: Option<Instant>,
    done_at: Option<Instant>,
    prompt_tokens: UsageTokenCount,
    completion_tokens: UsageTokenCount,
    line: Vec<u8>,
    data: Vec<u8>,
    frame_bytes: usize,
    previous_cr: bool,
    first_line: bool,
    error_event: bool,
    invalid: bool,
}

impl SseMeasurementCapture {
    fn new(started: Instant) -> Self {
        Self {
            started,
            first_output: None,
            last_output: None,
            done_at: None,
            prompt_tokens: UsageTokenCount::default(),
            completion_tokens: UsageTokenCount::default(),
            line: Vec::new(),
            data: Vec::new(),
            frame_bytes: 0,
            previous_cr: false,
            first_line: true,
            error_event: false,
            invalid: false,
        }
    }

    fn observe_bytes(&mut self, bytes: &[u8], now: Instant) {
        for &byte in bytes {
            if self.invalid {
                return;
            }
            if self.previous_cr {
                self.previous_cr = false;
                if byte == b'\n' {
                    continue;
                }
            }
            self.frame_bytes += 1;
            if self.frame_bytes > MAX_MEASUREMENT_FRAME_BYTES {
                self.invalid = true;
                self.line = Vec::new();
                self.data = Vec::new();
                return;
            }
            if byte == b'\r' || byte == b'\n' {
                self.observe_line(now);
                self.previous_cr = byte == b'\r';
            } else {
                self.line.push(byte);
            }
        }
    }

    fn observe_line(&mut self, now: Instant) {
        let mut line = std::mem::take(&mut self.line);
        if self.first_line {
            self.first_line = false;
            if line.starts_with(b"\xef\xbb\xbf") {
                line.drain(..3);
            }
        }
        if line.is_empty() {
            self.frame_bytes = 0;
            if self.error_event {
                self.invalid = true;
            } else if !self.data.is_empty() {
                let mut data = std::mem::take(&mut self.data);
                data.pop(); // SSE joins data fields with LF, omitting the final LF.
                self.observe_frame(&data, now);
            }
            self.error_event = false;
            return;
        }
        let colon = line.iter().position(|&b| b == b':').unwrap_or(line.len());
        let field = &line[..colon];
        let value = line.get(colon + 1..).unwrap_or_default();
        let value = value.strip_prefix(b" ").unwrap_or(value);
        match field {
            b"data" => {
                self.data.extend_from_slice(value);
                self.data.push(b'\n');
            }
            b"event" => self.error_event |= value == b"error",
            _ => {}
        }
    }

    fn observe_frame(&mut self, data: &[u8], now: Instant) {
        let Ok(data) = std::str::from_utf8(data) else {
            self.invalid = true;
            return;
        };
        let data = data.trim();
        if data.is_empty() {
            return;
        }
        if self.done_at.is_some() {
            self.invalid = true; // No further data is valid after the terminal marker.
            return;
        }
        if data == "[DONE]" {
            self.done_at = Some(now);
            return;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
            self.invalid = true;
            return;
        };
        if !value.is_object()
            || value.get("error").is_some_and(|error| !error.is_null())
            || matches!(value.get("type").and_then(|v| v.as_str()),
                Some("error" | "response.failed" | "response.incomplete" | "response.cancelled"))
        {
            self.invalid = true;
            return;
        }
        if let Some(usage) = value.get("usage").filter(|usage| !usage.is_null()) {
            if usage.is_object() {
                self.prompt_tokens.observe(usage.get("prompt_tokens"));
                self.completion_tokens.observe(usage.get("completion_tokens"));
            } else {
                self.prompt_tokens.observe(Some(&serde_json::Value::Null));
                self.completion_tokens.observe(Some(&serde_json::Value::Null));
            }
        }
        if let Some(choices) = value.get("choices").and_then(|v| v.as_array()) {
            for choice in choices {
                if matches!(choice.get("finish_reason").and_then(|v| v.as_str()),
                    Some("error" | "cancelled" | "canceled"))
                {
                    self.invalid = true;
                    return;
                }
                if choice.get("delta").is_some_and(delta_has_generated_output) {
                    self.first_output.get_or_insert(now);
                    self.last_output = Some(now);
                }
            }
        }
    }

    fn measurement(
        &self,
        event_id: u64,
        model_key: String,
        effective_provider: Option<String>,
    ) -> Option<CompletedStreamMeasurement> {
        if self.invalid || self.frame_bytes != 0 || !self.line.is_empty() || !self.data.is_empty() {
            return None;
        }
        let done_at = self.done_at?;
        let generation_tps = match (self.completion_tokens.value, self.first_output, self.last_output) {
            (Some(tokens), Some(first), Some(last)) if tokens > 1 && last > first => {
                Some((tokens - 1) as f64 / last.duration_since(first).as_secs_f64())
            }
            _ => None,
        };
        Some(CompletedStreamMeasurement {
            event_id,
            completed_at: chrono::Utc::now().to_rfc3339(),
            model_key,
            effective_provider,
            measured_ttft_ms: self.first_output.map(|first| {
                first.duration_since(self.started).as_secs_f64() * 1000.0
            }),
            generation_tps,
            prompt_tokens: self.prompt_tokens.value,
            completion_tokens: self.completion_tokens.value,
            stream_duration_ms: done_at.duration_since(self.started).as_secs_f64() * 1000.0,
        })
    }
}

fn delta_has_generated_output(delta: &serde_json::Value) -> bool {
    fn nonempty(value: Option<&serde_json::Value>) -> bool {
        value.and_then(|v| v.as_str()).is_some_and(|s| !s.is_empty())
    }
    ["content", "reasoning", "reasoning_content"].iter().any(|key| nonempty(delta.get(key)))
        || nonempty(delta.pointer("/function_call/arguments"))
        || delta.get("tool_calls").and_then(|v| v.as_array()).is_some_and(|calls| {
            calls.iter().any(|call| nonempty(call.pointer("/function/arguments")))
        })
}

/// Derive the Stage from the effective provider and Bonsai decision.
/// This is a best-effort reconstruction — the internal routing methods track
/// stage precisely, but here we reconstruct for the error path.
fn provider_to_stage(effective_provider: &Option<String>, bonsai_decision: &str) -> Stage {
    match (bonsai_decision, effective_provider.as_deref()) {
        ("cloud", Some("manifest")) | ("cloud-direct", Some("manifest")) => Stage::CloudPrimary,
        ("cloud", Some("llama-swap")) | ("cloud-direct", Some("llama-swap")) => Stage::CloudFallback,
        ("local-direct", Some("llama-swap")) | ("local", Some("llama-swap")) => Stage::LocalPrimary,
        _ => Stage::LocalPrimary,
    }
}

/// Stable per-conversation fingerprint for dashboard grouping.
///
/// Harnesses (OMP) send no session header, so without this every turn of one
/// conversation is a separate dashboard card. The first system message plus
/// the FIRST user message are identical on every turn of a conversation
/// (the whole history rides along each request), so hashing that prefix
/// yields one id for the whole conversation — stable across turns, retries,
/// and fallbacks. FNV-1a 64-bit: identity only, no security relevance.
/// "" when the request carries no user message.
pub(crate) fn conversation_fingerprint(request: &ChatCompletionRequest) -> String {
    let sys = request.messages.iter().find(|m| m.role == "system").map(|m| message_text(&m.content));
    let first_user = request.messages.iter().find(|m| m.role == "user").map(|m| message_text(&m.content));
    let (Some(sys), Some(first_user)) = (sys, first_user) else {
        return String::new();
    };
    if first_user.is_empty() {
        return String::new();
    }
    let mut h: u64 = 0xcbf29ce484222325;
    for text in [&sys, &first_user] {
        // Truncate at a char boundary — prompts are full of em dashes and
        // CJK; a naive byte slice would panic mid-codepoint.
        let mut end = text.len().min(400);
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        for b in text[..end].as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    format!("{:016x}", h)
}

/// Flatten a message's content (string or text-part array) to plain text.
fn message_text(content: &Option<serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(" "),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

/// Extract the last user message from the request, truncated to 200 chars.
/// Mirrors the logic in classifier.rs but with a shorter limit for the event log.
fn extract_prompt_excerpt(request: &ChatCompletionRequest) -> String {
    let last_user = request.messages.iter().rev().find(|m| m.role == "user");
    let raw = match last_user {
        Some(msg) => match &msg.content {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Array(parts)) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join(" "),
            Some(other) => other.to_string(),
            None => String::new(),
        },
        None => String::new(),
    };

    const MAX: usize = 200;
    if raw.len() > MAX {
        let mut end = MAX;
        while !raw.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        raw[..end].to_string()
    } else {
        raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use chrono::Utc;

    fn user(content: &str) -> ChatMessage {
        ChatMessage {
            role: "user".to_string(),
            content: Some(serde_json::Value::String(content.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn asst(content: &str) -> ChatMessage {
        ChatMessage {
            role: "assistant".to_string(),
            content: Some(serde_json::Value::String(content.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn req(messages: Vec<ChatMessage>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "auto".into(),
            messages,
            stream: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
            stop: None,
            extra: serde_json::Value::Null,
        }
    }

    #[tokio::test]
    async fn manifest_peek_reassembles_fragmented_model_frame() {
        let chunks = vec![
            Ok(Bytes::from_static(b"data: {\"id\":\"x\",\"mo")),
            Ok(Bytes::from_static(b"del\":\"claude-test\",\"choices\":[]}\n\n")),
        ];
        let peek = peek_manifest_model(Box::pin(fstream::iter(chunks))).await;
        let ManifestPeek::Accepted { mut stream, model } = peek else {
            panic!("fragmented metadata should be accepted");
        };
        assert_eq!(model.as_deref(), Some("claude-test"));
        let mut forwarded = Vec::new();
        while let Some(chunk) = stream.next().await {
            forwarded.extend_from_slice(&chunk.unwrap());
        }
        assert_eq!(
            forwarded,
            b"data: {\"id\":\"x\",\"model\":\"claude-test\",\"choices\":[]}\n\n"
        );
    }

    #[tokio::test]
    async fn manifest_peek_does_not_treat_unknown_metadata_as_error() {
        let stream = Box::pin(fstream::iter(vec![Ok(Bytes::from_static(
            b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
        ))]));
        match peek_manifest_model(stream).await {
            ManifestPeek::Accepted { model, .. } => assert!(model.is_none()),
            _ => panic!("missing model metadata must not trigger fallback"),
        }
    }

    #[tokio::test]
    async fn manifest_peek_rejects_explicit_pseudo_success() {
        let stream = Box::pin(fstream::iter(vec![Ok(Bytes::from_static(
            b"data: {\"model\":\"manifest\",\"error\":{\"message\":\"credits exhausted\"}}\n\n",
        ))]));
        assert!(matches!(
            peek_manifest_model(stream).await,
            ManifestPeek::PseudoError
        ));
    }

    #[test]
    fn server_mode_provider_metadata_uses_backend_name_and_v1_base_url() {
        let identity = ServingIdentity {
            toolbox_backend: "gufo",
            compute_api: "rocm".to_string(),
            runtime_profile_id: "strix-gufo".to_string(),
            served_model: "gufo-qwen38-27b-q4kxl".to_string(),
            endpoint: "http://127.0.0.1:8001/".to_string(),
            openai_compatible: true,
            registered_at: Utc::now(),
        };

        assert_eq!(server_mode_provider_name(&identity), "server-mode:gufo");
        assert_eq!(server_mode_provider_base_url(&identity), "http://127.0.0.1:8001/v1");
    }

    #[test]
    fn fingerprint_stable_across_turns_and_differs_per_conversation() {
        let sysm = sys("SYSTEM PROMPT");
        let turn1 = req(vec![
            sysm.clone(),
            user("first turn"),
            asst("ok"),
            user("second turn"),
        ]);
        // Same conversation, later turn: fingerprint must match (system + FIRST user unchanged).
        let mut later = turn1.clone();
        later.messages.push(asst("ok2"));
        later.messages.push(user("third turn"));
        assert_eq!(conversation_fingerprint(&turn1), conversation_fingerprint(&later));

        // Different conversation (different first user message) → different id.
        let other = req(vec![sysm, user("totally different")]);
        assert_ne!(conversation_fingerprint(&turn1), conversation_fingerprint(&other));

        // No user message → empty (falls back to old prompt-bucket grouping).
        assert_eq!(conversation_fingerprint(&req(vec![sys("only system")])), "");

        // Non-ASCII beyond the 400-byte cap: truncation must land on a char
        // boundary (a naive byte slice panics mid-codepoint).
        let wide = req(vec![
            sys(&"系统提示".repeat(200)),
            user(&format!("Automated capture turn — 用户 — {}", "…".repeat(200))),
        ]);
        let fp = conversation_fingerprint(&wide);
        assert_eq!(fp.len(), 16);

        // Truncation cap must not merge distinct conversations that share
        // the first 400 bytes: differ inside the cap → different id.
        let a = req(vec![sys("S"), user(&format!("X{}tail1", "p".repeat(300)))]);
        let b = req(vec![sys("S"), user(&format!("X{}tail2", "p".repeat(300)))]);
        assert_ne!(conversation_fingerprint(&a), conversation_fingerprint(&b));
    }

    fn sys(content: &str) -> ChatMessage {
        ChatMessage {
            role: "system".to_string(),
            content: Some(serde_json::Value::String(content.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn golden_body_shim_equals_native_sanitization() {
        // The shared llama-server-family contract: whatever the shim sends,
        // the native adapter must send byte-identically.
        let base: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "catalog-model",
            "messages": [
                {"role": "developer", "content": "dev instructions"},
                {"role": "system", "content": "system instructions"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": null}
            ]
        }))
        .unwrap();
        let mut shim = base.clone();
        sanitize_for_llama_family(&mut shim);
        let mut native = base;
        sanitize_for_llama_family(&mut native);
        let shim_bytes = serde_json::to_vec(&shim).unwrap();
        let native_bytes = serde_json::to_vec(&native).unwrap();
        assert_eq!(shim_bytes, native_bytes, "golden body is endpoint-independent");
        assert_eq!(shim.messages[0].role, "system");
        assert_eq!(
            shim.messages.iter().filter(|m| m.role == "system").count(),
            1,
            "developer→system and the system block merged"
        );
        // Idempotent (a re-applied sanitizer must not reshape again).
        let mut twice = shim.clone();
        sanitize_for_llama_family(&mut twice);
        assert_eq!(
            serde_json::to_vec(&twice).unwrap(),
            shim_bytes,
            "sanitization is idempotent"
        );
    }

    #[test]
    fn merge_empty() {
        let result = merge_system_messages(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn merge_single_passthrough() {
        let msgs = vec![sys("You are helpful.")];
        let result = merge_system_messages(msgs);
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].content.as_ref().unwrap().as_str().unwrap(),
            "You are helpful."
        );
    }

    #[test]
    fn merge_multiple_concatenates() {
        let msgs = vec![
            sys("You are a coding assistant."),
            sys("{\"type\":\"function\",\"function\":{\"name\":\"read\"}}"),
            sys("Extra instructions."),
        ];
        let result = merge_system_messages(msgs);
        assert_eq!(result.len(), 1);
        let text = result[0].content.as_ref().unwrap().as_str().unwrap();
        assert!(text.contains("coding assistant"));
        assert!(text.contains("\"function\""));
        assert!(text.contains("Extra instructions"));
        // Sections separated by double newlines
        assert!(text.contains("\n\n"));
    }

    #[test]
    fn merge_preserves_non_string_content() {
        let mut msgs = vec![sys("Prompt")];
        msgs.push(ChatMessage {
            role: "system".to_string(),
            content: Some(serde_json::json!({"tool": "schema"})),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
        let result = merge_system_messages(msgs);
        assert_eq!(result.len(), 1);
        let text = result[0].content.as_ref().unwrap().as_str().unwrap();
        assert!(text.contains("Prompt"));
        assert!(text.contains("tool"));
        assert!(text.contains("schema"));
    }

    #[test]
    fn merge_skips_none_content() {
        let msgs = vec![
            sys("Prompt"),
            ChatMessage {
                role: "system".to_string(),
                content: None,
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            sys("More"),
        ];
        let result = merge_system_messages(msgs);
        assert_eq!(result.len(), 1);
        let text = result[0].content.as_ref().unwrap().as_str().unwrap();
        assert_eq!(text, "Prompt\n\nMore");
    }

    #[test]
    fn sanitize_fills_empty_assistant_content() {
        let mut msgs = vec![
            sys("Prompt"),
            ChatMessage {
                role: "assistant".to_string(),
                content: None,
                name: None,
                tool_calls: Some(vec![serde_json::json!({"id": "1", "function": {"name": "read"}})]),
                tool_call_id: None,
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: Some(serde_json::Value::String("Hello".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ];
        sanitize_assistant_messages(&mut msgs);
        // First assistant had None content → now empty string
        assert_eq!(msgs[1].content.as_ref().unwrap().as_str().unwrap(), "");
        // Second assistant already had content → unchanged
        assert_eq!(msgs[2].content.as_ref().unwrap().as_str().unwrap(), "Hello");
        // System message untouched
        assert_eq!(msgs[0].role, "system");
    }

    #[tokio::test]
    async fn update_tps_backfills_matching_success_event() {
        let events = RoutingEvents::new();
        events.emit(RouteEvent {
            id: 0,
            timestamp: String::new(),
            prompt_excerpt: "hi".to_string(),
            requested_model: "auto".to_string(),
            effective_provider: Some("llama-swap".to_string()),
            model_key: "m".to_string(),
            latency_ms: 1,
            stage: Stage::LocalPrimary,
            success: true,
            error: String::new(),
            bonsai_decision: "local",
            routing_class: "bonsai → local",
            cwd: String::new(),
            session_id: None,
            user_agent: String::new(),
            source: None,
            conv_id: "abc123".to_string(),
            pp_tps: 0.0,
            tg_tps: 0.0,
            toolbox_backend: Some("llama_cpp".to_string()),
            compute_api: Some("vulkan".to_string()),
            role: "main".to_string(),
            decision: None,
            rationale: None,
            decision_fallback: None,
            chosen_model: None,
            swap_state: None,
            swap_source: None,
        });
        // Matching conversation backfills.
        assert!(events.update_tps("abc123", 41.5, 77.0));
        let evs = events.get_all();
        assert_eq!(evs[0].pp_tps, 41.5);
        assert_eq!(evs[0].tg_tps, 77.0);
        // Unknown conversation is a no-op.
        assert!(!events.update_tps("nope", 1.0, 1.0));
    }

    fn tps_test_event(model: &str) -> RouteEvent {
        RouteEvent {
            id: 0,
            timestamp: String::new(),
            prompt_excerpt: "hi".to_string(),
            requested_model: "auto".to_string(),
            effective_provider: Some("llama-swap".to_string()),
            model_key: model.to_string(),
            latency_ms: 1,
            stage: Stage::LocalPrimary,
            success: true,
            error: String::new(),
            bonsai_decision: "local",
            routing_class: "bonsai → local",
            cwd: String::new(),
            session_id: None,
            user_agent: String::new(),
            source: None,
            conv_id: "same-conversation".to_string(),
            pp_tps: 0.0,
            tg_tps: 0.0,
            toolbox_backend: Some("llama_cpp".to_string()),
            compute_api: Some("vulkan".to_string()),
            role: "main".to_string(),
            decision: None,
            rationale: None,
            decision_fallback: None,
            chosen_model: None,
            swap_state: None,
            swap_source: None,
        }
    }

    fn tps_fixture(
        events: &Arc<RoutingEvents>,
        model: &str,
        stream: crate::provider::SseStream,
    ) -> (u64, crate::provider::SseStream) {
        let event = tps_test_event(model);
        let provider = event.effective_provider.clone();
        let id = events.emit(event);
        let ProviderResponse::Stream(stream) = wrap_with_tps_capture(
            ProviderResponse::Stream(stream), Arc::clone(events), id,
            model.to_string(), provider, Instant::now() - Duration::from_millis(25),
        );
        (id, stream)
    }

    fn tps_output(content: &str) -> Bytes {
        Bytes::from(format!(
            "data: {}\n\n",
            serde_json::json!({"choices": [{"delta": {"content": content}}]}),
        ))
    }

    #[tokio::test]
    async fn tps_capture_stream_correlates_overlapping_same_conversation_requests() {
        use tokio_stream::wrappers::ReceiverStream;

        let events = Arc::new(RoutingEvents::new());
        let (tx_a, rx_a) = tokio::sync::mpsc::channel::<anyhow::Result<Bytes>>(8);
        let (tx_b, rx_b) = tokio::sync::mpsc::channel::<anyhow::Result<Bytes>>(8);
        let (id_a, mut stream_a) = tps_fixture(&events, "model-a", Box::pin(ReceiverStream::new(rx_a)));
        let (id_b, mut stream_b) = tps_fixture(&events, "model-b", Box::pin(ReceiverStream::new(rx_b)));
        assert_ne!(id_a, id_b);
        tx_a.send(Ok(tps_output("a"))).await.unwrap();
        tx_b.send(Ok(tps_output("b"))).await.unwrap();
        stream_a.next().await.unwrap().unwrap();
        stream_b.next().await.unwrap().unwrap();
        assert!(events.get_measurements().is_empty());

        // Finish the newer request first, then the older one. Conversation-based
        // backfill would overwrite model-b's rate with model-a's measurement.
        for (tx, stream, tokens) in [
            (tx_b, &mut stream_b, 11),
            (tx_a, &mut stream_a, 3),
        ] {
            tx.send(Ok(tps_output("tail"))).await.unwrap();
            tx.send(Ok(Bytes::from(format!(
                "data: {{\"usage\":{{\"prompt_tokens\":100,\"completion_tokens\":{tokens}}}}}\n\n",
            )))).await.unwrap();
            tx.send(Ok(Bytes::from_static(b"data: [DONE]\n\n"))).await.unwrap();
            drop(tx);
            while let Some(item) = stream.next().await {
                item.unwrap();
            }
        }
        let samples = events.get_measurements();
        assert_eq!(samples.len(), 2);
        assert_eq!((samples[0].event_id, samples[0].model_key.as_str()), (id_a, "model-a"));
        assert_eq!((samples[1].event_id, samples[1].model_key.as_str()), (id_b, "model-b"));
        assert_eq!(samples[0].completion_tokens, Some(3));
        assert_eq!(samples[1].completion_tokens, Some(11));
        for sample in &samples {
            assert!(sample.measured_ttft_ms.unwrap() >= 25.0);
            assert!(sample.generation_tps.unwrap() > 0.0);
            assert!(sample.stream_duration_ms >= sample.measured_ttft_ms.unwrap());
            assert_eq!(sample.effective_provider.as_deref(), Some("llama-swap"));
            chrono::DateTime::parse_from_rfc3339(&sample.completed_at).unwrap();
            let event = events.get_all().into_iter().find(|e| e.id == sample.event_id).unwrap();
            assert_eq!(event.tg_tps, sample.generation_tps.unwrap());
            assert_eq!(event.pp_tps, 0.0);
        }
        // Samples already exist before Drop, and subsequent EOF polls do not duplicate them.
        assert!(stream_a.next().await.is_none());
        drop((stream_a, stream_b));
        assert_eq!(events.get_measurements().len(), 2);
    }

    #[test]
    fn tps_capture_ignores_role_heartbeat_and_measures_exact_output_interval() {
        let start = Instant::now();
        let mut capture = SseMeasurementCapture::new(start);
        for bytes in [
            b": heartbeat with \"quotes\"\n\n".as_slice(),
            b"event: ping\ndata: {\"type\":\"ping\"}\n\n",
            b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
            b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"x\",\"function\":{\"name\":\"f\",\"arguments\":\"\"}}]}}]}\n\n",
        ] {
            capture.observe_bytes(bytes, start + Duration::from_millis(10));
            assert!(capture.first_output.is_none());
        }
        capture.observe_bytes(
            b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n",
            start + Duration::from_millis(100),
        );
        capture.observe_bytes(&tps_output("answer"), start + Duration::from_millis(300));
        capture.observe_bytes(
            b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":5}}\n\n",
            start + Duration::from_millis(600),
        );
        capture.observe_bytes(b"data: [DONE]\n\n", start + Duration::from_millis(800));
        let sample = capture.measurement(1, "model".into(), None).unwrap();
        assert_eq!(sample.measured_ttft_ms, Some(100.0));
        assert_eq!(sample.generation_tps, Some(20.0)); // (5 - 1) / 0.2; excludes usage/DONE tail.
        assert_eq!(sample.stream_duration_ms, 800.0);
        assert_eq!(sample.prompt_tokens, Some(100));
        assert_eq!(sample.completion_tokens, Some(5));
    }

    #[test]
    fn tps_capture_recognizes_reasoning_and_tool_argument_output() {
        for delta in [
            serde_json::json!({"content": " "}),
            serde_json::json!({"reasoning": "think"}),
            serde_json::json!({"reasoning_content": "think"}),
            serde_json::json!({"tool_calls": [{"function": {"arguments": "{\"a\":"}}]}),
            serde_json::json!({"function_call": {"arguments": "{}"}}),
        ] {
            let start = Instant::now();
            let mut capture = SseMeasurementCapture::new(start);
            let frame = format!("data: {}\n\n", serde_json::json!({"choices": [{"delta": delta}]}));
            capture.observe_bytes(frame.as_bytes(), start + Duration::from_millis(50));
            capture.observe_bytes(b"data: [DONE]\n\n", start + Duration::from_millis(100));
            let sample = capture.measurement(1, String::new(), None).unwrap();
            assert_eq!(sample.measured_ttft_ms, Some(50.0));
            assert_eq!(sample.generation_tps, None);
        }
    }

    #[test]
    fn tps_capture_decodes_split_utf8_crlf_multiline_and_multiple_frames() {
        let start = Instant::now();
        let wire = concat!(
            "\u{feff}: greeting\r\n\r\n",
            "data: {\"choices\":\r\n",
            "data: [{\"delta\":{\"content\":\"héllo\"}}]}\r\n\r\n",
            "data: {\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3}}\r\n\r\n",
            "data: [DONE]\r\n\r\n",
        );
        // Every split position includes UTF-8, CRLF, JSON, and DONE boundaries.
        for split in 0..=wire.len() {
            let mut capture = SseMeasurementCapture::new(start);
            capture.observe_bytes(&wire.as_bytes()[..split], start + Duration::from_millis(10));
            capture.observe_bytes(&wire.as_bytes()[split..], start + Duration::from_millis(20));
            let sample = capture.measurement(1, String::new(), None).unwrap();
            assert!(sample.measured_ttft_ms.is_some());
            assert_eq!(sample.prompt_tokens, Some(7));
            assert_eq!(sample.completion_tokens, Some(3));
            assert_eq!(sample.generation_tps, None);
        }
        for separator in ["\r", "\n", "\r\n"] {
            let wire = format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"é\"}}}}]}}{separator}{separator}data: [DONE]{separator}{separator}");
            let mut capture = SseMeasurementCapture::new(start);
            for (index, byte) in wire.bytes().enumerate() {
                capture.observe_bytes(&[byte], start + Duration::from_millis(index as u64 + 1));
            }
            assert!(capture.measurement(1, String::new(), None).unwrap().measured_ttft_ms.is_some());
        }
    }

    #[test]
    fn tps_capture_missing_malformed_conflicting_usage_stays_unknown() {
        for usage in [
            "",
            "data: {\"usage\":null}\n\n",
            "data: {\"usage\":{}}\n\n",
            "data: {\"usage\":false}\n\n",
            "data: {\"usage\":{\"prompt_tokens\":\"10\",\"completion_tokens\":-3}}\n\n",
            "data: {\"usage\":{\"prompt_tokens\":1.5,\"completion_tokens\":2.0}}\n\n",
            "data: {\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":4}}\n\ndata: {\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":5}}\n\n",
        ] {
            let start = Instant::now();
            let mut capture = SseMeasurementCapture::new(start);
            capture.observe_bytes(&tps_output("a"), start + Duration::from_millis(10));
            capture.observe_bytes(&tps_output("b"), start + Duration::from_millis(20));
            capture.observe_bytes(usage.as_bytes(), start + Duration::from_millis(30));
            capture.observe_bytes(b"data: [DONE]\n\n", start + Duration::from_millis(40));
            let sample = capture.measurement(1, String::new(), None).unwrap();
            assert_eq!(sample.measured_ttft_ms, Some(10.0), "{usage}");
            assert_eq!(sample.prompt_tokens, None, "{usage}");
            assert_eq!(sample.completion_tokens, None, "{usage}");
            assert_eq!(sample.generation_tps, None, "{usage}");
        }
        let start = Instant::now();
        let mut capture = SseMeasurementCapture::new(start);
        capture.observe_bytes(&tps_output("a"), start);
        capture.observe_bytes(&tps_output("b"), start + Duration::from_millis(100));
        capture.observe_bytes(
            b"data: {\"usage\":{\"prompt_tokens\":null,\"completion_tokens\":11}}\n\ndata: [DONE]\n\n",
            start + Duration::from_millis(200),
        );
        let sample = capture.measurement(1, String::new(), None).unwrap();
        assert_eq!(sample.prompt_tokens, None);
        assert_eq!(sample.completion_tokens, Some(11));
        assert_eq!(sample.generation_tps, Some(100.0));
    }

    #[test]
    fn tps_capture_requires_enough_tokens_and_distinct_output_times() {
        for tokens in [0, 1, 2] {
            for separate_times in [false, true] {
                let start = Instant::now();
                let mut capture = SseMeasurementCapture::new(start);
                capture.observe_bytes(&tps_output("a"), start);
                let last = start + Duration::from_millis(if separate_times { 100 } else { 0 });
                capture.observe_bytes(&tps_output("b"), last);
                let ending = format!("data: {{\"usage\":{{\"completion_tokens\":{tokens}}}}}\n\ndata: [DONE]\n\n");
                capture.observe_bytes(ending.as_bytes(), last);
                let sample = capture.measurement(1, String::new(), None).unwrap();
                assert_eq!(sample.completion_tokens, Some(tokens));
                assert_eq!(sample.generation_tps.is_some(), tokens > 1 && separate_times);
            }
        }
        let start = Instant::now();
        let mut capture = SseMeasurementCapture::new(start);
        capture.observe_bytes(b"data: [DONE]\n\n", start);
        let sample = capture.measurement(1, String::new(), None).unwrap();
        assert_eq!(sample.measured_ttft_ms, None);
        assert_eq!(sample.generation_tps, None);
    }

    #[tokio::test]
    async fn tps_capture_excludes_errors_incomplete_streams_and_cancellation() {
        let cases: Vec<Vec<anyhow::Result<Bytes>>> = vec![
            vec![],
            vec![Ok(tps_output("unfinished"))],
            vec![Ok(Bytes::from_static(b"data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\n"))],
            vec![Ok(Bytes::from_static(b"data: [DONE]"))],
            vec![Ok(Bytes::from_static(b"data: [DONE]\n"))],
            vec![Ok(Bytes::from_static(b"data: [DONE]\n\ndata: {"))],
            vec![Ok(Bytes::from_static(b"data: {bad json}\n\ndata: [DONE]\n\n"))],
            vec![Ok(Bytes::from_static(b"data: {\"error\":{\"message\":\"bad\"}}\n\ndata: [DONE]\n\n"))],
            vec![Ok(Bytes::from_static(b"event: error\ndata: {}\n\ndata: [DONE]\n\n"))],
            vec![Ok(Bytes::from_static(b"data: {\"type\":\"response.incomplete\"}\n\ndata: [DONE]\n\n"))],
            vec![Ok(Bytes::from_static(b"data: {\"choices\":[{\"finish_reason\":\"error\"}]}\n\ndata: [DONE]\n\n"))],
            vec![Err(anyhow!("transport failed")), Ok(Bytes::from_static(b"data: [DONE]\n\n"))],
            vec![Ok(Bytes::from_static(b"data: [DONE]\n\n")), Err(anyhow!("late transport failure"))],
            vec![Ok(Bytes::from_static(b"data: [DONE]\n\ndata: {\"error\":\"late failure\"}\n\n"))],
            vec![Ok(Bytes::from_static(b"data: \"\xff\"\n\ndata: [DONE]\n\n"))],
        ];
        for chunks in cases {
            let events = Arc::new(RoutingEvents::new());
            let (_, mut stream) = tps_fixture(&events, "model", Box::pin(fstream::iter(chunks)));
            while stream.next().await.is_some() {}
            drop(stream);
            assert!(events.get_measurements().is_empty());
            assert_eq!(events.get_all()[0].tg_tps, 0.0);
        }
        // Explicit cancellation before EOF, both before and after the DONE marker.
        for ending in ["", "data: [DONE]\n\n"] {
            let events = Arc::new(RoutingEvents::new());
            let chunks = vec![
                Ok(tps_output("a")), Ok(tps_output("b")),
                Ok(Bytes::from(format!(
                    "data: {{\"usage\":{{\"prompt_tokens\":10,\"completion_tokens\":20}}}}\n\n{ending}",
                ))),
            ];
            let (_, mut stream) = tps_fixture(&events, "model", Box::pin(fstream::iter(chunks)));
            for _ in 0..3 {
                stream.next().await.unwrap().unwrap();
            }
            drop(stream);
            assert!(events.get_measurements().is_empty());
            assert_eq!(events.get_all()[0].tg_tps, 0.0);
        }
    }

    #[tokio::test]
    async fn tps_capture_bounds_decoder_without_changing_forwarded_bytes() {
        for bytes in [
            vec![b'x'; MAX_MEASUREMENT_FRAME_BYTES + 1],
            b"data: \n".repeat(MAX_MEASUREMENT_FRAME_BYTES / 7 + 1),
            b": comment\n".repeat(MAX_MEASUREMENT_FRAME_BYTES / 10 + 1),
        ] {
            let start = Instant::now();
            let mut capture = SseMeasurementCapture::new(start);
            capture.observe_bytes(&bytes, start);
            assert!(capture.invalid);
            assert!(capture.line.is_empty());
            assert!(capture.data.is_empty());
            capture.observe_bytes(b"\n\ndata: [DONE]\n\n", start);
            assert!(capture.measurement(1, String::new(), None).is_none());

            let events = Arc::new(RoutingEvents::new());
            let input = Bytes::from(bytes);
            let chunks = vec![Ok(input.clone()), Ok(Bytes::from_static(b"\n\ndata: [DONE]\n\n"))];
            let (_, mut stream) = tps_fixture(&events, "model", Box::pin(fstream::iter(chunks)));
            assert_eq!(stream.next().await.unwrap().unwrap(), input);
            while stream.next().await.is_some() {}
            assert!(events.get_measurements().is_empty());
        }
    }

    #[tokio::test]
    async fn tps_capture_missing_usage_still_records_ttft_with_empty_conversation() {
        let events = Arc::new(RoutingEvents::new());
        let mut event = tps_test_event("model");
        event.conv_id.clear();
        event.session_id = Some("review-session".into());
        let id = events.emit(event);
        let chunks = vec![Ok(tps_output("a")), Ok(Bytes::from_static(b"data: [DONE]\n\n"))];
        let ProviderResponse::Stream(mut stream) = wrap_with_tps_capture(
            ProviderResponse::Stream(Box::pin(fstream::iter(chunks))),
            Arc::clone(&events), id, "model".into(), Some("llama-swap".into()), Instant::now(),
        );
        while let Some(item) = stream.next().await {
            item.unwrap();
        };
        let samples = events.get_measurements();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].event_id, id);
        assert!(samples[0].measured_ttft_ms.is_some());
        assert_eq!(samples[0].generation_tps, None);
        assert_eq!(samples[0].prompt_tokens, None);
        assert_eq!(samples[0].completion_tokens, None);
    }

    #[tokio::test]
    async fn tps_capture_anthropic_drains_usage_and_eof_before_terminal_events() {
        use crate::anthropic::AnthropicSseAdapter;
        use futures_util::FutureExt;
        use tokio_stream::wrappers::ReceiverStream;

        let events = Arc::new(RoutingEvents::new());
        let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<Bytes>>(8);
        let (id, stream) = tps_fixture(&events, "model", Box::pin(ReceiverStream::new(rx)));
        let mut adapter = AnthropicSseAdapter::new(stream, "model".into());
        tx.send(Ok(tps_output("hello"))).await.unwrap();
        loop {
            let frame = adapter.next().await.unwrap().unwrap();
            assert!(!String::from_utf8_lossy(&frame).contains("message_stop"));
            if String::from_utf8_lossy(&frame).contains("content_block_delta") {
                break;
            }
        }
        tx.send(Ok(tps_output(" world"))).await.unwrap();
        assert!(String::from_utf8_lossy(&adapter.next().await.unwrap().unwrap()).contains(" world"));
        for chunk in [
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
            "data: {\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":9}}\n\n",
            "data: [DO",
            "NE]\r\n\r\n",
        ] {
            tx.send(Ok(Bytes::from(chunk))).await.unwrap();
            assert!(adapter.next().now_or_never().is_none(), "terminal output before EOF");
            assert!(events.get_measurements().is_empty());
        }
        drop(tx);
        let closing = adapter.next().await.unwrap().unwrap();
        assert!(String::from_utf8_lossy(&closing).contains("content_block_stop"));
        // Even a client stopping at the terminal event cannot bypass the inner EOF.
        let samples = events.get_measurements();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].event_id, id);
        assert_eq!(samples[0].completion_tokens, Some(9));
        assert!(samples[0].generation_tps.unwrap() > 0.0);
        let delta = adapter.next().await.unwrap().unwrap();
        let delta = String::from_utf8_lossy(&delta);
        assert!(delta.contains("\"stop_reason\":\"max_tokens\""));
        assert!(delta.contains("\"output_tokens\":9"));
        assert!(String::from_utf8_lossy(&adapter.next().await.unwrap().unwrap()).contains("message_stop"));
        assert!(adapter.next().await.is_none());
        assert_eq!(events.get_measurements().len(), 1);
    }

    #[tokio::test]
    async fn tps_capture_anthropic_preserves_split_utf8_and_ending_frames() {
        let events = Arc::new(RoutingEvents::new());
        let wire = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"héllo\"}}]}\r\n\r\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\r\n\r\n",
            "data: {\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":5}}\r\n\r\n",
            "data: [DONE]\r\n\r\n",
        );
        let chunks: Vec<anyhow::Result<Bytes>> = wire.bytes().map(|byte| Ok(Bytes::from(vec![byte]))).collect();
        let (_, stream) = tps_fixture(&events, "model", Box::pin(fstream::iter(chunks)));
        let mut adapter = crate::anthropic::AnthropicSseAdapter::new(stream, "model".into());
        let mut output = String::new();
        while let Some(frame) = adapter.next().await {
            output.push_str(std::str::from_utf8(&frame.unwrap()).unwrap());
        }
        assert!(output.contains("héllo"));
        assert_eq!(output.matches("event: message_stop").count(), 1);
        assert!(output.contains("\"output_tokens\":5"));
        assert_eq!(events.get_measurements().len(), 1);
        assert_eq!(events.get_measurements()[0].completion_tokens, Some(5));
    }

    #[tokio::test]
    async fn tps_capture_anthropic_excludes_late_errors_and_missing_done() {
        let cases: Vec<(Vec<anyhow::Result<Bytes>>, bool)> = vec![
            (vec![Err(anyhow!("late provider failure"))], true),
            (vec![Ok(Bytes::from_static(b"data: {\"error\":\"late SSE error\"}\n\n"))], true),
            (vec![Ok(Bytes::from_static(b"event: error\ndata: {}\n\n"))], true),
            (vec![Ok(Bytes::from_static(b"data: {\"type\":\"response.cancelled\"}\n\n"))], true),
            (vec![Ok(Bytes::from_static(b"data: [DONE]\n\n")), Err(anyhow!("error after DONE"))], true),
            (vec![Ok(Bytes::from_static(b"data: {\"usage\":"))], true),
            (vec![], false),
        ];
        for (tail, expect_error) in cases {
            let events = Arc::new(RoutingEvents::new());
            let mut chunks = vec![
                Ok(tps_output("hello")),
                Ok(Bytes::from_static(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")),
                Ok(Bytes::from_static(b"data: {\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":5}}\n\n")),
            ];
            chunks.extend(tail);
            let (_, stream) = tps_fixture(&events, "model", Box::pin(fstream::iter(chunks)));
            let adapter = crate::anthropic::AnthropicSseAdapter::new(stream, "model".into());
            let frames: Vec<_> = adapter.collect().await;
            assert_eq!(frames.iter().any(|frame| frame.is_err()), expect_error);
            let terminal = frames.iter().filter_map(|frame| frame.as_ref().ok())
                .any(|frame| String::from_utf8_lossy(frame).contains("message_stop"));
            assert_eq!(terminal, !expect_error);
            assert!(events.get_measurements().is_empty());
        }
    }

    #[tokio::test]
    async fn tps_capture_anthropic_cancellation_during_tail_drain_records_nothing() {
        use futures_util::FutureExt;
        use tokio_stream::wrappers::ReceiverStream;

        for done in ["", "data: [DONE]\n\n"] {
            let events = Arc::new(RoutingEvents::new());
            let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<Bytes>>(8);
            let (_, stream) = tps_fixture(&events, "model", Box::pin(ReceiverStream::new(rx)));
            let mut adapter = crate::anthropic::AnthropicSseAdapter::new(stream, "model".into());
            tx.send(Ok(tps_output("hello"))).await.unwrap();
            loop {
                let frame = adapter.next().await.unwrap().unwrap();
                if String::from_utf8_lossy(&frame).contains("content_block_delta") {
                    break;
                }
            }
            tx.send(Ok(Bytes::from(format!(
                "data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\ndata: {{\"usage\":{{\"completion_tokens\":5}}}}\n\n{done}",
            )))).await.unwrap();
            assert!(adapter.next().now_or_never().is_none());
            drop(adapter);
            drop(tx);
            assert!(events.get_measurements().is_empty());
        }
    }

    #[tokio::test(start_paused = true)]
    async fn tps_capture_anthropic_bounds_tail_after_finish_or_done() {
        use crate::anthropic::{AnthropicSseAdapter, ANTHROPIC_EOF_GRACE};
        use futures_util::FutureExt;

        for ending in [
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
        ] {
            let events = Arc::new(RoutingEvents::new());
            let chunks = vec![Ok(tps_output("hello")), Ok(Bytes::from(ending))];
            let source = fstream::iter(chunks).chain(fstream::pending());
            let (_, stream) = tps_fixture(&events, "model", Box::pin(source));
            let mut adapter = AnthropicSseAdapter::new(stream, "model".into());
            loop {
                let frame = adapter.next().await.unwrap().unwrap();
                if String::from_utf8_lossy(&frame).contains("content_block_delta") {
                    break;
                }
            }
            assert!(adapter.next().now_or_never().is_none());
            tokio::time::advance(ANTHROPIC_EOF_GRACE + Duration::from_millis(1)).await;
            let error = adapter.next().await.unwrap().unwrap_err();
            assert!(error.to_string().contains("tail-drain deadline"));
            assert!(adapter.next().await.is_none());
            assert!(events.get_measurements().is_empty());
        }
    }

    #[tokio::test(start_paused = true)]
    async fn tps_capture_anthropic_done_does_not_extend_finish_deadline() {
        use crate::anthropic::{AnthropicSseAdapter, ANTHROPIC_EOF_GRACE};
        use futures_util::FutureExt;
        use tokio_stream::wrappers::ReceiverStream;

        let events = Arc::new(RoutingEvents::new());
        let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<Bytes>>(8);
        let (_, stream) = tps_fixture(&events, "model", Box::pin(ReceiverStream::new(rx)));
        let mut adapter = AnthropicSseAdapter::new(stream, "model".into());
        tx.send(Ok(tps_output("hello"))).await.unwrap();
        loop {
            let frame = adapter.next().await.unwrap().unwrap();
            if String::from_utf8_lossy(&frame).contains("content_block_delta") {
                break;
            }
        }
        tx.send(Ok(Bytes::from_static(
            b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        ))).await.unwrap();
        assert!(adapter.next().now_or_never().is_none());
        tokio::time::advance(ANTHROPIC_EOF_GRACE / 2).await;
        tx.send(Ok(Bytes::from_static(
            b"data: {\"usage\":{\"completion_tokens\":5}}\n\ndata: [DONE]\n\n",
        ))).await.unwrap();
        assert!(adapter.next().now_or_never().is_none());
        tokio::time::advance(ANTHROPIC_EOF_GRACE / 2 + Duration::from_millis(1)).await;
        let error = adapter.next().await.unwrap().unwrap_err();
        assert!(error.to_string().contains("tail-drain deadline"));
        assert!(adapter.next().await.is_none());
        assert!(events.get_measurements().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn tps_capture_anthropic_ready_tail_wins_after_consumer_backpressure() {
        use crate::anthropic::{AnthropicSseAdapter, ANTHROPIC_EOF_GRACE};

        for ending in ["ready-eof", "ready-data", "ready-error"] {
            let events = Arc::new(RoutingEvents::new());
            let first = "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\n";
            let usage = "data: {\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":5}}\n\n";
            let done = "data: [DONE]\n\n";
            let chunks = match ending {
                "ready-eof" => vec![Ok(Bytes::from(format!("{first}{usage}{done}")))],
                "ready-data" => vec![Ok(Bytes::from(first)), Ok(Bytes::from(usage)), Ok(Bytes::from(done))],
                _ => vec![Ok(Bytes::from(first)), Err(anyhow!("original upstream failure"))],
            };
            let (id, stream) = tps_fixture(&events, "model", Box::pin(fstream::iter(chunks)));
            let mut adapter = AnthropicSseAdapter::new(stream, "model".into());
            let first_frame = adapter.next().await.unwrap().unwrap();
            assert!(String::from_utf8_lossy(&first_frame).contains("message_start"));
            assert!(events.get_measurements().is_empty());
            // finish_reason was already decoded, but downstream has not yet
            // consumed the queued content/start frames or polled the ready tail.
            tokio::time::advance(ANTHROPIC_EOF_GRACE + Duration::from_secs(1)).await;
            let frames: Vec<_> = adapter.collect().await;
            if ending == "ready-error" {
                let error = frames.iter().find_map(|frame| frame.as_ref().err()).unwrap();
                assert_eq!(error.to_string(), "original upstream failure");
                assert!(events.get_measurements().is_empty());
            } else {
                assert!(frames.iter().all(|frame| frame.is_ok()));
                let text: String = frames.iter().map(|frame| {
                    String::from_utf8_lossy(frame.as_ref().unwrap()).into_owned()
                }).collect();
                assert!(text.contains("\"output_tokens\":5"));
                assert_eq!(text.matches("event: message_stop").count(), 1);
                let samples = events.get_measurements();
                assert_eq!(samples.len(), 1);
                assert_eq!(samples[0].event_id, id);
                assert_eq!(samples[0].completion_tokens, Some(5));
            }
        }
    }

    #[tokio::test]
    async fn tps_capture_anthropic_caps_ready_tail_bytes_and_partial_lines() {
        use crate::anthropic::{
            AnthropicSseAdapter, ANTHROPIC_TAIL_MAX_BYTES, ANTHROPIC_TAIL_MAX_LINE_BYTES,
        };

        for (tail, expected_error) in [
            (vec![b'x'; ANTHROPIC_TAIL_MAX_LINE_BYTES + 1], "line budget"),
            (b":\n".repeat(ANTHROPIC_TAIL_MAX_BYTES / 2 + 1), "byte budget"),
        ] {
            for same_chunk in [false, true] {
                let events = Arc::new(RoutingEvents::new());
                let finish = b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n";
                let mut chunks = vec![Ok(tps_output("hello"))];
                if same_chunk {
                    let mut bytes = finish.to_vec();
                    bytes.extend_from_slice(&tail);
                    chunks.push(Ok(Bytes::from(bytes)));
                } else {
                    chunks.push(Ok(Bytes::from_static(finish)));
                    // Split the tail across chunks to exercise cumulative limits.
                    chunks.extend(tail.chunks(1024).map(|bytes| Ok(Bytes::copy_from_slice(bytes))));
                }
                let (_, stream) = tps_fixture(&events, "model", Box::pin(fstream::iter(chunks)));
                let adapter = AnthropicSseAdapter::new(stream, "model".into());
                let frames: Vec<_> = adapter.collect().await;
                let error = frames.iter().find_map(|frame| frame.as_ref().err()).unwrap();
                assert!(error.to_string().contains(expected_error), "{error}");
                assert!(events.get_measurements().is_empty());
                assert!(!frames.iter().filter_map(|frame| frame.as_ref().ok())
                    .any(|frame| String::from_utf8_lossy(frame).contains("message_stop")));
            }
        }
    }

    #[tokio::test]
    async fn tps_capture_anthropic_caps_always_ready_empty_tail_work() {
        use crate::anthropic::{AnthropicSseAdapter, ANTHROPIC_TAIL_MAX_CHUNKS};
        use std::sync::atomic::AtomicUsize;

        let events = Arc::new(RoutingEvents::new());
        let polls = Arc::new(AtomicUsize::new(0));
        let tail_polls = Arc::clone(&polls);
        let tail = fstream::poll_fn(move |_| {
            let count = tail_polls.fetch_add(1, Ordering::Relaxed) + 1;
            // Fail rather than spin forever if the work guard regresses.
            assert!(count <= ANTHROPIC_TAIL_MAX_CHUNKS + 1);
            std::task::Poll::Ready(Some(Ok(Bytes::new())))
        });
        let chunks = vec![
            Ok(tps_output("hello")),
            Ok(Bytes::from_static(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")),
        ];
        let (_, stream) = tps_fixture(&events, "model", Box::pin(fstream::iter(chunks).chain(tail)));
        let adapter = AnthropicSseAdapter::new(stream, "model".into());
        let frames: Vec<_> = adapter.collect().await;
        let error = frames.iter().find_map(|frame| frame.as_ref().err()).unwrap();
        assert!(error.to_string().contains("chunk budget"));
        assert_eq!(polls.load(Ordering::Relaxed), ANTHROPIC_TAIL_MAX_CHUNKS + 1);
        assert!(events.get_measurements().is_empty());
    }

    #[tokio::test]
    async fn tps_capture_anthropic_tail_line_cap_does_not_restrict_prefinish_content() {
        use crate::anthropic::{AnthropicSseAdapter, ANTHROPIC_TAIL_MAX_LINE_BYTES};

        let events = Arc::new(RoutingEvents::new());
        let content = "x".repeat(ANTHROPIC_TAIL_MAX_LINE_BYTES + 1);
        let chunks = vec![
            Ok(tps_output(&content)),
            Ok(Bytes::from_static(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")),
        ];
        let (_, stream) = tps_fixture(&events, "model", Box::pin(fstream::iter(chunks)));
        let adapter = AnthropicSseAdapter::new(stream, "model".into());
        let frames: Vec<_> = adapter.collect().await;
        assert!(frames.iter().all(|frame| frame.is_ok()));
        assert!(frames.iter().any(|frame| String::from_utf8_lossy(frame.as_ref().unwrap()).contains(&content)));
        assert_eq!(events.get_measurements().len(), 1);
    }
}