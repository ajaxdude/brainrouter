//! Request router.
//!
//! The router is the core of brainrouter. For every incoming request it:
//!   1. Asks Bonsai (via `Classifier`) whether the query should go to Cloud
//!      (Manifest) or Local (llama-swap with a specific model).
//!   2. Dispatches to the chosen backend.
//!   3. On cloud failure / stall, falls through to llama-swap with the
//!      configured fallback model.
//!   4. Wraps the resulting stream in TimeoutStream so a stalled provider
//!      surfaces as an error instead of hanging the client. KeepaliveStream is
//!      applied in server.rs where the wire format (OpenAI/Anthropic) is known.

use crate::{
    classifier::{BudgetTier, Classifier, RoutingDecision},
    config::NudgeBudgets,
    health::HealthTracker,
    inference_state::{InferenceTracker, Phase},
    prompt_rewriter,
    provider::{openai::OpenAiProvider, Provider, ProviderResponse},
    routing_events::{RouteEvent, RoutingEvents, Stage},
    stream::TimeoutStream,
    types::{ChatCompletionRequest, ChatMessage},
};
use anyhow::{anyhow, Result};
use futures_util::{stream as fstream, StreamExt};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::{sync::Arc, time::{Duration, Instant}};
use tracing::{debug, info, warn};

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

/// Summary of what actually happened during a route call.
/// Returned alongside the response so callers (e.g. review loop) can
/// record which model handled the request.
pub struct RouteInfo {
    /// "cloud" or "local" — the Bonsai classification.
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

pub struct Router {
    classifier: Arc<Classifier>,
    manifest: Arc<OpenAiProvider>,
    manifest_enabled: bool,
    llama_swap: Arc<OpenAiProvider>,
    fallback_model: String,
    /// Model keys known to belong to llama-swap. Requests for these bypass Bonsai.
    local_models: Vec<String>,
    /// Subs-pool model key for `subs`/`brainrouter/subs` requests.
    subs_model: Option<String>,
    health: Arc<HealthTracker>,
    routing_events: Arc<RoutingEvents>,
    /// Optional custom system prompt for local routing mode.
    local_system_prompt: Option<String>,
    /// Dashboard inference progress tracker.
    pub inference_tracker: Arc<InferenceTracker>,
    /// Reasoning budgets per tier (nudge). Injected as
    /// `reasoning_budget_tokens` on auto-routed local requests when nudge is on.
    nudge_budgets: NudgeBudgets,
    /// Runtime nudge master switch (shared with dashboard).
    nudge_enabled: Arc<AtomicBool>,
    /// Runtime nudge tier override: 0 = auto (Bonsai), 1 = light, 2 = deep.
    nudge_tier: Arc<AtomicU8>,
    /// Runtime prompt-rewrite toggle. When off, local routes forward the
    /// incoming messages untouched (no system-prompt rewrite).
    prompt_rewrite: Arc<AtomicBool>,
}

pub struct RouterArgs {
    pub classifier: Arc<Classifier>,
    pub manifest: Arc<OpenAiProvider>,
    /// Whether the cloud backend (Manifest) is enabled. Off → cloud requests
    /// skip Manifest and fall straight back to llama-swap.
    pub manifest_enabled: bool,
    pub llama_swap: Arc<OpenAiProvider>,
    pub fallback_model: String,
    /// Model keys known to belong to llama-swap. See `LlamaSwapConfig::local_models`.
    pub local_models: Vec<String>,
    /// Subs-pool model key for `subs`/`brainrouter/subs` requests. See
    /// `LlamaSwapConfig::subs_model`.
    pub subs_model: Option<String>,
    pub health: Arc<HealthTracker>,
    pub routing_events: Arc<RoutingEvents>,
    pub local_system_prompt: Option<String>,
    pub inference_tracker: Arc<InferenceTracker>,
    pub nudge_budgets: NudgeBudgets,
    pub nudge_enabled: Arc<AtomicBool>,
    pub nudge_tier: Arc<AtomicU8>,
    pub prompt_rewrite: Arc<AtomicBool>,
}

impl Router {
    pub fn new(args: RouterArgs) -> Self {
        Self {
            classifier: args.classifier,
            manifest: args.manifest,
            manifest_enabled: args.manifest_enabled,
            llama_swap: args.llama_swap,
            fallback_model: args.fallback_model,
            local_models: args.local_models,
            subs_model: args.subs_model,
            health: args.health,
            routing_events: args.routing_events,
            local_system_prompt: args.local_system_prompt,
            inference_tracker: args.inference_tracker,
            nudge_budgets: args.nudge_budgets,
            nudge_enabled: args.nudge_enabled,
            nudge_tier: args.nudge_tier,
            prompt_rewrite: args.prompt_rewrite,
        }
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

    /// Inject the per-request reasoning budget for auto-routed local requests
    /// when nudge is enabled. The field flows through the request's flattened
    /// `extra` map into llama-swap and on to the nudge fork's
    /// `oaicompat_chat_params_parse`. A client-supplied value is never
    /// overridden, and direct/explicit model requests never reach this path.
    fn inject_nudge_budget(&self, request: &mut ChatCompletionRequest, tier: BudgetTier) {
        if !self.nudge_enabled.load(Ordering::Relaxed) {
            return;
        }
        let tier = match self.nudge_tier.load(Ordering::Relaxed) {
            1 => BudgetTier::Light,
            2 => BudgetTier::Deep,
            _ => tier,
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

    /// Route a request, returning the response and metadata about the routing decision.
    /// `session_id` tags the emitted RouteEvent (used by the review loop so events are
    /// linkable to review sessions).
    pub async fn route_tagged(
        &self,
        mut request: ChatCompletionRequest,
        session_id: Option<String>,
        cwd: String,
    ) -> Result<(ProviderResponse, RouteInfo)> {
        let start = Instant::now();
        let requested_model = request.model.clone();
        let prompt_excerpt = extract_prompt_excerpt(&request);

        let tracker = &self.inference_tracker;
        let max_tokens = request.max_tokens;
        let (bonsai_decision, result) = match requested_model.as_str() {
            // Managed local token: route to the local/fallback model, rewrite prompt.
            "local" | "brainrouter/local" => {
                info!("Direct local mode — rewriting system prompt");
                tracker.set(Phase::LocalWaiting, Some(self.fallback_model.clone()), Some("llama-swap".into()), max_tokens);
                request.messages = self.maybe_rewrite_local(request.messages);
                request.model = self.fallback_model.clone();
                ("local-direct", self.route_local(request, true).await)
            }
            // Direct cloud: skip Bonsai, go straight to Manifest
            "cloud" | "brainrouter/cloud" => {
                info!("Direct cloud mode — routing to Manifest");
                tracker.set(Phase::CloudWaiting, None, Some("Manifest".into()), max_tokens);
                ("cloud-direct", self.route_cloud(request).await)
            }
            // Managed routing: Bonsai classify + subs pool. Only these tokens get
            // nudge/bonsai/subs treatment. Direct model keys are authoritative.
            _ => {
                // Subs pool: `subs` or `brainrouter/subs` → subs_model, bypassing
                // Bonsai. Unconfigured → warn and fall back to auto below.
                if requested_model == "subs" || requested_model == "brainrouter/subs" {
                    if let Some(subs) = self.subs_model.clone() {
                        info!(model = %subs, "Subs pool routing — direct to llama-swap");
                        tracker.set(Phase::LocalWaiting, Some(subs.clone()), Some("llama-swap".into()), max_tokens);
                        request.model = subs;
                        ("local-subs", self.route_local(request, false).await)
                    } else {
                        warn!("Subs pool requested but llama_swap.subs_model is not configured — falling back to auto");
                        self.route_auto(request, tracker, max_tokens).await
                    }
                } else if requested_model == "auto" || requested_model == "brainrouter/auto" {
                    // Explicit "auto" — let the classifier decide (Bonsai off → local).
                    self.route_auto(request, tracker, max_tokens).await
                } else if let Some(specific) = requested_model.strip_prefix("brainrouter/") {
                    if !specific.is_empty() {
                        info!(model = specific, "Direct model mode — routing to llama-swap");
                        tracker.set(Phase::LocalWaiting, Some(specific.to_string()), Some("llama-swap".into()), max_tokens);
                        request.model = specific.to_string();
                        ("local-specific", self.route_local(request, false).await)
                    } else {
                        // Empty suffix, treat as auto — fall through to Bonsai
                        self.route_auto(request, tracker, max_tokens).await
                    }
                } else if self.local_models.contains(&requested_model) {
                    // Model is explicitly listed as a local model — route directly
                    // to llama-swap without consulting Bonsai. The user's model
                    // choice is authoritative.
                    info!(model = %requested_model, "Known local model — routing directly to llama-swap");
                    tracker.set(Phase::LocalWaiting, Some(requested_model.clone()), Some("llama-swap".into()), max_tokens);
                    // request.model is already correct (it's the llama-swap model key)
                    ("local-specific", self.route_local(request, false).await)
                } else {
                    // A named model that isn't a reserved routing token (auto/local/
                    // cloud/subs). The user picked it explicitly — route Local directly
                    // to that model in llama-swap. Lets you select any llama-swap model
                    // without a classifier hop; works even when Bonsai is off.
                    info!(model = %requested_model, "Named model — routing directly to llama-swap");
                    tracker.set(Phase::LocalWaiting, Some(requested_model.clone()), Some("llama-swap".into()), max_tokens);
                    ("local-specific", self.route_local(request, false).await)
                }
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
                        cwd: cwd.clone(),
                        session_id: session_id.clone(),
                    });
                }
                self.routing_events.emit(RouteEvent {
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
                    cwd: cwd.clone(),
                    session_id: session_id.clone(),
                });
                // Wrap the stream to clear the tracker when it completes
                let tracker_for_stream = Arc::clone(&self.inference_tracker);
                let resp = wrap_with_tracker_clear(resp, tracker_for_stream);
                (resp, info)
            }
            Err(e) => {
                tracker.clear();
                self.routing_events.emit(RouteEvent {
                    id: 0,
                    timestamp: String::new(),
                    prompt_excerpt,
                    requested_model,
                    effective_provider: None,
                    model_key: String::new(),
                    latency_ms,
                    stage: if bonsai_decision == "cloud" { Stage::CloudPrimary } else { Stage::LocalPrimary },
                    success: false,
                    error: e.to_string(),
                    bonsai_decision,
                    cwd,
                    session_id: session_id.clone(),
                });
                return Err(e);
            }
        };

        Ok((response, info))
    }

    /// Auto path: consult the Bonsai classifier and route Cloud or Local.
    /// Returns the (decision tag, result) pair consumed by `route_tagged`.
    async fn route_auto(
        &self,
        mut request: ChatCompletionRequest,
        tracker: &Arc<InferenceTracker>,
        max_tokens: Option<u32>,
    ) -> (&'static str, Result<(ProviderResponse, RouteInfo)>) {
        tracker.set(Phase::Classifying, None, None, max_tokens);
        let decision = self.classifier.classify_async(request.clone()).await;
        info!(?decision, "Bonsai routing decision");
        match decision {
            RoutingDecision::Cloud => {
                tracker.set(Phase::CloudWaiting, None, Some("Manifest".into()), max_tokens);
                ("cloud", self.route_cloud(request).await)
            }
            RoutingDecision::Local { model, tier } => {
                tracker.set(Phase::LocalWaiting, Some(model.clone()), Some("llama-swap".into()), max_tokens);
                request.model = model;
                request.messages = self.maybe_rewrite_local(request.messages);
                self.inject_nudge_budget(&mut request, tier);
                ("local", self.route_local(request, true).await)
            }
        }
    }

    /// Cloud path: try Manifest first. On error/circuit-open, fall back to
    /// llama-swap with the configured fallback model.
    async fn route_cloud(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Result<(ProviderResponse, RouteInfo)> {
        // Manifest expects "auto" — it does its own model selection.
        request.model = "auto".to_string();

        if !self.manifest_enabled {
            warn!(
                provider = MANIFEST_KEY,
                "Manifest is disabled (manifest.enabled: false) — cloud request falls back to llama-swap"
            );
        } else if self.health.is_healthy(MANIFEST_KEY) {
            info!(provider = MANIFEST_KEY, "Attempting Manifest");
            match self.manifest.chat_completion(request.clone()).await {
                Ok(ProviderResponse::Stream(stream)) => {
                    let (stream, model_key) = peek_manifest_model(stream).await;
                    // Manifest returns HTTP 200 even for errors (credits exhausted,
                    // missing auth) — it sends a single SSE chunk with model="manifest"
                    // and an error message as content. Detect this and fall through
                    // to llama-swap instead of forwarding the error to the client.
                    if model_key == "manifest" {
                        warn!(provider = MANIFEST_KEY, "Manifest returned pseudo-success (model=manifest) — likely credits exhausted or auth error, falling back");
                        // Don't report health failure — Manifest is reachable, just can't fulfill.
                    } else {
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

        // Cloud fallback → llama-swap with fallback_model
        request.model = self.fallback_model.clone();
        let model_key = self.fallback_model.clone();
        let (resp, _) = self.try_llama_swap(request, Stage::CloudFallback).await?;
        Ok((
            resp,
            RouteInfo {
                bonsai_decision: "cloud",
                effective_provider: Some("llama-swap".to_string()),
                model_key,
                failed_attempts: vec![FailedAttempt {
                    stage: Stage::CloudPrimary,
                    provider: "manifest".to_string(),
                    model_key: "auto".to_string(),
                    error: "manifest unavailable (disabled, circuit open, or error)".to_string(),
                }],
            },
        ))
    }

    /// Local path: go straight to llama-swap.
    ///
    /// `allow_fallback` gates the retry-with-fallback behavior. Direct model
    /// selections pass `false` so a failed explicit pick surfaces the error
    /// instead of silently switching to `fallback_model`. Managed routing
    /// (auto/bonsai/local tokens) passes `true` to keep the fallback hop.
    async fn route_local(
        &self,
        request: ChatCompletionRequest,
        allow_fallback: bool,
    ) -> Result<(ProviderResponse, RouteInfo)> {
        let requested = request.model.clone();
        match self.try_llama_swap(request.clone(), Stage::LocalPrimary).await {
            Ok((resp, model_key)) => Ok((
                resp,
                RouteInfo {
                    bonsai_decision: "local",
                    effective_provider: Some("llama-swap".to_string()),
                    model_key,
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
                let (resp, model_key) = self.try_llama_swap(fallback_req, Stage::LocalFallback).await?;
                Ok((
                    resp,
                    RouteInfo {
                        bonsai_decision: "local",
                        effective_provider: Some("llama-swap".to_string()),
                        model_key,
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
        // Sanitize messages for local llama-server compatibility:
        //  1. Normalize 'developer' role → 'system' (OpenAI-style role that Qwen3
        //     templates don't recognize).
        //  2. Merge system messages into one (Qwen3 requires single system msg at pos 0).
        //  3. Ensure assistant messages have content (llama-server rejects assistant
        //     messages with neither content nor tool_calls).
        for msg in &mut request.messages {
            if msg.role == "developer" {
                msg.role = "system".to_string();
            }
        }
        let (sys, rest): (Vec<_>, Vec<_>) = request
            .messages
            .into_iter()
            .partition(|m| m.role == "system");
        request.messages = merge_system_messages(sys)
            .into_iter()
            .chain(rest)
            .collect();
        sanitize_assistant_messages(&mut request.messages);
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

/// Consume the first chunk of a Manifest SSE stream, extract the `model` field
/// from the JSON payload, then reassemble the stream so the chunk is not lost.
///
/// Manifest's first SSE frame looks like:
///   `data: {"id":"...","model":"claude-3-7-sonnet-20250219","choices":[...]}\n\n`
///
/// Returns the reassembled stream and the model name, falling back to
/// `"manifest"` if the chunk is absent or the field cannot be parsed.
async fn peek_manifest_model(
    mut stream: crate::provider::SseStream,
) -> (crate::provider::SseStream, String) {
    let first = match tokio::time::timeout(TTFT_TIMEOUT, stream.next()).await {
        Ok(Some(Ok(chunk))) => chunk,
        Ok(Some(Err(e))) => {
            let err_stream: crate::provider::SseStream =
                Box::pin(fstream::once(async move { Err(e) }).chain(stream));
            return (err_stream, "manifest".to_string());
        }
        Ok(None) => return (Box::pin(fstream::empty()), "manifest".to_string()),
        Err(_elapsed) => {
            // First chunk timed out — surface as a stall error
            let err: anyhow::Error = anyhow!("Manifest stream stalled before first chunk ({}s timeout)", TTFT_TIMEOUT.as_secs());
            let err_stream: crate::provider::SseStream =
                Box::pin(fstream::once(async move { Err(err) }).chain(stream));
            return (err_stream, "manifest".to_string());
        }
    };

    // Scan the chunk for a `data: {` line and attempt to pull out `model`.
    let model = extract_model_from_sse_chunk(&first)
        .unwrap_or_else(|| "manifest".to_string());

    // Prepend the chunk back so downstream consumers see a complete stream.
    let reassembled: crate::provider::SseStream = Box::pin(
        fstream::once(async move { Ok(first) }).chain(stream),
    );
    (reassembled, model)
}

/// Scan raw SSE bytes for the first `data: {` line and extract the `model` field.
/// Returns `None` if parsing fails for any reason.
fn extract_model_from_sse_chunk(chunk: &bytes::Bytes) -> Option<String> {
    let text = std::str::from_utf8(chunk).ok()?;
    for line in text.lines() {
        let json_str = match line.strip_prefix("data: ") {
            Some(s) if s.starts_with('{') => s,
            _ => continue,
        };
        #[derive(serde::Deserialize)]
        struct ModelOnly {
            model: Option<String>,
        }
        if let Ok(parsed) = serde_json::from_str::<ModelOnly>(json_str) {
            if let Some(m) = parsed.model.filter(|s| !s.is_empty()) {
                return Some(m);
            }
        }
    }
    None
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
}