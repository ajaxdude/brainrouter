//! The decision-model slot (v2 L3, R4): features → {route, tier} with a
//! pluggable decision model.
//!
//! Kinds (`decision_model.kind`):
//! - `heuristic` — v1 selector-matrix parity (classifier-off anchor: local);
//!   always available, the default, and the fallback target for every model
//!   kind.
//! - `openai-compatible` — `{base_url, model, prompt_profile, timeout_ms}`.
//!   Bonsai migrates to exactly this with `prompt_profile: bonsai`, preserving
//!   the v1 prompt + one-word grammar verbatim (the wire code lives in
//!   `src/classifier.rs`).
//! - `structured-schema` — the typed Clef question (`schema_version: 1`),
//!   strict JSON reply.
//! - `webhook` — POST the features JSON (NO prompt excerpt) to `{url}`.
//! - `plugin` — server-initiated `decision.classify` over the plugin RPC
//!   (protocol v2; request-id correlation + `timeout_ms`).
//!
//! Failure matrix (binding, L3 spec): the bonsai profile answers every model
//! failure (HTTP error, timeout, parse fail, empty/missing last message, AND
//! server-down — v1's single failure mode) with `route=cloud` +
//! `fallback=v1-safe`; there is deliberately NO availability pre-check for it
//! (v1 has none). Fresh-config kinds (structured-schema / webhook / plugin)
//! fail SAFE: any failure, including unavailable, ⇒ heuristic +
//! `fallback=unavailable`. Low confidence (< `min_confidence`, default 0.4) ⇒
//! heuristic + `fallback=heuristic` for every model kind.
//!
//! Guardrail clamps apply to SLOT decisions only — never to explicit selectors
//! (explicit = owner intent): privacy `local-only` forces `route=local`, and
//! the quality floor raises the tier only.

use crate::{
    classifier::{self, BudgetTier, RoutingDecision},
    config::DecisionModelConfig,
    roles::{Privacy, Quality},
    types::ChatCompletionRequest,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::future::Future;
use std::{pin::Pin, sync::Arc, time::Duration};
use tracing::{debug, warn};

/// Canon decision vocabulary. `tier` maps: quality chips `fast→fast,
/// balanced→balanced, maximum→deep`; v1 classifier words `local→fast,
/// deep→deep`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Route {
    Local,
    Cloud,
}

impl Route {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Cloud => "cloud",
        }
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Fast,
    Balanced,
    Deep,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Balanced => "balanced",
            Self::Deep => "deep",
        }
    }

    /// Nudge budget map (L3 spec): `fast→Light, balanced→Light, deep→Deep`.
    pub fn budget_tier(self) -> BudgetTier {
        match self {
            Self::Fast | Self::Balanced => BudgetTier::Light,
            Self::Deep => BudgetTier::Deep,
        }
    }
}

/// The decision + provenance a slot returns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    pub route: Route,
    pub tier: Tier,
    pub confidence: f64,
    pub rationale: String,
}

/// Why the slot fell back (event `decision_fallback`; contract values).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallback {
    /// Bonsai profile: v1's safe default (every model failure ⇒ cloud).
    V1Safe,
    /// Low confidence ⇒ heuristic.
    Heuristic,
    /// Fresh-config kind: any failure incl. unavailable ⇒ heuristic.
    Unavailable,
}

impl Fallback {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::V1Safe => "v1-safe",
            Self::Heuristic => "heuristic",
            Self::Unavailable => "unavailable",
        }
    }
}

/// Slot outcome: the decision plus the fallback provenance (None = the kind
/// answered cleanly).
#[derive(Debug, Clone)]
pub struct SlotOutcome {
    pub decision: Decision,
    pub fallback: Option<Fallback>,
}

#[derive(Debug, Clone)]
pub enum DecisionError {
    /// HTTP error / timeout / connection failure / server down.
    Unavailable(String),
    /// Reply present but unparseable, empty, or schema-violating.
    Parse(String),
    /// No content to classify (e.g. missing last user message).
    Empty,
}

/// Everything a decision model may see. The prompt excerpt is included ONLY
/// for model kinds (`openai-compatible` / `structured-schema`); the webhook
/// payload carries the feature summary and MUST NOT carry
/// `last_user_excerpt` (design Security: "prompt content sent to classifiers =
/// feature summary").
#[derive(Debug, Clone)]
pub struct RequestFeatures {
    pub role: String,
    pub selector: String,
    pub privacy: Privacy,
    pub quality: Quality,
    pub project: String,
    pub msg_count: usize,
    pub approx_prompt_tokens: u64,
    pub has_tools: bool,
    pub code_density: usize,
    pub last_user_excerpt: Option<String>,
}

impl RequestFeatures {
    /// The features JSON (shared by webhook POST and the structured-schema
    /// `features` block). `with_excerpt` only for model kinds.
    pub fn to_json(&self, with_excerpt: bool) -> Value {
        let mut v = json!({
            "role": self.role,
            "selector": self.selector,
            "privacy": self.privacy.as_str(),
            "quality": self.quality.as_str(),
            "project": self.project,
            "msg_count": self.msg_count,
            "approx_prompt_tokens": self.approx_prompt_tokens,
            "has_tools": self.has_tools,
            "code_density": self.code_density,
        });
        if with_excerpt {
            if let Some(ex) = &self.last_user_excerpt {
                v["last_user_excerpt"] = json!(ex);
            }
        }
        v
    }

    /// `(bytes+3)/4` on the raw body at route time (L3 spec; precedent
    /// `src/benchmark_lab.rs` — L4 replaces with measured data).
    pub fn approx_tokens_of_body(bytes: usize) -> u64 {
        ((bytes as u64) + 3) / 4
    }

    /// Build the features from a request + routing context.
    pub fn extract(
        request: &ChatCompletionRequest,
        raw_body_bytes: usize,
        role: &str,
        selector: &str,
        privacy: Privacy,
        quality: Quality,
        project: &str,
    ) -> Self {
        let has_tools = request
            .extra
            .get("tools")
            .and_then(Value::as_array)
            .is_some_and(|t| !t.is_empty());
        let mut code_density = 0usize;
        for msg in &request.messages {
            if let Some(Value::String(text)) = &msg.content {
                code_density += text.matches("```").count() / 2;
            }
        }
        let last_user_excerpt = classifier::extract_last_user_message(request);
        Self {
            role: role.to_string(),
            selector: selector.to_string(),
            privacy,
            quality,
            project: project.to_string(),
            msg_count: request.messages.len(),
            approx_prompt_tokens: Self::approx_tokens_of_body(raw_body_bytes),
            has_tools,
            code_density,
            last_user_excerpt: (!last_user_excerpt.is_empty()).then_some(last_user_excerpt),
        }
    }
}

/// The pluggable decision model. Boxed futures keep the trait object-safe so
/// the slot can hold `Arc<dyn DecisionModel>`.
pub trait DecisionModel: Send + Sync {
    fn kind(&self) -> &'static str;
    /// Cheap config-level availability. NOT consulted for the bonsai profile
    /// (v1 has no pre-check — its server-down row IS the HTTP failure).
    fn available(&self) -> bool;
    fn classify<'a>(
        &'a self,
        features: &'a RequestFeatures,
    ) -> Pin<Box<dyn Future<Output = Result<Decision, DecisionError>> + Send + 'a>>;
}

// ── Heuristic (default; v1 selector-matrix parity) ───────────────────────────

/// v1 parity anchor: today's classifier-OFF routing (`route_auto` with Bonsai
/// disabled) is LOCAL to the default local model — the chain seed. Tier is
/// the Light-budget canon `fast`. The subs pool selector is local-direct, so
/// the parity table has no cloud row: with no classifier signal, everything
/// managed routes local.
pub struct HeuristicModel;

impl DecisionModel for HeuristicModel {
    fn kind(&self) -> &'static str {
        "heuristic"
    }
    fn available(&self) -> bool {
        true
    }
    fn classify<'a>(
        &'a self,
        features: &'a RequestFeatures,
    ) -> Pin<Box<dyn Future<Output = Result<Decision, DecisionError>> + Send + 'a>> {
        Box::pin(async move {
            Ok(Decision {
                route: Route::Local,
                tier: Tier::Fast,
                confidence: 1.0,
                rationale: format!(
                    "heuristic parity: {} routes local (v1 classifier-off anchor)",
                    features.role
                ),
            })
        })
    }
}

// ── OpenAI-compatible (incl. the migrated bonsai profile) ────────────────────

/// An OpenAI-compatible classifier endpoint. With `prompt_profile: bonsai` it
/// delegates to the verbatim v1 classifier wire (`classifier::classify_raw`);
/// the one-word grammar parse stays in that file too.
pub struct OpenAiCompatibleModel {
    config: DecisionModelConfig,
    classifier: Arc<crate::classifier::Classifier>,
}

impl OpenAiCompatibleModel {
    pub fn new(config: DecisionModelConfig, classifier: Arc<crate::classifier::Classifier>) -> Self {
        Self { config, classifier }
    }
}

impl DecisionModel for OpenAiCompatibleModel {
    fn kind(&self) -> &'static str {
        "openai-compatible"
    }
    fn available(&self) -> bool {
        // No HTTP probe: v1 has no availability pre-check, and the L3 spec
        // pins that for the bonsai profile ("the availability pre-check must
        // NOT short-circuit the bonsai profile to heuristic").
        true
    }
    fn classify<'a>(
        &'a self,
        features: &'a RequestFeatures,
    ) -> Pin<Box<dyn Future<Output = Result<Decision, DecisionError>> + Send + 'a>> {
        Box::pin(async move {
        if self.config.prompt_profile.as_deref() == Some("bonsai") {
            // One word reply: delegate to the v1 wire. A missing last user
            // message is v1's Cloud default (failure row); the classifier
            // handles the grammar (first-char parse) internally.
            let request = synthetic_request(features);
            let (decision, quality) = self.classifier.classify_with_provenance(request).await;
            let (route, tier) = match decision {
                RoutingDecision::Cloud => (Route::Cloud, Tier::Balanced),
                RoutingDecision::Local { tier, .. } => (
                    Route::Local,
                    match tier {
                        BudgetTier::Light => Tier::Fast,
                        BudgetTier::Deep => Tier::Deep,
                    },
                ),
            };
            Ok(Decision {
                route,
                tier,
                confidence: 1.0,
                rationale: match quality {
                    classifier::ClassifyQuality::Clean(w) =>
                        format!("bonsai one-word grammar: {w}"),
                    classifier::ClassifyQuality::V1Safe(reason) => {
                        // Binding failure matrix: every bonsai failure row is
                        // route=cloud + v1-safe.
                        return Err(DecisionError::Unavailable(reason));
                    }
                },
            })
        } else {
            // Generic prompt profile: the endpoint answers with the decision
            // JSON in choices[0].message.content.
            let base = self.config.base_url.clone().unwrap_or_default();
            let body = json!({
                "model": self.config.model.clone().unwrap_or_default(),
                "messages": [
                    {"role": "system", "content": "You are a routing classifier. Reply with JSON {\"route\":\"local|cloud\",\"tier\":\"fast|balanced|deep\",\"confidence\":0.0-1.0,\"rationale\":\"...\"} and nothing else."},
                    {"role": "user", "content": features.to_json(true)},
                ],
                "temperature": 0.0,
                "max_tokens": 64,
            });
            let text = post_json(&base, "/v1/chat/completions", body, timeout_of(&self.config)).await?;
            parse_decision_json(&text).map(|mut d| {
                d.rationale = format!("openai-compatible: {}", d.rationale);
                d
            })
        }
        })
    }
}

// ── Structured-schema (the Clef contract) ────────────────────────────────────

pub struct StructuredSchemaModel {
    config: DecisionModelConfig,
}

impl StructuredSchemaModel {
    pub fn new(config: DecisionModelConfig) -> Self {
        Self { config }
    }
}

impl DecisionModel for StructuredSchemaModel {
    fn kind(&self) -> &'static str {
        "structured-schema"
    }
    fn available(&self) -> bool {
        self.config.base_url.is_some() && self.config.model.is_some()
    }
    fn classify<'a>(
        &'a self,
        features: &'a RequestFeatures,
    ) -> Pin<Box<dyn Future<Output = Result<Decision, DecisionError>> + Send + 'a>> {
        Box::pin(async move {
        let base = self.config.base_url.clone().unwrap_or_default();
        // Pinned wire: schema_version 1 + feature summary (no excerpt — the
        // question carries only the schema) + the closed question.
        let body = json!({
            "model": self.config.model.clone().unwrap_or_default(),
            "messages": [
                {"role": "user", "content": json!({
                    "schema_version": 1,
                    "features": features.to_json(true),
                    "question": {
                        "route": ["local", "cloud"],
                        "tier": ["fast", "balanced", "deep"],
                    },
                })},
            ],
            "temperature": 0.0,
            "max_tokens": 128,
        });
        let text = post_json(&base, "/v1/chat/completions", body, timeout_of(&self.config)).await?;
        parse_decision_json(&text).map(|mut d| {
            d.rationale = format!("structured-schema: {}", d.rationale);
            d
        })
        })
    }
}

// ── Webhook ──────────────────────────────────────────────────────────────────

pub struct WebhookModel {
    config: DecisionModelConfig,
}

impl WebhookModel {
    pub fn new(config: DecisionModelConfig) -> Self {
        Self { config }
    }
}

impl DecisionModel for WebhookModel {
    fn kind(&self) -> &'static str {
        "webhook"
    }
    fn available(&self) -> bool {
        self.config.url.is_some()
    }
    fn classify<'a>(
        &'a self,
        features: &'a RequestFeatures,
    ) -> Pin<Box<dyn Future<Output = Result<Decision, DecisionError>> + Send + 'a>> {
        Box::pin(async move {
        let url = self.config.url.clone().unwrap_or_default();
        let timeout = timeout_of(&self.config);
        // NO excerpt: the webhook speaks the feature summary only.
        let body = features.to_json(false);
        let text = crate::provider::util::post_json_body(&url, &body, timeout)
            .await
            .map_err(|e| DecisionError::Unavailable(e.to_string()))?;
        let decision = decision_from_value_str(&text)?;
        // The webhook's rationale passes through VERBATIM (contract).
        Ok(decision)
        })
    }
}

// ── Plugin (server-initiated decision.classify over plugin RPC v2) ──────────

pub struct PluginModel {
    config: DecisionModelConfig,
    manager: Arc<crate::plugins::PluginManager>,
}

impl PluginModel {
    pub fn new(config: DecisionModelConfig, manager: Arc<crate::plugins::PluginManager>) -> Self {
        Self { config, manager }
    }
}

impl DecisionModel for PluginModel {
    fn kind(&self) -> &'static str {
        "plugin"
    }
    fn available(&self) -> bool {
        // The plugin must be installed, enabled, and RPC-connected
        // (protocol-version-2 sessions register the decision surface).
        self.config
            .plugin_id
            .as_deref()
            .is_some_and(|id| self.manager.has_live_session(id))
    }
    fn classify<'a>(
        &'a self,
        features: &'a RequestFeatures,
    ) -> Pin<Box<dyn Future<Output = Result<Decision, DecisionError>> + Send + 'a>> {
        Box::pin(async move {
        let id = self
            .config
            .plugin_id
            .clone()
            .ok_or_else(|| DecisionError::Unavailable("no plugin_id configured".into()))?;
        let timeout_ms = self.config.timeout_ms.unwrap_or(5_000);
        // Server-initiated request with request-id correlation + timeout
        // (plugin RPC protocol v2, negotiated at initialize).
        let result = self
            .manager
            .request_decision(&id, features.to_json(true), timeout_ms)
            .await
            .map_err(|e| DecisionError::Unavailable(e.to_string()))?;
        let decision = decision_from_value(result)
            .map_err(|e| DecisionError::Parse(format!("plugin reply: {e}")))?;
        Ok(Decision {
            rationale: format!("plugin {id}: {}", decision.rationale),
            ..decision
        })
        })
    }
}

// ── Slot: kind dispatch + failure matrix + clamps ────────────────────────────

pub struct DecisionSlot {
    config: DecisionModelConfig,
    model: Arc<dyn DecisionModel>,
    heuristic: HeuristicModel,
}

impl DecisionSlot {
    pub fn enabled(&self) -> bool {
        self.config.enabled
    }

    pub fn is_bonsai_profile(&self) -> bool {
        self.config.is_bonsai_profile()
    }

    pub fn new(
        config: DecisionModelConfig,
        classifier: Arc<crate::classifier::Classifier>,
        plugins: Option<Arc<crate::plugins::PluginManager>>,
    ) -> Self {
        let model: Arc<dyn DecisionModel> = match config.kind {
            crate::config::DecisionKind::Heuristic => Arc::new(HeuristicModel),
            crate::config::DecisionKind::OpenaiCompatible => {
                Arc::new(OpenAiCompatibleModel::new(config.clone(), classifier))
            }
            crate::config::DecisionKind::StructuredSchema => {
                Arc::new(StructuredSchemaModel::new(config.clone()))
            }
            crate::config::DecisionKind::Webhook => Arc::new(WebhookModel::new(config.clone())),
            crate::config::DecisionKind::Plugin => match plugins {
                Some(m) => Arc::new(PluginModel::new(config.clone(), m)),
                // A plugin kind without the plugin subsystem degrades to the
                // heuristic (its failures are all "unavailable" rows).
                None => Arc::new(HeuristicModel),
            },
        };
        Self {
            config,
            model,
            heuristic: HeuristicModel,
        }
    }

    /// The kind/config the prompt-rewrite gate needs (L3 spec invariant
    /// "prompt rewriting requires the decision slot"): the handler applies
    /// `enabled` + the bonsai-profile health check (async, server-side).
    pub fn rewrite_requires_launch_health(&self) -> bool {
        self.config.is_bonsai_profile() && self.config.launch.is_some()
    }

    pub fn config(&self) -> &DecisionModelConfig {
        &self.config
    }

    pub fn kind(&self) -> &'static str {
        self.model.kind()
    }

    pub fn available(&self) -> bool {
        self.model.available()
    }

    /// The public `decision show` shape (contract pin).
    pub fn describe(&self) -> Value {
        json!({
            "enabled": self.config.enabled,
            "kind": self.kind(),
            "available": self.available(),
            "min_confidence": self.config.min_confidence(),
            "base_url": self.config.base_url,
            "model": self.config.model,
            "prompt_profile": self.config.prompt_profile,
            "timeout_ms": self.config.timeout_ms,
            "url": self.config.url,
            "plugin_id": self.config.plugin_id,
        })
    }

    /// Run the slot for one request. A DISABLED slot, the heuristic kind, and
    /// every model-kind failure resolve through the same shape: a decision +
    /// an optional fallback label (event `decision_fallback`).
    pub async fn decide(&self, features: &RequestFeatures) -> SlotOutcome {
        if !self.config.enabled {
            // Disabled slot = parity anchor (v1: classifier off → local).
            let decision = self.heuristic.classify(features).await.expect("heuristic infallible");
            return SlotOutcome { decision, fallback: None };
        }
        if self.model.kind() == "heuristic" {
            let decision = self.model.classify(features).await.expect("heuristic infallible");
            return SlotOutcome { decision, fallback: None };
        }
        let is_bonsai = self.config.is_bonsai_profile();
        // Fresh-config kinds get the availability pre-check; the bonsai profile
        // NEVER does (spec: v1 has none — server-down is its v1-safe row).
        if !is_bonsai && !self.model.available() {
            return self.heuristic_fallback(features, Fallback::Unavailable).await;
        }
        match self.model.classify(features).await {
            Ok(decision) => {
                if decision.confidence < self.config.min_confidence() {
                    return self.heuristic_fallback(features, Fallback::Heuristic).await;
                }
                SlotOutcome { decision, fallback: None }
            }
            Err(e) => {
                if is_bonsai {
                    // Binding matrix: bonsai profile failures ⇒ cloud, v1-safe
                    // (route=cloud IS the v1 safe default).
                    let label = match &e {
                        DecisionError::Unavailable(m) => m.as_str(),
                        DecisionError::Parse(m) => m.as_str(),
                        DecisionError::Empty => "empty",
                    };
                    warn!(error = %label, "bonsai classifier failed; v1-safe cloud");
                    SlotOutcome {
                        decision: Decision {
                            route: Route::Cloud,
                            tier: Tier::Balanced,
                            confidence: 0.0,
                            rationale: format!("v1-safe: classifier failure ({label}) → cloud"),
                        },
                        fallback: Some(Fallback::V1Safe),
                    }
                } else {
                    warn!(error = ?e, "decision model failed; heuristic fallback");
                    self.heuristic_fallback(features, Fallback::Unavailable).await
                }
            }
        }
    }

    async fn heuristic_fallback(
        &self,
        features: &RequestFeatures,
        label: Fallback,
    ) -> SlotOutcome {
        let decision = self.heuristic.classify(features).await.expect("heuristic infallible");
        SlotOutcome {
            decision,
            fallback: Some(label),
        }
    }

    /// Guardrail clamps — SLOT decisions only, never explicit selectors.
    /// Privacy `local-only` forces route=local; the quality floor raises tier.
    pub fn clamp(outcome: &mut SlotOutcome, privacy: Privacy, quality: Quality) {
        if privacy == Privacy::LocalOnly && outcome.decision.route != Route::Local {
            debug!("privacy local-only: clamped slot decision to local");
            outcome.decision.route = Route::Local;
            outcome.decision.rationale =
                format!("privacy clamp → local; {}", outcome.decision.rationale);
        }
        let floor = match quality {
            Quality::Fast => Tier::Fast,
            Quality::Balanced => Tier::Balanced,
            Quality::Maximum => Tier::Deep,
        };
        if outcome.decision.tier < floor {
            outcome.decision.tier = floor;
        }
    }
}

// ── Wire helpers ─────────────────────────────────────────────────────────────

fn timeout_of(config: &DecisionModelConfig) -> Duration {
    Duration::from_millis(config.timeout_ms.unwrap_or(15_000))
}

/// Rebuild a single-message request from the features excerpt for the
/// verbatim v1 classifier wire (which reads the LAST user message itself).
fn synthetic_request(features: &RequestFeatures) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "auto".to_string(),
        messages: features
            .last_user_excerpt
            .as_ref()
            .map(|text| {
                vec![crate::types::ChatMessage {
                    role: "user".to_string(),
                    content: Some(Value::String(text.clone())),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                }]
            })
            .unwrap_or_default(),
        stream: None,
        temperature: None,
        max_tokens: None,
        top_p: None,
        stop: None,
        extra: json!({}),
    }
}

async fn post_json(base: &str, path: &str, body: Value, timeout: Duration) -> Result<String, DecisionError> {
    let url = format!("{}{}", base.trim_end_matches('/'), path);
    crate::provider::util::post_json_body(&url, &body, timeout)
        .await
        .map_err(|e| DecisionError::Unavailable(e.to_string()))
}

/// Parse a chat-completion reply, extracting the decision JSON from
/// `choices[0].message.content` (STRICT JSON — no fences, no prose).
fn parse_decision_json(raw: &str) -> Result<Decision, DecisionError> {
    let v: Value = serde_json::from_str(raw)
        .map_err(|e| DecisionError::Parse(format!("chat completion envelope: {e}")))?;
    let content = v
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if content.trim().is_empty() {
        return Err(DecisionError::Parse("empty classifier content".into()));
    }
    let decision = decision_from_value_str(content)?;
    Ok(decision)
}

fn decision_from_value(v: Value) -> Result<Decision, String> {
    let route = v.get("route").and_then(Value::as_str).unwrap_or("");
    let tier = v.get("tier").and_then(Value::as_str).unwrap_or("balanced");
    let confidence = v.get("confidence").and_then(Value::as_f64).unwrap_or(0.0);
    let rationale = v
        .get("rationale")
        .and_then(Value::as_str)
        .unwrap_or("no rationale")
        .to_string();
    let route = match route {
        "local" => Route::Local,
        "cloud" => Route::Cloud,
        other => return Err(format!("unknown route {other:?}")),
    };
    let tier = match tier {
        "fast" => Tier::Fast,
        "balanced" => Tier::Balanced,
        "deep" => Tier::Deep,
        _ => Tier::Balanced,
    };
    Ok(Decision {
        route,
        tier,
        confidence,
        rationale,
    })
}

fn decision_from_value_str(raw: &str) -> Result<Decision, DecisionError> {
    let v: Value = serde_json::from_str(raw.trim())
        .map_err(|e| DecisionError::Parse(format!("decision JSON: {e}")))?;
    decision_from_value(v).map_err(DecisionError::Parse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DecisionKind;

    fn features() -> RequestFeatures {
        RequestFeatures {
            role: "main".into(),
            selector: "auto".into(),
            privacy: Privacy::CloudOk,
            quality: Quality::Balanced,
            project: String::new(),
            msg_count: 1,
            approx_prompt_tokens: 12,
            has_tools: false,
            code_density: 0,
            last_user_excerpt: Some("hello".into()),
        }
    }

    #[tokio::test]
    async fn heuristic_parity_is_local_fast() {
        let slot = DecisionSlot::new(
            DecisionModelConfig {
                enabled: true,
                kind: DecisionKind::Heuristic,
                ..Default::default()
            },
            Arc::new(crate::classifier::Classifier::new(
                "http://127.0.0.1:1".into(),
                "fb".into(),
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                None,
            )),
            None,
        );
        let out = slot.decide(&features()).await;
        assert_eq!(out.decision.route, Route::Local);
        assert_eq!(out.decision.tier, Tier::Fast);
        assert_eq!(out.fallback, None);
    }

    #[tokio::test]
    async fn disabled_slot_is_parity_anchor() {
        let slot = DecisionSlot::new(
            DecisionModelConfig::default(),
            Arc::new(crate::classifier::Classifier::new(
                "http://127.0.0.1:1".into(),
                "fb".into(),
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                None,
            )),
            None,
        );
        let out = slot.decide(&features()).await;
        assert_eq!(out.decision.route, Route::Local);
        assert_eq!(out.fallback, None);
    }

    #[tokio::test]
    async fn bonsai_profile_server_down_is_v1_safe_cloud() {
        let slot = DecisionSlot::new(
            DecisionModelConfig {
                enabled: true,
                kind: DecisionKind::OpenaiCompatible,
                base_url: Some("http://127.0.0.1:1".into()), // refused = server down
                model: Some("bonsai".into()),
                prompt_profile: Some("bonsai".into()),
                timeout_ms: Some(1000),
                ..Default::default()
            },
            Arc::new(crate::classifier::Classifier::new(
                "http://127.0.0.1:1".into(),
                "fb".into(),
                Arc::new(std::sync::atomic::AtomicBool::new(true)),
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                None,
            )),
            None,
        );
        let out = slot.decide(&features()).await;
        assert_eq!(out.decision.route, Route::Cloud);
        assert_eq!(out.fallback, Some(Fallback::V1Safe));
    }

    #[tokio::test]
    async fn fresh_config_parse_failure_is_unavailable_heuristic() {
        let slot = DecisionSlot::new(
            DecisionModelConfig {
                enabled: true,
                kind: DecisionKind::StructuredSchema,
                base_url: Some("http://127.0.0.1:1".into()),
                model: Some("schema".into()),
                timeout_ms: Some(1000),
                ..Default::default()
            },
            Arc::new(crate::classifier::Classifier::new(
                "http://127.0.0.1:1".into(),
                "fb".into(),
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                None,
            )),
            None,
        );
        let out = slot.decide(&features()).await;
        assert_eq!(out.decision.route, Route::Local);
        assert_eq!(out.fallback, Some(Fallback::Unavailable));
    }

    #[tokio::test]
    async fn low_confidence_falls_back_to_heuristic() {
        // A webhook answering confidence 0.1 (< default 0.4).
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 65536];
            // Drain until the full header block is in.
            let mut text = String::new();
            loop {
                let n = sock.read(&mut buf).await.unwrap();
                if n == 0 {
                    break;
                }
                text.push_str(&String::from_utf8_lossy(&buf[..n]));
                if text.contains("\r\n\r\n") {
                    break;
                }
            }
            let body = json!({"route":"cloud","tier":"fast","confidence":0.1}).to_string();
            let _ = sock
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                        body.len(),
                        body
                    )
                    .as_bytes(),
                )
                .await;
        });
        let slot = DecisionSlot::new(
            DecisionModelConfig {
                enabled: true,
                kind: DecisionKind::Webhook,
                url: Some(format!("http://{addr}")),
                ..Default::default()
            },
            Arc::new(crate::classifier::Classifier::new(
                "http://127.0.0.1:1".into(),
                "fb".into(),
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                None,
            )),
            None,
        );
        let out = slot.decide(&features()).await;
        assert_eq!(out.decision.route, Route::Local, "heuristic parity");
        assert_eq!(out.fallback, Some(Fallback::Heuristic));
    }

    #[test]
    fn privacy_clamp_forces_local_never_affects_cloud_floor() {
        let mut out = SlotOutcome {
            decision: Decision {
                route: Route::Cloud,
                tier: Tier::Fast,
                confidence: 0.9,
                rationale: "x".into(),
            },
            fallback: None,
        };
        DecisionSlot::clamp(&mut out, Privacy::LocalOnly, Quality::Maximum);
        assert_eq!(out.decision.route, Route::Local);
        assert_eq!(out.decision.tier, Tier::Deep, "quality floor raises tier only");
    }

    #[test]
    fn webhook_payload_never_carries_excerpt() {
        let f = features();
        let v = f.to_json(false);
        assert!(v.get("last_user_excerpt").is_none());
        assert_eq!(v["msg_count"], json!(1));
        assert!(v.get("role").is_some() && v.get("selector").is_some() && v.get("privacy").is_some());
    }

    #[test]
    fn approx_tokens_rounds_up() {
        assert_eq!(RequestFeatures::approx_tokens_of_body(0), 0);
        assert_eq!(RequestFeatures::approx_tokens_of_body(1), 1);
        assert_eq!(RequestFeatures::approx_tokens_of_body(4), 1);
        assert_eq!(RequestFeatures::approx_tokens_of_body(5), 2);
    }
}
