//! Bonsai-powered query classifier.
//!
//! At request time, brainrouter sends the user's last message to the Bonsai
//! 27B llama-server (PrismML fork), which replies with "cloud", "local", or
//! "deep". "deep" routes local with the full reasoning budget when nudge is
//! enabled (voidsurfer/llama.cpp-nudge); "local" uses the lighter budget.

use crate::types::ChatCompletionRequest;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Decision returned by the classifier for an incoming request.
#[derive(Debug, Clone)]
pub enum RoutingDecision {
    /// Forward to Manifest (cloud router). Request model is rewritten to "auto".
    Cloud,
    /// Forward to llama-swap with the given model key.
    ///
    /// `tier` is the reasoning-budget tier picked by the classifier; it only
    /// matters when nudge is enabled (the router injects the matching budget).
    Local { model: String, tier: BudgetTier },
}

/// Reasoning-budget tier for auto-routed local requests (nudge).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetTier {
    /// Simple task — light thinking budget.
    Light,
    /// Complex task that should stay local — full thinking budget.
    Deep,
}

/// Maximum tokens to generate during classification. We only need one word.
const CLASSIFY_MAX_TOKENS: usize = 10;
/// Truncate the user message to this many characters before classifying.
const USER_MSG_TRUNCATE: usize = 800;

/// External Bonsai classifier. Sends prompts to a running llama-server process.
pub struct Classifier {
    /// Base URL of the Bonsai llama-server (e.g. http://127.0.0.1:9200).
    server_url: String,
    /// Default local model used when the request asks for "auto" but Bonsai
    /// chooses local routing without a specific suggestion.
    default_local_model: String,
    /// Shared HTTP client.
    http: Client,
    /// Shared "Bonsai server is on" flag. When the dashboard stops the
    /// llama-server this is cleared and classification skips HTTP, returning
    /// the local default directly (no cloud hop).
    enabled: Arc<AtomicBool>,
    /// Shared nudge master switch (dashboard/config). When on, auto-routed
    /// local requests use the nudge model key so per-request budgets apply.
    nudge_enabled: Arc<AtomicBool>,
    /// llama-swap model key that runs the nudge fork (started with
    /// `--reasoning-budget-enable`). `None` → fall back to the default model.
    nudge_model_key: Option<String>,
}

impl Classifier {
    /// Create a classifier pointing at the external Bonsai llama-server.
    /// `enabled` is the shared flag flipped by `BonsaiControl` when the
    /// server is started or stopped at runtime. `nudge_enabled` is the
    /// runtime nudge master switch; `nudge_model_key` is the llama-swap key
    /// that runs the nudge fork (used only when nudge is enabled).
    pub fn new(
        server_url: String,
        default_local_model: String,
        enabled: Arc<AtomicBool>,
        nudge_enabled: Arc<AtomicBool>,
        nudge_model_key: Option<String>,
    ) -> Self {
        Self {
            server_url,
            default_local_model,
            http: Client::new(),
            enabled,
            nudge_enabled,
            nudge_model_key,
        }
    }

    /// Classify a request asynchronously via the external llama-server.
    /// On any error, defaults to Cloud — the safe default.
    pub async fn classify_async(&self, request: ChatCompletionRequest) -> RoutingDecision {
        if !self.enabled.load(Ordering::Relaxed) {
            // No classifier → "auto" means local. Route directly to llama-swap
            // (Manifest is off by default too) instead of making a pointless
            // cloud hop that would immediately fall back.
            debug!("Bonsai server is off; routing auto requests directly to local");
            let model = if self.nudge_enabled.load(Ordering::Relaxed) {
                self.nudge_model_key
                    .clone()
                    .unwrap_or_else(|| self.default_local_model.clone())
            } else {
                self.default_local_model.clone()
            };
            return RoutingDecision::Local { model, tier: BudgetTier::Light };
        }
        let requested_model = request.model.clone();

        let last_user_msg = extract_last_user_message(&request);
        if last_user_msg.is_empty() {
            debug!("No user message found; defaulting to Cloud");
            return RoutingDecision::Cloud;
        }

        let server_url = self.server_url.clone();
        let http = self.http.clone();
        let result = http
            .post(format!("{}/v1/chat/completions", server_url))
            .json(&ChatCompletionInput {
                model: "bonsai".to_string(),
                messages: vec![
                    ChatMessageInput {
                        role: "system".to_string(),
                        content: format!(
                            "You are a routing classifier. Reply with exactly one word: \"cloud\" for complex tasks that need the cloud (large architecture, multi-system debugging, heavy refactoring), \"local\" for simple tasks (short answers, simple questions, single-line code, quick explanations), or \"deep\" for complex tasks that should still run locally (focused multi-step reasoning, moderate debugging). Output nothing else."
                        ),
                    },
                    ChatMessageInput {
                        role: "user".to_string(),
                        content: format!(
                            "Classify this request: {}",
                            last_user_msg
                        ),
                    },
                ],
                max_tokens: Some(CLASSIFY_MAX_TOKENS),
                temperature: Some(0.0),
                stop: Some(vec!["\n".to_string()]),
                // The Bonsai chat template has thinking mode on by default;
                // the thinking preamble would eat the whole 10-token
                // classification budget before the one-word answer.
                chat_template_kwargs: Some(serde_json::json!({ "enable_thinking": false })),
            })
            .send()
            .await;

        let raw = match result {
            Ok(resp) => match resp.text().await {
                Ok(s) => s,
                Err(e) => {
                    warn!("Classifier HTTP response read failed: {}; defaulting to Cloud", e);
                    return RoutingDecision::Cloud;
                }
            },
            Err(e) => {
                warn!("Classifier HTTP request failed: {}; defaulting to Cloud", e);
                return RoutingDecision::Cloud;
            }
        };

        // Parse the llama-server response: extract the first choice's message content
        let parsed = match serde_json::from_str::<ChatCompletionResponse>(&raw) {
            Ok(r) => r,
            Err(e) => {
                warn!("Classifier response parse failed (raw={:?}): {}; defaulting to Cloud", raw, e);
                return RoutingDecision::Cloud;
            }
        };

        let choice_text = parsed.choices.first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("");

        let decision = parse_decision(choice_text);
        debug!(raw = %choice_text.trim(), ?decision, "Bonsai classification");

        match decision {
            ParsedDecision::Cloud => RoutingDecision::Cloud,
            ParsedDecision::Local | ParsedDecision::Deep => {
                let tier = match decision {
                    ParsedDecision::Deep => BudgetTier::Deep,
                    _ => BudgetTier::Light,
                };
                let user_requested_specific = !matches!(
                    requested_model.as_str(),
                    "auto" | "brainrouter/auto" | "brainrouter" | ""
                );
                let model = if user_requested_specific {
                    requested_model
                } else if self.nudge_enabled.load(Ordering::Relaxed) {
                    // Nudge on: use the fork key so per-request budgets apply.
                    self.nudge_model_key
                        .clone()
                        .unwrap_or_else(|| self.default_local_model.clone())
                } else {
                    self.default_local_model.clone()
                };
                RoutingDecision::Local { model, tier }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
enum ParsedDecision {
    Cloud,
    Local,
    Deep,
}

/// Parse Bonsai's raw output into a decision. Looks at the first non-whitespace
/// character: 'l'/'L' → Local, 'd'/'D' → Deep, anything else → Cloud (safe
/// default).
fn parse_decision(raw: &str) -> ParsedDecision {
    let trimmed = raw.trim_start();
    match trimmed.chars().next() {
        Some('l') | Some('L') => ParsedDecision::Local,
        Some('d') | Some('D') => ParsedDecision::Deep,
        _ => ParsedDecision::Cloud,
    }
}

/// Extract the last user message from the request, truncated for prompt brevity.
fn extract_last_user_message(request: &ChatCompletionRequest) -> String {
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

    if raw.len() > USER_MSG_TRUNCATE {
        let mut end = USER_MSG_TRUNCATE;
        while !raw.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        raw[..end].to_string()
    } else {
        raw
    }
}

#[derive(Serialize)]
struct ChatCompletionInput {
    model: String,
    messages: Vec<ChatMessageInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct ChatMessageInput {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Debug, Deserialize)]
struct Message {
    content: Option<String>,
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cloud() {
        assert!(matches!(parse_decision("cloud"), ParsedDecision::Cloud));
        assert!(matches!(parse_decision("Cloud"), ParsedDecision::Cloud));
        assert!(matches!(parse_decision("  cloud "), ParsedDecision::Cloud));
        // Anything not starting with l/L/d/D defaults to Cloud
        assert!(matches!(parse_decision("xyz"), ParsedDecision::Cloud));
        assert!(matches!(parse_decision(""), ParsedDecision::Cloud));
    }

    #[test]
    fn parse_local() {
        assert!(matches!(parse_decision("local"), ParsedDecision::Local));
        assert!(matches!(parse_decision("Local"), ParsedDecision::Local));
        assert!(matches!(parse_decision("  local"), ParsedDecision::Local));
        assert!(matches!(parse_decision("l"), ParsedDecision::Local));
    }

    #[test]
    fn parse_deep() {
        assert!(matches!(parse_decision("deep"), ParsedDecision::Deep));
        assert!(matches!(parse_decision("Deep"), ParsedDecision::Deep));
        assert!(matches!(parse_decision("  deep"), ParsedDecision::Deep));
        assert!(matches!(parse_decision("d"), ParsedDecision::Deep));
    }
}
