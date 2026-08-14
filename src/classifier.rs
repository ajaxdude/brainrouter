//! Bonsai-powered query classifier.
//!
//! At request time, brainrouter sends the user's last message to the Bonsai
//! 27B llama-server (PrismML fork), which replies with "cloud" or "local".

use crate::types::ChatCompletionRequest;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

/// Decision returned by the classifier for an incoming request.
#[derive(Debug, Clone)]
pub enum RoutingDecision {
    /// Forward to Manifest (cloud router). Request model is rewritten to "auto".
    Cloud,
    /// Forward directly to llama-swap with this specific model name.
    Local { model: String },
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
}

impl Classifier {
    /// Create a classifier pointing at the external Bonsai llama-server.
    pub fn new(server_url: String, default_local_model: String) -> Self {
        Self {
            server_url,
            default_local_model,
            http: Client::new(),
        }
    }

    /// Classify a request asynchronously via the external llama-server.
    /// On any error, defaults to Cloud — the safe default.
    pub async fn classify_async(&self, request: ChatCompletionRequest) -> RoutingDecision {
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
                            "You are a routing classifier. Reply with exactly one word: \"cloud\" for complex tasks (architecture, debugging large systems, multi-step reasoning, refactoring) or \"local\" for simple tasks (short answers, simple questions, single-line code, explanations). Output nothing else."
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
            ParsedDecision::Local => {
                let user_requested_specific = !matches!(
                    requested_model.as_str(),
                    "auto" | "brainrouter/auto" | "brainrouter" | ""
                );
                let model = if user_requested_specific {
                    requested_model
                } else {
                    self.default_local_model.clone()
                };
                RoutingDecision::Local { model }
            }
        }
    }
}

#[derive(Debug, Deserialize)]
enum ParsedDecision {
    Cloud,
    Local,
}

/// Parse Bonsai's raw output into a decision. Looks at the first non-whitespace
/// character: 'l' or 'L' → Local, anything else → Cloud (safe default).
fn parse_decision(raw: &str) -> ParsedDecision {
    let trimmed = raw.trim_start();
    if trimmed.starts_with('l') || trimmed.starts_with('L') {
        ParsedDecision::Local
    } else {
        ParsedDecision::Cloud
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
        // Anything not starting with l/L defaults to Cloud
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
}
