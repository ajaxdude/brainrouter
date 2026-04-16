//! Bonsai-powered query classifier.
//!
//! At request time, brainrouter feeds the user's last message to Bonsai 8B,
//! which decides whether the query is complex enough to warrant cloud routing
//! (via Manifest) or simple enough to handle locally on llama-swap.

use crate::types::ChatCompletionRequest;
use anyhow::{Context, Result};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    sampling::LlamaSampler,
};
use std::{
    num::NonZeroU32,
    path::Path,
    sync::Arc,
};
use tracing::{debug, warn};

/// Decision returned by the classifier for an incoming request.
#[derive(Debug, Clone)]
pub enum RoutingDecision {
    /// Forward to Manifest (cloud router). Request model is rewritten to "auto".
    Cloud,
    /// Forward directly to llama-swap with this specific model name.
    Local { model: String },
}

/// Embedded Bonsai classifier. Holds the model and backend in Arcs so they can
/// be shared across blocking-thread inference invocations.
pub struct Classifier {
    model: Arc<LlamaModel>,
    backend: Arc<LlamaBackend>,
    /// Default local model used when the request asks for "auto" but Bonsai
    /// chooses local routing without a specific suggestion.
    default_local_model: String,
}

/// Maximum tokens to generate during classification. We only need one word.
const CLASSIFY_MAX_TOKENS: usize = 5;
/// Truncate the user message to this many characters before classifying.
/// Keeps the prompt short so classification stays under ~200ms.
const USER_MSG_TRUNCATE: usize = 800;
/// Context window for the classifier — small, since prompts are short.
const CLASSIFIER_CTX_SIZE: u32 = 2048;

impl Classifier {
    /// Load the Bonsai model from disk. Blocking; call once at startup.
    pub fn new(model_path: &Path, default_local_model: String) -> Result<Self> {
        let backend =
            LlamaBackend::init().context("Failed to initialize llama.cpp backend")?;

        // Offload to GPU. AMD Strix Halo with Vulkan handles this fine.
        let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);

        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .with_context(|| {
                format!("Failed to load Bonsai model from {}", model_path.display())
            })?;

        Ok(Self {
            model: Arc::new(model),
            backend: Arc::new(backend),
            default_local_model,
        })
    }

    /// Classify a request asynchronously. Inference runs on a blocking thread
    /// so it doesn't stall the tokio runtime.
    ///
    /// On any error, defaults to `Cloud` — the safe default that preserves
    /// quality at the cost of an extra hop.
    pub async fn classify_async(&self, request: ChatCompletionRequest) -> RoutingDecision {
        let model = Arc::clone(&self.model);
        let backend = Arc::clone(&self.backend);
        let default_local = self.default_local_model.clone();
        let requested_model = request.model.clone();

        // Determine whether the user explicitly requested a model. If they
        // did NOT request "auto" or "brainrouter/auto", we still let Bonsai
        // classify — the requested model only matters when the decision is
        // Local and we need to pick a name.
        let user_requested_specific = !matches!(
            requested_model.as_str(),
            "auto" | "brainrouter/auto" | "brainrouter" | ""
        );

        let last_user_msg = extract_last_user_message(&request);
        if last_user_msg.is_empty() {
            debug!("No user message found; defaulting to Cloud");
            return RoutingDecision::Cloud;
        }

        let result = tokio::task::spawn_blocking(move || {
            classify_blocking(&model, &backend, &last_user_msg)
        })
        .await;

        let raw = match result {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                warn!("Classifier inference failed: {:#}; defaulting to Cloud", e);
                return RoutingDecision::Cloud;
            }
            Err(e) => {
                warn!("Classifier task panicked: {}; defaulting to Cloud", e);
                return RoutingDecision::Cloud;
            }
        };

        let decision = parse_decision(&raw);
        debug!(raw = %raw.trim(), ?decision, "Bonsai classification");

        match decision {
            ParsedDecision::Cloud => RoutingDecision::Cloud,
            ParsedDecision::Local => {
                // Pick the model name: prefer the user's explicit choice if any,
                // otherwise fall back to the configured default.
                let model = if user_requested_specific {
                    requested_model
                } else {
                    default_local
                };
                RoutingDecision::Local { model }
            }
        }
    }
}

#[derive(Debug)]
enum ParsedDecision {
    Cloud,
    Local,
}

/// Parse Bonsai's raw output into a decision. Looks at the first non-whitespace
/// character: 'l' or 'L' → Local, anything else → Cloud (safe default).
fn parse_decision(raw: &str) -> ParsedDecision {
    let trimmed = raw.trim_start();
    match trimmed.chars().next() {
        Some('l') | Some('L') => ParsedDecision::Local,
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
        // Truncate at a UTF-8 char boundary
        let mut end = USER_MSG_TRUNCATE;
        while !raw.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        raw[..end].to_string()
    } else {
        raw
    }
}

/// Synchronous classification inference. Runs on a blocking thread.
fn classify_blocking(
    model: &LlamaModel,
    backend: &LlamaBackend,
    user_message: &str,
) -> Result<String> {
    let prompt = format!(
        "<|im_start|>system\n\
You are a routing classifier. Reply with exactly one word: \"cloud\" for complex \
tasks (architecture, debugging large systems, multi-step reasoning, refactoring) \
or \"local\" for simple tasks (short answers, simple questions, single-line code, \
explanations). Output nothing else.<|im_end|>\n\
<|im_start|>user\n\
Classify this request: {}<|im_end|>\n\
<|im_start|>assistant\n",
        user_message
    );

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4);

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(CLASSIFIER_CTX_SIZE))
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads);

    let mut ctx = model
        .new_context(backend, ctx_params)
        .context("Failed to create classifier llama context")?;

    let tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .context("Failed to tokenize classifier prompt")?;

    let mut batch = LlamaBatch::new(tokens.len().max(8), 1);
    let last_index = tokens.len() as i32 - 1;
    for (i, token) in (0_i32..).zip(tokens.into_iter()) {
        batch
            .add(token, i, &[0], i == last_index)
            .context("Failed to add token to batch")?;
    }
    ctx.decode(&mut batch)
        .context("Failed to decode classifier prompt")?;

    let mut n_cur = batch.n_tokens();
    let mut sampler = LlamaSampler::greedy();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut out = String::new();

    for _ in 0..CLASSIFY_MAX_TOKENS {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if token == model.token_eos() {
            break;
        }

        let piece = model
            .token_to_piece(token, &mut decoder, false, None)
            .context("Failed to decode token")?;
        out.push_str(&piece);

        // Stop early if we already have a clear word
        if out.trim().len() >= 5 {
            break;
        }

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .context("Failed to add token to batch")?;
        n_cur += 1;
        ctx.decode(&mut batch)
            .context("Failed to decode classifier token")?;
    }

    Ok(out)
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
