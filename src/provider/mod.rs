use crate::types::ChatCompletionRequest;
use anyhow::Result;
use bytes::Bytes;
use futures_util::Stream;
use std::future::Future;
use std::pin::Pin;

/// A stream of SSE-formatted bytes in OpenAI chat completion format.
pub type SseStream = Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>;

/// Response from a provider. Currently always streaming.
pub enum ProviderResponse {
    Stream(SseStream),
}

/// Error returned by a provider.
#[derive(Debug)]
pub struct ProviderError {
    /// Human-readable error message.
    pub message: String,
    /// True when the error is a backend infrastructure fault (connection refused,
    /// timeout, 5xx server crash). False for 4xx/5xx responses caused by a bad
    /// *request* (wrong message order, invalid JSON, oversized prompt) — these
    /// mean the backend is healthy and should not trip the circuit breaker.
    pub is_backend_fault: bool,
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ProviderError {}

/// Trait implemented by HTTP-based LLM provider adapters.
pub trait Provider: Send + Sync {
    fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ProviderResponse, ProviderError>> + Send + '_>>;

    fn name(&self) -> &str;
}

pub mod openai;