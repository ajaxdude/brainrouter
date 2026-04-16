use crate::types::ChatCompletionRequest;
use anyhow::Result;
use bytes::Bytes;
use futures_util::Stream;
use std::future::Future;
use std::pin::Pin;

/// A stream of SSE-formatted bytes (OpenAI chat completion streaming format)
pub type SseStream = Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>;

/// Response from a provider — either streaming or complete
pub enum ProviderResponse {
    /// SSE byte stream in OpenAI chat completion format
    Stream(SseStream),
}

/// Trait that all LLM provider adapters implement
pub trait Provider: Send + Sync {
    /// Send a chat completion request and return a streaming response.
    /// The stream yields SSE-formatted bytes in OpenAI chat completion format.
    /// Adapters that speak non-OpenAI protocols must translate to OpenAI format.
    fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ProviderResponse>> + Send + '_>>;

    /// Human-readable name for logging
    fn name(&self) -> &str;
}

pub mod anthropic;
// TODO: embedded provider has thread safety issues and needs to be fixed
pub mod embedded;
pub mod google;
pub mod openai;
