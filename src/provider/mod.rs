//! Provider adapter trait and common types.
//!
//! brainrouter uses exactly two OpenAI-compatible backends:
//! - Manifest (cloud router at localhost:3001)
//! - llama-swap (local model runner at localhost:8080)
//!
//! Both are served by `OpenAiProvider`. The embedded Bonsai model is NOT a
//! provider in the routing path — it's used by the `classifier` module to
//! make the routing decision, and is not surfaced as an LLM backend itself.

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

/// Trait implemented by HTTP-based LLM provider adapters.
pub trait Provider: Send + Sync {
    fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ProviderResponse>> + Send + '_>>;

    fn name(&self) -> &str;
}

pub mod openai;
