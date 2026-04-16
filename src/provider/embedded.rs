use super::{Provider, ProviderResponse};
use crate::types::ChatCompletionRequest;
use anyhow::Result;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

/// Embedded Bonsai provider using llama-cpp-2 (stub for Phase 3)
pub struct EmbeddedProvider {
    model_path: PathBuf,
    // llama-cpp-2 model and context will be added in Phase 3
}

impl EmbeddedProvider {
    pub fn new(model_path: PathBuf) -> Self {
        Self { model_path }
    }
}

impl Provider for EmbeddedProvider {
    fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ProviderResponse>> + Send + '_>> {
        Box::pin(async move {
            anyhow::bail!("Embedded Bonsai provider not yet initialized (Phase 3)")
        })
    }

    fn name(&self) -> &str {
        "embedded-bonsai"
    }
}
