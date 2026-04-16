//! Request router.
//!
//! The router is the core of brainrouter. For every incoming request it:
//!   1. Asks Bonsai (via `Classifier`) whether the query should go to Cloud
//!      (Manifest) or Local (llama-swap with a specific model).
//!   2. Dispatches to the chosen backend.
//!   3. On cloud failure / stall, falls through to llama-swap with the
//!      configured fallback model.
//!   4. Wraps the resulting stream in a TimeoutStream so a stalled provider
//!      surfaces as an error instead of hanging the client.

use crate::{
    classifier::{Classifier, RoutingDecision},
    health::HealthTracker,
    provider::{openai::OpenAiProvider, Provider, ProviderResponse},
    stream::TimeoutStream,
    types::ChatCompletionRequest,
};
use anyhow::{anyhow, Result};
use std::{sync::Arc, time::Duration};
use tracing::{info, warn};

/// Stream chunk inactivity threshold. If no chunk is received for this long,
/// the stream is considered stalled and failed.
const STREAM_STALL_TIMEOUT: Duration = Duration::from_secs(15);

/// Provider health-tracker keys. Used for circuit breaking.
const MANIFEST_KEY: &str = "manifest";
const LLAMA_SWAP_KEY: &str = "llama-swap";

pub struct Router {
    classifier: Arc<Classifier>,
    manifest: Arc<OpenAiProvider>,
    llama_swap: Arc<OpenAiProvider>,
    /// Model name passed to llama-swap when we fall back from Manifest,
    /// or when the classifier picks Local but the user didn't request
    /// a specific model.
    fallback_model: String,
    health: Arc<HealthTracker>,
}

impl Router {
    pub fn new(
        classifier: Arc<Classifier>,
        manifest: Arc<OpenAiProvider>,
        llama_swap: Arc<OpenAiProvider>,
        fallback_model: String,
        health: Arc<HealthTracker>,
    ) -> Self {
        Self {
            classifier,
            manifest,
            llama_swap,
            fallback_model,
            health,
        }
    }

    pub async fn route(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Result<ProviderResponse> {
        let decision = self
            .classifier
            .classify_async(request.clone())
            .await;

        info!(?decision, "Bonsai routing decision");

        match decision {
            RoutingDecision::Cloud => self.route_cloud(request).await,
            RoutingDecision::Local { model } => {
                request.model = model;
                self.route_local(request).await
            }
        }
    }

    /// Cloud path: try Manifest first. On error/circuit-open, fall back to
    /// llama-swap with the configured fallback model.
    async fn route_cloud(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Result<ProviderResponse> {
        // Manifest expects "auto" — it does its own model selection.
        request.model = "auto".to_string();

        if self.health.is_healthy(MANIFEST_KEY) {
            info!(provider = MANIFEST_KEY, "Attempting Manifest");
            match self.manifest.chat_completion(request.clone()).await {
                Ok(ProviderResponse::Stream(stream)) => {
                    self.health.report_success(MANIFEST_KEY);
                    info!(provider = MANIFEST_KEY, "Manifest accepted request");
                    return Ok(wrap_with_timeout(stream));
                }
                Err(e) => {
                    warn!(provider = MANIFEST_KEY, error = %e, "Manifest failed, falling back");
                    self.health.report_failure(MANIFEST_KEY);
                    // fall through to llama-swap
                }
            }
        } else {
            warn!(provider = MANIFEST_KEY, "Manifest circuit open, skipping");
        }

        // Cloud fallback → llama-swap with fallback_model
        request.model = self.fallback_model.clone();
        self.try_llama_swap(request, "cloud-fallback").await
    }

    /// Local path: go straight to llama-swap. On failure, retry once with the
    /// fallback model if it differs from what was requested.
    async fn route_local(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ProviderResponse> {
        let requested = request.model.clone();
        match self.try_llama_swap(request.clone(), "local-primary").await {
            Ok(resp) => Ok(resp),
            Err(e) => {
                if requested == self.fallback_model {
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
                self.try_llama_swap(fallback_req, "local-fallback").await
            }
        }
    }

    async fn try_llama_swap(
        &self,
        request: ChatCompletionRequest,
        stage: &'static str,
    ) -> Result<ProviderResponse> {
        if !self.health.is_healthy(LLAMA_SWAP_KEY) {
            return Err(anyhow!(
                "llama-swap circuit open, no backend available (stage={})",
                stage
            ));
        }
        info!(provider = LLAMA_SWAP_KEY, stage, model = %request.model, "Attempting llama-swap");
        match self.llama_swap.chat_completion(request).await {
            Ok(ProviderResponse::Stream(stream)) => {
                self.health.report_success(LLAMA_SWAP_KEY);
                info!(provider = LLAMA_SWAP_KEY, stage, "llama-swap accepted request");
                Ok(wrap_with_timeout(stream))
            }
            Err(e) => {
                warn!(provider = LLAMA_SWAP_KEY, stage, error = %e, "llama-swap failed");
                self.health.report_failure(LLAMA_SWAP_KEY);
                Err(e)
            }
        }
    }
}

fn wrap_with_timeout(
    stream: crate::provider::SseStream,
) -> ProviderResponse {
    let timeout_stream = TimeoutStream::new(stream, STREAM_STALL_TIMEOUT);
    ProviderResponse::Stream(Box::pin(timeout_stream))
}
