use crate::{
    config::ModelConfig,
    health::HealthTracker,
    provider::{Provider, ProviderResponse},
    stream::TimeoutStream,
};
use anyhow::{anyhow, Result};
use std::{collections::HashMap, sync::Arc, time::Duration};

pub struct Router {
    providers: HashMap<String, Arc<dyn Provider>>,
    failover_chain: Vec<String>, // Stores unique provider-model IDs
    health_tracker: Arc<HealthTracker>,
}

impl Router {
    pub fn new(
        providers: HashMap<String, Arc<dyn Provider>>,
        ranked_models: Vec<String>,
        model_configs: &[ModelConfig],
        health_tracker: Arc<HealthTracker>,
    ) -> Self {
        let mut failover_chain = Vec::new();
        let model_map: HashMap<_, _> = model_configs.iter().map(|m| (&m.name, m)).collect();

        for model_name in &ranked_models {
            if let Some(model_config) = model_map.get(model_name) {
                for provider_name in &model_config.providers {
                    // Unique ID for health tracking, e.g., "openai-claude-sonnet-4-6"
                    let provider_id = format!("{}-{}", provider_name, model_name);
                    failover_chain.push(provider_id);
                }
            }
        }

        // Add bonsai as the last resort if it exists
        if providers.contains_key("embedded-bonsai") {
            failover_chain.push("embedded-bonsai-bonsai".to_string());
        }

        tracing::info!("Built failover chain: {:?}", failover_chain);

        Self {
            providers,
            failover_chain,
            health_tracker,
        }
    }

    pub async fn route(
        &self,
        mut request: crate::types::ChatCompletionRequest,
    ) -> Result<ProviderResponse> {
        for provider_id in &self.failover_chain {
            if !self.health_tracker.is_healthy(provider_id) {
                tracing::warn!(%provider_id, "Skipping unhealthy provider");
                continue;
            }

            let (provider_name, model_name) = match provider_id.rsplit_once('-') {
                Some(split) => split,
                None => {
                    tracing::error!(%provider_id, "Invalid provider_id format");
                    continue;
                }
            };
            
            if let Some(provider) = self.providers.get(provider_name) {
                tracing::info!(%provider_id, "Attempting provider");
                request.model = model_name.to_string();

                // Clone the request for this attempt
                let attempt_request = request.clone();
                let response_future = provider.chat_completion(attempt_request);
                
                match response_future.await {
                    Ok(ProviderResponse::Stream(stream)) => {
                        self.health_tracker.report_success(provider_id);
                        tracing::info!(%provider_id, "Provider succeeded");
                        // Wrap the stream with a timeout for each chunk to detect stalls
                        let timeout_stream = TimeoutStream::new(stream, Duration::from_secs(15));
                        return Ok(ProviderResponse::Stream(Box::pin(timeout_stream)));
                    }
                    Err(e) => {
                        tracing::error!(%provider_id, error = ?e, "Provider failed");
                        self.health_tracker.report_failure(provider_id);
                        continue; // Try next provider
                    }
                }
            }
        }

        Err(anyhow!("All providers in the failover chain failed"))
    }
}
