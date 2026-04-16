use crate::config::ModelConfig;
use crate::provider::{Provider, ProviderResponse};
use crate::types::ChatCompletionRequest;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;

pub struct Router {
    failover_chain: Vec<Arc<dyn Provider>>,
}

impl Router {
    pub fn new(
        providers: HashMap<String, Arc<dyn Provider>>,
        ranked_models: Vec<String>,
        model_configs: &[ModelConfig],
    ) -> Self {
        let mut failover_chain = Vec::new();
        
        // Build the failover chain from ranked models
        for model_name in ranked_models {
            // Find the model config
            if let Some(model_config) = model_configs.iter().find(|m| m.name == model_name) {
                // Add each provider for this model to the chain
                for provider_name in &model_config.providers {
                    if let Some(provider) = providers.get(provider_name) {
                        failover_chain.push(Arc::clone(provider));
                    }
                }
            }
        }
        
        Self { failover_chain }
    }

    pub async fn route(&self, request: &ChatCompletionRequest) -> Result<ProviderResponse> {
        // For now, just try the first provider in the failover chain
        let provider = self
            .failover_chain
            .first()
            .ok_or_else(|| anyhow!("No providers in failover chain"))?;
        
        provider.chat_completion(request.clone()).await
    }
}
