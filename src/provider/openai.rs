use super::{Provider, ProviderResponse, SseStream};
use crate::types::ChatCompletionRequest;
use anyhow::{anyhow, Result};
use bytes::Bytes;
use futures_util::StreamExt;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

/// OpenAI-compatible provider adapter.
/// Handles OpenAI, GitHub Copilot, Mistral, llama-swap, and other OpenAI-compatible APIs.
pub struct OpenAiProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    name: String,
}

impl OpenAiProvider {
    /// Create a new OpenAI-compatible provider.
    ///
    /// # Arguments
    /// * `name` - Human-readable name (e.g. "mistral", "llama-swap", "openai")
    /// * `base_url` - Base URL (e.g. "https://api.openai.com/v1" or "http://localhost:8080")
    /// * `api_key` - Optional API key for authentication
    pub fn new(name: String, base_url: String, api_key: Option<String>) -> Self {
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(10)
            .pool_idle_timeout(Duration::from_secs(90))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build reqwest client");

        // Trim trailing slash from base_url
        let base_url = base_url.trim_end_matches('/').to_string();

        Self {
            client,
            base_url,
            api_key,
            name,
        }
    }
}

impl Provider for OpenAiProvider {
    fn chat_completion(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ProviderResponse>> + Send + '_>> {
        Box::pin(async move {
            // Force streaming mode
            request.stream = Some(true);

            // Build the request URL
            let url = format!("{}/chat/completions", self.base_url);

            tracing::debug!(
                provider = %self.name,
                url = %url,
                model = %request.model,
                "Sending chat completion request"
            );

            // Build HTTP request
            let mut req_builder = self
                .client
                .post(&url)
                .header("Content-Type", "application/json");

            // Add authorization header if API key is present
            if let Some(ref api_key) = self.api_key {
                req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
            }

            // Send the request
            let response = req_builder
                .json(&request)
                .send()
                .await
                .map_err(|e| anyhow!("Failed to send request to {}: {}", self.name, e))?;

            // Check response status
            let status = response.status();
            if !status.is_success() {
                // Read error body for better error messages
                let error_body = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "<failed to read error body>".to_string());

                return Err(anyhow!(
                    "{} returned error status {}: {}",
                    self.name,
                    status,
                    error_body
                ));
            }

            tracing::debug!(
                provider = %self.name,
                "Received successful response, streaming chunks"
            );

            // Convert response byte stream to our SseStream type
            let stream: SseStream = Box::pin(
                response
                    .bytes_stream()
                    .map(|result| result.map_err(|e| anyhow!("Stream error: {}", e))),
            );

            Ok(ProviderResponse::Stream(stream))
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}
