use super::{Provider, ProviderError, ProviderResponse, SseStream};
use crate::types::ChatCompletionRequest;
use anyhow::anyhow;
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
    ) -> Pin<Box<dyn Future<Output = Result<ProviderResponse, ProviderError>> + Send + '_>> {
        Box::pin(async move {
            // Force streaming mode
            request.stream = Some(true);

            let url = format!("{}/chat/completions", self.base_url);

            tracing::debug!(
                provider = %self.name,
                url = %url,
                model = %request.model,
                "Sending chat completion request"
            );

            let mut req_builder = self
                .client
                .post(&url)
                .header("Content-Type", "application/json");

            if let Some(ref api_key) = self.api_key {
                req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
            }

            let response = req_builder
                .json(&request)
                .send()
                .await
                .map_err(|e| ProviderError {
                    message: format!("Failed to connect to {}: {}", self.name, e),
                    is_backend_fault: true,
                })?;

            let status = response.status();
            if !status.is_success() {
                let error_body = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "<failed to read error body>".to_string());
                // 4xx = bad request (wrong message format, invalid params) — backend is
                // healthy. 5xx = server error — treat as backend fault to trip circuit.
                let is_backend_fault = status.is_server_error();
                return Err(ProviderError {
                    message: format!(
                        "{} returned error status {}: {}",
                        self.name, status, error_body
                    ),
                    is_backend_fault,
                });
            }

            tracing::debug!(provider = %self.name, "Received successful response, streaming chunks");

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
