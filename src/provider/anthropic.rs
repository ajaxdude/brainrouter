use crate::provider::{Provider, ProviderResponse, SseStream};
use crate::types::{ChatCompletionChunk, ChatCompletionRequest, ChatMessage, ChunkChoice, ChunkDelta};
use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
}

impl AnthropicProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
        }
    }

    fn translate_request(&self, req: &ChatCompletionRequest) -> Result<serde_json::Value> {
        // Extract system messages and concatenate them
        let mut system_messages = Vec::new();
        let mut non_system_messages = Vec::new();

        for msg in &req.messages {
            if msg.role == "system" {
                if let Some(content) = &msg.content {
                    let text = match content {
                        serde_json::Value::String(s) => s.clone(),
                        serde_json::Value::Array(_) => content.to_string(),
                        _ => content.to_string(),
                    };
                    system_messages.push(text);
                }
            } else {
                non_system_messages.push(msg.clone());
            }
        }

        let system = if system_messages.is_empty() {
            None
        } else {
            Some(system_messages.join("\n"))
        };

        // Convert messages to Anthropic format
        let messages: Vec<serde_json::Value> = non_system_messages
            .iter()
            .map(|msg| {
                let mut message = json!({
                    "role": msg.role,
                });

                if let Some(content) = &msg.content {
                    message["content"] = content.clone();
                }

                message
            })
            .collect();

        // Build the Anthropic request
        let mut body = json!({
            "model": req.model,
            "messages": messages,
            "stream": true,
            "max_tokens": req.max_tokens.unwrap_or(4096),
        });

        if let Some(system_text) = system {
            body["system"] = json!(system_text);
        }

        if let Some(temperature) = req.temperature {
            body["temperature"] = json!(temperature);
        }

        if let Some(top_p) = req.top_p {
            body["top_p"] = json!(top_p);
        }

        if let Some(stop) = &req.stop {
            body["stop_sequences"] = json!(stop);
        }

        Ok(body)
    }

    fn translate_sse_stream(
        &self,
        response: reqwest::Response,
        model: String,
    ) -> SseStream {
        let stream = response.bytes_stream();

        let translated = stream
            .map(move |chunk_result| {
                chunk_result.map_err(|e| anyhow!("Stream error: {}", e))
            })
            .scan(
                (String::new(), String::new(), model.clone()),
                |(buffer, current_event, model), chunk_result| {
                    let chunk = match chunk_result {
                        Ok(c) => c,
                        Err(e) => return std::future::ready(Some(vec![Err(e)])),
                    };

                    buffer.push_str(&String::from_utf8_lossy(&chunk));

                    let mut output = Vec::new();

                    // Process complete lines (ending with \n)
                    while let Some(newline_pos) = buffer.find('\n') {
                        let line = buffer[..newline_pos].to_string();
                        *buffer = buffer[newline_pos + 1..].to_string();

                        let trimmed = line.trim();

                        if trimmed.is_empty() {
                            // Empty line signals end of SSE block
                            continue;
                        }

                        if let Some(event_name) = trimmed.strip_prefix("event:") {
                            *current_event = event_name.trim().to_string();
                        } else if let Some(data_content) = trimmed.strip_prefix("data:") {
                            let data_str = data_content.trim();

                            if let Ok(data) = serde_json::from_str::<serde_json::Value>(data_str) {
                                if let Some(sse_chunk) = Self::translate_event(
                                    current_event.as_str(),
                                    &data,
                                    model.clone(),
                                ) {
                                    output.push(Ok(sse_chunk));
                                }
                            }

                            current_event.clear();
                        }
                    }

                    std::future::ready(Some(output))
                },
            )
            .flat_map(futures_util::stream::iter);

        Box::pin(translated)
    }

    fn translate_event(
        event_type: &str,
        data: &serde_json::Value,
        model: String,
    ) -> Option<Bytes> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let request_id = format!(
            "chatcmpl-{}",
            REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
        );

        match event_type {
            "content_block_delta" => {
                if let Some(delta) = data.get("delta") {
                    if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                        let chunk = ChatCompletionChunk {
                            id: request_id,
                            object: "chat.completion.chunk".to_string(),
                            created: timestamp,
                            model: model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: Some(text.to_string()),
                                },
                                finish_reason: None,
                            }],
                        };

                        let json = serde_json::to_string(&chunk).ok()?;
                        return Some(Bytes::from(format!("data: {}\n\n", json)));
                    }
                }
            }
            "message_delta" => {
                if let Some(delta) = data.get("delta") {
                    if let Some(stop_reason) = delta.get("stop_reason").and_then(|s| s.as_str()) {
                        let finish_reason = match stop_reason {
                            "end_turn" => "stop",
                            "max_tokens" => "length",
                            _ => stop_reason,
                        };

                        let chunk = ChatCompletionChunk {
                            id: request_id,
                            object: "chat.completion.chunk".to_string(),
                            created: timestamp,
                            model: model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: None,
                                },
                                finish_reason: Some(finish_reason.to_string()),
                            }],
                        };

                        let json = serde_json::to_string(&chunk).ok()?;
                        return Some(Bytes::from(format!("data: {}\n\n", json)));
                    }
                }
            }
            "message_stop" => {
                return Some(Bytes::from("data: [DONE]\n\n"));
            }
            _ => {}
        }

        None
    }
}

impl Provider for AnthropicProvider {
    fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ProviderResponse>> + Send + '_>> {
        Box::pin(async move {
            let model = request.model.clone();
            let body = self.translate_request(&request)
                .context("Failed to translate request to Anthropic format")?;

            tracing::debug!("Sending request to Anthropic: {:?}", body);

            let response = self
                .client
                .post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&body)
                .send()
                .await
                .context("Failed to send request to Anthropic")?;

            let status = response.status();
            if !status.is_success() {
                let error_body = response.text().await.unwrap_or_default();
                return Err(anyhow!(
                    "Anthropic API error ({}): {}",
                    status,
                    error_body
                ));
            }

            let stream = self.translate_sse_stream(response, model);
            Ok(ProviderResponse::Stream(stream))
        })
    }

    fn name(&self) -> &str {
        "anthropic"
    }
}
