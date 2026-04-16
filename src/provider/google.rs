use crate::provider::{Provider, ProviderResponse, SseStream};
use crate::types::{ChatCompletionRequest, ChatMessage};
use anyhow::{anyhow, Context, Result};
use bytes::{Bytes, BytesMut};
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::pin::Pin;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct GoogleProvider {
    client: reqwest::Client,
    api_key: String,
}

impl GoogleProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
        }
    }

    /// Convert OpenAI messages to Gemini contents format
    fn convert_messages(
        messages: &[ChatMessage],
    ) -> Result<(Vec<GeminiContent>, Option<GeminiSystemInstruction>)> {
        let mut contents = Vec::new();
        let mut system_instruction = None;

        for msg in messages {
            let role = msg.role.as_str();
            
            if role == "system" {
                // Extract system message
                let text = match &msg.content {
                    Some(serde_json::Value::String(s)) => s.clone(),
                    Some(serde_json::Value::Array(parts)) => {
                        // Handle multi-part content
                        parts.iter()
                            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                    _ => continue,
                };
                
                system_instruction = Some(GeminiSystemInstruction {
                    parts: vec![GeminiPart { text }],
                });
            } else {
                // Convert role: "assistant" -> "model", "user" -> "user"
                let gemini_role = if role == "assistant" {
                    "model"
                } else {
                    role
                };

                let parts = match &msg.content {
                    Some(serde_json::Value::String(s)) => {
                        vec![GeminiPart { text: s.clone() }]
                    }
                    Some(serde_json::Value::Array(arr)) => {
                        // Multi-part content (e.g., with images)
                        arr.iter()
                            .filter_map(|v| {
                                if let Some(text) = v.get("text").and_then(|t| t.as_str()) {
                                    Some(GeminiPart { text: text.to_string() })
                                } else {
                                    None
                                }
                            })
                            .collect()
                    }
                    _ => continue,
                };

                contents.push(GeminiContent {
                    role: gemini_role.to_string(),
                    parts,
                });
            }
        }

        Ok((contents, system_instruction))
    }

    /// Build Gemini request from OpenAI request
    fn build_gemini_request(&self, request: &ChatCompletionRequest) -> Result<GeminiRequest> {
        let (contents, system_instruction) = Self::convert_messages(&request.messages)?;

        let generation_config = GeminiGenerationConfig {
            temperature: request.temperature,
            max_output_tokens: request.max_tokens,
            top_p: request.top_p,
        };

        Ok(GeminiRequest {
            contents,
            system_instruction,
            generation_config,
        })
    }

    /// Translate Gemini streaming response to OpenAI SSE format
    fn translate_stream(
        &self,
        gemini_stream: impl Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
        model: String,
    ) -> SseStream {
        use futures_util::stream::{Stream, StreamExt};
        use std::task::{Context, Poll};
        
        struct TranslateStream<S> {
            inner: Pin<Box<S>>,
            buffer: BytesMut,
            chunk_id: String,
            created: u64,
            first_chunk: bool,
            model: String,
            done: bool,
        }

        impl<S> Stream for TranslateStream<S>
        where
            S: Stream<Item = reqwest::Result<Bytes>> + Send,
        {
            type Item = Result<Bytes>;

            fn poll_next(
                mut self: Pin<&mut Self>,
                cx: &mut Context<'_>,
            ) -> Poll<Option<Self::Item>> {
                loop {
                    // First, try to process buffered data
                    if let Some(newline_pos) = self.buffer.iter().position(|&b| b == b'\n') {
                        let line = self.buffer.split_to(newline_pos + 1);
                        let line_str = match std::str::from_utf8(&line) {
                            Ok(s) => s.trim(),
                            Err(e) => return Poll::Ready(Some(Err(anyhow::anyhow!("Invalid UTF-8: {}", e)))),
                        };

                        if line_str.is_empty() {
                            continue;
                        }

                        // Parse SSE data line
                        if let Some(data) = line_str.strip_prefix("data: ") {
                            if data == "[DONE]" {
                                self.done = true;
                                let sse_line = Bytes::from("data: [DONE]\n\n");
                                return Poll::Ready(Some(Ok(sse_line)));
                            }

                            // Parse Gemini response chunk
                            match serde_json::from_str::<GeminiStreamResponse>(data) {
                                Ok(gemini_chunk) => {
                                    if let Some(candidate) = gemini_chunk.candidates.first() {
                                        let text = candidate.content.parts.first()
                                            .map(|p| p.text.as_str())
                                            .unwrap_or("");

                                        let finish_reason = candidate.finish_reason.as_ref().and_then(|reason| {
                                            match reason.as_str() {
                                                "STOP" => Some("stop"),
                                                "MAX_TOKENS" => Some("length"),
                                                "SAFETY" => Some("content_filter"),
                                                _ => None,
                                            }
                                        }).map(String::from);

                                        // Build OpenAI chunk
                                        let openai_chunk = json!({
                                            "id": self.chunk_id,
                                            "object": "chat.completion.chunk",
                                            "created": self.created,
                                            "model": self.model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": if self.first_chunk {
                                                    json!({
                                                        "role": "assistant",
                                                        "content": text
                                                    })
                                                } else {
                                                    json!({
                                                        "content": text
                                                    })
                                                },
                                                "finish_reason": finish_reason,
                                            }]
                                        });

                                        self.first_chunk = false;

                                        // Emit as SSE
                                        match serde_json::to_string(&openai_chunk) {
                                            Ok(json_str) => {
                                                let sse_line = format!("data: {}\n\n", json_str);
                                                return Poll::Ready(Some(Ok(Bytes::from(sse_line))));
                                            }
                                            Err(e) => {
                                                return Poll::Ready(Some(Err(anyhow::anyhow!("JSON serialization error: {}", e))));
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!("Failed to parse Gemini chunk: {}", e);
                                    continue;
                                }
                            }
                        }

                        continue;
                    }

                    // No complete line in buffer, read more from upstream
                    if self.done {
                        return Poll::Ready(None);
                    }

                    match self.inner.as_mut().poll_next(cx) {
                        Poll::Ready(Some(Ok(bytes))) => {
                            self.buffer.extend_from_slice(&bytes);
                            continue;
                        }
                        Poll::Ready(Some(Err(e))) => {
                            return Poll::Ready(Some(Err(anyhow::anyhow!("Stream error: {}", e))));
                        }
                        Poll::Ready(None) => {
                            // Stream ended, emit [DONE] if not already sent
                            if !self.done {
                                self.done = true;
                                return Poll::Ready(Some(Ok(Bytes::from("data: [DONE]\n\n"))));
                            }
                            return Poll::Ready(None);
                        }
                        Poll::Pending => {
                            return Poll::Pending;
                        }
                    }
                }
            }
        }

        let chunk_id = format!(
            "chatcmpl-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let stream = TranslateStream {
            inner: Box::pin(gemini_stream),
            buffer: BytesMut::new(),
            chunk_id,
            created,
            first_chunk: true,
            model,
            done: false,
        };

        Box::pin(stream)
    }
}

impl Provider for GoogleProvider {
    fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ProviderResponse>> + Send + '_>> {
        Box::pin(async move {
            let model = request.model.clone();
            let gemini_request = self.build_gemini_request(&request)?;

            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?key={}&alt=sse",
                model, self.api_key
            );

            tracing::debug!("Sending request to Gemini: {}", url);

            let response = self
                .client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&gemini_request)
                .send()
                .await
                .context("Failed to send request to Gemini")?;

            if !response.status().is_success() {
                let status = response.status();
                let error_body = response.text().await.unwrap_or_default();
                return Err(anyhow!("Gemini API error {}: {}", status, error_body));
            }

            let stream = response.bytes_stream();
            let translated_stream = self.translate_stream(stream, model);

            Ok(ProviderResponse::Stream(translated_stream))
        })
    }

    fn name(&self) -> &str {
        "google"
    }
}

// === Gemini API Types ===

#[derive(Debug, Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiSystemInstruction>,
    generation_config: GeminiGenerationConfig,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiPart {
    text: String,
}

#[derive(Debug, Serialize)]
struct GeminiSystemInstruction {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct GeminiStreamResponse {
    candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}
