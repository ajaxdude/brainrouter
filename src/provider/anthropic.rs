//! v2 L6: NEW Anthropic outbound adapter (spec "Outbound").
//!
//! The router always speaks the canonical OpenAI request. This adapter
//! translates that request to the Anthropic Messages wire and translates the
//! Anthropic SSE stream back to OpenAI chat-completion chunks, so every
//! downstream consumer (stream sniffing, in-flight model stamping, TPS
//! capture) sees one protocol.
//!
//! Auth is `x-api-key` + `anthropic-version` — NOT Bearer (the inbound
//! `src/anthropic.rs` is the reverse-direction reference).

use super::{ProviderError, ProviderResponse, SseStream};
use crate::types::{ChatCompletionRequest, ChatMessage};
use anyhow::anyhow;
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

const RESPONSE_HEADER_TIMEOUT: Duration = Duration::from_secs(600);
/// Pinned Anthropic API version header value.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    name: String,
}

impl AnthropicProvider {
    pub fn new(name: String, base_url: String, api_key: Option<String>) -> Self {
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(10)
            .pool_idle_timeout(Duration::from_secs(90))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build reqwest client");
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            name,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// One non-streaming cheap probe body (provider Test).
    pub fn probe_request(model: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: model.to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some(Value::String("ping".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: Some(true),
            temperature: None,
            max_tokens: Some(16),
            top_p: None,
            stop: None,
            extra: Value::Null,
        }
    }

    pub async fn chat_completion(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        request.stream = Some(true);
        let body = to_anthropic_body(&request);
        let url = format!("{}/messages", self.base_url);
        tracing::debug!(provider = %self.name, url = %url, model = %request.model, "Sending Anthropic Messages request");

        let mut builder = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .header("anthropic-version", ANTHROPIC_VERSION);
        if let Some(key) = &self.api_key {
            builder = builder.header("x-api-key", key);
        }

        let response = tokio::time::timeout(RESPONSE_HEADER_TIMEOUT, builder.json(&body).send())
            .await
            .map_err(|_| ProviderError {
                message: format!(
                    "{} timed out waiting for response headers after {} seconds",
                    self.name,
                    RESPONSE_HEADER_TIMEOUT.as_secs()
                ),
                is_backend_fault: true,
            })?
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
            let is_backend_fault = status.is_server_error()
                || status == reqwest::StatusCode::TOO_MANY_REQUESTS;
            return Err(ProviderError {
                message: format!(
                    "{} returned error status {}: {}",
                    self.name, status, error_body
                ),
                is_backend_fault,
            });
        }

        let model = request.model.clone();
        let upstream: SseStream = Box::pin(
            response
                .bytes_stream()
                .map(|result| result.map_err(|e| anyhow!("Stream error: {}", e))),
        );
        let translated: SseStream = Box::pin(AnthropicToOpenAiStream::new(upstream, model));
        Ok(ProviderResponse::Stream(translated))
    }
}

/// Translate the canonical OpenAI request to the Anthropic Messages body.
pub fn to_anthropic_body(request: &ChatCompletionRequest) -> Value {
    let mut system_parts: Vec<String> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();
    for message in &request.messages {
        let role = message.role.as_str();
        if role == "system" || role == "developer" {
            if let Some(text) = content_text(message.content.as_ref()) {
                system_parts.push(text);
            }
            continue;
        }
        let role = if role == "assistant" { "assistant" } else { "user" };
        messages.push(json!({
            "role": role,
            "content": anthropic_content(message),
        }));
    }
    if messages.is_empty() {
        messages.push(json!({"role": "user", "content": ""}));
    }
    let mut body = json!({
        "model": request.model,
        "max_tokens": request.max_tokens.unwrap_or(4096),
        "messages": messages,
        "stream": true,
    });
    if !system_parts.is_empty() {
        body["system"] = Value::String(system_parts.join("\n\n"));
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(top_p) = request.top_p {
        body["top_p"] = json!(top_p);
    }
    if let Some(stop) = &request.stop {
        if !stop.is_empty() {
            body["stop_sequences"] = json!(stop);
        }
    }
    if let Some(tools) = request.extra.get("tools").and_then(Value::as_array) {
        let converted: Vec<Value> = tools
            .iter()
            .filter_map(|tool| {
                let function = tool.get("function")?;
                Some(json!({
                    "name": function.get("name")?.as_str()?,
                    "description": function.get("description").and_then(Value::as_str).unwrap_or(""),
                    "input_schema": function.get("parameters").cloned().unwrap_or_else(|| json!({"type": "object"})),
                }))
            })
            .collect();
        if !converted.is_empty() {
            body["tools"] = json!(converted);
        }
    }
    body
}

fn content_text(content: Option<&Value>) -> Option<String> {
    match content? {
        Value::String(text) => Some(text.clone()),
        Value::Array(blocks) => Some(
            blocks
                .iter()
                .filter_map(|block| block.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        other => Some(other.to_string()),
    }
}

fn anthropic_content(message: &ChatMessage) -> Value {
    match (&message.content, &message.tool_calls) {
        (Some(content), _) => content.clone(),
        (None, Some(calls)) => Value::Array(
            calls
                .iter()
                .filter_map(|call| {
                    let function = call.get("function")?;
                    Some(json!({
                        "type": "tool_use",
                        "id": call.get("id").and_then(Value::as_str).unwrap_or("tool"),
                        "name": function.get("name").and_then(Value::as_str).unwrap_or("tool"),
                        "input": function
                            .get("arguments")
                            .and_then(Value::as_str)
                            .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                            .unwrap_or_else(|| json!({})),
                    }))
                })
                .collect(),
        ),
        (None, None) => Value::String(String::new()),
    }
}

/// Translate Anthropic SSE frames to OpenAI chat-completion chunks.
pub struct AnthropicToOpenAiStream {
    inner: SseStream,
    model: String,
    buffer: Vec<u8>,
    pending: std::collections::VecDeque<Bytes>,
    started: bool,
    done: bool,
}

impl AnthropicToOpenAiStream {
    pub fn new(inner: SseStream, model: String) -> Self {
        Self {
            inner,
            model,
            buffer: Vec::new(),
            pending: std::collections::VecDeque::new(),
            started: false,
            done: false,
        }
    }

    fn openai_chunk(&self, delta: Value, finish: Option<&str>) -> Bytes {
        let mut choice = json!({"index": 0, "delta": delta});
        if let Some(finish) = finish {
            choice["finish_reason"] = json!(finish);
        } else {
            choice["finish_reason"] = Value::Null;
        }
        Bytes::from(format!(
            "data: {}\n\n",
            json!({
                "id": "chatcmpl-anthropic",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": self.model,
                "choices": [choice],
            })
        ))
    }

    fn start_frame(&mut self) {
        if !self.started {
            self.started = true;
            self.pending
                .push_back(self.openai_chunk(json!({"role": "assistant", "content": ""}), None));
        }
    }

    fn translate_frame(&mut self, event: &str, data: &str) {
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            return;
        };
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or(event);
        match kind {
            "message_start" => self.start_frame(),
            "content_block_start" => {
                self.start_frame();
                if let Some(model) = value
                    .get("message")
                    .and_then(|m| m.get("model"))
                    .and_then(Value::as_str)
                {
                    self.model = model.to_string();
                }
            }
            "content_block_delta" => {
                self.start_frame();
                let Some(delta) = value.get("delta") else {
                    return;
                };
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        if let Some(text) = delta.get("text").and_then(Value::as_str) {
                            self.pending
                                .push_back(self.openai_chunk(json!({"content": text}), None));
                        }
                    }
                    Some("thinking_delta") => {
                        if let Some(text) = delta.get("thinking").and_then(Value::as_str) {
                            self.pending.push_back(
                                self.openai_chunk(json!({"reasoning_content": text}), None),
                            );
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(partial) = delta.get("partial_json").and_then(Value::as_str) {
                            self.pending.push_back(self.openai_chunk(
                                json!({"tool_calls": [{
                                    "index": 0,
                                    "function": {"arguments": partial}
                                }]}),
                                None,
                            ));
                        }
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                let finish = match value
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(Value::as_str)
                {
                    Some("max_tokens") => Some("length"),
                    Some("tool_use") => Some("tool_calls"),
                    Some(_) => Some("stop"),
                    None => None,
                };
                if let Some(finish) = finish {
                    self.pending.push_back(self.openai_chunk(json!({}), Some(finish)));
                }
            }
            "message_stop" => {
                self.pending.push_back(Bytes::from_static(b"data: [DONE]\n\n"));
                self.done = true;
            }
            "error" => {
                let message = value
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("anthropic stream error");
                self.pending.push_back(Bytes::from(format!(
                    "data: {}\n\n",
                    json!({"error": {"message": message, "type": "provider_error"}})
                )));
            }
            _ => {}
        }
    }

    fn drain_buffer(&mut self) {
        while let Some(index) = find_frame_end(&self.buffer) {
            let frame: Vec<u8> = self.buffer.drain(..index).collect();
            // Consume the trailing blank line.
            let trimmed = String::from_utf8_lossy(&frame);
            let mut event = "";
            let mut data = String::new();
            for line in trimmed.lines() {
                if let Some(rest) = line.strip_prefix("event:") {
                    event = rest.trim();
                } else if let Some(rest) = line.strip_prefix("data:") {
                    if !data.is_empty() {
                        data.push('\n');
                    }
                    data.push_str(rest.trim());
                }
            }
            if !data.is_empty() {
                self.translate_frame(event, &data);
            }
        }
    }
}

/// Index just past the next `\n\n` (or `\r\n\r\n`) frame terminator.
fn find_frame_end(buffer: &[u8]) -> Option<usize> {
    buffer
        .windows(2)
        .position(|w| w == b"\n\n")
        .map(|p| p + 2)
        .or_else(|| {
            buffer
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
                .map(|p| p + 4)
        })
}

impl futures_util::Stream for AnthropicToOpenAiStream {
    type Item = anyhow::Result<Bytes>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(bytes) = this.pending.pop_front() {
                return Poll::Ready(Some(Ok(bytes)));
            }
            if this.done {
                return Poll::Ready(None);
            }
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    this.buffer.extend_from_slice(&bytes);
                    this.drain_buffer();
                    if this.pending.is_empty() && !this.done {
                        continue;
                    }
                }
                Poll::Ready(Some(Err(error))) => return Poll::Ready(Some(Err(error))),
                Poll::Ready(None) => {
                    this.drain_buffer();
                    if this.pending.is_empty() {
                        return Poll::Ready(None);
                    }
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}
