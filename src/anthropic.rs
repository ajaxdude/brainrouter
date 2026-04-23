//! Anthropic Messages API adapter.
//!
//! Provides two transformations:
//!   1. `anthropic_to_openai` — converts an incoming `POST /v1/messages` request
//!      (Anthropic format) into a `ChatCompletionRequest` (OpenAI format) that the
//!      Router can dispatch.
//!
//!   2. `openai_sse_to_anthropic_sse` — converts an OpenAI SSE stream
//!      (`data: {choices:[{delta:{content:"..."}}]}`)
//!      into the Anthropic SSE event sequence that the client expects:
//!      message_start → content_block_start → content_block_delta* → content_block_stop
//!      → message_delta (stop_reason) → message_stop
//!
//! Neither function touches the Router; they are pure protocol adapters.
//!
//! Supported v1 scope:
//!   - Text messages (user/assistant/system)
//!   - Tools (basic: input_schema → parameters mapping)
//!   - Streaming
//!   - stop_sequences, temperature, top_p, top_k, max_tokens
//!
//! Not supported (deferred):
//!   - Prompt caching, extended thinking blocks
//!   - Vision/image/PDF content
//!   - Batch API

use anyhow::Result;
use bytes::Bytes;
use futures_util::Stream;
use pin_project::pin_project;
use serde::Deserialize;
use serde_json::Value;
use std::pin::Pin;
use std::task::{Context, Poll};
use uuid::Uuid;
use tracing::warn;

use crate::types::{ChatCompletionRequest, ChatMessage};

// ─── Anthropic request types ─────────────────────────────────────────────────

/// Top-level Anthropic `/v1/messages` request.
#[derive(Debug, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<Value>,
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    #[serde(default)]
    pub tools: Vec<AnthropicTool>,
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: Value,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
}

// ─── Request translation ──────────────────────────────────────────────────────

/// Convert an Anthropic `/v1/messages` request to an OpenAI chat completion request.
pub fn anthropic_to_openai(req: AnthropicMessagesRequest) -> ChatCompletionRequest {
    let mut messages: Vec<ChatMessage> = Vec::new();

    // Anthropic top-level `system` field → prepend as system message
    if let Some(system) = &req.system {
        let system_text = match system {
            Value::String(s) => s.clone(),
            Value::Array(blocks) => {
                blocks.iter().filter_map(|block| {
                    if block.get("type").and_then(Value::as_str) == Some("text") {
                        block.get("text").and_then(Value::as_str).map(|s| s.to_string())
                    } else {
                        warn!(block_type = ?block.get("type"), "Dropping non-text block from system field");
                        None
                    }
                }).collect::<Vec<_>>().join("\n")
            }
            other => serde_json::to_string(other).unwrap_or_default(),
        };
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: Some(Value::String(system_text)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // Convert Anthropic messages
    for msg in req.messages {
        messages.push(anthropic_message_to_openai(msg));
    }

    // Convert Anthropic tools to OpenAI tools
    let tools: Option<Value> = if req.tools.is_empty() {
        None
    } else {
        let oai_tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema
                    }
                })
            })
            .collect();
        Some(Value::Array(oai_tools))
    };

    let mut extra = serde_json::Map::new();

    // tools + tool_choice
    if let Some(tools) = tools {
        extra.insert("tools".to_string(), tools);
    }
    if let Some(tc) = req.tool_choice {
        extra.insert("tool_choice".to_string(), map_tool_choice(tc));
    }

    ChatCompletionRequest {
        model: req.model,
        messages,
        stream: Some(true), // always stream at the backend; we adapt the response
        temperature: req.temperature,
        max_tokens: req.max_tokens,
        top_p: req.top_p,
        stop: if req.stop_sequences.is_empty() {
            None
        } else {
            Some(req.stop_sequences)
        },
        extra: Value::Object(extra),
    }
}

fn anthropic_message_to_openai(msg: AnthropicMessage) -> ChatMessage {
    let content = match &msg.content {
        // Simple string content (common for user/assistant turns)
        Value::String(s) => Some(Value::String(s.clone())),

        // Array of content blocks
        Value::Array(blocks) => {
            // Flatten text blocks into a single string; ignore non-text (image, etc.)
            let combined: String = blocks
                .iter()
                .filter_map(|block| {
                    if block.get("type").and_then(Value::as_str) == Some("text") {
                        block.get("text").and_then(Value::as_str).map(|s| s.to_string())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            if combined.is_empty() {
                // Fall through with the original array — downstream may handle it
                Some(msg.content.clone())
            } else {
                Some(Value::String(combined))
            }
        }
        other => Some(other.clone()),
    };

    ChatMessage {
        role: msg.role,
        content,
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

/// Map Anthropic `tool_choice` to OpenAI equivalent.
fn map_tool_choice(tc: Value) -> Value {
    match tc.get("type").and_then(Value::as_str) {
        Some("auto") => Value::String("auto".to_string()),
        Some("any") => Value::String("required".to_string()),
        Some("tool") => {
            let name = tc.get("name").and_then(Value::as_str).unwrap_or("");
            serde_json::json!({ "type": "function", "function": { "name": name } })
        }
        _ => Value::String("auto".to_string()),
    }
}

// ─── Response translation ─────────────────────────────────────────────────────

/// State machine for the SSE adapter.
#[derive(Debug, PartialEq, Clone, Copy)]
enum AdapterState {
    Initial,
    MessageStarted,
    ContentBlockStarted,
    Streaming,
    Done,
}

/// Adapts an OpenAI SSE stream to Anthropic SSE events.
///
/// Emits in order:
///   event: message_start         (once, first chunk)
///   event: content_block_start   (once, block index 0)
///   event: content_block_delta*  (per text chunk)
///   event: content_block_stop    (once, on finish_reason)
///   event: message_delta         (once, carries stop_reason + usage)
///   event: message_stop          (once, final)
#[pin_project]
pub struct AnthropicSseAdapter {
    #[pin]
    inner: Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>,
    state: AdapterState,
    message_id: String,
    model: String,
    input_tokens: u32,
    output_tokens: u32,
    /// Buffered bytes from partial SSE lines
    line_buf: String,
    /// Outgoing events queued to be flushed before pulling from inner
    pending: std::collections::VecDeque<Bytes>,
}

impl AnthropicSseAdapter {
    pub fn new(
        stream: Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>,
        model: String,
    ) -> Self {
        AnthropicSseAdapter {
            inner: stream,
            state: AdapterState::Initial,
            message_id: format!("msg_{}", Uuid::new_v4().to_string().replace('-', "")),
            model,
            input_tokens: 0,
            output_tokens: 0,
            line_buf: String::new(),
            pending: std::collections::VecDeque::new(),
        }
    }

    /// Emit a full Anthropic SSE frame.
    fn frame(event: &str, data: Value) -> Bytes {
        let data_str = serde_json::to_string(&data).unwrap_or_default();
        Bytes::from(format!("event: {}\ndata: {}\n\n", event, data_str))
    }

    /// Process one parsed OpenAI SSE JSON value, returning any Anthropic frames.
    fn process_openai_chunk(&mut self, chunk: &Value) -> Vec<Bytes> {
        let mut frames: Vec<Bytes> = Vec::new();

        // Extract content delta
        let choice = chunk.get("choices").and_then(|c| c.get(0));
        let delta = choice.and_then(|c| c.get("delta"));
        let content = delta
            .and_then(|d| d.get("content"))
            .and_then(Value::as_str);
        let finish_reason = choice
            .and_then(|c| c.get("finish_reason"))
            .and_then(Value::as_str);

        // On first chunk, emit message_start + content_block_start
        if self.state == AdapterState::Initial {
            self.state = AdapterState::MessageStarted;
            frames.push(Self::frame(
                "message_start",
                serde_json::json!({
                    "type": "message_start",
                    "message": {
                        "id": self.message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.model,
                        "stop_reason": null,
                        "stop_sequence": null,
                        "usage": { "input_tokens": 0, "output_tokens": 0 }
                    }
                }),
            ));
            frames.push(Self::frame(
                "content_block_start",
                serde_json::json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": { "type": "text", "text": "" }
                }),
            ));
            // Ping
            frames.push(Bytes::from("event: ping\ndata: {\"type\":\"ping\"}\n\n"));
            self.state = AdapterState::ContentBlockStarted;
        }

        // Emit content delta
        if let Some(text) = content {
            if !text.is_empty() {
                self.state = AdapterState::Streaming;
                self.output_tokens += 1; // rough estimate; real count comes from usage
                frames.push(Self::frame(
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": { "type": "text_delta", "text": text }
                    }),
                ));
            }
        }

        // Track usage from chunk if present (some providers include it)
        if let Some(usage) = chunk.get("usage") {
            if let Some(pt) = usage.get("prompt_tokens").and_then(Value::as_u64) {
                self.input_tokens = pt as u32;
            }
            if let Some(ct) = usage.get("completion_tokens").and_then(Value::as_u64) {
                self.output_tokens = ct as u32;
            }
        }

        // On finish_reason, close the stream
        if let Some(fr) = finish_reason {
            let stop_reason = match fr {
                "stop" => "end_turn",
                "length" => "max_tokens",
                "tool_calls" => "tool_use",
                _ => "end_turn",
            };

            frames.push(Self::frame(
                "content_block_stop",
                serde_json::json!({ "type": "content_block_stop", "index": 0 }),
            ));
            frames.push(Self::frame(
                "message_delta",
                serde_json::json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": stop_reason, "stop_sequence": null },
                    "usage": { "output_tokens": self.output_tokens }
                }),
            ));
            frames.push(Self::frame(
                "message_stop",
                serde_json::json!({ "type": "message_stop" }),
            ));
            self.state = AdapterState::Done;
        }

        frames
    }
}

impl Stream for AnthropicSseAdapter {
    type Item = Result<Bytes>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            // 1. Flush pending frames first
            if let Some(frame) = self.pending.pop_front() {
                return Poll::Ready(Some(Ok(frame)));
            }

            // 2. If we are already done, finish the stream
            if self.state == AdapterState::Done {
                return Poll::Ready(None);
            }

            // 3. Pull from inner stream
            match self.as_mut().project().inner.poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    // Inner stream ended — handle transition and closing logic once.
                    if self.state != AdapterState::Done {
                        let mut extra_frames: Vec<Bytes> = Vec::new();
                        
                        // Flush any remaining data in line buffer before closing
                        if !self.line_buf.is_empty() {
                            let line = self.line_buf.trim().to_string();
                            self.line_buf.clear();
                            if let Some(json_str) = line.strip_prefix("data: ") {
                                if let Ok(chunk_val) = serde_json::from_str::<Value>(json_str) {
                                    extra_frames.extend(self.process_openai_chunk(&chunk_val));
                                }
                            }
                        }

                        // Re-check state after potential process_openai_chunk transition.
                        // Always close gracefully if we haven't reached Done yet.
                        // This handles both empty responses (Initial) and interrupted streams.
                        if self.state != AdapterState::Done {
                            extra_frames.extend(close_stream_gracefully(&mut self));
                        }
                        
                        self.state = AdapterState::Done;
                        self.pending.extend(extra_frames);
                        
                        // Loop back once more to flush the newly added pending frames
                        continue;
                    }
                    
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(Some(Ok(bytes))) => {
                    // Process the SSE chunk bytes
                    let text = match std::str::from_utf8(&bytes) {
                        Ok(t) => t.to_string(),
                        Err(_) => continue,
                    };

                    // Accumulate in line buffer and process complete lines
                    self.line_buf.push_str(&text);
                    let mut new_frames: Vec<Bytes> = Vec::new();

                    // Process complete SSE events (terminated by \n)
                    // The .trim() on the extracted line handles \r\n line endings correctly.
                    while let Some(newline_pos) = self.line_buf.find('\n') {
                        let line = self.line_buf[..newline_pos].trim().to_string();
                        self.line_buf.drain(..newline_pos + 1);

                        if let Some(json_str) = line.strip_prefix("data: ") {
                            if json_str == "[DONE]" {
                                // Normal end — handled by finish_reason already
                                continue;
                            }
                            if let Ok(chunk_val) = serde_json::from_str::<Value>(json_str) {
                                let frames = self.process_openai_chunk(&chunk_val);
                                new_frames.extend(frames);
                            }
                        }
                    }

                    for f in new_frames {
                        self.pending.push_back(f);
                    }
                    // Loop back to flush pending
                }
            }
        }
    }
}

/// Called when the upstream stream ends without emitting finish_reason.
fn close_stream_gracefully(adapter: &mut AnthropicSseAdapter) -> Vec<Bytes> {
    if matches!(
        adapter.state,
        AdapterState::Initial | AdapterState::MessageStarted | AdapterState::ContentBlockStarted | AdapterState::Streaming
    ) {
        let mut frames = Vec::new();
        
        // Ensure protocol compliance: message_start -> content_block_start
        if adapter.state == AdapterState::Initial {
            frames.push(AnthropicSseAdapter::frame(
                "message_start",
                serde_json::json!({
                    "type": "message_start",
                    "message": {
                        "id": adapter.message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": adapter.model,
                        "stop_reason": null,
                        "stop_sequence": null,
                        "usage": { "input_tokens": 0, "output_tokens": 0 }
                    }
                }),
            ));
            frames.push(AnthropicSseAdapter::frame(
                "content_block_start",
                serde_json::json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": { "type": "text", "text": "" }
                }),
            ));
        } else if adapter.state == AdapterState::MessageStarted {
             // We started the message but not the block.
             frames.push(AnthropicSseAdapter::frame(
                "content_block_start",
                serde_json::json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": { "type": "text", "text": "" }
                }),
            ));
        }

        frames.push(AnthropicSseAdapter::frame(
                "content_block_stop",
                serde_json::json!({ "type": "content_block_stop", "index": 0 }),
        ));
        frames.push(AnthropicSseAdapter::frame(
                "message_delta",
                serde_json::json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": "end_turn", "stop_sequence": null },
                    "usage": { "output_tokens": adapter.output_tokens }
                }),
        ));
        frames.push(AnthropicSseAdapter::frame(
                "message_stop",
                serde_json::json!({ "type": "message_stop" }),
        ));
        frames
    } else {
        vec![]
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    fn make_request(model: &str, messages: Vec<(&str, &str)>) -> AnthropicMessagesRequest {
        AnthropicMessagesRequest {
            model: model.to_string(),
            messages: messages
                .into_iter()
                .map(|(role, content)| AnthropicMessage {
                    role: role.to_string(),
                    content: Value::String(content.to_string()),
                })
                .collect(),
            system: None,
            max_tokens: Some(1024),
            stop_sequences: vec![],
            temperature: None,
            top_p: None,
            top_k: None,
            tools: vec![],
            tool_choice: None,
            stream: true,
        }
    }

    #[test]
    fn translate_simple_request() {
        let req = make_request("claude-3-5-sonnet", vec![("user", "Hello")]);
        let oai = anthropic_to_openai(req);

        assert_eq!(oai.model, "claude-3-5-sonnet");
        assert_eq!(oai.messages.len(), 1);
        assert_eq!(oai.messages[0].role, "user");
        assert_eq!(oai.messages[0].content.as_ref().unwrap().as_str().unwrap(), "Hello");
        assert_eq!(oai.max_tokens, Some(1024));
    }

    #[test]
    fn system_message_prepended() {
        let mut req = make_request("model", vec![("user", "Hello")]);
        req.system = Some(Value::String("System instructions".to_string()));
        let oai = anthropic_to_openai(req);

        assert_eq!(oai.messages.len(), 2);
        assert_eq!(oai.messages[0].role, "system");
        assert_eq!(oai.messages[0].content.as_ref().unwrap().as_str().unwrap(), "System instructions");
    }

    #[test]
    fn translate_model_and_messages() {
        let req = make_request("claude-3", vec![("user", "hi"), ("assistant", "hello")]);
        let oai = anthropic_to_openai(req);
        assert_eq!(oai.model, "claude-3");
        assert_eq!(oai.messages.len(), 2);
    }

    #[test]
    fn translate_content_block_array() {
        let req = AnthropicMessagesRequest {
            model: "m".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: Value::Array(vec![
                    serde_json::json!({"type": "text", "text": "block 1"}),
                    serde_json::json!({"type": "text", "text": "block 2"}),
                ]),
            }],
            system: None,
            max_tokens: None,
            stop_sequences: vec![],
            temperature: None,
            top_p: None,
            top_k: None,
            tools: vec![],
            tool_choice: None,
            stream: false,
        };
        let oai = anthropic_to_openai(req);
        assert_eq!(oai.messages[0].content.as_ref().unwrap().as_str().unwrap(), "block 1\nblock 2");
    }

    #[test]
    fn translate_stop_sequences() {
        let mut req = make_request("m", vec![]);
        req.stop_sequences = vec!["###".to_string()];
        let oai = anthropic_to_openai(req);
        assert_eq!(oai.stop.unwrap(), vec!["###".to_string()]);
    }

    #[test]
    fn translate_temperature_and_top_p() {
        let mut req = make_request("m", vec![]);
        req.temperature = Some(0.7);
        req.top_p = Some(0.9);
        let oai = anthropic_to_openai(req);
        assert_eq!(oai.temperature, Some(0.7));
        assert_eq!(oai.top_p, Some(0.9));
    }

    #[test]
    fn translate_tools() {
        let req = AnthropicMessagesRequest {
            model: "m".to_string(),
            messages: vec![],
            system: None,
            max_tokens: None,
            stop_sequences: vec![],
            temperature: None,
            top_p: None,
            top_k: None,
            tools: vec![AnthropicTool {
                name: "test_tool".to_string(),
                description: Some("desc".to_string()),
                input_schema: serde_json::json!({"type": "object"}),
            }],
            tool_choice: None,
            stream: false,
        };
        let oai = anthropic_to_openai(req);
        let extra = oai.extra.as_object().unwrap();
        let tools = extra.get("tools").unwrap().as_array().unwrap();
        assert_eq!(tools[0]["function"]["name"], "test_tool");
    }

    #[test]
    fn translate_tool_choice_auto() {
        let mut req = make_request("m", vec![]);
        req.tool_choice = Some(serde_json::json!({"type": "auto"}));
        let oai = anthropic_to_openai(req);
        assert_eq!(oai.extra["tool_choice"], "auto");
    }

    #[test]
    fn translate_tool_choice_any_becomes_required() {
        let mut req = make_request("m", vec![]);
        req.tool_choice = Some(serde_json::json!({"type": "any"}));
        let oai = anthropic_to_openai(req);
        assert_eq!(oai.extra["tool_choice"], "required");
    }

    #[test]
    fn map_stop_reason() {
        let mut adapter = AnthropicSseAdapter::new(Box::pin(futures_util::stream::empty()), "model".to_string());
        
        let chunk = serde_json::json!({
            "choices": [{
                "delta": {"content": "finished"},
                "finish_reason": "stop"
            }]
        });
        
        let frames = adapter.process_openai_chunk(&chunk);
        
        // Find message_delta frame and check stop_reason
        let msg_delta_frame = frames.iter().find(|f| {
            let s = String::from_utf8_lossy(f);
            s.contains("message_delta")
        }).expect("Should contain message_delta frame");

        let frame_text = String::from_utf8_lossy(msg_delta_frame);
        let json_part = frame_text.lines()
            .find(|l| l.starts_with("data: "))
            .unwrap()
            .strip_prefix("data: ")
            .unwrap();
        let msg_delta: Value = serde_json::from_str(json_part).unwrap();
        assert_eq!(msg_delta["delta"]["stop_reason"], "end_turn");
    }

    #[tokio::test]
    async fn empty_stream_emits_required_anthropic_frames() {
        let adapter = AnthropicSseAdapter::new(
            Box::pin(futures_util::stream::empty()),
            "claude-3-5-sonnet".to_string(),
        );
        let frames: Vec<_> = adapter.collect().await;
        let frame_texts: Vec<String> = frames.iter()
            .filter_map(|r| r.as_ref().ok())
            .map(|b| String::from_utf8_lossy(b).to_string())
            .collect();
            
        // Check for specific events in correct order
        let event_types: Vec<&str> = frame_texts.iter().filter_map(|t| {
             t.lines().find(|l| l.starts_with("event: ")).map(|l| &l[7..])
        }).collect();

        assert_eq!(event_types, vec![
            "message_start",
            "content_block_start",
            "content_block_stop",
            "message_delta",
            "message_stop"
        ]);

        let all_text = frame_texts.join("");
        assert!(all_text.contains("end_turn"), "must emit end_turn for empty stream");
    }

    #[tokio::test]
    async fn message_started_stream_emits_content_block_start() {
        // Test state where message_start was emitted but nothing else
        let mut adapter = AnthropicSseAdapter::new(
            Box::pin(futures_util::stream::empty()),
            "model".to_string(),
        );
        adapter.state = AdapterState::MessageStarted;
        
        // collect() will hit the Poll::Ready(None) branch
        let frames: Vec<_> = adapter.collect().await;
        let event_types: Vec<String> = frames.iter()
            .filter_map(|r| r.as_ref().ok())
            .map(|b| {
                let s = String::from_utf8_lossy(b);
                s.lines().find(|l| l.starts_with("event: ")).map(|l| l[7..].to_string()).unwrap_or_default()
            })
            .collect();

        // Must contain content_block_start to be valid protocol
        assert!(event_types.contains(&"content_block_start".to_string()));
        assert_eq!(*event_types.last().unwrap(), "message_stop".to_string());
    }

    #[test]
    fn translate_system_field() {
        let mut req = make_request("m", vec![("user", "hi")]);
        req.system = Some(serde_json::json!([{"type": "text", "text": "sys"}]));
        let oai = anthropic_to_openai(req);
        assert_eq!(oai.messages[0].role, "system");
        assert_eq!(oai.messages[0].content.as_ref().unwrap().as_str().unwrap(), "sys");
    }

    #[test]
    fn tool_schema_translated() {
        let req = AnthropicMessagesRequest {
            model: "m".to_string(),
            messages: vec![],
            system: None,
            max_tokens: None,
            stop_sequences: vec![],
            temperature: None,
            top_p: None,
            top_k: None,
            tools: vec![AnthropicTool {
                name: "read".to_string(),
                description: Some("read file".to_string()),
                input_schema: serde_json::json!({"properties": {"path": {"type": "string"}}}),
            }],
            tool_choice: None,
            stream: false,
        };
        let oai = anthropic_to_openai(req);
        let extra = oai.extra.as_object().unwrap();
        let tools = extra.get("tools").unwrap().as_array().unwrap();
        assert_eq!(tools[0]["function"]["parameters"]["properties"]["path"]["type"], "string");
    }

    #[test]
    fn translate_content_block_mixed() {
        let req = AnthropicMessagesRequest {
            model: "m".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: Value::Array(vec![
                    serde_json::json!({"type": "text", "text": "text content"}),
                    serde_json::json!({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}),
                ]),
            }],
            system: None,
            max_tokens: None,
            stop_sequences: vec![],
            temperature: None,
            top_p: None,
            top_k: None,
            tools: vec![],
            tool_choice: None,
            stream: false,
        };
        let oai = anthropic_to_openai(req);
        // Image block should be dropped (not supported), text preserved.
        assert_eq!(oai.messages[0].content.as_ref().unwrap().as_str().unwrap(), "text content");
    }

    #[test]
    fn stop_sequences_translated() {
        let mut req = make_request("m", vec![]);
        req.stop_sequences = vec!["\nUser:".to_string()];
        let oai = anthropic_to_openai(req);
        assert_eq!(oai.stop.unwrap()[0], "\nUser:");
    }
}
