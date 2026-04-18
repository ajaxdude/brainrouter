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
//!        message_start → content_block_start → content_block_delta* → content_block_stop
//!        → message_delta (stop_reason) → message_stop
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
#[derive(Debug, PartialEq)]
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
            // Flush pending frames first
            if let Some(frame) = self.pending.pop_front() {
                return Poll::Ready(Some(Ok(frame)));
            }

            if self.state == AdapterState::Done {
                return Poll::Ready(None);
            }

            // Pull from inner stream
            match self.as_mut().project().inner.poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    // Stream ended without finish_reason — close gracefully
                    if self.state != AdapterState::Done {
                        let frames = close_stream_gracefully(&mut self);
                        for f in frames {
                            self.pending.push_back(f);
                        }
                        self.state = AdapterState::Done;
                        continue; // flush the pending
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

                    // Process complete SSE events (terminated by \n\n)
                    loop {
                        // Find a complete data line
                        if let Some(newline_pos) = self.line_buf.find('\n') {
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
                        } else {
                            break;
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
        AdapterState::ContentBlockStarted | AdapterState::Streaming
    ) {
        vec![
            AnthropicSseAdapter::frame(
                "content_block_stop",
                serde_json::json!({ "type": "content_block_stop", "index": 0 }),
            ),
            AnthropicSseAdapter::frame(
                "message_delta",
                serde_json::json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": "end_turn", "stop_sequence": null },
                    "usage": { "output_tokens": adapter.output_tokens }
                }),
            ),
            AnthropicSseAdapter::frame(
                "message_stop",
                serde_json::json!({ "type": "message_stop" }),
            ),
        ]
    } else {
        vec![]
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(
            oai.messages[0].content,
            Some(Value::String("Hello".to_string()))
        );
        assert_eq!(oai.max_tokens, Some(1024));
    }

    #[test]
    fn system_message_prepended() {
        let mut req = make_request("claude-3-5-sonnet", vec![("user", "Hi")]);
        req.system = Some(Value::String("You are helpful.".to_string()));
        let oai = anthropic_to_openai(req);

        assert_eq!(oai.messages.len(), 2);
        assert_eq!(oai.messages[0].role, "system");
        assert_eq!(
            oai.messages[0].content,
            Some(Value::String("You are helpful.".to_string()))
        );
        assert_eq!(oai.messages[1].role, "user");
    }

    #[test]
    fn stop_sequences_translated() {
        let mut req = make_request("claude-3-5-sonnet", vec![("user", "hi")]);
        req.stop_sequences = vec!["STOP".to_string(), "END".to_string()];
        let oai = anthropic_to_openai(req);

        assert_eq!(oai.stop, Some(vec!["STOP".to_string(), "END".to_string()]));
    }

    #[test]
    fn tool_schema_translated() {
        let mut req = make_request("claude-3-5-sonnet", vec![("user", "hi")]);
        req.tools = vec![AnthropicTool {
            name: "search".to_string(),
            description: Some("Search the web".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "query": { "type": "string" } }
            }),
        }];

        let oai = anthropic_to_openai(req);
        let tools = oai.extra.get("tools").and_then(Value::as_array).unwrap();
        assert_eq!(tools.len(), 1);
        let func = tools[0].get("function").unwrap();
        assert_eq!(func.get("name").and_then(Value::as_str), Some("search"));
        assert!(func.get("parameters").is_some());
    }

    #[test]
    fn map_stop_reason() {
        // Test via the state machine logic indirectly
        let chunk = serde_json::json!({
            "choices": [{ "delta": { "content": "" }, "finish_reason": "stop" }]
        });

        let mut adapter = AnthropicSseAdapter::new(
            Box::pin(futures_util::stream::empty()),
            "claude-3-5-sonnet".to_string(),
        );
        // Force state past initial
        adapter.state = AdapterState::ContentBlockStarted;
        let frames = adapter.process_openai_chunk(&chunk);

        // Should emit content_block_stop, message_delta, message_stop
        assert_eq!(frames.len(), 3);
        let frame_strs: Vec<_> = frames
            .iter()
            .map(|b| String::from_utf8_lossy(b).to_string())
            .collect();
        assert!(frame_strs[0].contains("content_block_stop"));
        assert!(frame_strs[1].contains("end_turn"));
        assert!(frame_strs[2].contains("message_stop"));
    }
}
