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

use anyhow::{anyhow, Context as _, Result};
use bytes::Bytes;
use futures_util::Stream;
use pin_project::pin_project;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;
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
        messages.extend(anthropic_message_to_openai(msg));
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

fn anthropic_message_to_openai(msg: AnthropicMessage) -> Vec<ChatMessage> {
    let Value::Array(blocks) = &msg.content else {
        return vec![ChatMessage {
            role: msg.role,
            content: Some(msg.content),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
    };

    if msg.role == "assistant" {
        let mut text = Vec::new();
        let mut tool_calls = Vec::new();
        for block in blocks {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(value) = block.get("text").and_then(Value::as_str) {
                        text.push(value.to_string());
                    }
                }
                Some("tool_use") => {
                    let Some(id) = block.get("id").and_then(Value::as_str) else {
                        warn!("Dropping Anthropic tool_use block without id");
                        continue;
                    };
                    let Some(name) = block.get("name").and_then(Value::as_str) else {
                        warn!(tool_id = id, "Dropping Anthropic tool_use block without name");
                        continue;
                    };
                    let input = block.get("input").cloned().unwrap_or_else(|| serde_json::json!({}));
                    tool_calls.push(serde_json::json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": serde_json::to_string(&input).unwrap_or_else(|_| "{}".into()),
                        }
                    }));
                }
                Some(other) => warn!(block_type = other, "Dropping unsupported Anthropic assistant block"),
                None => warn!("Dropping Anthropic assistant block without type"),
            }
        }
        return vec![ChatMessage {
            role: "assistant".into(),
            content: (!text.is_empty()).then(|| Value::String(text.join("\n"))),
            name: None,
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            tool_call_id: None,
        }];
    }

    if msg.role == "user" {
        let mut messages = Vec::new();
        let mut text = Vec::new();
        for block in blocks {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(value) = block.get("text").and_then(Value::as_str) {
                        text.push(value.to_string());
                    }
                }
                Some("tool_result") => {
                    let Some(tool_call_id) = block.get("tool_use_id").and_then(Value::as_str) else {
                        warn!("Dropping Anthropic tool_result block without tool_use_id");
                        continue;
                    };
                    let mut content = anthropic_block_text(block.get("content"));
                    if block
                        .get("is_error")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                    {
                        content = format!("Tool error: {content}");
                    }
                    messages.push(ChatMessage {
                        role: "tool".into(),
                        content: Some(Value::String(content)),
                        name: None,
                        tool_calls: None,
                        tool_call_id: Some(tool_call_id.to_string()),
                    });
                }
                Some(other) => warn!(block_type = other, "Dropping unsupported Anthropic user block"),
                None => warn!("Dropping Anthropic user block without type"),
            }
        }
        if !text.is_empty() {
            messages.push(ChatMessage {
                role: "user".into(),
                content: Some(Value::String(text.join("\n"))),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }
        return messages;
    }

    vec![ChatMessage {
        role: msg.role,
        content: Some(msg.content),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }]
}

fn anthropic_block_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(value)) => value.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|block| {
                (block.get("type").and_then(Value::as_str) == Some("text"))
                    .then(|| block.get("text").and_then(Value::as_str))
                    .flatten()
                    .map(str::to_string)
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(value) => serde_json::to_string(value).unwrap_or_default(),
        None => String::new(),
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

#[derive(Debug, Default)]
struct PendingToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

/// Bound waiting from the first finish_reason or [DONE]. Ready upstream items
/// take priority over elapsed time; continuously ready tails have separate caps.
pub(crate) const ANTHROPIC_EOF_GRACE: Duration = Duration::from_secs(2);
pub(crate) const ANTHROPIC_TAIL_MAX_BYTES: usize = 256 * 1024;
pub(crate) const ANTHROPIC_TAIL_MAX_LINE_BYTES: usize = 64 * 1024;
pub(crate) const ANTHROPIC_TAIL_MAX_CHUNKS: usize = 1024;

/// Adapts an OpenAI SSE stream to Anthropic SSE events.
///
/// Emits in order:
///   event: message_start         (once, first chunk)
///   event: content_block_start   (once, block index 0)
///   event: content_block_delta*  (per text chunk)
///   event: content_block_stop    (once, after upstream EOF)
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
    /// Finish metadata is held until usage and upstream EOF have been consumed.
    stop_reason: Option<&'static str>,
    done_seen: bool,
    eof_deadline: Option<Pin<Box<tokio::time::Sleep>>>,
    tail_bytes: usize,
    tail_chunks: usize,
    /// Preserve partial UTF-8 and SSE lines across transport chunks.
    line_buf: Vec<u8>,
    /// Outgoing events queued to be flushed before pulling from inner
    pending: std::collections::VecDeque<Bytes>,
    text_block_index: Option<u32>,
    text_block_open: bool,
    next_block_index: u32,
    tool_calls: BTreeMap<u64, PendingToolCall>,
    tool_blocks_emitted: bool,
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
            stop_reason: None,
            done_seen: false,
            eof_deadline: None,
            tail_bytes: 0,
            tail_chunks: 0,
            line_buf: Vec::new(),
            pending: std::collections::VecDeque::new(),
            text_block_index: None,
            text_block_open: false,
            next_block_index: 0,
            tool_calls: BTreeMap::new(),
            tool_blocks_emitted: false,
        }
    }

    /// Emit a full Anthropic SSE frame.
    fn frame(event: &str, data: Value) -> Bytes {
        let data_str = serde_json::to_string(&data).unwrap_or_default();
        Bytes::from(format!("event: {}\ndata: {}\n\n", event, data_str))
    }

    fn ensure_message_started(&mut self, frames: &mut Vec<Bytes>) {
        if self.state != AdapterState::Initial {
            return;
        }
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
        frames.push(Bytes::from("event: ping\ndata: {\"type\":\"ping\"}\n\n"));
        self.state = AdapterState::MessageStarted;
    }

    fn ensure_text_block(&mut self, frames: &mut Vec<Bytes>) -> u32 {
        if let Some(index) = self.text_block_index {
            return index;
        }
        let index = self.next_block_index;
        self.next_block_index = self.next_block_index.saturating_add(1);
        self.text_block_index = Some(index);
        self.text_block_open = true;
        frames.push(Self::frame(
            "content_block_start",
            serde_json::json!({
                "type": "content_block_start",
                "index": index,
                "content_block": { "type": "text", "text": "" }
            }),
        ));
        self.state = AdapterState::ContentBlockStarted;
        index
    }

    fn close_text_block(&mut self, frames: &mut Vec<Bytes>) {
        if self.text_block_open {
            frames.push(Self::frame(
                "content_block_stop",
                serde_json::json!({
                    "type": "content_block_stop",
                    "index": self.text_block_index.unwrap_or_default()
                }),
            ));
            self.text_block_open = false;
        }
    }

    fn collect_tool_call_deltas(&mut self, delta: Option<&Value>) {
        let Some(calls) = delta
            .and_then(|value| value.get("tool_calls"))
            .and_then(Value::as_array)
        else {
            return;
        };
        for call in calls {
            let index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
            let pending = self.tool_calls.entry(index).or_default();
            if let Some(id) = call.get("id").and_then(Value::as_str) {
                pending.id = Some(id.to_string());
            }
            if let Some(function) = call.get("function") {
                if let Some(name) = function.get("name").and_then(Value::as_str) {
                    pending.name = Some(name.to_string());
                }
                if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                    pending.arguments.push_str(arguments);
                }
            }
        }
    }

    fn emit_tool_blocks(&mut self, frames: &mut Vec<Bytes>) -> Result<()> {
        if self.tool_blocks_emitted || self.tool_calls.is_empty() {
            return Ok(());
        }
        self.close_text_block(frames);
        for (_, tool) in std::mem::take(&mut self.tool_calls) {
            let id = tool
                .id
                .context("OpenAI tool call completed without an id")?;
            let name = tool
                .name
                .context("OpenAI tool call completed without a function name")?;
            let index = self.next_block_index;
            self.next_block_index = self.next_block_index.saturating_add(1);
            frames.push(Self::frame(
                "content_block_start",
                serde_json::json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": id,
                        "name": name,
                        "input": {}
                    }
                }),
            ));
            if !tool.arguments.is_empty() {
                frames.push(Self::frame(
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": tool.arguments
                        }
                    }),
                ));
            }
            frames.push(Self::frame(
                "content_block_stop",
                serde_json::json!({ "type": "content_block_stop", "index": index }),
            ));
        }
        self.tool_blocks_emitted = true;
        Ok(())
    }

    /// Process one parsed OpenAI SSE JSON value, returning any Anthropic frames.
    fn process_openai_chunk(&mut self, chunk: &Value) -> Result<Vec<Bytes>> {
        let mut frames: Vec<Bytes> = Vec::new();
        if chunk.get("error").is_some_and(|error| !error.is_null())
            || matches!(chunk.get("type").and_then(Value::as_str),
                Some("error" | "response.failed" | "response.incomplete" | "response.cancelled"))
        {
            return Err(anyhow!("Upstream reported an SSE error"));
        }
        // DeferredStream supplies JSON keepalives while waiting/draining.
        if chunk.get("type").and_then(Value::as_str) == Some("ping") {
            return Ok(vec![Bytes::from(
                "event: ping\ndata: {\"type\":\"ping\"}\n\n",
            )]);
        }
        if self.done_seen {
            return Err(anyhow!("Unexpected upstream data after [DONE]"));
        }

        // Extract content delta
        let choice = chunk.get("choices").and_then(|c| c.get(0));
        let delta = choice.and_then(|c| c.get("delta"));
        let content = delta
            .and_then(|d| d.get("content"))
            .and_then(Value::as_str);
        self.collect_tool_call_deltas(delta);
        let finish_reason = choice
            .and_then(|c| c.get("finish_reason"))
            .and_then(Value::as_str);
        if matches!(finish_reason, Some("error" | "cancelled" | "canceled")) {
            return Err(anyhow!("Upstream ended generation with an error"));
        }

        self.ensure_message_started(&mut frames);

        // Emit content delta
        if let Some(text) = content {
            if !text.is_empty() {
                let index = self.ensure_text_block(&mut frames);
                self.state = AdapterState::Streaming;
                self.output_tokens += 1; // rough estimate; real count comes from usage
                frames.push(Self::frame(
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": index,
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

        // Do not stop polling here: final usage, [DONE], errors, and confirming
        // EOF can all arrive later. Text remains streaming through pending.
        if let Some(fr) = finish_reason {
            self.stop_reason = Some(match fr {
                "stop" => "end_turn",
                "length" => "max_tokens",
                "tool_calls" => "tool_use",
                _ => "end_turn",
            });
            if fr == "tool_calls" {
                self.emit_tool_blocks(&mut frames)?;
            }
            self.start_tail_drain();
        }

        Ok(frames)
    }

    fn start_tail_drain(&mut self) {
        if self.eof_deadline.is_none() {
            self.eof_deadline = Some(Box::pin(tokio::time::sleep(ANTHROPIC_EOF_GRACE)));
        }
    }

    fn process_openai_bytes(&mut self, bytes: &[u8]) -> Result<Vec<Bytes>> {
        if self.eof_deadline.is_some() {
            self.tail_chunks += 1;
            if self.tail_chunks > ANTHROPIC_TAIL_MAX_CHUNKS {
                return Err(anyhow!("Upstream exceeded the response tail-drain chunk budget"));
            }
        }
        let mut frames = Vec::new();
        for &byte in bytes {
            if self.eof_deadline.is_some() {
                self.tail_bytes += 1;
                if self.tail_bytes > ANTHROPIC_TAIL_MAX_BYTES {
                    return Err(anyhow!("Upstream exceeded the response tail-drain byte budget"));
                }
                if self.line_buf.len() >= ANTHROPIC_TAIL_MAX_LINE_BYTES {
                    return Err(anyhow!("Upstream exceeded the response tail-drain line budget"));
                }
            }
            // Buffer incrementally: a finish marker in the middle of this chunk
            // must activate the tail cap before the remainder is copied.
            self.line_buf.push(byte);
            if byte == b'\n' {
                let line = std::mem::take(&mut self.line_buf);
                frames.extend(self.process_openai_line(&line)?);
            }
        }
        Ok(frames)
    }

    fn process_openai_line(&mut self, line: &[u8]) -> Result<Vec<Bytes>> {
        let line = std::str::from_utf8(line)?.trim();
        if line.starts_with(':') {
            return Ok(vec![Bytes::from(
                "event: ping\ndata: {\"type\":\"ping\"}\n\n",
            )]);
        }
        if line.strip_prefix("event:").map(str::trim) == Some("error") {
            return Err(anyhow!("Upstream reported an SSE error event"));
        }
        let Some(data) = line.strip_prefix("data:") else {
            return Ok(Vec::new());
        };
        let data = data.trim();
        if data.is_empty() {
            return Ok(Vec::new());
        }
        if data == "[DONE]" {
            if !self.done_seen {
                self.done_seen = true;
                self.start_tail_drain();
            }
            return Ok(Vec::new());
        }
        self.process_openai_chunk(&serde_json::from_str::<Value>(data)?)
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

            // 3. Ready data/EOF/errors win over the timer, like tokio::timeout.
            // A slow consumer must not lose an already completed response.
            match self.as_mut().project().inner.poll_next(cx) {
                Poll::Pending => {
                    if self.eof_deadline.as_mut().is_some_and(|timer| timer.as_mut().poll(cx).is_ready()) {
                        self.state = AdapterState::Done;
                        return Poll::Ready(Some(Err(anyhow!("Upstream exceeded the response tail-drain deadline"))));
                    }
                    return Poll::Pending;
                }
                Poll::Ready(None) => {
                    let line = std::mem::take(&mut self.line_buf);
                    match self.process_openai_line(&line) {
                        Ok(frames) => self.pending.extend(frames),
                        Err(error) => {
                            self.state = AdapterState::Done;
                            return Poll::Ready(Some(Err(error)));
                        }
                    }
                    // Preserve graceful EOF compatibility for providers without
                    // [DONE]; the inner measurement wrapper excludes those samples.
                    let closing = close_stream_gracefully(&mut self);
                    self.pending.extend(closing);
                    self.state = AdapterState::Done;
                    continue;
                }
                Poll::Ready(Some(Err(e))) => {
                    self.state = AdapterState::Done;
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(Some(Ok(bytes))) => {
                    match self.process_openai_bytes(&bytes) {
                        Ok(frames) => self.pending.extend(frames),
                        Err(error) => {
                            self.line_buf = Vec::new();
                            self.state = AdapterState::Done;
                            return Poll::Ready(Some(Err(error)));
                        }
                    }
                    // Loop back to flush pending
                }
            }
        }
    }
}

/// Close only after actual upstream EOF, using any previously observed finish
/// reason and final usage. EOF without [DONE] retains legacy wire compatibility.
fn close_stream_gracefully(adapter: &mut AnthropicSseAdapter) -> Vec<Bytes> {
    if adapter.state != AdapterState::Done {
        let mut frames = Vec::new();
        adapter.ensure_message_started(&mut frames);
        if let Err(error) = adapter.emit_tool_blocks(&mut frames) {
            return vec![AnthropicSseAdapter::frame(
                "error",
                serde_json::json!({
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": error.to_string()
                    }
                }),
            )];
        }
        if adapter.next_block_index == 0 {
            adapter.ensure_text_block(&mut frames);
        }
        adapter.close_text_block(&mut frames);
        frames.push(AnthropicSseAdapter::frame(
            "message_delta",
            serde_json::json!({
                "type": "message_delta",
                "delta": { "stop_reason": adapter.stop_reason.unwrap_or("end_turn"), "stop_sequence": null },
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
    fn translates_tool_use_and_tool_result_messages() {
        let request = AnthropicMessagesRequest {
            model: "m".to_string(),
            messages: vec![
                AnthropicMessage {
                    role: "assistant".to_string(),
                    content: serde_json::json!([
                        {"type": "text", "text": "checking"},
                        {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {"q": "rust"}}
                    ]),
                },
                AnthropicMessage {
                    role: "user".to_string(),
                    content: serde_json::json!([
                        {"type": "tool_result", "tool_use_id": "tool-1", "content": [{"type": "text", "text": "found"}]},
                        {"type": "text", "text": "continue"}
                    ]),
                },
            ],
            system: None,
            max_tokens: Some(100),
            stop_sequences: vec![],
            temperature: None,
            top_p: None,
            top_k: None,
            tools: vec![],
            tool_choice: None,
            stream: true,
        };

        let translated = anthropic_to_openai(request);
        assert_eq!(translated.messages.len(), 3);
        assert_eq!(translated.messages[0].role, "assistant");
        assert_eq!(translated.messages[0].content.as_ref().and_then(Value::as_str), Some("checking"));
        let tool_call = &translated.messages[0].tool_calls.as_ref().unwrap()[0];
        assert_eq!(tool_call["id"], "tool-1");
        assert_eq!(tool_call["function"]["name"], "lookup");
        assert_eq!(tool_call["function"]["arguments"], r#"{"q":"rust"}"#);
        assert_eq!(translated.messages[1].role, "tool");
        assert_eq!(translated.messages[1].tool_call_id.as_deref(), Some("tool-1"));
        assert_eq!(translated.messages[1].content.as_ref().and_then(Value::as_str), Some("found"));
        assert_eq!(translated.messages[2].role, "user");
        assert_eq!(translated.messages[2].content.as_ref().and_then(Value::as_str), Some("continue"));
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

    #[tokio::test]
    async fn map_stop_reason() {
        let mut adapter = AnthropicSseAdapter::new(Box::pin(futures_util::stream::empty()), "model".to_string());
        
        let chunk = serde_json::json!({
            "choices": [{
                "delta": {"content": "finished"},
                "finish_reason": "stop"
            }]
        });
        
        let frames = adapter.process_openai_chunk(&chunk).unwrap();
        assert!(!frames.iter().any(|frame| String::from_utf8_lossy(frame).contains("message_stop")));
        assert_ne!(adapter.state, AdapterState::Done);
        let frames = close_stream_gracefully(&mut adapter);
        
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
            "ping",
            "content_block_start",
            "content_block_stop",
            "message_delta",
            "message_stop"
        ]);

        let all_text = frame_texts.join("");
        assert!(all_text.contains("end_turn"), "must emit end_turn for empty stream");
    }

    #[tokio::test]
    async fn fragmented_tool_calls_emit_anthropic_tool_blocks() {
        let chunks = vec![
            Ok(Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"q\\\":\"}}]},\"finish_reason\":null}]}\n\n",
            )),
            Ok(Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"rust\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n",
            )),
        ];
        let adapter = AnthropicSseAdapter::new(
            Box::pin(futures_util::stream::iter(chunks)),
            "model".to_string(),
        );
        let output = adapter
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let events = output
            .iter()
            .filter_map(|frame| {
                String::from_utf8_lossy(frame)
                    .lines()
                    .find_map(|line| line.strip_prefix("data: "))
                    .and_then(|data| serde_json::from_str::<Value>(data).ok())
            })
            .collect::<Vec<_>>();
        let tool_start = events
            .iter()
            .find(|event| event["content_block"]["type"] == "tool_use")
            .expect("tool_use block");
        assert_eq!(tool_start["content_block"]["id"], "call-1");
        assert_eq!(tool_start["content_block"]["name"], "lookup");
        let tool_delta = events
            .iter()
            .find(|event| event["delta"]["type"] == "input_json_delta")
            .expect("tool input delta");
        assert_eq!(tool_delta["delta"]["partial_json"], r#"{"q":"rust"}"#);
        assert!(events
            .iter()
            .any(|event| event["delta"]["stop_reason"] == "tool_use"));
    }

    #[tokio::test]
    async fn forwards_keepalive_before_first_provider_output() {
        let chunks = vec![
            Ok(Bytes::from_static(b": keepalive\n\n")),
            Ok(Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
            )),
        ];
        let adapter = AnthropicSseAdapter::new(
            Box::pin(futures_util::stream::iter(chunks)),
            "model".to_string(),
        );
        let output = adapter
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert!(String::from_utf8_lossy(&output[0]).contains("event: ping"));
        assert!(output
            .iter()
            .any(|frame| String::from_utf8_lossy(frame).contains("message_start")));
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
