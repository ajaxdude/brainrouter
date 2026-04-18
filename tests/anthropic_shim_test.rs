//! Integration tests for the Anthropic protocol shim.
//!
//! Tests request translation (Anthropic → OpenAI) and response translation
//! (OpenAI SSE stream → Anthropic SSE events) without requiring live providers.

use brainrouter::anthropic::{anthropic_to_openai, AnthropicMessagesRequest, AnthropicSseAdapter};
use bytes::Bytes;
use futures_util::{stream, StreamExt};
use serde_json::Value;

fn make_req(model: &str, content: &str) -> AnthropicMessagesRequest {
    serde_json::from_value(serde_json::json!({
        "model": model,
        "messages": [{ "role": "user", "content": content }],
        "max_tokens": 512,
        "stream": true
    }))
    .unwrap()
}

// ─── Request translation tests ────────────────────────────────────────────────

#[test]
fn translate_model_and_messages() {
    let req = make_req("claude-3-5-sonnet-20241022", "Hello");
    let oai = anthropic_to_openai(req);
    assert_eq!(oai.model, "claude-3-5-sonnet-20241022");
    assert_eq!(oai.messages.len(), 1);
    assert_eq!(oai.messages[0].role, "user");
}

#[test]
fn translate_system_field() {
    let req: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
        "model": "claude-3-5-sonnet",
        "system": "You are a test assistant.",
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 100
    }))
    .unwrap();

    let oai = anthropic_to_openai(req);
    assert_eq!(oai.messages.len(), 2);
    assert_eq!(oai.messages[0].role, "system");
    assert_eq!(
        oai.messages[0].content,
        Some(Value::String("You are a test assistant.".to_string()))
    );
}

#[test]
fn translate_temperature_and_top_p() {
    let req: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
        "model": "claude-3-5-sonnet",
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9
    }))
    .unwrap();

    let oai = anthropic_to_openai(req);
    assert_eq!(oai.temperature, Some(0.7));
    assert_eq!(oai.top_p, Some(0.9));
}

#[test]
fn translate_stop_sequences() {
    let req: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
        "model": "claude-3-5-sonnet",
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 100,
        "stop_sequences": ["STOP", "END"]
    }))
    .unwrap();

    let oai = anthropic_to_openai(req);
    assert_eq!(oai.stop, Some(vec!["STOP".to_string(), "END".to_string()]));
}

#[test]
fn translate_tools() {
    let req: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
        "model": "claude-3-5-sonnet",
        "messages": [{ "role": "user", "content": "search for rust" }],
        "max_tokens": 100,
        "tools": [{
            "name": "web_search",
            "description": "Search the web",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }
        }]
    }))
    .unwrap();

    let oai = anthropic_to_openai(req);
    let tools = oai.extra.get("tools").and_then(Value::as_array).unwrap();
    assert_eq!(tools.len(), 1);
    let func = &tools[0]["function"];
    assert_eq!(func["name"], "web_search");
    assert_eq!(func["description"], "Search the web");
    // input_schema → parameters
    assert!(func["parameters"].is_object());
}

#[test]
fn translate_tool_choice_auto() {
    let req: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
        "model": "claude-3-5-sonnet",
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 100,
        "tools": [{ "name": "t", "input_schema": {} }],
        "tool_choice": { "type": "auto" }
    }))
    .unwrap();

    let oai = anthropic_to_openai(req);
    assert_eq!(oai.extra.get("tool_choice"), Some(&Value::String("auto".to_string())));
}

#[test]
fn translate_tool_choice_any_becomes_required() {
    let req: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
        "model": "claude-3-5-sonnet",
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 100,
        "tools": [{ "name": "t", "input_schema": {} }],
        "tool_choice": { "type": "any" }
    }))
    .unwrap();

    let oai = anthropic_to_openai(req);
    assert_eq!(oai.extra.get("tool_choice"), Some(&Value::String("required".to_string())));
}

#[test]
fn translate_content_block_array() {
    // Anthropic content can be an array of blocks
    let req: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
        "model": "claude-3-5-sonnet",
        "messages": [{
            "role": "user",
            "content": [
                { "type": "text", "text": "Hello" },
                { "type": "text", "text": " world" }
            ]
        }],
        "max_tokens": 100
    }))
    .unwrap();

    let oai = anthropic_to_openai(req);
    // Text blocks should be joined
    assert_eq!(
        oai.messages[0].content,
        Some(Value::String("Hello\n world".to_string()))
    );
}

// ─── SSE adapter tests ────────────────────────────────────────────────────────

/// Build a fake OpenAI SSE stream from a vec of data strings.
fn make_openai_stream(chunks: Vec<&'static str>) -> Pin<Box<dyn futures_util::Stream<Item = anyhow::Result<Bytes>> + Send>> {
    let items: Vec<anyhow::Result<Bytes>> = chunks
        .into_iter()
        .map(|s| Ok(Bytes::from(format!("data: {}\n\n", s))))
        .collect();
    Box::pin(stream::iter(items))
}

use std::pin::Pin;

#[tokio::test]
async fn adapter_emits_message_start_on_first_chunk() {
    let openai_chunk = r#"{"choices":[{"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}"#;
    let stream = make_openai_stream(vec![openai_chunk]);
    let adapter = AnthropicSseAdapter::new(stream, "claude-3-5-sonnet".to_string());

    let chunks: Vec<Bytes> = adapter
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();

    let all_text: String = chunks
        .iter()
        .map(|b| String::from_utf8_lossy(b).to_string())
        .collect();

    assert!(all_text.contains("event: message_start"), "missing message_start in: {}", all_text);
    assert!(all_text.contains("event: content_block_start"), "missing content_block_start");
    assert!(all_text.contains("event: ping"), "missing ping");
    assert!(all_text.contains("event: content_block_delta"), "missing content_block_delta");
    assert!(all_text.contains("Hello"), "missing content text");
}

#[tokio::test]
async fn adapter_emits_stop_events_on_finish_reason() {
    let chunk_with_finish = r#"{"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}"#;
    let stream = make_openai_stream(vec![chunk_with_finish]);
    let adapter = AnthropicSseAdapter::new(stream, "claude-3-5-sonnet".to_string());

    let chunks: Vec<Bytes> = adapter
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();

    let all_text: String = chunks
        .iter()
        .map(|b| String::from_utf8_lossy(b).to_string())
        .collect();

    assert!(all_text.contains("event: content_block_stop"), "missing content_block_stop in: {}", all_text);
    assert!(all_text.contains("event: message_delta"), "missing message_delta");
    assert!(all_text.contains("end_turn"), "missing end_turn stop_reason");
    assert!(all_text.contains("event: message_stop"), "missing message_stop");
}

#[tokio::test]
async fn adapter_maps_finish_reason_length_to_max_tokens() {
    let chunk = r#"{"choices":[{"delta":{"content":""},"finish_reason":"length"}]}"#;
    let stream = make_openai_stream(vec![chunk]);
    let adapter = AnthropicSseAdapter::new(stream, "claude-3-5-sonnet".to_string());

    let all_text: String = adapter
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(|r| r.ok())
        .map(|b| String::from_utf8_lossy(&b).to_string())
        .collect();

    assert!(all_text.contains("max_tokens"), "expected max_tokens stop_reason in: {}", all_text);
}

#[tokio::test]
async fn adapter_multi_chunk_full_sequence() {
    let chunks = vec![
        r#"{"choices":[{"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}"#,
        r#"{"choices":[{"delta":{"content":" there"},"finish_reason":null}]}"#,
        r#"{"choices":[{"delta":{"content":"!"},"finish_reason":"stop"}]}"#,
    ];

    let stream = make_openai_stream(chunks);
    let adapter = AnthropicSseAdapter::new(stream, "claude-3-5-sonnet".to_string());

    let all_text: String = adapter
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(|r| r.ok())
        .map(|b| String::from_utf8_lossy(&b).to_string())
        .collect();

    // Full event sequence must be present
    assert!(all_text.contains("message_start"));
    assert!(all_text.contains("content_block_start"));
    assert!(all_text.contains("Hi"));
    assert!(all_text.contains(" there"));
    assert!(all_text.contains("!"));
    assert!(all_text.contains("content_block_stop"));
    assert!(all_text.contains("message_stop"));
}
