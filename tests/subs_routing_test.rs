//! Subs-pool routing tests.
//!
//! Verifies the `subs` / `brainrouter/subs` convention added for the dual-Dirk
//! plan: requests targeting the subs pool route directly to `llama_swap.subs_model`
//! and never touch the Bonsai classifier or the Manifest circuit breaker.
//!
//! No real Bonsai server or cloud backend is needed — the classifier points at a
//! dead port with `enabled=false` and the Manifest circuit is pre-opened; if the
//! subs path consulted either, the test would fail loudly instead of passing.

use brainrouter::classifier::Classifier;
use brainrouter::config::NudgeBudgets;
use brainrouter::health::HealthTracker;
use brainrouter::inference_state::InferenceTracker;
use brainrouter::provider::openai::OpenAiProvider;
use brainrouter::routing_events::RoutingEvents;
use brainrouter::router::{Router, RouterArgs};
use brainrouter::types::{ChatCompletionRequest, ChatMessage};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU8};
use std::sync::Arc;

const SUBS_KEY: &str = "dirk-qwen3.8-27b-q6-subs";
const FALLBACK_KEY: &str = "fallback-model";

/// Spawn a raw-TCP mock llama-swap: returns a URL and a shared list of every
/// `model` field it saw in request bodies. Responds with a minimal OpenAI SSE.
async fn spawn_mock_llama_swap() -> (String, Arc<Mutex<Vec<String>>>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let captured = Arc::new(Mutex::new(Vec::new()));

    let cap = captured.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                break;
            };
            let cap = cap.clone();
            tokio::spawn(async move {
                // Read the request head + body (chat completions bodies are small;
                // one read is enough for tests).
                let mut buf = vec![0u8; 65536];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                if n == 0 {
                    return;
                }
                let text = String::from_utf8_lossy(&buf[..n]);
                if let Some(body) = text.split("\r\n\r\n").nth(1) {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(body.trim_end()) {
                        if let Some(model) = v.get("model").and_then(|m| m.as_str()) {
                            cap.lock().push(model.to_string());
                        }
                    }
                }
                let body = "data: {\"id\":\"mock-1\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"mock\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"mock-1\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"mock\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                sock.write_all(resp.as_bytes()).await.ok();
            });
        }
    });

    (format!("http://{}", addr), captured)
}

/// Build a Router whose llama-swap points at the mock, whose classifier is
/// disabled and aimed at a dead port, and whose Manifest circuit is pre-open.
fn make_router(
    llama_swap_url: &str,
    subs_model: Option<&str>,
    local_models: Vec<&str>,
) -> Router {
    let classifier = Arc::new(Classifier::new(
        "http://127.0.0.1:1".to_string(), // dead — must never be called
        FALLBACK_KEY.to_string(),
        Arc::new(AtomicBool::new(false)), // classifier disabled
        Arc::new(AtomicBool::new(false)), // nudge off
        None,
    ));
    let llama_swap = Arc::new(OpenAiProvider::new(
        "llama-swap".to_string(),
        llama_swap_url.to_string(),
        None,
    ));
    let manifest = Arc::new(OpenAiProvider::new(
        "manifest".to_string(),
        "http://127.0.0.1:1".to_string(), // dead — must never be called
        None,
    ));

    let health = Arc::new(HealthTracker::new());
    // Pre-open the Manifest circuit: any cloud attempt would fail as a backend
    // fault and fall back — but the subs path must never even try.
    for _ in 0..3 {
        health.report_failure("manifest");
    }

    Router::new(RouterArgs {
        classifier,
        manifest,
        manifest_enabled: true,
        llama_swap,
        fallback_model: FALLBACK_KEY.to_string(),
        local_models: local_models.into_iter().map(|s| s.to_string()).collect(),
        subs_model: subs_model.map(|s| s.to_string()),
        health,
        routing_events: Arc::new(RoutingEvents::new()),
        local_system_prompt: None,
        inference_tracker: Arc::new(InferenceTracker::new()),
        nudge_budgets: NudgeBudgets::default(),
        nudge_enabled: Arc::new(AtomicBool::new(false)),
        nudge_tier: Arc::new(AtomicU8::new(0)),
        prompt_rewrite: Arc::new(AtomicBool::new(true)),
    })
}

fn chat_request(model: &str) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: model.to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: Some(serde_json::Value::String("hello".to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        stream: Some(true),
        temperature: None,
        max_tokens: None,
        top_p: None,
        stop: None,
        extra: serde_json::json!({}),
    }
}

#[tokio::test]
async fn brainrouter_subs_prefix_routes_to_subs_model() {
    let (url, captured) = spawn_mock_llama_swap().await;
    let router = make_router(&url, Some(SUBS_KEY), vec![]);

    let (resp, info) = router
        .route_tagged(chat_request("brainrouter/subs"), None, "/tmp".to_string(), String::new())
        .await
        .unwrap();

    // Routed to llama-swap with the subs model key, bypassing Bonsai + cloud.
    assert_eq!(info.bonsai_decision, "local-subs");
    assert_eq!(info.effective_provider.as_deref(), Some("llama-swap"));
    assert_eq!(info.model_key, SUBS_KEY);
    drop(resp);

    // Give the mock a moment to record the model, then assert what it saw.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let seen = captured.lock().clone();
    assert_eq!(seen, vec![SUBS_KEY.to_string()], "mock saw exactly the subs model");
}

#[tokio::test]
async fn bare_subs_model_also_routes_to_subs_pool() {
    let (url, captured) = spawn_mock_llama_swap().await;
    let router = make_router(&url, Some(SUBS_KEY), vec![]);

    let (resp, info) = router
        .route_tagged(chat_request("subs"), None, "/tmp".to_string(), String::new())
        .await
        .unwrap();

    assert_eq!(info.bonsai_decision, "local-subs");
    assert_eq!(info.model_key, SUBS_KEY);
    drop(resp);

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert_eq!(captured.lock().clone(), vec![SUBS_KEY.to_string()]);
}

#[tokio::test]
async fn direct_subs_model_key_via_local_models_also_bypasses_bonsai() {
    let (url, captured) = spawn_mock_llama_swap().await;
    // The subs key listed in local_models is a second direct path (Phase 3 wiring).
    let router = make_router(&url, None, vec![SUBS_KEY]);

    let (resp, info) = router
        .route_tagged(chat_request(SUBS_KEY), None, "/tmp".to_string(), String::new())
        .await
        .unwrap();

    assert_eq!(info.bonsai_decision, "local-specific");
    assert_eq!(info.model_key, SUBS_KEY);
    drop(resp);

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert_eq!(captured.lock().clone(), vec![SUBS_KEY.to_string()]);
}

#[tokio::test]
async fn unconfigured_subs_falls_back_to_auto_without_error() {
    let (url, captured) = spawn_mock_llama_swap().await;
    // No subs_model, no local_models: `brainrouter/subs` must fall back to the
    // auto path (classifier disabled → local default = fallback_model).
    let router = make_router(&url, None, vec![]);

    let (resp, info) = router
        .route_tagged(chat_request("brainrouter/subs"), None, "/tmp".to_string(), String::new())
        .await
        .unwrap();

    assert_eq!(info.bonsai_decision, "local");
    assert_eq!(info.model_key, FALLBACK_KEY);
    drop(resp);

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert_eq!(captured.lock().clone(), vec![FALLBACK_KEY.to_string()]);
}

/// Spawn a mock llama-swap that records the model in each request body and
/// replies HTTP 500 (a backend error) so the router's error path runs.
/// Returns the URL and the shared list of seen model keys.
async fn spawn_failing_llama_swap() -> (String, Arc<Mutex<Vec<String>>>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let captured = Arc::new(Mutex::new(Vec::new()));

    let cap = captured.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                break;
            };
            let cap = cap.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 65536];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                if n == 0 {
                    return;
                }
                let text = String::from_utf8_lossy(&buf[..n]);
                if let Some(body) = text.split("\r\n\r\n").nth(1) {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(body.trim_end()) {
                        if let Some(model) = v.get("model").and_then(|m| m.as_str()) {
                            cap.lock().push(model.to_string());
                        }
                    }
                }
                let resp = "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n";
                sock.write_all(resp.as_bytes()).await.ok();
            });
        }
    });

    (format!("http://{}", addr), captured)
}

#[tokio::test]
async fn direct_model_failure_does_not_fall_back_to_fallback_model() {
    let (url, captured) = spawn_failing_llama_swap().await;
    // A directly-selected model that fails to load must surface the error, NOT
    // silently retry with fallback_model. This is the "memory compaction"
    // surprise: brainrouter/ds4-deepseek-... was failing and switching to the
    // subs pool. Direct picks are authoritative — no fallback hop.
    let router = make_router(&url, None, vec![]);

    let result = router
        .route_tagged(chat_request("ds4-deepseek-v4-flash-0731-layers37"), None, "/tmp".to_string(), String::new())
        .await;
    assert!(result.is_err(), "direct model failure must surface, not fall back");

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let seen = captured.lock().clone();
    assert_eq!(
        seen,
        vec!["ds4-deepseek-v4-flash-0731-layers37".to_string()],
        "mock saw exactly the requested model — fallback_model must never be attempted"
    );
}

#[tokio::test]
async fn auto_local_still_falls_back_on_failure() {
    let (url, _captured) = spawn_failing_llama_swap().await;
    // Managed routing (auto → local) keeps the fallback hop: if Bonsai's chosen
    // model fails, route_local retries with fallback_model. Only direct picks
    // are authoritative.
    let router = make_router(&url, None, vec![]);

    let result = router
        .route_tagged(chat_request("auto"), None, "/tmp".to_string(), String::new())
        .await;
    // Classifier is off → auto routes local to default_local_model (fallback_key).
    // That fails → retry with fallback_model (same key here) → also fails.
    assert!(result.is_err());
    drop(result);
}
