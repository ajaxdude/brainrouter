use anyhow::Result;
use bytes::Bytes;
use futures_util::StreamExt;
use http_body_util::{BodyExt, Full, StreamBody, combinators::UnsyncBoxBody};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{body::Incoming, body::Frame, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde::Serialize;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::{TcpListener, UnixListener};
use tracing::{debug, error, info};

use crate::anthropic::{anthropic_to_openai, AnthropicMessagesRequest, AnthropicSseAdapter};
use crate::escalation;
use crate::peer_cwd::peer_cwd;
use crate::review::ReviewService;
use crate::router::Router;
use crate::routing_events::RoutingEvents;
use crate::session::SessionManager;
use crate::types::ChatCompletionRequest;
use crate::provider::ProviderResponse;

// Unified dashboard — embedded at compile time so the binary is self-contained.
const MAIN_DASHBOARD_HTML: &str = include_str!("escalation/templates/main_dashboard.html");

/// Shared state passed to all request handlers
pub struct AppState {
    pub router: Arc<Router>,
    pub session_manager: Arc<SessionManager>,
    pub review_service: Arc<ReviewService>,
    pub routing_events: Arc<RoutingEvents>,
    /// llama-swap root URL (without /v1 suffix) for status polling.
    pub llama_swap_url: String,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

#[derive(Serialize)]
struct ModelListResponse {
    object: &'static str,
    data: Vec<ModelObject>,
}

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

/// Create a JSON response with the given status code
fn json_response<T: Serialize>(status: StatusCode, body: &T) -> Response<Full<Bytes>> {
    let json = serde_json::to_vec(body).unwrap_or_else(|e| {
        error!("Failed to serialize response: {}", e);
        br#"{"error":"internal serialization error"}"#.to_vec()
    });

    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(json)))
        .expect("Failed to build response")
}

fn html_ok(html: &'static str) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/html; charset=utf-8")
        .body(
            Full::new(Bytes::from_static(html.as_bytes()))
                .map_err(|_| unreachable!())
                .boxed_unsync(),
        )
        .expect("Failed to build HTML response")
}

/// Handle incoming HTTP requests
async fn handle_request(
    req: Request<Incoming>,
    state: Arc<AppState>,
    cwd: String,
) -> Result<Response<UnsyncBoxBody<Bytes, anyhow::Error>>, Infallible> {
    let method = req.method().as_str();
    let path = req.uri().path();

    debug!("Request: {} {}", method, path);

    // Route /review/* to the escalation module
    if path.starts_with("/review") {
        let result = escalation::handle_review_request(req, Arc::clone(&state.review_service)).await;
        return result;
    }

    let response = match (method, path) {
        ("GET", "/health") => {
            let resp = json_response(StatusCode::OK, &HealthResponse { status: "ok" });
            resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
        }

        ("GET", "/v1/models") => {
            let models = ModelListResponse {
                object: "list",
                data: vec![
                    ModelObject { id: "auto".to_string(), object: "model", created: 0, owned_by: "brainrouter".to_string() },
                    ModelObject { id: "local".to_string(), object: "model", created: 0, owned_by: "brainrouter".to_string() },
                    ModelObject { id: "cloud".to_string(), object: "model", created: 0, owned_by: "brainrouter".to_string() },
                ],
            };
            let resp = json_response(StatusCode::OK, &models);
            resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
        }

        ("POST", "/v1/chat/completions") => {
            match handle_chat_completion(req, state, cwd).await {
                Ok(resp) => resp,
                Err(e) => {
                    error!("Error handling chat completion: {}", e);
                    let resp = json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &ErrorResponse { error: format!("Internal error: {}", e) },
                    );
                    resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
                }
            }
        }

        ("POST", "/v1/messages") => {
            match handle_anthropic_messages(req, state, cwd).await {
                Ok(resp) => resp,
                Err(e) => {
                    error!("Error handling Anthropic messages: {}", e);
                    let resp = json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &ErrorResponse { error: format!("Internal error: {}", e) },
                    );
                    resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
                }
            }
        }

        // ── Root redirect → dashboard ──────────────────────────────────────────
        ("GET", "/") => {
            let resp = Response::builder()
                .status(StatusCode::FOUND)
                .header("location", "/dashboard")
                .body(Full::new(Bytes::new()))
                .expect("Failed to build redirect");
            resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
        }

        // ── Unified dashboard ──────────────────────────────────────────────────
        ("GET", "/dashboard") => html_ok(MAIN_DASHBOARD_HTML),

        // ── Routing events API ─────────────────────────────────────────────────
        ("GET", "/api/routing-events") => {
            let resp = json_response(StatusCode::OK, &state.routing_events.get_all_as_response());
            resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
        }

        ("GET", "/api/routing-stats") => {
            let resp = json_response(StatusCode::OK, &state.routing_events.get_stats());
            resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
        }

        // ── Inference status API (polls llama-swap + llama-server) ────────────
        ("GET", "/api/inference-status") => {
            let resp = inference_status(&state.router.inference_tracker, &state.llama_swap_url).await;
            resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
        }

        // ── Service restart API ────────────────────────────────────────────────
        ("POST", "/api/restart/llama-swap") => {
            let resp = restart_service("llama-swap").await;
            resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
        }

        ("POST", "/api/restart/manifest") => {
            let resp = restart_service("manifest").await;
            resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
        }

        ("POST", "/api/restart/brainrouter") => {
            let resp = restart_service("brainrouter").await;
            resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
        }

        _ => {
            let resp = json_response(
                StatusCode::NOT_FOUND,
                &ErrorResponse { error: format!("Not found: {} {}", method, path) },
            );
            resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
        }
    };

    Ok(response)
}

/// Handle POST /v1/chat/completions
async fn handle_chat_completion(
    req: Request<Incoming>,
    state: Arc<AppState>,
    cwd: String,
) -> Result<Response<UnsyncBoxBody<Bytes, anyhow::Error>>, anyhow::Error> {
    let body_bytes = req.collect().await?.to_bytes();
    let request: ChatCompletionRequest = serde_json::from_slice(&body_bytes)?;
    let provider_response = state.router.route_tagged(request, None, cwd).await?.0;

    match provider_response {
        ProviderResponse::Stream(stream) => {
            let stream_body = StreamBody::new(stream.map(|chunk| chunk.map(Frame::data)));
            let response = Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "text/event-stream")
                .header("cache-control", "no-cache")
                .header("connection", "keep-alive")
                .body(stream_body.boxed_unsync())?;
            Ok(response)
        }
    }
}

/// Handle POST /v1/messages (Anthropic Messages API)
///
/// Translates the Anthropic request to OpenAI format, routes through Bonsai,
/// and translates the OpenAI SSE response back to Anthropic SSE events.
async fn handle_anthropic_messages(
    req: Request<Incoming>,
    state: Arc<AppState>,
    cwd: String,
) -> Result<Response<UnsyncBoxBody<Bytes, anyhow::Error>>, anyhow::Error> {
    let body_bytes = req.collect().await?.to_bytes();
    let anthropic_req: AnthropicMessagesRequest = serde_json::from_slice(&body_bytes)?;
    let model = anthropic_req.model.clone();
    let oai_request = anthropic_to_openai(anthropic_req);
    let provider_response = state.router.route_tagged(oai_request, None, cwd).await?.0;

    match provider_response {
        ProviderResponse::Stream(stream) => {
            let adapted = AnthropicSseAdapter::new(stream, model);
            let stream_body = StreamBody::new(adapted.map(|chunk| chunk.map(Frame::data)));
            let response = Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "text/event-stream")
                .header("cache-control", "no-cache")
                .header("connection", "keep-alive")
                .body(stream_body.boxed_unsync())?;
            Ok(response)
        }
    }
}

/// Poll llama-swap and the active model's llama-server for inference status.
/// Returns a combined view: which model is loaded, its state, and slot progress.
async fn inference_status(
    tracker: &crate::inference_state::InferenceTracker,
    llama_swap_url: &str,
) -> Response<Full<Bytes>> {
    use crate::inference_state::Phase;

    let snap = tracker.snapshot();

    match snap.phase {
        Phase::Idle => {
            // No active request in brainrouter. Check if llama-swap has a model loaded.
            let model_info = poll_llama_swap_running(llama_swap_url).await;
            match model_info {
                Some((name, display, swap_state)) if swap_state != "ready" => {
                    json_response(StatusCode::OK, &serde_json::json!({
                        "state": "loading",
                        "model": name,
                        "model_name": display,
                        "elapsed_ms": 0,
                    }))
                }
                Some((name, display, _)) => {
                    // Model is loaded and ready. Check /slots to detect activity
                    // from clients hitting llama-swap directly (bypassing brainrouter).
                    let slot_info = poll_llama_swap_slot(llama_swap_url).await;
                    let (state, n_decoded) = match &slot_info {
                        Some((true, 0)) => ("local_processing", 0u64),
                        Some((true, n)) => ("local_generating", *n),
                        _ => ("ready", 0),
                    };
                    json_response(StatusCode::OK, &serde_json::json!({
                        "state": state,
                        "model": name,
                        "model_name": display,
                        "n_decoded": n_decoded,
                    }))
                }
                None => json_response(StatusCode::OK, &serde_json::json!({
                    "state": "idle"
                })),
            }
        }
        Phase::Classifying => {
            json_response(StatusCode::OK, &serde_json::json!({
                "state": "classifying",
                "elapsed_ms": snap.elapsed_ms,
            }))
        }
        Phase::CloudWaiting => {
            json_response(StatusCode::OK, &serde_json::json!({
                "state": "cloud_waiting",
                "model": snap.model,
                "provider": snap.provider,
                "elapsed_ms": snap.elapsed_ms,
            }))
        }
        Phase::CloudStreaming => {
            json_response(StatusCode::OK, &serde_json::json!({
                "state": "cloud_streaming",
                "model": snap.model,
                "provider": snap.provider,
                "elapsed_ms": snap.elapsed_ms,
            }))
        }
        Phase::LocalWaiting | Phase::LocalStreaming => {
            // For local, enrich with llama-swap /slots data if available.
            let slot_info = poll_llama_swap_slot(llama_swap_url).await;
            let (sub_state, n_decoded) = match &slot_info {
                Some((true, 0)) => ("local_processing", 0u64),
                Some((true, n)) => ("local_generating", *n),
                Some((false, _)) if snap.phase == Phase::LocalStreaming => ("local_generating", 0),
                Some((false, _)) => ("ready", 0),
                // Slot poll failed (GPU busy, timeout) — infer from tracker phase.
                None if snap.phase == Phase::LocalStreaming => ("local_generating", 0),
                None => ("local_processing", 0),
            };
            json_response(StatusCode::OK, &serde_json::json!({
                "state": sub_state,
                "model": snap.model,
                "model_name": snap.model,
                "provider": snap.provider,
                "elapsed_ms": snap.elapsed_ms,
                "n_decoded": n_decoded,
            }))
        }
    }
}

/// Poll llama-swap /running for the active model's name, display name, and state.
async fn poll_llama_swap_running(llama_swap_url: &str) -> Option<(String, String, String)> {
    let url = format!("{}/running", llama_swap_url);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build().ok()?;
    let resp = client.get(&url).send().await.ok()?;
    let data: serde_json::Value = resp.json().await.ok()?;
    let entry = data.get("running")?.as_array()?.first()?;
    let name = entry.get("model")?.as_str()?.to_string();
    let display = entry.get("name").and_then(|n| n.as_str()).unwrap_or(&name).to_string();
    let state = entry.get("state").and_then(|s| s.as_str()).unwrap_or("unknown").to_string();
    Some((name, display, state))
}

/// Poll the active llama-server's /slots endpoint for processing state.
/// Returns (is_processing, n_decoded) if reachable.
async fn poll_llama_swap_slot(llama_swap_url: &str) -> Option<(bool, u64)> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build().ok()?;
    // First get the proxy URL from /running
    let running_url = format!("{}/running", llama_swap_url);
    let resp = client.get(&running_url).send().await.ok()?;
    let data: serde_json::Value = resp.json().await.ok()?;
    let entry = data.get("running")?.as_array()?.first()?;
    let proxy = entry.get("proxy")?.as_str()?;
    // Then poll /slots on the model's port
    let slots_url = format!("{}/slots", proxy);
    let resp = client.get(&slots_url).send().await.ok()?;
    let slots: serde_json::Value = resp.json().await.ok()?;
    let slot = slots.as_array()?.first()?;
    let is_proc = slot.get("is_processing")?.as_bool()?;
    let n_decoded = slot
        .get("next_token")
        .and_then(|nt| nt.get("n_decoded"))
        .and_then(|n| n.as_u64())
        .unwrap_or(0);
    Some((is_proc, n_decoded))
}

/// Restart a systemd user service. Only allows a fixed set of service names.
async fn restart_service(service: &str) -> Response<Full<Bytes>> {
    const ALLOWED: &[&str] = &["llama-swap", "manifest", "brainrouter"];
    if !ALLOWED.contains(&service) {
        return json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
            error: format!("Unknown service: {}", service),
        });
    }

    info!(service, "Restarting systemd user service");
    let output = tokio::process::Command::new("systemctl")
        .args(["--user", "restart", service])
        .output()
        .await;

    match output {
        Ok(out) if out.status.success() => {
            info!(service, "Service restarted successfully");
            json_response(StatusCode::OK, &serde_json::json!({
                "status": "ok",
                "service": service,
                "message": format!("{} restarted", service)
            }))
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            error!(service, %stderr, "Service restart failed");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Restart failed: {}", stderr.trim()),
            })
        }
        Err(e) => {
            error!(service, error = %e, "Failed to exec systemctl");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Failed to exec systemctl: {}", e),
            })
        }
    }
}

/// Run the HTTP server with dual listeners (TCP + Unix domain socket)
pub async fn run(
    tcp_addr: SocketAddr,
    uds_path: PathBuf,
    state: Arc<AppState>,
) -> Result<()> {
    if uds_path.exists() {
        info!("Removing existing Unix socket at {:?}", uds_path);
        std::fs::remove_file(&uds_path)?;
    }

    let tcp_listener = TcpListener::bind(tcp_addr).await?;
    info!("TCP listener bound to {}", tcp_addr);

    let uds_listener = UnixListener::bind(&uds_path)?;
    info!("Unix socket listener bound to {:?}", uds_path);

    let tcp_state = state.clone();
    let uds_state = state;

    let tcp_task = tokio::spawn(async move {
        loop {
            match tcp_listener.accept().await {
                Ok((stream, addr)) => {
                    info!("New TCP connection from {}", addr);
                    let io = TokioIo::new(stream);
                    let state = tcp_state.clone();
                    // Resolve the cwd of the connecting OMP process once per
                    // connection; all requests on this keep-alive connection
                    // share the same process and thus the same cwd.
                    let conn_cwd = peer_cwd(&addr).unwrap_or_default();
                    tokio::spawn(async move {
                        if let Err(e) = http1::Builder::new()
                            .serve_connection(io, service_fn(move |req| handle_request(req, state.clone(), conn_cwd.clone())))
                            .await
                        {
                            error!("Error serving TCP connection: {}", e);
                        }
                    });
                }
                Err(e) => error!("Failed to accept TCP connection: {}", e),
            }
        }
    });

    let uds_path_for_cleanup = uds_path.clone();
    let uds_task = tokio::spawn(async move {
        loop {
            match uds_listener.accept().await {
                Ok((stream, _addr)) => {
                    info!("New Unix socket connection");
                    let io = TokioIo::new(stream);
                    let state = uds_state.clone();
                    // UDS connections have no peer address; cwd is unknown.
                    let conn_cwd = String::new();
                    tokio::spawn(async move {
                        if let Err(e) = http1::Builder::new()
                            .serve_connection(io, service_fn(move |req| handle_request(req, state.clone(), conn_cwd.clone())))
                            .await
                        {
                            error!("Error serving Unix socket connection: {}", e);
                        }
                    });
                }
                Err(e) => error!("Failed to accept Unix socket connection: {}", e),
            }
        }
    });

    tokio::select! {
        _ = tcp_task => info!("TCP listener task ended"),
        _ = uds_task => info!("Unix socket listener task ended"),
    }

    if uds_path_for_cleanup.exists() {
        info!("Cleaning up Unix socket at {:?}", uds_path_for_cleanup);
        let _ = std::fs::remove_file(&uds_path_for_cleanup);
    }

    Ok(())
}
