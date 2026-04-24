// ── Runtime home-dir helpers ─────────────────────────────────────────────────
//
// The daemon may run under any user. Never hardcode /home/<user>.
// These helpers resolve paths relative to $HOME at runtime.

fn home_dir() -> String {
    std::env::var("HOME").unwrap_or_else(|_| "/root".to_string())
}

/// Resolve a binary name relative to ~/.local/bin, falling back to PATH.
fn home_bin(name: &str) -> String {
    let candidate = format!("{HOME}/.local/bin/{name}", HOME = home_dir());
    if std::path::Path::new(&candidate).exists() {
        candidate
    } else {
        name.to_string()
    }
}

/// Resolve a path relative to $HOME.
fn home_path(rel: &str) -> String {
    format!("{}/{}", home_dir(), rel)
}

// ─────────────────────────────────────────────────────────────────────────────

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
use tracing::{debug, error, info, warn};

use crate::anthropic::{anthropic_to_openai, AnthropicMessagesRequest, AnthropicSseAdapter};
use crate::escalation;
use crate::peer_cwd::peer_cwd;
use crate::review::ReviewService;
use crate::router::Router;
use crate::routing_events::RoutingEvents;
use crate::session::SessionManager;
use crate::types::ChatCompletionRequest;
use crate::provider::ProviderResponse;
use crate::stream::{SafeStream, StreamFormat};

// Unified dashboard — embedded at compile time so the binary is self-contained.
const MAIN_DASHBOARD_HTML: &str = include_str!("escalation/templates/main_dashboard.html");
const FAVICON_SVG: &[u8] = include_bytes!("escalation/templates/favicon.svg");
const LOGO_SVG: &[u8] = include_bytes!("escalation/templates/logo.svg");

/// Shared state passed to all request handlers
pub struct AppState {
    pub router: Arc<Router>,
    pub session_manager: Arc<SessionManager>,
    pub review_service: Arc<ReviewService>,
    pub routing_events: Arc<RoutingEvents>,
    /// llama-swap root URL (without /v1 suffix) for status polling.
    pub llama_swap_url: String,
    /// Path to the script that restarts the llama.cpp toolbox.
    pub llama_cpp_restart_script: Option<String>,
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
    peer_addr: SocketAddr,
) -> Result<Response<UnsyncBoxBody<Bytes, anyhow::Error>>, Infallible> {
    let method = req.method().as_str();
    let path = req.uri().path();

    debug!("Request: {} {}", method, path);

    // Security: Only allow localhost (127.0.0.1 or ::1) for destructive APIs.
    // UDS connections (peer_addr = 0.0.0.0:0) are always allowed as they are local.
    let is_local = peer_addr.ip().is_loopback() || peer_addr.port() == 0;
    let is_destructive = path.starts_with("/api/restart/") || path.starts_with("/api/upgrade/");

    if is_destructive {
        if !is_local {
            error!("Blocking destructive API request from non-local peer: {}", peer_addr);
            let resp = json_response(
                StatusCode::FORBIDDEN,
                &ErrorResponse { error: "Destructive APIs only allowed from localhost".to_string() },
            );
            return Ok(resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!()))));
        }
        
        // Anti-CSRF: Check Origin/Referer for browser-originated POSTs.
        let has_allowed_origin = if let Some(origin) = req.headers().get("Origin") {
            let s = origin.to_str().unwrap_or("");
            s == "null" || s.starts_with("http://localhost:") || s.starts_with("http://127.0.0.1:")
        } else if let Some(referer) = req.headers().get("Referer") {
            let s = referer.to_str().unwrap_or("");
            s.starts_with("http://localhost:") || s.starts_with("http://127.0.0.1:")
        } else {
            // Non-browser client (curl, MCP) doesn't send Origin usually.
            true
        };

        if !has_allowed_origin {
             error!("Blocking CSRF attempt on destructive API: Origin/Referer mismatch");
             let resp = json_response(
                StatusCode::FORBIDDEN,
                &ErrorResponse { error: "CSRF protection: Invalid Origin/Referer".to_string() },
            );
            return Ok(resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!()))));
        }
    }

    // Route /review/* to the escalation module
    if path.starts_with("/review") {
        let result = escalation::handle_review_request(req, Arc::clone(&state.review_service), cwd).await;
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

        ("GET", "/favicon.ico") | ("GET", "/favicon.svg") | ("GET", "/favicon.png") => {
            let resp = Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "image/svg+xml")
                .body(Full::new(Bytes::from_static(FAVICON_SVG)))
                .expect("Failed to build favicon response");
            resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
        }

        ("GET", "/logo.svg") => {
            let resp = Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "image/svg+xml")
                .body(Full::new(Bytes::from_static(LOGO_SVG)))
                .expect("Failed to build logo response");
            resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
        }

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

        ("POST", "/api/restart/llama-cpp") => {
            let resp = restart_llama_cpp(state.llama_cpp_restart_script.as_deref()).await;
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

        // ── System versions API ───────────────────────────────────────────────
        ("GET", "/api/versions") => {
            match handle_versions().await {
                Ok(resp) => resp,
                Err(e) => {
                    error!("Error getting versions: {}", e);
                    let resp = json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &ErrorResponse { error: format!("Internal error: {}", e) },
                    );
                    resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
                }
            }
        }

        ("POST", "/api/upgrade/llama-swap") => {
            let resp = upgrade_llama_swap().await;
            resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
        }

        ("POST", "/api/upgrade/manifest") => {
            let resp = upgrade_manifest().await;
            resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
        }

        ("POST", "/api/upgrade/toolbox") => {
            let resp = upgrade_toolbox().await;
            resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
        }

        // ── Review mode API ──────────────────────────────────────────────────
        ("GET", "/api/review-config") => {
            let config = &state.review_service.get_config();
            let resp = json_response(StatusCode::OK, config);
            resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
        }

        ("POST", "/api/review-config") => {
            match handle_update_review_config(req, &state.review_service).await {
                Ok(resp) => resp,
                Err(e) => {
                    error!("Error updating review config: {}", e);
                    let resp = json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &ErrorResponse { error: format!("Internal error: {}", e) },
                    );
                    resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
                }
            }
        }

        ("GET", "/api/models/llama-swap") => {
            match handle_llama_swap_models(&state.llama_swap_url).await {
                Ok(resp) => resp,
                Err(e) => {
                    error!("Error getting llama-swap models: {}", e);
                    let resp = json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &ErrorResponse { error: format!("Internal error: {}", e) },
                    );
                    resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!())))
                }
            }
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
            let safe_stream = SafeStream::new(stream, StreamFormat::OpenAi);
            let stream_body = StreamBody::new(safe_stream.map(|chunk| chunk.map(Frame::data)));
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
            let safe_stream = SafeStream::new(adapted, StreamFormat::Anthropic);
            let stream_body = StreamBody::new(safe_stream.map(|chunk| chunk.map(Frame::data)));
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

/// Restart the llama.cpp toolbox container.
async fn restart_llama_cpp(script_path: Option<&str>) -> Response<Full<Bytes>> {
    let script_path = match script_path {
        Some(p) => p,
        None => {
            return json_response(
                StatusCode::NOT_IMPLEMENTED,
                &ErrorResponse { error: "llama_cpp_restart_script not configured in brainrouter.yaml".to_string() },
            );
        }
    };
    info!("Restarting llama-vulkan-radv toolbox via {}...", script_path);
    let output = tokio::process::Command::new(script_path)
        .arg("llama-vulkan-radv")
        .output()
        .await;

    match output {
        Ok(out) if out.status.success() => {
            info!("Toolbox refreshed successfully");
            json_response(StatusCode::OK, &serde_json::json!({
                "status": "ok",
                "service": "llama-cpp",
                "message": "llama-vulkan-radv toolbox refreshed"
            }))
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            error!(%stderr, "Toolbox refresh failed");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Refresh failed: {}", stderr.trim()),
            })
        }
        Err(e) => {
            error!(error = %e, "Failed to exec refresh script");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Failed to exec refresh script {}: {}", script_path, e),
            })
        }
    }
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

/// Get current versions of llama-swap and llama-server (toolbox).
async fn handle_versions() -> Result<Response<UnsyncBoxBody<Bytes, anyhow::Error>>, anyhow::Error> {
    use tokio::process::Command;

    // 1. Get llama-swap version
    // First try the --version flag
    let swap_out = Command::new(home_bin("llama-swap"))
            .arg("--version")
        .output()
        .await;
    
    let mut swap_ver = match swap_out {
        Ok(out) if out.status.success() => {
            String::from_utf8_lossy(&out.stdout).trim().replace("version: ", "").to_string()
        }
        _ => String::new(),
    };

    // If --version failed (common in newer builds that removed the flag), 
    // try to extract version from build info via strings
    if swap_ver.is_empty() {
        let strings_out = Command::new("strings")
                    .arg(home_bin("llama-swap"))
            .output()
            .await;
            
        if let Ok(out) = strings_out {
            let s = String::from_utf8_lossy(&out.stdout);
            // Look for version tag like v0.1.5
            if let Some(line) = s.lines().find(|l| l.contains("github.com/mostlygeek/llama-swap") && (l.contains("v0.") || l.contains("v1."))) {
                // Example: mod github.com/mostlygeek/llama-swap v0.1.5 h1:...
                swap_ver = line.split_whitespace()
                    .find(|part| part.starts_with('v') && part.contains('.'))
                    .unwrap_or(line)
                    .to_string();
            }
        }
    }
    
    if swap_ver.is_empty() {
        swap_ver = "unknown".to_string();
    }

    // 2. Get llama.cpp version from toolbox container image.
    // Using 'podman run' directly on the image is more reliable than 'toolbox run'
    // which depends on a specific persistent container being in a startable state.
    //
    // IMPORTANT: use spawn() + kill_on_drop(true) rather than output() so that if
    // the 15-second timeout fires, the podman/conmon child processes are killed
    // immediately instead of becoming orphans attached to the service cgroup.
    const PODMAN_VERSION_TIMEOUT_SECS: u64 = 15;
    let toolbox_ver = {
        let child = Command::new("podman")
            .args(["run", "--rm", "docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv", "llama-server", "--version"])
            .kill_on_drop(true)  // ensures child is killed if the future is dropped on timeout
            .output();
        match tokio::time::timeout(std::time::Duration::from_secs(PODMAN_VERSION_TIMEOUT_SECS), child).await {
            Ok(Ok(out)) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);
                let combined = format!("{}{}", stdout, stderr);
                // Note: podman run may exit non-zero if no GPU is found (ggml_vulkan error)
                // but still output the version on stdout/stderr.
                if let Some(line) = combined.lines().find(|l| l.contains("version:")) {
                    line.replace("version:", "")
                        .replace("built with", "")
                        .trim()
                        .to_string()
                } else if let Some(line) = combined.lines().find(|l| l.contains('(') && l.contains(')') && !l.contains("Error")) {
                    line.replace("built with", "").trim().to_string()
                } else {
                    "unknown".to_string()
                }
            }
            Ok(Err(e)) => {
                error!(error = %e, "Failed to execute podman run for version check");
                "unknown".to_string()
            }
            Err(_elapsed) => {
                // kill_on_drop(true) above ensures the child is killed when this future is dropped.
                warn!(timeout_secs = PODMAN_VERSION_TIMEOUT_SECS, "podman run timed out during llama.cpp version check");
                "unknown".to_string()
            }
        }
    };

    // 3. Get toolbox container image version (OCI label, e.g. "43") and created date
    let (toolbox_image_ver, toolbox_image_created) = {
        let ver_out = Command::new("podman")
            .args(["image", "inspect",
                   "--format", "{{index .Labels \"org.opencontainers.image.version\"}}\t{{.Created}}",
                   "docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv"])
            .output()
            .await;
        match ver_out {
            Ok(o) => {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if let Some((ver, created)) = s.split_once('\t') {
                    // created looks like "2026-04-23 21:07:32.123 +0000 UTC" — take first 10 chars
                    let date = created.trim().get(..10).unwrap_or("").to_string();
                    (ver.trim().to_string(), date)
                } else {
                    (s, String::new())
                }
            }
            Err(_) => (String::new(), String::new()),
        }
    };

    // 4. Get locally running Manifest image digest (short SHA256)
    let manifest_ver = {
        let out = Command::new("docker")
            .args(["inspect", "--format",
                   "{{index .Config.Labels \"org.opencontainers.image.created\"}} {{slice .Id 7 19}}",
                   // slice .Id 7 19: full ID is "sha256:<hex>"; skip the 7-char prefix
                   // to get a 12-char short hash, e.g. "abc123456789"
                   "manifest-manifest-1"])
            .output()
            .await;
        match out {
            Ok(o) if o.status.success() => {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                // Format: "2026-04-15T22:47:22.44554494Z abc123456789"
                // Show as "Apr 15 · abc123"
                if let Some((date_part, hash)) = s.split_once(' ') {
                    let date = date_part.get(..10).unwrap_or(date_part);
                    format!("{} · {}", date, hash)
                } else {
                    s
                }
            }
            _ => "unknown".to_string(),
        }
    };

    // 5. Get remote Manifest digest to compare for update notification
    let (llama_swap_latest, manifest_latest, toolbox_latest) = tokio::join!(
        check_latest_llama_swap(),
        check_latest_manifest(),
        check_latest_toolbox(),
    );

    let data = serde_json::json!({
        "llama_swap": swap_ver,
        "llama_cpp": toolbox_ver,
        "toolbox_image_ver": toolbox_image_ver,
        "toolbox_image_created": toolbox_image_created,
        "manifest": manifest_ver,
        "llama_swap_latest": llama_swap_latest.unwrap_or_default(),
        "manifest_latest": manifest_latest.unwrap_or_default(),
        "toolbox_latest": toolbox_latest.unwrap_or_default(),
    });

    let resp = json_response(StatusCode::OK, &data);
    Ok(resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!()))))
}

async fn check_latest_llama_swap() -> Option<String> {
    let client = reqwest::Client::builder()
        .user_agent("brainrouter")
        .timeout(std::time::Duration::from_secs(3))
        .build().ok()?;
    let resp = client.get("https://api.github.com/repos/mostlygeek/llama-swap/releases/latest")
        .send().await.ok()?;
    let data: serde_json::Value = resp.json().await.ok()?;
    data.get("tag_name").and_then(|v| v.as_str()).map(|v| v.trim_start_matches('v').to_string())
}

/// Fetch the remote manifest Docker Hub tag's last-pushed date (used as a proxy for
/// "new version available"). Returns the ISO date string (first 10 chars) or None.
async fn check_latest_manifest() -> Option<String> {
    let client = reqwest::Client::builder()
        .user_agent("brainrouter")
        .timeout(std::time::Duration::from_secs(3))
        .build().ok()?;
    let resp = client.get("https://hub.docker.com/v2/repositories/manifestdotbuild/manifest/tags/latest")
        .send().await.ok()?;
    let data: serde_json::Value = resp.json().await.ok()?;
    // tag_last_pushed looks like "2026-04-23T23:27:27.60508Z"
    data.get("tag_last_pushed")
        .and_then(|v| v.as_str())
        .map(|s| s.get(..10).unwrap_or(s).to_string())
}

/// Fetch the remote toolbox image tag's last-pushed date.
async fn check_latest_toolbox() -> Option<String> {
    let client = reqwest::Client::builder()
        .user_agent("brainrouter")
        .timeout(std::time::Duration::from_secs(3))
        .build().ok()?;
    let resp = client.get("https://hub.docker.com/v2/repositories/kyuz0/amd-strix-halo-toolboxes/tags/vulkan-radv")
        .send().await.ok()?;
    let data: serde_json::Value = resp.json().await.ok()?;
    data.get("tag_last_pushed")
        .and_then(|v| v.as_str())
        .map(|s| s.get(..10).unwrap_or(s).to_string())
}

async fn upgrade_llama_swap() -> Response<Full<Bytes>> {
    info!("Upgrading llama-swap from GitHub releases...");

    let target_bin = home_bin("llama-swap");

    // 1. Fetch the latest release metadata from GitHub API
    let client = reqwest::Client::builder()
        .user_agent("brainrouter-upgrade/1.0")
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_default();

    let release_url = "https://api.github.com/repos/mostlygeek/llama-swap/releases/latest";
    let release: serde_json::Value = match client.get(release_url).send().await {
        Ok(r) => match r.json().await {
            Ok(j) => j,
            Err(e) => return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Failed to parse GitHub release JSON: {}", e),
            }),
        },
        Err(e) => return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
            error: format!("Failed to fetch latest release from GitHub: {}", e),
        }),
    };

    let tag = release.get("tag_name").and_then(|v| v.as_str()).unwrap_or("unknown");

    // 2. Find the linux_amd64 asset
    let download_url = release
        .get("assets")
        .and_then(|a| a.as_array())
        .and_then(|assets| {
            assets.iter().find(|a| {
                a.get("name")
                    .and_then(|n| n.as_str())
                    .map(|n| n.contains("linux_amd64") && n.ends_with(".tar.gz"))
                    .unwrap_or(false)
            })
        })
        .and_then(|a| a.get("browser_download_url"))
        .and_then(|u| u.as_str())
        .map(|s| s.to_string());

    let download_url = match download_url {
        Some(u) => u,
        None => return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
            error: "No linux_amd64 asset found in latest GitHub release".to_string(),
        }),
    };

    info!(tag, url = %download_url, "Downloading llama-swap");

    // 3. Download the tarball
    let tarball_bytes = match client.get(&download_url).send().await {
        Ok(r) => match r.bytes().await {
            Ok(b) => b,
            Err(e) => return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Failed to read tarball body: {}", e),
            }),
        },
        Err(e) => return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
            error: format!("Failed to download tarball: {}", e),
        }),
    };

    // 4. Extract the binary from the tarball via spawn_blocking (CPU-bound, sync)
    let target_bin_clone = target_bin.clone();
    let extract_result = tokio::task::spawn_blocking(move || -> Result<(), String> {
        use std::io::Read;
        let cursor = std::io::Cursor::new(tarball_bytes.as_ref());
        let gz = flate2::read::GzDecoder::new(cursor);
        let mut archive = tar::Archive::new(gz);
        for entry in archive.entries().map_err(|e| e.to_string())? {
            let mut entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path().map_err(|e| e.to_string())?;
            if path.file_name().and_then(|n| n.to_str()) == Some("llama-swap") {
                // Write to a temp file then atomically rename to avoid "Text file busy"
                let tmp = format!("{}.tmp", target_bin_clone);
                let mut file = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
                std::io::copy(&mut entry, &mut file).map_err(|e| e.to_string())?;
                drop(file);
                // Set executable bit
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))
                    .map_err(|e| e.to_string())?;
                std::fs::rename(&tmp, &target_bin_clone).map_err(|e| e.to_string())?;
                return Ok(());
            }
        }
        Err("'llama-swap' binary not found inside tarball".to_string())
    }).await;

    match extract_result {
        Err(join_err) => return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
            error: format!("Extract task panicked: {}", join_err),
        }),
        Ok(Err(e)) => return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
            error: format!("Failed to extract binary: {}", e),
        }),
        Ok(Ok(())) => {}
    }

    info!(tag, "llama-swap binary installed; restarting service");

    // 5. Restart the service (binary replaced atomically above so no stop needed)
    let restart = tokio::process::Command::new("systemctl")
        .args(["--user", "restart", "llama-swap"])
        .output()
        .await;

    match restart {
        Ok(out) if out.status.success() => json_response(StatusCode::OK, &serde_json::json!({
            "status": "ok",
            "message": format!("llama-swap upgraded to {} and restarted.", tag),
        })),
        _ => json_response(StatusCode::ACCEPTED, &serde_json::json!({
            "status": "partial",
            "message": format!("llama-swap upgraded to {} but restart failed — start manually.", tag),
        })),
    }
}

async fn upgrade_manifest() -> Response<Full<Bytes>> {
    info!("Upgrading Manifest via docker compose pull + up -d...");
    // Compose project lives at ~/ai/stack/manifest by convention.
    // Override with BRAINROUTER_MANIFEST_DIR env var if needed.
    let compose_dir = std::env::var("BRAINROUTER_MANIFEST_DIR")
        .unwrap_or_else(|_| home_path("ai/stack/manifest"));

    // Pull the latest image
    let pull = tokio::process::Command::new("docker")
        .args(["compose", "pull", "manifest"])
        .current_dir(&compose_dir)
        .output()
        .await;

    match pull {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            error!(%stderr, "docker compose pull failed");
            return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("docker compose pull failed: {}", stderr.trim()),
            });
        }
        Err(e) => {
            error!(error = %e, "Failed to exec docker compose pull");
            return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Failed to exec docker: {}", e),
            });
        }
    }

    // Recreate the container with the new image
    let up = tokio::process::Command::new("docker")
        .args(["compose", "up", "-d", "--force-recreate", "manifest"])
        .current_dir(&compose_dir)
        .output()
        .await;

    match up {
        Ok(out) if out.status.success() => {
            json_response(StatusCode::OK, &serde_json::json!({
                "status": "ok",
                "message": "Manifest upgraded and restarted successfully."
            }))
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            error!(%stderr, "docker compose up failed");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Pull succeeded but compose up failed: {}", stderr.trim()),
            })
        }
        Err(e) => {
            error!(error = %e, "Failed to exec docker compose up");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Failed to exec docker: {}", e),
            })
        }
    }
}

async fn upgrade_toolbox() -> Response<Full<Bytes>> {
    info!("Upgrading llama-vulkan-radv toolbox container...");
    let image = "docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv";
    let container = "llama-vulkan-radv";

    // 1. Pull the new image
    let pull = tokio::process::Command::new("podman")
        .args(["pull", image])
        .output()
        .await;

    match pull {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            error!(%stderr, "podman pull failed");
            return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("podman pull failed: {}", stderr.trim()),
            });
        }
        Err(e) => {
            error!(error = %e, "Failed to exec podman pull");
            return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Failed to exec podman: {}", e),
            });
        }
    }

    // 2. Remove the existing toolbox container (force, it may be running)
    let _ = tokio::process::Command::new("toolbox")
        .args(["rm", "--force", container])
        .output()
        .await;

    // 3. Recreate the toolbox container from the fresh image
    let create = tokio::process::Command::new("toolbox")
        .args(["create", "--image", image, container])
        .output()
        .await;

    match create {
        Ok(out) if out.status.success() => {
            json_response(StatusCode::OK, &serde_json::json!({
                "status": "ok",
                "message": "Toolbox container recreated with latest image."
            }))
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            error!(%stderr, "toolbox create failed");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Pull succeeded but toolbox create failed: {}", stderr.trim()),
            })
        }
        Err(e) => {
            error!(error = %e, "Failed to exec toolbox create");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Failed to exec toolbox: {}", e),
            })
        }
    }
}
async fn handle_update_review_config(
    req: Request<Incoming>,
    service: &ReviewService,
) -> Result<Response<UnsyncBoxBody<Bytes, anyhow::Error>>, anyhow::Error> {
    let body_bytes = req.collect().await?.to_bytes();
    let update: crate::config::ReviewConfig = serde_json::from_slice(&body_bytes)?;
    
    service.update_config(update).await;
    
    let resp = json_response(StatusCode::OK, &serde_json::json!({ "status": "ok" }));
    Ok(resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!()))))
}

async fn handle_llama_swap_models(
    llama_swap_url: &str,
) -> Result<Response<UnsyncBoxBody<Bytes, anyhow::Error>>, anyhow::Error> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()?;
        
    let url = format!("{}/v1/models", llama_swap_url);
    let resp = client.get(&url).send().await?;
    let data: serde_json::Value = resp.json().await?;
    
    let resp = json_response(StatusCode::OK, &data);
    Ok(resp.map(|body| BodyExt::boxed_unsync(body.map_err(|_| unreachable!()))))
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
                    // Runs in spawn_blocking because peer_cwd scans /proc synchronously.
                    let conn_cwd = tokio::task::spawn_blocking(move || peer_cwd(&addr).unwrap_or_default()).await.unwrap_or_default();
                    tokio::spawn(async move {
                        if let Err(e) = http1::Builder::new()
                            .serve_connection(io, service_fn(move |req| handle_request(req, state.clone(), conn_cwd.clone(), addr)))
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
                    // Resolve the cwd of the connecting process via UDS peer credentials.
                    // cwd_from_pid is a blocking readlink; run it off the async executor.
                    let conn_cwd = if let Ok(cred) = stream.peer_cred() {
                        if let Some(pid) = cred.pid() {
                            tokio::task::spawn_blocking(move || crate::peer_cwd::cwd_from_pid(pid).unwrap_or_default())
                                .await
                                .unwrap_or_default()
                        } else {
                            String::new()
                        }
                    } else {
                        String::new()
                    };
                    let io = TokioIo::new(stream);
                    let state = uds_state.clone();
                    let dummy_addr: SocketAddr = "0.0.0.0:0".parse().unwrap();
                    tokio::spawn(async move {
                        if let Err(e) = http1::Builder::new()
                            .serve_connection(io, service_fn(move |req| handle_request(req, state.clone(), conn_cwd.clone(), dummy_addr)))
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
