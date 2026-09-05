// ── Runtime home-dir helpers ─────────────────────────────────────────────────
//
// The daemon may run under any user. Never hardcode /home/<user>.
// These helpers resolve paths relative to $HOME at runtime.

fn home_dir() -> String {
    std::env::var("HOME").unwrap_or_else(|_| {
        tracing::warn!("$HOME is not set; upgrade and version-check paths will resolve under /root");
        "/root".to_string()
    })
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

/// Return the list of editable config/agent files: (display_name, path, exists).
fn config_file_list(home: &str, config_path: &std::path::Path, llama_swap_config_path: &std::path::Path) -> Vec<(&'static str, PathBuf, bool)> {
    let config_abs = std::fs::canonicalize(config_path)
        .unwrap_or_else(|_| config_path.to_path_buf());
    let ls_abs = std::fs::canonicalize(llama_swap_config_path)
        .unwrap_or_else(|_| llama_swap_config_path.to_path_buf());
    let entries: Vec<(&str, PathBuf)> = vec![
        ("Agent System Prompt", PathBuf::from(format!("{}/.omp/agent/APPEND_SYSTEM.md", home))),
        ("Review Prompt Template", PathBuf::from(format!("{}/.omp/agent/LLAMACPP.md", home))),
        ("Local System Prompt Override", PathBuf::from(format!("{}/.omp/agent/APPEND_SYSTEM.local.md", home))),
        ("Model Aliases", PathBuf::from(format!("{}/.config/omp-bridge/config.yaml", home))),
        ("brainrouter.yaml", config_abs),
        ("llama-swap config", ls_abs),
    ];
    entries.into_iter().map(|(name, path)| {
        let exists = path.exists();
        (name, path, exists)
    }).collect()
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
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering as AtomicOrdering};
use tracing::{debug, error, info, warn};
use std::sync::LazyLock;

/// Shared HTTP client for lightweight polling and version checks.
/// Each call site sets its own `.timeout()` on the request builder.
static VERSION_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .user_agent("brainrouter")
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .expect("Failed to build HTTP client")
});
use crate::anthropic::{anthropic_to_openai, AnthropicMessagesRequest, AnthropicSseAdapter};
use crate::escalation;
use crate::peer_cwd::peer_cwd;
use crate::review::ReviewService;
use crate::router::Router;
use crate::routing_events::RoutingEvents;
use crate::session::SessionManager;
use crate::types::ChatCompletionRequest;
use crate::provider::ProviderResponse;
use crate::stream::{DeferredStream, SafeStream, StreamFormat, KEEPALIVE_INTERVAL};
use crate::inflight::SniffStream;

// Unified dashboard — embedded at compile time so the binary is self-contained.
const MAIN_DASHBOARD_HTML: &str = include_str!("escalation/templates/main_dashboard.html");
const FAVICON_SVG: &[u8] = include_bytes!("escalation/templates/favicon.svg");
const LOGO_SVG: &[u8] = include_bytes!("escalation/templates/logo.svg");

/// Maximum time the DeferredStream will wait for a provider stream before
/// giving up.  Aligned with the TTFT_TIMEOUT in router.rs (600 s) so model
/// loading + prefill for large local models (qwen3-27b-mtp) can complete.
const DEFERRED_STREAM_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

/// Shared state passed to all request handlers
pub struct AppState {
    pub router: Arc<Router>,
    pub session_manager: Arc<SessionManager>,
    pub review_service: Arc<ReviewService>,
    pub routing_events: Arc<RoutingEvents>,
    /// llama-swap root URL (without /v1 suffix) for status polling.
    pub llama_swap_url: String,
    /// Manifest base URL for health checking.
    pub manifest_url: String,
    /// Whether the cloud backend (Manifest) is enabled from config.
    pub manifest_enabled: bool,
    /// Bridge transport manager (Discord, Signal status tracking).
    pub bridge_manager: Arc<crate::bridge::BridgeManager>,
    /// Path to brainrouter's own config file (used by the config UI and
    /// the self-restart endpoint).
    pub config_path: PathBuf,
    /// Path to llama-swap's config file (used by the restart-local-stack
    /// endpoint and the context-size setter).
    pub llama_swap_config_path: PathBuf,
    /// Our own TCP listen address (for the "open dashboard" button).
    pub tcp_addr: String,
    /// Runtime control of the Bonsai classifier llama-server (dashboard
    /// start/stop). Also read by the classifier for its enabled flag.
    pub bonsai: Arc<crate::bonsai_server::BonsaiControl>,
    /// Runtime routing mode: 0 = auto (Bonsai), 1 = cloud, 2 = local.
    /// Read by the proxy handlers to force-rewrite `request.model`.
    pub routing_mode: std::sync::Arc<AtomicU8>,
    /// Cached version/upgrade-check data (refreshed every 30 min).
    pub versions_cache: std::sync::Arc<tokio::sync::watch::Receiver<serde_json::Value>>,
    /// Runtime nudge master switch (initialized from `llama_swap.nudge.enabled`).
    pub nudge_enabled: Arc<AtomicBool>,
    /// Runtime nudge tier override: 0 = auto (Bonsai), 1 = light, 2 = deep.
    pub nudge_tier: Arc<AtomicU8>,
    /// Nudge model key from config (static; runtime changes use the config UI).
    pub nudge_model_key: Option<String>,
    pub nudge_budgets: crate::config::NudgeBudgets,
    /// Runtime prompt-rewrite toggle (default on). When off, local routes
    /// forward the incoming prompt untouched.
    pub prompt_rewrite: Arc<AtomicBool>,
    /// In-flight request registry (dashboard tracking + cancel).
    pub inflight: Arc<crate::inflight::InflightRegistry>,
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

/// Convert a `Response<Full<Bytes>>` into the handler return type.
/// `Full<Bytes>` is infallible, so the error mapping is a compile-time proof.
fn into_unsync(resp: Response<Full<Bytes>>) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    resp.map(|body| body.map_err(|e: Infallible| match e {}).boxed_unsync())
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
    let is_destructive = path.starts_with("/api/restart/") || path.starts_with("/api/upgrade/")
        || (method == "POST" && (
            path == "/api/config" || path == "/api/llama-swap-config"
            || path == "/api/open-editor" || path == "/api/models/sync-omp"
            || path == "/api/routing-mode" || path == "/api/review-config"
            || path == "/api/bridges/toggle"
            || path == "/api/bonsai/toggle" || path == "/api/models/flush"
            || path == "/api/nudge" || path == "/api/prompt-rewrite"
            // Review API is destructive too: it spawns reviews (arbitrary
            // project paths read into cloud prompts) and can approve/resolve
            // sessions. Gate it like the rest.
            || path.starts_with("/review/api/")
            || path == "/api/inflight/cancel"
        ));

    if is_destructive {
        if !is_local {
            error!("Blocking destructive API request from non-local peer: {}", peer_addr);
            let resp = json_response(
                StatusCode::FORBIDDEN,
                &ErrorResponse { error: "Destructive APIs only allowed from localhost".to_string() },
            );
            return Ok(into_unsync(resp));
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
            return Ok(into_unsync(resp));
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
            into_unsync(resp)
        }

        ("GET", "/v1/models") => {
            let mut data = vec![
                ModelObject { id: "auto".to_string(), object: "model", created: 0, owned_by: "brainrouter".to_string() },
                ModelObject { id: "local".to_string(), object: "model", created: 0, owned_by: "brainrouter".to_string() },
                ModelObject { id: "cloud".to_string(), object: "model", created: 0, owned_by: "brainrouter".to_string() },
            ];
            // Fetch llama-swap models and append them
            let ls_url = format!("{}/v1/models", &state.llama_swap_url);
            if let Ok(resp) = VERSION_CLIENT.get(&ls_url)
                .timeout(std::time::Duration::from_secs(2))
                .send().await
            {
                if let Ok(body) = resp.json::<serde_json::Value>().await {
                    if let Some(arr) = body.get("data").and_then(|d| d.as_array()) {
                        let skip = ["auto", "local", "cloud"];
                        for m in arr {
                            if let Some(id) = m.get("id").and_then(|v| v.as_str()) {
                                if !skip.contains(&id) {
                                    data.push(ModelObject {
                                        id: id.to_string(),
                                        object: "model",
                                        created: 0,
                                        owned_by: "llama-swap".to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
            let models = ModelListResponse { object: "list", data };
            let resp = json_response(StatusCode::OK, &models);
            into_unsync(resp)
        }

        ("POST", "/v1/chat/completions") => {
            let session_id = extract_session_id(req.headers());
            let user_agent = req.headers().get("user-agent").and_then(|v| v.to_str().ok()).unwrap_or("").trim().to_string();
            match handle_chat_completion(req, state, cwd, session_id, user_agent, peer_addr).await {
                Ok(resp) => resp,
                Err(e) => {
                    error!("Error handling chat completion: {}", e);
                    let resp = json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &ErrorResponse { error: format!("Internal error: {}", e) },
                    );
                    into_unsync(resp)
                }
            }
        }

        ("POST", "/v1/messages") => {
            let session_id = extract_session_id(req.headers());
            let user_agent = req.headers().get("user-agent").and_then(|v| v.to_str().ok()).unwrap_or("").trim().to_string();
            match handle_anthropic_messages(req, state, cwd, session_id, user_agent, peer_addr).await {
                Ok(resp) => resp,
                Err(e) => {
                    error!("Error handling Anthropic messages: {}", e);
                    let resp = json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &ErrorResponse { error: format!("Internal error: {}", e) },
                    );
                    into_unsync(resp)
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
            into_unsync(resp)
        }

        // ── Unified dashboard ──────────────────────────────────────────────────
        ("GET", "/dashboard") => {
            let resp = Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "text/html; charset=utf-8")
                .header("cache-control", "no-store")
                .body(Full::new(Bytes::from_static(MAIN_DASHBOARD_HTML.as_bytes())))
                .expect("Failed to build HTML response");
            into_unsync(resp)
        }

        ("GET", "/favicon.ico") | ("GET", "/favicon.svg") | ("GET", "/favicon.png") => {
            let resp = Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "image/svg+xml")
                .body(Full::new(Bytes::from_static(FAVICON_SVG)))
                .expect("Failed to build favicon response");
            into_unsync(resp)
        }

        ("GET", "/logo.svg") => {
            let resp = Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "image/svg+xml")
                .body(Full::new(Bytes::from_static(LOGO_SVG)))
                .expect("Failed to build logo response");
            into_unsync(resp)
        }

        // ── In-flight request tracking API ────────────────────────────────────────
        ("GET", "/api/inflight") => {
            let resp = json_response(StatusCode::OK, &state.inflight.json());
            into_unsync(resp)
        }

        ("POST", "/api/inflight/cancel") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let val: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();
            let ok = match val.get("id").and_then(|v| v.as_u64()) {
                Some(id) => state.inflight.cancel(id),
                None => false,
            };
            if !ok {
                let resp = json_response(
                    StatusCode::NOT_FOUND,
                    &ErrorResponse { error: "Unknown in-flight request id".to_string() },
                );
                into_unsync(resp)
            } else {
                let resp = json_response(StatusCode::OK, &serde_json::json!({"cancelled": true}));
                into_unsync(resp)
            }
        }

        ("GET", "/api/omp-sessions") => {
            let resp = json_response(StatusCode::OK, &omp_sessions());
            into_unsync(resp)
        }

        // ── Routing events API ─────────────────────────────────────────────────
        ("GET", "/api/routing-events") => {
            let resp = json_response(StatusCode::OK, &state.routing_events.get_all_as_response());
            into_unsync(resp)
        }

        ("GET", "/api/routing-stats") => {
            let resp = json_response(StatusCode::OK, &state.routing_events.get_stats());
            into_unsync(resp)
        }

        // ── Inference status API (polls llama-swap + llama-server) ────────────
        ("GET", "/api/inference-status") => {
            let resp = inference_status(&state.router.inference_tracker, &state.llama_swap_url).await;
            into_unsync(resp)
        }

        // ── Service health API ────────────────────────────────────────────────
        ("GET", "/api/service-health") => {
            let resp = service_health(
                &state.llama_swap_url,
                &state.manifest_url,
                &state.routing_events,
                state.manifest_enabled,
            )
            .await;
            into_unsync(resp)
        }

        // ── Service restart API ────────────────────────────────────────────────
        ("POST", "/api/restart/llama-swap") => {
            let resp = restart_service("llama-swap").await;
            into_unsync(resp)
        }

        ("POST", "/api/restart/llama-cpp") => {
            let resp = restart_llama_cpp().await;
            into_unsync(resp)
        }

        ("POST", "/api/restart/manifest") => {
            let resp = restart_service("manifest").await;
            into_unsync(resp)
        }

        ("POST", "/api/restart/brainrouter") => {
            // Brainrouter restarts itself: send a 200 immediately, then
            // schedule the actual restart after a short delay so the HTTP
            // response reaches the client before this process is killed.
            tokio::spawn(async {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                let _ = tokio::process::Command::new("systemctl")
                    .args(["--user", "restart", "brainrouter"])
                    .output()
                    .await;
            });
            let resp = json_response(StatusCode::OK, &serde_json::json!({
                "status": "ok",
                "service": "brainrouter",
                "message": "brainrouter restarting"
            }));
            into_unsync(resp)
        }

        // ── System versions API ───────────────────────────────────────────────
        ("GET", "/api/versions") => {
            let data = state.versions_cache.borrow().clone();
            let resp = json_response(StatusCode::OK, &data);
            into_unsync(resp)
        }

        ("POST", "/api/upgrade/llama-swap") => {
            let resp = upgrade_llama_swap().await;
            into_unsync(resp)
        }

        ("POST", "/api/upgrade/manifest") => {
            let resp = upgrade_manifest().await;
            into_unsync(resp)
        }

        ("POST", "/api/upgrade/toolbox") => {
            let resp = upgrade_toolbox(
                "llama-vulkan-radv",
                "docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv",
            )
            .await;
            into_unsync(resp)
        }

        // Per-container upgrade: /api/upgrade/toolbox/<container-name>. The
        // image is resolved from whatever the container currently runs.
        ("POST", p) if p.starts_with("/api/upgrade/toolbox/") => {
            let name = p.trim_start_matches("/api/upgrade/toolbox/").trim_end_matches('/');
            let resp = if name.is_empty() {
                json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
                    error: "Missing toolbox container name".into(),
                })
            } else if let Some(image) = toolbox_container_image(name).await {
                upgrade_toolbox(name, &image).await
            } else {
                json_response(StatusCode::NOT_FOUND, &ErrorResponse {
                    error: format!("No such toolbox container: {}", name),
                })
            };
            into_unsync(resp)
        }

        // ── Review mode API ──────────────────────────────────────────────────
        ("GET", "/api/review-config") => {
            let config = &state.review_service.get_config();
            let resp = json_response(StatusCode::OK, config);
            into_unsync(resp)
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
                    into_unsync(resp)
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
                    into_unsync(resp)
                }
            }
        }

        ("POST", "/api/models/sync-omp") => {
            match sync_omp_models(&state.llama_swap_url, &state.tcp_addr).await {
                Ok(count) => {
                    let resp = json_response(StatusCode::OK, &serde_json::json!({
                        "synced": true,
                        "model_count": count
                    }));
                    into_unsync(resp)
                }
                Err(e) => {
                    error!("Error syncing OMP models: {}", e);
                    let resp = json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &ErrorResponse { error: format!("Sync failed: {}", e) },
                    );
                    into_unsync(resp)
                }
            }
        }

        // \_\_ Bridge status API \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
        ("GET", "/api/bridge-status") => {
            let status = state.bridge_manager.status();
            let resp = json_response(StatusCode::OK, &status);
            into_unsync(resp)
        }

        ("POST", "/api/bridges/toggle") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let val: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();
            let bridge = val.get("bridge").and_then(|v| v.as_str()).unwrap_or("");
            let enabled = val.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true);
            match bridge {
                "discord" => {
                    state.bridge_manager.discord_enabled
                        .store(enabled, std::sync::atomic::Ordering::Relaxed);
                }
                "signal" => {
                    state.bridge_manager.signal_enabled
                        .store(enabled, std::sync::atomic::Ordering::Relaxed);
                }
                _ => {
                    let resp = json_response(
                        StatusCode::BAD_REQUEST,
                        &ErrorResponse { error: format!("Unknown bridge: {bridge}") },
                    );
                    return Ok(into_unsync(resp));
                }
            }
            let resp = json_response(StatusCode::OK, &serde_json::json!({ "ok": true, "bridge": bridge, "enabled": enabled }));
            into_unsync(resp)
        }

        // ── Routing mode override API ─────────────────────────────────────────
        ("GET", "/api/routing-mode") => {
            let mode = match state.routing_mode.load(AtomicOrdering::Relaxed) {
                1 => "cloud",
                2 => "local",
                _ => "auto",
            };
            let resp = json_response(StatusCode::OK, &serde_json::json!({ "mode": mode }));
            into_unsync(resp)
        }

        ("POST", "/api/routing-mode") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let val: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();
            let mode_str = val.get("mode").and_then(|v| v.as_str()).unwrap_or("auto");
            let code: u8 = match mode_str { "cloud" => 1, "local" => 2, _ => 0 };
            state.routing_mode.store(code, AtomicOrdering::Relaxed);
            let resp = json_response(StatusCode::OK, &serde_json::json!({ "mode": mode_str }));
            into_unsync(resp)
        }

        // ── Bonsai classifier server API ────────────────────────────────────
        ("GET", "/api/bonsai") => {
            let enabled = state.bonsai.is_running();
            let healthy = if enabled { state.bonsai.healthy().await } else { false };
            let resp = json_response(StatusCode::OK, &serde_json::json!({
                "enabled": enabled,
                "healthy": healthy,
                "url": state.bonsai.url(),
            }));
            into_unsync(resp)
        }

        ("POST", "/api/bonsai/toggle") => {
            match state.bonsai.toggle().await {
                Ok(running) => {
                    // Prompt rewrite is coupled to the classifier: drop it when
                    // Bonsai goes off so the invariant "rewrite on => Bonsai on"
                    // holds server-side, not just in the UI.
                    if !running {
                        state.prompt_rewrite.store(false, AtomicOrdering::Relaxed);
                    }
                    let resp = json_response(StatusCode::OK, &serde_json::json!({
                        "enabled": running,
                        "message": if running {
                            "Bonsai classifier server started"
                        } else {
                            "Bonsai classifier server stopped — auto routing defaults to cloud until re-enabled"
                        },
                    }));
                    into_unsync(resp)
                }
                Err(e) => {
                    error!(error = %e, "Bonsai toggle failed");
                    let resp = json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &ErrorResponse { error: format!("Bonsai toggle failed: {}", e) },
                    );
                    into_unsync(resp)
                }
            }
        }

        // ── Nudge (thinking budget) API ─────────────────────────────────────
        ("GET", "/api/nudge") => {
            let tier = match state.nudge_tier.load(AtomicOrdering::Relaxed) {
                1 => "light",
                2 => "deep",
                _ => "auto",
            };
            let resp = json_response(
                StatusCode::OK,
                &serde_json::json!({
                    "enabled": state.nudge_enabled.load(AtomicOrdering::Relaxed),
                    "tier": tier,
                    "model_key": state.nudge_model_key,
                    "budgets": {
                        "light": state.nudge_budgets.light,
                        "deep": state.nudge_budgets.deep,
                    },
                }),
            );
            into_unsync(resp)
        }

        ("POST", "/api/nudge") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let val: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();
            if let Some(enabled) = val.get("enabled").and_then(|v| v.as_bool()) {
                state.nudge_enabled.store(enabled, AtomicOrdering::Relaxed);
            }
            if let Some(tier) = val.get("tier").and_then(|v| v.as_str()) {
                let t = match tier {
                    // "local" is the legacy spelling of the light tier.
                    "light" | "local" => 1,
                    "deep" => 2,
                    _ => 0,
                };
                state.nudge_tier.store(t, AtomicOrdering::Relaxed);
            }
            let resp = json_response(StatusCode::OK, &serde_json::json!({
                "enabled": state.nudge_enabled.load(AtomicOrdering::Relaxed),
                "tier": match state.nudge_tier.load(AtomicOrdering::Relaxed) {
                    1 => "light",
                    2 => "deep",
                    _ => "auto",
                },
            }));
            into_unsync(resp)
        }

        // ── Prompt-rewrite toggle API (local pass-through mode) ─────────────
        // Independent of Bonsai: rewrite_for_local is a standalone local prompt
        // swap (applied to managed auto/local routes when enabled). Off →
        // forward the incoming prompt untouched.
        ("GET", "/api/prompt-rewrite") => {
            let resp = json_response(
                StatusCode::OK,
                &serde_json::json!({
                    "enabled": state.prompt_rewrite.load(AtomicOrdering::Relaxed),
                }),
            );
            into_unsync(resp)
        }

        ("POST", "/api/prompt-rewrite") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let val: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();
            // Rewriting the local system prompt is only meaningful with the
            // classifier driving auto/local routing. Refuse to turn it on while
            // Bonsai is off; turning off is always allowed.
            let rejected = match val.get("enabled").and_then(|v| v.as_bool()) {
                Some(true) if !(state.bonsai.is_running() && state.bonsai.healthy().await) => true,
                Some(enabled) => {
                    state.prompt_rewrite.store(enabled, AtomicOrdering::Relaxed);
                    false
                }
                None => false,
            };
            if rejected {
                into_unsync(json_response(
                    StatusCode::CONFLICT,
                    &ErrorResponse {
                        error: "Prompt rewrite requires the Bonsai classifier to be on".into(),
                    },
                ))
            } else {
                into_unsync(json_response(StatusCode::OK, &serde_json::json!({
                    "enabled": state.prompt_rewrite.load(AtomicOrdering::Relaxed),
                })))
            }
        }

        // ── Toolboxes API (all llama-* toolbox containers) ──────────────────
        ("GET", "/api/toolboxes") => {
            let resp = toolboxes_list().await;
            into_unsync(resp)
        }


        // ── Flush models API (free VRAM) ────────────────────────────────────
        ("POST", "/api/models/flush") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let val: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();
            let reload = val.get("reload").and_then(|v| v.as_bool()).unwrap_or(false);
            let local_models: Vec<String> = state.router.local_models().to_vec();
            let resp = flush_models(&state.llama_swap_url, reload, &local_models).await;
            into_unsync(resp)
        }


 // __ Config API _______________________________________________________
        ("GET", "/api/config") => {
            match std::fs::read_to_string(&state.config_path) {
                Ok(yaml) => {
                    
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("Content-Type", "text/yaml; charset=utf-8")
                        .body(Full::new(Bytes::from(yaml)).map_err(|e| anyhow::anyhow!(e)).boxed_unsync())
                        .unwrap()
                }
                Err(e) => {
                    let resp = json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &ErrorResponse { error: format!("Failed to read config: {}", e) },
                    );
                    into_unsync(resp)
                }
            }
        }

        ("POST", "/api/config") => {
            let body_bytes = req.collect().await
                .map(|c| c.to_bytes())
                .unwrap_or_default();
            if body_bytes.len() > 1_048_576 {
                let resp = json_response(
                    StatusCode::BAD_REQUEST,
                    &ErrorResponse { error: "Request body too large (max 1MB)".to_string() },
                );
                into_unsync(resp)
            } else {
                let body = String::from_utf8_lossy(&body_bytes).to_string();
                // Validate against the real config struct, not just generic YAML.
                match serde_yaml::from_str::<crate::config::BrainrouterConfig>(&body) {
                    Err(e) => {
                        let resp = json_response(
                            StatusCode::BAD_REQUEST,
                            &ErrorResponse { error: format!("Invalid config: {}", e) },
                        );
                        into_unsync(resp)
                    }
                    Ok(_) => {
                        // Atomic write: write to .tmp then rename.
                        let tmp_path = state.config_path.with_extension("yaml.tmp");
                        let write_result = std::fs::write(&tmp_path, body.as_bytes())
                            .and_then(|_| std::fs::rename(&tmp_path, &state.config_path));
                        if let Err(e) = write_result {
                            let _ = std::fs::remove_file(&tmp_path);
                            let resp = json_response(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                &ErrorResponse { error: format!("Failed to write config: {}", e) },
                            );
                            into_unsync(resp)
                        } else {
                            let resp = json_response(StatusCode::OK, &serde_json::json!({"status": "ok"}));
                            into_unsync(resp)
                        }
                    }
                }
            }
        }

        ("GET", "/api/llama-swap-config") => {
            match std::fs::read_to_string(&state.llama_swap_config_path) {
                Ok(yaml) => {
                    
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("Content-Type", "text/yaml; charset=utf-8")
                        .body(Full::new(Bytes::from(yaml)).map_err(|e| anyhow::anyhow!(e)).boxed_unsync())
                        .unwrap()
                }
                Err(e) => {
                    let resp = json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &ErrorResponse { error: format!("Failed to read llama-swap config: {}", e) },
                    );
                    into_unsync(resp)
                }
            }
        }

        ("POST", "/api/llama-swap-config") => {
            let body_bytes = req.collect().await
                .map(|c| c.to_bytes())
                .unwrap_or_default();
            if body_bytes.len() > 1_048_576 {
                let resp = json_response(
                    StatusCode::BAD_REQUEST,
                    &ErrorResponse { error: "Request body too large (max 1MB)".to_string() },
                );
                into_unsync(resp)
            } else {
                let body = String::from_utf8_lossy(&body_bytes).to_string();
                // Validate it's at least valid YAML.
                match serde_yaml::from_str::<serde_yaml::Value>(&body) {
                    Err(e) => {
                        let resp = json_response(
                            StatusCode::BAD_REQUEST,
                            &ErrorResponse { error: format!("Invalid YAML: {}", e) },
                        );
                        into_unsync(resp)
                    }
                    Ok(_) => {
                        let tmp_path = state.llama_swap_config_path.with_extension("yaml.tmp");
                        let write_result = std::fs::write(&tmp_path, body.as_bytes())
                            .and_then(|_| std::fs::rename(&tmp_path, &state.llama_swap_config_path));
                        if let Err(e) = write_result {
                            let _ = std::fs::remove_file(&tmp_path);
                            let resp = json_response(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                &ErrorResponse { error: format!("Failed to write llama-swap config: {}", e) },
                            );
                            into_unsync(resp)
                        } else {
                            let resp = json_response(StatusCode::OK, &serde_json::json!({"status": "ok"}));
                            into_unsync(resp)
                        }
                    }
                }
            }
        }

        ("POST", "/api/open-editor") => {
            let body_bytes = req.collect().await
                .map(|c| c.to_bytes())
                .unwrap_or_default();
            let parsed: Result<serde_json::Value, _> = serde_json::from_slice(&body_bytes);
            match parsed {
                Err(e) => {
                    let resp = json_response(
                        StatusCode::BAD_REQUEST,
                        &ErrorResponse { error: format!("Invalid JSON: {}", e) },
                    );
                    into_unsync(resp)
                }
                Ok(val) => {
                    let file_path = val.get("path").and_then(|v| v.as_str()).unwrap_or_default();
                    if file_path.is_empty() {
                        let resp = json_response(
                            StatusCode::BAD_REQUEST,
                            &ErrorResponse { error: "Missing 'path' field".to_string() },
                        );
                        into_unsync(resp)
                    } else {
                        // Allowlist: only files from the config-files list can be opened.
                        let home = home_dir();
                        let allowed = config_file_list(&home, &state.config_path, &state.llama_swap_config_path);
                        let canonical = std::fs::canonicalize(file_path).unwrap_or_default();
                        let is_allowed = allowed.iter().any(|(_, p, exists)| {
                            *exists && std::fs::canonicalize(p).ok().as_ref() == Some(&canonical)
                        });
                        if !is_allowed {
                            let resp = json_response(
                                StatusCode::FORBIDDEN,
                                &ErrorResponse { error: "Path not in the allowed file list".to_string() },
                            );
                            into_unsync(resp)
                        } else {
                            // Headless-friendly: fail loudly instead of returning
                            // "ok" with nothing opened (no silent `let _ =`).
                            use std::process::Stdio;
                            match tokio::process::Command::new("xdg-open")
                                .arg(file_path)
                                .stdin(Stdio::null())
                                .stdout(Stdio::null())
                                .stderr(Stdio::null())
                                .spawn()
                            {
                                Ok(_) => {
                                    let resp = json_response(StatusCode::OK, &serde_json::json!({"status": "ok"}));
                                    into_unsync(resp)
                                }
                                Err(e) => {
                                    let resp = json_response(
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        &ErrorResponse {
                                            error: format!("xdg-open is not available: {}", e),
                                        },
                                    );
                                    into_unsync(resp)
                                }
                            }
                        }
                    }
                }
            }
        }

        ("GET", "/api/config-files") => {
            let home = home_dir();
            let files: Vec<serde_json::Value> = config_file_list(&home, &state.config_path, &state.llama_swap_config_path)
                .into_iter()
                .map(|(name, path, exists)| serde_json::json!({
                    "name": name,
                    "path": path.to_string_lossy(),
                    "exists": exists
                }))
                .collect();
            let resp = json_response(StatusCode::OK, &files);
            into_unsync(resp)
        }

        _ => {
            let resp = json_response(
                StatusCode::NOT_FOUND,
                &ErrorResponse { error: format!("Not found: {} {}", method, path) },
            );
            into_unsync(resp)
        }
    };

    Ok(response)
}

/// Extract a client-provided conversation/session id from request headers.
///
/// The dashboard uses this (plus a stable hash of the conversation prefix)
/// to group events per conversation so each one renders as a single card.
/// If a client ever adds a session header, this picks it up with no code change.
fn extract_session_id(headers: &hyper::http::HeaderMap) -> Option<String> {
    const CANDIDATES: [&str; 6] = [
        "x-omp-session",
        "x-session-id",
        "x-conv-id",
        "x-conversation-id",
        "x-client-session",
        "x-request-conv",
    ];
    for name in CANDIDATES {
        if let Some(v) = headers.get(name) {
            let s = v.to_str().ok()?.trim().to_string();
            if !s.is_empty() {
                return Some(s);
            }
        }
    }
    None
}
/// Scan OMP session directories for their titles, used by the sankey
/// "SESSION" column. Each session is a directory under ~/.omp/agent/sessions
/// named by the cwd slug; the first JSONL line of any file carries
/// {"type":"title",...}. Returns {home, sessions:[{slug,title,updated_ms}]}.
fn omp_sessions() -> serde_json::Value {
    let home = std::env::var("HOME").unwrap_or_default();
    let base = std::path::Path::new(&home).join(".omp/agent/sessions");
    let mut sessions = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&base) {
        for dir in entries.flatten() {
            let slug = dir.file_name().to_string_lossy().into_owned();
            let mut title = String::new();
            let mut updated_ms = 0u64;
            let mut newest_mt: Option<std::time::SystemTime> = None;
            if let Ok(files) = std::fs::read_dir(dir.path()) {
                for f in files.flatten() {
                    let p = f.path();
                    if p.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                        continue;
                    }
                    if title.is_empty() {
                        if let Ok(content) = std::fs::read_to_string(&p) {
                            if let Some(first) = content.lines().next() {
                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(first) {
                                    if v.get("type").and_then(|t| t.as_str()) == Some("title") {
                                        title = v.get("title").and_then(|t| t.as_str())
                                            .unwrap_or("").to_string();
                                    }
                                }
                            }
                        }
                    }
                    if let Ok(md) = std::fs::metadata(&p) {
                        if let Ok(mt) = md.modified() {
                            if newest_mt.map_or(true, |n| mt > n) {
                                newest_mt = Some(mt);
                            }
                        }
                    }
                }
            }
            if let Some(mt) = newest_mt {
                updated_ms = mt.duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64).unwrap_or(0);
            }
            sessions.push(serde_json::json!({
                "slug": slug,
                "title": title,
                "updated_ms": updated_ms,
            }));
        }
    }
    serde_json::json!({"home": base.to_string_lossy().into_owned(), "sessions": sessions})
}

/// Handle POST /v1/chat/completions
async fn handle_chat_completion(
    req: Request<Incoming>,
    state: Arc<AppState>,
    cwd: String,
    session_id: Option<String>,
    user_agent: String,
    peer_addr: SocketAddr,
) -> Result<Response<UnsyncBoxBody<Bytes, anyhow::Error>>, anyhow::Error> {
    let body_bytes = req.collect().await?.to_bytes();
    let mut request: ChatCompletionRequest = serde_json::from_slice(&body_bytes)?;
    // Apply global routing override from dashboard — only when the harness
    // sends model="auto" (i.e. no explicit model preference). Specific model
    // selections like "brainrouter/qwen-coder" or "cloud" are always honoured.
    let is_auto = matches!(request.model.as_str(), "auto" | "" | "brainrouter/auto");
    if is_auto {
        match state.routing_mode.load(AtomicOrdering::Relaxed) {
            1 => request.model = "cloud".to_string(),
            2 => request.model = "local".to_string(),
            _ => {}
        }
    }
    // Spawn routing in a background task so we can return SSE headers immediately.
    // This prevents OMP's "first event" timeout from firing while llama-swap loads
    // a model (which can take minutes for large models like qwen3-27b-mtp).
    // Register the request in the in-flight registry before routing so the
    // dashboard sees it during model loading. The handle lives in the spawned
    // task and (via SniffStream) the response body; the row drops when both end.
    let handle = state.inflight.register(
        "POST /v1/chat/completions".to_string(),
        request.model.clone(),
        user_agent.clone(),
        peer_addr.to_string(),
        session_id.clone().unwrap_or_default(),
        crate::router::conversation_fingerprint(&request),
        0,
    );
    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let result = state.router.route_tagged(request, session_id, cwd, user_agent).await;
        let stream_result = result.map(|(resp, info)| {
            if !info.model_key.is_empty() {
                handle.set_model(info.model_key.clone());
            }
            match resp {
                ProviderResponse::Stream(s) => Box::pin(SniffStream::new(s, Arc::clone(&handle)))
                    as crate::provider::SseStream,
            }
        });
        let _ = tx.send(stream_result);
    });

    let deferred = DeferredStream::new(rx, KEEPALIVE_INTERVAL, DEFERRED_STREAM_TIMEOUT, StreamFormat::OpenAi);
    let safe_stream = SafeStream::new(deferred, StreamFormat::OpenAi);
    let stream_body = StreamBody::new(safe_stream.map(|chunk| chunk.map(Frame::data)));
    let response = Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .body(stream_body.boxed_unsync())?;
    Ok(response)
}

/// Handle POST /v1/messages (Anthropic Messages API)
///
/// Translates the Anthropic request to OpenAI format, routes through Bonsai,
/// and translates the OpenAI SSE response back to Anthropic SSE events.
async fn handle_anthropic_messages(
    req: Request<Incoming>,
    state: Arc<AppState>,
    cwd: String,
    session_id: Option<String>,
    user_agent: String,
    peer_addr: SocketAddr,
) -> Result<Response<UnsyncBoxBody<Bytes, anyhow::Error>>, anyhow::Error> {
    let body_bytes = req.collect().await?.to_bytes();
    let anthropic_req: AnthropicMessagesRequest = serde_json::from_slice(&body_bytes)?;
    let model = anthropic_req.model.clone();
    let mut oai_request = anthropic_to_openai(anthropic_req);
    // Apply global routing override — only for auto-routing requests.
    let is_auto = matches!(oai_request.model.as_str(), "auto" | "" | "brainrouter/auto");
    if is_auto {
        match state.routing_mode.load(AtomicOrdering::Relaxed) {
            1 => oai_request.model = "cloud".to_string(),
            2 => oai_request.model = "local".to_string(),
            _ => {}
        }
    }
    // Spawn routing so SSE headers are returned immediately (same rationale as OpenAI path).
    let handle = state.inflight.register(
        "POST /v1/messages".to_string(),
        oai_request.model.clone(),
        user_agent.clone(),
        peer_addr.to_string(),
        session_id.clone().unwrap_or_default(),
        crate::router::conversation_fingerprint(&oai_request),
        0,
    );
    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let result = state.router.route_tagged(oai_request, session_id, cwd, user_agent).await;
        let stream_result = result.map(|(resp, info)| {
            if !info.model_key.is_empty() {
                handle.set_model(info.model_key.clone());
            }
            match resp {
                ProviderResponse::Stream(s) => Box::pin(SniffStream::new(s, Arc::clone(&handle)))
                    as crate::provider::SseStream,
            }
        });
        let _ = tx.send(stream_result);
    });

    let deferred = DeferredStream::new(rx, KEEPALIVE_INTERVAL, DEFERRED_STREAM_TIMEOUT, StreamFormat::Anthropic);
    let adapted = AnthropicSseAdapter::new(Box::pin(deferred), model);
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
                Some((name, display, swap_state, ref proxy)) if swap_state != "ready" => {
                    let load_progress = poll_llama_server_health(proxy).await;
                    json_response(StatusCode::OK, &serde_json::json!({
                        "state": "loading",
                        "model": name,
                        "model_name": display,
                        "elapsed_ms": 0,
                        "progress": load_progress,
                    }))
                }
                Some((name, display, _, _)) => {
                    // Model is loaded and ready. Check /slots to detect activity
                    // from clients hitting llama-swap directly (bypassing brainrouter).
                    let slot_info = poll_llama_swap_slot(llama_swap_url).await;
                    let (state, n_decoded, n_tokens) = match &slot_info {
                        Some((true, 0, t)) => ("local_processing", 0u64, *t),
                        Some((true, n, t)) => ("local_generating", *n, *t),
                        _ => ("ready", 0, 0),
                    };
                    let progress = match (n_tokens, snap.max_tokens) {
                        (n, Some(max)) if max > 0 => Some(n as f32 / max as f32),
                        _ => None,
                    };
                    json_response(StatusCode::OK, &serde_json::json!({
                        "state": state,
                        "model": name,
                        "model_name": display,
                        "n_decoded": n_decoded,
                        "max_tokens": snap.max_tokens,
                        "progress": progress,
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
            // For local, enrich with llama-swap /slots and /health data if available.
            let (slot_info, running_info) = tokio::join!(
                poll_llama_swap_slot(llama_swap_url),
                poll_llama_swap_running(llama_swap_url),
            );
            let (sub_state, n_decoded, n_tokens) = match &slot_info {
                Some((true, 0, t)) => ("local_processing", 0u64, *t),
                Some((true, n, t)) => ("local_generating", *n, *t),
                Some((false, _, _)) if snap.phase == Phase::LocalStreaming => ("local_generating", 0, 0),
                Some((false, _, _)) => ("ready", 0, 0),
                // Slot poll failed (GPU busy, timeout) — infer from tracker phase.
                None if snap.phase == Phase::LocalStreaming => ("local_generating", 0, 0),
                None => ("local_processing", 0, 0),
            };
            // If the slot shows no token generation yet, check if the model is still loading.
            let progress: Option<f32> = if n_decoded == 0 {
                let proxy = running_info.as_ref().map(|(_, _, _, pr)| pr.as_str()).unwrap_or("");
                poll_llama_server_health(proxy).await
            } else {
                // Token generation in progress: slot n_tokens (total decoded so
                // far) / max_tokens. Monotonic within the request.
                match (n_tokens, snap.max_tokens) {
                    (t, Some(max)) if max > 0 => Some(t as f32 / max as f32),
                    _ => None,
                }
            };
            json_response(StatusCode::OK, &serde_json::json!({
                "state": sub_state,
                "model": snap.model,
                "model_name": snap.model,
                "provider": snap.provider,
                "elapsed_ms": snap.elapsed_ms,
                "n_decoded": n_decoded,
                "n_tokens": n_tokens,
                "max_tokens": snap.max_tokens,
                "progress": progress,
            }))
        }
    }
}

/// Probe the services and return their health status.
/// Called by the dashboard every 10s to render status dots.
/// States: "healthy", "unhealthy", "idle" (service up but no model loaded),
/// "loading" (model is loading), "disabled" (Bonsai/Manifest off in config).
/// Also reports whether the last cloud request fell back to local.
async fn service_health(
    llama_swap_url: &str,
    manifest_url: &str,
    routing_events: &RoutingEvents,
    manifest_enabled: bool,
) -> Response<Full<Bytes>> {
    let timeout = std::time::Duration::from_secs(3);

    // Probe all services in parallel
    let (swap_ok, manifest_ok, llama_cpp_state) = tokio::join!(
        // llama-swap: GET /running
        async {
            VERSION_CLIENT.get(format!("{}/running", llama_swap_url))
                .timeout(timeout).send().await
                .map(|r| r.status().is_success())
                .unwrap_or(false)
        },
        // Manifest: GET /api/v1/health
        async {
            let url = format!("{}/api/v1/health", manifest_url);
            match VERSION_CLIENT.get(&url).timeout(timeout).send().await {
                Ok(r) if r.status().is_success() => {
                    r.json::<serde_json::Value>().await
                        .map(|v| v.get("status").and_then(|s| s.as_str()) == Some("healthy"))
                        .unwrap_or(false)
                }
                _ => false,
            }
        },
        // llama.cpp (toolbox): tri-state check via llama-swap proxy
        async {
            let running_url = format!("{}/running", llama_swap_url);
            let resp = match VERSION_CLIENT.get(&running_url).timeout(timeout).send().await {
                Ok(r) => r,
                Err(_) => return "unhealthy",
            };
            let data: serde_json::Value = match resp.json().await {
                Ok(v) => v,
                Err(_) => return "unhealthy",
            };
            let entries = data.get("running")
                .and_then(|r| r.as_array());
            match entries {
                Some(arr) if arr.is_empty() => "idle",
                Some(arr) => {
                    let proxy = arr.first()
                        .and_then(|e| e.get("proxy"))
                        .and_then(|p| p.as_str());
                    match proxy {
                        Some(proxy_url) => {
                            let health_url = format!("{}/health", proxy_url);
                            if VERSION_CLIENT.get(&health_url).timeout(timeout).send().await
                                .map(|r| r.status().is_success()).unwrap_or(false)
                            {
                                "healthy"
                            } else {
                                let state = arr.first()
                                    .and_then(|e| e.get("state"))
                                    .and_then(|s| s.as_str())
                                    .unwrap_or("unknown");
                                if state == "loading" { "loading" } else { "unhealthy" }
                            }
                        }
                        None => "unhealthy",
                    }
                }
                None => "unhealthy",
            }
        },
    );

    // Check if the most recent cloud request fell back to local
    let cloud_fallback = {
        let events = routing_events.get_all();
        events.iter()
            .find(|e| e.bonsai_decision == "cloud" || e.bonsai_decision == "cloud-direct")
            .map(|e| {
                e.effective_provider.as_deref() == Some("llama-swap")
                    || !e.success
            })
            .unwrap_or(false)
    };

    json_response(StatusCode::OK, &serde_json::json!({
        "llama_swap": if swap_ok { "healthy" } else { "unhealthy" },
        "manifest": if !manifest_enabled {
            "disabled"
        } else if manifest_ok { "healthy" } else { "unhealthy" },
        "llama_cpp": llama_cpp_state,
        "toolbox": llama_cpp_state,
        "cloud_fallback": cloud_fallback,
    }))
}

/// Poll llama-swap /running for the active model's name, display name, state, and proxy URL.
async fn poll_llama_swap_running(llama_swap_url: &str) -> Option<(String, String, String, String)> {
    let url = format!("{}/running", llama_swap_url);
    let resp = VERSION_CLIENT.get(&url).timeout(std::time::Duration::from_secs(2)).send().await.ok()?;
    let data: serde_json::Value = resp.json().await.ok()?;
    let entry = data.get("running")?.as_array()?.first()?;
    let name = entry.get("model")?.as_str()?.to_string();
    let display = entry.get("name").and_then(|n| n.as_str()).unwrap_or(&name).to_string();
    let state = entry.get("state").and_then(|s| s.as_str()).unwrap_or("unknown").to_string();
    let proxy = entry.get("proxy").and_then(|p| p.as_str()).unwrap_or("").to_string();
    Some((name, display, state, proxy))
}

/// Poll a llama-server's /health endpoint for model load progress.
/// Returns Some(progress) where progress is 0.0–1.0 when status is "loading_model".
/// Returns None when the server is not reachable or not currently loading.
async fn poll_llama_server_health(proxy_url: &str) -> Option<f32> {
    if proxy_url.is_empty() { return None; }
    let health_url = format!("{}/health", proxy_url);
    let resp = VERSION_CLIENT.get(&health_url).timeout(std::time::Duration::from_secs(2)).send().await.ok()?;
    let data: serde_json::Value = resp.json().await.ok()?;
    // llama-server returns {"status": "loading_model", "progress": 0.75} while loading.
    if data.get("status").and_then(|s| s.as_str()) == Some("loading_model") {
        data.get("progress").and_then(|p| p.as_f64()).map(|p| p as f32)
    } else {
        None
    }
}

/// Poll the active llama-server's /slots endpoint for progress.
/// Returns (is_active, n_decoded, n_tokens_total).
///
/// /slots exposes `n_tokens` = total tokens decoded on the slot, which is
/// monotonic within a request. The old code read `next_token.n_decoded`,
/// which only counts tokens in the *current chunk*, so the dashboard
/// progress bar saw sawtooth jumps instead of steady progress.
async fn poll_llama_swap_slot(llama_swap_url: &str) -> Option<(bool, u64, u64)> {
    let running_url = format!("{}/running", llama_swap_url);
    let resp = VERSION_CLIENT.get(&running_url).timeout(std::time::Duration::from_secs(2)).send().await.ok()?;
    let data: serde_json::Value = resp.json().await.ok()?;
    let entry = data.get("running")?.as_array()?.first()?;
    let proxy = entry.get("proxy")?.as_str()?.to_string();
    let slots_url = format!("{}/slots", proxy);
    let resp = VERSION_CLIENT.get(&slots_url).timeout(std::time::Duration::from_secs(2)).send().await.ok()?;
    let body = resp.text().await.ok()?;
    parse_llama_slots(&body)
}

/// Parse the llama-server /slots payload, returning the active slot's
/// (is_active, n_decoded, n_ctx).
///
/// This llama-server build leaves the top-level `n_tokens`/`n_decoded`
/// fields null while generating; the live per-request count is
/// `next_token.n_decoded` (tokens decoded in the current prompt run),
/// which is monotonic within a request. `is_active` comes from
/// `is_processing` (with the max `n_prompt_tokens` slot as a fallback
/// for a slot between chunks).
fn parse_llama_slots(body: &str) -> Option<(bool, u64, u64)> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    // llama-server serves /slots as a bare JSON array; accept a
    // {"slots":[...]} wrapper too for older/newer builds.
    let slots = if v.is_array() {
        v.as_array()?
    } else {
        v.get("slots")?.as_array()?
    };
    let mut best: Option<(bool, u64, u64)> = None; // (active, decoded, prompt_tokens)
    for s in slots {
        let processing = s.get("is_processing").and_then(|p| p.as_bool()).unwrap_or(false);
        let prompt = s.get("n_prompt_tokens").and_then(|n| n.as_u64()).unwrap_or(0);
        // next_token may be an object or a single-element array depending on
        // the llama-server build.
        let decoded = s
            .get("next_token")
            .and_then(|nt| if nt.is_array() { nt.as_array()?.first() } else { Some(nt) })
            .and_then(|nt| nt.get("n_decoded"))
            .and_then(|n| n.as_u64())
            .unwrap_or(0);
        let active = processing || decoded > 0;
        match &mut best {
            None => best = Some((active, decoded, prompt)),
            Some(b) => {
                // Prefer an active slot; otherwise keep the one furthest
                // along in prompt tokens (most recently served).
                if (active && !b.0) || (active == b.0 && prompt > b.2) {
                    *b = (active, decoded, prompt);
                }
            }
        }
    }
    let (active, decoded, _prompt) = best?;
    Some((active, decoded, decoded))
}

/// Unload every model llama-swap currently holds in memory (VRAM), without
/// restarting the service. Proxies to llama-swap's `POST /api/models/unload`.
/// When `reload` is set, warm-loads each key in `local_models` afterwards so the
/// local working set (e.g. the dual-Dirk main + subs group) is resident again
/// instead of the user hand-starting each model.
async fn flush_models(
    llama_swap_url: &str,
    reload: bool,
    local_models: &[String],
) -> Response<Full<Bytes>> {
    let base = llama_swap_url.trim_end_matches('/').to_string();
    let url = format!("{}/api/models/unload", base);
    info!(%url, "Flushing all llama-swap models");
    match VERSION_CLIENT
        .post(&url)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            info!("llama-swap flushed all models");
            if reload && !local_models.is_empty() {
                // Re-establish the local working set: warm-load each model so
                // the dual-Dirk group (main + subs) is resident again.
                let mut reloaded = Vec::new();
                for key in local_models {
                    let warm = format!("{}/v1/chat/completions", base);
                    let ok = VERSION_CLIENT
                        .post(&warm)
                        .timeout(std::time::Duration::from_secs(180))
                        .json(&serde_json::json!({
                            "model": key,
                            "messages": [{"role": "user", "content": "warm"}],
                            "max_tokens": 1,
                            "stream": false
                        }))
                        .send()
                        .await
                        .map(|r| r.status().is_success())
                        .unwrap_or(false);
                    if ok {
                        info!(model = %key, "reloaded local model after flush");
                        reloaded.push(key.clone());
                    } else {
                        warn!(model = %key, "reload of local model after flush failed");
                    }
                }
                json_response(
                    StatusCode::OK,
                    &serde_json::json!({
                        "status": "ok",
                        "message": "flushed, then reloaded the local working set",
                        "reloaded": reloaded
                    }),
                )
            } else {
                json_response(
                    StatusCode::OK,
                    &serde_json::json!({
                        "status": "ok",
                        "message": "all llama-swap models unloaded from memory"
                    }),
                )
            }
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            warn!(%status, %body, "llama-swap flush rejected");
            json_response(
                StatusCode::BAD_GATEWAY,
                &ErrorResponse { error: format!("llama-swap returned {}: {}", status, body.trim()) },
            )
        }
        Err(e) => {
            warn!(error = %e, "llama-swap unreachable during flush");
            json_response(
                StatusCode::SERVICE_UNAVAILABLE,
                &ErrorResponse { error: format!("llama-swap unreachable: {}", e) },
            )
        }
    }
}


/// Restart the llama.cpp toolbox by restarting llama-swap.
/// llama-swap manages the toolbox container lifecycle; restarting it
/// kills the current model and lets llama-swap spawn fresh on next request.
async fn restart_llama_cpp() -> Response<Full<Bytes>> {
    info!("Restarting llama.cpp toolbox via llama-swap restart");
    let output = tokio::process::Command::new("systemctl")
        .args(["--user", "restart", "llama-swap"])
        .output()
        .await;

    match output {
        Ok(out) if out.status.success() => {
            info!("llama-swap restarted (toolbox will reload on next request)");
            json_response(StatusCode::OK, &serde_json::json!({
                "status": "ok",
                "service": "llama-cpp",
                "message": "llama-swap restarted — toolbox will reload on next model request"
            }))
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            error!(%stderr, "llama-swap restart failed");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Restart failed: {}", stderr.trim()),
            })
        }
        Err(e) => {
            error!(error = %e, "Failed to exec systemctl");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Failed to exec systemctl: {}", e),
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
            error!(service, %stderr, "systemctl restart returned non-success");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Failed to restart {}: {}", service, stderr.trim()),
            })
        }
        Err(e) => {
            error!(service, error = %e, "systemctl restart failed");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Failed to restart {}: {}", service, e),
            })
        }
    }
}
pub async fn toolboxes_list() -> Response<Full<Bytes>> {
    use tokio::process::Command;
    let containers = Command::new("podman")
        .args(["ps", "-a", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"])
        .output()
        .await;

    let mut list: Vec<serde_json::Value> = Vec::new();
    if let Ok(o) = containers {
        for line in String::from_utf8_lossy(&o.stdout).lines() {
            let mut parts = line.splitn(3, '\t');
            let (Some(name), Some(image), Some(status)) =
                (parts.next(), parts.next(), parts.next())
            else {
                continue;
            };
            // Only llama-* toolboxes (skip comfyui, ds4, etc.).
            if !name.starts_with("llama-") {
                continue;
            }
            list.push(serde_json::json!({
                "name": name,
                "short_name": name.strip_prefix("llama-").unwrap_or(name),
                "image": image,
                "running": status.starts_with("Up"),
                "status": status,
            }));
        }
    }
    list.sort_by(|a, b| a["name"].as_str().unwrap_or("").cmp(b["name"].as_str().unwrap_or("")));

    // Enrich each entry with local image creation date + Docker Hub latest
    // per tag, so the dashboard can show installed vs latest and flag updates.
    // Local created is read from `podman inspect` (absolute date) per distinct
    // image; the Hub date is the tag_last_pushed of the image's tag, fetched
    // once per repo (batched) rather than once per container.
    let repo_tag_map = hub_toolbox_tag_dates().await;
    for tb in list.iter_mut() {
        let image = tb["image"].as_str().unwrap_or("").to_string();
        // local_created: absolute YYYY-MM-DD the local image was pulled.
        let local = image_created_date(&image).await.unwrap_or_default();
        tb["local_created"] = serde_json::Value::String(local.clone());

        // latest_created: Docker Hub push date for this exact tag.
        let (repo, tag) = split_repo_tag(&image);
        let latest = repo_tag_map
            .get(&format!("{}/{}", repo, tag))
            .cloned()
            .unwrap_or_default();
        tb["latest_created"] = serde_json::Value::String(latest.clone());

        // update_available: local image predates the newest Hub push.
        tb["update_available"] =
            serde_json::Value::Bool(!local.is_empty() && !latest.is_empty() && local < latest);
    }

    json_response(StatusCode::OK, &serde_json::json!({ "toolboxes": list }))
}

/// Split a full image reference into (repository, tag). Defaults to `latest`
/// when no tag is present.
fn split_repo_tag(image: &str) -> (&str, &str) {
    match image.rsplit_once(':') {
        Some((repo, tag)) if !tag.is_empty() => (repo, tag),
        _ => (image, "latest"),
    }
}

/// Absolute creation date (YYYY-MM-DD) of a local image, via podman inspect.
async fn image_created_date(image: &str) -> Option<String> {
    let out = tokio::process::Command::new("podman")
        .args(["inspect", "--format", "{{.Created}}", image])
        .output()
        .await
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let created = String::from_utf8_lossy(&out.stdout).to_string();
    // e.g. "2026-08-26T14:03:11.234Z" -> "2026-08-26"
    created.trim().get(..10).filter(|s| s.len() == 10).map(|s| s.to_string())
}

/// Batched Docker Hub map of `repository:tag` -> last-push date (YYYY-MM-DD)
/// for every toolbox repo currently in use. A single `tags` call per repo
/// (all tags, capped) instead of one call per container.
async fn hub_toolbox_tag_dates() -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    let resp = VERSION_CLIENT
        .get("https://hub.docker.com/v2/repositories/kyuz0/amd-strix-halo-toolboxes/tags?page_size=100")
        .send()
        .await
        .ok();
    if let Some(r) = resp {
        if let Ok(data) = r.json::<serde_json::Value>().await {
            for t in data.get("results").and_then(|v| v.as_array()).into_iter().flatten() {
                let name = t.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let pushed = t
                    .get("tag_last_pushed")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.get(..10))
                    .unwrap_or("");
                if !name.is_empty() && !pushed.is_empty() {
                    map.insert(format!("kyuz0/amd-strix-halo-toolboxes:{}", name), pushed.to_string());
                }
            }
        }
    }
    map
}

/// Compute local versions and "latest available" metadata for the /api/versions endpoint.
/// Called periodically by a background task in daemon.rs.
pub async fn compute_versions_json(bonsai_fork_path: &std::path::Path) -> serde_json::Value {
    use tokio::process::Command;

    // 1. llama-swap version (timeout so a hung binary can't freeze the task)
    const LOCAL_VERSION_TIMEOUT_SECS: u64 = 5;
    let swap_ver = {
        let child = Command::new(home_bin("llama-swap"))
            .arg("--version")
            .output();
        match tokio::time::timeout(std::time::Duration::from_secs(LOCAL_VERSION_TIMEOUT_SECS), child).await {
            Ok(Ok(out)) if out.status.success() => {
                String::from_utf8_lossy(&out.stdout).trim()
                    .replace("version: ", "")
                    .to_string()
            }
            _ => "unknown".to_string(),
        }
    };

    // 2. llama.cpp version from toolbox container image.
    const PODMAN_VERSION_TIMEOUT_SECS: u64 = 15;
    let toolbox_ver = {
        let child = Command::new("podman")
            .args(["run", "--rm", "docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv", "llama-server", "--version"])
            .kill_on_drop(true)
            .output();
        match tokio::time::timeout(std::time::Duration::from_secs(PODMAN_VERSION_TIMEOUT_SECS), child).await {
            Ok(Ok(out)) => {
                let combined = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
                if let Some(line) = combined.lines().find(|l| l.contains("version:")) {
                    line.replace("version:", "").replace("built with", "").trim().to_string()
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
            Err(_) => {
                warn!(timeout_secs = PODMAN_VERSION_TIMEOUT_SECS, "podman run timed out during llama.cpp version check");
                "unknown".to_string()
            }
        }
    };

    // 3. Toolbox container image version and created date
    let (toolbox_image_ver, toolbox_image_created) = {
        let inspect = Command::new("podman")
            .args(["image", "inspect",
                   "--format", "{{index .Labels \"org.opencontainers.image.version\"}}\t{{.Created}}",
                   "docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv"])
            .output();
        let ver_out = match tokio::time::timeout(
            std::time::Duration::from_secs(LOCAL_VERSION_TIMEOUT_SECS),
            inspect,
        )
        .await
        {
            Ok(Ok(o)) => Some(o),
            _ => None,
        };
        match ver_out {
            Some(o) => {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if let Some((ver, created)) = s.split_once('\t') {
                    let date = created.trim().get(..10).unwrap_or("").to_string();
                    (ver.trim().to_string(), date)
                } else {
                    (s, String::new())
                }
            }
            None => (String::new(), String::new()),
        }
    };

    // 4. Locally running Manifest image info
    let manifest_ver = {
        let inspect = Command::new("docker")
            .args(["inspect", "--format",
                   "{{index .Config.Labels \"org.opencontainers.image.created\"}} {{slice .Id 7 19}}",
                   "manifest-manifest-1"])
            .output();
        let out = match tokio::time::timeout(
            std::time::Duration::from_secs(LOCAL_VERSION_TIMEOUT_SECS),
            inspect,
        )
        .await
        {
            Ok(Ok(o)) => Some(o),
            _ => None,
        };
        match out {
            Some(o) if o.status.success() => {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if let Some((date_part, hash)) = s.split_once(' ') {
                    format!("{} · {}", date_part.get(..10).unwrap_or(date_part), hash)
                } else {
                    s
                }
            }
            _ => "unknown".to_string(),
        }
    };
    // 4b. Bonsai classifier fork (PrismML llama.cpp) — the binary that serves
    //     the classifier, distinct from the toolbox llama-server above.
    let bonsai_fork_ver = {
        let version = Command::new(bonsai_fork_path)
            .arg("--version")
            .output();
        let out = match tokio::time::timeout(
            std::time::Duration::from_secs(LOCAL_VERSION_TIMEOUT_SECS),
            version,
        )
        .await
        {
            Ok(Ok(o)) => Some(o),
            _ => None,
        };
        match out {
            Some(o) => {
                let combined = format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
                combined
                    .lines()
                    .find(|l| l.contains("version:"))
                    .map(|l| {
                        let s = l.replace("version:", "").trim().to_string();
                        s.split("built with").next().unwrap_or(&s).trim().to_string()
                    })
                    .unwrap_or_else(|| "unknown".to_string())
            }
            None => {
                warn!(path = %bonsai_fork_path.display(), "Failed to read Bonsai fork version (missing or timed out)");
                "unknown".to_string()
            }
        }
    };

    // 5. Remote "latest available" versions (GitHub / Docker Hub)
    let (llama_swap_latest, manifest_latest, toolbox_latest) = tokio::join!(
        fetch_latest_llama_swap(),
        fetch_latest_manifest(),
        fetch_latest_toolbox(),
    );

    serde_json::json!({
        "brainrouter": env!("CARGO_PKG_VERSION"),
        "llama_swap": swap_ver,
        "llama_cpp": toolbox_ver,
        "bonsai_fork": bonsai_fork_ver,
        "toolbox_image_ver": toolbox_image_ver,
        "toolbox_image_created": toolbox_image_created,
        "manifest": manifest_ver,
        "llama_swap_latest": llama_swap_latest.unwrap_or_default(),
        "manifest_latest": manifest_latest.unwrap_or_default(),
        "toolbox_latest": toolbox_latest.unwrap_or_default(),
    })
}

async fn fetch_latest_llama_swap() -> Option<String> {
    let resp = VERSION_CLIENT.get("https://api.github.com/repos/mostlygeek/llama-swap/releases/latest")
        .send().await.ok()?;
    let data: serde_json::Value = resp.json().await.ok()?;
    data.get("tag_name").and_then(|v| v.as_str()).map(|v| v.trim_start_matches('v').to_string())
}

async fn fetch_latest_manifest() -> Option<String> {
    let resp = VERSION_CLIENT.get("https://hub.docker.com/v2/repositories/manifestdotbuild/manifest/tags/latest")
        .send().await.ok()?;
    let data: serde_json::Value = resp.json().await.ok()?;
    data.get("tag_last_pushed")
        .and_then(|v| v.as_str())
        .map(|s| s.get(..10).unwrap_or(s).to_string())
}

async fn fetch_latest_toolbox() -> Option<String> {
    let resp = VERSION_CLIENT.get("https://hub.docker.com/v2/repositories/kyuz0/amd-strix-halo-toolboxes/tags/vulkan-radv")
        .send().await.ok()?;
    let data: serde_json::Value = resp.json().await.ok()?;
    data.get("tag_last_pushed")
        .and_then(|v| v.as_str())
        .map(|s| s.get(..10).unwrap_or(s).to_string())
}

async fn upgrade_llama_swap() -> Response<Full<Bytes>> {
    info!("Upgrading llama-swap from GitHub releases...");

    let target_bin = home_bin("llama-swap");
    // home_bin falls back to the bare name when ~/.local/bin lacks the binary;
    // the extract step writes "{target_bin}.tmp" + rename, which MUST be an
    // absolute path or it resolves against the daemon's cwd. Pin it here.
    let target_bin = if std::path::Path::new(&target_bin).is_absolute() {
        target_bin
    } else {
        format!("{}/.local/bin/llama-swap", home_dir())
    };

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

/// Look up the image a podman container is running from (full `docker.io/…`
/// reference, as listed by `podman ps`).
async fn toolbox_container_image(container: &str) -> Option<String> {
    let out = tokio::process::Command::new("podman")
        .args(["inspect", "--format", "{{.Image}}", container])
        .output()
        .await
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let image = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!image.is_empty()).then_some(image)
}

async fn upgrade_toolbox(container: &str, image: &str) -> Response<Full<Bytes>> {
    info!(%container, %image, "Upgrading toolbox container...");

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
    Ok(into_unsync(resp))
}

async fn handle_llama_swap_models(
    llama_swap_url: &str,
) -> Result<Response<UnsyncBoxBody<Bytes, anyhow::Error>>, anyhow::Error> {
    let url = format!("{}/v1/models", llama_swap_url);
    let resp = VERSION_CLIENT.get(&url).timeout(std::time::Duration::from_secs(3)).send().await?;
    let data: serde_json::Value = resp.json().await?;
    
    let resp = json_response(StatusCode::OK, &data);
    Ok(into_unsync(resp))
}

/// Fetch live models from llama-swap and update the brainrouter section of
/// `~/.omp/agent/models.yml`. Preserves all other provider sections.
///
/// Returns the total number of models written to the brainrouter section.
pub async fn sync_omp_models(llama_swap_url: &str, tcp_addr: &str) -> anyhow::Result<usize> {
    let home = home_dir();
    // Refuse to write to /root — brainrouter is a user-facing daemon and
    // should not modify root's home directory.
    if home.is_empty() || home == "/root" {
        anyhow::bail!("$HOME is not set; cannot locate models.yml");
    }

    // Fetch live models from llama-swap (async HTTP).
    let url = format!("{}/v1/models", llama_swap_url);
    let resp = VERSION_CLIENT.get(&url)
        .timeout(std::time::Duration::from_secs(5))
        .send().await
        .map_err(|e| anyhow::anyhow!("Failed to fetch llama-swap models: {}", e))?;
    let body: serde_json::Value = resp.json().await
        .map_err(|e| anyhow::anyhow!("Failed to parse llama-swap models response: {}", e))?;
    let model_ids: Vec<String> = body
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|v| v.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();

    // All filesystem + YAML work runs off the async executor.
    let tcp_addr_owned = tcp_addr.to_string();
    tokio::task::spawn_blocking(move || write_omp_models_yml(&home, &model_ids, &tcp_addr_owned))
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking panicked: {}", e))?
}

/// Blocking: read models.yml, merge brainrouter models, write atomically.
fn write_omp_models_yml(home: &str, model_ids: &[String], tcp_addr: &str) -> anyhow::Result<usize> {
    let models_path = format!("{}/.omp/agent/models.yml", home);
    let path = std::path::Path::new(&models_path);

    // Read existing models.yml (or start fresh).
    let mut doc: serde_yaml::Value = if path.exists() {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", models_path, e))?;
        serde_yaml::from_str(&content).unwrap_or(serde_yaml::Value::Mapping(serde_yaml::Mapping::new()))
    } else {
        serde_yaml::Value::Mapping(serde_yaml::Mapping::new())
    };

    // Build the brainrouter models list.
    let mut models = Vec::new();

    // Fixed routing models.
    for (id, name) in [("auto", "Brainrouter (auto)"), ("local", "Brainrouter (local)"), ("cloud", "Brainrouter (cloud)")] {
        let mut entry = serde_yaml::Mapping::new();
        entry.insert(ykey("id"), yval(id));
        entry.insert(ykey("name"), yval(name));
        entry.insert(ykey("reasoning"), serde_yaml::Value::Bool(false));
        let mut input = serde_yaml::Sequence::new();
        input.push(yval("text"));
        entry.insert(ykey("input"), serde_yaml::Value::Sequence(input));
        models.push(serde_yaml::Value::Mapping(entry));
    }

    // llama-swap models (skip the fixed ones).
    let skip = ["auto", "local", "cloud"];
    for id in model_ids {
        if skip.contains(&id.as_str()) {
            continue;
        }
        let mut entry = serde_yaml::Mapping::new();
        entry.insert(ykey("id"), yval(id));
        entry.insert(ykey("name"), yval(&model_id_to_display_name(id)));
        models.push(serde_yaml::Value::Mapping(entry));
    }

    let total = models.len();

    // Build the brainrouter provider entry.
    let mut br_provider = serde_yaml::Mapping::new();
    br_provider.insert(ykey("baseUrl"), yval(&format!("http://{}/v1", tcp_addr)));
    br_provider.insert(ykey("api"), yval("openai-completions"));
    br_provider.insert(ykey("auth"), yval("none"));
    br_provider.insert(ykey("models"), serde_yaml::Value::Sequence(models));

    // Merge into the document, preserving other providers.
    let providers = doc
        .as_mapping_mut()
        .ok_or_else(|| anyhow::anyhow!("models.yml root is not a mapping"))?
        .entry(ykey("providers"))
        .or_insert(serde_yaml::Value::Mapping(serde_yaml::Mapping::new()));
    let providers_map = providers
        .as_mapping_mut()
        .ok_or_else(|| anyhow::anyhow!("models.yml 'providers' is not a mapping"))?;
    providers_map.insert(ykey("brainrouter"), serde_yaml::Value::Mapping(br_provider));

    // Atomic write: tempfile (PID-qualified to avoid races) then rename.
    let yaml_str = serde_yaml::to_string(&doc)
        .map_err(|e| anyhow::anyhow!("Failed to serialize models.yml: {}", e))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow::anyhow!("Failed to create directory {}: {}", parent.display(), e))?;
    }
    let tmp_path = format!("{}.{}.tmp", models_path, std::process::id());
    std::fs::write(&tmp_path, &yaml_str)
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {}", tmp_path, e))?;
    std::fs::rename(&tmp_path, path)
        .map_err(|e| anyhow::anyhow!("Failed to rename {} -> {}: {}", tmp_path, models_path, e))?;

    info!(path = %models_path, model_count = total, "Synced OMP models.yml");
    Ok(total)
}

/// Convert a model ID like "qwen3.6-27b-q6-amdvlk" to "Qwen3.6 27B Q6 AMDVLK".
fn model_id_to_display_name(id: &str) -> String {
    id.split('-')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let upper = part.to_uppercase();
            // Parameter-count suffixes: 27b, 120b, 128B → 27B, 120B
            if part.len() >= 2 && (part.ends_with('b') || part.ends_with('B')) {
                let prefix = &part[..part.len() - 1];
                if !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit()) {
                    return upper;
                }
            }
            match upper.as_str() {
                // Quant tags: Q6, Q8, Q4, etc.
                s if s.starts_with('Q') && s.len() <= 4 && s[1..].chars().all(|c| c.is_ascii_digit()) => upper,
                // Known all-caps abbreviations
                "A4B" | "A3B" | "A10B" | "E2B" | "E4B" | "AIR" | "OSS" | "GOOG" | "AMDVLK" | "DRAFT" => upper,
                _ => {
                    // Titlecase: capitalize first char, leave rest as-is
                    let mut chars = part.chars();
                    match chars.next() {
                        None => String::new(),
                        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
                    }
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn ykey(s: &str) -> serde_yaml::Value {
    serde_yaml::Value::String(s.to_string())
}

fn yval(s: &str) -> serde_yaml::Value {
    serde_yaml::Value::String(s.to_string())
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_name_full_model_id() {
        assert_eq!(model_id_to_display_name("qwen3.6-27b-q6-amdvlk"), "Qwen3.6 27B Q6 AMDVLK");
    }

    #[test]
    fn display_name_size_suffixes() {
        assert_eq!(model_id_to_display_name("llama-3.1-8b"), "Llama 3.1 8B");
        assert_eq!(model_id_to_display_name("gpt-oss-120b"), "Gpt OSS 120B");
        assert_eq!(model_id_to_display_name("mistral-medium-3.5-128B"), "Mistral Medium 3.5 128B");
    }

    #[test]
    fn display_name_quant_tags() {
        assert_eq!(model_id_to_display_name("gemma-4-31b-q6"), "Gemma 4 31B Q6");
        assert_eq!(model_id_to_display_name("gemma-4-31b-q8-heretic"), "Gemma 4 31B Q8 Heretic");
    }

    #[test]
    fn display_name_known_abbreviations() {
        assert_eq!(model_id_to_display_name("gemma-4-e2b"), "Gemma 4 E2B");
        assert_eq!(model_id_to_display_name("glm4.5-air"), "Glm4.5 AIR");
        assert_eq!(model_id_to_display_name("qwen3.6-27b-q6-draft"), "Qwen3.6 27B Q6 DRAFT");
    }

    #[test]
    fn display_name_empty_and_edge_cases() {
        assert_eq!(model_id_to_display_name(""), "");
        assert_eq!(model_id_to_display_name("foo--bar"), "Foo Bar");  // consecutive hyphens
        assert_eq!(model_id_to_display_name("stepfun"), "Stepfun");  // single word
    }

    #[test]
    fn display_name_moe_parts() {
        assert_eq!(model_id_to_display_name("qwen3.6-35b-a3b"), "Qwen3.6 35B A3B");
        assert_eq!(model_id_to_display_name("qwen3.5-122b-a10b"), "Qwen3.5 122B A10B");
    }

    #[test]
    fn write_omp_preserves_other_providers() {
        // Create a temp dir to simulate ~/.omp/agent/
        let tmp = std::env::temp_dir().join(format!("brainrouter-test-{}", std::process::id()));
        let agent_dir = tmp.join(".omp/agent");
        std::fs::create_dir_all(&agent_dir).unwrap();
        let models_file = agent_dir.join("models.yml");

        // Write initial YAML with a manifest provider
        std::fs::write(&models_file, r#"providers:
  manifest:
    baseUrl: http://localhost:3001/v1
    models:
    - id: auto
      name: Manifest
"#).unwrap();

        // Sync with some fake model IDs
        let result = write_omp_models_yml(tmp.to_str().unwrap(), &[
            "auto".to_string(), "local".to_string(), "my-model-27b".to_string(),
        ], "127.0.0.1:9099");
        assert!(result.is_ok());
        let count = result.unwrap();
        assert_eq!(count, 4); // auto + local + cloud + my-model-27b

        // Verify manifest provider survived
        let content = std::fs::read_to_string(&models_file).unwrap();
        assert!(content.contains("manifest"), "manifest provider should be preserved");
        assert!(content.contains("brainrouter"), "brainrouter provider should exist");
        assert!(content.contains("my-model-27b"), "synced model should be present");
        assert!(content.contains("http://127.0.0.1:9099/v1"), "brainrouter baseUrl must use the daemon's tcp_addr");

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp);
    }
}