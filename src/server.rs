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
use std::fs::{File, OpenOptions};
use std::net::SocketAddr;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::{TcpListener, UnixListener, UnixStream};
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
use crate::benchmark::{self, BenchmarkStore};
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
use crate::toolbox_catalog::{
    self, SupportedServingBackend, ToolboxCatalog, ToolboxDefinition,
};

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
    /// Legacy compatibility mirror. Routing decisions use Router's profile store.
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
    /// FR-A: code-review master switch (default on). Shared with the review
    /// dispatch so a review request short-circuits to `disabled` when off.
    pub code_review_enabled: Arc<AtomicBool>,
    /// FR-D: PR-generation-guideline switch (default off). When on, the two
    /// public proxy handlers inject a PR-structuring directive into requests.
    pub pr_guidelines_enabled: Arc<AtomicBool>,
    /// In-flight request registry (dashboard tracking + cancel).
    pub inflight: Arc<crate::inflight::InflightRegistry>,
    /// Optional benchmark storage; an initialization error disables only the explorer.
    pub benchmark_store: Result<Arc<BenchmarkStore>, String>,
    /// Optional native benchmark execution; disabled or invalid configuration
    /// does not affect the proxy or imported benchmark explorer.
    pub benchmark_lab: Result<Arc<crate::benchmark_lab::BenchmarkLab>, String>,
    /// Read-only model observations and separately persisted operator settings.
    pub observability: Arc<crate::observability::Observability>,
    /// Per-container-name mutation lock for toolbox create/update/delete/adopt,
    /// so two concurrent requests for the same container name queue instead of
    /// racing `podman`/`toolbox` invocations against each other. Different
    /// container names never contend (see design doc §5c).
    pub toolbox_container_locks: Arc<std::sync::Mutex<std::collections::HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
    /// Model-download job registry (PR6, §10): single-flight `hf download`
    /// orchestration for the download-capable catalog backends
    /// (ds4/halogen/llama_cpp/r9v; vllm is out of scope, see
    /// `model_downloads.rs` module docs).
    pub model_downloads: Arc<crate::model_downloads::ModelDownloadRegistry>,
    /// Serving-identity registry (PR11a, §5b/§16): bookkeeping-only record
    /// of currently-running toolbox Server Mode containers'
    /// `{toolbox_backend, compute_api, runtime_profile_id, endpoint}`.
    /// **Does not make any backend a routable Router upstream** — see
    /// `serving_identity` module docs and design doc §16 before extending
    /// this to affect request routing.
    pub serving_identities: Arc<crate::serving_identity::ServingIdentityRegistry>,
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

fn is_loopback_http_url(value: &str) -> bool {
    let Ok(url) = url::Url::parse(value) else {
        return false;
    };
    if url.scheme() != "http" || !url.username().is_empty() || url.password().is_some() {
        return false;
    }
    match url.host() {
        Some(url::Host::Domain(host)) => host.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(address)) => address.is_loopback(),
        Some(url::Host::Ipv6(address)) => address.is_loopback(),
        None => false,
    }
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
        // Generalized toolbox container management (create/update/delete/adopt,
        // PR2 §5c) — prefix-gated like /api/upgrade/ so a future sub-path under
        // this same prefix is never accidentally left ungated.
        || (method == "POST" && path.starts_with("/api/toolbox-containers"))
        // PR3: cockpit config.json explicit "apply" writes (§4) — these write
        // to a file outside brainrouter's own state, so gate them the same way.
        || (method == "POST" && path.starts_with("/api/cockpit-config/"))
        // PR6: model-download orchestration (§10) — starting/cancelling a
        // job spawns `hf download`/reads local files; prefix-gated like
        // /api/toolbox-containers so any future sub-path stays covered.
        || (method == "POST" && path.starts_with("/api/model-downloads"))
        // PR7: server-mode start/stop (§12) — launches/removes a detached
        // `podman run` container; prefix-gated the same way.
        || (method == "POST" && path.starts_with("/api/server-mode/"))
        // FR-A: code-review master switch write path — prefix-gated like the
        // other local-only mutating APIs.
        || (method == "POST" && path.starts_with("/api/review/"))
        || (method == "POST" && (
            path == "/api/config" || path == "/api/llama-swap-config"
            || path == "/api/open-editor" || path == "/api/models/sync-omp"
            || path == "/api/routing-mode" || path == "/api/review-config"
            || path == "/api/routing-profile"
            || path == "/api/bridges/toggle"
            || path == "/api/bonsai/toggle" || path == "/api/models/flush"
            || path == "/api/nudge" || path == "/api/prompt-rewrite"
            // Review API is destructive too: it spawns reviews (arbitrary
            // project paths read into cloud prompts) and can approve/resolve
            // sessions. Gate it like the rest.
            || path.starts_with("/review/api/")
            || path.starts_with("/api/benchmarks/")
            || path == "/api/inflight/cancel"
            || path.starts_with("/api/observability/")
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
        
        // Local port forwarding can change the browser port without changing the
        // trusted loopback host. Null and non-loopback origins remain forbidden.
        let has_allowed_origin = if let Some(origin) = req.headers().get("Origin") {
            is_loopback_http_url(origin.to_str().unwrap_or(""))
        } else if let Some(referer) = req.headers().get("Referer") {
            is_loopback_http_url(referer.to_str().unwrap_or(""))
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
        let result = escalation::handle_review_request(
            req,
            Arc::clone(&state.review_service),
            cwd,
            state.code_review_enabled.load(AtomicOrdering::Relaxed),
        ).await;
        return result;
    }

    if path.starts_with("/api/benchmarks/lab/") || path == "/api/benchmarks/lab" {
        return match &state.benchmark_lab {
            Ok(lab) => crate::benchmark_lab::handle_request(req, Arc::clone(lab)).await,
            Err(reason) => Ok(crate::benchmark_lab::unavailable_response(reason)),
        };
    }

    if path == "/benchmarks" || path == "/benchmarks/" || path.starts_with("/api/benchmarks/") {
        return match &state.benchmark_store {
            Ok(store) => benchmark::handle_request(req, store).await,
            Err(reason) => Ok(benchmark::unavailable_response(reason)),
        };
    }

    if path == "/models" || path == "/models/" || path.starts_with("/api/observability/") {
        return crate::observability::handle_request(req, &state).await;
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
            let choice = state.review_service.preferences().profile().main;
            let resp = json_response(StatusCode::OK, &serde_json::json!({
                "mode": choice.backend(), "model": choice.model(),
            }));
            into_unsync(resp)
        }

        ("POST", "/api/routing-mode") => {
            #[derive(serde::Deserialize)]
            #[serde(deny_unknown_fields)]
            struct ModeUpdate { mode: String }
            let update: ModeUpdate = match read_routing_json(req).await {
                Ok(update) => update,
                Err(error) => return Ok(routing_error(StatusCode::BAD_REQUEST, error)),
            };
            let choice = match crate::routing_profile::ModelChoice::from_legacy(&update.mode, None) {
                Ok(choice) => choice,
                Err(error) => return Ok(routing_error(StatusCode::BAD_REQUEST, error)),
            };
            let store = Arc::clone(state.review_service.preferences());
            match tokio::task::spawn_blocking(move || store.update_main(choice)).await
                .map_err(anyhow::Error::from).and_then(|result| result) {
                Ok(()) => {
                    let code = match update.mode.as_str() { "cloud" => 1, "local" => 2, _ => 0 };
                    state.routing_mode.store(code, AtomicOrdering::Relaxed);
                    into_unsync(json_response(StatusCode::OK, &serde_json::json!({ "mode": update.mode })))
                }
                Err(error) => routing_error(StatusCode::INTERNAL_SERVER_ERROR, error),
            }
        }

        ("GET", "/api/routing-profile") => {
            into_unsync(json_response(StatusCode::OK, &serde_json::json!({
                "profile": state.review_service.preferences().profile(),
                "cloud_enabled": state.manifest_enabled,
                "cloud_policy": "Disabled or unavailable cloud falls back to the configured local default; selecting a profile never enables cloud.",
                "review_policy": "New sessions snapshot the reviewer; continuations keep that choice.",
            })))
        }

        ("POST", "/api/routing-profile") => {
            let profile: crate::routing_profile::RoutingProfile = match read_routing_json(req).await {
                Ok(profile) => profile,
                Err(error) => return Ok(routing_error(StatusCode::BAD_REQUEST, error)),
            };
            if let Err(error) = profile.validate() {
                return Ok(routing_error(StatusCode::BAD_REQUEST, error));
            }
            let store = Arc::clone(state.review_service.preferences());
            let saved = profile.clone();
            match tokio::task::spawn_blocking(move || store.update_profile(saved)).await
                .map_err(anyhow::Error::from).and_then(|result| result) {
                Ok(()) => {
                    let code = match profile.main.backend() { "cloud" => 1, "local" => 2, _ => 0 };
                    state.routing_mode.store(code, AtomicOrdering::Relaxed);
                    into_unsync(json_response(StatusCode::OK, &serde_json::json!({ "profile": profile })))
                }
                Err(error) => routing_error(StatusCode::INTERNAL_SERVER_ERROR, error),
            }
        }

        ("GET", "/api/routing-models") => {
            into_unsync(json_response(StatusCode::OK, &state.router.model_catalog().await))
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

        // ── Code-review master switch API (FR-A) ───────────────────────────
        // Default on. When off, review requests short-circuit to a terminal
        // `disabled` result without contacting any model. GET is readable;
        // POST is gated by the local-only destructive guard above.
        ("GET", "/api/review/enabled") => {
            into_unsync(json_response(StatusCode::OK, &serde_json::json!({
                "enabled": state.code_review_enabled.load(AtomicOrdering::Relaxed),
            })))
        }

        ("POST", "/api/review/enabled") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let val: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();
            if let Some(enabled) = val.get("enabled").and_then(|v| v.as_bool()) {
                state.code_review_enabled.store(enabled, AtomicOrdering::Relaxed);
                // Persist BOTH live flags (full snapshot) so the choice survives a
                // restart and never clobbers the sibling flag. Write failure is
                // logged, not fatal — the in-memory switch already took effect.
                if let Err(e) = crate::review::runtime_state::save_state(
                    &crate::review::runtime_state::state_path(),
                    enabled,
                    state.pr_guidelines_enabled.load(AtomicOrdering::Relaxed),
                ) {
                    tracing::warn!(error = %e, "Failed to persist review_runtime_state.json");
                }
            }
            into_unsync(json_response(StatusCode::OK, &serde_json::json!({
                "enabled": state.code_review_enabled.load(AtomicOrdering::Relaxed),
            })))
        }

        // ── FR-D: PR-generation guideline toggle ───────────────────────────
        ("GET", "/api/review/pr-guidelines") => {
            into_unsync(json_response(StatusCode::OK, &serde_json::json!({
                "enabled": state.pr_guidelines_enabled.load(AtomicOrdering::Relaxed),
            })))
        }

        ("POST", "/api/review/pr-guidelines") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let val: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();
            if let Some(enabled) = val.get("enabled").and_then(|v| v.as_bool()) {
                state.pr_guidelines_enabled.store(enabled, AtomicOrdering::Relaxed);
                // Full-snapshot persist of both live flags (see /enabled above).
                if let Err(e) = crate::review::runtime_state::save_state(
                    &crate::review::runtime_state::state_path(),
                    state.code_review_enabled.load(AtomicOrdering::Relaxed),
                    enabled,
                ) {
                    tracing::warn!(error = %e, "Failed to persist review_runtime_state.json");
                }
            }
            into_unsync(json_response(StatusCode::OK, &serde_json::json!({
                "enabled": state.pr_guidelines_enabled.load(AtomicOrdering::Relaxed),
            })))
        }

        // ── Review status + verdict ledger (design G2/G4 / H8) ─────────────
        // Local-only: audit toggles + recent review-outcome ledger events.
        ("GET", "/api/review/status") => {
            let is_local = peer_addr.ip().is_loopback() || peer_addr.port() == 0;
            let resp = if !is_local {
                json_response(StatusCode::FORBIDDEN, &ErrorResponse {
                    error: "review status is local-only".into(),
                })
            } else {
                let cfg = state.review_service.get_config();
                let events = crate::review::ledger::recent(
                    &crate::review::ledger::ledger_path(),
                    50,
                );
                json_response(StatusCode::OK, &serde_json::json!({
                    "code_review_enabled": state.code_review_enabled.load(AtomicOrdering::Relaxed),
                    "hankndory_integration": cfg.hankndory_integration,
                    "design_doc_path": cfg.design_doc_path,
                    "recent_events": events,
                }))
            };
            into_unsync(resp)
        }

        // ── Design-aware review approval (design G1 / H5) ──────────────────
        ("POST", "/api/review/approve-design") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let val: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();
            let project_dir = val.get("project_dir").and_then(|v| v.as_str()).unwrap_or("");
            let resp = match val.get("path").and_then(|v| v.as_str()) {
                None => json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
                    error: "Missing \"path\" (repo-relative design doc under docs/design/)".into(),
                }),
                Some(path) => review_approve_design(project_dir, path),
            };
            into_unsync(resp)
        }

        // Report the current design-approval outcome without recording anything.
        ("POST", "/api/review/design-status") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let val: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();
            let project_dir = val.get("project_dir").and_then(|v| v.as_str()).unwrap_or("");
            let resp = match val.get("path").and_then(|v| v.as_str()) {
                None => json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
                    error: "Missing \"path\"".into(),
                }),
                Some(path) => review_design_status(project_dir, path),
            };
            into_unsync(resp)
        }

        // ── Toolboxes API (all llama-* toolbox containers) ──────────────────
        ("GET", "/api/toolboxes") => {
            let resp = toolboxes_list().await;
            into_unsync(resp)
        }

        // ── Generalized toolbox catalog/container API (PR2, all 5 supported
        // backends: llama_cpp/ds4/halogen/vllm/r9v — see design doc §5c) ────
        ("GET", "/api/toolbox-catalog") => {
            let resp = toolbox_catalog_response().await;
            into_unsync(resp)
        }

        ("GET", "/api/toolbox-models") => {
            let resp = toolbox_models_response().await;
            into_unsync(resp)
        }

        ("GET", "/api/toolbox-containers") => {
            let resp = toolbox_containers_list().await;
            into_unsync(resp)
        }

        ("POST", "/api/toolbox-containers") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let val: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();
            let resp = match val.get("toolbox_id").and_then(|v| v.as_str()) {
                Some(id) => create_toolbox_container(&state, id).await,
                None => json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
                    error: "Missing \"toolbox_id\" in request body".into(),
                }),
            };
            into_unsync(resp)
        }

        ("POST", p) if p.starts_with("/api/toolbox-containers/") && p.ends_with("/update") => {
            let name = p.trim_start_matches("/api/toolbox-containers/").trim_end_matches("/update").trim_end_matches('/');
            let resp = update_toolbox_container(&state, name).await;
            into_unsync(resp)
        }

        ("POST", p) if p.starts_with("/api/toolbox-containers/") && p.ends_with("/delete") => {
            let name = p.trim_start_matches("/api/toolbox-containers/").trim_end_matches("/delete").trim_end_matches('/');
            let resp = delete_toolbox_container(&state, name).await;
            into_unsync(resp)
        }

        ("POST", p) if p.starts_with("/api/toolbox-containers/") && p.ends_with("/adopt") => {
            let name = p.trim_start_matches("/api/toolbox-containers/").trim_end_matches("/adopt").trim_end_matches('/');
            let resp = adopt_toolbox_container(&state, name).await;
            into_unsync(resp)
        }

        // ── PR6: model-download orchestration (§10) ──────────────────────────
        ("GET", "/api/model-downloads/status") => {
            let resp = model_downloads_status_response().await;
            into_unsync(resp)
        }

        ("POST", "/api/model-downloads/verify") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let resp = model_downloads_verify_response(&body_bytes).await;
            into_unsync(resp)
        }

        ("GET", "/api/model-downloads") => {
            let resp = model_downloads_list_response(&state).await;
            into_unsync(resp)
        }

        ("POST", "/api/model-downloads") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let resp = model_downloads_start_response(&state, &body_bytes).await;
            into_unsync(resp)
        }

        ("POST", p) if p.starts_with("/api/model-downloads/") && p.ends_with("/cancel") => {
            let id = p.trim_start_matches("/api/model-downloads/").trim_end_matches("/cancel").trim_end_matches('/');
            let resp = model_downloads_cancel_response(&state, id).await;
            into_unsync(resp)
        }

        ("GET", p) if p.starts_with("/api/model-downloads/") => {
            let id = p.trim_start_matches("/api/model-downloads/").trim_end_matches('/');
            let resp = model_downloads_get_response(&state, id).await;
            into_unsync(resp)
        }

        // ── PR7: ds4 Server Mode (§11/§12) ────────────────────────────────────
        ("GET", "/api/server-mode/ds4/status") => {
            let resp = server_mode_ds4_status_response().await;
            into_unsync(resp)
        }

        ("POST", "/api/server-mode/ds4/start") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let resp = server_mode_ds4_start_response(&state, &body_bytes).await;
            into_unsync(resp)
        }

        ("POST", "/api/server-mode/ds4/stop") => {
            let resp = server_mode_ds4_stop_response(&state).await;
            into_unsync(resp)
        }

        // ── PR8: halogen Server Mode (§13) ────────────────────────────────────
        ("GET", "/api/server-mode/halogen/status") => {
            let resp = server_mode_halogen_status_response().await;
            into_unsync(resp)
        }

        ("POST", "/api/server-mode/halogen/start") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let resp = server_mode_halogen_start_response(&state, &body_bytes).await;
            into_unsync(resp)
        }

        ("POST", "/api/server-mode/halogen/stop") => {
            let resp = server_mode_halogen_stop_response(&state).await;
            into_unsync(resp)
        }

        // ── PR9: vllm Server Mode (§14) ────────────────────────────────────────
        ("GET", "/api/server-mode/vllm/status") => {
            let resp = server_mode_vllm_status_response().await;
            into_unsync(resp)
        }

        ("POST", "/api/server-mode/vllm/start") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let resp = server_mode_vllm_start_response(&state, &body_bytes).await;
            into_unsync(resp)
        }

        ("POST", "/api/server-mode/vllm/stop") => {
            let resp = server_mode_vllm_stop_response(&state).await;
            into_unsync(resp)
        }

        ("POST", "/api/server-mode/vllm/cache-paths") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let resp = server_mode_vllm_cache_paths_response(&body_bytes).await;
            into_unsync(resp)
        }

        // ── PR11: r9v Server Mode (§15/§15a) ───────────────────────────────────
        ("GET", "/api/server-mode/r9v/status") => {
            let resp = server_mode_r9v_status_response().await;
            into_unsync(resp)
        }

        ("POST", "/api/server-mode/r9v/start") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let resp = server_mode_r9v_start_response(&state, &body_bytes).await;
            into_unsync(resp)
        }

        ("POST", "/api/server-mode/r9v/stop") => {
            let resp = server_mode_r9v_stop_response(&state).await;
            into_unsync(resp)
        }

        ("POST", "/api/server-mode/r9v/paths") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let resp = server_mode_r9v_paths_response(&body_bytes).await;
            into_unsync(resp)
        }

        ("POST", "/api/model-downloads/r9v/prepare-ple") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let resp = model_downloads_prepare_ple_response(&state, &body_bytes).await;
            into_unsync(resp)
        }

        // ── PR11a: serving-identity registry (§16) ──────────────────────────
        ("GET", "/api/serving-identities") => {
            let resp = serving_identities_response(&state).await;
            into_unsync(resp)
        }

        // ── PR3: cockpit config.json Phase-1 read + explicit apply (§4) ─────
        ("GET", "/api/cockpit-config") => {
            let resp = cockpit_config_status_response().await;
            into_unsync(resp)
        }

        ("POST", "/api/cockpit-config/default-toolbox") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let resp = apply_cockpit_default_toolbox(&body_bytes).await;
            into_unsync(resp)
        }

        ("POST", "/api/cockpit-config/active-platform") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let resp = apply_cockpit_active_platform(&body_bytes).await;
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
                    Ok(mut config) => {
                        if let Err(error) = config.review.validate() {
                            return Ok(routing_error(StatusCode::BAD_REQUEST, error));
                        }
                        if let Err(error) = crate::config::validate(&mut config, &state.config_path) {
                            return Ok(routing_error(StatusCode::BAD_REQUEST, error.to_string()));
                        }
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

/// FR-D marker that prefixes the injected guideline. Reserved: any system
/// message already containing this string suppresses re-injection (idempotency).
const PR_GUIDELINE_MARKER: &str = "<!--BRAINROUTER:PR-GUIDELINES v1-->";

/// FR-D PR-generation guideline injected (when the toggle is on) into agent
/// proxy requests. Instructs the coding agent how to structure a PR's commits
/// and to surface explicit notes whenever it stretches or breaks a rule.
const PR_GUIDELINE_PROMPT: &str = "<!--BRAINROUTER:PR-GUIDELINES v1-->
# PR GENERATION GUIDELINE

When you create a pull request or a sequence of commits, follow these rules:

1. Split the change into commits by layer, function, or cross-cutting concern —
   one coherent, self-contained change per commit, never a single mixed dump.
2. Keep each commit human-reviewable in size and complexity: a small, focused
   diff a reviewer can fully understand in one sitting.
3. Make each commit individually deployable in sequence — build and tests pass at
   every commit, and no commit depends on a later one to be correct.
4. If you must stretch or break any of rules 1-3, add an explicit note in the PR
   description (and the relevant commit body) stating which rule, why, and the
   trade-off, so the reviewer can see exactly where and why the rules bent.";

/// FR-D: when `enabled`, insert the PR-generation guideline as one `system`
/// message after the contiguous leading run of `system` messages (index 0 if
/// none; end if all-system). Idempotent: skips if any system message already
/// carries the marker in string content. No-op when disabled ⇒ the request is
/// unchanged from the pre-FR-D path.
fn maybe_inject_pr_guidelines(request: &mut ChatCompletionRequest, enabled: bool) {
    if !enabled {
        return;
    }
    let already_present = request.messages.iter().any(|m| {
        m.role == "system"
            && m.content
                .as_ref()
                .and_then(serde_json::Value::as_str)
                .is_some_and(|s| s.contains(PR_GUIDELINE_MARKER))
    });
    if already_present {
        return;
    }
    let pos = request
        .messages
        .iter()
        .position(|m| m.role != "system")
        .unwrap_or(request.messages.len());
    request.messages.insert(
        pos,
        crate::types::ChatMessage {
            role: "system".to_string(),
            content: Some(serde_json::Value::String(PR_GUIDELINE_PROMPT.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        },
    );
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
    // FR-D: inject the PR-generation guideline before the fingerprint/registry so
    // in-flight and routing fingerprints see identical messages.
    maybe_inject_pr_guidelines(
        &mut request,
        state.pr_guidelines_enabled.load(AtomicOrdering::Relaxed),
    );
    // Router resolves managed defaults for both protocols and direct callers.
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
        request.max_tokens,
    );
    let (tx, rx) = tokio::sync::oneshot::channel();
    let routing_task = tokio::spawn(async move {
        let result = tokio::select! {
            result = state.router.route_tagged(request, session_id, cwd, user_agent) => result,
            _ = handle.cancelled() => Err(anyhow::anyhow!("Request cancelled")),
        };
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

    let deferred = DeferredStream::new(
        rx,
        routing_task.abort_handle(),
        KEEPALIVE_INTERVAL,
        DEFERRED_STREAM_TIMEOUT,
        StreamFormat::OpenAi,
    );
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
    // FR-D: inject before the in-flight fingerprint (same rationale as OpenAI).
    maybe_inject_pr_guidelines(
        &mut oai_request,
        state.pr_guidelines_enabled.load(AtomicOrdering::Relaxed),
    );
    // Spawn routing so SSE headers are returned immediately (same rationale as OpenAI path).
    let handle = state.inflight.register(
        "POST /v1/messages".to_string(),
        oai_request.model.clone(),
        user_agent.clone(),
        peer_addr.to_string(),
        session_id.clone().unwrap_or_default(),
        crate::router::conversation_fingerprint(&oai_request),
        0,
        oai_request.max_tokens,
    );
    let (tx, rx) = tokio::sync::oneshot::channel();
    let routing_task = tokio::spawn(async move {
        let result = tokio::select! {
            result = state.router.route_tagged(oai_request, session_id, cwd, user_agent) => result,
            _ = handle.cancelled() => Err(anyhow::anyhow!("Request cancelled")),
        };
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

    let deferred = DeferredStream::new(
        rx,
        routing_task.abort_handle(),
        KEEPALIVE_INTERVAL,
        DEFERRED_STREAM_TIMEOUT,
        StreamFormat::Anthropic,
    );
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
/// Approve the current design document by binding its SHA-256 in the
/// approved-record. Fails if the doc cannot be safely loaded or is not marked
/// approved-for-implementation with a human approval.
fn review_approve_design(project_dir: &str, path: &str) -> Response<Full<Bytes>> {
    use crate::review::design_doc;
    match design_doc::load_design_doc(project_dir, path) {
        Err(e) => json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
            error: format!("Cannot load design document: {e}"),
        }),
        Ok(doc) if !doc.doc_marks_approved() => json_response(StatusCode::CONFLICT, &ErrorResponse {
            error: "Design is not marked approved-for-implementation with a human approval; cannot record approval.".into(),
        }),
        Ok(doc) => match design_doc::record_approval(
            &design_doc::approvals_path(),
            &doc.repo_rel_path,
            &doc.sha256,
            doc.approved_version.clone(),
            "dashboard",
        ) {
            Ok(()) => json_response(StatusCode::OK, &serde_json::json!({
                "approved": true,
                "path": doc.repo_rel_path,
                "sha256": doc.sha256,
                "version": doc.approved_version,
            })),
            Err(e) => json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Could not record approval: {e}"),
            }),
        },
    }
}

/// Report the current design-approval outcome without recording anything.
fn review_design_status(project_dir: &str, path: &str) -> Response<Full<Bytes>> {
    use crate::review::design_doc::{self, ApprovalOutcome};
    match design_doc::load_design_doc(project_dir, path) {
        Err(e) => json_response(StatusCode::OK, &serde_json::json!({
            "state": "unavailable",
            "detail": e.to_string(),
        })),
        Ok(doc) => {
            let approvals = design_doc::load_approvals(&design_doc::approvals_path());
            let outcome = design_doc::evaluate_approval(&doc, approvals.get(&doc.repo_rel_path));
            let (state, detail) = match outcome {
                ApprovalOutcome::Approved { .. } => ("approved", String::new()),
                ApprovalOutcome::NotApproved(reason) => ("not_approved", reason),
            };
            json_response(StatusCode::OK, &serde_json::json!({
                "state": state,
                "detail": detail,
                "path": doc.repo_rel_path,
                "sha256": doc.sha256,
                "version": doc.approved_version,
                "doc_marks_approved": doc.doc_marks_approved(),
            }))
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

// ─── Generalized toolbox-catalog-driven container management (PR2) ─────────
//
// Everything below generalizes `toolboxes_list()`/`upgrade_toolbox()` above
// (kept, unmodified, for backward compatibility — see design doc §5c) from
// one hardcoded llama_cpp/Vulkan image to every toolbox in the vendored
// catalog, across all 5 supported backends. See
// `docs/design/ai-toolbox-cockpit-integration.md` §5c for the API contract
// and the decisions recorded while implementing this.

/// Podman label brainrouter attaches to every toolbox container it creates,
/// so it can tell "brainrouter made this" apart from "a container that
/// happens to share a catalog `container_name` but was made some other way
/// (e.g. by cockpit directly)". See §5c: attaching these via `toolbox
/// create --label` is unverified in this environment (Open question 6) and
/// deliberately allowed to fail loudly rather than being silently skipped.
const LABEL_MANAGED: &str = "io.brainrouter.managed";
const LABEL_CATALOG_ID: &str = "io.brainrouter.catalog_id";
const LABEL_CATALOG_REVISION: &str = "io.brainrouter.catalog_revision";

/// Loads and type-parses the vendored toolbox catalog, restricted to
/// entries brainrouter can act on. Returns a human-readable error string
/// (used directly in `ErrorResponse`s) rather than a custom error type,
/// matching this file's existing preference for cheap, situational errors
/// over a dedicated error enum for read paths that should essentially never
/// fail (the catalog is embedded at compile time and covered by unit tests).
fn load_typed_toolbox_catalog() -> Result<ToolboxCatalog, String> {
    let vendored = toolbox_catalog::load_vendored_catalog();
    if !vendored.report.is_ok() {
        warn!(
            errors = ?vendored.report.errors,
            warnings = ?vendored.report.warnings,
            "Vendored toolbox catalog has structural validation issues"
        );
    }
    let (toolboxes, _models) = vendored
        .typed()
        .map_err(|e| format!("failed to parse vendored toolbox catalog: {e}"))?;
    Ok(toolboxes)
}

/// Whether a podman container carries brainrouter's ownership label.
/// Absence (including "no such container") is treated as unmanaged, not an
/// error — callers already know whether the container exists from `podman
/// ps`, this only answers the ownership question for ones that do.
async fn toolbox_container_is_managed(name: &str) -> bool {
    let out = tokio::process::Command::new("podman")
        .args(["inspect", "--format", &format!("{{{{ index .Config.Labels \"{LABEL_MANAGED}\" }}}}"), name])
        .output()
        .await;
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim() == "true",
        _ => false,
    }
}

/// `Some(repo)` if `image` is hosted on Docker Hub in the implicit
/// `docker.io/<namespace>/<name>` form — the only registry shape this
/// freshness check knows how to query (§3, critic finding #12). Anything
/// else (e.g. `ghcr.io/...`) gets an explicit "freshness unavailable"
/// status downstream rather than a guessed Hub-only request that would
/// just fail or, worse, hit the wrong repo on Hub.
fn hub_repo_for_image(image: &str) -> Option<&str> {
    let (repo, _tag) = split_repo_tag(image);
    repo.strip_prefix("docker.io/")
}

/// How long a per-repository Docker Hub tag-listing is cached before being
/// re-fetched. Looping over every vendored toolbox's image on every
/// dashboard poll (today every 30s) without this would multiply outbound
/// Hub requests by the number of distinct repos in the catalog, which is
/// exactly the scaling problem §3/critic finding #12 flagged.
const HUB_TAG_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(15 * 60);

/// repo -> (fetched_at, tag -> last-push-date).
type HubTagCache = std::collections::HashMap<String, (std::time::Instant, std::collections::HashMap<String, String>)>;

static HUB_TAG_CACHE: LazyLock<std::sync::Mutex<HubTagCache>> =
    LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Docker Hub tag -> last-push-date map for one repo, generalizing
/// `hub_toolbox_tag_dates()`'s single hardcoded repo to any repo, cached
/// per-repo, and — unlike the original — with an explicit request timeout
/// (the original had none; see §3, critic finding #12).
async fn hub_repo_tag_dates(repo: &str) -> std::collections::HashMap<String, String> {
    if let Ok(cache) = HUB_TAG_CACHE.lock() {
        if let Some((fetched_at, dates)) = cache.get(repo) {
            if fetched_at.elapsed() < HUB_TAG_CACHE_TTL {
                return dates.clone();
            }
        }
    }

    let mut map = std::collections::HashMap::new();
    let url = format!("https://hub.docker.com/v2/repositories/{repo}/tags?page_size=100");
    let resp = VERSION_CLIENT
        .get(&url)
        .timeout(std::time::Duration::from_secs(5))
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
                    map.insert(name.to_string(), pushed.to_string());
                }
            }
        }
    }

    if let Ok(mut cache) = HUB_TAG_CACHE.lock() {
        cache.insert(repo.to_string(), (std::time::Instant::now(), map.clone()));
    }
    map
}

/// Freshness (`local_created`, `latest_created`, `update_available`) for a
/// set of distinct images, fetching each distinct Hub repo at most once
/// (bounded concurrency, not one call per container) rather than once per
/// image/container as a naive generalization of `toolboxes_list()` would.
async fn image_freshness_map(
    images: &std::collections::HashSet<String>,
) -> std::collections::HashMap<String, (String, String, bool)> {
    let repos: std::collections::HashSet<&str> =
        images.iter().filter_map(|i| hub_repo_for_image(i)).collect();

    let semaphore = Arc::new(tokio::sync::Semaphore::new(4));
    let mut tasks = Vec::new();
    for repo in repos {
        let repo = repo.to_string();
        let semaphore = Arc::clone(&semaphore);
        tasks.push(tokio::spawn(async move {
            let _permit = semaphore.acquire_owned().await.ok();
            let dates = hub_repo_tag_dates(&repo).await;
            (repo, dates)
        }));
    }
    let mut repo_tag_map: std::collections::HashMap<String, std::collections::HashMap<String, String>> =
        std::collections::HashMap::new();
    for task in tasks {
        if let Ok((repo, dates)) = task.await {
            repo_tag_map.insert(repo, dates);
        }
    }

    let mut result = std::collections::HashMap::new();
    for image in images {
        let local = image_created_date(image).await.unwrap_or_default();
        let (_repo, tag) = split_repo_tag(image);
        let latest = hub_repo_for_image(image)
            .and_then(|r| repo_tag_map.get(r))
            .and_then(|m| m.get(tag))
            .cloned()
            .unwrap_or_default();
        let update_available = !local.is_empty() && !latest.is_empty() && local < latest;
        result.insert(image.clone(), (local, latest, update_available));
    }
    result
}

/// `GET /api/toolbox-catalog` — the vendored catalog, restricted to the 5
/// supported backends (comfyui entries omitted entirely, not just flagged).
pub async fn toolbox_catalog_response() -> Response<Full<Bytes>> {
    let catalog = match load_typed_toolbox_catalog() {
        Ok(c) => c,
        Err(e) => {
            error!(error = %e, "Failed to load toolbox catalog");
            return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse { error: e });
        }
    };
    let toolboxes: Vec<&ToolboxDefinition> = catalog
        .toolboxes
        .iter()
        .filter(|t| t.supported_backend().is_some())
        .collect();
    json_response(StatusCode::OK, &serde_json::json!({
        "schema_version": catalog.schema_version,
        "toolboxes": toolboxes,
        "platforms": catalog.platforms,
    }))
}

/// `GET /api/toolbox-models` — the vendored model catalog, restricted to
/// the 5 supported backends.
pub async fn toolbox_models_response() -> Response<Full<Bytes>> {
    let vendored = toolbox_catalog::load_vendored_catalog();
    let (_toolboxes, models) = match vendored.typed() {
        Ok(v) => v,
        Err(e) => {
            error!(error = %e, "Failed to load toolbox model catalog");
            return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("failed to parse vendored model catalog: {e}"),
            });
        }
    };
    let backends: Vec<_> = models
        .backends
        .iter()
        .filter(|b| SupportedServingBackend::try_from(&b.backend).is_ok())
        .collect();
    json_response(StatusCode::OK, &serde_json::json!({
        "schema_version": models.schema_version,
        "backends": backends,
    }))
}

/// `GET /api/toolbox-containers` — generalizes `toolboxes_list()` above
/// from one hardcoded `llama-*` name prefix to every catalog toolbox across
/// all 5 supported backends. A podman container whose name matches no
/// catalog `container_name` is omitted, same effective behavior as the
/// old prefix filter, now catalog-driven instead of hardcoded.
pub async fn toolbox_containers_list() -> Response<Full<Bytes>> {
    let catalog = match load_typed_toolbox_catalog() {
        Ok(c) => c,
        Err(e) => {
            error!(error = %e, "Failed to load toolbox catalog");
            return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse { error: e });
        }
    };
    let by_container_name: std::collections::HashMap<&str, &ToolboxDefinition> = catalog
        .toolboxes
        .iter()
        .filter(|t| t.supported_backend().is_some())
        .map(|t| (t.container_name.as_str(), t))
        .collect();

    let containers = tokio::process::Command::new("podman")
        .args(["ps", "-a", "--format", "{{.Names}}|{{.Image}}|{{.Status}}|{{.CreatedAt}}"])
        .output()
        .await;

    // (container_name, toolbox, image, status, created_at)
    let mut matched: Vec<(String, &ToolboxDefinition, String, String, String)> = Vec::new();
    if let Ok(o) = containers {
        for line in String::from_utf8_lossy(&o.stdout).lines() {
            let mut parts = line.splitn(4, '|');
            let (Some(name), Some(image), Some(status), Some(created_at)) =
                (parts.next(), parts.next(), parts.next(), parts.next())
            else {
                continue;
            };
            let Some(tb) = by_container_name.get(name) else {
                continue;
            };
            matched.push((name.to_string(), tb, image.to_string(), status.to_string(), created_at.to_string()));
        }
    }
    matched.sort_by(|a, b| a.0.cmp(&b.0));

    let images: std::collections::HashSet<String> = matched.iter().map(|m| m.2.clone()).collect();
    let freshness = image_freshness_map(&images).await;

    let mut list: Vec<serde_json::Value> = Vec::with_capacity(matched.len());
    for (name, tb, image, status, created_at) in &matched {
        let managed = toolbox_container_is_managed(name).await;
        let (local_created, latest_created, update_available) =
            freshness.get(image).cloned().unwrap_or_default();
        list.push(serde_json::json!({
            "container_name": name,
            "toolbox_id": tb.id,
            "backend": tb.backend.as_str(),
            "image": image,
            "running": status.starts_with("Up"),
            "status": status,
            "created_at": created_at,
            "local_created": local_created,
            "latest_created": latest_created,
            "update_available": update_available,
            "managed": managed,
        }));
    }

    json_response(StatusCode::OK, &serde_json::json!({ "containers": list }))
}

/// Acquires the per-container-name mutation lock, creating one on first use.
/// The outer `std::sync::Mutex` only ever guards inserting a new per-name
/// entry (a fast, non-blocking operation); the actual create/update/delete/
/// adopt work is done while holding the returned async guard, so a second
/// concurrent request for the *same* name queues behind it, while different
/// names never contend (§5c).
async fn lock_toolbox_container(state: &AppState, name: &str) -> tokio::sync::OwnedMutexGuard<()> {
    let entry = {
        let mut locks = state.toolbox_container_locks.lock().unwrap_or_else(|e| e.into_inner());
        Arc::clone(locks.entry(name.to_string()).or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))))
    };
    entry.lock_owned().await
}

/// Shared create/update/adopt primitive: pulls (if `pull`) the toolbox's
/// catalog image, force-removes any existing container of that name, then
/// recreates it via `toolbox create` with brainrouter's ownership labels.
/// Assumes the caller already holds the per-container-name lock.
async fn recreate_toolbox_container(tb: &ToolboxDefinition, pull: bool) -> Response<Full<Bytes>> {
    let container = &tb.container_name;
    let image = &tb.image;

    if pull {
        let pull_out = tokio::process::Command::new("podman").args(["pull", image]).output().await;
        match pull_out {
            Ok(out) if out.status.success() => {}
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                error!(%stderr, %container, %image, "podman pull failed");
                return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                    error: format!("podman pull failed: {}", stderr.trim()),
                });
            }
            Err(e) => {
                error!(error = %e, %container, "Failed to exec podman pull");
                return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                    error: format!("Failed to exec podman: {}", e),
                });
            }
        }
    }

    // Force-remove any existing container of this name (idempotent: fine if absent).
    let _ = tokio::process::Command::new("toolbox").args(["rm", "--force", container]).output().await;

    let revision = toolbox_catalog::vendored_catalog_revision();
    let create = tokio::process::Command::new("toolbox")
        .args([
            "create",
            "--image",
            image,
            "--label",
            &format!("{LABEL_MANAGED}=true"),
            "--label",
            &format!("{LABEL_CATALOG_ID}={}", tb.id),
            "--label",
            &format!("{LABEL_CATALOG_REVISION}={revision}"),
            container,
        ])
        .output()
        .await;

    match create {
        Ok(out) if out.status.success() => json_response(StatusCode::OK, &serde_json::json!({
            "status": "ok",
            "message": format!("Toolbox container '{container}' created."),
        })),
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            error!(%stderr, %container, %image, "toolbox create failed");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!(
                    "toolbox create failed: {} (if this mentions an unrecognized --label flag, \
                     see design doc §5c / Open question 6 — toolbox's --label support is unverified)",
                    stderr.trim()
                ),
            })
        }
        Err(e) => {
            error!(error = %e, %container, "Failed to exec toolbox create");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Failed to exec toolbox: {}", e),
            })
        }
    }
}

/// `POST /api/toolbox-containers` — create a new container for `toolbox_id`.
pub async fn create_toolbox_container(state: &AppState, toolbox_id: &str) -> Response<Full<Bytes>> {
    let catalog = match load_typed_toolbox_catalog() {
        Ok(c) => c,
        Err(e) => return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse { error: e }),
    };
    let Some(tb) = catalog.toolbox_by_id(toolbox_id).filter(|t| t.supported_backend().is_some()) else {
        return json_response(StatusCode::NOT_FOUND, &ErrorResponse {
            error: format!("No such supported toolbox catalog id: {toolbox_id}"),
        });
    };
    let _guard = lock_toolbox_container(state, &tb.container_name).await;
    if toolbox_container_image(&tb.container_name).await.is_some() {
        return json_response(StatusCode::CONFLICT, &ErrorResponse {
            error: format!(
                "Container '{}' already exists — use update or adopt instead.",
                tb.container_name
            ),
        });
    }
    recreate_toolbox_container(tb, false).await
}

/// `POST /api/toolbox-containers/{name}/update` — pull latest image + recreate.
pub async fn update_toolbox_container(state: &AppState, container_name: &str) -> Response<Full<Bytes>> {
    let catalog = match load_typed_toolbox_catalog() {
        Ok(c) => c,
        Err(e) => return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse { error: e }),
    };
    let Some(tb) = catalog
        .toolboxes
        .iter()
        .find(|t| t.container_name == container_name && t.supported_backend().is_some())
    else {
        return json_response(StatusCode::NOT_FOUND, &ErrorResponse {
            error: format!("No catalog toolbox with container_name: {container_name}"),
        });
    };
    let _guard = lock_toolbox_container(state, container_name).await;
    recreate_toolbox_container(tb, true).await
}

/// `POST /api/toolbox-containers/{name}/adopt` — recreate-in-place (no
/// pull) to attach ownership labels to a pre-existing unmanaged container.
/// See §5c for why this can't be a label-only no-op.
pub async fn adopt_toolbox_container(state: &AppState, container_name: &str) -> Response<Full<Bytes>> {
    let catalog = match load_typed_toolbox_catalog() {
        Ok(c) => c,
        Err(e) => return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse { error: e }),
    };
    let Some(tb) = catalog
        .toolboxes
        .iter()
        .find(|t| t.container_name == container_name && t.supported_backend().is_some())
    else {
        return json_response(StatusCode::NOT_FOUND, &ErrorResponse {
            error: format!("No catalog toolbox with container_name: {container_name}"),
        });
    };
    let _guard = lock_toolbox_container(state, container_name).await;
    if toolbox_container_image(container_name).await.is_none() {
        return json_response(StatusCode::NOT_FOUND, &ErrorResponse {
            error: format!("No such container: {container_name}"),
        });
    }
    if toolbox_container_is_managed(container_name).await {
        return json_response(StatusCode::CONFLICT, &ErrorResponse {
            error: format!("Container '{container_name}' is already brainrouter-managed."),
        });
    }
    recreate_toolbox_container(tb, false).await
}

/// `POST /api/toolbox-containers/{name}/delete` — idempotent force-remove.
pub async fn delete_toolbox_container(state: &AppState, container_name: &str) -> Response<Full<Bytes>> {
    let _guard = lock_toolbox_container(state, container_name).await;
    if toolbox_container_image(container_name).await.is_none() {
        return json_response(StatusCode::OK, &serde_json::json!({
            "status": "ok",
            "message": "already removed",
        }));
    }
    let remove = tokio::process::Command::new("toolbox").args(["rm", "--force", container_name]).output().await;
    match remove {
        Ok(out) if out.status.success() => json_response(StatusCode::OK, &serde_json::json!({
            "status": "ok",
            "message": format!("Toolbox container '{container_name}' removed."),
        })),
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            error!(%stderr, %container_name, "toolbox rm failed");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("toolbox rm failed: {}", stderr.trim()),
            })
        }
        Err(e) => {
            error!(error = %e, %container_name, "Failed to exec toolbox rm");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Failed to exec toolbox: {}", e),
            })
        }
    }
}

// ── PR6: model-download orchestration (§10) ──────────────────────────────

/// `GET /api/model-downloads/status` — read-only local-presence sweep
/// across every download-capable backend's catalog entries (ds4/halogen/
/// r9v; llama_cpp and vllm are excluded — see `model_downloads.rs` docs).
pub async fn model_downloads_status_response() -> Response<Full<Bytes>> {
    match crate::model_downloads::local_presence_snapshot() {
        Ok(presence) => json_response(StatusCode::OK, &serde_json::json!({ "models": presence })),
        Err(e) => {
            error!(error = %e, "Failed to compute model-download presence snapshot");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse { error: e })
        }
    }
}

/// `GET /api/model-downloads` — list all known jobs (most-recent-first,
/// bounded by the registry's own `MAX_JOB_HISTORY`).
pub async fn model_downloads_list_response(state: &AppState) -> Response<Full<Bytes>> {
    let jobs = state.model_downloads.list(crate::model_downloads::MAX_JOB_HISTORY).await;
    json_response(StatusCode::OK, &serde_json::json!({ "jobs": jobs }))
}

/// `GET /api/model-downloads/{id}` — poll a single job's current state.
pub async fn model_downloads_get_response(state: &AppState, id: &str) -> Response<Full<Bytes>> {
    match state.model_downloads.get(id).await {
        Ok(job) => json_response(StatusCode::OK, &job),
        Err(e) => json_response(StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), &ErrorResponse {
            error: e.message().to_string(),
        }),
    }
}

/// `POST /api/model-downloads` — start a new download job. Body:
/// `{"backend": "...", "model_id": "...", "quant_pattern": "..." (llama_cpp only)}`.
pub async fn model_downloads_start_response(state: &AppState, body: &Bytes) -> Response<Full<Bytes>> {
    let request: crate::model_downloads::StartDownloadRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            return json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
                error: format!("invalid request body: {e}"),
            });
        }
    };
    match state.model_downloads.start(request).await {
        Ok(job) => json_response(StatusCode::OK, &job),
        Err(e) => json_response(StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), &ErrorResponse {
            error: e.message().to_string(),
        }),
    }
}

/// `POST /api/model-downloads/{id}/cancel` — request cancellation of a
/// running (or queued) job. Idempotent-ish: cancelling an already-terminal
/// job returns a Conflict, not a silent no-op, so the caller's UI can
/// surface it plainly.
pub async fn model_downloads_cancel_response(state: &AppState, id: &str) -> Response<Full<Bytes>> {
    match state.model_downloads.cancel(id).await {
        Ok(job) => json_response(StatusCode::OK, &job),
        Err(e) => json_response(StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), &ErrorResponse {
            error: e.message().to_string(),
        }),
    }
}

/// `POST /api/model-downloads/verify` — explicit SHA256 verification pass,
/// only meaningful for backends whose catalog entries carry a `sha256`
/// (today, only `r9v`). Body: `{"backend": "...", "model_id": "..."}`.
pub async fn model_downloads_verify_response(body: &Bytes) -> Response<Full<Bytes>> {
    let val: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
                error: format!("invalid request body: {e}"),
            });
        }
    };
    let (Some(backend_raw), Some(model_id)) = (
        val.get("backend").and_then(|v| v.as_str()),
        val.get("model_id").and_then(|v| v.as_str()),
    ) else {
        return json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
            error: "Missing \"backend\" or \"model_id\" in request body".into(),
        });
    };
    let catalog_id = crate::toolbox_catalog::types::CatalogBackendId::from_str(backend_raw);
    let Ok(backend) = SupportedServingBackend::try_from(&catalog_id) else {
        return json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
            error: format!("unsupported or unknown backend `{backend_raw}`"),
        });
    };
    match crate::model_downloads::verify_checksums(backend, model_id).await {
        Ok(results) => json_response(StatusCode::OK, &serde_json::json!({ "files": results })),
        Err(e) => json_response(StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), &ErrorResponse {
            error: e.message().to_string(),
        }),
    }
}

// ── PR3: cockpit config.json Phase-1 read + explicit "apply" write (§4) ────

/// `GET /api/cockpit-config` — read-only snapshot of cockpit's shared
/// config.json, including the resolved path/owner so a HOME/config-path
/// mismatch between brainrouter and an interactively-run cockpit is
/// visible in the dashboard rather than silent.
pub async fn cockpit_config_status_response() -> Response<Full<Bytes>> {
    json_response(StatusCode::OK, &crate::cockpit_config::load())
}

#[derive(serde::Deserialize)]
struct ApplyDefaultToolboxRequest {
    backend_id: String,
    platform_id: String,
    toolbox_id: String,
}

/// `POST /api/cockpit-config/default-toolbox` — the explicit, user-
/// triggered single-write "apply" action for one backend's default toolbox
/// on one platform. Only ever touches
/// `backends.<backend_id>.default_toolboxes.<platform_id>`; every other key
/// in the file round-trips untouched (§4).
pub async fn apply_cockpit_default_toolbox(body: &[u8]) -> Response<Full<Bytes>> {
    let req: ApplyDefaultToolboxRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            return json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
                error: format!("invalid request body: {e}"),
            });
        }
    };
    match crate::cockpit_config::apply_default_toolbox(&req.backend_id, &req.platform_id, &req.toolbox_id) {
        Ok(()) => json_response(StatusCode::OK, &serde_json::json!({ "status": "ok" })),
        Err(crate::cockpit_config::ApplyError::NotAvailable) => {
            json_response(StatusCode::NOT_FOUND, &ErrorResponse { error: crate::cockpit_config::ApplyError::NotAvailable.to_string() })
        }
        Err(e) => {
            error!(error = %e, "Failed to apply cockpit default-toolbox setting");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse { error: e.to_string() })
        }
    }
}

#[derive(serde::Deserialize)]
struct ApplyActivePlatformRequest {
    platform_id: String,
}

/// `POST /api/cockpit-config/active-platform` — same contract as
/// [`apply_cockpit_default_toolbox`], for the top-level `active_platform` key.
pub async fn apply_cockpit_active_platform(body: &[u8]) -> Response<Full<Bytes>> {
    let req: ApplyActivePlatformRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            return json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
                error: format!("invalid request body: {e}"),
            });
        }
    };
    match crate::cockpit_config::apply_active_platform(&req.platform_id) {
        Ok(()) => json_response(StatusCode::OK, &serde_json::json!({ "status": "ok" })),
        Err(crate::cockpit_config::ApplyError::NotAvailable) => {
            json_response(StatusCode::NOT_FOUND, &ErrorResponse { error: crate::cockpit_config::ApplyError::NotAvailable.to_string() })
        }
        Err(e) => {
            error!(error = %e, "Failed to apply cockpit active-platform setting");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse { error: e.to_string() })
        }
    }
}


/// Fixed name for the transient container used to probe the installed llama.cpp version.
/// Combined with `--replace`, this lets a fresh check safely reuse (rather than collide with)
/// any leftover container of the same name from a prior run whose cleanup failed, and gives
/// operators a stable, greppable name instead of podman's randomly-assigned pet-names.
const PODMAN_VERSION_CHECK_CONTAINER_NAME: &str = "brainrouter-llama-version-check";

/// Outcome of [`run_with_timeout_and_cleanup`].
enum TimedCommandOutcome {
    Completed(std::process::Output),
    Failed(std::io::Error),
    TimedOut,
}

/// Runs `cmd` under `timeout`. `cmd` must already have `.kill_on_drop(true)` set by the caller.
///
/// `podman run --rm` only removes its own container when the podman client exits normally; if
/// the client is killed first (as `kill_on_drop` does when this future is dropped on timeout),
/// `--rm`'s cleanup never runs and the container is left running, orphaned, under whatever name
/// podman assigned it. On timeout this function therefore also runs a best-effort `cleanup`
/// command (e.g. `podman rm -f <name>`) to remove that container; the cleanup's own failure is
/// logged but never propagated, since this is already a best-effort fallback path.
async fn run_with_timeout_and_cleanup(
    mut cmd: tokio::process::Command,
    timeout: std::time::Duration,
    mut cleanup: tokio::process::Command,
) -> TimedCommandOutcome {
    match tokio::time::timeout(timeout, cmd.output()).await {
        Ok(Ok(out)) => TimedCommandOutcome::Completed(out),
        Ok(Err(e)) => TimedCommandOutcome::Failed(e),
        Err(_) => {
            if let Err(e) = cleanup.output().await {
                warn!(error = %e, "best-effort cleanup command failed after timeout");
            }
            TimedCommandOutcome::TimedOut
        }
    }
}

/// Compute local versions and "latest available" metadata for the /api/versions endpoint.
/// Called periodically by a background task in daemon.rs.
/// Cooperative lock checked before spawning any llama-server process for a
/// version probe. Written by `~/.local/bin/gpu-exclusive-lock acquire` around
/// GPU-exclusive confirmation/benchmark runs; a stray `llama-server --version`
/// container is enough to trip those runs' conflict guard. Treated as absent
/// once older than GPU_EXCLUSIVE_LOCK_MAX_AGE, so a crashed acquirer can't
/// wedge this off forever.
fn gpu_exclusive_lock_active() -> bool {
    const GPU_EXCLUSIVE_LOCK_MAX_AGE: std::time::Duration = std::time::Duration::from_secs(12 * 3600);
    let Some(home) = std::env::var_os("HOME") else { return false };
    let lock_path = std::path::Path::new(&home).join(".local/state/gpu-exclusive.lock");
    let Ok(meta) = std::fs::metadata(&lock_path) else { return false };
    let Ok(modified) = meta.modified() else { return true };
    modified.elapsed().map(|age| age < GPU_EXCLUSIVE_LOCK_MAX_AGE).unwrap_or(true)
}

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
    let toolbox_ver = if gpu_exclusive_lock_active() {
        "unknown".to_string()
    } else {
        let mut child = Command::new("podman");
        child
            .args([
                "run", "--rm", "--replace", "--name", PODMAN_VERSION_CHECK_CONTAINER_NAME,
                "docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv", "llama-server", "--version",
            ])
            .kill_on_drop(true);
        let mut cleanup = Command::new("podman");
        cleanup.args(["rm", "-f", PODMAN_VERSION_CHECK_CONTAINER_NAME]);

        match run_with_timeout_and_cleanup(
            child,
            std::time::Duration::from_secs(PODMAN_VERSION_TIMEOUT_SECS),
            cleanup,
        ).await {
            TimedCommandOutcome::Completed(out) => {
                let combined = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
                if let Some(line) = combined.lines().find(|l| l.contains("version:")) {
                    line.replace("version:", "").replace("built with", "").trim().to_string()
                } else if let Some(line) = combined.lines().find(|l| l.contains('(') && l.contains(')') && !l.contains("Error")) {
                    line.replace("built with", "").trim().to_string()
                } else {
                    "unknown".to_string()
                }
            }
            TimedCommandOutcome::Failed(e) => {
                error!(error = %e, "Failed to execute podman run for version check");
                "unknown".to_string()
            }
            TimedCommandOutcome::TimedOut => {
                warn!(
                    timeout_secs = PODMAN_VERSION_TIMEOUT_SECS,
                    container = PODMAN_VERSION_CHECK_CONTAINER_NAME,
                    "podman run timed out during llama.cpp version check; ran best-effort cleanup"
                );
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

// ── PR7: ds4 Server Mode (§11/§12) ────────────────────────────────────────

/// Shared helper for the 4 backends' `start_*_response()` handlers
/// (§16/PR11a): builds a [`crate::serving_identity::ServingIdentity`] from
/// an already-resolved toolbox/runtime-profile pair and registers it.
/// Called only after `start_*_server()` itself already succeeded — a
/// failure here (e.g. an unexpected catalog-resolution error for a
/// toolbox_id that just successfully started) is logged, not surfaced as
/// an HTTP error, since the server itself is genuinely running either way
/// (§16: registration is bookkeeping, never gates the start/stop action).
async fn register_serving_identity(
    state: &AppState,
    backend: crate::toolbox_catalog::SupportedServingBackend,
    container_name: &str,
    toolbox_id: String,
    profile: &crate::toolbox_catalog::RuntimeProfile,
    endpoint: String,
) {
    let identity = crate::serving_identity::ServingIdentity {
        toolbox_backend: backend.as_str(),
        compute_api: crate::server_mode::compute_api_for_runtime_profile(profile).to_string(),
        runtime_profile_id: toolbox_id,
        endpoint,
        openai_compatible: crate::serving_identity::openai_compatible_for_backend(backend),
        registered_at: chrono::Utc::now(),
    };
    state.serving_identities.register(container_name, identity).await;
}

/// `"http://host:port"`, normalizing an all-interfaces bind address
/// (`0.0.0.0`) to `127.0.0.1` — the dashboard/API reader needs a real
/// destination to (eventually) reach, and `0.0.0.0` is never a valid one
/// (§16).
fn serving_identity_endpoint(host: &str, port: u16) -> String {
    let host = if host == "0.0.0.0" { "127.0.0.1" } else { host };
    format!("http://{host}:{port}")
}

/// `GET /api/server-mode/ds4/status` — always reads live `podman inspect`
/// state (§12 item 5: no persisted registry).
pub async fn server_mode_ds4_status_response() -> Response<Full<Bytes>> {
    let status = crate::server_mode::ds4_server_status().await;
    json_response(StatusCode::OK, &status)
}

/// `POST /api/server-mode/ds4/start` — body:
/// `{"toolbox_id": "...", "model_id": "...", "ctx": <n>, "host": "...", "port": <n>, "custom_args": "..."}`.
pub async fn server_mode_ds4_start_response(state: &AppState, body: &Bytes) -> Response<Full<Bytes>> {
    let request: crate::server_mode::StartDs4ServerRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            return json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
                error: format!("invalid request body: {e}"),
            });
        }
    };
    match crate::server_mode::start_ds4_server(&request).await {
        Ok(()) => {
            if let Ok((_, profile)) = crate::server_mode::resolve_ds4_toolbox(&request.toolbox_id) {
                register_serving_identity(
                    state,
                    crate::toolbox_catalog::SupportedServingBackend::Ds4,
                    crate::server_mode::DS4_SERVER_CONTAINER_NAME,
                    request.toolbox_id.clone(),
                    &profile,
                    serving_identity_endpoint(&request.host, request.port),
                )
                .await;
            }
            let status = crate::server_mode::ds4_server_status().await;
            json_response(StatusCode::OK, &status)
        }
        Err(e) => json_response(StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), &ErrorResponse {
            error: e.message().to_string(),
        }),
    }
}

/// `POST /api/server-mode/ds4/stop` — graceful `podman stop` then `podman
/// rm -f`, idempotent if the container is already gone (§12 item 5).
pub async fn server_mode_ds4_stop_response(state: &AppState) -> Response<Full<Bytes>> {
    match crate::server_mode::stop_ds4_server().await {
        Ok(()) => {
            state.serving_identities.deregister(crate::server_mode::DS4_SERVER_CONTAINER_NAME).await;
            json_response(StatusCode::OK, &serde_json::json!({
                "status": "ok",
                "message": "ds4 server stopped.",
            }))
        }
        Err(e) => json_response(StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), &ErrorResponse {
            error: e.message().to_string(),
        }),
    }
}

// ── PR8: halogen Server Mode (§13) ────────────────────────────────────────

/// `GET /api/server-mode/halogen/status` — always reads live `podman
/// inspect` state, same no-persisted-registry contract as ds4's status
/// endpoint above.
pub async fn server_mode_halogen_status_response() -> Response<Full<Bytes>> {
    let status = crate::server_mode::halogen_server_status().await;
    json_response(StatusCode::OK, &status)
}

/// `POST /api/server-mode/halogen/start` — body:
/// `{"toolbox_id": "...", "bundle_id": "...", "host": "...", "port": <n>,
/// "context_size": <n>, "kv_pool_positions": <n>, "kv_slots": <n>,
/// "prompt_cache": "0"|"1"|"2"}`.
pub async fn server_mode_halogen_start_response(state: &AppState, body: &Bytes) -> Response<Full<Bytes>> {
    let request: crate::server_mode::StartHalogenServerRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            return json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
                error: format!("invalid request body: {e}"),
            });
        }
    };
    match crate::server_mode::start_halogen_server(&request).await {
        Ok(()) => {
            if let Ok((_, profile, _platform_id)) = crate::server_mode::resolve_halogen_toolbox(&request.toolbox_id) {
                register_serving_identity(
                    state,
                    crate::toolbox_catalog::SupportedServingBackend::Halogen,
                    crate::server_mode::HALOGEN_SERVER_CONTAINER_NAME,
                    request.toolbox_id.clone(),
                    &profile,
                    serving_identity_endpoint(&request.host, request.port),
                )
                .await;
            }
            let status = crate::server_mode::halogen_server_status().await;
            json_response(StatusCode::OK, &status)
        }
        Err(e) => json_response(StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), &ErrorResponse {
            error: e.message().to_string(),
        }),
    }
}

/// `POST /api/server-mode/halogen/stop` — graceful `podman stop` then
/// `podman rm -f`, idempotent if the container is already gone.
pub async fn server_mode_halogen_stop_response(state: &AppState) -> Response<Full<Bytes>> {
    match crate::server_mode::stop_halogen_server().await {
        Ok(()) => {
            state.serving_identities.deregister(crate::server_mode::HALOGEN_SERVER_CONTAINER_NAME).await;
            json_response(StatusCode::OK, &serde_json::json!({
                "status": "ok",
                "message": "halogen server stopped.",
            }))
        }
        Err(e) => json_response(StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), &ErrorResponse {
            error: e.message().to_string(),
        }),
    }
}

// ── PR9: vllm Server Mode (§14) ────────────────────────────────────────────

/// `GET /api/server-mode/vllm/status` — always reads live `podman inspect`
/// state, same no-persisted-registry contract as ds4/halogen's status
/// endpoints above.
pub async fn server_mode_vllm_status_response() -> Response<Full<Bytes>> {
    let status = crate::server_mode::vllm_server_status().await;
    json_response(StatusCode::OK, &status)
}

/// `POST /api/server-mode/vllm/start` — body:
/// `{"toolbox_id": "...", "model_id"|"custom_repo": "...", "host": "...",
/// "port": <n>, "tensor_parallel": <n>, "max_num_seqs": <n>,
/// "max_model_len": "auto"|"<n>", "gpu_memory_utilization": <f>,
/// "attention_backend": "...", "enforce_eager": <bool>, "dtype": "...",
/// "api_key": "...", "extra_args": "...", "reset_caches": <bool>}`.
pub async fn server_mode_vllm_start_response(state: &AppState, body: &Bytes) -> Response<Full<Bytes>> {
    let request: crate::server_mode::StartVllmServerRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            return json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
                error: format!("invalid request body: {e}"),
            });
        }
    };
    match crate::server_mode::start_vllm_server(&request).await {
        Ok(()) => {
            if let Ok((_, profile)) = crate::server_mode::resolve_vllm_toolbox(&request.toolbox_id) {
                register_serving_identity(
                    state,
                    crate::toolbox_catalog::SupportedServingBackend::Vllm,
                    crate::server_mode::VLLM_SERVER_CONTAINER_NAME,
                    request.toolbox_id.clone(),
                    &profile,
                    serving_identity_endpoint(&request.host, request.port),
                )
                .await;
            }
            let status = crate::server_mode::vllm_server_status().await;
            json_response(StatusCode::OK, &status)
        }
        Err(e) => json_response(StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), &ErrorResponse {
            error: e.message().to_string(),
        }),
    }
}

/// `POST /api/server-mode/vllm/stop` — graceful `podman stop` then `podman
/// rm -f`, idempotent if the container is already gone.
pub async fn server_mode_vllm_stop_response(state: &AppState) -> Response<Full<Bytes>> {
    match crate::server_mode::stop_vllm_server().await {
        Ok(()) => {
            state.serving_identities.deregister(crate::server_mode::VLLM_SERVER_CONTAINER_NAME).await;
            json_response(StatusCode::OK, &serde_json::json!({
                "status": "ok",
                "message": "vllm server stopped.",
            }))
        }
        Err(e) => json_response(StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), &ErrorResponse {
            error: e.message().to_string(),
        }),
    }
}

/// `POST /api/server-mode/vllm/cache-paths` — body:
/// `{"hf_cache"?, "vllm_cache"?, "triton_cache"?, "aiter_cache"?: "..."}`,
/// the direct analogue of upstream's separate "Save Cache Paths" action
/// (§14 item 7) — unlike ds4/halogen, vllm does not persist its settings
/// as a side effect of `start`.
pub async fn server_mode_vllm_cache_paths_response(body: &Bytes) -> Response<Full<Bytes>> {
    let request: crate::server_mode::VllmCachePathsRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            return json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
                error: format!("invalid request body: {e}"),
            });
        }
    };
    match crate::server_mode::save_vllm_cache_paths(&request) {
        Ok(()) => json_response(StatusCode::OK, &serde_json::json!({
            "status": "ok",
            "message": "vllm cache paths saved.",
        })),
        Err(e) => json_response(StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), &ErrorResponse {
            error: e.message().to_string(),
        }),
    }
}

// ── PR11: r9v Server Mode (§15/§15a) ───────────────────────────────────────

/// `GET /api/server-mode/r9v/status` — always reads live `podman inspect`
/// state, same no-persisted-registry contract as the other three backends'
/// status endpoints above.
pub async fn server_mode_r9v_status_response() -> Response<Full<Bytes>> {
    let status = crate::server_mode::r9v_server_status().await;
    json_response(StatusCode::OK, &status)
}

/// `POST /api/server-mode/r9v/start` — body:
/// `{"toolbox_id": "...", "package_id": "...", "host"?, "port"?,
/// "devices"?, "context"?, "batch"?, "sequences"?, "kv_bytes"?,
/// "expert_cache_slots"?, "offload"?, "offload_devices"?, "served_model"?,
/// "api_key"?, "extra_args"?}` — every tuning field is optional, falling
/// back to upstream's own literal defaults (§15a).
pub async fn server_mode_r9v_start_response(state: &AppState, body: &Bytes) -> Response<Full<Bytes>> {
    let request: crate::server_mode::StartR9vServerRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            return json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
                error: format!("invalid request body: {e}"),
            });
        }
    };
    match crate::server_mode::start_r9v_server(&request).await {
        Ok(()) => {
            if let Ok((_, profile, _platform_id)) = crate::server_mode::resolve_r9v_toolbox(&request.toolbox_id) {
                let host = request.host.as_deref().unwrap_or(crate::server_mode::R9V_DEFAULT_HOST);
                let port = request.port.unwrap_or(crate::server_mode::R9V_DEFAULT_PORT);
                register_serving_identity(
                    state,
                    crate::toolbox_catalog::SupportedServingBackend::R9v,
                    crate::server_mode::R9V_SERVER_CONTAINER_NAME,
                    request.toolbox_id.clone(),
                    &profile,
                    serving_identity_endpoint(host, port),
                )
                .await;
            }
            let status = crate::server_mode::r9v_server_status().await;
            json_response(StatusCode::OK, &status)
        }
        Err(e) => json_response(StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), &ErrorResponse {
            error: e.message().to_string(),
        }),
    }
}

/// `POST /api/server-mode/r9v/stop` — graceful `podman stop` then `podman
/// rm -f`, idempotent if the container is already gone.
pub async fn server_mode_r9v_stop_response(state: &AppState) -> Response<Full<Bytes>> {
    match crate::server_mode::stop_r9v_server().await {
        Ok(()) => {
            state.serving_identities.deregister(crate::server_mode::R9V_SERVER_CONTAINER_NAME).await;
            json_response(StatusCode::OK, &serde_json::json!({
                "status": "ok",
                "message": "r9v server stopped.",
            }))
        }
        Err(e) => json_response(StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), &ErrorResponse {
            error: e.message().to_string(),
        }),
    }
}

/// `POST /api/server-mode/r9v/paths` — body:
/// `{"models_dir"?, "ple_dir"?, "cache_dir"?: "..."}`, the r9v analogue of
/// vllm's `/cache-paths` action.
pub async fn server_mode_r9v_paths_response(body: &Bytes) -> Response<Full<Bytes>> {
    let request: crate::server_mode::R9vPathsRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            return json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
                error: format!("invalid request body: {e}"),
            });
        }
    };
    match crate::server_mode::save_r9v_paths(&request) {
        Ok(()) => json_response(StatusCode::OK, &serde_json::json!({
            "status": "ok",
            "message": "r9v paths saved.",
        })),
        Err(e) => json_response(StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), &ErrorResponse {
            error: e.message().to_string(),
        }),
    }
}

/// `POST /api/model-downloads/r9v/prepare-ple` — body:
/// `{"toolbox_id": "...", "package_id": "..."}`. Starts the "Prepare PLE"
/// job (`podman run ... r9v-model prepare`) through the same
/// [`crate::model_downloads::ModelDownloadRegistry`] job registry as
/// ordinary downloads (§15's job-registry-widening decision) — the
/// returned [`crate::model_downloads::ModelDownloadJob`] is polled the
/// same way via the existing `GET /api/model-downloads/{id}` endpoint.
pub async fn model_downloads_prepare_ple_response(state: &AppState, body: &Bytes) -> Response<Full<Bytes>> {
    let request: crate::model_downloads::StartPreparePleRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            return json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
                error: format!("invalid request body: {e}"),
            });
        }
    };
    match state.model_downloads.start_prepare_ple(&request).await {
        Ok(job) => json_response(StatusCode::OK, &job),
        Err(e) => json_response(StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), &ErrorResponse {
            error: e.message().to_string(),
        }),
    }
}

/// `GET /api/serving-identities` — read-only snapshot of every
/// currently-registered server-mode backend's serving identity (§16/PR11a).
/// Purely informational: the dashboard panel this feeds has no start/stop
/// controls, and no request routing ever consults this endpoint's data
/// (see `serving_identity` module docs).
pub async fn serving_identities_response(state: &AppState) -> Response<Full<Bytes>> {
    let identities = state.serving_identities.snapshot().await;
    json_response(StatusCode::OK, &serde_json::json!({ "identities": identities }))
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
    let update: crate::config::ReviewConfig = match read_routing_json(req).await {
        Ok(update) => update,
        Err(error) => return Ok(routing_error(StatusCode::BAD_REQUEST, error)),
    };
    if let Err(error) = update.validate() {
        return Ok(routing_error(StatusCode::BAD_REQUEST, error));
    }
    match service.update_config(update).await {
        Ok(()) => Ok(into_unsync(json_response(StatusCode::OK, &serde_json::json!({ "status": "ok" })))),
        Err(error) => Ok(routing_error(StatusCode::INTERNAL_SERVER_ERROR, error)),
    }
}

fn routing_error(status: StatusCode, error: impl std::fmt::Display) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    into_unsync(json_response(status, &ErrorResponse { error: error.to_string() }))
}

async fn read_routing_json<T: serde::de::DeserializeOwned>(
    req: Request<Incoming>,
) -> anyhow::Result<T> {
    let bytes = http_body_util::Limited::new(req.into_body(), 16 * 1024).collect().await
        .map_err(anyhow::Error::from_boxed)?.to_bytes();
    Ok(serde_json::from_slice(&bytes)?)
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
        .map_err(|e| anyhow::anyhow!("Failed to fetch llama-swap models: {}", e))?
        .error_for_status()
        .map_err(|e| anyhow::anyhow!("llama-swap models request failed: {}", e))?;
    let body: serde_json::Value = resp.json().await
        .map_err(|e| anyhow::anyhow!("Failed to parse llama-swap models response: {}", e))?;
    let model_ids = parse_llama_swap_model_ids(&body)?;

    // All filesystem + YAML work runs off the async executor.
    let tcp_addr_owned = tcp_addr.to_string();
    tokio::task::spawn_blocking(move || write_omp_models_yml(&home, &model_ids, &tcp_addr_owned))
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking panicked: {}", e))?
}

fn parse_llama_swap_model_ids(body: &serde_json::Value) -> anyhow::Result<Vec<String>> {
    body
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| anyhow::anyhow!("llama-swap models response is missing a data array"))?
        .iter()
        .enumerate()
        .map(|(index, model)| {
            model
                .get("id")
                .and_then(|value| value.as_str())
                .filter(|id| !id.is_empty())
                .map(String::from)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "llama-swap models response has an invalid id at data[{}]",
                        index
                    )
                })
        })
        .collect()
}

/// Blocking: read models.yml, merge brainrouter models, write atomically.
fn write_omp_models_yml(home: &str, model_ids: &[String], tcp_addr: &str) -> anyhow::Result<usize> {
    let models_path = format!("{}/.omp/agent/models.yml", home);
    let path = std::path::Path::new(&models_path);

    // Read existing models.yml (or start fresh).
    let mut doc: serde_yaml::Value = if path.exists() {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", models_path, e))?;
        serde_yaml::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse {}: {}", models_path, e))?
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
    let tmp_path = format!("{}.{}.tmp", models_path, uuid::Uuid::new_v4());
    std::fs::write(&tmp_path, &yaml_str)
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {}", tmp_path, e))?;
    if let Ok(metadata) = std::fs::metadata(path) {
        std::fs::set_permissions(&tmp_path, metadata.permissions())
            .map_err(|e| anyhow::anyhow!("Failed to preserve permissions on {}: {}", tmp_path, e))?;
    }
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
    let tcp_listener = TcpListener::bind(tcp_addr).await?;
    info!("TCP listener bound to {}", tcp_addr);

    let _uds_lock = acquire_uds_lock(&uds_path)?;
    prepare_uds_path(&uds_path).await?;
    let uds_listener = UnixListener::bind(&uds_path)?;
    let uds_identity = socket_identity(&uds_path)?;
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

    remove_owned_uds(&uds_path_for_cleanup, uds_identity);

    Ok(())
}

fn acquire_uds_lock(uds_path: &std::path::Path) -> Result<File> {
    let mut lock_name = uds_path.as_os_str().to_os_string();
    lock_name.push(".lock");
    let lock_path = PathBuf::from(lock_name);
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&lock_path)?;

    lock_uds_file(&lock, uds_path)?;
    Ok(lock)
}

#[cfg(not(target_os = "solaris"))]
fn lock_uds_file(lock: &File, uds_path: &std::path::Path) -> Result<()> {
    // flock is tied to the open file description, so holding `lock` for the
    // lifetime of `run` serializes socket inspection, removal, and binding.
    let result = unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if result != 0 {
        let error = std::io::Error::last_os_error();
        if error.kind() == std::io::ErrorKind::WouldBlock {
            anyhow::bail!(
                "Another brainrouter instance is using Unix socket {}",
                uds_path.display()
            );
        }
        return Err(error.into());
    }
    Ok(())
}

#[cfg(target_os = "solaris")]
fn lock_uds_file(_lock: &File, uds_path: &std::path::Path) -> Result<()> {
    anyhow::bail!(
        "Unix socket startup locking is not supported on Solaris for {}",
        uds_path.display()
    )
}

fn socket_identity(path: &std::path::Path) -> Result<(u64, u64)> {
    let metadata = std::fs::symlink_metadata(path)?;
    Ok((metadata.dev(), metadata.ino()))
}

fn remove_owned_uds(path: &std::path::Path, expected: (u64, u64)) {
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return;
    };
    if (metadata.dev(), metadata.ino()) != expected {
        warn!(
            path = %path.display(),
            "Skipping Unix socket cleanup because the path is now owned by another socket"
        );
        return;
    }
    info!("Cleaning up Unix socket at {:?}", path);
    if let Err(error) = std::fs::remove_file(path) {
        warn!(path = %path.display(), error = %error, "Failed to clean up Unix socket");
    }
}

async fn prepare_uds_path(uds_path: &std::path::Path) -> Result<()> {
    let metadata = match std::fs::symlink_metadata(uds_path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    if !metadata.file_type().is_socket() {
        anyhow::bail!(
            "Refusing to replace non-socket path at {}",
            uds_path.display()
        );
    }

    match tokio::time::timeout(
        std::time::Duration::from_millis(250),
        UnixStream::connect(uds_path),
    )
    .await
    {
        Ok(Ok(_)) => anyhow::bail!(
            "Another brainrouter instance is already listening at {}",
            uds_path.display()
        ),
        Ok(Err(error))
            if matches!(
                error.kind(),
                std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::NotFound
            ) =>
        {
            info!("Removing stale Unix socket at {:?}", uds_path);
            std::fs::remove_file(uds_path)?;
            Ok(())
        }
        Ok(Err(error)) => Err(error.into()),
        Err(_) => anyhow::bail!(
            "Timed out probing existing Unix socket at {}",
            uds_path.display()
        ),
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    // ── FR-D: maybe_inject_pr_guidelines ─────────────────────────────────────
    fn req(messages: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({ "model": "auto", "messages": messages })).unwrap()
    }
    fn marker_at(r: &ChatCompletionRequest, i: usize) -> bool {
        r.messages[i].role == "system"
            && r.messages[i]
                .content
                .as_ref()
                .and_then(serde_json::Value::as_str)
                .is_some_and(|s| s.contains(PR_GUIDELINE_MARKER))
    }

    #[test]
    fn pr_guidelines_off_leaves_messages_unchanged() {
        let mut r = req(serde_json::json!([
            {"role": "system", "content": "You are OMP."},
            {"role": "user", "content": "hi"},
        ]));
        maybe_inject_pr_guidelines(&mut r, false);
        assert_eq!(r.messages.len(), 2);
        assert!(!r.messages.iter().any(marker_present));
    }

    fn marker_present(m: &crate::types::ChatMessage) -> bool {
        m.content.as_ref().and_then(serde_json::Value::as_str).is_some_and(|s| s.contains(PR_GUIDELINE_MARKER))
    }

    #[test]
    fn pr_guidelines_inject_after_leading_system_block() {
        let mut r = req(serde_json::json!([
            {"role": "system", "content": "sys A"},
            {"role": "system", "content": "sys B"},
            {"role": "user", "content": "hi"},
        ]));
        maybe_inject_pr_guidelines(&mut r, true);
        // Inserted at index 2 (after the two leading systems, before the user).
        assert_eq!(r.messages.len(), 4);
        assert!(marker_at(&r, 2));
        assert_eq!(r.messages[3].role, "user");
        // Original leading systems untouched.
        assert_eq!(r.messages[0].content.as_ref().unwrap().as_str().unwrap(), "sys A");
    }

    #[test]
    fn pr_guidelines_no_leading_system_inserts_at_front() {
        let mut r = req(serde_json::json!([{"role": "user", "content": "hi"}]));
        maybe_inject_pr_guidelines(&mut r, true);
        assert_eq!(r.messages.len(), 2);
        assert!(marker_at(&r, 0));
        assert_eq!(r.messages[1].role, "user");
    }

    #[test]
    fn pr_guidelines_all_system_appends_at_end() {
        let mut r = req(serde_json::json!([
            {"role": "system", "content": "a"},
            {"role": "system", "content": "b"},
        ]));
        maybe_inject_pr_guidelines(&mut r, true);
        assert_eq!(r.messages.len(), 3);
        assert!(marker_at(&r, 2));
    }

    #[test]
    fn pr_guidelines_are_idempotent() {
        let mut r = req(serde_json::json!([
            {"role": "system", "content": "You are OMP."},
            {"role": "user", "content": "hi"},
        ]));
        maybe_inject_pr_guidelines(&mut r, true);
        let after_first = r.messages.len();
        maybe_inject_pr_guidelines(&mut r, true);
        assert_eq!(r.messages.len(), after_first, "second call must not duplicate");
        assert_eq!(r.messages.iter().filter(|m| marker_present(m)).count(), 1);
    }

    #[test]
    fn pr_guidelines_structured_content_is_not_a_false_positive() {
        // A system message with non-string (array) content must not be scanned as
        // carrying the marker, so injection still proceeds and nothing panics.
        let mut r = req(serde_json::json!([
            {"role": "system", "content": [{"type": "text", "text": "structured"}]},
            {"role": "user", "content": "hi"},
        ]));
        maybe_inject_pr_guidelines(&mut r, true);
        assert_eq!(r.messages.len(), 3);
        assert!(marker_at(&r, 1));
    }

    #[test]
    fn display_name_full_model_id() {
        assert_eq!(model_id_to_display_name("qwen3.6-27b-q6-amdvlk"), "Qwen3.6 27B Q6 AMDVLK");
    }

    /// Regression test for the podman version-check leak (design doc
    /// `docs/design/ai-toolbox-cockpit-integration.md` §5a): a normal completion within the
    /// timeout must NOT trigger the best-effort cleanup command. This must actually exercise the
    /// success path, not just assert the timeout path is correct in isolation, since either half
    /// failing silently would defeat the fix.
    #[tokio::test]
    async fn timed_command_skips_cleanup_when_command_completes_in_time() {
        let marker = std::env::temp_dir().join(format!("brainrouter-test-marker-{}", uuid::Uuid::new_v4()));
        let mut cmd = tokio::process::Command::new("sh");
        cmd.args(["-c", "true"]).kill_on_drop(true);
        let mut cleanup = tokio::process::Command::new("sh");
        cleanup.args(["-c", &format!("touch {}", marker.display())]);

        let outcome = run_with_timeout_and_cleanup(
            cmd,
            std::time::Duration::from_secs(5),
            cleanup,
        ).await;

        assert!(matches!(outcome, TimedCommandOutcome::Completed(_)));
        assert!(!marker.exists(), "cleanup must not run when the command completes in time");
    }

    /// Regression test forcing the **timeout branch itself** (not just running the check twice
    /// normally, which the Dory critic review found would pass even without the fix): a
    /// deliberately slow command under an artificially short timeout must be killed and must
    /// trigger the best-effort cleanup command exactly once.
    #[tokio::test]
    async fn timed_command_runs_cleanup_on_timeout() {
        let marker = std::env::temp_dir().join(format!("brainrouter-test-marker-{}", uuid::Uuid::new_v4()));
        let mut cmd = tokio::process::Command::new("sh");
        cmd.args(["-c", "sleep 5"]).kill_on_drop(true);
        let mut cleanup = tokio::process::Command::new("sh");
        cleanup.args(["-c", &format!("touch {}", marker.display())]);

        let outcome = run_with_timeout_and_cleanup(
            cmd,
            std::time::Duration::from_millis(50),
            cleanup,
        ).await;

        assert!(matches!(outcome, TimedCommandOutcome::TimedOut));
        assert!(marker.exists(), "best-effort cleanup must run on timeout");
        let _ = std::fs::remove_file(&marker);
    }

    #[test]
    fn destructive_api_origins_require_real_loopback_urls() {
        for value in [
            "http://localhost:9099", "http://127.0.0.1:9099/dashboard", "http://[::1]:9099",
            "http://localhost:8080", "http://127.0.0.1:12345", "http://[::1]:12345/dashboard",
            "http://localhost",
        ] {
            assert!(is_loopback_http_url(value), "{value}");
        }
        for value in [
            "null", "http://localhost.example.com:9099", "https://localhost:9099",
            "http://192.0.2.1:8080", "http://[2001:db8::1]:8080", "not a URL",
            "http://localhost@evil.example:9099", "http://evil.example@localhost:9099",
        ] {
            assert!(!is_loopback_http_url(value), "{value}");
        }
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
        let tmp = std::env::temp_dir().join(format!(
            "brainrouter-test-{}",
            uuid::Uuid::new_v4()
        ));
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

    #[test]
    fn malformed_omp_models_file_is_not_replaced() {
        let tmp = std::env::temp_dir().join(format!(
            "brainrouter-test-{}",
            uuid::Uuid::new_v4()
        ));
        let agent_dir = tmp.join(".omp/agent");
        std::fs::create_dir_all(&agent_dir).unwrap();
        let models_file = agent_dir.join("models.yml");
        let original = "providers: [invalid\n";
        std::fs::write(&models_file, original).unwrap();

        let result = write_omp_models_yml(
            tmp.to_str().unwrap(),
            &["model-a".to_string()],
            "127.0.0.1:9099",
        );
        assert!(result.is_err());
        assert_eq!(std::fs::read_to_string(&models_file).unwrap(), original);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn invalid_llama_swap_model_payload_is_rejected() {
        assert!(parse_llama_swap_model_ids(&serde_json::json!({})).is_err());
        assert!(parse_llama_swap_model_ids(&serde_json::json!({
            "data": [{"id": "model-a"}, {"name": "missing-id"}]
        }))
        .is_err());
        assert_eq!(
            parse_llama_swap_model_ids(&serde_json::json!({
                "data": [{"id": "model-a"}, {"id": "model-b"}]
            }))
            .unwrap(),
            vec!["model-a", "model-b"]
        );
    }

    #[tokio::test]
    async fn live_uds_is_not_unlinked() {
        let dir = PathBuf::from(format!("/tmp/br-uds-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let socket_path = dir.join("brainrouter.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();

        let error = prepare_uds_path(&socket_path).await.unwrap_err();
        assert!(error.to_string().contains("already listening"));
        assert!(UnixStream::connect(&socket_path).await.is_ok());

        drop(listener);
        let _ = std::fs::remove_file(&socket_path);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn stale_uds_is_removed_before_binding() {
        let dir = PathBuf::from(format!("/tmp/br-uds-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let socket_path = dir.join("brainrouter.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();
        drop(listener);

        prepare_uds_path(&socket_path).await.unwrap();
        assert!(!socket_path.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(not(target_os = "solaris"))]
    #[test]
    fn uds_lock_rejects_a_second_owner() {
        let path = PathBuf::from(format!("/tmp/br-uds-{}.sock", uuid::Uuid::new_v4()));
        let first = acquire_uds_lock(&path).unwrap();
        let second = acquire_uds_lock(&path);
        assert!(second.is_err());

        drop(first);
        let mut lock_name = path.as_os_str().to_os_string();
        lock_name.push(".lock");
        let _ = std::fs::remove_file(PathBuf::from(lock_name));
    }

    #[tokio::test]
    async fn cleanup_does_not_unlink_replaced_socket() {
        let dir = PathBuf::from(format!("/tmp/br-uds-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let socket_path = dir.join("brainrouter.sock");
        let first = UnixListener::bind(&socket_path).unwrap();
        let first_identity = socket_identity(&socket_path).unwrap();
        drop(first);
        std::fs::remove_file(&socket_path).unwrap();

        let replacement = UnixListener::bind(&socket_path).unwrap();
        remove_owned_uds(&socket_path, first_identity);
        assert!(UnixStream::connect(&socket_path).await.is_ok());

        drop(replacement);
        let _ = std::fs::remove_file(&socket_path);
        let _ = std::fs::remove_dir_all(&dir);
    }
}