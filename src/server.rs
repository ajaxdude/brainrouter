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
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::convert::Infallible;
use std::fs::{File, OpenOptions};
use std::net::SocketAddr;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::net::{TcpListener, UnixListener, UnixStream};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering as AtomicOrdering};
use tracing::{debug, error, info, warn};
use std::sync::LazyLock;
use sha2::{Digest, Sha256};

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
use crate::managed_toolboxes::OwnershipEntry;
use crate::toolbox_catalog::{
    self, SupportedServingBackend, ToolboxCatalog, ToolboxDefinition,
};

// Unified dashboard — embedded at compile time so the binary is self-contained.
const MAIN_DASHBOARD_HTML: &str = include_str!("escalation/templates/main_dashboard.html");
/// v2 R2: token the dashboard's nav script reads to hide the Benchmarks
/// explorer link in builds without the `benchmarks` feature.
const BENCHMARKS_FEATURE_TOKEN: &str = "__BENCHMARKS_FEATURE_ENABLED__";
/// v2 L4: the probe's catalog platform id replaces the hardcoded literal.
const CURRENT_PLATFORM_TOKEN: &str = "__CURRENT_PLATFORM_ID__";

fn main_dashboard_html(platform_id: &str) -> String {
    MAIN_DASHBOARD_HTML
        .replace(
            BENCHMARKS_FEATURE_TOKEN,
            if cfg!(feature = "benchmarks") { "true" } else { "false" },
        )
        .replace(CURRENT_PLATFORM_TOKEN, platform_id)
}
const FAVICON_SVG: &[u8] = include_bytes!("escalation/templates/favicon.svg");
const LOGO_SVG: &[u8] = include_bytes!("escalation/templates/logo.svg");

/// Maximum time the DeferredStream will wait for a provider stream before
/// giving up.  Aligned with the TTFT_TIMEOUT in router.rs (600 s) so model
/// loading + prefill for large local models (qwen3-27b-mtp) can complete.
const DEFERRED_STREAM_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

/// Shared state passed to all request handlers
/// The project-scope retry grace window for one UDS CONNECTION, anchored
/// at ACCEPT time: a connection accepted at t has a live peer (its CLI
/// process completes its request-response exchange before exiting), so a
/// cwd resolvable at accept cannot vanish mid-exchange, and queueing
/// anywhere below (reqwest accept queues, tokio task scheduling under
/// full-suite load) does NOT burn the window. Per-connection state keeps
/// concurrent CLI connections independent.
#[derive(Default)]
pub struct ScopeClock(std::sync::Mutex<Option<std::time::Instant>>);

impl ScopeClock {
    pub fn elapsed(&self) -> Option<std::time::Duration> {
        self.0.lock().unwrap().map(|t| t.elapsed())
    }
}



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
    /// v2 L2 plugin subsystem: registry, supervisor, RPC UDS servers.
    pub plugins: Arc<crate::plugins::PluginManager>,
    /// Path to brainrouter's own config file (used by the config UI and
    /// the self-restart endpoint).
    pub config_path: PathBuf,
    /// Path to llama-swap's config file (used by the restart-local-stack
    /// endpoint and the context-size setter).
    pub llama_swap_config_path: PathBuf,
    /// Serializes all llama-swap config writers, including raw edits and generated registrations.
    pub llama_swap_config_lock: Arc<tokio::sync::Mutex<()>>,
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
    /// does not affect the proxy or imported benchmark explorer. Compiled
    /// only with the `benchmarks` feature (v2 R2).
    #[cfg(feature = "benchmarks")]
    pub benchmark_lab: Result<Arc<crate::benchmark_lab::BenchmarkLab>, String>,
    /// Read-only model observations and separately persisted operator settings.
    pub observability: Arc<crate::observability::Observability>,
    /// Local sidecar mapping toolbox container names to exact podman IDs.
    pub managed_toolboxes_path: PathBuf,
    /// Per-container-name mutation lock for toolbox create/update/delete/adopt,
    /// so two concurrent requests for the same container name queue instead of
    /// racing `podman`/`toolbox` invocations against each other. Different
    /// container names never contend (see design doc §5c).
    pub toolbox_container_locks: Arc<std::sync::Mutex<std::collections::HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
    /// Model-download job registry (PR6, §10): single-flight `hf download`
    /// orchestration for the download-capable catalog backends
    /// (ds4/llama_cpp/gufo after the v2 L1 purge; the HF-pulling server
    /// backends have no download step, see `model_downloads.rs` module docs).
    pub model_downloads: Arc<crate::model_downloads::ModelDownloadRegistry>,
    /// v2 L3: role cards (the single model config — assignments + policy +
    /// review card extras).
    pub roles: Arc<crate::roles::RoleCardStore>,
    /// v2 L3: the decision-model slot (features → {route, tier}).
    pub decision: Arc<crate::decision::DecisionSlot>,
    /// v2 L3: the decision slot config (for `decision show`/`test`).
    pub decision_config: crate::config::DecisionModelConfig,
    /// v2 L2 credentials store — exposed to the L3 credentials UI endpoints.
    pub credentials: Arc<crate::credentials::CredentialStore>,
    /// v2 L4: the boot platform probe (id, GPUs, budgets, engine offers).
    pub platform: Arc<crate::platform::PlatformProbe>,
    /// v2 L4: swap supervisor (catalog, residency, budget, queue, adapters).
    pub swap: Arc<crate::swap_scheduler::SwapScheduler>,
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
pub(crate) struct ModelListResponse {
    object: &'static str,
    data: Vec<ModelObject>,
}

#[derive(Serialize)]
pub(crate) struct ModelObject {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

/// v2 L4: one canonical path per engine operation so the documented route
/// names (list/queue/pin/unpin/import/logs) and their obvious aliases hit the
/// same handlers (`/api/engines/list` → `/api/engines`,
/// `/api/engines/import` GET/POST → preview/apply, trailing slashes ignored).
fn normalize_engines_path<'a>(method: &str, path: &'a str) -> &'a str {
    if !(path.starts_with("/api/engines") || path.starts_with("/api/platform")) {
        return path;
    }
    let trimmed = path.trim_end_matches('/');
    match (method, trimmed) {
        ("GET", "/api/engines/list") => "/api/engines",
        ("GET", "/api/engines/import") => "/api/engines/import/preview",
        ("POST", "/api/engines/import") => "/api/engines/import/apply",
        _ => {
            if trimmed.is_empty() {
                path
            } else {
                trimmed
            }
        }
    }
}

/// v2 L4: a swap admission failure is an explicit 503 with the shared
/// `-32005` busy code (never a generic 502). Returns `None` for other errors.
fn swap_busy_response(error: &anyhow::Error) -> Option<Response<Full<Bytes>>> {
    let text = error.to_string();
    if !text.contains(&format!("{}", crate::swap_scheduler::BUSY_CODE))
        && !text.contains("no-fit")
        && !text.contains("swap queue")
    {
        return None;
    }
    Some(json_response(
        StatusCode::SERVICE_UNAVAILABLE,
        &serde_json::json!({
            "error": {
                "message": text,
                "type": "insufficient_capacity",
                "code": crate::swap_scheduler::BUSY_CODE,
            }
        }),
    ))
}

/// v2 L4 import source path: the daemon's configured llama-swap config, or an
/// explicit `path` confined to the caller's HOME (never an arbitrary read).
fn engines_import_path(
    state: &AppState,
    requested: Option<&str>,
) -> anyhow::Result<PathBuf> {
    let Some(requested) = requested.map(str::trim).filter(|p| !p.is_empty()) else {
        return Ok(state.llama_swap_config_path.clone());
    };
    let path = std::fs::canonicalize(requested)
        .map_err(|e| anyhow::anyhow!("resolving import path {requested}: {e}"))?;
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    let home = std::fs::canonicalize(&home).unwrap_or_else(|_| PathBuf::from(home));
    let state_root = crate::paths::brainrouter_state();
    if !path.starts_with(&home) && !path.starts_with(&state_root) {
        anyhow::bail!(
            "import path must live under $HOME or the brainrouter state directory"
        );
    }
    Ok(path)
}

/// v2 L4: the Engines tab / `swap status` payload.
fn engines_json(state: &AppState) -> serde_json::Value {
    let status = state.swap.status();
    serde_json::json!({
        "platform": &*state.platform,
        "models": status.models,
        "queue": status.queue,
        "budget": status.budget,
        "adoption": status.adoption,
        "shim_error": status.shim_error,
        "adapters": status.adapter_names,
        "groups": state.swap.catalog().groups(),
        "pins": state.swap.catalog().pins(),
        "logs_dir": state.swap.logs_dir().display().to_string(),
    })
}

/// Map a swap failure onto its HTTP status (`404` unknown model, `503`
/// no-fit/queue-full/spawn-failure with `-32005`, `504` timeout).
fn swap_api_error(error: anyhow::Error) -> Response<Full<Bytes>> {
    use crate::swap_scheduler::SwapError;
    if let Some(swap_error) = error.downcast_ref::<SwapError>() {
        return json_response(
            StatusCode::from_u16(swap_error.status_code())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            &serde_json::json!({
                "error": swap_error.to_string(),
                "code": swap_error.code(),
            }),
        );
    }
    json_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        &serde_json::json!({ "error": error.to_string() }),
    )
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

/// Classify a request as destructive (loopback + CSRF gated). Pure so the
/// security boundary is unit-tested. `has_query` is `req.uri().query().is_some()`
/// — used only to gate the per-project-pin `?path=` resolver while leaving the
/// plain no-query GET ungated.
fn request_is_destructive(method: &str, path: &str, _has_query: bool) -> bool {
    path.starts_with("/api/restart/") || path.starts_with("/api/upgrade/")
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
        // v2 L3 role cards / policy / credentials / decision slot: every
        // mutation is loopback + CSRF gated (credentials material NEVER
        // appears in GETs unmasked).
        || (matches!(method, "POST" | "DELETE") && path.starts_with("/api/roles"))
        || (matches!(method, "POST" | "DELETE") && path.starts_with("/api/policy"))
        || (matches!(method, "POST" | "DELETE") && path.starts_with("/api/credentials"))
        || (method == "POST" && path.starts_with("/api/decision/"))
        // v2 L4 swap supervisor mutations (spawn/stop/pin/import/add/rm).
        || (matches!(method, "POST" | "DELETE") && path.starts_with("/api/engines"))
        || (method == "POST" && path.starts_with("/api/llama-swap/"))
        || (method == "POST" && (
            path == "/api/config" || path == "/api/llama-swap-config"
            || path == "/api/open-editor"
            || path == "/api/routing-mode" || path == "/api/review-config"
            || path.starts_with("/api/plugins")
            || path == "/api/bonsai/toggle" || path == "/api/models/flush"
            || path == "/api/nudge" || path == "/api/prompt-rewrite"
            // Review API is destructive too: it spawns reviews (arbitrary
            // project paths read into cloud prompts) and can approve/resolve
            // sessions. Gate it like the rest.
            || path.starts_with("/review/api/")
            || path.starts_with("/api/benchmarks/")
            || path == "/api/inflight/cancel"
            || path.starts_with("/api/observability/")
        ))
}

/// Handle incoming HTTP requests
async fn handle_request(
    req: Request<Incoming>,
    state: Arc<AppState>,
    cwd: String,
    peer_addr: SocketAddr,
    scope_clock: Option<Arc<ScopeClock>>,
) -> Result<Response<UnsyncBoxBody<Bytes, anyhow::Error>>, Infallible> {
    // v2 L3 Step 9 I4: the grace clock lives in a task-local scope around
    // the whole request (per-connection task; concurrent requests on the
    // shared Router each see their own window).
    Box::pin(Router::scope_clock_scope(scope_clock, handle_request_inner(req, state, cwd, peer_addr))).await
}

async fn handle_request_inner(
    req: Request<Incoming>,
    state: Arc<AppState>,
    cwd: String,
    peer_addr: SocketAddr,
) -> Result<Response<UnsyncBoxBody<Bytes, anyhow::Error>>, Infallible> {
    let method = req.method().as_str();
    let path = normalize_engines_path(method, req.uri().path());

    debug!("Request: {} {}", method, path);

    // Security: Only allow localhost (127.0.0.1 or ::1) for destructive APIs.
    // UDS connections (peer_addr = 0.0.0.0:0) are always allowed as they are local.
    let is_local = peer_addr.ip().is_loopback() || peer_addr.port() == 0;
    let is_destructive = request_is_destructive(method, path, req.uri().query().is_some());

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

    #[cfg(feature = "benchmarks")]
    if path.starts_with("/api/benchmarks/lab/") || path == "/api/benchmarks/lab" {
        return match &state.benchmark_lab {
            Ok(lab) => crate::benchmark_lab::handle_request(req, Arc::clone(lab)).await,
            Err(reason) => Ok(crate::benchmark_lab::unavailable_response(reason)),
        };
    }

    if path == "/benchmarks" || path == "/benchmarks/" || path.starts_with("/api/benchmarks/") {
        return match &state.benchmark_store {
            // v2 R2: with the `benchmarks` feature off the daemon never
            // opens the store, so the `Err` arm below answers the same
            // documented 503 degradation envelope as unavailable storage;
            // core routing is untouched.
            #[cfg(feature = "benchmarks")]
            Ok(store) => benchmark::handle_request(req, store).await,
            Err(reason) => Ok(benchmark::unavailable_response(reason)),
            #[cfg(not(feature = "benchmarks"))]
            Ok(_) => Ok(benchmark::unavailable_response(
                "the `benchmarks` cargo feature is not compiled into this build \
                 (rebuild with --features benchmarks to enable the explorer)",
            )),
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
            let models = build_model_list(&state).await;
            let resp = json_response(StatusCode::OK, &models);
            into_unsync(resp)
        }

        ("POST", "/v1/chat/completions") => {
            let session_id = extract_session_id(req.headers());
            let user_agent = req.headers().get("user-agent").and_then(|v| v.to_str().ok()).unwrap_or("").trim().to_string();
            // v2 L3 role signal: x-brainrouter-role (default main).
            let role = req.headers().get("x-brainrouter-role").and_then(|v| v.to_str().ok()).unwrap_or("main").trim().to_string();
            match handle_chat_completion(req, state, cwd, session_id, user_agent, peer_addr, role).await {
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
            let _l3_role = req.headers().get("x-brainrouter-role").and_then(|v| v.to_str().ok()).unwrap_or("main").trim().to_string();
            let user_agent = req.headers().get("user-agent").and_then(|v| v.to_str().ok()).unwrap_or("").trim().to_string();
            match handle_anthropic_messages(req, state, cwd, session_id, user_agent, peer_addr, _l3_role).await {
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
                .body(Full::new(Bytes::from(
                    main_dashboard_html(&state.platform.id).into_bytes(),
                )))
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

        // ── v2 L4: platform probe + swap supervisor API ──────────────────────
        ("GET", "/api/platform") => {
            into_unsync(json_response(StatusCode::OK, &*state.platform))
        }

        ("GET", "/api/engines") | ("GET", "/api/engines/status") => {
            into_unsync(json_response(StatusCode::OK, &engines_json(&state)))
        }

        ("GET", "/api/engines/queue") => {
            into_unsync(json_response(
                StatusCode::OK,
                &serde_json::json!({ "queue": state.swap.status().queue }),
            ))
        }

        ("POST", "/api/engines/queue") | ("POST", "/api/engines/ensure") => {
            let val = read_json_body(req).await;
            let Some(model) = val.get("model").and_then(|m| m.as_str()) else {
                return Ok(into_unsync(json_response(
                    StatusCode::BAD_REQUEST,
                    &serde_json::json!({ "error": "model is required" }),
                )));
            };
            match state.swap.ensure(model).await {
                Ok(outcome) => into_unsync(json_response(
                    StatusCode::OK,
                    &serde_json::json!({
                        "model": match &outcome {
                            crate::swap_scheduler::EnsureOutcome::Attach { model, .. } => model,
                        },
                        "endpoint": outcome.endpoint(),
                        "spawned": outcome.spawned(),
                    }),
                )),
                Err(error) => into_unsync(swap_api_error(error)),
            }
        }

        ("POST", "/api/engines/evict") => {
            let val = read_json_body(req).await;
            let Some(model) = val.get("model").and_then(|m| m.as_str()) else {
                return Ok(into_unsync(json_response(
                    StatusCode::BAD_REQUEST,
                    &serde_json::json!({ "error": "model is required" }),
                )));
            };
            match state.swap.evict(model, "explicit evict") {
                Ok(stopped) => into_unsync(json_response(
                    StatusCode::OK,
                    &serde_json::json!({ "model": model, "stopped": stopped }),
                )),
                Err(error) => into_unsync(swap_api_error(error)),
            }
        }

        ("DELETE", p) if p.starts_with("/api/engines/queue/") => {
            match p["/api/engines/queue/".len()..].parse::<u64>() {
                Ok(id) if state.swap.cancel(id) => into_unsync(json_response(
                    StatusCode::OK,
                    &serde_json::json!({ "cancelled": true, "id": id }),
                )),
                _ => into_unsync(json_response(
                    StatusCode::NOT_FOUND,
                    &serde_json::json!({ "error": "Unknown queue id" }),
                )),
            }
        }

        ("POST", "/api/engines/pin") => {
            let val = read_json_body(req).await;
            let model = val.get("model").and_then(|m| m.as_str()).unwrap_or("");
            let gib = val
                .get("reserved_gib")
                .or_else(|| val.get("gib"))
                .and_then(|g| g.as_f64())
                .unwrap_or(0.0);
            match state.swap.catalog().pin(model, gib) {
                Ok(()) => {
                    let _ = state
                        .swap
                        .ledger()
                        .record("pin", model, gib, "sized pin reservation");
                    into_unsync(json_response(
                        StatusCode::OK,
                        &serde_json::json!({ "pins": state.swap.catalog().pins() }),
                    ))
                }
                Err(error) => into_unsync(json_response(
                    StatusCode::BAD_REQUEST,
                    &serde_json::json!({ "error": error.to_string() }),
                )),
            }
        }

        ("POST", "/api/engines/unpin") => {
            let val = read_json_body(req).await;
            let model = val.get("model").and_then(|m| m.as_str()).unwrap_or("");
            match state.swap.catalog().unpin(model) {
                Ok(removed) => {
                    if removed {
                        let _ = state
                            .swap
                            .ledger()
                            .record("unpin", model, 0.0, "pin removed");
                    }
                    into_unsync(json_response(
                        StatusCode::OK,
                        &serde_json::json!({ "removed": removed, "pins": state.swap.catalog().pins() }),
                    ))
                }
                Err(error) => into_unsync(json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &serde_json::json!({ "error": error.to_string() }),
                )),
            }
        }

        ("GET", "/api/engines/import/preview") => {
            let requested = query_param(req.uri().query().unwrap_or(""), "path");
            let config = match engines_import_path(&state, Some(&requested)) {
                Ok(path) => path,
                Err(error) => {
                    return Ok(into_unsync(json_response(
                        StatusCode::BAD_REQUEST,
                        &serde_json::json!({ "error": error.to_string() }),
                    )))
                }
            };
            match crate::swap_catalog::import_llama_swap_file(&config) {
                Ok(plan) => into_unsync(json_response(
                    StatusCode::OK,
                    &serde_json::json!({
                        "config": config.display().to_string(),
                        "would_import": plan.entries.len(),
                        "entries": plan.entries.keys().collect::<Vec<_>>(),
                        "groups": plan.groups,
                        "warnings": plan.warnings,
                        "lines": plan.summary_lines(),
                    }),
                )),
                Err(error) => into_unsync(json_response(
                    StatusCode::BAD_REQUEST,
                    &serde_json::json!({ "error": error.to_string() }),
                )),
            }
        }

        ("POST", "/api/engines/import/apply") => {
            let val = read_json_body(req).await;
            let requested = val.get("path").and_then(|p| p.as_str());
            let config = match engines_import_path(&state, requested) {
                Ok(path) => path,
                Err(error) => {
                    return Ok(into_unsync(json_response(
                        StatusCode::BAD_REQUEST,
                        &serde_json::json!({ "error": error.to_string() }),
                    )))
                }
            };
            match crate::swap_catalog::import_llama_swap_file(&config)
                .and_then(|plan| state.swap.catalog().apply_import(&plan))
            {
                Ok(outcome) => into_unsync(json_response(
                    StatusCode::OK,
                    &serde_json::json!({
                        "config": config.display().to_string(),
                        "added": outcome.added,
                        "updated": outcome.updated,
                        "unchanged": outcome.unchanged,
                        "skipped": outcome.skipped,
                    }),
                )),
                Err(error) => into_unsync(json_response(
                    StatusCode::BAD_REQUEST,
                    &serde_json::json!({ "error": error.to_string() }),
                )),
            }
        }

        ("POST", "/api/engines/catalog/add") => {
            let val = read_json_body(req).await;
            match serde_json::from_value::<crate::swap_catalog::CatalogEntry>(val) {
                Ok(entry) => match crate::swap_catalog::validate_entry(&entry)
                    .and_then(|()| state.swap.catalog().upsert_entry(entry))
                {
                    Ok(()) => into_unsync(json_response(
                        StatusCode::OK,
                        &serde_json::json!({ "ok": true }),
                    )),
                    Err(error) => into_unsync(json_response(
                        StatusCode::BAD_REQUEST,
                        &serde_json::json!({ "error": error.to_string() }),
                    )),
                },
                Err(error) => into_unsync(json_response(
                    StatusCode::BAD_REQUEST,
                    &serde_json::json!({ "error": error.to_string() }),
                )),
            }
        }

        ("POST", "/api/engines/catalog/rm") => {
            let val = read_json_body(req).await;
            let model = val.get("model").and_then(|m| m.as_str()).unwrap_or("");
            match state.swap.catalog().remove_entry(model) {
                Ok(removed) => into_unsync(json_response(
                    StatusCode::OK,
                    &serde_json::json!({ "removed": removed }),
                )),
                Err(error) => into_unsync(json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &serde_json::json!({ "error": error.to_string() }),
                )),
            }
        }

        ("GET", p) if p == "/api/engines/logs" => {
            let query = req.uri().query().unwrap_or("");
            let model = query_param(query, "model");
            let lines = query_param(query, "lines")
                .parse::<usize>()
                .unwrap_or(200)
                .min(2000);
            into_unsync(json_response(
                StatusCode::OK,
                &serde_json::json!({
                    "model": model,
                    "lines": state.swap.logs(&model, lines),
                }),
            ))
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
                &state,
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
                upgrade_toolbox(&state, name, &image).await
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

        // ── v2 L2 Plugins API (replaces the v1 bridge status/toggle pair) ──
        ("GET", "/api/plugins") => {
            let resp = json_response(StatusCode::OK, &state.plugins.dashboard_json());
            into_unsync(resp)
        }

        ("GET", p) if p.starts_with("/api/plugins/logs") => {
            let query = req.uri().query().unwrap_or("");
            let id = query_param(query, "id");
            if id.is_empty() {
                let resp = json_response(StatusCode::BAD_REQUEST, &ErrorResponse { error: "missing ?id=".into() });
                return Ok(into_unsync(resp));
            }
            let tail: usize = query_param(query, "tail").parse().unwrap_or(400);
            let path = match state.plugins.log_path_checked(&id) {
                Ok(p) => p,
                Err(e) => {
                    let resp = json_response(StatusCode::BAD_REQUEST, &ErrorResponse { error: e.to_string() });
                    return Ok(into_unsync(resp));
                }
            };
            match std::fs::read_to_string(&path) {
                Ok(text) => {
                    let lines: Vec<&str> = text.lines().collect();
                    let start = lines.len().saturating_sub(tail.max(1).min(5000));
                    let body = lines[start..].join("\n");
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("Content-Type", "text/plain; charset=utf-8")
                        .body(Full::new(Bytes::from(body)).map_err(|e| anyhow::anyhow!(e)).boxed_unsync())
                        .unwrap()
                }
                Err(e) => {
                    let resp = json_response(StatusCode::NOT_FOUND, &ErrorResponse { error: format!("no plugin logs for {id}: {e}") });
                    into_unsync(resp)
                }
            }
        }

        ("POST", "/api/plugins/install") => {
            let body = read_json_body(req).await;
            let dir = body.get("dir").and_then(|v| v.as_str()).unwrap_or("").to_string();
            plugins_reply(state.plugins.install(std::path::Path::new(&dir)))
        }

        ("POST", "/api/plugins/enable") => {
            let body = read_json_body(req).await;
            let id = body.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            plugins_reply(state.plugins.enable(&id).await)
        }

        ("POST", "/api/plugins/disable") => {
            let body = read_json_body(req).await;
            let id = body.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            plugins_reply(state.plugins.disable(&id).await)
        }

        ("POST", "/api/plugins/remove") => {
            let body = read_json_body(req).await;
            let id = body.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            plugins_reply(state.plugins.remove(&id).await)
        }

        ("POST", "/api/plugins/config") => {
            let body = read_json_body(req).await;
            let id = body.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let key = body.get("key").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let value = body.get("value").cloned().unwrap_or(serde_json::Value::Null);
            plugins_reply(state.plugins.config_set(&id, &key, value))
        }

        // ── Routing mode override API ─────────────────────────────────────────
        ("GET", "/api/routing-mode") => {
            // Legacy mirror of the global main card's backend.
            let assignment = state.roles.assignment(crate::roles::Role::Main, None);
            let selector = crate::roles::parse_selector(&assignment);
            let resp = json_response(StatusCode::OK, &serde_json::json!({
                "mode": selector.backend(),
                "assignment": assignment,
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
            let selector = crate::roles::parse_selector(&update.mode);
            if !matches!(selector, crate::roles::Selector::Auto | crate::roles::Selector::Local | crate::roles::Selector::Cloud) {
                return Ok(routing_error(StatusCode::BAD_REQUEST, "mode must be auto, local, or cloud"));
            }
            let cards = Arc::clone(&state.roles);
            let mode = update.mode.clone();
            match tokio::task::spawn_blocking(move || cards.set_assignment(crate::roles::GLOBAL_SCOPE, crate::roles::Role::Main, &mode)).await
                .map_err(anyhow::Error::from).and_then(|result| result) {
                Ok(()) => {
                    let code = match selector.backend() { "cloud" => 1, "local" => 2, _ => 0 };
                    state.routing_mode.store(code, AtomicOrdering::Relaxed);
                    into_unsync(json_response(StatusCode::OK, &serde_json::json!({ "mode": update.mode })))
                }
                Err(error) => routing_error(StatusCode::INTERNAL_SERVER_ERROR, error),
            }
        }

        // ── v2 L3: role cards + policy (replaces routing-profile + pins) ────
        ("GET", "/api/roles") => {
            into_unsync(json_response(StatusCode::OK, &roles_json(&state)))
        }

        ("POST", "/api/roles/set") => {
            #[derive(serde::Deserialize)]
            #[serde(deny_unknown_fields)]
            struct RoleSet {
                role: String,
                assignment: String,
                #[serde(default)]
                scope: Option<String>,
            }
            let body: RoleSet = match read_routing_json(req).await {
                Ok(body) => body,
                Err(error) => return Ok(routing_error(StatusCode::BAD_REQUEST, error)),
            };
            let role = match crate::roles::Role::parse(&body.role) {
                Ok(role) => role,
                Err(error) => return Ok(routing_error(StatusCode::BAD_REQUEST, error)),
            };
            let scope = body.scope.unwrap_or_else(|| crate::roles::GLOBAL_SCOPE.to_string());
            let assignment = body.assignment.clone();
            let cards = Arc::clone(&state.roles);
            match tokio::task::spawn_blocking(move || cards.set_assignment(&scope, role, &assignment)).await
                .map_err(anyhow::Error::from).and_then(|result| result)
            {
                Ok(()) => into_unsync(json_response(StatusCode::OK, &roles_json(&state))),
                Err(error) => routing_error(StatusCode::BAD_REQUEST, error),
            }
        }

        ("POST", "/api/roles/remove") => {
            #[derive(serde::Deserialize)]
            #[serde(deny_unknown_fields)]
            struct RoleRemove {
                role: String,
                #[serde(default)]
                scope: Option<String>,
            }
            let body: RoleRemove = match read_routing_json(req).await {
                Ok(body) => body,
                Err(error) => return Ok(routing_error(StatusCode::BAD_REQUEST, error)),
            };
            let role = match crate::roles::Role::parse(&body.role) {
                Ok(role) => role,
                Err(error) => return Ok(routing_error(StatusCode::BAD_REQUEST, error)),
            };
            let scope = body.scope.unwrap_or_else(|| crate::roles::GLOBAL_SCOPE.to_string());
            let cards = Arc::clone(&state.roles);
            match tokio::task::spawn_blocking(move || cards.remove_assignment(&scope, role)).await
                .map_err(anyhow::Error::from).and_then(|result| result)
            {
                Ok(removed) => into_unsync(json_response(StatusCode::OK, &serde_json::json!({
                    "removed": removed, "roles": roles_json(&state),
                }))),
                Err(error) => routing_error(StatusCode::BAD_REQUEST, error),
            }
        }

        // Policy set: the contract pins POST /api/policy/set; the dashboard
        // probes the short aliases — all routes share one handler.
        ("POST", "/api/policy/set") | ("POST", "/api/policy") | ("POST", "/api/roles/policy") | ("POST", "/api/role-cards/policy") => {
            let body: serde_json::Value = match read_routing_json(req).await {
                Ok(body) => body,
                Err(error) => return Ok(routing_error(StatusCode::BAD_REQUEST, error)),
            };
            let scope = body.get("scope").and_then(|v| v.as_str()).unwrap_or(crate::roles::GLOBAL_SCOPE);
            let privacy = match body.get("privacy").and_then(|v| v.as_str()) {
                Some("cloud-ok") => crate::roles::Privacy::CloudOk,
                Some("local-only") => crate::roles::Privacy::LocalOnly,
                _ => return Ok(routing_error(StatusCode::BAD_REQUEST, "privacy must be cloud-ok or local-only")),
            };
            let quality = match body.get("quality").and_then(|v| v.as_str()) {
                Some("fast") => crate::roles::Quality::Fast,
                Some("balanced") => crate::roles::Quality::Balanced,
                Some("maximum") => crate::roles::Quality::Maximum,
                _ => return Ok(routing_error(StatusCode::BAD_REQUEST, "quality must be fast, balanced, or maximum")),
            };
            let cards = Arc::clone(&state.roles);
            let scope = scope.to_string();
            match tokio::task::spawn_blocking(move || cards.set_policy(&scope, privacy, quality)).await
                .map_err(anyhow::Error::from).and_then(|result| result)
            {
                Ok(()) => into_unsync(json_response(StatusCode::OK, &policy_json(&state))),
                Err(error) => routing_error(StatusCode::BAD_REQUEST, error),
            }
        }

        ("GET", "/api/policy") => into_unsync(json_response(StatusCode::OK, &policy_json(&state))),

        // ── v2 L3: credentials UI (masked list + add/update/remove/test) ────
        ("GET", "/api/credentials") | ("GET", "/api/credentials/list") => {
            into_unsync(json_response(StatusCode::OK, &credentials_json(&state)))
        }

        ("POST", "/api/credentials/add") | ("POST", "/api/credentials/update") | ("POST", "/api/credentials/save") => {
            let record: crate::credentials::CredentialRecord = match read_routing_json(req).await {
                Ok(record) => record,
                Err(error) => return Ok(routing_error(StatusCode::BAD_REQUEST, error)),
            };
            if record.id.trim().is_empty() {
                return Ok(routing_error(StatusCode::BAD_REQUEST, "credential id must not be empty"));
            }
            let store = Arc::clone(&state.credentials);
            match tokio::task::spawn_blocking(move || store.put(record)).await
                .map_err(anyhow::Error::from).and_then(|result| result)
            {
                Ok(()) => into_unsync(json_response(StatusCode::OK, &credentials_json(&state))),
                Err(error) => routing_error(StatusCode::INTERNAL_SERVER_ERROR, error),
            }
        }

        ("POST", "/api/credentials/remove") => {
            let body: serde_json::Value = match read_routing_json(req).await {
                Ok(body) => body,
                Err(error) => return Ok(routing_error(StatusCode::BAD_REQUEST, error)),
            };
            let Some(id) = body.get("id").and_then(|v| v.as_str()) else {
                return Ok(routing_error(StatusCode::BAD_REQUEST, "missing id"));
            };
            let store = Arc::clone(&state.credentials);
            let id = id.to_string();
            match tokio::task::spawn_blocking(move || store.remove(&id)).await
                .map_err(anyhow::Error::from).and_then(|result| result)
            {
                Ok(removed) => into_unsync(json_response(StatusCode::OK, &serde_json::json!({
                    "removed": removed, "credentials": credentials_json(&state),
                }))),
                Err(error) => routing_error(StatusCode::INTERNAL_SERVER_ERROR, error),
            }
        }

        ("POST", "/api/credentials/test") => {
            let body: serde_json::Value = match read_routing_json(req).await {
                Ok(body) => body,
                Err(error) => return Ok(routing_error(StatusCode::BAD_REQUEST, error)),
            };
            let Some(id) = body.get("id").and_then(|v| v.as_str()) else {
                return Ok(routing_error(StatusCode::BAD_REQUEST, "missing id"));
            };
            into_unsync(json_response(StatusCode::OK, &credential_test_json(&state, id)))
        }

        // REST alias (design "add/edit/delete"): DELETE /api/credentials/{id}.
        ("DELETE", p) if p.starts_with("/api/credentials/") => {
            let id = &p["/api/credentials/".len()..];
            if id.is_empty() || id.contains('/') {
                return Ok(routing_error(StatusCode::BAD_REQUEST, "invalid credential id"));
            }
            let store = Arc::clone(&state.credentials);
            let id = id.to_string();
            match tokio::task::spawn_blocking(move || store.remove(&id)).await
                .map_err(anyhow::Error::from).and_then(|result| result)
            {
                Ok(removed) => into_unsync(json_response(StatusCode::OK, &serde_json::json!({
                    "removed": removed, "credentials": credentials_json(&state),
                }))),
                Err(error) => routing_error(StatusCode::INTERNAL_SERVER_ERROR, error),
            }
        }

        // Per-credential test alias: POST /api/credentials/{id}/test.
        ("POST", p) if p.starts_with("/api/credentials/") && p.ends_with("/test") => {
            let id = &p["/api/credentials/".len()..p.len() - "/test".len()];
            if id.is_empty() || id.contains('/') {
                return Ok(routing_error(StatusCode::BAD_REQUEST, "invalid credential id"));
            }
            into_unsync(json_response(StatusCode::OK, &credential_test_json(&state, id)))
        }

        ("POST", "/api/credentials/audit") => {
            // v2 L3 Step 9 I6: a usage audit — every masked record plus the
            // plugins (grants consumers) that reference it. No plaintext,
            // ever: records are the store's own masked shape. Consumers are
            // the credential record's OWN grants (plugin ids) intersected
            // with registered plugin ids — NOT the manifests' RPC-method
            // grants (disjoint vocabularies; Bruce r2 Finding 1).
            let plugins = state.plugins.list().unwrap_or_default();
            let plugin_ids: Vec<String> = plugins
                .iter()
                .filter_map(|c| c.get("id").and_then(|v| v.as_str()).map(str::to_string))
                .collect();
            let mut audit = state
                .credentials
                .list_masked()
                .unwrap_or_default()
                .into_iter()
                .map(|r| {
                    let consumers: Vec<String> = r
                        .get("grants")
                        .and_then(|g| g.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|g| g.as_str())
                                .filter(|g| plugin_ids.iter().any(|p| p == g))
                                .map(str::to_string)
                                .collect()
                        })
                        .unwrap_or_default();
                    let mut v = r;
                    if let Some(obj) = v.as_object_mut() {
                        obj.insert("consumers".into(), serde_json::json!(consumers));
                    }
                    v
                })
                .collect::<Vec<_>>();
            audit.sort_by_key(|v| {
                v.get("id")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string()
            });
            into_unsync(json_response(
                StatusCode::OK,
                &serde_json::json!({ "credentials": audit }),
            ))
        }

        // ── v2 L3: decision slot (read-only describe + test classify) ───────
        ("GET", "/api/decision") => into_unsync(json_response(
            StatusCode::OK,
            &state.decision.describe(),
        )),

        // v2 L3 Step 9 I6: the Decisions panel's feature summary (what the
        // slot sees per request shape; excerpt-free).
        ("POST", "/api/decision/features") => {
            #[derive(serde::Deserialize)]
            struct FeaturesProbe {
                #[serde(default)]
                text: Option<String>,
                #[serde(default)]
                role: Option<String>,
                #[serde(default)]
                selector: Option<String>,
            }
            let body: FeaturesProbe = match read_routing_json(req).await {
                Ok(body) => body,
                Err(error) => return Ok(routing_error(StatusCode::BAD_REQUEST, error)),
            };
            let request = ChatCompletionRequest {
                model: body.selector.unwrap_or_else(|| "auto".into()),
                messages: vec![crate::types::ChatMessage {
                    role: "user".into(),
                    content: Some(serde_json::Value::String(
                        body.text.unwrap_or_default(),
                    )),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                }],
                stream: Some(false),
                temperature: None,
                max_tokens: None,
                top_p: None,
                stop: None,
                extra: serde_json::Value::Null,
            };
            let features = crate::decision::RequestFeatures::extract(
                &request,
                64,
                body.role.as_deref().unwrap_or("main"),
                &request.model,
                crate::roles::Privacy::CloudOk,
                crate::roles::Quality::Fast,
                "",
            );
            into_unsync(json_response(StatusCode::OK, &features.to_json(false)))
        }

        ("POST", "/api/decision/test") => {
            #[derive(serde::Deserialize)]
            struct DecisionTest { text: String }
            let body: DecisionTest = match read_routing_json(req).await {
                Ok(body) => body,
                Err(error) => return Ok(routing_error(StatusCode::BAD_REQUEST, error)),
            };
            let request = ChatCompletionRequest {
                model: "auto".into(),
                messages: vec![crate::types::ChatMessage {
                    role: "user".into(),
                    content: Some(serde_json::Value::String(body.text)),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                }],
                stream: None,
                temperature: None,
                max_tokens: None,
                top_p: None,
                stop: None,
                extra: serde_json::json!({}),
            };
            let policy = state.roles.policy_for(None);
            let features = crate::decision::RequestFeatures::extract(
                &request, 0, "main", "auto", policy.privacy, policy.quality, "",
            );
            let outcome = state.decision.decide(&features).await;
            let mut out = serde_json::to_value(&outcome.decision).unwrap_or_default();
            if let Some(map) = out.as_object_mut() {
                map.insert("fallback".into(), serde_json::json!(outcome.fallback.map(|f| f.as_str())));
            }
            into_unsync(json_response(StatusCode::OK, &out))
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
            // v2 L3 Step 9 B2: rewrite requires the DECISION SLOT enabled
            // (any kind — the heuristic slot routes auto/local exactly like
            // the classifier did). The server health check applies only to
            // the bonsai profile, the one kind backed by the external
            // llama-server. Turning off is always allowed.
            let slot_enabled = state.decision.config().enabled;
            let bonsai_health_needed = state.decision.rewrite_requires_launch_health();
            let slot_ok = slot_enabled
                && (!bonsai_health_needed
                    || (state.bonsai.is_running() && state.bonsai.healthy().await));
            let rejected = match val.get("enabled").and_then(|v| v.as_bool()) {
                Some(true) if !slot_ok => true,
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
                        error: "Prompt rewrite requires the decision slot to be enabled (and, for the bonsai classifier profile, its server healthy)".into(),
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
                // Persist ALL live flags (full snapshot) so the choice survives a
                // restart and never clobbers a sibling flag. Write failure is
                // logged, not fatal — the in-memory switch already took effect.
                if let Err(e) = crate::review::runtime_state::save_state(
                    &crate::review::runtime_state::state_path(),
                    enabled,
                    state.pr_guidelines_enabled.load(AtomicOrdering::Relaxed),
                    state.review_service.get_config().hankndory_integration,
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
                // Full-snapshot persist of all live flags (see /enabled above).
                if let Err(e) = crate::review::runtime_state::save_state(
                    &crate::review::runtime_state::state_path(),
                    state.code_review_enabled.load(AtomicOrdering::Relaxed),
                    enabled,
                    state.review_service.get_config().hankndory_integration,
                ) {
                    tracing::warn!(error = %e, "Failed to persist review_runtime_state.json");
                }
            }
            into_unsync(json_response(StatusCode::OK, &serde_json::json!({
                "enabled": state.pr_guidelines_enabled.load(AtomicOrdering::Relaxed),
            })))
        }

        // ── Phase-1b: design-aware (HankNDory) review runtime toggle ────────
        // Runtime override for `review.hankndory_integration`, persisted so the
        // dashboard choice supersedes the YAML seed across restarts. The live
        // value lives in the ProfileStore (what `run_loop` reads).
        ("GET", "/api/review/hankndory") => {
            into_unsync(json_response(StatusCode::OK, &serde_json::json!({
                "enabled": state.review_service.get_config().hankndory_integration,
            })))
        }

        ("POST", "/api/review/hankndory") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let val: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();
            if let Some(enabled) = val.get("enabled").and_then(|v| v.as_bool()) {
                // Update the live value read by the review loop, then persist all
                // flags (the override becomes authoritative over the YAML seed).
                if let Err(e) = state.roles.set_hankndory_integration(enabled) {
                    tracing::warn!(error = %e, "Failed to persist the review card extras");
                }
                if let Err(e) = crate::review::runtime_state::save_state(
                    &crate::review::runtime_state::state_path(),
                    state.code_review_enabled.load(AtomicOrdering::Relaxed),
                    state.pr_guidelines_enabled.load(AtomicOrdering::Relaxed),
                    enabled,
                ) {
                    tracing::warn!(error = %e, "Failed to persist review_runtime_state.json");
                }
            }
            into_unsync(json_response(StatusCode::OK, &serde_json::json!({
                "enabled": state.review_service.get_config().hankndory_integration,
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

        // ── Generalized toolbox catalog/container API (PR2, the supported
        // serving backends — see design doc §5c + the v2 L1 purge) ────
        ("GET", "/api/toolbox-catalog") => {
            let resp = toolbox_catalog_response().await;
            into_unsync(resp)
        }

        ("GET", "/api/toolbox-models") => {
            let resp = toolbox_models_response().await;
            into_unsync(resp)
        }

        ("GET", "/api/toolbox-containers") => {
            let resp = toolbox_containers_list(&state).await;
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

        // ── gufo Server Mode (design doc DI-5/DI-7) ───────────────────────────
        ("GET", "/api/server-mode/gufo/status") => {
            let resp = server_mode_gufo_status_response().await;
            into_unsync(resp)
        }

        ("POST", "/api/server-mode/gufo/start") => {
            let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let resp = server_mode_gufo_start_response(&state, &body_bytes).await;
            into_unsync(resp)
        }

        ("POST", "/api/server-mode/gufo/stop") => {
            let resp = server_mode_gufo_stop_response(&state).await;
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

        ("POST", "/api/llama-swap/register-model") => {
            let resp = register_llama_swap_model_request(req, &state).await;
            into_unsync(resp)
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
                        let _guard = state.llama_swap_config_lock.lock().await;
                        let tmp_path = unique_sibling_path(&state.llama_swap_config_path, "tmp");
                        let write_result = atomic_write_path(&state.llama_swap_config_path, &tmp_path, body.as_bytes());
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
    role: String,
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
    // v2 L4 additive: stamp the swap-catalog id at route time so the LRU
    // never evicts a model with an active (or loading) stream.
    if let Some(catalog_id) = state.swap.catalog().resolve_alias(&request.model) {
        handle.set_catalog_id(catalog_id);
    }
    let (tx, rx) = tokio::sync::oneshot::channel();
    let routing_task = tokio::spawn(async move {
        let result = tokio::select! {
            result = state.router.route_tagged_role(request, session_id, cwd, user_agent, &role) => result,
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

    // v2 L3 fast-fail window: a routing failure that lands within the first
    // few seconds (explicit selector refused, chain exhausted, reviewer strict
    // bail, provider 5xx on a fast path) surfaces as a real HTTP status.
    // Slower resolutions (model loading — minutes) keep v1's deferred
    // headers: the stream commits 200 and the error arrives in-body, exactly
    // as before, so harness first-event timeouts never fire during a load.
    let mut rx = Box::pin(rx);
    match tokio::time::timeout(L3_FAST_FAIL_WINDOW, rx.as_mut()).await {
        Ok(Ok(Ok(stream))) => {
            let stream_body = StreamBody::new(
                stream
                    .map(|chunk| chunk.map(Frame::data).map_err(|e| e)),
            );
            let response = Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "text/event-stream")
                .header("cache-control", "no-cache")
                .header("connection", "keep-alive")
                .body(stream_body.boxed_unsync())?;
            return Ok(response);
        }
        Ok(Ok(Err(error))) => {
            routing_task.abort();
            if let Some(response) = swap_busy_response(&error) {
                return Ok(into_unsync(response));
            }
            return Ok(into_unsync(json_response(
                StatusCode::BAD_GATEWAY,
                &ErrorResponse { error: format!("routing failed: {error}") },
            )));
        }
        Ok(Err(_)) => {
            routing_task.abort();
            return Ok(into_unsync(json_response(
                StatusCode::BAD_GATEWAY,
                &ErrorResponse { error: "routing task ended".into() },
            )));
        }
        Err(_elapsed) => {} // headline still pending (model load): defer, v1 path
    }

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

/// v2 L3: how long the proxy waits for the routing headline (connect /
/// response-headline failure) before committing streamed 200 headers.
const L3_FAST_FAIL_WINDOW: std::time::Duration = std::time::Duration::from_secs(15);

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
    role: String,
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
    if let Some(catalog_id) = state.swap.catalog().resolve_alias(&oai_request.model) {
        handle.set_catalog_id(catalog_id);
    }
    let (tx, rx) = tokio::sync::oneshot::channel();
    let routing_task = tokio::spawn(async move {
        let result = tokio::select! {
            result = state.router.route_tagged_role(oai_request, session_id, cwd, user_agent, &role) => result,
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
    match restart_systemd_user_service(service).await {
        Ok(()) => json_response(StatusCode::OK, &serde_json::json!({
            "status": "ok",
            "service": service,
            "message": format!("{} restarted", service)
        })),
        Err(e) if e.starts_with("Unknown service:") => json_response(StatusCode::BAD_REQUEST, &ErrorResponse { error: e }),
        Err(e) => json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse { error: e }),
    }
}

async fn restart_systemd_user_service(service: &str) -> Result<(), String> {
    const ALLOWED: &[&str] = &["llama-swap", "manifest", "brainrouter"];
    if !ALLOWED.contains(&service) {
        return Err(format!("Unknown service: {}", service));
    }

    info!(service, "Restarting systemd user service");
    let output = tokio::process::Command::new("systemctl")
        .args(["--user", "restart", service])
        .output()
        .await;

    match output {
        Ok(out) if out.status.success() => {
            info!(service, "Service restarted successfully");
            Ok(())
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            error!(service, %stderr, "systemctl restart returned non-success");
            Err(format!("Failed to restart {}: {}", service, stderr.trim()))
        }
        Err(e) => {
            error!(service, error = %e, "systemctl restart failed");
            Err(format!("Failed to restart {}: {}", service, e))
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

/// Legacy podman label still honored for backwards compatibility.
const LABEL_MANAGED: &str = "io.brainrouter.managed";

type ToolboxOpResult = Result<(), Box<Response<Full<Bytes>>>>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ContainerSnapshot {
    id: String,
    managed_label: String,
    image: String,
}

/// Loads and type-parses the vendored toolbox catalog, restricted to
/// entries brainrouter can act on. Returns a human-readable error string
/// (used directly in `ErrorResponse`s) rather than a custom error type,
/// matching this file's existing preference for cheap, situational errors
/// over a dedicated error enum for read paths that should essentially never
/// fail (the catalog is embedded at compile time and covered by unit tests).
fn load_typed_toolbox_catalog() -> Result<ToolboxCatalog, String> {
    // Effective catalog = vendored cockpit feed + brainrouter's gufo overlay,
    // validated as a whole and fail-closed (see
    // `toolbox_catalog::load_effective_typed_catalog`).
    let (toolboxes, _models) =
        toolbox_catalog::load_effective_typed_catalog().map_err(|e| e.to_string())?;
    Ok(toolboxes)
}

async fn inspect_container(name: &str) -> Option<ContainerSnapshot> {
    let out = tokio::process::Command::new("podman")
        .args([
            "inspect",
            "--format",
            &format!("{{{{.Id}}}}|{{{{ index .Config.Labels \"{LABEL_MANAGED}\" }}}}|{{{{.Image}}}}"),
            name,
        ])
        .output()
        .await
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut parts = stdout.trim().splitn(3, '|');
    let id = parts.next()?.to_string();
    let managed_label = parts.next().unwrap_or_default().to_string();
    let image = parts.next().unwrap_or_default().to_string();
    (!id.is_empty()).then_some(ContainerSnapshot { id, managed_label, image })
}

fn authorize(snapshot: Option<&ContainerSnapshot>, entry: Option<&OwnershipEntry>) -> bool {
    let Some(snapshot) = snapshot else { return false; };
    snapshot.managed_label == "true" || entry.map(|entry| entry.container_id == snapshot.id).unwrap_or(false)
}

async fn toolbox_container_is_managed(name: &str, sidecar_path: &std::path::Path) -> bool {
    let snapshot = inspect_container(name).await;
    let entry = crate::managed_toolboxes::get(sidecar_path, name);
    authorize(snapshot.as_ref(), entry.as_ref())
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


#[derive(Debug, Deserialize)]
struct RegisterLlamaSwapModelRequest {
    model_id: String,
    quant_pattern: String,
}

#[derive(Debug, Serialize)]
struct RegisterLlamaSwapModelResponse {
    status: &'static str,
    llama_swap_id: String,
    model_path: String,
    message: String,
    reloaded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    warning: Option<String>,
}

async fn register_llama_swap_model_request(req: Request<Incoming>, state: &AppState) -> Response<Full<Bytes>> {
    let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
    if body_bytes.len() > 16_384 {
        return json_response(StatusCode::BAD_REQUEST, &ErrorResponse { error: "Request body too large (max 16KB)".to_string() });
    }
    let request: RegisterLlamaSwapModelRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => return json_response(StatusCode::BAD_REQUEST, &ErrorResponse { error: format!("invalid request body: {e}") }),
    };
    match register_llama_swap_model(state, request).await {
        Ok(resp) => json_response(StatusCode::OK, &resp),
        Err(e) => json_response(StatusCode::from_u16(e.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), &ErrorResponse { error: e.message().to_string() }),
    }
}

async fn register_llama_swap_model(state: &AppState, request: RegisterLlamaSwapModelRequest) -> Result<RegisterLlamaSwapModelResponse, crate::llama_swap_register::RegisterError> {
    use crate::llama_swap_register::{build_entry, insert_model_entry, path_visibility_warning, resolve_primary_gguf, RegisterError};
    use crate::model_downloads::{effective_models_dir, resolve_catalog_entry};
    use crate::toolbox_catalog::{ModelPayload, SupportedServingBackend};

    if request.model_id.trim().is_empty() || request.quant_pattern.trim().is_empty() {
        return Err(RegisterError::BadRequest("model_id and quant_pattern are required".to_string()));
    }
    let (payload, storage) = resolve_catalog_entry(SupportedServingBackend::LlamaCpp, &request.model_id).map_err(register_error_from_download)?;
    let ModelPayload::LlamaCpp(model) = &payload else {
        return Err(RegisterError::BadRequest("only llama_cpp catalog entries can be registered in llama-swap; use Server Mode for ds4/gufo".to_string()));
    };
    let models_dir = effective_models_dir(SupportedServingBackend::LlamaCpp, &storage);
    let repo_basename = model.repo.rsplit('/').next().unwrap_or(&model.repo);
    let destination = models_dir.join(repo_basename);
    let model_path = resolve_primary_gguf(&destination, &request.quant_pattern)?;
    let warning = path_visibility_warning(&model_path);
    let entry = build_entry(&request.model_id, &model.name, &model_path);

    let _guard = state.llama_swap_config_lock.lock().await;
    let config_path = &state.llama_swap_config_path;
    let original = std::fs::read(config_path).map_err(|e| RegisterError::Internal(format!("failed to read llama-swap config `{}`: {e}", config_path.display())))?;
    let original_hash = sha256_hex(&original);
    let original_text = String::from_utf8(original.clone()).map_err(|e| RegisterError::BadRequest(format!("llama-swap config is not valid UTF-8: {e}")))?;
    let new_text = insert_model_entry(&original_text, &request.model_id, &entry)?;
    let changed = config_hash_changed(config_path, &original_hash).map_err(|e| RegisterError::Internal(format!("failed to re-read llama-swap config `{}`: {e}", config_path.display())))?;
    if changed {
        return Err(RegisterError::Conflict("llama-swap config changed while preparing registration; retry".to_string()));
    }

    let bak_path = rolling_backup_path(config_path);
    std::fs::write(&bak_path, &original).map_err(|e| RegisterError::Internal(format!("failed to write llama-swap config backup `{}`: {e}", bak_path.display())))?;
    let tmp_path = unique_sibling_path(config_path, "tmp");
    if let Err(e) = atomic_write_path(config_path, &tmp_path, new_text.as_bytes()) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(RegisterError::Internal(format!("failed to write llama-swap config: {e}")));
    }

    if let Err(e) = restart_systemd_user_service("llama-swap").await {
        restore_llama_swap_config(config_path, &bak_path).map_err(|restore_err| RegisterError::Internal(format!("registered config failed reload ({e}) and restore failed: {restore_err}; backup is `{}`", bak_path.display())))?;
        let _ = restart_systemd_user_service("llama-swap").await;
        return Ok(RegisterLlamaSwapModelResponse { status: "registered_reload_failed_restored", llama_swap_id: request.model_id, model_path: model_path.display().to_string(), message: format!("llama-swap reload failed and the previous config was restored: {e}"), reloaded: false, warning });
    }
    if let Err(e) = poll_llama_swap_model(&state.llama_swap_url, &request.model_id).await {
        restore_llama_swap_config(config_path, &bak_path).map_err(|restore_err| RegisterError::Internal(format!("registered config did not become healthy ({e}) and restore failed: {restore_err}; backup is `{}`", bak_path.display())))?;
        let _ = restart_systemd_user_service("llama-swap").await;
        return Ok(RegisterLlamaSwapModelResponse { status: "registered_reload_failed_restored", llama_swap_id: request.model_id, model_path: model_path.display().to_string(), message: format!("llama-swap reload did not report the new model and the previous config was restored: {e}"), reloaded: false, warning });
    }
    Ok(RegisterLlamaSwapModelResponse { status: "registered", llama_swap_id: request.model_id, model_path: model_path.display().to_string(), message: "Registered baseline in llama-swap and verified it appears in /v1/models".to_string(), reloaded: true, warning })
}

fn register_error_from_download(e: crate::model_downloads::DownloadError) -> crate::llama_swap_register::RegisterError {
    match e {
        crate::model_downloads::DownloadError::Validation(m) | crate::model_downloads::DownloadError::NotFound(m) => crate::llama_swap_register::RegisterError::BadRequest(m),
        crate::model_downloads::DownloadError::Conflict(m) => crate::llama_swap_register::RegisterError::Conflict(m),
        crate::model_downloads::DownloadError::Internal(m) => crate::llama_swap_register::RegisterError::Internal(m),
    }
}

fn unique_sibling_path(path: &Path, suffix: &str) -> PathBuf {
    let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("config.yaml");
    path.with_file_name(format!("{file_name}.{pid}.{nonce}.{suffix}", pid = std::process::id(), nonce = uuid::Uuid::new_v4()))
}

fn rolling_backup_path(path: &Path) -> PathBuf {
    let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("config.yaml");
    path.with_file_name(format!("{file_name}.bak"))
}

fn config_hash_changed(path: &Path, original_hash: &str) -> Result<bool, std::io::Error> {
    let current = std::fs::read(path)?;
    Ok(sha256_hex(&current) != original_hash)
}

fn atomic_write_path(path: &Path, tmp_path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    std::fs::write(tmp_path, bytes).and_then(|_| std::fs::rename(tmp_path, path))
}

fn restore_llama_swap_config(path: &Path, bak_path: &Path) -> std::io::Result<()> {
    let bytes = std::fs::read(bak_path)?;
    let tmp_path = unique_sibling_path(path, "restore.tmp");
    atomic_write_path(path, &tmp_path, &bytes)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

async fn poll_llama_swap_model(llama_swap_url: &str, id: &str) -> Result<(), String> {
    let url = format!("{}/v1/models", llama_swap_url);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut last_error = String::new();
    while std::time::Instant::now() < deadline {
        match VERSION_CLIENT.get(&url).timeout(std::time::Duration::from_secs(3)).send().await {
            Ok(resp) => match resp.json::<serde_json::Value>().await {
                Ok(body) if llama_swap_models_contains(&body, id) => return Ok(()),
                Ok(_) => last_error = format!("model `{id}` not present in /v1/models yet"),
                Err(e) => last_error = format!("failed to parse /v1/models response: {e}"),
            },
            Err(e) => last_error = format!("failed to fetch /v1/models: {e}"),
        }
        tokio::time::sleep(std::time::Duration::from_millis(750)).await;
    }
    Err(last_error)
}

fn llama_swap_models_contains(body: &serde_json::Value, id: &str) -> bool {
    body.get("data").and_then(|d| d.as_array()).is_some_and(|arr| arr.iter().any(|m| m.get("id").and_then(|v| v.as_str()) == Some(id)))
}

/// `GET /api/toolbox-models` — the effective model catalog (vendored + gufo
/// overlay), restricted to the supported backends.
pub async fn toolbox_models_response() -> Response<Full<Bytes>> {
    let (_toolboxes, models) = match toolbox_catalog::load_effective_typed_catalog() {
        Ok(v) => v,
        Err(e) => {
            error!(error = %e, "Failed to load effective toolbox model catalog");
            return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("failed to load effective model catalog: {e}"),
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
pub async fn toolbox_containers_list(state: &AppState) -> Response<Full<Bytes>> {
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
        let managed = toolbox_container_is_managed(name, &state.managed_toolboxes_path).await;
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

fn ownership_entry(container_id: String, image: String, catalog_id: String) -> OwnershipEntry {
    OwnershipEntry {
        container_id,
        image,
        catalog_id,
        catalog_revision: toolbox_catalog::vendored_catalog_revision(),
        created_at: chrono::Utc::now().to_rfc3339(),
    }
}

async fn pull_toolbox_image(container: &str, image: &str) -> ToolboxOpResult {
    let pull_out = tokio::process::Command::new("podman").args(["pull", image]).output().await;
    match pull_out {
        Ok(out) if out.status.success() => Ok(()),
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            error!(%stderr, %container, %image, "podman pull failed");
            Err(Box::new(json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("podman pull failed: {}", stderr.trim()),
            })))
        }
        Err(e) => {
            error!(error = %e, %container, "Failed to exec podman pull");
            Err(Box::new(json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Failed to exec podman: {}", e),
            })))
        }
    }
}

async fn toolbox_create(container: &str, image: &str) -> ToolboxOpResult {
    let create = tokio::process::Command::new("toolbox")
        .args(["create", "--image", image, container])
        .output()
        .await;
    match create {
        Ok(out) if out.status.success() => Ok(()),
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            error!(%stderr, %container, %image, "toolbox create failed");
            Err(Box::new(json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("toolbox create failed: {}", stderr.trim()),
            })))
        }
        Err(e) => {
            error!(error = %e, %container, "Failed to exec toolbox create");
            Err(Box::new(json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Failed to exec toolbox: {}", e),
            })))
        }
    }
}

async fn podman_rm_force(container_id: &str, container_name: &str) -> ToolboxOpResult {
    let remove = tokio::process::Command::new("podman")
        .args(["rm", "--force", container_id])
        .output()
        .await;
    match remove {
        Ok(out) if out.status.success() => Ok(()),
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            error!(%stderr, %container_name, container_id, "podman rm failed");
            Err(Box::new(json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("podman rm failed: {}", stderr.trim()),
            })))
        }
        Err(e) => {
            error!(error = %e, %container_name, container_id, "Failed to exec podman rm");
            Err(Box::new(json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
                error: format!("Failed to exec podman: {}", e),
            })))
        }
    }
}

fn record_toolbox_ownership(sidecar_path: &std::path::Path, container: &str, entry: OwnershipEntry) -> ToolboxOpResult {
    crate::managed_toolboxes::record(sidecar_path, container, entry).map_err(|e| {
        error!(error = %e, %container, "Failed to write managed-toolboxes sidecar");
        Box::new(json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
            error: format!("failed to persist toolbox ownership: {e}"),
        }))
    })
}

fn remove_toolbox_ownership(sidecar_path: &std::path::Path, container: &str) -> ToolboxOpResult {
    crate::managed_toolboxes::remove(sidecar_path, container).map_err(|e| {
        error!(error = %e, %container, "Failed to remove managed-toolboxes sidecar entry");
        Box::new(json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
            error: format!("failed to remove toolbox ownership: {e}"),
        }))
    })
}

async fn recreate_toolbox_container(
    sidecar_path: &std::path::Path,
    tb: &ToolboxDefinition,
    pull: bool,
    cleanup_on_record_failure: bool,
) -> Response<Full<Bytes>> {
    let container = &tb.container_name;
    let image = &tb.image;
    if pull {
        if let Err(resp) = pull_toolbox_image(container, image).await { return *resp; }
    }
    if let Err(resp) = toolbox_create(container, image).await { return *resp; }
    let Some(snapshot) = inspect_container(container).await else {
        return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
            error: format!("toolbox create succeeded but container '{container}' could not be inspected"),
        });
    };
    let entry = ownership_entry(snapshot.id.clone(), image.clone(), tb.id.clone());
    if let Err(resp) = record_toolbox_ownership(sidecar_path, container, entry) {
        if cleanup_on_record_failure {
            let _ = podman_rm_force(&snapshot.id, container).await;
        } else {
            error!(container_id = %snapshot.id, %container, "Toolbox ownership persist failed after recreate");
        }
        return *resp;
    }
    json_response(StatusCode::OK, &serde_json::json!({
        "status": "ok",
        "message": format!("Toolbox container '{container}' created."),
    }))
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
    if inspect_container(&tb.container_name).await.is_some() {
        return json_response(StatusCode::CONFLICT, &ErrorResponse {
            error: format!("Container '{}' already exists — use update or adopt instead.", tb.container_name),
        });
    }
    recreate_toolbox_container(&state.managed_toolboxes_path, tb, true, true).await
}

/// `POST /api/toolbox-containers/{name}/update` — pull latest image + recreate.
pub async fn update_toolbox_container(state: &AppState, container_name: &str) -> Response<Full<Bytes>> {
    let catalog = match load_typed_toolbox_catalog() {
        Ok(c) => c,
        Err(e) => return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse { error: e }),
    };
    let Some(tb) = catalog.toolboxes.iter().find(|t| t.container_name == container_name && t.supported_backend().is_some()) else {
        return json_response(StatusCode::NOT_FOUND, &ErrorResponse {
            error: format!("No catalog toolbox with container_name: {container_name}"),
        });
    };
    let _guard = lock_toolbox_container(state, container_name).await;
    let Some(snapshot) = inspect_container(container_name).await else {
        return json_response(StatusCode::NOT_FOUND, &ErrorResponse { error: format!("No such container: {container_name}") });
    };
    let entry = crate::managed_toolboxes::get(&state.managed_toolboxes_path, container_name);
    if !authorize(Some(&snapshot), entry.as_ref()) {
        return json_response(StatusCode::FORBIDDEN, &ErrorResponse { error: format!("Container '{container_name}' is not brainrouter-managed.") });
    }
    if let Err(resp) = pull_toolbox_image(container_name, &tb.image).await { return *resp; }
    if let Err(resp) = podman_rm_force(&snapshot.id, container_name).await { return *resp; }
    recreate_toolbox_container(&state.managed_toolboxes_path, tb, false, false).await
}

/// `POST /api/toolbox-containers/{name}/adopt` — record ownership of the exact live container ID.
pub async fn adopt_toolbox_container(state: &AppState, container_name: &str) -> Response<Full<Bytes>> {
    let catalog = match load_typed_toolbox_catalog() {
        Ok(c) => c,
        Err(e) => return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse { error: e }),
    };
    let Some(tb) = catalog.toolboxes.iter().find(|t| t.container_name == container_name && t.supported_backend().is_some()) else {
        return json_response(StatusCode::NOT_FOUND, &ErrorResponse {
            error: format!("No catalog toolbox with container_name: {container_name}"),
        });
    };
    let _guard = lock_toolbox_container(state, container_name).await;
    let Some(snapshot) = inspect_container(container_name).await else {
        return json_response(StatusCode::NOT_FOUND, &ErrorResponse { error: format!("No such container: {container_name}") });
    };
    let existing_entry = crate::managed_toolboxes::get(&state.managed_toolboxes_path, container_name);
    if authorize(Some(&snapshot), existing_entry.as_ref()) {
        return json_response(StatusCode::CONFLICT, &ErrorResponse { error: format!("Container '{container_name}' is already brainrouter-managed.") });
    }
    let entry = ownership_entry(snapshot.id, snapshot.image, tb.id.clone());
    if let Err(resp) = record_toolbox_ownership(&state.managed_toolboxes_path, container_name, entry) { return *resp; }
    json_response(StatusCode::OK, &serde_json::json!({ "status": "ok", "message": format!("Toolbox container '{container_name}' adopted.") }))
}

/// `POST /api/toolbox-containers/{name}/delete` — idempotent force-remove.
pub async fn delete_toolbox_container(state: &AppState, container_name: &str) -> Response<Full<Bytes>> {
    let _guard = lock_toolbox_container(state, container_name).await;
    let Some(snapshot) = inspect_container(container_name).await else {
        if let Err(resp) = remove_toolbox_ownership(&state.managed_toolboxes_path, container_name) { return *resp; }
        return json_response(StatusCode::OK, &serde_json::json!({ "status": "ok", "message": "already removed" }));
    };
    let entry = crate::managed_toolboxes::get(&state.managed_toolboxes_path, container_name);
    if !authorize(Some(&snapshot), entry.as_ref()) {
        return json_response(StatusCode::FORBIDDEN, &ErrorResponse { error: format!("Container '{container_name}' is not brainrouter-managed.") });
    }
    if let Err(resp) = podman_rm_force(&snapshot.id, container_name).await { return *resp; }
    if let Err(resp) = remove_toolbox_ownership(&state.managed_toolboxes_path, container_name) { return *resp; }
    json_response(StatusCode::OK, &serde_json::json!({ "status": "ok", "message": format!("Toolbox container '{container_name}' removed.") }))
}

// ── PR6: model-download orchestration (§10) ──────────────────────────────

/// `GET /api/model-downloads/status` — read-only local-presence sweep
/// across every download-capable backend's catalog entries (ds4/gufo;
/// llama_cpp is excluded — see `model_downloads.rs` docs).
/// Also returns an advisory `hf` PATH-preflight (see [`status_response_value`])
/// so the dashboard can warn when the Hugging Face CLI downloads depend on is
/// missing from the service PATH.
pub async fn model_downloads_status_response() -> Response<Full<Bytes>> {
    match crate::model_downloads::local_presence_snapshot() {
        Ok(presence) => json_response(
            StatusCode::OK,
            &status_response_value(presence, crate::model_downloads::hf_preflight()),
        ),
        Err(e) => {
            error!(error = %e, "Failed to compute model-download presence snapshot");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse { error: e })
        }
    }
}

/// Builds the `/api/model-downloads/status` body: the local-presence `models`
/// array (unchanged semantics) plus the additive advisory `hf` preflight
/// object. Pure so the top-level response shape is deterministically testable
/// without a live handler or filesystem.
fn status_response_value(
    presence: Vec<crate::model_downloads::ModelPresence>,
    hf: crate::model_downloads::HfPreflight,
) -> serde_json::Value {
    serde_json::json!({ "models": presence, "hf": hf })
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
/// (the optional feed field). Body: `{"backend": "...", "model_id": "..."}`.
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
    served_model: String,
    endpoint: String,
) {
    let identity = crate::serving_identity::ServingIdentity {
        toolbox_backend: backend.as_str(),
        compute_api: crate::server_mode::compute_api_for_runtime_profile(profile).to_string(),
        runtime_profile_id: toolbox_id,
        served_model,
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
                    request.model_id.clone(),
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

// ── gufo Server Mode (design doc DI-7) ────────────────────────────────────

/// `GET /api/server-mode/gufo/status` — live `podman inspect`, no persisted state.
pub async fn server_mode_gufo_status_response() -> Response<Full<Bytes>> {
    let status = crate::server_mode::gufo_server_status().await;
    json_response(StatusCode::OK, &status)
}

/// `POST /api/server-mode/gufo/start` — body:
/// `{"toolbox_id","model_id","ctx":<n>,"sessions":<n?>,"host","port":<n>,"custom_args"?}`.
pub async fn server_mode_gufo_start_response(state: &AppState, body: &Bytes) -> Response<Full<Bytes>> {
    let request: crate::server_mode::StartGufoServerRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            return json_response(StatusCode::BAD_REQUEST, &ErrorResponse {
                error: format!("invalid request body: {e}"),
            });
        }
    };
    match crate::server_mode::start_gufo_server(&request).await {
        Ok(()) => {
            if let Ok((_, profile)) = crate::server_mode::resolve_gufo_toolbox(&request.toolbox_id) {
                register_serving_identity(
                    state,
                    crate::toolbox_catalog::SupportedServingBackend::Gufo,
                    crate::server_mode::GUFO_SERVER_CONTAINER_NAME,
                    request.toolbox_id.clone(),
                    &profile,
                    request.model_id.clone(),
                    serving_identity_endpoint(&request.host, request.port),
                )
                .await;
            }
            let status = crate::server_mode::gufo_server_status().await;
            json_response(StatusCode::OK, &status)
        }
        Err(e) => json_response(StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), &ErrorResponse {
            error: e.message().to_string(),
        }),
    }
}

/// `POST /api/server-mode/gufo/stop` — graceful stop then `rm -f`, and
/// deregister the serving identity.
pub async fn server_mode_gufo_stop_response(state: &AppState) -> Response<Full<Bytes>> {
    match crate::server_mode::stop_gufo_server().await {
        Ok(()) => {
            state.serving_identities.deregister(crate::server_mode::GUFO_SERVER_CONTAINER_NAME).await;
            json_response(StatusCode::OK, &serde_json::json!({
                "status": "ok",
                "message": "gufo server stopped.",
            }))
        }
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

async fn upgrade_toolbox(state: &AppState, container: &str, image: &str) -> Response<Full<Bytes>> {
    info!(%container, %image, "Upgrading toolbox container...");
    let _guard = lock_toolbox_container(state, container).await;
    let Some(snapshot) = inspect_container(container).await else {
        return json_response(StatusCode::NOT_FOUND, &ErrorResponse { error: format!("No such toolbox container: {}", container) });
    };
    let entry = crate::managed_toolboxes::get(&state.managed_toolboxes_path, container);
    if !authorize(Some(&snapshot), entry.as_ref()) {
        return json_response(StatusCode::FORBIDDEN, &ErrorResponse { error: format!("Container '{container}' is not brainrouter-managed.") });
    }
    if let Err(resp) = pull_toolbox_image(container, image).await { return *resp; }
    if let Err(resp) = podman_rm_force(&snapshot.id, container).await { return *resp; }
    if let Err(resp) = toolbox_create(container, image).await { return *resp; }
    let Some(new_snapshot) = inspect_container(container).await else {
        return json_response(StatusCode::INTERNAL_SERVER_ERROR, &ErrorResponse {
            error: format!("toolbox create succeeded but container '{container}' could not be inspected"),
        });
    };
    let catalog_id = entry.as_ref().map(|entry| entry.catalog_id.clone()).unwrap_or_else(|| "legacy-upgrade".to_string());
    let ownership = ownership_entry(new_snapshot.id.clone(), image.to_string(), catalog_id);
    if let Err(resp) = record_toolbox_ownership(&state.managed_toolboxes_path, container, ownership) {
        error!(container_id = %new_snapshot.id, %container, "Toolbox ownership persist failed after legacy upgrade");
        return *resp;
    }
    json_response(StatusCode::OK, &serde_json::json!({
        "status": "ok",
        "message": "Toolbox container recreated with latest image."
    }))
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

/// The `GET /api/roles` payload (contract: assignments + policy; the review
/// card's extras ride along for the dashboard panel).
fn roles_json(state: &AppState) -> serde_json::Value {
    serde_json::json!({
        "assignments": state.roles.assignments_snapshot(),
        "policy": state.roles.policy_snapshot(),
        "review_extras": state.roles.review_extras(),
    })
}

fn policy_json(state: &AppState) -> serde_json::Value {
    serde_json::json!({ "policy": state.roles.policy_snapshot() })
}

/// `GET /api/credentials` — masked list (the store's own masking; plaintext
/// is NEVER serializable through `SecretString`).
fn credentials_json(state: &AppState) -> serde_json::Value {
    serde_json::json!({ "credentials": state.credentials.list_masked().unwrap_or_default() })
}

/// The credentials UI test verdict: resolve through the store (store value /
/// env var set / file readable) and answer `{ "ok": bool }` — the value
/// itself never leaves the process.
fn credential_test_json(state: &AppState, id: &str) -> serde_json::Value {
    let ok = match state.credentials.resolve(id) {
        Ok(secret) => !secret.expose().is_empty(),
        Err(_) => false,
    };
    serde_json::json!({ "ok": ok, "id": id })
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

/// Shared builder for the proxy model list (`/v1/models`) and the plugin RPC
/// `models.list` — one payload, two surfaces (design: models.list returns the
/// daemon /v1/models payload verbatim).
pub(crate) async fn build_model_list(state: &AppState) -> ModelListResponse {
    let mut data = vec![
        ModelObject { id: "auto".to_string(), object: "model", created: 0, owned_by: "brainrouter".to_string() },
        ModelObject { id: "local".to_string(), object: "model", created: 0, owned_by: "brainrouter".to_string() },
        ModelObject { id: "cloud".to_string(), object: "model", created: 0, owned_by: "brainrouter".to_string() },
    ];
    let mut seen: HashSet<String> = data.iter().map(|m| m.id.clone()).collect();
    let ls_url = format!("{}/v1/models", state.llama_swap_url);
    if let Ok(resp) = VERSION_CLIENT.get(&ls_url)
        .timeout(std::time::Duration::from_secs(2))
        .send().await
    {
        if let Ok(body) = resp.json::<serde_json::Value>().await {
            if let Some(arr) = body.get("data").and_then(|d| d.as_array()) {
                let skip = ["auto", "local", "cloud"];
                for m in arr {
                    if let Some(id) = m.get("id").and_then(|v| v.as_str()) {
                        if !skip.contains(&id) && seen.insert(id.to_string()) {
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
    for id in state.serving_identities.openai_compatible_served_models().await {
        if seen.insert(id.clone()) {
            data.push(ModelObject {
                id,
                object: "model",
                created: 0,
                owned_by: "server-mode".to_string(),
            });
        }
    }
    ModelListResponse { object: "list", data }
}

/// The `/v1/models` payload as JSON (plugin RPC `models.list`).
pub(crate) async fn model_list_json(state: &AppState) -> serde_json::Value {
    serde_json::to_value(build_model_list(state).await)
        .unwrap_or_else(|_| serde_json::json!({ "object": "list", "data": [] }))
}

/// Read a JSON body (empty object on any failure).
async fn read_json_body(req: hyper::Request<hyper::body::Incoming>) -> serde_json::Value {
    let bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
    serde_json::from_slice(&bytes).unwrap_or_else(|_| serde_json::json!({}))
}

/// Uniform reply for the plugin management endpoints.
fn plugins_reply(result: anyhow::Result<serde_json::Value>) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    match result {
        Ok(v) => into_unsync(json_response(StatusCode::OK, &v)),
        Err(e) => into_unsync(json_response(
            StatusCode::BAD_REQUEST,
            &ErrorResponse { error: e.to_string() },
        )),
    }
}

/// Query-string parameter lookup (no percent-decoding needed for the plugin
/// endpoints' numeric/id params).
fn query_param(query: &str, key: &str) -> String {
    query
        .split('&')
        .find_map(|pair| {
            let mut kv = pair.splitn(2, '=');
            match (kv.next(), kv.next()) {
                (Some(k), Some(v)) if k == key => Some(v.to_string()),
                _ => None,
            }
        })
        .unwrap_or_default()
}

/// Run the HTTP server with dual listeners (TCP + Unix domain socket)
pub struct BoundListeners {
    pub tcp: TcpListener,
    pub uds: UnixListener,
    pub uds_path: PathBuf,
    pub uds_identity: (u64, u64),
    pub uds_lock: File,
}

/// Bind the TCP + UDS listeners. The daemon calls this BEFORE the boot work
/// (migration, plugin registry, model resolution) so a CLI invoked the
/// instant `/health` answers always finds the UDS socket file — the bind
/// used to sit behind all of that, and under parallel load the window was
/// long enough for "Daemon socket not found".
pub async fn bind_listeners(tcp_addr: SocketAddr, uds_path: PathBuf) -> Result<BoundListeners> {
    // Bind the UDS socket FIRST: between TCP bind and UDS bind a CLI (the
    // gate's black-box tests call verbs the instant /health answers) could
    // connect, be accepted, and have its socket file unlinked by the
    // stale-socket cleanup below — the request would then hit the accept
    // queue of a dead inode and hang until the daemon exits.
    let uds_lock = acquire_uds_lock(&uds_path)?;
    prepare_uds_path(&uds_path).await?;
    let uds_listener = UnixListener::bind(&uds_path)?;
    let uds_identity = socket_identity(&uds_path)?;
    info!("Unix socket listener bound to {:?}", uds_path);

    let tcp_listener = TcpListener::bind(tcp_addr).await?;
    info!("TCP listener bound to {}", tcp_addr);

    Ok(BoundListeners {
        tcp: tcp_listener,
        uds: uds_listener,
        uds_path,
        uds_identity,
        uds_lock,
    })
}

pub async fn run(
    tcp_addr: SocketAddr,
    uds_path: PathBuf,
    state: Arc<AppState>,
) -> Result<()> {
    let bound = bind_listeners(tcp_addr, uds_path).await?;
    run_with_listeners(bound.tcp, bound.uds_path, bound.uds, bound.uds_identity, bound.uds_lock, state).await
}

/// The accept loops + shutdown contract, split out so the daemon can bind
/// the listeners BEFORE the boot work (migration, plugin registry, model
/// resolution) — a CLI invoked the instant `/health` answers must always
/// find the UDS socket file (the bind used to sit behind all of it).
pub async fn run_with_listeners(
    tcp_listener: TcpListener,
    uds_path: PathBuf,
    uds_listener: UnixListener,
    uds_identity: (u64, u64),
    _uds_lock: File,
    state: Arc<AppState>,
) -> Result<()> {

    let tcp_state = state.clone();
    let uds_state = state.clone();

    let tcp_task = tokio::spawn(async move {
        loop {
            match tcp_listener.accept().await {
                Ok((stream, addr)) => {
                    info!("New TCP connection from {}", addr);
                    let local_addr = stream.local_addr().ok();
                    let io = TokioIo::new(stream);
                    let state = tcp_state.clone();
                    // Resolve the cwd of the connecting OMP process once per
                    // connection; all requests on this keep-alive connection
                    // share the same process and thus the same cwd.
                    // Synchronous on this task: the default blocking pool
                    // saturates under full-suite load, and queueing there
                    // burned the project-scope grace window before the
                    // first probe ran (the scope flakes). The /proc scans
                    // are a few reads of hot pseudo-files.
                    // Bruce r2 nit 2: a local_addr() error skips the scan
                    // entirely (empty cwd) — a port-0 fallback would match
                    // LISTEN rows, the exact collision class removed.
                    let conn_cwd = local_addr
                        .as_ref()
                        .and_then(|la| peer_cwd(&addr, la))
                        .unwrap_or_default();
                    tokio::spawn(async move {
                        if let Err(e) = http1::Builder::new()
                            .serve_connection(io, service_fn(move |req| handle_request(req, state.clone(), conn_cwd.clone(), addr, None)))
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
                    // The peer pid (UDS credentials). The cwd is read PER
                    // REQUEST, not cached at accept: a CLI process that
                    // chdir'd after connecting (the gate's black-box scope
                    // tests spawn the daemon, then change directory) must
                    // resolve its CURRENT project scope. cwd_from_pid is a
                    // cheap readlink on /proc of a live local process.
                    let conn_pid = stream
                        .peer_cred()
                        .ok()
                        .and_then(|cred| cred.pid())
                        .unwrap_or(0);
                    // Grace anchor = ACCEPT time: the window for this
                    // connection's requests (see `ScopeClock`).
                    let conn_clock = Arc::new(ScopeClock(std::sync::Mutex::new(Some(
                        std::time::Instant::now(),
                    ))));
                    let io = TokioIo::new(stream);
                    let state = uds_state.clone();
                    let dummy_addr: SocketAddr = "0.0.0.0:0".parse().unwrap();
                    let pid_for_request = conn_pid;
                    tokio::spawn(async move {
                        let io = http1::Builder::new().serve_connection(io, service_fn(move |req| {
                            let clock = conn_clock.clone();
                            let cwd = {
                                // Read the peer cwd at REQUEST time (a CLI
                                // process that chdir'd after connecting —
                                // the gate's black-box scope tests — must
                                // resolve its CURRENT project scope; cheap
                                // readlink on a live local process).
                                if pid_for_request > 0 {
                                    crate::peer_cwd::cwd_from_pid(pid_for_request).unwrap_or_default()
                                } else {
                                    String::new()
                                }
                            };
                            let state2 = state.clone();
                            async move { handle_request(req, state2, cwd, dummy_addr, Some(clock)).await }
                        }));
                        if let Err(e) = io.await {
                            error!("Error serving Unix socket connection: {}", e);
                        }
                    });
                }
                Err(e) => error!("Failed to accept Unix socket connection: {}", e),
            }
        }
    });

    let mut tcp_task = tcp_task;
    let mut uds_task = uds_task;
    tokio::select! {
        _ = &mut tcp_task => info!("TCP listener task ended"),
        _ = &mut uds_task => info!("Unix socket listener task ended"),
        _ = shutdown_signal() => info!("Shutdown signal received; stopping plugins"),
    }
    tcp_task.abort();
    uds_task.abort();

    // v2 L2: SIGTERM with grace, then kill (by PID) — every supervised
    // plugin child is stopped before the daemon exits.
    state.plugins.shutdown().await;

    remove_owned_uds(&uds_path_for_cleanup, uds_identity);

    Ok(())
}

/// SIGTERM (the systemd stop path) / Ctrl-C.
async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = signal(SignalKind::terminate()).expect("installing SIGTERM handler");
    tokio::select! {
        _ = term.recv() => {}
        _ = tokio::signal::ctrl_c() => {}
    }
}

pub(crate) fn acquire_uds_lock(uds_path: &std::path::Path) -> Result<File> {
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

pub(crate) fn socket_identity(path: &std::path::Path) -> Result<(u64, u64)> {
    let metadata = std::fs::symlink_metadata(path)?;
    Ok((metadata.dev(), metadata.ino()))
}

pub(crate) fn remove_owned_uds(path: &std::path::Path, expected: (u64, u64)) {
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

pub(crate) async fn prepare_uds_path(uds_path: &std::path::Path) -> Result<()> {
    let metadata = match std::fs::symlink_metadata(uds_path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        // ENOTDIR: an ancestor of the path is not a directory — the
        // caller's state dir (e.g. a test tmpdir) was deleted out from
        // under the boot. Clean startup failure, not a bind success.
        Err(error) if error.kind() == std::io::ErrorKind::NotADirectory => {
            anyhow::bail!(
                "the parent directory of {} no longer exists; not binding (the daemon's                      state directory was deleted during startup)",
                uds_path.display()
            )
        }
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

    fn ownership(id: &str) -> OwnershipEntry {
        OwnershipEntry {
            container_id: id.to_string(),
            image: "docker.io/example/toolbox:latest".to_string(),
            catalog_id: "catalog-toolbox".to_string(),
            catalog_revision: "rev-1".to_string(),
            created_at: "2026-09-24T00:00:00Z".to_string(),
        }
    }

    fn snapshot(id: &str, managed_label: &str) -> ContainerSnapshot {
        ContainerSnapshot {
            id: id.to_string(),
            managed_label: managed_label.to_string(),
            image: "docker.io/example/toolbox:latest".to_string(),
        }
    }

    #[test]
    fn toolbox_authorize_matrix_uses_legacy_label_or_exact_sidecar_id() {
        let labeled = snapshot("live-id", "true");
        assert!(authorize(Some(&labeled), None), "legacy label authorizes");

        let unlabeled = snapshot("live-id", "");
        let matching = ownership("live-id");
        assert!(authorize(Some(&unlabeled), Some(&matching)), "matching sidecar id authorizes");

        let mismatched = ownership("old-id");
        assert!(!authorize(Some(&unlabeled), Some(&mismatched)), "same-name replacement is unmanaged");
        assert!(!authorize(Some(&unlabeled), None), "neither source is unmanaged");
        assert!(!authorize(None, Some(&matching)), "absent live container is not authorized");
    }

    #[test]
    fn status_response_value_has_models_and_additive_hf() {
        // `models` passes through with unchanged semantics; `hf` is additive
        // with exactly the four keys and correct nullability for each state.
        let missing = status_response_value(Vec::new(), crate::model_downloads::build_preflight(false));
        assert_eq!(missing["models"], serde_json::json!([]));
        let obj = missing["hf"].as_object().expect("hf is an object");
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["binary", "found_on_path", "install_command", "message"]);
        assert_eq!(missing["hf"]["found_on_path"], serde_json::json!(false));
        assert!(missing["hf"]["message"].is_string());
        assert!(missing["hf"]["install_command"].is_string());

        let found = status_response_value(Vec::new(), crate::model_downloads::build_preflight(true));
        assert_eq!(found["hf"]["found_on_path"], serde_json::json!(true));
        assert!(found["hf"]["message"].is_null());
        assert!(found["hf"]["install_command"].is_null());
    }

    // ── Per-project model pins: gate + POST decoder (Phase 2) ────────────────

    #[test]
    fn llama_swap_gate_covers_register_subpaths() {
        assert!(request_is_destructive("POST", "/api/llama-swap/register-model", false));
        assert!(request_is_destructive("POST", "/api/llama-swap/future", false));
        assert!(request_is_destructive("POST", "/api/llama-swap-config", false));
        assert!(!request_is_destructive("GET", "/api/llama-swap/register-model", false));
    }

    fn server_test_dir(name: &str) -> PathBuf {
        let p = std::env::current_dir().unwrap().join("target").join("server_file_transaction_tests").join(name);
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn llama_swap_atomic_write_replaces_file_and_removes_temp() {
        let dir = server_test_dir("atomic-write");
        let path = dir.join("config.yaml");
        let tmp = unique_sibling_path(&path, "tmp");
        std::fs::write(&path, b"old").unwrap();
        atomic_write_path(&path, &tmp, b"new").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        assert!(!tmp.exists());
    }

    #[test]
    fn llama_swap_restore_is_byte_identical_to_backup() {
        let dir = server_test_dir("restore");
        let path = dir.join("config.yaml");
        let bak = rolling_backup_path(&path);
        let original = b"models:
  old:
    cmd: old
";
        std::fs::write(&path, b"changed").unwrap();
        std::fs::write(&bak, original).unwrap();
        restore_llama_swap_config(&path, &bak).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), original);
    }

    #[test]
    fn llama_swap_external_change_detection_flags_hash_mismatch() {
        let dir = server_test_dir("external-change");
        let path = dir.join("config.yaml");
        std::fs::write(&path, b"original").unwrap();
        let captured = sha256_hex(&std::fs::read(&path).unwrap());
        assert!(!config_hash_changed(&path, &captured).unwrap());
        std::fs::write(&path, b"changed").unwrap();
        assert!(config_hash_changed(&path, &captured).unwrap());
    }

    #[test]
    fn project_pin_gate_covers_mutations_and_the_resolver() {
        // v2 L3 retarget (spec "Removals: routing_profile store/CLI/widgets;
        // project_pins store (migrated)"): the L3 mutation surface replaces the
        // pins/profile pairs and is gated the same way.
        assert!(request_is_destructive("POST", "/api/roles/set", false));
        assert!(request_is_destructive("POST", "/api/roles/policy", false));
        assert!(request_is_destructive("POST", "/api/policy/set", false));
        assert!(request_is_destructive("DELETE", "/api/credentials/x", false));
        assert!(request_is_destructive("POST", "/api/credentials/add", false));
        assert!(request_is_destructive("POST", "/api/decision/test", false));
        // Reads stay ungated.
        assert!(!request_is_destructive("GET", "/api/roles", false));
        assert!(!request_is_destructive("GET", "/api/credentials", false));
        assert!(!request_is_destructive("GET", "/api/policy", false));
        assert!(!request_is_destructive("GET", "/api/decision", false));
        // Removed v1 endpoints answer from the 404 catch-all (not gated code).
        assert!(!request_is_destructive("POST", "/api/routing-profile", false));
    }

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
        // Under the full parallel suite another test may fork a child while
        // this listener exists; until that child execs (CLOEXEC closes the
        // inherited fd) a connect can briefly succeed. Wait for the kernel to
        // report the socket dead — the condition the test is about.
        for _ in 0..400 {
            if UnixStream::connect(&socket_path).await.is_err() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }

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
    #[test]
    fn engines_mutations_are_gated_and_reads_are_not() {
        assert!(request_is_destructive("POST", "/api/engines/ensure", false));
        assert!(request_is_destructive("POST", "/api/engines/import/apply", false));
        assert!(request_is_destructive("DELETE", "/api/engines/queue/3", false));
        assert!(!request_is_destructive("GET", "/api/engines", false));
        assert!(!request_is_destructive("GET", "/api/engines/import/preview", false));
        assert!(!request_is_destructive("GET", "/api/platform", false));
    }

    #[test]
    fn engine_route_aliases_normalize_to_one_handler() {
        assert_eq!(
            normalize_engines_path("GET", "/api/engines/list"),
            "/api/engines"
        );
        assert_eq!(
            normalize_engines_path("GET", "/api/engines/import"),
            "/api/engines/import/preview"
        );
        assert_eq!(
            normalize_engines_path("POST", "/api/engines/import"),
            "/api/engines/import/apply"
        );
        assert_eq!(
            normalize_engines_path("GET", "/api/engines/logs/"),
            "/api/engines/logs"
        );
        assert_eq!(normalize_engines_path("GET", "/health"), "/health");
    }

}