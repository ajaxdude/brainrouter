//! v2 L6: LAN federation — the SECOND listener (spec "LAN federation").
//!
//! The primary `--tcp_addr` surface is never touched: this module registers a
//! separate route table containing ONLY `/v1/chat/completions`, `/v1/messages`,
//! `/v1/models` (expose-filtered), and `/api/peer-info`. Admin, dashboard,
//! roles, credentials, and provider endpoints are not registered here at all
//! (listener-level 404, not middleware-level).
//!
//! Auth is `Authorization: Bearer <token>` compared constant-time against the
//! resolved `lan.token_credential` on every request (rotation is immediate).
//! The per-token bucket (`lan.rate`) answers 429 on excess. Expose is
//! double-enforced: the RAW requested model AND the RESOLVED model must both
//! be listed, so an alias/qualified selector can never reach unexposed cloud
//! spend.

use crate::config::LanConfig;
use crate::server::AppState;
use anyhow::{Context, Result};
use bytes::Bytes;
use futures_util::StreamExt;
use http_body_util::{BodyExt, Full, StreamBody, combinators::UnsyncBoxBody};
use hyper::body::{Frame, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::net::TcpListener;

/// The LAN bearer id when the operator configured none (never matches).
pub struct LanState {
    expose: HashSet<String>,
    token_credential: String,
    burst: f64,
    per_sec: f64,
    bucket: Mutex<Bucket>,
}

struct Bucket {
    tokens: f64,
    last: Instant,
}

impl Bucket {
    fn new(burst: u64) -> Self {
        Self {
            tokens: burst as f64,
            last: Instant::now(),
        }
    }

    fn allow(&mut self, burst: f64, per_sec: f64) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last).as_secs_f64();
        self.last = now;
        self.tokens = (self.tokens + elapsed * per_sec).min(burst);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

impl LanState {
    fn new(config: &LanConfig) -> Self {
        Self {
            expose: config.expose.iter().cloned().collect(),
            token_credential: config.token_credential.clone(),
            burst: config.rate.burst as f64,
            per_sec: config.rate.per_sec as f64,
            bucket: Mutex::new(Bucket::new(config.rate.burst)),
        }
    }

    fn allows_rate(&self) -> bool {
        self.bucket
            .lock()
            .unwrap()
            .allow(self.burst, self.per_sec)
    }
}

/// A bound LAN listener.
pub struct LanListener {
    pub listener: TcpListener,
    pub addr: SocketAddr,
    state: Arc<LanState>,
}

/// Bind the LAN listener (called only when `lan.enabled`).
pub async fn bind(config: &LanConfig) -> Result<LanListener> {
    let addr: SocketAddr = config
        .bind
        .parse()
        .with_context(|| format!("lan.bind `{}` is not a socket address", config.bind))?;
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding the LAN listener on {addr}"))?;
    let addr = listener.local_addr()?;
    Ok(LanListener {
        listener,
        addr,
        state: Arc::new(LanState::new(config)),
    })
}

/// Serve the LAN accept loop forever.
pub async fn serve(listener: LanListener, app: Arc<AppState>) -> Result<()> {
    let state = Arc::clone(&listener.state);
    let listener = listener.listener;
    loop {
        let Ok((stream, peer)) = listener.accept().await else {
            continue;
        };
        let io = TokioIo::new(stream);
        let app = Arc::clone(&app);
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            let service = service_fn(move |req| {
                let app = Arc::clone(&app);
                let state = Arc::clone(&state);
                async move { handle(req, app, state, peer).await }
            });
            if let Err(error) = http1::Builder::new().serve_connection(io, service).await {
                tracing::debug!(%error, "LAN connection ended");
            }
        });
    }
}

fn json_response(status: StatusCode, value: &Value) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(value.to_string())))
        .expect("build LAN JSON response")
}

fn into_unsync(
    response: Response<Full<Bytes>>,
) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    response.map(|body| body.map_err(|never| match never {}).boxed_unsync())
}

/// Constant-time byte comparison (bearer tokens). Returns false early only on
/// a LENGTH difference, which leaks nothing about the token's content.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn bearer(headers: &hyper::HeaderMap) -> Option<&str> {
    headers
        .get(hyper::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim)
}

fn authorized(app: &AppState, lane: &LanState, headers: &hyper::HeaderMap) -> bool {
    let Some(presented) = bearer(headers) else {
        return false;
    };
    let Ok(secret) = app.credentials.resolve(&lane.token_credential) else {
        return false;
    };
    constant_time_eq(presented.as_bytes(), secret.expose().as_bytes())
}

async fn handle(
    req: Request<Incoming>,
    app: Arc<AppState>,
    lane: Arc<LanState>,
    _peer: SocketAddr,
) -> Result<Response<UnsyncBoxBody<Bytes, anyhow::Error>>, Infallible> {
    let method = req.method().as_str().to_string();
    let path = req.uri().path().to_string();

    // Listener-level route table: everything else is a 404 and never reaches
    // an admin handler (the primary surface is untouched).
    let known = match (method.as_str(), path.as_str()) {
        ("GET", "/api/peer-info") | ("GET", "/v1/models") => true,
        ("POST", "/v1/chat/completions") | ("POST", "/v1/messages") => true,
        (_, "/api/peer-info")
        | (_, "/v1/models")
        | (_, "/v1/chat/completions")
        | (_, "/v1/messages") => {
            return Ok(into_unsync(json_response(
                StatusCode::METHOD_NOT_ALLOWED,
                &json!({"error": "method not allowed on the LAN listener"}),
            )))
        }
        _ => false,
    };
    if !known {
        return Ok(into_unsync(json_response(
            StatusCode::NOT_FOUND,
            &json!({"error": "not found on the LAN listener"}),
        )));
    }

    if !authorized(&app, &lane, req.headers()) {
        return Ok(into_unsync(json_response(
            StatusCode::UNAUTHORIZED,
            &json!({"error": "missing or invalid bearer token"}),
        )));
    }

    // Rate limit the inference/discovery routes (per-token bucket). The
    // read-only peer-info discovery probe is not rate-limited: it is the
    // readiness/health feed, not an inference path.
    if path != "/api/peer-info" && !lane.allows_rate() {
        return Ok(into_unsync(json_response(
            StatusCode::TOO_MANY_REQUESTS,
            &json!({"error": "LAN rate limit exceeded", "code": -32005}),
        )));
    }

    match (method.as_str(), path.as_str()) {
        ("GET", "/api/peer-info") => Ok(into_unsync(json_response(
            StatusCode::OK,
            &peer_info(&app, &lane),
        ))),
        ("GET", "/v1/models") => {
            let data: Vec<Value> = lane
                .expose
                .iter()
                .map(|key| {
                    json!({"id": key, "object": "model", "created": 0, "owned_by": "brainrouter-lan"})
                })
                .collect();
            Ok(into_unsync(json_response(
                StatusCode::OK,
                &json!({"object": "list", "data": data}),
            )))
        }
        ("POST", "/v1/chat/completions") => {
            let headers = req.headers().clone();
            let bytes = match req.collect().await {
                Ok(body) => body.to_bytes(),
                Err(error) => {
                    return Ok(into_unsync(json_response(
                        StatusCode::BAD_REQUEST,
                        &json!({"error": format!("reading body: {error}")}),
                    )))
                }
            };
            let request: crate::types::ChatCompletionRequest =
                match serde_json::from_slice(&bytes) {
                    Ok(request) => request,
                    Err(error) => {
                        return Ok(into_unsync(json_response(
                            StatusCode::BAD_REQUEST,
                            &json!({"error": format!("invalid JSON: {error}")}),
                        )))
                    }
                };
            if let Err(error) = enforce_expose(&app, &lane, &request.model) {
                return Ok(into_unsync(json_response(
                    StatusCode::BAD_REQUEST,
                    &json!({"error": error}),
                )));
            }
            let session_id = crate::server::extract_session_id(&headers);
            let user_agent = headers
                .get(hyper::header::USER_AGENT)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .trim()
                .to_string();
            // LAN inbound headers are harness headers (L3): the peer applies
            // its OWN role cards to the named role.
            let role = headers
                .get("x-brainrouter-role")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("main")
                .trim()
                .to_string();
            match app
                .router
                .route_tagged_role(request, session_id, String::new(), user_agent, &role)
                .await
            {
                Ok((crate::provider::ProviderResponse::Stream(stream), _info)) => {
                    let stream_body =
                        StreamBody::new(stream.map(|chunk| chunk.map(Frame::data)));
                    let response = Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "text/event-stream")
                        .header("cache-control", "no-cache")
                        .body(stream_body.boxed_unsync())
                        .expect("build LAN SSE response");
                    Ok(response)
                }
                Err(error) => Ok(into_unsync(json_response(
                    StatusCode::BAD_GATEWAY,
                    &json!({"error": format!("routing failed: {error}")}),
                ))),
            }
        }
        ("POST", "/v1/messages") => {
            let headers = req.headers().clone();
            let bytes = match req.collect().await {
                Ok(body) => body.to_bytes(),
                Err(error) => {
                    return Ok(into_unsync(json_response(
                        StatusCode::BAD_REQUEST,
                        &json!({"error": format!("reading body: {error}")}),
                    )))
                }
            };
            let parsed: crate::anthropic::AnthropicMessagesRequest =
                match serde_json::from_slice(&bytes) {
                    Ok(parsed) => parsed,
                    Err(error) => {
                        return Ok(into_unsync(json_response(
                            StatusCode::BAD_REQUEST,
                            &json!({"error": format!("invalid JSON: {error}")}),
                        )))
                    }
                };
            if let Err(error) = enforce_expose(&app, &lane, &parsed.model) {
                return Ok(into_unsync(json_response(
                    StatusCode::BAD_REQUEST,
                    &json!({"error": error}),
                )));
            }
            let session_id = crate::server::extract_session_id(&headers);
            let user_agent = headers
                .get(hyper::header::USER_AGENT)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .trim()
                .to_string();
            let role = headers
                .get("x-brainrouter-role")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("main")
                .trim()
                .to_string();
            crate::server::anthropic_messages_from_bytes(
                app,
                bytes,
                String::new(),
                session_id,
                user_agent,
                "0.0.0.0:0".parse().unwrap(),
                role,
            )
            .await
            .map_err(|error| {
                tracing::warn!(%error, "LAN Anthropic request failed");
                error
            })
            .or_else(|_| {
                Ok(into_unsync(json_response(
                    StatusCode::BAD_GATEWAY,
                    &json!({"error": "routing failed"}),
                )))
            })
        }
        _ => Ok(into_unsync(json_response(
            StatusCode::NOT_FOUND,
            &json!({"error": "not found on the LAN listener"}),
        ))),
    }
}

/// Expose double-enforcement: the RAW requested string AND the RESOLVED model
/// must both be listed.
fn enforce_expose(app: &AppState, lane: &LanState, raw: &str) -> std::result::Result<(), String> {
    if !lane.expose.contains(raw) {
        return Err(format!(
            "model `{raw}` is not in the LAN expose list"
        ));
    }
    let resolved = resolved_model(app, raw);
    if let Some(resolved) = resolved {
        if !lane.expose.contains(&resolved) {
            return Err(format!(
                "model `{raw}` resolves to `{resolved}`, which is not in the LAN expose list"
            ));
        }
    }
    Ok(())
}

fn resolved_model(app: &AppState, raw: &str) -> Option<String> {
    let records = app.providers.snapshot();
    match crate::roles::parse_selector(raw) {
        crate::roles::Selector::Cloud => Some("auto".to_string()),
        crate::roles::Selector::CloudId(rest) => {
            let query = crate::providers::PoolQuery::default();
            crate::providers::resolve_cloud_target(&records, Some(&rest), &query)
                .ok()
                .map(|target| target.model)
        }
        crate::roles::Selector::Bare(key) => {
            Some(crate::providers::resolve_bare_model(&records, &key).map(|(_, m)| m).unwrap_or(key))
        }
        _ => None,
    }
}

fn host_id() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok().filter(|s| !s.is_empty()))
        .unwrap_or_else(|| "brainrouter".to_string())
}

/// `peer-info`: `{host_id, version, models: [{key, ctx, tools, tier, state}],
/// serving: [...]}` — read-only model + health + tier for token holders.
fn peer_info(app: &AppState, lane: &LanState) -> Value {
    let records = app.providers.snapshot();
    let mut models = Vec::new();
    for key in &lane.expose {
        let provider_model = records
            .iter()
            .filter(|p| p.enabled)
            .find_map(|p| p.model_match(key).map(|m| (p.clone(), m.clone())));
        match provider_model {
            Some((provider, model)) => {
                models.push(json!({
                    "key": key,
                    "ctx": model.ctx,
                    "tools": model.tools,
                    "tier": model.tier.as_str(),
                    "state": app.router.health_state(&provider.id),
                }));
            }
            None => {
                models.push(json!({
                    "key": key,
                    "ctx": 0,
                    "tools": true,
                    "tier": "balanced",
                    "state": "ready",
                }));
            }
        }
    }
    let serving: Vec<&String> = lane.expose.iter().collect();
    json!({
        "host_id": host_id(),
        "version": env!("CARGO_PKG_VERSION"),
        "models": models,
        "serving": serving,
    })
}
