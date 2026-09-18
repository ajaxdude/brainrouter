//! Escalation UI — HTTP handlers for /review/* routes.
//!
//! Routes:
//!   GET  /review/                       — redirect to /dashboard
//!   GET  /review/session/:id             — session detail (HTML)
//!   POST /review/session/:id/resolve     — human submits feedback
//!   POST /review/api/request             — legacy: start review (blocks until done; no client uses it)
//!   POST /review/api/request-async       — start review, return session ID immediately (CLI/MCP)
//!   POST /review/api/resolve             — resolve session with feedback
//!   POST /review/api/continue            — additional LLM review rounds
//!   POST /review/api/lgtm               — quick-approve a session
//!   GET  /review/api/sessions            — JSON session list
//!   GET  /review/api/sessions/:id        — JSON session detail
use bytes::Bytes;
use http_body_util::{combinators::UnsyncBoxBody, BodyExt, Full};
use hyper::{Request, Response, StatusCode};
use hyper::body::Incoming;
use serde::Serialize;
use std::convert::Infallible;
use std::sync::Arc;

use crate::{
    review::ReviewService,
    session::Session,
};

// Embed templates at compile time so the binary is self-contained.
const SESSION_HTML: &str = include_str!("templates/session.html");

/// Handle all /review/* requests. Called from the main request dispatcher.
pub async fn handle_review_request(
    req: Request<Incoming>,
    review_service: Arc<ReviewService>,
    cwd: String,
    code_review_enabled: bool,
) -> Result<Response<UnsyncBoxBody<Bytes, anyhow::Error>>, Infallible> {
    let method = req.method().as_str();
    let path = req.uri().path().to_string();

    // FR-A: code-review master switch. When off, every review-START endpoint
    // short-circuits to a terminal `disabled` result WITHOUT routing to any
    // model or spawning a loop. Read/list/resolve/lgtm endpoints are
    // unaffected, so any in-flight or historical session stays inspectable.
    if review_start_is_gated(code_review_enabled, method, path.as_str()) {
        return Ok(review_disabled_response());
    }

    let response = match (method, path.as_str()) {
        // Review dashboard — redirect to unified dashboard
        ("GET", "/review/" | "/review") => {
            Response::builder()
                .status(StatusCode::FOUND)
                .header("location", "/dashboard")
                .body(
                    Full::new(Bytes::new())
                        .map_err(|e: Infallible| match e {})
                        .boxed_unsync(),
                )
                .expect("Failed to build redirect")
        }

        // Session detail page
        ("GET", p) if p.starts_with("/review/session/") && !p.ends_with("/resolve") => {
            html_response(SESSION_HTML)
        }

        // JSON API — MCP: resolve a session with human feedback
        // Must come BEFORE the wildcard /resolve guard to avoid shadowing.
        ("POST", "/review/api/resolve") => {
            handle_api_resolve(req, review_service).await
        }

        // Human resolve endpoint
        ("POST", p) if p.ends_with("/resolve") => {
            let session_id = extract_session_id_from_resolve_path(&path);
            handle_resolve(req, review_service, session_id).await
        }

        // JSON API — MCP: start review (long-running, blocks until loop completes)
        ("POST", "/review/api/request") => {
            handle_request_review(req, review_service, cwd).await
        }

        // JSON API — MCP: start review non-blocking, return session ID immediately.
        // The caller should poll GET /review/api/sessions/:id until terminal status.
        ("POST", "/review/api/request-async") => {
            handle_request_review_async(req, review_service, cwd).await
        }

        // JSON API — continue iterating (additional LLM rounds, no human feedback needed)
        ("POST", "/review/api/continue") => {
            handle_continue_review(req, review_service).await
        }

        // JSON API — quick approve (LGTM, no feedback needed)
        ("POST", "/review/api/lgtm") => {
            handle_lgtm(req, review_service).await
        }

        // JSON API — session list
        ("GET", "/review/api/sessions") => {
            let sessions = review_service.session_manager().list_sessions();
            let data: Vec<SessionSummary> = sessions.iter().map(SessionSummary::from).collect();
            json_ok(&ApiSessionList { sessions: data })
        }

        // JSON API — session detail
        ("GET", p) if p.starts_with("/review/api/sessions/") => {
            let session_id = p.trim_start_matches("/review/api/sessions/");
            match review_service.session_manager().get_session(session_id) {
                Some(s) => json_ok(&SessionDetail::from(&s)),
                None => json_error(StatusCode::NOT_FOUND, "Session not found"),
            }
        }
        _ => json_error(StatusCode::NOT_FOUND, "Not found"),
    };

    Ok(response)
}

/// FR-A: exactly the review-START endpoints gated by the code-review master
/// switch. Read/list/resolve/lgtm endpoints are never gated, so existing
/// sessions remain inspectable and human-resolvable even while the switch is
/// off. Pure and side-effect free for direct unit testing.
fn review_start_is_gated(code_review_enabled: bool, method: &str, path: &str) -> bool {
    !code_review_enabled
        && method == "POST"
        && matches!(
            path,
            "/review/api/request" | "/review/api/request-async" | "/review/api/continue"
        )
}

/// FR-A: uniform terminal response when the code reviewer is switched off.
///
/// Every review-START endpoint returns HTTP 200 with a terminal `disabled`
/// status (and a null `sessionId`, since no session is created) so the MCP
/// thin client and the CLI stop immediately instead of polling a session that
/// never exists. No model is contacted.
fn review_disabled_response() -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    json_ok(&serde_json::json!({
        "sessionId": serde_json::Value::Null,
        "status": "disabled",
        "feedback": "Code review is disabled in brainrouter (enable it in the dashboard or via POST /api/review/enabled).",
    }))
}

/// POST /review/api/request — legacy blocking review trigger.
/// Long-running: blocks until the review loop completes.
/// CLI and the MCP thin client use /review/api/request-async + polling instead.
async fn handle_request_review(
    req: Request<Incoming>,
    review_service: Arc<ReviewService>,
    cwd: String,
) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    // Read body
    let body_bytes = match req.collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("Failed to read body: {}", e)),
    };

    #[derive(serde::Deserialize)]
    struct ReviewRequest {
        #[serde(rename = "taskId")]
        task_id: String,
        summary: String,
        details: Option<String>,
        #[serde(rename = "conversationHistory", default)]
        conversation_history: Vec<String>,
        /// Explicit project directory from the agent; overrides peer-cred-resolved cwd.
        cwd: Option<String>,
    }

    let body: ReviewRequest = match serde_json::from_slice(&body_bytes) {
        Ok(b) => b,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("Invalid JSON: {}", e)),
    };

    // Determine project directory: prefer the explicitly-provided cwd from the
    // agent payload (which knows the actual project dir) over the peer-cred-
    // resolved cwd (which is the brainrouter mcp subprocess's own cwd and is
    // typically wrong or empty when the MCP client is a sub-process launched
    // from a fixed location).
    let candidate_cwd = body.cwd.unwrap_or(cwd);

    // Security: Sanitize and validate cwd
    let mut safe_cwd = candidate_cwd;
    let limit = 4096;
    if safe_cwd.len() > limit {
        let boundary = (0..=limit).rev().find(|&i| safe_cwd.is_char_boundary(i)).unwrap_or(0);
        safe_cwd.truncate(boundary);
    }
    
    // Robust path traversal and security check.
    // 1. Block null bytes.
    // 2. Block non-absolute paths.
    // 3. Block path traversal components (..).
    let is_valid = !safe_cwd.contains('\0')
        && !safe_cwd.is_empty() 
        && safe_cwd.starts_with('/') 
        && !std::path::Path::new(&safe_cwd).components().any(|c| matches!(c, std::path::Component::ParentDir));

    if !is_valid {
        // Reject rather than silently falling back to the daemon's own cwd:
        // a review would read the WRONG project's PRD and git diff.
        return json_error(
            StatusCode::BAD_REQUEST,
            "Invalid cwd: must be an absolute path without '..' components",
        );
    }

    match review_service
        .start_review(body.task_id, body.summary, body.details, body.conversation_history, safe_cwd)
        .await
    {
        Ok(result) => json_ok(&serde_json::json!({
            "status": result.status.as_str(),
            "feedback": result.feedback,
            "sessionId": result.session_id,
            "iterationCount": result.iteration_count,
            "reviewerType": format!("{:?}", result.reviewer_type).to_lowercase()
        })),
        Err(e) => {
            let message = e.to_string();
            let status = if message.contains("is already running") {
                StatusCode::CONFLICT
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            json_error(status, &message)
        }
    }
}

/// POST /review/api/request-async — Non-blocking variant of handle_request_review.
///
/// Creates the session and spawns the review loop in the background, then
/// returns immediately with `{ "sessionId": "…", "status": "pending" }`.
/// The caller must poll GET /review/api/sessions/:id until status is terminal.
async fn handle_request_review_async(
    req: Request<Incoming>,
    review_service: Arc<ReviewService>,
    cwd: String,
) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    let body_bytes = match req.collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("Failed to read body: {}", e)),
    };

    #[derive(serde::Deserialize)]
    struct ReviewRequest {
        #[serde(rename = "taskId")]
        task_id: String,
        summary: String,
        details: Option<String>,
        #[serde(rename = "conversationHistory", default)]
        conversation_history: Vec<String>,
        cwd: Option<String>,
    }

    let body: ReviewRequest = match serde_json::from_slice(&body_bytes) {
        Ok(b) => b,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("Invalid JSON: {}", e)),
    };

    let candidate_cwd = body.cwd.unwrap_or(cwd);
    let mut safe_cwd = candidate_cwd;
    if safe_cwd.len() > 4096 {
        let boundary = (0..=4096).rev().find(|&i| safe_cwd.is_char_boundary(i)).unwrap_or(0);
        safe_cwd.truncate(boundary);
    }
    let is_valid = !safe_cwd.contains('\0')
        && !safe_cwd.is_empty()
        && safe_cwd.starts_with('/')
        && !std::path::Path::new(&safe_cwd).components().any(|c| matches!(c, std::path::Component::ParentDir));
    if !is_valid {
        // Reject rather than silently falling back to the daemon's own cwd:
        // a review would read the WRONG project's PRD and git diff.
        return json_error(
            StatusCode::BAD_REQUEST,
            "Invalid cwd: must be an absolute path without '..' components",
        );
    }

    let session_id = review_service.start_review_async(
        body.task_id,
        body.summary,
        body.details,
        body.conversation_history,
        safe_cwd,
    );

    json_ok(&serde_json::json!({
        "sessionId": session_id,
        "status": "pending"
    }))
}

/// POST /review/api/resolve — MCP thin client calls this to resolve a session.
async fn handle_api_resolve(
    req: Request<Incoming>,
    review_service: Arc<ReviewService>,
) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    let body_bytes = match req.collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("Failed to read body: {}", e)),
    };

    #[derive(serde::Deserialize)]
    struct ResolveApiBody {
        #[serde(rename = "sessionId")]
        session_id: String,
        feedback: String,
    }

    let body: ResolveApiBody = match serde_json::from_slice(&body_bytes) {
        Ok(b) => b,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("Invalid JSON: {}", e)),
    };

    match review_service.resolve_session(&body.session_id, body.feedback) {
        Ok(()) => json_ok(&serde_json::json!({ "success": true })),
        Err(e) => json_error(StatusCode::NOT_FOUND, &e.to_string()),
    }
}

async fn handle_resolve(
    req: Request<Incoming>,
    review_service: Arc<ReviewService>,
    session_id: String,
) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    // Read body
    let body_bytes = match req.collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("Failed to read body: {}", e)),
    };

    #[derive(serde::Deserialize)]
    struct ResolveBody {
        feedback: String,
    }

    let body: ResolveBody = match serde_json::from_slice(&body_bytes) {
        Ok(b) => b,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("Invalid JSON: {}", e)),
    };

    if body.feedback.trim().is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "feedback must not be empty");
    }

    match review_service.resolve_session(&session_id, body.feedback.trim().to_string()) {
        Ok(()) => json_ok(&serde_json::json!({ "success": true, "session_id": session_id })),
        Err(e) => json_error(StatusCode::NOT_FOUND, &e.to_string()),
    }
}

async fn handle_continue_review(
    req: Request<Incoming>,
    review_service: Arc<ReviewService>,
) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    use http_body_util::BodyExt;

    let body_bytes = match req.collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("Failed to read body: {}", e)),
    };

    #[derive(serde::Deserialize)]
    struct ContinueBody {
        #[serde(rename = "sessionId")]
        session_id: String,
        #[serde(default = "default_extra_iterations")]
        iterations: u32,
    }
    fn default_extra_iterations() -> u32 { 4 }

    let body: ContinueBody = match serde_json::from_slice(&body_bytes) {
        Ok(b) => b,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("Invalid JSON: {}", e)),
    };

    match review_service.continue_review(&body.session_id, body.iterations).await {
        Ok(result) => json_ok(&serde_json::json!({
            "status": result.status.as_str(),
            "feedback": result.feedback,
            "sessionId": result.session_id,
            "iterationCount": result.iteration_count,
        })),
        Err(e) => {
            let message = e.to_string();
            let status = if message.contains("is already running") {
                StatusCode::CONFLICT
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            json_error(status, &message)
        }
    }
}

async fn handle_lgtm(
    req: Request<Incoming>,
    review_service: Arc<ReviewService>,
) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    use http_body_util::BodyExt;

    let body_bytes = match req.collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("Failed to read body: {}", e)),
    };

    #[derive(serde::Deserialize)]
    struct LgtmBody {
        #[serde(rename = "sessionId")]
        session_id: String,
    }

    let body: LgtmBody = match serde_json::from_slice(&body_bytes) {
        Ok(b) => b,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("Invalid JSON: {}", e)),
    };

    match review_service.resolve_session(&body.session_id, "lgtm".to_string()) {
        Ok(()) => json_ok(&serde_json::json!({ "success": true, "sessionId": body.session_id })),
        Err(e) => json_error(StatusCode::NOT_FOUND, &e.to_string()),
    }
}

fn extract_session_id_from_resolve_path(path: &str) -> String {
    // Path: /review/session/:id/resolve
    path.trim_start_matches("/review/session/")
        .trim_end_matches("/resolve")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::review_start_is_gated;

    #[test]
    fn master_switch_gates_only_review_start_endpoints_when_disabled() {
        // Disabled: the three START endpoints short-circuit to `disabled`.
        for p in [
            "/review/api/request",
            "/review/api/request-async",
            "/review/api/continue",
        ] {
            assert!(review_start_is_gated(false, "POST", p), "{p} should be gated when disabled");
        }
        // Disabled but non-START endpoints stay live so existing sessions
        // remain inspectable and human-resolvable.
        for p in [
            "/review/api/resolve",
            "/review/api/lgtm",
            "/review/api/sessions",
            "/review/api/sessions/abc",
        ] {
            assert!(!review_start_is_gated(false, "POST", p), "{p} must not be gated");
        }
        // GET is never gated.
        assert!(!review_start_is_gated(false, "GET", "/review/api/request-async"));
        // Enabled (default): nothing is gated.
        for p in [
            "/review/api/request",
            "/review/api/request-async",
            "/review/api/continue",
        ] {
            assert!(!review_start_is_gated(true, "POST", p), "{p} must not be gated when enabled");
        }
    }

    #[test]
    fn dashboard_has_code_review_toggle() {
        let html = include_str!("templates/main_dashboard.html");
        assert!(html.contains("toggle-codereview"), "missing code-review toggle button");
        assert!(html.contains("toggleCodeReview()"), "missing toggle handler");
        assert!(html.contains("/api/review/enabled"), "dashboard must call the toggle API");
    }

    #[test]
    fn dashboard_review_actions_use_api_field_names_and_surface_errors() {
        let html = include_str!("templates/main_dashboard.html");
        assert!(html.contains("body: JSON.stringify({ sessionId })"));
        assert!(!html.contains("body: JSON.stringify({ session_id: sessionId })"));
        assert!(html.contains("if (!response.ok) throw new Error"));
    }
}

// ─── Response helpers ────────────────────────────────────────────────────────

fn html_response(html: &'static str) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/html; charset=utf-8")
        .body(
            Full::new(Bytes::from_static(html.as_bytes()))
                .map_err(|e: std::convert::Infallible| match e {})
                .boxed_unsync(),
        )
        .expect("Failed to build HTML response")
}

fn json_ok<T: Serialize>(body: &T) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    let json = serde_json::to_vec(body).unwrap_or_default();
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(
            Full::new(Bytes::from(json))
                .map_err(|e: std::convert::Infallible| match e {})
                .boxed_unsync(),
        )
        .expect("Failed to build JSON response")
}

fn json_error(status: StatusCode, message: &str) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    let body = serde_json::json!({ "error": message });
    let json = serde_json::to_vec(&body).unwrap_or_default();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(
            Full::new(Bytes::from(json))
                .map_err(|e: std::convert::Infallible| match e {})
                .boxed_unsync(),
        )
        .expect("Failed to build error response")
}

// ─── API types ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ApiSessionList {
    sessions: Vec<SessionSummary>,
}

#[derive(Serialize)]
struct SessionSummary {
    id: String,
    task_id: String,
    status: String,
    summary: String,
    details: Option<String>,
    llm_feedback: Option<String>,
    human_feedback: Option<String>,
    llm_turns: Vec<String>,
    escalation_reason: Option<String>,
    iteration_count: u32,
    reviewer_type: Option<String>,
    review_model: Option<String>,
    review_config: Option<crate::config::ReviewConfig>,
    created_at: String,
    updated_at: String,
    cwd: String,
}

impl From<&Session> for SessionSummary {
    fn from(s: &Session) -> Self {
        SessionSummary {
            id: s.id.clone(),
            task_id: s.task_id.clone(),
            status: s.status.to_string(),
            summary: s.summary.clone(),
            details: s.details.clone(),
            llm_feedback: s.llm_feedback.clone(),
            human_feedback: s.human_feedback.clone(),
            llm_turns: s.llm_turns.clone(),
            escalation_reason: s
                .escalation_reason
                .as_ref()
                .map(|reason| reason.as_str().to_string()),
            iteration_count: s.iteration_count,
            reviewer_type: s
                .reviewer_type
                .as_ref()
                .map(|reviewer| reviewer.as_str().to_string()),
            review_model: s.review_model.clone(),
            review_config: s.review_config.clone(),
            created_at: s.created_at.clone(),
            updated_at: s.updated_at.clone(),
            cwd: s.cwd.clone(),
        }
    }
}

#[derive(Serialize)]
struct SessionDetail {
    id: String,
    task_id: String,
    status: String,
    summary: String,
    details: Option<String>,
    llm_feedback: Option<String>,
    human_feedback: Option<String>,
    llm_turns: Vec<String>,
    escalation_reason: Option<String>,
    iteration_count: u32,
    reviewer_type: Option<String>,
    review_model: Option<String>,
    review_config: Option<crate::config::ReviewConfig>,
    created_at: String,
    updated_at: String,
    cwd: String,
}

impl From<&Session> for SessionDetail {
    fn from(s: &Session) -> Self {
        SessionDetail {
            id: s.id.clone(),
            task_id: s.task_id.clone(),
            status: s.status.to_string(),
            summary: s.summary.clone(),
            details: s.details.clone(),
            llm_feedback: s.llm_feedback.clone(),
            human_feedback: s.human_feedback.clone(),
            llm_turns: s.llm_turns.clone(),
            escalation_reason: s
                .escalation_reason
                .as_ref()
                .map(|reason| reason.as_str().to_string()),
            iteration_count: s.iteration_count,
            reviewer_type: s
                .reviewer_type
                .as_ref()
                .map(|reviewer| reviewer.as_str().to_string()),
            review_model: s.review_model.clone(),
            review_config: s.review_config.clone(),
            created_at: s.created_at.clone(),
            updated_at: s.updated_at.clone(),
            cwd: s.cwd.clone(),
        }
    }
}
