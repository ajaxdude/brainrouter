//! Escalation UI — HTTP handlers for /review/* routes.
//!
//! Routes:
//!   GET  /review/              — session dashboard (HTML, auto-refresh)
//!   GET  /review/session/:id   — session detail (HTML, with human-resolve form)
//!   POST /review/session/:id/resolve — human submits feedback
//!   GET  /review/api/sessions  — JSON session list
//!   GET  /review/api/sessions/:id — JSON session detail

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
const DASHBOARD_HTML: &str = include_str!("templates/dashboard.html");
const SESSION_HTML: &str = include_str!("templates/session.html");

/// Handle all /review/* requests. Called from the main request dispatcher.
pub async fn handle_review_request(
    req: Request<Incoming>,
    review_service: Arc<ReviewService>,
) -> Result<Response<UnsyncBoxBody<Bytes, anyhow::Error>>, Infallible> {
    let method = req.method().as_str();
    let path = req.uri().path().to_string();

    let response = match (method, path.as_str()) {
        // Dashboard
        ("GET", "/review/" | "/review") => html_response(DASHBOARD_HTML),

        // Session detail page
        ("GET", p) if p.starts_with("/review/session/") && !p.ends_with("/resolve") => {
            html_response(SESSION_HTML)
        }

        // Human resolve endpoint
        ("POST", p) if p.ends_with("/resolve") => {
            let session_id = extract_session_id_from_resolve_path(&path);
            handle_resolve(req, review_service, session_id).await
        }

        // JSON API — MCP: start review (long-running, blocks until loop completes)
        ("POST", "/review/api/request") => {
            handle_request_review(req, review_service).await
        }

        // JSON API — MCP: resolve a session with human feedback
        ("POST", "/review/api/resolve") => {
            handle_api_resolve(req, review_service).await
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

/// POST /review/api/request — MCP thin client calls this to trigger a review.
/// Long-running: blocks until the review loop completes.
async fn handle_request_review(
    req: Request<Incoming>,
    review_service: Arc<ReviewService>,
) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    use http_body_util::BodyExt;

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
    }

    let body: ReviewRequest = match serde_json::from_slice(&body_bytes) {
        Ok(b) => b,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("Invalid JSON: {}", e)),
    };

    match review_service
        .start_review(body.task_id, body.summary, body.details, body.conversation_history)
        .await
    {
        Ok(result) => json_ok(&serde_json::json!({
            "status": result.status.as_str(),
            "feedback": result.feedback,
            "sessionId": result.session_id,
            "iterationCount": result.iteration_count,
            "reviewerType": format!("{:?}", result.reviewer_type).to_lowercase()
        })),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// POST /review/api/resolve — MCP thin client calls this to resolve a session.
async fn handle_api_resolve(
    req: Request<Incoming>,
    review_service: Arc<ReviewService>,
) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    use http_body_util::BodyExt;

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
    use http_body_util::BodyExt;

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

fn extract_session_id_from_resolve_path(path: &str) -> String {
    // Path: /review/session/:id/resolve
    path.trim_start_matches("/review/session/")
        .trim_end_matches("/resolve")
        .to_string()
}

// ─── Response helpers ────────────────────────────────────────────────────────

fn html_response(html: &'static str) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
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

fn json_ok<T: Serialize>(body: &T) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    let json = serde_json::to_vec(body).unwrap_or_default();
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(
            Full::new(Bytes::from(json))
                .map_err(|_| unreachable!())
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
                .map_err(|_| unreachable!())
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
    iteration_count: u32,
    updated_at: String,
    review_model: Option<String>,
}

impl From<&Session> for SessionSummary {
    fn from(s: &Session) -> Self {
        SessionSummary {
            id: s.id.clone(),
            task_id: s.task_id.clone(),
            status: s.status.to_string(),
            summary: s.summary.clone(),
            iteration_count: s.iteration_count,
            updated_at: s.updated_at.clone(),
            review_model: s.review_model.clone(),
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
    escalation_reason: Option<String>,
    iteration_count: u32,
    reviewer_type: Option<String>,
    created_at: String,
    updated_at: String,
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
            escalation_reason: s
                .escalation_reason
                .as_ref()
                .map(|r| format!("{:?}", r).to_lowercase()),
            iteration_count: s.iteration_count,
            reviewer_type: s.reviewer_type.as_ref().map(|r| format!("{:?}", r).to_lowercase()),
            created_at: s.created_at.clone(),
            updated_at: s.updated_at.clone(),
        }
    }
}
