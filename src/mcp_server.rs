//! MCP stdio server — `brainrouter mcp` subcommand.
//!
//! Implements the Model Context Protocol over stdio using the `rmcp` crate.
//! This is a thin client: it does NOT load Bonsai or run the review loop.
//! All heavy lifting stays in the daemon; this process forwards tool calls to
//! the daemon via HTTP over the Unix domain socket.
//!
//! Phase 3 wires the real rmcp implementation. For now this binary connects
//! to the daemon socket and dispatches the four tool calls.

use anyhow::{bail, Context, Result};
use brainrouter::daemon_client::{DaemonClient, DaemonEndpoint};
use clap::Args;
use serde_json::{json, Value};
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{error, info};

/// Arguments for the `mcp` subcommand.
#[derive(Args)]
pub struct McpArgs {
    /// Path to the Unix domain socket used to reach the daemon.
    /// Defaults to $XDG_RUNTIME_DIR/brainrouter.sock (or /run/brainrouter.sock).
    #[arg(long)]
    pub socket: Option<PathBuf>,
}

/// Entry point for `brainrouter mcp`.
///
/// Implements the MCP JSON-RPC stdio protocol directly without an external
/// crate dependency. The rmcp crate will be added in Phase 3; this
/// implementation is fully functional in the meantime.
pub async fn run(args: McpArgs) -> Result<()> {
    let socket = args.socket.unwrap_or_else(brainrouter::config::default_socket_path);
    info!(socket = %socket.display(), "Starting brainrouter MCP server");

    // Verify the daemon socket is reachable before starting the protocol loop
    if !socket.exists() {
        bail!(
            "Daemon socket not found at {}. Is `brainrouter serve` running?",
            socket.display()
        );
    }

    let client = DaemonClient::new(DaemonEndpoint::Socket(socket.clone()));

    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);
    let mut writer = tokio::io::BufWriter::new(stdout);

    // MCP JSON-RPC stdio loop
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            // EOF — client disconnected
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let msg: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to parse MCP message: {}", e);
                continue;
            }
        };

        let response = handle_message(&client, msg).await;
        // Notifications return Value::Null — skip writing a response for them.
        if response.is_null() {
            continue;
        }
        let mut response_str = serde_json::to_string(&response)?;
        response_str.push('\n');
        writer.write_all(response_str.as_bytes()).await?;
        writer.flush().await?;
    }

    Ok(())
}

async fn handle_message(client: &DaemonClient, msg: Value) -> Value {
    let id = msg.get("id").cloned().unwrap_or(Value::Null);
    let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");

    match method {
        "initialize" => json_result(id, json!({
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "brainrouter", "version": env!("CARGO_PKG_VERSION") }
        })),

        "notifications/initialized" | "notifications/cancelled" => {
            // One-way notification — no response.
            Value::Null
        }

        // JSON-RPC liveness check — MUST return {"result":{}} (a response is
        // required; only *notifications* go unanswered).
        "ping" => json_result(id, json!({})),

        "tools/list" => json_result(id, json!({
            "tools": [
                {
                    "name": "request_review",
                    "description": "Request a code review for a task. By default waits for expert feedback. Set wait:false to return immediately with {status:\"pending\", sessionId} and keep working; fetch the verdict later with wait_review (preferred) or get_session_details. NEVER re-call request_review to fetch a verdict - it starts a duplicate review.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "taskId": { "type": "string", "description": "The task ID to review" },
                            "summary": { "type": "string", "description": "Brief summary of the task" },
                            "details": { "type": "string", "description": "Additional details about the task" },
                            "conversationHistory": { "type": "array", "items": { "type": "string" }, "description": "Conversation history as context" },
                            "cwd": { "type": "string", "description": "Absolute path to the project directory being reviewed. Optional but strongly recommended for accurate git diff collection; falls back to peer-cred-resolved cwd if omitted." },
                            "wait": { "type": "boolean", "description": "true (default): block until the review reaches a terminal status. false: return immediately with {status:\"pending\", sessionId} so the caller can keep working, then call wait_review(sessionId) before committing." }
                        },
                        "required": ["taskId", "summary"]
                    }
                },
                {
                    "name": "wait_review",
                    "description": "Wait for a review session (from request_review with wait:false) to reach a terminal status. Returns the verdict (status/feedback/reviewerType/reviewModel) when terminal, or the current status with timedOut:true after the timeout (call again). Keys on sessionId - do NOT re-call request_review to fetch a verdict.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "sessionId": { "type": "string", "description": "The sessionId returned by request_review with wait:false" },
                            "timeoutSeconds": { "type": "number", "description": "Max seconds to wait (default 300, range 1-1800)." }
                        },
                        "required": ["sessionId"]
                    }
                },
                {
                    "name": "get_session_list",
                    "description": "Returns list of all review sessions",
                    "inputSchema": { "type": "object", "properties": {} }
                },
                {
                    "name": "get_session_details",
                    "description": "Returns details for a specific session",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "sessionId": { "type": "string", "description": "The session ID to retrieve" }
                        },
                        "required": ["sessionId"]
                    }
                },
                {
                    "name": "resolve_session",
                    "description": "Resolve a session with feedback",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "sessionId": { "type": "string", "description": "The session ID to resolve" },
                            "feedback": { "type": "string", "description": "Feedback for resolving the session" }
                        },
                        "required": ["sessionId", "feedback"]
                    }
                }
            ]
        })),

        "tools/call" => {
            let tool_name = msg
                .get("params")
                .and_then(|p| p.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("");
            let arguments = msg
                .get("params")
                .and_then(|p| p.get("arguments"))
                .cloned()
                .unwrap_or(json!({}));

            match dispatch_tool(client, tool_name, arguments).await {
                Ok(result) => json_result(id, json!({
                    "content": [{ "type": "text", "text": serde_json::to_string(&result).unwrap_or_default() }]
                })),
                Err(e) => json_error(id, -32000, &e.to_string()),
            }
        }

        _ => {
            if msg.get("id").is_none() {
                // Unknown *notification*: JSON-RPC forbids responding.
                Value::Null
            } else {
                json_error(id, -32601, &format!("Method not found: {}", method))
            }
        }
    }
}

/// Terminal review statuses (v1 spelling kept: `failed` is reserved for the
/// background-task error path).
const REVIEW_TERMINAL: &[&str] = &["approved", "needs_revision", "escalated", "failed"];

/// Path-traversal guard for session ids used in URL paths.
fn validate_session_id(session_id: &str) -> Result<()> {
    if !session_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        bail!("Invalid sessionId: must be alphanumeric, dashes, or underscores");
    }
    Ok(())
}

/// The shared terminal-review verdict shape (same as the old blocking
/// endpoint, plus the reviewer provenance `reviewModel`).
fn normalized_terminal(detail: &Value, session_id: &str) -> Value {
    serde_json::json!({
        "status": detail.get("status").cloned().unwrap_or(serde_json::Value::Null),
        "feedback": detail.get("llm_feedback").or_else(|| detail.get("human_feedback")).cloned().unwrap_or(serde_json::Value::Null),
        "sessionId": session_id,
        "iterationCount": detail.get("iteration_count").cloned().unwrap_or(serde_json::Value::Null),
        "reviewerType": detail.get("reviewer_type").cloned().unwrap_or(serde_json::Value::Null),
        "reviewModel": brainrouter::daemon_client::review_model_from_detail(detail),
    })
}

/// Forward a tool call to the daemon via the shared thin client.
async fn dispatch_tool(client: &DaemonClient, tool: &str, args: Value) -> Result<Value> {
    match tool {
        "request_review" => {
            // Use the non-blocking async endpoint so we don't hold a 5-minute
            // HTTP connection open on the UDS socket. Instead we:
            //   1. POST /review/api/request-async → get session ID immediately
            //   2. Poll GET /review/api/sessions/:id every 5s until terminal status
            //   3. Return the same shape as the old blocking endpoint
            //
            // Terminal statuses: approved, needs_revision, escalated, failed.
            // We poll for up to 30 minutes (360 × 5s) before giving up.
            // `wait:false` (parallel workflow) skips step 2 entirely: the caller
            // keeps working and fetches the verdict via wait_review(sessionId).
            let wait = args
                .get("wait")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let mut request_body = args;
            if let Some(obj) = request_body.as_object_mut() {
                obj.remove("wait"); // client-side flag; not part of the daemon's ReviewRequest
            }
            let start_resp = client.post_json("/review/api/request-async", request_body).await?;

            // FR-A: the code reviewer can be switched off. The async start then
            // returns a terminal `disabled` status with no session to poll, so
            // return immediately instead of demanding a sessionId or polling.
            if start_resp.get("status").and_then(|v| v.as_str()) == Some("disabled") {
                return Ok(serde_json::json!({
                    "status": "disabled",
                    "feedback": start_resp.get("feedback").cloned().unwrap_or(serde_json::Value::Null),
                    "sessionId": serde_json::Value::Null,
                }));
            }

            let session_id = start_resp
                .get("sessionId")
                .and_then(|v| v.as_str())
                .context("request-async response missing sessionId")?
                .to_string();

            validate_session_id(&session_id)?;

            if !wait {
                return Ok(serde_json::json!({
                    "status": "pending",
                    "sessionId": session_id,
                }));
            }

            let poll_path = format!("/review/api/sessions/{}", session_id);
            const POLL_INTERVAL_SECS: u64 = 5;
            const MAX_POLLS: u32 = 360; // 30 minutes

            for _ in 0..MAX_POLLS {
                tokio::time::sleep(tokio::time::Duration::from_secs(POLL_INTERVAL_SECS)).await;

                let detail = client.get_json(&poll_path).await?;

                let status = detail
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                if REVIEW_TERMINAL.contains(&status) {
                    return Ok(normalized_terminal(&detail, &session_id));
                }
            }

            bail!(
                "Review timed out after {} minutes. Session {} is still in progress.                  Check /review/api/sessions/{} for status.",
                MAX_POLLS as u64 * POLL_INTERVAL_SECS / 60,
                session_id,
                session_id
            )
        }
        "wait_review" => {
            // Verdict fetch for the async (wait:false) workflow. Keys on the
            // sessionId so the review loop is never duplicated. Check first
            // (the verdict is usually already in by the time the caller asks),
            // then long-poll to the deadline. On timeout, report the current
            // status — NOT an error: the caller may keep waiting.
            let session_id = args
                .get("sessionId")
                .and_then(|v| v.as_str())
                .context("missing sessionId")?
                .to_string();
            validate_session_id(&session_id)?;
            let timeout_secs = args
                .get("timeoutSeconds")
                .and_then(|v| v.as_u64())
                .unwrap_or(300)
                .clamp(1, 1800);

            let poll_path = format!("/review/api/sessions/{}", session_id);
            let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(timeout_secs);
            loop {
                let detail = client.get_json(&poll_path).await?;
                let status = detail
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                if REVIEW_TERMINAL.contains(&status) {
                    return Ok(normalized_terminal(&detail, &session_id));
                }

                let now = tokio::time::Instant::now();
                if now >= deadline {
                    return Ok(serde_json::json!({
                        "status": status,
                        "sessionId": session_id,
                        "timedOut": true,
                        "iterationCount": detail.get("iteration_count").cloned().unwrap_or(serde_json::Value::Null),
                        "reviewModel": brainrouter::daemon_client::review_model_from_detail(&detail),
                    }));
                }
                // Sleep at most until the deadline (2s cadence).
                let nap = tokio::time::Duration::from_secs(2).min(deadline - now);
                tokio::time::sleep(nap).await;
            }
        }
        "get_session_list" => {
            client.get_json("/review/api/sessions").await
        }
        "get_session_details" => {
            let session_id = args
                .get("sessionId")
                .and_then(|v| v.as_str())
                .context("missing sessionId")?
                .to_string();
            validate_session_id(&session_id)?; // path-traversal guard
            let path = format!("/review/api/sessions/{}", session_id);
            client.get_json(&path).await
        }
        "resolve_session" => {
            client.post_json("/review/api/resolve", args).await
        }
        _ => bail!("Unknown tool: {}", tool),
    }
}

// ─── JSON-RPC helpers ────────────────────────────────────────────────────────

fn json_result(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    })
}

fn json_error(id: Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message }
    })
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// Minimal stub daemon: `request-async` hands out a session id; the detail
    /// poll returns `detail_json` (a terminal status). One request per
    /// connection (`connection: close`); the accept loop lives for the test.
    fn stub_daemon(detail_json: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut stream = stream;
                let mut text = String::new();
                let mut buf = [0u8; 8192];
                while !text.contains("\r\n\r\n") {
                    match stream.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => text.push_str(&String::from_utf8_lossy(&buf[..n])),
                    }
                }
                let content_length = text
                    .lines()
                    .find_map(|l| {
                        let (k, v) = l.split_once(':')?;
                        if k.trim().eq_ignore_ascii_case("content-length") {
                            v.trim().parse().ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0usize);
                let mut have = text.split_once("\r\n\r\n").map(|(_, b)| b.len()).unwrap_or(0);
                while have < content_length {
                    match stream.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            text.push_str(&String::from_utf8_lossy(&buf[..n]));
                            have += n;
                        }
                    }
                }
                let path = text.split_whitespace().nth(1).unwrap_or("").to_string();
                let body = if path.starts_with("/review/api/request-async") {
                    r#"{"sessionId":"stub-session-1","status":"pending"}"#.to_string()
                } else {
                    detail_json.to_string()
                };
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(resp.as_bytes());
                let _ = stream.flush();
            }
        });
        url
    }

    #[tokio::test]
    async fn request_review_result_names_the_reviewer_model() {
        let detail = r#"{"status":"approved","llm_feedback":"SHIP IT","iteration_count":1,"reviewer_type":"llm","review_model":"requested cloud/vendor/rev; actual cloud → mock (vendor/rev)","review_selector":"cloud/vendor/rev"}"#;
        let client = DaemonClient::new(DaemonEndpoint::Url(stub_daemon(detail)));
        let out = dispatch_tool(&client, "request_review", json!({ "taskId": "t", "summary": "s" }))
            .await
            .expect("stubbed review must terminate");
        assert_eq!(out["status"], "approved");
        // The OMP tool result must name the reviewer model and its backend.
        assert_eq!(
            out["reviewModel"],
            "requested cloud/vendor/rev; actual cloud → mock (vendor/rev)"
        );
    }

    #[tokio::test]
    async fn review_model_falls_back_to_the_requested_card_selector() {
        // No LLM call ever succeeded (escalated llm_error at iteration 0): the
        // requested card is the best available reviewer provenance.
        let detail = r#"{"status":"escalated","escalation_reason":"llm_error","iteration_count":0,"reviewer_type":null,"review_model":null,"review_selector":"cloud/vendor/rev"}"#;
        let client = DaemonClient::new(DaemonEndpoint::Url(stub_daemon(detail)));
        let out = dispatch_tool(&client, "request_review", json!({ "taskId": "t", "summary": "s" }))
            .await
            .expect("stubbed review must terminate");
        assert_eq!(out["reviewModel"], "cloud/vendor/rev");
    }

    #[tokio::test]
    async fn wait_false_returns_pending_immediately_without_polling() {
        // The stub's detail route would answer `approved` if polled; the async
        // path must instead return `pending` + sessionId without ever polling
        // (that is the whole point: the caller keeps working).
        let detail = r#"{"status":"approved","llm_feedback":"SHIP IT","iteration_count":1,"reviewer_type":"llm","review_model":"requested cloud/vendor/rev; actual cloud → mock (vendor/rev)","review_selector":"cloud/vendor/rev"}"#;
        let client = DaemonClient::new(DaemonEndpoint::Url(stub_daemon(detail)));
        let out = dispatch_tool(
            &client,
            "request_review",
            json!({ "taskId": "t", "summary": "s", "wait": false }),
        )
        .await
        .expect("async start must return immediately");
        assert_eq!(out["status"], "pending");
        assert_eq!(out["sessionId"], "stub-session-1");
    }

    #[tokio::test]
    async fn wait_review_returns_terminal_verdict_on_first_check() {
        let detail = r#"{"status":"approved","llm_feedback":"SHIP IT","iteration_count":1,"reviewer_type":"llm","review_model":"requested cloud/vendor/rev; actual cloud → mock (vendor/rev)","review_selector":"cloud/vendor/rev"}"#;
        let client = DaemonClient::new(DaemonEndpoint::Url(stub_daemon(detail)));
        let out = dispatch_tool(
            &client,
            "wait_review",
            json!({ "sessionId": "stub-session-1", "timeoutSeconds": 5 }),
        )
        .await
        .expect("terminal session must resolve immediately");
        assert_eq!(out["status"], "approved");
        assert_eq!(out["feedback"], "SHIP IT");
        assert_eq!(out["reviewModel"], "requested cloud/vendor/rev; actual cloud → mock (vendor/rev)");
        assert!(out.get("timedOut").is_none());
    }

    #[tokio::test]
    async fn wait_review_timeout_reports_current_status_not_an_error() {
        // Still pending: the caller gets the live status with timedOut:true and
        // may call again — a timeout is not a failure.
        let detail = r#"{"status":"pending","iteration_count":0,"reviewer_type":null,"review_model":null,"review_selector":"cloud/vendor/rev"}"#;
        let client = DaemonClient::new(DaemonEndpoint::Url(stub_daemon(detail)));
        let started = std::time::Instant::now();
        let out = dispatch_tool(
            &client,
            "wait_review",
            json!({ "sessionId": "stub-session-1", "timeoutSeconds": 1 }),
        )
        .await
        .expect("timeout must be a value, not an error");
        assert!(started.elapsed() < std::time::Duration::from_secs(5), "must bound by the deadline");
        assert_eq!(out["status"], "pending");
        assert_eq!(out["timedOut"], true);
        assert_eq!(out["reviewModel"], "cloud/vendor/rev", "provenance visible while pending too");
    }
}
