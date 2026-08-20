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
                    "description": "Request a code review for a task. After calling this tool, wait for expert feedback.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "taskId": { "type": "string", "description": "The task ID to review" },
                            "summary": { "type": "string", "description": "Brief summary of the task" },
                            "details": { "type": "string", "description": "Additional details about the task" },
                            "conversationHistory": { "type": "array", "items": { "type": "string" }, "description": "Conversation history as context" },
                            "cwd": { "type": "string", "description": "Absolute path to the project directory being reviewed. Optional but strongly recommended for accurate git diff collection; falls back to peer-cred-resolved cwd if omitted." }
                        },
                        "required": ["taskId", "summary"]
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
            let start_resp = client.post_json("/review/api/request-async", args).await?;

            let session_id = start_resp
                .get("sessionId")
                .and_then(|v| v.as_str())
                .context("request-async response missing sessionId")?
                .to_string();

            // Validate before using in a URL path.
            if !session_id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                bail!("Invalid sessionId returned by daemon: {:?}", session_id);
            }

            let poll_path = format!("/review/api/sessions/{}", session_id);
            const TERMINAL: &[&str] = &["approved", "needs_revision", "escalated", "failed"];
            const POLL_INTERVAL_SECS: u64 = 5;
            const MAX_POLLS: u32 = 360; // 30 minutes

            for _ in 0..MAX_POLLS {
                tokio::time::sleep(tokio::time::Duration::from_secs(POLL_INTERVAL_SECS)).await;

                let detail = client.get_json(&poll_path).await?;

                let status = detail
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                if TERMINAL.contains(&status) {
                    // Normalise to the shape the old blocking endpoint returned.
                    return Ok(serde_json::json!({
                        "status": status,
                        "feedback": detail.get("llm_feedback").or_else(|| detail.get("human_feedback")).unwrap_or(&serde_json::Value::Null),
                        "sessionId": session_id,
                        "iterationCount": detail.get("iteration_count").unwrap_or(&serde_json::Value::Null),
                        "reviewerType": detail.get("reviewer_type").unwrap_or(&serde_json::Value::Null),
                    }));
                }
            }

            bail!(
                "Review timed out after {} minutes. Session {} is still in progress.                  Check /review/api/sessions/{} for status.",
                MAX_POLLS as u64 * POLL_INTERVAL_SECS / 60,
                session_id,
                session_id
            )
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
            // Validate session_id to prevent path traversal
            if !session_id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
                bail!("Invalid sessionId: must be alphanumeric, dashes, or underscores");
            }
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
