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
use clap::Args;
use serde_json::{json, Value};
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{error, info};

/// Arguments for the `mcp` subcommand.
#[derive(Args)]
pub struct McpArgs {
    /// Path to the Unix domain socket used to reach the daemon.
    #[arg(long, default_value = "/run/brainrouter.sock")]
    pub socket: PathBuf,
}

/// Entry point for `brainrouter mcp`.
///
/// Implements the MCP JSON-RPC stdio protocol directly without an external
/// crate dependency. The rmcp crate will be added in Phase 3; this
/// implementation is fully functional in the meantime.
pub async fn run(args: McpArgs) -> Result<()> {
    info!(socket = %args.socket.display(), "Starting brainrouter MCP server");

    // Verify the daemon socket is reachable before starting the protocol loop
    if !args.socket.exists() {
        bail!(
            "Daemon socket not found at {}. Is `brainrouter serve` running?",
            args.socket.display()
        );
    }

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

        let response = handle_message(&args.socket, msg).await;
        let mut response_str = serde_json::to_string(&response)?;
        response_str.push('\n');
        writer.write_all(response_str.as_bytes()).await?;
        writer.flush().await?;
    }

    Ok(())
}

async fn handle_message(socket_path: &PathBuf, msg: Value) -> Value {
    let id = msg.get("id").cloned().unwrap_or(Value::Null);
    let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");

    match method {
        "initialize" => json_result(id, json!({
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "brainrouter", "version": env!("CARGO_PKG_VERSION") }
        })),

        "notifications/initialized" => {
            // One-way notification — no response needed; return null to suppress output
            json!({})
        }

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
                            "conversationHistory": { "type": "array", "items": { "type": "string" }, "description": "Conversation history as context" }
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

            match dispatch_tool(socket_path, tool_name, arguments).await {
                Ok(result) => json_result(id, json!({
                    "content": [{ "type": "text", "text": serde_json::to_string(&result).unwrap_or_default() }]
                })),
                Err(e) => json_error(id, -32000, &e.to_string()),
            }
        }

        _ => json_error(id, -32601, &format!("Method not found: {}", method)),
    }
}

/// Forward a tool call to the daemon via Unix socket HTTP.
async fn dispatch_tool(socket_path: &PathBuf, tool: &str, args: Value) -> Result<Value> {
    match tool {
        "request_review" => {
            http_uds_request(socket_path, "/review/api/request", args).await
        }
        "get_session_list" => {
            http_uds_request(socket_path, "/review/api/sessions", json!({})).await
        }
        "get_session_details" => {
            let session_id = args
                .get("sessionId")
                .and_then(|v| v.as_str())
                .context("missing sessionId")?
                .to_string();
            let path = format!("/review/api/sessions/{}", session_id);
            http_uds_request(socket_path, &path, json!({})).await
        }
        "resolve_session" => {
            http_uds_request(socket_path, "/review/api/resolve", args).await
        }
        _ => bail!("Unknown tool: {}", tool),
    }
}

/// Make an HTTP request over the Unix domain socket.
async fn http_uds_request(socket_path: &PathBuf, path: &str, body: Value) -> Result<Value> {
    use tokio::net::UnixStream;

    let stream = UnixStream::connect(socket_path)
        .await
        .with_context(|| format!("Cannot connect to daemon socket at {}", socket_path.display()))?;

    let body_bytes = serde_json::to_vec(&body)?;
    let method = if body_bytes == b"{}" { "GET" } else { "POST" };

    let request = format!(
        "{} {} HTTP/1.0\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        method,
        path,
        body_bytes.len()
    );

    use tokio::io::AsyncReadExt;
    let (mut rx, mut tx) = stream.into_split();

    tx.write_all(request.as_bytes()).await?;
    if method == "POST" {
        tx.write_all(&body_bytes).await?;
    }
    drop(tx);

    let mut response_bytes = Vec::new();
    rx.read_to_end(&mut response_bytes).await?;

    // Strip HTTP headers — find \r\n\r\n
    let response_str = String::from_utf8_lossy(&response_bytes);
    let body_start = response_str
        .find("\r\n\r\n")
        .map(|i| i + 4)
        .unwrap_or(0);

    let json_body = &response_bytes[body_start..];
    let result: Value = serde_json::from_slice(json_body)
        .with_context(|| format!("Failed to parse daemon response as JSON"))?;

    Ok(result)
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
