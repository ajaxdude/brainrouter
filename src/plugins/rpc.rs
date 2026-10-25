//! Plugin RPC v1 surface (JSON-RPC 2.0 over the per-plugin UDS).
//!
//! Framing: one JSON object per line, `\n` terminated. Additive-only
//! evolution. Body cap 1 MiB; per-plugin in-flight 4 + queue 16, excess →
//! JSON-RPC limit error (−32005). Unknown method → −32601. Connect handshake:
//! `initialize` (constant-time token check, peer uid, manifest
//! `min_daemon_version`) → grants + protocol version. A failed handshake
//! closes the connection and is logged — it is NOT a plugin crash (it never
//! reaches the supervisor).
//!
//! Methods (token + per-method grant gating): `chat.completions`,
//! `models.list`, `routing.events.tail`, `inflight.list`, `config.get/set`,
//! `kv.get/set`, `secrets.get` (grant-enforced per credential), `notify`,
//! `set_status`. Daemon→plugin pushes: `config.changed`. No system ops.

use super::{
    daemon_version, manifest::daemon_version_satisfies, store, Inner, PluginManager,
    JSONRPC_INTERNAL_ERROR, JSONRPC_INVALID_REQUEST, JSONRPC_LIMIT, JSONRPC_METHOD_NOT_FOUND,
    JSONRPC_NOT_AUTHED, JSONRPC_PARSE_ERROR, MAX_BODY_BYTES, PROTOCOL_VERSION, RPC_DELIM,
};
use crate::server::AppState;
use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

pub(crate) async fn accept_loop(inner: Arc<Inner>, id: String, listener: Arc<UnixListener>) {
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let inner = inner.clone();
                let id = id.clone();
                tokio::spawn(async move {
                    if let Err(e) = serve_connection(inner, id, stream).await {
                        debug!(error = %e, "plugin RPC connection ended");
                    }
                });
            }
            Err(e) => {
                // Accept errors (EMFILE-style storms) must not spin the loop.
                warn!(plugin = %id, error = %e, "plugin socket accept failed");
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }
    }
}

#[derive(Clone)]
struct Session {
    plugin_id: String,
    grants: Arc<HashSet<String>>,
}

pub(crate) async fn serve_connection(inner: Arc<Inner>, plugin_id: String, stream: UnixStream) -> Result<()> {
    // Peer-uid check (design: "daemon checks token ..., peer uid"). A foreign
    // uid gets a closed connection + a log line; it is not a plugin crash.
    let cred = stream.peer_cred().context("peer_cred on plugin socket")?;
    let euid = unsafe { libc::geteuid() };
    let peer_uid = cred.uid();
    if peer_uid != euid {
        warn!(
            plugin = %plugin_id,
            peer_uid = peer_uid,
            expected_uid = euid,
            "plugin RPC connection rejected: peer uid mismatch (connection closed)"
        );
        drop(stream);
        return Ok(());
    }

    let (reader, writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<String>();
    let writer_task = tokio::spawn(async move {
        let mut writer = writer;
        while let Some(line) = out_rx.recv().await {
            if writer.write_all(line.as_bytes()).await.is_err() {
                break;
            }
            if writer.flush().await.is_err() {
                break;
            }
        }
    });

    let manager = PluginManager::from_inner(inner.clone());
    let mut session: Option<Session> = None;

    loop {
        // Read one request line, bounded by the 1 MiB body cap.
        let mut line = Vec::with_capacity(16 * 1024);
        let mut over_cap = false;
        loop {
            match reader.read_until(RPC_DELIM, &mut line).await {
                Ok(0) => {
                    // EOF: connection done.
                    writer_task.abort();
                    return Ok(());
                }
                Ok(_) => {}
                Err(e) => {
                    let _ = out_tx.send(rpc_error(Value::Null, JSONRPC_INTERNAL_ERROR, &e.to_string()));
                    writer_task.abort();
                    return Ok(());
                }
            }
            let ended = line.ends_with(&[RPC_DELIM]);
            if ended {
                line.pop();
            }
            if line.len() > MAX_BODY_BYTES {
                over_cap = true;
                if ended {
                    break;
                }
                // Drain the remainder of the oversized line.
                loop {
                    let mut scratch = [0u8; 64 * 1024];
                    let n = reader.read(&mut scratch).await?;
                    if n == 0 {
                        writer_task.abort();
                        return Ok(());
                    }
                    if scratch[..n].contains(&RPC_DELIM) {
                        break;
                    }
                }
                break;
            }
            if ended {
                break;
            }
        }
        if over_cap {
            // Answer with the request's id when the (oversized) line begins
            // with a parseable one, so correlation-keyed clients see the
            // refusal as the answer to their request.
            let id = scan_id_prefix(&line).unwrap_or(Value::Null);
            let _ = out_tx.send(rpc_error(
                id,
                JSONRPC_INVALID_REQUEST,
                "request body exceeds the 1 MiB RPC cap",
            ));
            continue;
        }
        if line.is_empty() {
            continue;
        }
        let msg: Value = match serde_json::from_slice(&line) {
            Ok(v) => v,
            Err(e) => {
                let _ = out_tx.send(rpc_error(
                    Value::Null,
                    JSONRPC_PARSE_ERROR,
                    &format!("parse error: {e}"),
                ));
                continue;
            }
        };
        // Response frame to a server-initiated request (protocol v2,
        // `decision.classify`): has result/error, no method. Route it to the
        // pending correlation map instead of treating it as a request.
        let is_response = msg.get("method").is_none()
            && (msg.get("result").is_some() || msg.get("error").is_some());
        if is_response {
            let manager = PluginManager::from_inner(inner.clone());
            if !manager.resolve_pending(&plugin_id, &msg) {
                debug!(plugin = %plugin_id, "unmatched plugin response frame dropped");
            }
            continue;
        }
        let Some(id) = msg.get("id").cloned() else {
            let _ = out_tx.send(rpc_error(
                Value::Null,
                JSONRPC_INVALID_REQUEST,
                "plugin RPC requests must carry an id",
            ));
            continue;
        };
        let method: String = msg
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let params = msg.get("params").cloned().unwrap_or_else(|| json!({}));
        if method.is_empty() {
            let _ = out_tx.send(rpc_error(id, JSONRPC_INVALID_REQUEST, "missing method"));
            continue;
        }

        if method == "initialize" {
            match handle_initialize(&inner, &plugin_id, &params) {
                Ok(result) => {
                    let grants: Arc<HashSet<String>> = Arc::new(
                        result
                            .get("grants")
                            .and_then(Value::as_array)
                            .map(|a| {
                                a.iter()
                                    .filter_map(Value::as_str)
                                    .map(str::to_string)
                                    .collect()
                            })
                            .unwrap_or_default(),
                    );
                    let negotiated = result
                        .get("protocol_version")
                        .and_then(Value::as_u64)
                        .unwrap_or(1) as u32;
                    session = Some(Session {
                        plugin_id: plugin_id.clone(),
                        grants,
                    });
                    let _ = out_tx.send(rpc_result(id, result));
                    // Push notifications only reach authenticated sessions.
                    manager.register_session(&plugin_id, out_tx.clone(), negotiated);
                }
                Err(e) => {
                    // Design: "auth failure closes the connection and is
                    // logged — it is NOT counted as a crash". No response.
                    warn!(plugin = %plugin_id, reason = %e,
                        "plugin initialize rejected (connection closed)");
                    writer_task.abort();
                    return Ok(());
                }
            }
            continue;
        }

        let Some(current) = session.clone() else {
            let _ = out_tx.send(rpc_error(
                id,
                JSONRPC_NOT_AUTHED,
                "not authenticated: call initialize first",
            ));
            return Ok(());
        };
        if !is_rpc_method(&method) {
            let _ = out_tx.send(rpc_error(
                id,
                JSONRPC_METHOD_NOT_FOUND,
                &format!("unknown method {method:?}"),
            ));
            continue;
        }
        if !current.grants.contains(method.as_str()) {
            let _ = out_tx.send(rpc_error(
                id,
                JSONRPC_METHOD_NOT_FOUND,
                &format!("method {method:?} not granted to plugin {}", current.plugin_id),
            ));
            continue;
        }

        // Requests run concurrently (in-flight caps are per plugin, shared by
        // all sessions of this plugin).
        let inner2 = inner.clone();
        let manager2 = manager.clone();
        let out = out_tx.clone();
        tokio::spawn(async move {
            let resp = dispatch(inner2, manager2, current, method, params).await;
            let _ = out.send(framed(id, resp));
        });
    }
}

fn handle_initialize(inner: &Arc<Inner>, socket_plugin: &str, params: &Value) -> Result<Value> {
    let claimed = params
        .get("id")
        .or_else(|| params.get("plugin_id"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let token = params.get("token").and_then(Value::as_str).unwrap_or("");
    if claimed.is_empty() {
        anyhow::bail!("initialize missing plugin id");
    }
    if claimed != socket_plugin {
        anyhow::bail!("initialize id {claimed:?} does not own the socket for {socket_plugin:?}");
    }
    let token_path = store::plugin_state_dir(&inner.state_root, socket_plugin).join("token");
    let expected = std::fs::read_to_string(&token_path)
        .context("reading minted plugin token")?;
    let expected = expected.trim();
    if expected.is_empty() || !store::ct_eq(token, expected) {
        anyhow::bail!("invalid token");
    }
    let (grants, min_version) = {
        let reg = inner.registry.lock().unwrap();
        let entry = reg
            .plugins
            .get(socket_plugin)
            .context("plugin is not in the registry")?;
        (entry.grants.clone(), entry.min_daemon_version.clone())
    };
    if !daemon_version_satisfies(daemon_version(), &min_version) {
        anyhow::bail!(
            "daemon {} is older than the plugin's min_daemon_version {}",
            daemon_version(),
            min_version
        );
    }
    // Protocol negotiation (v2 L3): the plugin declares what it speaks
    // (`protocol_version`, default 1 = L2 bridge); the daemon answers with
    // the minimum. v2 adds server-initiated `decision.classify`; the
    // manifest's `min_daemon_version` keeps a plugin that REQUIRES v2 from
    // pairing with an old daemon.
    let requested = params
        .get("protocol_version")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .min(u32::MAX as u64) as u32;
    let negotiated = requested.min(PROTOCOL_VERSION);
    info!(plugin = %socket_plugin, grants = ?grants, protocol = negotiated, "plugin initialize accepted");
    let mut result = json!({
        "plugin_id": socket_plugin,
        "protocol_version": negotiated,
        "grants": grants,
        "daemon_version": daemon_version(),
    });
    if negotiated >= 2 {
        result["server_methods"] = json!(["decision.classify"]);
    }
    Ok(result)
}

async fn dispatch(
    inner: Arc<Inner>,
    manager: PluginManager,
    session: Session,
    method: String,
    params: Value,
) -> String {
    let app = match inner.app.get() {
        Some(a) => a.clone(),
        None => return error_body(JSONRPC_INTERNAL_ERROR, "daemon not ready"),
    };
    let plugin_id = session.plugin_id.as_str();
    let result: std::result::Result<Value, (i32, String)> = match method.as_str() {
        "chat.completions" => chat_completions(&app, &manager, plugin_id, params)
            .await
            .map_err(limit_or(-32000)),
        "models.list" => Ok(crate::server::model_list_json(&app).await),
        "routing.events.tail" => Ok(routing_events_tail(&app, &params)),
        "inflight.list" => Ok(app.inflight.json()),
        "config.get" => manager
            .config_get(plugin_id, param_key(&params))
            .map(|v| json!({ "key": param_key(&params), "value": v }))
            .map_err(plain),
        "config.set" => manager
            .config_set(
                plugin_id,
                param_key(&params),
                params.get("value").cloned().unwrap_or(Value::Null),
            )
            .map_err(plain),
        "kv.get" => manager
            .kv_get(plugin_id, param_key(&params))
            .map(|v| json!({ "key": param_key(&params), "value": v }))
            .map_err(plain),
        "kv.set" => manager
            .kv_set(
                plugin_id,
                param_key(&params),
                params.get("value").cloned().unwrap_or(Value::Null),
            )
            .map_err(plain),
        "secrets.get" => secrets_get(&inner, plugin_id, &params).map_err(plain),
        "notify" => {
            let text = scrub_text(
                params
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            );
            let toast = json!({
                "plugin": plugin_id,
                "text": text,
                "level": params.get("level").and_then(Value::as_str).unwrap_or("info"),
                "ts": chrono::Utc::now().to_rfc3339(),
            });
            manager.record_toast(toast.clone());
            info!(plugin = %plugin_id, text = %text, "plugin notify");
            Ok(json!({ "ok": true, "toast": toast }))
        }
        "set_status" => {
            let text = scrub_text(
                params
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            );
            manager
                .set_status_line(plugin_id, &text)
                .map(|()| json!({ "ok": true, "status_line": text }))
                .map_err(plain)
        }
        other => Err((JSONRPC_METHOD_NOT_FOUND, format!("unknown method {other:?}"))),
    };
    match result {
        Ok(value) => json!({ "result": value }).to_string(),
        Err((code, message)) => json!({ "error": { "code": code, "message": message } }).to_string(),
    }
}

fn plain(e: anyhow::Error) -> (i32, String) {
    (-32000, e.to_string())
}

/// Map a limit breach to the JSON-RPC busy code (Hank ruling: −32005 naming
/// the cap); everything else keeps `fallback`.
fn limit_or(fallback: i32) -> impl Fn(anyhow::Error) -> (i32, String) {
    move |e: anyhow::Error| {
        let text = e.to_string();
        if text.contains("in-flight cap") {
            (JSONRPC_LIMIT, text)
        } else {
            (fallback, text)
        }
    }
}

/// The closed RPC v1 method surface (design: additive-only; no system ops).
/// Best-effort `"id":<number>` extraction from the head of a (request) line —
/// used for the over-cap refusal so the error frame is correlatable.
fn scan_id_prefix(bytes: &[u8]) -> Option<Value> {
    let head = &bytes[..bytes.len().min(8192)];
    let text = std::str::from_utf8(head).ok()?;
    let at = text.find("\"id\"")?;
    let rest = text[at + 4..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse::<u64>().ok().map(Value::from)
}

fn is_rpc_method(method: &str) -> bool {
    matches!(
        method,
        "chat.completions"
            | "models.list"
            | "routing.events.tail"
            | "inflight.list"
            | "config.get"
            | "config.set"
            | "kv.get"
            | "kv.set"
            | "secrets.get"
            | "notify"
            | "set_status"
    )
}

/// N7: scrub control characters (newlines, CR, tabs, ANSI escapes) from
/// plugin-supplied text before it lands in the journal/CLI surfaces — a
/// plugin log line must not forge daemon log lines.
fn scrub_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // Drop a full ANSI escape sequence (ESC [ ... final-byte).
            if chars.clone().next() == Some('[') {
                chars.next();
                for c2 in chars.by_ref() {
                    if c2.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            continue;
        }
        if c.is_control() {
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out.chars().take(2000).collect()
}

fn param_key(params: &Value) -> &str {
    ["key", "name"]
        .iter()
        .find_map(|k| params.get(*k).and_then(Value::as_str))
        .unwrap_or("")
}

fn secrets_get(inner: &Arc<Inner>, plugin_id: &str, params: &Value) -> Result<Value> {
    let cred_id = ["id", "credential_id", "key", "name"]
        .iter()
        .find_map(|k| params.get(*k).and_then(Value::as_str))
        .context("secrets.get needs a credential id")?;
    // Credential-grant axis: the credential must explicitly grant this plugin.
    let masked = inner.creds.get_masked(cred_id)?;
    let granted = masked
        .get("grants")
        .and_then(Value::as_array)
        .map(|a| a.iter().any(|g| g.as_str() == Some(plugin_id)))
        .unwrap_or(false);
    if !granted {
        anyhow::bail!("credential '{cred_id}' does not grant plugin '{plugin_id}'");
    }
    let secret = inner.creds.resolve(cred_id)?;
    Ok(json!({ "id": cred_id, "value": secret.expose() }))
}

fn routing_events_tail(app: &AppState, params: &Value) -> Value {
    let since = params.get("since").and_then(Value::as_u64).unwrap_or(0);
    let limit = params
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(100)
        .min(500) as usize;
    let events: Vec<Value> = app
        .routing_events
        .get_all()
        .into_iter()
        .filter(|e| e.id > since)
        .take(limit)
        .map(|e| serde_json::to_value(&e).unwrap_or(Value::Null))
        .collect();
    json!({ "events": events })
}

async fn chat_completions(
    app: &Arc<AppState>,
    manager: &PluginManager,
    plugin_id: &str,
    params: Value,
) -> Result<Value> {
    let runtime = manager.runtime(plugin_id);
    // Per-plugin in-flight cap 4 + queue 16 (design). Total permits = cap +
    // queue; beyond that the JSON-RPC limit error answers.
    let _total = runtime.chat_total.try_acquire().map_err(|_| {
        anyhow::anyhow!(
            "plugin in-flight cap reached ({} active + {} queued)",
            super::INFLIGHT_CAP,
            super::INFLIGHT_QUEUE
        )
    })?;
    let _active = runtime.chat_active.acquire().await.map_err(|_| {
        anyhow::anyhow!("plugin in-flight cap closed ({} active)", super::INFLIGHT_CAP)
    })?;
    let request: crate::types::ChatCompletionRequest =
        serde_json::from_value(params).context("invalid chat.completions params")?;
    let source = format!("plugin {plugin_id}");
    let conv_id = crate::router::conversation_fingerprint(&request);
    let handle = app.inflight.register(
        "POST rpc chat.completions".to_string(),
        request.model.clone(),
        source.clone(),
        "uds:plugins".to_string(),
        String::new(),
        conv_id,
        0,
        request.max_tokens,
    );
    let result = app
        .router
        .route_plugin(request, source)
        .await
        .map(|(resp, _info)| resp);
    match result {
        Ok(crate::provider::ProviderResponse::Stream(stream)) => {
            let completion = collect_completion(stream).await?;
            handle.add_bytes(completion.to_string().len() as u64);
            Ok(completion)
        }
        Err(e) => {
            handle.add_bytes(0);
            Err(e).context("chat.completions routing failed")
        }
    }
}

/// Consume the provider SSE (or plain JSON) stream and return a final
/// `chat.completion` object — the RPC surface speaks request/response.
pub(crate) async fn collect_completion(mut stream: crate::provider::SseStream) -> Result<Value> {
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        buf.extend_from_slice(&chunk?);
    }
    let text = String::from_utf8_lossy(&buf).to_string();
    if text.trim_start().starts_with('{') && !text.contains("data:") {
        // Plain JSON completion body (non-streaming provider responses).
        return serde_json::from_str(&text).context("provider returned invalid JSON");
    }
    // OpenAI SSE: aggregate deltas.
    let mut id = String::new();
    let mut model = String::new();
    let mut created = 0i64;
    let mut usage: Option<Value> = None;
    let mut contents: Vec<String> = Vec::new();
    let mut roles: Vec<String> = Vec::new();
    let mut finish: Vec<Option<String>> = Vec::new();
    for frame in text.split("\n\n") {
        for raw in frame.lines() {
            let Some(data) = raw.trim().strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                continue;
            }
            let Ok(chunk) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            if id.is_empty() {
                if let Some(v) = chunk.get("id").and_then(Value::as_str) {
                    id = v.to_string();
                }
            }
            if model.is_empty() {
                if let Some(v) = chunk.get("model").and_then(Value::as_str) {
                    model = v.to_string();
                }
            }
            if created == 0 {
                if let Some(v) = chunk.get("created").and_then(Value::as_i64) {
                    created = v;
                }
            }
            if chunk.get("usage").map(|u| !u.is_null()).unwrap_or(false) {
                usage = chunk.get("usage").cloned();
            }
            if let Some(choices) = chunk.get("choices").and_then(Value::as_array) {
                for c in choices {
                    // N6: clamp a hostile index (SSE is untrusted upstream
                    // data) so `index: 1e9` cannot drive a giant allocation.
                    let idx = (c.get("index").and_then(Value::as_u64).unwrap_or(0) as usize).min(64);
                    while contents.len() <= idx {
                        contents.push(String::new());
                        roles.push("assistant".to_string());
                        finish.push(None);
                    }
                    if let Some(d) = c.get("delta") {
                        if let Some(role) = d.get("role").and_then(Value::as_str) {
                            if !role.is_empty() {
                                roles[idx] = role.to_string();
                            }
                        }
                        if let Some(content) = d.get("content").and_then(Value::as_str) {
                            contents[idx].push_str(content);
                        }
                    }
                    if let Some(f) = c.get("finish_reason").and_then(Value::as_str) {
                        finish[idx] = Some(f.to_string());
                    }
                }
            }
        }
    }
    let choices: Vec<Value> = contents
        .iter()
        .enumerate()
        .map(|(i, content)| {
            json!({
                "index": i,
                "message": {
                    "role": roles.get(i).cloned().unwrap_or_else(|| "assistant".into()),
                    "content": content,
                },
                "finish_reason": finish.get(i).and_then(|f| f.clone()),
            })
        })
        .collect();
    let mut out = json!({
        "id": if id.is_empty() { "chatcmpl-plugin".to_string() } else { id },
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": choices,
    });
    if let Some(u) = usage {
        out["usage"] = u;
    }
    Ok(out)
}

fn rpc_result(id: Value, result: Value) -> String {
    json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string() + "\n"
}

fn rpc_error(id: Value, code: i32, message: &str) -> String {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
        .to_string()
        + "\n"
}

fn error_body(code: i32, message: &str) -> String {
    json!({ "error": { "code": code, "message": message } }).to_string()
}

/// Attach the JSON-RPC envelope (id + version) to a dispatch body.
fn framed(id: Value, body: String) -> String {
    let mut v: Value = serde_json::from_str(&body).unwrap_or_else(|_| {
        json!({ "error": { "code": JSONRPC_INTERNAL_ERROR, "message": "dispatcher bug" } })
    });
    if let Some(obj) = v.as_object_mut() {
        obj.insert("jsonrpc".into(), json!("2.0"));
        obj.insert("id".into(), id);
    }
    v.to_string() + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_and_helpers() {
        let msg = framed(json!(7), json!({ "result": { "ok": true } }).to_string());
        let parsed: Value = serde_json::from_str(msg.trim_end()).unwrap();
        assert_eq!(parsed["id"], json!(7));
        assert_eq!(parsed["jsonrpc"], json!("2.0"));
        assert_eq!(parsed["result"]["ok"], json!(true));
        assert!(store::ct_eq("abc", "abc"));
        assert!(!store::ct_eq("abc", "abd"));
    }

    #[test]
    fn scrub_text_strips_control_chars_and_ansi() {
        assert_eq!(scrub_text("a\nb\tc"), "a b c");
        assert_eq!(scrub_text("red\u{1b}[31malert\u{1b}[0m"), "redalert");
        assert_eq!(scrub_text("plain"), "plain");
        assert_eq!(scrub_text(&"x".repeat(5000)).len(), 2000);
    }

    #[test]
    fn rpc_method_surface_is_closed() {
        assert!(is_rpc_method("chat.completions"));
        assert!(is_rpc_method("set_status"));
        assert!(!is_rpc_method("system.restart"));
        assert!(!is_rpc_method("daemon.flush"));
    }

    #[test]
    fn param_key_prefers_key_then_name() {
        assert_eq!(param_key(&json!({"key": "k", "name": "n"})), "k");
        assert_eq!(param_key(&json!({"name": "n"})), "n");
        assert_eq!(param_key(&json!({})), "");
    }

    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Runtime::new().unwrap().block_on(f)
    }

    #[test]
    fn sse_aggregation_builds_completion() {
        let frames: Vec<Result<bytes::Bytes, anyhow::Error>> = vec![
            Ok(bytes::Bytes::from_static(
                br#"data: {"id":"c1","model":"m","created":5,"choices":[{"index":0,"delta":{"role":"assistant"}}]}

"#,
            )),
            Ok(bytes::Bytes::from_static(
                br#"data: {"choices":[{"index":0,"delta":{"content":"he"}}]}

"#,
            )),
            Ok(bytes::Bytes::from_static(
                br#"data: {"choices":[{"index":0,"delta":{"content":"llo"},"finish_reason":"stop"}]}

"#,
            )),
            Ok(bytes::Bytes::from_static(
                br#"data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2}}

"#,
            )),
            Ok(bytes::Bytes::from_static(b"data: [DONE]\n\n")),
        ];
        let stream: crate::provider::SseStream = Box::pin(futures_util::stream::iter(frames));
        let out = block_on(collect_completion(stream)).unwrap();
        assert_eq!(out["choices"][0]["message"]["content"], json!("hello"));
        assert_eq!(out["choices"][0]["finish_reason"], json!("stop"));
        assert_eq!(out["usage"]["completion_tokens"], json!(2));
    }

    #[test]
    fn plain_json_body_passes_through() {
        let stream: crate::provider::SseStream = Box::pin(futures_util::stream::iter(vec![
            Ok::<_, anyhow::Error>(bytes::Bytes::from_static(
                br#"{"choices":[{"message":{"content":"x"}}]}"#,
            )),
        ]));
        let out = block_on(collect_completion(stream)).unwrap();
        assert_eq!(out["choices"][0]["message"]["content"], json!("x"));
    }

    #[test]
    fn limit_error_maps_to_minus_32005() {
        let f = limit_or(-32000);
        let (code, msg) = f(anyhow::anyhow!("plugin in-flight cap reached (4 active + 16 queued)"));
        assert_eq!(code, JSONRPC_LIMIT);
        assert!(msg.contains("in-flight cap"));
        let (code, _) = f(anyhow::anyhow!("something else"));
        assert_eq!(code, -32000);
    }
}
