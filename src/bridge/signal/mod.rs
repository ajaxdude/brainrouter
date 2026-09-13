//! Signal transport — keeps a persistent `signal-cli daemon` connection,
//! dispatches incoming messages to OMP, and replies over JSON-RPC.

use std::collections::HashMap;
use std::ffi::OsString;
use std::os::unix::fs::FileTypeExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::UnixStream;
use tokio::process::{Child, ChildStderr, Command};
use tokio::sync::{mpsc, oneshot, Mutex};
use tracing::{debug, info, warn};

use crate::bridge::core::{invoke_omp, load_model_aliases, resolve_model};
use crate::bridge::persist::{
    load_channel_models, load_sessions, save_channel_models, save_sessions,
};

const TRANSPORT: &str = "signal";
const RPC_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const DAEMON_START_TIMEOUT: Duration = Duration::from_secs(30);
const DAEMON_RESTART_DELAY: Duration = Duration::from_secs(5);
const DAEMON_STOP_TIMEOUT: Duration = Duration::from_secs(5);

// Persistence is sequenced and coalesced by bridge::persist.
fn spawn_save_sessions(sessions: &HashMap<String, String>) {
    save_sessions(TRANSPORT, sessions);
}
fn spawn_save_channel_models(models: &HashMap<String, String>) {
    save_channel_models(TRANSPORT, models);
}

type SessionMap = Arc<Mutex<HashMap<String, String>>>;
type ModelMap = Arc<Mutex<HashMap<String, String>>>;

fn signal_message_allowed(configured_group: Option<&str>, reply_to: &str, is_group: bool) -> bool {
    configured_group.is_none_or(|group| is_group && reply_to == group)
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct SignalConfig {
    pub enabled: Option<bool>,
    /// Phone number in E.164 format (e.g. +1234567890).
    pub account: Option<String>,
    pub group_id: Option<String>,
    pub prefix: Option<String>,
    pub storage_path: Option<String>,
    pub llama_swap_url: Option<String>,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            enabled: Some(false),
            account: None,
            group_id: None,
            prefix: Some("!".to_string()),
            storage_path: None,
            llama_swap_url: Some("http://localhost:8081".to_string()),
        }
    }
}

impl SignalConfig {
    pub fn is_enabled(&self) -> bool {
        self.enabled.unwrap_or(false)
    }

    fn account_or_err(&self) -> anyhow::Result<&str> {
        self.account
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow::anyhow!("signal account (phone number) is required"))
    }

    fn prefix_str(&self) -> &str {
        self.prefix.as_deref().unwrap_or("!")
    }

    fn llama_url(&self) -> &str {
        self.llama_swap_url
            .as_deref()
            .unwrap_or("http://localhost:8081")
    }
}

// ---------------------------------------------------------------------------
// signal-cli helpers
// ---------------------------------------------------------------------------

struct RpcRequest {
    method: String,
    params: Value,
    response: oneshot::Sender<Result<Value, String>>,
}

#[derive(Clone)]
struct SignalRpcClient {
    requests: mpsc::Sender<RpcRequest>,
}

impl SignalRpcClient {
    async fn request(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        let (response_tx, response_rx) = oneshot::channel();
        self.requests
            .send(RpcRequest {
                method: method.to_string(),
                params,
                response: response_tx,
            })
            .await
            .map_err(|_| anyhow!("signal-cli daemon connection is unavailable"))?;

        match tokio::time::timeout(RPC_REQUEST_TIMEOUT, response_rx).await {
            Ok(Ok(Ok(result))) => Ok(result),
            Ok(Ok(Err(error))) => Err(anyhow!("signal-cli {method} failed: {error}")),
            Ok(Err(_)) => Err(anyhow!(
                "signal-cli daemon disconnected while handling {method}"
            )),
            Err(_) => Err(anyhow!(
                "signal-cli {method} timed out after {} seconds",
                RPC_REQUEST_TIMEOUT.as_secs()
            )),
        }
    }
}

fn recipient_params(identifier: &str, is_group: bool) -> serde_json::Map<String, Value> {
    let mut params = serde_json::Map::new();
    if is_group {
        params.insert("groupId".to_string(), json!(identifier));
    } else {
        params.insert("recipient".to_string(), json!([identifier]));
    }
    params
}

async fn send_typing(
    client: &SignalRpcClient,
    identifier: &str,
    is_group: bool,
    stop: bool,
) -> anyhow::Result<()> {
    let mut params = recipient_params(identifier, is_group);
    if stop {
        params.insert("stop".to_string(), Value::Bool(true));
    }
    client
        .request("sendTyping", Value::Object(params))
        .await
        .map(|_| ())
}

async fn send_message_to(
    client: &SignalRpcClient,
    identifier: &str,
    is_group: bool,
    content: &str,
) -> anyhow::Result<()> {
    let mut params = recipient_params(identifier, is_group);
    params.insert("message".to_string(), json!(content));
    client
        .request("send", Value::Object(params))
        .await
        .map(|_| ())
}

/// Split long text into Signal messages (max 4 000 chars each).
async fn send_chunked(
    client: &SignalRpcClient,
    identifier: &str,
    is_group: bool,
    text: &str,
) -> anyhow::Result<()> {
    const MAX_CHARS: usize = 4_000;
    let mut rest = text;
    while !rest.is_empty() {
        let split = if rest.chars().count() <= MAX_CHARS {
            rest.len()
        } else {
            rest.char_indices()
                .nth(MAX_CHARS)
                .map(|(i, _)| i)
                .unwrap_or(rest.len())
        };
        let (chunk, remainder) = rest.split_at(split);
        send_message_to(client, identifier, is_group, chunk).await?;
        rest = remainder;
    }
    Ok(())
}

#[cfg(test)]
fn parse_received_messages(json_output: &[u8]) -> anyhow::Result<Vec<(String, String, bool)>> {
    let mut messages = Vec::new();
    let content = String::from_utf8_lossy(json_output);

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(val) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if let Some(message) = parse_received_message(&val) {
            messages.push(message);
        }
    }
    Ok(messages)
}

fn parse_received_message(value: &Value) -> Option<(String, String, bool)> {
    let payload = if value.get("method").and_then(Value::as_str) == Some("receive") {
        let params = value.get("params")?;
        params.get("result").unwrap_or(params)
    } else {
        value
    };
    let envelope = payload.get("envelope")?;
    let data_msg = envelope.get("dataMessage")?;
    let message = data_msg.get("message").and_then(Value::as_str)?;
    if message.is_empty() {
        return None;
    }

    let source_number = envelope
        .get("sourceNumber")
        .and_then(Value::as_str)?
        .to_string();
    if source_number.is_empty() {
        return None;
    }

    let (reply_to, is_group) = if let Some(group_info) = data_msg.get("groupInfo") {
        let group_id = group_info
            .get("groupId")
            .and_then(Value::as_str)
            .unwrap_or(&source_number)
            .to_string();
        (group_id, true)
    } else {
        (source_number, false)
    };

    Some((reply_to, message.to_string(), is_group))
}

async fn run_rpc_connection(
    stream: UnixStream,
    mut requests: mpsc::Receiver<RpcRequest>,
    incoming: mpsc::UnboundedSender<(String, String, bool)>,
) -> anyhow::Result<()> {
    let (read_half, write_half) = stream.into_split();
    let mut lines = BufReader::new(read_half).lines();
    let mut writer = BufWriter::new(write_half);
    let mut pending = HashMap::new();
    let mut next_id = 1_u64;

    loop {
        tokio::select! {
            request = requests.recv() => {
                let Some(request) = request else {
                    anyhow::bail!("Signal JSON-RPC request channel closed");
                };
                let id = next_id;
                next_id = next_id.wrapping_add(1).max(1);
                let frame = json!({
                    "jsonrpc": "2.0",
                    "method": request.method,
                    "params": request.params,
                    "id": id,
                });
                pending.insert(id, request.response);
                writer
                    .write_all(serde_json::to_string(&frame)?.as_bytes())
                    .await
                    .context("failed to write Signal JSON-RPC request")?;
                writer
                    .write_all(b"\n")
                    .await
                    .context("failed to terminate Signal JSON-RPC request")?;
                writer
                    .flush()
                    .await
                    .context("failed to flush Signal JSON-RPC request")?;
            }
            line = lines.next_line() => {
                let Some(line) = line.context("failed to read Signal JSON-RPC response")? else {
                    anyhow::bail!("signal-cli daemon closed its JSON-RPC socket");
                };
                let value: Value = match serde_json::from_str(&line) {
                    Ok(value) => value,
                    Err(error) => {
                        warn!("Ignoring malformed signal-cli JSON-RPC output: {error}");
                        continue;
                    }
                };
                if let Some(id) = value.get("id").and_then(Value::as_u64) {
                    if let Some(response) = pending.remove(&id) {
                        let result = if let Some(error) = value.get("error") {
                            Err(format_rpc_error(error))
                        } else {
                            Ok(value.get("result").cloned().unwrap_or(Value::Null))
                        };
                        let _ = response.send(result);
                    } else {
                        debug!(id, "Ignoring response for unknown Signal JSON-RPC request");
                    }
                } else if value.get("method").and_then(Value::as_str) == Some("receive") {
                    if let Some(message) = parse_received_message(&value) {
                        incoming
                            .send(message)
                            .map_err(|_| anyhow!("Signal message dispatcher stopped"))?;
                    }
                }
            }
        }
    }
}

fn format_rpc_error(error: &Value) -> String {
    let code = error.get("code").and_then(Value::as_i64);
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("unknown JSON-RPC error");
    match code {
        Some(code) => format!("{message} (code {code})"),
        None => message.to_string(),
    }
}

fn expand_home(path: &str) -> PathBuf {
    if let Some(suffix) = path.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(suffix);
        }
    }
    PathBuf::from(path)
}

fn signal_daemon_args(
    account: &str,
    storage_path: Option<&Path>,
    socket_path: &Path,
) -> Vec<OsString> {
    let mut args = vec![OsString::from("--scrub-log")];
    if let Some(storage_path) = storage_path {
        args.push(OsString::from("--config"));
        args.push(storage_path.as_os_str().to_owned());
    }
    args.extend([
        OsString::from("-a"),
        OsString::from(account),
        OsString::from("daemon"),
        OsString::from("--socket"),
        socket_path.as_os_str().to_owned(),
        OsString::from("--no-receive-stdout"),
        OsString::from("--receive-mode=on-connection"),
    ]);
    args
}

fn signal_daemon_socket_path() -> PathBuf {
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .unwrap_or_else(std::env::temp_dir);
    base.join(format!("brainrouter-signal-{}.sock", std::process::id()))
}

fn remove_stale_socket(path: &Path) -> anyhow::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_socket() => std::fs::remove_file(path)
            .with_context(|| format!("failed to remove stale Signal socket {}", path.display())),
        Ok(_) => anyhow::bail!(
            "refusing to replace non-socket Signal daemon path {}",
            path.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error)
            .with_context(|| format!("failed to inspect Signal socket {}", path.display())),
    }
}

fn spawn_signal_daemon(
    account: &str,
    storage_path: Option<&Path>,
    socket_path: &Path,
) -> anyhow::Result<Child> {
    let mut command = Command::new("signal-cli");
    command
        .args(signal_daemon_args(account, storage_path, socket_path))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    command.process_group(0);
    command.spawn().context("failed to start signal-cli daemon")
}

async fn connect_signal_daemon(
    child: &mut Child,
    socket_path: &Path,
) -> anyhow::Result<UnixStream> {
    let deadline = tokio::time::Instant::now() + DAEMON_START_TIMEOUT;
    loop {
        match UnixStream::connect(socket_path).await {
            Ok(stream) => return Ok(stream),
            Err(error) if tokio::time::Instant::now() < deadline => {
                if let Some(status) = child
                    .try_wait()
                    .context("failed to inspect signal-cli daemon")?
                {
                    anyhow::bail!("signal-cli daemon exited during startup with {status}");
                }
                debug!("Waiting for signal-cli daemon socket: {error}");
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "signal-cli daemon socket {} was not ready within {} seconds",
                        socket_path.display(),
                        DAEMON_START_TIMEOUT.as_secs()
                    )
                });
            }
        }
    }
}

async fn read_daemon_stderr(stderr: ChildStderr) -> String {
    const RETAIN_BYTES: usize = 8 * 1024;
    let mut lines = BufReader::new(stderr).lines();
    let mut retained = String::new();
    while let Ok(Some(line)) = lines.next_line().await {
        debug!("signal-cli daemon: {line}");
        retained.push_str(&line);
        retained.push('\n');
        if retained.len() > RETAIN_BYTES {
            let overflow = retained.len() - RETAIN_BYTES;
            let split = retained
                .char_indices()
                .find_map(|(index, _)| (index >= overflow).then_some(index))
                .unwrap_or(retained.len());
            retained.drain(..split);
        }
    }
    retained
}

fn signal_process_group(pid: Option<u32>, signal: i32) {
    if let Some(pid) = pid {
        unsafe {
            libc::kill(-(pid as i32), signal);
        }
    }
}

async fn stop_signal_daemon(child: &mut Child, pid: Option<u32>) {
    if child.try_wait().ok().flatten().is_some() {
        return;
    }
    signal_process_group(pid, libc::SIGTERM);
    if tokio::time::timeout(DAEMON_STOP_TIMEOUT, child.wait())
        .await
        .is_err()
    {
        signal_process_group(pid, libc::SIGKILL);
        let _ = child.wait().await;
    }
}

struct SignalProcessGroupGuard {
    pid: Option<u32>,
}

impl SignalProcessGroupGuard {
    fn new(pid: Option<u32>) -> Self {
        Self { pid }
    }

    fn disarm(&mut self) {
        self.pid = None;
    }
}

impl Drop for SignalProcessGroupGuard {
    fn drop(&mut self) {
        signal_process_group(self.pid, libc::SIGKILL);
    }
}

async fn list_llama_models(base_url: &str) -> anyhow::Result<Vec<String>> {
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
    let body: serde_json::Value = reqwest::get(&url).await?.json().await?;
    Ok(body
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    m.get("id")
                        .and_then(|id| id.as_str())
                        .map(|s| s.to_string())
                })
                .collect()
        })
        .unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Message handler
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn handle_message(
    sender: &str,
    content: &str,
    is_group: bool,
    sessions: SessionMap,
    channel_models: ModelMap,
    signal: &SignalRpcClient,
    account: &str,
    prefix: &str,
    llama_swap_url: &str,
    omp_path: &str,
    work_dir: &str,
    timeout_secs: u64,
    default_model: &str,
    model_aliases: &HashMap<String, String>,
    manager: &super::BridgeManager,
) {
    // Don't respond to our own messages.
    if sender == account {
        return;
    }

    let text = content.trim();

    // !br ping
    if text == format!("{prefix}ping") || text == format!("{prefix}br ping") {
        let _ = send_message_to(signal, sender, is_group, "Pong! (Signal)").await;
        return;
    }

    // !br help / ?
    if text == format!("{prefix}br help") || text == format!("{prefix}br ?") {
        let help = format!(
            "brainrouter commands:\n\n\
            {prefix}br reset      \u{2014} Clear session\n\
            {prefix}br status     \u{2014} Show current model\n\
            {prefix}br auto|local|cloud \u{2014} Set routing mode\n\
            {prefix}br <model>    \u{2014} Use specific local model\n\
            {prefix}br list       \u{2014} List available models\n\
            {prefix}br review [mode] \u{2014} Set/show review mode\n\
            {prefix}br ? / help   \u{2014} Show this help\n\n\
            Tip: Use --model <name> at the start of any message for a one-time override."
        );
        let _ = send_message_to(signal, sender, is_group, &help).await;
        return;
    }

    // !br reset
    if text == format!("{prefix}br reset") {
        let had_session = {
            let mut sessions = sessions.lock().await;
            let removed = sessions.remove(sender).is_some();
            if removed {
                spawn_save_sessions(&sessions);
            }
            removed
        };
        let reply = if had_session {
            "Session cleared. Starting fresh on your next message."
        } else {
            "No active session for this contact/group."
        };
        let _ = send_message_to(signal, sender, is_group, reply).await;
        return;
    }

    // !br status \ show current model
    if text == format!("{prefix}br status") {
        let prefs = channel_models.lock().await;
        let reply = match prefs.get(sender) {
            Some(m) => format!("Current model: {m}"),
            None => format!("No model set (using default: {default_model})."),
        };
        let _ = send_message_to(signal, sender, is_group, &reply).await;
        return;
    }

    // !br model <name> \ set sticky model (clears session)
    if let Some(rest) = text.strip_prefix(&format!("{prefix}br model ")) {
        let name = rest.trim();
        if name.is_empty() {
            let _ = send_message_to(signal, sender, is_group, "Usage: !br model <name>").await;
            return;
        }
        let resolved = resolve_model(name, model_aliases);
        {
            let mut prefs = channel_models.lock().await;
            prefs.insert(sender.to_string(), resolved.clone());
            spawn_save_channel_models(&prefs);
        }
        {
            let mut sessions = sessions.lock().await;
            if sessions.remove(sender).is_some() {
                spawn_save_sessions(&sessions);
            }
        }
        let _ = send_message_to(
            signal,
            sender,
            is_group,
            &format!("Model set to: {resolved}\nSession cleared."),
        )
        .await;
        return;
    }

    // !br list
    if text == format!("{prefix}br list") {
        let reply = match list_llama_models(llama_swap_url).await {
            Ok(models) if models.is_empty() => "No models found.".to_string(),
            Ok(models) => format!("Available models:\n{}", models.join("\n")),
            Err(e) => format!("Failed to list models: {e}"),
        };
        let _ = send_message_to(signal, sender, is_group, &reply).await;
        return;
    }

    // !br auto | !br local | !br cloud
    for mode in ["auto", "local", "cloud"] {
        if text == format!("{prefix}br {mode}") {
            let model_id = format!("brainrouter/{mode}");
            {
                let mut prefs = channel_models.lock().await;
                prefs.insert(sender.to_string(), model_id.clone());
                spawn_save_channel_models(&prefs);
            }
            {
                let mut sessions = sessions.lock().await;
                if sessions.remove(sender).is_some() {
                    spawn_save_sessions(&sessions);
                }
            }
            let _ = send_message_to(
                signal,
                sender,
                is_group,
                &format!("Routing set to {model_id}. Session cleared."),
            )
            .await;
            return;
        }
    }

    // !br review [auto|local|cloud]
    if let Some(rest) = text.strip_prefix(&format!("{prefix}br review ")) {
        let mode = rest.trim();
        if mode != "auto" && mode != "local" && mode != "cloud" {
            let _ = send_message_to(
                signal,
                sender,
                is_group,
                "Usage: !br review auto|local|cloud",
            )
            .await;
            return;
        }
        let body = serde_json::json!({"forced_mode": mode, "max_iterations": 5});
        match reqwest::Client::new()
            .post("http://127.0.0.1:9099/api/review-config")
            .json(&body)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                let _ = send_message_to(
                    signal,
                    sender,
                    is_group,
                    &format!("Review mode set to {mode}."),
                )
                .await;
            }
            _ => {
                let _ =
                    send_message_to(signal, sender, is_group, "Failed to set review mode.").await;
            }
        }
        return;
    }
    if text == format!("{prefix}br review") {
        match reqwest::get("http://127.0.0.1:9099/api/review-config").await {
            Ok(resp) => {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    let mode = json
                        .get("forced_mode")
                        .and_then(|v| v.as_str())
                        .unwrap_or("auto");
                    let _ =
                        send_message_to(signal, sender, is_group, &format!("Review mode: {mode}"))
                            .await;
                }
            }
            _ => {
                let _ =
                    send_message_to(signal, sender, is_group, "Failed to get review config.").await;
            }
        }
        return;
    }

    // !br <model-name> — set specific llama-swap model
    if let Some(rest) = text.strip_prefix(&format!("{prefix}br ")) {
        let arg = rest.trim();
        let reserved = [
            "ping", "reset", "help", "?", "status", "list", "auto", "local", "cloud", "review",
            "model",
        ];
        if !reserved.contains(&arg)
            && !arg.contains(' ')
            && (arg.contains('-') || arg.contains('.'))
        {
            let model_id = format!("brainrouter/{arg}");
            {
                let mut prefs = channel_models.lock().await;
                prefs.insert(sender.to_string(), model_id.clone());
                spawn_save_channel_models(&prefs);
            }
            {
                let mut sessions = sessions.lock().await;
                if sessions.remove(sender).is_some() {
                    spawn_save_sessions(&sessions);
                }
            }
            let _ = send_message_to(
                signal,
                sender,
                is_group,
                &format!("Model set to {model_id}. Session cleared."),
            )
            .await;
            return;
        }
    }

    // Strip !br prefix if present, leaving the bare query.
    let mut query_text = text;
    if let Some(rest) = text.strip_prefix(&format!("{prefix}br")) {
        query_text = rest.trim();
    } else if let Some(rest) = text.strip_prefix(account) {
        query_text = rest.trim();
    }

    if query_text.is_empty() {
        let _ = send_message_to(signal, sender, is_group, "(No query provided)").await;
        return;
    }

    // Inline --model override
    let (model_owned, actual_query) = {
        let mut q = query_text;
        let mut m: Option<String> = None;
        if q.starts_with("--model ") {
            let parts: Vec<&str> = q.splitn(3, ' ').collect();
            if parts.len() >= 3 {
                m = Some(resolve_model(parts[1], model_aliases));
                q = parts[2];
            }
        }
        (m, q.to_string())
    };

    // Resolve model: inline override > sticky per-sender preference > default_model
    let effective_model: Option<String> = if model_owned.is_some() {
        model_owned.clone()
    } else {
        let prefs = channel_models.lock().await;
        prefs.get(sender).cloned()
    }
    .or_else(|| Some(default_model.to_string()));

    let session_id: Option<String> = {
        let sessions = sessions.lock().await;
        sessions.get(sender).cloned()
    };

    let _ = send_typing(signal, sender, is_group, false).await;

    let result = invoke_omp(
        omp_path,
        work_dir,
        effective_model.as_deref(),
        &actual_query,
        session_id.as_deref(),
        timeout_secs,
    )
    .await;

    let _ = send_typing(signal, sender, is_group, true).await;

    match result {
        Ok((response, new_session, model_info)) => {
            if let Some(sid) = new_session {
                let mut sessions = sessions.lock().await;
                sessions.insert(sender.to_string(), sid);
                spawn_save_sessions(&sessions);
            }
            let text = if response.is_empty() {
                "(OMP returned an empty response)".to_string()
            } else if let (Some(_), Some((provider, mdl))) = (model_owned.as_deref(), model_info) {
                format!("{response}\n\n_({provider}/{mdl})_")
            } else {
                response
            };
            if let Err(error) = send_chunked(signal, sender, is_group, &text).await {
                warn!("Failed to send Signal response: {error}");
            }
            manager.record_signal_message();
        }
        Err(e) => {
            warn!("OMP invocation failed: {e}");
            let _ = send_message_to(signal, sender, is_group, &format!("OMP error: {e}")).await;
        }
    }
}

// ---------------------------------------------------------------------------
// SignalService
// ---------------------------------------------------------------------------

pub struct Contact {
    pub identifier: String,
    pub name: String,
    pub is_group: bool,
}

pub struct SignalService {
    account: String,
    storage_path: Option<PathBuf>,
    group_id: Option<String>,
    prefix: String,
    llama_swap_url: String,
    omp_path: String,
    work_dir: String,
    timeout_secs: u64,
    default_model: String,
    sessions: SessionMap,
    model_aliases: HashMap<String, String>,
    channel_models: ModelMap,
    manager: Arc<super::BridgeManager>,
    rpc_client: Arc<Mutex<Option<SignalRpcClient>>>,
}

impl SignalService {
    pub fn new(
        signal_config: &SignalConfig,
        omp_path: &str,
        work_dir: &str,
        aliases_config: &str,
        timeout_secs: u64,
        default_model: &str,
        manager: Arc<super::BridgeManager>,
    ) -> anyhow::Result<Self> {
        let account = signal_config.account_or_err()?.to_string();
        let group_id = signal_config
            .group_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        let prefix = signal_config.prefix_str().to_string();
        let llama_swap_url = signal_config.llama_url().to_string();
        let storage_path = signal_config
            .storage_path
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(expand_home);

        if let Some(storage) = storage_path.as_deref() {
            std::fs::create_dir_all(storage).with_context(|| {
                format!(
                    "failed to create signal-cli storage directory {}",
                    storage.display()
                )
            })?;
        }

        let sessions: SessionMap = Arc::new(Mutex::new(load_sessions(TRANSPORT)));
        let model_aliases = load_model_aliases(aliases_config);
        let channel_models: ModelMap = Arc::new(Mutex::new(load_channel_models(TRANSPORT)));

        info!("Signal service ready");
        Ok(Self {
            account,
            storage_path,
            group_id,
            prefix,
            llama_swap_url,
            omp_path: omp_path.to_string(),
            work_dir: work_dir.to_string(),
            timeout_secs,
            default_model: default_model.to_string(),
            sessions,
            model_aliases,
            channel_models,
            manager,
            rpc_client: Arc::new(Mutex::new(None)),
        })
    }

    pub async fn send_message(&self, identifier: &str, content: &str) -> anyhow::Result<()> {
        let client = self
            .rpc_client
            .lock()
            .await
            .clone()
            .ok_or_else(|| anyhow!("signal-cli daemon is not connected"))?;
        send_message_to(&client, identifier, !identifier.starts_with('+'), content).await
    }

    pub async fn list_contacts(&self) -> anyhow::Result<Vec<Contact>> {
        warn!("Signal contact listing not fully implemented via signal-cli");
        Ok(Vec::new())
    }

    fn spawn_message_dispatcher(
        &self,
        client: SignalRpcClient,
        mut messages: mpsc::UnboundedReceiver<(String, String, bool)>,
    ) -> tokio::task::JoinHandle<anyhow::Result<()>> {
        let account = self.account.clone();
        let group_id = self.group_id.clone();
        let sessions = self.sessions.clone();
        let channel_models = self.channel_models.clone();
        let prefix = self.prefix.clone();
        let llama_swap_url = self.llama_swap_url.clone();
        let omp_path = self.omp_path.clone();
        let work_dir = self.work_dir.clone();
        let timeout_secs = self.timeout_secs;
        let default_model = self.default_model.clone();
        let model_aliases = self.model_aliases.clone();
        let manager = self.manager.clone();

        tokio::spawn(async move {
            while let Some((reply_to, content, is_group)) = messages.recv().await {
                if !manager
                    .signal_enabled
                    .load(std::sync::atomic::Ordering::Relaxed)
                {
                    debug!("Signal bridge paused; skipping incoming message");
                    continue;
                }
                if !signal_message_allowed(group_id.as_deref(), &reply_to, is_group) {
                    debug!(
                        reply_to,
                        is_group, "Ignoring Signal message outside the configured group"
                    );
                    continue;
                }
                handle_message(
                    &reply_to,
                    &content,
                    is_group,
                    sessions.clone(),
                    channel_models.clone(),
                    &client,
                    &account,
                    &prefix,
                    &llama_swap_url,
                    &omp_path,
                    &work_dir,
                    timeout_secs,
                    &default_model,
                    &model_aliases,
                    &manager,
                )
                .await;
            }
            anyhow::bail!("Signal message stream closed")
        })
    }

    async fn run_daemon_session(&self) -> anyhow::Result<()> {
        let socket_path = signal_daemon_socket_path();
        remove_stale_socket(&socket_path)?;

        let mut child =
            spawn_signal_daemon(&self.account, self.storage_path.as_deref(), &socket_path)?;
        let pid = child.id();
        let mut process_group = SignalProcessGroupGuard::new(pid);
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("signal-cli daemon stderr was not captured"))?;
        let stderr_task = tokio::spawn(read_daemon_stderr(stderr));

        let stream = match connect_signal_daemon(&mut child, &socket_path).await {
            Ok(stream) => stream,
            Err(error) => {
                stop_signal_daemon(&mut child, pid).await;
                process_group.disarm();
                let stderr = stderr_task.await.unwrap_or_default();
                let _ = remove_stale_socket(&socket_path);
                if stderr.trim().is_empty() {
                    return Err(error);
                }
                return Err(error.context(format!("signal-cli daemon stderr: {}", stderr.trim())));
            }
        };

        let (request_tx, request_rx) = mpsc::channel(32);
        let client = SignalRpcClient {
            requests: request_tx,
        };
        let (message_tx, message_rx) = mpsc::unbounded_channel();
        let mut connection_task = tokio::spawn(run_rpc_connection(stream, request_rx, message_tx));
        let mut dispatcher_task = self.spawn_message_dispatcher(client.clone(), message_rx);

        *self.rpc_client.lock().await = Some(client);
        self.manager.clear_signal_error();
        self.manager.set_signal_connected(true);
        info!(
            socket = %socket_path.display(),
            "Signal daemon connected; receiving messages continuously"
        );

        let failure = tokio::select! {
            status = child.wait() => {
                match status {
                    Ok(status) => anyhow!("signal-cli daemon exited with {status}"),
                    Err(error) => anyhow!("failed waiting for signal-cli daemon: {error}"),
                }
            }
            result = &mut connection_task => {
                match result {
                    Ok(Ok(())) => anyhow!("Signal JSON-RPC connection ended unexpectedly"),
                    Ok(Err(error)) => error,
                    Err(error) => anyhow!("Signal JSON-RPC task failed: {error}"),
                }
            }
            result = &mut dispatcher_task => {
                match result {
                    Ok(Ok(())) => anyhow!("Signal message dispatcher ended unexpectedly"),
                    Ok(Err(error)) => error,
                    Err(error) => anyhow!("Signal message dispatcher failed: {error}"),
                }
            }
        };

        self.manager.set_signal_connected(false);
        *self.rpc_client.lock().await = None;
        connection_task.abort();
        dispatcher_task.abort();
        stop_signal_daemon(&mut child, pid).await;
        process_group.disarm();
        let stderr = stderr_task.await.unwrap_or_default();
        if let Err(error) = remove_stale_socket(&socket_path) {
            warn!("{error}");
        }

        if stderr.trim().is_empty() {
            Err(failure)
        } else {
            Err(failure.context(format!("signal-cli daemon stderr: {}", stderr.trim())))
        }
    }

    async fn run(&self) -> anyhow::Result<()> {
        loop {
            if let Err(error) = self.run_daemon_session().await {
                self.manager.set_signal_connected(false);
                self.manager.set_signal_error(error.to_string());
                warn!(
                    "Signal daemon session failed: {error}; retrying in {} seconds",
                    DAEMON_RESTART_DELAY.as_secs()
                );
            }
            tokio::time::sleep(DAEMON_RESTART_DELAY).await;
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Start the Signal bridge transport. Blocks forever.
pub async fn start(
    signal_config: &SignalConfig,
    omp_path: &str,
    work_dir: &str,
    aliases_config: &str,
    timeout_secs: u64,
    default_model: &str,
    manager: Arc<super::BridgeManager>,
) -> anyhow::Result<()> {
    let _ = signal_config.account_or_err()?;

    let service = SignalService::new(
        signal_config,
        omp_path,
        work_dir,
        aliases_config,
        timeout_secs,
        default_model,
        manager,
    )?;

    service.run().await
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_received_messages_dm() {
        let json = r#"{"envelope":{"source":"+1555000001","sourceNumber":"+1555000001","sourceUuid":"abc","sourceName":"Alice","sourceDevice":1,"timestamp":1000,"dataMessage":{"timestamp":1000,"message":"!ping","expiresInSeconds":0,"viewOnce":false}}}"#;
        let msgs = parse_received_messages(json.as_bytes()).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].0, "+1555000001");
        assert_eq!(msgs[0].1, "!ping");
        assert!(!msgs[0].2);
    }

    #[test]
    fn test_parse_received_messages_group() {
        let json = r#"{"envelope":{"source":"+1555000001","sourceNumber":"+1555000001","sourceUuid":"abc","sourceName":"Alice","sourceDevice":1,"timestamp":1000,"dataMessage":{"timestamp":1000,"message":"!br hello","groupInfo":{"groupId":"AAAA=","type":"DELIVER"}}}}"#;
        let msgs = parse_received_messages(json.as_bytes()).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].0, "AAAA=");
        assert_eq!(msgs[0].1, "!br hello");
        assert!(msgs[0].2);
    }

    #[test]
    fn test_parse_json_rpc_receive_notification() {
        let notification = json!({
            "jsonrpc": "2.0",
            "method": "receive",
            "params": {
                "envelope": {
                    "sourceNumber": "+1555000001",
                    "dataMessage": {
                        "message": "!br hello",
                        "groupInfo": {"groupId": "AAAA="}
                    }
                }
            }
        });
        let message = parse_received_message(&notification).unwrap();
        assert_eq!(
            message,
            ("AAAA=".to_string(), "!br hello".to_string(), true)
        );
    }

    #[test]
    fn test_parse_subscription_receive_notification() {
        let notification = json!({
            "jsonrpc": "2.0",
            "method": "receive",
            "params": {
                "subscription": 0,
                "result": {
                    "account": "+1555000999",
                    "envelope": {
                        "sourceNumber": "+1555000001",
                        "dataMessage": {"message": "!ping"}
                    }
                }
            }
        });
        let message = parse_received_message(&notification).unwrap();
        assert_eq!(
            message,
            ("+1555000001".to_string(), "!ping".to_string(), false)
        );
    }

    #[test]
    fn daemon_arguments_enable_persistent_socket_mode() {
        let args = signal_daemon_args(
            "+1555000999",
            Some(Path::new("/var/lib/signal")),
            Path::new("/run/user/1000/brainrouter-signal.sock"),
        );
        let args: Vec<_> = args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            args,
            vec![
                "--scrub-log",
                "--config",
                "/var/lib/signal",
                "-a",
                "+1555000999",
                "daemon",
                "--socket",
                "/run/user/1000/brainrouter-signal.sock",
                "--no-receive-stdout",
                "--receive-mode=on-connection",
            ]
        );
    }

    #[test]
    fn rpc_recipient_parameters_distinguish_groups() {
        assert_eq!(
            Value::Object(recipient_params("+1555000001", false)),
            json!({"recipient": ["+1555000001"]})
        );
        assert_eq!(
            Value::Object(recipient_params("AAAA=", true)),
            json!({"groupId": "AAAA="})
        );
    }

    #[tokio::test]
    async fn json_rpc_connection_multiplexes_responses_and_notifications() {
        let (client_stream, server_stream) = UnixStream::pair().unwrap();
        let (request_tx, request_rx) = mpsc::channel(4);
        let client = SignalRpcClient {
            requests: request_tx,
        };
        let (message_tx, mut message_rx) = mpsc::unbounded_channel();
        let connection = tokio::spawn(run_rpc_connection(client_stream, request_rx, message_tx));

        let (server_read, mut server_write) = server_stream.into_split();
        let mut server_lines = BufReader::new(server_read).lines();
        let request = tokio::spawn(async move {
            client
                .request(
                    "send",
                    json!({"recipient": ["+1555000001"], "message": "hi"}),
                )
                .await
        });

        let request_frame: Value =
            serde_json::from_str(&server_lines.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(request_frame["method"], "send");
        assert_eq!(request_frame["params"]["message"], "hi");
        let id = request_frame["id"].as_u64().unwrap();

        let notification = json!({
            "jsonrpc": "2.0",
            "method": "receive",
            "params": {
                "envelope": {
                    "sourceNumber": "+1555000002",
                    "dataMessage": {"message": "!br status"}
                }
            }
        });
        server_write
            .write_all(format!("{notification}\n").as_bytes())
            .await
            .unwrap();
        server_write
            .write_all(
                format!(
                    "{}\n",
                    json!({"jsonrpc": "2.0", "result": {"timestamp": 1234}, "id": id})
                )
                .as_bytes(),
            )
            .await
            .unwrap();

        assert_eq!(
            message_rx.recv().await.unwrap(),
            ("+1555000002".to_string(), "!br status".to_string(), false)
        );
        assert_eq!(request.await.unwrap().unwrap()["timestamp"], 1234);
        connection.abort();
    }

    #[test]
    fn test_parse_received_messages_skips_empty_body() {
        let json = r#"{"envelope":{"source":"+1555000001","sourceNumber":"+1555000001","sourceUuid":"abc","sourceName":"Alice","sourceDevice":1,"timestamp":1000,"dataMessage":{"timestamp":1000,"message":"","expiresInSeconds":0,"viewOnce":false}}}"#;
        let msgs = parse_received_messages(json.as_bytes()).unwrap();
        assert_eq!(msgs.len(), 0);
    }

    #[test]
    fn test_parse_received_messages_skips_non_data() {
        let json = r#"{"envelope":{"source":"+1555000001","sourceNumber":"+1555000001","sourceUuid":"abc","sourceName":"Alice","sourceDevice":1,"timestamp":1000,"typingMessage":{"action":"STARTED","timestamp":1000}}}"#;
        let msgs = parse_received_messages(json.as_bytes()).unwrap();
        assert_eq!(msgs.len(), 0);
    }

    #[test]
    fn configured_group_rejects_other_groups_and_direct_messages() {
        assert!(signal_message_allowed(Some("AAAA="), "AAAA=", true));
        assert!(!signal_message_allowed(Some("AAAA="), "BBBB=", true));
        assert!(!signal_message_allowed(Some("AAAA="), "+1555000001", false));
        assert!(signal_message_allowed(None, "+1555000001", false));
    }
}
