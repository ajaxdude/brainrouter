//! Signal transport — receives messages via `signal-cli receive` polling,
//! dispatches queries to OMP, and replies via `signal-cli send`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::bridge::core::{invoke_omp, load_model_aliases, resolve_model};
use crate::bridge::persist::{
    load_channel_models, load_sessions, save_channel_models, save_sessions,
};

const TRANSPORT: &str = "signal";

// Non-blocking persistence — fire-and-forget on a background thread.
fn spawn_save_sessions(sessions: &HashMap<String, String>) {
    let snapshot = sessions.clone();
    tokio::task::spawn_blocking(move || save_sessions(TRANSPORT, &snapshot));
}
fn spawn_save_channel_models(models: &HashMap<String, String>) {
    let snapshot = models.clone();
    tokio::task::spawn_blocking(move || save_channel_models(TRANSPORT, &snapshot));
}

type SessionMap = Arc<Mutex<HashMap<String, String>>>;
type ModelMap = Arc<Mutex<HashMap<String, String>>>;

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

async fn send_typing(account: &str, identifier: &str, stop: bool) -> anyhow::Result<()> {
    let is_group = !identifier.starts_with('+');
    let mut cmd = tokio::process::Command::new("signal-cli");
    cmd.args(["-a", account]);
    if stop {
        cmd.arg("-s");
    }
    if is_group {
        cmd.args(["-g", identifier]);
    } else {
        cmd.arg(identifier);
    }
    cmd.arg("sendTyping");
    let output = cmd.output().await?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        anyhow::bail!("signal-cli sendTyping failed: {}", stderr.trim());
    }
    Ok(())
}

async fn send_message_to(account: &str, identifier: &str, content: &str) -> anyhow::Result<()> {
    let is_group = !identifier.starts_with('+');
    let mut cmd = tokio::process::Command::new("signal-cli");
    cmd.args(["-a", account, "send", "-m", content]);
    if is_group {
        cmd.args(["-g", identifier]);
    } else {
        cmd.arg(identifier);
    }
    let output = cmd.output().await?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        anyhow::bail!("signal-cli send failed: {}", stderr.trim());
    }
    Ok(())
}

/// Split long text into Signal messages (max 4 000 chars each).
async fn send_chunked(account: &str, identifier: &str, text: &str) -> anyhow::Result<()> {
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
        send_message_to(account, identifier, chunk).await?;
        rest = remainder;
    }
    Ok(())
}

async fn receive_messages(account: &str) -> anyhow::Result<Vec<(String, String, bool)>> {
    let output = tokio::process::Command::new("signal-cli")
        .args(["-a", account, "-o", "json", "receive"])
        .output()
        .await?;

    if !output.status.success() && output.stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if !stderr.is_empty() {
            warn!("signal-cli receive stderr: {}", stderr);
        }
    }

    parse_received_messages(&output.stdout)
}

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
        let Some(envelope) = val.get("envelope") else {
            continue;
        };
        let Some(data_msg) = envelope.get("dataMessage") else {
            continue;
        };
        let Some(message) = data_msg.get("message").and_then(|m| m.as_str()) else {
            continue;
        };
        if message.is_empty() {
            continue;
        }

        let source_number = envelope
            .get("sourceNumber")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        if source_number.is_empty() {
            continue;
        }

        let (reply_to, is_group) = if let Some(group_info) = data_msg.get("groupInfo") {
            let group_id = group_info
                .get("groupId")
                .and_then(|id| id.as_str())
                .unwrap_or(&source_number)
                .to_string();
            (group_id, true)
        } else {
            (source_number, false)
        };

        messages.push((reply_to, message.to_string(), is_group));
    }
    Ok(messages)
}

async fn list_llama_models(base_url: &str) -> anyhow::Result<Vec<String>> {
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
    let body: serde_json::Value = reqwest::get(&url).await?.json().await?;
    Ok(body
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|id| id.as_str()).map(|s| s.to_string()))
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
    _is_group: bool,
    sessions: SessionMap,
    channel_models: ModelMap,
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

    // !ping
    if text == format!("{prefix}ping") {
        let _ = send_message_to(account, sender, "Pong! (Signal)").await;
        return;
    }

    // !omp help / ?
    if text == format!("{prefix}omp help") || text == format!("{prefix}omp ?") {
        let help = format!(
            "Available {prefix}omp commands:\n\n\
            {prefix}omp reset      \u{2014} Clear current conversation session\n\
            {prefix}omp model      \u{2014} Show current model\n\
            {prefix}omp model <n>  \u{2014} Set model (clears session)\n\
            {prefix}omp llama-list \u{2014} List available models\n\
            {prefix}omp ? / help   \u{2014} Show this help\n\n\
            Tip: Use --model <name> at the start of any message for a one-time override."
        );
        let _ = send_message_to(account, sender, &help).await;
        return;
    }

    // !omp reset
    if text == format!("{prefix}omp reset") {
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
        let _ = send_message_to(account, sender, reply).await;
        return;
    }

    // !omp model — show current model
    if text == format!("{prefix}omp model") {
        let prefs = channel_models.lock().await;
        let reply = match prefs.get(sender) {
            Some(m) => format!("Current model: {m}"),
            None => format!("No model set (using default: {default_model})."),
        };
        let _ = send_message_to(account, sender, &reply).await;
        return;
    }

    // !omp model <name>
    if let Some(rest) = text.strip_prefix(&format!("{prefix}omp model ")) {
        let name = rest.trim();
        if name.is_empty() {
            let _ = send_message_to(account, sender, "Usage: !omp model <name>").await;
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
            account,
            sender,
            &format!("Model set to: {resolved}\nSession cleared."),
        )
        .await;
        return;
    }

    // !omp llama-list
    if text == format!("{prefix}omp llama-list") {
        let reply = match list_llama_models(llama_swap_url).await {
            Ok(models) if models.is_empty() => "No models found.".to_string(),
            Ok(models) => format!("Available models:\n{}", models.join("\n")),
            Err(e) => format!("Failed to list models: {e}"),
        };
        let _ = send_message_to(account, sender, &reply).await;
        return;
    }

    // Strip !omp prefix if present, leaving the bare query.
    let mut query_text = text;
    if let Some(rest) = text.strip_prefix(&format!("{prefix}omp")) {
        query_text = rest.trim();
    } else if let Some(rest) = text.strip_prefix(account) {
        query_text = rest.trim();
    }

    if query_text.is_empty() {
        let _ = send_message_to(account, sender, "(No query provided)").await;
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

    let _ = send_typing(account, sender, false).await;

    let result = invoke_omp(
        omp_path,
        work_dir,
        effective_model.as_deref(),
        &actual_query,
        session_id.as_deref(),
        timeout_secs,
    )
    .await;

    let _ = send_typing(account, sender, true).await;

    match result {
        Ok((response, new_session, model_info)) => {
            if let Some(sid) = new_session {
                let mut sessions = sessions.lock().await;
                sessions.insert(sender.to_string(), sid);
                spawn_save_sessions(&sessions);
            }
            let text = if response.is_empty() {
                "(OMP returned an empty response)".to_string()
            } else if let (Some(_), Some((provider, mdl))) =
                (model_owned.as_deref(), model_info)
            {
                format!("{response}\n\n_({provider}/{mdl})_")
            } else {
                response
            };
            send_chunked(account, sender, &text).await.ok();
            manager.record_signal_message();
        }
        Err(e) => {
            warn!("OMP invocation failed: {e}");
            let _ = send_message_to(account, sender, &format!("OMP error: {e}")).await;
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
    prefix: String,
    llama_swap_url: String,
    omp_path: String,
    work_dir: String,
    aliases_config: String,
    timeout_secs: u64,
    default_model: String,
    sessions: SessionMap,
    model_aliases: HashMap<String, String>,
    channel_models: ModelMap,
    manager: Arc<super::BridgeManager>,
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
        let prefix = signal_config.prefix_str().to_string();
        let llama_swap_url = signal_config.llama_url().to_string();

        if let Some(ref storage) = signal_config.storage_path {
            if let Err(e) = std::fs::create_dir_all(storage) {
                warn!("Could not create storage dir {storage}: {e}");
            }
        }

        let sessions: SessionMap = Arc::new(Mutex::new(load_sessions(TRANSPORT)));
        let model_aliases = load_model_aliases(aliases_config);
        let channel_models: ModelMap = Arc::new(Mutex::new(load_channel_models(TRANSPORT)));

        info!("Signal service ready");
        Ok(Self {
            account,
            prefix,
            llama_swap_url,
            omp_path: omp_path.to_string(),
            work_dir: work_dir.to_string(),
            aliases_config: aliases_config.to_string(),
            timeout_secs,
            default_model: default_model.to_string(),
            sessions,
            model_aliases,
            channel_models,
            manager,
        })
    }

    pub async fn send_message(&self, identifier: &str, content: &str) -> anyhow::Result<()> {
        send_message_to(&self.account, identifier, content).await
    }

    pub async fn list_contacts(&self) -> anyhow::Result<Vec<Contact>> {
        warn!("Signal contact listing not fully implemented via signal-cli");
        Ok(Vec::new())
    }

    /// Spawn a background task polling signal-cli for incoming messages every 3 seconds.
    pub fn start_receive_loop(&self) {
        let account = self.account.clone();
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
            info!("Signal receive loop started (polling every 3 s)");
            loop {
                match receive_messages(&account).await {
                    Ok(msgs) => {
                        for (reply_to, content, is_group) in msgs {
                            handle_message(
                                &reply_to,
                                &content,
                                is_group,
                                sessions.clone(),
                                channel_models.clone(),
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
                    }
                    Err(e) => warn!("Signal receive error: {e}"),
                }
                tokio::time::sleep(Duration::from_secs(3)).await;
            }
        });
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

    service.start_receive_loop();

    // Keep alive forever — the receive loop runs in a spawned task.
    loop {
        tokio::time::sleep(Duration::from_secs(3600)).await;
    }
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
        let json = r#"{"envelope":{"source":"+1555000001","sourceNumber":"+1555000001","sourceUuid":"abc","sourceName":"Alice","sourceDevice":1,"timestamp":1000,"dataMessage":{"timestamp":1000,"message":"!omp hello","groupInfo":{"groupId":"AAAA=","type":"DELIVER"}}}}"#;
        let msgs = parse_received_messages(json.as_bytes()).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].0, "AAAA=");
        assert_eq!(msgs[0].1, "!omp hello");
        assert!(msgs[0].2);
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
}
