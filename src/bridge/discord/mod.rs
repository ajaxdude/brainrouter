//! Discord transport for the OMP bridge.
//!
//! Connects to Discord's gateway via serenity and forwards `!omp` commands
//! (and @-mentions) to the OMP subprocess.  REST helpers (`send_message`,
//! `read_channel`, etc.) are exposed for programmatic use.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serenity::all::{
    ChannelId, Context, EventHandler, GatewayIntents, GetMessages, Message, Ready,
};
use serenity::async_trait;
use serenity::client::ClientBuilder;
use tokio::sync::Mutex;
use tracing::info;

use crate::bridge::core::{invoke_omp, load_model_aliases, resolve_model};
use crate::bridge::persist::{
    display_path, load_channel_models, load_sessions, load_work_dirs, save_channel_models,
    save_sessions, save_work_dirs,
};

const TRANSPORT: &str = "discord";


// ---------------------------------------------------------------------------
// Non-blocking persistence — fire-and-forget on a background thread.
// Clones the map so the caller's mutex is released immediately.
// ---------------------------------------------------------------------------

fn spawn_save_sessions(sessions: &HashMap<String, String>) {
    let snapshot = sessions.clone();
    tokio::task::spawn_blocking(move || save_sessions(TRANSPORT, &snapshot));
}

fn spawn_save_channel_models(models: &HashMap<String, String>) {
    let snapshot = models.clone();
    tokio::task::spawn_blocking(move || save_channel_models(TRANSPORT, &snapshot));
}

fn spawn_save_work_dirs(dirs: &HashMap<String, String>) {
    let snapshot = dirs.clone();
    tokio::task::spawn_blocking(move || save_work_dirs(TRANSPORT, &snapshot));
}
// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Discord-specific configuration, deserialized from the brainrouter config.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct DiscordConfig {
    /// Whether the Discord transport is enabled.
    #[serde(default)]
    pub enabled: Option<bool>,
    /// Bot token.  Required when `enabled` is true.
    pub token: Option<String>,
    /// Command prefix (default `!`).
    #[serde(default = "default_prefix")]
    pub prefix: Option<String>,
    /// Optional channel restriction (unused today, reserved).
    pub channel_id: Option<String>,
}

fn default_prefix() -> Option<String> {
    Some("!".to_string())
}

impl DiscordConfig {
    pub fn is_enabled(&self) -> bool {
        self.enabled.unwrap_or(false)
    }

    pub fn prefix(&self) -> &str {
        self.prefix.as_deref().unwrap_or("!")
    }
}

// ---------------------------------------------------------------------------
// User-facing types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ChannelMessage {
    pub id: String,
    pub author: String,
    pub content: String,
    pub timestamp: String,
}

#[derive(Debug, Clone)]
pub struct ServerInfo {
    pub id: String,
    pub name: String,
    pub member_count: u64,
}

// ---------------------------------------------------------------------------
// send_chunked
// ---------------------------------------------------------------------------

/// Send a long string as successive Discord messages, each at most 1 500 chars.
///
/// Discord enforces a 2 000-character hard limit per message.  Splitting on
/// character count (not bytes) is correct: Discord counts Unicode scalar values.
async fn send_chunked(ctx: &Context, channel_id: ChannelId, text: &str) {
    const MAX_CHARS: usize = 1_500;
    let mut rest = text;
    while !rest.is_empty() {
        let split = rest
            .char_indices()
            .nth(MAX_CHARS)
            .map(|(idx, _)| idx)
            .unwrap_or(rest.len());
        let (chunk, remainder) = rest.split_at(split);
        let _ = channel_id.say(&ctx.http, chunk).await;
        rest = remainder;
    }
}

// ---------------------------------------------------------------------------
// Gateway event handler
// ---------------------------------------------------------------------------

/// Internal handler config — combines the Discord-specific fields with the
/// shared bridge parameters that the handler needs at runtime.
struct HandlerConfig {
    prefix: String,
    omp_path: String,
    omp_work_dir: String,
    omp_timeout_secs: u64,
    default_model: String,
}

struct DiscordHandler {
    config: HandlerConfig,
    bot_id: Arc<std::sync::OnceLock<serenity::model::id::UserId>>,
    sessions: SessionMap,
    channel_models: ModelMap,
    model_aliases: HashMap<String, String>,
    omp_root: PathBuf,
    work_dirs: WorkdirMap,
}

type SessionMap = Arc<Mutex<HashMap<String, String>>>;
type ModelMap = Arc<Mutex<HashMap<String, String>>>;
type WorkdirMap = Arc<Mutex<HashMap<String, String>>>;

#[async_trait]
impl EventHandler for DiscordHandler {
    async fn ready(&self, _ctx: Context, ready: Ready) {
        let _ = self.bot_id.set(ready.user.id);
        info!(
            "Discord bot connected as: {} (ID: {})",
            ready.user.name, ready.user.id
        );
    }

    async fn message(&self, ctx: Context, msg: Message) {
        tracing::debug!(
            channel = %msg.channel_id,
            author = %msg.author.name,
            is_bot = msg.author.bot,
            is_dm = msg.guild_id.is_none(),
            guild_id = ?msg.guild_id,
            "Discord message event"
        );
        if msg.author.bot {
            return;
        }

        let prefix = &self.config.prefix;
        let is_dm = msg.guild_id.is_none();
        // Normalize em-dash (U+2014) → "--" so command parsing always sees ASCII hyphens.
        let content_normalized = msg.content.replace('\u{2014}', "--");
        let mut text = content_normalized.trim();

        // !ping  (works in both DMs and guilds)
        if text == format!("{}ping", prefix) {
            let now = serenity::all::Timestamp::now();
            let now_f64 = now.unix_timestamp() as f64 + now.nanosecond() as f64 / 1e9;
            let msg_f64 =
                msg.timestamp.unix_timestamp() as f64 + msg.timestamp.nanosecond() as f64 / 1e9;
            let latency = (now_f64 - msg_f64).max(0.0);
            let _ = msg
                .channel_id
                .say(&ctx.http, format!("Pong! {:.3}s", latency))
                .await;
            return;
        }

        // !omp reset  (works in both DMs and guilds)
        if text == format!("{}omp reset", prefix) || text == "reset" {
            let channel_key = msg.channel_id.to_string();
            let had_session = {
                let mut sessions = self.sessions.lock().await;
                let removed = sessions.remove(&channel_key).is_some();
                if removed {
                    spawn_save_sessions(&sessions);
                }
                removed
            };
            let reply = if had_session {
                "Session cleared. Starting fresh on your next message."
            } else {
                "No active session for this channel."
            };
            let _ = msg.channel_id.say(&ctx.http, reply).await;
            return;
        }

        // Strip !omp prefix or @mention if present; bare text is always accepted.
        if let Some(rest) = text.strip_prefix(&format!("{}omp", prefix)) {
            text = rest.trim();
        } else if let Some(bot_id) = self.bot_id.get() {
            let long_mention = format!("<@{}>", bot_id);
            let nick_mention = format!("<@!{}>", bot_id);
            if let Some(rest) = text.strip_prefix(&long_mention) {
                text = rest.trim();
            } else if let Some(rest) = text.strip_prefix(&nick_mention) {
                text = rest.trim();
            }
            // No prefix/mention: use full text as query.
        }
        // In DMs: bare text falls through here as the query.

        if text.is_empty() {
            return;
        }

        // !omp ? / help  (also bare "help"/"?" in DMs)
        if text == "?" || text == "help" {
            let help_msg = if is_dm {
                "**OMP Bridge \u{2014} Discord Commands**\n\
                Just type your message \u{2014} no prefix needed in DMs.\n\n\
                `reset` \u{2014} Clear session\n\
                `model <alias> <query>` \u{2014} One-off model override\n\
                `brainrouter` \u{2014} Show current model\n\
                `brainrouter auto|local|cloud` \u{2014} Set routing mode\n\
                `brainrouter <model>` \u{2014} Use specific local model\n\
                `brainrouter list` \u{2014} List all available models\n\
                `review [auto|local|cloud]` \u{2014} Set/show review mode\n\
                `ls` \u{2014} List working directory\n\
                `cd <dir>` \u{2014} Change directory\n\
                `..` \u{2014} Go up one directory\n\
                `mkdir <name>` \u{2014} Create a directory\n\
                `help` / `?` \u{2014} Show this help"
            } else {
                "**OMP Bridge — Discord Commands**\n\
                `!ping` — Health check\n\
                `!omp reset` — Clear session for this channel\n\
                `!omp <query>` — Send a query to OMP\n\
                `@bot <query>` — Same as `!omp <query>`\n\
                `!omp model <alias> <query>` — One-off model override\n\
                `!omp brainrouter` \u{2014} Show current model\n\
                `!omp brainrouter auto|local|cloud` \u{2014} Set routing mode\n\
                `!omp brainrouter <model>` \u{2014} Use specific local model\n\
                `!omp brainrouter list` \u{2014} List all available models\n\
                `!omp review [auto|local|cloud]` \u{2014} Set/show review mode\n\
                `!omp ls` — List current working directory\n\
                `!omp cd <dir>` — Change directory (sandboxed)\n\
                `!omp ..` — Go up one directory\n\
                `!omp mkdir <name>` — Create a directory\n\
                `!omp help` / `!omp ?` — Show this help"
            };
            let _ = msg.channel_id.say(&ctx.http, help_msg).await;
            return;
        }

		// !omp brainrouter [auto|cloud|local|list|<model>]
		if let Some(rest) = text.strip_prefix("brainrouter ") {
		    let arg = rest.trim();
		    let arg = if arg.is_empty() { "auto" } else { arg };
		    if arg == "list" {
		        // List all available models from brainrouter /v1/models
		        let _ = msg.channel_id.broadcast_typing(&ctx.http).await;
		        match reqwest::get("http://127.0.0.1:9099/v1/models").await {
		            Ok(resp) => {
		                if let Ok(json) = resp.json::<serde_json::Value>().await {
		                    let mut routing = Vec::new();
		                    let mut local = Vec::new();
		                    if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
		                        for item in data {
		                            if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
		                                let owner = item.get("owned_by").and_then(|v| v.as_str()).unwrap_or("");
		                                if owner == "brainrouter" {
		                                    routing.push(format!("  `{}` \u{2014} routing mode", id));
		                                } else {
		                                    local.push(format!("  `{}`", id));
		                                }
		                            }
		                        }
		                    }
		                    let mut reply = String::from("**Routing modes:**\n");
		                    reply.push_str(&routing.join("\n"));
		                    if !local.is_empty() {
		                        reply.push_str("\n\n**Local models (llama-swap):**\n");
		                        reply.push_str(&local.join("\n"));
		                    }
		                    reply.push_str("\n\n**Cloud:** Manifest (auto-selects provider)");
		                    send_chunked(&ctx, msg.channel_id, &reply).await;
		                } else {
		                    let _ = msg.channel_id.say(&ctx.http, "Failed to parse models.").await;
		                }
		            }
		            Err(e) => {
		                let _ = msg.channel_id.say(&ctx.http, format!("Failed: {}", e)).await;
		            }
		        }
		        return;
		    }
		    // auto, local, cloud, or specific model name
		    let model_id = format!("brainrouter/{}", arg);
		    {
		        let mut channel_models = self.channel_models.lock().await;
		        channel_models.insert(msg.channel_id.to_string(), model_id.clone());
		        spawn_save_channel_models(&channel_models);
		    }
		    {
		        let mut sessions = self.sessions.lock().await;
		        if sessions.remove(&msg.channel_id.to_string()).is_some() {
		            spawn_save_sessions(&sessions);
		        }
		    }
		    let _ = msg.channel_id.say(&ctx.http,
		        format!("Model set to `{}`. Session cleared.", model_id)).await;
		    return;
		}
		// Also handle bare "brainrouter" (no arg) — show current model
		if text == "brainrouter" {
		    let channel_key = msg.channel_id.to_string();
		    let channel_models = self.channel_models.lock().await;
		    let current = channel_models.get(&channel_key)
		        .map(|m| m.as_str())
		        .unwrap_or(&self.config.default_model);
		    let _ = msg.channel_id.say(&ctx.http,
		        format!("Current model: `{}`\nUse `brainrouter list` to see options.", current)).await;
		    return;
		}

        // !omp review [auto|local|cloud] — set the code review routing mode
        if let Some(rest) = text.strip_prefix("review ") {
            let mode = rest.trim();
            if mode != "auto" && mode != "local" && mode != "cloud" {
                let _ = msg.channel_id.say(&ctx.http,
                    "Usage: `review auto|local|cloud`").await;
                return;
            }
            let body = serde_json::json!({
                "forced_mode": mode,
                "max_iterations": 5
            });
            match reqwest::Client::new()
                .post("http://127.0.0.1:9099/api/review-config")
                .json(&body)
                .send().await
            {
                Ok(resp) if resp.status().is_success() => {
                    let _ = msg.channel_id.say(&ctx.http,
                        format!("Review mode set to `{}`.", mode)).await;
                }
                Ok(resp) => {
                    let err = resp.text().await.unwrap_or_default();
                    let _ = msg.channel_id.say(&ctx.http,
                        format!("Failed to set review mode: {}", err)).await;
                }
                Err(e) => {
                    let _ = msg.channel_id.say(&ctx.http,
                        format!("Failed to reach brainrouter: {}", e)).await;
                }
            }
            return;
        }
        if text == "review" {
            match reqwest::get("http://127.0.0.1:9099/api/review-config").await {
                Ok(resp) => {
                    if let Ok(json) = resp.json::<serde_json::Value>().await {
                        let mode = json.get("forced_mode").and_then(|v| v.as_str()).unwrap_or("auto");
                        let model = json.get("forced_model").and_then(|v| v.as_str());
                        let reply = if let Some(m) = model {
                            format!("Review mode: `{}` (model: `{}`)", mode, m)
                        } else {
                            format!("Review mode: `{}`", mode)
                        };
                        let _ = msg.channel_id.say(&ctx.http, reply).await;
                    } else {
                        let _ = msg.channel_id.say(&ctx.http, "Failed to parse review config.").await;
                    }
                }
                Err(e) => {
                    let _ = msg.channel_id.say(&ctx.http,
                        format!("Failed to reach brainrouter: {}", e)).await;
                }
            }
            return;
        }

        // !omp ls
        if text == "ls" {
            let current_dir = self.channel_workdir(&msg.channel_id.to_string()).await;
            match std::fs::read_dir(&current_dir) {
                Ok(entries) => {
                    let mut names: Vec<String> = entries
                        .filter_map(|e| e.ok())
                        .map(|e| {
                            let name = e.file_name().to_string_lossy().to_string();
                            if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                                format!("{}/", name)
                            } else {
                                name
                            }
                        })
                        .collect();
                    names.sort();
                    let header = display_path(&current_dir, &self.omp_root);
                    let body = if names.is_empty() {
                        "(empty)".to_string()
                    } else {
                        names.join("\n")
                    };
                    let reply = format!("**{}**\n```\n{}\n```", header, body);
                    send_chunked(&ctx, msg.channel_id, &reply).await;
                }
                Err(e) => {
                    let _ = msg
                        .channel_id
                        .say(&ctx.http, format!("Cannot list directory: {}", e))
                        .await;
                }
            }
            return;
        }

        // !omp ..
        if text == ".." {
            let channel_key = msg.channel_id.to_string();
            let current_dir = self.channel_workdir(&channel_key).await;
            if current_dir == self.omp_root {
                let _ = msg
                    .channel_id
                    .say(&ctx.http, "Already at the root of the allowed directory tree.")
                    .await;
                return;
            }
            let parent = match current_dir.parent() {
                Some(p) if p.starts_with(&self.omp_root) => p.to_path_buf(),
                _ => {
                    let _ = msg
                        .channel_id
                        .say(&ctx.http, "Already at the root of the allowed directory tree.")
                        .await;
                    return;
                }
            };
            self.set_workdir(&channel_key, &parent).await;
            self.clear_session(&channel_key).await;
            let _ = msg
                .channel_id
                .say(
                    &ctx.http,
                    format!("Now in `{}`", display_path(&parent, &self.omp_root)),
                )
                .await;
            return;
        }

        // !omp cd <dir>
        if let Some(target) = text.strip_prefix("cd ") {
            let target = target.trim();
            let channel_key = msg.channel_id.to_string();
            let current_dir = self.channel_workdir(&channel_key).await;
            let candidate = if target == "/" {
                self.omp_root.clone()
            } else {
                current_dir.join(target)
            };
            let resolved = match candidate.canonicalize() {
                Ok(p) => p,
                Err(_) => {
                    let _ = msg
                        .channel_id
                        .say(&ctx.http, format!("`{}`: no such directory", target))
                        .await;
                    return;
                }
            };
            if !resolved.starts_with(&self.omp_root) {
                let _ = msg
                    .channel_id
                    .say(&ctx.http, "Cannot navigate outside the root directory.")
                    .await;
                return;
            }
            if !resolved.is_dir() {
                let _ = msg
                    .channel_id
                    .say(&ctx.http, format!("`{}`: not a directory", target))
                    .await;
                return;
            }
            self.set_workdir(&channel_key, &resolved).await;
            self.clear_session(&channel_key).await;
            let _ = msg
                .channel_id
                .say(
                    &ctx.http,
                    format!("Now in `{}`", display_path(&resolved, &self.omp_root)),
                )
                .await;
            return;
        }

        // !omp mkdir <name>
        if let Some(name) = text.strip_prefix("mkdir ") {
            let name = name.trim();
            let channel_key = msg.channel_id.to_string();
            let current_dir = self.channel_workdir(&channel_key).await;
            let new_dir = current_dir.join(name);
            if new_dir.components().any(|c| c.as_os_str() == "..") {
                let _ = msg
                    .channel_id
                    .say(&ctx.http, "Directory name cannot contain `..`.")
                    .await;
                return;
            }
            match std::fs::create_dir_all(&new_dir) {
                Ok(_) => {
                    let _ = msg
                        .channel_id
                        .say(
                            &ctx.http,
                            format!("Created `{}`", display_path(&new_dir, &self.omp_root)),
                        )
                        .await;
                }
                Err(e) => {
                    let _ = msg
                        .channel_id
                        .say(&ctx.http, format!("Cannot create directory: {}", e))
                        .await;
                }
            }
            return;
        }

        // --- OMP query ---
        self.handle_omp_query(ctx, msg, text).await;
    }
}

impl DiscordHandler {
    /// Resolve the current working directory for a channel, staying within omp_root.
    async fn channel_workdir(&self, channel_key: &str) -> PathBuf {
        let dirs = self.work_dirs.lock().await;
        dirs.get(channel_key)
            .map(PathBuf::from)
            .filter(|p| p.starts_with(&self.omp_root) && p.is_dir())
            .unwrap_or_else(|| self.omp_root.clone())
    }

    async fn set_workdir(&self, channel_key: &str, path: &Path) {
        let mut dirs = self.work_dirs.lock().await;
        dirs.insert(channel_key.to_string(), path.to_string_lossy().to_string());
        spawn_save_work_dirs(&dirs);
    }

    async fn clear_session(&self, channel_key: &str) {
        let mut sessions = self.sessions.lock().await;
        if sessions.remove(channel_key).is_some() {
            spawn_save_sessions(&sessions);
        }
    }

    async fn handle_omp_query(&self, ctx: Context, msg: Message, text: &str) {
        let channel_key = msg.channel_id.to_string();

        // Resolve model: inline override > sticky channel model > default_model.
        let mut is_override = false;
        let (model_owned, actual_query) = {
            let mut q = text;
            let mut m: Option<String> = None;
            if q.starts_with("model ") {
                let parts: Vec<&str> = q.splitn(3, ' ').collect();
                if parts.len() >= 3 {
                    let resolved = resolve_model(parts[1], &self.model_aliases);
                    m = Some(if resolved.starts_with("brainrouter/") { resolved } else { self.config.default_model.clone() });
                    q = parts[2];
                    is_override = true;
                }
            }
            if m.is_none() {
                let channel_models = self.channel_models.lock().await;
                m = channel_models.get(&channel_key).cloned();
                // Sanitize stale models that OMP can't route (e.g. llama.cpp/*).
                if let Some(ref model) = m {
                    if !model.starts_with("brainrouter/") {
                        m = Some(self.config.default_model.clone());
                    }
                }
            }
            // Fall back to configured default (brainrouter/auto by default).
            if m.is_none() {
                m = Some(self.config.default_model.clone());
            }
            (m, q.to_string())
        };
        let model = model_owned.as_deref();

        // Session resume — skip when using a one-off model override.
        let session_id: Option<String> = if is_override {
            info!(
                "Model override {:?} — skipping session resume for channel {}",
                model, channel_key
            );
            None
        } else {
            let sessions = self.sessions.lock().await;
            sessions.get(&channel_key).cloned()
        };

        let omp_work_dir = self.channel_workdir(&channel_key).await;
        let omp_work_dir = omp_work_dir.to_string_lossy().to_string();

        // Keep typing indicator alive for the duration of the OMP call.
        let (typing_cancel_tx, mut typing_cancel_rx) = tokio::sync::oneshot::channel::<()>();
        let typing_http = ctx.http.clone();
        let typing_channel = msg.channel_id;
        tokio::spawn(async move {
            loop {
                let _ = typing_channel.broadcast_typing(&typing_http).await;
                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_secs(8)) => {}
                    _ = &mut typing_cancel_rx => break,
                }
            }
        });

        // Invoke OMP, retrying once without --resume on stale-session errors.
        let result = {
            let first = invoke_omp(
                &self.config.omp_path,
                &omp_work_dir,
                model,
                &actual_query,
                session_id.as_deref(),
                self.config.omp_timeout_secs,
            )
            .await;

            match first {
                Ok(v) => Ok(v),
                Err(ref e)
                    if (e.contains("not found")
                        || e.contains("exceeds the available context size"))
                        && session_id.is_some() =>
                {
                    tracing::warn!(
                        "Session for channel {} cleared ({}); retrying without --resume",
                        channel_key,
                        if e.contains("context size") {
                            "context overflow"
                        } else {
                            "stale session"
                        }
                    );
                    self.clear_session(&channel_key).await;
                    invoke_omp(
                        &self.config.omp_path,
                        &omp_work_dir,
                        model,
                        &actual_query,
                        None,
                        self.config.omp_timeout_secs,
                    )
                    .await
                }
                Err(e) => Err(e),
            }
        };

        let _ = typing_cancel_tx.send(());

        match result {
            Ok((response, new_session, model_info)) => {
                if let Some(sid) = new_session {
                    let mut sessions = self.sessions.lock().await;
                    sessions.insert(channel_key.clone(), sid.clone());
                    spawn_save_sessions(&sessions);
                    info!("Saved session {} for channel {}", sid, channel_key);
                }
                let text = if response.is_empty() {
                    "(OMP returned an empty response)".to_string()
                } else if let (Some(_), Some((provider, mdl))) =
                    (model_owned.as_deref(), model_info)
                {
                    if is_override {
                        format!("{response}\n\n-# {provider}/{mdl}")
                    } else {
                        response
                    }
                } else {
                    response
                };
                send_chunked(&ctx, msg.channel_id, &text).await;
            }
            Err(e) => {
                self.clear_session(&channel_key).await;
                tracing::error!("OMP invocation failed: {}", e);
                let _ = msg
                    .channel_id
                    .say(&ctx.http, format!("OMP error: {}", e))
                    .await;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DiscordService — REST API surface
// ---------------------------------------------------------------------------

pub struct DiscordService {
    pub http: Arc<serenity::http::Http>,
    _gateway_task: tokio::task::JoinHandle<()>,
}

impl DiscordService {
    pub async fn send_message(
        &self,
        channel_id: &str,
        content: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let cid = channel_id
            .parse::<serenity::model::id::ChannelId>()
            .map_err(|e| format!("Invalid channel ID: {}", e))?;
        let message = cid.say(&self.http, content).await?;
        Ok(message.id.to_string())
    }

    pub async fn read_channel(
        &self,
        channel_id: &str,
        limit: u32,
    ) -> Result<Vec<ChannelMessage>, Box<dyn std::error::Error + Send + Sync>> {
        let cid = channel_id
            .parse::<serenity::model::id::ChannelId>()
            .map_err(|e| format!("Invalid channel ID: {}", e))?;
        let messages = cid
            .messages(
                &self.http,
                GetMessages::default().limit(limit.min(100) as u8),
            )
            .await?;
        Ok(messages
            .into_iter()
            .map(|m| ChannelMessage {
                id: m.id.to_string(),
                author: m.author.name.to_string(),
                content: m.content,
                timestamp: m.timestamp.to_rfc3339().unwrap_or_default(),
            })
            .collect())
    }

    pub async fn list_servers(
        &self,
    ) -> Result<Vec<ServerInfo>, Box<dyn std::error::Error + Send + Sync>> {
        let guilds = self.http.get_guilds(None, None).await?;
        Ok(guilds
            .into_iter()
            .map(|g| ServerInfo {
                id: g.id.to_string(),
                name: g.name,
                member_count: 0,
            })
            .collect())
    }

    pub async fn mention_user(
        &self,
        channel_id: &str,
        user_id: &str,
        content: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        self.send_message(channel_id, &format!("<@{}> {}", user_id, content))
            .await
    }

    pub async fn post_file(
        &self,
        channel_id: &str,
        file_path: &str,
        description: Option<String>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        use serenity::all::{CreateAttachment, CreateMessage};
        let cid = channel_id
            .parse::<serenity::model::id::ChannelId>()
            .map_err(|e| format!("Invalid channel ID: {}", e))?;
        let file_data = std::fs::read(file_path)?;
        let file_name = std::path::Path::new(file_path)
            .file_name()
            .unwrap_or(std::ffi::OsStr::new("file"))
            .to_string_lossy()
            .to_string();
        let attachment = CreateAttachment::bytes(file_data, &file_name);
        let mut message = CreateMessage::default();
        if let Some(desc) = description {
            message = message.content(desc);
        }
        cid.send_files(&self.http, [attachment], message).await?;
        Ok(format!("File {} uploaded successfully", file_name))
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Start the Discord transport.
///
/// This builds a serenity client, spawns the gateway, and loops forever to
/// keep the transport alive.  Returns `Err` only if the token is missing or
/// the client cannot be built.
pub async fn start(
    discord_config: &DiscordConfig,
    omp_path: &str,
    work_dir: &str,
    aliases_config: &str,
    timeout_secs: u64,
    default_model: &str,
) -> anyhow::Result<()> {
    let token = discord_config
        .token
        .as_deref()
        .filter(|t| !t.is_empty())
        .ok_or_else(|| anyhow::anyhow!("Discord token is required but not configured"))?;

    let prefix = discord_config.prefix().to_string();

    info!("Building Discord client...");

    let bot_id = Arc::new(std::sync::OnceLock::new());
    let sessions: SessionMap = Arc::new(Mutex::new(load_sessions(TRANSPORT)));
    let channel_models: ModelMap = Arc::new(Mutex::new(load_channel_models(TRANSPORT)));
    let model_aliases = load_model_aliases(aliases_config);

    let omp_root = PathBuf::from(work_dir)
        .canonicalize()
        .unwrap_or_else(|e| {
            tracing::warn!(
                "Could not canonicalize work_dir {:?}: {}; using as-is",
                work_dir,
                e
            );
            PathBuf::from(work_dir)
        });

    let work_dirs: WorkdirMap = Arc::new(Mutex::new(load_work_dirs(TRANSPORT)));

    let handler = DiscordHandler {
        config: HandlerConfig {
            prefix,
            omp_path: omp_path.to_string(),
            omp_work_dir: work_dir.to_string(),
            omp_timeout_secs: timeout_secs,
            default_model: default_model.to_string(),
        },
        bot_id: bot_id.clone(),
        sessions,
        channel_models,
        model_aliases,
        omp_root,
        work_dirs,
    };

    let mut client = ClientBuilder::new(
        token,
        GatewayIntents::GUILD_MESSAGES
            | GatewayIntents::DIRECT_MESSAGES
            | GatewayIntents::MESSAGE_CONTENT,
    )
    .event_handler(handler)
    .await?;

    let http = client.http.clone();

    info!("Spawning Discord gateway in background...");
    let gateway_task = tokio::spawn(async move {
        if let Err(e) = client.start_autosharded().await {
            tracing::error!("Discord gateway exited with error: {}", e);
        }
    });

    info!("Discord transport started — gateway connecting in background");

    // Keep the transport alive; the gateway task runs in the background.
    // When the future is dropped (e.g. by BridgeManager), the gateway shuts down.
    let _service = DiscordService {
        http,
        _gateway_task: gateway_task,
    };

    // Park forever — brainrouter's lifecycle owns the shutdown.
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
    }
}
