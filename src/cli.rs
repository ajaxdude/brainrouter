//! brainrouter CLI — `brainrouter cli <command>`.
//!
//! Total-parity headless control plane. Every action the dashboard can perform
//! is available here; the CLI is a thin client that forwards commands to the
//! running daemon over its Unix domain socket (default) or TCP listener
//! (`--url`). It never loads Bonsai or runs the review loop itself.
//!
//! For the headless setup: run `brainrouter serve` (via systemd) and drive
//! everything else from this CLI — status, routing, Bonsai, nudge, context,
//! restarts, upgrades, reviews — no browser required.

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand};
use serde_json::{json, Value};
use std::path::PathBuf;
use tracing::debug;

use brainrouter::daemon_client::{DaemonClient, DaemonEndpoint};
use brainrouter::plugins::PluginManager;
use std::net::TcpStream;
use std::sync::Arc;
use brainrouter::routing_profile::{ModelChoice, RoutingPreset, RoutingProfile};

/// Arguments for the `cli` subcommand.
#[derive(Args)]
pub struct CliArgs {
    /// Path to the daemon's Unix domain socket.
    /// Defaults to the same path `brainrouter serve` uses (config::default_socket_path).
    #[arg(long)]
    pub socket: Option<PathBuf>,

    /// Talk to the daemon over TCP instead of the Unix socket, e.g.
    /// `--url http://127.0.0.1:9099`.
    #[arg(long)]
    pub url: Option<String>,

    #[command(subcommand)]
    pub command: CliCommand,
}

#[derive(Subcommand)]
pub enum CliCommand {
    /// Overall system health: llama-swap, manifest, llama.cpp, cloud fallback
    Status,
    /// Version info for brainrouter, llama-swap, manifest, toolbox, Bonsai
    Versions,
    /// Live inference status: loaded model, state, slot progress
    Inference,
    /// Recent routing events (per-request decisions)
    Events,
    /// Aggregated routing statistics
    Stats,
    /// Model list (auto/local/cloud + llama-swap keys)
    Models {
        /// Fetch the raw llama-swap model list instead of the proxy view
        #[arg(long)]
        llama_swap: bool,
    },
    /// Bonsai classifier server status / start / stop
    Bonsai {
        #[command(subcommand)]
        action: BonsaiAction,
    },
    /// Thinking-budget nudge status / toggle / tier
    Nudge {
        #[command(subcommand)]
        action: NudgeAction,
    },
    /// Prompt-rewrite toggle (local pass-through mode)
    PromptRewrite {
        #[command(subcommand)]
        action: PromptRewriteAction,
    },
    /// llama-swap context size (auto = 131072)
    Context {
        #[command(subcommand)]
        action: ContextAction,
    },
    /// Routing mode override (auto/cloud/local)
    RoutingMode {
        #[command(subcommand)]
        action: RoutingModeAction,
    },
    /// Independent main/reviewer/subagent choices and named presets
    RoutingProfile {
        #[command(subcommand)]
        action: RoutingProfileAction,
    },
    /// Manage sidecar plugins (v2 L2): install/list/enable/disable/remove/
    /// status/config. Talks to the running daemon; falls back to direct
    /// state-dir management when no daemon is running.
    Plugin {
        #[command(subcommand)]
        action: PluginAction,
    },
    /// Config migration controls (per-leg snapshots + named restore).
    Migration {
        #[command(subcommand)]
        action: MigrationAction,
    },
    /// List llama-* toolbox containers
    Toolboxes,
    /// Restart a service: llama-swap, llama-cpp (dashboard shows "llama.cpp"), manifest, brainrouter
    Restart {
        #[arg(value_parser = clap::builder::PossibleValuesParser::new(
            ["llama-swap", "llama-cpp", "manifest", "brainrouter"]
        ))]
        service: String,
    },
    /// Upgrade a component: llama-swap, manifest, toolbox
    Upgrade {
        #[arg(value_parser = clap::builder::PossibleValuesParser::new(
            ["llama-swap", "manifest", "toolbox"]
        ))]
        component: String,
    },
    /// Flush loaded models to free VRAM
    FlushModels,
    /// Push llama-swap models into OMP's models.yml (one-way sync)
    SyncOmp,
    /// List config files the daemon manages
    ConfigFiles,
    /// View brainrouter.yaml, or write a new one from a file or stdin.
    /// Alias: `config` (short form)
    #[command(alias = "config")]
    BrainrouterConfig {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// View llama-swap's config.yaml, or write a new one from a file or stdin
    LlamaSwapConfig {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Review service configuration
    ReviewConfig {
        #[command(subcommand)]
        action: ReviewConfigAction,
    },
    /// Review sessions: list, inspect, request, continue, approve, resolve
    Review {
        #[command(subcommand)]
        action: ReviewAction,
    },
}

#[derive(Subcommand)]
pub enum BonsaiAction {
    /// Current status
    Status,
    /// Enable the classifier server (start it)
    Enable,
    /// Disable the classifier server (stop it; frees VRAM; auto-routing goes local)
    Disable,
    /// Start if stopped, stop if running
    Toggle,
}

#[derive(Subcommand)]
pub enum NudgeAction {
    /// Current status
    Status,
    /// Enable nudge
    Enable,
    /// Disable nudge
    Disable,
    /// Toggle the master switch
    Toggle,
    /// Set the tier override: auto (Bonsai picks), light, deep
    Tier {
        #[arg(value_parser = clap::builder::PossibleValuesParser::new(
            ["auto", "light", "deep"]
        ))]
        tier: String,
    },
}

#[derive(Subcommand)]
pub enum PromptRewriteAction {
    /// Current status
    Status,
    /// Enable prompt rewriting for local routes
    Enable,
    /// Disable prompt rewriting (pass-through mode)
    Disable,
}

#[derive(Subcommand)]
pub enum ContextAction {
    /// Current context size
    Status,
    /// Set a context size in tokens (2048–262144); "auto" = 131072
    Set {
        #[arg(value_parser = parse_context)]
        value: u64,
    },
}

#[derive(Subcommand)]
pub enum RoutingModeAction {
    /// Current routing mode
    Status,
    /// Set the routing mode override
    Set {
        #[arg(value_parser = clap::builder::PossibleValuesParser::new(
            ["auto", "cloud", "local"]
        ))]
        mode: String,
    },
}

#[derive(Subcommand)]
pub enum RoutingProfileAction {
    /// Current profile and cloud opt-in policy
    Status,
    /// Discover local and cloud model IDs (metadata only; no inference)
    Models,
    /// Replace the profile with a JSON file ("-" = stdin)
    Set { path: String },
    /// Apply a named preset, preserving the subagent pool
    Preset {
        #[arg(value_enum)]
        preset: RoutingPreset,
        /// Explicit local main model required by local_custom
        #[arg(long)]
        main_model: Option<String>,
    },
    /// Select one independent role; omit --model for backend default/auto
    Choose {
        #[arg(value_parser = ["main", "reviewer"])]
        role: String,
        #[arg(value_parser = ["auto", "local", "cloud"])]
        backend: String,
        #[arg(long)]
        model: Option<String>,
    },
    /// Set the local subagent pool; omit MODEL to clear it (legacy auto behavior)
    Pool { model: Option<String> },
}

#[derive(Subcommand)]
pub enum PluginAction {
    /// Install a plugin from a directory containing plugin.toml + binary
    Install {
        /// Path to the plugin directory
        dir: String,
    },
    /// List installed plugins
    List,
    /// Enable (and start) a plugin
    Enable {
        id: String,
    },
    /// Disable (and stop) a plugin
    Disable {
        id: String,
    },
    /// Remove a plugin (cleans state, drops credential grants naming it)
    Remove {
        id: String,
    },
    /// Show plugin status (all, or one by id)
    Status {
        id: Option<String>,
    },
    /// Plugin config (namespaced per plugin)
    #[command(name = "config")]
    Config {
        #[command(subcommand)]
        action: PluginConfigAction,
    },
}

#[derive(Subcommand)]
pub enum PluginConfigAction {
    /// Set config keys: `plugin config set <id> key=value [key=value ...]`
    Set {
        id: String,
        /// `key=value` pairs (value parsed as JSON, falling back to string)
        assignments: Vec<String>,
    },
    /// Show a plugin's config
    Show {
        id: String,
    },
}

#[derive(Subcommand)]
pub enum MigrationAction {
    /// Restore a leg snapshot (`v2.0-p2` restores config YAML + credentials
    /// from the newest migration snapshot for that phase)
    Restore {
        /// Phase tag, e.g. v2.0-p2
        phase: String,
    },
}

#[derive(Subcommand)]
pub enum ConfigAction {
    /// Print the current config (YAML)
    Show,
    /// Write a new config from a file ("-" = stdin)
    Set { path: String },
}

#[derive(Subcommand)]
pub enum ReviewConfigAction {
    /// Current review configuration
    Status,
    /// Update fields (partial merge; unspecified fields are kept)
    Update {
        /// Maximum LLM review iterations before human escalation
        #[arg(long)]
        max_iterations: Option<u32>,
        /// Forced review mode: auto | cloud | local
        #[arg(long, value_parser = ["auto", "cloud", "local"])]
        forced_mode: Option<String>,
        /// Explicit local or cloud model ID
        #[arg(long)]
        forced_model: Option<String>,
        /// Clear a prior explicit model and use the backend default
        #[arg(long, conflicts_with = "forced_model")]
        clear_model: bool,
    },
}

#[derive(Subcommand)]
pub enum ReviewAction {
    /// List all review sessions
    List,
    /// Show one review session's details
    Get {
        session_id: String,
    },
    /// Request a code review. Blocks until the review completes unless --async.
    Request {
        /// Task ID, e.g. feature-20260819-001
        task_id: String,
        /// Summary of the change
        summary: String,
        /// Optional details
        #[arg(long)]
        details: Option<String>,
        /// Project directory (defaults to the current directory)
        #[arg(long)]
        cwd: Option<String>,
        /// Return immediately with a session ID; poll `review get` for status
        #[arg(long = "async")]
        async_: bool,
    },
    /// Continue iterating a review (additional LLM rounds)
    Continue {
        session_id: String,
    },
    /// Quick-approve a review (LGTM)
    Lgtm {
        session_id: String,
    },
    /// Resolve a review session with feedback
    Resolve {
        session_id: String,
        feedback: String,
    },
}

/// "auto" → 131072; otherwise parse a token count within the accepted range.
fn parse_context(s: &str) -> Result<u64> {
    if s == "auto" {
        return Ok(131072);
    }
    let v: u64 = s.parse().context("context value must be an integer or 'auto'")?;
    if !(2048..=262144).contains(&v) {
        bail!("context value must be between 2048 and 262144");
    }
    Ok(v)
}

/// Entry point for `brainrouter cli`.
pub async fn run(args: CliArgs) -> Result<()> {
    let endpoint = if let Some(url) = &args.url {
        DaemonEndpoint::Url(url.clone())
    } else {
        let socket = args.socket.unwrap_or_else(brainrouter::config::default_socket_path);
        if !socket.exists() {
            bail!(
                "Daemon socket not found at {} (is `brainrouter serve` running?).\n\
                 Use --socket <path> or --url http://127.0.0.1:9099 to point at a running daemon.",
                socket.display()
            );
        }
        DaemonEndpoint::Socket(socket)
    };

    let client = DaemonClient::new(endpoint);

    match args.command {
        CliCommand::Status => {
            print_json(&client.get_json("/api/service-health").await?);
        }
        CliCommand::Versions => {
            print_json(&client.get_json("/api/versions").await?);
        }
        CliCommand::Inference => {
            print_json(&client.get_json("/api/inference-status").await?);
        }
        CliCommand::Events => {
            print_json(&client.get_json("/api/routing-events").await?);
        }
        CliCommand::Stats => {
            print_json(&client.get_json("/api/routing-stats").await?);
        }
        CliCommand::Models { llama_swap } => {
            if llama_swap {
                print_json(&client.get_json("/api/models/llama-swap").await?);
            } else {
                print_json(&client.get_json("/v1/models").await?);
            }
        }
        CliCommand::Bonsai { action } => match action {
            BonsaiAction::Status => print_json(&client.get_json("/api/bonsai").await?),
            BonsaiAction::Enable => {
                let current = client.get_json("/api/bonsai").await?;
                if current.get("enabled").and_then(|e| e.as_bool()).unwrap_or(false) {
                    print_json(&json!({ "enabled": true, "message": "already running" }));
                } else {
                    print_json(&client.post_json("/api/bonsai/toggle", json!({})).await?);
                }
            }
            BonsaiAction::Disable => {
                let current = client.get_json("/api/bonsai").await?;
                if current.get("enabled").and_then(|e| e.as_bool()).unwrap_or(false) {
                    print_json(&client.post_json("/api/bonsai/toggle", json!({})).await?);
                } else {
                    print_json(&json!({ "enabled": false, "message": "already stopped" }));
                }
            }
            BonsaiAction::Toggle => {
                print_json(&client.post_json("/api/bonsai/toggle", json!({})).await?);
            }
        },
        CliCommand::Nudge { action } => {
            // Read-modify-write for on/off/tier so CLI stays idempotent.
            let current = client.get_json("/api/nudge").await?;
            let enabled = current.get("enabled").and_then(|e| e.as_bool()).unwrap_or(false);
            let tier = current.get("tier").and_then(|t| t.as_str()).unwrap_or("auto").to_string();

            match action {
                NudgeAction::Status => print_json(&current),
                NudgeAction::Enable => {
                    let resp = client
                        .post_json("/api/nudge", json!({ "enabled": true, "tier": tier }))
                        .await?;
                    print_json(&resp);
                }
                NudgeAction::Disable => {
                    let resp = client
                        .post_json("/api/nudge", json!({ "enabled": false, "tier": tier }))
                        .await?;
                    print_json(&resp);
                }
                NudgeAction::Toggle => {
                    let resp = client
                        .post_json("/api/nudge", json!({ "enabled": !enabled, "tier": tier }))
                        .await?;
                    print_json(&resp);
                }
                NudgeAction::Tier { tier: t } => {
                    let resp = client
                        .post_json("/api/nudge", json!({ "enabled": enabled, "tier": t }))
                        .await?;
                    print_json(&resp);
                }
            }
        }
        CliCommand::PromptRewrite { action } => match action {
            PromptRewriteAction::Status => {
                print_json(&client.get_json("/api/prompt-rewrite").await?);
            }
            PromptRewriteAction::Enable => {
                print_json(&client.post_json("/api/prompt-rewrite", json!({ "enabled": true })).await?);
            }
            PromptRewriteAction::Disable => {
                print_json(&client.post_json("/api/prompt-rewrite", json!({ "enabled": false })).await?);
            }
        },
        CliCommand::Context { action } => match action {
            ContextAction::Status => {
                print_json(&client.get_json("/api/context").await?);
            }
            ContextAction::Set { value } => {
                print_json(&client.post_json("/api/context", json!({ "value": value })).await?);
            }
        },
        CliCommand::RoutingMode { action } => match action {
            RoutingModeAction::Status => {
                print_json(&client.get_json("/api/routing-mode").await?);
            }
            RoutingModeAction::Set { mode } => {
                print_json(&client.post_json("/api/routing-mode", json!({ "mode": mode })).await?);
            }
        },
        CliCommand::RoutingProfile { action } => {
            match action {
                RoutingProfileAction::Status => print_json(&client.get_json("/api/routing-profile").await?),
                RoutingProfileAction::Models => print_json(&client.get_json("/api/routing-models").await?),
                action => {
                    let mut profile: RoutingProfile = if let RoutingProfileAction::Set { ref path } = action {
                        let text = if path == "-" {
                            use std::io::Read;
                            let mut text = String::new();
                            std::io::stdin().read_to_string(&mut text)?;
                            text
                        } else { std::fs::read_to_string(path)? };
                        serde_json::from_str(&text)?
                    } else {
                        serde_json::from_value(client.get_json("/api/routing-profile").await?["profile"].clone())?
                    };
                    match action {
                        RoutingProfileAction::Preset { preset, main_model } => {
                            if let Some(model) = main_model {
                                if preset != RoutingPreset::LocalCustom { bail!("--main-model requires local_custom"); }
                                profile.main = ModelChoice::Local { model: Some(model) };
                            }
                            profile.apply_preset(preset)?;
                        }
                        RoutingProfileAction::Choose { role, backend, model } => {
                            let choice = ModelChoice::from_legacy(&backend, model)?;
                            if role == "main" { profile.main = choice; } else { profile.reviewer = choice; }
                            profile.preset = RoutingPreset::Custom;
                        }
                        RoutingProfileAction::Pool { model } => profile.subagent_model = model,
                        _ => {}
                    }
                    profile.validate()?;
                    print_json(&client.post_json("/api/routing-profile", serde_json::to_value(profile)?).await?);
                }
            }
        }
        CliCommand::Plugin { action } => run_plugin(&client, action).await?,
        CliCommand::Migration { action } => run_migration(action)?,
        CliCommand::Toolboxes => {
            print_json(&client.get_json("/api/toolboxes").await?);
        }
        CliCommand::Restart { service } => {
            let path = format!("/api/restart/{}", service);
            print_json(&client.post_json(&path, json!({})).await?);
        }
        CliCommand::Upgrade { component } => {
            let path = format!("/api/upgrade/{}", component);
            print_json(&client.post_json(&path, json!({})).await?);
        }
        CliCommand::FlushModels => {
            print_json(&client.post_json("/api/models/flush", json!({})).await?);
        }
        CliCommand::SyncOmp => {
            print_json(&client.post_json("/api/models/sync-omp", json!({})).await?);
        }
        CliCommand::ConfigFiles => {
            print_json(&client.get_json("/api/config-files").await?);
        }
        CliCommand::BrainrouterConfig { action } => {
            run_config_cmd(&client, action, "/api/config", "brainrouter.yaml").await?;
        }
        CliCommand::LlamaSwapConfig { action } => {
            run_config_cmd(&client, action, "/api/llama-swap-config", "llama-swap config.yaml").await?;
        }
        CliCommand::ReviewConfig { action } => match action {
            ReviewConfigAction::Status => {
                print_json(&client.get_json("/api/review-config").await?);
            }
            ReviewConfigAction::Update { max_iterations, forced_mode, forced_model, clear_model } => {
                // Merge into the current config — the daemon endpoint replaces it.
                let current = client.get_json("/api/review-config").await?;
                let mut update = current.clone();
                if let Some(v) = max_iterations {
                    update["max_iterations"] = json!(v);
                }
                if let Some(v) = forced_mode {
                    update["forced_mode"] = json!(v);
                }
                if let Some(v) = forced_model {
                    update["forced_model"] = json!(v);
                }
                if clear_model { update["forced_model"] = Value::Null; }
                print_json(&client.post_json("/api/review-config", update).await?);
            }
        },
        CliCommand::Review { action } => match action {
            ReviewAction::List => {
                print_json(&client.get_json("/review/api/sessions").await?);
            }
            ReviewAction::Get { session_id } => {
                validate_session_id(&session_id)?;
                let path = format!("/review/api/sessions/{}", session_id);
                print_json(&client.get_json(&path).await?);
            }
            ReviewAction::Request { task_id, summary, details, cwd, async_ } => {
                // The daemon needs a real project directory to read the PRD and
                // git diff. Default to the CLI's own cwd — peer-cwd resolution
                // can't see a UDS client's directory.
                let cwd = cwd.or_else(|| {
                    std::env::current_dir()
                        .ok()
                        .map(|p| p.to_string_lossy().into_owned())
                });
                let body = json!({
                    "taskId": task_id,
                    "summary": summary,
                    "details": details,
                    "cwd": cwd,
                });
                let resp = client.post_json("/review/api/request-async", body).await?;
                if resp.get("status").and_then(|v| v.as_str()) == Some("disabled") {
                    // FR-A: the reviewer is switched off — terminal `disabled`,
                    // with no session to poll. Print and stop.
                    print_json(&resp);
                } else if async_ {
                    print_json(&resp);
                } else {
                    let session_id = resp
                        .get("sessionId")
                        .and_then(|v| v.as_str())
                        .context("request-async response missing sessionId")?
                        .to_string();
                    validate_session_id(&session_id)?;
                    print_review_progress(&client, &session_id).await?;
                }
            }
            ReviewAction::Continue { session_id } => {
                validate_session_id(&session_id)?;
                print_json(&client
                    .post_json("/review/api/continue", json!({ "sessionId": session_id }))
                    .await?);
            }
            ReviewAction::Lgtm { session_id } => {
                validate_session_id(&session_id)?;
                print_json(&client
                    .post_json("/review/api/lgtm", json!({ "sessionId": session_id }))
                    .await?);
            }
            ReviewAction::Resolve { session_id, feedback } => {
                validate_session_id(&session_id)?;
                print_json(&client
                    .post_json("/review/api/resolve", json!({ "sessionId": session_id, "feedback": feedback }))
                    .await?);
            }
        },
    }

    Ok(())
}

/// Shared handler for the two YAML config commands.
async fn run_config_cmd(client: &DaemonClient, action: ConfigAction, api_path: &str, label: &str) -> Result<()> {
    match action {
        ConfigAction::Show => {
            let yaml = client.get_raw(api_path).await?;
            println!("{}", yaml);
        }
        ConfigAction::Set { path } => {
            let contents = if path == "-" {
                use std::io::Read;
                let mut buf = String::new();
                std::io::stdin().read_to_string(&mut buf).context("Failed to read stdin")?;
                buf
            } else {
                std::fs::read_to_string(&path)
                    .with_context(|| format!("Failed to read {} from {}", label, path))?
            };
            let resp = client.post_raw(api_path, &contents).await?;
            println!("{}", resp);
        }
    }
    Ok(())
}

/// Poll a review session until it reaches a terminal status, printing updates.
/// Mirrors the MCP thin client's behaviour: request-async, then poll every 5 s.
async fn print_review_progress(client: &DaemonClient, session_id: &str) -> Result<()> {
    const TERMINAL: &[&str] = &["approved", "needs_revision", "escalated", "failed"];
    const POLL_INTERVAL_SECS: u64 = 5;
    const MAX_POLLS: u32 = 360; // 30 minutes

    let path = format!("/review/api/sessions/{}", session_id);
    eprintln!("Review session {} started; polling every {}s…", session_id, POLL_INTERVAL_SECS);

    for _ in 0..MAX_POLLS {
        tokio::time::sleep(std::time::Duration::from_secs(POLL_INTERVAL_SECS)).await;
        let detail = client.get_json(&path).await?;
        let status = detail.get("status").and_then(|v| v.as_str()).unwrap_or("");

        if TERMINAL.contains(&status) {
            let feedback = detail
                .get("llm_feedback")
                .or_else(|| detail.get("human_feedback"))
                .unwrap_or(&Value::Null);
            print_json(&json!({
                "status": status,
                "feedback": feedback,
                "sessionId": session_id,
                "iterationCount": detail.get("iteration_count").unwrap_or(&Value::Null),
                "reviewerType": detail.get("reviewer_type").unwrap_or(&Value::Null),
            }));
            return Ok(());
        }
        debug!(status = %status, session = %session_id, "Review still in progress");
    }

    bail!(
        "Review timed out after {} minutes. Session {} is still in progress; poll `brainrouter cli review get {}`.",
        MAX_POLLS as u64 * POLL_INTERVAL_SECS / 60,
        session_id,
        session_id
    )
}

/// Validate a session ID before using it in a URL path (path-traversal guard).
fn validate_session_id(session_id: &str) -> Result<()> {
    if !session_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        bail!("Invalid sessionId: must be alphanumeric, dashes, or underscores");
    }
    Ok(())
}

/// v2 L2 `plugin` verbs. Thin client over the daemon; when the daemon is
/// unreachable, install/list/enable/disable/remove operate directly on the
/// plugin state (the supervisor picks enabled plugins up at next boot).
async fn run_plugin(client: &DaemonClient, action: PluginAction) -> Result<()> {
    use PluginAction::*;
    let daemon_live = client.get_json("/health").await.is_ok();
    if daemon_live {
        match action {
            Install { dir } => {
                print_json(&client.post_json("/api/plugins/install", json!({ "dir": dir })).await?);
            }
            List => print_json(&client.get_json("/api/plugins").await?),
            Enable { id } => {
                print_json(&client.post_json("/api/plugins/enable", json!({ "id": id })).await?)
            }
            Disable { id } => {
                print_json(&client.post_json("/api/plugins/disable", json!({ "id": id })).await?)
            }
            Remove { id } => {
                print_json(&client.post_json("/api/plugins/remove", json!({ "id": id })).await?)
            }
            Status { id } => {
                let all = client.get_json("/api/plugins").await?;
                print_plugin_status(&all, id.as_deref())?;
            }
            Config { action } => match action {
                PluginConfigAction::Set { id, assignments } => {
                    for pair in assignments {
                        let (key, value) = parse_kv(&pair)?;
                        print_json(
                            &client
                                .post_json("/api/plugins/config", json!({ "id": id, "key": key, "value": value }))
                                .await?,
                        );
                    }
                }
                PluginConfigAction::Show { id } => {
                    let all = client.get_json("/api/plugins").await?;
                    print_json(&plugin_field(&all, &id, "config"));
                }
            }
        }
        return Ok(());
    }
    // Unreachable socket mid-call: direct-state management.
    drop(client);
    run_plugin_direct(action).await
}

fn parse_kv(pair: &str) -> Result<(String, Value)> {
    let (key, raw) = pair
        .split_once('=')
        .with_context(|| format!("expected key=value, got {pair:?}"))?;
    if key.trim().is_empty() {
        bail!("empty config key in {pair:?}");
    }
    let value = serde_json::from_str(raw).unwrap_or_else(|_| json!(raw));
    Ok((key.to_string(), value))
}

fn plugin_field(all: &Value, id: &str, field: &str) -> Value {
    all.get("plugins")
        .and_then(Value::as_array)
        .and_then(|arr| arr.iter().find(|p| p.get("id").and_then(Value::as_str) == Some(id)))
        .and_then(|p| p.get(field))
        .cloned()
        .unwrap_or(Value::Null)
}

fn print_plugin_status(all: &Value, id: Option<&str>) -> Result<()> {
    let plugins = all.get("plugins").and_then(Value::as_array).cloned().unwrap_or_default();
    match id {
        Some(id) => {
            let entry = plugins
                .iter()
                .find(|p| p.get("id").and_then(Value::as_str) == Some(id))
                .with_context(|| format!("plugin '{id}' is not installed"))?;
            let state = entry.get("state").and_then(Value::as_str).unwrap_or("stopped");
            println!(
                "{id}: state={state} enabled={} version={} grants={} pid={} status={}",
                entry.get("enabled").and_then(Value::as_bool).unwrap_or(false),
                entry.get("version").and_then(Value::as_str).unwrap_or("?"),
                entry.get("grants").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(",")).unwrap_or_default(),
                entry.get("pid").and_then(Value::as_u64).map(|p| p.to_string()).unwrap_or("-".into()),
                entry.get("status_line").and_then(Value::as_str).unwrap_or(""),
            );
        }
        None => {
            if plugins.is_empty() {
                println!("no plugins installed");
            }
            for p in &plugins {
                println!(
                    "{:<16} {:<9} enabled={:<5} v{}",
                    p.get("id").and_then(Value::as_str).unwrap_or("?"),
                    p.get("state").and_then(Value::as_str).unwrap_or("stopped"),
                    p.get("enabled").and_then(Value::as_bool).unwrap_or(false).to_string(),
                    p.get("version").and_then(Value::as_str).unwrap_or("?"),
                );
            }
        }
    }
    Ok(())
}

/// Top-level `brainrouter plugin ...` (Hank ruling #4: the plugin verbs ship
/// at the top level; `brainrouter cli plugin ...` remains the daemon-client
/// spelling). Same handling as the `cli plugin` subcommand.
#[derive(Args)]
pub struct PluginCliArgs {
    /// Path to the daemon's Unix domain socket.
    #[arg(long, global = true)]
    pub socket: Option<PathBuf>,
    /// Talk to the daemon over TCP instead of the Unix socket.
    #[arg(long, global = true)]
    pub url: Option<String>,
    /// Config file (accepted for symmetry with `serve`; used for the
    /// daemon-less path).
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,
    #[command(subcommand)]
    pub action: PluginAction,
}

/// Top-level `brainrouter migration ...`.
#[derive(Args)]
pub struct MigrationArgs {
    /// Path to the YAML config file (restore target).
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[command(subcommand)]
    pub action: MigrationAction,
}

pub async fn run_plugin_standalone(args: PluginCliArgs) -> Result<()> {
    // The daemon-client path requires a live socket; when the daemon is not
    // running (socket absent / unconnectable), operate directly on the plugin
    // state (install/list/status/config + registry-only enable/disable).
    if daemon_reachable(args.socket.as_deref(), args.url.as_deref()) {
        let cli_args = CliArgs {
            socket: args.socket,
            url: args.url,
            command: CliCommand::Plugin { action: args.action },
        };
        return run(cli_args).await;
    }
    run_plugin_direct(args.action).await
}

/// Probe: is a daemon accepting at the endpoint (default socket when none
/// given)? A UDS connect on a live listener succeeds instantly.
fn daemon_reachable(socket: Option<&std::path::Path>, url: Option<&str>) -> bool {
    use std::os::unix::net::UnixStream;
    if let Some(u) = url {
        return TcpStream::connect(u.trim_start_matches("http://").trim_end_matches('/'))
            .is_ok();
    }
    let path = socket
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(brainrouter::config::default_socket_path);
    std::thread::spawn(move || UnixStream::connect(&path).is_ok())
        .join()
        .unwrap_or(false)
}

/// The daemon-less `plugin` verbs: direct state management. Enable/disable
/// persist the registry flag ONLY (I1: the supervisor is the daemon's job).
async fn run_plugin_direct(action: PluginAction) -> Result<()> {
    use PluginAction::*;
    let creds = Arc::new(
        brainrouter::credentials::CredentialStore::load(
            brainrouter::credentials::CredentialStore::default_path(),
        )
        .context("loading credentials store")?,
    );
    let manager = PluginManager::new(creds).context("opening the plugin registry")?;
    match action {
        Install { dir } => print_json(&manager.install(std::path::Path::new(&dir))?),
        List => print_json(&json!({ "plugins": manager.list()? })),
        Enable { id } => print_json(&manager.set_enabled_registry_only(&id, true)?),
        Disable { id } => print_json(&manager.set_enabled_registry_only(&id, false)?),
        Remove { id } => print_json(&manager.remove(&id).await?),
        Status { id } => print_plugin_status(&json!({ "plugins": manager.list()? }), id.as_deref())?,
        Config { action } => match action {
            PluginConfigAction::Set { id, assignments } => {
                for pair in assignments {
                    let (key, value) = parse_kv(&pair)?;
                    print_json(&manager.config_set(&id, &key, value)?);
                }
            }
            PluginConfigAction::Show { id } => print_json(&manager.config_get_full(&id)?),
        },
    }
    Ok(())
}

pub fn run_migration_standalone(args: MigrationArgs) -> Result<()> {
    match &args.action {
        MigrationAction::Restore { phase } => {
            let config_path = args
                .config
                .clone()
                .unwrap_or_else(brainrouter::config::default_config_path);
            let store = brainrouter::credentials::CredentialStore::load(
                brainrouter::credentials::CredentialStore::default_path(),
            )
            .context("loading credentials store")?;
            let msg = brainrouter::migration::restore(
                phase,
                &config_path,
                &store,
                &brainrouter::paths::brainrouter_state(),
            )?;
            println!("{msg}");
            Ok(())
        }
    }
}

/// v2 L2 `migration` verbs (always direct state operations).
fn run_migration(action: MigrationAction) -> Result<()> {
    match action {
        MigrationAction::Restore { phase } => {
            let config_path = brainrouter::config::default_config_path();
            let store = brainrouter::credentials::CredentialStore::load(
                brainrouter::credentials::CredentialStore::default_path(),
            )
            .context("loading credentials store")?;
            let msg = brainrouter::migration::restore(
                &phase,
                &config_path,
                &store,
                &brainrouter::paths::brainrouter_state(),
            )?;
            println!("{msg}");
        }
    }
    Ok(())
}

/// Pretty-print a JSON value to stdout.
fn print_json(value: &Value) {
    match serde_json::to_string_pretty(value) {
        Ok(s) => println!("{}", s),
        Err(_) => println!("{}", value),
    }
}
