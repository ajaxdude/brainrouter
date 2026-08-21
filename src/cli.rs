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
    /// Discord/Signal bridge status / toggle
    Bridges {
        #[command(subcommand)]
        action: BridgesAction,
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
pub enum BridgesAction {
    /// Current bridge status
    Status,
    /// Enable a bridge
    Enable {
        #[arg(value_parser = clap::builder::PossibleValuesParser::new(
            ["discord", "signal"]
        ))]
        bridge: String,
    },
    /// Disable a bridge
    Disable {
        #[arg(value_parser = clap::builder::PossibleValuesParser::new(
            ["discord", "signal"]
        ))]
        bridge: String,
    },
    /// Start if stopped, stop if running
    Toggle {
        #[arg(value_parser = clap::builder::PossibleValuesParser::new(
            ["discord", "signal"]
        ))]
        bridge: String,
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
        #[arg(long)]
        forced_mode: Option<String>,
        /// Forced model key (used when forced_mode is local)
        #[arg(long)]
        forced_model: Option<String>,
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
        CliCommand::Bridges { action } => match action {
            BridgesAction::Status => {
                print_json(&client.get_json("/api/bridge-status").await?);
            }
            BridgesAction::Enable { bridge } => {
                print_json(&client
                    .post_json("/api/bridges/toggle", json!({ "bridge": bridge, "enabled": true }))
                    .await?);
            }
            BridgesAction::Disable { bridge } => {
                print_json(&client
                    .post_json("/api/bridges/toggle", json!({ "bridge": bridge, "enabled": false }))
                    .await?);
            }
            BridgesAction::Toggle { bridge } => {
                // Read current state so toggle flips it, not just enables it.
                let current = client.get_json("/api/bridge-status").await?;
                let enabled = current
                    .get(&bridge)
                    .and_then(|b| b.get("enabled"))
                    .and_then(|e| e.as_bool())
                    .unwrap_or(false);
                print_json(&client
                    .post_json("/api/bridges/toggle", json!({ "bridge": bridge, "enabled": !enabled }))
                    .await?);
            }
        },
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
            ReviewConfigAction::Update { max_iterations, forced_mode, forced_model } => {
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
                if async_ {
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

/// Pretty-print a JSON value to stdout.
fn print_json(value: &Value) {
    match serde_json::to_string_pretty(value) {
        Ok(s) => println!("{}", s),
        Err(_) => println!("{}", value),
    }
}
