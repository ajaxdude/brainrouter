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
    /// Role cards (v2 L3): the single model config — `get` prints
    /// `{assignments, policy}`, `set` assigns a selector to a role card
    /// (mutations go through the running daemon; reads direct-state when down).
    Roles {
        #[command(subcommand)]
        action: RolesAction,
    },
    /// Decision-model slot (v2 L3): `show` describes the slot, `test` runs
    /// one feature set through it.
    Decision {
        #[command(subcommand)]
        action: DecisionAction,
    },
    /// Credentials vault (v2 L3 UI surface): list/add/remove/test; mutations
    /// go through the running daemon, values are never printed.
    Credential {
        #[command(subcommand)]
        action: CredentialAction,
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
pub enum RolesAction {
    /// Print all card assignments + policy (reads via daemon; direct-state fallback)
    Get,
    /// Assign a selector to a role card: `roles set <main|review|subagent> <selector> [--project <path>]`
    Set {
        role: String,
        assignment: String,
        /// Project scope (git root); omit for the global card
        #[arg(long)]
        project: Option<String>,
    },
    /// Remove a card assignment (falls back to the next scope / default)
    Remove {
        role: String,
        #[arg(long)]
        project: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum DecisionAction {
    /// Describe the configured slot ({enabled, kind, available, min_confidence, ...})
    Show,
    /// Run one prompt through the slot and print the full Decision JSON
    Test { text: String },
}

#[derive(Subcommand)]
pub enum CredentialAction {
    /// List credentials (masked; values are never printed)
    List,
    /// Add/rotate a credential: `credential add <id> --kind store|env|file [--value V | --env VAR | --file PATH] [--grants a,b]`
    Add {
        id: String,
        #[arg(long, value_parser = ["store", "env", "file"])]
        kind: String,
        #[arg(long)]
        value: Option<String>,
        #[arg(long)]
        env: Option<String>,
        #[arg(long)]
        file: Option<String>,
        #[arg(long, value_delimiter = ',')]
        grants: Vec<String>,
        #[arg(long)]
        description: Option<String>,
    },
    /// Remove a credential
    Remove { id: String },
    /// Resolve-test a credential ({ok: bool}) — the value is never printed
    Test { id: String },
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
        // A daemon that is mid-boot (migration, plugin registry, model
        // resolution) has not reached the accept loop; a CLI invoked in that
        // window sees no socket file. Poll before declaring the daemon
        // absent (the gate's black-box tests call CLI verbs the instant
        // `/health` answers; under parallel load boot takes seconds).
        let poll = std::time::Duration::from_millis(50);
        let deadline = std::time::Instant::now()
            + std::time::Duration::from_millis(
                std::env::var("BRAINROUTER_CLI_SOCKET_WAIT_MS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(10_000),
            );
        while !socket.exists() && std::time::Instant::now() < deadline {
            std::thread::sleep(poll);
        }
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
        CliCommand::Roles { action } => run_roles(&client, action).await?,
        CliCommand::Decision { action } => run_decision(&client, action).await?,
        CliCommand::Credential { action } => run_credential(&client, action).await?,
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
            ReviewConfigAction::Update { max_iterations } => {
                // v2 L3: the reviewer MODEL lives on the review card
                // (`roles set review <selector>`); this config carries the
                // surviving extras only.
                let current = client.get_json("/api/review-config").await?;
                let mut update = current.clone();
                if let Some(v) = max_iterations {
                    update["max_iterations"] = json!(v);
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
/// v2 L3 role-card verbs. Mutations go through the daemon (single-writer
/// store); `get` falls back to a direct store read when the daemon is down.
async fn run_roles(client: &DaemonClient, action: RolesAction) -> Result<()> {
    match action {
        RolesAction::Get => {
            match client.get_json("/api/roles").await {
                Ok(v) => print_json(&v),
                Err(_) => {
                    // Daemon down: reads are direct (contract).
                    let path = brainrouter::roles::RoleCardStore::default_path();
                    let store = brainrouter::roles::RoleCardStore::load(path);
                    print_json(&json!({
                        "assignments": store.assignments_snapshot(),
                        "policy": store.policy_snapshot(),
                        "review_extras": store.review_extras(),
                    }));
                }
            }
        }
        RolesAction::Set { role, assignment, project } => {
            // Client-side validation mirrors the daemon's (typos never reach
            // the wire, matching v1 `routing-profile choose` behavior).
            brainrouter::roles::Role::parse(&role)?;
            brainrouter::roles::validate_card_assignment(&assignment)?;
            let scope = project.unwrap_or_else(|| "global".into());
            print_json(&client.post_json(
                "/api/roles/set",
                json!({ "role": role, "assignment": assignment, "scope": scope }),
            ).await?);
        }
        RolesAction::Remove { role, project } => {
            let scope = project.unwrap_or_else(|| "global".into());
            print_json(&client.post_json(
                "/api/roles/remove",
                json!({ "role": role, "scope": scope }),
            ).await?);
        }
    }
    Ok(())
}

async fn run_decision(client: &DaemonClient, action: DecisionAction) -> Result<()> {
    match action {
        DecisionAction::Show => match client.get_json("/api/decision").await {
            Ok(v) => print_json(&v),
            Err(_) => {
                // Daemon down: reads are direct — describe the config.
                let config_path = brainrouter::config::default_config_path();
                let config = brainrouter::config::load(&config_path)
                    .context("loading config for decision show")?;
                let slot = brainrouter::decision::DecisionSlot::new(
                    config.decision_model,
                    std::sync::Arc::new(brainrouter::classifier::Classifier::new(
                        "http://127.0.0.1:1".into(),
                        String::new(),
                        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                        None,
                    )),
                    None,
                );
                print_json(&slot.describe());
            }
        },
        DecisionAction::Test { text } => {
            print_json(&client.post_json("/api/decision/test", json!({ "text": text })).await?);
        }
    }
    Ok(())
}

async fn run_credential(client: &DaemonClient, action: CredentialAction) -> Result<()> {
    match action {
        CredentialAction::List => {
            print_json(&client.get_json("/api/credentials").await?);
        }
        CredentialAction::Add { id, kind, value, env, file, grants, description } => {
            let mut record = json!({ "id": id, "kind": kind, "grants": grants });
            if let Some(d) = description {
                record["description"] = json!(d);
            }
            match kind.as_str() {
                "store" => record["value"] = json!(value.context("--value is required for kind store")?),
                "env" => record["var"] = json!(env.context("--env is required for kind env")?),
                "file" => record["path"] = json!(file.context("--file is required for kind file")?),
                other => bail!("unknown credential kind {other:?}"),
            }
            print_json(&client.post_json("/api/credentials/add", record).await?);
        }
        CredentialAction::Remove { id } => {
            print_json(&client.post_json("/api/credentials/remove", json!({ "id": id })).await?);
        }
        CredentialAction::Test { id } => {
            print_json(&client.post_json("/api/credentials/test", json!({ "id": id })).await?);
        }
    }
    Ok(())
}

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
    // Unreachable socket mid-call: direct-state management (the client's
    // connections die with this frame at the tail expression's drop).
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
    /// Brainrouter state root (where `migrations/<phase>/` snapshots live).
    /// Default: `$XDG_STATE_HOME/brainrouter`.
    #[arg(long)]
    pub state: Option<PathBuf>,
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
            let config_dir = config_path
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(std::path::PathBuf::new);
            // State-root resolution: --state, then the env default, then a
            // snapshot probe against the config-adjacent XDG tree (so a
            // restore aimed at an isolated `--config` finds the snapshots
            // written by a daemon booted with that same isolated environment).
            let mut candidates: Vec<PathBuf> = Vec::new();
            if let Some(state) = &args.state {
                candidates.push(state.clone());
            }
            candidates.push(brainrouter::paths::brainrouter_state());
            let derived = [
                brainrouter::paths::env_dir("XDG_STATE_HOME", "", ".local/state"),
            ];
            for base in derived {
                let cand = base.join("brainrouter");
                if !candidates.contains(&cand) {
                    candidates.push(cand);
                }
            }
            // Config-adjacent probes: <config-dir>/../state/brainrouter and
            // <config-dir>/brainrouter-state (isolated-deployment layouts).
            for probe in [
                config_dir.join("state/brainrouter"),
                config_dir.join("../state/brainrouter"),
                config_dir.join("brainrouter-state"),
            ] {
                let probe = probe.canonicalize().unwrap_or(probe.clone());
                if !candidates.contains(&probe) {
                    candidates.push(probe);
                }
            }
            let has_snap = |root: &std::path::Path| -> bool {
                let dir = root.join("migrations").join(phase.as_str());
                dir.is_dir()
                    && std::fs::read_dir(dir)
                        .map(|mut r| r.next().is_some())
                        .unwrap_or(false)
            };
            let state_root = candidates
                .iter()
                .find(|root| has_snap(root))
                .cloned()
                .unwrap_or_else(brainrouter::paths::brainrouter_state);
            let store = brainrouter::credentials::CredentialStore::load(
                state_root.join("credentials.json"),
            )
            .context("loading credentials store")?;
            let msg = brainrouter::migration::restore(
                phase,
                &config_path,
                &store,
                &state_root,
                &config_dir,
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
            let config_dir = brainrouter::config::default_config_path()
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(std::path::PathBuf::new);
            let msg = brainrouter::migration::restore(
                &phase,
                &config_path,
                &store,
                &brainrouter::paths::brainrouter_state(),
                &config_dir,
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
