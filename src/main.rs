//! brainrouter entry point.
//!
//! Dispatches between two subcommands:
//!   `brainrouter serve`  — run the HTTP proxy daemon (default)
//!   `brainrouter mcp`    — run the MCP stdio server (thin client of daemon)

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::{ffi::OsString, net::SocketAddr, path::PathBuf};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod cli;
mod daemon;
mod install;
mod mcp_server;
#[derive(Parser)]
#[command(name = "brainrouter", version, about = "Bonsai-routed LLM failover proxy")]
struct Cli {
    /// Default log level if RUST_LOG is not set.
    #[arg(long, default_value = "info", global = true)]
    log_level: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the HTTP proxy daemon (TCP + UDS listeners)
    Serve(daemon::ServeArgs),
    /// Run the MCP stdio server (forwards tool calls to the daemon)
    Mcp(mcp_server::McpArgs),
    /// Headless CLI: control the running daemon (total dashboard parity)
    Cli(cli::CliArgs),
    /// Manage sidecar plugins (v2 L2). Thin client of the running daemon;
    /// falls back to direct state management when the daemon is not running.
    Plugin(cli::PluginCliArgs),
    /// Config migration controls (`migration restore <phase>`).
    Migration(cli::MigrationArgs),
    /// Credential owner commands that read the vault directly
    /// (`credential export --env-line <file>`).
    Credential(cli::CredentialDirectArgs),
    /// Swap supervisor control (v2 L4): status/ensure/evict/catalog/import/pin
    Swap(cli::SwapArgs),
    /// Platform probe (v2 L4): detected hardware, budgets, engine offers
    Platform(cli::PlatformArgs),
    /// Install brainrouter into a coding harness config
    Install(install::InstallArgs),
    /// Internal inference-only proxy used inside Benchmark Lab sandboxes.
    #[command(hide = true)]
    BenchmarkSandbox {
        #[arg(long)]
        socket: PathBuf,
        #[arg(long)]
        listen: SocketAddr,
        #[arg(trailing_var_arg = true, required = true)]
        command: Vec<OsString>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Tracing: MCP mode must write logs to stderr (stdout is the JSON-RPC channel).
    // Serve/Install modes can use stdout.
    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(&cli.log_level))
        .expect("Invalid log level");
    let fmt_layer = tracing_subscriber::fmt::layer().with_writer(std::io::stderr);
    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .init();

    match cli.command {
        Command::Serve(args) => daemon::run(args).await,
        Command::Mcp(args) => mcp_server::run(args).await,
        Command::Cli(args) => cli::run(args).await,
        Command::Plugin(args) => cli::run_plugin_standalone(args).await,
        Command::Migration(args) => cli::run_migration_standalone(args),
        Command::Credential(args) => cli::run_credential_direct(args),
        Command::Swap(args) => cli::run_swap_standalone(args).await,
        Command::Platform(args) => cli::run_platform_standalone(args),
        Command::Install(args) => {
            install::run(args)?;
            Ok(())
        }
        Command::BenchmarkSandbox {
            socket,
            listen,
            command,
        } => brainrouter::inference_proxy::run_sandbox_command(listen, socket, command).await,
    }
}
