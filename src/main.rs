//! brainrouter entry point.
//!
//! Dispatches between two subcommands:
//!   `brainrouter serve`  — run the HTTP proxy daemon (default)
//!   `brainrouter mcp`    — run the MCP stdio server (thin client of daemon)

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod daemon;
mod install;
mod mcp_server;
#[derive(Parser)]
#[command(name = "brainrouter", about = "Bonsai-routed LLM failover proxy")]
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
    /// Install brainrouter into a coding harness config
    Install(install::InstallArgs),
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
        Command::Install(args) => {
            install::run(args)?;
            Ok(())
        }
    }
}
