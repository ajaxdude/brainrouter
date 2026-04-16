use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod config;
mod server;
mod types;
mod provider;

#[derive(clap::Parser)]
#[command(name = "brainrouter", about = "LLM failover proxy daemon")]
struct Args {
    /// Path to config file
    #[arg(short, long, default_value = "brainrouter.yaml")]
    config: PathBuf,

    /// TCP listen address
    #[arg(long, default_value = "127.0.0.1:9099")]
    tcp_addr: String,

    /// Unix socket path
    #[arg(long, default_value = "/run/brainrouter.sock")]
    socket: PathBuf,

    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,
}


#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing subscriber with env filter (RUST_LOG overrides --log-level)
    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(&args.log_level))
        .context("Invalid log level")?;

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config_path = &args.config;
    let config = config::load(config_path)
        .with_context(|| format!("Failed to load config from {}", config_path.display()))?;

    // Parse TCP address early to fail fast
    let tcp_addr = args
        .tcp_addr
        .parse()
        .with_context(|| format!("Invalid TCP address: {}", args.tcp_addr))?;

    // Log startup information
    info!(
        config_path = %config_path.display(),
        tcp_addr = %tcp_addr,
        uds_path = %args.socket.display(),
        providers = config.providers.len(),
        models = config.models.len(),
        "Starting brainrouter daemon"
    );

    // Create shared application state
    let state = Arc::new(server::AppState {});

    // Start the server (handles both TCP and Unix socket listeners)
    server::run(tcp_addr, args.socket, state).await?;

    Ok(())
}
