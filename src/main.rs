//! brainrouter daemon entry point.

use anyhow::{Context, Result};
use brainrouter::{
    classifier::Classifier,
    config,
    health::HealthTracker,
    provider::openai::OpenAiProvider,
    router::Router,
    server::{self, AppState},
};
use clap::Parser;
use std::{path::PathBuf, sync::Arc};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[derive(clap::Parser)]
#[command(name = "brainrouter", about = "Bonsai-routed LLM failover proxy")]
struct Args {
    /// Path to the YAML config file.
    #[arg(short, long, default_value = "brainrouter.yaml")]
    config: PathBuf,

    /// TCP listen address.
    #[arg(long, default_value = "127.0.0.1:9099")]
    tcp_addr: String,

    /// Unix domain socket path.
    #[arg(long, default_value = "/run/brainrouter.sock")]
    socket: PathBuf,

    /// Default log level if RUST_LOG is not set.
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Tracing: RUST_LOG overrides --log-level
    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(&args.log_level))
        .context("Invalid log level")?;
    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Config
    let config = config::load(&args.config).with_context(|| {
        format!("Failed to load config from {}", args.config.display())
    })?;

    let tcp_addr: std::net::SocketAddr = args
        .tcp_addr
        .parse()
        .with_context(|| format!("Invalid TCP address: {}", args.tcp_addr))?;

    info!(
        config_path = %args.config.display(),
        tcp_addr = %tcp_addr,
        uds_path = %args.socket.display(),
        manifest_url = %config.manifest.base_url,
        llama_swap_url = %config.llama_swap.base_url,
        fallback_model = %config.llama_swap.fallback_model,
        "Starting brainrouter daemon"
    );

    // Load Bonsai classifier. This is blocking but only happens once at startup.
    info!("Loading Bonsai classifier from {}...", config.bonsai.model_path.display());
    let classifier = Classifier::new(
        &config.bonsai.model_path,
        config.llama_swap.fallback_model.clone(),
    )
    .context("Failed to initialize Bonsai classifier")?;
    let classifier = Arc::new(classifier);
    info!("Bonsai classifier loaded");

    // Manifest provider (cloud)
    let manifest_api_key = config.resolve_manifest_api_key();
    let manifest = Arc::new(OpenAiProvider::new(
        "manifest".to_string(),
        config.manifest.base_url.clone(),
        manifest_api_key,
    ));

    // llama-swap provider (local)
    let llama_swap = Arc::new(OpenAiProvider::new(
        "llama-swap".to_string(),
        config.llama_swap.base_url.clone(),
        None,
    ));

    // Health tracker (circuit breaker)
    let health = Arc::new(HealthTracker::new());

    // Router
    let router = Arc::new(Router::new(
        classifier,
        manifest,
        llama_swap,
        config.llama_swap.fallback_model.clone(),
        health,
    ));

    let state = Arc::new(AppState { router });

    // Server (TCP + UDS)
    server::run(tcp_addr, args.socket, state).await?;

    Ok(())
}
