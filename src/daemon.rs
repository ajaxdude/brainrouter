//! brainrouter daemon — the `serve` subcommand.
//!
//! Constructs all shared state (classifier, router, review service) and runs
//! the dual-listener HTTP server (TCP + UDS). Extracted from main.rs so that
//! main.rs can dispatch between `serve` and `mcp` subcommands without carrying
//! startup logic.

use anyhow::{Context, Result};
use clap::Args;
use std::{path::PathBuf, sync::Arc};
use tracing::{info, warn};

use brainrouter::{
    classifier::Classifier,
    config,
    health::HealthTracker,
    provider::openai::OpenAiProvider,
    review::ReviewService,
    router::Router,
    routing_events::RoutingEvents,
    server::{self, AppState},
    session::SessionManager,
};

/// Arguments for the `serve` subcommand.
#[derive(Args)]
pub struct ServeArgs {
    /// Path to the YAML config file.
    #[arg(short, long, default_value = "brainrouter.yaml")]
    pub config: PathBuf,

    /// TCP listen address.
    #[arg(long, default_value = "127.0.0.1:9099")]
    pub tcp_addr: String,

    /// Unix domain socket path.
    #[arg(long, default_value = "/run/brainrouter.sock")]
    pub socket: PathBuf,
}

/// Entry point for `brainrouter serve`.
pub async fn run(args: ServeArgs) -> Result<()> {
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

    // Load Bonsai classifier. Blocking but happens once at startup.
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

    // Routing event ring buffer — shared between Router (writes) and HTTP API (reads)
    let routing_events = Arc::new(RoutingEvents::new());

    // Load optional custom local system prompt
    let local_system_prompt = config
        .llama_swap
        .local_system_prompt
        .as_ref()
        .and_then(|path| match std::fs::read_to_string(path) {
            Ok(content) => {
                info!(path = %path, "Loaded custom local system prompt");
                Some(content)
            }
            Err(e) => {
                warn!(path = %path, error = %e, "Failed to load custom local system prompt, using built-in");
                None
            }
        });

    // Router — shared between the proxy and the review service
    let router = Arc::new(Router::new(
        classifier,
        manifest,
        llama_swap,
        config.llama_swap.fallback_model.clone(),
        health,
        Arc::clone(&routing_events),
        local_system_prompt,
    ));

    // Session manager (in-memory; ephemeral per process lifetime)
    let session_manager = Arc::new(SessionManager::new());

    // Review service
    let review_config = config.review.clone();
    let review_service = Arc::new(ReviewService::new(
        Arc::clone(&router),
        Arc::clone(&session_manager),
        review_config,
    ));

    let state = Arc::new(AppState {
        router,
        session_manager,
        review_service,
        routing_events,
    });

    // Server (TCP + UDS)
    server::run(tcp_addr, args.socket, state).await?;

    Ok(())
}
