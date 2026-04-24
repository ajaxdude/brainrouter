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
    inference_state::InferenceTracker,
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

    // Operational safety: Validate toolbox restart script path if configured
    if let Some(ref script) = config.llama_swap.llama_cpp_restart_script {
        let path = std::path::Path::new(script);
        if !path.exists() {
            warn!(path = %script, "llama_cpp_restart_script path does not exist; toolbox restart will fail");
        } else {
            // Executable bit check is Unix-only; on other platforms only path existence is verified.
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                if let Ok(metadata) = std::fs::metadata(path) {
                    if metadata.mode() & 0o111 == 0 {
                        warn!(path = %script, "llama_cpp_restart_script is not executable; toolbox restart will fail");
                    }
                }
            }
        }
    }

    // Warn if $HOME is unset — upgrade and version-check paths fall back to /root.
    if std::env::var("HOME").is_err() {
        warn!("$HOME is not set; upgrade paths will fall back to /root. Set HOME in the service environment if running as a non-root system user.");
    }

    // Validate BRAINROUTER_MANIFEST_DIR if the operator set it, so misconfiguration
    // fails fast at startup rather than producing a confusing error at upgrade time.
    if let Ok(dir) = std::env::var("BRAINROUTER_MANIFEST_DIR") {
        let p = std::path::Path::new(&dir);
        if !p.exists() {
            warn!(path = %dir, "BRAINROUTER_MANIFEST_DIR does not exist; Manifest upgrade will fail");
        } else if !p.join("docker-compose.yml").exists() && !p.join("docker-compose.yaml").exists() {
            warn!(path = %dir, "BRAINROUTER_MANIFEST_DIR has no docker-compose.yml; Manifest upgrade will fail");
        }
    }

    // Inference state tracker — shared between Router (writes) and HTTP API (reads)
    let inference_tracker = Arc::new(InferenceTracker::new());

    // Router — shared between the proxy and the review service
    let router = Arc::new(Router::new(brainrouter::router::RouterArgs {
        classifier,
        manifest,
        llama_swap,
        fallback_model: config.llama_swap.fallback_model.clone(),
        health,
        routing_events: Arc::clone(&routing_events),
        local_system_prompt,
        inference_tracker: Arc::clone(&inference_tracker),
    }));

    // Session manager (in-memory; ephemeral per process lifetime)
    let session_manager = Arc::new(SessionManager::new());

    // Review service
    let review_config = config.review.clone();
    let review_service = Arc::new(ReviewService::new(
        Arc::clone(&router),
        Arc::clone(&session_manager),
        review_config,
    ));

    let llama_swap_url = config.llama_swap.base_url
        .trim_end_matches('/')
        .strip_suffix("/v1")
        .unwrap_or(&config.llama_swap.base_url)
        .to_string();

    let state = Arc::new(AppState {
        router,
        session_manager,
        review_service,
        routing_events,
        llama_swap_url,
        llama_cpp_restart_script: config.llama_swap.llama_cpp_restart_script,
    });

    // Server (TCP + UDS)
    server::run(tcp_addr, args.socket, state).await?;

    Ok(())
}
