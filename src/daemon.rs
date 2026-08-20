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
    bonsai_server::BonsaiControl,
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
    /// Defaults to ~/.config/brainrouter/brainrouter.yaml
    /// (or $XDG_CONFIG_HOME/brainrouter/brainrouter.yaml).
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// TCP listen address.
    #[arg(long, default_value = "127.0.0.1:9099")]
    pub tcp_addr: String,

    /// Unix domain socket path.
    /// Defaults to $XDG_RUNTIME_DIR/brainrouter.sock (or /run/brainrouter.sock).
    #[arg(long)]
    pub socket: Option<PathBuf>,
}

/// Entry point for `brainrouter serve`.
pub async fn run(args: ServeArgs) -> Result<()> {
    let socket = args.socket.unwrap_or_else(config::default_socket_path);
    let config_path = args.config.unwrap_or_else(config::default_config_path);

    // Config
    let config = config::load(&config_path).with_context(|| {
        format!("Failed to load config from {}", config_path.display())
    })?;

    let tcp_addr: std::net::SocketAddr = args
        .tcp_addr
        .parse()
        .with_context(|| format!("Invalid TCP address: {}", args.tcp_addr))?;

    info!(
        config_path = %config_path.display(),
        tcp_addr = %tcp_addr,
        uds_path = %socket.display(),
        manifest_url = %config.manifest.base_url,
        llama_swap_url = %config.llama_swap.base_url,
        fallback_model = %config.llama_swap.fallback_model,
        "Starting brainrouter daemon"
    );
    let bonsai_control = if config.bonsai.enabled {
        let model_path = config
            .bonsai
            .model_path
            .clone()
            .context("bonsai.enabled is true but bonsai.model_path is not set")?;
        BonsaiControl::start(
            config.bonsai.fork_path.clone(),
            model_path,
            config.bonsai.server_port,
        )
        .await
        .context("Failed to start Bonsai llama-server")?
    } else {
        warn!("Bonsai classifier is disabled (bonsai.enabled: false) — auto-routed requests go straight to llama-swap. Enable it with `brainrouter cli bonsai on` or set bonsai.enabled: true in brainrouter.yaml.");
        BonsaiControl::disabled(config.bonsai.server_port).await
    };

    // Nudge (per-request reasoning budget) runtime state — shared between the
    // classifier, the router, and the dashboard API.
    let nudge_enabled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(config.llama_swap.nudge.enabled));
    let nudge_tier = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)); // 0 = Bonsai picks
    let prompt_rewrite = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let context_value = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(131072));

    // Create classifier pointing at the external server
    let classifier = Classifier::new(
        bonsai_control.url(),
        config.llama_swap.fallback_model.clone(),
        bonsai_control.enabled(),
        Arc::clone(&nudge_enabled),
        config.llama_swap.nudge.model_key.clone(),
    );
    let classifier = Arc::new(classifier);
    info!("Bonsai classifier ready");

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
        manifest_enabled: config.manifest.enabled,
        llama_swap,
        fallback_model: config.llama_swap.fallback_model.clone(),
        local_models: config.llama_swap.local_models.clone(),
        health,
        routing_events: Arc::clone(&routing_events),
        local_system_prompt,
        inference_tracker: Arc::clone(&inference_tracker),
        nudge_budgets: config.llama_swap.nudge.budgets,
        nudge_enabled: Arc::clone(&nudge_enabled),
        nudge_tier: Arc::clone(&nudge_tier),
        prompt_rewrite: Arc::clone(&prompt_rewrite),
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

    let manifest_url = config.manifest.base_url
        .trim_end_matches('/')
        .strip_suffix("/v1")
        .unwrap_or(&config.manifest.base_url)
        .to_string();

    // Bridge manager (status tracking for Discord/Signal transports)
    let bridge_manager = Arc::new(brainrouter::bridge::BridgeManager::new());

    // Seed with an empty object, not null: /api/versions and `brainrouter cli
    // versions` read this before the first compute completes (up to ~15 s).
    let (versions_tx, versions_rx) = tokio::sync::watch::channel(serde_json::json!({}));
    let versions_tx = Arc::new(versions_tx);

    let state = Arc::new(AppState {
        router,
        session_manager,
        review_service,
        routing_events,
        llama_swap_url,
        manifest_url,
        bridge_manager: Arc::clone(&bridge_manager),
        bonsai: Arc::new(bonsai_control),
        config_path: std::fs::canonicalize(&config_path).unwrap_or(config_path.clone()),
        llama_swap_config_path: {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
            let p = std::path::PathBuf::from(format!("{}/.config/llama-swap/config.yaml", home));
            std::fs::canonicalize(&p).unwrap_or(p)
        },
        tcp_addr: tcp_addr.to_string(),
        manifest_enabled: config.manifest.enabled,
        routing_mode: std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)),
        versions_cache: Arc::new(versions_rx),
        nudge_enabled,
        nudge_tier,
        nudge_model_key: config.llama_swap.nudge.model_key.clone(),
        nudge_budgets: config.llama_swap.nudge.budgets,
        prompt_rewrite,
        context_value,
    });

    // Background task: compute versions once, then refresh every 30 minutes.
    {
        let tx = Arc::clone(&versions_tx);
        tokio::spawn(async move {
            let data = server::compute_versions_json(&config.bonsai.fork_path).await;
            let _ = tx.send_replace(data);
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(30 * 60)).await;
                let data = server::compute_versions_json(&config.bonsai.fork_path).await;
                let _ = tx.send_replace(data);
            }
        });
    }

    // Start bridge transports if configured
    if let Some(ref bridge_config) = config.bridge {
        let bm = Arc::clone(&bridge_manager);
        let bc = bridge_config.clone();
        tokio::spawn(async move {
            brainrouter::bridge::start(bc, bm).await;
        });
    }

    // Auto-sync OMP models.yml from live llama-swap model list.
    // Runs in background — non-fatal if llama-swap is not yet up.
    {
        let ls_url = state.llama_swap_url.clone();
        let own_addr = tcp_addr.to_string();
        tokio::spawn(async move {
            // Give llama-swap a moment to come up after boot.
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            match server::sync_omp_models(&ls_url, &own_addr).await {
                Ok(n) => tracing::info!(model_count = n, "Auto-synced OMP models.yml on startup"),
                Err(e) => tracing::warn!(error = %e, "Failed to auto-sync OMP models.yml (llama-swap may not be ready)"),
            }
        });
    }

    // Server (TCP + UDS)
    server::run(tcp_addr, socket, state).await?;

    Ok(())
}
