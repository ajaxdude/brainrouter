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
    benchmark::BenchmarkStore,
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

/// Progress exposed by the active llama-server slot.
#[derive(Debug, Clone, PartialEq)]
struct SlotProgress {
    model: String,
    prefill_progress: Option<f64>,
    generated_tokens: u64,
    max_tokens: Option<u64>,
}

/// Poll llama-swap /running for the active llama-server proxy, then its /slots
/// array for prompt-prefill and generation progress.
async fn fetch_slots(client: &reqwest::Client, ls_url: &str) -> Option<SlotProgress> {
    let running = client.get(format!("{}/running", ls_url))
        .timeout(std::time::Duration::from_secs(3))
        .send().await.ok()?
        .json::<serde_json::Value>().await.ok()?;
    let active_model = running["running"][0]["model"].as_str().unwrap_or("").to_string();
    let proxy = running["running"][0]["proxy"].as_str()?.to_string();
    let slots = client.get(format!("{}/slots", proxy))
        .timeout(std::time::Duration::from_secs(3))
        .send().await.ok()?
        .json::<serde_json::Value>().await.ok()?;
    parse_slot_progress(&slots, active_model)
}

fn parse_slot_progress(slots: &serde_json::Value, model: String) -> Option<SlotProgress> {
    let arr = slots.as_array().or_else(|| slots.get("slots")?.as_array())?;
    let mut best: Option<(u64, SlotProgress)> = None;
    for slot in arr {
        let processing = slot.get("is_processing").and_then(|v| v.as_bool()).unwrap_or(false);
        let total = slot.get("n_prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        let processed = slot.get("n_prompt_processed")
            .or_else(|| slot.get("n_prompt_tokens_processed"))
            .and_then(|v| v.as_u64()).unwrap_or(0);
        let next_token = slot.get("next_token").and_then(|value| {
            if value.is_array() { value.as_array()?.first() } else { Some(value) }
        });
        let generated_tokens = slot.get("n_decoded")
            .and_then(|v| v.as_u64())
            .or_else(|| next_token.and_then(|value| value.get("n_decoded")).and_then(|v| v.as_u64()))
            .unwrap_or(0);
        if !processing && generated_tokens == 0 {
            continue;
        }
        let remaining = next_token
            .and_then(|value| value.get("n_remain"))
            .and_then(|v| v.as_i64())
            .filter(|value| *value >= 0)
            .map(|value| value as u64);
        let configured_max = slot.pointer("/params/n_predict")
            .and_then(|v| v.as_i64())
            .filter(|value| *value > 0)
            .map(|value| value as u64);
        let max_tokens = remaining.map(|value| generated_tokens.saturating_add(value)).or(configured_max);
        let prefill_progress = if generated_tokens == 0 && total > 0 {
            Some((processed as f64 / total as f64).clamp(0.0, 1.0))
        } else {
            None
        };
        let score = generated_tokens.saturating_mul(1_000_000).saturating_add(processed);
        let progress = SlotProgress {
            model: model.clone(),
            prefill_progress,
            generated_tokens,
            max_tokens,
        };
        if best.as_ref().map_or(true, |(best_score, _)| score > *best_score) {
            best = Some((score, progress));
        }
    }
    best.map(|(_, progress)| progress)
}

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

    let profiles = Arc::new(brainrouter::routing_profile::ProfileStore::load(
        config::default_config_path().with_file_name("routing_state.json"),
        config.routing_profile()?,
        &config.review,
    )?);
    let routing_mode = match profiles.profile().main.backend() { "cloud" => 1, "local" => 2, _ => 0 };
    let benchmark_store = BenchmarkStore::open(config.benchmarks.database_path.clone())
        .map(Arc::new)
        .map_err(|error| {
            warn!(
                path = %config.benchmarks.database_path.display(),
                error = %error,
                "Benchmark explorer unavailable; continuing core daemon startup"
            );
            error.to_string()
        });

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
        BonsaiControl::disabled(
            config.bonsai.fork_path.clone(),
            config.bonsai.model_path.clone(),
            config.bonsai.server_port,
        )
        .await
    };

    // Nudge (per-request reasoning budget) runtime state — shared between the
    // classifier, the router, and the dashboard API.
    let nudge_enabled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(config.llama_swap.nudge.enabled));
    let nudge_tier = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(2)); // deep — Bonsai is off by default, so there is no classifier to pick a tier
    let prompt_rewrite = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)); // off by default; enabling requires the Bonsai classifier to be running

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
        subs_model: config.llama_swap.subs_model.clone(),
        health,
        routing_events: Arc::clone(&routing_events),
        local_system_prompt,
        inference_tracker: Arc::clone(&inference_tracker),
        nudge_budgets: config.llama_swap.nudge.budgets,
        nudge_enabled: Arc::clone(&nudge_enabled),
        nudge_tier: Arc::clone(&nudge_tier),
        prompt_rewrite: Arc::clone(&prompt_rewrite),
    }).with_profiles(profiles));

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
        routing_mode: std::sync::Arc::new(std::sync::atomic::AtomicU8::new(routing_mode)),
        versions_cache: Arc::new(versions_rx),
        nudge_enabled,
        nudge_tier,
        nudge_model_key: config.llama_swap.nudge.model_key.clone(),
        nudge_budgets: config.llama_swap.nudge.budgets,
        prompt_rewrite,
        inflight: Arc::new(brainrouter::inflight::InflightRegistry::new()),
        benchmark_store,
        observability: Arc::new(brainrouter::observability::Observability::new(&config_path)),
    });

    brainrouter::observability::start(&state);

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
    // Background task: feed the active llama-server's prefill and generation
    // progress into the in-flight registry. Builds without /slots no-op.
    {
        let ls_url = state.llama_swap_url.clone();
        let inflight = Arc::clone(&state.inflight);
        tokio::spawn(async move {
            let client = reqwest::Client::new();
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                let Some(progress) = fetch_slots(&client, &ls_url).await else {
                    continue;
                };
                inflight.set_slot_progress_for_model(
                    &progress.model,
                    progress.prefill_progress,
                    progress.generated_tokens,
                    progress.max_tokens,
                );
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

#[cfg(test)]
mod slot_progress_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_prefill_progress() {
        let slots = json!([{
            "is_processing": true,
            "n_prompt_tokens": 100,
            "n_prompt_processed": 40,
            "next_token": {"n_decoded": 0}
        }]);
        let progress = parse_slot_progress(&slots, "model-a".into()).unwrap();
        assert_eq!(progress.model, "model-a");
        assert_eq!(progress.prefill_progress, Some(0.4));
        assert_eq!(progress.generated_tokens, 0);
    }

    #[test]
    fn parses_generation_progress_and_limit() {
        let slots = json!({"slots": [{
            "is_processing": true,
            "n_prompt_tokens": 100,
            "next_token": [{"n_decoded": 24, "n_remain": 76}]
        }]});
        let progress = parse_slot_progress(&slots, "model-b".into()).unwrap();
        assert_eq!(progress.prefill_progress, None);
        assert_eq!(progress.generated_tokens, 24);
        assert_eq!(progress.max_tokens, Some(100));
    }

    #[test]
    fn ignores_idle_slots() {
        let slots = json!([{
            "is_processing": false,
            "n_prompt_tokens": 100,
            "next_token": {"n_decoded": 0}
        }]);
        assert!(parse_slot_progress(&slots, "model".into()).is_none());
    }
}
