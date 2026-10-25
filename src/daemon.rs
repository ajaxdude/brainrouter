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

#[cfg(feature = "benchmarks")]
use brainrouter::benchmark_lab::BenchmarkLab;
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
    let running = client
        .get(format!("{}/running", ls_url))
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
        .ok()?
        .json::<serde_json::Value>()
        .await
        .ok()?;
    let active_model = running["running"][0]["model"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let proxy = running["running"][0]["proxy"].as_str()?.to_string();
    let slots = client
        .get(format!("{}/slots", proxy))
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
        .ok()?
        .json::<serde_json::Value>()
        .await
        .ok()?;
    parse_slot_progress(&slots, active_model)
}

fn parse_slot_progress(slots: &serde_json::Value, model: String) -> Option<SlotProgress> {
    let arr = slots
        .as_array()
        .or_else(|| slots.get("slots")?.as_array())?;
    let mut best: Option<(u64, SlotProgress)> = None;
    for slot in arr {
        let processing = slot
            .get("is_processing")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let total = slot
            .get("n_prompt_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let processed = slot
            .get("n_prompt_processed")
            .or_else(|| slot.get("n_prompt_tokens_processed"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let next_token = slot.get("next_token").and_then(|value| {
            if value.is_array() {
                value.as_array()?.first()
            } else {
                Some(value)
            }
        });
        let generated_tokens = slot
            .get("n_decoded")
            .and_then(|v| v.as_u64())
            .or_else(|| {
                next_token
                    .and_then(|value| value.get("n_decoded"))
                    .and_then(|v| v.as_u64())
            })
            .unwrap_or(0);
        if !processing && generated_tokens == 0 {
            continue;
        }
        let remaining = next_token
            .and_then(|value| value.get("n_remain"))
            .and_then(|v| v.as_i64())
            .filter(|value| *value >= 0)
            .map(|value| value as u64);
        let configured_max = slot
            .pointer("/params/n_predict")
            .and_then(|v| v.as_i64())
            .filter(|value| *value > 0)
            .map(|value| value as u64);
        let max_tokens = remaining
            .map(|value| generated_tokens.saturating_add(value))
            .or(configured_max);
        let prefill_progress = if generated_tokens == 0 && total > 0 {
            Some((processed as f64 / total as f64).clamp(0.0, 1.0))
        } else {
            None
        };
        let score = generated_tokens
            .saturating_mul(1_000_000)
            .saturating_add(processed);
        let progress = SlotProgress {
            model: model.clone(),
            prefill_progress,
            generated_tokens,
            max_tokens,
        };
        if best
            .as_ref()
            .map_or(true, |(best_score, _)| score > *best_score)
        {
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

    /// v2 L2: preview the config migrations (v1 bridge.discord.token ->
    /// credentials store) and exit 0 without touching any state.
    #[arg(long)]
    pub dry_run: bool,
}

/// Entry point for `brainrouter serve`.
pub async fn run(args: ServeArgs) -> Result<()> {
    let socket = args.socket.unwrap_or_else(config::default_socket_path);
    let config_path = args.config.unwrap_or_else(config::default_config_path);
    let managed_toolboxes_path = config_path.with_file_name("managed_toolboxes.json");

    // v2 L2: config migrations run BEFORE validation so v1 YAML boots.
    // `--dry-run` prints the migration preview and exits without touching
    // state (snapshot layout, credentials store, YAML key removal).
    let credentials_path = brainrouter::credentials::CredentialStore::default_path();
    if args.dry_run {
        let config_dir = config::default_config_path()
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(std::path::PathBuf::new);
        for line in brainrouter::migration::preview_v2_p3(
            &config_path,
            &brainrouter::paths::brainrouter_state(),
            &config_dir,
        )? {
            println!("{line}");
        }
        for line in brainrouter::migration::preview_v2_p2(
            &config_path,
            &credentials_path,
            &brainrouter::paths::brainrouter_state(),
        )? {
            println!("{line}");
        }
        println!("dry-run complete: no state written");
        return Ok(());
    }
    // I3: a corrupt credentials.json must not brick the daemon — rename the
    // unreadable vault aside and boot with an empty store (resolve then
    // fails cleanly per consumer).
    let credentials_store = Arc::new(
        match brainrouter::credentials::CredentialStore::load_quarantining(credentials_path.clone()) {
            (Ok(store), _) => store,
            (Err(e), _) => {
                tracing::error!(error = %e,
                    "credentials store is unreadable and could not be quarantined;                      starting with an in-memory empty vault");
                brainrouter::credentials::CredentialStore::load(
                    credentials_path.with_file_name("credentials.disabled.json"),
                )
                .context("creating the empty fallback credentials store")?
            }
        },
    );
    // v2 L3 role cards: load first so the migration can seed them. Runs
    // BEFORE the L2 phase so the v2.0-p3 snapshot is the pristine v1 state.
    let config_dir = config::default_config_path()
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(std::path::PathBuf::new);
    let cards = Arc::new(brainrouter::roles::RoleCardStore::load(
        brainrouter::roles::RoleCardStore::default_path(),
    ));
    for action in brainrouter::migration::run_v2_p3(
        &config_path,
        &credentials_store,
        &brainrouter::paths::brainrouter_state(),
        &config_dir,
        &cards,
    )? {
        // warn-level: operators on default RUST_LOG must see that their v1
        // config was consumed (v1 logged the forced_model migration at warn).
        warn!("Migration {}: {action}", brainrouter::migration::PHASE_V2_P3);
    }
    for action in brainrouter::migration::run_v2_p2(
        &config_path,
        &credentials_store,
        &brainrouter::paths::brainrouter_state(),
    )? {
        info!("Migration {}: {action}", brainrouter::migration::PHASE_V2_P2);
    }

    // Config
    let config = config::load(&config_path)
        .with_context(|| format!("Failed to load config from {}", config_path.display()))?;

    let routing_mode = match brainrouter::roles::parse_selector(
        &cards.assignment(brainrouter::roles::Role::Main, None),
    )
    .backend()
    {
        "cloud" => 1,
        "local" => 2,
        _ => 0,
    };
    // v2 R2: without the `benchmarks` cargo feature the explorer lab is not
    // compiled and the store is never opened — the routes answer the
    // documented 503 degradation envelope (`benchmark::unavailable_response`)
    // exactly as an unopenable database does today.
    let benchmark_store: Result<Arc<BenchmarkStore>, String> = if !cfg!(feature = "benchmarks") {
        Err("the `benchmarks` cargo feature is not compiled into this build \
            (rebuild with --features benchmarks to enable the explorer)"
            .to_string())
    } else {
        BenchmarkStore::open(config.benchmarks.database_path.clone())
            .map(Arc::new)
            .map_err(|error| {
                warn!(
                    path = %config.benchmarks.database_path.display(),
                    error = %error,
                    "Benchmark explorer unavailable; continuing core daemon startup"
                );
                error.to_string()
            })
    };

    // Defensive, log-only check that the vendored ai-toolbox-cockpit catalog
    // (assets/cockpit-catalog/) still parses/validates. No behavior depends
    // on this yet (that starts in PR2) — it exists purely so a corrupted or
    // unexpectedly-edited vendored file is visible in the daemon's own logs,
    // not just in CI. Never fatal.
    let vendored_catalog = brainrouter::toolbox_catalog::load_vendored_catalog();
    if vendored_catalog.report.is_ok() {
        info!(
            toolboxes = vendored_catalog.toolbox_count().unwrap_or(0),
            model_backends = vendored_catalog.model_backend_count().unwrap_or(0),
            warnings = vendored_catalog.report.warnings.len(),
            "Vendored toolbox catalog loaded"
        );
    } else {
        warn!(
            errors = ?vendored_catalog.report.errors,
            "Vendored toolbox catalog failed structural validation; toolbox-catalog features \
             (none shipped yet) would be unavailable"
        );
    }

    // Also verify the *effective* catalog (vendored + brainrouter's gufo
    // overlay). This is the runtime source of truth for gufo-aware endpoints,
    // which fail closed on a broken overlay, so surface any problem loudly
    // here too. Never fatal.
    match brainrouter::toolbox_catalog::load_effective_typed_catalog() {
        Ok((toolboxes, models)) => info!(
            toolboxes = toolboxes.toolboxes.len(),
            model_backends = models.backends.len(),
            "Effective toolbox catalog (with gufo overlay) loaded"
        ),
        Err(error) => warn!(
            %error,
            "Effective catalog (vendored + gufo overlay) is invalid; gufo-aware endpoints \
             will fail closed until this is fixed"
        ),
    }

    let tcp_addr: std::net::SocketAddr = args
        .tcp_addr
        .parse()
        .with_context(|| format!("Invalid TCP address: {}", args.tcp_addr))?;

    // v2 L3: bind the listeners before ANY heavy boot work (migration,
    // plugin registry, decision-backend launch, model resolution) so
    // `/health` never answers while the UDS socket file is still pending —
    // the window that made a CLI invoked right after startup race the bind
    // under parallel load ("Daemon socket not found" flakes).
    let bound = server::bind_listeners(tcp_addr, socket.clone()).await?;

    info!(
        config_path = %config_path.display(),
        tcp_addr = %tcp_addr,
        uds_path = %socket.display(),
        manifest_url = %config.manifest.base_url,
        llama_swap_url = %config.llama_swap.base_url,
        fallback_model = %config.llama_swap.fallback_model,
        "Starting brainrouter daemon"
    );
    // The Bonsai llama-server launch backend (L3: config moved under
    // `decision_model.launch`; the server itself is unchanged).
    let launch = config.decision_model.launch.clone();
    let bonsai_control = match &launch {
        Some(l) if config.decision_model.enabled && config.decision_model.is_bonsai_profile() => {
            let model_path = l
                .model_path
                .clone()
                .context("decision_model is enabled with prompt_profile bonsai but launch.model_path is not set")?;
            BonsaiControl::start(l.fork_path.clone(), model_path, l.server_port)
                .await
                .context("Failed to start Bonsai llama-server")?
        }
        _ => {
            warn!("decision slot classifier server is disabled — the slot answers per its kind (heuristic parity routes auto requests local). Enable with decision_model.enabled: true.");
            BonsaiControl::disabled(
                launch.as_ref().map(|l| l.fork_path.clone()).unwrap_or_default(),
                launch.as_ref().and_then(|l| l.model_path.clone()),
                launch.as_ref().map(|l| l.server_port).unwrap_or(9200),
            )
            .await
        }
    };

    // Nudge (per-request reasoning budget) runtime state — shared between the
    // classifier, the router, and the dashboard API.
    let nudge_enabled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
        config.llama_swap.nudge.enabled,
    ));
    let nudge_tier = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(2)); // deep — Bonsai is off by default, so there is no classifier to pick a tier
    let prompt_rewrite = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)); // off by default; enabling requires the Bonsai classifier to be running
    // FR-A + FR-D + Phase-1b: read all runtime flags in one load. Reviewer on by
    // default; PR-guideline injection off by default; HankNDory design-aware
    // review = the runtime override if the dashboard has ever set it, else the
    // YAML `review.hankndory_integration` seed. Persisted in
    // review_runtime_state.json.
    let (code_review_flag, pr_guidelines_flag, hankndory_override) =
        brainrouter::review::runtime_state::load_state(
            &brainrouter::review::runtime_state::state_path(),
        );
    let code_review_enabled =
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(code_review_flag));
    let pr_guidelines_enabled =
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(pr_guidelines_flag));
    // Apply the effective HankNDory value onto the live review card extras
    // (what `run_loop` reads). The card seeded the YAML value at migration;
    // override only when the runtime file has a value.
    let hankndory_effective = hankndory_override.unwrap_or(config.review.hankndory_integration);
    cards.set_hankndory_integration(hankndory_effective).ok();

    // The decision slot (L3): an openai-compatible bonsai profile keeps the
    // v1 wire + shared "server on" flag; any other kind carries its own
    // endpoint config.
    let slot_classifier_enabled = if config.decision_model.launch.is_some() {
        bonsai_control.enabled()
    } else {
        Arc::new(std::sync::atomic::AtomicBool::new(true))
    };
    let slot_base_url = config
        .decision_model
        .base_url
        .clone()
        .unwrap_or_else(|| {
            format!(
                "http://127.0.0.1:{}",
                config
                    .decision_model
                    .launch
                    .as_ref()
                    .map(|l| l.server_port)
                    .unwrap_or(9200)
            )
        });
    let slot_classifier = Arc::new(Classifier::for_slot(
        slot_base_url,
        config.llama_swap.fallback_model.clone(),
        slot_classifier_enabled,
        Arc::clone(&nudge_enabled),
        config.llama_swap.nudge.model_key.clone(),
        std::time::Duration::from_millis(config.decision_model.timeout_ms.unwrap_or(15_000)),
    ));
    info!(kind = config.decision_model.kind.as_str(), enabled = config.decision_model.enabled, "Decision slot ready");

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
        } else if !p.join("docker-compose.yml").exists() && !p.join("docker-compose.yaml").exists()
        {
            warn!(path = %dir, "BRAINROUTER_MANIFEST_DIR has no docker-compose.yml; Manifest upgrade will fail");
        }
    }

    // Inference state tracker — shared between Router (writes) and HTTP API (reads)
    let inference_tracker = Arc::new(InferenceTracker::new());
    let serving_identities = Arc::new(brainrouter::serving_identity::ServingIdentityRegistry::new());

    // v2 L2 plugin subsystem (registry + supervisor + per-plugin UDS RPC).
    // Constructed before the router: the `plugin` decision kind speaks to it.
    let plugins = Arc::new(
        brainrouter::plugins::PluginManager::new(Arc::clone(&credentials_store))
            .context("loading the plugin registry")?,
    );
    let decision_slot = Arc::new(brainrouter::decision::DecisionSlot::new(
        config.decision_model.clone(),
        slot_classifier,
        Some(Arc::clone(&plugins)),
    ));

    // Router — shared between the proxy and the review service
    let llama_swap_config_path = {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        let p = std::path::PathBuf::from(format!("{}/.config/llama-swap/config.yaml", home));
        std::fs::canonicalize(&p).ok()
    };
    let router = Arc::new(Router::new(brainrouter::router::RouterArgs {
        slot: Arc::clone(&decision_slot),
        cards: Arc::clone(&cards),
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
        nudge_model_key: config.llama_swap.nudge.model_key.clone(),
        prompt_rewrite: Arc::clone(&prompt_rewrite),
        llama_swap_config_path,
        serving_identities: Some(Arc::clone(&serving_identities)),
    }));

    // Session manager (in-memory; ephemeral per process lifetime)
    let session_manager = Arc::new(SessionManager::new());

    // Review service
    let review_config = config.review.clone();
    let review_service = Arc::new(ReviewService::new(
        Arc::clone(&router),
        Arc::clone(&session_manager),
        Arc::clone(&cards),
        review_config,
        config.review_admission.clone(),
        config.manifest.enabled,
        config.llama_swap.fallback_model.clone(),
    ));

    #[cfg(feature = "benchmarks")]
    let benchmark_lab = if !config.benchmarks.lab.enabled {
        Err("disabled in benchmarks.lab.enabled".to_string())
    } else {
        match &benchmark_store {
            Ok(store) => BenchmarkLab::new(
                config.benchmarks.lab.clone(),
                socket.clone(),
                Arc::clone(&router),
                Arc::clone(store),
            )
            .map(Arc::new)
            .map_err(|error| {
                warn!(error = %error, "Benchmark Lab unavailable; continuing core daemon startup");
                error.to_string()
            }),
            Err(reason) => Err(format!("benchmark registry unavailable: {reason}")),
        }
    };

    let llama_swap_url = config
        .llama_swap
        .base_url
        .trim_end_matches('/')
        .strip_suffix("/v1")
        .unwrap_or(&config.llama_swap.base_url)
        .to_string();

    let manifest_url = config
        .manifest
        .base_url
        .trim_end_matches('/')
        .strip_suffix("/v1")
        .unwrap_or(&config.manifest.base_url)
        .to_string();

    // Seed with an empty object, not null: /api/versions and `brainrouter cli
    // versions` read this before the first compute completes (up to ~15 s).
    let (versions_tx, versions_rx) = tokio::sync::watch::channel(serde_json::json!({}));
    let versions_tx = Arc::new(versions_tx);

    let state = Arc::new(AppState {
        router,
        session_manager,
        review_service,
        roles: Arc::clone(&cards),
        decision: Arc::clone(&decision_slot),
        credentials: Arc::clone(&credentials_store),
        routing_events,

        llama_swap_url,
        manifest_url,
        plugins: Arc::clone(&plugins),
        bonsai: Arc::new(bonsai_control),
        config_path: std::fs::canonicalize(&config_path).unwrap_or(config_path.clone()),
        llama_swap_config_path: {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
            let p = std::path::PathBuf::from(format!("{}/.config/llama-swap/config.yaml", home));
            std::fs::canonicalize(&p).unwrap_or(p)
        },
        decision_config: config.decision_model.clone(),
        llama_swap_config_lock: Arc::new(tokio::sync::Mutex::new(())),
        tcp_addr: tcp_addr.to_string(),
        manifest_enabled: config.manifest.enabled,
        routing_mode: std::sync::Arc::new(std::sync::atomic::AtomicU8::new(routing_mode)),
        versions_cache: Arc::new(versions_rx),
        nudge_enabled,
        nudge_tier,
        nudge_model_key: config.llama_swap.nudge.model_key.clone(),
        nudge_budgets: config.llama_swap.nudge.budgets,
        prompt_rewrite,
        code_review_enabled,
        pr_guidelines_enabled,
        inflight: Arc::new(brainrouter::inflight::InflightRegistry::new()),
        benchmark_store,
        #[cfg(feature = "benchmarks")]
        benchmark_lab,
        observability: Arc::new(brainrouter::observability::Observability::new(&config_path)),
        managed_toolboxes_path,
        toolbox_container_locks: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        model_downloads: Arc::new(brainrouter::model_downloads::ModelDownloadRegistry::new()),
        serving_identities,
    });

    brainrouter::observability::start(&state);

    // v2 L2: wire the plugin subsystem to the live state, bind per-plugin
    // UDS sockets, and spawn the enabled plugins (supervisor).
    plugins.attach(Arc::clone(&state));
    plugins.start_at_boot().await;

    // Background task: compute versions once, then refresh every 30 minutes.
    {
        let tx = Arc::clone(&versions_tx);
        tokio::spawn(async move {
            let fork_path = config
                .decision_model
                .launch
                .as_ref()
                .map(|l| l.fork_path.clone())
                .unwrap_or_default();
            let data = server::compute_versions_json(&fork_path).await;
            let _ = tx.send_replace(data);
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(30 * 60)).await;
                let data = server::compute_versions_json(&fork_path).await;
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

    // Server: the accept loops on the listeners bound at the very start.
    server::run_with_listeners(bound.tcp, bound.uds_path, bound.uds, bound.uds_identity, bound.uds_lock, state).await?;

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
