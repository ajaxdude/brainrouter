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

/// Native-mode counterpart: the supervisor resident's own `/slots` endpoint
/// (no `:8081` hop). Same parser, same progress semantics.
async fn fetch_slots_native(
    client: &reqwest::Client,
    swap: &Arc<brainrouter::swap_scheduler::SwapScheduler>,
) -> Option<SlotProgress> {
    let status = swap.status();
    let resident = status
        .models
        .iter()
        .find(|m| m.state == "ready" && m.endpoint.is_some())?;
    let model = resident.id.clone();
    let endpoint = resident.endpoint.clone()?;
    let slots = client
        .get(format!("{}/slots", endpoint.trim_end_matches('/')))
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
        .ok()?
        .json::<serde_json::Value>()
        .await
        .ok()?;
    parse_slot_progress(&slots, model)
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
/// v2 L7 flip gate. `swap.mode: native` refuses to boot while the shim still
/// reports loaded models — the operator flushes first, so no model is
/// double-resident and retiring the shim Residency source is safe. An
/// unreachable shim holds nothing; boot proceeds.
async fn enforce_native_flip_gate(swap: &config::SwapConfig) -> Result<()> {
    if !swap.mode.is_native() {
        return Ok(());
    }
    // `base_url` may carry the OpenAI `/v1` suffix; `/running` is at the root.
    let base = swap.shim_base_url().trim_end_matches('/');
    let base = base.strip_suffix("/v1").unwrap_or(base);
    let url = format!("{base}/running");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .context("building the flip-gate client")?;
    let loaded: Vec<String> = match client.get(&url).send().await {
        Ok(response) => match response.json::<serde_json::Value>().await {
            Ok(json) => json
                .get("running")
                .and_then(|running| running.as_array())
                .map(|rows| {
                    rows.iter()
                        .filter_map(|row| {
                            row.get("model").and_then(|m| m.as_str()).map(str::to_string)
                        })
                        .collect()
                })
                .unwrap_or_default(),
            Err(_) => Vec::new(),
        },
        Err(_) => Vec::new(),
    };
    if !loaded.is_empty() {
        anyhow::bail!(
            "swap.mode: native refuses to boot — the llama-swap shim still reports loaded models: {}. Flush them first (dashboard \"Unload models\" or POST /api/models/flush), then restart. Native mode never double-resides a model.",
            loaded.join(", ")
        );
    }
    Ok(())
}

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
        let llama_swap_config =
            brainrouter::migration::read_swap_config_path(&config_path)?;
        for line in brainrouter::migration::preview_v2_p4(
            &llama_swap_config,
            &brainrouter::paths::brainrouter_state(),
            &brainrouter::swap_catalog::default_path(),
        )? {
            println!("{line}");
        }
        for line in brainrouter::migration::preview_v2_p6(
            &config_path,
            &brainrouter::paths::brainrouter_state(),
            &brainrouter::providers::default_path(),
        )? {
            println!("{line}");
        }
        for line in brainrouter::migration::preview_v2_p7(
            &config_path,
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

    // v2 L4: import the live llama-swap config into the swap catalog (one-shot,
    // snapshot + restore; CLI-owned entries are preserved). `swap.config_path`
    // is the import source (fedora hosts point it at their docker-mounted file;
    // default `~/.config/llama-swap/config.yaml`).
    {
        let llama_swap_config =
            brainrouter::migration::read_swap_config_path(&config_path)?;
        for action in brainrouter::migration::run_v2_p4(
            &llama_swap_config,
            &brainrouter::paths::brainrouter_state(),
            &brainrouter::swap_catalog::default_path(),
            &brainrouter::swap_scheduler::budget_path(),
        )? {
            warn!("Migration {}: {action}", brainrouter::migration::PHASE_V2_P4);
        }
    }

    // v2 L6: absorb the v1 manifest block into a provider record (one-shot;
    // the YAML keys are removed, the `.bak` + snapshot hold the pre-state).
    for action in brainrouter::migration::run_v2_p6(
        &config_path,
        &brainrouter::paths::brainrouter_state(),
        &brainrouter::providers::default_path(),
        &brainrouter::provider_budget::default_path(),
    )? {
        warn!("Migration {}: {action}", brainrouter::migration::PHASE_V2_P6);
    }

    // v2 L7: rename `llama_swap:` → `swap:` LAST, right before the config is
    // read — after the p3 snapshot (which must hold the pristine v1 YAML
    // byte-for-byte) and after p4/p6 (which inspect either spelling). Keys move
    // verbatim; a default `base_url` is dropped with a note. Idempotent.
    for action in brainrouter::migration::run_v2_p7(
        &config_path,
        &brainrouter::paths::brainrouter_state(),
    )? {
        warn!("Migration {}: {action}", brainrouter::migration::PHASE_V2_P7);
    }
    // Config
    let config = config::load(&config_path)
        .with_context(|| format!("Failed to load config from {}", config_path.display()))?;

    // v2 L7 flip gate: native mode REFUSES to boot while the shim still holds
    // loaded models (no silent double residency). The operator flushes the
    // shim (dashboard "Unload models" / POST /api/models/flush), then restarts.
    enforce_native_flip_gate(&config.swap).await?;
    if config.swap.mode.is_native() {
        info!(
            mode = config.swap.mode.as_str(),
            import_source = %config.swap.import_config_path().display(),
            "swap.mode: native — the supervisor owns every local model; the :8081 shim is retired"
        );
    }

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
        swap_mode = config.swap.mode.as_str(),
        shim_url = %config.swap.shim_base_url(),
        fallback_model = %config.swap.fallback_model,
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
        config.swap.nudge.enabled,
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
        config.swap.fallback_model.clone(),
        slot_classifier_enabled,
        Arc::clone(&nudge_enabled),
        config.swap.nudge.model_key.clone(),
        std::time::Duration::from_millis(config.decision_model.timeout_ms.unwrap_or(15_000)),
    ));
    info!(kind = config.decision_model.kind.as_str(), enabled = config.decision_model.enabled, "Decision slot ready");

    // v2 L6: providers replace the hardcoded Manifest backend. A corrupt
    // registry is quarantined and boot continues with an empty one (cloud
    // simply reports unavailable until healed).
    let providers_path = brainrouter::providers::default_path();
    let providers = Arc::new(
        match brainrouter::providers::ProviderStore::load(providers_path.clone()) {
            Ok(store) => store,
            Err(error) => {
                let quarantine = providers_path.with_extension("json.corrupt");
                let _ = std::fs::rename(&providers_path, &quarantine);
                warn!(%error, quarantine = %quarantine.display(),
                    "providers.json unreadable; quarantined and starting empty");
                brainrouter::providers::ProviderStore::load(providers_path.clone())
                    .unwrap_or_else(|_| brainrouter::providers::ProviderStore::in_memory())
            }
        },
    );
    let provider_budget = Arc::new(
        brainrouter::provider_budget::ProviderBudget::load(
            brainrouter::provider_budget::default_path(),
        )
        .unwrap_or_else(|error| {
            warn!(%error, "provider budget ledger unreadable; starting a fresh in-memory ledger");
            brainrouter::provider_budget::ProviderBudget::in_memory()
        }),
    );

    // Shim provider (local shim mode only; native mode never dials it)
    let llama_swap = Arc::new(OpenAiProvider::new(
        "llama-swap".to_string(),
        config.swap.shim_base_url().to_string(),
        None,
    ));

    // Health tracker (circuit breaker)
    let health = Arc::new(HealthTracker::new());

    // Routing event ring buffer — shared between Router (writes) and HTTP API (reads)
    let routing_events = Arc::new(RoutingEvents::new());

    // Load optional custom local system prompt
    let local_system_prompt = config
        .swap
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

    // v2 L4: platform probe (feed cache + optional remote URL, staleness
    // bounded) and the swap supervisor. A corrupt catalog is quarantined
    // beside the live file rather than silently dropped (pins are state).
    let platform_client = reqwest::Client::new();
    let (platform_feed, platform_feed_origin) = brainrouter::platform::resolve_feed(
        &platform_client,
        std::env::var(brainrouter::platform::FEED_URL_ENV).ok().as_deref(),
        &brainrouter::platform::feed_cache_path(),
        std::time::SystemTime::now(),
    )
    .await;
    let platform = Arc::new(brainrouter::platform::resolve(
        &brainrouter::platform::detect_facts(),
        &platform_feed,
        platform_feed_origin,
    ));
    if !platform.known {
        warn!(
            platform = %platform.id,
            "Unknown platform (fail-open): no feed platform matches this hardware;              all engines/toolboxes are offered and the dashboard shows the banner"
        );
    }
    let swap_catalog_path = brainrouter::swap_catalog::default_path();
    let swap_catalog = Arc::new(
        match brainrouter::swap_catalog::SwapCatalog::load(swap_catalog_path.clone()) {
            Ok(catalog) => catalog,
            Err(error) => {
                let quarantine = swap_catalog_path.with_extension("json.corrupt");
                let _ = std::fs::rename(&swap_catalog_path, &quarantine);
                warn!(
                    %error,
                    quarantine = %quarantine.display(),
                    "Swap catalog unreadable; quarantined and starting empty"
                );
                brainrouter::swap_catalog::SwapCatalog::in_memory(swap_catalog_path.clone())
            }
        },
    );
    // v2 L7: a native boot must resolve every ex-shim key through the registry.
    // The p4 one-shot import may have run against the default path on a host
    // whose live config lives elsewhere (`swap.config_path`), so an empty
    // catalog imports from the configured source now instead of stranding keys.
    if config.swap.mode.is_native() && swap_catalog.entries().is_empty() {
        let source = config.swap.import_config_path();
        if source.exists() {
            match brainrouter::swap_catalog::import_llama_swap_file(&source)
                .and_then(|plan| swap_catalog.apply_import(&plan))
            {
                Ok(outcome) => info!(
                    path = %source.display(),
                    added = outcome.added,
                    updated = outcome.updated,
                    "swap.mode: native — imported the catalog from the configured source"
                ),
                Err(error) => warn!(
                    path = %source.display(),
                    %error,
                    "swap.mode: native — import source unreadable; the catalog is empty"
                ),
            }
        } else {
            warn!(
                path = %source.display(),
                "swap.mode: native — import source missing and the catalog is empty; add keys                  with `swap catalog add`"
            );
        }
    }
    let budget_ledger = Arc::new(
        brainrouter::swap_scheduler::BudgetLedger::load(
            brainrouter::swap_scheduler::budget_path(),
        )
        .unwrap_or_else(|error| {
            warn!(%error, "Budget ledger unreadable; starting an in-memory ledger");
            brainrouter::swap_scheduler::BudgetLedger::in_memory(
                brainrouter::swap_scheduler::budget_path(),
            )
        }),
    );
    let mut swap_adapters: std::collections::BTreeMap<
        String,
        Arc<dyn brainrouter::swap_scheduler::EngineAdapter>,
    > = std::collections::BTreeMap::new();
    swap_adapters.insert(
        "fake".into(),
        Arc::new(brainrouter::swap_scheduler::FakeAdapter::new()),
    );
    swap_adapters.insert(
        "external".into(),
        Arc::new(brainrouter::swap_scheduler::ExternalAdapter),
    );
    let native_adapter = Arc::new(brainrouter::engine_native::NativeLlamaCppAdapter::new());
    swap_adapters.insert(
        "llama_cpp".into(),
        native_adapter as Arc<dyn brainrouter::swap_scheduler::EngineAdapter>,
    );
    // v2 L5: per-host fleet profile decides which icon-fleet engines become
    // supervisor-managed here (strixite on gfx1151, hipfire on gfx1100).
    let fleet = brainrouter::fleet_profile::FleetProfile::for_probe(&platform);
    if let Some(strixite) = &fleet.strixite {
        let adapter = Arc::new(brainrouter::engine_strixite::StrixiteAdapter::new(
            strixite.clone(),
        ));
        adapter.reserve_ports(brainrouter::engine_strixite::avoidance_from_catalog(
            &swap_catalog.snapshot(),
        ));
        swap_adapters.insert(
            "strixite".into(),
            adapter as Arc<dyn brainrouter::swap_scheduler::EngineAdapter>,
        );
    }
    if let Some(hipfire) = &fleet.hipfire {
        let adapter = Arc::new(brainrouter::engine_hipfire::HipfireAdapter::new(
            hipfire.clone(),
        ));
        swap_adapters.insert(
            "hipfire".into(),
            adapter as Arc<dyn brainrouter::swap_scheduler::EngineAdapter>,
        );
    }
    // L5: probe each registered adapter once at boot (install marker +
    // version-keyed schema derivation). Non-fatal, same as the L4 adapters.
    for (name, adapter) in &swap_adapters {
        if let Err(error) = adapter.probe() {
            warn!(adapter = %name, %error, "engine adapter probe failed");
        }
    }
    let active_consumers = Arc::new(brainrouter::active_consumers::ActiveConsumers::from_env_or(
        std::time::Duration::from_secs(fleet.active_consumer_grace_s),
    ));
    let mut scheduler_config = brainrouter::swap_scheduler::SwapSchedulerConfig {
        llama_swap_url: config
            .swap
            .shim_base_url()
            .trim_end_matches('/')
            .strip_suffix("/v1")
            .unwrap_or(config.swap.shim_base_url())
            .to_string(),
        shim_source_enabled: !config.swap.mode.is_native(),
        active_consumer_grace: active_consumers.grace(),
        gate_timeout: std::time::Duration::from_millis(fleet.gate_timeout_ms),
        gate_poll: std::time::Duration::from_millis(fleet.gate_poll_ms),
        settle: std::time::Duration::from_millis(fleet.settle_ms),
        cutover_recheck: std::time::Duration::from_millis(fleet.cutover_recheck_ms),
        ..Default::default()
    };
    if fleet.strixite.is_some() || fleet.hipfire.is_some() {
        scheduler_config.min_mem_gib = 0.0; // the adapter profile owns the floor
    }
    let external_observer = Arc::new(brainrouter::engine_observer::ExternalObserver::new(
        Arc::new(brainrouter::engine_observer::ProcFsSource),
        brainrouter::swap_scheduler::adoption_dir(),
        Arc::clone(&swap_catalog),
    ));
    let mut scheduler_builder = brainrouter::swap_scheduler::SwapScheduler::new(
        Arc::clone(&swap_catalog),
        budget_ledger,
        scheduler_config,
        swap_adapters,
    )
    .with_serving_identities(Arc::clone(&serving_identities))
    .with_active_consumers(Arc::clone(&active_consumers))
    .with_external_observer(external_observer)
    .with_unit_runtime(Arc::new(
        brainrouter::engine_systemd::SystemdUnitRuntime::cli(),
    ));
    if let Some(strixite) = &fleet.strixite {
        if let Some(unit) = &strixite.unit {
            scheduler_builder = scheduler_builder.with_takeover_unit("strixite", unit.clone());
        }
    }
    if let Some(hipfire) = &fleet.hipfire {
        if let Some(unit) = &hipfire.unit {
            scheduler_builder = scheduler_builder.with_takeover_unit("hipfire", unit.clone());
        }
    }
    let swap_scheduler = Arc::new(scheduler_builder);
    let adoption = swap_scheduler.scan_adoption();
    if !adoption.adopted.is_empty() {
        warn!(
            adopted = ?adoption.adopted,
            spec_stale = ?adoption.spec_stale,
            cleaned = ?adoption.cleaned,
            "Adopted native engines from the previous daemon (no kill, no respawn)"
        );
    }

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
        let p = config.swap.import_config_path();
        std::fs::canonicalize(&p).ok()
    };
    let mut router_builder = Router::new(brainrouter::router::RouterArgs {
        slot: Arc::clone(&decision_slot),
        cards: Arc::clone(&cards),
        providers: Arc::clone(&providers),
        credentials: Arc::clone(&credentials_store),
        budget: Arc::clone(&provider_budget),
        llama_swap,
        fallback_model: config.swap.fallback_model.clone(),
        local_models: config.swap.local_models.clone(),
        subs_model: config.swap.subs_model.clone(),
        health,
        routing_events: Arc::clone(&routing_events),
        local_system_prompt,
        inference_tracker: Arc::clone(&inference_tracker),
        nudge_budgets: config.swap.nudge.budgets,
        nudge_enabled: Arc::clone(&nudge_enabled),
        nudge_tier: Arc::clone(&nudge_tier),
        nudge_model_key: config.swap.nudge.model_key.clone(),
        prompt_rewrite: Arc::clone(&prompt_rewrite),
        llama_swap_config_path,
        serving_identities: Some(Arc::clone(&serving_identities)),
    });
    router_builder.install_swap(Arc::clone(&swap_scheduler));
    router_builder.install_swap_mode(config.swap.mode);
    router_builder.install_active_consumers(Arc::clone(&active_consumers));
    let router = Arc::new(router_builder);

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
        Arc::clone(&providers),
        config.swap.fallback_model.clone(),
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
        .swap
        .shim_base_url()
        .trim_end_matches('/')
        .strip_suffix("/v1")
        .unwrap_or(config.swap.shim_base_url())
        .to_string();

    // Prefer the migrated provider's URL; fall back to the live YAML value
    // (pre-migration) or an empty string (no cloud provider at all).
    let manifest_url = {
        let registry_url = providers
            .snapshot()
            .into_iter()
            .find(|p| p.id == "manifest")
            .map(|p| p.base_url)
            .unwrap_or_default();
        let url = if registry_url.is_empty() {
            config.manifest.base_url.clone()
        } else {
            registry_url
        };
        url.trim_end_matches('/')
            .strip_suffix("/v1")
            .unwrap_or(&url)
            .to_string()
    };

    // Seed with an empty object, not null: /api/versions and `brainrouter cli
    // versions` read this before the first compute completes (up to ~15 s).
    let (versions_tx, versions_rx) = tokio::sync::watch::channel(serde_json::json!({}));
    let versions_tx = Arc::new(versions_tx);

    let inflight = Arc::new(brainrouter::inflight::InflightRegistry::new());
    swap_scheduler.attach_inflight(Arc::clone(&inflight));
    {
        let scheduler = Arc::clone(&swap_scheduler);
        let consumers = Arc::clone(&active_consumers);
        inflight.set_completion_hook(Arc::new(move |catalog_id: &str| {
            scheduler.record_stream_completed(catalog_id);
            // L5: a completed proxy stream opens the Active-Consumer grace
            // window (the live span is covered by the in-flight row itself).
            consumers.note_completion(catalog_id);
        }));
    }
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
            let p = config.swap.import_config_path();
            std::fs::canonicalize(&p).unwrap_or(p)
        },
        swap_mode: config.swap.mode,
        decision_config: config.decision_model.clone(),
        llama_swap_config_lock: Arc::new(tokio::sync::Mutex::new(())),
        tcp_addr: tcp_addr.to_string(),
        providers: Arc::clone(&providers),
        budget: Arc::clone(&provider_budget),
        routing_mode: std::sync::Arc::new(std::sync::atomic::AtomicU8::new(routing_mode)),
        versions_cache: Arc::new(versions_rx),
        nudge_enabled,
        nudge_tier,
        nudge_model_key: config.swap.nudge.model_key.clone(),
        nudge_budgets: config.swap.nudge.budgets,
        prompt_rewrite,
        code_review_enabled,
        pr_guidelines_enabled,
        inflight,
        platform,
        swap: swap_scheduler,
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

    // v2 L6: LAN federation listener (second bound socket; the primary
    // loopback surface is untouched). Default off = v1 attack surface.
    if config.lan.enabled {
        let listener = brainrouter::lan::bind(&config.lan).await?;
        let lan_addr = listener.addr;
        // Named accepted risk (design Q3 + Map Security): bearer over
        // plaintext HTTP on a trusted LAN; peer-info enumerates exposed
        // models to any token holder.
        warn!(
            %lan_addr,
            expose = ?config.lan.expose,
            "LAN listener enabled: bearer-authenticated plaintext HTTP on the LAN \
             (trusted-LAN assumption; peer-info enumerates exposed models to token holders)"
        );
        let app = Arc::clone(&state);
        tokio::spawn(async move {
            if let Err(error) = brainrouter::lan::serve(listener, app).await {
                tracing::error!(%error, "LAN listener ended");
            }
        });
    }

    // v2 L2: wire the plugin subsystem to the live state, bind per-plugin
    // UDS sockets, and spawn the enabled plugins (supervisor).
    plugins.attach(Arc::clone(&state));
    plugins.start_at_boot().await;

    // Background task: compute versions once, then refresh every 30 minutes.
    {
        let tx = Arc::clone(&versions_tx);
        let fork_path = config
            .decision_model
            .launch
            .as_ref()
            .map(|l| l.fork_path.clone())
            .unwrap_or_default();
        let swap_mode = config.swap.mode;
        let swap = Arc::clone(&state.swap);
        tokio::spawn(async move {
            let data =
                server::compute_versions_json(&fork_path, swap_mode, swap.catalog().entries().len())
                    .await;
            let _ = tx.send_replace(data);
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(30 * 60)).await;
                let data = server::compute_versions_json(
                    &fork_path,
                    swap_mode,
                    swap.catalog().entries().len(),
                )
                .await;
                let _ = tx.send_replace(data);
            }
        });
    }
    // Background task: feed the active llama-server's prefill and generation
    // progress into the in-flight registry. Builds without /slots no-op.
    {
        let ls_url = state.llama_swap_url.clone();
        let swap = Arc::clone(&state.swap);
        let mode = state.swap_mode;
        let inflight = Arc::clone(&state.inflight);
        tokio::spawn(async move {
            let client = reqwest::Client::new();
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                let progress = if mode.is_native() {
                    fetch_slots_native(&client, &swap).await
                } else {
                    fetch_slots(&client, &ls_url).await
                };
                let Some(progress) = progress else {
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

    // Background task: refresh the Residency view (llama-swap /running +
    // container identities) and run crash detection/bounded restart every 2 s.
    {
        let swap = Arc::clone(&state.swap);
        tokio::spawn(async move {
            let client = reqwest::Client::new();
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                swap.refresh_residency(&client).await;
                // L5: read-only /proc observation of icon-era engines.
                swap.refresh_external();
                swap.maintenance().await;
            }
        });
    }

    // Server: the accept loops on the listeners bound at the very start.
    let serve_result = server::run_with_listeners(
        bound.tcp,
        bound.uds_path,
        bound.uds,
        bound.uds_identity,
        bound.uds_lock,
        Arc::clone(&state),
    )
    .await;

    // v2 L7 rollback contract: shutdown stops every native child (SIGTERM →
    // grace, flock released at child exit). Runs on the error path too, so a
    // failed serve never leaves orphan engines loading into VRAM.
    let stopped = state.swap.shutdown_all("daemon shutdown");
    if !stopped.is_empty() {
        info!(
            models = ?stopped,
            "swap supervisor stopped native children at shutdown"
        );
    }

    serve_result?;
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
