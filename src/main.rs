use anyhow::{Context, Result};
use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;
use crate::provider::{
    anthropic::AnthropicProvider, embedded::EmbeddedProvider, google::GoogleProvider,
    openai::OpenAiProvider, Provider,
};
use router::Router;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use brainrouter::{config, health, provider, ranker, router, server, stream, types};

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

    // Create providers
    let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();

    // Initialize configured providers
    for (provider_name, provider_config) in &config.providers {
        match provider_config.provider_type {
            config::ProviderType::OpenAiCompatible => {
                let base_url = provider_config.base_url.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("OpenAI-compatible provider '{}' requires base_url", provider_name))?;
                let api_key = config.resolve_api_key(provider_name);
                let provider = OpenAiProvider::new(
                    provider_name.clone(),
                    base_url.clone(),
                    api_key,
                );
                providers.insert(provider_name.clone(), Arc::new(provider));
                info!("Initialized OpenAI-compatible provider: {}", provider_name);
            }
            config::ProviderType::Anthropic => {
                let api_key = config.resolve_api_key(provider_name)
                    .ok_or_else(|| anyhow::anyhow!("Anthropic provider '{}' requires api_key", provider_name))?;
                let provider = AnthropicProvider::new(api_key);
                providers.insert(provider_name.clone(), Arc::new(provider));
                info!("Initialized Anthropic provider: {}", provider_name);
            }
            config::ProviderType::Google => {
                let api_key = config.resolve_api_key(provider_name)
                    .ok_or_else(|| anyhow::anyhow!("Google provider '{}' requires api_key", provider_name))?;
                let provider = GoogleProvider::new(api_key);
                providers.insert(provider_name.clone(), Arc::new(provider));
                info!("Initialized Google provider: {}", provider_name);
            }
        }
    }

    // Initialize embedded Bonsai provider
    info!("Initializing embedded Bonsai provider...");
    let embedded_provider = provider::embedded::EmbeddedProvider::new(&config.bonsai.model_path)
        .context("Failed to initialize embedded Bonsai provider")?;
    info!("Initialized embedded Bonsai provider");

    // Rank models using Bonsai
    let ranked_models = match ranker::rank_models(&config.models, &embedded_provider) {
        Ok(ranked) => {
            info!("Successfully ranked {} models", ranked.len());
            ranked
        }
        Err(e) => {
            tracing::warn!("Model ranking failed: {}. Using config file order.", e);
            config.models.iter().map(|m| m.name.clone()).collect()
        }
    };

    // Insert embedded provider into providers map
    providers.insert("embedded-bonsai".to_string(), Arc::new(embedded_provider));

    // Create health tracker
    let health_tracker = Arc::new(health::HealthTracker::new());

    // Create router with ranked models
    let router = Arc::new(Router::new(
        providers,
        ranked_models,
        &config.models,
        health_tracker.clone(),
    ));

    // Create shared application state
    let state = Arc::new(server::AppState { router });

    // Start the server (handles both TCP and Unix socket listeners)
    server::run(tcp_addr, args.socket, state).await?;

    Ok(())
}
