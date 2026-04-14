use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainrouterConfig {
    pub providers: HashMap<String, ProviderConfig>,
    pub models: Vec<ModelConfig>,
    pub bonsai: BonsaiConfig,
    pub llama_swap: Option<LlamaSwapConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    #[serde(rename = "type")]
    pub provider_type: ProviderType,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderType {
    Anthropic,
    Google,
    #[serde(rename = "openai-compatible")]
    OpenAiCompatible,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub providers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BonsaiConfig {
    pub model_path: PathBuf,
    #[serde(default = "default_always_last_resort")]
    pub always_last_resort: bool,
}

fn default_always_last_resort() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaSwapConfig {
    pub base_url: String,
    #[serde(default = "default_health_poll_interval_secs")]
    pub health_poll_interval_secs: u64,
    #[serde(default = "default_restart_on_stuck")]
    pub restart_on_stuck: bool,
    #[serde(default = "default_stuck_threshold_secs")]
    pub stuck_threshold_secs: u64,
}

fn default_health_poll_interval_secs() -> u64 {
    10
}

fn default_restart_on_stuck() -> bool {
    true
}

fn default_stuck_threshold_secs() -> u64 {
    60
}

pub fn load(path: &Path) -> Result<BrainrouterConfig> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;

    let config: BrainrouterConfig = serde_yaml::from_str(&contents)
        .with_context(|| format!("Failed to parse YAML config: {}", path.display()))?;

    // Validate at least one model
    if config.models.is_empty() {
        bail!("Configuration must define at least one model");
    }

    // Validate all model provider references exist
    for model in &config.models {
        for provider_ref in &model.providers {
            if !config.providers.contains_key(provider_ref) {
                bail!(
                    "Model '{}' references unknown provider '{}'",
                    model.name,
                    provider_ref
                );
            }
        }
    }

    // Validate bonsai model path exists
    if !config.bonsai.model_path.exists() {
        bail!(
            "Bonsai model path does not exist: {}",
            config.bonsai.model_path.display()
        );
    }

    Ok(config)
}

impl BrainrouterConfig {
    pub fn resolve_api_key(&self, provider_name: &str) -> Option<String> {
        let provider = self.providers.get(provider_name)?;
        let env_var_name = provider.api_key_env.as_ref()?;
        std::env::var(env_var_name).ok()
    }
}
