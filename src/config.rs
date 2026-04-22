use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Top-level brainrouter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainrouterConfig {
    pub manifest: ManifestConfig,
    pub llama_swap: LlamaSwapConfig,
    pub bonsai: BonsaiConfig,
    #[serde(default)]
    pub review: ReviewConfig,
}

/// Configuration for the Manifest cloud LLM router.
/// Manifest exposes an OpenAI-compatible endpoint; brainrouter delegates all
/// cloud routing decisions to Manifest by sending requests with model="auto".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestConfig {
    /// Base URL of the Manifest instance. Example: "http://localhost:3001".
    pub base_url: String,

    /// Name of the environment variable holding the Manifest API key (mnfst_*).
    /// Optional for local deployments that don't require auth.
    #[serde(default)]
    pub api_key_env: Option<String>,
}

/// Configuration for the local llama-swap server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaSwapConfig {
    /// Base URL of llama-swap. Example: "http://localhost:8080".
    pub base_url: String,

    /// The model key to use when falling back from Manifest, or when the user
    /// sends model="auto" and Bonsai classifies the query as local without a
    /// specific model hint. Must match an entry in the llama-swap config.
    pub fallback_model: String,

    /// Optional path to a custom system prompt file for local routing mode.
    /// If absent, the built-in lean prompt is used.
    #[serde(default)]
    pub local_system_prompt: Option<String>,
}

/// Configuration for the embedded Bonsai classifier model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BonsaiConfig {
    /// Path to the Bonsai GGUF model file.
    pub model_path: PathBuf,
}

/// Configuration for the review service and escalation dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewConfig {
    /// Maximum number of LLM review iterations before escalating to human.
    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,
}

fn default_max_iterations() -> u32 {
    5
}

impl Default for ReviewConfig {
    fn default() -> Self {
        ReviewConfig {
            max_iterations: default_max_iterations(),
        }
    }
}

impl BrainrouterConfig {
    /// Resolve the Manifest API key from the configured environment variable.
    /// Returns None if no env var is configured or it is unset.
    pub fn resolve_manifest_api_key(&self) -> Option<String> {
        let env_var = self.manifest.api_key_env.as_ref()?;
        std::env::var(env_var).ok()
    }
}

/// Load and validate the brainrouter configuration from a YAML file.
pub fn load(path: &Path) -> Result<BrainrouterConfig> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;

    let config: BrainrouterConfig = serde_yaml::from_str(&contents)
        .with_context(|| format!("Failed to parse YAML config: {}", path.display()))?;

    // Validate manifest.base_url
    if config.manifest.base_url.is_empty() {
        bail!("manifest.base_url must not be empty");
    }
    if !config.manifest.base_url.starts_with("http://")
        && !config.manifest.base_url.starts_with("https://")
    {
        bail!(
            "manifest.base_url must start with http:// or https://, got: {}",
            config.manifest.base_url
        );
    }

    // Validate llama_swap.base_url
    if config.llama_swap.base_url.is_empty() {
        bail!("llama_swap.base_url must not be empty");
    }
    if config.llama_swap.fallback_model.is_empty() {
        bail!("llama_swap.fallback_model must not be empty");
    }

    // Validate bonsai.model_path exists
    if !config.bonsai.model_path.exists() {
        bail!(
            "Bonsai model path does not exist: {}",
            config.bonsai.model_path.display()
        );
    }

    Ok(config)
}
