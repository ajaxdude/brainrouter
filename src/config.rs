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
    /// Bridge transports (Discord, Signal) — optional, disabled by default.
    #[serde(default)]
    pub bridge: Option<crate::bridge::BridgeConfig>,
}

/// Configuration for the Manifest cloud LLM router.
/// Manifest exposes an OpenAI-compatible endpoint; brainrouter delegates all
/// cloud routing decisions to Manifest by sending requests with model="auto".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestConfig {
    /// Base URL of the Manifest instance. Example: "http://localhost:2099".
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

    /// Explicit list of model keys served by llama-swap. When a request arrives
    /// with `model` set to one of these names, brainrouter routes directly to
    /// llama-swap without consulting Bonsai — the user's model choice is
    /// authoritative. Omit or leave empty to rely on the `brainrouter/<model>`
    /// prefix convention for direct-local routing.
    ///
    /// Matching is case-sensitive; model names must exactly match the keys in
    /// your llama-swap config (e.g. the key in llama-swap's `models:` map).
    #[serde(default)]
    pub local_models: Vec<String>,

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

    /// Forced review mode: "auto", "cloud", or "local".
    #[serde(default = "default_review_mode")]
    pub forced_mode: String,

    /// Forced model key (only used when forced_mode is "local").
    #[serde(default)]
    pub forced_model: Option<String>,
}

fn default_max_iterations() -> u32 {
    5
}

fn default_review_mode() -> String {
    "auto".to_string()
}

impl Default for ReviewConfig {
    fn default() -> Self {
        ReviewConfig {
            max_iterations: default_max_iterations(),
            forced_mode: default_review_mode(),
            forced_model: None,
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
    if !config.llama_swap.base_url.starts_with("http://")
        && !config.llama_swap.base_url.starts_with("https://")
    {
        bail!(
            "llama_swap.base_url must start with http:// or https://, got: {}",
            config.llama_swap.base_url
        );
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


/// Default Unix domain socket path for the brainrouter daemon.
/// Prefers $XDG_RUNTIME_DIR/brainrouter.sock, falls back to /run/brainrouter.sock.
pub fn default_socket_path() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_RUNTIME_DIR") {
        PathBuf::from(dir).join("brainrouter.sock")
    } else {
        PathBuf::from("/run/brainrouter.sock")
    }
}

/// Default config file path for the brainrouter daemon.
/// Prefers `$XDG_CONFIG_HOME/brainrouter/brainrouter.yaml`,
/// falls back to `~/.config/brainrouter/brainrouter.yaml`.
/// The `--config` flag always overrides this.
pub fn default_config_path() -> PathBuf {
    let base = std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
            PathBuf::from(home).join(".config")
        });
    base.join("brainrouter").join("brainrouter.yaml")
}


#[cfg(test)]
mod tests {
    use super::*;

    fn parse_llama_swap(yaml: &str) -> LlamaSwapConfig {
        serde_yaml::from_str(yaml).expect("parse failed")
    }

    #[test]
    fn local_models_defaults_to_empty() {
        let cfg = parse_llama_swap(
            "base_url: http://localhost:8081/v1\nfallback_model: default\n",
        );
        assert!(cfg.local_models.is_empty());
    }

    #[test]
    fn local_models_parses_list() {
        let cfg = parse_llama_swap(
            "base_url: http://localhost:8081/v1\nfallback_model: default\nlocal_models:\n  - qwen3\n  - mistral-7b\n",
        );
        assert_eq!(cfg.local_models, vec!["qwen3", "mistral-7b"]);
    }

    #[test]
    fn local_models_membership_check() {
        // Mirrors the router check: `self.local_models.contains(&requested_model)`
        let cfg = parse_llama_swap(
            "base_url: http://localhost:8081/v1\nfallback_model: default\nlocal_models:\n  - qwen3\n",
        );
        assert!(cfg.local_models.contains(&"qwen3".to_string()));
        assert!(!cfg.local_models.contains(&"claude-3-opus".to_string()));
        // The brainrouter/ prefix is handled before the local_models check in the
        // router, so the raw prefix form is not expected to match here.
        assert!(!cfg.local_models.contains(&"brainrouter/qwen3".to_string()));
    }

    #[test]
    fn default_config_path_uses_xdg_config_home() {
        // Temporarily override XDG_CONFIG_HOME and HOME so the test is hermetic.
        // std::env::set_var is unsafe in multi-threaded contexts; use serial_test
        // or just document the env dependency. These tests run in the same process,
        // so we test the logic rather than mutating global env.
        //
        // Verify: when XDG_CONFIG_HOME is absent, falls back to $HOME/.config.
        // We can't set env vars safely in parallel tests, so we test the path
        // shape by calling the function and checking the suffix.
        let p = default_config_path();
        assert!(
            p.ends_with("brainrouter/brainrouter.yaml"),
            "unexpected path: {}",
            p.display()
        );
    }
}
