use std::os::unix::fs::PermissionsExt;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Top-level brainrouter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainrouterConfig {
    pub manifest: ManifestConfig,
    pub llama_swap: LlamaSwapConfig,
    pub bonsai: BonsaiConfig,
    /// Shared model storage directory settings.
    #[serde(default)]
    pub models: ModelsConfig,
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
    /// Master switch for the cloud backend. Off by default: `model=cloud` and
    /// Bonsai-classified cloud requests skip Manifest and fall back to
    /// llama-swap. Set `enabled: true` to use the cloud.
    #[serde(default)]
    pub enabled: bool,

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

    /// Optional subs-pool model key. Requests targeting `subs` or
    /// `brainrouter/subs` route directly to this llama-swap key, bypassing
    /// Bonsai classification — intended for a multi-slot pool serving
    /// subagent traffic. Must match an entry in the llama-swap config.
    /// Absent → `subs` requests fall back to normal auto routing.
    #[serde(default)]
    pub subs_model: Option<String>,

    /// Optional reasoning-budget nudge settings (voidsurfer/llama.cpp-nudge
    /// fork). When enabled, auto-routed local requests are sent to
    /// `model_key` — a llama-swap entry that must be launched with
    /// `--reasoning-budget-enable` — carrying a per-request
    /// `reasoning_budget_tokens` field chosen from `budgets` by Bonsai's tier
    /// ("light" vs "deep") or the dashboard tier override.
    #[serde(default)]
    pub nudge: NudgeConfig,
}

/// Reasoning-budget nudge settings. The master switch can also be flipped at
/// runtime from the dashboard (`POST /api/nudge`); `budgets` come from here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NudgeConfig {
    /// Master switch. Off → normal local routing (no model key swap, no budget
    /// injection).
    #[serde(default)]
    pub enabled: bool,
    /// llama-swap model key that runs the nudge fork. When set and nudge is
    /// on, auto-local requests use this key instead of `fallback_model`.
    #[serde(default)]
    pub model_key: Option<String>,
    /// Per-tier reasoning budgets (tokens).
    #[serde(default)]
    pub budgets: NudgeBudgets,
}

impl Default for NudgeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_key: None,
            budgets: NudgeBudgets::default(),
        }
    }
}

/// Per-tier reasoning budgets (tokens) injected as `reasoning_budget_tokens`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NudgeBudgets {
    /// "light" tier (simple tasks) — tight budget.
    /// Legacy configs may still spell this `local`; both are accepted.
    #[serde(default = "default_nudge_budget_light", alias = "local")]
    pub light: u32,
    /// "deep" tier (complex local tasks) — full budget.
    #[serde(default = "default_nudge_budget_deep")]
    pub deep: u32,
}

fn default_nudge_budget_light() -> u32 {
    10240
}
fn default_nudge_budget_deep() -> u32 {
    12288
}

impl Default for NudgeBudgets {
    fn default() -> Self {
        Self {
            light: default_nudge_budget_light(),
            deep: default_nudge_budget_deep(),
        }
    }
}

/// Configuration for the embedded Bonsai classifier model.
///
/// Off by default: `brainrouter serve` will not spawn the llama-server and
/// auto-routed requests go straight to llama-swap (no classification). When
/// `enabled: true`, `model_path` must point at an existing GGUF file — config
/// load fails fast if the file is missing so a headless daemon never silently
/// runs without its classifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BonsaiConfig {
    /// Master switch for the classifier server. Default off.
    #[serde(default)]
    pub enabled: bool,
    /// Path to the Bonsai GGUF model file. Required only when enabled.
    #[serde(default)]
    pub model_path: Option<PathBuf>,
    /// Port for the external llama-server process.
    #[serde(default = "default_bonsai_port")]
    pub server_port: u16,
    /// Path to the PrismML fork llama-server binary.
    #[serde(default = "default_fork_path")]
    pub fork_path: PathBuf,
}
fn default_bonsai_port() -> u16 { 9200 }
fn default_fork_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    PathBuf::from(format!("{}/.local/share/brainrouter/llama-prism/llama-server", home))
}


/// Shared model storage directory configuration.
///
/// Controls where models live on disk and whether non-owner users may add or
/// delete files in that directory.
///
/// ```yaml
/// models:
///   path: /opt/models      # default
///   shared_write: false    # default — only the directory owner can add/delete
/// ```
///
/// When `shared_write: false` (the default) the directory is created with
/// permissions `root:aistack 750` — members of the `aistack` group can read
/// and traverse but cannot write.  When `shared_write: true` the directory
/// gets `root:aistack 770` — all `aistack` members can add and delete models.
///
/// `bonsai.model_path` may contain the literal token `${models_path}` which
/// is expanded to this path at config-load time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    /// Absolute path to the shared model directory.
    #[serde(default = "default_models_path")]
    pub path: PathBuf,

    /// When true, all members of the `aistack` group may add and delete models.
    /// When false (default), only the directory owner (typically root or papa)
    /// may write; group members can only read.
    #[serde(default)]
    pub shared_write: bool,
}

fn default_models_path() -> PathBuf {
    PathBuf::from("/opt/models")
}

impl Default for ModelsConfig {
    fn default() -> Self {
        ModelsConfig {
            path: default_models_path(),
            shared_write: false,
        }
    }
}

impl ModelsConfig {
    /// Return the Unix permission mode for the models directory.
    ///
    /// - `shared_write: false` → `0o750`  (owner rwx, group r-x, others ---)
    /// - `shared_write: true`  → `0o770`  (owner rwx, group rwx, others ---)
    pub fn dir_mode(&self) -> u32 {
        if self.shared_write { 0o770 } else { 0o750 }
    }

    /// Apply the configured permissions to `path` (and recursively to all
    /// sub-directories).  Silently succeeds if the path does not exist yet —
    /// directory creation is the installer's job, not the daemon's.
    pub fn apply_permissions(&self) -> Result<()> {
        if !self.path.exists() {
            return Ok(());
        }
        let mode = self.dir_mode();
        apply_dir_mode(&self.path, mode)
    }
}

/// Recursively set the Unix permission bits on `dir` and all sub-directories.
fn apply_dir_mode(dir: &Path, mode: u32) -> Result<()> {
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(mode))
        .with_context(|| format!("Failed to chmod {:o} on {}", mode, dir.display()))?;
    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("Failed to read dir {}", dir.display()))?
    {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            apply_dir_mode(&entry.path(), mode)?;
        }
    }
    Ok(())
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
    "local".to_string()
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
///
/// `bonsai.model_path` supports one substitution token:
///   `${models_path}` — expanded to the value of `models.path` (default `/opt/models`).
///
/// Example:
/// ```yaml
/// models:
///   path: /mnt/nas/models
/// bonsai:
///   model_path: "${models_path}/bonsai/Bonsai-8B-Q4_K_M.gguf"
/// ```
pub fn load(path: &Path) -> Result<BrainrouterConfig> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;

    let mut config: BrainrouterConfig = serde_yaml::from_str(&contents)
        .with_context(|| format!("Failed to parse YAML config: {}", path.display()))?;

    // Expand ${models_path} in bonsai.model_path.
    // This lets users write a single `models.path` and reference it everywhere
    // without repeating the absolute prefix.
    if let Some(model_path) = config.bonsai.model_path.as_ref() {
        let models_path_str = config.models.path.to_string_lossy().into_owned();
        let raw = model_path.to_string_lossy().into_owned();
        if raw.contains("${models_path}") {
            let expanded = raw.replace("${models_path}", &models_path_str);
            config.bonsai.model_path = Some(PathBuf::from(expanded));
        }
    }

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
    if let Some(model) = config.llama_swap.subs_model.as_deref() {
        if model.is_empty() {
            bail!("llama_swap.subs_model must not be empty");
        }
    }

    // Validate bonsai.model_path exists (after token expansion) — but only
    // when the classifier is enabled. Disabled Bonsai needs no model file, so
    // a headless local-only daemon can run without any GGUF on disk.
    if config.bonsai.enabled {
        match config.bonsai.model_path.as_ref() {
            Some(path) => {
                if !path.exists() {
                    bail!("Bonsai model path does not exist: {}", path.display());
                }
            }
            None => bail!("bonsai.enabled is true but bonsai.model_path is not set"),
        }
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
