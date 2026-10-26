use std::os::unix::fs::PermissionsExt;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Top-level brainrouter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainrouterConfig {
    /// v1 cloud gateway. Migrated into a `manifest`-kind provider record by
    /// `v2.0-p6`, which REMOVES this block from the YAML post-migration; the
    /// field stays parse-compatible so pre-migration files boot.
    #[serde(default)]
    pub manifest: ManifestConfig,
    /// v2 L7: the swap supervisor section. Pre-`v2.0-p7` configs spell the
    /// section `llama_swap:`; the alias keeps them booting until the migration
    /// renames it in place (keys move verbatim).
    #[serde(default, alias = "llama_swap")]
    pub swap: SwapConfig,
    /// v2 L3: the decision-model slot (R4). Replaces the v1 `bonsai:` block
    /// (which migrates into `kind: openai-compatible, prompt_profile: bonsai`,
    /// with the launch backend config preserved under `launch:`) and the v1
    /// `routing:` block (which migrates into role cards).
    #[serde(default)]
    pub decision_model: DecisionModelConfig,
    /// Shared model storage directory settings.
    #[serde(default)]
    pub models: ModelsConfig,
    #[serde(default)]
    pub review: ReviewConfig,
    /// Admission policy for the memory-gated local code reviewer (design H1).
    /// Separate from `review` (which is `deny_unknown_fields`) so these safety
    /// fields survive a review-card update.
    #[serde(default)]
    pub review_admission: ReviewAdmissionConfig,
    /// Persistent imported benchmark results and explorer.
    #[serde(default)]
    pub benchmarks: BenchmarkConfig,
    /// v2 L6: LAN federation listener (default off = v1 attack surface).
    #[serde(default)]
    pub lan: LanConfig,
}

/// v2 L6 LAN federation config (spec "LAN federation").
///
/// Default `enabled: false` means the listener is never created; the primary
/// loopback surface is byte-unchanged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Bind address for the SECOND listener, e.g. `0.0.0.0:9099`.
    #[serde(default)]
    pub bind: String,
    /// Credential id holding the bearer token (store-managed, rotate + revoke).
    #[serde(default)]
    pub token_credential: String,
    /// Model keys a token holder may address (double-enforced: raw + resolved).
    #[serde(default)]
    pub expose: Vec<String>,
    #[serde(default)]
    pub rate: LanRate,
}

impl Default for LanConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bind: String::new(),
            token_credential: String::new(),
            expose: Vec::new(),
            rate: LanRate::default(),
        }
    }
}

/// Per-token token bucket (spec defaults: burst 20, 4 requests/second).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanRate {
    #[serde(default = "default_lan_burst")]
    pub burst: u64,
    #[serde(default = "default_lan_per_sec")]
    pub per_sec: u64,
}

fn default_lan_burst() -> u64 {
    20
}

fn default_lan_per_sec() -> u64 {
    4
}

impl Default for LanRate {
    fn default() -> Self {
        Self {
            burst: default_lan_burst(),
            per_sec: default_lan_per_sec(),
        }
    }
}

/// Storage settings for the benchmark explorer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// SQLite database containing imported benchmark definitions and results.
    #[serde(default = "default_benchmark_database_path")]
    pub database_path: PathBuf,
    /// Optional native Riddllr and Plumebench execution.
    #[serde(default)]
    pub lab: BenchmarkLabConfig,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            database_path: default_benchmark_database_path(),
            lab: BenchmarkLabConfig::default(),
        }
    }
}

/// Native benchmark execution settings.
///
/// Execution is disabled by default. When enabled, Brainrouter reads suite
/// inputs from the configured roots, creates isolated per-job workspaces, and
/// stores results in the benchmark registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkLabConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub riddllr_root: Option<PathBuf>,
    #[serde(default)]
    pub plumebench_root: Option<PathBuf>,
    #[serde(default = "default_benchmark_workspace_path")]
    pub workspace_path: PathBuf,
    #[serde(default = "default_omp_binary")]
    pub omp_bin: PathBuf,
    #[serde(default = "default_plumebench_sandbox_binary")]
    pub plumebench_sandbox_bin: PathBuf,
    #[serde(default = "default_python_binary")]
    pub python_bin: PathBuf,
    #[serde(default = "default_benchmark_job_timeout_seconds")]
    pub max_job_seconds: u64,
    #[serde(default = "default_riddllr_max_tokens")]
    pub riddllr_max_tokens: u32,
    #[serde(default = "default_plumebench_max_turns")]
    pub plumebench_max_turns: u32,
    #[serde(default = "default_plumebench_thinking")]
    pub plumebench_thinking: String,
}

impl Default for BenchmarkLabConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            riddllr_root: None,
            plumebench_root: None,
            workspace_path: default_benchmark_workspace_path(),
            omp_bin: default_omp_binary(),
            plumebench_sandbox_bin: default_plumebench_sandbox_binary(),
            python_bin: default_python_binary(),
            max_job_seconds: default_benchmark_job_timeout_seconds(),
            riddllr_max_tokens: default_riddllr_max_tokens(),
            plumebench_max_turns: default_plumebench_max_turns(),
            plumebench_thinking: default_plumebench_thinking(),
        }
    }
}

fn default_benchmark_database_path() -> PathBuf {
    let base = std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
            PathBuf::from(home).join(".local/share")
        });
    base.join("brainrouter").join("benchmarks.sqlite3")
}

fn default_benchmark_workspace_path() -> PathBuf {
    let base = std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
            PathBuf::from(home).join(".local/share")
        });
    base.join("brainrouter").join("benchmark-lab")
}

fn default_omp_binary() -> PathBuf {
    PathBuf::from("omp")
}

fn default_plumebench_sandbox_binary() -> PathBuf {
    PathBuf::from("bwrap")
}

fn default_python_binary() -> PathBuf {
    PathBuf::from("python3")
}

fn default_benchmark_job_timeout_seconds() -> u64 {
    900
}

fn default_riddllr_max_tokens() -> u32 {
    4096
}

fn default_plumebench_max_turns() -> u32 {
    40
}

fn default_plumebench_thinking() -> String {
    "low".to_string()
}

/// Configuration for the Manifest cloud LLM router.
/// Manifest exposes an OpenAI-compatible endpoint; brainrouter delegates all
/// managed cloud routing decisions to Manifest with model="auto"; explicit IDs pass through.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ManifestConfig {
    /// Master switch for the cloud backend. Off by default: `model=cloud` and
    /// Bonsai-classified cloud requests skip Manifest and fall back to
    /// llama-swap. Set `enabled: true` to use the cloud.
    #[serde(default)]
    pub enabled: bool,

    /// Base URL of the Manifest instance. Example: "http://localhost:2099".
    /// Optional post-`v2.0-p6` (the migrated provider record owns it).
    #[serde(default)]
    pub base_url: String,

    /// Name of the environment variable holding the Manifest API key (mnfst_*).
    /// Optional for local deployments that don't require auth.
    #[serde(default)]
    pub api_key_env: Option<String>,
}

/// Engine mode for the local serving layer (v2 L7 "the flip").
///
/// * `shim` (default) — today's behavior: every local route dials the
///   llama-swap shim. Byte-identical to v1 through the whole leg.
/// * `native` — the swap supervisor owns every local model. Boot REFUSES while
///   the shim still reports loaded models, and no component dials `:8081`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SwapMode {
    /// llama-swap shim serves local traffic (default; floor mode).
    #[default]
    Shim,
    /// Native swap supervisor serves local traffic.
    Native,
}

impl SwapMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Shim => "shim",
            Self::Native => "native",
        }
    }
    pub fn is_native(self) -> bool {
        matches!(self, Self::Native)
    }
}

/// Configuration for the local serving layer (v2 L7 `swap:` section; the
/// pre-`v2.0-p7` name is `llama_swap:`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapConfig {
    /// Which engine serves local traffic. Default `shim` — the whole leg keeps
    /// the v1 floor; the flip is `mode: native` + restart.
    #[serde(default)]
    pub mode: SwapMode,

    /// Import source for the swap catalog: the llama-swap config the native
    /// supervisor imports at boot. Defaults to
    /// `~/.config/llama-swap/config.yaml`; fedora hosts point it at their
    /// docker-mounted config (e.g. `/opt/ai/llama-swap/config.yaml`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_path: Option<PathBuf>,

    /// Deprecated: the shim URL is fixed (`http://localhost:8081/v1`) in the
    /// end state. The key is still HONORED in shim mode so fixtures and
    /// non-default shim ports keep working, and `v2.0-p7` removes it from the
    /// YAML with a "no longer used" note.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,

    /// The model key to use when falling back from cloud, or when the user
    /// sends model="auto" and the classifier picks local without a specific
    /// model hint. Must match an entry in the swap catalog.
    pub fallback_model: String,

    /// Explicit list of model keys served locally. When a request arrives
    /// with `model` set to one of these names, brainrouter routes directly to
    /// the local engine without consulting the classifier — the user's model
    /// choice is authoritative. Omit or leave empty to rely on the
    /// `brainrouter/<model>` prefix convention for direct-local routing.
    ///
    /// Matching is case-sensitive; model names must exactly match catalog keys.
    #[serde(default)]
    pub local_models: Vec<String>,

    /// Optional path to a custom system prompt file for local routing mode.
    /// If absent, the built-in lean prompt is used.
    #[serde(default)]
    pub local_system_prompt: Option<String>,

    /// Optional subs-pool model key. Requests targeting `subs` or
    /// `brainrouter/subs` route directly to this catalog key, bypassing
    /// classification — intended for a multi-slot pool serving subagent
    /// traffic. Absent → `subs` requests fall back to normal auto routing.
    #[serde(default)]
    pub subs_model: Option<String>,

    /// Optional reasoning-budget nudge settings (voidsurfer/llama.cpp-nudge
    /// fork). When enabled, auto-routed local requests are sent to
    /// `model_key` — a catalog entry that must be launched with
    /// `--reasoning-budget-enable` — carrying a per-request
    /// `reasoning_budget_tokens` field chosen from `budgets` by the
    /// classifier's tier ("light" vs "deep") or the dashboard tier override.
    #[serde(default)]
    pub nudge: NudgeConfig,
}

/// The shim's fixed URL in the end state (no `base_url` key post-migration).
pub const DEFAULT_SHIM_BASE_URL: &str = "http://localhost:8081/v1";

impl SwapConfig {
    /// The shim URL (deprecated override honored, fixed default otherwise).
    pub fn shim_base_url(&self) -> &str {
        self.base_url
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(DEFAULT_SHIM_BASE_URL)
    }

    /// Import source for the swap catalog: the configured path or the
    /// pre-flip llama-swap default.
    pub fn import_config_path(&self) -> PathBuf {
        match self.config_path.as_ref() {
            Some(path) if !path.as_os_str().is_empty() => path.clone(),
            _ => default_import_config_path(),
        }
    }
}

impl Default for SwapConfig {
    fn default() -> Self {
        Self {
            mode: SwapMode::Shim,
            config_path: None,
            base_url: None,
            fallback_model: String::new(),
            local_models: Vec::new(),
            local_system_prompt: None,
            subs_model: None,
            nudge: NudgeConfig::default(),
        }
    }
}

/// Default import source: the llama-swap config the native supervisor takes
/// over. Fedora hosts override `swap.config_path` (docker-mounted file).
pub fn default_import_config_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    PathBuf::from(home).join(".config/llama-swap/config.yaml")
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

/// The decision-model slot kind (L3 spec "Decision-model slot").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum DecisionKind {
    /// v1 selector-matrix parity; always available. Default.
    #[default]
    Heuristic,
    /// OpenAI-compatible classifier endpoint. Bonsai migrates to exactly this
    /// with `prompt_profile: bonsai`.
    OpenaiCompatible,
    /// Typed question schema, strict JSON reply (the Clef contract).
    StructuredSchema,
    /// POST the features JSON to `{url}`; the reply is the decision.
    Webhook,
    /// Server-initiated `decision.classify` over the plugin RPC (protocol v2).
    Plugin,
}

impl DecisionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Heuristic => "heuristic",
            Self::OpenaiCompatible => "openai-compatible",
            Self::StructuredSchema => "structured-schema",
            Self::Webhook => "webhook",
            Self::Plugin => "plugin",
        }
    }
}

/// Launch config for the embedded Bonsai classifier llama-server. The
/// `bonsai_server.rs` launch backend survives L3; the v1 `bonsai:` keys migrate
/// here.
///
/// Off by default: `brainrouter serve` will not spawn the llama-server. When
/// `enabled: true`, `model_path` must point at an existing GGUF file — config
/// load fails fast if the file is missing.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BonsaiLaunch {
    /// Port for the external llama-server process.
    #[serde(default = "default_bonsai_port")]
    pub server_port: u16,
    /// Path to the PrismML fork llama-server binary.
    #[serde(default = "default_fork_path")]
    pub fork_path: PathBuf,
    /// Path to the Bonsai GGUF model file. Required only when the slot is
    /// enabled with the bonsai prompt profile.
    #[serde(default)]
    pub model_path: Option<PathBuf>,
}

/// Configuration of the decision-model slot. Kinds needing an endpoint carry
/// the fields marked optional below; `min_confidence` gates every model kind
/// (default 0.4).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DecisionModelConfig {
    pub enabled: bool,
    pub kind: DecisionKind,
    /// openai-compatible / structured-schema: classifier base URL.
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    /// `"bonsai"` preserves the v1 classifier prompt + one-word grammar.
    #[serde(default)]
    pub prompt_profile: Option<String>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// webhook: features endpoint URL.
    #[serde(default)]
    pub url: Option<String>,
    /// plugin: plugin id serving `decision.classify`.
    #[serde(default)]
    pub plugin_id: Option<String>,
    #[serde(default)]
    pub min_confidence: Option<f64>,
    /// Launch backend config for the migrated bonsai classifier server.
    #[serde(default)]
    pub launch: Option<BonsaiLaunch>,
}

impl Default for DecisionModelConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            kind: DecisionKind::Heuristic,
            base_url: None,
            model: None,
            prompt_profile: None,
            timeout_ms: None,
            url: None,
            plugin_id: None,
            min_confidence: None,
            launch: None,
        }
    }
}

/// The default low-confidence threshold (L3 spec: 0.4).
pub const DEFAULT_MIN_CONFIDENCE: f64 = 0.4;

impl DecisionModelConfig {
    pub fn min_confidence(&self) -> f64 {
        self.min_confidence.unwrap_or(DEFAULT_MIN_CONFIDENCE)
    }

    /// The bonsai profile: an openai-compatible slot carrying v1's prompt.
    pub fn is_bonsai_profile(&self) -> bool {
        self.kind == DecisionKind::OpenaiCompatible
            && self.prompt_profile.as_deref() == Some("bonsai")
    }
}

fn default_bonsai_port() -> u16 {
    9200
}
fn default_fork_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    PathBuf::from(format!(
        "{}/.local/share/brainrouter/llama-prism/llama-server",
        home
    ))
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
        if self.shared_write {
            0o770
        } else {
            0o750
        }
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
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("Failed to read dir {}", dir.display()))?
    {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            apply_dir_mode(&entry.path(), mode)?;
        }
    }
    Ok(())
}

/// Admission policy for a local code reviewer (design H1). Kept out of
/// `ReviewConfig` (which is `deny_unknown_fields` and round-trips through the
/// routing profile) so these safety fields survive a profile update.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ReviewAdmissionConfig {
    /// Measured peak resident memory per local reviewer model key, in MiB. A
    /// model with no entry here is never admitted locally (falls back to cloud
    /// or is blocked) — the gate never guesses a budget.
    pub local_model_budget_mb: std::collections::HashMap<String, u64>,
    /// System/other-workload headroom kept free above the reviewer budget, MiB.
    pub system_reserve_mb: u64,
    /// Concurrent local reviews permitted. Pinned to 1 in this release.
    pub local_review_permits: u32,
}

impl Default for ReviewAdmissionConfig {
    fn default() -> Self {
        ReviewAdmissionConfig {
            local_model_budget_mb: std::collections::HashMap::new(),
            system_reserve_mb: 4096,
            local_review_permits: 1,
        }
    }
}

impl ReviewAdmissionConfig {
    pub fn validate(&self) -> Result<()> {
        if self.local_review_permits != 1 {
            anyhow::bail!("review_admission.local_review_permits must be 1 in this release");
        }
        if self.local_model_budget_mb.values().any(|&v| v == 0) {
            anyhow::bail!("review_admission.local_model_budget_mb values must be greater than 0");
        }
        Ok(())
    }
}

/// Configuration for the review service and escalation dashboard.
///
/// v2 L3: reduced to the non-model survivors. The reviewer MODEL lives on the
/// `review` role card (the single reviewer source of truth); the v1
/// `forced_mode`/`forced_model` YAML keys migrate into it and are removed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewConfig {
    /// Maximum number of LLM review iterations before escalating to human.
    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,

    /// HankNDory design-aware review (design G1). When true, a review only
    /// proceeds if an *approved* design document is loaded, and the reviewer is
    /// told to judge the diff against it. Default off (opt-in).
    #[serde(default)]
    pub hankndory_integration: bool,

    /// Repo-relative path (under `docs/design/`) of the design document the
    /// reviewer should judge against when `hankndory_integration` is on. Unset
    /// while integration is on ⇒ the review fails closed (escalates).
    #[serde(default)]
    pub design_doc_path: Option<String>,

    /// Second-chance reviewer selector (e.g. `cloud/deepseek-ai/DeepSeek-V4-Flash`).
    /// When the primary reviewer backend fails — cloud credits exhausted, circuit
    /// open, unreachable — the review retries ONCE against this selector before
    /// escalating to a human. Cloud reviews only: a local review never migrates
    /// to another model (its rubric and admission permit are model-specific).
    /// Must be a `cloud/...` selector.
    #[serde(default)]
    pub fallback_selector: Option<String>,
}

fn default_max_iterations() -> u32 {
    5
}

impl Default for ReviewConfig {
    fn default() -> Self {
        ReviewConfig {
            max_iterations: default_max_iterations(),
            hankndory_integration: false,
            design_doc_path: None,
            fallback_selector: None,
        }
    }
}

impl ReviewConfig {
    pub fn validate(&self) -> Result<()> {
        crate::roles::validate_iterations(self.max_iterations)?;
        if let Some(sel) = self.fallback_selector.as_deref() {
            let choice = crate::roles::ModelChoice::from_selector(sel)
                .map_err(|e| anyhow::anyhow!("review.fallback_selector: {e}"))?;
            if choice.backend() != "cloud" {
                bail!(
                    "review.fallback_selector must be a cloud selector (cloud/...), got {sel:?}"
                );
            }
        }
        Ok(())
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
    validate(&mut config, path)?;
    Ok(config)
}

/// Normalize and semantically validate a parsed configuration.
///
/// Both daemon startup and the dashboard editor must call this before using or
/// persisting a configuration so a successful save cannot create a file that
/// the next daemon restart rejects.
pub fn validate(config: &mut BrainrouterConfig, path: &Path) -> Result<()> {
    // Expand ${models_path} in the decision slot's launch model_path (the
    // migrated Bonsai GGUF). This lets users write a single `models.path` and
    // reference it everywhere without repeating the absolute prefix.
    if let Some(launch) = config.decision_model.launch.as_mut() {
        if let Some(model_path) = launch.model_path.as_ref() {
            let models_path_str = config.models.path.to_string_lossy().into_owned();
            let raw = model_path.to_string_lossy().into_owned();
            if raw.contains("${models_path}") {
                let expanded = raw.replace("${models_path}", &models_path_str);
                launch.model_path = Some(PathBuf::from(expanded));
            }
        }
    }

    // Validate manifest.base_url. Post-`v2.0-p6` the YAML block is removed
    // (the provider record owns the URL), so only a present value is checked;
    // an enabled block still requires one.
    if config.manifest.base_url.is_empty() {
        if config.manifest.enabled {
            bail!("manifest.base_url must not be empty when manifest.enabled is true");
        }
    } else if !config.manifest.base_url.starts_with("http://")
        && !config.manifest.base_url.starts_with("https://")
    {
        bail!(
            "manifest.base_url must start with http:// or https://, got: {}",
            config.manifest.base_url
        );
    }
    // v2 L6: the LAN second listener is opt-in; when enabled every field it
    // needs must be present so a typo cannot silently bind an unauthenticated
    // or rate-unlimited surface.
    if config.lan.enabled {
        if config.lan.bind.trim().is_empty() {
            bail!("lan.bind must not be empty when lan.enabled is true");
        }
        if config.lan.bind.parse::<std::net::SocketAddr>().is_err() {
            bail!(
                "lan.bind must be a socket address (ip:port), got: {}",
                config.lan.bind
            );
        }
        if config.lan.token_credential.trim().is_empty() {
            bail!("lan.token_credential must not be empty when lan.enabled is true");
        }
        if config.lan.rate.burst == 0 {
            bail!("lan.rate.burst must be greater than zero");
        }
        if config.lan.rate.per_sec == 0 {
            bail!("lan.rate.per_sec must be greater than zero");
        }
    }

    // v2 L7: validate the swap section. `base_url` is optional post-`v2.0-p7`
    // (the shim URL has a fixed default); when present it must be a URL.
    if let Some(base_url) = config.swap.base_url.as_deref() {
        if !base_url.is_empty()
            && !base_url.starts_with("http://")
            && !base_url.starts_with("https://")
        {
            bail!(
                "swap.base_url must start with http:// or https://, got: {}",
                base_url
            );
        }
    }
    if let Some(path) = config.swap.config_path.as_ref() {
        if path.as_os_str().is_empty() {
            bail!("swap.config_path must not be empty when set");
        }
    }
    if config.swap.fallback_model.is_empty() {
        bail!("swap.fallback_model must not be empty");
    }
    if let Some(model) = config.swap.subs_model.as_deref() {
        if model.is_empty() {
            bail!("swap.subs_model must not be empty");
        }
    }
    if config.benchmarks.database_path.as_os_str().is_empty() {
        bail!("benchmarks.database_path must not be empty");
    }
    if config.benchmarks.lab.workspace_path.as_os_str().is_empty() {
        bail!("benchmarks.lab.workspace_path must not be empty");
    }
    if config.benchmarks.lab.omp_bin.as_os_str().is_empty() {
        bail!("benchmarks.lab.omp_bin must not be empty");
    }
    if config
        .benchmarks
        .lab
        .plumebench_sandbox_bin
        .as_os_str()
        .is_empty()
    {
        bail!("benchmarks.lab.plumebench_sandbox_bin must not be empty");
    }
    if config.benchmarks.lab.python_bin.as_os_str().is_empty() {
        bail!("benchmarks.lab.python_bin must not be empty");
    }
    if !(1..=86_400).contains(&config.benchmarks.lab.max_job_seconds) {
        bail!("benchmarks.lab.max_job_seconds must be between 1 and 86400");
    }
    if config.benchmarks.lab.riddllr_max_tokens == 0 {
        bail!("benchmarks.lab.riddllr_max_tokens must be positive");
    }
    if config.benchmarks.lab.plumebench_max_turns == 0 {
        bail!("benchmarks.lab.plumebench_max_turns must be positive");
    }
    if !matches!(
        config.benchmarks.lab.plumebench_thinking.as_str(),
        "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" | "auto"
    ) {
        bail!(
            "benchmarks.lab.plumebench_thinking must be off, minimal, low, medium, high, xhigh, max, or auto"
        );
    }
    config
        .review
        .validate()
        .with_context(|| format!("Invalid review configuration in {}", path.display()))?;
    validate_decision_model(&mut config.decision_model, path)?;
    Ok(())
}

/// Validate the decision-model slot (L3). Kinds needing an endpoint require
/// it; the migrated bonsai profile additionally keeps v1's fail-fast rule:
/// when the slot is enabled, the launch backend needs an existing GGUF.
fn validate_decision_model(slot: &mut DecisionModelConfig, path: &Path) -> Result<()> {
    let need = |field: &str, present: bool| -> Result<()> {
        if !present {
            bail!(
                "decision_model.{field} is required for kind {:?} (in {})",
                slot.kind,
                path.display()
            );
        }
        Ok(())
    };
    match slot.kind {
        DecisionKind::Heuristic => {}
        DecisionKind::OpenaiCompatible | DecisionKind::StructuredSchema => {
            need("base_url", slot.base_url.as_deref().is_some())?;
            need("model", slot.model.as_deref().is_some())?;
        }
        DecisionKind::Webhook => need("url", slot.url.as_deref().is_some())?,
        DecisionKind::Plugin => need("plugin_id", slot.plugin_id.as_deref().is_some())?,
    }
    if let Some(ms) = slot.timeout_ms {
        if ms == 0 {
            bail!("decision_model.timeout_ms must be greater than zero (in {})", path.display());
        }
    }
    if let Some(conf) = slot.min_confidence {
        if !(0.0..=1.0).contains(&conf) {
            bail!("decision_model.min_confidence must be within 0.0..=1.0 (in {})", path.display());
        }
    }
    // The v1 fail-fast rule, migrated — scoped to the launch backend: an
    // enabled slot whose LAUNCH config exists (daemon-managed classifier
    // server) needs an existing GGUF. An externally-managed endpoint (no
    // launch block) has no daemon-side model file to check.
    if slot.enabled && slot.is_bonsai_profile() {
        if let Some(launch) = slot.launch.as_ref() {
            match launch.model_path.as_ref() {
                Some(model_path) => {
                    if !model_path.exists() {
                        bail!(
                            "Bonsai model path does not exist: {} (decision_model.launch.model_path)",
                            model_path.display()
                        );
                    }
                }
                None => bail!(
                    "decision_model is enabled with prompt_profile bonsai and a launch backend, but decision_model.launch.model_path is not set (in {})",
                    path.display()
                ),
            }
        }
    }
    Ok(())
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

    fn parse_swap(yaml: &str) -> SwapConfig {
        serde_yaml::from_str(yaml).expect("parse failed")
    }

    #[test]
    fn local_models_defaults_to_empty() {
        let cfg = parse_swap("base_url: http://localhost:8081/v1\nfallback_model: default\n");
        assert!(cfg.local_models.is_empty());
    }

    #[test]
    fn local_models_parses_list() {
        let cfg = parse_swap(
            "base_url: http://localhost:8081/v1\nfallback_model: default\nlocal_models:\n  - qwen3\n  - mistral-7b\n",
        );
        assert_eq!(cfg.local_models, vec!["qwen3", "mistral-7b"]);
    }

    #[test]
    fn local_models_membership_check() {
        // Mirrors the router check: `self.local_models.contains(&requested_model)`
        let cfg = parse_swap(
            "base_url: http://localhost:8081/v1\nfallback_model: default\nlocal_models:\n  - qwen3\n",
        );
        assert!(cfg.local_models.contains(&"qwen3".to_string()));
        assert!(!cfg.local_models.contains(&"claude-3-opus".to_string()));
        // The brainrouter/ prefix is handled before the local_models check in the
        // router, so the raw prefix form is not expected to match here.
        assert!(!cfg.local_models.contains(&"brainrouter/qwen3".to_string()));
    }

    #[test]
    fn swap_mode_defaults_to_shim_and_parses_native() {
        let cfg = parse_swap("fallback_model: m\n");
        assert_eq!(cfg.mode, SwapMode::Shim);
        let cfg = parse_swap("mode: native\nfallback_model: m\n");
        assert_eq!(cfg.mode, SwapMode::Native);
        assert!(cfg.mode.is_native());
        // An unknown spelling is a parse error, not a silent shim fallback.
        assert!(serde_yaml::from_str::<SwapConfig>("mode: turbo\nfallback_model: m\n").is_err());
    }

    #[test]
    fn swap_section_accepts_the_pre_migration_spelling() {
        // v2.0-p7 renames `llama_swap:` → `swap:`; pre-migration files (and
        // frozen v1 fixtures) keep booting through the serde alias.
        let migrated: BrainrouterConfig =
            serde_yaml::from_str("swap:\n  fallback_model: a\n").expect("swap: parses");
        assert_eq!(migrated.swap.fallback_model, "a");
        let legacy: BrainrouterConfig =
            serde_yaml::from_str("llama_swap:\n  fallback_model: b\n").expect("alias parses");
        assert_eq!(legacy.swap.fallback_model, "b");
        // The end-state serialization writes `swap:` only, and the section
        // itself carries no `base_url`.
        let yaml = serde_yaml::to_string(&legacy).unwrap();
        assert!(yaml.contains("\nswap:"), "{yaml}");
        assert!(!yaml.contains("llama_swap:"), "{yaml}");
        let reparsed: serde_yaml::Value = serde_yaml::from_str(&yaml).unwrap();
        let swap = reparsed.get("swap").expect("swap: section");
        assert!(swap.get("base_url").is_none(), "{yaml}");
    }

    #[test]
    fn outbound_validation_accepts_a_post_migration_swap_section() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("brainrouter.yaml");
        std::fs::write(&path, "swap:\n  mode: shim\n  fallback_model: m\n").unwrap();
        let mut config = load(&path).expect("post-migration config boots");
        assert_eq!(config.swap.shim_base_url(), "http://localhost:8081/v1");
        assert_eq!(
            config.swap.import_config_path(),
            default_import_config_path()
        );
        assert!(validate(&mut config, &path).is_ok());
    }

    #[test]
    fn swap_base_url_override_is_honored_in_shim_mode_only_input() {
        let cfg = parse_swap("base_url: http://127.0.0.1:19999/v1\nfallback_model: m\n");
        assert_eq!(cfg.shim_base_url(), "http://127.0.0.1:19999/v1");
        let cfg = parse_swap("fallback_model: m\n");
        assert_eq!(cfg.shim_base_url(), DEFAULT_SHIM_BASE_URL);
    }

    #[test]
    fn swap_config_path_defaults_to_the_llama_swap_location() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("brainrouter.yaml");
        std::fs::write(
            &path,
            "swap:\n  config_path: /opt/ai/llama-swap/config.yaml\n  fallback_model: m\n",
        )
        .unwrap();
        let config = load(&path).unwrap();
        assert_eq!(
            config.swap.import_config_path(),
            PathBuf::from("/opt/ai/llama-swap/config.yaml")
        );
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

    #[test]
    fn benchmark_database_path_has_a_per_user_default() {
        let config = BenchmarkConfig::default();
        assert!(config
            .database_path
            .ends_with("brainrouter/benchmarks.sqlite3"));
        assert!(config
            .lab
            .workspace_path
            .ends_with("brainrouter/benchmark-lab"));
        assert!(!config.lab.enabled);
    }
}
