//! Persistent LLM benchmark registry, ingestion boundary, query API, and explorer.
//!
//! This module stores imported benchmark results only. It never launches a model
//! or benchmark process.

use bytes::Bytes;
use chrono::{DateTime, Utc};
use http_body_util::{combinators::UnsyncBoxBody, BodyExt, Full, LengthLimitError, Limited};
use hyper::{body::Incoming, Request, Response, StatusCode};
use rusqlite::{
    params, params_from_iter, types::Value as SqlValue, Connection, OptionalExtension, Transaction,
    TransactionBehavior,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    convert::Infallible,
    fmt,
    path::{Path, PathBuf},
};

const MIGRATION_0001: &str = include_str!("../migrations/0001_benchmark_explorer.sql");
const EXPLORER_HTML: &str = include_str!("escalation/templates/benchmarks.html");
const MAX_INGEST_BYTES: usize = 16 * 1024 * 1024;
const MAX_PAGE_SIZE: u32 = 100;

#[derive(Debug)]
pub enum BenchmarkError {
    Validation(String),
    Conflict(String),
    NotFound(String),
    Database(rusqlite::Error),
    Io(std::io::Error),
}

impl fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(message) | Self::Conflict(message) | Self::NotFound(message) => {
                f.write_str(message)
            }
            Self::Database(error) => write!(f, "benchmark database error: {error}"),
            Self::Io(error) => write!(f, "benchmark database I/O error: {error}"),
        }
    }
}

impl std::error::Error for BenchmarkError {}

impl From<rusqlite::Error> for BenchmarkError {
    fn from(error: rusqlite::Error) -> Self {
        if error.sqlite_error_code() == Some(rusqlite::ErrorCode::ConstraintViolation) {
            Self::Conflict(format!(
                "benchmark data conflicts with an existing record: {error}"
            ))
        } else {
            Self::Database(error)
        }
    }
}

impl From<std::io::Error> for BenchmarkError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

type BenchmarkResult<T> = Result<T, BenchmarkError>;

macro_rules! string_enum {
    ($name:ident { $($variant:ident => $value:literal),+ $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
        pub enum $name {
            $(#[serde(rename = $value)] $variant),+
        }

        impl $name {
            pub fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $value),+
                }
            }
        }
    };
}

string_enum!(ModelKind {
    Dense => "dense",
    Moe => "moe",
    Hybrid => "hybrid",
    Unknown => "unknown",
});
string_enum!(Backend {
    Cpu => "cpu",
    Cuda => "cuda",
    Rocm => "rocm",
    Vulkan => "vulkan",
    Metal => "metal",
    Sycl => "sycl",
    Rpc => "rpc",
    Other => "other",
});
string_enum!(FeatureState {
    Unsupported => "unsupported",
    Disabled => "disabled",
    Enabled => "enabled",
    RequestedUnavailable => "requested_unavailable",
});
string_enum!(SpeculatorType {
    None => "none",
    Mtp => "mtp",
    Ngram => "ngram",
    DraftModel => "draft_model",
    Eagle => "eagle",
    Dflash => "dflash",
    Combined => "combined",
    Other => "other",
});
string_enum!(WorkloadType {
    Performance => "performance",
    CodeQuality => "code_quality",
    Perplexity => "perplexity",
    Retrieval => "retrieval",
    Mixed => "mixed",
});
string_enum!(RunStatus {
    Planned => "planned",
    Running => "running",
    Succeeded => "succeeded",
    Failed => "failed",
    Oom => "oom",
    Timeout => "timeout",
    Cancelled => "cancelled",
    Skipped => "skipped",
});
string_enum!(NgramStorageLocation {
    Cpu => "cpu",
    Gpu => "gpu",
    Unified => "unified",
    Unknown => "unknown",
});

fn default_model_kind() -> ModelKind {
    ModelKind::Unknown
}

fn default_feature_disabled() -> FeatureState {
    FeatureState::Disabled
}

fn default_speculator_type() -> SpeculatorType {
    SpeculatorType::None
}

fn default_true() -> bool {
    true
}

fn default_one_u64() -> u64 {
    1
}

fn default_top_p() -> f64 {
    1.0
}

fn default_repetition_penalty() -> f64 {
    1.0
}

fn new_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

fn require_text(name: &str, value: &str) -> BenchmarkResult<()> {
    if value.trim().is_empty() {
        return Err(BenchmarkError::Validation(format!(
            "{name} must not be empty"
        )));
    }
    Ok(())
}

fn require_sha256(name: &str, value: &str) -> BenchmarkResult<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(BenchmarkError::Validation(format!(
            "{name} must be a lowercase 64-character SHA-256 digest"
        )));
    }
    Ok(())
}

fn require_finite_non_negative(name: &str, value: Option<f64>) -> BenchmarkResult<()> {
    if let Some(value) = value {
        if !value.is_finite() || value < 0.0 {
            return Err(BenchmarkError::Validation(format!(
                "{name} must be finite and non-negative"
            )));
        }
    }
    Ok(())
}

fn require_percentage(name: &str, value: Option<f64>) -> BenchmarkResult<()> {
    require_finite_non_negative(name, value)?;
    if value.is_some_and(|value| value > 100.0) {
        return Err(BenchmarkError::Validation(format!(
            "{name} must be between 0 and 100"
        )));
    }
    Ok(())
}

fn require_ratio(name: &str, value: Option<f64>) -> BenchmarkResult<()> {
    require_finite_non_negative(name, value)?;
    if value.is_some_and(|value| value > 1.0) {
        return Err(BenchmarkError::Validation(format!(
            "{name} must be between 0 and 1"
        )));
    }
    Ok(())
}

fn require_sqlite_integer(name: &str, value: u64) -> BenchmarkResult<()> {
    if value > i64::MAX as u64 {
        return Err(BenchmarkError::Validation(format!(
            "{name} exceeds SQLite's signed 64-bit integer limit"
        )));
    }
    Ok(())
}

fn require_optional_sqlite_integer(name: &str, value: Option<u64>) -> BenchmarkResult<()> {
    if let Some(value) = value {
        require_sqlite_integer(name, value)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDefinition {
    pub id: String,
    pub family: String,
    pub architecture: String,
    pub checkpoint: String,
    pub revision: String,
    pub tokenizer_id: String,
    #[serde(default)]
    pub tokenizer_revision: Option<String>,
    #[serde(default)]
    pub parameter_count_total: Option<u64>,
    #[serde(default)]
    pub parameter_count_active: Option<u64>,
    #[serde(default = "default_model_kind")]
    pub model_kind: ModelKind,
    #[serde(default)]
    pub native_context_tokens: Option<u64>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

impl ModelDefinition {
    fn validate(&self) -> BenchmarkResult<()> {
        for (name, value) in [
            ("model.id", self.id.as_str()),
            ("model.family", self.family.as_str()),
            ("model.architecture", self.architecture.as_str()),
            ("model.checkpoint", self.checkpoint.as_str()),
            ("model.revision", self.revision.as_str()),
            ("model.tokenizer_id", self.tokenizer_id.as_str()),
        ] {
            require_text(name, value)?;
        }
        if self.parameter_count_total == Some(0)
            || self.parameter_count_active == Some(0)
            || self.native_context_tokens == Some(0)
        {
            return Err(BenchmarkError::Validation(
                "model parameter and context counts must be positive".into(),
            ));
        }
        if matches!(
            (self.parameter_count_active, self.parameter_count_total),
            (Some(active), Some(total)) if active > total
        ) {
            return Err(BenchmarkError::Validation(
                "model.parameter_count_active cannot exceed parameter_count_total".into(),
            ));
        }
        require_optional_sqlite_integer("model.parameter_count_total", self.parameter_count_total)?;
        require_optional_sqlite_integer(
            "model.parameter_count_active",
            self.parameter_count_active,
        )?;
        require_optional_sqlite_integer("model.native_context_tokens", self.native_context_tokens)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactDefinition {
    pub id: String,
    pub model_id: String,
    pub format: String,
    pub quant_family: String,
    pub quant_name: String,
    #[serde(default)]
    pub average_bits_per_weight: Option<f64>,
    pub disk_bytes: u64,
    pub sha256: String,
    #[serde(default)]
    pub source_uri: Option<String>,
    #[serde(default)]
    pub imatrix_used: bool,
    #[serde(default)]
    pub conversion_tool: Option<String>,
    #[serde(default)]
    pub conversion_commit: Option<String>,
    #[serde(default)]
    pub conversion_command: Option<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

impl ArtifactDefinition {
    fn validate(&self) -> BenchmarkResult<()> {
        for (name, value) in [
            ("artifact.id", self.id.as_str()),
            ("artifact.model_id", self.model_id.as_str()),
            ("artifact.format", self.format.as_str()),
            ("artifact.quant_family", self.quant_family.as_str()),
            ("artifact.quant_name", self.quant_name.as_str()),
        ] {
            require_text(name, value)?;
        }
        require_sha256("artifact.sha256", &self.sha256)?;
        require_sqlite_integer("artifact.disk_bytes", self.disk_bytes)?;
        if self
            .average_bits_per_weight
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(BenchmarkError::Validation(
                "artifact.average_bits_per_weight must be finite and positive".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeDefinition {
    pub id: String,
    pub repository: String,
    pub fork_name: String,
    pub commit_sha: String,
    #[serde(default)]
    pub dirty_tree: bool,
    pub compiler: String,
    pub compiler_version: String,
    pub backend: Backend,
    #[serde(default)]
    pub build_flags: Vec<String>,
    #[serde(default)]
    pub capabilities: BTreeMap<String, FeatureState>,
    #[serde(default)]
    pub executable_sha256: Option<String>,
    #[serde(default)]
    pub container_digest: Option<String>,
    pub built_at: DateTime<Utc>,
}

impl RuntimeDefinition {
    fn validate(&self) -> BenchmarkResult<()> {
        for (name, value) in [
            ("runtime.id", self.id.as_str()),
            ("runtime.repository", self.repository.as_str()),
            ("runtime.fork_name", self.fork_name.as_str()),
            ("runtime.commit_sha", self.commit_sha.as_str()),
            ("runtime.compiler", self.compiler.as_str()),
            ("runtime.compiler_version", self.compiler_version.as_str()),
        ] {
            require_text(name, value)?;
        }
        if let Some(digest) = &self.executable_sha256 {
            require_sha256("runtime.executable_sha256", digest)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuDevice {
    pub index: u64,
    pub name: String,
    #[serde(default)]
    pub vram_bytes: Option<u64>,
    #[serde(default)]
    pub pci_id: Option<String>,
    #[serde(default)]
    pub driver: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareProfile {
    pub id: String,
    #[serde(default)]
    pub hostname_hash: Option<String>,
    pub cpu_model: String,
    #[serde(default)]
    pub physical_cores: Option<u64>,
    #[serde(default)]
    pub logical_cores: Option<u64>,
    pub system_ram_bytes: u64,
    #[serde(default)]
    pub gpus: Vec<GpuDevice>,
    #[serde(default)]
    pub unified_memory: bool,
    pub os_name: String,
    pub os_version: String,
    pub kernel: String,
    #[serde(default)]
    pub driver_versions: BTreeMap<String, String>,
    #[serde(default)]
    pub power_profile: Option<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
    pub captured_at: DateTime<Utc>,
}

impl HardwareProfile {
    fn validate(&self) -> BenchmarkResult<()> {
        for (name, value) in [
            ("hardware.id", self.id.as_str()),
            ("hardware.cpu_model", self.cpu_model.as_str()),
            ("hardware.os_name", self.os_name.as_str()),
            ("hardware.os_version", self.os_version.as_str()),
            ("hardware.kernel", self.kernel.as_str()),
        ] {
            require_text(name, value)?;
        }
        if self.system_ram_bytes == 0
            || self.physical_cores == Some(0)
            || self.logical_cores == Some(0)
            || self.gpus.iter().any(|gpu| gpu.vram_bytes == Some(0))
        {
            return Err(BenchmarkError::Validation(
                "hardware memory and core counts must be positive".into(),
            ));
        }
        if self.gpus.iter().any(|gpu| gpu.name.trim().is_empty()) {
            return Err(BenchmarkError::Validation(
                "hardware.gpus[].name must not be empty".into(),
            ));
        }
        require_optional_sqlite_integer("hardware.physical_cores", self.physical_cores)?;
        require_optional_sqlite_integer("hardware.logical_cores", self.logical_cores)?;
        require_sqlite_integer("hardware.system_ram_bytes", self.system_ram_bytes)?;
        for gpu in &self.gpus {
            require_sqlite_integer("hardware.gpus[].index", gpu.index)?;
            require_optional_sqlite_integer("hardware.gpus[].vram_bytes", gpu.vram_bytes)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadDefinition {
    pub id: String,
    pub name: String,
    pub version: String,
    pub workload_type: WorkloadType,
    pub manifest_sha256: String,
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub output_tokens: Option<u64>,
    #[serde(default)]
    pub corpus_bytes: Option<u64>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

impl WorkloadDefinition {
    fn validate(&self) -> BenchmarkResult<()> {
        for (name, value) in [
            ("workload.id", self.id.as_str()),
            ("workload.name", self.name.as_str()),
            ("workload.version", self.version.as_str()),
        ] {
            require_text(name, value)?;
        }
        require_sha256("workload.manifest_sha256", &self.manifest_sha256)
            .and_then(|_| {
                require_optional_sqlite_integer("workload.input_tokens", self.input_tokens)
            })
            .and_then(|_| {
                require_optional_sqlite_integer("workload.output_tokens", self.output_tokens)
            })
            .and_then(|_| {
                require_optional_sqlite_integer("workload.corpus_bytes", self.corpus_bytes)
            })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingConfig {
    #[serde(default)]
    pub seed: i64,
    #[serde(default)]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    #[serde(default)]
    pub top_k: u64,
    #[serde(default)]
    pub min_p: f64,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
        }
    }
}

impl SamplingConfig {
    fn validate(&self) -> BenchmarkResult<()> {
        require_finite_non_negative("sampling.temperature", Some(self.temperature))?;
        require_ratio("sampling.top_p", Some(self.top_p))?;
        require_ratio("sampling.min_p", Some(self.min_p))?;
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err(BenchmarkError::Validation(
                "sampling.repetition_penalty must be finite and positive".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizationConfig {
    #[serde(default = "default_feature_disabled")]
    pub flash_attention: FeatureState,
    #[serde(default)]
    pub kv_cache_type_k: Option<String>,
    #[serde(default)]
    pub kv_cache_type_v: Option<String>,
    #[serde(default)]
    pub gpu_layers: Option<i64>,
    #[serde(default)]
    pub tensor_split: Vec<f64>,
    #[serde(default = "default_true")]
    pub mmap: bool,
    #[serde(default)]
    pub mlock: bool,
    #[serde(default = "default_speculator_type")]
    pub speculator_type: SpeculatorType,
    #[serde(default)]
    pub draft_model_artifact_id: Option<String>,
    #[serde(default)]
    pub draft_tokens: Option<u64>,
    #[serde(default)]
    pub draft_threshold: Option<f64>,
    #[serde(default = "default_feature_disabled")]
    pub mtp: FeatureState,
    #[serde(default = "default_feature_disabled")]
    pub ngram: FeatureState,
    #[serde(default)]
    pub ngram_storage_location: Option<NgramStorageLocation>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            flash_attention: FeatureState::Disabled,
            kv_cache_type_k: None,
            kv_cache_type_v: None,
            gpu_layers: None,
            tensor_split: Vec::new(),
            mmap: true,
            mlock: false,
            speculator_type: SpeculatorType::None,
            draft_model_artifact_id: None,
            draft_tokens: None,
            draft_threshold: None,
            mtp: FeatureState::Disabled,
            ngram: FeatureState::Disabled,
            ngram_storage_location: None,
        }
    }
}

impl OptimizationConfig {
    fn validate(&self) -> BenchmarkResult<()> {
        if self
            .tensor_split
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(BenchmarkError::Validation(
                "optimization.tensor_split values must be finite and positive".into(),
            ));
        }
        require_ratio("optimization.draft_threshold", self.draft_threshold)?;
        if self.draft_tokens == Some(0) {
            return Err(BenchmarkError::Validation(
                "optimization.draft_tokens must be positive".into(),
            ));
        }
        if self.speculator_type == SpeculatorType::None
            && (self.mtp == FeatureState::Enabled || self.ngram == FeatureState::Enabled)
        {
            return Err(BenchmarkError::Validation(
                "optimization.speculator_type cannot be none when MTP or n-gram is enabled".into(),
            ));
        }
        if self.speculator_type == SpeculatorType::DraftModel
            && self
                .draft_model_artifact_id
                .as_deref()
                .is_none_or(|id| id.trim().is_empty())
        {
            return Err(BenchmarkError::Validation(
                "optimization.draft_model_artifact_id is required for draft_model speculation"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentConfig {
    #[serde(default = "new_id")]
    pub id: String,
    pub artifact_id: String,
    pub runtime_id: String,
    pub hardware_id: String,
    pub workload_id: String,
    pub context_tokens: u64,
    pub prompt_tokens: u64,
    pub generation_tokens: u64,
    #[serde(default = "default_one_u64")]
    pub batch_size: u64,
    #[serde(default = "default_one_u64")]
    pub micro_batch_size: u64,
    #[serde(default)]
    pub threads: Option<u64>,
    #[serde(default)]
    pub optimization: OptimizationConfig,
    #[serde(default)]
    pub sampling: SamplingConfig,
    pub command_template: String,
}

#[derive(Serialize)]
struct CanonicalExperiment<'a> {
    artifact_id: &'a str,
    runtime_id: &'a str,
    hardware_id: &'a str,
    workload_id: &'a str,
    context_tokens: u64,
    prompt_tokens: u64,
    generation_tokens: u64,
    batch_size: u64,
    micro_batch_size: u64,
    threads: Option<u64>,
    optimization: &'a OptimizationConfig,
    sampling: &'a SamplingConfig,
    command_template: &'a str,
}

impl ExperimentConfig {
    fn validate(&self) -> BenchmarkResult<()> {
        for (name, value) in [
            ("experiment.id", self.id.as_str()),
            ("experiment.artifact_id", self.artifact_id.as_str()),
            ("experiment.runtime_id", self.runtime_id.as_str()),
            ("experiment.hardware_id", self.hardware_id.as_str()),
            ("experiment.workload_id", self.workload_id.as_str()),
            (
                "experiment.command_template",
                self.command_template.as_str(),
            ),
        ] {
            require_text(name, value)?;
        }
        if self.context_tokens == 0
            || self.batch_size == 0
            || self.micro_batch_size == 0
            || self.threads == Some(0)
        {
            return Err(BenchmarkError::Validation(
                "experiment context, batch, micro-batch, and thread counts must be positive".into(),
            ));
        }
        if self.prompt_tokens > self.context_tokens {
            return Err(BenchmarkError::Validation(
                "experiment.prompt_tokens cannot exceed context_tokens".into(),
            ));
        }
        if self.batch_size != 1 && self.micro_batch_size > self.batch_size {
            return Err(BenchmarkError::Validation(
                "experiment.micro_batch_size cannot exceed batch_size".into(),
            ));
        }
        for (name, value) in [
            ("experiment.context_tokens", self.context_tokens),
            ("experiment.prompt_tokens", self.prompt_tokens),
            ("experiment.generation_tokens", self.generation_tokens),
            ("experiment.batch_size", self.batch_size),
            ("experiment.micro_batch_size", self.micro_batch_size),
        ] {
            require_sqlite_integer(name, value)?;
        }
        require_optional_sqlite_integer("experiment.threads", self.threads)?;
        self.optimization.validate()?;
        self.sampling.validate()
    }
}

fn default_repetitions() -> u64 {
    5
}

fn default_optimizations() -> Vec<OptimizationConfig> {
    vec![OptimizationConfig::default()]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentMatrix {
    pub name: String,
    pub artifacts: Vec<String>,
    pub runtimes: Vec<String>,
    pub hardware: Vec<String>,
    pub workloads: Vec<String>,
    pub contexts: Vec<u64>,
    pub prompt_tokens: Vec<u64>,
    pub generation_tokens: Vec<u64>,
    #[serde(default = "default_optimizations")]
    pub optimizations: Vec<OptimizationConfig>,
    #[serde(default = "default_repetitions")]
    pub repetitions: u64,
    #[serde(default = "default_true")]
    pub randomize_order: bool,
    #[serde(default = "default_true")]
    pub capture_telemetry: bool,
    #[serde(default = "default_one_u64")]
    pub batch_size: u64,
    #[serde(default = "default_one_u64")]
    pub micro_batch_size: u64,
    #[serde(default)]
    pub threads: Option<u64>,
    pub command_template: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlanExclusion {
    pub candidate: Value,
    pub reason_code: &'static str,
    pub reason: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExperimentPlan {
    pub name: String,
    pub experiments: Vec<ExperimentConfig>,
    pub exclusions: Vec<PlanExclusion>,
    pub repetitions: u64,
    pub run_count: u64,
    pub randomize_order: bool,
    pub capture_telemetry: bool,
}

impl ExperimentMatrix {
    pub fn expand(&self) -> BenchmarkResult<ExperimentPlan> {
        require_text("matrix.name", &self.name)?;
        require_text("matrix.command_template", &self.command_template)?;
        if self.repetitions == 0
            || self.batch_size == 0
            || self.micro_batch_size == 0
            || self.threads == Some(0)
        {
            return Err(BenchmarkError::Validation(
                "matrix repetitions, batch sizes, and threads must be positive".into(),
            ));
        }
        for (name, values) in [
            ("artifacts", &self.artifacts),
            ("runtimes", &self.runtimes),
            ("hardware", &self.hardware),
            ("workloads", &self.workloads),
        ] {
            if values.is_empty() || values.iter().any(|value| value.trim().is_empty()) {
                return Err(BenchmarkError::Validation(format!(
                    "matrix.{name} must contain at least one non-empty ID"
                )));
            }
        }
        if self.contexts.is_empty()
            || self.prompt_tokens.is_empty()
            || self.generation_tokens.is_empty()
            || self.optimizations.is_empty()
            || self.contexts.contains(&0)
        {
            return Err(BenchmarkError::Validation(
                        "matrix contexts, token counts, and optimizations must not be empty; contexts must be positive".into(),
                    ));
        }
        for optimization in &self.optimizations {
            optimization.validate()?;
        }
        for (name, unique, total) in [
            (
                "artifacts",
                self.artifacts.iter().collect::<BTreeSet<_>>().len(),
                self.artifacts.len(),
            ),
            (
                "runtimes",
                self.runtimes.iter().collect::<BTreeSet<_>>().len(),
                self.runtimes.len(),
            ),
            (
                "hardware",
                self.hardware.iter().collect::<BTreeSet<_>>().len(),
                self.hardware.len(),
            ),
            (
                "workloads",
                self.workloads.iter().collect::<BTreeSet<_>>().len(),
                self.workloads.len(),
            ),
            (
                "contexts",
                self.contexts.iter().collect::<BTreeSet<_>>().len(),
                self.contexts.len(),
            ),
            (
                "prompt_tokens",
                self.prompt_tokens.iter().collect::<BTreeSet<_>>().len(),
                self.prompt_tokens.len(),
            ),
            (
                "generation_tokens",
                self.generation_tokens.iter().collect::<BTreeSet<_>>().len(),
                self.generation_tokens.len(),
            ),
        ] {
            if unique != total {
                return Err(BenchmarkError::Validation(format!(
                    "matrix.{name} must not contain duplicate values"
                )));
            }
        }
        let optimization_keys = self
            .optimizations
            .iter()
            .map(to_json)
            .collect::<BenchmarkResult<BTreeSet<_>>>()?;
        if optimization_keys.len() != self.optimizations.len() {
            return Err(BenchmarkError::Validation(
                "matrix.optimizations must not contain duplicate configurations".into(),
            ));
        }

        let dimensions = [
            self.artifacts.len(),
            self.runtimes.len(),
            self.hardware.len(),
            self.workloads.len(),
            self.contexts.len(),
            self.prompt_tokens.len(),
            self.generation_tokens.len(),
            self.optimizations.len(),
        ];
        let candidate_count = dimensions
            .iter()
            .try_fold(1usize, |count, size| count.checked_mul(*size));
        if candidate_count.is_none_or(|count| count > 10_000) {
            return Err(BenchmarkError::Validation(
                "matrix expands beyond the 10,000-experiment planning limit".into(),
            ));
        }

        let mut experiments = Vec::new();
        let mut exclusions = Vec::new();
        for artifact_id in &self.artifacts {
            for runtime_id in &self.runtimes {
                for hardware_id in &self.hardware {
                    for workload_id in &self.workloads {
                        for &context_tokens in &self.contexts {
                            for &prompt_tokens in &self.prompt_tokens {
                                for &generation_tokens in &self.generation_tokens {
                                    for optimization in &self.optimizations {
                                        if prompt_tokens > context_tokens {
                                            exclusions.push(PlanExclusion {
                                                candidate: json!({
                                                    "artifact_id": artifact_id,
                                                    "runtime_id": runtime_id,
                                                    "hardware_id": hardware_id,
                                                    "workload_id": workload_id,
                                                    "context_tokens": context_tokens,
                                                    "prompt_tokens": prompt_tokens,
                                                    "generation_tokens": generation_tokens,
                                                }),
                                                reason_code: "prompt_exceeds_context",
                                                reason:
                                                    "prompt_tokens cannot exceed context_tokens",
                                            });
                                            continue;
                                        }
                                        let mut experiment = ExperimentConfig {
                                            id: "pending".into(),
                                            artifact_id: artifact_id.clone(),
                                            runtime_id: runtime_id.clone(),
                                            hardware_id: hardware_id.clone(),
                                            workload_id: workload_id.clone(),
                                            context_tokens,
                                            prompt_tokens,
                                            generation_tokens,
                                            batch_size: self.batch_size,
                                            micro_batch_size: self.micro_batch_size,
                                            threads: self.threads,
                                            optimization: optimization.clone(),
                                            sampling: SamplingConfig::default(),
                                            command_template: self.command_template.clone(),
                                        };
                                        let hash = experiment.experiment_hash()?;
                                        experiment.id = format!("experiment-{hash}");
                                        experiments.push(experiment);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        experiments.sort_by(|left, right| left.id.cmp(&right.id));
        let run_count = (experiments.len() as u64)
            .checked_mul(self.repetitions)
            .ok_or_else(|| BenchmarkError::Validation("planned run count overflowed".into()))?;
        Ok(ExperimentPlan {
            name: self.name.clone(),
            experiments,
            exclusions,
            repetitions: self.repetitions,
            run_count,
            randomize_order: self.randomize_order,
            capture_telemetry: self.capture_telemetry,
        })
    }
}

impl ExperimentConfig {
    fn canonical(&self) -> CanonicalExperiment<'_> {
        CanonicalExperiment {
            artifact_id: &self.artifact_id,
            runtime_id: &self.runtime_id,
            hardware_id: &self.hardware_id,
            workload_id: &self.workload_id,
            context_tokens: self.context_tokens,
            prompt_tokens: self.prompt_tokens,
            generation_tokens: self.generation_tokens,
            batch_size: self.batch_size,
            micro_batch_size: self.micro_batch_size,
            threads: self.threads,
            optimization: &self.optimization,
            sampling: &self.sampling,
            command_template: &self.command_template,
        }
    }

    pub fn canonical_json(&self) -> BenchmarkResult<String> {
        serde_json::to_string(&self.canonical()).map_err(|error| {
            BenchmarkError::Validation(format!("experiment cannot be canonicalized: {error}"))
        })
    }

    pub fn experiment_hash(&self) -> BenchmarkResult<String> {
        self.validate()?;
        let mut hasher = Sha256::new();
        hasher.update(self.canonical_json()?.as_bytes());
        Ok(format!("{:x}", hasher.finalize()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunRecord {
    #[serde(default = "new_id")]
    pub id: String,
    pub experiment_id: String,
    pub repetition: u64,
    #[serde(default = "default_run_status")]
    pub status: RunStatus,
    #[serde(default)]
    pub started_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub ended_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub exit_code: Option<i64>,
    #[serde(default)]
    pub random_seed: Option<i64>,
    #[serde(default)]
    pub warmup_count: u64,
    pub exact_command: String,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    #[serde(default)]
    pub stdout_path: Option<String>,
    #[serde(default)]
    pub stderr_path: Option<String>,
    #[serde(default)]
    pub failure_reason: Option<String>,
    #[serde(default)]
    pub raw_result: Option<BTreeMap<String, Value>>,
}

fn default_run_status() -> RunStatus {
    RunStatus::Planned
}

impl RunRecord {
    fn validate(&self) -> BenchmarkResult<()> {
        for (name, value) in [
            ("run.id", self.id.as_str()),
            ("run.experiment_id", self.experiment_id.as_str()),
            ("run.exact_command", self.exact_command.as_str()),
        ] {
            require_text(name, value)?;
        }
        if self.ended_at.is_some() && self.started_at.is_none() {
            return Err(BenchmarkError::Validation(
                "run.ended_at requires started_at".into(),
            ));
        }
        if matches!((self.started_at, self.ended_at), (Some(start), Some(end)) if end < start) {
            return Err(BenchmarkError::Validation(
                "run.ended_at cannot be before started_at".into(),
            ));
        }
        require_sqlite_integer("run.repetition", self.repetition)?;
        require_sqlite_integer("run.warmup_count", self.warmup_count)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceMetrics {
    pub run_id: String,
    #[serde(default)]
    pub model_load_ms: Option<f64>,
    #[serde(default)]
    pub prompt_processing_ms: Option<f64>,
    #[serde(default)]
    pub prompt_tps: Option<f64>,
    #[serde(default)]
    pub ttft_ms: Option<f64>,
    #[serde(default)]
    pub generation_ms: Option<f64>,
    #[serde(default)]
    pub generation_tps: Option<f64>,
    #[serde(default)]
    pub inter_token_p50_ms: Option<f64>,
    #[serde(default)]
    pub inter_token_p95_ms: Option<f64>,
    #[serde(default)]
    pub inter_token_p99_ms: Option<f64>,
    #[serde(default)]
    pub peak_rss_bytes: Option<u64>,
    #[serde(default)]
    pub peak_vram_bytes: Option<u64>,
    #[serde(default)]
    pub kv_cache_bytes: Option<u64>,
    #[serde(default)]
    pub energy_joules: Option<f64>,
    #[serde(default)]
    pub avg_power_watts: Option<f64>,
}

impl PerformanceMetrics {
    fn validate(&self) -> BenchmarkResult<()> {
        require_text("performance_metrics.run_id", &self.run_id)?;
        for (name, value) in [
            ("model_load_ms", self.model_load_ms),
            ("prompt_processing_ms", self.prompt_processing_ms),
            ("prompt_tps", self.prompt_tps),
            ("ttft_ms", self.ttft_ms),
            ("generation_ms", self.generation_ms),
            ("generation_tps", self.generation_tps),
            ("inter_token_p50_ms", self.inter_token_p50_ms),
            ("inter_token_p95_ms", self.inter_token_p95_ms),
            ("inter_token_p99_ms", self.inter_token_p99_ms),
            ("energy_joules", self.energy_joules),
            ("avg_power_watts", self.avg_power_watts),
        ] {
            require_finite_non_negative(&format!("performance_metrics.{name}"), value)?;
        }
        if matches!(
            (
                self.inter_token_p50_ms,
                self.inter_token_p95_ms,
                self.inter_token_p99_ms
            ),
            (Some(p50), Some(p95), Some(p99)) if !(p50 <= p95 && p95 <= p99)
        ) {
            return Err(BenchmarkError::Validation(
                "latency percentiles must satisfy p50 <= p95 <= p99".into(),
            ));
        }
        for (name, value) in [
            ("peak_rss_bytes", self.peak_rss_bytes),
            ("peak_vram_bytes", self.peak_vram_bytes),
            ("kv_cache_bytes", self.kv_cache_bytes),
        ] {
            require_optional_sqlite_integer(&format!("performance_metrics.{name}"), value)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeMetrics {
    pub run_id: String,
    pub speculator_type: SpeculatorType,
    #[serde(default)]
    pub proposals: Option<u64>,
    #[serde(default)]
    pub proposed_tokens: Option<u64>,
    #[serde(default)]
    pub accepted_tokens: Option<u64>,
    #[serde(default)]
    pub target_evaluations: Option<u64>,
    #[serde(default)]
    pub acceptance_rate: Option<f64>,
    #[serde(default)]
    pub speculator_memory_bytes: Option<u64>,
    #[serde(default)]
    pub overhead_ms: Option<f64>,
}

impl SpeculativeMetrics {
    fn validate(&self) -> BenchmarkResult<()> {
        require_text("speculative_metrics.run_id", &self.run_id)?;
        require_ratio("speculative_metrics.acceptance_rate", self.acceptance_rate)?;
        require_finite_non_negative("speculative_metrics.overhead_ms", self.overhead_ms)?;
        if matches!(
            (self.accepted_tokens, self.proposed_tokens),
            (Some(accepted), Some(proposed)) if accepted > proposed
        ) {
            return Err(BenchmarkError::Validation(
                "speculative_metrics.accepted_tokens cannot exceed proposed_tokens".into(),
            ));
        }
        for (name, value) in [
            ("proposals", self.proposals),
            ("proposed_tokens", self.proposed_tokens),
            ("accepted_tokens", self.accepted_tokens),
            ("target_evaluations", self.target_evaluations),
            ("speculator_memory_bytes", self.speculator_memory_bytes),
        ] {
            require_optional_sqlite_integer(&format!("speculative_metrics.{name}"), value)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualityResult {
    #[serde(default = "new_id")]
    pub id: String,
    pub run_id: String,
    pub task_id: String,
    pub metric_name: String,
    #[serde(default)]
    pub metric_value: Option<f64>,
    #[serde(default)]
    pub passed: Option<bool>,
    #[serde(default)]
    pub compile_succeeded: Option<bool>,
    #[serde(default)]
    pub tests_passed: Option<u64>,
    #[serde(default)]
    pub tests_total: Option<u64>,
    #[serde(default)]
    pub generated_tokens: Option<u64>,
    #[serde(default)]
    pub duration_ms: Option<f64>,
    #[serde(default)]
    pub output_path: Option<String>,
    #[serde(default)]
    pub log_path: Option<String>,
    #[serde(default)]
    pub details: BTreeMap<String, Value>,
}

impl QualityResult {
    fn validate(&self) -> BenchmarkResult<()> {
        for (name, value) in [
            ("quality_results[].id", self.id.as_str()),
            ("quality_results[].run_id", self.run_id.as_str()),
            ("quality_results[].task_id", self.task_id.as_str()),
            ("quality_results[].metric_name", self.metric_name.as_str()),
        ] {
            require_text(name, value)?;
        }
        if self.metric_value.is_some_and(|value| !value.is_finite()) {
            return Err(BenchmarkError::Validation(
                "quality_results[].metric_value must be finite".into(),
            ));
        }
        if self.metric_name == "pass@1" {
            require_ratio("quality_results[].metric_value", self.metric_value)?;
        }
        require_finite_non_negative("quality_results[].duration_ms", self.duration_ms)?;
        if matches!(
            (self.tests_passed, self.tests_total),
            (Some(passed), Some(total)) if passed > total
        ) {
            return Err(BenchmarkError::Validation(
                "quality_results[].tests_passed cannot exceed tests_total".into(),
            ));
        }
        for (name, value) in [
            ("tests_passed", self.tests_passed),
            ("tests_total", self.tests_total),
            ("generated_tokens", self.generated_tokens),
        ] {
            require_optional_sqlite_integer(&format!("quality_results[].{name}"), value)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetrySample {
    pub run_id: String,
    pub sampled_at: DateTime<Utc>,
    #[serde(default)]
    pub cpu_percent: Option<f64>,
    #[serde(default)]
    pub rss_bytes: Option<u64>,
    #[serde(default)]
    pub gpu_index: Option<u64>,
    #[serde(default)]
    pub gpu_util_percent: Option<f64>,
    #[serde(default)]
    pub vram_used_bytes: Option<u64>,
    #[serde(default)]
    pub temperature_c: Option<f64>,
    #[serde(default)]
    pub power_watts: Option<f64>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

impl TelemetrySample {
    fn validate(&self) -> BenchmarkResult<()> {
        require_text("telemetry_samples[].run_id", &self.run_id)?;
        require_percentage("telemetry_samples[].cpu_percent", self.cpu_percent)?;
        require_percentage(
            "telemetry_samples[].gpu_util_percent",
            self.gpu_util_percent,
        )?;
        require_finite_non_negative("telemetry_samples[].power_watts", self.power_watts)?;
        if self.temperature_c.is_some_and(|value| !value.is_finite()) {
            return Err(BenchmarkError::Validation(
                "telemetry_samples[].temperature_c must be finite".into(),
            ));
        }
        require_optional_sqlite_integer("telemetry_samples[].rss_bytes", self.rss_bytes)?;
        require_optional_sqlite_integer("telemetry_samples[].gpu_index", self.gpu_index)?;
        require_optional_sqlite_integer(
            "telemetry_samples[].vram_used_bytes",
            self.vram_used_bytes,
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IngestBundle {
    pub model: ModelDefinition,
    pub artifact: ArtifactDefinition,
    pub runtime: RuntimeDefinition,
    pub hardware: HardwareProfile,
    pub workload: WorkloadDefinition,
    pub experiment: ExperimentConfig,
    pub run: RunRecord,
    #[serde(default)]
    pub performance_metrics: Option<PerformanceMetrics>,
    #[serde(default)]
    pub speculative_metrics: Option<SpeculativeMetrics>,
    #[serde(default)]
    pub quality_results: Vec<QualityResult>,
    #[serde(default)]
    pub telemetry_samples: Vec<TelemetrySample>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlamaBenchIngest {
    bundle: IngestBundle,
    llama_bench: Value,
}

fn apply_llama_bench_metrics(bundle: &mut IngestBundle, output: &Value) -> BenchmarkResult<()> {
    let rows = output
        .as_array()
        .cloned()
        .or_else(|| output.get("results").and_then(Value::as_array).cloned())
        .unwrap_or_else(|| vec![output.clone()]);
    let mut prompt_tps = None;
    let mut generation_tps = None;
    let mut ttft_ms = None;
    for row in rows {
        let Some(object) = row.as_object() else {
            return Err(BenchmarkError::Validation(
                "llama_bench must be an object, an array of objects, or contain a results array"
                    .into(),
            ));
        };
        let value = ["avg_ts", "tokens_per_second", "tps"]
            .iter()
            .find_map(|key| object.get(*key).and_then(Value::as_f64));
        let test = object
            .get("test")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_ascii_lowercase();
        let prompt_tokens = object
            .get("n_prompt")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let generation_tokens = object
            .get("n_gen")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let explicit_prompt_tps = object.get("prompt_tps").and_then(Value::as_f64);
        let explicit_generation_tps = object.get("generation_tps").and_then(Value::as_f64);
        let is_combined = (test.contains("pp") && test.contains("tg"))
            || (prompt_tokens > 0 && generation_tokens > 0);
        if is_combined && (explicit_prompt_tps.is_none() || explicit_generation_tps.is_none()) {
            return Err(BenchmarkError::Validation(
                "combined llama_bench rows require distinct prompt_tps and generation_tps fields"
                    .into(),
            ));
        }
        if let Some(value) = explicit_prompt_tps {
            set_llama_bench_metric("prompt TPS", &mut prompt_tps, value)?;
        } else if (test.starts_with("pp") || prompt_tokens > 0) && generation_tokens == 0 {
            if let Some(value) = value {
                set_llama_bench_metric("prompt TPS", &mut prompt_tps, value)?;
            }
        }
        if let Some(value) = explicit_generation_tps {
            set_llama_bench_metric("generation TPS", &mut generation_tps, value)?;
        } else if (test.starts_with("tg") || generation_tokens > 0) && prompt_tokens == 0 {
            if let Some(value) = value {
                set_llama_bench_metric("generation TPS", &mut generation_tps, value)?;
            }
        }
        if let Some(value) = object.get("ttft_ms").and_then(Value::as_f64) {
            set_llama_bench_metric("TTFT", &mut ttft_ms, value)?;
        }
    }
    if prompt_tps.is_none() && generation_tps.is_none() && ttft_ms.is_none() {
        return Err(BenchmarkError::Validation(
            "llama_bench output contains no recognized throughput or TTFT metrics".into(),
        ));
    }
    let metrics = bundle
        .performance_metrics
        .get_or_insert_with(|| PerformanceMetrics {
            run_id: bundle.run.id.clone(),
            model_load_ms: None,
            prompt_processing_ms: None,
            prompt_tps: None,
            ttft_ms: None,
            generation_ms: None,
            generation_tps: None,
            inter_token_p50_ms: None,
            inter_token_p95_ms: None,
            inter_token_p99_ms: None,
            peak_rss_bytes: None,
            peak_vram_bytes: None,
            kv_cache_bytes: None,
            energy_joules: None,
            avg_power_watts: None,
        });
    if prompt_tps.is_some() {
        metrics.prompt_tps = prompt_tps;
    }
    if generation_tps.is_some() {
        metrics.generation_tps = generation_tps;
    }
    if ttft_ms.is_some() {
        metrics.ttft_ms = ttft_ms;
    }
    metrics.validate()
}

fn set_llama_bench_metric(name: &str, target: &mut Option<f64>, value: f64) -> BenchmarkResult<()> {
    if target.is_some() {
        return Err(BenchmarkError::Validation(format!(
            "llama_bench output has multiple {name} rows; supply one configuration per run"
        )));
    }
    *target = Some(value);
    Ok(())
}

impl IngestBundle {
    fn validate(&self) -> BenchmarkResult<()> {
        self.model.validate()?;
        self.artifact.validate()?;
        self.runtime.validate()?;
        self.hardware.validate()?;
        self.workload.validate()?;
        self.experiment.validate()?;
        self.run.validate()?;
        if self.artifact.model_id != self.model.id {
            return Err(BenchmarkError::Validation(
                "artifact.model_id must match model.id".into(),
            ));
        }
        for (name, actual, expected) in [
            (
                "experiment.artifact_id",
                self.experiment.artifact_id.as_str(),
                self.artifact.id.as_str(),
            ),
            (
                "experiment.runtime_id",
                self.experiment.runtime_id.as_str(),
                self.runtime.id.as_str(),
            ),
            (
                "experiment.hardware_id",
                self.experiment.hardware_id.as_str(),
                self.hardware.id.as_str(),
            ),
            (
                "experiment.workload_id",
                self.experiment.workload_id.as_str(),
                self.workload.id.as_str(),
            ),
            (
                "run.experiment_id",
                self.run.experiment_id.as_str(),
                self.experiment.id.as_str(),
            ),
        ] {
            if actual != expected {
                return Err(BenchmarkError::Validation(format!(
                    "{name} must reference the matching object in this bundle"
                )));
            }
        }
        if let Some(metrics) = &self.performance_metrics {
            metrics.validate()?;
            if metrics.run_id != self.run.id {
                return Err(BenchmarkError::Validation(
                    "performance_metrics.run_id must match run.id".into(),
                ));
            }
        }
        if let Some(metrics) = &self.speculative_metrics {
            metrics.validate()?;
            if metrics.run_id != self.run.id {
                return Err(BenchmarkError::Validation(
                    "speculative_metrics.run_id must match run.id".into(),
                ));
            }
        }
        for result in &self.quality_results {
            result.validate()?;
            if result.run_id != self.run.id {
                return Err(BenchmarkError::Validation(
                    "quality_results[].run_id must match run.id".into(),
                ));
            }
        }
        for sample in &self.telemetry_samples {
            sample.validate()?;
            if sample.run_id != self.run.id {
                return Err(BenchmarkError::Validation(
                    "telemetry_samples[].run_id must match run.id".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub run_id: String,
    pub status: String,
    pub repetition: u64,
    pub experiment_hash: String,
    pub family: String,
    pub architecture: String,
    pub quant_name: String,
    pub fork_name: String,
    pub backend: String,
    pub context_tokens: u64,
    pub workload: String,
    pub prompt_tps: Option<f64>,
    pub generation_tps: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
    pub peak_vram_bytes: Option<u64>,
    pub speculator_type: Option<String>,
    pub acceptance_rate: Option<f64>,
    pub started_at: Option<String>,
    pub ended_at: Option<String>,
    pub failure_reason: Option<String>,
    pub disk_bytes: u64,
    pub quality_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunPage {
    pub items: Vec<RunSummary>,
    pub page: u32,
    pub per_page: u32,
    pub total: u64,
    pub total_pages: u64,
}

#[derive(Debug, Clone, Default)]
pub struct RunQuery {
    pub page: u32,
    pub per_page: u32,
    pub status: Option<String>,
    pub family: Option<String>,
    pub backend: Option<String>,
    pub workload: Option<String>,
    pub quant_name: Option<String>,
    pub speculator_type: Option<String>,
    pub q: Option<String>,
    pub sort: String,
    pub order: String,
}

impl RunQuery {
    pub fn parse(query: Option<&str>) -> BenchmarkResult<Self> {
        let mut result = Self {
            page: 1,
            per_page: 25,
            sort: "started_at".into(),
            order: "desc".into(),
            ..Self::default()
        };
        for (key, value) in url::form_urlencoded::parse(query.unwrap_or_default().as_bytes()) {
            let value = value.into_owned();
            match key.as_ref() {
                "page" => {
                    result.page = value.parse().map_err(|_| {
                        BenchmarkError::Validation("page must be a positive integer".into())
                    })?;
                }
                "per_page" => {
                    result.per_page = value.parse().map_err(|_| {
                        BenchmarkError::Validation("per_page must be a positive integer".into())
                    })?;
                }
                "status" => result.status = non_empty_filter(value),
                "family" => result.family = non_empty_filter(value),
                "backend" => result.backend = non_empty_filter(value),
                "workload" => result.workload = non_empty_filter(value),
                "quant_name" => result.quant_name = non_empty_filter(value),
                "speculator_type" => result.speculator_type = non_empty_filter(value),
                "q" => result.q = non_empty_filter(value),
                "sort" => result.sort = value,
                "order" => result.order = value,
                unknown => {
                    return Err(BenchmarkError::Validation(format!(
                        "unknown query parameter: {unknown}"
                    )))
                }
            }
        }
        if result.page == 0 {
            return Err(BenchmarkError::Validation(
                "page must be greater than zero".into(),
            ));
        }
        if result.per_page == 0 || result.per_page > MAX_PAGE_SIZE {
            return Err(BenchmarkError::Validation(format!(
                "per_page must be between 1 and {MAX_PAGE_SIZE}"
            )));
        }
        if !matches!(
            result.sort.as_str(),
            "started_at"
                | "family"
                | "workload"
                | "context_tokens"
                | "prompt_tps"
                | "generation_tps"
                | "ttft_ms"
        ) {
            return Err(BenchmarkError::Validation(
                "sort must be one of started_at, family, workload, context_tokens, prompt_tps, generation_tps, ttft_ms".into(),
            ));
        }
        if !matches!(result.order.as_str(), "asc" | "desc") {
            return Err(BenchmarkError::Validation(
                "order must be asc or desc".into(),
            ));
        }
        Ok(result)
    }
}

fn non_empty_filter(value: String) -> Option<String> {
    (!value.trim().is_empty()).then_some(value)
}

#[derive(Debug, Clone)]
pub struct BenchmarkStore {
    path: PathBuf,
}

impl BenchmarkStore {
    pub fn open(path: impl Into<PathBuf>) -> BenchmarkResult<Self> {
        let store = Self { path: path.into() };
        if let Some(parent) = store
            .path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        let connection = store.connect()?;
        let migrated = connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations')",
                [],
                |row| row.get::<_, bool>(0),
            )?;
        if !migrated {
            connection.execute_batch(MIGRATION_0001)?;
        } else {
            let version = connection
                .query_row("SELECT MAX(version) FROM schema_migrations", [], |row| {
                    row.get::<_, Option<i64>>(0)
                })?
                .unwrap_or_default();
            let expected_migration = connection.query_row(
                "SELECT EXISTS(
                   SELECT 1 FROM schema_migrations
                    WHERE version=1 AND name='benchmark_explorer_initial'
                 )",
                [],
                |row| row.get::<_, bool>(0),
            )?;
            let required_tables = connection.query_row(
                "SELECT COUNT(*) FROM sqlite_master
                  WHERE type='table'
                    AND name IN ('models','artifacts','experiments','runs','entity_fingerprints')",
                [],
                |row| row.get::<_, u64>(0),
            )?;
            if version != 1 || !expected_migration || required_tables != 5 {
                return Err(BenchmarkError::Validation(format!(
                    "database is not a complete benchmark explorer schema at version 1 (found version {version})"
                )));
            }
        }
        let foreign_keys: bool =
            connection.query_row("PRAGMA foreign_keys", [], |row| row.get(0))?;
        if !foreign_keys {
            return Err(BenchmarkError::Validation(
                "SQLite foreign key enforcement could not be enabled".into(),
            ));
        }
        Ok(store)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn connect(&self) -> BenchmarkResult<Connection> {
        let connection = Connection::open(&self.path)?;
        connection.busy_timeout(std::time::Duration::from_secs(5))?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        Ok(connection)
    }

    pub fn ingest(&self, bundle: &IngestBundle) -> BenchmarkResult<String> {
        bundle.validate()?;
        let experiment_hash = bundle.experiment.experiment_hash()?;
        let canonical_spec = bundle.experiment.canonical_json()?;
        let spec_id = format!("spec-{experiment_hash}");

        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let existing_run = transaction
            .query_row(
                "SELECT status, experiment_id, repetition FROM runs WHERE id=?1",
                params![bundle.run.id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, u64>(2)?,
                    ))
                },
            )
            .optional()?;
        let repetition_run_id = transaction
            .query_row(
                "SELECT id FROM runs WHERE experiment_id=?1 AND repetition=?2",
                params![bundle.run.experiment_id, bundle.run.repetition],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        if repetition_run_id
            .as_deref()
            .is_some_and(|run_id| run_id != bundle.run.id)
        {
            return Err(BenchmarkError::Conflict(format!(
                "experiment repetition already belongs to run {}",
                repetition_run_id.unwrap_or_default()
            )));
        }
        if let Some((status, experiment_id, repetition)) = &existing_run {
            if experiment_id != &bundle.run.experiment_id || repetition != &bundle.run.repetition {
                return Err(BenchmarkError::Conflict(
                    "run experiment_id and repetition are immutable".into(),
                ));
            }
            if status == "succeeded" {
                return Err(BenchmarkError::Conflict(
                    "successful run payloads are immutable; ingest a new repetition or experiment"
                        .into(),
                ));
            }
        }

        ensure_fingerprint(&transaction, "model", &bundle.model.id, &bundle.model)?;
        ensure_fingerprint(
            &transaction,
            "artifact",
            &bundle.artifact.id,
            &bundle.artifact,
        )?;
        ensure_fingerprint(&transaction, "runtime", &bundle.runtime.id, &bundle.runtime)?;
        ensure_fingerprint(
            &transaction,
            "hardware",
            &bundle.hardware.id,
            &bundle.hardware,
        )?;
        ensure_fingerprint(
            &transaction,
            "workload",
            &bundle.workload.id,
            &bundle.workload,
        )?;
        ensure_fingerprint(
            &transaction,
            "experiment",
            &bundle.experiment.id,
            &bundle.experiment.canonical(),
        )?;

        let model_metadata = to_json(&bundle.model.metadata)?;
        transaction.execute(
            "INSERT OR IGNORE INTO models(
                id,family,architecture,checkpoint,revision,tokenizer_id,tokenizer_revision,
                parameter_count_total,parameter_count_active,model_kind,native_context_tokens,metadata_json
             ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
            params![
                bundle.model.id,
                bundle.model.family,
                bundle.model.architecture,
                bundle.model.checkpoint,
                bundle.model.revision,
                bundle.model.tokenizer_id,
                bundle.model.tokenizer_revision,
                bundle.model.parameter_count_total,
                bundle.model.parameter_count_active,
                bundle.model.model_kind.as_str(),
                bundle.model.native_context_tokens,
                model_metadata,
            ],
        )?;

        let artifact_metadata = to_json(&bundle.artifact.metadata)?;
        transaction.execute(
            "INSERT OR IGNORE INTO artifacts(
                id,model_id,format,quant_family,quant_name,average_bits_per_weight,disk_bytes,
                sha256,source_uri,imatrix_used,conversion_tool,conversion_commit,
                conversion_command,metadata_json
             ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",
            params![
                bundle.artifact.id,
                bundle.artifact.model_id,
                bundle.artifact.format,
                bundle.artifact.quant_family,
                bundle.artifact.quant_name,
                bundle.artifact.average_bits_per_weight,
                bundle.artifact.disk_bytes,
                bundle.artifact.sha256,
                bundle.artifact.source_uri,
                bundle.artifact.imatrix_used,
                bundle.artifact.conversion_tool,
                bundle.artifact.conversion_commit,
                bundle.artifact.conversion_command,
                artifact_metadata,
            ],
        )?;

        let build_flags = to_json(&bundle.runtime.build_flags)?;
        let capabilities = to_json(&bundle.runtime.capabilities)?;
        transaction.execute(
            "INSERT OR IGNORE INTO runtimes(
                id,repository,fork_name,commit_sha,dirty_tree,compiler,compiler_version,backend,
                build_flags_json,capabilities_json,executable_sha256,container_digest,built_at
             ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
            params![
                bundle.runtime.id,
                bundle.runtime.repository,
                bundle.runtime.fork_name,
                bundle.runtime.commit_sha,
                bundle.runtime.dirty_tree,
                bundle.runtime.compiler,
                bundle.runtime.compiler_version,
                bundle.runtime.backend.as_str(),
                build_flags,
                capabilities,
                bundle.runtime.executable_sha256,
                bundle.runtime.container_digest,
                bundle.runtime.built_at.to_rfc3339(),
            ],
        )?;

        let gpus = to_json(&bundle.hardware.gpus)?;
        let driver_versions = to_json(&bundle.hardware.driver_versions)?;
        let hardware_metadata = to_json(&bundle.hardware.metadata)?;
        transaction.execute(
            "INSERT OR IGNORE INTO hardware_profiles(
                id,hostname_hash,cpu_model,physical_cores,logical_cores,system_ram_bytes,gpu_json,
                unified_memory,os_name,os_version,kernel,driver_versions_json,power_profile,
                metadata_json,captured_at
             ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)",
            params![
                bundle.hardware.id,
                bundle.hardware.hostname_hash,
                bundle.hardware.cpu_model,
                bundle.hardware.physical_cores,
                bundle.hardware.logical_cores,
                bundle.hardware.system_ram_bytes,
                gpus,
                bundle.hardware.unified_memory,
                bundle.hardware.os_name,
                bundle.hardware.os_version,
                bundle.hardware.kernel,
                driver_versions,
                bundle.hardware.power_profile,
                hardware_metadata,
                bundle.hardware.captured_at.to_rfc3339(),
            ],
        )?;

        let workload_metadata = to_json(&bundle.workload.metadata)?;
        transaction.execute(
            "INSERT OR IGNORE INTO workloads(
                id,name,version,workload_type,manifest_sha256,input_tokens,output_tokens,
                corpus_bytes,license,metadata_json
             ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            params![
                bundle.workload.id,
                bundle.workload.name,
                bundle.workload.version,
                bundle.workload.workload_type.as_str(),
                bundle.workload.manifest_sha256,
                bundle.workload.input_tokens,
                bundle.workload.output_tokens,
                bundle.workload.corpus_bytes,
                bundle.workload.license,
                workload_metadata,
            ],
        )?;

        if let Some(draft_artifact_id) = bundle
            .experiment
            .optimization
            .draft_model_artifact_id
            .as_deref()
        {
            let draft_exists = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM artifacts WHERE id=?1)",
                params![draft_artifact_id],
                |row| row.get::<_, bool>(0),
            )?;
            if !draft_exists {
                return Err(BenchmarkError::Validation(format!(
                    "draft model artifact {draft_artifact_id} does not exist"
                )));
            }
        }

        transaction.execute(
            "INSERT OR IGNORE INTO experiment_specs(id,name,spec_sha256,canonical_spec_json)
             VALUES (?1,?2,?3,?4)",
            params![
                spec_id,
                format!(
                    "{} / {} / {}",
                    bundle.model.family, bundle.workload.name, bundle.artifact.quant_name
                ),
                experiment_hash,
                canonical_spec,
            ],
        )?;

        let optimization_json = to_json(&bundle.experiment.optimization)?;
        let sampling_json = to_json(&bundle.experiment.sampling)?;
        transaction.execute(
            "INSERT OR IGNORE INTO experiments(
                id,experiment_hash,spec_id,artifact_id,runtime_id,hardware_id,workload_id,
                context_tokens,prompt_tokens,generation_tokens,batch_size,micro_batch_size,
                threads,gpu_layers,flash_attention,kv_cache_type_k,kv_cache_type_v,
                optimization_json,sampling_json,command_template
             ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20)",
            params![
                bundle.experiment.id,
                experiment_hash,
                spec_id,
                bundle.experiment.artifact_id,
                bundle.experiment.runtime_id,
                bundle.experiment.hardware_id,
                bundle.experiment.workload_id,
                bundle.experiment.context_tokens,
                bundle.experiment.prompt_tokens,
                bundle.experiment.generation_tokens,
                bundle.experiment.batch_size,
                bundle.experiment.micro_batch_size,
                bundle.experiment.threads,
                bundle.experiment.optimization.gpu_layers,
                bundle.experiment.optimization.flash_attention == FeatureState::Enabled,
                bundle.experiment.optimization.kv_cache_type_k,
                bundle.experiment.optimization.kv_cache_type_v,
                optimization_json,
                sampling_json,
                bundle.experiment.command_template,
            ],
        )?;

        let started_at = bundle.run.started_at.map(|value| value.to_rfc3339());
        let ended_at = bundle.run.ended_at.map(|value| value.to_rfc3339());
        let environment_json = to_json(&bundle.run.environment)?;
        let raw_result_json = bundle.run.raw_result.as_ref().map(to_json).transpose()?;
        if existing_run.is_some() {
            transaction.execute(
                "UPDATE runs SET experiment_id=?2,repetition=?3,status=?4,started_at=?5,
                    ended_at=?6,exit_code=?7,random_seed=?8,warmup_count=?9,exact_command=?10,
                    cwd=?11,environment_json=?12,stdout_path=?13,stderr_path=?14,
                    failure_reason=?15,raw_result_json=?16 WHERE id=?1",
                params![
                    bundle.run.id,
                    bundle.run.experiment_id,
                    bundle.run.repetition,
                    bundle.run.status.as_str(),
                    started_at,
                    ended_at,
                    bundle.run.exit_code,
                    bundle.run.random_seed,
                    bundle.run.warmup_count,
                    bundle.run.exact_command,
                    bundle.run.cwd,
                    environment_json,
                    bundle.run.stdout_path,
                    bundle.run.stderr_path,
                    bundle.run.failure_reason,
                    raw_result_json,
                ],
            )?;
        } else {
            transaction.execute(
                "INSERT INTO runs(
                    id,experiment_id,repetition,status,started_at,ended_at,exit_code,random_seed,
                    warmup_count,exact_command,cwd,environment_json,stdout_path,stderr_path,
                    failure_reason,raw_result_json
                 ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16)",
                params![
                    bundle.run.id,
                    bundle.run.experiment_id,
                    bundle.run.repetition,
                    bundle.run.status.as_str(),
                    started_at,
                    ended_at,
                    bundle.run.exit_code,
                    bundle.run.random_seed,
                    bundle.run.warmup_count,
                    bundle.run.exact_command,
                    bundle.run.cwd,
                    environment_json,
                    bundle.run.stdout_path,
                    bundle.run.stderr_path,
                    bundle.run.failure_reason,
                    raw_result_json,
                ],
            )?;
        }

        transaction.execute(
            "DELETE FROM performance_metrics WHERE run_id=?1",
            params![bundle.run.id],
        )?;
        if let Some(metrics) = &bundle.performance_metrics {
            transaction.execute(
                "INSERT INTO performance_metrics(
                    run_id,model_load_ms,prompt_processing_ms,prompt_tps,ttft_ms,generation_ms,
                    generation_tps,inter_token_p50_ms,inter_token_p95_ms,inter_token_p99_ms,
                    peak_rss_bytes,peak_vram_bytes,kv_cache_bytes,energy_joules,avg_power_watts
                 ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)",
                params![
                    metrics.run_id,
                    metrics.model_load_ms,
                    metrics.prompt_processing_ms,
                    metrics.prompt_tps,
                    metrics.ttft_ms,
                    metrics.generation_ms,
                    metrics.generation_tps,
                    metrics.inter_token_p50_ms,
                    metrics.inter_token_p95_ms,
                    metrics.inter_token_p99_ms,
                    metrics.peak_rss_bytes,
                    metrics.peak_vram_bytes,
                    metrics.kv_cache_bytes,
                    metrics.energy_joules,
                    metrics.avg_power_watts,
                ],
            )?;
        }

        transaction.execute(
            "DELETE FROM speculative_metrics WHERE run_id=?1",
            params![bundle.run.id],
        )?;
        if let Some(metrics) = &bundle.speculative_metrics {
            transaction.execute(
                "INSERT INTO speculative_metrics(
                    run_id,speculator_type,proposals,proposed_tokens,accepted_tokens,
                    target_evaluations,acceptance_rate,speculator_memory_bytes,overhead_ms
                 ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
                params![
                    metrics.run_id,
                    metrics.speculator_type.as_str(),
                    metrics.proposals,
                    metrics.proposed_tokens,
                    metrics.accepted_tokens,
                    metrics.target_evaluations,
                    metrics.acceptance_rate,
                    metrics.speculator_memory_bytes,
                    metrics.overhead_ms,
                ],
            )?;
        }

        transaction.execute(
            "DELETE FROM quality_results WHERE run_id=?1",
            params![bundle.run.id],
        )?;
        for result in &bundle.quality_results {
            let details = to_json(&result.details)?;
            transaction.execute(
                "INSERT INTO quality_results(
                    id,run_id,task_id,metric_name,metric_value,passed,compile_succeeded,
                    tests_passed,tests_total,generated_tokens,duration_ms,output_path,log_path,
                    details_json
                 ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",
                params![
                    result.id,
                    result.run_id,
                    result.task_id,
                    result.metric_name,
                    result.metric_value,
                    result.passed,
                    result.compile_succeeded,
                    result.tests_passed,
                    result.tests_total,
                    result.generated_tokens,
                    result.duration_ms,
                    result.output_path,
                    result.log_path,
                    details,
                ],
            )?;
        }

        transaction.execute(
            "DELETE FROM telemetry_samples WHERE run_id=?1",
            params![bundle.run.id],
        )?;
        for sample in &bundle.telemetry_samples {
            let metadata = to_json(&sample.metadata)?;
            transaction.execute(
                "INSERT INTO telemetry_samples(
                    run_id,sampled_at,cpu_percent,rss_bytes,gpu_index,gpu_util_percent,
                    vram_used_bytes,temperature_c,power_watts,metadata_json
                 ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
                params![
                    sample.run_id,
                    sample.sampled_at.to_rfc3339(),
                    sample.cpu_percent,
                    sample.rss_bytes,
                    sample.gpu_index,
                    sample.gpu_util_percent,
                    sample.vram_used_bytes,
                    sample.temperature_c,
                    sample.power_watts,
                    metadata,
                ],
            )?;
        }

        transaction.commit()?;
        Ok(bundle.run.id.clone())
    }

    pub fn query_runs(&self, query: &RunQuery) -> BenchmarkResult<RunPage> {
        let connection = self.connect()?;
        self.query_runs_connection(&connection, query)
    }

    fn query_runs_connection(
        &self,
        connection: &Connection,
        query: &RunQuery,
    ) -> BenchmarkResult<RunPage> {
        let (where_sql, values) = query_where(query);
        let total = connection.query_row(
            &format!(
                "SELECT COUNT(*) FROM run_summary rs
                 JOIN runs r ON r.id=rs.run_id {where_sql}"
            ),
            params_from_iter(values.iter()),
            |row| row.get::<_, u64>(0),
        )?;

        let sort_column = match query.sort.as_str() {
            "family" => "rs.family",
            "workload" => "rs.workload",
            "context_tokens" => "rs.context_tokens",
            "prompt_tps" => "rs.prompt_tps",
            "generation_tps" => "rs.generation_tps",
            "ttft_ms" => "rs.ttft_ms",
            _ => "COALESCE(r.started_at, r.created_at)",
        };
        let direction = if query.order == "asc" { "ASC" } else { "DESC" };
        let offset = u64::from(query.page - 1) * u64::from(query.per_page);
        let mut paged_values = values;
        paged_values.push(SqlValue::Integer(i64::from(query.per_page)));
        paged_values.push(SqlValue::Integer(offset as i64));
        let sql = format!(
            "SELECT rs.run_id,rs.status,rs.repetition,rs.experiment_hash,rs.family,
                    rs.architecture,rs.quant_name,rs.fork_name,rs.backend,rs.context_tokens,
                    rs.workload,rs.prompt_tps,rs.generation_tps,rs.ttft_ms,
                    rs.peak_rss_bytes,rs.peak_vram_bytes,rs.speculator_type,
                    rs.acceptance_rate,r.started_at,r.ended_at,r.failure_reason,
                    (SELECT a.disk_bytes FROM experiments e
                       JOIN artifacts a ON a.id=e.artifact_id
                      WHERE e.id=r.experiment_id),
                    (SELECT AVG(q.metric_value) FROM quality_results q
                      WHERE q.run_id=r.id AND q.metric_name='pass@1')
             FROM run_summary rs JOIN runs r ON r.id=rs.run_id
             {where_sql}
             ORDER BY {sort_column} {direction} NULLS LAST, rs.run_id ASC
             LIMIT ?{} OFFSET ?{}",
            paged_values.len() - 1,
            paged_values.len()
        );
        let mut statement = connection.prepare(&sql)?;
        let rows = statement.query_map(params_from_iter(paged_values.iter()), summary_from_row)?;
        let items = rows.collect::<Result<Vec<_>, _>>()?;
        Ok(RunPage {
            items,
            page: query.page,
            per_page: query.per_page,
            total,
            total_pages: total.div_ceil(u64::from(query.per_page)),
        })
    }

    pub fn run_detail(&self, run_id: &str) -> BenchmarkResult<Value> {
        require_text("run_id", run_id)?;
        let connection = self.connect()?;
        let summary = connection
            .query_row(
                "SELECT rs.run_id,rs.status,rs.repetition,rs.experiment_hash,rs.family,
                        rs.architecture,rs.quant_name,rs.fork_name,rs.backend,rs.context_tokens,
                        rs.workload,rs.prompt_tps,rs.generation_tps,rs.ttft_ms,
                        rs.peak_rss_bytes,rs.peak_vram_bytes,rs.speculator_type,
                        rs.acceptance_rate,r.started_at,r.ended_at,r.failure_reason,
                        (SELECT a.disk_bytes FROM experiments e
                           JOIN artifacts a ON a.id=e.artifact_id
                          WHERE e.id=r.experiment_id),
                        (SELECT AVG(q.metric_value) FROM quality_results q
                          WHERE q.run_id=r.id AND q.metric_name='pass@1')
                 FROM run_summary rs JOIN runs r ON r.id=rs.run_id WHERE rs.run_id=?1",
                params![run_id],
                summary_from_row,
            )
            .optional()?
            .ok_or_else(|| BenchmarkError::NotFound(format!("benchmark run {run_id} not found")))?;

        let (configuration, run_record, performance, speculation): (
            String,
            String,
            Option<String>,
            Option<String>,
        ) = connection.query_row(
            "SELECT
               json_object(
                 'model', json_object(
                   'id',m.id,'family',m.family,'architecture',m.architecture,
                   'checkpoint',m.checkpoint,'revision',m.revision,
                   'tokenizer_id',m.tokenizer_id,'tokenizer_revision',m.tokenizer_revision,
                   'parameter_count_total',m.parameter_count_total,
                   'parameter_count_active',m.parameter_count_active,
                   'model_kind',m.model_kind,'native_context_tokens',m.native_context_tokens,
                   'metadata',json(m.metadata_json)),
                 'artifact', json_object(
                   'id',a.id,'model_id',a.model_id,'format',a.format,
                   'quant_family',a.quant_family,'quant_name',a.quant_name,
                   'average_bits_per_weight',a.average_bits_per_weight,
                   'disk_bytes',a.disk_bytes,'sha256',a.sha256,'source_uri',a.source_uri,
                   'imatrix_used',json(CASE a.imatrix_used WHEN 1 THEN 'true' ELSE 'false' END),
                   'conversion_tool',a.conversion_tool,'conversion_commit',a.conversion_commit,
                   'conversion_command',a.conversion_command,'metadata',json(a.metadata_json)),
                 'runtime', json_object(
                   'id',rt.id,'repository',rt.repository,'fork_name',rt.fork_name,
                   'commit_sha',rt.commit_sha,
                   'dirty_tree',json(CASE rt.dirty_tree WHEN 1 THEN 'true' ELSE 'false' END),
                   'compiler',rt.compiler,'compiler_version',rt.compiler_version,
                   'backend',rt.backend,'build_flags',json(rt.build_flags_json),
                   'capabilities',json(rt.capabilities_json),
                   'executable_sha256',rt.executable_sha256,
                   'container_digest',rt.container_digest,'built_at',rt.built_at),
                 'hardware', json_object(
                   'id',h.id,'hostname_hash',h.hostname_hash,'cpu_model',h.cpu_model,
                   'physical_cores',h.physical_cores,'logical_cores',h.logical_cores,
                   'system_ram_bytes',h.system_ram_bytes,'gpus',json(h.gpu_json),
                   'unified_memory',json(CASE h.unified_memory WHEN 1 THEN 'true' ELSE 'false' END),
                   'os_name',h.os_name,'os_version',h.os_version,'kernel',h.kernel,
                   'driver_versions',json(h.driver_versions_json),
                   'power_profile',h.power_profile,'metadata',json(h.metadata_json),
                   'captured_at',h.captured_at),
                 'workload', json_object(
                   'id',w.id,'name',w.name,'version',w.version,
                   'workload_type',w.workload_type,'manifest_sha256',w.manifest_sha256,
                   'input_tokens',w.input_tokens,'output_tokens',w.output_tokens,
                   'corpus_bytes',w.corpus_bytes,'license',w.license,
                   'metadata',json(w.metadata_json)),
                 'experiment', json_object(
                   'id',e.id,'experiment_hash',e.experiment_hash,
                   'context_tokens',e.context_tokens,'prompt_tokens',e.prompt_tokens,
                   'generation_tokens',e.generation_tokens,'batch_size',e.batch_size,
                   'micro_batch_size',e.micro_batch_size,'threads',e.threads,
                   'optimization',json(e.optimization_json),
                   'sampling',json(e.sampling_json),'command_template',e.command_template)
               ),
               json_object(
                 'id',r.id,'experiment_id',r.experiment_id,'repetition',r.repetition,
                 'status',r.status,'started_at',r.started_at,'ended_at',r.ended_at,
                 'exit_code',r.exit_code,'random_seed',r.random_seed,
                 'warmup_count',r.warmup_count,'exact_command',r.exact_command,
                 'cwd',r.cwd,'environment',json(r.environment_json),
                 'stdout_path',r.stdout_path,'stderr_path',r.stderr_path,
                 'failure_reason',r.failure_reason,
                 'raw_result',CASE WHEN r.raw_result_json IS NULL THEN NULL ELSE json(r.raw_result_json) END
               ),
               CASE WHEN pm.run_id IS NULL THEN NULL ELSE json_object(
                 'model_load_ms',pm.model_load_ms,
                 'prompt_processing_ms',pm.prompt_processing_ms,'prompt_tps',pm.prompt_tps,
                 'ttft_ms',pm.ttft_ms,'generation_ms',pm.generation_ms,
                 'generation_tps',pm.generation_tps,
                 'inter_token_p50_ms',pm.inter_token_p50_ms,
                 'inter_token_p95_ms',pm.inter_token_p95_ms,
                 'inter_token_p99_ms',pm.inter_token_p99_ms,
                 'peak_rss_bytes',pm.peak_rss_bytes,'peak_vram_bytes',pm.peak_vram_bytes,
                 'kv_cache_bytes',pm.kv_cache_bytes,'energy_joules',pm.energy_joules,
                 'avg_power_watts',pm.avg_power_watts) END,
               CASE WHEN sm.run_id IS NULL THEN NULL ELSE json_object(
                 'speculator_type',sm.speculator_type,'proposals',sm.proposals,
                 'proposed_tokens',sm.proposed_tokens,'accepted_tokens',sm.accepted_tokens,
                 'target_evaluations',sm.target_evaluations,
                 'acceptance_rate',sm.acceptance_rate,
                 'speculator_memory_bytes',sm.speculator_memory_bytes,
                 'overhead_ms',sm.overhead_ms) END
             FROM runs r
             JOIN experiments e ON e.id=r.experiment_id
             JOIN artifacts a ON a.id=e.artifact_id
             JOIN models m ON m.id=a.model_id
             JOIN runtimes rt ON rt.id=e.runtime_id
             JOIN hardware_profiles h ON h.id=e.hardware_id
             JOIN workloads w ON w.id=e.workload_id
             LEFT JOIN performance_metrics pm ON pm.run_id=r.id
             LEFT JOIN speculative_metrics sm ON sm.run_id=r.id
             WHERE r.id=?1",
            params![run_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )?;

        let mut quality_statement = connection.prepare(
            "SELECT id,task_id,metric_name,metric_value,passed,compile_succeeded,
                    tests_passed,tests_total,generated_tokens,duration_ms,output_path,
                    log_path,details_json
             FROM quality_results WHERE run_id=?1 ORDER BY task_id,metric_name,id",
        )?;
        let quality = quality_statement
            .query_map(params![run_id], |row| {
                let details: String = row.get(12)?;
                Ok(json!({
                    "id": row.get::<_, String>(0)?,
                    "task_id": row.get::<_, String>(1)?,
                    "metric_name": row.get::<_, String>(2)?,
                    "metric_value": row.get::<_, Option<f64>>(3)?,
                    "passed": row.get::<_, Option<bool>>(4)?,
                    "compile_succeeded": row.get::<_, Option<bool>>(5)?,
                    "tests_passed": row.get::<_, Option<u64>>(6)?,
                    "tests_total": row.get::<_, Option<u64>>(7)?,
                    "generated_tokens": row.get::<_, Option<u64>>(8)?,
                    "duration_ms": row.get::<_, Option<f64>>(9)?,
                    "output_path": row.get::<_, Option<String>>(10)?,
                    "log_path": row.get::<_, Option<String>>(11)?,
                    "details": parse_json(&details),
                }))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let mut telemetry_statement = connection.prepare(
            "SELECT sampled_at,cpu_percent,rss_bytes,gpu_index,gpu_util_percent,
                    vram_used_bytes,temperature_c,power_watts,metadata_json
             FROM telemetry_samples WHERE run_id=?1 ORDER BY sampled_at,id",
        )?;
        let telemetry = telemetry_statement
            .query_map(params![run_id], |row| {
                let metadata: String = row.get(8)?;
                Ok(json!({
                    "sampled_at": row.get::<_, String>(0)?,
                    "cpu_percent": row.get::<_, Option<f64>>(1)?,
                    "rss_bytes": row.get::<_, Option<u64>>(2)?,
                    "gpu_index": row.get::<_, Option<u64>>(3)?,
                    "gpu_util_percent": row.get::<_, Option<f64>>(4)?,
                    "vram_used_bytes": row.get::<_, Option<u64>>(5)?,
                    "temperature_c": row.get::<_, Option<f64>>(6)?,
                    "power_watts": row.get::<_, Option<f64>>(7)?,
                    "metadata": parse_json(&metadata),
                }))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(json!({
            "run": summary,
            "configuration": parse_json(&configuration),
            "run_record": parse_json(&run_record),
            "performance_metrics": performance.as_deref().map(parse_json),
            "speculative_metrics": speculation.as_deref().map(parse_json),
            "quality_results": quality,
            "telemetry_samples": telemetry,
        }))
    }

    pub fn filter_options(&self) -> BenchmarkResult<Value> {
        let connection = self.connect()?;
        Ok(json!({
            "statuses": distinct_values(&connection, "SELECT DISTINCT status FROM runs ORDER BY status")?,
            "families": distinct_values(&connection, "SELECT DISTINCT family FROM run_summary ORDER BY family")?,
            "backends": distinct_values(&connection, "SELECT DISTINCT backend FROM run_summary ORDER BY backend")?,
            "workloads": distinct_values(&connection, "SELECT DISTINCT workload FROM run_summary ORDER BY workload")?,
            "quant_names": distinct_values(&connection, "SELECT DISTINCT quant_name FROM run_summary ORDER BY quant_name")?,
            "speculator_types": distinct_values(&connection, "SELECT DISTINCT speculator_type FROM run_summary WHERE speculator_type IS NOT NULL ORDER BY speculator_type")?,
        }))
    }

    pub fn export_runs(&self, mut query: RunQuery, format: &str) -> BenchmarkResult<String> {
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Deferred)?;
        query.page = 1;
        query.per_page = MAX_PAGE_SIZE;
        let mut all = Vec::new();
        loop {
            let page = self.query_runs_connection(&transaction, &query)?;
            all.extend(page.items);
            if u64::from(query.page) >= page.total_pages {
                break;
            }
            query.page += 1;
        }
        let output = match format {
            "jsonl" => {
                let mut output = String::new();
                for run in all {
                    output.push_str(&to_json(&run)?);
                    output.push('\n');
                }
                Ok(output)
            }
            "csv" => {
                let mut output = String::from(
                    "run_id,status,repetition,experiment_hash,family,architecture,quant_name,fork_name,backend,context_tokens,workload,prompt_tps,generation_tps,ttft_ms,peak_rss_bytes,peak_vram_bytes,speculator_type,acceptance_rate,started_at,ended_at,failure_reason,disk_bytes,quality_score\n",
                );
                for run in all {
                    let row = [
                        run.run_id,
                        run.status,
                        run.repetition.to_string(),
                        run.experiment_hash,
                        run.family,
                        run.architecture,
                        run.quant_name,
                        run.fork_name,
                        run.backend,
                        run.context_tokens.to_string(),
                        run.workload,
                        option_string(run.prompt_tps),
                        option_string(run.generation_tps),
                        option_string(run.ttft_ms),
                        option_string(run.peak_rss_bytes),
                        option_string(run.peak_vram_bytes),
                        run.speculator_type.unwrap_or_default(),
                        option_string(run.acceptance_rate),
                        run.started_at.unwrap_or_default(),
                        run.ended_at.unwrap_or_default(),
                        run.failure_reason.unwrap_or_default(),
                        run.disk_bytes.to_string(),
                        option_string(run.quality_score),
                    ];
                    output.push_str(
                        &row.into_iter()
                            .map(|value| csv_field(&value))
                            .collect::<Vec<_>>()
                            .join(","),
                    );
                    output.push('\n');
                }
                Ok(output)
            }
            _ => Err(BenchmarkError::Validation(
                "format must be jsonl or csv".into(),
            )),
        }?;
        transaction.commit()?;
        Ok(output)
    }
}

fn to_json<T: Serialize>(value: &T) -> BenchmarkResult<String> {
    serde_json::to_string(value)
        .map_err(|error| BenchmarkError::Validation(format!("invalid JSON value: {error}")))
}

fn ensure_fingerprint<T: Serialize>(
    transaction: &Transaction<'_>,
    entity_type: &str,
    entity_id: &str,
    value: &T,
) -> BenchmarkResult<()> {
    let mut hasher = Sha256::new();
    hasher.update(to_json(value)?.as_bytes());
    let fingerprint = format!("{:x}", hasher.finalize());
    let existing = transaction
        .query_row(
            "SELECT sha256 FROM entity_fingerprints WHERE entity_type=?1 AND entity_id=?2",
            params![entity_type, entity_id],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    if existing
        .as_deref()
        .is_some_and(|existing| existing != fingerprint)
    {
        return Err(BenchmarkError::Conflict(format!(
            "{entity_type} {entity_id} is immutable and the supplied payload differs"
        )));
    }
    transaction.execute(
        "INSERT OR IGNORE INTO entity_fingerprints(entity_type,entity_id,sha256)
         VALUES (?1,?2,?3)",
        params![entity_type, entity_id, fingerprint],
    )?;
    Ok(())
}

fn parse_json(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or(Value::Null)
}

fn option_string<T: ToString>(value: Option<T>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn csv_field(value: &str) -> String {
    let value = if value.chars().next().is_some_and(|character| {
        character.is_whitespace()
            || character.is_control()
            || matches!(character, '=' | '+' | '-' | '@')
    }) {
        format!("'{value}")
    } else {
        value.to_string()
    };
    if value
        .chars()
        .any(|character| matches!(character, ',' | '"' | '\n' | '\r'))
    {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value
    }
}

fn summary_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<RunSummary> {
    Ok(RunSummary {
        run_id: row.get(0)?,
        status: row.get(1)?,
        repetition: row.get(2)?,
        experiment_hash: row.get(3)?,
        family: row.get(4)?,
        architecture: row.get(5)?,
        quant_name: row.get(6)?,
        fork_name: row.get(7)?,
        backend: row.get(8)?,
        context_tokens: row.get(9)?,
        workload: row.get(10)?,
        prompt_tps: row.get(11)?,
        generation_tps: row.get(12)?,
        ttft_ms: row.get(13)?,
        peak_rss_bytes: row.get(14)?,
        peak_vram_bytes: row.get(15)?,
        speculator_type: row.get(16)?,
        acceptance_rate: row.get(17)?,
        started_at: row.get(18)?,
        ended_at: row.get(19)?,
        failure_reason: row.get(20)?,
        disk_bytes: row.get(21)?,
        quality_score: row.get(22)?,
    })
}

fn query_where(query: &RunQuery) -> (String, Vec<SqlValue>) {
    let mut clauses = Vec::new();
    let mut values = Vec::new();
    for (column, value) in [
        ("rs.status", query.status.as_ref()),
        ("rs.family", query.family.as_ref()),
        ("rs.backend", query.backend.as_ref()),
        ("rs.workload", query.workload.as_ref()),
        ("rs.quant_name", query.quant_name.as_ref()),
        ("rs.speculator_type", query.speculator_type.as_ref()),
    ] {
        if let Some(value) = value {
            values.push(SqlValue::Text(value.clone()));
            clauses.push(format!("{column} = ?{}", values.len()));
        }
    }
    if let Some(search) = &query.q {
        values.push(SqlValue::Text(format!("%{}%", escape_like(search))));
        let parameter = values.len();
        clauses.push(format!(
            "(rs.family LIKE ?{parameter} ESCAPE '\\' COLLATE NOCASE
              OR rs.architecture LIKE ?{parameter} ESCAPE '\\' COLLATE NOCASE
              OR rs.quant_name LIKE ?{parameter} ESCAPE '\\' COLLATE NOCASE
              OR rs.fork_name LIKE ?{parameter} ESCAPE '\\' COLLATE NOCASE
              OR rs.workload LIKE ?{parameter} ESCAPE '\\' COLLATE NOCASE
              OR rs.run_id LIKE ?{parameter} ESCAPE '\\' COLLATE NOCASE)"
        ));
    }
    let where_sql = if clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", clauses.join(" AND "))
    };
    (where_sql, values)
}

fn escape_like(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

fn distinct_values(connection: &Connection, sql: &str) -> BenchmarkResult<Vec<String>> {
    let mut statement = connection.prepare(sql)?;
    let values = statement
        .query_map([], |row| row.get(0))?
        .collect::<Result<Vec<String>, _>>()?;
    Ok(values)
}

enum BodyReadError {
    TooLarge,
    Read(String),
}

async fn collect_body(req: Request<Incoming>) -> Result<Bytes, BodyReadError> {
    Limited::new(req.into_body(), MAX_INGEST_BYTES)
        .collect()
        .await
        .map(|body| body.to_bytes())
        .map_err(|error| {
            if error.downcast_ref::<LengthLimitError>().is_some() {
                BodyReadError::TooLarge
            } else {
                BodyReadError::Read(error.to_string())
            }
        })
}

fn body_error_response(
    error: BodyReadError,
    payload_name: &str,
) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    match error {
        BodyReadError::TooLarge => json_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            &json!({"error": format!("{payload_name} payload exceeds {MAX_INGEST_BYTES} bytes")}),
        ),
        BodyReadError::Read(error) => json_response(
            StatusCode::BAD_REQUEST,
            &json!({"error": format!("failed to read {payload_name} payload: {error}")}),
        ),
    }
}

pub async fn handle_request(
    req: Request<Incoming>,
    store: &BenchmarkStore,
) -> Result<Response<UnsyncBoxBody<Bytes, anyhow::Error>>, Infallible> {
    let method = req.method().as_str();
    let path = req.uri().path().to_string();
    let response = match (method, path.as_str()) {
        ("GET", "/benchmarks") | ("GET", "/benchmarks/") => html_response(EXPLORER_HTML),
        ("GET", "/api/benchmarks/filters") => result_response(store.filter_options()),
        ("GET", "/api/benchmarks/export") => {
            match parse_export_query(req.uri().query()).and_then(|(query, format)| {
                store.export_runs(query, &format).map(|body| (format, body))
            }) {
                Ok((format, body)) => download_response(&format, body),
                Err(error) => error_response(error),
            }
        }
        ("GET", "/api/benchmarks/runs") => {
            match RunQuery::parse(req.uri().query()).and_then(|query| store.query_runs(&query)) {
                Ok(page) => json_response(StatusCode::OK, &page),
                Err(error) => error_response(error),
            }
        }
        ("GET", path) if path.starts_with("/api/benchmarks/runs/") => {
            let encoded = path.trim_start_matches("/api/benchmarks/runs/");
            match url::form_urlencoded::parse(format!("id={encoded}").as_bytes())
                .next()
                .map(|(_, value)| value.into_owned())
            {
                Some(run_id) => result_response(store.run_detail(&run_id)),
                None => json_response(StatusCode::BAD_REQUEST, &json!({"error": "invalid run id"})),
            }
        }
        ("POST", "/api/benchmarks/ingest") => {
            if req
                .headers()
                .get(hyper::header::CONTENT_LENGTH)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<usize>().ok())
                .is_some_and(|length| length > MAX_INGEST_BYTES)
            {
                json_response(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    &json!({"error": format!("ingest payload exceeds {MAX_INGEST_BYTES} bytes")}),
                )
            } else {
                match collect_body(req).await {
                    Ok(bytes) => match serde_json::from_slice::<IngestBundle>(&bytes) {
                        Ok(bundle) => match store.ingest(&bundle) {
                            Ok(run_id) => {
                                json_response(StatusCode::CREATED, &json!({"run_id": run_id}))
                            }
                            Err(error) => error_response(error),
                        },
                        Err(error) => json_response(
                            StatusCode::BAD_REQUEST,
                            &json!({"error": format!("invalid ingest payload: {error}")}),
                        ),
                    },
                    Err(error) => body_error_response(error, "ingest"),
                }
            }
        }
        ("POST", "/api/benchmarks/ingest/llama-bench") => match collect_body(req).await {
            Ok(bytes) => match serde_json::from_slice::<LlamaBenchIngest>(&bytes) {
                Ok(mut ingest) => {
                    match apply_llama_bench_metrics(&mut ingest.bundle, &ingest.llama_bench)
                        .and_then(|_| store.ingest(&ingest.bundle))
                    {
                        Ok(run_id) => {
                            json_response(StatusCode::CREATED, &json!({"run_id": run_id}))
                        }
                        Err(error) => error_response(error),
                    }
                }
                Err(error) => json_response(
                    StatusCode::BAD_REQUEST,
                    &json!({"error": format!("invalid llama-bench ingest payload: {error}")}),
                ),
            },
            Err(error) => body_error_response(error, "ingest"),
        },
        ("POST", "/api/benchmarks/plan") => {
            let is_yaml = req
                .headers()
                .get(hyper::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.contains("yaml"));
            match collect_body(req).await {
                Ok(bytes) => {
                    let matrix = if is_yaml {
                        serde_yaml::from_slice::<ExperimentMatrix>(&bytes)
                            .map_err(|error| error.to_string())
                    } else {
                        serde_json::from_slice::<ExperimentMatrix>(&bytes)
                            .map_err(|error| error.to_string())
                    };
                    match matrix {
                        Ok(matrix) => match matrix.expand() {
                            Ok(plan) => json_response(StatusCode::OK, &plan),
                            Err(error) => error_response(error),
                        },
                        Err(error) => json_response(
                            StatusCode::BAD_REQUEST,
                            &json!({"error": format!("invalid experiment matrix: {error}")}),
                        ),
                    }
                }
                Err(error) => body_error_response(error, "plan"),
            }
        }
        _ => json_response(StatusCode::NOT_FOUND, &json!({"error": "not found"})),
    };
    Ok(response)
}

fn parse_export_query(query: Option<&str>) -> BenchmarkResult<(RunQuery, String)> {
    let mut format = None;
    let mut serializer = url::form_urlencoded::Serializer::new(String::new());
    for (key, value) in url::form_urlencoded::parse(query.unwrap_or_default().as_bytes()) {
        match key.as_ref() {
            "format" => format = Some(value.into_owned()),
            "page" | "per_page" => {}
            _ => {
                serializer.append_pair(&key, &value);
            }
        }
    }
    let query = RunQuery::parse(Some(&serializer.finish()))?;
    Ok((query, format.unwrap_or_else(|| "jsonl".into())))
}

fn result_response(
    result: BenchmarkResult<Value>,
) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    match result {
        Ok(value) => json_response(StatusCode::OK, &value),
        Err(error) => error_response(error),
    }
}

fn error_response(error: BenchmarkError) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    let status = match &error {
        BenchmarkError::Validation(_) => StatusCode::BAD_REQUEST,
        BenchmarkError::Conflict(_) => StatusCode::CONFLICT,
        BenchmarkError::NotFound(_) => StatusCode::NOT_FOUND,
        BenchmarkError::Database(_) | BenchmarkError::Io(_) => {
            tracing::error!(error = %error, "Benchmark API failure");
            StatusCode::INTERNAL_SERVER_ERROR
        }
    };
    json_response(status, &json!({"error": error.to_string()}))
}

fn download_response(format: &str, body: String) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    let (content_type, extension) = if format == "csv" {
        ("text/csv; charset=utf-8", "csv")
    } else {
        ("application/x-ndjson; charset=utf-8", "jsonl")
    };
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", content_type)
        .header(
            "content-disposition",
            format!("attachment; filename=\"brainrouter-benchmarks.{extension}\""),
        )
        .header("cache-control", "no-store")
        .body(
            Full::new(Bytes::from(body))
                .map_err(|error: Infallible| match error {})
                .boxed_unsync(),
        )
        .expect("failed to build benchmark download response")
}

fn json_response<T: Serialize>(
    status: StatusCode,
    value: &T,
) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    let body = match serde_json::to_vec(value) {
        Ok(body) => body,
        Err(error) => {
            tracing::error!(error = %error, "Failed to serialize benchmark response");
            br#"{"error":"internal serialization error"}"#.to_vec()
        }
    };
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .header("cache-control", "no-store")
        .body(
            Full::new(Bytes::from(body))
                .map_err(|error: Infallible| match error {})
                .boxed_unsync(),
        )
        .expect("failed to build benchmark JSON response")
}

fn html_response(html: &'static str) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/html; charset=utf-8")
        .header("cache-control", "no-store")
        .body(
            Full::new(Bytes::from_static(html.as_bytes()))
                .map_err(|error: Infallible| match error {})
                .boxed_unsync(),
        )
        .expect("failed to build benchmark HTML response")
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestStore {
        store: BenchmarkStore,
        path: PathBuf,
    }

    impl Drop for TestStore {
        fn drop(&mut self) {
            for suffix in ["", "-wal", "-shm"] {
                let _ = std::fs::remove_file(format!("{}{}", self.path.display(), suffix));
            }
        }
    }

    fn test_store() -> TestStore {
        let path = std::env::temp_dir().join(format!("brainrouter-bench-{}.sqlite3", new_id()));
        TestStore {
            store: BenchmarkStore::open(&path).expect("create benchmark store"),
            path,
        }
    }

    fn sha(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn bundle(run_id: &str, repetition: u64, status: RunStatus) -> IngestBundle {
        let now = Utc::now();
        IngestBundle {
            model: ModelDefinition {
                id: "model-1".into(),
                family: "Qwen".into(),
                architecture: "transformer".into(),
                checkpoint: "org/model".into(),
                revision: "main".into(),
                tokenizer_id: "org/model".into(),
                tokenizer_revision: None,
                parameter_count_total: Some(30_000_000_000),
                parameter_count_active: Some(3_000_000_000),
                model_kind: ModelKind::Moe,
                native_context_tokens: Some(131_072),
                metadata: BTreeMap::new(),
            },
            artifact: ArtifactDefinition {
                id: "artifact-1".into(),
                model_id: "model-1".into(),
                format: "gguf".into(),
                quant_family: "k-quant".into(),
                quant_name: "Q6_K".into(),
                average_bits_per_weight: Some(6.5),
                disk_bytes: 24_000_000_000,
                sha256: sha('a'),
                source_uri: None,
                imatrix_used: true,
                conversion_tool: Some("llama-quantize".into()),
                conversion_commit: None,
                conversion_command: None,
                metadata: BTreeMap::new(),
            },
            runtime: RuntimeDefinition {
                id: "runtime-1".into(),
                repository: "https://github.com/ggerganov/llama.cpp".into(),
                fork_name: "upstream".into(),
                commit_sha: "abc123".into(),
                dirty_tree: false,
                compiler: "clang".into(),
                compiler_version: "18".into(),
                backend: Backend::Vulkan,
                build_flags: vec!["GGML_VULKAN=ON".into()],
                capabilities: BTreeMap::new(),
                executable_sha256: None,
                container_digest: None,
                built_at: now,
            },
            hardware: HardwareProfile {
                id: "hardware-1".into(),
                hostname_hash: None,
                cpu_model: "AMD Ryzen AI MAX+ PRO 395".into(),
                physical_cores: Some(16),
                logical_cores: Some(32),
                system_ram_bytes: 128_000_000_000,
                gpus: Vec::new(),
                unified_memory: true,
                os_name: "Linux".into(),
                os_version: "6".into(),
                kernel: "6.18".into(),
                driver_versions: BTreeMap::new(),
                power_profile: None,
                metadata: BTreeMap::new(),
                captured_at: now,
            },
            workload: WorkloadDefinition {
                id: "workload-1".into(),
                name: "long-context".into(),
                version: "1".into(),
                workload_type: WorkloadType::Performance,
                manifest_sha256: sha('b'),
                input_tokens: Some(128_000),
                output_tokens: Some(512),
                corpus_bytes: None,
                license: None,
                metadata: BTreeMap::new(),
            },
            experiment: ExperimentConfig {
                id: "experiment-1".into(),
                artifact_id: "artifact-1".into(),
                runtime_id: "runtime-1".into(),
                hardware_id: "hardware-1".into(),
                workload_id: "workload-1".into(),
                context_tokens: 131_072,
                prompt_tokens: 128_000,
                generation_tokens: 512,
                batch_size: 1,
                micro_batch_size: 1,
                threads: Some(16),
                optimization: OptimizationConfig::default(),
                sampling: SamplingConfig::default(),
                command_template: "llama-bench ...".into(),
            },
            run: RunRecord {
                id: run_id.into(),
                experiment_id: "experiment-1".into(),
                repetition,
                status,
                started_at: Some(now),
                ended_at: Some(now),
                exit_code: Some(0),
                random_seed: Some(42),
                warmup_count: 1,
                exact_command: "llama-bench --no-run-by-brainrouter".into(),
                cwd: None,
                environment: BTreeMap::new(),
                stdout_path: None,
                stderr_path: None,
                failure_reason: None,
                raw_result: None,
            },
            performance_metrics: Some(PerformanceMetrics {
                run_id: run_id.into(),
                model_load_ms: Some(1.0),
                prompt_processing_ms: Some(2.0),
                prompt_tps: Some(1000.0),
                ttft_ms: Some(3.0),
                generation_ms: Some(4.0),
                generation_tps: Some(50.0 + repetition as f64),
                inter_token_p50_ms: Some(10.0),
                inter_token_p95_ms: Some(20.0),
                inter_token_p99_ms: Some(30.0),
                peak_rss_bytes: Some(1),
                peak_vram_bytes: Some(2),
                kv_cache_bytes: Some(3),
                energy_joules: None,
                avg_power_watts: None,
            }),
            speculative_metrics: None,
            quality_results: Vec::new(),
            telemetry_samples: Vec::new(),
        }
    }

    #[test]
    fn migration_creates_contract_schema_and_enables_pragmas() {
        let test = test_store();
        let connection = test.store.connect().unwrap();
        let tables = distinct_values(
            &connection,
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        )
        .unwrap();
        for table in [
            "models",
            "artifacts",
            "experiments",
            "runs",
            "performance_metrics",
            "quality_results",
            "telemetry_samples",
        ] {
            assert!(tables.iter().any(|name| name == table), "missing {table}");
        }
        assert!(connection
            .query_row("PRAGMA foreign_keys", [], |row| row.get::<_, bool>(0))
            .unwrap());
        assert_eq!(
            connection
                .query_row("PRAGMA journal_mode", [], |row| row.get::<_, String>(0))
                .unwrap(),
            "wal"
        );
    }

    #[test]
    fn unrelated_schema_migration_table_is_rejected() {
        let path =
            std::env::temp_dir().join(format!("brainrouter-bench-unrelated-{}.sqlite3", new_id()));
        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE schema_migrations(
                   version INTEGER PRIMARY KEY,
                   name TEXT NOT NULL UNIQUE,
                   applied_at TEXT NOT NULL
                 ) STRICT;
                 INSERT INTO schema_migrations VALUES(1,'other_application','now');",
            )
            .unwrap();
        drop(connection);
        assert!(BenchmarkStore::open(&path).is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn experiment_hash_is_stable_and_excludes_id() {
        let a = bundle("run-a", 0, RunStatus::Planned).experiment;
        let mut b = a.clone();
        b.id = "different".into();
        assert_eq!(a.experiment_hash().unwrap(), b.experiment_hash().unwrap());
    }

    #[test]
    fn model_and_speculation_validation_matches_contract() {
        let mut invalid_model = bundle("run-a", 0, RunStatus::Planned);
        invalid_model.model.parameter_count_active = Some(31_000_000_000);
        assert!(invalid_model.validate().is_err());

        let mut invalid_optimization = bundle("run-b", 0, RunStatus::Planned);
        invalid_optimization.experiment.optimization.mtp = FeatureState::Enabled;
        assert!(invalid_optimization.validate().is_err());

        let mut too_large = bundle("run-c", 0, RunStatus::Planned);
        too_large.artifact.disk_bytes = i64::MAX as u64 + 1;
        assert!(too_large.validate().is_err());
    }

    #[test]
    fn metrics_validation_matches_contract() {
        let mut invalid = bundle("run-a", 0, RunStatus::Planned);
        let metrics = invalid.performance_metrics.as_mut().unwrap();
        metrics.inter_token_p50_ms = Some(30.0);
        metrics.inter_token_p95_ms = Some(20.0);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn ingestion_queries_with_stable_pagination_and_filters() {
        let test = test_store();
        let first = bundle("run-a", 0, RunStatus::Failed);
        test.store.ingest(&first).unwrap();
        let mut second = first.clone();
        second.run.id = "run-b".into();
        second.run.repetition = 1;
        second.run.status = RunStatus::Succeeded;
        second.run.started_at = second
            .run
            .started_at
            .map(|time| time + chrono::Duration::seconds(1));
        second.run.ended_at = second.run.started_at;
        second.performance_metrics.as_mut().unwrap().run_id = "run-b".into();
        second.performance_metrics.as_mut().unwrap().generation_tps = Some(51.0);
        second.quality_results = vec![
            QualityResult {
                id: "quality-pass".into(),
                run_id: "run-b".into(),
                task_id: "task-1".into(),
                metric_name: "pass@1".into(),
                metric_value: Some(0.5),
                passed: Some(true),
                compile_succeeded: Some(true),
                tests_passed: Some(1),
                tests_total: Some(1),
                generated_tokens: Some(10),
                duration_ms: Some(1.0),
                output_path: None,
                log_path: None,
                details: BTreeMap::new(),
            },
            QualityResult {
                id: "quality-perplexity".into(),
                run_id: "run-b".into(),
                task_id: "task-1".into(),
                metric_name: "perplexity".into(),
                metric_value: Some(100.0),
                passed: None,
                compile_succeeded: None,
                tests_passed: None,
                tests_total: None,
                generated_tokens: None,
                duration_ms: None,
                output_path: None,
                log_path: None,
                details: BTreeMap::new(),
            },
        ];
        test.store.ingest(&second).unwrap();

        let page = test
            .store
            .query_runs(
                &RunQuery::parse(Some("per_page=1&sort=generation_tps&order=desc")).unwrap(),
            )
            .unwrap();
        assert_eq!(page.total, 2);
        assert_eq!(page.total_pages, 2);
        assert_eq!(page.items[0].run_id, "run-b");
        assert_eq!(page.items[0].quality_score, Some(0.5));

        let filtered = test
            .store
            .query_runs(&RunQuery::parse(Some("status=failed&family=Qwen")).unwrap())
            .unwrap();
        assert_eq!(filtered.total, 1);
        assert_eq!(filtered.items[0].run_id, "run-a");

        let detail = test.store.run_detail("run-b").unwrap();
        assert_eq!(detail["configuration"]["model"]["family"], "Qwen");
        assert_eq!(detail["performance_metrics"]["generation_tps"], json!(51.0));

        let csv = test
            .store
            .export_runs(RunQuery::parse(None).unwrap(), "csv")
            .unwrap();
        assert!(csv.contains("run-a"));
        assert!(csv.contains("run-b"));
    }

    #[test]
    fn successful_runs_are_immutable_but_failed_runs_can_be_completed() {
        let test = test_store();
        let failed = bundle("run-a", 0, RunStatus::Failed);
        test.store.ingest(&failed).unwrap();
        let mut succeeded = failed.clone();
        succeeded.run.status = RunStatus::Succeeded;
        test.store.ingest(&succeeded).unwrap();
        assert!(matches!(
            test.store.ingest(&succeeded),
            Err(BenchmarkError::Conflict(_))
        ));
    }

    #[test]
    fn registry_entities_are_immutable() {
        let test = test_store();
        let initial = bundle("run-a", 0, RunStatus::Failed);
        test.store.ingest(&initial).unwrap();
        let mut changed = initial.clone();
        changed.run.status = RunStatus::Running;
        changed.model.architecture = "changed".into();
        assert!(matches!(
            test.store.ingest(&changed),
            Err(BenchmarkError::Conflict(_))
        ));
    }

    #[test]
    fn mutable_run_identity_cannot_move_between_repetitions() {
        let test = test_store();
        let initial = bundle("run-a", 0, RunStatus::Failed);
        test.store.ingest(&initial).unwrap();
        let mut moved = initial.clone();
        moved.run.status = RunStatus::Running;
        moved.run.repetition = 1;
        assert!(matches!(
            test.store.ingest(&moved),
            Err(BenchmarkError::Conflict(_))
        ));
    }

    #[test]
    fn draft_model_artifacts_must_exist() {
        let test = test_store();
        let mut invalid = bundle("run-a", 0, RunStatus::Planned);
        invalid.experiment.optimization.speculator_type = SpeculatorType::DraftModel;
        invalid.experiment.optimization.draft_model_artifact_id = Some("missing-draft".into());
        assert!(matches!(
            test.store.ingest(&invalid),
            Err(BenchmarkError::Validation(_))
        ));
    }

    #[test]
    fn query_validation_rejects_unbounded_or_ambiguous_requests() {
        assert!(RunQuery::parse(Some("page=0")).is_err());
        assert!(RunQuery::parse(Some("per_page=101")).is_err());
        assert!(RunQuery::parse(Some("sort=drop_table")).is_err());
        assert!(RunQuery::parse(Some("unknown=value")).is_err());
    }

    #[test]
    fn csv_export_neutralizes_spreadsheet_formulas() {
        assert_eq!(csv_field("=cmd()"), "'=cmd()");
        assert_eq!(csv_field("@sum,1"), "\"'@sum,1\"");
        assert_eq!(csv_field("\t=cmd()"), "'\t=cmd()");
    }

    #[test]
    fn matrix_planning_is_deterministic_and_records_invalid_candidates() {
        let matrix = ExperimentMatrix {
            name: "contexts".into(),
            artifacts: vec!["artifact-1".into()],
            runtimes: vec!["runtime-1".into()],
            hardware: vec!["hardware-1".into()],
            workloads: vec!["workload-1".into()],
            contexts: vec![8_192, 32_768],
            prompt_tokens: vec![8_000, 16_000],
            generation_tokens: vec![128],
            optimizations: default_optimizations(),
            repetitions: 3,
            randomize_order: false,
            capture_telemetry: true,
            batch_size: 1,
            micro_batch_size: 1,
            threads: Some(16),
            command_template: "llama-bench ...".into(),
        };
        let first = matrix.expand().unwrap();
        let second = matrix.expand().unwrap();
        assert_eq!(
            first
                .experiments
                .iter()
                .map(|experiment| &experiment.id)
                .collect::<Vec<_>>(),
            second
                .experiments
                .iter()
                .map(|experiment| &experiment.id)
                .collect::<Vec<_>>()
        );
        assert_eq!(first.experiments.len(), 3);
        assert_eq!(first.exclusions.len(), 1);
        assert_eq!(first.run_count, 9);
    }

    #[test]
    fn matrix_planning_rejects_duplicate_dimensions() {
        let mut matrix = ExperimentMatrix {
            name: "duplicate".into(),
            artifacts: vec!["artifact-1".into()],
            runtimes: vec!["runtime-1".into()],
            hardware: vec!["hardware-1".into()],
            workloads: vec!["workload-1".into()],
            contexts: vec![8_192, 8_192],
            prompt_tokens: vec![8_000],
            generation_tokens: vec![128],
            optimizations: default_optimizations(),
            repetitions: 1,
            randomize_order: false,
            capture_telemetry: true,
            batch_size: 1,
            micro_batch_size: 1,
            threads: None,
            command_template: "llama-bench ...".into(),
        };
        assert!(matrix.expand().is_err());
        matrix.contexts = vec![8_192];
        assert!(matrix.expand().is_ok());
    }

    #[test]
    fn llama_bench_adapter_extracts_prompt_and_generation_rates() {
        let mut bundle = bundle("run-a", 0, RunStatus::Succeeded);
        bundle.performance_metrics = None;
        apply_llama_bench_metrics(
            &mut bundle,
            &json!([
                {"test": "pp512", "avg_ts": 1234.5, "n_prompt": 512, "n_gen": 0},
                {"test": "tg128", "avg_ts": 55.25, "n_prompt": 0, "n_gen": 128}
            ]),
        )
        .unwrap();
        let metrics = bundle.performance_metrics.unwrap();
        assert_eq!(metrics.prompt_tps, Some(1234.5));
        assert_eq!(metrics.generation_tps, Some(55.25));
    }

    #[test]
    fn llama_bench_adapter_rejects_ambiguous_rows() {
        let mut bundle = bundle("run-a", 0, RunStatus::Succeeded);
        bundle.performance_metrics = None;
        assert!(apply_llama_bench_metrics(
            &mut bundle,
            &json!([
                {"test": "tg128", "avg_ts": 50.0, "n_gen": 128},
                {"test": "tg128", "avg_ts": 55.0, "n_gen": 128}
            ]),
        )
        .is_err());

        assert!(apply_llama_bench_metrics(
            &mut bundle,
            &json!({"test": "pp512+tg128", "avg_ts": 70.0, "n_prompt": 512, "n_gen": 128}),
        )
        .is_err());
    }
}
