//! Native Riddllr and Plumebench execution.
//!
//! The lab is opt-in. It reads configured suite roots without modifying them,
//! runs one heavy job at a time in a private workspace, and ingests completed
//! evidence through the benchmark registry's normal validation boundary.

use crate::{
    benchmark::{
        ArtifactDefinition, Backend, BenchmarkError, BenchmarkJobRecord, BenchmarkStore,
        ExperimentConfig, FeatureState, HardwareProfile, IngestBundle, ModelDefinition, ModelKind,
        OptimizationConfig, PerformanceMetrics, QualityResult, RunRecord, RunStatus,
        RuntimeDefinition, SamplingConfig, WorkloadDefinition, WorkloadType,
    },
    config::BenchmarkLabConfig,
    provider::ProviderResponse,
    router::Router,
    types::{ChatCompletionRequest, ChatMessage},
};
use anyhow::{anyhow, bail, Context};
use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use http_body_util::{combinators::UnsyncBoxBody, BodyExt, Full, LengthLimitError, Limited};
use hyper::{body::Incoming, Method, Request, Response, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    convert::Infallible,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    process::ExitStatus,
    sync::Arc,
    time::Duration,
};
use tokio::{
    io::AsyncReadExt,
    process::Command,
    sync::{watch, Mutex, Notify, RwLock, Semaphore},
    time::Instant,
};
use tracing::{error, warn};

const MAX_LAB_REQUEST_BYTES: usize = 64 * 1024;
const MAX_CAPTURE_BYTES: usize = 4 * 1024 * 1024;
const MAX_SUITE_FILE_BYTES: u64 = 8 * 1024 * 1024;
const MAX_SUITE_TREE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SSE_BUFFER_BYTES: usize = 1024 * 1024;
const MAX_JOB_HISTORY: u32 = 100;

type HttpResponse = Response<UnsyncBoxBody<Bytes, anyhow::Error>>;

#[derive(Debug)]
enum LabError {
    Validation(String),
    Limit(String),
    Conflict(String),
    NotFound(String),
    Busy(String),
    Internal(String),
}

impl LabError {
    fn status(&self) -> StatusCode {
        match self {
            Self::Validation(_) => StatusCode::BAD_REQUEST,
            Self::Limit(_) => StatusCode::PAYLOAD_TOO_LARGE,
            Self::Conflict(_) => StatusCode::CONFLICT,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::Busy(_) => StatusCode::TOO_MANY_REQUESTS,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn message(&self) -> &str {
        match self {
            Self::Validation(message)
            | Self::Limit(message)
            | Self::Conflict(message)
            | Self::NotFound(message)
            | Self::Busy(message)
            | Self::Internal(message) => message,
        }
    }
}

impl From<BenchmarkError> for LabError {
    fn from(error: BenchmarkError) -> Self {
        match error {
            BenchmarkError::Validation(message) => Self::Validation(message),
            BenchmarkError::Limit(message) => Self::Limit(message),
            BenchmarkError::Conflict(message) => Self::Conflict(message),
            BenchmarkError::NotFound(message) => Self::NotFound(message),
            BenchmarkError::Busy(message) => Self::Busy(message),
            BenchmarkError::Database(error) => {
                Self::Internal(format!("benchmark database error: {error}"))
            }
            BenchmarkError::Io(error) => {
                Self::Internal(format!("benchmark storage I/O error: {error}"))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SuiteCase {
    id: String,
    name: String,
    manifest_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SuiteDescriptor {
    id: &'static str,
    name: &'static str,
    description: &'static str,
    available: bool,
    reason: Option<String>,
    cases: Vec<SuiteCase>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StartJobRequest {
    suite: String,
    case_id: String,
    model: String,
    #[serde(default)]
    repetition: u64,
}

#[derive(Debug, Clone)]
enum ResolvedCase {
    Riddllr {
        id: String,
        prompt: String,
        solution: String,
        manifest_sha256: String,
        prompt_name: String,
        solution_name: String,
    },
    Plumebench {
        id: String,
        task_dir: PathBuf,
        prompt: String,
        manifest_sha256: String,
    },
}

impl ResolvedCase {
    fn id(&self) -> &str {
        match self {
            Self::Riddllr { id, .. } | Self::Plumebench { id, .. } => id,
        }
    }

    fn manifest_sha256(&self) -> &str {
        match self {
            Self::Riddllr {
                manifest_sha256, ..
            }
            | Self::Plumebench {
                manifest_sha256, ..
            } => manifest_sha256,
        }
    }

    fn workload_name(&self) -> String {
        match self {
            Self::Riddllr { id, .. } => format!("Riddllr · {id}"),
            Self::Plumebench { id, .. } => format!("Plumebench · {id}"),
        }
    }

    fn suite(&self) -> &'static str {
        match self {
            Self::Riddllr { .. } => "riddllr",
            Self::Plumebench { .. } => "plumebench",
        }
    }

    fn prompt_bytes(&self) -> usize {
        match self {
            Self::Riddllr { prompt, .. } | Self::Plumebench { prompt, .. } => prompt.len(),
        }
    }
}

struct JobControl {
    record: RwLock<BenchmarkJobRecord>,
    cancel: watch::Sender<bool>,
}

#[derive(Debug)]
struct ExecutionResult {
    run_status: RunStatus,
    exit_code: Option<i64>,
    passed: Option<bool>,
    duration_ms: f64,
    stdout_path: Option<PathBuf>,
    stderr_path: Option<PathBuf>,
    failure_reason: Option<String>,
    details: Value,
    quality: Vec<QualityMeasurement>,
}

#[derive(Debug)]
struct QualityMeasurement {
    task_id: String,
    metric_name: String,
    metric_value: Option<f64>,
    passed: Option<bool>,
    compile_succeeded: Option<bool>,
    tests_passed: Option<u64>,
    tests_total: Option<u64>,
    generated_tokens: Option<u64>,
    duration_ms: Option<f64>,
    output_path: Option<PathBuf>,
    log_path: Option<PathBuf>,
    details: Value,
}

#[derive(Debug)]
enum ProcessEnd {
    Completed(ExitStatus),
    Cancelled,
    Timeout,
    TurnLimit,
}

#[derive(Debug)]
struct ProcessResult {
    end: ProcessEnd,
    stdout: Vec<u8>,
}

#[derive(Debug, Clone)]
struct SuiteRoot {
    configured: Option<PathBuf>,
    canonical: Option<PathBuf>,
    error: Option<String>,
}

impl SuiteRoot {
    fn new(configured: Option<PathBuf>) -> Self {
        let Some(path) = configured.clone() else {
            return Self {
                configured,
                canonical: None,
                error: Some("not configured".into()),
            };
        };
        match fs::canonicalize(&path) {
            Ok(canonical) if canonical.is_dir() => Self {
                configured,
                canonical: Some(canonical),
                error: None,
            },
            Ok(_) => Self {
                configured,
                canonical: None,
                error: Some(format!("{} is not a directory", path.display())),
            },
            Err(error) => Self {
                configured,
                canonical: None,
                error: Some(format!("{}: {error}", path.display())),
            },
        }
    }

    fn path(&self) -> Result<&Path, LabError> {
        self.canonical.as_deref().ok_or_else(|| {
            LabError::Validation(
                self.error
                    .clone()
                    .unwrap_or_else(|| "suite root is unavailable".into()),
            )
        })
    }
}

pub struct BenchmarkLab {
    config: BenchmarkLabConfig,
    brainrouter_socket_path: PathBuf,
    riddllr: SuiteRoot,
    plumebench: SuiteRoot,
    router: Arc<Router>,
    store: Arc<BenchmarkStore>,
    execution_slot: Arc<Semaphore>,
    controls: Mutex<HashMap<String, Arc<JobControl>>>,
    runtime: RuntimeDefinition,
    hardware: HardwareProfile,
}

impl BenchmarkLab {
    pub fn new(
        mut config: BenchmarkLabConfig,
        brainrouter_socket_path: PathBuf,
        router: Arc<Router>,
        store: Arc<BenchmarkStore>,
    ) -> anyhow::Result<Self> {
        fs::create_dir_all(config.workspace_path.join("jobs")).with_context(|| {
            format!(
                "failed to create benchmark workspace {}",
                config.workspace_path.display()
            )
        })?;
        config.workspace_path = fs::canonicalize(&config.workspace_path).with_context(|| {
            format!(
                "failed to resolve benchmark workspace {}",
                config.workspace_path.display()
            )
        })?;
        let riddllr = SuiteRoot::new(config.riddllr_root.clone());
        let plumebench = SuiteRoot::new(config.plumebench_root.clone());
        for (name, root) in [("Riddllr", &riddllr), ("Plumebench", &plumebench)] {
            if let Some(source) = root.canonical.as_deref() {
                if paths_overlap(&config.workspace_path, source) {
                    bail!(
                        "benchmark workspace {} must not overlap the {name} source root {}",
                        config.workspace_path.display(),
                        source.display()
                    );
                }
            }
        }
        let interrupted = store
            .interrupt_incomplete_jobs()
            .context("failed to recover incomplete benchmark jobs")?;
        if interrupted > 0 {
            warn!(
                interrupted,
                "Marked benchmark jobs interrupted after daemon restart"
            );
        }
        let runtime = runtime_definition()?;
        let hardware = hardware_profile();
        Ok(Self {
            riddllr,
            plumebench,
            config,
            brainrouter_socket_path,
            router,
            store,
            execution_slot: Arc::new(Semaphore::new(1)),
            controls: Mutex::new(HashMap::new()),
            runtime,
            hardware,
        })
    }

    pub fn suites(&self) -> Vec<SuiteDescriptor> {
        vec![self.riddllr_descriptor(), self.plumebench_descriptor()]
    }

    fn riddllr_descriptor(&self) -> SuiteDescriptor {
        match discover_riddllr_cases(&self.riddllr) {
            Ok(cases) => SuiteDescriptor {
                id: "riddllr",
                name: "Riddllr",
                description:
                    "Prompt-and-answer reasoning evaluation with deterministic local grading.",
                available: !cases.is_empty(),
                reason: cases
                    .is_empty()
                    .then(|| "No matched prompt/solution files were found".into()),
                cases,
            },
            Err(error) => SuiteDescriptor {
                id: "riddllr",
                name: "Riddllr",
                description:
                    "Prompt-and-answer reasoning evaluation with deterministic local grading.",
                available: false,
                reason: Some(error.message().to_string()),
                cases: Vec::new(),
            },
        }
    }

    fn plumebench_descriptor(&self) -> SuiteDescriptor {
        match discover_plumebench_cases(&self.plumebench) {
            Ok(cases) => SuiteDescriptor {
                id: "plumebench",
                name: "Plumebench",
                description:
                    "Agentic coding tasks with isolated hidden tests and optional elegance metrics.",
                available: !cases.is_empty(),
                reason: cases
                    .is_empty()
                    .then(|| "No complete task directories were found".into()),
                cases,
            },
            Err(error) => SuiteDescriptor {
                id: "plumebench",
                name: "Plumebench",
                description:
                    "Agentic coding tasks with isolated hidden tests and optional elegance metrics.",
                available: false,
                reason: Some(error.message().to_string()),
                cases: Vec::new(),
            },
        }
    }

    async fn start_job(
        self: &Arc<Self>,
        mut request: StartJobRequest,
    ) -> Result<BenchmarkJobRecord, LabError> {
        request.suite = request.suite.trim().to_ascii_lowercase();
        request.case_id = request.case_id.trim().to_string();
        request.model = normalize_model(&request.model)?;
        if request.repetition > i64::MAX as u64 {
            return Err(LabError::Validation(
                "repetition exceeds SQLite's signed 64-bit integer limit".into(),
            ));
        }
        let resolved = self.resolve_case(&request)?;
        let identity_sha256 = job_identity(&request, resolved.manifest_sha256(), &self.config)?;
        let run_id = format!("run-{}", &identity_sha256[..32]);
        let existing_run_id = run_id.clone();
        let existing_status = self
            .store
            .run_blocking(move |store| store.run_status(&existing_run_id))
            .await
            .map_err(LabError::from)?;
        if existing_status.as_deref() == Some("succeeded") {
            return Err(LabError::Conflict(format!(
                "this suite, case, model, and repetition already has a successful run ({run_id}); increment repetition"
            )));
        }

        let now = Utc::now();
        let job_id = uuid::Uuid::new_v4().to_string();
        let work_dir = self.config.workspace_path.join("jobs").join(&job_id);
        let record = BenchmarkJobRecord {
            id: job_id.clone(),
            identity_sha256,
            suite: request.suite.clone(),
            case_id: request.case_id.clone(),
            model: request.model.clone(),
            repetition: request.repetition,
            status: "queued".into(),
            progress: 0.0,
            message: "Waiting for the benchmark execution slot".into(),
            queued_at: now,
            started_at: None,
            ended_at: None,
            run_id: Some(run_id),
            work_dir: Some(work_dir.to_string_lossy().into_owned()),
            request: json!({
                "suite": request.suite,
                "case_id": request.case_id,
                "model": request.model,
                "repetition": request.repetition,
                "manifest_sha256": resolved.manifest_sha256(),
            }),
            result: None,
            error: None,
            updated_at: now,
        };
        let stored = record.clone();
        self.store
            .run_blocking(move |store| store.save_job(&stored))
            .await
            .map_err(LabError::from)?;

        let (cancel, cancel_rx) = watch::channel(false);
        let control = Arc::new(JobControl {
            record: RwLock::new(record.clone()),
            cancel,
        });
        self.controls
            .lock()
            .await
            .insert(job_id.clone(), Arc::clone(&control));

        let lab = Arc::clone(self);
        tokio::spawn(async move {
            lab.execute_job(control, request, resolved, cancel_rx).await;
        });
        Ok(record)
    }

    async fn execute_job(
        self: Arc<Self>,
        control: Arc<JobControl>,
        request: StartJobRequest,
        resolved: ResolvedCase,
        mut cancel: watch::Receiver<bool>,
    ) {
        let job_id = control.record.read().await.id.clone();
        let permit = tokio::select! {
            permit = Arc::clone(&self.execution_slot).acquire_owned() => match permit {
                Ok(permit) => permit,
                Err(_) => {
                    let persisted = self.finish_without_ingest(
                        &control,
                        "failed",
                        "Benchmark execution slot closed",
                        Some("Benchmark execution slot closed".into()),
                    ).await.is_ok();
                    if persisted {
                        self.controls.lock().await.remove(&job_id);
                    }
                    return;
                }
            },
            changed = cancel.changed() => {
                let persisted = if changed.is_ok() && *cancel.borrow() {
                    self.finish_without_ingest(
                        &control,
                        "cancelled",
                        "Cancelled before execution started",
                        None,
                    ).await
                } else {
                    self.finish_without_ingest(
                        &control,
                        "failed",
                        "Benchmark cancellation channel closed",
                        Some("Benchmark cancellation channel closed".into()),
                    ).await
                }.is_ok();
                if persisted {
                    self.controls.lock().await.remove(&job_id);
                }
                return;
            }
        };

        let started_at = Utc::now();
        if let Err(error) = self
            .update_job(&control, |job| {
                job.status = "running".into();
                job.progress = 0.05;
                job.message = "Preparing benchmark inputs".into();
                job.started_at = Some(started_at);
                job.updated_at = started_at;
            })
            .await
        {
            error!(job_id, error = %error.message(), "Failed to mark benchmark job running");
            let persisted = self
                .finish_without_ingest(
                    &control,
                    "failed",
                    "Benchmark execution could not persist its running state",
                    Some(error.message().to_string()),
                )
                .await
                .is_ok();
            drop(permit);
            if persisted {
                self.controls.lock().await.remove(&job_id);
            }
            return;
        }

        let execution = match &resolved {
            ResolvedCase::Riddllr { .. } => {
                self.run_riddllr(&control, &request, &resolved, cancel.clone())
                    .await
            }
            ResolvedCase::Plumebench { .. } => {
                self.run_plumebench(&control, &request, &resolved, cancel.clone())
                    .await
            }
        };
        let ended_at = Utc::now();

        let terminal_persisted = match execution {
            Ok(result) => {
                let job_snapshot = control.record.read().await.clone();
                let bundle = self.build_bundle(
                    &job_snapshot,
                    &request,
                    &resolved,
                    started_at,
                    ended_at,
                    &result,
                );
                match bundle {
                    Ok(bundle) => {
                        let run_id = bundle.run.id.clone();
                        let stored_bundle = bundle.clone();
                        match self
                            .store
                            .run_critical(move |store| store.ingest(&stored_bundle))
                            .await
                        {
                            Ok(_) => {
                                let status = job_status(result.run_status);
                                let result_json = json!({
                                    "run_id": run_id,
                                    "passed": result.passed,
                                    "duration_ms": result.duration_ms,
                                    "details": result.details,
                                });
                                match self
                                    .update_job(&control, |job| {
                                        job.status = status.into();
                                        job.progress = 1.0;
                                        job.message = completion_message(&result).into();
                                        job.ended_at = Some(ended_at);
                                        job.result = Some(result_json);
                                        job.error = result.failure_reason.clone();
                                        job.updated_at = ended_at;
                                    })
                                    .await
                                {
                                    Ok(()) => true,
                                    Err(error) => {
                                        error!(job_id, error = %error.message(), "Failed to persist completed benchmark job");
                                        false
                                    }
                                }
                            }
                            Err(error) => self
                                .finish_without_ingest(
                                    &control,
                                    "failed",
                                    "Benchmark completed, but registry ingestion failed",
                                    Some(error.to_string()),
                                )
                                .await
                                .is_ok(),
                        }
                    }
                    Err(error) => self
                        .finish_without_ingest(
                            &control,
                            "failed",
                            "Benchmark completed, but its result bundle was invalid",
                            Some(error.to_string()),
                        )
                        .await
                        .is_ok(),
                }
            }
            Err(error) => self
                .finish_without_ingest(
                    &control,
                    "failed",
                    "Benchmark execution failed",
                    Some(error.to_string()),
                )
                .await
                .is_ok(),
        };

        drop(permit);
        if terminal_persisted {
            self.controls.lock().await.remove(&job_id);
        }
    }

    async fn finish_without_ingest(
        &self,
        control: &Arc<JobControl>,
        status: &str,
        message: &str,
        error: Option<String>,
    ) -> Result<(), LabError> {
        let now = Utc::now();
        let result = self
            .update_job(control, |job| {
                job.status = status.into();
                job.progress = 1.0;
                job.message = message.into();
                job.started_at.get_or_insert(now);
                job.ended_at = Some(now);
                job.error = error;
                job.updated_at = now;
            })
            .await;
        if let Err(save_error) = &result {
            error!(error = %save_error.message(), "Failed to persist benchmark job terminal state");
        }
        result
    }

    async fn update_job<F>(&self, control: &Arc<JobControl>, update: F) -> Result<(), LabError>
    where
        F: FnOnce(&mut BenchmarkJobRecord),
    {
        let snapshot = {
            let job = control.record.read().await;
            let mut snapshot = job.clone();
            update(&mut snapshot);
            snapshot
        };
        let stored = snapshot.clone();
        self.store
            .run_critical(move |store| store.save_job(&stored))
            .await
            .map_err(LabError::from)?;
        *control.record.write().await = snapshot;
        Ok(())
    }

    async fn cancel_job(&self, id: &str) -> Result<BenchmarkJobRecord, LabError> {
        if let Some(control) = self.controls.lock().await.get(id).cloned() {
            let _ = control.cancel.send(true);
            let snapshot = control.record.read().await.clone();
            return Ok(snapshot);
        }
        let id = id.to_string();
        let job = self
            .store
            .run_blocking(move |store| store.benchmark_job(&id))
            .await
            .map_err(LabError::from)?;
        if matches!(job.status.as_str(), "queued" | "running") {
            return Err(LabError::Conflict(
                "job is no longer controlled by this daemon; restart recovery will mark it interrupted"
                    .into(),
            ));
        }
        Err(LabError::Conflict(format!(
            "job is already {} and cannot be cancelled",
            job.status
        )))
    }

    async fn jobs(&self, limit: u32) -> Result<Vec<BenchmarkJobRecord>, LabError> {
        self.store
            .run_blocking(move |store| store.benchmark_jobs(limit))
            .await
            .map_err(LabError::from)
    }

    async fn job(&self, id: &str) -> Result<BenchmarkJobRecord, LabError> {
        if let Some(control) = self.controls.lock().await.get(id).cloned() {
            return Ok(control.record.read().await.clone());
        }
        let id = id.to_string();
        self.store
            .run_blocking(move |store| store.benchmark_job(&id))
            .await
            .map_err(LabError::from)
    }

    fn resolve_case(&self, request: &StartJobRequest) -> Result<ResolvedCase, LabError> {
        match request.suite.as_str() {
            "riddllr" => resolve_riddllr_case(&self.riddllr, &request.case_id),
            "plumebench" => resolve_plumebench_case(&self.plumebench, &request.case_id),
            _ => Err(LabError::Validation(
                "suite must be riddllr or plumebench".into(),
            )),
        }
    }

    async fn run_riddllr(
        &self,
        control: &Arc<JobControl>,
        request: &StartJobRequest,
        resolved: &ResolvedCase,
        mut cancel: watch::Receiver<bool>,
    ) -> anyhow::Result<ExecutionResult> {
        let ResolvedCase::Riddllr {
            prompt,
            solution,
            prompt_name,
            solution_name,
            ..
        } = resolved
        else {
            return Err(anyhow!("invalid Riddllr case"));
        };
        let work_dir = PathBuf::from(
            control
                .record
                .read()
                .await
                .work_dir
                .clone()
                .context("job has no work directory")?,
        );
        tokio::fs::create_dir_all(&work_dir).await?;
        let output_path = work_dir.join("answer.txt");
        let log_path = work_dir.join("route.json");
        self.update_job(control, |job| {
            job.progress = 0.15;
            job.message = "Running reasoning prompt through Brainrouter".into();
            job.updated_at = Utc::now();
        })
        .await
        .map_err(|error| anyhow!(error.message().to_string()))?;

        let model = router_model_selector(&request.model);
        let completion_request = ChatCompletionRequest {
            model,
            messages: vec![
                ChatMessage {
                    role: "system".into(),
                    content: Some(Value::String(
                        "Solve the supplied benchmark problem independently. Follow its output format exactly and do not add commentary unless requested."
                            .into(),
                    )),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
                ChatMessage {
                    role: "user".into(),
                    content: Some(Value::String(prompt.clone())),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
            stream: Some(true),
            temperature: Some(0.0),
            max_tokens: Some(self.config.riddllr_max_tokens),
            top_p: Some(1.0),
            stop: None,
            extra: json!({}),
        };
        let deadline = Instant::now() + Duration::from_secs(self.config.max_job_seconds);
        let started = Instant::now();
        let route = tokio::select! {
            result = self.router.route_tagged(
                completion_request,
                Some(format!("benchmark:{}", control.record.read().await.id)),
                work_dir.to_string_lossy().into_owned(),
                "brainrouter-benchmark-lab/riddllr".into(),
            ) => result?,
            changed = cancel.changed() => {
                if changed.is_ok() && *cancel.borrow() {
                    return Ok(aborted_result(RunStatus::Cancelled, started.elapsed(), "Riddllr job cancelled"));
                }
                return Err(anyhow!("benchmark cancellation channel closed"));
            }
            _ = tokio::time::sleep_until(deadline) => {
                return Ok(aborted_result(RunStatus::Timeout, started.elapsed(), "Riddllr job timed out"));
            }
        };
        let (response, route_info) = route;
        let (answer, generated_tokens) =
            match collect_completion(response, deadline, &mut cancel).await {
                Ok(result) => result,
                Err(_) if *cancel.borrow() => {
                    return Ok(aborted_result(
                        RunStatus::Cancelled,
                        started.elapsed(),
                        "Riddllr job cancelled",
                    ))
                }
                Err(_) if Instant::now() >= deadline => {
                    return Ok(aborted_result(
                        RunStatus::Timeout,
                        started.elapsed(),
                        "Riddllr job timed out",
                    ))
                }
                Err(error) => return Err(error),
            };
        tokio::fs::write(&output_path, answer.as_bytes()).await?;
        tokio::fs::write(
            &log_path,
            serde_json::to_vec_pretty(&json!({
                "provider": route_info.effective_provider,
                "model": route_info.model_key,
                "decision": route_info.bonsai_decision,
                "failed_attempts": route_info.failed_attempts.iter().map(|attempt| json!({
                    "provider": attempt.provider,
                    "model": attempt.model_key,
                    "error": attempt.error,
                })).collect::<Vec<_>>(),
            }))?,
        )
        .await?;
        let grade = grade_riddllr_answer(&answer, solution);
        let duration = started.elapsed();
        let grading_details = json!({
            "grader": grade.grader,
            "expected": json_preview(&grade.expected, 64 * 1024),
            "actual": json_preview(&grade.actual, 64 * 1024),
            "errors": grade.errors,
            "prompt_file": prompt_name,
            "solution_file": solution_name,
        });
        Ok(ExecutionResult {
            run_status: RunStatus::Succeeded,
            exit_code: Some(0),
            passed: Some(grade.passed),
            duration_ms: duration.as_secs_f64() * 1000.0,
            stdout_path: Some(output_path.clone()),
            stderr_path: None,
            failure_reason: None,
            details: json!({
                "suite": "riddllr",
                "grade": grading_details,
                "route_log": log_path,
            }),
            quality: vec![QualityMeasurement {
                task_id: resolved.id().to_string(),
                metric_name: "pass@1".into(),
                metric_value: Some(if grade.passed { 1.0 } else { 0.0 }),
                passed: Some(grade.passed),
                compile_succeeded: None,
                tests_passed: None,
                tests_total: None,
                generated_tokens: Some(generated_tokens),
                duration_ms: Some(duration.as_secs_f64() * 1000.0),
                output_path: Some(output_path),
                log_path: Some(log_path),
                details: grading_details,
            }],
        })
    }

    async fn run_plumebench(
        &self,
        control: &Arc<JobControl>,
        request: &StartJobRequest,
        resolved: &ResolvedCase,
        cancel: watch::Receiver<bool>,
    ) -> anyhow::Result<ExecutionResult> {
        let ResolvedCase::Plumebench {
            task_dir,
            manifest_sha256,
            ..
        } = resolved
        else {
            return Err(anyhow!("invalid Plumebench case"));
        };
        let work_dir = PathBuf::from(
            control
                .record
                .read()
                .await
                .work_dir
                .clone()
                .context("job has no work directory")?,
        );
        tokio::fs::create_dir_all(&work_dir).await?;
        let job_name = work_dir
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or("benchmark");
        let source_snapshot = work_dir.with_file_name(format!("{job_name}-source"));
        tokio::fs::create_dir_all(&source_snapshot).await?;
        snapshot_plumebench_case(
            self.plumebench
                .path()
                .map_err(|error| anyhow!(error.message().to_string()))?,
            task_dir,
            &source_snapshot,
        )?;
        let snapshot_hash = hash_plumebench_snapshot(&source_snapshot)?;
        if &snapshot_hash != manifest_sha256 {
            return Err(anyhow!(
                "Plumebench task or grader changed after it was queued; submit a new job"
            ));
        }
        let snapshot_task = source_snapshot.join("task");
        copy_tree(&snapshot_task.join("starter"), &work_dir)?;
        let snapshot_prompt = read_limited_text(&snapshot_task.join("task.md"))
            .map_err(|error| anyhow!(error.message().to_string()))?;
        let prompt_path = work_dir.join(".brainrouter-task.md");
        tokio::fs::write(&prompt_path, snapshot_prompt.as_bytes()).await?;
        let stdout_path = work_dir.join("omp.jsonl");
        let stderr_path = work_dir.join("omp.stderr.log");
        let deadline = Instant::now() + Duration::from_secs(self.config.max_job_seconds);
        let started = Instant::now();

        ensure_omp_cli_compatible(&self.config.omp_bin).await?;
        let omp_profile = work_dir.with_file_name(format!("{job_name}-omp-profile"));
        let inference_socket = PathBuf::from("/tmp").join(format!(
            "brb-{}.sock",
            uuid::Uuid::new_v4().simple()
        ));
        let _inference_proxy = crate::inference_proxy::start(
            inference_socket.clone(),
            self.brainrouter_socket_path.clone(),
        )
        .await?;
        let inference_port = available_loopback_port()?;
        write_omp_profile(
            &omp_profile,
            &request.model,
            &format!("http://127.0.0.1:{inference_port}/v1"),
        )?;

        self.update_job(control, |job| {
            job.progress = 0.15;
            job.message = "Running OMP in the isolated Plumebench sandbox".into();
            job.updated_at = Utc::now();
        })
        .await
        .map_err(|error| anyhow!(error.message().to_string()))?;

        let mut omp = sandboxed_omp_command(
            &self.config.plumebench_sandbox_bin,
            &self.config.omp_bin,
            &work_dir,
            &omp_profile,
            &inference_socket,
            inference_port,
        )?;
        omp.arg("-p")
            .arg("--model")
            .arg(format!("brainrouter/{}", request.model))
            .arg("--mode")
            .arg("json")
            .arg("--max-time")
            .arg(self.config.max_job_seconds.to_string())
            .arg("--thinking")
            .arg(&self.config.plumebench_thinking)
            .arg("--auto-approve")
            .arg("--no-session")
            .arg("--no-extensions")
            .arg("--no-skills")
            .arg("--no-rules")
            .arg("--cwd")
            .arg("/work")
            .arg("@.brainrouter-task.md");
        let omp_result = run_command_with_turn_limit(
            omp,
            deadline,
            cancel.clone(),
            &stdout_path,
            &stderr_path,
            self.config.plumebench_max_turns,
        )
        .await?;
        match &omp_result.end {
            ProcessEnd::Cancelled => {
                return Ok(aborted_result(
                    RunStatus::Cancelled,
                    started.elapsed(),
                    "Plumebench job cancelled",
                ))
            }
            ProcessEnd::Timeout => {
                return Ok(aborted_result(
                    RunStatus::Timeout,
                    started.elapsed(),
                    "Plumebench OMP run timed out",
                ))
            }
            ProcessEnd::Completed(_) | ProcessEnd::TurnLimit => {}
        }
        let omp_status = match &omp_result.end {
            ProcessEnd::Completed(status) => Some(status),
            ProcessEnd::TurnLimit => None,
            _ => unreachable!(),
        };

        self.update_job(control, |job| {
            job.progress = 0.65;
            job.message = "Running hidden tests outside the model-visible workspace".into();
            job.updated_at = Utc::now();
        })
        .await
        .map_err(|error| anyhow!(error.message().to_string()))?;

        let stage_dir = work_dir.with_file_name(format!("{}-stage", job_name));
        tokio::fs::create_dir_all(&stage_dir).await?;
        copy_tree(&work_dir, &stage_dir)?;
        copy_tree(
            &snapshot_task.join("tests_hidden"),
            &stage_dir.join("tests_hidden"),
        )?;
        let test_stdout_path = work_dir.join("pytest.log");
        let test_stderr_path = work_dir.join("pytest.stderr.log");
        let mut pytest = sandboxed_python_command(
            &self.config.plumebench_sandbox_bin,
            &self.config.python_bin,
            &stage_dir,
            None,
        )?;
        pytest
            .arg("-m")
            .arg("pytest")
            .arg("-q")
            .arg("tests_hidden")
            .arg("--rootdir")
            .arg("/work")
            .env("PYTHONPATH", "/work");
        let test_result = run_command(
            pytest,
            deadline,
            cancel.clone(),
            &test_stdout_path,
            &test_stderr_path,
        )
        .await?;
        match &test_result.end {
            ProcessEnd::Cancelled => {
                return Ok(aborted_result(
                    RunStatus::Cancelled,
                    started.elapsed(),
                    "Plumebench job cancelled during hidden tests",
                ))
            }
            ProcessEnd::Timeout => {
                return Ok(aborted_result(
                    RunStatus::Timeout,
                    started.elapsed(),
                    "Plumebench hidden tests timed out",
                ))
            }
            ProcessEnd::Completed(_) => {}
            ProcessEnd::TurnLimit => unreachable!("pytest has no turn limit"),
        }
        let test_status = match &test_result.end {
            ProcessEnd::Completed(status) => status,
            ProcessEnd::TurnLimit => unreachable!("pytest has no turn limit"),
            _ => unreachable!(),
        };
        let test_summary = String::from_utf8_lossy(&test_result.stdout).into_owned();
        let test_summary_preview = text_preview(&test_summary, 64 * 1024);
        let (tests_passed, tests_total) = parse_pytest_counts(&test_summary);
        let passed = test_status.success();

        let mut elegance = None;
        let elegance_script = Some(source_snapshot.join("elegance.py"));
        if passed && elegance_script.as_ref().is_some_and(|path| path.is_file()) {
            self.update_job(control, |job| {
                job.progress = 0.85;
                job.message = "Collecting Plumebench elegance metrics".into();
                job.updated_at = Utc::now();
            })
            .await
            .map_err(|error| anyhow!(error.message().to_string()))?;
            let elegance_path = stage_dir.join("elegance.json");
            let elegance_stdout = work_dir.join("elegance.log");
            let elegance_stderr = work_dir.join("elegance.stderr.log");
            let mut command = sandboxed_python_command(
                &self.config.plumebench_sandbox_bin,
                &self.config.python_bin,
                &stage_dir,
                Some(&source_snapshot),
            )?;
            command
                .arg("/suite/elegance.py")
                .arg("scan")
                .arg("/suite/task")
                .arg("/work")
                .arg("--out")
                .arg("/work/elegance.json")
                .arg("--tag")
                .arg(control.record.read().await.id.clone());
            match run_command(
                command,
                deadline,
                cancel.clone(),
                &elegance_stdout,
                &elegance_stderr,
            )
            .await
            {
                Ok(result) if matches!(result.end, ProcessEnd::Completed(status) if status.success()) =>
                {
                    elegance = read_elegance_metrics(
                        &elegance_path,
                        resolved.id(),
                        &control.record.read().await.id,
                    )
                    .ok();
                }
                Ok(result) => {
                    let job_id = control.record.read().await.id.clone();
                    warn!(job_id, end = ?result.end, "Elegance scan did not complete successfully");
                }
                Err(error) => {
                    let job_id = control.record.read().await.id.clone();
                    warn!(job_id, error = %error, "Elegance scan failed");
                }
            }
        }

        let duration = started.elapsed();
        let transcript = parse_omp_transcript(&omp_result.stdout);
        let mut quality = vec![QualityMeasurement {
            task_id: resolved.id().to_string(),
            metric_name: "pass@1".into(),
            metric_value: Some(if passed { 1.0 } else { 0.0 }),
            passed: Some(passed),
            compile_succeeded: Some(omp_status.is_some_and(ExitStatus::success)),
            tests_passed,
            tests_total,
            generated_tokens: transcript.generated_tokens,
            duration_ms: Some(duration.as_secs_f64() * 1000.0),
            output_path: Some(stage_dir.clone()),
            log_path: Some(test_stdout_path.clone()),
            details: json!({
                "pytest_summary": test_summary_preview,
                "omp_exit_code": omp_status.and_then(ExitStatus::code),
                "omp": transcript,
            }),
        }];
        if let Some(metrics) = elegance.as_ref().and_then(Value::as_object) {
            for (name, value) in metrics {
                if let Some(metric_value) = value.as_f64() {
                    quality.push(QualityMeasurement {
                        task_id: resolved.id().to_string(),
                        metric_name: format!("elegance.{name}"),
                        metric_value: Some(metric_value),
                        passed: None,
                        compile_succeeded: None,
                        tests_passed: None,
                        tests_total: None,
                        generated_tokens: None,
                        duration_ms: None,
                        output_path: Some(stage_dir.clone()),
                        log_path: None,
                        details: json!({}),
                    });
                }
            }
        }
        let harness_ok = omp_status.is_some_and(ExitStatus::success);
        let turn_limited = matches!(omp_result.end, ProcessEnd::TurnLimit);
        Ok(ExecutionResult {
            run_status: if harness_ok {
                RunStatus::Succeeded
            } else {
                RunStatus::Failed
            },
            exit_code: omp_status.and_then(ExitStatus::code).map(i64::from),
            passed: Some(passed),
            duration_ms: duration.as_secs_f64() * 1000.0,
            stdout_path: Some(stdout_path),
            stderr_path: Some(stderr_path),
            failure_reason: (!harness_ok).then(|| {
                if turn_limited {
                    format!(
                        "OMP exceeded the configured {}-turn limit",
                        self.config.plumebench_max_turns
                    )
                } else {
                    format!(
                        "OMP exited with status {}",
                        omp_status
                            .and_then(ExitStatus::code)
                            .map_or_else(|| "signal".into(), |code| code.to_string())
                    )
                }
            }),
            details: json!({
                "suite": "plumebench",
                "passed": passed,
                "tests_passed": tests_passed,
                "tests_total": tests_total,
                "omp": transcript,
                "elegance": elegance,
                "stage_dir": stage_dir,
            }),
            quality,
        })
    }

    fn build_bundle(
        &self,
        job: &BenchmarkJobRecord,
        request: &StartJobRequest,
        resolved: &ResolvedCase,
        started_at: DateTime<Utc>,
        ended_at: DateTime<Utc>,
        result: &ExecutionResult,
    ) -> anyhow::Result<IngestBundle> {
        let run_id = job.run_id.clone().context("job has no run id")?;
        let model_slug = slug(&request.model);
        let model_hash = sha256_text(&format!("benchmark-model:{}", request.model));
        let model_id = format!("model-{model_slug}-{}", &model_hash[..12]);
        let artifact_id = format!("artifact-{model_slug}-{}", &model_hash[12..24]);
        let workload_id = format!(
            "workload-{}-{}-{}",
            resolved.suite(),
            slug(resolved.id()),
            &resolved.manifest_sha256()[..12]
        );
        let prompt_tokens = ((resolved.prompt_bytes() as u64).saturating_add(3) / 4).max(1);
        let generation_tokens = match resolved {
            ResolvedCase::Riddllr { .. } => u64::from(self.config.riddllr_max_tokens),
            ResolvedCase::Plumebench { .. } => 0,
        };
        let context_tokens = 131_072_u64.max(prompt_tokens.saturating_add(generation_tokens));
        let mut model_metadata = BTreeMap::new();
        model_metadata.insert("served_model_key".into(), json!(request.model));
        model_metadata.insert(
            "declaration_note".into(),
            json!("Runtime-served model metadata; artifact properties are not file attestation."),
        );
        let model = ModelDefinition {
            id: model_id.clone(),
            family: model_family(&request.model),
            architecture: "unknown".into(),
            checkpoint: request.model.clone(),
            revision: "runtime-served".into(),
            tokenizer_id: format!("runtime:{}", request.model),
            tokenizer_revision: None,
            parameter_count_total: None,
            parameter_count_active: None,
            model_kind: ModelKind::Unknown,
            native_context_tokens: None,
            metadata: model_metadata,
        };
        let mut artifact_metadata = BTreeMap::new();
        artifact_metadata.insert(
            "declaration_note".into(),
            json!("Logical served artifact identity; disk size and source file are unknown."),
        );
        let artifact = ArtifactDefinition {
            id: artifact_id.clone(),
            model_id: model_id.clone(),
            format: "runtime-served".into(),
            quant_family: "unknown".into(),
            quant_name: inferred_quantization(&request.model),
            average_bits_per_weight: None,
            disk_bytes: 0,
            sha256: model_hash,
            source_uri: Some(format!("brainrouter://model/{}", request.model)),
            imatrix_used: false,
            conversion_tool: None,
            conversion_commit: None,
            conversion_command: None,
            metadata: artifact_metadata,
        };
        let mut workload_metadata = BTreeMap::new();
        workload_metadata.insert("suite".into(), json!(resolved.suite()));
        workload_metadata.insert("case_id".into(), json!(resolved.id()));
        workload_metadata.insert(
            "source_root".into(),
            json!(match resolved {
                ResolvedCase::Riddllr { .. } => self.riddllr.configured.as_ref(),
                ResolvedCase::Plumebench { .. } => self.plumebench.configured.as_ref(),
            }),
        );
        let workload = WorkloadDefinition {
            id: workload_id.clone(),
            name: resolved.workload_name(),
            version: resolved.manifest_sha256()[..12].to_string(),
            workload_type: WorkloadType::CodeQuality,
            manifest_sha256: resolved.manifest_sha256().to_string(),
            input_tokens: Some(prompt_tokens),
            output_tokens: match resolved {
                ResolvedCase::Riddllr { .. } => Some(generation_tokens),
                ResolvedCase::Plumebench { .. } => None,
            },
            corpus_bytes: Some(resolved.prompt_bytes() as u64),
            license: None,
            metadata: workload_metadata,
        };
        let mut experiment = ExperimentConfig {
            id: "pending-benchmark-lab-experiment".into(),
            artifact_id,
            runtime_id: self.runtime.id.clone(),
            hardware_id: self.hardware.id.clone(),
            workload_id,
            context_tokens,
            prompt_tokens,
            generation_tokens,
            batch_size: 1,
            micro_batch_size: 1,
            threads: None,
            optimization: OptimizationConfig {
                flash_attention: FeatureState::Unsupported,
                ..OptimizationConfig::default()
            },
            sampling: SamplingConfig::default(),
            command_template: format!(
                "brainrouter benchmark-lab --suite {} --case {{case}} --model {{model}}",
                resolved.suite()
            ),
        };
        let experiment_hash = experiment
            .experiment_hash()
            .map_err(|error| anyhow!(error.to_string()))?;
        let experiment_id = format!("experiment-{}", &experiment_hash[..32]);
        experiment.id = experiment_id.clone();
        let mut raw_result = BTreeMap::new();
        raw_result.insert("benchmark_lab_job_id".into(), json!(job.id));
        raw_result.insert("passed".into(), json!(result.passed));
        raw_result.insert("details".into(), result.details.clone());
        let run = RunRecord {
            id: run_id.clone(),
            experiment_id,
            repetition: request.repetition,
            status: result.run_status,
            started_at: Some(started_at),
            ended_at: Some(ended_at),
            exit_code: result.exit_code,
            random_seed: Some(0),
            warmup_count: 0,
            exact_command: exact_command(request, resolved),
            cwd: job.work_dir.clone(),
            environment: BTreeMap::new(),
            stdout_path: result
                .stdout_path
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned()),
            stderr_path: result
                .stderr_path
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned()),
            failure_reason: result.failure_reason.clone(),
            raw_result: Some(raw_result),
        };
        let quality_results = result
            .quality
            .iter()
            .map(|measurement| QualityResult {
                id: format!(
                    "quality-{}",
                    &sha256_text(&format!(
                        "{}:{}:{}",
                        run_id, measurement.task_id, measurement.metric_name
                    ))[..32]
                ),
                run_id: run_id.clone(),
                task_id: measurement.task_id.clone(),
                metric_name: measurement.metric_name.clone(),
                metric_value: measurement.metric_value,
                passed: measurement.passed,
                compile_succeeded: measurement.compile_succeeded,
                tests_passed: measurement.tests_passed,
                tests_total: measurement.tests_total,
                generated_tokens: measurement.generated_tokens,
                duration_ms: measurement.duration_ms,
                output_path: measurement
                    .output_path
                    .as_ref()
                    .map(|path| path.to_string_lossy().into_owned()),
                log_path: measurement
                    .log_path
                    .as_ref()
                    .map(|path| path.to_string_lossy().into_owned()),
                details: value_object(measurement.details.clone()),
            })
            .collect();
        Ok(IngestBundle {
            model,
            artifact,
            runtime: self.runtime.clone(),
            hardware: self.hardware.clone(),
            workload,
            experiment,
            run,
            performance_metrics: Some(PerformanceMetrics {
                run_id,
                model_load_ms: None,
                prompt_processing_ms: None,
                prompt_tps: None,
                ttft_ms: None,
                generation_ms: Some(result.duration_ms),
                generation_tps: None,
                inter_token_p50_ms: None,
                inter_token_p95_ms: None,
                inter_token_p99_ms: None,
                peak_rss_bytes: None,
                peak_vram_bytes: None,
                kv_cache_bytes: None,
                energy_joules: None,
                avg_power_watts: None,
            }),
            speculative_metrics: None,
            quality_results,
            telemetry_samples: Vec::new(),
        })
    }
}

pub async fn handle_request(
    req: Request<Incoming>,
    lab: Arc<BenchmarkLab>,
) -> Result<HttpResponse, Infallible> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let query = req.uri().query().map(str::to_string);
    let response = match dispatch_http(req, lab, method, path, query).await {
        Ok(response) => response,
        Err(error) => error_response(error),
    };
    Ok(response)
}

pub fn unavailable_response(reason: &str) -> HttpResponse {
    json_response(
        StatusCode::SERVICE_UNAVAILABLE,
        &json!({"error": format!("Benchmark Lab unavailable: {reason}")}),
    )
}

async fn dispatch_http(
    req: Request<Incoming>,
    lab: Arc<BenchmarkLab>,
    method: Method,
    path: String,
    query: Option<String>,
) -> Result<HttpResponse, LabError> {
    match (method, path.as_str()) {
        (Method::GET, "/api/benchmarks/lab/suites") => {
            let suites = lab.suites();
            Ok(json_response(StatusCode::OK, &json!({"suites": suites})))
        }
        (Method::GET, "/api/benchmarks/lab/jobs") => {
            let limit = query
                .as_deref()
                .and_then(|query| {
                    url::form_urlencoded::parse(query.as_bytes())
                        .find(|(key, _)| key == "limit")
                        .and_then(|(_, value)| value.parse::<u32>().ok())
                })
                .unwrap_or(25)
                .min(MAX_JOB_HISTORY);
            let jobs = lab.jobs(limit).await?;
            Ok(json_response(StatusCode::OK, &json!({"jobs": jobs})))
        }
        (Method::GET, path) if path.starts_with("/api/benchmarks/lab/jobs/") => {
            let id = parse_job_path(path, "")?;
            Ok(json_response(StatusCode::OK, &lab.job(&id).await?))
        }
        (Method::POST, "/api/benchmarks/lab/jobs") => {
            let bytes = collect_request_body(req).await?;
            let request: StartJobRequest = serde_json::from_slice(&bytes).map_err(|error| {
                LabError::Validation(format!("invalid benchmark job payload: {error}"))
            })?;
            let job = lab.start_job(request).await?;
            Ok(json_response(StatusCode::ACCEPTED, &job))
        }
        (Method::POST, path)
            if path.starts_with("/api/benchmarks/lab/jobs/") && path.ends_with("/cancel") =>
        {
            let id = parse_job_path(path, "/cancel")?;
            let job = lab.cancel_job(&id).await?;
            Ok(json_response(StatusCode::ACCEPTED, &job))
        }
        _ => Err(LabError::NotFound("Benchmark Lab route not found".into())),
    }
}

fn parse_job_path(path: &str, suffix: &str) -> Result<String, LabError> {
    let id = path
        .strip_prefix("/api/benchmarks/lab/jobs/")
        .and_then(|value| value.strip_suffix(suffix))
        .unwrap_or_default();
    if id.is_empty()
        || id.len() > 128
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
    {
        return Err(LabError::Validation("invalid benchmark job id".into()));
    }
    Ok(id.to_string())
}

async fn collect_request_body(req: Request<Incoming>) -> Result<Bytes, LabError> {
    let limited = Limited::new(req.into_body(), MAX_LAB_REQUEST_BYTES);
    limited
        .collect()
        .await
        .map(|body| body.to_bytes())
        .map_err(|error| {
            if error.downcast_ref::<LengthLimitError>().is_some() {
                LabError::Limit(format!(
                    "benchmark job request exceeds {MAX_LAB_REQUEST_BYTES} bytes"
                ))
            } else {
                LabError::Validation(format!("invalid benchmark job request body: {error}"))
            }
        })
}

fn json_response<T: Serialize>(status: StatusCode, body: &T) -> HttpResponse {
    let bytes = serde_json::to_vec(body)
        .unwrap_or_else(|_| br#"{"error":"internal serialization error"}"#.to_vec());
    Response::builder()
        .status(status)
        .header("content-type", "application/json; charset=utf-8")
        .header("cache-control", "no-store")
        .body(
            Full::new(Bytes::from(bytes))
                .map_err(|error: Infallible| match error {})
                .boxed_unsync(),
        )
        .expect("benchmark lab JSON response")
}

fn error_response(error: LabError) -> HttpResponse {
    if matches!(error, LabError::Internal(_)) {
        error!(error = error.message(), "Benchmark Lab request failed");
    }
    json_response(error.status(), &json!({"error": error.message()}))
}

fn discover_riddllr_cases(root: &SuiteRoot) -> Result<Vec<SuiteCase>, LabError> {
    let root = root.path()?;
    let prompts = root.join("prompts");
    let solutions = root.join("solutions");
    ensure_directory_within(root, &prompts)?;
    ensure_directory_within(root, &solutions)?;
    let mut cases = Vec::new();
    for entry in sorted_entries(&prompts)? {
        let path = entry.path();
        if entry
            .file_type()
            .map_err(|error| LabError::Internal(error.to_string()))?
            .is_symlink()
            || !path.is_file()
            || path.extension() != Some(OsStr::new("txt"))
        {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(OsStr::to_str) else {
            continue;
        };
        if !valid_case_id(stem) {
            continue;
        }
        let solution = solutions.join(format!("{stem}-solution.txt"));
        if !solution.is_file() {
            continue;
        }
        let (_, _, manifest_sha256) = read_riddllr_pair(root, stem)?;
        cases.push(SuiteCase {
            id: stem.to_string(),
            name: stem.replace(['_', '-'], " "),
            manifest_sha256,
        });
    }
    Ok(cases)
}

fn discover_plumebench_cases(root: &SuiteRoot) -> Result<Vec<SuiteCase>, LabError> {
    let root = root.path()?;
    let tasks = root.join("tasks");
    ensure_directory_within(root, &tasks)?;
    let mut cases = Vec::new();
    for entry in sorted_entries(&tasks)? {
        let file_type = entry
            .file_type()
            .map_err(|error| LabError::Internal(error.to_string()))?;
        if file_type.is_symlink() || !file_type.is_dir() {
            continue;
        }
        let Some(id) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if !valid_case_id(&id) {
            continue;
        }
        let task_dir = entry.path();
        if ["task.md", "starter", "tests_hidden", "reference"]
            .iter()
            .all(|name| task_dir.join(name).exists())
        {
            cases.push(SuiteCase {
                id: id.clone(),
                name: id.replace(['_', '-'], " "),
                manifest_sha256: hash_plumebench_case(root, &task_dir)
                    .map_err(|error| LabError::Validation(error.to_string()))?,
            });
        }
    }
    Ok(cases)
}

fn resolve_riddllr_case(root: &SuiteRoot, id: &str) -> Result<ResolvedCase, LabError> {
    if !valid_case_id(id) {
        return Err(LabError::Validation("invalid Riddllr case id".into()));
    }
    let root_path = root.path()?;
    let (prompt, solution, manifest_sha256) = read_riddllr_pair(root_path, id)?;
    Ok(ResolvedCase::Riddllr {
        id: id.to_string(),
        prompt,
        solution,
        manifest_sha256,
        prompt_name: format!("{id}.txt"),
        solution_name: format!("{id}-solution.txt"),
    })
}

fn resolve_plumebench_case(root: &SuiteRoot, id: &str) -> Result<ResolvedCase, LabError> {
    if !valid_case_id(id) {
        return Err(LabError::Validation("invalid Plumebench case id".into()));
    }
    let root_path = root.path()?;
    let task_dir = root_path.join("tasks").join(id);
    ensure_directory_within(root_path, &task_dir)?;
    for name in ["starter", "tests_hidden", "reference"] {
        ensure_directory_within(root_path, &task_dir.join(name))?;
    }
    let prompt_path = task_dir.join("task.md");
    ensure_file_within(root_path, &prompt_path)?;
    let prompt = read_limited_text(&prompt_path)?;
    let manifest_sha256 = hash_plumebench_case(root_path, &task_dir)
        .map_err(|error| LabError::Validation(error.to_string()))?;
    Ok(ResolvedCase::Plumebench {
        id: id.to_string(),
        task_dir,
        prompt,
        manifest_sha256,
    })
}

fn read_riddllr_pair(root: &Path, id: &str) -> Result<(String, String, String), LabError> {
    let prompt_path = root.join("prompts").join(format!("{id}.txt"));
    let solution_path = root.join("solutions").join(format!("{id}-solution.txt"));
    ensure_file_within(root, &prompt_path)?;
    ensure_file_within(root, &solution_path)?;
    let prompt = read_limited_text(&prompt_path)?;
    let solution = read_limited_text(&solution_path)?;
    let manifest_sha256 = sha256_text(&format!("prompt\0{}\0solution\0{}", prompt, solution));
    Ok((prompt, solution, manifest_sha256))
}

fn hash_plumebench_case(root: &Path, task_dir: &Path) -> anyhow::Result<String> {
    let task_hash = hash_tree(task_dir)?;
    let elegance_path = root.join("elegance.py");
    let elegance_hash = if elegance_path.exists() {
        ensure_file_within(root, &elegance_path)
            .map_err(|error| anyhow!(error.message().to_string()))?;
        sha256_bytes(&fs::read(elegance_path)?)
    } else {
        "absent".to_string()
    };
    Ok(sha256_text(&format!(
        "task\0{task_hash}\0elegance\0{elegance_hash}"
    )))
}

fn snapshot_plumebench_case(
    root: &Path,
    task_dir: &Path,
    destination: &Path,
) -> anyhow::Result<()> {
    copy_tree(task_dir, &destination.join("task"))?;
    let elegance = root.join("elegance.py");
    if elegance.exists() {
        ensure_file_within(root, &elegance)
            .map_err(|error| anyhow!(error.message().to_string()))?;
        fs::copy(&elegance, destination.join("elegance.py")).with_context(|| {
            format!(
                "failed to snapshot Plumebench grader {}",
                elegance.display()
            )
        })?;
    }
    Ok(())
}

fn hash_plumebench_snapshot(snapshot: &Path) -> anyhow::Result<String> {
    let task_hash = hash_tree(&snapshot.join("task"))?;
    let elegance = snapshot.join("elegance.py");
    let elegance_hash = if elegance.exists() {
        let metadata = fs::symlink_metadata(&elegance)?;
        if metadata.file_type().is_symlink()
            || !metadata.is_file()
            || metadata.len() > MAX_SUITE_FILE_BYTES
        {
            return Err(anyhow!("invalid snapshotted Plumebench grader"));
        }
        sha256_bytes(&fs::read(elegance)?)
    } else {
        "absent".to_string()
    };
    Ok(sha256_text(&format!(
        "task\0{task_hash}\0elegance\0{elegance_hash}"
    )))
}

fn ensure_directory_within(root: &Path, path: &Path) -> Result<(), LabError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        LabError::Validation(format!("{} is unavailable: {error}", path.display()))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(LabError::Validation(format!(
            "{} must be a regular, non-symlink directory",
            path.display()
        )));
    }
    let canonical = fs::canonicalize(path).map_err(|error| {
        LabError::Validation(format!("{} is unavailable: {error}", path.display()))
    })?;
    if !canonical.starts_with(root) || !canonical.is_dir() {
        return Err(LabError::Validation(format!(
            "{} is not a directory within the configured suite root",
            path.display()
        )));
    }
    Ok(())
}

fn ensure_file_within(root: &Path, path: &Path) -> Result<(), LabError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        LabError::Validation(format!("{} is unavailable: {error}", path.display()))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(LabError::Validation(format!(
            "{} must be a regular, non-symlink file",
            path.display()
        )));
    }
    let canonical = fs::canonicalize(path).map_err(|error| {
        LabError::Validation(format!("{} cannot be resolved: {error}", path.display()))
    })?;
    if !canonical.starts_with(root) {
        return Err(LabError::Validation(format!(
            "{} escapes the configured suite root",
            path.display()
        )));
    }
    if metadata.len() > MAX_SUITE_FILE_BYTES {
        return Err(LabError::Validation(format!(
            "{} exceeds the {} byte suite file limit",
            path.display(),
            MAX_SUITE_FILE_BYTES
        )));
    }
    Ok(())
}

fn read_limited_text(path: &Path) -> Result<String, LabError> {
    let bytes = fs::read(path).map_err(|error| {
        LabError::Internal(format!("failed to read {}: {error}", path.display()))
    })?;
    if bytes.len() as u64 > MAX_SUITE_FILE_BYTES {
        return Err(LabError::Validation(format!(
            "{} exceeds the suite file limit",
            path.display()
        )));
    }
    String::from_utf8(bytes).map_err(|error| {
        LabError::Validation(format!("{} is not valid UTF-8: {error}", path.display()))
    })
}

fn sorted_entries(path: &Path) -> Result<Vec<fs::DirEntry>, LabError> {
    let mut entries = fs::read_dir(path)
        .map_err(|error| LabError::Internal(format!("failed to read {}: {error}", path.display())))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| LabError::Internal(error.to_string()))?;
    entries.sort_by_key(|entry| entry.file_name());
    Ok(entries)
}

fn hash_tree(root: &Path) -> anyhow::Result<String> {
    let canonical_root =
        fs::canonicalize(root).with_context(|| format!("failed to resolve {}", root.display()))?;
    if !canonical_root.is_dir() {
        return Err(anyhow!("{} is not a directory", root.display()));
    }
    let mut files = Vec::new();
    collect_tree_files(&canonical_root, &canonical_root, &mut files)?;
    files.sort();
    let mut total = 0_u64;
    let mut hasher = Sha256::new();
    for relative in files {
        let path = canonical_root.join(&relative);
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            return Err(anyhow!("suite tree contains symlink {}", path.display()));
        }
        total = total.saturating_add(metadata.len());
        if metadata.len() > MAX_SUITE_FILE_BYTES || total > MAX_SUITE_TREE_BYTES {
            return Err(anyhow!("suite tree exceeds configured size limits"));
        }
        hasher.update(relative.to_string_lossy().as_bytes());
        hasher.update([0]);
        hasher.update(fs::read(path)?);
        hasher.update([0]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn collect_tree_files(root: &Path, path: &Path, files: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let metadata = fs::symlink_metadata(entry.path())?;
        if metadata.file_type().is_symlink() {
            return Err(anyhow!(
                "suite tree contains symlink {}",
                entry.path().display()
            ));
        }
        if metadata.is_dir() {
            collect_tree_files(root, &entry.path(), files)?;
        } else if metadata.is_file() {
            files.push(entry.path().strip_prefix(root)?.to_path_buf());
        }
    }
    Ok(())
}

fn copy_tree(source: &Path, destination: &Path) -> anyhow::Result<()> {
    let source = fs::canonicalize(source)
        .with_context(|| format!("failed to resolve {}", source.display()))?;
    if !source.is_dir() {
        return Err(anyhow!("{} is not a directory", source.display()));
    }
    fs::create_dir_all(destination)?;
    let mut total = 0_u64;
    copy_tree_inner(&source, &source, destination, &mut total)
}

fn copy_tree_inner(
    root: &Path,
    source: &Path,
    destination: &Path,
    total: &mut u64,
) -> anyhow::Result<()> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let metadata = fs::symlink_metadata(entry.path())?;
        if metadata.file_type().is_symlink() {
            return Err(anyhow!(
                "refusing to copy symlink from benchmark input: {}",
                entry.path().display()
            ));
        }
        let relative = entry.path().strip_prefix(root)?.to_path_buf();
        let target = destination.join(relative);
        if metadata.is_dir() {
            fs::create_dir_all(&target)?;
            copy_tree_inner(root, &entry.path(), destination, total)?;
        } else if metadata.is_file() {
            *total = total.saturating_add(metadata.len());
            if metadata.len() > MAX_SUITE_FILE_BYTES || *total > MAX_SUITE_TREE_BYTES {
                return Err(anyhow!("benchmark workspace exceeds copy size limits"));
            }
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(entry.path(), target)?;
        } else {
            return Err(anyhow!(
                "refusing to copy unsupported benchmark file type: {}",
                entry.path().display()
            ));
        }
    }
    Ok(())
}

fn paths_overlap(left: &Path, right: &Path) -> bool {
    left.starts_with(right) || right.starts_with(left)
}

async fn collect_completion(
    response: ProviderResponse,
    deadline: Instant,
    cancel: &mut watch::Receiver<bool>,
) -> anyhow::Result<(String, u64)> {
    let ProviderResponse::Stream(mut stream) = response;
    let mut buffer = Vec::new();
    let mut output = String::new();
    let mut completion_tokens = None;
    loop {
        let next = tokio::select! {
            chunk = stream.next() => chunk,
            changed = cancel.changed() => {
                if changed.is_ok() && *cancel.borrow() {
                    return Err(anyhow!("benchmark job cancelled"));
                }
                return Err(anyhow!("benchmark cancellation channel closed"));
            }
            _ = tokio::time::sleep_until(deadline) => {
                return Err(anyhow!("benchmark generation timed out"));
            }
        };
        let Some(chunk) = next else {
            break;
        };
        buffer.extend_from_slice(&chunk?);
        if buffer.len() > MAX_SSE_BUFFER_BYTES {
            return Err(anyhow!("benchmark SSE frame exceeded the buffer limit"));
        }
        while let Some(position) = buffer.iter().position(|byte| *byte == b'\n') {
            let line = buffer.drain(..=position).collect::<Vec<_>>();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim();
            let Some(data) = line
                .strip_prefix("data: ")
                .or_else(|| line.strip_prefix("data:"))
            else {
                continue;
            };
            if data.trim() == "[DONE]" {
                let tokens = completion_tokens.unwrap_or_else(|| count_tokens(&output));
                return Ok((output, tokens));
            }
            let Ok(value) = serde_json::from_str::<Value>(data.trim()) else {
                continue;
            };
            if let Some(error) = value.get("error") {
                return Err(anyhow!("provider returned an SSE error: {error}"));
            }
            if let Some(content) = value
                .pointer("/choices/0/delta/content")
                .and_then(Value::as_str)
            {
                output.push_str(content);
                if output.len() > MAX_CAPTURE_BYTES {
                    return Err(anyhow!("benchmark completion exceeded the output limit"));
                }
            }
            if let Some(tokens) = value
                .pointer("/usage/completion_tokens")
                .and_then(Value::as_u64)
            {
                completion_tokens = Some(tokens);
            }
        }
    }
    let tokens = completion_tokens.unwrap_or_else(|| count_tokens(&output));
    Ok((output, tokens))
}

async fn ensure_omp_cli_compatible(binary: &Path) -> anyhow::Result<()> {
    let binary = resolve_executable(binary)?;
    let mut command = Command::new(&binary);
    command
        .arg("--help")
        .kill_on_drop(true)
        .env_clear()
        .env("PATH", "/usr/bin:/bin");
    let output = tokio::time::timeout(Duration::from_secs(5), command.output())
        .await
        .context("timed out checking the configured OMP CLI")?
        .with_context(|| format!("failed to run {} --help", binary.display()))?;
    if !output.status.success() {
        return Err(anyhow!(
            "{} --help exited with status {}",
            binary.display(),
            output.status
        ));
    }
    let help = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let missing = [
        "--model",
        "--mode",
        "--max-time",
        "--thinking",
        "--auto-approve",
        "--no-session",
        "--no-extensions",
        "--no-skills",
        "--no-rules",
        "--cwd",
    ]
    .into_iter()
    .filter(|flag| !help.contains(flag))
    .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(anyhow!(
            "configured OMP CLI is incompatible with Benchmark Lab; missing flags: {}",
            missing.join(", ")
        ));
    }
    Ok(())
}

fn write_omp_profile(profile_dir: &Path, model: &str, base_url: &str) -> anyhow::Result<()> {
    fs::create_dir_all(profile_dir)?;
    let models = json!({
        "providers": {
            "brainrouter": {
                "baseUrl": base_url,
                "api": "openai-completions",
                "auth": "none",
                "models": [{
                    "id": model,
                    "name": format!("Brainrouter benchmark: {model}"),
                    "reasoning": false,
                    "input": ["text"],
                }],
            },
        },
    });
    fs::write(
        profile_dir.join("models.yml"),
        serde_yaml::to_string(&models)?,
    )?;
    Ok(())
}

fn sandboxed_omp_command(
    sandbox_binary: &Path,
    omp_binary: &Path,
    work_dir: &Path,
    profile_dir: &Path,
    inference_socket: &Path,
    inference_port: u16,
) -> anyhow::Result<Command> {
    let sandbox_binary = resolve_executable(sandbox_binary)?;
    let omp_binary = resolve_executable(omp_binary)?;
    let brainrouter_binary = fs::canonicalize(std::env::current_exe()?)
        .context("failed to resolve the running brainrouter executable")?;
    let work_dir = fs::canonicalize(work_dir)
        .with_context(|| format!("failed to resolve {}", work_dir.display()))?;
    let profile_dir = fs::canonicalize(profile_dir)
        .with_context(|| format!("failed to resolve {}", profile_dir.display()))?;
    let inference_socket = fs::canonicalize(inference_socket)
        .with_context(|| format!("failed to resolve {}", inference_socket.display()))?;

    let mut command = Command::new(sandbox_binary);
    command
        .env_clear()
        .current_dir(&work_dir)
        .arg("--die-with-parent")
        .arg("--unshare-all")
        .arg("--new-session")
        .arg("--proc")
        .arg("/proc")
        .arg("--dev")
        .arg("/dev")
        .arg("--tmpfs")
        .arg("/tmp")
        .arg("--dir")
        .arg("/opt")
        .arg("--dir")
        .arg("/run")
        .arg("--dir")
        .arg("/etc")
        .arg("--dir")
        .arg("/home")
        .arg("--dir")
        .arg("/home/benchmark")
        .arg("--dir")
        .arg("/home/benchmark/.config")
        .arg("--dir")
        .arg("/home/benchmark/.local")
        .arg("--dir")
        .arg("/home/benchmark/.local/share")
        .arg("--ro-bind")
        .arg("/usr")
        .arg("/usr");
    for path in ["/bin", "/lib", "/lib64"] {
        add_bwrap_system_path(&mut command, Path::new(path))?;
    }
    for path in [
        "/etc/ssl",
        "/etc/pki",
        "/etc/ca-certificates",
        "/etc/resolv.conf",
        "/etc/hosts",
        "/etc/nsswitch.conf",
        "/etc/passwd",
        "/etc/group",
        "/etc/localtime",
        "/etc/os-release",
        "/etc/protocols",
        "/etc/services",
    ] {
        add_bwrap_etc_path(&mut command, Path::new(path))?;
    }
    command
        .arg("--ro-bind")
        .arg(omp_binary)
        .arg("/opt/omp")
        .arg("--ro-bind")
        .arg(brainrouter_binary)
        .arg("/opt/brainrouter")
        .arg("--ro-bind")
        .arg(inference_socket)
        .arg("/run/brainrouter-inference.sock")
        .arg("--bind")
        .arg(work_dir)
        .arg("/work")
        .arg("--bind")
        .arg(profile_dir)
        .arg("/profile")
        .arg("--chdir")
        .arg("/work")
        .arg("--setenv")
        .arg("HOME")
        .arg("/home/benchmark")
        .arg("--setenv")
        .arg("USER")
        .arg("benchmark")
        .arg("--setenv")
        .arg("LOGNAME")
        .arg("benchmark")
        .arg("--setenv")
        .arg("PATH")
        .arg("/usr/bin:/bin:/opt")
        .arg("--setenv")
        .arg("PI_CODING_AGENT_DIR")
        .arg("/profile")
        .arg("--setenv")
        .arg("XDG_CONFIG_HOME")
        .arg("/home/benchmark/.config")
        .arg("--setenv")
        .arg("XDG_DATA_HOME")
        .arg("/home/benchmark/.local/share")
        .arg("--setenv")
        .arg("XDG_CACHE_HOME")
        .arg("/tmp")
        .arg("--setenv")
        .arg("NO_PROXY")
        .arg("127.0.0.1,localhost")
        .arg("--")
        .arg("/opt/brainrouter")
        .arg("benchmark-sandbox")
        .arg("--socket")
        .arg("/run/brainrouter-inference.sock")
        .arg("--listen")
        .arg(format!("127.0.0.1:{inference_port}"))
        .arg("--")
        .arg("/opt/omp");
    Ok(command)
}

fn available_loopback_port() -> anyhow::Result<u16> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))
        .context("failed to reserve a benchmark inference proxy port")?;
    Ok(listener.local_addr()?.port())
}

fn sandboxed_python_command(
    sandbox_binary: &Path,
    python_binary: &Path,
    work_dir: &Path,
    suite_dir: Option<&Path>,
) -> anyhow::Result<Command> {
    let sandbox_binary = resolve_executable(sandbox_binary)?;
    let python_binary = resolve_executable(python_binary)?;
    let work_dir = fs::canonicalize(work_dir)
        .with_context(|| format!("failed to resolve {}", work_dir.display()))?;
    let suite_dir = suite_dir
        .map(fs::canonicalize)
        .transpose()
        .context("failed to resolve benchmark suite snapshot")?;

    let mut command = Command::new(sandbox_binary);
    command
        .env_clear()
        .current_dir(&work_dir)
        .arg("--die-with-parent")
        .arg("--unshare-all")
        .arg("--new-session")
        .arg("--proc")
        .arg("/proc")
        .arg("--dev")
        .arg("/dev")
        .arg("--tmpfs")
        .arg("/tmp")
        .arg("--dir")
        .arg("/tmp/home")
        .arg("--dir")
        .arg("/opt")
        .arg("--dir")
        .arg("/etc")
        .arg("--ro-bind")
        .arg("/usr")
        .arg("/usr");
    for path in ["/bin", "/lib", "/lib64"] {
        add_bwrap_system_path(&mut command, Path::new(path))?;
    }
    for path in [
        "/etc/passwd",
        "/etc/group",
        "/etc/localtime",
        "/etc/os-release",
    ] {
        add_bwrap_etc_path(&mut command, Path::new(path))?;
    }
    command
        .arg("--ro-bind")
        .arg(python_binary)
        .arg("/opt/python")
        .arg("--bind")
        .arg(work_dir)
        .arg("/work");
    if let Some(suite_dir) = suite_dir {
        command
            .arg("--ro-bind")
            .arg(suite_dir)
            .arg("/suite");
    }
    command
        .arg("--chdir")
        .arg("/work")
        .arg("--setenv")
        .arg("HOME")
        .arg("/tmp/home")
        .arg("--setenv")
        .arg("PATH")
        .arg("/usr/bin:/bin:/opt")
        .arg("--setenv")
        .arg("PYTHONDONTWRITEBYTECODE")
        .arg("1")
        .arg("--")
        .arg("/opt/python");
    Ok(command)
}

fn add_bwrap_system_path(command: &mut Command, path: &Path) -> anyhow::Result<()> {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return Ok(());
    };
    if metadata.file_type().is_symlink() {
        command.arg("--symlink").arg(fs::read_link(path)?).arg(path);
    } else if metadata.is_dir() {
        command.arg("--ro-bind").arg(path).arg(path);
    }
    Ok(())
}

fn add_bwrap_etc_path(command: &mut Command, destination: &Path) -> anyhow::Result<()> {
    if !destination.exists() {
        return Ok(());
    }
    let source = fs::canonicalize(destination)?;
    command.arg("--ro-bind").arg(source).arg(destination);
    Ok(())
}

fn resolve_executable(path: &Path) -> anyhow::Result<PathBuf> {
    if path.is_absolute() || path.components().count() > 1 {
        return fs::canonicalize(path)
            .with_context(|| format!("failed to resolve executable {}", path.display()));
    }
    let search_path = std::env::var_os("PATH").context("PATH is unavailable")?;
    for directory in std::env::split_paths(&search_path) {
        let candidate = directory.join(path);
        if candidate.is_file() {
            return fs::canonicalize(&candidate)
                .with_context(|| format!("failed to resolve executable {}", candidate.display()));
        }
    }
    Err(anyhow!(
        "executable {} was not found in PATH",
        path.display()
    ))
}

async fn run_command(
    mut command: Command,
    deadline: Instant,
    mut cancel: watch::Receiver<bool>,
    stdout_path: &Path,
    stderr_path: &Path,
) -> anyhow::Result<ProcessResult> {
    run_command_inner(
        &mut command,
        deadline,
        &mut cancel,
        stdout_path,
        stderr_path,
        None,
    )
    .await
}

async fn run_command_with_turn_limit(
    mut command: Command,
    deadline: Instant,
    mut cancel: watch::Receiver<bool>,
    stdout_path: &Path,
    stderr_path: &Path,
    max_turns: u32,
) -> anyhow::Result<ProcessResult> {
    run_command_inner(
        &mut command,
        deadline,
        &mut cancel,
        stdout_path,
        stderr_path,
        Some(max_turns),
    )
    .await
}

async fn run_command_inner(
    command: &mut Command,
    deadline: Instant,
    cancel: &mut watch::Receiver<bool>,
    stdout_path: &Path,
    stderr_path: &Path,
    max_turns: Option<u32>,
) -> anyhow::Result<ProcessResult> {
    if *cancel.borrow() {
        return Ok(ProcessResult {
            end: ProcessEnd::Cancelled,
            stdout: Vec::new(),
        });
    }
    command
        .kill_on_drop(true)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    #[cfg(unix)]
    command.process_group(0);
    let mut child = command
        .spawn()
        .context("failed to start benchmark process")?;
    let pid = child.id();
    let stdout = child
        .stdout
        .take()
        .context("benchmark stdout unavailable")?;
    let stderr = child
        .stderr
        .take()
        .context("benchmark stderr unavailable")?;
    let turn_limit = Arc::new(Notify::new());
    let mut stdout_reader = tokio::spawn(read_bounded(
        stdout,
        max_turns,
        Some(Arc::clone(&turn_limit)),
    ));
    let mut stderr_reader = tokio::spawn(read_bounded(stderr, None, None));
    #[cfg(unix)]
    let mut process_group = ProcessGroupGuard::new(pid);

    let end = tokio::select! {
        status = child.wait() => ProcessEnd::Completed(status?),
        changed = cancel.changed() => {
            if changed.is_ok() && *cancel.borrow() {
                terminate_child(&mut child, pid).await;
                ProcessEnd::Cancelled
            } else {
                terminate_child(&mut child, pid).await;
                return Err(anyhow!("benchmark cancellation channel closed"));
            }
        }
        _ = tokio::time::sleep_until(deadline) => {
            terminate_child(&mut child, pid).await;
            ProcessEnd::Timeout
        }
        _ = turn_limit.notified(), if max_turns.is_some() => {
            terminate_child(&mut child, pid).await;
            ProcessEnd::TurnLimit
        }
    };
    #[cfg(unix)]
    force_kill_process_group(pid);
    let (stdout, stdout_total) = join_reader(&mut stdout_reader, "stdout").await?;
    let (stderr, stderr_total) = join_reader(&mut stderr_reader, "stderr").await?;
    #[cfg(unix)]
    process_group.disarm();
    if let Some(parent) = stdout_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    if let Some(parent) = stderr_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(stdout_path, &stdout).await?;
    tokio::fs::write(stderr_path, &stderr).await?;
    if stdout_total > MAX_CAPTURE_BYTES || stderr_total > MAX_CAPTURE_BYTES {
        return Err(anyhow!(
            "benchmark process output exceeded the {} byte capture limit (stdout {}, stderr {}); truncated logs were saved",
            MAX_CAPTURE_BYTES,
            stdout_total,
            stderr_total
        ));
    }
    Ok(ProcessResult { end, stdout })
}

async fn join_reader(
    reader: &mut tokio::task::JoinHandle<std::io::Result<(Vec<u8>, usize)>>,
    stream: &str,
) -> anyhow::Result<(Vec<u8>, usize)> {
    match tokio::time::timeout(Duration::from_secs(3), &mut *reader).await {
        Ok(result) => result
            .with_context(|| format!("benchmark {stream} reader failed"))?
            .with_context(|| format!("benchmark {stream} read failed")),
        Err(_) => {
            reader.abort();
            Err(anyhow!(
                "benchmark {stream} pipe did not close after process-group termination"
            ))
        }
    }
}

async fn read_bounded<R>(
    mut reader: R,
    max_turns: Option<u32>,
    turn_limit: Option<Arc<Notify>>,
) -> std::io::Result<(Vec<u8>, usize)>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut captured = Vec::new();
    let mut total = 0_usize;
    let mut buffer = [0_u8; 16 * 1024];
    let mut event_buffer = Vec::new();
    let mut turns = 0_u32;
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        total = total.saturating_add(read);
        let remaining = MAX_CAPTURE_BYTES.saturating_sub(captured.len());
        captured.extend_from_slice(&buffer[..read.min(remaining)]);
        if let (Some(max_turns), Some(turn_limit)) = (max_turns, turn_limit.as_ref()) {
            event_buffer.extend_from_slice(&buffer[..read]);
            while let Some(position) = event_buffer.iter().position(|byte| *byte == b'\n') {
                let line = event_buffer.drain(..=position).collect::<Vec<_>>();
                if serde_json::from_slice::<Value>(&line)
                    .ok()
                    .and_then(|value| {
                        value
                            .get("type")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    })
                    .as_deref()
                    == Some("turn_start")
                {
                    turns = turns.saturating_add(1);
                    if turns > max_turns {
                        turn_limit.notify_one();
                    }
                }
            }
            if event_buffer.len() > MAX_SSE_BUFFER_BYTES {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "OMP JSON event exceeded the buffer limit",
                ));
            }
        }
    }
    Ok((captured, total))
}

async fn terminate_child(child: &mut tokio::process::Child, pid: Option<u32>) {
    #[cfg(unix)]
    if let Some(pid) = pid {
        unsafe {
            libc::kill(-(pid as i32), libc::SIGTERM);
        }
    } else {
        let _ = child.start_kill();
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        let _ = child.start_kill();
    }
    tokio::time::sleep(Duration::from_secs(3)).await;
    #[cfg(unix)]
    force_kill_process_group(pid);
    #[cfg(not(unix))]
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(1), child.wait()).await;
}

#[cfg(unix)]
fn force_kill_process_group(pid: Option<u32>) {
    if let Some(pid) = pid {
        unsafe {
            libc::kill(-(pid as i32), libc::SIGKILL);
        }
    }
}

#[cfg(unix)]
struct ProcessGroupGuard {
    pid: Option<u32>,
}

#[cfg(unix)]
impl ProcessGroupGuard {
    fn new(pid: Option<u32>) -> Self {
        Self { pid }
    }

    fn disarm(&mut self) {
        self.pid = None;
    }
}

#[cfg(unix)]
impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        force_kill_process_group(self.pid);
    }
}

#[derive(Debug)]
struct GradeResult {
    passed: bool,
    grader: &'static str,
    expected: Value,
    actual: Value,
    errors: Vec<String>,
}

fn grade_riddllr_answer(answer: &str, solution: &str) -> GradeResult {
    let expected_cardinals = cardinal_answers(solution);
    if expected_cardinals.len() == 4 {
        let actual_cardinals = cardinal_answers(answer);
        let mut errors = Vec::new();
        for position in ["north", "east", "south", "west"] {
            match (
                expected_cardinals.get(position),
                actual_cardinals.get(position),
            ) {
                (Some(expected), Some(actual)) if expected == actual => {}
                (Some(expected), Some(actual)) => errors.push(format!(
                    "{position} mismatch: expected {}, got {}",
                    expected.join(", "),
                    actual.join(", ")
                )),
                (Some(_), None) => errors.push(format!("{position} answer is missing")),
                _ => {}
            }
        }
        return GradeResult {
            passed: errors.is_empty(),
            grader: "cardinal-assignments",
            expected: json!(expected_cardinals),
            actual: json!(actual_cardinals),
            errors,
        };
    }
    let expected = normalized_answer_lines(solution);
    let actual = normalized_answer_lines(answer);
    if expected.is_empty() {
        return GradeResult {
            passed: false,
            grader: "ordered-lines",
            expected: json!(expected),
            actual: json!(actual),
            errors: vec!["solution does not contain a gradeable answer line".into()],
        };
    }
    let passed = expected == actual;
    GradeResult {
        passed,
        grader: "ordered-lines",
        expected: json!(expected),
        actual: json!(actual),
        errors: if passed {
            Vec::new()
        } else {
            vec!["answer lines do not match the solution".into()]
        },
    }
}

fn cardinal_answers(text: &str) -> BTreeMap<String, Vec<String>> {
    let mut answers = BTreeMap::new();
    for line in text.lines() {
        let normalized = normalize_latin(line);
        for position in ["north", "east", "south", "west"] {
            let marker = format!("{position}:");
            if let Some(index) = normalized.find(&marker) {
                let values = normalized[index + marker.len()..]
                    .split(|character: char| !character.is_ascii_alphanumeric())
                    .filter(|value| {
                        !value.is_empty() && !matches!(*value, "from" | "with" | "the" | "and")
                    })
                    .map(str::to_string)
                    .collect::<Vec<_>>();
                if !values.is_empty() {
                    answers.insert(position.to_string(), values);
                }
            }
        }
    }
    answers
}

fn normalized_answer_lines(text: &str) -> Vec<String> {
    text.lines()
        .map(strip_list_prefix)
        .map(normalize_latin)
        .map(|line| {
            line.chars()
                .filter(|character| character.is_ascii_alphanumeric())
                .collect::<String>()
        })
        .filter(|line| !line.is_empty())
        .collect()
}

fn strip_list_prefix(line: &str) -> &str {
    let line = line.trim();
    for prefix in ["- ", "* ", "# "] {
        if let Some(rest) = line.strip_prefix(prefix) {
            return rest.trim_start();
        }
    }
    let digit_count = line.bytes().take_while(u8::is_ascii_digit).count();
    if digit_count > 0 {
        let rest = &line[digit_count..];
        if let Some(rest) = rest
            .strip_prefix('.')
            .or_else(|| rest.strip_prefix(')'))
            .or_else(|| rest.strip_prefix(':'))
        {
            if rest.chars().next().is_some_and(char::is_whitespace) {
                return rest.trim_start();
            }
        }
    }
    line
}

fn normalize_latin(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    for character in text.chars() {
        match character.to_ascii_lowercase() {
            'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' => normalized.push('a'),
            'ç' => normalized.push('c'),
            'é' | 'è' | 'ê' | 'ë' => normalized.push('e'),
            'í' | 'ì' | 'î' | 'ï' => normalized.push('i'),
            'ñ' => normalized.push('n'),
            'ó' | 'ò' | 'ô' | 'ö' | 'õ' => normalized.push('o'),
            'ú' | 'ù' | 'û' | 'ü' => normalized.push('u'),
            'ý' | 'ÿ' => normalized.push('y'),
            other => normalized.extend(other.to_lowercase()),
        }
    }
    normalized
}

fn parse_pytest_counts(output: &str) -> (Option<u64>, Option<u64>) {
    let mut passed = 0_u64;
    let mut failed = 0_u64;
    let mut errors = 0_u64;
    let tokens = output.split_whitespace().collect::<Vec<_>>();
    for window in tokens.windows(2) {
        let Ok(count) = window[0]
            .trim_matches(|character: char| !character.is_ascii_digit())
            .parse()
        else {
            continue;
        };
        match window[1].trim_matches(|character: char| !character.is_ascii_alphabetic()) {
            "passed" => passed = count,
            "failed" => failed = count,
            "error" | "errors" => errors = count,
            _ => {}
        }
    }
    let total = passed.saturating_add(failed).saturating_add(errors);
    ((total > 0).then_some(passed), (total > 0).then_some(total))
}

#[derive(Debug, Serialize)]
struct OmpTranscriptSummary {
    session_id: Option<String>,
    stop_reason: Option<String>,
    turns: u64,
    generated_tokens: Option<u64>,
}

fn parse_omp_transcript(bytes: &[u8]) -> OmpTranscriptSummary {
    let mut summary = OmpTranscriptSummary {
        session_id: None,
        stop_reason: None,
        turns: 0,
        generated_tokens: None,
    };
    let mut output_tokens = 0_u64;
    for line in String::from_utf8_lossy(bytes).lines() {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let event_type = value.get("type").and_then(Value::as_str);
        if summary.session_id.is_none() && matches!(event_type, Some("session" | "session_start")) {
            summary.session_id = value.get("id").and_then(Value::as_str).map(str::to_string);
        }
        if let Some(reason) = value
            .get("stopReason")
            .or_else(|| value.get("stop_reason"))
            .and_then(Value::as_str)
        {
            summary.stop_reason = Some(reason.to_string());
        }
        if event_type == Some("turn_end") {
            summary.turns = summary.turns.saturating_add(1);
            output_tokens = output_tokens.saturating_add(
                value
                    .pointer("/usage/output")
                    .and_then(Value::as_u64)
                    .unwrap_or_default(),
            );
        }
    }
    summary.generated_tokens = (output_tokens > 0).then_some(output_tokens);
    summary
}

fn read_elegance_metrics(path: &Path, task_id: &str, tag: &str) -> anyhow::Result<Value> {
    let value: Value = serde_json::from_slice(&fs::read(path)?)?;
    value
        .pointer(&format!(
            "/solutions/{}/{}",
            escape_json_pointer(task_id),
            escape_json_pointer(tag)
        ))
        .cloned()
        .context("elegance output did not contain this job")
}

fn escape_json_pointer(value: &str) -> String {
    value.replace('~', "~0").replace('/', "~1")
}

fn aborted_result(status: RunStatus, elapsed: Duration, reason: &str) -> ExecutionResult {
    ExecutionResult {
        run_status: status,
        exit_code: None,
        passed: None,
        duration_ms: elapsed.as_secs_f64() * 1000.0,
        stdout_path: None,
        stderr_path: None,
        failure_reason: Some(reason.into()),
        details: json!({"reason": reason}),
        quality: Vec::new(),
    }
}

fn completion_message(result: &ExecutionResult) -> &'static str {
    match result.run_status {
        RunStatus::Succeeded if result.passed == Some(true) => "Benchmark completed and passed",
        RunStatus::Succeeded if result.passed == Some(false) => {
            "Benchmark completed; quality checks failed"
        }
        RunStatus::Succeeded => "Benchmark completed",
        RunStatus::Failed => "Benchmark harness failed",
        RunStatus::Timeout => "Benchmark timed out",
        RunStatus::Cancelled => "Benchmark cancelled",
        _ => "Benchmark completed",
    }
}

fn job_status(status: RunStatus) -> &'static str {
    match status {
        RunStatus::Succeeded => "succeeded",
        RunStatus::Failed | RunStatus::Oom => "failed",
        RunStatus::Timeout => "timeout",
        RunStatus::Cancelled => "cancelled",
        RunStatus::Planned | RunStatus::Running | RunStatus::Skipped => "failed",
    }
}

fn normalize_model(model: &str) -> Result<String, LabError> {
    let model = model.trim();
    let model = model.strip_prefix("brainrouter/").unwrap_or(model);
    if model.is_empty()
        || model.len() > 160
        || !model.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b'/' | b':')
        })
    {
        return Err(LabError::Validation(
            "model must be 1-160 ASCII letters, digits, '.', '_', '-', '/', or ':'".into(),
        ));
    }
    Ok(model.to_string())
}

fn valid_case_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
        && value != "."
        && value != ".."
}

fn router_model_selector(model: &str) -> String {
    match model {
        "auto" | "local" | "cloud" => model.to_string(),
        _ => format!("brainrouter/{model}"),
    }
}

fn job_identity(
    request: &StartJobRequest,
    manifest_sha256: &str,
    config: &BenchmarkLabConfig,
) -> Result<String, LabError> {
    let value = json!({
        "suite": request.suite,
        "case_id": request.case_id,
        "model": request.model,
        "repetition": request.repetition,
        "manifest_sha256": manifest_sha256,
        "max_job_seconds": config.max_job_seconds,
        "riddllr_max_tokens": config.riddllr_max_tokens,
        "plumebench_max_turns": config.plumebench_max_turns,
        "plumebench_thinking": config.plumebench_thinking,
    });
    let bytes = serde_json::to_vec(&value)
        .map_err(|error| LabError::Internal(format!("cannot serialize job identity: {error}")))?;
    Ok(sha256_bytes(&bytes))
}

fn exact_command(request: &StartJobRequest, resolved: &ResolvedCase) -> String {
    match resolved {
        ResolvedCase::Riddllr { .. } => format!(
            "brainrouter internal-chat --model {} --suite riddllr --case {}",
            router_model_selector(&request.model),
            request.case_id
        ),
        ResolvedCase::Plumebench { .. } => format!(
            "bwrap <isolated mounts and private network> -- brainrouter benchmark-sandbox <inference-only proxy> -- omp -p --model brainrouter/{} --mode json --max-time <configured> --thinking <configured> --auto-approve --no-session --cwd /work @.brainrouter-task.md; Brainrouter enforces the configured turn limit from OMP turn_start events",
            request.model
        ),
    }
}

fn runtime_definition() -> anyhow::Result<RuntimeDefinition> {
    let executable_sha256 = std::env::current_exe()
        .ok()
        .and_then(|path| fs::read(path).ok())
        .map(|bytes| sha256_bytes(&bytes));
    let version = env!("CARGO_PKG_VERSION");
    let repository = "https://github.com/ajaxdude/brainrouter".to_string();
    let dirty_tree = env!("BRAINROUTER_GIT_DIRTY") == "true";
    let git_sha = env!("BRAINROUTER_GIT_SHA");
    let commit_sha = if dirty_tree {
        format!(
            "{git_sha}-dirty-{}",
            executable_sha256
                .as_deref()
                .and_then(|hash| hash.get(..12))
                .unwrap_or("unknown")
        )
    } else {
        git_sha.to_string()
    };
    let compiler_version = env!("BRAINROUTER_RUSTC_VERSION").to_string();
    let build_flags = vec![
        format!("package={version}"),
        format!("target={}-{}", std::env::consts::ARCH, std::env::consts::OS),
        format!("compiler={compiler_version}"),
    ];
    let identity = sha256_bytes(&serde_json::to_vec(&(
        &repository,
        &commit_sha,
        "other",
        &build_flags,
        dirty_tree,
    ))?);
    let built_at = DateTime::parse_from_rfc3339(env!("BRAINROUTER_GIT_COMMIT_DATE"))
        .map(|value| value.with_timezone(&Utc))
        .unwrap_or_else(|_| epoch());
    Ok(RuntimeDefinition {
        id: format!("runtime-brainrouter-{}", &identity[..16]),
        repository,
        fork_name: "brainrouter-benchmark-lab".into(),
        commit_sha,
        dirty_tree,
        compiler: "rustc".into(),
        compiler_version,
        backend: Backend::Other,
        build_flags,
        capabilities: BTreeMap::from([
            ("native_benchmark_lab".into(), FeatureState::Enabled),
            ("hidden_test_isolation".into(), FeatureState::Enabled),
            ("grader_sandbox".into(), FeatureState::Enabled),
            ("private_network_namespace".into(), FeatureState::Enabled),
        ]),
        executable_sha256,
        container_digest: None,
        built_at,
    })
}

fn hardware_profile() -> HardwareProfile {
    let logical_cores = num_cpus::get().max(1) as u64;
    let system_ram_bytes = system_ram_bytes().max(1);
    let identity = sha256_text(&format!(
        "{}:{}:{}:{}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        logical_cores,
        system_ram_bytes
    ));
    HardwareProfile {
        id: format!("hardware-{}", &identity[..24]),
        hostname_hash: None,
        cpu_model: std::env::consts::ARCH.into(),
        physical_cores: None,
        logical_cores: Some(logical_cores),
        system_ram_bytes,
        gpus: Vec::new(),
        unified_memory: false,
        os_name: std::env::consts::OS.into(),
        os_version: "unknown".into(),
        kernel: "unknown".into(),
        driver_versions: BTreeMap::new(),
        power_profile: None,
        metadata: BTreeMap::from([(
            "declaration_note".into(),
            json!("Minimal host snapshot captured without external probes."),
        )]),
        captured_at: epoch(),
    }
}

fn system_ram_bytes() -> u64 {
    unsafe {
        let pages = libc::sysconf(libc::_SC_PHYS_PAGES);
        let page_size = libc::sysconf(libc::_SC_PAGESIZE);
        if pages > 0 && page_size > 0 {
            (pages as u64).saturating_mul(page_size as u64)
        } else {
            1
        }
    }
}

fn epoch() -> DateTime<Utc> {
    DateTime::parse_from_rfc3339("1970-01-01T00:00:00Z")
        .expect("valid epoch")
        .with_timezone(&Utc)
}

fn model_family(model: &str) -> String {
    model
        .split(['/', '-', '_'])
        .find(|part| !part.is_empty())
        .unwrap_or("unknown")
        .to_ascii_lowercase()
}

fn inferred_quantization(model: &str) -> String {
    let lower = model.to_ascii_lowercase();
    for quant in [
        "iq1", "iq2", "iq3", "iq4", "q2", "q3", "q4", "q5", "q6", "q8", "fp8", "fp16", "bf16",
    ] {
        if lower.contains(quant) {
            return quant.to_ascii_uppercase();
        }
    }
    "unknown".into()
}

fn count_tokens(text: &str) -> u64 {
    text.split_whitespace().count() as u64
}

fn value_object(value: Value) -> BTreeMap<String, Value> {
    value
        .as_object()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .collect()
}

fn text_preview(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }
    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n[truncated: {} bytes total]",
        &value[..end],
        value.len()
    )
}

fn json_preview(value: &Value, max_bytes: usize) -> Value {
    match serde_json::to_string(value) {
        Ok(serialized) if serialized.len() <= max_bytes => value.clone(),
        Ok(serialized) => json!({
            "truncated_json": text_preview(&serialized, max_bytes),
            "original_bytes": serialized.len(),
        }),
        Err(error) => json!({"preview_error": error.to_string()}),
    }
}

fn slug(value: &str) -> String {
    let mut result = String::new();
    let mut separator = false;
    for character in value.chars() {
        if character.is_ascii_alphanumeric() {
            result.push(character.to_ascii_lowercase());
            separator = false;
        } else if !separator && !result.is_empty() {
            result.push('-');
            separator = true;
        }
        if result.len() >= 48 {
            break;
        }
    }
    result.trim_matches('-').to_string()
}

fn sha256_text(value: &str) -> String {
    sha256_bytes(value.as_bytes())
}

fn sha256_bytes(value: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grades_cardinal_answers_semantically() {
        let solution = "* North: Aivas from Zartharim with crossbow\n* South: Tristan from Karville with sword\n* East: Camorra from Saranthium with axe\n* West: Robert from Arkney with bow\n";
        let answer = "North: Aivas, Zartharim, crossbow\nEast: Camorra, Saranthium, axe\nSouth: Tristan, Karville, sword\nWest: Robert, Arkney, bow\n";
        let grade = grade_riddllr_answer(answer, solution);
        assert!(grade.passed, "{:?}", grade.errors);
    }

    #[test]
    fn grades_ordered_lines_and_normalizes_accents() {
        let solution = "1. Paris\n2. Tokyo\n3. Brasilia\n";
        let answer = "Paris\nTokyo\nBrasília\n";
        assert!(grade_riddllr_answer(answer, solution).passed);
    }

    #[test]
    fn rejects_extra_ordered_answer_text() {
        let solution = "1. Paris\n2. Tokyo\n";
        let answer = "Here you go:\nParis\nTokyo\n";
        assert!(!grade_riddllr_answer(answer, solution).passed);
    }

    #[test]
    fn ordered_grader_preserves_numeric_answers_and_signs() {
        assert!(grade_riddllr_answer("42\n-42\n", "1. 42\n2. -42\n").passed);
        assert!(!grade_riddllr_answer("", "1. 42\n").passed);
        assert!(!grade_riddllr_answer("", "1.\n").passed);
    }

    #[test]
    fn parses_pytest_summary() {
        assert_eq!(
            parse_pytest_counts("2 failed, 7 passed in 0.12s"),
            (Some(7), Some(9))
        );
        assert_eq!(parse_pytest_counts("5 passed in 0.04s"), (Some(5), Some(5)));
        assert_eq!(parse_pytest_counts("collection failed"), (None, None));
    }

    #[test]
    fn parses_current_omp_event_schema_without_double_counting() {
        let transcript = parse_omp_transcript(
            br#"{"type":"session","id":"session-1"}
{"type":"turn_start"}
{"type":"turn_end","usage":{"input":5,"output":7}}
{"type":"turn_start"}
{"type":"turn_end","stopReason":"stop","usage":{"input":9,"output":3}}
{"type":"session_end","usage":{"input":14,"output":10}}
"#,
        );
        assert_eq!(transcript.session_id.as_deref(), Some("session-1"));
        assert_eq!(transcript.stop_reason.as_deref(), Some("stop"));
        assert_eq!(transcript.turns, 2);
        assert_eq!(transcript.generated_tokens, Some(10));
    }

    #[test]
    fn model_validation_blocks_shell_metacharacters() {
        assert!(normalize_model("qwen3.6-35b-q4").is_ok());
        assert!(normalize_model("brainrouter/qwen3.6-35b-q4").is_ok());
        assert!(normalize_model("qwen; rm -rf /").is_err());
        assert!(normalize_model("../qwen").is_ok());
    }

    #[test]
    fn case_ids_cannot_traverse() {
        assert!(valid_case_id("t1_greenfield"));
        assert!(!valid_case_id("../hidden"));
        assert!(!valid_case_id("nested/task"));
        assert!(!valid_case_id(".."));
    }

    #[cfg(unix)]
    #[test]
    fn workspace_copy_rejects_symlinks() {
        use std::os::unix::fs::symlink;

        let root =
            std::env::temp_dir().join(format!("brainrouter-lab-copy-{}", uuid::Uuid::new_v4()));
        let source = root.join("source");
        let destination = root.join("destination");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("safe.py"), "print('safe')\n").unwrap();
        symlink("/etc/passwd", source.join("escape")).unwrap();
        let error = copy_tree(&source, &destination).unwrap_err();
        assert!(error.to_string().contains("symlink"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn overlapping_workspace_and_suite_roots_are_detected() {
        let suite = Path::new("/srv/benchmarks/plumebench");
        assert!(paths_overlap(suite, &suite.join("workspace")));
        assert!(paths_overlap(&suite.join("workspace"), suite));
        assert!(!paths_overlap(
            suite,
            Path::new("/srv/brainrouter/benchmark-lab")
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn omp_turn_limit_is_enforced_from_json_events() {
        let root = std::env::temp_dir().join(format!("brainrouter-turns-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let mut command = Command::new("/bin/sh");
        command.arg("-c").arg(
            "printf '%s\n' \
             '{\"type\":\"turn_start\"}' \
             '{\"type\":\"turn_end\",\"usage\":{\"output\":1}}' \
             '{\"type\":\"turn_start\"}' \
             '{\"type\":\"turn_end\",\"usage\":{\"output\":1}}' \
             '{\"type\":\"turn_start\"}' \
             '{\"type\":\"turn_end\",\"usage\":{\"output\":1}}' \
             '{\"type\":\"turn_start\"}'; /bin/sleep 30",
        );
        let (_cancel_tx, cancel_rx) = watch::channel(false);
        let started = Instant::now();
        let result = run_command_with_turn_limit(
            command,
            Instant::now() + Duration::from_secs(10),
            cancel_rx,
            &root.join("stdout"),
            &root.join("stderr"),
            3,
        )
        .await
        .unwrap();
        assert!(matches!(result.end, ProcessEnd::TurnLimit));
        assert!(started.elapsed() < Duration::from_secs(5));
        let _ = fs::remove_dir_all(root);
    }
}
