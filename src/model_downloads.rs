//! Native Rust model-download orchestration for the ai-toolbox-cockpit
//! backends that support it (`ds4`/`halogen`/`r9v`/`llama_cpp`).
//!
//! `vllm` is deliberately excluded — upstream's own `vllm/models.py` is
//! "deliberately non-downloading" (its server pulls from Hugging Face on
//! demand at container start), so brainrouter mirrors that division of
//! responsibility rather than inventing a download step cockpit itself
//! doesn't have.
//!
//! This shells out to the `hf` CLI (`huggingface_hub`'s official CLI) with
//! per-backend argument shapes mirrored exactly from upstream's
//! `backends/{ds4,halogen,llama_cpp,r9v}/model_manager.py::get_download_cmd()`
//! (verified live against upstream source, not guessed). It does **not**
//! attempt cockpit's own foreground-suspend-the-TUI execution model
//! (`self.app.suspend()` + synchronous `subprocess.run`) — brainrouter is a
//! headless HTTP server with no TTY to suspend into, so downloads run as
//! background jobs tracked in an in-memory registry, mirroring
//! `src/benchmark_lab.rs`'s job-registry shape (single-flight semaphore,
//! poll-based status, real OS-level cancellation).
//!
//! See `docs/design/ai-toolbox-cockpit-integration.md` §10 for the full
//! design rationale, including the explicitly-flagged open items (no
//! byte-level progress — see the "Design note" below — concurrency default,
//! and HF-token write-path scope).

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::{watch, Mutex, RwLock, Semaphore};
use tracing::{error, warn};

use crate::toolbox_catalog::{
    self, CatalogModelFile, GufoModel, GufoRole, GufoSpeculativeMode, ModelPayload,
    SupportedServingBackend,
};

/// In-memory job history cap. Unlike `benchmark_lab.rs`'s SQLite-persisted
/// job store, this registry is deliberately in-memory only for v1 — a
/// daemon restart loses download job history (an active download's own
/// child process is also lost, same as any other subprocess would be on
/// restart). Flagged as an accepted v1 scope trade-off in design doc §10.
pub(crate) const MAX_JOB_HISTORY: usize = 100;

/// Bounded raw stdout+stderr tail kept per job, for the dashboard's
/// "raw tool output" reassurance panel (§10). Deliberately much smaller
/// than `benchmark_lab.rs`'s 4 MiB capture buffer: this is a best-effort
/// debug tail, not a verified progress signal — `hf`'s own `\r`-based tqdm
/// redraws aren't a stable format to parse a percentage out of.
const MAX_OUTPUT_TAIL_BYTES: usize = 64 * 1024;

/// Coarse download lifecycle state machine (§10) — deliberately not a byte
/// percentage. `hf download --local-dir` downloads to a temp/cache location
/// and only atomically renames into the destination on success (confirmed
/// via upstream maintainer statements during PR6's design pass), so no
/// partial-byte progress is observable by polling the destination directory.
///
/// `Running` (PR11, §15 item 3) is the generic in-progress state for job
/// kinds that aren't an `hf download` — today, only r9v's "Prepare PLE"
/// job (see [`JobKind`]) — kept distinct from `Downloading` so the
/// dashboard never shows a misleading "downloading" badge for a podman
/// extraction job. `Verifying` stays shared: both an `hf download`'s
/// post-transfer file check and a prepare-PLE job's post-run size check
/// are the same kind of "confirm the output looks right" step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadStatus {
    Queued,
    Downloading,
    Running,
    Verifying,
    Complete,
    Failed,
    Cancelled,
}

/// Which real-world action a [`ModelDownloadJob`] record represents (PR11,
/// §15 item 3). The registry was widened to carry both kinds rather than
/// standing up a second, parallel job registry for r9v's one "Prepare PLE"
/// action — see the module-level design note referenced above for the
/// explicit tradeoff (~150 lines of proven registry/cancellation/history
/// code reused, at the cost of `ModelDownloadJob` no longer being *only*
/// downloads despite its name).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Download,
    PreparePle,
}

/// A single job's externally-visible state — despite the name (kept for
/// minimal diff/history, PR6-8), this now also represents r9v's "Prepare
/// PLE" job (PR11, [`JobKind::PreparePle`]), disambiguated by `kind`.
#[derive(Clone, Debug, Serialize)]
pub struct ModelDownloadJob {
    pub id: String,
    pub kind: JobKind,
    pub backend: SupportedServingBackend,
    /// The catalog entry this job concerns — a model id for `Download`
    /// jobs, r9v's package id for `PreparePle` jobs.
    pub model_id: String,
    /// Where the job's output lands — the model's destination directory
    /// for `Download` jobs, r9v's `ple_dir` for `PreparePle` jobs.
    pub destination: String,
    pub status: DownloadStatus,
    pub message: String,
    /// The exact subprocess invocation, for transparency/debugging — not a
    /// secret (the destination path and repo id are already visible
    /// elsewhere; no token is ever included here, see `hf_token_env`).
    pub command_display: String,
    /// Best-effort bounded tail of the subprocess's combined stdout+stderr,
    /// explicitly labeled as raw tool output, never a verified percentage.
    pub output_tail: String,
    pub queued_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub ended_at: Option<DateTime<Utc>>,
}


struct JobControl {
    record: RwLock<ModelDownloadJob>,
    cancel: watch::Sender<bool>,
}

/// Why a requested download/verify/status action didn't happen.
#[derive(Debug)]
pub enum DownloadError {
    /// The requested backend/model/quant combination doesn't resolve to a
    /// catalog entry, or the backend doesn't support downloads at all
    /// (`vllm` — see module docs).
    Validation(String),
    NotFound(String),
    /// A cancel/verify request racing a job that already reached a
    /// terminal state.
    Conflict(String),
    Internal(String),
}

impl DownloadError {
    pub fn status(&self) -> u16 {
        match self {
            Self::Validation(_) => 400,
            Self::NotFound(_) => 404,
            Self::Conflict(_) => 409,
            Self::Internal(_) => 500,
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Validation(m) | Self::NotFound(m) | Self::Conflict(m) | Self::Internal(m) => m,
        }
    }
}

impl std::fmt::Display for DownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for DownloadError {}

/// A caller-supplied request to start a download.
#[derive(Debug, Clone, Deserialize)]
pub struct StartDownloadRequest {
    pub backend: String,
    pub model_id: String,
    /// Required only for `llama_cpp` (the catalog's `inference_profiles`
    /// aren't fully typed yet — see design doc §10 — so the client passes
    /// the exact quant/pattern string it read out of the catalog's raw
    /// `extra` field for the chosen entry). Ignored for every other backend
    /// (their catalog entries already fully specify what to download).
    #[serde(default)]
    pub quant_pattern: Option<String>,
}

/// Per-file expectation used both to build the `hf download` command and
/// to check post-download completeness.
#[derive(Debug)]
struct ExpectedFile {
    relative_path: String,
    expected_size_bytes: Option<u64>,
    sha256: Option<String>,
}

/// A fully-resolved, not-yet-executed download.
#[derive(Debug)]
struct BuiltDownload {
    args: Vec<String>,
    destination: PathBuf,
    expected_files: Vec<ExpectedFile>,
}

/// Result of a (non-cryptographic) completeness check — mirrors upstream's
/// own per-backend matrix exactly (§10), not a uniform stronger check.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CompletenessResult {
    /// `llama_cpp` has no manifest-based completeness check upstream
    /// either — only the download subprocess's own exit code is used.
    NotChecked { reason: String },
    Complete,
    Incomplete { missing_or_mismatched: Vec<String> },
}

/// One file's result from an explicit, user-triggered SHA256 verification
/// pass (only meaningful for backends whose catalog entries carry a
/// `sha256` — today, only `r9v`; mirrors upstream's separate, expensive,
/// not-run-automatically `verify_package()`/`verify_file()`).
#[derive(Debug, Clone, Serialize)]
pub struct FileVerification {
    pub relative_path: String,
    pub expected_sha256: String,
    pub actual_sha256: Option<String>,
    pub matched: bool,
}

/// Local on-disk presence for one catalog model entry, independent of any
/// download job (used by the read-only status sweep the dashboard polls to
/// show "already downloaded" badges).
#[derive(Debug, Clone, Serialize)]
pub struct ModelPresence {
    pub backend: SupportedServingBackend,
    pub model_id: String,
    pub destination: String,
    pub completeness: CompletenessResult,
    /// R9V's second readiness gate (design doc §15 item 8): `Some(true)`/
    /// `Some(false)` for r9v entries that carry a typed `ple` manifest
    /// field, `None` for every other backend (and for an r9v entry that
    /// somehow lacks one) — lets the dashboard show "PLE: ready/not
    /// prepared" and gate the Server Mode Start button without a second
    /// round-trip per package.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ple_ready: Option<bool>,
}

/// Expands a leading `~` the same way cockpit's own `Path(...).expanduser()`
/// does. `$HOME` falls back to `/root` if unset, matching
/// `cockpit_config.rs::config_path()`'s existing convention. `pub(crate)`
/// (PR9+) so `server_mode.rs`'s vllm cache-directory resolution reuses this
/// exact expansion rather than duplicating it (§14).
pub(crate) fn expand_tilde(raw: &str) -> PathBuf {
    let home = || std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    if raw == "~" {
        return PathBuf::from(home());
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        return PathBuf::from(home()).join(rest);
    }
    PathBuf::from(raw)
}

/// Resolves the effective models directory for `backend`, mirroring
/// upstream's own precedence exactly: a cockpit `config.json`
/// `backends.<id>.models_dir` override first (read via the same shared
/// config source of truth as PR3 — requirement 3, "ideally merging
/// config.json"), falling back to the catalog's own
/// `backends.<id>.storage.default` (verified present for all four
/// download-capable backends in the vendored fixture — see design doc §10).
fn effective_models_dir(backend: SupportedServingBackend, storage: &serde_json::Value) -> PathBuf {
    let cockpit = crate::cockpit_config::load();
    let override_dir = cockpit.config.as_ref().and_then(|c| c.models_dir(backend.as_str()));
    let catalog_default = storage.get("default").and_then(serde_json::Value::as_str);
    let raw = override_dir.or(catalog_default).unwrap_or("~/models");
    expand_tilde(raw)
}

/// Looks up one catalog model entry by backend + id, returning its payload
/// and the backend's `storage` metadata (§10). Returns a `Validation` error
/// (not `NotFound`) for `vllm`/unsupported backends, since that's a request
/// shape problem, not a missing-resource one.
fn resolve_catalog_entry(
    backend: SupportedServingBackend,
    model_id: &str,
) -> Result<(ModelPayload, serde_json::Value), DownloadError> {
    if backend == SupportedServingBackend::Vllm {
        return Err(DownloadError::Validation(
            "vllm has no download step in ai-toolbox-cockpit — its server pulls from Hugging \
             Face on demand at container start; browse only (design doc §10)"
                .to_string(),
        ));
    }
    let (_toolboxes, models) = toolbox_catalog::load_effective_typed_catalog()
        .map_err(|e| DownloadError::Internal(e.to_string()))?;
    let backend_catalog = models
        .backends
        .iter()
        .find(|b| b.backend.as_str() == backend.as_str())
        .ok_or_else(|| DownloadError::Internal(format!("catalog has no `{}` backend section", backend.as_str())))?;
    let entry = backend_catalog
        .entries
        .iter()
        .find(|e| e.id == model_id)
        .ok_or_else(|| DownloadError::NotFound(format!("no `{}` catalog entry with id `{model_id}`", backend.as_str())))?;
    let payload = entry
        .payload
        .clone()
        .ok_or_else(|| DownloadError::Internal(format!("catalog entry `{model_id}` has no typed `{}` payload", backend.as_str())))?;
    Ok((payload, backend_catalog.storage.clone()))
}

/// The `hf` binary to invoke. Upstream tries the executable colocated with
/// its own Python interpreter first, falling back to `PATH` — not
/// applicable to a compiled Rust binary, so brainrouter simply resolves via
/// `PATH`, the same ultimate fallback upstream itself uses when its
/// colocated-executable guess misses.
fn hf_binary() -> &'static str {
    "hf"
}

/// The exact command that installs the `hf` CLI onto the brainrouter service
/// account's PATH, verified on the Strix Halo host (`~/.local/bin`, which is
/// on the per-user systemd service PATH). Single runtime source of truth for
/// the install hint surfaced by [`hf_preflight`]; a `#[test]` below asserts
/// README.md and PRD.md both contain this exact string so docs cannot drift.
const HF_INSTALL_COMMAND: &str = "python3 -m pip install --user -U \"huggingface_hub[cli]\"";

/// Advisory PATH-preflight for the `hf` CLI, surfaced on `/api/model-downloads/status`.
///
/// `found_on_path` is a best-effort, conservative diagnostic — whether an
/// executable file named [`hf_binary`] is visible on the service's PATH — not
/// a guarantee the binary will launch (effective-user perms, `noexec`, or a
/// bad interpreter can still make the real download spawn fail; that spawn
/// remains authoritative). When not found, `message`/`install_command` carry
/// the remediation; when found they are `None`.
#[derive(Debug, Clone, Serialize)]
pub struct HfPreflight {
    pub found_on_path: bool,
    pub binary: String,
    pub message: Option<String>,
    pub install_command: Option<String>,
}

/// True when `p` resolves (following symlinks) to a regular file that carries
/// an execute bit on Unix. A dangling symlink or missing file yields `false`.
/// The execute-bit check is a heuristic (any of the three bits, not the
/// effective-user bit); the per-job download spawn is the authoritative check.
fn is_executable_file(p: &Path) -> bool {
    match std::fs::metadata(p) {
        Ok(md) => {
            if !md.is_file() {
                return false;
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                md.permissions().mode() & 0o111 != 0
            }
            #[cfg(not(unix))]
            {
                true
            }
        }
        Err(_) => false,
    }
}

/// Resolve `name` against `entries` using an injected executable predicate.
/// If `name` contains a path separator it is treated as a direct path; empty
/// PATH entries are skipped (a stray `./name` must never be reported as
/// "found"). The predicate injection keeps the empty-skip invariant testable
/// without touching the filesystem or the process working directory.
fn resolve_in_with<I, F>(entries: I, name: &str, is_exec: F) -> bool
where
    I: IntoIterator<Item = PathBuf>,
    F: Fn(&Path) -> bool,
{
    if name.contains('/') {
        return is_exec(Path::new(name));
    }
    for entry in entries {
        if entry.as_os_str().is_empty() {
            continue;
        }
        if is_exec(&entry.join(name)) {
            return true;
        }
    }
    false
}

fn resolve_in<I: IntoIterator<Item = PathBuf>>(entries: I, name: &str) -> bool {
    resolve_in_with(entries, name, is_executable_file)
}

fn resolve_on_path(name: &str) -> bool {
    match std::env::var_os("PATH") {
        Some(path) => resolve_in(std::env::split_paths(&path), name),
        None => false,
    }
}

/// Build an [`HfPreflight`] from an already-resolved `found_on_path` bool.
/// Separated from [`resolve_on_path`] so the field/nullability logic is unit
/// testable without touching the process environment.
pub(crate) fn build_preflight(found_on_path: bool) -> HfPreflight {
    if found_on_path {
        HfPreflight {
            found_on_path: true,
            binary: hf_binary().to_owned(),
            message: None,
            install_command: None,
        }
    } else {
        HfPreflight {
            found_on_path: false,
            binary: hf_binary().to_owned(),
            message: Some(
                "The Hugging Face CLI (`hf`) was not found on the brainrouter service's PATH. \
                 Model downloads run `hf`, so they need it installed and on the service PATH to work."
                    .to_owned(),
            ),
            install_command: Some(HF_INSTALL_COMMAND.to_owned()),
        }
    }
}

/// Advisory preflight recomputed on each `/api/model-downloads/status` poll.
pub fn hf_preflight() -> HfPreflight {
    build_preflight(resolve_on_path(hf_binary()))
}

/// Builds the exact `hf download` argument list + expected post-download
/// file set for one backend/model/quant combination, mirroring upstream's
/// `get_download_cmd()` shapes verified live during PR6's design pass
/// (design doc §10). Does not execute anything.
fn build_download(
    backend: SupportedServingBackend,
    payload: &ModelPayload,
    models_dir: &Path,
    quant_pattern: Option<&str>,
) -> Result<BuiltDownload, DownloadError> {
    match (backend, payload) {
        (SupportedServingBackend::LlamaCpp, ModelPayload::LlamaCpp(m)) => {
            let pattern = quant_pattern.ok_or_else(|| {
                DownloadError::Validation("llama_cpp downloads require `quant_pattern`".to_string())
            })?;
            let repo_basename = m.repo.rsplit('/').next().unwrap_or(&m.repo);
            let destination = models_dir.join(repo_basename);
            let mut args = vec![
                "download".to_string(),
                m.repo.clone(),
                "--local-dir".to_string(),
                destination.display().to_string(),
            ];
            if pattern.ends_with(".gguf") {
                if pattern.contains('*') {
                    args.push("--include".to_string());
                    args.push(pattern.to_string());
                } else {
                    args.push(pattern.to_string());
                }
            } else {
                // Folder-based quant (e.g. "BF16", "UD-IQ2_M"): glob the whole subfolder.
                args.push("--include".to_string());
                args.push(format!("{pattern}/*"));
            }
            // llama_cpp has no manifest-based completeness check upstream
            // either (§10) — `expected_files` stays empty on purpose.
            Ok(BuiltDownload { args, destination, expected_files: Vec::new() })
        }
        (SupportedServingBackend::Ds4, ModelPayload::Ds4(m)) => {
            let destination = models_dir.to_path_buf();
            let args = vec![
                "download".to_string(),
                m.repo.clone(),
                m.filename.clone(),
                "--local-dir".to_string(),
                destination.display().to_string(),
            ];
            Ok(BuiltDownload {
                args,
                destination,
                expected_files: vec![ExpectedFile {
                    relative_path: m.filename.clone(),
                    // ds4's own completeness check is file-exists-only, no
                    // size check (§10) — expected_size_bytes stays None.
                    expected_size_bytes: None,
                    sha256: None,
                }],
            })
        }
        (SupportedServingBackend::Halogen, ModelPayload::Halogen(m)) => {
            Ok(build_multifile_download(&m.repo, &m.revision, &m.files, models_dir))
        }
        (SupportedServingBackend::R9v, ModelPayload::R9v(m)) => {
            Ok(build_multifile_download(&m.repo, &m.revision, &m.files, models_dir))
        }
        (SupportedServingBackend::Gufo, ModelPayload::Gufo(m)) => {
            // Each gufo entry (main or draft) is a single-repo, single-revision
            // download of its one GGUF, verified by an exact-size completeness
            // check (the overlay carries the exact LFS `size_bytes`).
            Ok(build_multifile_download(&m.repo, &m.revision, &m.files, models_dir))
        }
        _ => Err(DownloadError::Internal(
            "catalog payload variant does not match the requested backend".to_string(),
        )),
    }
}

/// Shared halogen/r9v shape: every file in the bundle's `files[]` manifest,
/// pinned to `revision`, downloaded flat into the backend's single models
/// directory (not a per-bundle subdirectory — confirmed against upstream's
/// `halogen/models.py`/`r9v/models.py` call sites, which both pass
/// `get_models_dir()` directly, not a per-entry subdirectory).
fn build_multifile_download(
    repo: &str,
    revision: &str,
    files: &[CatalogModelFile],
    models_dir: &Path,
) -> BuiltDownload {
    let destination = models_dir.to_path_buf();
    let mut args = vec!["download".to_string(), repo.to_string()];
    for file in files {
        args.push(file.path.clone());
    }
    args.push("--revision".to_string());
    args.push(revision.to_string());
    args.push("--local-dir".to_string());
    args.push(destination.display().to_string());
    let expected_files = files
        .iter()
        .map(|f| ExpectedFile {
            relative_path: f.path.clone(),
            expected_size_bytes: Some(f.size_bytes),
            sha256: f.sha256.clone(),
        })
        .collect();
    BuiltDownload { args, destination, expected_files }
}

/// Checks whether every expected file is present with the expected size
/// (or simply present, for backends whose catalog carries no size at all —
/// `ds4`). Never reads file contents — mirrors upstream's own deliberately
/// size-based-not-checksum-based check ("Size checks catch missing/partial
/// downloads without reading 118 GiB" — halogen's own source comment).
fn check_completeness(expected_files: &[ExpectedFile], destination: &Path) -> CompletenessResult {
    if expected_files.is_empty() {
        return CompletenessResult::NotChecked {
            reason: "no manifest-based completeness check for this backend (design doc §10)".to_string(),
        };
    }
    let mut bad = Vec::new();
    for file in expected_files {
        let path = destination.join(&file.relative_path);
        let ok = match std::fs::metadata(&path) {
            Ok(meta) => meta.is_file() && file.expected_size_bytes.is_none_or(|size| meta.len() == size),
            Err(_) => false,
        };
        if !ok {
            bad.push(file.relative_path.clone());
        }
    }
    if bad.is_empty() {
        CompletenessResult::Complete
    } else {
        CompletenessResult::Incomplete { missing_or_mismatched: bad }
    }
}

/// The full HF-auth environment for the `hf` subprocess, mirroring
/// upstream's `huggingface.py::huggingface_environment()`: `HF_TOKEN` from
/// the process environment first, falling back to cockpit's own saved
/// top-level `hf_token` config.json setting (read-only reuse — see design
/// doc §10's flagged open item on whether brainrouter should ever *write*
/// this value). Always sets `HF_XET_HIGH_PERFORMANCE=1`, matching upstream.
fn hf_subprocess_env() -> Vec<(String, String)> {
    let mut env = vec![("HF_XET_HIGH_PERFORMANCE".to_string(), "1".to_string())];
    if std::env::var_os("HF_TOKEN").is_none() {
        if let Some(token) = crate::cockpit_config::load().config.and_then(|c| c.hf_token) {
            env.push(("HF_TOKEN".to_string(), token));
        }
    }
    env
}

/// Read-only local-presence sweep across every download-capable backend's
/// catalog entries (`llama_cpp` excluded — no completeness check is
/// possible without a chosen quant/pattern; `vllm` excluded per module
/// docs). Used by `GET /api/model-downloads/status`.
pub fn local_presence_snapshot() -> Result<Vec<ModelPresence>, String> {
    let (_toolboxes, models) =
        toolbox_catalog::load_effective_typed_catalog().map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for backend_catalog in &models.backends {
        let Ok(backend) = SupportedServingBackend::try_from(&backend_catalog.backend) else {
            continue;
        };
        // llama_cpp needs a quant_pattern to know what "complete" even
        // means for a given entry — omitted from this backend-wide sweep,
        // same as the per-backend completeness matrix already documents.
        if backend == SupportedServingBackend::LlamaCpp {
            continue;
        }
        let models_dir = effective_models_dir(backend, &backend_catalog.storage);
        for entry in &backend_catalog.entries {
            let Some(payload) = &entry.payload else { continue };
            let built = match build_download(backend, payload, &models_dir, None) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let ple_ready = match payload {
                ModelPayload::R9v(m) if backend == SupportedServingBackend::R9v => m
                    .ple
                    .as_ref()
                    .map(|ple| r9v_ple_ready(&effective_r9v_paths(&backend_catalog.storage).ple_dir, ple)),
                _ => None,
            };
            out.push(ModelPresence {
                backend,
                model_id: entry.id.clone(),
                destination: built.destination.display().to_string(),
                completeness: check_completeness(&built.expected_files, &built.destination),
                ple_ready,
            });
        }
    }
    Ok(out)
}

/// An already-downloaded ds4 catalog model's on-disk location, resolved for
/// PR7's Server Mode (`src/server_mode.rs`) — analogous to what `-m
/// /models/<rel_path>` needs in upstream's `build_server_cmd()`, but
/// computed from the catalog + local completeness check rather than
/// accepting a raw filesystem path from the request body (§12 item 3).
pub struct ResolvedDs4Model {
    pub models_dir: PathBuf,
    pub filename: String,
}

/// Resolves `model_id` (a ds4 catalog entry id, same id space as
/// [`StartDownloadRequest::model_id`]) to its on-disk location, rejecting
/// the request if the model is unknown or not yet fully downloaded — Server
/// Mode must never be pointed at a partial/missing download. Mirrors
/// [`local_presence_snapshot`]'s own completeness check for a single entry
/// instead of every entry.
pub fn resolve_downloaded_ds4_model(model_id: &str) -> Result<ResolvedDs4Model, DownloadError> {
    let (payload, storage) = resolve_catalog_entry(SupportedServingBackend::Ds4, model_id)?;
    let ModelPayload::Ds4(m) = &payload else {
        return Err(DownloadError::Internal(
            "ds4 catalog entry did not carry a Ds4 payload".to_string(),
        ));
    };
    let models_dir = effective_models_dir(SupportedServingBackend::Ds4, &storage);
    let built = build_download(SupportedServingBackend::Ds4, &payload, &models_dir, None)?;
    match check_completeness(&built.expected_files, &built.destination) {
        CompletenessResult::Complete => {}
        other => {
            return Err(DownloadError::Validation(format!(
                "ds4 model `{model_id}` is not fully downloaded yet ({other:?}) — download it \
                 first via the Models tab before starting a server for it"
            )));
        }
    }
    Ok(ResolvedDs4Model { models_dir, filename: m.filename.clone() })
}

/// How to serve a resolved gufo model: autoregressive (no draft) or DFlash2
/// speculative decoding with a downloaded draft GGUF. Filenames are relative to
/// the shared gufo models directory (mounted at `/models` in the server
/// container). Design doc DI-6 (round-4 B1/B2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GufoServePlan {
    Autoregressive { main_filename: String },
    Dflash2 { main_filename: String, draft_filename: String },
}

/// An already-downloaded gufo model resolved for Server Mode: the shared models
/// directory (host side of the `-v <models_dir>:/models:ro` mount) plus the
/// serve plan. Analogous to [`ResolvedDs4Model`], but carries the
/// main + optional-draft pairing.
pub struct ResolvedGufoModel {
    pub models_dir: PathBuf,
    pub plan: GufoServePlan,
}

/// Pure core of [`resolve_downloaded_gufo_model`] (design doc round-4 I2):
/// decides the serve plan for a `role=main` gufo entry given a way to look up
/// its draft entry and a way to check any entry's on-disk completeness. Kept
/// filesystem-free so all three cases (AR, complete DFlash2, missing draft) are
/// unit-testable without downloads. A `role=main` model with a declared
/// speculative draft is servable **only** when both the main and the draft are
/// complete (design doc D7); a model with no `speculative` serves AR.
fn plan_from_entry(
    main: &GufoModel,
    lookup_draft: impl Fn(&str) -> Option<GufoModel>,
    is_complete: impl Fn(&GufoModel) -> bool,
) -> Result<GufoServePlan, DownloadError> {
    if main.role != GufoRole::Main {
        return Err(DownloadError::Validation(format!(
            "gufo model `{}` is a draft model, not a servable main model",
            main.id
        )));
    }
    let main_filename = main
        .files
        .first()
        .ok_or_else(|| DownloadError::Internal(format!("gufo model `{}` has no file", main.id)))?
        .path
        .clone();
    if !is_complete(main) {
        return Err(DownloadError::Validation(format!(
            "gufo model `{}` is not fully downloaded yet — download it first via the Models tab \
             before starting a server for it",
            main.id
        )));
    }

    let Some(spec) = &main.speculative else {
        return Ok(GufoServePlan::Autoregressive { main_filename });
    };

    let draft = lookup_draft(&spec.draft_model_id).ok_or_else(|| {
        DownloadError::Internal(format!(
            "gufo model `{}` references unknown draft catalog entry `{}`",
            main.id, spec.draft_model_id
        ))
    })?;
    if draft.role != GufoRole::Draft {
        return Err(DownloadError::Internal(format!(
            "gufo draft catalog entry `{}` is not role=draft",
            draft.id
        )));
    }
    let draft_filename = draft
        .files
        .first()
        .ok_or_else(|| DownloadError::Internal(format!("gufo draft `{}` has no file", draft.id)))?
        .path
        .clone();
    if !is_complete(&draft) {
        return Err(DownloadError::Validation(format!(
            "gufo model `{}` needs its DFlash2 draft `{}` downloaded first — download the draft \
             via the Models tab before starting this server",
            main.id, draft.id
        )));
    }
    match spec.mode {
        GufoSpeculativeMode::Dflash2 => Ok(GufoServePlan::Dflash2 {
            main_filename,
            draft_filename,
        }),
    }
}

/// Resolves `model_id` (a `role=main` gufo catalog entry id) to its on-disk
/// serve plan, rejecting the request if the main — or, when it declares one,
/// its DFlash2 draft — is unknown or not fully downloaded. Server Mode must
/// never be pointed at a partial/missing download. Mirrors
/// [`resolve_downloaded_ds4_model`], but returns the models dir + serve plan.
pub fn resolve_downloaded_gufo_model(model_id: &str) -> Result<ResolvedGufoModel, DownloadError> {
    let (payload, storage) = resolve_catalog_entry(SupportedServingBackend::Gufo, model_id)?;
    let ModelPayload::Gufo(main) = &payload else {
        return Err(DownloadError::Internal(
            "gufo catalog entry did not carry a Gufo payload".to_string(),
        ));
    };
    let models_dir = effective_models_dir(SupportedServingBackend::Gufo, &storage);

    // All gufo entries share the one gufo models dir, so completeness is a
    // per-entry build_download + check_completeness in that dir.
    let is_complete = |m: &GufoModel| -> bool {
        let payload = ModelPayload::Gufo(m.clone());
        match build_download(SupportedServingBackend::Gufo, &payload, &models_dir, None) {
            Ok(built) => matches!(
                check_completeness(&built.expected_files, &built.destination),
                CompletenessResult::Complete
            ),
            Err(_) => false,
        }
    };
    let lookup_draft = |id: &str| -> Option<GufoModel> {
        match resolve_catalog_entry(SupportedServingBackend::Gufo, id) {
            Ok((ModelPayload::Gufo(d), _)) => Some(d),
            _ => None,
        }
    };

    let plan = plan_from_entry(main, lookup_draft, is_complete)?;
    Ok(ResolvedGufoModel { models_dir, plan })
}

/// An already-downloaded halogen catalog bundle's on-disk location,
/// resolved for PR8's Server Mode — analogous to [`ResolvedDs4Model`] but
/// shaped for halogen's multi-file "HGN bundle" (a checkpoint + precision
/// overlay + flat tokenizer, optionally a vision tower), matching what
/// upstream's `runner.py::build_server_cmd()` reads off its own `bundle`
/// dict (`bundle['checkpoint']`/`bundle['overlay']`/`bundle['tokenizer_dir']`/
/// `bundle.get('vision_tower')`) to build the `HALOGEN_*` env vars.
pub struct ResolvedHalogenBundle {
    pub models_dir: PathBuf,
    pub checkpoint: String,
    pub overlay: String,
    pub tokenizer_dir: String,
    /// Only 2 of the 4 vendored bundles carry this (the "+vision" variants)
    /// — absent for the other 2, mirroring upstream's `bundle.get(...)`
    /// (not a required key).
    pub vision_tower: Option<String>,
}

/// Resolves `bundle_id` (a halogen catalog entry id) to its on-disk
/// location, rejecting the request if the bundle is unknown or any of its
/// manifest files is missing/incomplete — mirrors upstream's own
/// `incomplete_files(bundle, directory)` gate inside `build_server_cmd`,
/// which raises before ever constructing the podman command.
pub fn resolve_downloaded_halogen_bundle(bundle_id: &str) -> Result<ResolvedHalogenBundle, DownloadError> {
    let (payload, storage) = resolve_catalog_entry(SupportedServingBackend::Halogen, bundle_id)?;
    let ModelPayload::Halogen(m) = &payload else {
        return Err(DownloadError::Internal(
            "halogen catalog entry did not carry a Halogen payload".to_string(),
        ));
    };
    let models_dir = effective_models_dir(SupportedServingBackend::Halogen, &storage);
    let built = build_download(SupportedServingBackend::Halogen, &payload, &models_dir, None)?;
    match check_completeness(&built.expected_files, &built.destination) {
        CompletenessResult::Complete => {}
        other => {
            return Err(DownloadError::Validation(format!(
                "halogen bundle `{bundle_id}` is not fully downloaded yet ({other:?}) — download it \
                 first via the Models tab before starting a server for it"
            )));
        }
    }
    let vision_tower = m.extra.get("vision_tower").and_then(serde_json::Value::as_str).map(str::to_string);
    Ok(ResolvedHalogenBundle {
        models_dir,
        checkpoint: m.checkpoint.clone(),
        overlay: m.overlay.clone(),
        tokenizer_dir: m.tokenizer_dir.clone(),
        vision_tower,
    })
}

/// R9V's three named host directories (design doc §15 item 6 — not one or
/// four): `models_dir` (the downloaded package, resolved the same
/// override-then-catalog-default precedence as every other backend via
/// [`effective_models_dir`]), `ple_dir` (where the prepared PLE file
/// lives) and `cache_dir` (compilation/runtime cache, mounted read-write).
/// `ple_dir`/`cache_dir` have no catalog `storage.default` — same as
/// vllm's cache directories — so they fall back to upstream's own
/// hardcoded defaults (`runner.py::default_paths()`) rather than a
/// catalog-driven one.
#[derive(Debug, Clone, PartialEq)]
pub struct R9vPaths {
    pub models_dir: PathBuf,
    pub ple_dir: PathBuf,
    pub cache_dir: PathBuf,
}

/// Resolves R9V's effective paths: a cockpit `backends.r9v.<key>`
/// config.json override first, falling back to upstream's own defaults —
/// mirrors `server_mode.rs::effective_vllm_cache_dirs()`'s exact
/// precedence and reuses this module's own [`effective_models_dir`] for
/// `models_dir` (it *does* have a catalog `storage.default`, verified
/// against the vendored fixture's r9v entry).
pub fn effective_r9v_paths(storage: &serde_json::Value) -> R9vPaths {
    let models_dir = effective_models_dir(SupportedServingBackend::R9v, storage);
    let cockpit = crate::cockpit_config::load();
    let settings = cockpit.config.as_ref().and_then(|c| c.backends.get("r9v"));
    let get = |key: &str, default: &str| -> PathBuf {
        settings
            .and_then(|s| s.extra.get(key))
            .and_then(serde_json::Value::as_str)
            .map(expand_tilde)
            .unwrap_or_else(|| expand_tilde(default))
    };
    R9vPaths {
        models_dir,
        ple_dir: get("ple_dir", "~/r9v-data"),
        cache_dir: get("cache_dir", "~/r9v-data/cache-toolbox-rocm10"),
    }
}

/// The single large file R9V's "Prepare PLE" job extracts into `ple_dir`
/// (`per_layer_token_embd.iq4_nl.bin` in the vendored fixture). Mirrors
/// upstream's own `model_manager.py::ple_ready()`: existence + exact
/// `size_bytes` match, never a hash check (design doc §15's flagged open
/// item — `sha256` is carried on [`toolbox_catalog::R9vPleFile`] but not
/// verified here, a possible future strengthening, not silently added).
fn r9v_ple_ready(ple_dir: &Path, ple: &toolbox_catalog::R9vPleFile) -> bool {
    let expected = [ExpectedFile {
        relative_path: ple.filename.clone(),
        expected_size_bytes: Some(ple.size_bytes),
        sha256: None,
    }];
    matches!(check_completeness(&expected, ple_dir), CompletenessResult::Complete)
}

/// An already-downloaded R9V package's on-disk location plus its PLE
/// (per-layer token embedding) preparation readiness — resolved for PR11's
/// Server Mode (design doc §15) and for the Downloads tab's own
/// "package: complete/incomplete, PLE: ready/not prepared" status display.
/// Two independent gates, mirrored from upstream's own
/// `incomplete_files()` + `ple_ready()` checks in `model_manager.py`.
pub struct ResolvedR9vPackage {
    pub models_dir: PathBuf,
    pub ple_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub ple_filename: String,
    pub ple_size_bytes: u64,
    pub ple_ready: bool,
}

/// Resolves `package_id` (an r9v catalog entry id) to its on-disk package
/// location and PLE readiness, rejecting the request if the package itself
/// isn't fully downloaded yet — but **not** if the PLE isn't prepared yet;
/// that's a distinct, actionable state the caller (Server Mode) surfaces
/// separately, so a user blocked on "prepare PLE first" isn't told the
/// misleading "download it first" message instead (§15 item 2).
pub fn resolve_downloaded_r9v_package(package_id: &str) -> Result<ResolvedR9vPackage, DownloadError> {
    let (payload, storage) = resolve_catalog_entry(SupportedServingBackend::R9v, package_id)?;
    let ModelPayload::R9v(m) = &payload else {
        return Err(DownloadError::Internal("r9v catalog entry did not carry an R9v payload".to_string()));
    };
    let ple = m.ple.as_ref().ok_or_else(|| {
        DownloadError::Internal(format!("r9v catalog entry `{package_id}` has no `ple` metadata"))
    })?;
    let paths = effective_r9v_paths(&storage);
    let built = build_download(SupportedServingBackend::R9v, &payload, &paths.models_dir, None)?;
    match check_completeness(&built.expected_files, &built.destination) {
        CompletenessResult::Complete => {}
        other => {
            return Err(DownloadError::Validation(format!(
                "r9v package `{package_id}` is not fully downloaded yet ({other:?}) — download it \
                 first via the Models tab before preparing PLE or starting a server for it"
            )));
        }
    }
    let ple_ready = r9v_ple_ready(&paths.ple_dir, ple);
    Ok(ResolvedR9vPackage {
        models_dir: paths.models_dir,
        ple_dir: paths.ple_dir,
        cache_dir: paths.cache_dir,
        ple_filename: ple.filename.clone(),
        ple_size_bytes: ple.size_bytes,
        ple_ready,
    })
}

/// Looks up an r9v catalog toolbox's image by id, for the "Prepare PLE"
/// job's `podman run <image> r9v-model prepare` invocation — a plain
/// catalog lookup, not [`crate::server_mode::resolve_toolbox_for_server`]'s
/// `features.server`-gated one, since preparing the PLE has nothing to do
/// with Server Mode's own feature gate.
fn resolve_r9v_toolbox_image(toolbox_id: &str) -> Result<String, DownloadError> {
    let (catalog, _models) = toolbox_catalog::load_effective_typed_catalog()
        .map_err(|e| DownloadError::Internal(e.to_string()))?;
    let tb = catalog
        .toolbox_by_id(toolbox_id)
        .filter(|t| t.supported_backend() == Some(SupportedServingBackend::R9v))
        .ok_or_else(|| DownloadError::NotFound(format!("no r9v catalog toolbox with id `{toolbox_id}`")))?;
    Ok(tb.image.clone())
}

/// `POST /api/model-downloads/r9v/prepare-ple` request body.
#[derive(Debug, Clone, Deserialize)]
pub struct StartPreparePleRequest {
    /// Which vendored r9v toolbox to source the image from (same id space
    /// as [`crate::server_mode::StartR9vServerRequest::toolbox_id`]).
    pub toolbox_id: String,
    /// Which already-downloaded r9v catalog package to prepare the PLE
    /// for (same id space as [`StartDownloadRequest::model_id`]).
    pub package_id: String,
}

/// Streaming SHA256 verification for a backend's catalog entry (only
/// meaningful when its files carry `sha256` — today, only `r9v`; mirrors
/// upstream's explicit, expensive, not-run-automatically `verify_package()`).
/// Runs on a blocking thread since it reads entire files.
pub async fn verify_checksums(backend: SupportedServingBackend, model_id: &str) -> Result<Vec<FileVerification>, DownloadError> {
    let (payload, storage) = resolve_catalog_entry(backend, model_id)?;
    let models_dir = effective_models_dir(backend, &storage);
    let built = build_download(backend, &payload, &models_dir, None)?;
    let with_hashes: Vec<ExpectedFile> = built
        .expected_files
        .into_iter()
        .filter(|f| f.sha256.is_some())
        .collect();
    if with_hashes.is_empty() {
        return Err(DownloadError::Validation(format!(
            "no sha256 checksums available in the catalog for `{}`/{model_id} (design doc §10 — only r9v carries per-file hashes today)",
            backend.as_str()
        )));
    }
    let destination = built.destination;
    tokio::task::spawn_blocking(move || {
        with_hashes
            .into_iter()
            .map(|file| {
                let path = destination.join(&file.relative_path);
                let expected = file.sha256.clone().unwrap_or_default();
                let actual = sha256_file(&path).ok();
                let matched = actual.as_deref() == Some(expected.as_str());
                FileVerification { relative_path: file.relative_path, expected_sha256: expected, actual_sha256: actual, matched }
            })
            .collect::<Vec<_>>()
    })
    .await
    .map_err(|e| DownloadError::Internal(format!("verification task panicked: {e}")))
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// The job registry: single-flight execution (mirrors `benchmark_lab.rs`'s
/// `Semaphore::new(1)`), in-memory job records, real OS-level cancellation.
pub struct ModelDownloadRegistry {
    execution_slot: Arc<Semaphore>,
    jobs: Mutex<HashMap<String, Arc<JobControl>>>,
    /// Insertion order, for the history cap — oldest terminal jobs are
    /// evicted from `jobs` once the count exceeds `MAX_JOB_HISTORY`.
    history: Mutex<VecDeque<String>>,
}

impl Default for ModelDownloadRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelDownloadRegistry {
    pub fn new() -> Self {
        Self {
            execution_slot: Arc::new(Semaphore::new(1)),
            jobs: Mutex::new(HashMap::new()),
            history: Mutex::new(VecDeque::new()),
        }
    }

    pub async fn start(self: &Arc<Self>, request: StartDownloadRequest) -> Result<ModelDownloadJob, DownloadError> {
        let catalog_id = crate::toolbox_catalog::types::CatalogBackendId::from_str(&request.backend);
        let backend = SupportedServingBackend::try_from(&catalog_id)
            .map_err(|_| DownloadError::Validation(format!("unsupported or unknown backend `{}`", request.backend)))?;
        let (payload, storage) = resolve_catalog_entry(backend, &request.model_id)?;
        let models_dir = effective_models_dir(backend, &storage);
        let built = build_download(backend, &payload, &models_dir, request.quant_pattern.as_deref())?;

        let id = uuid::Uuid::new_v4().to_string();
        let command_display = format!("{} {}", hf_binary(), built.args.join(" "));
        let now = Utc::now();
        let record = ModelDownloadJob {
            id,
            kind: JobKind::Download,
            backend,
            model_id: request.model_id.clone(),
            destination: built.destination.display().to_string(),
            status: DownloadStatus::Queued,
            message: "Waiting for the download execution slot".to_string(),
            command_display,
            output_tail: String::new(),
            queued_at: now,
            started_at: None,
            ended_at: None,
        };

        let (control, cancel_rx) = self.register_job(record.clone()).await;

        let registry = Arc::clone(self);
        let expected_files = built.expected_files;
        let args = built.args;
        let destination = built.destination;
        tokio::spawn(async move {
            registry.execute(control, args, destination, expected_files, cancel_rx).await;
        });

        Ok(record)
    }

    /// `POST /api/model-downloads/r9v/prepare-ple`: runs r9v's one-shot
    /// `podman run --network=none ... r9v-model prepare` job, sharing this
    /// registry's execution slot/history/cancellation with `hf download`
    /// jobs (PR11, §15 item 3 — see [`JobKind`]'s doc comment for the
    /// explicit tradeoff). Rejects the request if the package itself isn't
    /// fully downloaded yet (mirrors [`resolve_downloaded_r9v_package`]'s
    /// own gate) — the same "download it first" message a Server Mode
    /// start attempt would get.
    pub async fn start_prepare_ple(self: &Arc<Self>, req: &StartPreparePleRequest) -> Result<ModelDownloadJob, DownloadError> {
        let image = resolve_r9v_toolbox_image(&req.toolbox_id)?;
        let resolved = resolve_downloaded_r9v_package(&req.package_id)?;

        let args: Vec<String> = vec![
            "run".to_string(),
            "--rm".to_string(),
            "--network=none".to_string(),
            "--user".to_string(),
            "0:0".to_string(),
            "--security-opt".to_string(),
            "label=disable".to_string(),
            "-v".to_string(),
            format!("{}:/models:ro", resolved.models_dir.display()),
            "-v".to_string(),
            format!("{}:/ple", resolved.ple_dir.display()),
            image,
            "r9v-model".to_string(),
            "prepare".to_string(),
        ];

        let id = uuid::Uuid::new_v4().to_string();
        let command_display = format!("podman {}", args.join(" "));
        let now = Utc::now();
        let record = ModelDownloadJob {
            id,
            kind: JobKind::PreparePle,
            backend: SupportedServingBackend::R9v,
            model_id: req.package_id.clone(),
            destination: resolved.ple_dir.display().to_string(),
            status: DownloadStatus::Queued,
            message: "Waiting for the job execution slot".to_string(),
            command_display,
            output_tail: String::new(),
            queued_at: now,
            started_at: None,
            ended_at: None,
        };

        let (control, cancel_rx) = self.register_job(record.clone()).await;

        let registry = Arc::clone(self);
        let ple_dir = resolved.ple_dir;
        let expected_files = vec![ExpectedFile {
            relative_path: resolved.ple_filename,
            expected_size_bytes: Some(resolved.ple_size_bytes),
            sha256: None,
        }];
        tokio::spawn(async move {
            registry.execute_prepare_ple(control, args, ple_dir, expected_files, cancel_rx).await;
        });

        Ok(record)
    }

    /// Inserts a freshly-built job record into the registry (job map,
    /// history, and eviction) and opens its cancellation channel — shared
    /// by [`Self::start`] and [`Self::start_prepare_ple`]; everything from
    /// here on is identical regardless of which subprocess the job runs.
    async fn register_job(self: &Arc<Self>, record: ModelDownloadJob) -> (Arc<JobControl>, watch::Receiver<bool>) {
        let id = record.id.clone();
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let control = Arc::new(JobControl { record: RwLock::new(record), cancel: cancel_tx });
        {
            let mut jobs = self.jobs.lock().await;
            jobs.insert(id.clone(), Arc::clone(&control));
        }
        {
            let mut history = self.history.lock().await;
            history.push_back(id);
        }
        self.evict_old_jobs().await;
        (control, cancel_rx)
    }

    async fn evict_old_jobs(&self) {
        let mut history = self.history.lock().await;
        if history.len() <= MAX_JOB_HISTORY {
            return;
        }
        let mut jobs = self.jobs.lock().await;
        while history.len() > MAX_JOB_HISTORY {
            if let Some(oldest) = history.pop_front() {
                // Never evict a job that's still active — only terminal
                // ones count against the cap. If the oldest is still
                // running, put it back and stop (it'll be evicted once
                // terminal and no longer the oldest).
                let still_active = jobs
                    .get(&oldest)
                    .map(|c| {
                        let record = c.record.try_read();
                        record.map(|r| is_terminal(r.status)).unwrap_or(true)
                    })
                    .unwrap_or(true);
                if still_active {
                    jobs.remove(&oldest);
                } else {
                    history.push_front(oldest);
                    break;
                }
            }
        }
    }

    async fn execute(
        self: Arc<Self>,
        control: Arc<JobControl>,
        args: Vec<String>,
        destination: PathBuf,
        expected_files: Vec<ExpectedFile>,
        mut cancel_rx: watch::Receiver<bool>,
    ) {
        let permit = tokio::select! {
            permit = Arc::clone(&self.execution_slot).acquire_owned() => match permit {
                Ok(permit) => permit,
                Err(_) => {
                    self.finish(&control, DownloadStatus::Failed, "execution slot closed".to_string()).await;
                    return;
                }
            },
            _ = cancel_rx.changed() => {
                self.finish(&control, DownloadStatus::Cancelled, "cancelled while queued".to_string()).await;
                return;
            }
        };

        if let Err(e) = tokio::fs::create_dir_all(&destination).await {
            drop(permit);
            self.finish(&control, DownloadStatus::Failed, format!("failed to create destination directory: {e}")).await;
            return;
        }

        {
            let mut record = control.record.write().await;
            record.status = DownloadStatus::Downloading;
            record.message = "Downloading".to_string();
            record.started_at = Some(Utc::now());
        }

        let mut command = Command::new(hf_binary());
        command.args(&args).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
        for (key, value) in hf_subprocess_env() {
            command.env(key, value);
        }

        let mut child = match command.spawn() {
            Ok(c) => c,
            Err(e) => {
                drop(permit);
                self.finish(
                    &control,
                    DownloadStatus::Failed,
                    format!("failed to exec `{}`: {e} (is the `hf` CLI installed and on PATH?)", hf_binary()),
                )
                .await;
                return;
            }
        };

        let tail: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        if let Some(stdout) = child.stdout.take() {
            tokio::spawn(drain_into_tail(stdout, Arc::clone(&tail)));
        }
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(drain_into_tail(stderr, Arc::clone(&tail)));
        }

        let outcome = tokio::select! {
            status = child.wait() => Outcome::Exited(status),
            _ = cancel_rx.changed() => {
                if let Err(e) = child.start_kill() {
                    warn!(error = %e, "Failed to send kill to model-download subprocess");
                }
                let _ = child.wait().await;
                Outcome::Cancelled
            }
        };
        drop(permit);

        let final_tail = tail.lock().await.clone();
        {
            let mut record = control.record.write().await;
            record.output_tail = final_tail;
        }

        match outcome {
            Outcome::Cancelled => {
                self.finish(&control, DownloadStatus::Cancelled, "cancelled by request".to_string()).await;
            }
            Outcome::Exited(Ok(status)) if status.success() => {
                {
                    let mut record = control.record.write().await;
                    record.status = DownloadStatus::Verifying;
                    record.message = "Checking completeness".to_string();
                }
                match check_completeness(&expected_files, &destination) {
                    CompletenessResult::Incomplete { missing_or_mismatched } => {
                        self.finish(
                            &control,
                            DownloadStatus::Failed,
                            format!("hf download exited successfully but files are missing/wrong size: {}", missing_or_mismatched.join(", ")),
                        )
                        .await;
                    }
                    CompletenessResult::Complete => {
                        self.finish(&control, DownloadStatus::Complete, "Download complete".to_string()).await;
                    }
                    CompletenessResult::NotChecked { reason } => {
                        self.finish(&control, DownloadStatus::Complete, format!("Download finished ({reason})")).await;
                    }
                }
            }
            Outcome::Exited(Ok(status)) => {
                self.finish(&control, DownloadStatus::Failed, format!("hf download exited with {status}")).await;
            }
            Outcome::Exited(Err(e)) => {
                error!(error = %e, "Failed to wait on model-download subprocess");
                self.finish(&control, DownloadStatus::Failed, format!("failed to wait on subprocess: {e}")).await;
            }
        }
    }

    /// The r9v "Prepare PLE" counterpart to [`Self::execute`] — same
    /// slot/cancel/tail/history-registry shape, but runs `podman` (not
    /// `hf`) with no special env, and reports `Running`/"Preparing PLE"
    /// instead of `Downloading`/"Downloading" so the dashboard never shows
    /// a misleading download badge for this job (PR11, §15 item 3). Kept
    /// as its own method rather than further-parametrizing [`Self::execute`]
    /// — the two subprocess shapes (binary, env, in-progress wording) differ
    /// enough that threading yet more parameters through the existing,
    /// well-tested download path was judged a worse tradeoff than this
    /// small amount of duplication.
    async fn execute_prepare_ple(
        self: Arc<Self>,
        control: Arc<JobControl>,
        args: Vec<String>,
        destination: PathBuf,
        expected_files: Vec<ExpectedFile>,
        mut cancel_rx: watch::Receiver<bool>,
    ) {
        let permit = tokio::select! {
            permit = Arc::clone(&self.execution_slot).acquire_owned() => match permit {
                Ok(permit) => permit,
                Err(_) => {
                    self.finish(&control, DownloadStatus::Failed, "execution slot closed".to_string()).await;
                    return;
                }
            },
            _ = cancel_rx.changed() => {
                self.finish(&control, DownloadStatus::Cancelled, "cancelled while queued".to_string()).await;
                return;
            }
        };

        if let Err(e) = tokio::fs::create_dir_all(&destination).await {
            drop(permit);
            self.finish(&control, DownloadStatus::Failed, format!("failed to create ple_dir: {e}")).await;
            return;
        }

        {
            let mut record = control.record.write().await;
            record.status = DownloadStatus::Running;
            record.message = "Preparing PLE".to_string();
            record.started_at = Some(Utc::now());
        }

        let mut command = Command::new("podman");
        command.args(&args).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);

        let mut child = match command.spawn() {
            Ok(c) => c,
            Err(e) => {
                drop(permit);
                self.finish(
                    &control,
                    DownloadStatus::Failed,
                    format!("failed to exec `podman`: {e} (is podman installed and on PATH?)"),
                )
                .await;
                return;
            }
        };

        let tail: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        if let Some(stdout) = child.stdout.take() {
            tokio::spawn(drain_into_tail(stdout, Arc::clone(&tail)));
        }
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(drain_into_tail(stderr, Arc::clone(&tail)));
        }

        let outcome = tokio::select! {
            status = child.wait() => Outcome::Exited(status),
            _ = cancel_rx.changed() => {
                if let Err(e) = child.start_kill() {
                    warn!(error = %e, "Failed to send kill to prepare-ple subprocess");
                }
                let _ = child.wait().await;
                Outcome::Cancelled
            }
        };
        drop(permit);

        let final_tail = tail.lock().await.clone();
        {
            let mut record = control.record.write().await;
            record.output_tail = final_tail;
        }

        match outcome {
            Outcome::Cancelled => {
                self.finish(&control, DownloadStatus::Cancelled, "cancelled by request".to_string()).await;
            }
            Outcome::Exited(Ok(status)) if status.success() => {
                {
                    let mut record = control.record.write().await;
                    record.status = DownloadStatus::Verifying;
                    record.message = "Checking PLE readiness".to_string();
                }
                match check_completeness(&expected_files, &destination) {
                    CompletenessResult::Incomplete { missing_or_mismatched } => {
                        self.finish(
                            &control,
                            DownloadStatus::Failed,
                            format!(
                                "r9v-model prepare exited successfully but the PLE file is missing/wrong size: {}",
                                missing_or_mismatched.join(", ")
                            ),
                        )
                        .await;
                    }
                    CompletenessResult::Complete | CompletenessResult::NotChecked { .. } => {
                        self.finish(&control, DownloadStatus::Complete, "PLE prepared".to_string()).await;
                    }
                }
            }
            Outcome::Exited(Ok(status)) => {
                self.finish(&control, DownloadStatus::Failed, format!("r9v-model prepare exited with {status}")).await;
            }
            Outcome::Exited(Err(e)) => {
                error!(error = %e, "Failed to wait on prepare-ple subprocess");
                self.finish(&control, DownloadStatus::Failed, format!("failed to wait on subprocess: {e}")).await;
            }
        }
    }

    async fn finish(&self, control: &Arc<JobControl>, status: DownloadStatus, message: String) {
        let mut record = control.record.write().await;
        record.status = status;
        record.message = message;
        record.ended_at = Some(Utc::now());
    }

    pub async fn get(&self, id: &str) -> Result<ModelDownloadJob, DownloadError> {
        let jobs = self.jobs.lock().await;
        let control = jobs.get(id).ok_or_else(|| DownloadError::NotFound(format!("no such job `{id}`")))?;
        let record = control.record.read().await.clone();
        Ok(record)
    }

    pub async fn list(&self, limit: usize) -> Vec<ModelDownloadJob> {
        let jobs = self.jobs.lock().await;
        let mut records: Vec<ModelDownloadJob> = Vec::with_capacity(jobs.len());
        for control in jobs.values() {
            records.push(control.record.read().await.clone());
        }
        records.sort_by_key(|r| std::cmp::Reverse(r.queued_at));
        records.truncate(limit);
        records
    }

    pub async fn cancel(&self, id: &str) -> Result<ModelDownloadJob, DownloadError> {
        let control = {
            let jobs = self.jobs.lock().await;
            Arc::clone(jobs.get(id).ok_or_else(|| DownloadError::NotFound(format!("no such job `{id}`")))?)
        };
        {
            let record = control.record.read().await;
            if is_terminal(record.status) {
                return Err(DownloadError::Conflict(format!("job `{id}` already reached a terminal state ({:?})", record.status)));
            }
        }
        let _ = control.cancel.send(true);
        let record = control.record.read().await.clone();
        Ok(record)
    }
}

enum Outcome {
    Exited(std::io::Result<std::process::ExitStatus>),
    Cancelled,
}

fn is_terminal(status: DownloadStatus) -> bool {
    matches!(status, DownloadStatus::Complete | DownloadStatus::Failed | DownloadStatus::Cancelled)
}

/// Drains a child process's stdout/stderr into a shared, bounded tail
/// buffer. Both streams are captured (not just stderr) because it isn't
/// verified in this sandbox which stream `hf`'s own progress output uses
/// (design doc §10) — capturing both is strictly safer and also prevents
/// the child from blocking on a full, unread pipe either way.
async fn drain_into_tail(mut reader: impl tokio::io::AsyncRead + Unpin, tail: Arc<Mutex<String>>) {
    let mut buf = [0u8; 8192];
    loop {
        match reader.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let mut guard = tail.lock().await;
                guard.push_str(&String::from_utf8_lossy(&buf[..n]));
                if guard.len() > MAX_OUTPUT_TAIL_BYTES {
                    let excess = guard.len() - MAX_OUTPUT_TAIL_BYTES;
                    let boundary = guard
                        .char_indices()
                        .map(|(i, _)| i)
                        .find(|&i| i >= excess)
                        .unwrap_or(guard.len());
                    guard.drain(..boundary);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toolbox_catalog::{load_vendored_catalog, ModelPayload};

    #[test]
    fn expand_tilde_handles_bare_and_prefixed_paths() {
        std::env::set_var("HOME", "/home/testuser");
        assert_eq!(expand_tilde("~"), PathBuf::from("/home/testuser"));
        assert_eq!(expand_tilde("~/models"), PathBuf::from("/home/testuser/models"));
        assert_eq!(expand_tilde("/opt/models"), PathBuf::from("/opt/models"));
    }

    fn gufo_model(id: &str, role: GufoRole, draft: Option<&str>) -> GufoModel {
        GufoModel {
            id: id.to_string(),
            name: id.to_string(),
            repo: "repo/x".to_string(),
            revision: "rev".to_string(),
            role,
            files: vec![CatalogModelFile {
                path: format!("{id}.gguf"),
                size_bytes: 100,
                role: None,
                sha256: None,
            }],
            recommended: false,
            ctx_default: Some(32768),
            sessions_default: Some(2),
            speculative: draft.map(|d| crate::toolbox_catalog::GufoSpeculative {
                mode: GufoSpeculativeMode::Dflash2,
                draft_model_id: d.to_string(),
            }),
            extra: serde_json::Map::new(),
        }
    }

    #[test]
    fn plan_from_entry_autoregressive_when_no_speculative() {
        let main = gufo_model("m", GufoRole::Main, None);
        let plan = plan_from_entry(&main, |_| None, |_| true).unwrap();
        assert_eq!(
            plan,
            GufoServePlan::Autoregressive { main_filename: "m.gguf".to_string() }
        );
    }

    #[test]
    fn plan_from_entry_dflash2_when_main_and_draft_complete() {
        let main = gufo_model("m", GufoRole::Main, Some("d"));
        let draft = gufo_model("d", GufoRole::Draft, None);
        let plan = plan_from_entry(&main, move |id| (id == "d").then(|| draft.clone()), |_| true).unwrap();
        assert_eq!(
            plan,
            GufoServePlan::Dflash2 {
                main_filename: "m.gguf".to_string(),
                draft_filename: "d.gguf".to_string(),
            }
        );
    }

    #[test]
    fn plan_from_entry_errors_when_draft_incomplete() {
        let main = gufo_model("m", GufoRole::Main, Some("d"));
        let draft = gufo_model("d", GufoRole::Draft, None);
        // main complete, draft not.
        let err = plan_from_entry(&main, move |id| (id == "d").then(|| draft.clone()), |g: &GufoModel| g.id == "m")
            .unwrap_err();
        match err {
            DownloadError::Validation(msg) => assert!(msg.contains("draft"), "{msg}"),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn plan_from_entry_errors_when_main_incomplete() {
        let main = gufo_model("m", GufoRole::Main, None);
        assert!(matches!(
            plan_from_entry(&main, |_| None, |_| false),
            Err(DownloadError::Validation(_))
        ));
    }

    #[test]
    fn plan_from_entry_rejects_a_draft_role_used_as_main() {
        let draft = gufo_model("d", GufoRole::Draft, None);
        assert!(matches!(
            plan_from_entry(&draft, |_| None, |_| true),
            Err(DownloadError::Validation(_))
        ));
    }

    #[test]
    fn build_download_gufo_matches_single_repo_shape() {
        let m = gufo_model("gufo-x", GufoRole::Main, None);
        let built = build_download(
            SupportedServingBackend::Gufo,
            &ModelPayload::Gufo(m),
            std::path::Path::new("/models/gufo"),
            None,
        )
        .expect("must build");
        assert_eq!(
            built.args,
            vec![
                "download",
                "repo/x",
                "gufo-x.gguf",
                "--revision",
                "rev",
                "--local-dir",
                "/models/gufo",
            ]
        );
        // Exact-size completeness expectation carried through.
        assert_eq!(built.expected_files.len(), 1);
        assert_eq!(built.expected_files[0].expected_size_bytes, Some(100));
    }

    #[test]
    fn ds4_download_command_matches_upstream_shape() {
        let vendored = load_vendored_catalog();
        let (_toolboxes, models) = vendored.typed().expect("vendored catalog parses");
        let backend_catalog = models.backends.iter().find(|b| b.backend.as_str() == "ds4").expect("ds4 present");
        let entry = backend_catalog.entries.first().expect("at least one ds4 entry");
        let payload = entry.payload.clone().expect("typed payload");
        let ModelPayload::Ds4(_) = &payload else {
            panic!("expected a ds4 payload");
        };
        let built = build_download(SupportedServingBackend::Ds4, &payload, Path::new("/models/ds4"), None)
            .expect("ds4 needs no quant_pattern");
        assert_eq!(built.args[0], "download");
        assert!(built.args.contains(&"--local-dir".to_string()));
        assert!(!built.args.contains(&"--revision".to_string()), "ds4 pins no revision upstream");
        assert_eq!(built.expected_files.len(), 1);
        assert!(built.expected_files[0].expected_size_bytes.is_none(), "ds4's own check is file-exists-only, no size");
    }

    #[test]
    fn halogen_download_command_includes_every_file_and_revision() {
        let vendored = load_vendored_catalog();
        let (_toolboxes, models) = vendored.typed().expect("vendored catalog parses");
        let backend_catalog = models.backends.iter().find(|b| b.backend.as_str() == "halogen").expect("halogen present");
        let entry = backend_catalog.entries.first().expect("at least one halogen bundle");
        let ModelPayload::Halogen(m) = entry.payload.clone().expect("typed payload") else {
            panic!("expected halogen payload");
        };
        let built = build_multifile_download(&m.repo, &m.revision, &m.files, Path::new("/models/halogen"));
        assert!(built.args.contains(&"--revision".to_string()));
        assert!(built.args.contains(&m.revision));
        assert_eq!(built.destination, PathBuf::from("/models/halogen"), "halogen downloads flat into the backend directory, not a per-bundle subdir");
        assert_eq!(built.expected_files.len(), m.files.len());
        for f in &built.expected_files {
            assert!(f.expected_size_bytes.is_some(), "halogen entries carry size_bytes");
        }
    }

    #[test]
    fn r9v_download_command_carries_sha256_when_present() {
        let vendored = load_vendored_catalog();
        let (_toolboxes, models) = vendored.typed().expect("vendored catalog parses");
        let backend_catalog = models.backends.iter().find(|b| b.backend.as_str() == "r9v").expect("r9v present");
        let entry = backend_catalog.entries.first().expect("at least one r9v package");
        let ModelPayload::R9v(m) = entry.payload.clone().expect("typed payload") else {
            panic!("expected r9v payload");
        };
        let built = build_multifile_download(&m.repo, &m.revision, &m.files, Path::new("/models/r9v"));
        assert!(built.expected_files.iter().any(|f| f.sha256.is_some()), "r9v is the only backend with catalog sha256s");
    }

    #[test]
    fn llama_cpp_requires_quant_pattern() {
        let vendored = load_vendored_catalog();
        let (_toolboxes, models) = vendored.typed().expect("vendored catalog parses");
        let backend_catalog = models.backends.iter().find(|b| b.backend.as_str() == "llama_cpp").expect("llama_cpp present");
        let entry = backend_catalog.entries.first().expect("at least one llama_cpp entry");
        let payload = entry.payload.clone().expect("typed payload");
        let err = build_download(SupportedServingBackend::LlamaCpp, &payload, Path::new("/models"), None)
            .expect_err("missing quant_pattern must be rejected");
        assert!(matches!(err, DownloadError::Validation(_)));
    }

    #[test]
    fn llama_cpp_exact_filename_vs_glob_pattern() {
        let vendored = load_vendored_catalog();
        let (_toolboxes, models) = vendored.typed().expect("vendored catalog parses");
        let backend_catalog = models.backends.iter().find(|b| b.backend.as_str() == "llama_cpp").expect("llama_cpp present");
        let entry = backend_catalog.entries.first().expect("at least one llama_cpp entry");
        let payload = entry.payload.clone().expect("typed payload");

        let exact = build_download(SupportedServingBackend::LlamaCpp, &payload, Path::new("/models"), Some("model.q4_k_m.gguf")).unwrap();
        assert!(exact.args.last().map(String::as_str) == Some("model.q4_k_m.gguf"));
        assert!(!exact.args.contains(&"--include".to_string()));

        let glob = build_download(SupportedServingBackend::LlamaCpp, &payload, Path::new("/models"), Some("*-of-*.gguf")).unwrap();
        assert!(glob.args.contains(&"--include".to_string()));

        let folder = build_download(SupportedServingBackend::LlamaCpp, &payload, Path::new("/models"), Some("BF16")).unwrap();
        assert!(folder.args.contains(&"BF16/*".to_string()));
    }

    #[test]
    fn vllm_is_rejected_as_unsupported() {
        let err = resolve_catalog_entry(SupportedServingBackend::Vllm, "anything").expect_err("vllm has no download step");
        assert!(matches!(err, DownloadError::Validation(_)));
    }

    #[test]
    fn completeness_check_flags_missing_files() {
        let expected = vec![ExpectedFile { relative_path: "does-not-exist.bin".to_string(), expected_size_bytes: Some(123), sha256: None }];
        let result = check_completeness(&expected, Path::new("/tmp/definitely-does-not-exist-brainrouter-test"));
        assert!(matches!(result, CompletenessResult::Incomplete { .. }));
    }

    #[test]
    fn completeness_check_empty_manifest_is_not_checked() {
        let result = check_completeness(&[], Path::new("/tmp"));
        assert!(matches!(result, CompletenessResult::NotChecked { .. }));
    }

    #[test]
    fn r9v_ple_ready_checks_existence_and_exact_size_only() {
        let dir = std::env::temp_dir().join(format!("brainrouter-r9v-ple-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("create test dir");
        let ple = toolbox_catalog::R9vPleFile {
            filename: "per_layer_token_embd.iq4_nl.bin".to_string(),
            size_bytes: 5,
            sha256: "deadbeef".to_string(),
        };

        // Not present yet.
        assert!(!r9v_ple_ready(&dir, &ple));

        // Present but wrong size.
        std::fs::write(dir.join(&ple.filename), b"1234").expect("write short file");
        assert!(!r9v_ple_ready(&dir, &ple));

        // Present with the exact expected size — sha256 is deliberately
        // never checked (§15's flagged v1 scope decision).
        std::fs::write(dir.join(&ple.filename), b"12345").expect("write exact-size file");
        assert!(r9v_ple_ready(&dir, &ple));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_r9v_toolbox_image_finds_the_real_vendored_entry() {
        let image = resolve_r9v_toolbox_image("r9700-r9v-rocm-10-0").expect("must resolve");
        assert_eq!(image, "docker.io/kyuz0/amd-r9700-toolboxes:r9v-rocm-10.0");
    }

    #[test]
    fn resolve_r9v_toolbox_image_rejects_unknown_id() {
        let err = resolve_r9v_toolbox_image("no-such-toolbox").unwrap_err();
        assert!(matches!(err, DownloadError::NotFound(_)));
    }

    #[test]
    fn resolve_r9v_toolbox_image_rejects_non_r9v_backend() {
        let err = resolve_r9v_toolbox_image("strix-halo-llama-rocm-10-0").unwrap_err();
        assert!(matches!(err, DownloadError::NotFound(_)));
    }

    #[tokio::test]
    async fn registry_rejects_unknown_backend() {
        let registry = Arc::new(ModelDownloadRegistry::new());
        let err = registry
            .start(StartDownloadRequest { backend: "comfyui".to_string(), model_id: "x".to_string(), quant_pattern: None })
            .await
            .expect_err("comfyui is not a supported serving backend");
        assert!(matches!(err, DownloadError::Validation(_)));
    }

    #[tokio::test]
    async fn registry_cancel_of_unknown_job_is_not_found() {
        let registry = ModelDownloadRegistry::new();
        let err = registry.cancel("no-such-job").await.expect_err("must not find a nonexistent job");
        assert!(matches!(err, DownloadError::NotFound(_)));
    }

    #[test]
    fn resolve_in_with_skips_empty_path_entries() {
        // A leading empty PATH entry must never be probed (a stray ./hf must
        // not be reported as "found"). The spy records exactly which paths the
        // predicate is asked about — filesystem-/cwd-free, so it is race-safe.
        use std::cell::RefCell;
        let seen: RefCell<Vec<PathBuf>> = RefCell::new(Vec::new());
        let found = resolve_in_with(vec![PathBuf::new(), PathBuf::from("/real")], "hf", |p| {
            seen.borrow_mut().push(p.to_path_buf());
            false
        });
        assert!(!found);
        assert_eq!(seen.into_inner(), vec![PathBuf::from("/real/hf")]);
    }

    #[test]
    fn resolve_in_with_treats_separatored_name_as_direct_path() {
        use std::cell::RefCell;
        let seen: RefCell<Vec<PathBuf>> = RefCell::new(Vec::new());
        let _ = resolve_in_with(vec![PathBuf::from("/ignored")], "/abs/hf", |p| {
            seen.borrow_mut().push(p.to_path_buf());
            false
        });
        assert_eq!(seen.into_inner(), vec![PathBuf::from("/abs/hf")]);
    }

    #[cfg(unix)]
    #[test]
    fn resolve_in_detects_executable_and_rejects_non_executable() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let hf = dir.path().join("hf");
        std::fs::write(&hf, b"#!/bin/sh\n").expect("write");
        std::fs::set_permissions(&hf, std::fs::Permissions::from_mode(0o644)).expect("chmod 644");
        assert!(!resolve_in(vec![dir.path().to_path_buf()], "hf"));
        std::fs::set_permissions(&hf, std::fs::Permissions::from_mode(0o755)).expect("chmod 755");
        assert!(resolve_in(vec![dir.path().to_path_buf()], "hf"));
    }

    #[test]
    fn resolve_in_returns_false_when_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(!resolve_in(vec![dir.path().to_path_buf()], "hf"));
    }

    #[test]
    fn build_preflight_populates_hint_only_when_missing() {
        let found = build_preflight(true);
        assert!(found.found_on_path);
        assert_eq!(found.binary, "hf");
        assert!(found.message.is_none());
        assert!(found.install_command.is_none());

        let missing = build_preflight(false);
        assert!(!missing.found_on_path);
        assert_eq!(missing.install_command.as_deref(), Some(HF_INSTALL_COMMAND));
        assert!(missing.message.is_some());
    }

    #[test]
    fn install_command_is_documented_in_readme_and_prd() {
        // R10: bind the runtime source of truth to the docs so it cannot drift.
        assert!(
            include_str!("../README.md").contains(HF_INSTALL_COMMAND),
            "README.md must document HF_INSTALL_COMMAND verbatim",
        );
        assert!(
            include_str!("../PRD.md").contains(HF_INSTALL_COMMAND),
            "PRD.md must document HF_INSTALL_COMMAND verbatim",
        );
    }
}
