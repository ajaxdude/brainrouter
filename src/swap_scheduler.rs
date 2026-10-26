//! Swap supervisor scheduler (v2 L4).
//!
//! The scheduler owns model lifecycle: `Stopped | Loading | Ready | Draining |
//! Failed`, the in-memory ensure queue, the VRAM/GTT budget ledger
//! (`budget.json`), sized pins, in-flight-aware LRU eviction, child-held
//! per-GGUF flocks, adoption markers across daemon restarts, and the adapter
//! dispatch.
//!
//! Binding ordering (L4 spec): **Residency before budget**. `ensure(X)` first
//! asks the Residency view whether X is already serving anywhere; a resident
//! model is attached with no spawn and no budget gate. Budget admission (and
//! LRU evict-to-fit) applies only to ensures that would spawn. Pins that
//! cannot fit make every spawning ensure an explicit `NoFit`; resident
//! attaches are unaffected.
//!
//! The queue is in-memory and lost on daemon restart (clients retry); adopted
//! running loads keep serving. The flock is **held by the child process** (an
//! inherited fd with close-on-exec cleared): a surviving child keeps it across
//! daemon restarts (that is what adoption relies on) and releases it on exit.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::os::fd::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::active_consumers::{protection_keys, ActiveConsumers, ConsumerSpan};
use crate::engine_observer::{ExternalObserver, ObservedEngine};
use crate::engine_systemd::{UnitRuntime, UnitState};
use crate::inflight::InflightRegistry;
use crate::residency::{
    self, ExternalResidencyRegistry, Resident, ResidentSource, ResidencyView,
};
use crate::serving_identity::ServingIdentityRegistry;
use crate::swap_catalog::{CatalogEntry, LaunchSpec, SwapCatalog};

/// Budget ledger state file (under `$XDG_STATE_HOME/brainrouter`).
pub const BUDGET_FILE: &str = "budget.json";
/// Append-only, but bounded: the newest N entries survive a rotation at write
/// (5000 rows ≈ months of admissions/spawns at this box's rate).
pub const LEDGER_MAX_ENTRIES: usize = 5000;
/// Adoption markers directory (under the state root).
pub const ADOPTION_DIR: &str = "adoption";
/// JSON-RPC-style busy code reused from the plugin convention (Hank ruling
/// #7): the HTTP layer answers 503 with this code when a swap cannot admit.
pub const BUSY_CODE: i32 = -32005;

pub fn budget_path() -> PathBuf {
    crate::paths::brainrouter_state().join(BUDGET_FILE)
}

pub fn adoption_dir() -> PathBuf {
    crate::paths::brainrouter_state().join(ADOPTION_DIR)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum SwapError {
    UnknownModel(String),
    NoFit(String),
    QueueFull { model: String, cap: usize },
    /// Refused a stop because the model is protected (active stream or a
    /// sized pin) or another actor currently owns its spawn lock.
    Busy(String),
    SpawnFailed(String),
    Timeout { model: String, waited_ms: u64 },
    Cancelled(String),
    /// A live (or grace-protected) consumer blocks the operation.
    /// Block reason: `blocked:active-consumer`.
    BlockedActiveConsumer { model: String, detail: String },
    /// A systemd unit owns the engine and the caller did not pass the
    /// single-action `--override`.
    UnitOwned { unit: String },
    /// `MemAvailable` is below the adapter's `min_mem_gib` floor.
    GateBlocked {
        model: String,
        threshold_gib: f64,
        available_gib: f64,
    },
    /// An unregistered external engine occupies the exclusive device. Only an
    /// explicit operator action (adopt / stop it) or `--override` may proceed.
    /// Block reason: `blocked:external-occupant`.
    ExternalOccupant {
        model: String,
        pid: u32,
        endpoint: String,
    },
}

impl SwapError {
    pub fn status_code(&self) -> u16 {
        match self {
            Self::UnknownModel(_) => 404,
            Self::NoFit(_) | Self::QueueFull { .. } | Self::SpawnFailed(_) => 503,
            Self::Busy(_) => 409,
            Self::Timeout { .. } => 504,
            Self::Cancelled(_) => 499,
            Self::BlockedActiveConsumer { .. }
            | Self::GateBlocked { .. }
            | Self::ExternalOccupant { .. } => 503,
            Self::UnitOwned { .. } => 409,
        }
    }

    pub fn code(&self) -> i32 {
        match self {
            Self::NoFit(_)
            | Self::QueueFull { .. }
            | Self::BlockedActiveConsumer { .. }
            | Self::GateBlocked { .. }
            | Self::ExternalOccupant { .. } => BUSY_CODE,
            _ => 0,
        }
    }
}

impl std::fmt::Display for SwapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownModel(id) => write!(f, "model `{id}` is not in the swap catalog"),
            Self::NoFit(message) => write!(f, "no-fit ({BUSY_CODE} busy): {message}"),
            Self::QueueFull { model, cap } => write!(
                f,
                "{BUSY_CODE} busy: the swap queue is full (cap {cap}, global) — `{model}` refused"
            ),
            Self::Busy(reason) => write!(f, "refused: {reason}"),
            Self::SpawnFailed(message) => write!(f, "spawn failed: {message}"),
            Self::Timeout { model, waited_ms } => {
                write!(f, "timed out waiting for `{model}` after {waited_ms} ms")
            }
            Self::Cancelled(model) => write!(f, "cancelled while waiting for `{model}`"),
            Self::BlockedActiveConsumer { model, detail } => write!(
                f,
                "blocked:active-consumer: `{model}` has a live or grace-protected stream ({detail})"
            ),
            Self::UnitOwned { unit } => write!(
                f,
                "blocked:unit-owned: `{unit}` owns the engine; pass --override to take it over"
            ),
            Self::GateBlocked {
                model,
                threshold_gib,
                available_gib,
            } => write!(
                f,
                "blocked:min-mem-gate: `{model}` needs MemAvailable ≥ {threshold_gib:.1} GiB \
                 (probe reports {available_gib:.1} GiB); pass --force-gate to override"
            ),
            Self::ExternalOccupant {
                model,
                pid,
                endpoint,
            } => write!(
                f,
                "blocked:external-occupant: `{model}` needs the device occupied by an \
                 unregistered engine (pid {pid} at {endpoint}); run `swap adopt` or stop it, \
                 or pass --override"
            ),
        }
    }
}

impl std::error::Error for SwapError {}

// ── Child-held GGUF flock ───────────────────────────────────────────────────

/// An exclusive `flock` on the canonical GGUF, keyed on the file. The lock is
/// associated with the open file description, so passing the fd to a child
/// (with close-on-exec cleared) and closing the parent's copy leaves the child
/// holding it until it exits. The daemon must therefore spawn with
/// [`Self::inherit_fd_in`] before dropping the handle.
pub struct GgufLock {
    file: std::fs::File,
    path: PathBuf,
}

impl GgufLock {
    /// `Ok(None)` = another process (or an earlier child) holds the lock.
    pub fn try_acquire(path: &Path) -> std::io::Result<Option<Self>> {
        let file = std::fs::OpenOptions::new().read(true).open(path)?;
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc == 0 {
            Ok(Some(Self {
                file,
                path: path.to_path_buf(),
            }))
        } else {
            let error = std::io::Error::last_os_error();
            match error.raw_os_error() {
                Some(code) if code == libc::EWOULDBLOCK || code == libc::EAGAIN => Ok(None),
                _ => Err(error),
            }
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }

    /// Arrange for `command`'s child to inherit the locked fd (close-on-exec
    /// cleared). Call before `spawn`; dropping this handle afterwards keeps
    /// the lock alive in the child only.
    pub fn inherit_fd_in(&self, command: &mut std::process::Command) {
        let fd = self.fd();
        unsafe {
            use std::os::unix::process::CommandExt;
            command.pre_exec(move || {
                let flags = libc::fcntl(fd, libc::F_GETFD);
                if flags < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
}

// ── Adapter contract ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HealthReport {
    pub ready: bool,
    /// The stem SET the engine advertises (spawn-time assertion target).
    pub stems: BTreeSet<String>,
}

impl HealthReport {
    pub fn loading() -> Self {
        Self {
            ready: false,
            stems: BTreeSet::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadProgress {
    /// Scripted/known fraction 0..=1; `None` = indeterminate (native
    /// llama-server offers no load-progress endpoint).
    pub fraction: Option<f64>,
    pub elapsed_ms: u64,
    /// EWMA-derived ETA when known.
    pub eta_ms: Option<u64>,
}

impl LoadProgress {
    pub fn indeterminate() -> Self {
        Self {
            fraction: None,
            elapsed_ms: 0,
            eta_ms: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineHandle {
    pub pid: u32,
    pub endpoint: String,
    pub started_at: u64,
    #[serde(default)]
    pub scope_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SpawnRequest {
    pub model_id: String,
    pub spec: LaunchSpec,
    pub gguf: Option<PathBuf>,
    pub port: u16,
    pub endpoint: String,
    pub log_path: PathBuf,
    pub scope_name: Option<String>,
    /// Extra env from the catalog entry (fake-adapter scripting, engine knobs).
    pub env: BTreeMap<String, String>,
}

impl SpawnRequest {
    /// Resolved args with `${PORT}` substituted.
    pub fn resolved_args(&self) -> Vec<String> {
        let port = self.port.to_string();
        let mut args: Vec<String> = self
            .spec
            .args
            .iter()
            .map(|a| a.replace("${PORT}", &port))
            .collect();
        if !args.iter().any(|a| a == "--port") && !self.spec.program.is_empty() {
            args.push("--port".into());
            args.push(port);
        }
        args
    }
}

pub trait EngineAdapter: Send + Sync {
    fn name(&self) -> &'static str;
    fn probe(&self) -> Result<()> {
        Ok(())
    }
    fn spawn(&self, request: &SpawnRequest) -> Result<EngineHandle>;
    fn stop(&self, handle: &EngineHandle) -> Result<()>;
    fn health(&self, endpoint: &str, handle: &EngineHandle) -> Result<HealthReport>;
    fn progress(&self, endpoint: &str, handle: &EngineHandle) -> LoadProgress {
        let _ = (endpoint, handle);
        LoadProgress::indeterminate()
    }
    fn vram_estimate(&self, entry: &CatalogEntry) -> Option<f64> {
        entry.vram_estimate_gib
    }
    fn capabilities(&self, entry: &CatalogEntry) -> Vec<String> {
        let _ = entry;
        Vec::new()
    }
    /// L5: `MemAvailable` floor for a spawning ensure. `None` = no gate
    /// (llama.cpp/fake/shim behave exactly as L4).
    fn min_mem_gib(&self) -> Option<f64> {
        None
    }
    /// L5: exclusive device class: one resident per device. Ensures of a
    /// second model in the same class evict-then-spawn, never co-reside.
    fn exclusive_key(&self) -> Option<String> {
        None
    }
    /// L5: adapter-owned port allocation (strixite conf scope, hipfire
    /// single-owner port). `None` = the scheduler's generic free-port scan.
    fn allocate_port(&self, avoid: &BTreeSet<u16>) -> Option<u16> {
        let _ = avoid;
        None
    }
    /// L5: the systemd user unit this engine may be taken over from.
    fn takeover_unit(&self) -> Option<String> {
        None
    }
    fn conf_path_for(&self, model: &str) -> Option<PathBuf> {
        let _ = model;
        None
    }
    /// L5: readiness poll budget (`/v1/models` stem assertion). `None` = the
    /// scheduler's global `wait_timeout` (L4 behavior).
    fn readiness_timeout(&self) -> Option<Duration> {
        None
    }
    /// L5: adapter detail badge (schema/version/conf-dir) for the Engines tab.
    fn describe(&self) -> Option<serde_json::Value> {
        None
    }
}

fn pid_alive(pid: u32) -> bool {
    pid > 0 && unsafe { libc::kill(pid as i32, 0) == 0 }
}

fn process_start_time(pid: u32) -> Option<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // Field 22 (1-based) after the comm field which may contain spaces and is
    // wrapped in parens: split at the last ')'.
    let tail = &stat[stat.rfind(')')? + 1..];
    let mut fields = tail.split_whitespace();
    // tail starts at field 3 (state), so field 22 is index 19.
    fields.nth(19)?.parse::<u64>().ok()
}

/// The argv as NUL-separated on the wire — adoption matches a FULL argument
/// (never a substring: `/models/a.gguf` must not match `/models/a.gguf.bak`).
pub(crate) fn process_args(pid: u32) -> Vec<String> {
    std::fs::read(format!("/proc/{pid}/cmdline"))
        .map(|bytes| {
            bytes
                .split(|b| *b == 0)
                .filter(|part| !part.is_empty())
                .map(|part| String::from_utf8_lossy(part).into_owned())
                .collect()
        })
        .unwrap_or_default()
}

fn process_cmdline(pid: u32) -> String {
    std::fs::read(format!("/proc/{pid}/cmdline"))
        .map(|bytes| {
            bytes
                .split(|b| *b == 0)
                .map(|part| String::from_utf8_lossy(part).into_owned())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default()
}

// ── Fake adapter (acceptance substrate) ─────────────────────────────────────

/// Scripted behavior. Keys may also arrive through the catalog entry's
/// `launch_spec.env`: `FAKE_LOAD_MS`, `FAKE_STEMS` (comma-separated),
/// `FAKE_CRASH_AFTER_MS`, `FAKE_PROGRESS` (`1`/`0`).
#[derive(Debug, Clone)]
pub struct FakeScript {
    pub load_ms: u64,
    pub stems: Vec<String>,
    pub crash_after_ms: Option<u64>,
    pub progress: bool,
}

impl Default for FakeScript {
    fn default() -> Self {
        Self {
            load_ms: 0,
            stems: Vec::new(),
            crash_after_ms: None,
            progress: true,
        }
    }
}

impl FakeScript {
    pub fn from_env(env: &BTreeMap<String, String>) -> Self {
        let mut script = Self::default();
        if let Some(ms) = env.get("FAKE_LOAD_MS").and_then(|v| v.parse().ok()) {
            script.load_ms = ms;
        }
        if let Some(stems) = env.get("FAKE_STEMS") {
            script.stems = stems
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
        script.crash_after_ms = env
            .get("FAKE_CRASH_AFTER_MS")
            .and_then(|v| v.parse().ok());
        if let Some(progress) = env.get("FAKE_PROGRESS") {
            script.progress = progress != "0";
        }
        script
    }
}

struct FakeProc {
    child: std::process::Child,
    endpoint: String,
    model: String,
    spawned_ms: u64,
    script: FakeScript,
}

/// Scripted adapter with a REAL child process: `sleep` holds the process (and
/// the inherited GGUF flock) so lifecycle/ownership/flock semantics are the
/// same as a native engine. Health, stems, load duration, progress, and crashes
/// are scripted.
pub struct FakeAdapter {
    /// Keyed by PID: fake engines do not bind their endpoint, so two models
    /// can be handed the same probe-free port; the pid is the identity.
    inner: Mutex<BTreeMap<u32, FakeProc>>,
    spawns: AtomicU64,
    /// Test knob: make `stop` fail (verifies marker retention + state
    /// restoration on a failed stop).
    fail_stop: AtomicBool,
}

impl Default for FakeAdapter {
    fn default() -> Self {
        Self {
            inner: Mutex::new(BTreeMap::new()),
            spawns: AtomicU64::new(0),
            fail_stop: AtomicBool::new(false),
        }
    }
}

impl FakeAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Test knob: `stop` returns an error (child stays alive).
    pub fn set_fail_stop(&self, fail: bool) {
        self.fail_stop.store(fail, Ordering::SeqCst);
    }

    /// Number of child processes actually spawned (concurrency assertions).
    pub fn spawn_count(&self) -> u64 {
        self.spawns.load(Ordering::SeqCst)
    }

    /// Read the script for an endpoint (tests assert progress/state).
    pub fn script_for(&self, endpoint: &str) -> Option<FakeScript> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .find(|p| p.endpoint == endpoint)
            .map(|p| p.script.clone())
    }
}

impl Drop for FakeAdapter {
    fn drop(&mut self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let children: Vec<std::process::Child> = std::mem::take(&mut *inner)
            .into_values()
            .map(|p| p.child)
            .collect();
        drop(inner);
        for mut child in children {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl EngineAdapter for FakeAdapter {
    fn name(&self) -> &'static str {
        "fake"
    }

    fn spawn(&self, request: &SpawnRequest) -> Result<EngineHandle> {
        let script = FakeScript::from_env(&request.env);
        // Child-held flock: open + lock here, pass the fd to the child, then
        // drop the parent handle. The lock outlives the daemon call.
        let lock = match &request.gguf {
            Some(gguf) => match GgufLock::try_acquire(gguf) {
                Ok(Some(lock)) => Some(lock),
                Ok(None) => anyhow::bail!(
                    "GGUF `{}` is locked by another running engine",
                    gguf.display()
                ),
                Err(e) => anyhow::bail!("failed to lock `{}`: {e}", gguf.display()),
            },
            None => None,
        };
        // The child's argv must name the GGUF: it is both the flock target
        // and the adoption identity check (`/proc/<pid>/cmdline` contains the
        // model path) after a daemon restart.
        let mut command = match &request.gguf {
            Some(gguf) => {
                let mut command = std::process::Command::new("tail");
                command.arg("-f").arg(gguf);
                command.stdin(std::process::Stdio::null());
                command.stdout(std::process::Stdio::null());
                command
            }
            None => {
                let mut command = std::process::Command::new("sleep");
                command.arg("3600");
                command.stdout(std::process::Stdio::null());
                command.stderr(std::process::Stdio::null());
                command.stdin(std::process::Stdio::null());
                command
            }
        };
        if let Some(lock) = &lock {
            lock.inherit_fd_in(&mut command);
        }
        let child = command.spawn().context("spawning the fake engine child")?;
        let pid = child.id();
        drop(lock);
        self.spawns.fetch_add(1, Ordering::SeqCst);
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.insert(
            pid,
            FakeProc {
                child,
                endpoint: request.endpoint.clone(),
                model: request.model_id.clone(),
                spawned_ms: now_ms(),
                script: script.clone(),
            },
        );
        if let Some(after) = script.crash_after_ms {
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(after));
                if pid_alive(pid) {
                    unsafe {
                        libc::kill(pid as i32, libc::SIGKILL);
                    }
                }
            });
        }
        Ok(EngineHandle {
            pid,
            endpoint: request.endpoint.clone(),
            started_at: now_ms(),
            scope_name: request.scope_name.clone(),
        })
    }

    fn stop(&self, handle: &EngineHandle) -> Result<()> {
        if self.fail_stop.load(Ordering::SeqCst) {
            anyhow::bail!("simulated stop failure for pid {}", handle.pid);
        }
        let child = self
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&handle.pid)
            .map(|p| p.child);
        if let Some(mut child) = child {
            let _ = child.kill();
            let _ = child.wait();
        } else {
            terminate_pid(handle.pid, Duration::from_millis(500));
        }
        if pid_alive(handle.pid) {
            let cmd = process_cmdline(handle.pid);
            anyhow::bail!(
                "fake engine pid {} is still alive after stop (cmdline={cmd:?})",
                handle.pid
            );
        }
        Ok(())
    }

    fn health(&self, endpoint: &str, handle: &EngineHandle) -> Result<HealthReport> {
        let proc = {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.get_mut(&handle.pid).map(|p| {
                let exited = matches!(p.child.try_wait(), Ok(Some(_)));
                (exited, p.child.id(), p.model.clone(), p.spawned_ms, p.script.clone())
            })
        };
        let Some((exited, pid, model, spawned_ms, script)) = proc else {
            anyhow::bail!("unknown fake engine pid {} for {endpoint}", handle.pid)
        };
        if exited {
            anyhow::bail!("fake engine pid {pid} exited");
        }
        let elapsed = now_ms().saturating_sub(spawned_ms);
        if elapsed < script.load_ms {
            return Ok(HealthReport::loading());
        }
        let mut stems: BTreeSet<String> = script.stems.into_iter().collect();
        if stems.is_empty() {
            // Default script = the serving engine advertises its own model id.
            stems.insert(model);
        }
        Ok(HealthReport {
            ready: true,
            stems,
        })
    }

    fn progress(&self, _endpoint: &str, handle: &EngineHandle) -> LoadProgress {
        let proc = self
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&handle.pid)
            .map(|p| (p.spawned_ms, p.script.clone()));
        let Some((spawned_ms, script)) = proc else {
            return LoadProgress::indeterminate();
        };
        let elapsed = now_ms().saturating_sub(spawned_ms);
        if !script.progress || script.load_ms == 0 {
            return LoadProgress {
                fraction: None,
                elapsed_ms: elapsed,
                eta_ms: None,
            };
        }
        let fraction = (elapsed as f64 / script.load_ms as f64).clamp(0.0, 1.0);
        LoadProgress {
            fraction: Some(fraction),
            elapsed_ms: elapsed,
            eta_ms: Some(script.load_ms.saturating_sub(elapsed)),
        }
    }
}

/// Engine owned by another subsystem (llama-swap, ds4/gufo containers): the
/// scheduler records it from Residency but never spawns it.
pub struct ExternalAdapter;

impl EngineAdapter for ExternalAdapter {
    fn name(&self) -> &'static str {
        "external"
    }
    fn spawn(&self, _request: &SpawnRequest) -> Result<EngineHandle> {
        anyhow::bail!("this engine is managed outside the swap supervisor")
    }
    fn stop(&self, _handle: &EngineHandle) -> Result<()> {
        Ok(())
    }
    fn health(&self, _endpoint: &str, _handle: &EngineHandle) -> Result<HealthReport> {
        Ok(HealthReport::loading())
    }
}

fn terminate_pid(pid: u32, grace: Duration) {
    if !pid_alive(pid) {
        return;
    }
    unsafe {
        libc::kill(pid as i32, libc::SIGTERM);
    }
    let deadline = std::time::Instant::now() + grace;
    while std::time::Instant::now() < deadline {
        if !pid_alive(pid) {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    if pid_alive(pid) {
        unsafe {
            libc::kill(pid as i32, libc::SIGKILL);
        }
    }
}

// ── Budget ledger (budget.json, append-only) ────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BudgetEntry {
    pub ts: String,
    /// `admit` | `deny` | `evict` | `pin` | `unpin` | `spawn` | `stop` |
    /// `adopt`
    pub kind: String,
    pub model: String,
    pub gib: f64,
    #[serde(default)]
    pub reason: String,
    /// Who caused this entry (L5 audit: `who`, `when` = ts, `target` = model).
    /// API mutations carry `admin@<origin>` (or the CLI-supplied `cli:uid=N`);
    /// internal daemon actions leave it absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct BudgetFile {
    #[serde(default)]
    entries: Vec<BudgetEntry>,
}

/// Keep the newest `max` rows (append-only semantics, bounded file size).
fn push_bounded(entries: &mut Vec<BudgetEntry>, max: usize) {
    if entries.len() > max {
        let excess = entries.len() - max;
        entries.drain(0..excess);
    }
}

pub struct BudgetLedger {
    path: PathBuf,
    inner: Mutex<Vec<BudgetEntry>>,
}

impl BudgetLedger {
    pub fn load(path: PathBuf) -> Result<Self> {
        let entries = match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str::<BudgetFile>(&text)
                .with_context(|| format!("parsing budget ledger {}", path.display()))?
                .entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        Ok(Self {
            path,
            inner: Mutex::new(entries),
        })
    }

    pub fn in_memory(path: PathBuf) -> Self {
        Self {
            path,
            inner: Mutex::new(Vec::new()),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn record(&self, kind: &str, model: &str, gib: f64, reason: impl Into<String>) -> Result<()> {
        self.record_as(kind, model, gib, reason, None)
    }

    /// Audit form: records the acting identity beside ts/kind/model/reason.
    pub fn record_as(
        &self,
        kind: &str,
        model: &str,
        gib: f64,
        reason: impl Into<String>,
        actor: Option<&str>,
    ) -> Result<()> {
        let mut entries = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        entries.push(BudgetEntry {
            ts: chrono::Utc::now().to_rfc3339(),
            kind: kind.to_string(),
            model: model.to_string(),
            gib,
            reason: reason.into(),
            actor: actor.map(|value| value.to_string()),
        });
        push_bounded(&mut entries, LEDGER_MAX_ENTRIES);
        let text = serde_json::to_string_pretty(&BudgetFile {
            entries: entries.clone(),
        })
        .context("serializing budget ledger")?;
        crate::plugins::store::atomic_write_0600(&self.path, text.as_bytes())
            .with_context(|| format!("writing budget ledger {}", self.path.display()))
    }

    pub fn entries(&self) -> Vec<BudgetEntry> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

// ── Status shapes ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecycleState {
    Stopped,
    Loading,
    Ready,
    Draining,
    Failed,
    /// Admission is waiting on the `min_mem_gib` gate (`Waiting(gate)`).
    Waiting,
}

impl LifecycleState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Stopped => "stopped",
            Self::Loading => "loading",
            Self::Ready => "ready",
            Self::Draining => "draining",
            Self::Failed => "failed",
            Self::Waiting => "waiting",
        }
    }
}

/// The visible `Waiting(gate)` payload: the threshold, the current probe value
/// and the deadline (the Engines tab renders all three; spec binding).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateStatus {
    pub threshold_gib: f64,
    pub current_gib: f64,
    pub waited_ms: u64,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelStatus {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub engine: String,
    pub state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    pub resident: bool,
    pub spec_stale: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    pub queue_position: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ewma_load_ms: Option<f64>,
    pub last_stream_completed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_estimate_gib: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pinned_gib: Option<f64>,
    /// Adapter-reported progress for a loading model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<LoadProgress>,
    /// `Waiting(gate)` admission details (L5 `min_mem_gib` gate).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gate: Option<GateStatus>,
    /// Supervisor-allocated port for a managed child.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    /// Generated conf path (strixite: the port lives in the conf).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conf_path: Option<String>,
    /// Active-consumer protection span (live stream or idle grace).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_consumer: Option<ConsumerSpan>,
}

/// An externally-launched engine process (icon-era server): visible,
/// budget-counted, attach-only until `swap adopt`.
#[derive(Debug, Clone, Serialize)]
pub struct ExternalEngineView {
    pub pid: u32,
    pub engine: String,
    pub endpoint: String,
    pub port: u16,
    pub stems: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub catalog_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gguf: Option<String>,
    /// True once the operator adopted the pid (marker written; the pid is then
    /// supervisor-owned and a cutover may signal it).
    pub adopted: bool,
}

/// One configured systemd unit (takeover surface).
#[derive(Debug, Clone, Serialize)]
pub struct UnitView {
    pub engine: String,
    pub unit: String,
    pub active: bool,
    pub main_pid: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueueStatus {
    pub id: u64,
    pub model: String,
    pub position: usize,
    pub waited_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BudgetLine {
    pub model: String,
    pub source: String,
    pub gib: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pinned_gib: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BudgetReport {
    pub cap_gib: f64,
    pub used_gib: f64,
    pub headroom_gib: f64,
    pub residents: Vec<BudgetLine>,
    /// Pins with no observed resident count their full `reserved_gib`.
    pub pins: Vec<BudgetLine>,
    pub evictions_today: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EnginesStatus {
    pub models: Vec<ModelStatus>,
    pub queue: Vec<QueueStatus>,
    pub budget: BudgetReport,
    pub adoption: AdoptionReport,
    pub shim_error: Option<String>,
    pub adapter_names: Vec<String>,
    /// L5: icon-era engines observed read-only (attach-only until adopted).
    pub external: Vec<ExternalEngineView>,
    /// L5: configured systemd units + their state (takeover surface).
    pub units: Vec<UnitView>,
    /// L5: Active-Consumer Set spans (live in-process streams + idle grace).
    pub active_consumers: Vec<ConsumerSpan>,
    /// L5: adapter detail badges (hipfire version-keyed schema, strixite confs).
    pub adapters: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AdoptionReport {
    pub adopted: Vec<String>,
    pub spec_stale: Vec<String>,
    pub cleaned: Vec<String>,
    /// Adopted children with no catalog entry (unnamed residents): they run
    /// and are never killed, but cannot be spec-hashed; surfaced as warnings.
    #[serde(default)]
    pub catalog_missing: Vec<String>,
    /// Human-readable refusals (e.g. a live marker with no recorded
    /// start_time — identity cannot be proven, so adoption is refused).
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdoptionMarker {
    pub model: String,
    pub pid: u32,
    pub gguf: String,
    pub start_time: u64,
    pub spec_hash: String,
    #[serde(default)]
    pub scope_name: Option<String>,
    pub endpoint: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnsureOutcome {
    /// Model is (or became) ready at this endpoint. `spawned` distinguishes a
    /// fresh child from an attach.
    Attach {
        model: String,
        endpoint: String,
        source: ResidentSource,
        spawned: bool,
    },
}

impl EnsureOutcome {
    pub fn endpoint(&self) -> &str {
        match self {
            Self::Attach { endpoint, .. } => endpoint,
        }
    }

    pub fn spawned(&self) -> bool {
        match self {
            Self::Attach { spawned, .. } => *spawned,
        }
    }
}

/// Admission options. Both flags are single-action operator overrides
/// (audit-logged); defaults are the safe behavior.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EnsureOptions {
    /// `swap ensure <model> --override` / the dashboard "Swap anyway" button:
    /// bypass the Active-Consumer span for this operation only.
    pub override_active_consumers: bool,
    /// `--force-gate`: proceed below the adapter's `min_mem_gib` floor.
    pub force_gate: bool,
    /// `swap adopt` / takeover: never spawn, only attach to an existing
    /// resident (external observer or unit process).
    pub attach_only: bool,
    /// Audit identity for override/blocked entries (`admin@<origin>` or
    /// `cli:uid=N`).
    pub actor: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct TakeoverOptions {
    pub unit: Option<String>,
    pub override_unit: bool,
    pub force_gate: bool,
    pub override_active_consumers: bool,
    /// Audit identity for the takeover/override ledger entries.
    pub actor: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TakeoverReport {
    pub model: String,
    pub endpoint: String,
    pub attached: bool,
    pub spawned: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stopped_unit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adopted_pid: Option<u32>,
    /// Who performed the takeover (audit; surfaced to the Engines tab).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actor: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdoptReport {
    pub model: String,
    pub pid: u32,
    pub endpoint: String,
    pub gguf: Option<String>,
    pub spec_hash: String,
}

// ── Scheduler ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SwapSchedulerConfig {
    /// How long `ensure` waits for a loading model before timing out.
    pub wait_timeout: Duration,
    pub queue_cap: usize,
    pub stop_grace: Duration,
    pub max_restarts: u32,
    pub port_min: u16,
    pub port_max: u16,
    pub llama_swap_url: String,
    /// v2 L7: the shim Residency source retires at the flip. `false` = native
    /// mode: never dial the shim for `/running`, never list shim residents.
    pub shim_source_enabled: bool,
    /// L5: idle grace after the last routed stream (Active-Consumer span).
    pub active_consumer_grace: Duration,
    /// L5: `MemAvailable` floor applied to spawning ensures (`0` = disabled).
    pub min_mem_gib: f64,
    /// L5: how long a below-floor admission stays `Waiting(gate)`.
    pub gate_timeout: Duration,
    pub gate_poll: Duration,
    /// L5: settle sleep after the previous pid is gone (spec: 2 s).
    pub settle: Duration,
    /// L5: bounded single ACS re-check at the SIGTERM point.
    pub cutover_recheck: Duration,
}

impl Default for SwapSchedulerConfig {
    fn default() -> Self {
        Self {
            wait_timeout: Duration::from_secs(
                std::env::var("BRAINROUTER_SWAP_WAIT_MS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(600_000),
            ),
            queue_cap: std::env::var("BRAINROUTER_SWAP_QUEUE_CAP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8),
            stop_grace: Duration::from_millis(
                std::env::var("BRAINROUTER_SWAP_STOP_GRACE_MS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(2_000),
            ),
            max_restarts: 3,
            port_min: 5900,
            port_max: 5999,
            llama_swap_url: "http://127.0.0.1:8081".into(),
            shim_source_enabled: true,
            active_consumer_grace: Duration::from_secs(
                std::env::var("BRAINROUTER_ACTIVE_CONSUMER_GRACE_S")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(crate::active_consumers::DEFAULT_GRACE_S),
            ),
            min_mem_gib: 0.0,
            gate_timeout: Duration::from_millis(
                std::env::var("BRAINROUTER_GATE_TIMEOUT_MS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(600_000),
            ),
            gate_poll: Duration::from_millis(
                std::env::var("BRAINROUTER_GATE_POLL_MS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1_000),
            ),
            settle: Duration::from_millis(
                std::env::var("BRAINROUTER_CUTOVER_SETTLE_MS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(2_000),
            ),
            cutover_recheck: Duration::from_millis(
                std::env::var("BRAINROUTER_CUTOVER_RECHECK_MS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(5_000),
            ),
        }
    }
}

struct ModelRuntime {
    state: LifecycleState,
    endpoint: Option<String>,
    handle: Option<EngineHandle>,
    source: Option<ResidentSource>,
    last_error: Option<String>,
    restarts: u32,
    load_started_ms: u64,
    spec_stale: bool,
    port: Option<u16>,
    conf_path: Option<String>,
    gate: Option<GateStatus>,
}

impl Default for ModelRuntime {
    fn default() -> Self {
        Self {
            state: LifecycleState::Stopped,
            endpoint: None,
            handle: None,
            source: None,
            last_error: None,
            restarts: 0,
            load_started_ms: 0,
            spec_stale: false,
            port: None,
            conf_path: None,
            gate: None,
        }
    }
}

struct QueueEntry {
    id: u64,
    model: String,
    requested_ms: u64,
    cancelled: Arc<AtomicBool>,
}

#[derive(Default)]
struct Inner {
    models: BTreeMap<String, ModelRuntime>,
    queue: Vec<QueueEntry>,
    next_queue_id: u64,
}

struct QueueGuard<'a> {
    scheduler: &'a SwapScheduler,
    id: u64,
}

impl Drop for SwapScheduler {
    fn drop(&mut self) {
        if !self.kill_children_on_drop {
            return;
        }
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        for runtime in inner.models.values() {
            if let Some(handle) = &runtime.handle {
                if pid_alive(handle.pid) {
                    unsafe {
                        libc::kill(handle.pid as i32, libc::SIGTERM);
                    }
                }
            }
        }
    }
}

impl Drop for QueueGuard<'_> {
    fn drop(&mut self) {
        self.scheduler.remove_queue(self.id);
    }
}

pub struct SwapScheduler {
    catalog: Arc<SwapCatalog>,
    ledger: Arc<BudgetLedger>,
    adapters: BTreeMap<String, Arc<dyn EngineAdapter>>,
    pub config: SwapSchedulerConfig,
    inner: Mutex<Inner>,
    spawn_locks: Mutex<BTreeMap<String, Arc<tokio::sync::Mutex<()>>>>,
    serving_identities: Option<Arc<ServingIdentityRegistry>>,
    external: Arc<ExternalResidencyRegistry>,
    shim_view: Mutex<ResidencyView>,
    container_view: Mutex<Vec<Resident>>,
    inflight: Mutex<Option<Arc<InflightRegistry>>>,
    last_adoption: Mutex<AdoptionReport>,
    cap_override: Option<f64>,
    native_adapter: Option<Arc<dyn EngineAdapter>>,
    logs_dir: PathBuf,
    adoption_dir: PathBuf,
    evictions_today: AtomicU64,
    /// Ports assigned to live/pending engines. A managed port is reserved
    /// before spawn so concurrent spawns (and fake engines, which never bind)
    /// cannot share an endpoint.
    reserved_ports: Mutex<BTreeSet<u16>>,
    /// Monotonic counter the daemon poller uses to refresh shim residency.
    shim_refreshed_ms: AtomicU64,
    /// L5 Active-Consumer Set (in-process streams + idle grace).
    consumers: Mutex<Option<Arc<ActiveConsumers>>>,
    /// L5 external-engine observer (read-only `/proc` scan + `/v1/models`).
    external_observer: Mutex<Option<Arc<ExternalObserver>>>,
    /// Last observation (Engines tab detail: pid/engine/stems/adopted).
    observed_external: Mutex<Vec<ObservedEngine>>,
    /// L5 systemd user-unit runtime (pid-scoped takeover).
    unit_runtime: Mutex<Option<Arc<dyn UnitRuntime>>>,
    /// engine → configured unit name (from the fleet profile).
    takeover_units: Mutex<BTreeMap<String, String>>,
    /// L5 injectable `MemAvailable` probe (fixtures); env wins over it.
    mem_source: Mutex<Option<Arc<dyn Fn() -> f64 + Send + Sync>>>,
    /// Serialized per-device swap locks (exclusive adapters).
    device_locks: Mutex<BTreeMap<String, Arc<tokio::sync::Mutex<()>>>>,
    /// Adapters already probed (once per adapter, at the first spawn).
    probed_adapters: Mutex<BTreeSet<String>>,
    /// Unit probe cache (a `systemctl show` subprocess per configured unit;
    /// `/api/engines` is polled by the dashboard, so keep it short-lived).
    unit_cache: Mutex<Option<(u64, Vec<UnitView>)>>,
    /// Test-only hygiene: kill children when the scheduler is dropped. The
    /// daemon MUST leave this false — native children are scoped so they
    /// survive a daemon restart (that is what adoption relies on).
    kill_children_on_drop: bool,
}

impl SwapScheduler {
    pub fn new(
        catalog: Arc<SwapCatalog>,
        ledger: Arc<BudgetLedger>,
        config: SwapSchedulerConfig,
        adapters: BTreeMap<String, Arc<dyn EngineAdapter>>,
    ) -> Self {
        Self {
            catalog,
            ledger,
            adapters,
            config,
            inner: Mutex::new(Inner::default()),
            spawn_locks: Mutex::new(BTreeMap::new()),
            serving_identities: None,
            external: Arc::new(ExternalResidencyRegistry::new()),
            shim_view: Mutex::new(ResidencyView::new()),
            container_view: Mutex::new(Vec::new()),
            inflight: Mutex::new(None),
            last_adoption: Mutex::new(AdoptionReport::default()),
            cap_override: None,
            native_adapter: None,
            logs_dir: crate::swap_catalog::logs_dir(),
            adoption_dir: adoption_dir(),
            evictions_today: AtomicU64::new(0),
            reserved_ports: Mutex::new(BTreeSet::new()),
            shim_refreshed_ms: AtomicU64::new(0),
            consumers: Mutex::new(None),
            external_observer: Mutex::new(None),
            observed_external: Mutex::new(Vec::new()),
            unit_runtime: Mutex::new(None),
            takeover_units: Mutex::new(BTreeMap::new()),
            mem_source: Mutex::new(None),
            device_locks: Mutex::new(BTreeMap::new()),
            probed_adapters: Mutex::new(BTreeSet::new()),
            unit_cache: Mutex::new(None),
            kill_children_on_drop: false,
        }
    }

    /// Test-only: terminate owned children on Drop so `cargo test` leaves no
    /// `sleep` processes behind. Never enable in the daemon.
    pub fn with_kill_children_on_drop(mut self, kill: bool) -> Self {
        self.kill_children_on_drop = kill;
        self
    }

    /// Attach the live in-flight registry (LRU must never evict an active
    /// stream).
    pub fn attach_inflight(&self, inflight: Arc<InflightRegistry>) {
        *self.inflight.lock().unwrap_or_else(|e| e.into_inner()) = Some(inflight);
    }

    /// L5: attach the Active-Consumer Set (router in-process streams + grace).
    pub fn with_active_consumers(mut self, consumers: Arc<ActiveConsumers>) -> Self {
        self.consumers = Mutex::new(Some(consumers));
        self
    }

    pub fn consumers(&self) -> Option<Arc<ActiveConsumers>> {
        self.consumers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// L5: attach the external-engine observer (icon-era servers).
    pub fn with_external_observer(mut self, observer: Arc<ExternalObserver>) -> Self {
        self.external_observer = Mutex::new(Some(observer));
        self
    }

    /// L5: attach the systemd user-unit runtime (pid-scoped takeover).
    pub fn with_unit_runtime(mut self, runtime: Arc<dyn UnitRuntime>) -> Self {
        self.unit_runtime = Mutex::new(Some(runtime));
        self
    }

    /// L5: register the unit an engine may be taken over from.
    pub fn with_takeover_unit(self, engine: &str, unit: impl Into<String>) -> Self {
        self.takeover_units
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(engine.to_string(), unit.into());
        self
    }

    /// L5: injectable `MemAvailable` probe (fixtures). `BRAINROUTER_MEM_AVAILABLE_GIB`
    /// still wins, so the fixture tier can hold one value for a whole run.
    pub fn with_mem_source(mut self, source: Arc<dyn Fn() -> f64 + Send + Sync>) -> Self {
        self.mem_source = Mutex::new(Some(source));
        self
    }

    pub fn mem_available_gib(&self) -> f64 {
        if let Some(value) = std::env::var("BRAINROUTER_MEM_AVAILABLE_GIB")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
        {
            return value;
        }
        let source = self
            .mem_source
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        source
            .map(|f| f())
            .unwrap_or_else(crate::platform::mem_available_gib)
    }

    /// Refresh the external-engine slot: `/proc` scan → Residency + detail rows.
    /// Classification precedence is the observer's (markers/registry pids are
    /// native, never double-listed).
    pub fn refresh_external(&self) {
        let observer = self
            .external_observer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let Some(observer) = observer else {
            return;
        };
        let native_pids: Vec<u32> = {
            let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner
                .models
                .values()
                .filter_map(|runtime| runtime.handle.as_ref().map(|h| h.pid))
                .collect()
        };
        observer.set_native_pids(native_pids);
        let (residents, observed) = observer.snapshot();
        self.external.replace(residents);
        *self
            .observed_external
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = observed;
    }

    pub fn observed_external(&self) -> Vec<ObservedEngine> {
        self.observed_external
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Budget cap override for tests/daemon policy (beats the env var).
    pub fn with_budget_cap(mut self, gib: f64) -> Self {
        self.cap_override = Some(gib);
        self
    }

    pub fn with_serving_identities(
        mut self,
        identities: Arc<ServingIdentityRegistry>,
    ) -> Self {
        self.serving_identities = Some(identities);
        self
    }

    pub fn external_registry(&self) -> Arc<ExternalResidencyRegistry> {
        Arc::clone(&self.external)
    }

    pub fn catalog(&self) -> &Arc<SwapCatalog> {
        &self.catalog
    }

    pub fn ledger(&self) -> &Arc<BudgetLedger> {
        &self.ledger
    }

    /// Override the state root (tests): logs + adoption markers.
    pub fn with_state_root(mut self, root: PathBuf) -> Self {
        self.logs_dir = root.join("swap_logs");
        self.adoption_dir = root.join(ADOPTION_DIR);
        self
    }

    pub fn install_native_adapter(&mut self, adapter: Arc<dyn EngineAdapter>) {
        self.native_adapter = Some(Arc::clone(&adapter));
        self.adapters
            .insert("llama_cpp".to_string(), Arc::clone(&adapter));
    }

    /// Replace the cached shim residency (daemon poller calls this every few
    /// seconds; tests call it directly for the fixture tier).
    pub fn set_shim_view(&self, view: ResidencyView) {
        self.shim_refreshed_ms.store(now_ms(), Ordering::SeqCst);
        *self.shim_view.lock().unwrap_or_else(|e| e.into_inner()) = view;
    }

    /// Refresh the async sources (llama-swap `/running`, server-mode container
    /// identities) into the cached sync view. The daemon poller calls this;
    /// tests drive the shim view directly.
    pub async fn refresh_residency(&self, client: &reqwest::Client) {
        // v2 L7: after the flip the shim source retires. Native mode keeps an
        // empty shim view and never dials `:8081` (the boot gate made this
        // safe by refusing while shim residents existed).
        let view = if self.config.shim_source_enabled {
            residency::poll_shim(client, &self.config.llama_swap_url, &self.catalog).await
        } else {
            residency::ResidencyView::new()
        };
        self.set_shim_view(view);
        if let Some(registry) = &self.serving_identities {
            let snapshot = registry.snapshot().await;
            *self
                .container_view
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = residency::observe_containers(&snapshot);
        }
    }

    /// v2 L7 rollback contract: stop every supervisor-owned child (adapters do
    /// SIGTERM → grace → KILL; the per-GGUF flock is released when the child
    /// exits). Called on daemon shutdown so a native→shim rollback leaves no
    /// native resident behind.
    pub fn shutdown_all(&self, reason: &str) -> Vec<String> {
        let ids: Vec<String> = {
            let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner
                .models
                .iter()
                .filter(|(_, runtime)| runtime.handle.is_some())
                .map(|(id, _)| id.clone())
                .collect()
        };
        let mut stopped = Vec::new();
        for id in ids {
            if let Err(error) = self.stop_native_inner(&id, reason, true) {
                tracing::warn!(model = %id, %error, "swap shutdown: could not stop native child");
            } else {
                stopped.push(id);
            }
        }
        stopped
    }

    /// Combined Residency view: shim + own children + containers + external.
    pub fn residency(&self) -> ResidencyView {
        let shim = self
            .shim_view
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let native = {
            let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner
                .models
                .iter()
                .filter_map(|(id, runtime)| {
                    let handle = runtime.handle.as_ref()?;
                    if runtime.state != LifecycleState::Ready {
                        return None;
                    }
                    Some(Resident {
                        catalog_id: id.clone(),
                        source: ResidentSource::Native,
                        endpoint: handle.endpoint.clone(),
                        gguf_path: self
                            .catalog
                            .entry(id)
                            .and_then(|e| e.canonical_gguf())
                            .map(|p| p.display().to_string()),
                        started_at: handle.started_at,
                        last_stream_completed: self
                            .catalog
                            .entry(id)
                            .map(|e| e.last_stream_completed)
                            .unwrap_or(0),
                        state: "ready".into(),
                        spec_stale: runtime.spec_stale,
                        engine: Some(
                            self.catalog
                                .entry(id)
                                .map(|e| e.engine)
                                .unwrap_or_default(),
                        ),
                    })
                })
                .collect()
        };
        let containers = self
            .container_view
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        residency::compose(shim, native, containers, self.external.snapshot())
    }

    // ── Budget ─────────────────────────────────────────────────────────────

    pub fn budget_cap_gib(&self) -> f64 {
        if let Some(cap) = self.cap_override {
            return cap;
        }
        std::env::var(crate::platform::BUDGET_ENV)
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| *v > 0.0)
            .unwrap_or_else(|| crate::platform::budget_gib(&crate::platform::detect_facts()))
    }

    pub fn need_gib(&self, entry: &CatalogEntry) -> f64 {
        let estimate = entry.vram_estimate_gib.unwrap_or(0.0);
        let pinned = self
            .catalog
            .pins()
            .get(&entry.id)
            .map(|p| p.reserved_gib)
            .unwrap_or(0.0);
        // Count each model id ONCE: observed resident ⇒ max(estimate,
        // reserved_gib); the pin alone is not added on top.
        estimate.max(pinned)
    }

    /// Budget report: every observed resident counted once (max of estimate
    /// and its pin), then unmatched pins at full reserved size.
    pub fn budget_report(&self) -> BudgetReport {
        let view = self.residency();
        let data = self.catalog.snapshot();
        let cap = self.budget_cap_gib();
        let mut by_id: BTreeMap<String, (&'static str, f64, Option<f64>)> = BTreeMap::new();
        for resident in &view.residents {
            let entry = data.entries.get(&resident.catalog_id);
            let estimate = entry.and_then(|e| e.vram_estimate_gib).unwrap_or(0.0);
            let pin = data
                .pins
                .get(&resident.catalog_id)
                .map(|p| p.reserved_gib);
            let line = by_id
                .entry(resident.catalog_id.clone())
                .or_insert((resident.source.as_str(), 0.0, pin));
            line.1 = line.1.max(estimate.max(pin.unwrap_or(0.0)));
            if line.2.is_none() {
                line.2 = pin;
            }
        }
        let mut residents: Vec<BudgetLine> = by_id
            .into_iter()
            .map(|(model, (source, gib, pinned))| BudgetLine {
                model,
                source: source.to_string(),
                gib,
                pinned_gib: pinned,
            })
            .collect();
        residents.sort_by(|a, b| a.model.cmp(&b.model));
        let resident_ids: BTreeSet<&String> = residents.iter().map(|r| &r.model).collect();
        let pins: Vec<BudgetLine> = data
            .pins
            .iter()
            .filter(|(id, _)| !resident_ids.contains(id))
            .map(|(id, pin)| BudgetLine {
                model: id.clone(),
                source: "pin".into(),
                gib: pin.reserved_gib,
                pinned_gib: Some(pin.reserved_gib),
            })
            .collect();
        let used = residents.iter().map(|r| r.gib).sum::<f64>()
            + pins.iter().map(|r| r.gib).sum::<f64>();
        BudgetReport {
            cap_gib: cap,
            used_gib: used,
            headroom_gib: (cap - used).max(0.0),
            residents,
            pins,
            evictions_today: self.evictions_today.load(Ordering::SeqCst),
        }
    }

    fn is_inflight(&self, id: &str) -> bool {
        let guard = self.inflight.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .as_ref()
            .map(|registry| {
                registry
                    .snapshot()
                    .iter()
                    .any(|row| row.catalog_id.as_deref() == Some(id))
            })
            .unwrap_or(false)
    }

    /// The endpoint a model is serving at (runtime handle first, then the
    /// Residency view, external residents included).
    fn endpoint_for(&self, id: &str) -> Option<String> {
        if let Some(runtime) = self.runtime(id) {
            if let Some(endpoint) = runtime.endpoint {
                return Some(endpoint);
            }
        }
        self.residency()
            .resident_for(id)
            .map(|resident| resident.endpoint.clone())
    }

    /// Active-Consumer Set check for one model. `Some(detail)` = protected
    /// (live proxy row, live in-process stream, or the idle grace window).
    pub fn protection(&self, id: &str) -> Option<String> {
        if self.is_inflight(id) {
            return Some("proxy in-flight row".to_string());
        }
        let consumers = self.consumers()?;
        let endpoint = self.endpoint_for(id);
        for key in protection_keys(id, endpoint.as_deref()) {
            if !consumers.protected(&key) {
                continue;
            }
            let remaining = consumers.span_remaining_ms(&key);
            return Some(if remaining == u64::MAX {
                format!("live in-process stream on `{key}`")
            } else {
                format!("idle grace ({remaining} ms remaining) on `{key}`")
            });
        }
        None
    }

    fn record_blocked_as(&self, model: &str, reason: &str, actor: Option<&str>) {
        let gib = self
            .catalog
            .entry(model)
            .map(|entry| self.need_gib(&entry))
            .unwrap_or(0.0);
        let _ = self
            .ledger
            .record_as("blocked", model, gib, reason, actor);
        self.log(model, reason);
    }

    /// Evict least-recent-STREAM-COMPLETED native children until `needed_gib`
    /// fits. Never evicts pinned models or models inside the Active-Consumer
    /// Set (unless the one-action override is set).
    fn evict_to_fit(&self, needed_gib: f64) -> Result<Vec<String>> {
        self.evict_to_fit_with(needed_gib, &EnsureOptions::default())
    }

    fn evict_to_fit_with(
        &self,
        needed_gib: f64,
        options: &EnsureOptions,
    ) -> Result<Vec<String>> {
        let mut evicted: Vec<String> = Vec::new();
        let mut skipped: Vec<String> = Vec::new();
        let mut logged_blocked: BTreeSet<String> = BTreeSet::new();
        loop {
            let report = self.budget_report();
            if report.used_gib + needed_gib <= report.cap_gib + 1e-9 {
                // One aggregate ledger row per evict pass (never one row per
                // candidate; a single-victim pass keeps the victim's model id
                // so the sequence stays greppable).
                if !evicted.is_empty() {
                    let gib: f64 = evicted
                        .iter()
                        .map(|id| {
                            self.catalog
                                .entry(id)
                                .map(|e| self.need_gib(&e))
                                .unwrap_or(0.0)
                        })
                        .sum();
                    let model = if evicted.len() == 1 {
                        evicted[0].clone()
                    } else {
                        "*".to_string()
                    };
                    let _ = self.ledger.record(
                        "evict",
                        &model,
                        gib,
                        format!("lru eviction pass: {}", evicted.join(", ")),
                    );
                }
                return Ok(evicted);
            }
            let pins = self.catalog.pins();
            let data = self.catalog.snapshot();
            let mut blocked: Vec<(String, String)> = Vec::new();
            let mut candidates: Vec<(u64, String, f64)> = report
                .residents
                .iter()
                .filter(|line| line.source == "native")
                .filter(|line| !pins.contains_key(&line.model))
                .filter(|line| options.override_active_consumers || !self.is_inflight(&line.model))
                .filter(|line| !evicted.contains(&line.model) && !skipped.contains(&line.model))
                .filter(|line| {
                    if options.override_active_consumers {
                        return true;
                    }
                    match self.protection(&line.model) {
                        Some(detail) => {
                            blocked.push((line.model.clone(), detail));
                            false
                        }
                        None => true,
                    }
                })
                .map(|line| {
                    (
                        data.entries
                            .get(&line.model)
                            .map(|e| e.last_stream_completed)
                            .unwrap_or(0),
                        line.model.clone(),
                        line.gib,
                    )
                })
                .collect();
            for (model, detail) in &blocked {
                if logged_blocked.insert(model.clone()) {
                    self.record_blocked_as(
                        model,
                        &format!("blocked:active-consumer: skipped eviction ({detail})"),
                        options.actor.as_deref(),
                    );
                }
            }
            candidates.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
            let Some((_, model, _)) = candidates.first().cloned() else {
                if let Some((model, detail)) = blocked.first().cloned() {
                    return Err(SwapError::BlockedActiveConsumer { model, detail }.into());
                }
                let pins_total: f64 = pins.values().map(|p| p.reserved_gib).sum();
                return Err(SwapError::NoFit(format!(
                    "need {needed_gib:.1} GiB but the platform budget is {:.1} GiB \
                     (used {:.1} GiB; pins reserve {:.1} GiB; no evictable native \
                     candidates remain)",
                    report.cap_gib, report.used_gib, pins_total
                ))
                .into());
            };
            match self.stop_native(
                &model,
                "lru eviction",
                false,
                options.override_active_consumers,
            ) {
                Ok(()) => evicted.push(model),
                // Locked by a concurrent ensure or protected between the
                // candidate snapshot and now: skip it, never block the pass.
                Err(e) if e.downcast_ref::<SwapError>().is_some_and(|e| {
                    matches!(e, SwapError::Busy(_))
                }) =>
                {
                    skipped.push(model)
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Stop a model under its spawn lock, re-checking protection immediately
    /// before signalling (the LRU candidate list may be stale: a stream can
    /// start between selection and eviction).
    /// `force` is the operator's one-action override: it bypasses the
    /// in-flight guard (a sized pin is never bypassed implicitly).
    fn stop_native(&self, id: &str, reason: &str, record_ledger: bool, force: bool) -> Result<()> {
        let lock = self.spawn_lock(id);
        let _guard = lock.try_lock().map_err(|_| {
            SwapError::Busy(format!(
                "`{id}` is currently being started or stopped by another request"
            ))
        })?;
        if self.catalog.pins().contains_key(id) {
            return Err(SwapError::Busy(format!(
                "`{id}` carries a sized pin and is never evicted implicitly"
            ))
            .into());
        }
        if !force && self.is_inflight(id) {
            return Err(SwapError::Busy(format!(
                "`{id}` has an active stream and is never evicted"
            ))
            .into());
        }
        self.stop_native_inner(id, reason, record_ledger)
    }

    /// Unprotected stop body: only callers that already own the lifecycle
    /// (the spawning ensure, an explicit protected stop) use this.
    fn stop_native_inner(&self, id: &str, reason: &str, record_ledger: bool) -> Result<()> {
        let (handle, adapter_name, prev_state, prev_source) = {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            let Some(runtime) = inner.models.get_mut(id) else {
                return Ok(());
            };
            let handle = runtime.handle.take();
            let prev_state = runtime.state;
            let prev_source = runtime.source;
            runtime.state = LifecycleState::Stopped;
            runtime.endpoint = None;
            runtime.source = None;
            let adapter_name = self
                .catalog
                .entry(id)
                .map(|e| e.engine)
                .unwrap_or_else(|| "llama_cpp".into());
            (handle, adapter_name, prev_state, prev_source)
        };
        if let Some(handle) = handle {
            if let Some(adapter) = self.adapter_for(&adapter_name) {
                // A failed stop MUST keep the marker: the child may still own
                // the GGUF flock, and a boot scan is the only recovery path.
                if let Err(e) = adapter.stop(&handle) {
                    // Keep BOTH the marker and the live registry row: the
                    // child may still hold the flock, and the Engines tab
                    // must still show it as serving.
                    let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(runtime) = inner.models.get_mut(id) {
                        runtime.state = prev_state;
                        runtime.endpoint = Some(handle.endpoint.clone());
                        runtime.handle = Some(handle.clone());
                        runtime.source = prev_source;
                    }
                    return Err(SwapError::SpawnFailed(format!(
                        "stop `{id}` failed: {e}"
                    ))
                    .into());
                }
            } else {
                terminate_pid(handle.pid, self.config.stop_grace);
                if pid_alive(handle.pid) {
                    let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(runtime) = inner.models.get_mut(id) {
                        runtime.state = prev_state;
                        runtime.endpoint = Some(handle.endpoint.clone());
                        runtime.handle = Some(handle.clone());
                        runtime.source = prev_source;
                    }
                    return Err(SwapError::SpawnFailed(format!(
                        "stop `{id}` failed: process {} is still alive",
                        handle.pid
                    ))
                    .into());
                }
            }
            self.release_endpoint_port(&handle.endpoint);
            self.evictions_today.fetch_add(1, Ordering::SeqCst);
            self.remove_marker(id);
            if record_ledger {
                let gib = self
                    .catalog
                    .entry(id)
                    .map(|e| self.need_gib(&e))
                    .unwrap_or(0.0);
                let _ = self.ledger.record("evict", id, gib, reason);
            }
        }
        Ok(())
    }

    /// Reserve a managed port for a spawn (unique across models).
    fn reserve_port(&self) -> Option<u16> {
        let mut reserved = self.reserved_ports.lock().unwrap_or_else(|e| e.into_inner());
        for port in self.config.port_min..=self.config.port_max {
            if reserved.contains(&port) {
                continue;
            }
            if free_port(port, port).is_some() {
                reserved.insert(port);
                return Some(port);
            }
        }
        None
    }

    fn release_endpoint_port(&self, endpoint: &str) {
        if let Some(port) = endpoint
            .trim_end_matches('/')
            .rsplit(':')
            .next()
            .and_then(|p| p.parse::<u16>().ok())
        {
            self.reserved_ports
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&port);
        }
    }

    fn adapter_for(&self, engine: &str) -> Option<Arc<dyn EngineAdapter>> {
        if let Some(adapter) = self.adapters.get(engine) {
            return Some(Arc::clone(adapter));
        }
        match engine {
            "llama_cpp" => self.adapters.get("fake").cloned(),
            _ => None,
        }
    }

    // ── Ensure ─────────────────────────────────────────────────────────────

    pub async fn ensure(&self, requested: &str) -> Result<EnsureOutcome> {
        self.ensure_with_options(requested, None, EnsureOptions::default())
            .await
    }

    pub async fn ensure_with_cancel(
        &self,
        requested: &str,
        cancel: Option<&crate::inflight::InflightHandle>,
    ) -> Result<EnsureOutcome> {
        self.ensure_with_options(requested, cancel, EnsureOptions::default())
            .await
    }

    /// The L5 admission path: Residency-first attach, then (for a would-spawn
    /// ensure) exclusive cutover → drain/settle → `min_mem_gib` gate → budget
    /// admission (ACS-aware) → spawn.
    pub async fn ensure_with_options(
        &self,
        requested: &str,
        cancel: Option<&crate::inflight::InflightHandle>,
        options: EnsureOptions,
    ) -> Result<EnsureOutcome> {
        let resolved = self
            .catalog
            .resolve_alias(requested)
            .ok_or_else(|| SwapError::UnknownModel(requested.to_string()))?;
        let entry = self
            .catalog
            .entry(&resolved)
            .ok_or_else(|| SwapError::UnknownModel(resolved.clone()))?;

        // 1. Residency BEFORE budget: a resident model attaches, always.
        if let Some(resident) = self.resident_ready(&entry) {
            return Ok(EnsureOutcome::Attach {
                model: entry.id.clone(),
                endpoint: resident.endpoint.clone(),
                source: resident.source,
                spawned: false,
            });
        }
        if options.attach_only {
            return Err(SwapError::SpawnFailed(format!(
                "attach-only requested for `{}` but no resident serves it",
                entry.id
            ))
            .into());
        }

        // Serialized per-device swap lock: two exclusive ensures never race
        // their drain/spawn sequences on one device.
        let _device_guard = match self
            .adapter_for(&entry.engine)
            .and_then(|adapter| adapter.exclusive_key())
        {
            Some(device) => Some(self.device_lock(&device).lock_owned().await),
            None => None,
        };

        let spawn_lock = self.spawn_lock(&entry.id);
        let mut is_spawner = false;
        let mut handle = None;
        let queue_entry;
        // `try_lock`: an ensure that arrives while another owns the decision
        // joins the queue (cap/position/cancel apply to it) instead of
        // blocking invisibly on the mutex.
        match spawn_lock.try_lock() {
            Ok(_guard) => {
            // Re-check under the spawn lock: a concurrent ensure may have won.
            if let Some(resident) = self.resident_ready(&entry) {
                return Ok(EnsureOutcome::Attach {
                    model: entry.id.clone(),
                    endpoint: resident.endpoint.clone(),
                    source: resident.source,
                    spawned: false,
                });
            }
            let state = self.runtime_state(&entry.id);
            match state {
                LifecycleState::Ready => {
                    let runtime = self.runtime(&entry.id);
                    let endpoint = runtime
                        .as_ref()
                        .and_then(|r| r.endpoint.clone())
                        .unwrap_or_default();
                    let source = runtime
                        .as_ref()
                        .and_then(|r| r.source)
                        .unwrap_or(ResidentSource::Native);
                    return Ok(EnsureOutcome::Attach {
                        model: entry.id.clone(),
                        endpoint,
                        source,
                        spawned: false,
                    });
                }
                LifecycleState::Loading | LifecycleState::Waiting => {
                    queue_entry = self.enqueue(&entry.id)?;
                }
                _ => {
                    // A stale adopted spec converges on an explicit ensure.
                    if self.runtime(&entry.id).map(|r| r.spec_stale).unwrap_or(false) {
                        self.stop_native_inner(&entry.id, "spec_stale convergence", true)?;
                    }
                    if options.override_active_consumers {
                        let _ = self.ledger.record_as(
                            "override",
                            &entry.id,
                            self.need_gib(&entry),
                            "active-consumer override (ensure admission)",
                            options.actor.as_deref(),
                        );
                    }
                    // L5 cutover: exclusive drain → pid-gone → settle → gate.
                    self.cutover_prepare(&entry, &options, cancel).await?;
                    // Budget admission applies ONLY here (would spawn).
                    self.evict_to_fit_with(self.need_gib(&entry), &options)?;
                    self.wait_gate(&entry, &options, cancel).await?;
                    let gib = self.need_gib(&entry);
                    self.ledger.record("admit", &entry.id, gib, "ensure admitted")?;
                    handle = Some(self.start_spawn(&entry)?);
                    is_spawner = true;
                    queue_entry = self.enqueue(&entry.id)?;
                }
            }
            }
            Err(_) => {
                // Another ensure owns the spawn lock: queue behind it.
                queue_entry = self.enqueue(&entry.id)?;
            }
        }

        let _queue_guard = QueueGuard {
            scheduler: self,
            id: queue_entry.id,
        };
        if is_spawner {
            let handle = handle.expect("spawner has a handle");
            self.finish_spawn(&entry, handle, cancel, &queue_entry).await
        } else {
            self.wait_ready(&entry.id, cancel, &queue_entry).await
        }
    }

    fn device_lock(&self, device: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self
            .device_locks
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        Arc::clone(
            locks
                .entry(device.to_string())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
        )
    }

    /// Ready models that share an exclusive device class with `entry`.
    fn exclusive_predecessors(&self, entry: &CatalogEntry) -> Vec<(String, Option<u32>)> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner
            .models
            .iter()
            .filter(|(id, runtime)| runtime.state == LifecycleState::Ready && **id != entry.id)
            .filter_map(|(id, runtime)| {
                let engine = self.catalog.entry(id)?.engine.clone();
                let device = self.adapter_for(&engine)?.exclusive_key()?;
                Some((id.clone(), runtime.handle.as_ref().map(|h| h.pid), device))
            })
            .filter(|(_, _, device)| {
                self.adapter_for(&entry.engine)
                    .and_then(|adapter| adapter.exclusive_key())
                    .as_deref()
                    == Some(device.as_str())
            })
            .map(|(id, pid, _)| (id, pid))
            .collect()
    }

    /// An unregistered (never adopted) external engine of this entry's engine
    /// class. It owns the device but not the supervisor: the cutover must
    /// refuse until the operator drains/adopts/stops it, and must never signal
    /// its pid (start-ownership).
    fn external_occupant(&self, entry: &CatalogEntry) -> Option<ObservedEngine> {
        let markers = crate::engine_observer::marker_pids(&self.adoption_dir);
        self.observed_external().into_iter().find(|engine| {
            // "Unregistered" = the observer could not resolve the process to
            // any catalog entry (no catalog_id). An external that DOES map to
            // an entry is a known resident: ensure attaches to it when it is
            // the target, and an operator can adopt/adopt-or-stop it through
            // the normal catalog vocabulary.
            engine.engine == entry.engine
                && engine.catalog_id.is_none()
                && !markers.contains(&engine.pid)
        })
    }

    /// The L5 cutover sequence for an exclusive adapter: an unregistered
    /// external occupant ⇒ `blocked:external-occupant` (unless `--override`);
    /// at the SIGTERM point re-read the Active-Consumer Set (bounded single
    /// re-check ⇒ `blocked:active-consumer`); SIGTERM **only**
    /// supervisor-owned marker pids; wait for pid-gone; then the injectable
    /// settle sleep. The memory gate and spawn follow in
    /// [`Self::wait_gate`]/[`Self::start_spawn`].
    async fn cutover_prepare(
        &self,
        entry: &CatalogEntry,
        options: &EnsureOptions,
        cancel: Option<&crate::inflight::InflightHandle>,
    ) -> Result<()> {
        let exclusive = self
            .adapter_for(&entry.engine)
            .and_then(|adapter| adapter.exclusive_key())
            .is_some();
        if !exclusive {
            return Ok(());
        }
        if let Some(occupant) = self.external_occupant(entry) {
            if options.override_active_consumers {
                self.log(
                    &entry.id,
                    &format!(
                        "cutover: --override: unregistered external pid {} at {} bypassed",
                        occupant.pid, occupant.endpoint
                    ),
                );
                let _ = self.ledger.record_as(
                    "override",
                    &entry.id,
                    0.0,
                    format!(
                        "external occupant override (pid {} at {})",
                        occupant.pid, occupant.endpoint
                    ),
                    options.actor.as_deref(),
                );
            } else {
                self.record_blocked_as(
                    &entry.id,
                    &format!(
                        "blocked:external-occupant: unregistered {} pid {} at {};                          run `swap adopt` or stop it, or pass --override",
                        occupant.engine, occupant.pid, occupant.endpoint
                    ),
                    options.actor.as_deref(),
                );
                return Err(SwapError::ExternalOccupant {
                    model: entry.id.clone(),
                    pid: occupant.pid,
                    endpoint: occupant.endpoint,
                }
                .into());
            }
        }
        let predecessors = self.exclusive_predecessors(entry);
        if predecessors.is_empty() {
            return Ok(());
        }
        for (model, pid) in predecessors {
            if let Some(cancel) = cancel {
                if cancel.is_cancelled() {
                    return Err(SwapError::Cancelled(entry.id.clone()).into());
                }
            }
            if options.override_active_consumers {
                self.log(&model, "cutover: --override: protection span bypassed");
                let _ = self.ledger.record_as(
                    "override",
                    &model,
                    0.0,
                    "active-consumer override (cutover)",
                    options.actor.as_deref(),
                );
            } else if let Some(detail) = self.protection(&model) {
                self.log(
                    &model,
                    &format!("cutover: active consumer at SIGTERM point ({detail}); re-check queued"),
                );
                tokio::time::sleep(self.config.cutover_recheck).await;
                if let Some(detail) = self.protection(&model) {
                    self.record_blocked_as(
                        &model,
                        &format!("blocked:active-consumer: cutover re-check still protected ({detail})"),
                        options.actor.as_deref(),
                    );
                    return Err(SwapError::BlockedActiveConsumer { model, detail }.into());
                }
                self.log(&model, "cutover: re-check passed");
            }
            self.set_state(&model, LifecycleState::Draining, None);
            self.log(
                &model,
                &format!(
                    "cutover: SIGTERM pid={}",
                    pid.map(|p| p.to_string()).unwrap_or_else(|| "adopted".into())
                ),
            );
            self.stop_native_inner(&model, "exclusive cutover", true)?;
            if let Some(pid) = pid {
                match crate::engine_process::wait_pid_gone(pid, self.config.stop_grace) {
                    Some(ms) => self.log(&model, &format!("cutover: pid gone after {ms} ms")),
                    None => self.log(
                        &model,
                        "cutover: pid still alive after the stop grace (escalated to SIGKILL)",
                    ),
                }
            }
        }
        if !self.config.settle.is_zero() {
            self.log(
                &entry.id,
                &format!("cutover: settle {} ms", self.config.settle.as_millis()),
            );
            tokio::time::sleep(self.config.settle).await;
        }
        Ok(())
    }

    fn set_gate(&self, id: &str, gate: GateStatus) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let runtime = inner.models.entry(id.to_string()).or_default();
        runtime.state = LifecycleState::Waiting;
        runtime.gate = Some(gate);
    }

    fn clear_gate(&self, id: &str) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(runtime) = inner.models.get_mut(id) {
            runtime.gate = None;
            if runtime.state == LifecycleState::Waiting {
                runtime.state = LifecycleState::Stopped;
            }
        }
    }

    /// `MemAvailable ≥ adapter.min_mem_gib` (spec: after drain + settle;
    /// 10-minute timeout ⇒ Failed with reason + `--force-gate`).
    async fn wait_gate(
        &self,
        entry: &CatalogEntry,
        options: &EnsureOptions,
        cancel: Option<&crate::inflight::InflightHandle>,
    ) -> Result<()> {
        let Some(adapter) = self.adapter_for(&entry.engine) else {
            return Ok(());
        };
        let threshold = adapter.min_mem_gib().unwrap_or(self.config.min_mem_gib);
        if threshold <= 0.0 {
            return Ok(());
        }
        let started = std::time::Instant::now();
        let mut announced = false;
        loop {
            let available = self.mem_available_gib();
            if available + 1e-9 >= threshold {
                self.clear_gate(&entry.id);
                self.log(
                    &entry.id,
                    &format!(
                        "cutover: mem gate ok (available {available:.1} ≥ {threshold:.1} GiB)"
                    ),
                );
                return Ok(());
            }
            if options.force_gate {
                let _ = self.ledger.record_as(
                    "override",
                    &entry.id,
                    threshold,
                    format!(
                        "force-gate: MemAvailable {available:.1} GiB < {threshold:.1} GiB floor"
                    ),
                    options.actor.as_deref(),
                );
                self.log(&entry.id, "gate: --force-gate override (audit-logged)");
                let _ = self.ledger.record_as(
                    "gate",
                    &entry.id,
                    threshold,
                    format!("gate bypassed by force-gate at {available:.1} GiB"),
                    options.actor.as_deref(),
                );
                self.clear_gate(&entry.id);
                return Ok(());
            }
            let waited = started.elapsed().as_millis() as u64;
            self.set_gate(
                &entry.id,
                GateStatus {
                    threshold_gib: threshold,
                    current_gib: available,
                    waited_ms: waited,
                    timeout_ms: self.config.gate_timeout.as_millis() as u64,
                },
            );
            if !announced {
                announced = true;
                self.record_blocked_as(
                    &entry.id,
                    &format!(
                        "blocked:min-mem-gate: needs MemAvailable ≥ {threshold:.1} GiB \
                         (probe reports {available:.1} GiB)"
                    ),
                    options.actor.as_deref(),
                );
            }
            if started.elapsed() >= self.config.gate_timeout {
                self.clear_gate(&entry.id);
                let reason = format!(
                    "min_mem gate: `{}` needs MemAvailable ≥ {threshold:.1} GiB but the probe \
                     reports {available:.1} GiB after {} ms (pass --force-gate to override)",
                    entry.id,
                    self.config.gate_timeout.as_millis()
                );
                self.log(&entry.id, &format!("gate: failed: {reason}"));
                self.set_state(&entry.id, LifecycleState::Failed, Some(reason.clone()));
                let _ = self.ledger.record("gate-fail", &entry.id, threshold, reason);
                return Err(SwapError::GateBlocked {
                    model: entry.id.clone(),
                    threshold_gib: threshold,
                    available_gib: available,
                }
                .into());
            }
            if let Some(cancel) = cancel {
                if cancel.is_cancelled() {
                    self.clear_gate(&entry.id);
                    return Err(SwapError::Cancelled(entry.id.clone()).into());
                }
            }
            tokio::time::sleep(self.config.gate_poll).await;
        }
    }

    fn resident_ready(&self, entry: &CatalogEntry) -> Option<Resident> {
        let view = self.residency();
        view.ready_for(
            &entry.id,
            entry.canonical_gguf().as_deref(),
            Some(&entry.served_alias()),
        )
        .cloned()
    }

    fn spawn_lock(&self, id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.spawn_locks.lock().unwrap_or_else(|e| e.into_inner());
        Arc::clone(
            locks
                .entry(id.to_string())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
        )
    }

    /// Register a queue row (cap checked). The returned entry is removed when
    /// the ensure returns (its guard drops).
    fn enqueue(&self, model: &str) -> Result<Arc<QueueEntry>> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.queue.len() >= self.config.queue_cap {
            return Err(SwapError::QueueFull {
                model: model.to_string(),
                cap: self.config.queue_cap,
            }
            .into());
        }
        inner.next_queue_id += 1;
        let entry = Arc::new(QueueEntry {
            id: inner.next_queue_id,
            model: model.to_string(),
            requested_ms: now_ms(),
            cancelled: Arc::new(AtomicBool::new(false)),
        });
        inner.queue.push(QueueEntry {
            id: entry.id,
            model: entry.model.clone(),
            requested_ms: entry.requested_ms,
            cancelled: Arc::clone(&entry.cancelled),
        });
        Ok(entry)
    }

    fn remove_queue(&self, id: u64) {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .queue
            .retain(|q| q.id != id);
    }

    /// Public queue cancellation (dashboard / inflight row).
    pub fn cancel(&self, queue_id: u64) -> bool {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        match inner.queue.iter().find(|q| q.id == queue_id) {
            Some(entry) => {
                entry.cancelled.store(true, Ordering::SeqCst);
                true
            }
            None => false,
        }
    }

    fn start_spawn(&self, entry: &CatalogEntry) -> Result<EngineHandle> {
        let adapter = self.adapter_for(&entry.engine).ok_or_else(|| {
            SwapError::SpawnFailed(format!(
                "no adapter for engine `{}` (managed externally)",
                entry.engine
            ))
        })?;
        // L5: probe each adapter once (engine install marker / version-keyed
        // schema derivation). A probe failure is logged, never fatal: the
        // spawn attempt itself is the authority (L4 behavior preserved).
        if self
            .probed_adapters
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(adapter.name().to_string())
        {
            if let Err(error) = adapter.probe() {
                self.log(
                    &entry.id,
                    &format!("adapter `{}` probe failed: {error}", adapter.name()),
                );
            }
        }
        // L5: adapter-owned allocation (strixite conf scope, hipfire
        // single-owner port) avoids every Residency endpoint, imported
        // explicit ports and llama-swap's dynamic range; otherwise the L4
        // generic scan, whose result is RESERVED in the port set so a
        // concurrent spawn cannot collide with it.
        let avoid = self.avoided_ports();
        let port = adapter
            .allocate_port(&avoid)
            .or_else(|| self.reserve_port())
            .ok_or_else(|| {
                SwapError::SpawnFailed("no free port in the swap port range".into())
            })?;
        let endpoint = format!("http://127.0.0.1:{port}");
        let gguf = entry.canonical_gguf();
        let scope_name = Some(format!(
            "brainrouter-swap-{}-{}",
            sanitize(&entry.id),
            uuid::Uuid::new_v4().simple()
        ));
        let log_path = self.logs_dir.join(format!("{}.log", sanitize(&entry.id)));
        let request = SpawnRequest {
            model_id: entry.id.clone(),
            spec: entry.launch_spec.clone(),
            gguf: gguf.clone(),
            port,
            endpoint: endpoint.clone(),
            log_path,
            scope_name: scope_name.clone(),
            env: {
                let mut env = entry.launch_spec.env.clone();
                // Fake-adapter scripting keys may also live at top level of
                // the entry's env (same map today; kept explicit).
                env.extend(std::env::vars().filter(|(k, _)| {
                    k.starts_with("FAKE_") || k.starts_with("BRAINROUTER_SWAP_")
                }));
                env
            },
        };
        // Marker is written BEFORE spawn (atomic): a daemon crash between the
        // two leaves a marker with a dead pid → cleaned on boot; a spawn
        // without a marker would be an orphan we could not adopt.
        let marker = AdoptionMarker {
            model: entry.id.clone(),
            pid: 0,
            gguf: gguf
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            start_time: 0,
            spec_hash: crate::swap_catalog::launch_spec_hash(entry),
            scope_name: scope_name.clone(),
            endpoint: endpoint.clone(),
        };
        self.write_marker(&marker)?;
        let handle = match adapter.spawn(&request) {
            Ok(handle) => handle,
            Err(e) => {
                self.release_endpoint_port(&endpoint);
                self.remove_marker(&entry.id);
                self.set_state(
                    &entry.id,
                    LifecycleState::Failed,
                    Some(format!("{e}")),
                );
                return Err(SwapError::SpawnFailed(format!("{e}")).into());
            }
        };
        let mut marker = marker;
        marker.pid = handle.pid;
        marker.start_time = process_start_time(handle.pid).unwrap_or(0);
        let _ = self.write_marker(&marker);
        self.ledger
            .record("spawn", &entry.id, self.need_gib(entry), format!("pid {}", handle.pid))?;
        {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            let restarts = inner
                .models
                .get(&entry.id)
                .map(|r| r.restarts)
                .unwrap_or(0);
            inner.models.insert(
                entry.id.clone(),
                ModelRuntime {
                    state: LifecycleState::Loading,
                    endpoint: Some(handle.endpoint.clone()),
                    handle: Some(handle.clone()),
                    source: Some(ResidentSource::Native),
                    last_error: None,
                    restarts,
                    load_started_ms: now_ms(),
                    spec_stale: false,
                    port: Some(port),
                    conf_path: adapter
                        .conf_path_for(&entry.id)
                        .map(|path| path.display().to_string()),
                    gate: None,
                },
            );
        }
        self.log(
            &entry.id,
            &format!(
                "spawn {} engine={} pid={} endpoint={} scope={}",
                entry.id,
                entry.engine,
                handle.pid,
                handle.endpoint,
                handle.scope_name.clone().unwrap_or_default()
            ),
        );
        Ok(handle)
    }

    /// Ports that must never be allocated for a managed child: every
    /// Residency endpoint, imported explicit `--port` values and llama-swap's
    /// configured dynamic `${PORT}` range.
    fn avoided_ports(&self) -> BTreeSet<u16> {
        let mut ports = crate::engine_strixite::avoidance_from_catalog(&self.catalog.snapshot());
        let view = self.residency();
        ports.extend(crate::engine_strixite::endpoint_ports(
            view.residents.iter().map(|resident| resident.endpoint.as_str()),
        ));
        ports
    }

    async fn finish_spawn(
        &self,
        entry: &CatalogEntry,
        handle: EngineHandle,
        cancel: Option<&crate::inflight::InflightHandle>,
        queue: &QueueEntry,
    ) -> Result<EnsureOutcome> {
        let adapter = self
            .adapter_for(&entry.engine)
            .ok_or_else(|| SwapError::SpawnFailed("adapter disappeared".into()))?;
        let readiness_timeout = adapter.readiness_timeout().unwrap_or(self.config.wait_timeout);
        let started = std::time::Instant::now();
        loop {
            if cancel_requested(cancel, queue) {
                self.stop_native_inner(&entry.id, "request cancelled", true)?;
                return Err(SwapError::Cancelled(entry.id.clone()).into());
            }
            if !pid_alive(handle.pid) {
                return self
                    .handle_spawn_failure(entry, &handle, "engine process exited during load", cancel, queue)
                    .await;
            }
            match adapter.health(&handle.endpoint, &handle) {
                Ok(report) if report.ready => {
                    if !report.stems.is_empty()
                        && !report.stems.contains(&entry.served_alias())
                        && !report.stems.contains(&entry.id)
                    {
                        return self
                            .handle_spawn_failure(
                                entry,
                                &handle,
                                &format!(
                                    "stem assertion failed: engine serves {:?}, catalog expects `{}`",
                                    report.stems,
                                    entry.served_alias()
                                ),
                                cancel,
                                queue,
                            )
                            .await;
                    }
                    let load_ms = started.elapsed().as_millis() as f64;
                    let _ = self.catalog.record_load_ms(&entry.id, load_ms);
                    {
                        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                        if let Some(runtime) = inner.models.get_mut(&entry.id) {
                            runtime.state = LifecycleState::Ready;
                            runtime.endpoint = Some(handle.endpoint.clone());
                            runtime.handle = Some(handle.clone());
                            runtime.source = Some(ResidentSource::Native);
                            runtime.last_error = None;
                            runtime.spec_stale = false;
                            // A successful load closes the restart window.
                            runtime.restarts = 0;
                        }
                    }
                    self.log(
                        &entry.id,
                        &format!("ready in {:.0} ms at {}", load_ms, handle.endpoint),
                    );
                    return Ok(EnsureOutcome::Attach {
                        model: entry.id.clone(),
                        endpoint: handle.endpoint,
                        source: ResidentSource::Native,
                        spawned: true,
                    });
                }
                Ok(_) => {}
                Err(e) => {
                    // Adapters answer `Ok(loading)` for a live-but-not-yet-
                    // listening process; `Err` means the child is gone (a
                    // reaped crash, not a transient connection).
                    return self
                        .handle_spawn_failure(entry, &handle, &format!("{e}"), cancel, queue)
                        .await;
                }
            }
            if started.elapsed() > readiness_timeout {
                self.stop_native_inner(&entry.id, "load timeout", true)?;
                return Err(SwapError::Timeout {
                    model: entry.id.clone(),
                    waited_ms: started.elapsed().as_millis() as u64,
                }
                .into());
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    async fn handle_spawn_failure(
        &self,
        entry: &CatalogEntry,
        handle: &EngineHandle,
        reason: &str,
        cancel: Option<&crate::inflight::InflightHandle>,
        queue: &QueueEntry,
    ) -> Result<EnsureOutcome> {
        // Stop (and reap) the failed child BEFORE the retry: a live holder of
        // the GGUF flock would make every restart fail with "locked".
        if let Some(adapter) = self.adapter_for(&entry.engine) {
            adapter
                .stop(handle)
                .map_err(|e| SwapError::SpawnFailed(format!("stop failed: {e}")))?;
        } else {
            terminate_pid(handle.pid, self.config.stop_grace);
            if pid_alive(handle.pid) {
                return Err(SwapError::SpawnFailed(format!(
                    "stop failed: process {} is still alive",
                    handle.pid
                ))
                .into());
            }
        }
        self.release_endpoint_port(&handle.endpoint);
        self.remove_marker(&entry.id);
        let restarts = {
            let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.models.get(&entry.id).map(|r| r.restarts).unwrap_or(0)
        };
        self.log(&entry.id, &format!("spawn failure: {reason} (restarts {restarts})"));
        if restarts < self.config.max_restarts {
            if let Some(runtime) = self.inner.lock().unwrap_or_else(|e| e.into_inner()).models.get_mut(&entry.id) {
                runtime.restarts += 1;
                runtime.state = LifecycleState::Stopped;
                runtime.handle = None;
            }
            let _ = self.evict_to_fit(self.need_gib(entry));
            // Hung/assert-failed starts release their memory before the retry:
            // re-run the `min_mem_gib` gate (spec: "re-run gate").
            self.wait_gate(entry, &EnsureOptions::default(), cancel).await?;
            let handle = self.start_spawn(entry)?;
            return Box::pin(self.finish_spawn(entry, handle, cancel, queue)).await;
        }
        self.set_state(
            &entry.id,
            LifecycleState::Failed,
            Some(reason.to_string()),
        );
        Err(SwapError::SpawnFailed(reason.to_string()).into())
    }

    async fn wait_ready(
        &self,
        id: &str,
        cancel: Option<&crate::inflight::InflightHandle>,
        queue: &QueueEntry,
    ) -> Result<EnsureOutcome> {
        let deadline = std::time::Instant::now() + self.config.wait_timeout;
        loop {
            if cancel_requested(cancel, queue) {
                return Err(SwapError::Cancelled(id.to_string()).into());
            }
            {
                let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(runtime) = inner.models.get(id) {
                    match runtime.state {
                        LifecycleState::Ready => {
                            return Ok(EnsureOutcome::Attach {
                                model: id.to_string(),
                                endpoint: runtime.endpoint.clone().unwrap_or_default(),
                                source: runtime
                                    .source
                                    .unwrap_or(ResidentSource::Native),
                                spawned: false,
                            })
                        }
                        LifecycleState::Failed => {
                            return Err(SwapError::SpawnFailed(
                                runtime
                                    .last_error
                                    .clone()
                                    .unwrap_or_else(|| "model failed".into()),
                            )
                            .into())
                        }
                        _ => {}
                    }
                }
            }
            if std::time::Instant::now() > deadline {
                return Err(SwapError::Timeout {
                    model: id.to_string(),
                    waited_ms: self.config.wait_timeout.as_millis() as u64,
                }
                .into());
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// Explicit eviction (`swap evict` / dashboard). Refuses an unknown model.
    pub fn evict(&self, id: &str, reason: &str) -> Result<bool> {
        self.evict_with_options(id, reason, false)
    }

    /// Explicit eviction. Refuses a protected consumer unless the one-action
    /// override is set (block reason `blocked:active-consumer`).
    pub fn evict_with_options(
        &self,
        id: &str,
        reason: &str,
        override_active_consumers: bool,
    ) -> Result<bool> {
        self.evict_with_actor(id, reason, override_active_consumers, None)
    }

    pub fn evict_with_actor(
        &self,
        id: &str,
        reason: &str,
        override_active_consumers: bool,
        actor: Option<&str>,
    ) -> Result<bool> {
        let resolved = self
            .catalog
            .resolve_alias(id)
            .ok_or_else(|| SwapError::UnknownModel(id.to_string()))?;
        if !override_active_consumers {
            if let Some(detail) = self.protection(&resolved) {
                self.record_blocked_as(
                    &resolved,
                    &format!("blocked:active-consumer: explicit evict refused ({detail})"),
                    actor,
                );
                return Err(SwapError::BlockedActiveConsumer {
                    model: resolved,
                    detail,
                }
                .into());
            }
        } else {
            let _ = self.ledger.record_as(
                "override",
                &resolved,
                0.0,
                format!("active-consumer override ({reason})"),
                actor,
            );
        }
        let had = self
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .models
            .get(&resolved)
            .map(|r| r.handle.is_some())
            .unwrap_or(false);
        self.stop_native(&resolved, reason, true, override_active_consumers)?;
        Ok(had)
    }

    fn runtime(&self, id: &str) -> Option<ModelRuntime> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.models.get(id).map(|r| ModelRuntime {
            state: r.state,
            endpoint: r.endpoint.clone(),
            handle: r.handle.clone(),
            source: r.source,
            last_error: r.last_error.clone(),
            restarts: r.restarts,
            load_started_ms: r.load_started_ms,
            spec_stale: r.spec_stale,
            port: r.port,
            conf_path: r.conf_path.clone(),
            gate: r.gate.clone(),
        })
    }

    fn runtime_state(&self, id: &str) -> LifecycleState {
        self.runtime(id)
            .map(|r| r.state)
            .unwrap_or(LifecycleState::Stopped)
    }

    fn set_state(&self, id: &str, state: LifecycleState, error: Option<String>) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let runtime = inner.models.entry(id.to_string()).or_default();
        runtime.state = state;
        if error.is_some() {
            runtime.last_error = error;
        }
    }

    /// Boot scan: live pid + gguf match ⇒ ADOPT (spec_hash mismatch ⇒ adopt +
    /// `spec_stale`); dead pid / stale marker ⇒ cleanup. Never kills a child.
    pub fn scan_adoption(&self) -> AdoptionReport {
        let mut report = AdoptionReport::default();
        let entries = match std::fs::read_dir(&self.adoption_dir) {
            Ok(dir) => dir.flatten().map(|e| e.path()).collect::<Vec<_>>(),
            Err(_) => return report,
        };
        for path in entries {
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok(marker) = serde_json::from_str::<AdoptionMarker>(&text) else {
                let _ = std::fs::remove_file(&path);
                report
                    .cleaned
                    .push(path.file_stem().unwrap_or_default().to_string_lossy().into_owned());
                continue;
            };
            let live_pid = pid_alive(marker.pid);
            // Identity = live pid + the recorded kernel start-time + the GGUF
            // as a FULL argv element. An unknown start_time cannot prove the
            // pid is the process we started (pid reuse), so adoption is
            // refused and the marker left for the operator.
            let start_time_matches = marker.start_time != 0
                && process_start_time(marker.pid) == Some(marker.start_time);
            let args = if live_pid && start_time_matches {
                process_args(marker.pid)
            } else {
                Vec::new()
            };
            let gguf_matches = if marker.gguf.is_empty() {
                true
            } else {
                let want = std::fs::canonicalize(&marker.gguf)
                    .unwrap_or_else(|_| PathBuf::from(&marker.gguf));
                args.iter().any(|arg| {
                    arg == &marker.gguf
                        || std::fs::canonicalize(arg).is_ok_and(|arg| arg == want)
                })
            };
            if live_pid && marker.start_time == 0 {
                let warning = format!(
                    "refusing to adopt `{}` (pid {}): the marker has no start_time,                      so pid identity cannot be proven",
                    marker.model, marker.pid
                );
                self.log(&marker.model, &warning);
                report.warnings.push(warning);
                continue;
            }
            if !live_pid || !start_time_matches || !gguf_matches {
                let _ = std::fs::remove_file(&path);
                report.cleaned.push(marker.model.clone());
                let _ = self.ledger.record("stop", &marker.model, 0.0, "stale adoption marker cleaned");
                continue;
            }
            let expected_hash = self
                .catalog
                .entry(&marker.model)
                .map(|e| crate::swap_catalog::launch_spec_hash(&e));
            // An adopted icon-era external carries the PORT-LESS hash
            // (profile template + weights); it must not wear a permanent
            // `spec_stale` badge (L5 binding).
            let adopted_hash = self
                .catalog
                .entry(&marker.model)
                .map(|e| crate::swap_catalog::adopted_spec_hash(&e));
            let stale = expected_hash
                .as_deref()
                .is_some_and(|hash| hash != marker.spec_hash)
                && adopted_hash
                    .as_deref()
                    .is_some_and(|hash| hash != marker.spec_hash);
            if expected_hash.is_none() {
                let warning = format!(
                    "adopted unnamed resident `{}` (pid {}): it is serving but has no \
                     catalog entry, so its launch spec cannot be compared",
                    marker.model, marker.pid
                );
                self.log(&marker.model, &warning);
                report.catalog_missing.push(marker.model.clone());
                report.warnings.push(warning);
            }
            {
                let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                inner.models.insert(
                    marker.model.clone(),
                    ModelRuntime {
                        state: LifecycleState::Ready,
                        endpoint: Some(marker.endpoint.clone()),
                        handle: Some(EngineHandle {
                            pid: marker.pid,
                            endpoint: marker.endpoint.clone(),
                            started_at: now_ms(),
                            scope_name: marker.scope_name.clone(),
                        }),
                        source: Some(ResidentSource::Native),
                        last_error: None,
                        restarts: 0,
                        load_started_ms: 0,
                        spec_stale: stale,
                        port: Some(
                            crate::engine_strixite::endpoint_ports([marker.endpoint.as_str()])
                                .into_iter()
                                .next()
                                .unwrap_or(0),
                        )
                        .filter(|port| *port > 0),
                        conf_path: None,
                        gate: None,
                    },
                );
            }
            if let Some(port) = marker
                .endpoint
                .trim_end_matches('/')
                .rsplit(':')
                .next()
                .and_then(|p| p.parse::<u16>().ok())
            {
                self.reserved_ports
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(port);
            }
            let _ = self
                .ledger
                .record("adopt", &marker.model, 0.0, format!("pid {}", marker.pid));
            report.adopted.push(marker.model.clone());
            if stale {
                report.spec_stale.push(marker.model);
            }
        }
        *self.last_adoption.lock().unwrap_or_else(|e| e.into_inner()) = report.clone();
        report
    }

    /// Periodic maintenance: crash detection + bounded restart, and native
    /// residency/health refresh. Safe to call every second.
    pub async fn maintenance(self: &Arc<Self>) {
        let ids: Vec<String> = {
            let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner
                .models
                .iter()
                .filter(|(_, r)| {
                    matches!(r.state, LifecycleState::Ready | LifecycleState::Loading)
                })
                .map(|(id, _)| id.clone())
                .collect()
        };
        for id in ids {
            let runtime = match self.runtime(&id) {
                Some(r) => r,
                None => continue,
            };
            let Some(handle) = runtime.handle.clone() else {
                continue;
            };
            let alive = pid_alive(handle.pid);
            let healthy = alive
                && self
                    .adapter_for(
                        &self
                            .catalog
                            .entry(&id)
                            .map(|e| e.engine)
                            .unwrap_or_else(|| "llama_cpp".into()),
                    )
                    .map(|adapter| {
                        adapter
                            .health(&handle.endpoint, &handle)
                            .map(|r| r.ready)
                            .unwrap_or(false)
                    })
                    .unwrap_or(false);
            if runtime.state == LifecycleState::Loading && healthy {
                // A cancelled ensure future may have dropped mid-load while
                // the child kept loading: finalize it here (stem assertion
                // included) instead of leaving a permanent Loading row.
                let stems_ok = self
                    .adapter_for(
                        &self
                            .catalog
                            .entry(&id)
                            .map(|e| e.engine)
                            .unwrap_or_else(|| "llama_cpp".into()),
                    )
                    .and_then(|adapter| adapter.health(&handle.endpoint, &handle).ok())
                    .map(|report| {
                        report.ready
                            && (report.stems.is_empty()
                                || self
                                    .catalog
                                    .entry(&id)
                                    .map(|e| {
                                        report.stems.contains(&e.served_alias())
                                            || report.stems.contains(&e.id)
                                    })
                                    .unwrap_or(false))
                    })
                    .unwrap_or(false);
                if stems_ok {
                    let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(r) = inner.models.get_mut(&id) {
                        // Re-check under the same lock: a cutover may have
                        // drained this model between the health probe and here.
                        if r.state == LifecycleState::Loading {
                            r.state = LifecycleState::Ready;
                            r.last_error = None;
                            r.restarts = 0;
                        }
                    }
                    self.log(&id, "recovered ready state after a dropped ensure future");
                }
            }
            if runtime.state != LifecycleState::Ready || healthy {
                continue;
            }
            // Crash restart is serialized like an admission: the per-device
            // lock and the model's spawn lock are taken in the same order as
            // `ensure`, then the state is re-read. Without this, a cutover
            // that drained the model while the health probe ran could be
            // resurrected here (co-residency on an exclusive device).
            let engine = self
                .catalog
                .entry(&id)
                .map(|e| e.engine)
                .unwrap_or_else(|| "llama_cpp".into());
            let _device_guard = match self
                .adapter_for(&engine)
                .and_then(|adapter| adapter.exclusive_key())
            {
                Some(device) => Some(self.device_lock(&device).lock_owned().await),
                None => None,
            };
            let lock = self.spawn_lock(&id);
            let _guard = lock.lock().await;
            let current = match self.runtime(&id) {
                Some(r) => r,
                None => continue,
            };
            if current.state != LifecycleState::Ready {
                continue;
            }
            if current
                .handle
                .as_ref()
                .is_some_and(|handle| pid_alive(handle.pid))
            {
                continue; // the probe raced a recovery
            }
            let (restarts, old_endpoint) = {
                let mut inner =
                    self.inner.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(r) = inner.models.get_mut(&id) {
                    r.restarts += 1;
                    r.state = LifecycleState::Stopped;
                    r.handle = None;
                    r.last_error = Some("crashed; restarting".into());
                    (r.restarts, r.endpoint.take())
                } else {
                    (0, None)
                }
            };
            if let Some(endpoint) = old_endpoint {
                self.release_endpoint_port(&endpoint);
            }
            self.remove_marker(&id);
            self.log(&id, "crash detected; bounded restart");
            if restarts <= self.config.max_restarts {
                if let Some(entry) = self.catalog.entry(&id) {
                    if let Ok(handle) = self.start_spawn(&entry) {
                        // Validate in the background; ensure callers that
                        // race this still see Loading → Ready.
                        let this = Arc::clone(self);
                        let restart_queue = QueueEntry {
                            id: 0,
                            model: entry.id.clone(),
                            requested_ms: now_ms(),
                            cancelled: Arc::new(AtomicBool::new(false)),
                        };
                        tokio::spawn(async move {
                            let _ = this
                                .finish_spawn(&entry, handle, None, &restart_queue)
                                .await;
                        });
                    }
                }
            } else {
                self.set_state(&id, LifecycleState::Failed, Some("restart budget exhausted".into()));
            }
        }
    }

    /// Snapshot for the Engines tab / `swap status`.
    pub fn status(&self) -> EnginesStatus {
        let data = self.catalog.snapshot();
        let view = self.residency();
        let queue = {
            let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner
                .queue
                .iter()
                .enumerate()
                .map(|(index, entry)| QueueStatus {
                    id: entry.id,
                    model: entry.model.clone(),
                    position: index + 1,
                    waited_ms: now_ms().saturating_sub(entry.requested_ms),
                })
                .collect::<Vec<_>>()
        };
        let models = data
            .entries
            .values()
            .map(|entry| {
                let runtime = self.runtime(&entry.id);
                let resident = view.resident_for(&entry.id);
                let state = runtime
                    .as_ref()
                    .map(|r| r.state)
                    .filter(|s| *s != LifecycleState::Stopped)
                    .map(|s| s.as_str().to_string())
                    .or_else(|| resident.map(|r| r.state.clone()))
                    .unwrap_or_else(|| "stopped".into());
                let endpoint = runtime
                    .as_ref()
                    .and_then(|r| r.endpoint.clone())
                    .or_else(|| resident.map(|r| r.endpoint.clone()));
                let source = runtime
                    .as_ref()
                    .and_then(|r| r.source)
                    .or_else(|| resident.map(|r| r.source))
                    .map(|s| s.as_str().to_string());
                let progress = runtime.as_ref().and_then(|r| {
                    if r.state != LifecycleState::Loading {
                        return None;
                    }
                    let handle = r.handle.as_ref()?;
                    let adapter = self.adapter_for(&entry.engine)?;
                    let mut progress = adapter.progress(&handle.endpoint, handle);
                    // Native llama-server has no progress endpoint: fill the
                    // ETA from the registry's EWMA of measured past loads.
                    if progress.eta_ms.is_none() {
                        if let Some(ewma) = entry.ewma_load_ms {
                            progress.eta_ms =
                                Some((ewma as u64).saturating_sub(progress.elapsed_ms));
                        }
                    }
                    Some(progress)
                });
                ModelStatus {
                    id: entry.id.clone(),
                    name: entry.name.clone(),
                    engine: entry.engine.clone(),
                    state,
                    endpoint,
                    source,
                    resident: resident.is_some(),
                    spec_stale: runtime.as_ref().map(|r| r.spec_stale).unwrap_or(false)
                        || (resident.is_some() && entry.fingerprint_stale().is_some()),
                    last_error: runtime.as_ref().and_then(|r| r.last_error.clone()),
                    queue_position: queue
                        .iter()
                        .find(|q| q.model == entry.id)
                        .map(|q| q.position),
                    ewma_load_ms: entry.ewma_load_ms,
                    last_stream_completed: entry.last_stream_completed,
                    vram_estimate_gib: entry.vram_estimate_gib,
                    pinned_gib: data.pins.get(&entry.id).map(|p| p.reserved_gib),
                    progress,
                    gate: runtime.as_ref().and_then(|r| r.gate.clone()),
                    port: runtime.as_ref().and_then(|r| r.port),
                    conf_path: runtime
                        .as_ref()
                        .and_then(|r| r.conf_path.clone())
                        .or_else(|| {
                            self.adapter_for(&entry.engine)
                                .and_then(|adapter| adapter.conf_path_for(&entry.id))
                                .map(|path| path.display().to_string())
                        }),
                    active_consumer: self.consumer_span(&entry.id),
                }
            })
            .collect();
        let external = {
            let observed = self.observed_external();
            let markers = crate::engine_observer::marker_pids(&self.adoption_dir);
            observed
                .iter()
                .map(|engine| ExternalEngineView {
                    pid: engine.pid,
                    engine: engine.engine.clone(),
                    endpoint: engine.endpoint.clone(),
                    port: engine.port,
                    stems: engine.stems.clone(),
                    catalog_id: engine.catalog_id.clone(),
                    gguf: engine.gguf.clone(),
                    adopted: markers.contains(&engine.pid),
                })
                .collect()
        };
        let units = self.probe_units();
        let active_consumers = self
            .consumers()
            .map(|consumers| consumers.snapshot())
            .unwrap_or_default();
        let adapters = self
            .adapters
            .values()
            .filter_map(|adapter| adapter.describe())
            .collect();
        EnginesStatus {
            models,
            queue,
            budget: self.budget_report(),
            adoption: self
                .last_adoption
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
            shim_error: view.shim_error,
            adapter_names: self.adapters.keys().cloned().collect(),
            external,
            units,
            active_consumers,
            adapters,
        }
    }

    /// The ACS span for one model (live in-process stream or grace window).
    pub fn consumer_span(&self, id: &str) -> Option<ConsumerSpan> {
        let consumers = self.consumers()?;
        let endpoint = self.endpoint_for(id);
        for key in protection_keys(id, endpoint.as_deref()) {
            if let Some(span) = consumers
                .snapshot()
                .into_iter()
                .find(|span| span.key == key)
            {
                return Some(span);
            }
        }
        None
    }

    /// Configured units, probed pid-scoped (never by name pattern). Cached for
    /// two seconds (`/api/engines` is polled; one `systemctl show` per unit
    /// per poll is wasteful).
    pub fn probe_units(&self) -> Vec<UnitView> {
        if let Some((at, units)) = self
            .unit_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            if now_ms().saturating_sub(*at) < 2_000 {
                return units.clone();
            }
        }
        let units = self.probe_units_now();
        *self.unit_cache.lock().unwrap_or_else(|e| e.into_inner()) =
            Some((now_ms(), units.clone()));
        units
    }

    pub fn probe_units_now(&self) -> Vec<UnitView> {
        let runtime = self
            .unit_runtime
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let Some(runtime) = runtime else {
            return Vec::new();
        };
        let units: BTreeMap<String, String> = self
            .takeover_units
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        units
            .into_iter()
            .map(|(engine, unit)| {
                let state: UnitState = runtime.probe(&unit);
                UnitView {
                    engine,
                    unit,
                    active: state.active,
                    main_pid: state.main_pid,
                    error: state.error,
                }
            })
            .collect()
    }

    /// LRU clock: a completed stream for the model id (called from the
    /// SniffStream drop hook via the inflight registry).
    pub fn record_stream_completed(&self, id: &str) {
        let resolved = self.catalog.resolve_alias(id).unwrap_or_else(|| id.to_string());
        let _ = self
            .catalog
            .record_stream_completed(&resolved, now_ms());
    }

    // ── Takeover / adopt (L5) ──────────────────────────────────────────────

    /// Register an observed external pid as a supervisor-owned resident:
    /// marker with the PORT-LESS adopted hash (no permanent `spec_stale`),
    /// runtime Ready at the external endpoint. Never signals the pid.
    fn adopt_pid(
        &self,
        entry: &CatalogEntry,
        pid: u32,
        endpoint: &str,
        gguf: Option<String>,
        actor: Option<&str>,
    ) -> Result<AdoptReport> {
        if pid == 0 || !pid_alive(pid) {
            return Err(SwapError::SpawnFailed(format!(
                "cannot adopt pid {pid} for `{}`: process is not running",
                entry.id
            ))
            .into());
        }
        let gguf = gguf.or_else(|| {
            entry
                .canonical_gguf()
                .map(|path| path.display().to_string())
        });
        let spec_hash = crate::swap_catalog::adopted_spec_hash(entry);
        let marker = AdoptionMarker {
            model: entry.id.clone(),
            pid,
            gguf: gguf.clone().unwrap_or_default(),
            start_time: process_start_time(pid).unwrap_or(0),
            spec_hash: spec_hash.clone(),
            scope_name: None,
            endpoint: endpoint.to_string(),
        };
        self.write_marker(&marker)?;
        {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.models.insert(
                entry.id.clone(),
                ModelRuntime {
                    state: LifecycleState::Ready,
                    endpoint: Some(endpoint.to_string()),
                    handle: Some(EngineHandle {
                        pid,
                        endpoint: endpoint.to_string(),
                        started_at: now_ms(),
                        scope_name: None,
                    }),
                    source: Some(ResidentSource::Native),
                    last_error: None,
                    restarts: 0,
                    load_started_ms: 0,
                    spec_stale: false,
                    port: crate::engine_strixite::endpoint_ports([endpoint])
                        .into_iter()
                        .next(),
                    conf_path: None,
                    gate: None,
                },
            );
        }
        let _ = self.ledger.record_as(
            "adopt",
            &entry.id,
            0.0,
            format!("pid {pid} adopted from an external engine at {endpoint}"),
            actor,
        );
        self.log(
            &entry.id,
            &format!("adopt: external pid {pid} at {endpoint} (marker port-less hash)"),
        );
        Ok(AdoptReport {
            model: entry.id.clone(),
            pid,
            endpoint: endpoint.to_string(),
            gguf,
            spec_hash,
        })
    }

    /// `swap adopt <model> [--pid N]`: register an icon-launched resident so a
    /// later cutover may signal it (start-ownership transfers explicitly).
    pub fn adopt_external(&self, model: &str, pid: Option<u32>) -> Result<AdoptReport> {
        self.adopt_external_as(model, pid, None)
    }

    pub fn adopt_external_as(
        &self,
        model: &str,
        pid: Option<u32>,
        actor: Option<&str>,
    ) -> Result<AdoptReport> {
        let resolved = self
            .catalog
            .resolve_alias(model)
            .ok_or_else(|| SwapError::UnknownModel(model.to_string()))?;
        let entry = self
            .catalog
            .entry(&resolved)
            .ok_or_else(|| SwapError::UnknownModel(resolved.clone()))?;
        let observed = self.observed_external();
        let served_alias = entry.served_alias();
        let target = observed.iter().find(|engine| {
            if let Some(pid) = pid {
                return engine.pid == pid;
            }
            engine.catalog_id.as_deref() == Some(resolved.as_str())
                || engine.stems.iter().any(|stem| {
                    stem == &resolved || stem == &served_alias || stem == &entry.id
                })
                || engine.gguf.as_deref().is_some_and(|gguf| {
                    entry
                        .files
                        .iter()
                        .any(|file| file == gguf || file.ends_with(gguf))
                })
        });
        let Some(engine) = target.cloned() else {
            return Err(SwapError::SpawnFailed(format!(
                "no observed external engine serves `{model}` (refresh /api/engines/external)"
            ))
            .into());
        };
        self.adopt_pid(&entry, engine.pid, &engine.endpoint, engine.gguf.clone(), actor)
    }

    /// `swap adopt --endpoint <http://...>`: adopt whatever the observer sees
    /// at that endpoint. The catalog id comes from the observer's own
    /// resolution (gguf or served stems); an unregistered endpoint with no
    /// catalog entry is a named error, never a guessed model.
    pub fn adopt_endpoint(
        &self,
        endpoint: &str,
        pid: Option<u32>,
        actor: Option<&str>,
    ) -> Result<AdoptReport> {
        let normalized = endpoint.trim_end_matches('/');
        let observed = self.observed_external();
        let target = observed.iter().find(|engine| {
            engine.endpoint.trim_end_matches('/') == normalized
                || pid.is_some_and(|pid| engine.pid == pid)
        });
        let Some(engine) = target.cloned() else {
            return Err(SwapError::SpawnFailed(format!(
                "no observed external engine at `{endpoint}` (refresh /api/engines/external)"
            ))
            .into());
        };
        let id = engine
            .catalog_id
            .clone()
            .or_else(|| engine.stems.first().cloned())
            .ok_or_else(|| {
                SwapError::SpawnFailed(format!(
                    "observed external pid {} at {} serves no resolvable model;                      add a catalog entry first",
                    engine.pid, engine.endpoint
                ))
            })?;
        if self.catalog.resolve_alias(&id).is_none() {
            return Err(SwapError::SpawnFailed(format!(
                "observed external pid {} at {} serves `{id}`, which is not in the                  swap catalog; add the entry first",
                engine.pid, engine.endpoint
            ))
            .into());
        }
        self.adopt_external_as(&id, Some(engine.pid), actor)
    }

    /// `swap takeover <model>` / the dashboard takeover action. Attach-not-spawn
    /// when a resident (external observer or unit process) already serves the
    /// model; otherwise take over the operator unit pid-scoped (requires
    /// `--override`) and spawn through the normal admission path.
    pub async fn takeover(
        &self,
        requested: &str,
        options: TakeoverOptions,
    ) -> Result<TakeoverReport> {
        let resolved = self
            .catalog
            .resolve_alias(requested)
            .ok_or_else(|| SwapError::UnknownModel(requested.to_string()))?;
        let entry = self
            .catalog
            .entry(&resolved)
            .ok_or_else(|| SwapError::UnknownModel(resolved.clone()))?;

        // 1. Attach-not-spawn: anything already serving this model.
        if let Some(resident) = self.resident_ready(&entry) {
            self.log(
                &entry.id,
                &format!("takeover: attach at {} (no spawn)", resident.endpoint),
            );
            return Ok(TakeoverReport {
                model: entry.id.clone(),
                endpoint: resident.endpoint,
                attached: true,
                spawned: false,
                unit: None,
                stopped_unit: None,
                adopted_pid: None,
                actor: options.actor.clone(),
            });
        }

        // 2. Unit owned by the operator: attach, adopt, or (override) stop.
        let adapter = self.adapter_for(&entry.engine);
        let unit = options
            .unit
            .clone()
            .or_else(|| adapter.as_ref().and_then(|adapter| adapter.takeover_unit()))
            .or_else(|| {
                self.takeover_units
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(&entry.engine)
                    .cloned()
            });
        let runtime = self
            .unit_runtime
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let mut stopped_unit = None;
        let mut adopted_pid = None;
        if let (Some(unit), Some(runtime)) = (unit.clone(), runtime) {
            let state = runtime.probe(&unit);
            if state.active {
                let pids = runtime.pids(&unit);
                let endpoint = pids.iter().find_map(|pid| {
                    crate::engine_observer::parse_port(&process_cmdline(*pid))
                        .map(|port| format!("http://127.0.0.1:{port}"))
                });
                let serves_expected = endpoint.as_deref().is_some_and(|endpoint| {
                    crate::engine_process::probe_stems(endpoint, Duration::from_millis(800))
                        .is_some_and(|stems| {
                            stems.iter().any(|stem| {
                                stem == &entry.id || stem == &entry.served_alias()
                            })
                        })
                });
                if serves_expected {
                    let pid = pids
                        .first()
                        .copied()
                        .or(state.main_pid)
                        .ok_or_else(|| {
                            SwapError::SpawnFailed(format!(
                                "unit `{unit}` serves `{}` but exposes no pid to adopt",
                                entry.id
                            ))
                        })?;
                    let endpoint = endpoint.expect("serves_expected implies an endpoint");
                    let report = self.adopt_pid(
                        &entry,
                        pid,
                        &endpoint,
                        None,
                        options.actor.as_deref(),
                    )?;
                    self.log(
                        &entry.id,
                        &format!("takeover: adopted unit pid {pid} at {endpoint} (attach)"),
                    );
                    let _ = self.ledger.record_as(
                        "takeover",
                        &entry.id,
                        0.0,
                        format!("unit `{unit}` adopted pid {pid} at {endpoint}"),
                        options.actor.as_deref(),
                    );
                    return Ok(TakeoverReport {
                        model: entry.id.clone(),
                        endpoint: report.endpoint,
                        attached: true,
                        spawned: false,
                        unit: Some(unit),
                        stopped_unit: None,
                        adopted_pid: Some(pid),
                        actor: options.actor.clone(),
                    });
                }
                if !options.override_unit {
                    self.record_blocked_as(
                        &entry.id,
                        &format!("blocked:unit-owned: `{unit}` is active and not serving `{}`", entry.id),
                        options.actor.as_deref(),
                    );
                    return Err(SwapError::UnitOwned { unit }.into());
                }
                // Single-action takeover: pid-scoped SIGTERM to the pids the
                // unit itself reports, then stop the unit as cleanup.
                self.log(
                    &entry.id,
                    &format!(
                        "takeover: --override: stopping unit `{unit}` pid-scoped ({pids:?})"
                    ),
                );
                for pid in &pids {
                    crate::engine_process::terminate(*pid, self.config.stop_grace);
                    let _ = crate::engine_process::wait_pid_gone(*pid, self.config.stop_grace);
                }
                if let Err(error) = runtime.stop(&unit) {
                    self.log(
                        &entry.id,
                        &format!("takeover: systemctl stop `{unit}` reported: {error}"),
                    );
                }
                let _ = self.ledger.record_as(
                    "takeover",
                    &entry.id,
                    0.0,
                    format!("unit `{unit}` taken over (pids {pids:?})"),
                    options.actor.as_deref(),
                );
                stopped_unit = Some(unit);
                adopted_pid = None;
            }
        }

        // 3. Normal admission (spawn under our own scope).
        let outcome = self
            .ensure_with_options(
                &entry.id,
                None,
                EnsureOptions {
                    override_active_consumers: options.override_active_consumers
                        || options.override_unit,
                    force_gate: options.force_gate,
                    attach_only: false,
                    actor: options.actor.clone(),
                },
            )
            .await?;
        Ok(TakeoverReport {
            model: entry.id.clone(),
            endpoint: outcome.endpoint().to_string(),
            attached: !outcome.spawned(),
            spawned: outcome.spawned(),
            unit,
            stopped_unit,
            adopted_pid,
            actor: options.actor.clone(),
        })
    }

    // ── Markers + logs ─────────────────────────────────────────────────────

    fn marker_path(&self, id: &str) -> PathBuf {
        self.adoption_dir.join(format!("{}.json", sanitize(id)))
    }

    fn write_marker(&self, marker: &AdoptionMarker) -> Result<()> {
        let text = serde_json::to_string_pretty(marker)?;
        crate::plugins::store::atomic_write_0600(&self.marker_path(&marker.model), text.as_bytes())
            .with_context(|| format!("writing adoption marker for {}", marker.model))
    }

    fn remove_marker(&self, id: &str) {
        let _ = std::fs::remove_file(self.marker_path(id));
    }

    fn log(&self, id: &str, line: &str) {
        let _ = std::fs::create_dir_all(&self.logs_dir);
        let path = self.logs_dir.join(format!("{}.log", sanitize(id)));
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let _ = writeln!(
                file,
                "{} {}",
                chrono::Utc::now().to_rfc3339(),
                line
            );
        }
    }

    /// Tail a model's log (`GET /api/engines/logs?model=`).
    pub fn logs(&self, id: &str, lines: usize) -> Vec<String> {
        let path = self.logs_dir.join(format!("{}.log", sanitize(id)));
        let text = std::fs::read_to_string(path).unwrap_or_default();
        let all: Vec<&str> = text.lines().collect();
        all[all.len().saturating_sub(lines)..]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    pub fn logs_dir(&self) -> &Path {
        &self.logs_dir
    }

    #[cfg(test)]
    fn adoption_dir_for_test(&self) -> &Path {
        &self.adoption_dir
    }
}

/// Abort signal for a wait. A routed request (in-flight handle attached)
/// honors both its row cancellation and its queue-row cancellation. A direct
/// `POST /api/engines/ensure` caller carries no handle: its own disconnect
/// drops the future, and the queue row is informational for it (aborting the
/// synchronous API call would surface an error to a caller that never asked
/// to cancel).
fn cancel_requested(
    cancel: Option<&crate::inflight::InflightHandle>,
    queue: &QueueEntry,
) -> bool {
    match cancel {
        Some(handle) => {
            handle.is_cancelled() || queue.cancelled.load(Ordering::SeqCst)
        }
        None => false,
    }
}

fn sanitize(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn free_port(min: u16, max: u16) -> Option<u16> {
    (min..=max).find(|port| std::net::TcpListener::bind(("127.0.0.1", *port)).is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swap_catalog::{import_llama_swap_config, FileFingerprint};

    fn test_root(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "brainrouter-scheduler-{name}-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn catalog_with(entries: &[(&str, &str)]) -> (Arc<SwapCatalog>, PathBuf) {
        let root = test_root("catalog");
        let catalog = Arc::new(SwapCatalog::load(root.join("swap_catalog.json")).unwrap());
        for (id, extra) in entries {
            let gguf = root.join(format!("{id}.gguf"));
            std::fs::write(&gguf, b"gguf").unwrap();
            let text = format!(
                "models:\n  {id}:\n    cmd: |\n      /usr/bin/llama-server --port ${{PORT}} {extra} --model {}\n",
                gguf.display()
            );
            catalog
                .apply_import(&import_llama_swap_config(&text).unwrap())
                .unwrap();
        }
        let mut data = catalog.snapshot();
        for entry in data.entries.values_mut() {
            if entry.engine == "llama_cpp" {
                entry.engine = "fake".into();
                entry.launch_spec.program = "sleep".into();
            }
            entry.vram_estimate_gib = Some(10.0);
        }
        // Persist the fake engine swap through a fresh apply: use upsert.
        for entry in data.entries.values().cloned() {
            catalog.upsert_entry(entry).unwrap();
        }
        (catalog, root)
    }

    fn scheduler(name: &str, entries: &[(&str, &str)]) -> (Arc<SwapScheduler>, Arc<FakeAdapter>, PathBuf) {
        scheduler_cap(name, entries, None)
    }

    fn scheduler_cap(
        name: &str,
        entries: &[(&str, &str)],
        cap: Option<f64>,
    ) -> (Arc<SwapScheduler>, Arc<FakeAdapter>, PathBuf) {
        scheduler_with(name, entries, cap, true)
    }

    fn scheduler_with(
        name: &str,
        entries: &[(&str, &str)],
        cap: Option<f64>,
        kill_on_drop: bool,
    ) -> (Arc<SwapScheduler>, Arc<FakeAdapter>, PathBuf) {
        let (catalog, root) = catalog_with(entries);
        let ledger = Arc::new(BudgetLedger::load(root.join(BUDGET_FILE)).unwrap());
        let fake = Arc::new(FakeAdapter::new());
        let mut adapters: BTreeMap<String, Arc<dyn EngineAdapter>> = BTreeMap::new();
        adapters.insert("fake".into(), fake.clone());
        adapters.insert("external".into(), Arc::new(ExternalAdapter));
        let config = SwapSchedulerConfig {
            wait_timeout: Duration::from_secs(5),
            queue_cap: 2,
            stop_grace: Duration::from_millis(200),
            max_restarts: 2,
            port_min: 12500,
            port_max: 12550,
            llama_swap_url: "http://127.0.0.1:1".into(),
            // Fixtures: no settle sleep, zero grace (L5 seams are injectable).
            settle: Duration::ZERO,
            cutover_recheck: Duration::from_millis(50),
            gate_poll: Duration::from_millis(10),
            gate_timeout: Duration::from_millis(200),
            ..SwapSchedulerConfig::default()
        };
        let mut builder = SwapScheduler::new(catalog, ledger, config, adapters)
            .with_state_root(root.clone())
            .with_kill_children_on_drop(kill_on_drop);
        if let Some(cap) = cap {
            builder = builder.with_budget_cap(cap);
        }
        let scheduler = Arc::new(builder);
        let _ = name;
        (scheduler, fake, root)
    }

    #[tokio::test]
    async fn ensure_spawns_a_real_child_and_reaches_ready() {
        let (scheduler, fake, _root) = scheduler("spawn", &[("m", "")]);
        let outcome = scheduler.ensure("m").await.unwrap();
        assert!(outcome.spawned());
        assert!(
            outcome.endpoint().starts_with("http://127.0.0.1:"),
            "endpoint is a loopback port, got {}",
            outcome.endpoint()
        );
        assert_eq!(fake.spawn_count(), 1);
        assert_eq!(scheduler.status().models[0].state, "ready");
        assert!(scheduler.status().adoption.adopted.is_empty());
    }

    #[tokio::test]
    async fn concurrent_ensures_spawn_exactly_once() {
        let (scheduler, fake, _root) = scheduler("concurrent", &[("m", "")]);
        let futures = (0..5).map(|_| {
            let scheduler = Arc::clone(&scheduler);
            async move { scheduler.ensure("m").await.unwrap() }
        });
        let outcomes = futures_util::future::join_all(futures).await;
        assert_eq!(fake.spawn_count(), 1, "exactly one child spawn");
        assert_eq!(
            outcomes.iter().filter(|o| o.spawned()).count(),
            1,
            "exactly one caller owns the spawn; the rest attach"
        );
        let first_endpoint = outcomes[0].endpoint().to_string();
        for outcome in &outcomes {
            assert!(outcome.endpoint().starts_with("http://127.0.0.1:"));
            assert_eq!(outcome.endpoint(), first_endpoint, "one residency");
        }
    }

    #[tokio::test]
    async fn stem_assertion_failure_marks_the_model_failed() {
        let (scheduler, _fake, _root) = scheduler("stem", &[("m", "")]);
        // Script a stem set that does not contain the catalog id.
        let entry = scheduler.catalog().entry("m").unwrap();
        let mut entry = entry;
        entry
            .launch_spec
            .env
            .insert("FAKE_STEMS".into(), "someone-else".into());
        scheduler.catalog().upsert_entry(entry).unwrap();
        let error = scheduler.ensure("m").await.unwrap_err();
        assert!(error.to_string().contains("stem assertion failed"), "{error}");
        assert_eq!(scheduler.status().models[0].state, "failed");
    }

    #[tokio::test]
    async fn scripted_crash_is_restarted_within_the_budget() {
        let (scheduler, fake, _root) = scheduler("crash", &[("m", "")]);
        let mut entry = scheduler.catalog().entry("m").unwrap();
        entry
            .launch_spec
            .env
            .insert("FAKE_CRASH_AFTER_MS".into(), "60".into());
        entry
            .launch_spec
            .env
            .insert("FAKE_LOAD_MS".into(), "500".into());
        scheduler.catalog().upsert_entry(entry).unwrap();
        let error = scheduler.ensure("m").await.unwrap_err();
        assert!(error.to_string().contains("spawn failed"), "{error}");
        assert_eq!(fake.spawn_count(), 3, "initial + two bounded restarts");
        assert_eq!(scheduler.status().models[0].state, "failed");
    }

    #[tokio::test]
    async fn scripted_load_progress_is_reported_while_loading() {
        let (scheduler, _fake, _root) = scheduler("progress", &[("m", "")]);
        let mut entry = scheduler.catalog().entry("m").unwrap();
        entry
            .launch_spec
            .env
            .insert("FAKE_LOAD_MS".into(), "300".into());
        scheduler.catalog().upsert_entry(entry).unwrap();
        let scheduler2 = Arc::clone(&scheduler);
        let task = tokio::spawn(async move { scheduler2.ensure("m").await });
        tokio::time::sleep(Duration::from_millis(80)).await;
        let status = scheduler.status();
        assert_eq!(status.models[0].state, "loading");
        let progress = status.models[0].progress.clone().unwrap();
        assert!(progress.fraction.unwrap() > 0.0);
        task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn queue_cap_and_cancel() {
        let (scheduler, _fake, _root) = scheduler("queue", &[("m", "")]);
        let mut entry = scheduler.catalog().entry("m").unwrap();
        entry
            .launch_spec
            .env
            .insert("FAKE_LOAD_MS".into(), "200".into());
        scheduler.catalog().upsert_entry(entry).unwrap();
        let first = {
            let scheduler = Arc::clone(&scheduler);
            tokio::spawn(async move { scheduler.ensure("m").await })
        };
        tokio::time::sleep(Duration::from_millis(30)).await;
        // Cap is 2; the first ensure already holds one row.
        let second = {
            let scheduler = Arc::clone(&scheduler);
            tokio::spawn(async move { scheduler.ensure("m").await })
        };
        tokio::time::sleep(Duration::from_millis(30)).await;
        let third = {
            let scheduler = Arc::clone(&scheduler);
            tokio::spawn(async move { scheduler.ensure("m").await })
        };
        tokio::time::sleep(Duration::from_millis(30)).await;
        let error = third.await.unwrap().unwrap_err();
        assert!(error.to_string().contains("queue"), "{error}");
        let queue = scheduler.status().queue;
        assert!(!queue.is_empty());
        assert!(scheduler.cancel(queue[0].id));
        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn residency_attach_never_spawns_or_gates_budget() {
        let (scheduler, fake, _root) = scheduler("residency", &[("m", "")]);
        // Shim already serves the model.
        scheduler.set_shim_view(ResidencyView {
            residents: vec![Resident {
                catalog_id: "m".into(),
                source: ResidentSource::Shim,
                endpoint: "http://127.0.0.1:5801".into(),
                gguf_path: scheduler
                    .catalog()
                    .entry("m")
                    .and_then(|e| e.canonical_gguf())
                    .map(|p| p.display().to_string()),
                started_at: 0,
                last_stream_completed: 0,
                state: "ready".into(),
                spec_stale: false,
                engine: None,
            }],
            shim_error: None,
            observed_at: String::new(),
        });
        // Pin larger than the cap: an attach must still succeed.
        scheduler.catalog().pin("m", 10_000.0).unwrap();
        let outcome = scheduler.ensure("m").await.unwrap();
        assert!(!outcome.spawned());
        assert_eq!(outcome.endpoint(), "http://127.0.0.1:5801");
        assert_eq!(fake.spawn_count(), 0);
    }

    #[tokio::test]
    async fn pins_over_budget_make_spawning_ensures_no_fit() {
        let (scheduler, fake, _root) = scheduler_cap("nofit", &[("m", "")], Some(25.0));
        scheduler.catalog().pin("m", 100.0).unwrap();
        let error = scheduler.ensure("m").await.unwrap_err();
        assert!(error.to_string().contains("no-fit"), "{error}");
        assert_eq!(fake.spawn_count(), 0);
    }

    #[tokio::test]
    async fn lru_evicts_least_recent_stream() {
        let (scheduler, fake, _root) =
            scheduler_cap("lru", &[("a", ""), ("b", ""), ("c", "")], Some(20.0));
        scheduler.ensure("a").await.unwrap();
        scheduler.ensure("b").await.unwrap();
        // b completed a stream most recently; a never did (clock 0).
        scheduler.record_stream_completed("b");
        let outcome = scheduler.ensure("c").await.unwrap();
        assert!(outcome.spawned());
        let status = scheduler.status();
        let state = |id: &str| {
            status
                .models
                .iter()
                .find(|m| m.id == id)
                .unwrap()
                .state
                .clone()
        };
        assert_eq!(state("a"), "stopped", "least-recent stream is evicted");
        assert_eq!(state("b"), "ready");
        assert_eq!(state("c"), "ready");
        assert_eq!(status.budget.evictions_today, 1);
        assert_eq!(fake.spawn_count(), 3);
        // One aggregate ledger row per evict pass; a single victim keeps its
        // model id (greppable sequence).
        let evicts: Vec<String> = scheduler
            .ledger()
            .entries()
            .into_iter()
            .filter(|e| e.kind == "evict")
            .map(|e| e.model)
            .collect();
        assert_eq!(evicts, vec!["a".to_string()]);
    }

    #[tokio::test]
    async fn in_flight_models_are_never_evicted() {
        let (scheduler, _fake, _root) =
            scheduler_cap("inflight", &[("a", ""), ("b", ""), ("c", "")], Some(20.0));
        let registry = Arc::new(InflightRegistry::new());
        scheduler.attach_inflight(Arc::clone(&registry));
        scheduler.ensure("a").await.unwrap();
        scheduler.ensure("b").await.unwrap();
        // a is the LRU candidate but carries an active stream.
        let handle = registry.register(
            "POST /v1/chat/completions".into(),
            "a".into(),
            "ua".into(),
            "127.0.0.1:1".into(),
            String::new(),
            String::new(),
            0,
            None,
        );
        handle.set_catalog_id("a");
        scheduler.ensure("c").await.unwrap();
        let status = scheduler.status();
        let state = |id: &str| {
            status
                .models
                .iter()
                .find(|m| m.id == id)
                .unwrap()
                .state
                .clone()
        };
        assert_eq!(state("a"), "ready", "in-flight model survives eviction");
        assert_eq!(state("b"), "stopped");
        assert_eq!(state("c"), "ready");
        drop(handle);
    }

    #[tokio::test]
    async fn budget_ledger_is_append_only_and_persisted() {
        let (scheduler, _fake, root) = scheduler("ledger", &[("m", "")]);
        scheduler.ensure("m").await.unwrap();
        scheduler.evict("m", "test").unwrap();
        let reloaded = BudgetLedger::load(root.join(BUDGET_FILE)).unwrap();
        let kinds: Vec<String> = reloaded.entries().iter().map(|e| e.kind.clone()).collect();
        assert!(kinds.contains(&"admit".into()));
        assert!(kinds.contains(&"spawn".into()));
        assert!(kinds.contains(&"evict".into()));
    }

    #[tokio::test]
    async fn adoption_scan_adopts_matching_and_cleans_dead_markers() {
        let (scheduler, _fake, root) = scheduler("adoption", &[("m", "")]);
        let entry = scheduler.catalog().entry("m").unwrap();
        let dir = root.join(ADOPTION_DIR);
        std::fs::create_dir_all(&dir).unwrap();
        let gguf = entry.canonical_gguf().unwrap();
        // A live child whose cmdline names the GGUF (the marker's identity
        // check). `#` makes the path a shell comment so sleep still runs.
        // A live process whose argv names the GGUF (`tail -f` does not exec
        // away its own argv, unlike `sh -c 'sleep ...'`).
        let mut child = std::process::Command::new("tail")
            .arg("-f")
            .arg(&gguf)
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
        // Spawn returns before exec; wait for the argv the adoption identity
        // check reads (a real daemon restart is always long past this point).
        for _ in 0..200 {
            if process_cmdline(child.id()).contains(&gguf.display().to_string()) {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let marker = AdoptionMarker {
            model: "m".into(),
            pid: child.id(),
            gguf: gguf.display().to_string(),
            start_time: process_start_time(child.id()).unwrap_or(0),
            spec_hash: crate::swap_catalog::launch_spec_hash(&entry),
            scope_name: None,
            endpoint: "http://127.0.0.1:5801".into(),
        };
        std::fs::write(
            dir.join("m.json"),
            serde_json::to_string(&marker).unwrap(),
        )
        .unwrap();
        let mut dead = marker.clone();
        dead.model = "dead".into();
        dead.pid = 999_999;
        std::fs::write(dir.join("dead.json"), serde_json::to_string(&dead).unwrap()).unwrap();
        let report = scheduler.scan_adoption();
        assert_eq!(report.adopted, vec!["m".to_string()]);
        assert!(report.cleaned.contains(&"dead".to_string()));
        assert!(!report.spec_stale.contains(&"m".to_string()));
        let status = scheduler.status();
        let m = status.models.iter().find(|s| s.id == "m").unwrap();
        assert_eq!(m.state, "ready");
        unsafe {
            libc::kill(child.id() as i32, libc::SIGKILL);
        }
        let _ = child.wait();
    }

    #[tokio::test]
    async fn adoption_spec_hash_mismatch_marks_spec_stale_without_killing() {
        let (scheduler, _fake, root) = scheduler("adoption-stale", &[("m", "")]);
        let entry = scheduler.catalog().entry("m").unwrap();
        let dir = root.join(ADOPTION_DIR);
        std::fs::create_dir_all(&dir).unwrap();
        let gguf = entry.canonical_gguf().unwrap();
        // A live process whose argv names the GGUF (`tail -f` does not exec
        // away its own argv, unlike `sh -c 'sleep ...'`).
        let mut child = std::process::Command::new("tail")
            .arg("-f")
            .arg(&gguf)
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
        // Spawn returns before exec; wait for the argv the adoption identity
        // check reads (a real daemon restart is always long past this point).
        for _ in 0..200 {
            if process_cmdline(child.id()).contains(&gguf.display().to_string()) {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let marker = AdoptionMarker {
            model: "m".into(),
            pid: child.id(),
            gguf: gguf.display().to_string(),
            start_time: process_start_time(child.id()).unwrap_or(0),
            spec_hash: "outdated".into(),
            scope_name: None,
            endpoint: "http://127.0.0.1:5801".into(),
        };
        std::fs::write(dir.join("m.json"), serde_json::to_string(&marker).unwrap()).unwrap();
        let report = scheduler.scan_adoption();
        assert_eq!(report.spec_stale, vec!["m".to_string()]);
        let status = scheduler.status();
        assert!(status.models[0].spec_stale);
        assert_eq!(status.models[0].state, "ready");
        unsafe {
            libc::kill(child.id() as i32, libc::SIGKILL);
        }
        let _ = child.wait();
    }

    #[test]
    fn child_held_flock_releases_only_when_the_child_exits() {
        let root = test_root("flock");
        let gguf = root.join("model.gguf");
        std::fs::write(&gguf, b"gguf").unwrap();
        let lock = GgufLock::try_acquire(&gguf).unwrap().unwrap();
        let mut command = std::process::Command::new("sleep");
        command.arg("5");
        lock.inherit_fd_in(&mut command);
        let child = command.spawn().unwrap();
        let pid = child.id();
        drop(child);
        drop(lock);
        // Parent's fd is closed; the child still holds the lock.
        assert!(GgufLock::try_acquire(&gguf).unwrap().is_none());
        unsafe {
            libc::kill(pid as i32, libc::SIGKILL);
        }
        for _ in 0..100 {
            if GgufLock::try_acquire(&gguf).unwrap().is_some() {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("the lock was not released after the child died");
    }

    #[tokio::test]
    async fn second_engine_for_the_same_gguf_attaches_or_is_lock_refused() {
        let (scheduler, fake, _root) = scheduler("flock-refuse", &[("m", "")]);
        // First child holds the lock via the fake adapter.
        scheduler.ensure("m").await.unwrap();
        let entry_a = scheduler.catalog().entry("m").unwrap();
        // (a) A second catalog id pointing at the same canonical GGUF is a
        // RESIDENCY ATTACH (single residency) — no second process.
        let mut clone = entry_a.clone();
        clone.id = "m2".into();
        clone.aliases = vec![];
        clone.provenance.source = "cli".into();
        clone.launch_spec.served_model_alias = "m2".into();
        scheduler.catalog().upsert_entry(clone).unwrap();
        let outcome = scheduler.ensure("m2").await.unwrap();
        assert!(!outcome.spawned(), "same GGUF = attach, not a second residency");
        assert!(outcome.endpoint().starts_with("http://127.0.0.1:"));
        assert_eq!(fake.spawn_count(), 1);
        // (b) A direct adapter spawn for the same GGUF is refused by the
        // child-held flock.
        let request = SpawnRequest {
            model_id: "m3".into(),
            spec: LaunchSpec {
                program: "sleep".into(),
                ..LaunchSpec::default()
            },
            gguf: entry_a.canonical_gguf(),
            port: 12549,
            endpoint: "http://127.0.0.1:12549".into(),
            log_path: _root.join("m3.log"),
            scope_name: None,
            env: BTreeMap::new(),
        };
        let error = fake.spawn(&request).unwrap_err();
        assert!(error.to_string().contains("locked"), "{error}");
        assert_eq!(fake.spawn_count(), 1);
    }

    #[tokio::test]
    async fn fake_child_is_adopted_after_a_simulated_restart() {
        let (scheduler, fake, root) =
            scheduler_with("adopt-restart", &[("m", "")], None, false);
        let first = scheduler.ensure("m").await.unwrap();
        // Keep the adapter (and therefore the child) alive across the
        // simulated daemon death; the scheduler itself must not kill it.
        std::mem::forget(Arc::clone(&fake));
        drop(scheduler);

        let catalog = Arc::new(SwapCatalog::load(root.join("swap_catalog.json")).unwrap());
        let ledger = Arc::new(BudgetLedger::load(root.join(BUDGET_FILE)).unwrap());
        let mut adapters: BTreeMap<String, Arc<dyn EngineAdapter>> = BTreeMap::new();
        adapters.insert("fake".into(), Arc::new(FakeAdapter::new()));
        adapters.insert("external".into(), Arc::new(ExternalAdapter));
        let restarted = Arc::new(
            SwapScheduler::new(
                catalog,
                ledger,
                SwapSchedulerConfig {
                    wait_timeout: Duration::from_secs(5),
                    queue_cap: 2,
                    stop_grace: Duration::from_millis(200),
                    max_restarts: 1,
                    port_min: 12600,
                    port_max: 12650,
                    llama_swap_url: "http://127.0.0.1:1".into(),
                    settle: Duration::ZERO,
                    ..SwapSchedulerConfig::default()
                },
                adapters,
            )
            .with_state_root(root.clone())
            .with_kill_children_on_drop(true),
        );
        let report = restarted.scan_adoption();
        assert!(report.adopted.contains(&"m".to_string()), "{report:?}");
        let status = restarted.status();
        let model = status.models.iter().find(|m| m.id == "m").unwrap();
        assert_eq!(model.state, "ready");
        assert_eq!(model.endpoint.as_deref(), Some(first.endpoint()));
    }

    #[tokio::test]
    async fn routed_queue_cancel_aborts_but_direct_ensure_still_completes() {
        let (scheduler, _fake, _root) = scheduler("cancel-routed", &[("m", "")]);
        let mut entry = scheduler.catalog().entry("m").unwrap();
        entry
            .launch_spec
            .env
            .insert("FAKE_LOAD_MS".into(), "600".into());
        scheduler.catalog().upsert_entry(entry).unwrap();

        // Routed request: in-flight row attached.
        let registry = Arc::new(InflightRegistry::new());
        let handle = Arc::new(registry.register(
            "POST /v1/chat/completions".into(),
            "m".into(),
            "ua".into(),
            "127.0.0.1:1".into(),
            String::new(),
            String::new(),
            0,
            None,
        ));
        handle.set_catalog_id("m");
        let routed = {
            let scheduler = Arc::clone(&scheduler);
            let handle = Arc::clone(&handle);
            tokio::spawn(async move {
                scheduler.ensure_with_cancel("m", Some(&handle)).await
            })
        };
        let queue_id = poll_queue_id(&scheduler, "m").await;
        assert!(scheduler.cancel(queue_id), "queue row cancelled");
        let error = routed.await.unwrap().unwrap_err();
        assert!(error.to_string().contains("cancelled"), "{error}");
        drop(handle);

        // The queue-row cancellation was consumed; a DIRECT ensure (no
        // in-flight row) owns its request and completes normally.
        let outcome = scheduler.ensure("m").await.unwrap();
        assert!(outcome.spawned());
    }

    async fn poll_queue_id(scheduler: &Arc<SwapScheduler>, model: &str) -> u64 {
        for _ in 0..500 {
            if let Some(row) = scheduler
                .status()
                .queue
                .into_iter()
                .find(|q| q.model == model)
            {
                return row.id;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("no queue row for {model}");
    }

    #[tokio::test]
    async fn stop_refuses_inflight_and_pinned_models_at_the_last_moment() {
        let (scheduler, _fake, _root) = scheduler("protect", &[("m", "")]);
        scheduler.ensure("m").await.unwrap();
        let registry = Arc::new(InflightRegistry::new());
        scheduler.attach_inflight(Arc::clone(&registry));
        let handle = registry.register(
            "POST /v1/chat/completions".into(),
            "m".into(),
            "ua".into(),
            "127.0.0.1:1".into(),
            String::new(),
            String::new(),
            0,
            None,
        );
        handle.set_catalog_id("m");
        let error = scheduler.evict("m", "test").unwrap_err();
        assert!(error.to_string().contains("stream"), "{error}");
        assert_eq!(scheduler.status().models[0].state, "ready");
        drop(handle);
        scheduler.catalog().pin("m", 5.0).unwrap();
        let error = scheduler.evict("m", "test").unwrap_err();
        assert!(error.to_string().contains("pin"), "{error}");
        scheduler.catalog().unpin("m").unwrap();
        assert!(scheduler.evict("m", "test").unwrap());
    }

    #[tokio::test]
    async fn adoption_refuses_unknown_start_time_and_full_arg_mismatch() {
        let (scheduler, _fake, root) = scheduler("adoption-identity", &[("m", "")]);
        let entry = scheduler.catalog().entry("m").unwrap();
        let gguf = entry.canonical_gguf().unwrap();
        let dir = root.join(ADOPTION_DIR);
        std::fs::create_dir_all(&dir).unwrap();
        let mut child = std::process::Command::new("tail")
            .arg("-f")
            .arg(&gguf)
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
        for _ in 0..200 {
            if process_args(child.id()).iter().any(|a| a == &gguf.display().to_string()) {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        // (a) live pid, unknown start_time → refuse, keep the marker.
        let mut marker = AdoptionMarker {
            model: "m".into(),
            pid: child.id(),
            gguf: gguf.display().to_string(),
            start_time: 0,
            spec_hash: crate::swap_catalog::launch_spec_hash(&entry),
            scope_name: None,
            endpoint: "http://127.0.0.1:5801".into(),
        };
        std::fs::write(dir.join("m.json"), serde_json::to_string(&marker).unwrap()).unwrap();
        let report = scheduler.scan_adoption();
        assert!(report.adopted.is_empty(), "{report:?}");
        assert!(report.warnings.iter().any(|w| w.contains("no start_time")), "{report:?}");
        assert!(dir.join("m.json").exists(), "unproven marker is kept");
        // (b) known start_time, but the GGUF is not a FULL argv element
        // (`m.gguf.bak` must not match `m.gguf`).
        marker.start_time = process_start_time(child.id()).unwrap();
        marker.gguf = format!("{}.bak", gguf.display());
        std::fs::write(dir.join("m.json"), serde_json::to_string(&marker).unwrap()).unwrap();
        let report = scheduler.scan_adoption();
        assert!(report.adopted.is_empty(), "suffix must not match: {report:?}");
        assert!(report.cleaned.contains(&"m".to_string()), "{report:?}");
        unsafe {
            libc::kill(child.id() as i32, libc::SIGKILL);
        }
        let _ = child.wait();
    }

    #[tokio::test]
    async fn failed_stop_keeps_marker_and_restores_the_live_row() {
        let (scheduler, fake, _root) = scheduler("stop-failure", &[("m", "")]);
        scheduler.ensure("m").await.unwrap();
        assert!(scheduler
            .adoption_dir_for_test()
            .join("m.json")
            .exists());
        fake.set_fail_stop(true);
        let error = scheduler.evict("m", "test").unwrap_err();
        assert!(error.to_string().contains("stop `m` failed"), "{error}");
        assert!(
            scheduler.adoption_dir_for_test().join("m.json").exists(),
            "a failed stop must never delete the ownership marker"
        );
        let model = scheduler.status().models.into_iter().next().unwrap();
        assert_eq!(model.state, "ready", "the live row is restored: {model:?}");
        assert!(model.resident, "still resident: {model:?}");
        fake.set_fail_stop(false);
        assert!(scheduler.evict("m", "test").unwrap());
        assert!(!scheduler.adoption_dir_for_test().join("m.json").exists());
    }

    #[test]
    fn ledger_rotation_keeps_the_newest_rows() {
        let mut entries: Vec<BudgetEntry> = (0..3)
            .map(|i| BudgetEntry {
                ts: format!("{i}"),
                kind: "admit".into(),
                model: format!("m{i}"),
                gib: 1.0,
                reason: String::new(),
                actor: None,
            })
            .collect();
        for i in 3..6 {
            entries.push(BudgetEntry {
                ts: format!("{i}"),
                kind: "admit".into(),
                model: format!("m{i}"),
                gib: 1.0,
                reason: String::new(),
                actor: None,
            });
            push_bounded(&mut entries, 3);
        }
        assert_eq!(entries.len(), 3);
        assert_eq!(
            entries.iter().map(|e| e.ts.as_str()).collect::<Vec<_>>(),
            vec!["3", "4", "5"]
        );
    }

    #[test]
    fn fingerprint_staleness_answers_spec_stale() {
        let root = test_root("fingerprint");
        let gguf = root.join("m.gguf");
        std::fs::write(&gguf, b"gguf").unwrap();
        let mut entry = CatalogEntry {
            id: "m".into(),
            engine: "fake".into(),
            files: vec![gguf.display().to_string()],
            ..CatalogEntry::default()
        };
        entry.fingerprints.push(FileFingerprint {
            path: gguf.display().to_string(),
            mtime: 1,
            size: 1,
        });
        assert!(entry.fingerprint_stale().is_some());
    }

    #[test]
    fn budget_report_counts_each_model_once_and_sizes_pins() {
        let (scheduler, _fake, _root) = scheduler_cap("budget-report", &[("m", "")], Some(100.0));
        scheduler.catalog().pin("m", 40.0).unwrap();
        scheduler.set_shim_view(ResidencyView {
            residents: vec![
                Resident {
                    catalog_id: "m".into(),
                    source: ResidentSource::Shim,
                    endpoint: "http://127.0.0.1:1".into(),
                    gguf_path: None,
                    started_at: 0,
                    last_stream_completed: 0,
                    state: "ready".into(),
                    spec_stale: false,
                    engine: None,
                },
                Resident {
                    catalog_id: "m".into(),
                    source: ResidentSource::Native,
                    endpoint: "http://127.0.0.1:2".into(),
                    gguf_path: None,
                    started_at: 0,
                    last_stream_completed: 0,
                    state: "ready".into(),
                    spec_stale: false,
                    engine: None,
                },
            ],
            shim_error: None,
            observed_at: String::new(),
        });
        let report = scheduler.budget_report();
        // One line for `m`, max(estimate 10, pin 40) = 40 — not doubled.
        assert_eq!(report.residents.len(), 1);
        assert_eq!(report.used_gib, 40.0);
        assert_eq!(report.headroom_gib, 60.0);
    }
}
