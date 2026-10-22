//! Plugin core (v2 L2, design `docs/design/v2-streamlining.L2.md`).
//!
//! Sidecar plugins: separate executables supervised by the daemon and spoken
//! to over a per-plugin Unix domain socket using newline-delimited JSON-RPC
//! 2.0. The RPC v1 surface is minimal and additive-only; plugins get NO
//! system ops.
//!
//! Layout:
//! - [`manifest`] — `plugin.toml` parsing.
//! - [`registry`] — installed-plugin state (survives restarts; terminal
//!   `failed` is sticky until explicit `enable`).
//! - [`supervisor`] — spawn/backoff/crash-loop/stop semantics.
//! - [`rpc`] — the UDS server, `initialize` handshake, grant gating, the
//!   method surface, size/in-flight caps.

pub mod manifest;
pub mod registry;
pub mod rpc;
pub mod store;
pub mod supervisor;

use anyhow::{bail, Context, Result};
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::net::UnixListener;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::credentials::CredentialStore;
use manifest::PluginManifest;
use registry::{PluginEntry, PluginState, Registry};

/// JSON-RPC framing: one object per line, `\n` terminated (Hank ruling).
pub const RPC_DELIM: u8 = b'\n';
/// Body cap for every RPC request line (design: "body cap 1 MiB").
pub const MAX_BODY_BYTES: usize = 1024 * 1024;
/// Per-plugin in-flight RPC working requests (design: "in-flight cap 4").
pub const INFLIGHT_CAP: usize = 4;
/// Per-plugin queue beyond the in-flight cap (design: "queue 16").
pub const INFLIGHT_QUEUE: usize = 16;
/// The RPC protocol version returned by `initialize`.
pub const PROTOCOL_VERSION: u32 = 1;

/// Standard JSON-RPC error codes used by this surface.
pub const JSONRPC_PARSE_ERROR: i32 = -32700;
pub const JSONRPC_INVALID_REQUEST: i32 = -32600;
pub const JSONRPC_METHOD_NOT_FOUND: i32 = -32601;
pub const JSONRPC_INTERNAL_ERROR: i32 = -32603;
/// Server busy (design: "per-plugin in-flight 4 + queue 16 → JSON-RPC limit
/// error"; Hank ruling #7: −32005 naming the cap).
pub const JSONRPC_LIMIT: i32 = -32005;
/// Not authenticated (initialize not performed).
pub const JSONRPC_NOT_AUTHED: i32 = -32002;

/// The live `AppState` handle. Daemon wiring: `PluginManager::new` runs
/// before `AppState` is built; `attach` wires it in right after.
pub type AppHandle = OnceLock<Arc<crate::server::AppState>>;

/// Supervisor mailbox commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupCmd {
    /// Spawn + monitor (no-op if a child is already running).
    Start,
    /// Stop the child (SIGTERM → grace → kill); the supervisor task exits.
    Stop,
    /// Daemon shutdown: stop the child, the task exits.
    Shutdown,
}

/// Per-plugin shared runtime state (caps live across connections).
pub struct PluginRuntime {
    /// Active chat completions (`chat.completions` is the only long-running
    /// method; every RPC request also counts against the total cap).
    pub chat_active: tokio::sync::Semaphore,
    /// Total active+queued requests: `INFLIGHT_CAP + INFLIGHT_QUEUE`.
    pub chat_total: tokio::sync::Semaphore,
}

impl Default for PluginRuntime {
    fn default() -> Self {
        Self {
            chat_active: tokio::sync::Semaphore::new(INFLIGHT_CAP),
            chat_total: tokio::sync::Semaphore::new(INFLIGHT_CAP + INFLIGHT_QUEUE),
        }
    }
}

impl PluginRuntime {
    pub fn new() -> Self {
        Self::default()
    }
}

struct Inner {
    state_root: PathBuf,
    runtime_root: PathBuf,
    registry: Mutex<Registry>,
    creds: Arc<CredentialStore>,
    app: Arc<AppHandle>,
    supervisors: Mutex<HashMap<String, mpsc::UnboundedSender<SupCmd>>>,
    /// Live RPC sessions per plugin, for daemon→plugin notifications.
    sessions: Mutex<HashMap<String, Vec<mpsc::UnboundedSender<String>>>>,
    runtimes: Mutex<HashMap<String, Arc<PluginRuntime>>>,
    listeners: Mutex<HashMap<String, Arc<UnixListener>>>,
    toasts: Mutex<VecDeque<Value>>,
    shutting_down: AtomicBool,
}

/// The daemon-side plugin subsystem.
#[derive(Clone)]
pub struct PluginManager {
    inner: Arc<Inner>,
}

impl PluginManager {
    /// Load the registry and prepare for `attach` + `start_at_boot`.
    pub fn new(creds: Arc<CredentialStore>) -> Result<Self> {
        Self::new_at(
            creds,
            crate::paths::brainrouter_state(),
            crate::paths::brainrouter_runtime(),
        )
    }

    /// Root-injected constructor (tests; also the shape for multi-instance
    /// development runs).
    pub fn new_at(
        creds: Arc<CredentialStore>,
        state_root: PathBuf,
        runtime_root: PathBuf,
    ) -> Result<Self> {
        // I3: corrupt registry → quarantine + boot empty, never brick.
        let (registry, _) = Registry::load_quarantining(&state_root);
        Ok(Self {
            inner: Arc::new(Inner {
                state_root,
                runtime_root,
                registry: Mutex::new(registry),
                creds,
                app: Arc::new(OnceLock::new()),
                supervisors: Mutex::new(HashMap::new()),
                sessions: Mutex::new(HashMap::new()),
                runtimes: Mutex::new(HashMap::new()),
                listeners: Mutex::new(HashMap::new()),
                toasts: Mutex::new(VecDeque::new()),
                shutting_down: AtomicBool::new(false),
            }),
        })
    }

    /// Re-handle from the shared inner (used by the RPC dispatcher).
    pub(crate) fn from_inner(inner: Arc<Inner>) -> Self {
        Self { inner }
    }

    pub fn app(&self) -> &Arc<AppHandle> {
        &self.inner.app
    }

    pub fn creds(&self) -> &Arc<CredentialStore> {
        &self.inner.creds
    }

    pub fn state_root(&self) -> &Path {
        &self.inner.state_root
    }

    pub fn runtime_root(&self) -> &Path {
        &self.inner.runtime_root
    }

    pub fn runtime(&self, id: &str) -> Arc<PluginRuntime> {
        let mut map = self.inner.runtimes.lock().unwrap();
        map.entry(id.to_string())
            .or_insert_with(|| Arc::new(PluginRuntime::new()))
            .clone()
    }

    /// Wire the live AppState (called by the daemon right after it is built).
    pub fn attach(&self, state: Arc<crate::server::AppState>) {
        let _ = self.inner.app.set(state);
    }

    /// Bind sockets + spawn supervisors for installed plugins at daemon boot.
    pub async fn start_at_boot(&self) {
        let ids: Vec<(String, bool, PluginState)> = {
            let reg = self.inner.registry.lock().unwrap();
            reg.plugins
                .values()
                .map(|e| (e.id.clone(), e.enabled, e.state))
                .collect()
        };
        if ids.is_empty() {
            return;
        }
        if let Err(e) = store::ensure_plugins_runtime_dir(&self.inner.runtime_root) {
            warn!(error = %e, "plugin runtime directory unavailable; plugins cannot connect");
            return;
        }
        for (id, enabled, state) in ids {
            if let Err(e) = self.bind_plugin_socket(&id) {
                warn!(plugin = %id, error = %e, "failed to bind plugin socket");
                continue;
            }
            if enabled && state != PluginState::Failed {
                // Stale `Running` from a previous boot just means: spawn again.
                self.start_supervisor(&id);
            }
        }
    }

    /// Graceful daemon shutdown: stop every supervised child (SIGTERM with
    /// grace, then kill), then remove the plugin sockets.
    pub async fn shutdown(&self) {
        self.inner
            .shutting_down
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let senders: Vec<(String, mpsc::UnboundedSender<SupCmd>)> = {
            let mut sup = self.inner.supervisors.lock().unwrap();
            std::mem::take(&mut *sup).into_iter().collect()
        };
        for (id, tx) in senders {
            let _ = tx.send(SupCmd::Shutdown);
            // The task exits once the child is reaped. Bounded so a wedged
            // child can never stall daemon shutdown past the SIGKILL grace.
            info!(plugin = %id, "plugin supervisor stop requested");
        }
        // Give supervisors time to SIGTERM→SIGKILL their children (grace is
        // 5 s per child, all in parallel).
        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(8);
        loop {
            let n = self.inner.supervisors.lock().unwrap().len();
            if n == 0 || tokio::time::Instant::now() >= deadline {
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
        // Remove sockets we own.
        let listeners: HashMap<String, Arc<UnixListener>> = {
            let mut l = self.inner.listeners.lock().unwrap();
            std::mem::take(&mut *l)
        };
        for (id, _) in listeners {
            let sock = store::plugin_socket_path(&self.inner.runtime_root, &id);
            let _ = std::fs::remove_file(&sock);
        }
    }

    pub fn is_shutting_down(&self) -> bool {
        self.inner.shutting_down.load(std::sync::atomic::Ordering::Relaxed)
    }

    // ── Registry queries ──────────────────────────────────────────────────

    pub fn list(&self) -> Result<Vec<Value>> {
        let reg = self.inner.registry.lock().unwrap();
        Ok(reg
            .plugins
            .values()
            .map(|e| {
                let mut v = e.masked_json();
                // The dashboard tab and `plugin config show` read the
                // plugin-namespaced config file (never a credential value).
                if let Some(obj) = v.as_object_mut() {
                    obj.insert(
                        "config".to_string(),
                        self.config_get(&e.id, "*").unwrap_or_else(|_| json!({})),
                    );
                }
                v
            })
            .collect())
    }

    pub fn entry_json(&self, id: &str) -> Result<Value> {
        check_plugin_id(id)?;
        let reg = self.inner.registry.lock().unwrap();
        reg.plugins
            .get(id)
            .map(|e| e.masked_json())
            .with_context(|| format!("plugin '{id}' is not installed"))
    }

    pub fn set_status_line(&self, id: &str, text: &str) -> Result<()> {
        check_plugin_id(id)?;
        let mut reg = self.inner.registry.lock().unwrap();
        let entry = reg
            .plugins
            .get_mut(id)
            .with_context(|| format!("plugin '{id}' is not installed"))?;
        entry.status_line = text.trim().chars().take(2000).collect();
        reg.save(&self.inner.state_root)
    }

    pub fn record_toast(&self, toast: Value) {
        let mut t = self.inner.toasts.lock().unwrap();
        t.push_back(toast);
        while t.len() > 20 {
            t.pop_front();
        }
    }

    pub fn toasts(&self) -> Vec<Value> {
        self.inner.toasts.lock().unwrap().iter().rev().cloned().collect()
    }

    /// All plugin entries as JSON for the dashboard (`/api/plugins`).
    pub fn dashboard_json(&self) -> Value {
        let plugins: Vec<Value> = self.list().unwrap_or_default();
        json!({ "plugins": plugins, "toasts": self.toasts() })
    }

    // ── Install / remove ──────────────────────────────────────────────────

    /// Install a plugin from a directory containing `plugin.toml` + binary.
    /// Copies everything into the plugin state dir; mints a token on first
    /// install; a reinstall preserves the token and all plugin state.
    pub fn install(&self, dir: &Path) -> Result<Value> {
        let manifest = PluginManifest::load(dir)?;
        let id = manifest.id.clone();
        let state_dir = store::ensure_plugin_state_dir(&self.inner.state_root, &id)?;
        let bin_src = dir.join(&manifest.binary);
        if !bin_src.is_file() {
            bail!(
                "plugin manifest names binary {:?} but {} has no such file",
                manifest.binary,
                dir.display()
            );
        }
        // Reinstall-safe: copy manifest + every regular file from the source
        // dir (token/config/kv/logs live in the STATE dir and are untouched).
        copy_tree(dir, &state_dir)?;
        let bin_path = state_dir.join(&manifest.binary);
        std::fs::set_permissions(&bin_path, std::fs::Permissions::from_mode(0o755))
            .context("chmod 0755 on plugin binary")?;

        let token_path = state_dir.join("token");
        let minted = if token_path.exists() {
            false
        } else {
            store::atomic_write_0600(&token_path, format!("{}\n", store::mint_token()).as_bytes())?;
            true
        };

        let mut reg = self.inner.registry.lock().unwrap();
        let entry = reg
            .plugins
            .entry(id.clone())
            .or_insert_with(|| PluginEntry::from_manifest(&manifest));
        // Reinstall refreshes the manifest fields, preserves token/state/
        // enabled/crash counters.
        entry.name = manifest.name.clone();
        entry.version = manifest.version.clone();
        entry.binary = manifest.binary.clone();
        entry.min_daemon_version = manifest.min_daemon_version.clone();
        entry.grants = manifest.grants.clone();
        entry.config_schema = manifest.config_schema.clone();
        reg.save(&self.inner.state_root)?;
        drop(reg);

        // Bind the socket immediately so `plugin enable` + the plugin's own
        // connect both work without a daemon restart.
        if let Err(e) = self.bind_plugin_socket(&id) {
            warn!(plugin = %id, error = %e, "installed, but plugin socket bind failed");
        }
        info!(plugin = %id, version = %entry_version(&self.inner, &id), minted_token = minted,
            "plugin installed");
        Ok(json!({
            "ok": true,
            "id": id,
            "installed_to": state_dir.display().to_string(),
            "token_minted": minted,
        }))
    }

    /// Remove a plugin: stop it, delete its state dir, drop credential grants
    /// naming it, remove the registry entry.
    pub async fn remove(&self, id: &str) -> Result<Value> {
        check_plugin_id(id)?;
        self.stop_supervisor(id).await;
        self.close_listener(id);
        let state_dir = store::plugin_state_dir(&self.inner.state_root, id);
        if state_dir.exists() {
            std::fs::remove_dir_all(&state_dir)
                .with_context(|| format!("removing plugin state dir {}", state_dir.display()))?;
        }
        let dropped = self.inner.creds.drop_grants_for_plugin(id)?;
        if !dropped.is_empty() {
            info!(plugin = %id, credentials = ?dropped,
                "dropped credential grants naming the removed plugin");
        }
        let mut reg = self.inner.registry.lock().unwrap();
        let removed = reg.plugins.remove(id).is_some();
        reg.save(&self.inner.state_root)?;
        drop(reg);
        if !removed {
            bail!("plugin '{id}' is not installed");
        }
        self.inner.sessions.lock().unwrap().remove(id);
        self.inner.runtimes.lock().unwrap().remove(id);
        info!(plugin = %id, "plugin removed");
        Ok(json!({ "ok": true, "id": id, "dropped_grants_from": dropped }))
    }

    // ── Enable / disable ──────────────────────────────────────────────────

    pub async fn enable(&self, id: &str) -> Result<Value> {
        check_plugin_id(id)?;
        let mut reg = self.inner.registry.lock().unwrap();
        let entry = reg
            .plugins
            .get_mut(id)
            .with_context(|| format!("plugin '{id}' is not installed"))?;
        let was_failed = entry.state == PluginState::Failed;
        entry.enabled = true;
        if was_failed {
            // Explicit enable is the documented recovery from terminal failed.
            entry.state = PluginState::Stopped;
            entry.consecutive_crashes = 0;
            entry.last_exit = None;
        }
        reg.save(&self.inner.state_root)?;
        drop(reg);
        if self.bind_plugin_socket(id).is_err() {
            warn!(plugin = %id, "plugin socket already bound");
        }
        self.start_supervisor(id);
        info!(plugin = %id, reset_from_failed = was_failed, "plugin enabled");
        Ok(json!({ "ok": true, "id": id, "enabled": true, "reset_from_failed": was_failed }))
    }

    pub async fn disable(&self, id: &str) -> Result<Value> {
        check_plugin_id(id)?;
        {
            let mut reg = self.inner.registry.lock().unwrap();
            let entry = reg
                .plugins
                .get_mut(id)
                .with_context(|| format!("plugin '{id}' is not installed"))?;
            entry.enabled = false;
            reg.save(&self.inner.state_root)?;
        }
        self.stop_supervisor(id).await;
        {
            let mut reg = self.inner.registry.lock().unwrap();
            if let Some(entry) = reg.plugins.get_mut(id) {
                entry.state = PluginState::Stopped;
                entry.consecutive_crashes = 0;
                entry.pid = None;
                reg.save(&self.inner.state_root)?;
            }
        }
        info!(plugin = %id, "plugin disabled");
        Ok(json!({ "ok": true, "id": id, "enabled": false }))
    }

    /// I1: daemon-less `plugin enable/disable` persists the enabled flag
    /// ONLY. The supervisor is the daemon's job; spawning from the CLI would
    /// create an unsupervised orphan (the daemon spawns a second copy at the
    /// next boot — double transports).
    pub fn set_enabled_registry_only(&self, id: &str, enabled: bool) -> Result<Value> {
        check_plugin_id(id)?;
        let mut reg = self.inner.registry.lock().unwrap();
        let entry = reg
            .plugins
            .get_mut(id)
            .with_context(|| format!("plugin '{id}' is not installed"))?;
        entry.enabled = enabled;
        if enabled && entry.state == registry::PluginState::Failed {
            entry.state = registry::PluginState::Stopped;
            entry.consecutive_crashes = 0;
            entry.last_exit = None;
        }
        if !enabled {
            entry.state = registry::PluginState::Stopped;
            entry.consecutive_crashes = 0;
            entry.pid = None;
        }
        reg.save(&self.inner.state_root)?;
        info!(plugin = %id, enabled, "plugin enablement persisted (spawn deferred to daemon boot)");
        Ok(json!({ "ok": true, "id": id, "enabled": enabled, "spawn_deferred": true }))
    }

    pub async fn set_enabled(&self, id: &str, enabled: bool) -> Result<Value> {
        if enabled {
            self.enable(id).await
        } else {
            self.disable(id).await
        }
    }

    // ── Plugin config / kv (shared by RPC and the dashboard/CLI) ─────────

    /// Log path for the dashboard/CLI; id-validated (B2: the HTTP layer
    /// joins the id straight into a read path).
    pub fn log_path_checked(&self, id: &str) -> Result<PathBuf> {
        check_plugin_id(id)?;
        Ok(self.log_path(id))
    }

    pub fn config_path(&self, id: &str) -> PathBuf {
        store::plugin_state_dir(&self.inner.state_root, id).join("config.json")
    }

    pub fn kv_path(&self, id: &str) -> PathBuf {
        store::plugin_state_dir(&self.inner.state_root, id).join("kv.json")
    }

    pub fn log_path(&self, id: &str) -> PathBuf {
        store::plugin_state_dir(&self.inner.state_root, id).join("plugin.log")
    }

    /// Read a (dotted) config key. `*` returns the whole object.
    pub fn config_get(&self, id: &str, key: &str) -> Result<Value> {
        check_plugin_id(id)?;
        let path = self.config_path(id);
        let obj = store::read_json(&path)?.unwrap_or_else(|| json!({}));
        if key == "*" {
            return Ok(obj);
        }
        Ok(dotted_get(&obj, key).cloned().unwrap_or(Value::Null))
    }

    pub fn config_get_full(&self, id: &str) -> Result<Value> {
        check_plugin_id(id)?;
        self.config_get(id, "*")
    }

    /// Set a (dotted) config key and push `config.changed` to live sessions.
    pub fn config_set(&self, id: &str, key: &str, value: Value) -> Result<Value> {
        check_plugin_id(id)?;
        if key == "*" || key.is_empty() {
            bail!("config key must name a field, not the whole object");
        }
        let path = self.config_path(id);
        let mut obj = store::read_json(&path)?.unwrap_or_else(|| json!({}));
        dotted_set(&mut obj, key, value.clone())?;
        store::write_json_0600(&path, &obj)?;
        self.push_config_changed(id, key);
        Ok(json!({ "ok": true, "key": key, "value": obj }))
    }

    pub fn kv_get(&self, id: &str, key: &str) -> Result<Value> {
        check_plugin_id(id)?;
        let path = self.kv_path(id);
        let obj = store::read_json(&path)?.unwrap_or_else(|| json!({}));
        Ok(obj.get(key).cloned().unwrap_or(Value::Null))
    }

    pub fn kv_set(&self, id: &str, key: &str, value: Value) -> Result<Value> {
        check_plugin_id(id)?;
        if key.is_empty() {
            bail!("kv key must not be empty");
        }
        let path = self.kv_path(id);
        let mut obj = store::read_json(&path)?.unwrap_or_else(|| json!({}));
        if let Some(map) = obj.as_object_mut() {
            map.insert(key.to_string(), value.clone());
        }
        store::write_json_0600(&path, &obj)?;
        Ok(json!({ "ok": true, "key": key, "value": value }))
    }

    /// Daemon→plugin `config.changed` push (design: the bridge reacts to it).
    pub fn push_config_changed(&self, id: &str, key: &str) {
        let msg = json!({
            "jsonrpc": "2.0",
            "method": "config.changed",
            "params": { "key": key },
        })
        .to_string()
            + "\n";
        let mut sessions = self.inner.sessions.lock().unwrap();
        if let Some(senders) = sessions.get_mut(id) {
            senders.retain(|tx| tx.send(msg.clone()).is_ok());
        }
    }

    pub(crate) fn register_session(
        &self,
        id: &str,
        tx: mpsc::UnboundedSender<String>,
    ) {
        self.inner
            .sessions
            .lock()
            .unwrap()
            .entry(id.to_string())
            .or_default()
            .push(tx);
    }

    // ── Sockets ───────────────────────────────────────────────────────────

    /// Bind the plugin's UDS server: dir 0700, socket 0600, stale socket
    /// unlinked (repo precedent `prepare_uds_path` in `src/server.rs`).
    pub fn bind_plugin_socket(&self, id: &str) -> Result<()> {
        check_plugin_id(id)?;
        let sock = store::plugin_socket_path(&self.inner.runtime_root, id);
        {
            let listeners = self.inner.listeners.lock().unwrap();
            if listeners.contains_key(id) {
                return Ok(()); // already bound by this daemon
            }
        }
        store::ensure_plugins_runtime_dir(&self.inner.runtime_root)?;
        prepare_plugin_socket_path(&sock)?;
        let listener = UnixListener::bind(&sock)
            .with_context(|| format!("binding plugin socket {}", sock.display()))?;
        std::fs::set_permissions(&sock, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("chmod 0600 on plugin socket {}", sock.display()))?;
        let listener = Arc::new(listener);
        self.inner
            .listeners
            .lock()
            .unwrap()
            .insert(id.to_string(), listener.clone());
        let inner = self.inner.clone();
        let id = id.to_string();
        tokio::spawn(async move {
            rpc::accept_loop(inner, id, listener).await;
        });
        Ok(())
    }

    fn close_listener(&self, id: &str) {
        if let Some(listener) = self.inner.listeners.lock().unwrap().remove(id) {
            drop(listener);
        }
        let sock = store::plugin_socket_path(&self.inner.runtime_root, id);
        let _ = std::fs::remove_file(&sock);
    }

    // ── Supervisors ───────────────────────────────────────────────────────

    fn start_supervisor(&self, id: &str) {
        // N4: a stop/enable race with daemon shutdown must not spawn a child
        // the daemon is about to stop (and whose socket is being unlinked).
        if self.is_shutting_down() {
            warn!(plugin = %id, "daemon is shutting down; not spawning plugin");
            return;
        }
        let mut sup = self.inner.supervisors.lock().unwrap();
        if let Some(tx) = sup.get(id) {
            let _ = tx.send(SupCmd::Start);
            return;
        }
        let (tx, rx) = mpsc::unbounded_channel();
        sup.insert(id.to_string(), tx);
        drop(sup);
        let inner = self.inner.clone();
        let id2 = id.to_string();
        tokio::spawn(async move {
            supervisor::supervisor_loop(inner, id2, rx).await;
        });
    }

    async fn stop_supervisor(&self, id: &str) {
        let tx = self.inner.supervisors.lock().unwrap().remove(id);
        if let Some(tx) = tx {
            let _ = tx.send(SupCmd::Stop);
            // N1: the supervisor removes itself from the map when the child
            // is reaped — poll for that (returns instantly when there was no
            // child), bounded by the SIGTERM grace + kill window.
            let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(12);
            loop {
                if !self.inner.supervisors.lock().unwrap().contains_key(id)
                    || tokio::time::Instant::now() >= deadline
                {
                    break;
                }
                tokio::time::sleep(tokio::time::Duration::from_millis(25)).await;
            }
        }
    }
}

fn entry_version(inner: &Inner, id: &str) -> String {
    let reg = inner.registry.lock().unwrap();
    reg.plugins.get(id).map(|e| e.version.clone()).unwrap_or_default()
}

/// `brainrouter --version` reports `clap`-style "name version"; the plugin
/// core compares the bare semver.
pub fn daemon_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Validate a plugin id used as a filesystem path component. Every API
/// surface that takes an `id` from the wire (HTTP, CLI, RPC) must run it
/// through this BEFORE touching the registry, the state dir, or the
/// runtime dir: ids name directories and sockets, so a traversal id
/// ("..", "a/b", absolute) must be a 4xx, never a `remove_dir_all` target.
pub fn check_plugin_id(id: &str) -> Result<()> {
    if id.is_empty() {
        bail!("plugin id must not be empty");
    }
    if id.len() > 64 {
        bail!("plugin id too long ({})", id.len());
    }
    if !id
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        bail!("plugin id {id:?} is invalid: use letters, digits, '-' and '_'");
    }
    Ok(())
}

/// I3: a corrupt registry must never brick the daemon. Rename the unreadable
/// file aside (`.corrupt-<UTC ts>`, kept for forensics) and boot with an
/// empty registry (precedent: the project-pins store).
impl Registry {
    pub fn load_quarantining(state_root: &Path) -> (Registry, Option<PathBuf>) {
        match Registry::load(state_root) {
            Ok(reg) => (reg, None),
            Err(e) => {
                let path = Registry::path(state_root);
                match quarantine_file(&path) {
                    Ok(moved) => {
                        warn!(path = %path.display(), quarantined = %moved.display(), error = %e,
                            "plugin registry was unreadable; quarantined and continuing with an empty registry");
                        (Registry::default(), Some(moved))
                    }
                    Err(qe) => {
                        warn!(path = %path.display(), error = %e, quarantine_error = %qe,
                            "plugin registry unreadable and could not be quarantined; booting with an empty registry");
                        (Registry::default(), None)
                    }
                }
            }
        }
    }
}

/// Rename `path` to `<path>.corrupt-<UTC ts>`; returns the new path.
pub fn quarantine_file(path: &Path) -> Result<PathBuf> {
    let ts = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let mut name = path.as_os_str().to_os_string();
    name.push(format!(".corrupt-{ts}"));
    let moved = PathBuf::from(name);
    std::fs::rename(path, &moved)
        .with_context(|| format!("renaming {} for quarantine", path.display()))?;
    Ok(moved)
}

/// Copy a plugin directory into the state dir, skipping dot/temp files.
fn copy_tree(from: &Path, to: &Path) -> Result<()> {
    let mut stack = vec![(from.to_path_buf(), to.to_path_buf())];
    while let Some((src_dir, dst_dir)) = stack.pop() {
        std::fs::create_dir_all(&dst_dir)?;
        for entry in std::fs::read_dir(&src_dir)
            .with_context(|| format!("reading {}", src_dir.display()))?
        {
            let entry = entry?;
            let name = entry.file_name();
            if name.to_string_lossy().starts_with('.') {
                continue;
            }
            let src = entry.path();
            let dst = dst_dir.join(&name);
            if src.is_dir() {
                stack.push((src, dst));
            } else {
                std::fs::copy(&src, &dst).with_context(|| {
                    format!("copying {} -> {}", src.display(), dst.display())
                })?;
            }
        }
    }
    Ok(())
}

pub(crate) fn dotted_get<'a>(root: &'a Value, key: &str) -> Option<&'a Value> {
    let mut cur = root;
    for part in key.split('.') {
        cur = cur.get(part)?;
    }
    Some(cur)
}

pub(crate) fn dotted_set(root: &mut Value, key: &str, value: Value) -> Result<()> {
    let parts: Vec<&str> = key.split('.').collect();
    let mut cur = root;
    for (i, part) in parts.iter().enumerate() {
        if cur.is_null() {
            *cur = json!({});
        }
        // N3: a mid-path scalar (config.json `{"a": 1}` then set "a.b") is a
        // caller error, reported as a JSON-RPC error — never a panic.
        let Some(map) = cur.as_object_mut() else {
            bail!(
                "config key {key:?} descends through non-object value at {part:?}"
            );
        };
        if i == parts.len() - 1 {
            map.insert((*part).to_string(), value);
            return Ok(());
        }
        cur = map.entry(*part).or_insert_with(|| json!({}));
    }
    Ok(())
}

/// Unlink a stale (unconnectable) socket; refuse to clobber a live one.
fn prepare_plugin_socket_path(path: &Path) -> Result<()> {
    let meta = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    if !meta.file_type().is_socket() {
        bail!("refusing to replace non-socket path at {}", path.display());
    }
    // A UDS connect on a dead socket returns ECONNREFUSED without blocking;
    // run it off the async executor (precedent: `prepare_uds_path` probes with
    // a timeout in `src/server.rs`).
    let probe = path.to_path_buf();
    let alive = std::thread::spawn(move || std::os::unix::net::UnixStream::connect(&probe).is_ok())
        .join()
        .map_err(|_| anyhow::anyhow!("plugin socket probe thread panicked"))?;
    if alive {
        bail!("a plugin listener is already live at {}", path.display());
    }
    info!(path = %path.display(), "removing stale plugin socket");
    std::fs::remove_file(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manager(dir: &Path) -> PluginManager {
        let creds = Arc::new(CredentialStore::load(dir.join("credentials.json")).unwrap());
        PluginManager::new_at(
            creds,
            dir.join("state"),
            dir.join("run"),
        )
        .unwrap()
    }

    #[test]
    fn dotted_paths() {
        let mut v = json!({ "a": { "b": 1 } });
        dotted_set(&mut v, "a.b", json!(2)).unwrap();
        dotted_set(&mut v, "c.d", json!("x")).unwrap();
        assert_eq!(dotted_get(&v, "a.b"), Some(&json!(2)));
        assert_eq!(dotted_get(&v, "c.d"), Some(&json!("x")));
        assert_eq!(dotted_get(&v, "zz"), None);
        // N3: descending through a scalar is an error, not a panic.
        let mut w = json!({ "a": 1 });
        assert!(dotted_set(&mut w, "a.b", json!(2)).is_err());
        assert_eq!(w, json!({ "a": 1 }));
    }

    #[test]
    fn plugin_id_validation_blocks_traversal() {
        assert!(check_plugin_id("bridge").is_ok());
        assert!(check_plugin_id("rayA").is_ok());
        assert!(check_plugin_id("a-b_c9").is_ok());
        for bad in ["", "..", ".", "a/b", "/etc/passwd", "../evil", "a\\b", "id;rm", &"x".repeat(65)] {
            assert!(check_plugin_id(bad).is_err(), "must reject {bad:?}");
        }
    }

    #[tokio::test]
    async fn remove_rejects_traversal_id_before_touching_disk() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path());
        // B2: an id-validating entry point must refuse BEFORE any state touch.
        let err = m.remove("..").await.unwrap_err().to_string();
        assert!(err.contains("invalid"), "{err}");
        // The state tree was never created (install never ran).
        assert!(!store::plugins_state_root(&dir.path().join("state")).join("bridge").exists());
        assert!(m.list().unwrap().is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn install_enable_config_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path());
        // Build a source dir.
        let src = dir.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(
            src.join("plugin.toml"),
            "id = \"demo\"\nname = \"Demo\"\nversion = \"0.1.0\"\nbinary = \"demo.bin\"\nmin_daemon_version = \"1.0.0\"\ngrants = [\"config.get\", \"config.set\"]\n",
        )
        .unwrap();
        std::fs::write(src.join("demo.bin"), b"#!/bin/sh\nexit 0\n").unwrap();

        m.install(&src).unwrap();
        let listed = m.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0]["id"], "demo");
        assert_eq!(listed[0]["enabled"], json!(false));

        // Token minted 0600.
        let token = store::plugin_state_dir(&m.state_root().to_path_buf(), "demo").join("token");
        assert_eq!(
            std::os::unix::fs::PermissionsExt::mode(
                &std::fs::metadata(&token).unwrap().permissions()
            ) & 0o777,
            0o600
        );
        // config set/get
        m.config_set("demo", "discord.enabled", json!(true)).unwrap();
        m.config_set("demo", "timeout_secs", json!(60)).unwrap();
        assert_eq!(m.config_get("demo", "discord.enabled").unwrap(), json!(true));
        assert_eq!(m.config_get_full("demo").unwrap()["timeout_secs"], json!(60));
        // kv set/get
        m.kv_set("demo", "k", json!("v")).unwrap();
        assert_eq!(m.kv_get("demo", "k").unwrap(), json!("v"));

        // Reinstall preserves token + config.
        let token_text = std::fs::read_to_string(&token).unwrap();
        m.install(&src).unwrap();
        assert_eq!(std::fs::read_to_string(&token).unwrap(), token_text);
        assert_eq!(m.config_get("demo", "timeout_secs").unwrap(), json!(60));

        // Remove cleans state + registry.
        m.remove("demo").await.unwrap();
        assert!(m.list().unwrap().is_empty());
        assert!(!store::plugin_state_dir(&m.state_root().to_path_buf(), "demo").exists());
    }

    #[test]
    fn manifest_grants_validated_at_install() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path());
        let src = dir.path().join("bad");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(
            src.join("plugin.toml"),
            "id = \"bad\"\nname = \"Bad\"\nversion = \"1\"\nbinary = \"b\"\ngrants = [\"system.restart\"]\n",
        )
        .unwrap();
        std::fs::write(src.join("b"), b"x").unwrap();
        assert!(m.install(&src).is_err());
    }
}
