//! Swap model registry + launch specs (v2 L4).
//!
//! `swap_catalog.json` under `$XDG_STATE_HOME/brainrouter/` is the supervisor's
//! model registry: one [`CatalogEntry`] per servable model with a fully
//! resolved launch spec, plus the sized **pins** (budget reservations) and the
//! **groups** metadata imported from llama-swap (L7 slot-sharing input).
//!
//! Seeds:
//! 1. [`import_llama_swap_config`] — the inverse of `llama_swap_register.rs`.
//!    It parses llama-swap's `config.yaml`, resolves the macro chain
//!    (top-level `macros:` shadowed by per-model `macros:`, recursively, so
//!    `${common}` → `${ctx}` → `-c 131072` lands as concrete args) while
//!    preserving runtime placeholders (`${PORT}`), and records groups as
//!    metadata. Import is an idempotent merge: re-running it against the same
//!    file changes nothing; `--dry-run` renders the plan without writing.
//! 2. The toolbox/server-mode catalog (future legs).
//! 3. CLI additions (`swap catalog add <spec.json>`).
//!
//! All writes are atomic owner-only files (the migration/credentials
//! convention). Readers take a snapshot; writers serialize on one mutex.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// State file name (under `$XDG_STATE_HOME/brainrouter`).
pub const CATALOG_FILE: &str = "swap_catalog.json";
/// Current on-disk schema version.
pub const CATALOG_VERSION: u32 = 1;

pub fn default_path() -> PathBuf {
    crate::paths::brainrouter_state().join(CATALOG_FILE)
}

/// `$XDG_STATE_HOME/brainrouter/swap_logs` — engine spawn/stop log files.
pub fn logs_dir() -> PathBuf {
    crate::paths::brainrouter_state().join("swap_logs")
}

// ── Data shapes ─────────────────────────────────────────────────────────────

/// Fully-resolved engine launch. `${PORT}` is the one placeholder a launch
/// spec may still carry: the scheduler assigns the port at spawn time (port
/// policy `managed`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct LaunchSpec {
    #[serde(default)]
    pub program: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default = "default_port_policy")]
    pub port_policy: String,
    /// The model name the engine serves (stem-assertion target). Defaults to
    /// the catalog id.
    #[serde(default)]
    pub served_model_alias: String,
    /// llama-swap `proxy:` template (external engines — recorded, not spawned).
    #[serde(default)]
    pub proxy_template: Option<String>,
    /// llama-swap `cmdStop:` (external engines — recorded, not spawned).
    #[serde(default)]
    pub stop_command: Option<String>,
}

fn default_port_policy() -> String {
    "managed".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Provenance {
    /// `llama-swap-import` | `cli` | `toolbox` | ...
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub imported_at: Option<String>,
    #[serde(default)]
    pub spec_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileFingerprint {
    pub path: String,
    /// Unix mtime seconds (fractional part dropped deliberately: filesystem
    /// granularity varies).
    pub mtime: i64,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CatalogEntry {
    pub id: String,
    #[serde(default = "default_engine")]
    pub engine: String,
    /// Primary GGUF(s); the first is the canonical lock/identity path.
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default)]
    pub ctx: Option<u64>,
    #[serde(default)]
    pub quant: Option<String>,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub tools: Option<bool>,
    #[serde(default)]
    pub vram_estimate_gib: Option<f64>,
    #[serde(default)]
    pub launch_spec: LaunchSpec,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub groups: Vec<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub provenance: Provenance,
    /// LRU clock: monotonic millis of the last completed stream (0 = never).
    #[serde(default)]
    pub last_stream_completed: u64,
    /// EWMA of measured load durations (native adapter ETA; absent = unknown).
    #[serde(default)]
    pub ewma_load_ms: Option<f64>,
    #[serde(default)]
    pub fingerprints: Vec<FileFingerprint>,
}

fn default_engine() -> String {
    "llama_cpp".to_string()
}

impl Default for CatalogEntry {
    fn default() -> Self {
        Self {
            id: String::new(),
            engine: default_engine(),
            files: Vec::new(),
            ctx: None,
            quant: None,
            family: None,
            tools: None,
            vram_estimate_gib: None,
            launch_spec: LaunchSpec::default(),
            aliases: Vec::new(),
            groups: Vec::new(),
            name: None,
            provenance: Provenance::default(),
            last_stream_completed: 0,
            ewma_load_ms: None,
            fingerprints: Vec::new(),
        }
    }
}

impl CatalogEntry {
    /// Canonical primary GGUF path (first file that exists, canonicalized).
    pub fn canonical_gguf(&self) -> Option<PathBuf> {
        self.files.iter().find_map(|f| {
            let p = Path::new(f);
            std::fs::canonicalize(p).ok().or_else(|| p.exists().then(|| p.to_path_buf()))
        })
    }

    /// The name the engine is expected to advertise (`--alias`, else id).
    pub fn served_alias(&self) -> String {
        if self.launch_spec.served_model_alias.is_empty() {
            self.id.clone()
        } else {
            self.launch_spec.served_model_alias.clone()
        }
    }

    /// True when a tracked GGUF file changed size/mtime since import — the
    /// `spec_stale` signal (event + re-import hint; a serving engine is never
    /// killed for this).
    pub fn fingerprint_stale(&self) -> Option<String> {
        for fp in &self.fingerprints {
            let Ok(meta) = std::fs::metadata(&fp.path) else {
                return Some(format!("{} is missing", fp.path));
            };
            let mtime = meta
                .modified()
                .ok()
                .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            if mtime != fp.mtime || meta.len() != fp.size {
                return Some(format!(
                    "{} changed (size {} → {}, mtime {} → {})",
                    fp.path,
                    fp.size,
                    meta.len(),
                    fp.mtime,
                    mtime
                ));
            }
        }
        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Pin {
    pub reserved_gib: f64,
    #[serde(default)]
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GroupMeta {
    #[serde(default)]
    pub members: Vec<String>,
    #[serde(default = "default_true")]
    pub swap: bool,
    #[serde(default = "default_true")]
    pub exclusive: bool,
    #[serde(default)]
    pub persistent: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SwapCatalogData {
    #[serde(default = "default_version")]
    pub version: u32,
    #[serde(default)]
    pub entries: BTreeMap<String, CatalogEntry>,
    #[serde(default)]
    pub pins: BTreeMap<String, Pin>,
    #[serde(default)]
    pub groups: BTreeMap<String, GroupMeta>,
    /// llama-swap settings captured at import (`startPort`, timeouts...).
    #[serde(default)]
    pub settings: BTreeMap<String, serde_json::Value>,
    /// Phase markers (`v2.0-p4` → timestamp) make migrations one-shot.
    #[serde(default)]
    pub migrations: BTreeMap<String, String>,
}

fn default_version() -> u32 {
    CATALOG_VERSION
}

impl Default for SwapCatalogData {
    fn default() -> Self {
        Self {
            version: CATALOG_VERSION,
            entries: BTreeMap::new(),
            pins: BTreeMap::new(),
            groups: BTreeMap::new(),
            settings: BTreeMap::new(),
            migrations: BTreeMap::new(),
        }
    }
}

// ── Store ───────────────────────────────────────────────────────────────────

pub struct SwapCatalog {
    path: PathBuf,
    inner: Mutex<SwapCatalogData>,
}

impl SwapCatalog {
    /// Default path under the real state root.
    pub fn load_default() -> Result<Self> {
        Self::load(default_path())
    }

    /// Load (missing file = empty catalog). A corrupt file is an error: the
    /// caller decides whether to quarantine (never silently drop pins).
    pub fn load(path: PathBuf) -> Result<Self> {
        let data = match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text)
                .with_context(|| format!("parsing swap catalog {}", path.display()))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => SwapCatalogData::default(),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        Ok(Self {
            path,
            inner: Mutex::new(data),
        })
    }

    /// In-memory catalog (tests / daemon fallback when the file is corrupt).
    pub fn in_memory(path: PathBuf) -> Self {
        Self {
            path,
            inner: Mutex::new(SwapCatalogData::default()),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn snapshot(&self) -> SwapCatalogData {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn entry(&self, id: &str) -> Option<CatalogEntry> {
        self.snapshot().entries.remove(id)
    }

    pub fn entries(&self) -> Vec<CatalogEntry> {
        self.snapshot().entries.into_values().collect()
    }

    pub fn has_entry(&self, id: &str) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entries
            .contains_key(id)
    }

    /// Resolve a served name (catalog id or imported alias) to a catalog id.
    pub fn resolve_alias(&self, name: &str) -> Option<String> {
        let data = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if data.entries.contains_key(name) {
            return Some(name.to_string());
        }
        data.entries
            .values()
            .find(|e| e.aliases.iter().any(|a| a == name) || e.served_alias() == name)
            .map(|e| e.id.clone())
    }

    pub fn pin(&self, id: &str, reserved_gib: f64) -> Result<()> {
        anyhow::ensure!(reserved_gib > 0.0, "reserved_gib must be > 0");
        let mut data = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        anyhow::ensure!(
            data.entries.contains_key(id),
            "`{id}` is not in the swap catalog — import or add it first"
        );
        data.pins.insert(
            id.to_string(),
            Pin {
                reserved_gib,
                created_at: chrono::Utc::now().to_rfc3339(),
            },
        );
        self.write_locked(&data)
    }

    pub fn unpin(&self, id: &str) -> Result<bool> {
        let mut data = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let removed = data.pins.remove(id).is_some();
        self.write_locked(&data)?;
        Ok(removed)
    }

    pub fn pins(&self) -> BTreeMap<String, Pin> {
        self.snapshot().pins
    }

    pub fn groups(&self) -> BTreeMap<String, GroupMeta> {
        self.snapshot().groups
    }

    pub fn record_stream_completed(&self, id: &str, at_ms: u64) -> Result<bool> {
        let mut data = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let Some(entry) = data.entries.get_mut(id) else {
            return Ok(false);
        };
        if at_ms > entry.last_stream_completed {
            entry.last_stream_completed = at_ms;
        }
        self.write_locked(&data)?;
        Ok(true)
    }

    pub fn record_load_ms(&self, id: &str, load_ms: f64) -> Result<()> {
        let mut data = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let Some(entry) = data.entries.get_mut(id) else {
            anyhow::bail!("`{id}` is not in the swap catalog");
        };
        entry.ewma_load_ms = Some(match entry.ewma_load_ms {
            // EWMA α = 0.3: fast enough to follow hardware changes, slow
            // enough that one cold disk read does not dominate the ETA.
            Some(prev) => prev * 0.7 + load_ms * 0.3,
            None => load_ms,
        });
        self.write_locked(&data)
    }

    pub fn upsert_entry(&self, entry: CatalogEntry) -> Result<()> {
        let mut data = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        data.entries.insert(entry.id.clone(), entry);
        self.write_locked(&data)
    }

    pub fn remove_entry(&self, id: &str) -> Result<bool> {
        let mut data = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let removed = data.entries.remove(id).is_some();
        data.pins.remove(id);
        self.write_locked(&data)?;
        Ok(removed)
    }

    pub fn mark_migration(&self, phase: &str) -> Result<()> {
        let mut data = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        data.migrations
            .insert(phase.to_string(), chrono::Utc::now().to_rfc3339());
        self.write_locked(&data)
    }

    pub fn migration_done(&self, phase: &str) -> bool {
        self.snapshot().migrations.contains_key(phase)
    }

    /// Idempotent merge of an import plan. Entries whose provenance is not
    /// `llama-swap-import` are never overwritten (CLI additions win).
    pub fn apply_import(&self, plan: &ImportPlan) -> Result<ImportOutcome> {
        let mut data = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut outcome = ImportOutcome::default();
        let now = chrono::Utc::now().to_rfc3339();
        for (id, incoming) in &plan.entries {
            match data.entries.get(id) {
                None => {
                    let mut entry = incoming.clone();
                    entry.provenance.imported_at = Some(now.clone());
                    data.entries.insert(id.clone(), entry);
                    outcome.added += 1;
                }
                Some(existing) => {
                    if !existing.provenance.source.is_empty()
                        && existing.provenance.source != "llama-swap-import"
                    {
                        outcome.skipped.push(id.clone());
                        continue;
                    }
                    let mut merged = incoming.clone();
                    merged.provenance.imported_at = existing
                        .provenance
                        .imported_at
                        .clone()
                        .or_else(|| Some(now.clone()));
                    merged.last_stream_completed = existing.last_stream_completed;
                    merged.ewma_load_ms = existing.ewma_load_ms;
                    // Stable comparison ignores the import timestamp.
                    if entries_equal(&merged, existing) {
                        outcome.unchanged += 1;
                    } else {
                        data.entries.insert(id.clone(), merged);
                        outcome.updated += 1;
                    }
                }
            }
        }
        for (name, group) in &plan.groups {
            data.groups.insert(name.clone(), group.clone());
        }
        for (key, value) in &plan.settings {
            data.settings.insert(key.clone(), value.clone());
        }
        self.write_locked(&data)?;
        Ok(outcome)
    }

    fn write_locked(&self, data: &SwapCatalogData) -> Result<()> {
        let text = serde_json::to_string_pretty(data).context("serializing swap catalog")?;
        crate::plugins::store::atomic_write_0600(&self.path, text.as_bytes())
            .with_context(|| format!("writing swap catalog {}", self.path.display()))
    }
}

fn entries_equal(a: &CatalogEntry, b: &CatalogEntry) -> bool {
    let mut a = a.clone();
    let mut b = b.clone();
    a.provenance.imported_at = None;
    b.provenance.imported_at = None;
    a == b
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ImportOutcome {
    pub added: usize,
    pub updated: usize,
    pub unchanged: usize,
    pub skipped: Vec<String>,
}

// ── llama-swap import (inverse of llama_swap_register) ──────────────────────

#[derive(Debug, Clone, Default)]
pub struct ImportPlan {
    pub entries: BTreeMap<String, CatalogEntry>,
    pub groups: BTreeMap<String, GroupMeta>,
    pub settings: BTreeMap<String, serde_json::Value>,
    pub warnings: Vec<String>,
}

impl ImportPlan {
    /// Stable human-readable preview (`--dry-run`). Never contains timestamps,
    /// so repeated dry-runs print byte-identical output.
    pub fn summary_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push(format!(
            "would import {} model(s), {} group(s) from llama-swap",
            self.entries.len(),
            self.groups.len()
        ));
        for entry in self.entries.values() {
            lines.push(format!(
                "  {} [engine={}] ctx={} files={} args={}{}",
                entry.id,
                entry.engine,
                entry
                    .ctx
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "?".into()),
                entry.files.len(),
                entry.launch_spec.args.len(),
                if entry.groups.is_empty() {
                    String::new()
                } else {
                    format!(" groups={}", entry.groups.join(","))
                }
            ));
        }
        for (name, group) in &self.groups {
            lines.push(format!(
                "  group {}: members={} swap={} exclusive={}",
                name,
                group.members.len(),
                group.swap,
                group.exclusive
            ));
        }
        for warning in &self.warnings {
            lines.push(format!("  warning: {warning}"));
        }
        lines
    }
}

/// Parse llama-swap `config.yaml` into a plan. Accepts both config shapes:
/// `models:` as a mapping (current) and as a sequence (legacy).
pub fn import_llama_swap_config(text: &str) -> Result<ImportPlan> {
    let root: serde_yaml::Value = serde_yaml::from_str(text).context("invalid llama-swap YAML")?;
    let mut plan = ImportPlan::default();
    let Some(root_map) = root.as_mapping() else {
        anyhow::bail!("llama-swap config must be a top-level YAML mapping");
    };

    for key in [
        "startPort",
        "healthCheckTimeout",
        "globalTTL",
        "sendLoadingState",
        "logLevel",
    ] {
        if let Some(value) = root_map.get(serde_yaml::Value::String(key.into())) {
            if let Ok(json) = serde_json::to_value(value) {
                plan.settings.insert(key.to_string(), json);
            }
        }
    }

    let global_macros = yaml_string_map(root_map.get(serde_yaml::Value::String("macros".into())));

    // groups: {name: {members: [...], swap:, exclusive:, persistent:}}
    if let Some(groups) = root_map
        .get(serde_yaml::Value::String("groups".into()))
        .and_then(|g| g.as_mapping())
    {
        for (name, body) in groups {
            let Some(name) = name.as_str() else { continue };
            let members: Vec<String> = body
                .get("members")
                .and_then(|m| m.as_sequence())
                .map(|seq| {
                    seq.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            plan.groups.insert(
                name.to_string(),
                GroupMeta {
                    members,
                    swap: body.get("swap").and_then(|v| v.as_bool()).unwrap_or(true),
                    exclusive: body
                        .get("exclusive")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true),
                    persistent: body
                        .get("persistent")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                },
            );
        }
    }

    let models: Vec<(String, serde_yaml::Value)> = match root_map
        .get(serde_yaml::Value::String("models".into()))
    {
        Some(serde_yaml::Value::Mapping(map)) => map
            .iter()
            .filter_map(|(k, v)| k.as_str().map(|k| (k.to_string(), v.clone())))
            .collect(),
        Some(serde_yaml::Value::Sequence(seq)) => seq
            .iter()
            .filter_map(|model| {
                model
                    .get("name")
                    .or_else(|| model.get("id"))
                    .and_then(|v| v.as_str())
                    .map(|k| (k.to_string(), model.clone()))
            })
            .collect(),
        _ => Vec::new(),
    };

    for (id, model) in models {
        match import_model(&id, &model, &global_macros, &plan.groups) {
            Ok(entry) => {
                plan.entries.insert(id, entry);
            }
            Err(e) => plan.warnings.push(format!("{id}: {e}")),
        }
    }
    Ok(plan)
}

pub fn import_llama_swap_file(path: &Path) -> Result<ImportPlan> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading llama-swap config {}", path.display()))?;
    import_llama_swap_config(&text)
}

fn import_model(
    id: &str,
    model: &serde_yaml::Value,
    global_macros: &BTreeMap<String, String>,
    groups: &BTreeMap<String, GroupMeta>,
) -> Result<CatalogEntry> {
    let cmd = model
        .get("cmd")
        .and_then(|v| v.as_str())
        .context("model has no `cmd` (external/container entries are not importable)")?;
    let mut macros = global_macros.clone();
    macros.extend(yaml_string_map(model.get("macros")));
    let expanded = expand_macros(cmd, &macros);

    let tokens = shlex::split(&expanded).unwrap_or_else(|| {
        expanded
            .split_whitespace()
            .map(str::to_string)
            .collect::<Vec<_>>()
    });
    let mut program = String::new();
    let mut args = Vec::new();
    if let Some(first) = tokens.first() {
        if !first.starts_with('-') {
            program = first.clone();
            args.extend(tokens.iter().skip(1).cloned());
        } else {
            args.extend(tokens.iter().cloned());
        }
    }

    let mut files: Vec<String> = Vec::new();
    let mut ctx = None;
    let mut alias = None;
    let mut tools = None;
    let mut fixed_port = false;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let next = args.get(i + 1).cloned();
        match arg {
            "--model" | "-m" => {
                if let Some(path) = next {
                    if !files.contains(&path) {
                        files.push(path);
                    }
                    i += 1;
                }
            }
            "--ctx-size" | "-c" | "--fit-ctx" => {
                if let Some(value) = next {
                    if let Ok(parsed) = value.parse::<u64>() {
                        ctx = Some(ctx.map_or(parsed, |c: u64| c.max(parsed)));
                    }
                    i += 1;
                }
            }
            "--alias" | "--served-model-name" => {
                if let Some(value) = next {
                    alias = Some(value);
                    i += 1;
                }
            }
            "--jinja" => {
                tools = Some(true);
                i += 0;
            }
            "--port" => {
                if let Some(value) = next {
                    if value != "${PORT}" {
                        fixed_port = true;
                    }
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    if alias.is_none() {
        // llama-swap serves the config key as the model name; imported aliases
        // are extra accepted names, never the canonical served stem.
        alias = Some(id.to_string());
    }
    let mut env = BTreeMap::new();
    match model.get("env") {
        Some(serde_yaml::Value::Sequence(seq)) => {
            for item in seq {
                if let Some(pair) = item.as_str() {
                    if let Some((k, v)) = pair.split_once('=') {
                        env.insert(k.trim().to_string(), v.trim().to_string());
                    }
                }
            }
        }
        Some(serde_yaml::Value::Mapping(map)) => {
            for (k, v) in map {
                if let (Some(k), Some(v)) = (k.as_str(), v.as_str()) {
                    env.insert(k.to_string(), v.to_string());
                }
            }
        }
        _ => {}
    }
    let aliases: Vec<String> = model
        .get("aliases")
        .and_then(|a| a.as_sequence())
        .map(|seq| {
            seq.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .filter(|a| a != id)
                .collect()
        })
        .unwrap_or_default();

    let quant = files.first().and_then(|f| quant_from_filename(f));
    let vram_estimate_gib = files.first().and_then(|f| {
        std::fs::metadata(f)
            .ok()
            .map(|m| (m.len() as f64 / 1024.0 / 1024.0 / 1024.0) * 1.1)
    });
    let fingerprints = files
        .iter()
        .filter_map(|f| {
            let meta = std::fs::metadata(f).ok()?;
            let mtime = meta
                .modified()
                .ok()
                .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            Some(FileFingerprint {
                path: f.clone(),
                mtime,
                size: meta.len(),
            })
        })
        .collect();

    let mut entry = CatalogEntry {
        id: id.to_string(),
        engine: engine_for_program(&program),
        files,
        ctx,
        quant,
        family: None,
        tools,
        vram_estimate_gib,
        launch_spec: LaunchSpec {
            program,
            args,
            env,
            port_policy: if fixed_port { "fixed".into() } else { "managed".into() },
            served_model_alias: alias.unwrap_or_else(|| id.to_string()),
            proxy_template: model
                .get("proxy")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            stop_command: model
                .get("cmdStop")
                .and_then(|v| v.as_str())
                .map(str::to_string),
        },
        aliases,
        groups: groups
            .iter()
            .filter(|(_, g)| g.members.iter().any(|m| m == id))
            .map(|(name, _)| name.clone())
            .collect(),
        name: model.get("name").and_then(|v| v.as_str()).map(str::to_string),
        provenance: Provenance {
            source: "llama-swap-import".into(),
            imported_at: None,
            spec_hash: String::new(),
        },
        last_stream_completed: 0,
        ewma_load_ms: None,
        fingerprints,
    };
    entry.provenance.spec_hash = launch_spec_hash(&entry);
    Ok(entry)
}

fn yaml_string_map(value: Option<&serde_yaml::Value>) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    if let Some(serde_yaml::Value::Mapping(map)) = value {
        for (k, v) in map {
            if let Some(k) = k.as_str() {
                let text = match v {
                    serde_yaml::Value::String(s) => s.clone(),
                    other => serde_yaml::to_string(other)
                        .unwrap_or_default()
                        .trim_end()
                        .to_string(),
                };
                out.insert(k.to_string(), text);
            }
        }
    }
    out
}

/// Recursively substitute `${name}` from the macro map. Runtime placeholders
/// not present in the macros (`${PORT}`, `${MODEL_ID}`) are preserved.
pub fn expand_macros(text: &str, macros: &BTreeMap<String, String>) -> String {
    let mut current = text.to_string();
    for _ in 0..16 {
        let mut changed = false;
        let mut next = String::with_capacity(current.len());
        let bytes = current.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'$' && bytes.get(i + 1) == Some(&b'{') {
                if let Some(end_rel) = current[i + 2..].find('}') {
                    let end = i + 2 + end_rel;
                    let name = &current[i + 2..end];
                    if let Some(value) = macros.get(name) {
                        next.push_str(value);
                        i = end + 1;
                        changed = true;
                        continue;
                    }
                }
            }
            let ch = current[i..].chars().next().unwrap();
            next.push(ch);
            i += ch.len_utf8();
        }
        current = next;
        if !changed {
            break;
        }
    }
    current
}

/// Engine classification from the resolved program basename.
pub fn engine_for_program(program: &str) -> String {
    let base = Path::new(program)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(program)
        .to_ascii_lowercase();
    if base.contains("gufo") {
        "gufo".into()
    } else if base.contains("ds4") {
        "ds4".into()
    } else if base.contains("llama") {
        "llama_cpp".into()
    } else if base.is_empty() {
        "unknown".into()
    } else {
        "external".into()
    }
}

fn quant_from_filename(path: &str) -> Option<String> {
    let upper = path.to_ascii_uppercase();
    for marker in [
        "UD-Q8_K_XL", "Q8_K_XL", "UD-Q6_K_XL", "Q6_K_XL", "Q6_K_L", "Q5_K_M", "Q5_K_S",
        "UD-Q4_K_XL", "Q4_K_XL", "Q4_K_M", "Q4_K_S", "Q3_K_M", "Q2_K", "BF16", "F16", "Q8_0",
    ] {
        if upper.contains(marker) {
            return Some(marker.to_string());
        }
    }
    None
}

/// sha256 over the resolved launch identity. Adoption compares this hash:
/// same pid + same gguf + same hash = adopt silently; a mismatch adopts +
/// marks `spec_stale` (a serving engine is never killed by an upgrade).
pub fn launch_spec_hash(entry: &CatalogEntry) -> String {
    let mut hasher = Sha256::new();
    hasher.update(entry.launch_spec.program.as_bytes());
    for arg in &entry.launch_spec.args {
        hasher.update(b"\x00");
        hasher.update(arg.as_bytes());
    }
    for (k, v) in &entry.launch_spec.env {
        hasher.update(b"\x01");
        hasher.update(k.as_bytes());
        hasher.update(b"=");
        hasher.update(v.as_bytes());
    }
    hasher.update(b"\x02");
    hasher.update(entry.ctx.unwrap_or(0).to_string().as_bytes());
    hasher.update(b"\x03");
    hasher.update(entry.served_alias().as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Port-less launch hash for an ADOPTED icon-era external: the marker records
/// profile/template + weights, EXCLUDING the allocated port, so an adopted
/// resident never carries a permanent `spec_stale` badge (L5 binding).
pub fn adopted_spec_hash(entry: &CatalogEntry) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"adopted\x00");
    hasher.update(entry.engine.as_bytes());
    hasher.update(b"\x00");
    hasher.update(entry.launch_spec.program.as_bytes());
    hasher.update(b"\x00");
    if let Some(gguf) = entry.canonical_gguf() {
        hasher.update(gguf.display().to_string().as_bytes());
    }
    for file in &entry.files {
        hasher.update(b"\x04");
        hasher.update(file.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Cheap sanity checks for adapter-owned launch specs (engine-validated flags
/// are the adapter's job; this rejects structurally unusable entries).
pub fn validate_entry(entry: &CatalogEntry) -> Result<()> {
    anyhow::ensure!(!entry.id.is_empty(), "entry id must not be empty");
    anyhow::ensure!(
        [
            "llama_cpp",
            "fake",
            "ds4",
            "gufo",
            "external",
            "strixite",
            "hipfire"
        ]
        .contains(&entry.engine.as_str()),
        "unknown engine `{}`",
        entry.engine
    );
    if matches!(entry.engine.as_str(), "llama_cpp" | "fake") {
        anyhow::ensure!(
            !entry.files.is_empty() || entry.engine == "fake",
            "engine `{}` needs at least one model file",
            entry.engine
        );
        anyhow::ensure!(
            !entry.launch_spec.program.is_empty() || entry.engine == "fake",
            "engine `{}` needs launch_spec.program",
            entry.engine
        );
    }
    anyhow::ensure!(
        entry.groups.iter().all(|g| !g.is_empty()),
        "group names must not be empty"
    );
    Ok(())
}

/// Group membership as a set, for the Engines tab / L7 input.
pub fn group_members(data: &SwapCatalogData) -> BTreeMap<String, BTreeSet<String>> {
    data.groups
        .iter()
        .map(|(name, meta)| {
            (
                name.clone(),
                meta.members.iter().cloned().collect::<BTreeSet<_>>(),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"
startPort: 5800
macros:
  "ls": "/usr/bin/llama-server"
  "common": >-
    --jinja -t 8 -ngl 999
    --ctx-size ${ctx}
  "ctx": "16384"
groups:
  pair:
    swap: false
    exclusive: false
    members:
      - model-a
      - model-b
models:
  model-a:
    name: "Model A"
    aliases:
      - "a-alias"
    env:
      - "LLAMA_CONTAINER=a"
    macros:
      "ctx": "32768"
    cmd: |
      ${ls}
      --port ${PORT}
      ${common}
      --model /models/a.gguf
  model-b:
    name: "Model B"
    cmd: |
      ${ls} --jinja --port ${PORT} --ctx-size 8192 --model /models/b.gguf
"#;

    fn test_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "brainrouter-swap-catalog-{name}-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(CATALOG_FILE)
    }

    #[test]
    fn import_expands_the_macro_chain_and_records_groups() {
        let plan = import_llama_swap_config(FIXTURE).unwrap();
        let a = plan.entries.get("model-a").unwrap();
        assert_eq!(a.launch_spec.program, "/usr/bin/llama-server");
        // ${common} expanded recursively: ${ctx} became the per-model 32768.
        assert!(a.launch_spec.args.contains(&"--ctx-size".to_string()));
        let idx = a
            .launch_spec
            .args
            .iter()
            .position(|x| x == "--ctx-size")
            .unwrap();
        assert_eq!(a.launch_spec.args[idx + 1], "32768");
        assert!(!a.launch_spec.args.iter().any(|x| x.contains("${common}")));
        // ${PORT} is a runtime placeholder and survives expansion.
        assert!(a.launch_spec.args.iter().any(|x| x == "${PORT}"));
        assert_eq!(a.ctx, Some(32768));
        assert_eq!(a.files, vec!["/models/a.gguf"]);
        assert_eq!(a.aliases, vec!["a-alias"]);
        assert_eq!(a.launch_spec.env.get("LLAMA_CONTAINER").unwrap(), "a");
        assert_eq!(a.groups, vec!["pair"]);
        assert_eq!(a.engine, "llama_cpp");
        assert_eq!(a.served_alias(), "model-a");
        let group = plan.groups.get("pair").unwrap();
        assert!(!group.swap);
        assert!(!group.exclusive);
        assert_eq!(group.members, vec!["model-a", "model-b"]);
        assert_eq!(plan.entries.get("model-b").unwrap().ctx, Some(8192));
        assert_eq!(plan.settings.get("startPort").unwrap(), 5800);
    }

    #[test]
    fn re_import_is_idempotent_and_dry_run_is_stable() {
        let path = test_path("idempotent");
        let catalog = SwapCatalog::load(path).unwrap();
        let plan = import_llama_swap_config(FIXTURE).unwrap();
        let first = catalog.apply_import(&plan).unwrap();
        assert_eq!(first.added, 2);
        let before = catalog.snapshot();
        let second = catalog.apply_import(&plan).unwrap();
        assert_eq!(second.added, 0);
        assert_eq!(second.updated, 0);
        assert_eq!(second.unchanged, 2);
        assert_eq!(before, catalog.snapshot());
        // Dry-run rendering has no timestamps and is byte-stable.
        let a = import_llama_swap_config(FIXTURE).unwrap().summary_lines();
        let b = import_llama_swap_config(FIXTURE).unwrap().summary_lines();
        assert_eq!(a, b);
    }

    #[test]
    fn re_import_never_overwrites_cli_entries() {
        let path = test_path("provenance");
        let catalog = SwapCatalog::load(path).unwrap();
        let plan = import_llama_swap_config(FIXTURE).unwrap();
        catalog.apply_import(&plan).unwrap();
        let mut custom = catalog.entry("model-a").unwrap();
        custom.provenance.source = "cli".into();
        custom.ctx = Some(999);
        catalog.upsert_entry(custom).unwrap();
        let outcome = catalog.apply_import(&plan).unwrap();
        assert!(outcome.skipped.contains(&"model-a".to_string()));
        assert_eq!(catalog.entry("model-a").unwrap().ctx, Some(999));
    }

    #[test]
    fn pins_and_stream_clock_persist_across_reload() {
        let path = test_path("pins");
        let catalog = SwapCatalog::load(path.clone()).unwrap();
        catalog.apply_import(&import_llama_swap_config(FIXTURE).unwrap()).unwrap();
        catalog.pin("model-a", 24.0).unwrap();
        catalog.record_stream_completed("model-a", 1234).unwrap();
        catalog.record_load_ms("model-a", 1000.0).unwrap();
        catalog.record_load_ms("model-a", 2000.0).unwrap();
        let reloaded = SwapCatalog::load(path).unwrap();
        assert_eq!(reloaded.pins().get("model-a").unwrap().reserved_gib, 24.0);
        let entry = reloaded.entry("model-a").unwrap();
        assert_eq!(entry.last_stream_completed, 1234);
        // EWMA α=0.3: 1000 → 1300.
        assert!((entry.ewma_load_ms.unwrap() - 1300.0).abs() < 0.01);
    }

    #[test]
    fn pin_requires_a_catalog_entry_and_positive_size() {
        let path = test_path("pin-validation");
        let catalog = SwapCatalog::load(path).unwrap();
        assert!(catalog.pin("missing", 10.0).is_err());
        assert!(catalog.pin("missing", 0.0).is_err());
    }

    #[test]
    fn alias_resolution_covers_ids_aliases_and_served_names() {
        let path = test_path("aliases");
        let catalog = SwapCatalog::load(path).unwrap();
        catalog.apply_import(&import_llama_swap_config(FIXTURE).unwrap()).unwrap();
        assert_eq!(catalog.resolve_alias("model-a").as_deref(), Some("model-a"));
        assert_eq!(catalog.resolve_alias("a-alias").as_deref(), Some("model-a"));
        assert_eq!(catalog.resolve_alias("nope"), None);
    }

    #[test]
    fn sequence_shaped_models_import_too() {
        let plan = import_llama_swap_config(
            r#"
models:
  - name: legacy-model
    cmd: |
      /usr/bin/llama-server --port ${PORT} --model /x.gguf
"#,
        )
        .unwrap();
        assert!(plan.entries.contains_key("legacy-model"));
    }

    #[test]
    fn external_entries_without_cmd_are_warned_not_fatal() {
        let plan = import_llama_swap_config(
            r#"
models:
  gufo-thing:
    name: Gufo
    proxy: http://127.0.0.1:${PORT}
"#,
        )
        .unwrap();
        assert!(plan.entries.is_empty());
        assert_eq!(plan.warnings.len(), 1);
    }

    #[test]
    fn fingerprint_detects_changed_files() {
        let dir = std::env::temp_dir().join(format!(
            "brainrouter-swap-fingerprint-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let gguf = dir.join("a.gguf");
        std::fs::write(&gguf, b"data").unwrap();
        let mut entry = import_llama_swap_config(&format!(
            "models:\n  m:\n    cmd: |\n      /usr/bin/llama-server --model {}\n",
            gguf.display()
        ))
        .unwrap()
        .entries
        .into_values()
        .next()
        .unwrap();
        assert!(entry.fingerprint_stale().is_none());
        entry.fingerprints[0].size = 999;
        assert!(entry.fingerprint_stale().unwrap().contains("changed"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn spec_hash_covers_args_env_and_ctx() {
        let plan = import_llama_swap_config(FIXTURE).unwrap();
        let mut entry = plan.entries.get("model-a").unwrap().clone();
        let hash = launch_spec_hash(&entry);
        entry.ctx = Some(1);
        assert_ne!(hash, launch_spec_hash(&entry));
        let mut args_changed = plan.entries.get("model-a").unwrap().clone();
        args_changed.launch_spec.args.push("--extra".into());
        assert_ne!(hash, launch_spec_hash(&args_changed));
        let mut env_changed = plan.entries.get("model-a").unwrap().clone();
        env_changed
            .launch_spec
            .env
            .insert("NEW".into(), "1".into());
        assert_ne!(hash, launch_spec_hash(&env_changed));
    }
}
