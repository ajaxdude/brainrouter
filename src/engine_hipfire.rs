//! hipfire engine adapter (v2 L5).
//!
//! hipfire is the fedora1 (gfx1100) engine. Its model registry is a directory
//! of card files (`*.mq4`, an `.mtp` sidecar auto-loaded by stem). Launch is
//! `hipfire serve` with an **engine-version-keyed flag schema**: the schema is
//! re-derived when the probed engine version changes (an upgrade cannot
//! silently leave a poisoned stored schema) and unknown flags, notably
//! `--mtp`, which does not exist, are rejected at spec build.
//!
//! Port ownership is a single-owner rule on `:8080`. A foreign server already
//! answering on the port is a **squat**: the adapter refuses to spawn and the
//! supervisor surfaces `Failed` + a banner instead of losing a suite to a
//! port collision. Stop is SIGTERM plus a port-free gate (`/health` must fail
//! before the next spawn).
//!
//! Context math is pre-forward: `max_tokens` defaults to the engine's
//! remaining context (max-seq − measured prompt); a prompt that does not fit
//! yields a typed [`ContextShortfall`] the proxy surfaces as a 400.

use anyhow::Result;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::engine_process;
use crate::swap_scheduler::{
    EngineAdapter, EngineHandle, GgufLock, HealthReport, LoadProgress, SpawnRequest,
};

pub const ENGINE: &str = "hipfire";
pub const DEFAULT_PORT: u16 = 8080;
/// Default engine max sequence length when a card/entry does not say.
pub const DEFAULT_MAX_SEQ: u64 = 8192;
/// The composite/systemd unit the operator disables during migration.
pub const DEFAULT_UNIT: &str = "hipfire-serve.service";

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ── Card registry ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HipfireCard {
    pub stem: String,
    pub weights: PathBuf,
    /// `.mtp` sidecar path when present (auto-loaded by stem).
    pub mtp: Option<PathBuf>,
    pub sidecar_present: bool,
}

/// Scan `~/.hipfire/models/` for `*.mq4` cards. A missing `.mtp` sidecar is a
/// warning, never an error: the engine auto-loads by stem either way.
pub fn scan_cards(dir: &Path) -> (Vec<HipfireCard>, Vec<String>) {
    let mut cards = Vec::new();
    let mut warnings = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return (cards, warnings);
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let is_mq4 = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("mq4"));
        if !is_mq4 {
            continue;
        }
        let Some(stem) = path.file_stem().map(|s| s.to_string_lossy().into_owned()) else {
            continue;
        };
        let sidecar = dir.join(format!("{stem}.mtp"));
        let mtp = sidecar.is_file().then(|| sidecar.clone());
        if mtp.is_none() {
            warnings.push(format!(
                "card `{stem}` has no `.mtp` sidecar (engine auto-loads by stem)"
            ));
        }
        cards.push(HipfireCard {
            stem,
            weights: path,
            sidecar_present: mtp.is_some(),
            mtp,
        });
    }
    cards.sort_by(|a, b| a.stem.cmp(&b.stem));
    (cards, warnings)
}

// ── Version-keyed flag schema ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct HipfireSchema {
    pub version: String,
    pub allowed_flags: BTreeSet<String>,
}

impl HipfireSchema {
    /// Derive the flag schema for an engine version marker. Schema drift is a
    /// first-class outcome (the Engines tab badges it), so the rule is data:
    /// the base flag set plus version-gated additions.
    pub fn for_version(version: &str) -> Self {
        let mut allowed_flags: BTreeSet<String> = [
            "--model",
            "--host",
            "--port",
            "--ctx",
            "--max-seq",
            "--ngl",
            "--threads",
            "--temp",
            "--top-p",
            "--top-k",
            "--repeat-penalty",
            "--verbose",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let major: u64 = version
            .split('.')
            .next()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        if major >= 1 {
            // Documented v1 flag: an upgrade re-derives the schema so the
            // flag becomes legal without a rebuild.
            allowed_flags.insert("--flash-attn".to_string());
        }
        Self {
            version: version.to_string(),
            allowed_flags,
        }
    }

    /// Reject unknown flags and the non-existent speculative flag. The stored
    /// schema is the validator, so an engine upgrade that changes the surface
    /// is caught by re-derivation rather than a silently poisoned spec.
    pub fn validate_args(&self, args: &[String]) -> Result<()> {
        for arg in args {
            if !arg.starts_with('-') {
                continue;
            }
            let flag = arg.split('=').next().unwrap_or(arg);
            if flag == "--mtp" {
                anyhow::bail!(
                    "hipfire has no `--mtp` speculative flag (schema {} rejects it by construction)",
                    self.version
                );
            }
            if !self.allowed_flags.contains(flag) {
                anyhow::bail!(
                    "hipfire schema {} does not know flag `{flag}` (engine-validated args only)",
                    self.version
                );
            }
        }
        Ok(())
    }

    pub fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "version": self.version,
            "flags": self.allowed_flags,
        })
    }
}

// ── Context math ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct ContextShortfall {
    pub prompt_tokens: u64,
    pub ctx: u64,
}

impl std::fmt::Display for ContextShortfall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "context shortfall: prompt is {} tokens but the engine context is {} (needs {} more)",
            self.prompt_tokens,
            self.ctx,
            self.prompt_tokens.saturating_sub(self.ctx)
        )
    }
}

impl std::error::Error for ContextShortfall {}

/// Remaining context for generation after the prompt.
pub fn remaining_context(max_seq: u64, prompt_tokens: u64) -> u64 {
    max_seq.saturating_sub(prompt_tokens)
}

/// Pre-forward shaping: when the client omits `max_tokens`, default it to the
/// engine's remaining context. A prompt that cannot fit is a shortfall (the
/// proxy answers 400 with this text).
pub fn default_max_tokens(
    ctx: u64,
    prompt_tokens: u64,
) -> std::result::Result<u64, ContextShortfall> {
    let remaining = remaining_context(ctx, prompt_tokens);
    if remaining == 0 {
        return Err(ContextShortfall {
            prompt_tokens,
            ctx,
        });
    }
    Ok(remaining)
}

/// Apply the default to a request's optional `max_tokens`, also clamping a
/// client-supplied value that overshoots the remaining context.
pub fn shape_max_tokens(
    requested: Option<u32>,
    ctx: u64,
    prompt_tokens: u64,
) -> std::result::Result<u32, ContextShortfall> {
    let remaining = default_max_tokens(ctx, prompt_tokens)?;
    let remaining = remaining.min(u32::MAX as u64) as u32;
    Ok(match requested {
        Some(value) if value > 0 => value.min(remaining),
        _ => remaining,
    })
}

// ── Profile + adapter ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct HipfireProfile {
    pub program: String,
    pub models_dir: Option<PathBuf>,
    /// Single-owner port (`:8080` by fleet profile; env override for fixtures).
    pub port: u16,
    pub min_mem_gib: f64,
    pub device: String,
    pub unit: Option<String>,
    /// Engine max sequence length (card metadata when available).
    pub max_seq: u64,
    /// How long `/health` may still answer after a stop before it is an error.
    pub port_free_timeout: Duration,
}

impl Default for HipfireProfile {
    fn default() -> Self {
        Self::from_env()
    }
}

impl HipfireProfile {
    pub fn from_env() -> Self {
        let models_dir = std::env::var("BRAINROUTER_HIPFIRE_MODELS_DIR")
            .map(PathBuf::from)
            .ok()
            .or_else(|| {
                std::env::var("HOME")
                    .ok()
                    .map(|home| PathBuf::from(home).join(".hipfire/models"))
            });
        Self {
            program: std::env::var("BRAINROUTER_HIPFIRE_BIN")
                .unwrap_or_else(|_| "hipfire".to_string()),
            models_dir,
            port: engine_process::env_u64("BRAINROUTER_HIPFIRE_PORT")
                .map(|p| p as u16)
                .unwrap_or(DEFAULT_PORT),
            min_mem_gib: engine_process::env_f64("BRAINROUTER_HIPFIRE_MIN_MEM_GIB")
                .unwrap_or(16.0),
            device: std::env::var("BRAINROUTER_HIPFIRE_DEVICE")
                .unwrap_or_else(|_| "hipfire:gfx1100".to_string()),
            unit: Some(
                std::env::var("BRAINROUTER_HIPFIRE_UNIT")
                    .unwrap_or_else(|_| DEFAULT_UNIT.to_string()),
            ),
            max_seq: engine_process::env_u64("BRAINROUTER_HIPFIRE_MAX_SEQ")
                .unwrap_or(DEFAULT_MAX_SEQ),
            port_free_timeout: Duration::from_millis(
                engine_process::env_u64("BRAINROUTER_HIPFIRE_PORT_FREE_MS").unwrap_or(10_000),
            ),
        }
    }
}

/// Why a port is squatted: a foreign stem set at the endpoint. Pure so the
/// Engines tab and unit tests share the classification.
pub fn squat_reason(stems: &[String], expected: &[&str]) -> Option<String> {
    if stems.is_empty() {
        return None;
    }
    if stems
        .iter()
        .any(|stem| expected.iter().any(|e| stem == e))
    {
        return None;
    }
    Some(format!(
        "port is owned by a foreign server serving {:?} (expected {expected:?})",
        stems
    ))
}

pub struct HipfireAdapter {
    profile: HipfireProfile,
    schema: std::sync::Mutex<HipfireSchema>,
    /// Last version the probe reported when the schema was derived.
    derived_from: std::sync::Mutex<Option<String>>,
    schema_drift: std::sync::Mutex<bool>,
    /// When the engine version marker was last probed (the Engines tab must
    /// not spawn `<program> --version` on every poll).
    last_probe: std::sync::Mutex<Option<std::time::Instant>>,
    children: std::sync::Mutex<HashMap<String, std::process::Child>>,
    use_scope: bool,
}

impl HipfireAdapter {
    pub fn new(profile: HipfireProfile) -> Self {
        // The schema is derived from the installed engine's version marker at
        // construction (`<program> --version`), so the Engines tab and the
        // first ensure already use the right flag set. A missing binary
        // leaves the version `unknown`; `probe()`/`spawn()` re-derive later.
        let version = Self::read_version(&profile.program).unwrap_or_else(|| "unknown".into());
        Self {
            profile,
            schema: std::sync::Mutex::new(HipfireSchema::for_version(&version)),
            derived_from: std::sync::Mutex::new(Some(version)),
            schema_drift: std::sync::Mutex::new(false),
            last_probe: std::sync::Mutex::new(Some(std::time::Instant::now())),
            children: std::sync::Mutex::new(HashMap::new()),
            use_scope: !matches!(
                std::env::var("BRAINROUTER_SWAP_NO_SYSTEMD").as_deref(),
                Ok("1")
            ) && crate::engine_native::systemd_scope_available(),
        }
    }

    /// `<program> --version` → version marker (`hipfire 1.4.2` → `1.4.2`).
    fn read_version(program: &str) -> Option<String> {
        let output = std::process::Command::new(program)
            .arg("--version")
            .stdin(std::process::Stdio::null())
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&output.stdout);
        let line = text.lines().next()?.trim();
        if line.is_empty() {
            return None;
        }
        Some(
            line.split_whitespace()
                .find(|token| token.chars().next().is_some_and(|c| c.is_ascii_digit()))
                .unwrap_or(line)
                .trim_matches(|c: char| !(c.is_ascii_alphanumeric() || c == '.' || c == '-'))
                .to_string(),
        )
    }

    /// Re-probe at most once per `ttl` (Engines-tab detail path).
    pub fn maybe_refresh_schema(&self, ttl: Duration) -> bool {
        let stale = {
            let last = self
                .last_probe
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            last.map(|at| at.elapsed() >= ttl).unwrap_or(true)
        };
        if stale {
            self.refresh_schema()
        } else {
            false
        }
    }

    pub fn profile(&self) -> &HipfireProfile {
        &self.profile
    }

    /// Probe the installed engine's version marker (`<program> --version`).
    pub fn probed_version(&self) -> Option<String> {
        Self::read_version(&self.profile.program)
    }

    /// Re-derive the schema when the engine version changed. Returns true when
    /// a drift was detected (the Engines tab badges it).
    pub fn refresh_schema(&self) -> bool {
        *self
            .last_probe
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(std::time::Instant::now());
        let version = self
            .probed_version()
            .or_else(|| {
                self.derived_from
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone()
            })
            .unwrap_or_else(|| "unknown".to_string());
        let mut schema = self.schema.lock().unwrap_or_else(|e| e.into_inner());
        if schema.version != version {
            *schema = HipfireSchema::for_version(&version);
            *self
                .derived_from
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(version);
            *self.schema_drift.lock().unwrap_or_else(|e| e.into_inner()) = true;
            true
        } else {
            false
        }
    }

    /// Test/tool seam: force the schema's version marker.
    pub fn set_schema_version(&self, version: &str) -> bool {
        let mut schema = self.schema.lock().unwrap_or_else(|e| e.into_inner());
        let changed = schema.version != version;
        if changed {
            *schema = HipfireSchema::for_version(version);
            *self
                .derived_from
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(version.to_string());
            *self.schema_drift.lock().unwrap_or_else(|e| e.into_inner()) = true;
        }
        changed
    }

    pub fn schema(&self) -> HipfireSchema {
        self.schema
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn schema_drift(&self) -> bool {
        *self.schema_drift.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Find the card whose stem or weights path matches `model`.
    pub fn card_for(&self, model: &str, weights: Option<&Path>) -> Option<HipfireCard> {
        if let Some(dir) = &self.profile.models_dir {
            let (cards, _) = scan_cards(dir);
            if let Some(card) = cards.into_iter().find(|card| {
                card.stem == model
                    || weights.is_some_and(|w| card.weights == w)
                    || weights.is_some_and(|w| w.file_stem() == Some(card.stem.as_ref()))
            }) {
                return Some(card);
            }
        }
        None
    }

    /// Refuse a spawn when a foreign server already owns the endpoint.
    fn check_squat(&self, endpoint: &str, expected: &[&str]) -> Result<()> {
        if let Some(stems) = engine_process::probe_stems(endpoint, Duration::from_millis(600)) {
            if let Some(reason) = squat_reason(&stems, expected) {
                anyhow::bail!("{reason}");
            }
            anyhow::bail!(
                "endpoint {endpoint} already serves {stems:?}; refusing to spawn a second owner"
            );
        }
        if std::net::TcpListener::bind(("127.0.0.1", self.profile.port)).is_err()
            && endpoint.ends_with(&self.profile.port.to_string())
        {
            anyhow::bail!(
                "port {} is already in use (single-owner rule; stop the foreign server first)",
                self.profile.port
            );
        }
        Ok(())
    }

    fn reaped(&self, endpoint: &str) -> Option<std::process::ExitStatus> {
        let mut children = self.children.lock().unwrap_or_else(|e| e.into_inner());
        match children.get_mut(endpoint) {
            Some(child) => match child.try_wait() {
                Ok(Some(status)) => Some(status),
                _ => None,
            },
            None => None,
        }
    }

    /// Port-free gate after a stop: `/health` must fail before the next spawn.
    pub fn wait_port_free(&self, endpoint: &str) -> bool {
        let deadline = std::time::Instant::now() + self.profile.port_free_timeout;
        loop {
            let health = crate::engine_native::http_get(
                endpoint,
                "/health",
                Duration::from_millis(400),
            );
            if health.is_none() {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

impl Drop for HipfireAdapter {
    fn drop(&mut self) {
        let children: Vec<std::process::Child> = self
            .children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .drain()
            .map(|(_, child)| child)
            .collect();
        for mut child in children {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl EngineAdapter for HipfireAdapter {
    fn name(&self) -> &'static str {
        ENGINE
    }

    fn probe(&self) -> Result<()> {
        if !crate::engine_native::program_exists(&self.profile.program) {
            anyhow::bail!("`{}` not found on PATH", self.profile.program);
        }
        // Re-derive on every probe: an upgrade changes the version marker and
        // therefore the legal flag set (schema drift is badged, never silent).
        self.refresh_schema();
        Ok(())
    }

    fn spawn(&self, request: &SpawnRequest) -> Result<EngineHandle> {
        // A spawn is the authority on the engine marker: re-derive first so an
        // upgraded engine cannot be poisoned by a stale stored schema.
        self.refresh_schema();
        let schema = self.schema();
        schema.validate_args(&request.spec.args)?;
        let expected_alias = request
            .spec
            .served_model_alias
            .clone();
        let expected = [request.model_id.as_str(), expected_alias.as_str()];
        self.check_squat(&request.endpoint, &expected)?;

        let weights = request
            .gguf
            .clone()
            .or_else(|| {
                self.card_for(&request.model_id, None)
                    .map(|card| card.weights)
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "hipfire needs a card ({}.mq4) or a weights path for `{}`",
                    request.model_id,
                    request.model_id
                )
            })?;
        let program = if request.spec.program.is_empty() {
            self.profile.program.clone()
        } else {
            request.spec.program.clone()
        };
        let mut args = vec![
            "serve".to_string(),
            "--model".to_string(),
            weights.display().to_string(),
            "--host".to_string(),
            "127.0.0.1".to_string(),
            "--port".to_string(),
            request.port.to_string(),
        ];
        args.extend(request.spec.args.iter().cloned());

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
        let mut env = request.spec.env.clone();
        env.extend(request.env.clone());
        env.entry("BRAINROUTER_SWAP_PORT".to_string())
            .or_insert_with(|| request.port.to_string());
        env.entry("PORT".to_string())
            .or_insert_with(|| request.port.to_string());
        let child = engine_process::spawn(engine_process::EngineProcessRequest {
            program: &program,
            args,
            env,
            log_path: &request.log_path,
            scope_name: request.scope_name.as_deref(),
            use_scope: self.use_scope,
            lock: lock.as_ref(),
        })?;
        let pid = child.id();
        self.children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(request.endpoint.clone(), child);
        drop(lock);
        Ok(EngineHandle {
            pid,
            endpoint: request.endpoint.clone(),
            started_at: now_ms(),
            scope_name: request.scope_name.clone(),
        })
    }

    fn stop(&self, handle: &EngineHandle) -> Result<()> {
        if let Some(scope) = &handle.scope_name {
            let _ = std::process::Command::new("systemctl")
                .args(["--user", "stop", scope])
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
        }
        let child = self
            .children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&handle.endpoint);
        if let Some(mut child) = child {
            engine_process::stop_child(&mut child, Duration::from_secs(10));
        } else {
            engine_process::terminate(handle.pid, Duration::from_secs(10));
        }
        // Port-free gate (binding): `/health` must fail before the next spawn.
        if !self.wait_port_free(&handle.endpoint) {
            anyhow::bail!(
                "port-free gate failed for {} (a server still answers /health)",
                handle.endpoint
            );
        }
        Ok(())
    }

    fn health(&self, endpoint: &str, handle: &EngineHandle) -> Result<HealthReport> {
        if let Some(status) = self.reaped(endpoint) {
            anyhow::bail!("engine pid {} exited ({status})", handle.pid);
        }
        let alive = handle.pid > 0 && unsafe { libc::kill(handle.pid as i32, 0) == 0 };
        match engine_process::probe_stems(endpoint, Duration::from_millis(800)) {
            Some(stems) => Ok(HealthReport {
                ready: true,
                stems: stems.into_iter().collect(),
            }),
            None if alive => Ok(HealthReport::loading()),
            None => anyhow::bail!("engine pid {} is not running", handle.pid),
        }
    }

    fn progress(&self, _endpoint: &str, handle: &EngineHandle) -> LoadProgress {
        LoadProgress {
            fraction: None,
            elapsed_ms: now_ms().saturating_sub(handle.started_at),
            eta_ms: None,
        }
    }

    fn min_mem_gib(&self) -> Option<f64> {
        Some(self.profile.min_mem_gib)
    }

    fn exclusive_key(&self) -> Option<String> {
        Some(self.profile.device.clone())
    }

    fn allocate_port(&self, avoid: &BTreeSet<u16>) -> Option<u16> {
        let port = self.profile.port;
        if avoid.contains(&port) {
            return None;
        }
        Some(port)
    }

    fn readiness_timeout(&self) -> Option<Duration> {
        Some(Duration::from_millis(
            engine_process::env_u64("BRAINROUTER_READINESS_TIMEOUT_MS")
                .unwrap_or(30_000),
        ))
    }

    fn takeover_unit(&self) -> Option<String> {
        self.profile.unit.clone()
    }

    fn describe(&self) -> Option<serde_json::Value> {
        self.maybe_refresh_schema(Duration::from_secs(30));
        let schema = self.schema();
        let (cards, warnings) = self
            .profile
            .models_dir
            .as_deref()
            .map(scan_cards)
            .unwrap_or_default();
        Some(serde_json::json!({
            "engine": ENGINE,
            "program": self.profile.program,
            "port": self.profile.port,
            "min_mem_gib": self.profile.min_mem_gib,
            "device": self.profile.device,
            "unit": self.profile.unit,
            "max_seq": self.profile.max_seq,
            "schema": schema.describe(),
            "schema_drift": self.schema_drift(),
            "models_dir": self.profile.models_dir.as_ref().map(|d| d.display().to_string()),
            "cards": cards,
            "warnings": warnings,
            "scoped": self.use_scope,
        }))
    }
}

/// The engine's context window for a catalog entry (entry `ctx` wins).
pub fn context_window(entry: &crate::swap_catalog::CatalogEntry, profile: &HipfireProfile) -> u64 {
    entry.ctx.unwrap_or(profile.max_seq)
}

#[derive(Debug, Clone, Default)]
pub struct HipfireRuntimeState {
    pub cards: Vec<HipfireCard>,
    pub warnings: Vec<String>,
    pub version: Option<String>,
    pub schema: Option<HipfireSchema>,
    pub schema_drift: bool,
}

/// Snapshot for `/api/engines` (adapter states + card/schema badges).
pub fn runtime_state(adapter: &HipfireAdapter) -> HipfireRuntimeState {
    let (cards, warnings) = adapter
        .profile
        .models_dir
        .as_deref()
        .map(scan_cards)
        .unwrap_or_default();
    HipfireRuntimeState {
        cards,
        warnings,
        version: adapter
            .derived_from
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone(),
        schema: Some(adapter.schema()),
        schema_drift: adapter.schema_drift(),
    }
}

/// Extra launch args map (unused placeholder to keep the shape explicit for
/// future schema-driven flags; never forwards unknown keys).
pub fn args_from_map(map: &BTreeMap<String, String>) -> Vec<String> {
    map.iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(dir: &Path, stem: &str, sidecar: bool) {
        std::fs::write(dir.join(format!("{stem}.mq4")), b"MQ4").unwrap();
        if sidecar {
            std::fs::write(dir.join(format!("{stem}.mtp")), b"MTP").unwrap();
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "brainrouter-hipfire-{name}-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn cards_scan_sidecar_and_warn_when_missing() {
        let dir = temp_dir("cards");
        card(&dir, "swift", true);
        card(&dir, "muse", false);
        std::fs::write(dir.join("README.txt"), b"x").unwrap();
        let (cards, warnings) = scan_cards(&dir);
        assert_eq!(cards.len(), 2);
        assert_eq!(cards[0].stem, "muse");
        assert!(cards[0].mtp.is_none());
        assert_eq!(cards[1].mtp.as_ref().unwrap().file_name().unwrap(), "swift.mtp");
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("muse"));
    }

    #[test]
    fn schema_is_version_keyed_and_rejects_mtp_by_construction() {
        let v0 = HipfireSchema::for_version("0.9.0");
        let v1 = HipfireSchema::for_version("1.4.2");
        assert!(!v0.allowed_flags.contains("--flash-attn"));
        assert!(v1.allowed_flags.contains("--flash-attn"));
        v1.validate_args(&["--ctx".into(), "8192".into(), "--flash-attn".into()])
            .unwrap();
        for bad in [vec!["--mtp".to_string()], vec!["--mtp=true".to_string()], vec!["--wat".to_string()]] {
            assert!(v1.validate_args(&bad).is_err(), "{bad:?}");
        }
        // An upgrade re-derives: the same argv that failed under v0 is legal
        // under v1 (and the caller badges the drift).
        assert!(v0.validate_args(&["--flash-attn".into()]).is_err());
    }

    #[test]
    fn version_bump_rederives_and_badges() {
        let profile = HipfireProfile {
            program: "/nonexistent/hipfire".into(),
            models_dir: None,
            port: DEFAULT_PORT,
            min_mem_gib: 16.0,
            device: "hipfire:gfx1100".into(),
            unit: Some(DEFAULT_UNIT.into()),
            max_seq: 8192,
            port_free_timeout: Duration::from_millis(100),
        };
        let adapter = HipfireAdapter::new(profile);
        assert!(
            adapter.set_schema_version("1.0.0"),
            "deriving from the unknown marker counts as a change"
        );
        assert!(!adapter.set_schema_version("1.0.0"), "same version is stable");
        assert!(adapter.schema().allowed_flags.contains("--flash-attn"));
        assert!(adapter.set_schema_version("2.0.0"), "version change = drift");
        assert!(adapter.schema_drift());
        assert_eq!(adapter.schema().version, "2.0.0");
    }

    #[test]
    fn context_math_defaults_and_reports_shortfall() {
        assert_eq!(remaining_context(8192, 1000), 7192);
        assert_eq!(default_max_tokens(8192, 1000).unwrap(), 7192);
        assert_eq!(shape_max_tokens(None, 8192, 1000).unwrap(), 7192);
        assert_eq!(shape_max_tokens(Some(4000), 8192, 1000).unwrap(), 4000);
        assert_eq!(
            shape_max_tokens(Some(90_000), 8192, 1000).unwrap(),
            7192,
            "client overshoot clamps to the remaining context"
        );
        let shortfall = default_max_tokens(8192, 8192).unwrap_err();
        assert_eq!(shortfall.ctx, 8192);
        assert_eq!(shortfall.prompt_tokens, 8192);
        assert!(shortfall.to_string().contains("shortfall"));
    }

    #[test]
    fn squat_detection_names_the_foreign_stems() {
        let expected = ["swift", "model-x"];
        assert!(squat_reason(&["swift".into()], &expected).is_none());
        assert!(squat_reason(&[], &expected).is_none());
        let reason = squat_reason(&["someone-else".into()], &expected).unwrap();
        assert!(reason.contains("someone-else"), "{reason}");
        assert!(reason.contains("foreign"), "{reason}");
    }
}
