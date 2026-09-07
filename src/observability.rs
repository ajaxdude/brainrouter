//! Bounded, process-local model observations. No request history is persisted.
//!
//! Live trends use disjoint time windows from the same measurement source.
//! Benchmark references are operator-selected, never inferred from model names.

use crate::benchmark::{BenchmarkError, BenchmarkStore};
use crate::inflight::InflightRow;
use crate::routing_events::{RouteEvent, Stage};
use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures_util::{stream::FuturesUnordered, StreamExt};
use http_body_util::{combinators::UnsyncBoxBody, BodyExt, Full, Limited};
use hyper::{body::Incoming, Request, Response, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Semaphore;

const HTML: &str = include_str!("escalation/templates/model_observability.html");
const SETTINGS_LIMIT: usize = 128 * 1024;
const BACKEND_LIMIT: usize = 1024 * 1024;
const POLL_SECONDS: u64 = 5;
const MODEL_LIMIT: usize = 256;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AlertPolicy {
    pub window_seconds: i64,
    pub min_samples: usize,
    pub generation_drop_percent: f64,
    pub ttft_rise_percent: f64,
    pub rate_increase_points: f64,
    pub repeated_count: usize,
    pub debounce_samples: u32,
    pub recovery_samples: u32,
    pub baseline_max_age_days: i64,
}

impl Default for AlertPolicy {
    fn default() -> Self {
        Self {
            window_seconds: 300,
            min_samples: 5,
            generation_drop_percent: 20.0,
            ttft_rise_percent: 30.0,
            rate_increase_points: 20.0,
            repeated_count: 3,
            debounce_samples: 2,
            recovery_samples: 2,
            baseline_max_age_days: 30,
        }
    }
}

impl AlertPolicy {
    fn validate(&self) -> Result<(), String> {
        if !(30..=3600).contains(&self.window_seconds)
            || !(3..=100).contains(&self.min_samples)
            || !(2..=100).contains(&self.repeated_count)
            || !(1..=10).contains(&self.debounce_samples)
            || !(1..=10).contains(&self.recovery_samples)
            || !(1..=3650).contains(&self.baseline_max_age_days)
            || !self.generation_drop_percent.is_finite()
            || !(5.0..=95.0).contains(&self.generation_drop_percent)
            || !self.ttft_rise_percent.is_finite()
            || !(5.0..=500.0).contains(&self.ttft_rise_percent)
            || !self.rate_increase_points.is_finite()
            || !(1.0..=100.0).contains(&self.rate_increase_points)
        {
            return Err("Invalid policy: window 30..3600s, samples 3..100, repeats 2..100, debounce/recovery 1..10, generation drop 5..95%, TTFT rise 5..500%, rate increase 1..100 points, baseline age 1..3650 days".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Baseline {
    run_id: String,
    experiment_hash: String,
    configuration: Value,
    performance_metrics: Value,
    ended_at: DateTime<Utc>,
    selected_at: DateTime<Utc>,
    note: String,
}

#[derive(Clone, Default, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    policy: AlertPolicy,
    baselines: BTreeMap<String, Baseline>,
}

impl Settings {
    fn validate(&self) -> Result<(), String> {
        self.policy.validate()?;
        if self.baselines.len() > 32 {
            return Err("At most 32 explicit baseline mappings are supported".into());
        }
        for (model, baseline) in &self.baselines {
            validate_text(model, "model_key", 256)?;
            validate_text(&baseline.run_id, "run_id", 256)?;
            validate_text(&baseline.note, "note", 1000)?;
            validate_configuration(&baseline.configuration)?;
            if baseline.configuration["experiment"]["experiment_hash"].as_str()
                != Some(&baseline.experiment_hash)
                || (positive(baseline.performance_metrics["generation_tps"].as_f64()).is_none()
                    && positive(baseline.performance_metrics["ttft_ms"].as_f64()).is_none())
            {
                return Err("Stored reference identity or measurements are invalid".into());
            }
        }
        Ok(())
    }
}

struct SettingsState {
    settings: Settings,
    revision: u64,
    read_error: Option<String>,
    write_error: Option<String>,
}

#[derive(Clone, Default, Serialize)]
struct BackendModel {
    model_key: String,
    state: String,
    reported_state: Option<String>,
    quantization: Option<String>,
    artifact_path: Option<String>,
    runtime_build: Option<Value>,
    context_tokens: Option<u64>,
    metadata_error: Option<String>,
}

#[derive(Clone, Serialize)]
struct BackendSnapshot {
    sampled_at: Option<DateTime<Utc>>,
    running_error: Option<String>,
    catalog_error: Option<String>,
    models: BTreeMap<String, BackendModel>,
}

impl Default for BackendSnapshot {
    fn default() -> Self {
        Self {
            sampled_at: None,
            running_error: Some("Awaiting first backend status poll".into()),
            catalog_error: None,
            models: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug)]
struct Measurement {
    id: u64,
    at: DateTime<Utc>,
    model: String,
    generation_tps: Option<f64>,
    ttft_ms: Option<f64>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    stream_duration_ms: Option<f64>,
}

#[derive(Default)]
struct Latch {
    seen: BTreeSet<u64>,
    bad: u32,
    good: u32,
    active: bool,
    recovered: bool,
}

impl Latch {
    fn evaluate(
        &mut self,
        condition: Option<bool>,
        evidence: &[u64],
        policy: &AlertPolicy,
    ) -> &'static str {
        let Some(bad) = condition else {
            *self = Self::default();
            return "insufficient_data";
        };
        // Evidence itself is bounded by the source ring. Keep exactly this set:
        // event IDs are creation-ordered, not completion-ordered, so removing
        // the smallest ID could discard a late completion still being polled.
        let current: BTreeSet<_> = evidence.iter().copied().collect();
        let has_new_evidence = !current.is_subset(&self.seen);
        self.seen = current;
        if has_new_evidence {
            if bad {
                self.good = 0;
                self.bad += 1;
                self.recovered = false;
                if self.bad >= policy.debounce_samples {
                    self.active = true;
                }
            } else {
                self.bad = 0;
                self.good += 1;
                if self.active && self.good >= policy.recovery_samples {
                    self.active = false;
                    self.recovered = true;
                }
            }
        }
        if self.active {
            if self.good > 0 {
                "recovering"
            } else {
                "active"
            }
        } else if self.bad > 0 {
            "pending"
        } else if self.recovered {
            "recovered"
        } else {
            "ok"
        }
    }
}

pub struct Observability {
    path: PathBuf,
    settings: Arc<Mutex<SettingsState>>,
    writer: Arc<Semaphore>,
    snapshot: Mutex<Value>,
}

impl Observability {
    /// Settings failures are visible but never prevent daemon startup.
    pub fn new(config_path: &Path) -> Self {
        let path = config_path.with_extension("observability.json");
        let loaded = load_settings(&path);
        let (settings, read_error) = match loaded {
            Ok(settings) => (settings, None),
            Err(error) => {
                tracing::warn!(%error, path = %path.display(), "Model observation settings unavailable");
                (Settings::default(), Some(error))
            }
        };
        Self {
            path,
            settings: Arc::new(Mutex::new(SettingsState {
                settings,
                revision: 0,
                read_error,
                write_error: None,
            })),
            writer: Arc::new(Semaphore::new(1)),
            snapshot: Mutex::new(
                json!({"sampled_at": null, "models": [], "status": "Awaiting first observation", "alerts": []}),
            ),
        }
    }

    fn settings_response(&self) -> Value {
        let state = self.settings.lock().unwrap();
        json!({
            "settings": state.settings, "revision": state.revision,
            "read_error": state.read_error, "write_error": state.write_error,
            "path": self.path.display().to_string(), "scope": "Only operator baseline mappings and alert policy persist."
        })
    }

    async fn save(&self, revision: u64, settings: Settings) -> Result<Value, ApiError> {
        settings.validate().map_err(ApiError::bad_request)?;
        let permit = Arc::clone(&self.writer).try_acquire_owned().map_err(|_| {
            ApiError(
                StatusCode::SERVICE_UNAVAILABLE,
                "Settings writer busy; retry".into(),
            )
        })?;
        let path = self.path.clone();
        let state = Arc::clone(&self.settings);
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            {
                let current = state.lock().unwrap();
                if let Some(error) = &current.read_error {
                    return Err(ApiError(
                        StatusCode::SERVICE_UNAVAILABLE,
                        format!("Repair the settings file and restart before saving: {error}"),
                    ));
                }
                if current.revision != revision {
                    return Err(ApiError(
                        StatusCode::CONFLICT,
                        "Settings changed; reload before saving".into(),
                    ));
                }
            }
            let bytes = serde_json::to_vec_pretty(&settings).map_err(ApiError::internal)?;
            if bytes.len() > SETTINGS_LIMIT {
                return Err(ApiError(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "Settings exceed 128 KiB".into(),
                ));
            }
            let result = atomic_write(&path, &bytes);
            let mut current = state.lock().unwrap();
            match result {
                Ok(durability) => {
                    // Rename committed the new file; keep memory consistent even if
                    // the directory fsync reports uncertain crash durability.
                    current.settings = settings;
                    current.revision += 1;
                    current.write_error = durability.err().map(|error| error.to_string());
                    if let Some(error) = &current.write_error {
                        return Err(ApiError::internal(format!(
                            "Settings replaced, but durability is uncertain: {error}"
                        )));
                    }
                    Ok(json!({"saved": true, "revision": current.revision}))
                }
                Err(error) => {
                    current.write_error = Some(error.to_string());
                    Err(ApiError::internal(format!("Settings not saved: {error}")))
                }
            }
        })
        .await
        .map_err(ApiError::internal)?
    }
}

fn load_settings(path: &Path) -> Result<Settings, String> {
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Settings::default())
        }
        Err(error) => return Err(error.to_string()),
    };
    let mut bytes = Vec::new();
    file.take((SETTINGS_LIMIT + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() > SETTINGS_LIMIT {
        return Err("Settings exceed 128 KiB".into());
    }
    let settings: Settings = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    settings.validate()?;
    Ok(settings)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<std::io::Result<()>> {
    let parent = path
        .parent()
        .ok_or_else(|| std::io::Error::other("Settings path has no parent"))?;
    let temp = parent.join(format!(".observability-{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&temp, path)?;
        Ok(std::fs::File::open(parent).and_then(|directory| directory.sync_all()))
    })();
    if result.is_err() {
        if let Err(error) = std::fs::remove_file(&temp) {
            if error.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(%error, path = %temp.display(), "Cannot remove failed settings temporary file");
            }
        }
    }
    result
}

fn validate_text(value: &str, name: &str, max: usize) -> Result<(), String> {
    if value.trim().is_empty() || value.len() > max || value.chars().any(char::is_control) {
        return Err(format!(
            "{name} must be nonempty, at most {max} bytes, without control characters"
        ));
    }
    Ok(())
}

async fn fetch_json(client: &reqwest::Client, url: &str) -> Result<Value, String> {
    let mut response = client
        .get(url)
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| error.to_string())?;
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|error| error.to_string())? {
        if bytes.len() + chunk.len() > BACKEND_LIMIT {
            return Err("Backend status response exceeds 1 MiB".into());
        }
        bytes.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&bytes).map_err(|error| format!("Invalid backend status JSON: {error}"))
}

fn parse_catalog(value: &Value) -> Result<BTreeMap<String, BackendModel>, String> {
    let rows = value["data"]
        .as_array()
        .ok_or("Missing models data array")?;
    if rows.len() > MODEL_LIMIT {
        return Err("Model catalog exceeds 256 models".into());
    }
    let mut models = BTreeMap::new();
    for row in rows {
        let key = row["id"].as_str().ok_or("Model catalog entry has no id")?;
        validate_text(key, "backend model id", 256)?;
        models.insert(
            key.to_owned(),
            BackendModel {
                model_key: key.to_owned(),
                state: "idle".into(),
                quantization: row["quantization"].as_str().map(str::to_owned),
                ..BackendModel::default()
            },
        );
    }
    Ok(models)
}

fn apply_running(models: &mut BTreeMap<String, BackendModel>, value: &Value) -> Result<(), String> {
    let rows = value["running"].as_array().ok_or("Missing running array")?;
    if rows.len() > MODEL_LIMIT {
        return Err("Running status exceeds 256 models".into());
    }
    for row in rows {
        let key = row["model"]
            .as_str()
            .ok_or("Running entry has no model key")?;
        validate_text(key, "running model key", 256)?;
        let model = models
            .entry(key.to_owned())
            .or_insert_with(|| BackendModel {
                model_key: key.to_owned(),
                ..BackendModel::default()
            });
        model.reported_state = row["state"].as_str().map(str::to_owned);
        model.state = match model.reported_state.as_deref() {
            Some("ready") => "loaded",
            Some("starting" | "loading" | "loading_model") => "loading",
            Some("stopping" | "stopped") => "stopping",
            _ => "unknown",
        }
        .into();
    }
    Ok(())
}

fn allowed_proxy(proxy: &str, backend: &str) -> bool {
    let (Ok(proxy), Ok(backend)) = (url::Url::parse(proxy), url::Url::parse(backend)) else {
        return false;
    };
    if !matches!(proxy.scheme(), "http" | "https")
        || !proxy.username().is_empty()
        || proxy.password().is_some()
    {
        return false;
    }
    proxy.host() == backend.host()
        || matches!(proxy.host(),
        Some(url::Host::Ipv4(ip)) if ip.is_loopback())
        || matches!(proxy.host(),
        Some(url::Host::Ipv6(ip)) if ip.is_loopback())
}

async fn poll_backend(client: &reqwest::Client, url: &str) -> BackendSnapshot {
    let catalog_url = format!("{url}/v1/models");
    let running_url = format!("{url}/running");
    let (catalog, running) = tokio::join!(
        fetch_json(client, &catalog_url),
        fetch_json(client, &running_url)
    );
    let mut snapshot = BackendSnapshot {
        sampled_at: Some(Utc::now()),
        ..BackendSnapshot::default()
    };
    match catalog.and_then(|value| parse_catalog(&value)) {
        Ok(models) => snapshot.models = models,
        Err(error) => snapshot.catalog_error = Some(error),
    }
    match running {
        Ok(value) => {
            snapshot.running_error = apply_running(&mut snapshot.models, &value).err();
            // /props is read only and is queried only for already-ready proxies.
            // Limit fanout, and never follow arbitrary backend-provided URLs.
            if snapshot.running_error.is_none() {
                let rows = value["running"]
                    .as_array()
                    .expect("validated running array");
                let mut pending = FuturesUnordered::new();
                for row in rows.iter().filter(|row| row["state"] == "ready").take(4) {
                    let Some(key) = row["model"].as_str() else {
                        continue;
                    };
                    let Some(proxy) = row["proxy"].as_str() else {
                        continue;
                    };
                    let model = snapshot
                        .models
                        .get_mut(key)
                        .expect("validated running model");
                    if !allowed_proxy(proxy, url) {
                        model.metadata_error = Some(
                            "Proxy metadata URL is outside the configured backend host/loopback"
                                .into(),
                        );
                        continue;
                    }
                    pending.push(async move {
                        (
                            key.to_owned(),
                            fetch_json(client, &format!("{}/props", proxy.trim_end_matches('/')))
                                .await,
                        )
                    });
                }
                while let Some((key, result)) = pending.next().await {
                    let model = snapshot
                        .models
                        .get_mut(&key)
                        .expect("validated running model");
                    match result {
                        Ok(props) => {
                            model.artifact_path = props["model_path"]
                                .as_str()
                                .filter(|path| path.len() <= 4096)
                                .map(str::to_owned);
                            model.runtime_build = props
                                .get("build_info")
                                .filter(|value| !value.is_null() && value.to_string().len() <= 4096)
                                .cloned();
                            model.context_tokens = props["default_generation_settings"]["n_ctx"]
                                .as_u64()
                                .filter(|n| *n > 0);
                            model.quantization = props["quantization"]
                                .as_str()
                                .map(str::to_owned)
                                .or(model.quantization.take());
                        }
                        Err(error) => model.metadata_error = Some(error),
                    }
                }
            }
        }
        Err(error) => snapshot.running_error = Some(error),
    }
    if snapshot.running_error.is_some() {
        for model in snapshot.models.values_mut() {
            model.state = "unreachable".into();
        }
    }
    snapshot
}

fn positive(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite() && *value > 0.0)
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        values[middle - 1] / 2.0 + values[middle] / 2.0
    } else {
        values[middle]
    })
}

fn window(at: DateTime<Utc>, now: DateTime<Utc>, seconds: i64) -> Option<usize> {
    let age = now.signed_duration_since(at).num_milliseconds();
    if age < 0 || age >= seconds * 2000 {
        None
    } else if age < seconds * 1000 {
        Some(1)
    } else {
        Some(0)
    }
}

fn performance_alert(
    name: &str,
    samples: &[&Measurement],
    now: DateTime<Utc>,
    policy: &AlertPolicy,
    latch: &mut Latch,
) -> Value {
    let mut values = [Vec::new(), Vec::new()];
    let mut evidence = Vec::new();
    for sample in samples {
        let value = if name == "generation" {
            sample.generation_tps
        } else {
            sample.ttft_ms
        };
        if let (Some(value), Some(index)) = (
            positive(value),
            window(sample.at, now, policy.window_seconds),
        ) {
            values[index].push(value);
            if index == 1 {
                evidence.push(sample.id);
            }
        }
    }
    let counts = [values[0].len(), values[1].len()];
    let reference = median(&mut values[0]);
    let current = median(&mut values[1]);
    let percent = reference
        .zip(current)
        .map(|(reference, current)| (current / reference - 1.0) * 100.0);
    let condition = if counts.iter().all(|count| *count >= policy.min_samples) {
        percent.map(|percent| {
            if name == "generation" {
                percent <= -policy.generation_drop_percent
            } else {
                percent >= policy.ttft_rise_percent
            }
        })
    } else {
        None
    };
    json!({
        "kind": name, "status": latch.evaluate(condition, &evidence, policy),
        "reference_count": counts[0], "current_count": counts[1],
        "reference": reference, "current": current, "change_percent": percent,
        "unit": if name == "generation" { "tokens/s" } else { "ms" },
        "scope": "Indicative same-model live trend, NOT a controlled benchmark regression. Artifact, runtime, hardware, context, workload and concurrency are not verified per sample."
    })
}

fn error_evidence(event: &RouteEvent, name: &str) -> bool {
    if name == "fallback" {
        return matches!(event.stage, Stage::CloudFallback | Stage::LocalFallback);
    }
    if event.success {
        return false;
    }
    let error = event.error.to_ascii_lowercase();
    match name {
        "error" => true,
        "oom" => {
            error.contains("out of memory")
                || error.contains("out_of_memory")
                || error
                    .split(|c: char| !c.is_ascii_alphanumeric())
                    .any(|word| word == "oom")
        }
        "timeout" => {
            error.contains("timed out")
                || error
                    .split(|c: char| !c.is_ascii_alphanumeric())
                    .any(|word| word == "timeout")
        }
        _ => false,
    }
}

fn rate_alert(
    name: &str,
    events: &[&RouteEvent],
    now: DateTime<Utc>,
    policy: &AlertPolicy,
    latch: &mut Latch,
) -> Value {
    let mut totals = [0usize; 2];
    let mut counts = [0usize; 2];
    let mut evidence = Vec::new();
    let mut ids = Vec::new();
    for event in events {
        let Ok(at) = DateTime::parse_from_rfc3339(&event.timestamp) else {
            continue;
        };
        let Some(index) = window(at.with_timezone(&Utc), now, policy.window_seconds) else {
            continue;
        };
        totals[index] += 1;
        if index == 1 {
            evidence.push(event.id);
        }
        if error_evidence(event, name) {
            counts[index] += 1;
            if index == 1 && ids.len() < 10 {
                ids.push(event.id);
            }
        }
    }
    let rates = [0, 1].map(|index| {
        (totals[index] > 0).then(|| counts[index] as f64 * 100.0 / totals[index] as f64)
    });
    let increase = rates[0].zip(rates[1]).map(|(old, new)| new - old);
    let condition = if totals.iter().all(|count| *count >= policy.min_samples) {
        increase.map(|increase| {
            counts[1] >= policy.repeated_count && increase >= policy.rate_increase_points
        })
    } else {
        None
    };
    json!({
        "kind": name, "status": latch.evaluate(condition, &evidence, policy),
        "reference_count": totals[0], "current_count": totals[1],
        "reference_occurrences": counts[0], "current_occurrences": counts[1],
        "reference": rates[0], "current": rates[1], "increase_points": increase,
        "event_ids": ids, "unit": "%",
        "scope": if name == "fallback" {
            "Fallback-served routing stages, not failures attributed to the selected fallback model."
        } else {
            "Router-result errors only; stream errors may not appear in RouteEvent. OOM/timeout categories require explicit error text, not a hardware diagnosis."
        }
    })
}

fn build_snapshot(
    backend: &BackendSnapshot,
    events: &[RouteEvent],
    measurements: &[Measurement],
    inflight: &[InflightRow],
    settings: &Settings,
    now: DateTime<Utc>,
    latches: &mut BTreeMap<(String, String), Latch>,
) -> Value {
    let mut keys: BTreeSet<String> = backend
        .models
        .keys()
        .chain(settings.baselines.keys())
        .cloned()
        .collect();
    for event in events.iter().filter(|event| {
        event.effective_provider.as_deref() == Some("llama-swap") && !event.model_key.is_empty()
    }) {
        keys.insert(event.model_key.clone());
    }
    keys.extend(measurements.iter().map(|sample| sample.model.clone()));
    let mut models = Vec::new();
    for key in &keys {
        let recent: Vec<_> = events
            .iter()
            .filter(|event| {
                event.model_key == *key && event.effective_provider.as_deref() == Some("llama-swap")
            })
            .collect();
        let samples: Vec<_> = measurements
            .iter()
            .filter(|sample| sample.model == *key)
            .collect();
        let mut sample_rows: Vec<_> = samples
            .iter()
            .filter(|sample| window(sample.at, now, settings.policy.window_seconds).is_some())
            .collect();
        sample_rows.sort_by(|left, right| right.at.cmp(&left.at).then(right.id.cmp(&left.id)));
        let sample_rows: Vec<_> = sample_rows.into_iter().take(25).map(|sample| json!({
            "event_id": sample.id, "completed_at": sample.at,
            "measured_ttft_ms": sample.ttft_ms, "generation_tps": sample.generation_tps,
            "prompt_tokens": sample.prompt_tokens, "completion_tokens": sample.completion_tokens,
            "stream_duration_ms": sample.stream_duration_ms
        })).collect();
        let active: Vec<_> = inflight
            .iter()
            .filter(|row| row.model == *key)
            .map(|row| {
                json!({
                    "id": row.id, "activity": row.activity, "elapsed_ms": row.elapsed_ms,
                    "bytes_received": row.bytes_received,
                    "pp_progress": if row.pp_progress > 0.0 { Some(row.pp_progress) } else { None }
                })
            })
            .collect();
        let mut alerts = Vec::new();
        for kind in ["generation", "ttft"] {
            alerts.push(performance_alert(
                kind,
                &samples,
                now,
                &settings.policy,
                latches.entry((key.clone(), kind.into())).or_default(),
            ));
        }
        for kind in ["error", "oom", "timeout", "fallback"] {
            alerts.push(rate_alert(
                kind,
                &recent,
                now,
                &settings.policy,
                latches.entry((key.clone(), kind.into())).or_default(),
            ));
        }
        let state = backend
            .models
            .get(key)
            .map(|model| model.state.as_str())
            .unwrap_or(if backend.running_error.is_some() {
                "unreachable"
            } else {
                "unknown"
            });
        let recent_rows: Vec<_> = recent.iter()
            .filter(|event| DateTime::parse_from_rfc3339(&event.timestamp).ok()
                .and_then(|at| window(at.with_timezone(&Utc), now, settings.policy.window_seconds)).is_some())
            .take(25).map(|event| json!({
                "id": event.id, "timestamp": event.timestamp, "router_latency_ms": event.latency_ms,
                "router_success": event.success, "error": event.error.chars().take(1000).collect::<String>(), "stage": event.stage
            })).collect();
        models.push(json!({
            "model_key": key, "provider": "llama-swap",
            "state": if !active.is_empty() && state == "loaded" { "running" } else { state },
            "backend": backend.models.get(key), "active_requests": active,
            "active_request_scope": "Exact registry model labels only; unresolved aliases are shown separately.",
            "memory": { "rss_bytes": null, "vram_bytes": null, "source": "Not reported by available backend status APIs; model file size is not resident memory." },
            "live_generation_tps": null, "live_tps_reason": "No reliable live token counter. Recent rates use completed stream measurements.",
            "alerts": alerts, "recent_events": recent_rows, "recent_measurements": sample_rows,
            "baseline_run_id": settings.baselines.get(key).map(|baseline| &baseline.run_id)
        }));
    }
    latches.retain(|(model, _), _| keys.contains(model) || model == "\0unattributed");
    let unassigned: Vec<_> = events
        .iter()
        .filter(|event| {
            event.model_key.is_empty() || event.effective_provider.as_deref() != Some("llama-swap")
        })
        .collect();
    let unresolved: Vec<_> = inflight.iter().filter(|row| !keys.contains(&row.model)).map(|row| json!({
        "id": row.id, "model_label": row.model, "activity": row.activity, "elapsed_ms": row.elapsed_ms
    })).collect();
    let unattributed_alerts: Vec<_> = ["error", "oom", "timeout", "fallback"]
        .into_iter()
        .map(|kind| {
            rate_alert(
                kind,
                &unassigned,
                now,
                &settings.policy,
                latches
                    .entry(("\0unattributed".into(), kind.into()))
                    .or_default(),
            )
        })
        .collect();
    json!({
        "sampled_at": now, "backend": backend, "models": models, "unresolved_requests": unresolved,
        "unattributed_route_count": unassigned.len(),
        "unattributed_error_count": unassigned.iter().filter(|event| !event.success).count(),
        "unattributed_alerts": unattributed_alerts,
        "policy": settings.policy,
        "retention": {
            "max_route_events": 500, "max_completed_measurements": 500,
            "window_seconds": settings.policy.window_seconds,
            "reference_from": now - chrono::Duration::seconds(2 * settings.policy.window_seconds),
            "current_from": now - chrono::Duration::seconds(settings.policy.window_seconds),
            "until": now,
            "limits": "Two disjoint time windows, at most 500 routes and 500 completed measurements globally. Memory only; restart resets samples and alert debounce. Busy traffic can evict reference samples early. Policy changes reset debounce; new evidence, not repeated polls, advances it."
        }
    })
}

fn completed_measurements(events: &crate::routing_events::RoutingEvents) -> Vec<Measurement> {
    events
        .get_measurements()
        .into_iter()
        .filter_map(|sample| {
            if sample.effective_provider.as_deref() != Some("llama-swap")
                || sample.model_key.is_empty()
            {
                return None;
            }
            Some(Measurement {
                id: sample.event_id,
                at: DateTime::parse_from_rfc3339(&sample.completed_at)
                    .ok()?
                    .with_timezone(&Utc),
                model: sample.model_key,
                generation_tps: sample.generation_tps,
                ttft_ms: sample.measured_ttft_ms,
                prompt_tokens: sample.prompt_tokens,
                completion_tokens: sample.completion_tokens,
                stream_duration_ms: Some(sample.stream_duration_ms),
            })
        })
        .collect()
}

/// Start read-only status polling independently of routing and benchmark storage.
pub fn start(state: &Arc<crate::server::AppState>) {
    let state = Arc::clone(state);
    tokio::spawn(async move {
        let client = match reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .redirect(reqwest::redirect::Policy::none())
            .build()
        {
            Ok(client) => client,
            Err(error) => {
                tracing::error!(%error, "Cannot initialize model observability client");
                *state.observability.snapshot.lock().unwrap() =
                    json!({"models": [], "error": error.to_string()});
                return;
            }
        };
        let mut latches = BTreeMap::new();
        let mut revision = 0;
        loop {
            let backend = poll_backend(&client, &state.llama_swap_url).await;
            let (settings, next_revision) = {
                let current = state.observability.settings.lock().unwrap();
                (current.settings.clone(), current.revision)
            };
            if revision != next_revision {
                latches.clear();
                revision = next_revision;
            }
            let events = state.routing_events.get_all();
            let measurements = completed_measurements(&state.routing_events);
            let mut snapshot = build_snapshot(
                &backend,
                &events,
                &measurements,
                &state.inflight.snapshot(),
                &settings,
                Utc::now(),
                &mut latches,
            );
            snapshot["latest_inference_phase"] =
                serde_json::to_value(state.router.inference_tracker.snapshot())
                    .expect("serializable inference snapshot");
            snapshot["inference_scope"] =
                json!("Single most recent inference tracker, not a complete request registry.");
            snapshot["measurement_definitions"] =
                json!(crate::routing_events::STREAM_MEASUREMENT_DEFINITION);
            snapshot["service_versions"] = state.versions_cache.borrow().clone();
            *state.observability.snapshot.lock().unwrap() = snapshot;
            tokio::time::sleep(Duration::from_secs(POLL_SECONDS)).await;
        }
    });
}

fn validate_configuration(configuration: &Value) -> Result<(), String> {
    for (section, field) in [
        ("model", "id"),
        ("artifact", "sha256"),
        ("runtime", "id"),
        ("runtime", "commit_sha"),
        ("hardware", "id"),
        ("workload", "id"),
        ("experiment", "experiment_hash"),
    ] {
        let value = configuration[section][field]
            .as_str()
            .ok_or_else(|| format!("Reference missing {section}.{field}"))?;
        validate_text(value, &format!("{section}.{field}"), 1024)?;
    }
    if configuration["experiment"]["context_tokens"]
        .as_u64()
        .filter(|value| *value > 0)
        .is_none()
    {
        return Err("Reference missing positive context_tokens".into());
    }
    Ok(())
}

fn reference_from_detail(
    detail: &Value,
    note: String,
    now: DateTime<Utc>,
) -> Result<Baseline, String> {
    if detail["run_record"]["status"] != "succeeded" {
        return Err("Baseline must be a succeeded run".into());
    }
    let configuration = &detail["configuration"];
    validate_configuration(configuration)?;
    let ended_at = detail["run_record"]["ended_at"]
        .as_str()
        .and_then(|at| DateTime::parse_from_rfc3339(at).ok())
        .map(|at| at.with_timezone(&Utc))
        .ok_or("Baseline must have a completion timestamp")?;
    if ended_at > now {
        return Err("Baseline completion timestamp is in the future".into());
    }
    let metrics = &detail["performance_metrics"];
    if positive(metrics["generation_tps"].as_f64()).is_none()
        && positive(metrics["ttft_ms"].as_f64()).is_none()
    {
        return Err("Baseline requires a reported generation or TTFT measurement".into());
    }
    let run_id = detail["run_record"]["id"]
        .as_str()
        .ok_or("Baseline missing run id")?
        .to_owned();
    Ok(Baseline {
        run_id,
        experiment_hash: configuration["experiment"]["experiment_hash"]
            .as_str()
            .unwrap()
            .to_owned(),
        configuration: configuration.clone(),
        performance_metrics: metrics.clone(),
        ended_at,
        selected_at: now,
        note,
    })
}

fn baseline_comparison(
    saved: &Baseline,
    current: &Baseline,
    now: DateTime<Utc>,
    policy: &AlertPolicy,
) -> Value {
    let status = if saved.configuration != current.configuration
        || saved.performance_metrics != current.performance_metrics
        || saved.ended_at != current.ended_at
        || saved.run_id != current.run_id
    {
        "mismatched"
    } else if now.signed_duration_since(current.ended_at).num_seconds()
        > policy.baseline_max_age_days * 86400
    {
        "stale"
    } else {
        "not_comparable"
    };
    json!({
        "status": status,
        "reference": saved,
        "verified_at": now,
        "sample_count": 1,
        "reason": "Explicit operator reference, not proof of live configuration. Live samples do not attest artifact SHA, runtime build, hardware profile, context length, workload, concurrency, or benchmark-equivalent measurement definitions.",
        "missing_dimensions": ["artifact SHA-256", "runtime/build", "hardware profile", "context/prompt/generation lengths", "workload/sampling", "concurrency", "equivalent metric definitions"],
        "controlled_generation_regression": null, "controlled_ttft_regression": null
    })
}

#[derive(Debug)]
struct ApiError(StatusCode, String);
impl ApiError {
    fn bad_request(message: impl ToString) -> Self {
        Self(StatusCode::BAD_REQUEST, message.to_string())
    }
    fn internal(message: impl ToString) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, message.to_string())
    }
}

fn response(status: StatusCode, data: Value) -> Response<UnsyncBoxBody<Bytes, anyhow::Error>> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .header("cache-control", "no-store")
        .body(
            Full::new(Bytes::from(data.to_string()))
                .map_err(|error: Infallible| match error {})
                .boxed_unsync(),
        )
        .expect("valid observation response")
}

async fn read_body<T: for<'de> Deserialize<'de>>(req: Request<Incoming>) -> Result<T, ApiError> {
    let bytes = tokio::time::timeout(
        Duration::from_secs(5),
        Limited::new(req.into_body(), SETTINGS_LIMIT).collect(),
    )
    .await
    .map_err(|_| ApiError(StatusCode::REQUEST_TIMEOUT, "Request body timed out".into()))?
    .map_err(|error| {
        if error
            .downcast_ref::<http_body_util::LengthLimitError>()
            .is_some()
        {
            ApiError(
                StatusCode::PAYLOAD_TOO_LARGE,
                "Request body exceeds 128 KiB".into(),
            )
        } else {
            ApiError::bad_request(error)
        }
    })?
    .to_bytes();
    serde_json::from_slice(&bytes).map_err(ApiError::bad_request)
}

fn query_value(req: &Request<Incoming>, key: &str) -> Result<String, ApiError> {
    let pairs: Vec<_> =
        url::form_urlencoded::parse(req.uri().query().unwrap_or_default().as_bytes()).collect();
    if pairs.len() != 1 || pairs[0].0 != key {
        return Err(ApiError::bad_request(format!(
            "Expected exactly one {key} query parameter"
        )));
    }
    let value = pairs[0].1.to_string();
    validate_text(&value, key, 256).map_err(ApiError::bad_request)?;
    Ok(value)
}

async fn lookup_reference(
    store: &Result<Arc<BenchmarkStore>, String>,
    id: String,
) -> Result<Baseline, ApiError> {
    let store = store.as_ref().map_err(|error| {
        ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("Benchmark store unavailable: {error}"),
        )
    })?;
    let detail = tokio::time::timeout(
        Duration::from_secs(3),
        store.run_blocking(move |store| {
            let detail = store.run_detail(&id)?;
            Ok(json!({
                "run_record": detail["run_record"],
                "configuration": detail["configuration"],
                "performance_metrics": detail["performance_metrics"]
            }))
        }),
    )
    .await
    .map_err(|_| {
        ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "Benchmark reference lookup timed out".into(),
        )
    })?
    .map_err(|error| {
        let status = match &error {
            BenchmarkError::NotFound(_) => StatusCode::NOT_FOUND,
            BenchmarkError::Validation(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::SERVICE_UNAVAILABLE,
        };
        ApiError(status, error.to_string())
    })?;
    reference_from_detail(&detail, String::new(), Utc::now()).map_err(ApiError::bad_request)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyWrite {
    revision: u64,
    policy: AlertPolicy,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BaselineWrite {
    revision: u64,
    model_key: String,
    run_id: Option<String>,
    expected_experiment_hash: Option<String>,
    note: String,
}

async fn api(
    req: Request<Incoming>,
    observations: &Observability,
    benchmark_store: &Result<Arc<BenchmarkStore>, String>,
) -> Result<Value, ApiError> {
    match (req.method().as_str(), req.uri().path()) {
        ("GET", "/api/observability/models") => Ok(observations.snapshot.lock().unwrap().clone()),
        ("GET", "/api/observability/settings") => Ok(observations.settings_response()),
        ("GET", "/api/observability/reference") => {
            let id = query_value(&req, "run_id")?;
            Ok(json!(lookup_reference(benchmark_store, id).await?))
        }
        ("GET", "/api/observability/baseline") => {
            let model = query_value(&req, "model_key")?;
            let settings = observations.settings.lock().unwrap().settings.clone();
            let Some(saved) = settings.baselines.get(&model) else {
                return Ok(
                    json!({"status": "not_selected", "reason": "Select a successful run explicitly; model-family names are never matched automatically."}),
                );
            };
            match lookup_reference(benchmark_store, saved.run_id.clone()).await {
                Ok(current) => Ok(baseline_comparison(
                    saved,
                    &current,
                    Utc::now(),
                    &settings.policy,
                )),
                Err(error) => Ok(
                    json!({"status": "unavailable", "reference": saved, "reason": error.1, "lookup_status": error.0.as_u16()}),
                ),
            }
        }
        ("POST", "/api/observability/settings") => {
            let write: PolicyWrite = read_body(req).await?;
            let mut settings = observations.settings.lock().unwrap().settings.clone();
            settings.policy = write.policy;
            observations.save(write.revision, settings).await
        }
        ("POST", "/api/observability/baseline") => {
            let write: BaselineWrite = read_body(req).await?;
            validate_text(&write.model_key, "model_key", 256).map_err(ApiError::bad_request)?;
            let mut settings = observations.settings.lock().unwrap().settings.clone();
            if let Some(id) = write.run_id {
                validate_text(&id, "run_id", 256).map_err(ApiError::bad_request)?;
                validate_text(&write.note, "note", 1000).map_err(ApiError::bad_request)?;
                let known = observations.snapshot.lock().unwrap()["models"]
                    .as_array()
                    .is_some_and(|models| {
                        models
                            .iter()
                            .any(|model| model["model_key"] == write.model_key)
                    });
                if !known {
                    return Err(ApiError::bad_request(
                        "Unknown exact local model key; wait for a successful status poll or route",
                    ));
                }
                let mut baseline = lookup_reference(benchmark_store, id).await?;
                if write.expected_experiment_hash.as_deref() != Some(&baseline.experiment_hash) {
                    return Err(ApiError(StatusCode::CONFLICT, "Reference identity changed or was not previewed; inspect and confirm its experiment hash".into()));
                }
                if Utc::now()
                    .signed_duration_since(baseline.ended_at)
                    .num_seconds()
                    > settings.policy.baseline_max_age_days * 86400
                {
                    return Err(ApiError::bad_request(
                        "Reference exceeds policy maximum age",
                    ));
                }
                baseline.note = write.note;
                settings.baselines.insert(write.model_key, baseline);
            } else {
                settings.baselines.remove(&write.model_key);
            }
            observations.save(write.revision, settings).await
        }
        (
            _,
            "/api/observability/models"
            | "/api/observability/settings"
            | "/api/observability/reference"
            | "/api/observability/baseline",
        ) => Err(ApiError(
            StatusCode::METHOD_NOT_ALLOWED,
            "Method not allowed".into(),
        )),
        _ => Err(ApiError(
            StatusCode::NOT_FOUND,
            "Unknown observability endpoint".into(),
        )),
    }
}

pub async fn handle_request(
    req: Request<Incoming>,
    state: &crate::server::AppState,
) -> Result<Response<UnsyncBoxBody<Bytes, anyhow::Error>>, Infallible> {
    if matches!(req.uri().path(), "/models" | "/models/") {
        if req.method() != hyper::Method::GET {
            return Ok(response(
                StatusCode::METHOD_NOT_ALLOWED,
                json!({"error": "Method not allowed"}),
            ));
        }
        return Ok(Response::builder()
            .header("content-type", "text/html; charset=utf-8")
            .header("cache-control", "no-store")
            .body(
                Full::new(Bytes::from_static(HTML.as_bytes()))
                    .map_err(|error: Infallible| match error {})
                    .boxed_unsync(),
            )
            .expect("model page"));
    }
    Ok(
        match api(req, &state.observability, &state.benchmark_store).await {
            Ok(value) => response(StatusCode::OK, value),
            Err(error) => response(error.0, json!({"error": error.1})),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::{server::conn::http1, service::service_fn};
    use hyper_util::rt::TokioIo;
    use tokio::net::TcpListener;

    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("br-observation-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn event(id: u64, age: i64, model: &str, error: &str, now: DateTime<Utc>) -> RouteEvent {
        RouteEvent {
            id,
            timestamp: (now - chrono::Duration::seconds(age)).to_rfc3339(),
            prompt_excerpt: "not included in observation payloads".into(),
            requested_model: model.into(),
            effective_provider: Some("llama-swap".into()),
            model_key: model.into(),
            latency_ms: 120_000,
            stage: Stage::LocalPrimary,
            success: error.is_empty(),
            error: error.into(),
            bonsai_decision: "local",
            cwd: String::new(),
            session_id: None,
            user_agent: String::new(),
            conv_id: "same-conversation".into(),
            pp_tps: 0.0,
            tg_tps: 0.0,
        }
    }

    fn measurement(
        id: u64,
        age: i64,
        model: &str,
        tps: Option<f64>,
        ttft: Option<f64>,
        now: DateTime<Utc>,
    ) -> Measurement {
        Measurement {
            id,
            at: now - chrono::Duration::seconds(age),
            model: model.into(),
            generation_tps: tps,
            ttft_ms: ttft,
            prompt_tokens: None,
            completion_tokens: None,
            stream_duration_ms: None,
        }
    }

    fn reference_fixture(now: DateTime<Utc>) -> Value {
        json!({
            "run_record": {"id": "fixture-run", "status": "succeeded", "ended_at": now - chrono::Duration::hours(1)},
            "configuration": {
                "model": {"id": "fixture-model"}, "artifact": {"sha256": "a".repeat(64)},
                "runtime": {"id": "fixture-runtime", "commit_sha": "fixture-build"},
                "hardware": {"id": "fixture-hardware"}, "workload": {"id": "fixture-workload"},
                "experiment": {"experiment_hash": "b".repeat(64), "context_tokens": 4096}
            },
            "performance_metrics": {"generation_tps": 100.0, "ttft_ms": 25.0}
        })
    }

    #[test]
    fn unknown_measurements_exact_matching_and_predictable_sorting() {
        let now = Utc::now();
        let backend = BackendSnapshot {
            models: parse_catalog(&json!({"data":[{"id":"z-model"},{"id":"a-model"}]})).unwrap(),
            ..BackendSnapshot::default()
        };
        let events = vec![event(1, 5, "a-model-extra", "", now)];
        let samples = vec![measurement(1, 5, "a-model-extra", Some(7.0), None, now)];
        let snapshot = build_snapshot(
            &backend,
            &events,
            &samples,
            &[],
            &Settings::default(),
            now,
            &mut BTreeMap::new(),
        );
        let models = snapshot["models"].as_array().unwrap();
        assert_eq!(
            models
                .iter()
                .map(|model| model["model_key"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["a-model", "a-model-extra", "z-model"]
        );
        assert_eq!(models[0]["alerts"][0]["current_count"], 0);
        assert_eq!(models[0]["alerts"][0]["status"], "insufficient_data");
        assert!(models[0]["memory"]["rss_bytes"].is_null());
        assert!(models[0]["memory"]["vram_bytes"].is_null());
        assert!(models[0]["backend"]["quantization"].is_null());
        assert_eq!(models[1]["alerts"][0]["current_count"], 1);
        assert_eq!(models[1]["alerts"][1]["current_count"], 0);
        assert!(!snapshot
            .to_string()
            .contains("not included in observation payloads"));
    }

    #[test]
    fn generation_and_ttft_require_samples_and_new_evidence_for_recovery() {
        let now = Utc::now();
        let policy = AlertPolicy {
            window_seconds: 30,
            ..AlertPolicy::default()
        };
        let mut samples = Vec::new();
        for id in 1..=5 {
            samples.push(measurement(id, 45, "model", Some(100.0), Some(20.0), now));
        }
        for id in 20..=24 {
            samples.push(measurement(id, 5, "model", Some(40.0), Some(80.0), now));
        }
        for name in ["generation", "ttft"] {
            let mut latch = Latch::default();
            let refs = samples.iter().collect::<Vec<_>>();
            assert_eq!(
                performance_alert(name, &refs[..9], now, &policy, &mut latch)["status"],
                "insufficient_data"
            );
            assert_eq!(
                performance_alert(name, &refs, now, &policy, &mut latch)["status"],
                "pending"
            );
            for _ in 0..5 {
                assert_eq!(
                    performance_alert(name, &refs, now, &policy, &mut latch)["status"],
                    "pending"
                );
            }
            // An older event ID may finish after newer same-conversation requests.
            let late = measurement(10, 2, "model", Some(40.0), Some(80.0), now);
            let mut expanded = refs.clone();
            expanded.push(&late);
            assert_eq!(
                performance_alert(name, &expanded, now, &policy, &mut latch)["status"],
                "active"
            );
            let good: Vec<_> = (30..35)
                .map(|id| measurement(id, 1, "model", Some(100.0), Some(20.0), now))
                .collect();
            let mut recovered: Vec<_> = samples[..5].iter().chain(good.iter()).collect();
            assert_eq!(
                performance_alert(name, &recovered, now, &policy, &mut latch)["status"],
                "recovering"
            );
            let next = measurement(35, 0, "model", Some(100.0), Some(20.0), now);
            recovered.push(&next);
            assert_eq!(
                performance_alert(name, &recovered, now, &policy, &mut latch)["status"],
                "recovered"
            );
            assert_eq!(
                performance_alert(name, &[], now, &policy, &mut latch)["status"],
                "insufficient_data"
            );
        }
    }

    #[test]
    fn missing_zero_nonfinite_and_expired_samples_do_not_trigger() {
        let now = Utc::now();
        let samples: Vec<_> = [
            None,
            Some(0.0),
            Some(-1.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
        ]
        .into_iter()
        .enumerate()
        .map(|(id, value)| measurement(id as u64, 5, "model", value, value, now))
        .collect();
        for name in ["generation", "ttft"] {
            let result = performance_alert(
                name,
                &samples.iter().collect::<Vec<_>>(),
                now,
                &AlertPolicy::default(),
                &mut Latch::default(),
            );
            assert_eq!(result["current_count"], 0);
            assert!(result["current"].is_null());
            assert_eq!(result["status"], "insufficient_data");
        }

        assert_eq!(window(now + chrono::Duration::seconds(1), now, 30), None);
        assert_eq!(window(now - chrono::Duration::seconds(60), now, 30), None);
        assert_eq!(
            window(now - chrono::Duration::seconds(30), now, 30),
            Some(0)
        );
    }

    #[test]
    fn unchanged_full_window_with_late_completion_never_advances_debounce() {
        let policy = AlertPolicy::default();
        let mut latch = Latch::default();
        let original: Vec<_> = (2..=501).collect();
        assert_eq!(latch.evaluate(Some(false), &original, &policy), "ok");
        let late: Vec<_> = std::iter::once(1).chain(3..=501).collect();
        assert_eq!(latch.evaluate(Some(true), &late, &policy), "pending");
        for _ in 0..5 {
            assert_eq!(latch.evaluate(Some(true), &late, &policy), "pending");
            assert_eq!(latch.seen.len(), 500);
        }
        let new: Vec<_> = std::iter::once(1).chain(4..=502).collect();
        assert_eq!(latch.evaluate(Some(true), &new, &policy), "active");
        let recovery: Vec<_> = std::iter::once(1).chain(5..=503).collect();
        assert_eq!(
            latch.evaluate(Some(false), &recovery, &policy),
            "recovering"
        );
        for _ in 0..5 {
            assert_eq!(
                latch.evaluate(Some(false), &recovery, &policy),
                "recovering"
            );
        }
    }

    #[test]
    fn error_and_fallback_alerts_use_recorded_evidence_and_prior_rates() {
        let now = Utc::now();
        let policy = AlertPolicy {
            window_seconds: 30,
            ..AlertPolicy::default()
        };
        let mut events: Vec<_> = (1..=5).map(|id| event(id, 45, "model", "", now)).collect();
        for id in 6..=10 {
            events.push(event(id, 5, "model", "CUDA out of memory", now));
        }
        let mut latch = Latch::default();
        let result = rate_alert(
            "oom",
            &events.iter().collect::<Vec<_>>(),
            now,
            &policy,
            &mut latch,
        );
        assert_eq!(result["status"], "pending");
        assert_eq!(result["current_occurrences"], 5);
        assert_eq!(result["increase_points"], 100.0);
        events.push(event(11, 1, "model", "OOM", now));
        assert_eq!(
            rate_alert(
                "oom",
                &events.iter().collect::<Vec<_>>(),
                now,
                &policy,
                &mut latch
            )["status"],
            "active"
        );
        assert!(!error_evidence(
            &event(12, 1, "model", "room unavailable; GPU hot", now),
            "oom"
        ));
        assert!(!error_evidence(
            &event(13, 1, "model", "deadline maybe", now),
            "timeout"
        ));
        assert!(error_evidence(
            &event(14, 1, "model", "request timed out", now),
            "timeout"
        ));
        let mut fallback = event(15, 1, "model", "", now);
        fallback.stage = Stage::LocalFallback;
        assert!(error_evidence(&fallback, "fallback"));
        assert!(!error_evidence(&fallback, "error"));
        // Constant failure rates are not claimed to be a newly measured regression.
        for event in &mut events {
            event.error = "OOM".into();
            event.success = false;
        }
        assert_eq!(
            rate_alert(
                "oom",
                &events.iter().collect::<Vec<_>>(),
                now,
                &policy,
                &mut Latch::default()
            )["status"],
            "ok"
        );
    }

    #[test]
    fn references_are_explicit_and_never_claim_unverified_comparability() {
        let now = Utc::now();
        let fixture = reference_fixture(now);
        let baseline =
            reference_from_detail(&fixture, "Verified operator mapping rationale".into(), now)
                .unwrap();
        let result = baseline_comparison(&baseline, &baseline, now, &AlertPolicy::default());
        assert_eq!(result["status"], "not_comparable");
        assert!(result["controlled_ttft_regression"].is_null());
        assert_eq!(result["sample_count"], 1);
        let mut current = baseline.clone();
        current.configuration["runtime"]["commit_sha"] = json!("changed-build");
        assert_eq!(
            baseline_comparison(&baseline, &current, now, &AlertPolicy::default())["status"],
            "mismatched"
        );
        assert_eq!(
            baseline_comparison(
                &baseline,
                &baseline,
                now + chrono::Duration::days(31),
                &AlertPolicy::default()
            )["status"],
            "stale"
        );
        for state in ["failed", "oom", "running", "cancelled"] {
            let mut invalid = fixture.clone();
            invalid["run_record"]["status"] = json!(state);
            assert!(reference_from_detail(&invalid, "mapping".into(), now).is_err());
        }
        let mut invalid = fixture.clone();
        invalid["configuration"]["hardware"] = Value::Null;
        assert!(reference_from_detail(&invalid, "mapping".into(), now).is_err());
        invalid = fixture.clone();
        invalid["performance_metrics"] = json!({"ttft_ms": null, "generation_tps": 0});
        assert!(reference_from_detail(&invalid, "mapping".into(), now).is_err());
        invalid = fixture;
        invalid["run_record"]["ended_at"] = json!(now + chrono::Duration::seconds(1));
        assert!(reference_from_detail(&invalid, "mapping".into(), now).is_err());
    }

    #[tokio::test]
    async fn settings_are_atomic_cas_guarded_and_failures_remain_visible() {
        let directory = Directory::new();
        let config = directory.0.join("config.yaml");
        let observations = Observability::new(&config);
        let mut settings = Settings::default();
        settings.policy.min_samples = 9;
        observations.save(0, settings.clone()).await.unwrap();
        assert_eq!(
            load_settings(&config.with_extension("observability.json"))
                .unwrap()
                .policy
                .min_samples,
            9
        );
        assert_eq!(
            observations
                .save(0, Settings::default())
                .await
                .unwrap_err()
                .0,
            StatusCode::CONFLICT
        );
        assert_eq!(
            observations.settings_response()["settings"]["policy"]["min_samples"],
            9
        );
        assert_eq!(
            std::fs::read_dir(&directory.0).unwrap().count(),
            1,
            "no temporary files remain"
        );
        let bytes = std::fs::read(observations.path.clone()).unwrap();
        assert!(!String::from_utf8(bytes).unwrap().contains("prompt_excerpt"));
        std::fs::write(&observations.path, "corrupt").unwrap();
        let broken = Observability::new(&config);
        assert!(broken.settings_response()["read_error"].is_string());
        assert_eq!(
            broken.save(0, Settings::default()).await.unwrap_err().0,
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert!(broken.snapshot.lock().unwrap()["models"].is_array());
        let impossible = Observability::new(&directory.0.join("missing").join("config.yaml"));
        assert_eq!(
            impossible.save(0, settings).await.unwrap_err().0,
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert!(impossible.settings_response()["write_error"].is_string());
        assert_eq!(impossible.settings_response()["revision"], 0);
    }

    #[test]
    fn policy_bounds_and_status_parsing_are_conservative() {
        let policy = AlertPolicy {
            min_samples: 1,
            ..AlertPolicy::default()
        };
        assert!(policy.validate().is_err());
        let policy = AlertPolicy {
            generation_drop_percent: f64::NAN,
            ..AlertPolicy::default()
        };
        assert!(policy.validate().is_err());
        let mut models = parse_catalog(&json!({"data":[{"id":"model"}]})).unwrap();
        assert_eq!(models["model"].state, "idle");
        apply_running(
            &mut models,
            &json!({"running":[{"model":"model","state":"starting"}]}),
        )
        .unwrap();
        assert_eq!(models["model"].state, "loading");
        apply_running(
            &mut models,
            &json!({"running":[{"model":"model","state":"ready"}]}),
        )
        .unwrap();
        assert_eq!(models["model"].state, "loaded");
        apply_running(
            &mut models,
            &json!({"running":[{"model":"model","state":"unexpected"}]}),
        )
        .unwrap();
        assert_eq!(models["model"].state, "unknown");
        assert!(apply_running(&mut models, &json!({})).is_err());
        assert!(!allowed_proxy(
            "http://unrelated.invalid:99",
            "http://127.0.0.1:9999"
        ));
        assert!(!allowed_proxy(
            "file:///etc/passwd",
            "http://127.0.0.1:9999"
        ));
        assert!(allowed_proxy(
            "http://127.0.0.1:99",
            "http://127.0.0.1:9999"
        ));
    }

    async fn serve_api(
        observations: Arc<Observability>,
        store: Result<Arc<BenchmarkStore>, String>,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            let mut connections = tokio::task::JoinSet::new();
            loop {
                tokio::select! {
                    connection = listener.accept() => {
                        let (stream, _) = connection.unwrap();
                        let observations = Arc::clone(&observations); let store = store.clone();
                        connections.spawn(async move {
                            let service = service_fn(move |req| {
                                let observations = Arc::clone(&observations); let store = store.clone();
                                async move {
                                    Ok::<_, Infallible>(match api(req, &observations, &store).await {
                                        Ok(data) => response(StatusCode::OK, data),
                                        Err(error) => response(error.0, json!({"error":error.1})),
                                    })
                                }
                            });
                            http1::Builder::new().serve_connection(TokioIo::new(stream), service).await.unwrap();
                        });
                    }
                    Some(result) = connections.join_next() => { result.unwrap(); }
                }
            }
        });
        (url, task)
    }

    #[tokio::test]
    async fn api_stays_available_without_benchmarks_and_rejects_invalid_requests() {
        let directory = Directory::new();
        let observations = Arc::new(Observability::new(&directory.0.join("config.yaml")));
        let (url, task) = serve_api(observations, Err("fixture unavailable database".into())).await;
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        for route in ["models", "settings", "baseline?model_key=unknown"] {
            assert_eq!(
                client
                    .get(format!("{url}/api/observability/{route}"))
                    .send()
                    .await
                    .unwrap()
                    .status(),
                StatusCode::OK
            );
        }
        assert_eq!(
            client
                .get(format!("{url}/api/observability/reference?run_id=missing"))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            client
                .get(format!(
                    "{url}/api/observability/reference?run_id=a&run_id=b"
                ))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            client
                .get(format!("{url}/api/observability/missing"))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            client
                .delete(format!("{url}/api/observability/settings"))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::METHOD_NOT_ALLOWED
        );
        assert_eq!(
            client
                .post(format!("{url}/api/observability/settings"))
                .body("invalid-json")
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            client
                .post(format!("{url}/api/observability/settings"))
                .body("a".repeat(SETTINGS_LIMIT + 1))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::PAYLOAD_TOO_LARGE
        );
        assert_eq!(
            client
                .post(format!("{url}/api/observability/settings"))
                .json(&json!({"revision":0,"policy":AlertPolicy::default()}))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            client
                .post(format!("{url}/api/observability/settings"))
                .json(&json!({"revision":0,"policy":AlertPolicy::default()}))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::CONFLICT
        );
        task.abort();
        let _ = task.await;
    }

    #[tokio::test]
    async fn api_validates_and_persists_explicit_reference_not_fuzzy_names() {
        let directory = Directory::new();
        let store = Arc::new(BenchmarkStore::open(directory.0.join("benchmarks.sqlite")).unwrap());
        let mut bundle: crate::benchmark::IngestBundle =
            serde_json::from_str(include_str!("../examples/benchmarks/synthetic-bundle.json"))
                .unwrap();
        bundle.run.started_at = Some(Utc::now() - chrono::Duration::seconds(10));
        bundle.run.ended_at = Some(Utc::now() - chrono::Duration::seconds(5));
        let id = store.ingest(&bundle).unwrap();
        let observations = Arc::new(Observability::new(&directory.0.join("config.yaml")));
        *observations.snapshot.lock().unwrap() = json!({"models":[{"model_key":"exact-key"}]});
        let (url, task) = serve_api(Arc::clone(&observations), Ok(Arc::clone(&store))).await;
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let reference: Value = client
            .get(format!("{url}/api/observability/reference?run_id={id}"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let mut write = json!({"revision":0,"model_key":"exact-key-extra","run_id":id,
            "expected_experiment_hash":reference["experiment_hash"],"note":"Synthetic fixture mapping"});
        assert_eq!(
            client
                .post(format!("{url}/api/observability/baseline"))
                .json(&write)
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
        write["model_key"] = json!("exact-key");
        write["expected_experiment_hash"] = json!("wrong");
        assert_eq!(
            client
                .post(format!("{url}/api/observability/baseline"))
                .json(&write)
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::CONFLICT
        );
        write["expected_experiment_hash"] = reference["experiment_hash"].clone();
        assert_eq!(
            client
                .post(format!("{url}/api/observability/baseline"))
                .json(&write)
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        let selected: Value = client
            .get(format!(
                "{url}/api/observability/baseline?model_key=exact-key"
            ))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(selected["status"], "not_comparable");
        assert_eq!(selected["reference"]["run_id"], id);
        assert_eq!(
            load_settings(&observations.path).unwrap().baselines["exact-key"].run_id,
            id
        );
        task.abort();
        let _ = task.await;
        let (url, task) = serve_api(
            Arc::clone(&observations),
            Err("database later unavailable".into()),
        )
        .await;
        let selected: Value = client
            .get(format!(
                "{url}/api/observability/baseline?model_key=exact-key"
            ))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(selected["status"], "unavailable");
        write["revision"] = json!(1);
        write["run_id"] = Value::Null;
        assert_eq!(
            client
                .post(format!("{url}/api/observability/baseline"))
                .json(&write)
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        task.abort();
        let _ = task.await;
    }

    #[tokio::test]
    async fn stub_backend_reports_metadata_without_running_any_model_work() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let stub_url = url.clone();
        let task = tokio::spawn(async move {
            let mut connections = tokio::task::JoinSet::new();
            loop {
                tokio::select! {
                    connection = listener.accept() => {
                        let (stream, _) = connection.unwrap(); let stub_url = stub_url.clone();
                        connections.spawn(async move {
                            let service = service_fn(move |req: Request<Incoming>| {
                                let data = match req.uri().path() {
                                    "/v1/models" => json!({"data":[{"id":"fixture"}]}),
                                    "/running" => json!({"running":[{"model":"fixture","state":"ready","proxy":stub_url}]}),
                                    "/props" => json!({"model_path":"fixture.gguf","build_info":"fixture-build","default_generation_settings":{"n_ctx":4096}}),
                                    other => panic!("Unexpected backend request, especially inference: {other}"),
                                };
                                async move { Ok::<_, Infallible>(response(StatusCode::OK, data)) }
                            });
                            http1::Builder::new().serve_connection(TokioIo::new(stream), service).await.unwrap();
                        });
                    }
                    Some(result) = connections.join_next() => { result.unwrap(); }
                }
            }
        });
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_millis(100))
            .build()
            .unwrap();
        let snapshot = poll_backend(&client, &url).await;
        assert!(snapshot.running_error.is_none());
        assert!(snapshot.catalog_error.is_none());
        assert_eq!(snapshot.models["fixture"].state, "loaded");
        assert_eq!(
            snapshot.models["fixture"].artifact_path.as_deref(),
            Some("fixture.gguf")
        );
        assert_eq!(
            snapshot.models["fixture"].runtime_build,
            Some(json!("fixture-build"))
        );
        assert!(snapshot.models["fixture"].quantization.is_none());
        task.abort();
        let _ = task.await;
        let failed = poll_backend(&client, &url).await;
        assert!(failed.running_error.is_some());
        assert!(failed.catalog_error.is_some());
    }
}
