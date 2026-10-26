//! v2 L6: cloud/LAN provider registry (`providers.json`, 0600 atomic).
//!
//! Providers replace the v1 Manifest special case: an owner-registered pool of
//! OpenAI-compatible, Anthropic, and migrated-manifest endpoints. The store is
//! the single source of truth for selector resolution (`cloud` /
//! `cloud/<id>` / `cloud/<provider>/<id>`), for the auto cloud pool, and for
//! per-provider breakers (the health tracker keys on the provider id; the
//! migrated Manifest provider keeps the literal `manifest` key, preserving
//! live breaker semantics).
//!
//! Mutations follow the L3 single-writer pattern: the daemon owns the file and
//! the CLI talks to it. Every write is atomic and 0600.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// State-dir file name (L6 spec "Providers": `providers.json`).
pub const PROVIDERS_FILE: &str = "providers.json";

/// `$XDG_STATE_HOME/brainrouter/providers.json`.
pub fn default_path() -> PathBuf {
    crate::paths::brainrouter_state().join(PROVIDERS_FILE)
}

/// Outbound wire family of a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderKind {
    /// OpenAI-compatible `/chat/completions` (llama-server, DeepInfra, a peer
    /// brainrouter, ...).
    #[serde(rename = "openai-compatible")]
    OpenaiCompatible,
    /// Anthropic Messages wire (`x-api-key` + `anthropic-version`).
    #[serde(rename = "anthropic")]
    Anthropic,
    /// The migrated v1 Manifest gateway. `auto` records the v1 hardcoded
    /// `model:"auto"` behavior; any other model id passes through verbatim.
    #[serde(rename = "manifest")]
    Manifest,
}

impl ProviderKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenaiCompatible => "openai-compatible",
            Self::Anthropic => "anthropic",
            Self::Manifest => "manifest",
        }
    }
}

/// Operator-entered quality tier; ranks cloud-pool candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelTier {
    Fast,
    #[default]
    Balanced,
    Deep,
}

impl ModelTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Balanced => "balanced",
            Self::Deep => "deep",
        }
    }

    fn rank(self) -> i32 {
        match self {
            Self::Fast => 0,
            Self::Balanced => 1,
            Self::Deep => 2,
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "fast" => Some(Self::Fast),
            "balanced" => Some(Self::Balanced),
            "deep" => Some(Self::Deep),
            _ => None,
        }
    }
}

/// `$/1K in,out` (L6 spec "Cost"; the fixture spelling is `{in, out}`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Price {
    #[serde(rename = "in", alias = "input")]
    pub input: f64,
    #[serde(rename = "out", alias = "output")]
    pub output: f64,
}

/// One model served by a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderModel {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,
    #[serde(default)]
    pub ctx: u64,
    #[serde(default = "default_true")]
    pub tools: bool,
    #[serde(default)]
    pub tier: ModelTier,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub price: Option<Price>,
}

/// Budget enforcement mode (default `warn`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BudgetMode {
    #[default]
    Warn,
    Stop,
}

impl BudgetMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Warn => "warn",
            Self::Stop => "stop",
        }
    }
}

/// One provider record (L6 spec "Providers" record shape).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderRecord {
    pub id: String,
    #[serde(default = "default_kind")]
    pub kind: ProviderKind,
    pub base_url: String,
    /// Credential id (store record id, or `env:NAME` for the migrated v1
    /// `manifest.api_key_env` indirection). Absent = auth-less local gateway.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential_id: Option<String>,
    /// Manifest kind only: the record answers the v1 wire value `model:"auto"`.
    #[serde(default)]
    pub auto: bool,
    #[serde(default)]
    pub models: Vec<ProviderModel>,
    /// Federation peer: role headers are forwarded verbatim; peers never join
    /// the cloud auto pool.
    #[serde(default)]
    pub lan_peer: bool,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monthly_budget_usd: Option<f64>,
    #[serde(default)]
    pub budget_mode: BudgetMode,
}

fn default_true() -> bool {
    true
}

fn default_kind() -> ProviderKind {
    ProviderKind::OpenaiCompatible
}

impl ProviderRecord {
    /// The model entry matching an id OR alias, when the provider serves it.
    pub fn model_match(&self, model: &str) -> Option<&ProviderModel> {
        self.models
            .iter()
            .find(|m| m.id == model || m.alias.as_deref() == Some(model))
    }
}

#[derive(Default, Serialize, Deserialize)]
struct StoreData {
    #[serde(default)]
    providers: Vec<ProviderRecord>,
    #[serde(default)]
    schema_version: u32,
}

/// The provider registry. In-memory copy guarded by a mutex; every mutation
/// persists atomically with mode 0600.
pub struct ProviderStore {
    path: PathBuf,
    data: Mutex<StoreData>,
}

impl ProviderStore {
    /// Load the registry. A missing file is an empty registry.
    pub fn load(path: PathBuf) -> Result<Self> {
        let data = match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str::<StoreData>(&text)
                .with_context(|| format!("providers store {} is corrupt", path.display()))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => StoreData::default(),
            Err(e) => {
                return Err(e).with_context(|| format!("reading providers store {}", path.display()))
            }
        };
        Ok(Self {
            path,
            data: Mutex::new(data),
        })
    }

    /// In-memory registry (unit tests / degraded boot).
    pub fn in_memory() -> Self {
        Self::in_memory_at(PathBuf::new(), Vec::new())
    }

    pub fn in_memory_with(records: Vec<ProviderRecord>) -> Self {
        Self::in_memory_at(PathBuf::new(), records)
    }

    fn in_memory_at(path: PathBuf, records: Vec<ProviderRecord>) -> Self {
        Self {
            path,
            data: Mutex::new(StoreData {
                providers: records,
                schema_version: 1,
            }),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn snapshot(&self) -> Vec<ProviderRecord> {
        self.data.lock().unwrap().providers.clone()
    }

    pub fn get(&self, id: &str) -> Option<ProviderRecord> {
        self.data
            .lock()
            .unwrap()
            .providers
            .iter()
            .find(|p| p.id == id)
            .cloned()
    }

    /// `cloud_available` (L6 spec): at least one enabled non-`lan_peer`
    /// provider (the migrated manifest provider included).
    pub fn cloud_available(&self) -> bool {
        self.data
            .lock()
            .unwrap()
            .providers
            .iter()
            .any(|p| p.enabled && !p.lan_peer)
    }

    /// The enabled manifest-kind auto provider (v1 `cloud` parity), when unique.
    pub fn auto_manifest(&self) -> Option<ProviderRecord> {
        let data = self.data.lock().unwrap();
        let mut found = data
            .providers
            .iter()
            .filter(|p| p.enabled && p.kind == ProviderKind::Manifest && p.auto);
        let first = found.next()?.clone();
        Some(first)
    }

    /// Add or replace one provider. Validation runs before the write.
    pub fn upsert(&self, mut record: ProviderRecord) -> Result<()> {
        validate_record(&mut record)?;
        let mut data = self.data.lock().unwrap();
        if record.enabled && record.kind == ProviderKind::Manifest && record.auto {
            if let Some(other) = data.providers.iter().find(|p| {
                p.id != record.id && p.enabled && p.kind == ProviderKind::Manifest && p.auto
            }) {
                bail!(
                    "a second enabled manifest provider with auto: true is a configuration \
                     error (rule (b) requires uniqueness): `{}` already answers the v1 \
                     `cloud` selector; `{}` was refused",
                    other.id,
                    record.id
                );
            }
        }
        match data.providers.iter_mut().find(|p| p.id == record.id) {
            Some(existing) => *existing = record,
            None => data.providers.push(record),
        }
        persist(&self.path, &data)
    }

    pub fn remove(&self, id: &str) -> Result<bool> {
        let mut data = self.data.lock().unwrap();
        let before = data.providers.len();
        data.providers.retain(|p| p.id != id);
        let removed = data.providers.len() != before;
        if removed {
            persist(&self.path, &data)?;
        }
        Ok(removed)
    }

    /// Replace the whole registry (v2.0-p6 migration).
    pub fn replace_all(&self, records: Vec<ProviderRecord>) -> Result<()> {
        let mut data = self.data.lock().unwrap();
        data.providers = records;
        data.schema_version = 1;
        persist(&self.path, &data)
    }
}

fn validate_record(record: &mut ProviderRecord) -> Result<()> {
    record.id = record.id.trim().to_string();
    if record.id.is_empty()
        || record.id.len() > 128
        || record
            .id
            .chars()
            .any(|c| c.is_whitespace() || c == '/' || c.is_control())
    {
        bail!("provider id must be 1-128 bytes without whitespace, '/' or control characters");
    }
    if record.base_url.is_empty() {
        bail!("provider `{}`: base_url must not be empty", record.id);
    }
    if !record.base_url.starts_with("http://") && !record.base_url.starts_with("https://") {
        bail!(
            "provider `{}`: base_url must start with http:// or https://, got {}",
            record.id,
            record.base_url
        );
    }
    record.base_url = record.base_url.trim_end_matches('/').to_string();
    if record.kind != ProviderKind::Manifest && record.auto {
        bail!(
            "provider `{}`: `auto` is a manifest-kind behavior (wire value `model:\"auto\"`)",
            record.id
        );
    }
    if let Some(budget) = record.monthly_budget_usd {
        if !budget.is_finite() || budget < 0.0 {
            bail!("provider `{}`: monthly_budget_usd must be a finite non-negative number", record.id);
        }
    }
    for model in &mut record.models {
        model.id = model.id.trim().to_string();
        if model.id.is_empty() {
            bail!("provider `{}`: model id must not be empty", record.id);
        }
        if let Some(price) = model.price {
            if !price.input.is_finite() || !price.output.is_finite() {
                bail!(
                    "provider `{}` model `{}`: price must be finite",
                    record.id,
                    model.id
                );
            }
        }
    }
    if let Some(credential) = record.credential_id.as_deref() {
        if credential.trim().is_empty() {
            record.credential_id = None;
        }
    }
    Ok(())
}

fn persist(path: &Path, data: &StoreData) -> Result<()> {
    if path.as_os_str().is_empty() {
        return Ok(());
    }
    let text = serde_json::to_string_pretty(data).context("serializing providers store")?;
    crate::plugins::store::atomic_write_0600(path, text.as_bytes())
        .with_context(|| format!("writing providers store {}", path.display()))
}

// ── Selector resolution (the L6 matrix) ─────────────────────────────────────

/// A request's pool-relevant features.
#[derive(Debug, Clone, Default)]
pub struct PoolQuery {
    pub approx_tokens: u64,
    pub needs_tools: bool,
    /// The decision slot's tier (`fast|balanced|deep`); `None` = balanced.
    pub tier: Option<ModelTier>,
    /// Providers excluded because a `stop`-mode budget is exhausted.
    pub over_budget: HashSet<String>,
}

/// The concrete provider+model a `cloud` selector resolved to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloudTarget {
    pub provider_id: String,
    pub model: String,
    pub base_url: String,
    pub lan_peer: bool,
    pub kind: ProviderKind,
}

/// L6 rule (c): multi-match / no-provider are explicit 4xx client errors, not
/// backend failures (the server maps this to 400).
#[derive(Debug, Clone)]
pub struct CloudSelectorError(pub String);

impl std::fmt::Display for CloudSelectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CloudSelectorError {}

fn target(record: &ProviderRecord, model: String) -> CloudTarget {
    CloudTarget {
        provider_id: record.id.clone(),
        model,
        base_url: record.base_url.clone(),
        lan_peer: record.lan_peer,
        kind: record.kind,
    }
}

/// Resolve a cloud selector against the enabled registry.
///
/// * `cloud_model = None` — bare `cloud`: the enabled manifest auto provider
///   (v1 parity) else the auto cloud pool's top candidate.
/// * `Some(rest)` — `cloud/<id>`: (a) unique model/alias match; (d) qualified
///   `cloud/<provider>/<model>` when the first segment names an enabled
///   provider; (b) one enabled manifest auto provider ⇒ verbatim passthrough;
///   (c) multi-match ⇒ 4xx naming the providers.
pub fn resolve_cloud_target(
    records: &[ProviderRecord],
    cloud_model: Option<&str>,
    query: &PoolQuery,
) -> Result<CloudTarget, CloudSelectorError> {
    let enabled: Vec<&ProviderRecord> = records.iter().filter(|p| p.enabled).collect();

    let Some(rest) = cloud_model else {
        if let Some(manifest) = enabled
            .iter()
            .find(|p| p.kind == ProviderKind::Manifest && p.auto)
        {
            return Ok(target(manifest, "auto".to_string()));
        }
        return pool_top(&enabled, query).ok_or_else(|| {
            CloudSelectorError(
                "no cloud candidate: no enabled non-peer provider model matches the request \
                 (ctx/tools/budget)"
                    .to_string(),
            )
        });
    };

    // (d) qualified form: only when the first segment names an ENABLED provider.
    if let Some((first, remainder)) = rest.split_once('/') {
        if !remainder.is_empty() {
            if let Some(provider) = enabled.iter().find(|p| p.id == first) {
                return Ok(target(provider, remainder.to_string()));
            }
        }
    }

    // (a) unique match on any enabled provider's models[].id / alias.
    let matches: Vec<(&ProviderRecord, &ProviderModel)> = enabled
        .iter()
        .filter_map(|p| p.model_match(rest).map(|m| (*p, m)))
        .collect();
    let distinct: HashSet<&str> = matches.iter().map(|(p, _)| p.id.as_str()).collect();
    if distinct.len() > 1 {
        let mut names: Vec<&str> = distinct.into_iter().collect();
        names.sort_unstable();
        return Err(CloudSelectorError(format!(
            "model id `{}` matches multiple enabled providers ({}); use the qualified form \
             `cloud/<provider>/{}` — no silent pick",
            rest,
            names.join(", "),
            rest
        )));
    }
    if let Some((provider, model)) = matches.first() {
        return Ok(target(provider, model.id.clone()));
    }

    // (b) v1 parity: exactly one enabled manifest auto provider accepts the
    // undeclared id verbatim ("accepted without discovery").
    let auto_manifests: Vec<&ProviderRecord> = enabled
        .iter()
        .copied()
        .filter(|p| p.kind == ProviderKind::Manifest && p.auto)
        .collect();
    match auto_manifests.len() {
        0 => Err(CloudSelectorError(format!(
            "no enabled provider serves `{}` and no manifest provider is configured for \
             passthrough",
            rest
        ))),
        1 => Ok(target(auto_manifests[0], rest.to_string())),
        _ => {
            let names: Vec<&str> = auto_manifests.iter().map(|p| p.id.as_str()).collect();
            Err(CloudSelectorError(format!(
                "two enabled manifest providers with auto: true ({}) — rule (b) requires \
                 uniqueness",
                names.join(", ")
            )))
        }
    }
}

/// Top of the auto cloud pool: enabled non-peer providers, ctx/tools filtered,
/// tier-fit ranked, stable on provider record order then model list order.
fn pool_top(enabled: &[&ProviderRecord], query: &PoolQuery) -> Option<CloudTarget> {
    let wanted_tier = query.tier.unwrap_or_default();
    let mut candidates: Vec<(i32, usize, usize, &ProviderRecord, &ProviderModel)> = Vec::new();
    for (provider_index, provider) in enabled.iter().enumerate() {
        if provider.lan_peer || query.over_budget.contains(&provider.id) {
            continue;
        }
        for (model_index, model) in provider.models.iter().enumerate() {
            if query.needs_tools && !model.tools {
                continue;
            }
            if query.approx_tokens > 0 && model.ctx > 0 && model.ctx < query.approx_tokens {
                continue;
            }
            let fit = (model.tier.rank() - wanted_tier.rank()).abs();
            candidates.push((fit, provider_index, model_index, *provider, model));
        }
    }
    candidates.sort_by_key(|(fit, provider_index, model_index, _, _)| {
        (*fit, *provider_index, *model_index)
    });
    candidates
        .first()
        .map(|(_, _, _, provider, model)| target(provider, model.id.clone()))
}

/// Resolve a bare key/alias against enabled providers (LAN expose
/// double-enforcement: the RESOLVED model id).
pub fn resolve_bare_model(records: &[ProviderRecord], key: &str) -> Option<(String, String)> {
    for provider in records.iter().filter(|p| p.enabled) {
        if let Some(model) = provider.model_match(key) {
            return Some((provider.id.clone(), model.id.clone()));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(id: &str, kind: ProviderKind, models: Vec<ProviderModel>) -> ProviderRecord {
        ProviderRecord {
            id: id.into(),
            kind,
            base_url: format!("http://127.0.0.1:1/{id}/v1"),
            credential_id: None,
            auto: kind == ProviderKind::Manifest,
            models,
            lan_peer: false,
            enabled: true,
            monthly_budget_usd: None,
            budget_mode: BudgetMode::Warn,
        }
    }

    fn model(id: &str) -> ProviderModel {
        ProviderModel {
            id: id.into(),
            alias: None,
            ctx: 131072,
            tools: true,
            tier: ModelTier::Balanced,
            price: None,
        }
    }

    #[test]
    fn bare_cloud_prefers_manifest_then_pool_order() {
        let records = vec![
            provider("p1", ProviderKind::OpenaiCompatible, vec![model("p1-m1")]),
            provider("manifest", ProviderKind::Manifest, vec![]),
        ];
        let q = PoolQuery::default();
        assert_eq!(
            resolve_cloud_target(&records, None, &q).unwrap().provider_id,
            "manifest"
        );
        let records = records[..1].to_vec();
        assert_eq!(
            resolve_cloud_target(&records, None, &q).unwrap().model,
            "p1-m1"
        );
    }

    #[test]
    fn qualified_requires_an_enabled_provider_then_cascades() {
        let records = vec![provider(
            "manifest",
            ProviderKind::Manifest,
            vec![],
        )];
        let q = PoolQuery::default();
        let passthrough = resolve_cloud_target(
            &records,
            Some("deepinfra/deepseek-ai/DeepSeek-V4-Flash"),
            &q,
        )
        .unwrap();
        assert_eq!(passthrough.provider_id, "manifest");
        assert_eq!(passthrough.model, "deepinfra/deepseek-ai/DeepSeek-V4-Flash");

        let mut with_provider = records.clone();
        with_provider.push(provider(
            "deepinfra",
            ProviderKind::OpenaiCompatible,
            vec![model("deepseek-ai/DeepSeek-V4-Flash")],
        ));
        let qualified = resolve_cloud_target(
            &with_provider,
            Some("deepinfra/deepseek-ai/DeepSeek-V4-Flash"),
            &q,
        )
        .unwrap();
        assert_eq!(qualified.provider_id, "deepinfra");
        assert_eq!(qualified.model, "deepseek-ai/DeepSeek-V4-Flash");
    }

    #[test]
    fn multi_match_is_an_error_naming_providers() {
        let records = vec![
            provider("a", ProviderKind::OpenaiCompatible, vec![model("shared")]),
            provider("b", ProviderKind::OpenaiCompatible, vec![model("shared")]),
        ];
        let err = resolve_cloud_target(&records, Some("shared"), &PoolQuery::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains('a') && err.contains('b'), "{err}");
    }
}
