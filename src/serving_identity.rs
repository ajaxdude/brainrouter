//! PR11a (design doc §5b/§16): serving-identity registration for toolbox
//! Server Mode backends (v2 L1: ds4/gufo).
//!
//! **Scope, precisely (see §16 for the full reasoning — do not extend this
//! module to route real traffic without re-reading it first):** this is a
//! read-only bookkeeping registry, not a routing mechanism. It records a
//! `{toolbox_backend, compute_api, runtime_profile_id, endpoint}` fact per
//! currently-running, brainrouter-managed server-mode container, so the
//! dashboard (and, eventually, a future PR) has a real data source instead
//! of "no registration exists at all". It deliberately does **not** modify
//! `daemon.rs`'s `Router`/`route()` or `routing_events::derive_serving_identity()`
//! — no backend becomes a genuinely routable upstream as a side effect of
//! this module existing. That is Rollout PR 11b's job, not this one's.
//!
//! Modeled as an in-memory registry, not a new DB table, mirroring §5c's
//! "live container state is not persisted to SQLite" convention (the same
//! reasoning that kept PR2's toolbox-container state podman-derived). It is
//! populated at container-start/stop time only — not checked per HTTP
//! request — since no request currently reaches any of these backends
//! (§16, resolving §747's "request-time vs. container-start-time" open
//! item in favor of the latter).

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use serde::Serialize;
use tokio::sync::RwLock;

use crate::toolbox_catalog::SupportedServingBackend;

/// One registered server-mode container's serving identity.
#[derive(Debug, Clone, Serialize)]
pub struct ServingIdentity {
    pub toolbox_backend: &'static str,
    /// `"vulkan" | "rocm" | "cuda"` today — derived from the resolved
    /// toolbox's [`crate::toolbox_catalog::RuntimeProfile`] at registration
    /// time (see [`crate::server_mode::compute_api_for_runtime_profile`]),
    /// not a live probe.
    pub compute_api: String,
    /// The catalog `toolbox_id` actually running (not the backend id —
    /// several toolbox ids can share one backend, e.g. ds4's rocm/therock
    /// variants).
    pub runtime_profile_id: String,
    /// The exact OpenAI `model` string served by this backend endpoint.
    /// Matching is exact and in-memory only; no live podman inspection is
    /// performed on the request routing hot path.
    pub served_model: String,
    /// `"http://host:port"`, from the start request (falling back to each
    /// backend's own documented default host/port when the request omitted
    /// them).
    pub endpoint: String,
    /// Static, per-backend fact recorded by §16's research — **not** a live
    /// protocol probe, and registering an identity confers no routing
    /// capability regardless of this value (see module docs above).
    pub openai_compatible: bool,
    pub registered_at: DateTime<Utc>,
}

/// One entry in `GET /api/serving-identities`'s response — [`ServingIdentity`]
/// plus a live `container_running` cross-check (§16: staleness — a
/// container dying without brainrouter noticing — is handled by re-checking
/// live podman state at read time, not by trusting the in-memory map alone).
#[derive(Debug, Clone, Serialize)]
pub struct ServingIdentityView {
    pub container_name: String,
    #[serde(flatten)]
    pub identity: ServingIdentity,
    pub container_running: bool,
}

/// Static per-backend OpenAI-API-compatibility classification, per §16's
/// research. Recorded here, not derived, and not treated as ground truth
/// for anything beyond dashboard/inspection purposes.
pub fn openai_compatible_for_backend(backend: SupportedServingBackend) -> bool {
    match backend {
        // Already the router's one real local provider today.
        SupportedServingBackend::LlamaCpp => true,
        // gufo is genuinely OpenAI-compatible (`/v1/chat/completions`,
        // `/v1/completions`, `/v1/responses`, `/v1/models`, `/health` —
        // github.com/gufo-org/gufo). This is accurate bookkeeping only; it
        // does not make gufo a routable upstream (design doc D5, same as the
        // other backends). A strong future routing candidate (Feature B).
        SupportedServingBackend::Gufo => true,
        // No new evidence surfaced beyond §12a's original "unverified"
        // findings; §16 deliberately does not upgrade this to `true`
        // without real evidence.
        SupportedServingBackend::Ds4 => false,
    }
}

/// In-memory registry, keyed by the backend's fixed container name (each
/// kept backend — ds4/gufo — is single-instance-per-host in v1, per
/// §12 — one entry per backend is therefore sufficient, but the
/// registry is keyed generically so a future multi-instance backend needs
/// no shape change).
#[derive(Default)]
pub struct ServingIdentityRegistry {
    inner: RwLock<HashMap<String, ServingIdentity>>,
}

impl ServingIdentityRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Called on a backend's `start_*_server()` success. Overwrites any
    /// existing entry for the same container name (a restart with
    /// different tuning replaces, not duplicates, the identity).
    pub async fn register(&self, container_name: &str, identity: ServingIdentity) {
        self.inner.write().await.insert(container_name.to_string(), identity);
    }

    /// Called on a backend's `stop_*_server()` success. Removing an
    /// already-absent entry is a no-op, matching the idempotent
    /// stop-already-stopped contract every `stop_*_server()` already has.
    pub async fn deregister(&self, container_name: &str) {
        self.inner.write().await.remove(container_name);
    }

    /// Hot-path lookup for Feature B routing. This reads only the in-memory
    /// registry and only returns OpenAI-compatible identities whose served
    /// model exactly matches the requested model.
    pub async fn find_by_served_model(&self, model: &str) -> Option<ServingIdentity> {
        self.inner
            .read()
            .await
            .values()
            .find(|identity| identity.openai_compatible && identity.served_model == model)
            .cloned()
    }

    /// In-memory served-model list for `GET /v1/models`; intentionally does
    /// not call [`Self::snapshot`] because that shells out to podman.
    pub async fn openai_compatible_served_models(&self) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut models: Vec<String> = self
            .inner
            .read()
            .await
            .values()
            .filter(|identity| identity.openai_compatible)
            .filter_map(|identity| {
                let model = identity.served_model.clone();
                seen.insert(model.clone()).then_some(model)
            })
            .collect();
        models.sort();
        models
    }

    /// Read-only snapshot for `GET /api/serving-identities` and the
    /// dashboard panel. Cross-checks each entry's container against live
    /// podman state via `container_running`, so a container that died
    /// without brainrouter noticing (§16, §747) reads as stale rather than
    /// silently trusted.
    pub async fn snapshot(&self) -> Vec<ServingIdentityView> {
        let entries: Vec<(String, ServingIdentity)> = self
            .inner
            .read()
            .await
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let mut views = Vec::with_capacity(entries.len());
        for (container_name, identity) in entries {
            let running = crate::server_mode::container_running(&container_name).await;
            views.push(ServingIdentityView {
                container_name,
                identity,
                container_running: running,
            });
        }
        views
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_compatible_for_backend_matches_16_research() {
        assert!(openai_compatible_for_backend(SupportedServingBackend::LlamaCpp));
        assert!(openai_compatible_for_backend(SupportedServingBackend::Gufo));
        assert!(!openai_compatible_for_backend(SupportedServingBackend::Ds4));
    }

    fn sample_identity() -> ServingIdentity {
        ServingIdentity {
            toolbox_backend: "gufo",
            compute_api: "rocm".to_string(),
            runtime_profile_id: "strix-halo-gufo-runtime".to_string(),
            served_model: "gufo-org/Qwen3.8-27B-Q4_K_XL".to_string(),
            endpoint: "http://127.0.0.1:8002".to_string(),
            openai_compatible: true,
            registered_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn register_then_snapshot_returns_the_entry() {
        let registry = ServingIdentityRegistry::default();
        registry.register("brainrouter-gufo-server", sample_identity()).await;
        let snap = registry.snapshot().await;
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].container_name, "brainrouter-gufo-server");
        assert_eq!(snap[0].identity.toolbox_backend, "gufo");
        // No real podman on this host, so container_running is always
        // false here — asserting it's present, not asserting a specific
        // podman-dependent value.
        let _ = snap[0].container_running;
    }

    #[tokio::test]
    async fn register_overwrites_same_container_name() {
        let registry = ServingIdentityRegistry::default();
        registry.register("brainrouter-gufo-server", sample_identity()).await;
        let mut second = sample_identity();
        second.endpoint = "http://127.0.0.1:9999".to_string();
        registry.register("brainrouter-gufo-server", second).await;
        let snap = registry.snapshot().await;
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].identity.endpoint, "http://127.0.0.1:9999");
    }

    #[tokio::test]
    async fn deregister_removes_the_entry() {
        let registry = ServingIdentityRegistry::default();
        registry.register("brainrouter-gufo-server", sample_identity()).await;
        registry.deregister("brainrouter-gufo-server").await;
        assert!(registry.snapshot().await.is_empty());
    }

    #[tokio::test]
    async fn deregister_of_absent_entry_is_a_no_op() {
        let registry = ServingIdentityRegistry::default();
        registry.deregister("brainrouter-does-not-exist").await;
        assert!(registry.snapshot().await.is_empty());
    }

    #[tokio::test]
    async fn find_by_served_model_matches_openai_compatible_exactly() {
        let registry = ServingIdentityRegistry::default();
        registry.register("brainrouter-gufo-server", sample_identity()).await;
        let found = registry
            .find_by_served_model("gufo-org/Qwen3.8-27B-Q4_K_XL")
            .await
            .expect("expected exact compatible match");
        assert_eq!(found.toolbox_backend, "gufo");
        assert!(registry.find_by_served_model("gufo-org/qwen3.8-27b-q4_k_xl").await.is_none());
        assert!(registry.find_by_served_model("Qwen3.8-27B").await.is_none());
    }

    #[tokio::test]
    async fn find_by_served_model_ignores_non_openai_compatible_identity() {
        let registry = ServingIdentityRegistry::default();
        let mut identity = sample_identity();
        identity.toolbox_backend = "ds4";
        identity.openai_compatible = false;
        registry.register("brainrouter-ds4-server", identity).await;
        assert!(registry.find_by_served_model("gufo-org/Qwen3.8-27B-Q4_K_XL").await.is_none());
    }

    #[tokio::test]
    async fn openai_compatible_served_models_dedups_and_excludes_incompatible() {
        let registry = ServingIdentityRegistry::default();
        registry.register("brainrouter-gufo-server", sample_identity()).await;
        let mut duplicate = sample_identity();
        duplicate.toolbox_backend = "llama_cpp";
        duplicate.endpoint = "http://127.0.0.1:8003".to_string();
        registry.register("brainrouter-llama-server", duplicate).await;
        let mut incompatible = sample_identity();
        incompatible.toolbox_backend = "ds4";
        incompatible.openai_compatible = false;
        incompatible.served_model = "deepseek-v4-flash".to_string();
        registry.register("brainrouter-ds4-server", incompatible).await;

        assert_eq!(
            registry.openai_compatible_served_models().await,
            vec!["gufo-org/Qwen3.8-27B-Q4_K_XL".to_string()]
        );
    }
}
