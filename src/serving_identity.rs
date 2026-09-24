//! PR11a (design doc §5b/§16): serving-identity registration for toolbox
//! Server Mode backends (ds4/halogen/vllm/r9v).
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

use std::collections::HashMap;

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
    /// `"http://host:port"`, from the start request (falling back to each
    /// backend's own documented default host/port when the request omitted
    /// them, e.g. r9v's optional `host`/`port` fields).
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
        // Confirmed via `server_mode.rs`'s own `served-model-name` /
        // OpenAI-compatible-API doc comment plus vLLM's well-established
        // wire protocol (§16).
        SupportedServingBackend::Vllm => true,
        // gufo is genuinely OpenAI-compatible (`/v1/chat/completions`,
        // `/v1/completions`, `/v1/responses`, `/v1/models`, `/health` —
        // github.com/gufo-org/gufo). This is accurate bookkeeping only; it
        // does not make gufo a routable upstream (design doc D5, same as the
        // other backends). A strong future routing candidate (Feature B).
        SupportedServingBackend::Gufo => true,
        // Plausible (vLLM-lineage env-var surface) but not verified against
        // a live deployment — §16 deliberately does not upgrade this to
        // `true` without real evidence.
        SupportedServingBackend::R9v => false,
        // No new evidence surfaced beyond §12a/§13a's original "unverified"
        // findings.
        SupportedServingBackend::Ds4 => false,
        SupportedServingBackend::Halogen => false,
    }
}

/// In-memory registry, keyed by the backend's fixed container name (each
/// of ds4/halogen/vllm/r9v is single-instance-per-host in v1, per
/// §12/§13/§14/§15 — one entry per backend is therefore sufficient, but the
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
        assert!(openai_compatible_for_backend(SupportedServingBackend::Vllm));
        assert!(openai_compatible_for_backend(SupportedServingBackend::Gufo));
        assert!(!openai_compatible_for_backend(SupportedServingBackend::Ds4));
        assert!(!openai_compatible_for_backend(SupportedServingBackend::Halogen));
        assert!(!openai_compatible_for_backend(SupportedServingBackend::R9v));
    }

    fn sample_identity() -> ServingIdentity {
        ServingIdentity {
            toolbox_backend: "vllm",
            compute_api: "rocm".to_string(),
            runtime_profile_id: "strix-vllm-latest".to_string(),
            endpoint: "http://127.0.0.1:8002".to_string(),
            openai_compatible: true,
            registered_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn register_then_snapshot_returns_the_entry() {
        let registry = ServingIdentityRegistry::default();
        registry.register("brainrouter-vllm-server", sample_identity()).await;
        let snap = registry.snapshot().await;
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].container_name, "brainrouter-vllm-server");
        assert_eq!(snap[0].identity.toolbox_backend, "vllm");
        // No real podman on this host, so container_running is always
        // false here — asserting it's present, not asserting a specific
        // podman-dependent value.
        let _ = snap[0].container_running;
    }

    #[tokio::test]
    async fn register_overwrites_same_container_name() {
        let registry = ServingIdentityRegistry::default();
        registry.register("brainrouter-vllm-server", sample_identity()).await;
        let mut second = sample_identity();
        second.endpoint = "http://127.0.0.1:9999".to_string();
        registry.register("brainrouter-vllm-server", second).await;
        let snap = registry.snapshot().await;
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].identity.endpoint, "http://127.0.0.1:9999");
    }

    #[tokio::test]
    async fn deregister_removes_the_entry() {
        let registry = ServingIdentityRegistry::default();
        registry.register("brainrouter-vllm-server", sample_identity()).await;
        registry.deregister("brainrouter-vllm-server").await;
        assert!(registry.snapshot().await.is_empty());
    }

    #[tokio::test]
    async fn deregister_of_absent_entry_is_a_no_op() {
        let registry = ServingIdentityRegistry::default();
        registry.deregister("brainrouter-does-not-exist").await;
        assert!(registry.snapshot().await.is_empty());
    }
}
