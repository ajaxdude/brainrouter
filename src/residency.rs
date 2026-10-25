//! Residency view (v2 L4) — the input side of every budget/eviction/spawn
//! decision.
//!
//! A [`Resident`] is a model that is provably serving *somewhere*:
//!
//! * **shim** — llama-swap `/running` (every entry, not just `running[0]`),
//!   canonical GGUF + key resolved through the imported catalog;
//! * **native** — the swap supervisor's own live children;
//! * **container** — server-mode serving identities (ds4/gufo);
//! * **external** — the L5 external-engine observer's slot (strixite/hipfire
//!   pid-scoped cutover); L4 ships the registry, L5 feeds it.
//!
//! The scheduler asks [`ResidencyView::ready_for`] FIRST: a resident model is
//! Ready-at-that-endpoint and is attached — no native spawn and **no budget
//! gate** (budget admission applies only to ensures that would spawn).

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::serving_identity::ServingIdentityView;
use crate::swap_catalog::SwapCatalog;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidentSource {
    Shim,
    Native,
    Container,
    External,
}

impl ResidentSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Shim => "shim",
            Self::Native => "native",
            Self::Container => "container",
            Self::External => "external",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Resident {
    pub catalog_id: String,
    pub source: ResidentSource,
    /// Base OpenAI-compatible endpoint (`http://host:port`).
    pub endpoint: String,
    #[serde(default)]
    pub gguf_path: Option<String>,
    /// Unix millis the resident was first observed (best effort).
    #[serde(default)]
    pub started_at: u64,
    #[serde(default)]
    pub last_stream_completed: u64,
    /// `ready` | `loading` | `stopped` (source vocabulary).
    #[serde(default)]
    pub state: String,
    /// Adopted native child whose resolved launch spec no longer matches the
    /// catalog (upgrade changed args) — badge only, never killed.
    #[serde(default)]
    pub spec_stale: bool,
    /// L5: the engine adapter vocabulary for external residents
    /// (`strixite` | `hipfire`); `None` for shim/native/container rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
}

impl Resident {
    pub fn is_ready(&self) -> bool {
        matches!(self.state.as_str(), "ready" | "serving" | "running" | "")
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResidencyView {
    pub residents: Vec<Resident>,
    /// Last shim poll error, surfaced in the Engines tab (never fatal).
    #[serde(default)]
    pub shim_error: Option<String>,
    #[serde(default)]
    pub observed_at: String,
}

impl ResidencyView {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, resident: Resident) {
        self.residents.push(resident);
    }

    /// Ready-at-endpoint answer. Matches, in order: catalog id, canonical GGUF
    /// path, served alias. Returns the first READY resident.
    pub fn ready_for(
        &self,
        catalog_id: &str,
        canonical_gguf: Option<&Path>,
        served_alias: Option<&str>,
    ) -> Option<&Resident> {
        let canonical = canonical_gguf.and_then(|p| std::fs::canonicalize(p).ok());
        self.residents.iter().find(|resident| {
            resident.is_ready()
                && (resident.catalog_id == catalog_id
                    || served_alias.is_some_and(|alias| resident.catalog_id == alias)
                    || match (&canonical, &resident.gguf_path) {
                        (Some(want), Some(have)) => {
                            std::fs::canonicalize(have)
                                .map(|have| have == *want)
                                .unwrap_or_else(|_| Path::new(have) == want.as_path())
                        }
                        _ => false,
                    })
        })
    }

    /// Any resident (ready or loading) for a catalog id.
    pub fn resident_for(&self, catalog_id: &str) -> Option<&Resident> {
        self.residents
            .iter()
            .find(|resident| resident.catalog_id == catalog_id)
    }

    pub fn source_counts(&self) -> Vec<(String, usize)> {
        let mut counts: Vec<(String, usize)> = Vec::new();
        for resident in &self.residents {
            let name = resident.source.as_str().to_string();
            match counts.iter_mut().find(|(n, _)| *n == name) {
                Some((_, c)) => *c += 1,
                None => counts.push((name, 1)),
            }
        }
        counts
    }
}

/// Parse llama-swap `/running` into shim residents. Every entry is considered
/// (`running[0]` was the v1 poller's single-model assumption). The canonical
/// GGUF comes from the entry's `cmd`; the catalog id from the imported catalog
/// (model key or GGUF match), falling back to the wire model name.
pub fn observe_shim(running: &serde_json::Value, catalog: &SwapCatalog) -> Vec<Resident> {
    let Some(entries) = running.get("running").and_then(|r| r.as_array()) else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let wire_model = entry.get("model").and_then(|m| m.as_str()).unwrap_or("");
            let endpoint = entry
                .get("proxy")
                .and_then(|p| p.as_str())
                .unwrap_or("")
                .trim_end_matches('/')
                .to_string();
            let state = entry
                .get("state")
                .and_then(|s| s.as_str())
                .unwrap_or("ready")
                .to_string();
            let gguf = entry
                .get("cmd")
                .and_then(|c| c.as_str())
                .and_then(gguf_from_cmd);
            let resolved = catalog
                .resolve_alias(wire_model)
                .or_else(|| {
                    gguf.as_ref().and_then(|path| {
                        let canonical = std::fs::canonicalize(path).ok();
                        catalog.entries().into_iter().find(|e| {
                            e.files.iter().any(|f| {
                                let fp = std::fs::canonicalize(f).unwrap_or_else(|_| Path::new(f).to_path_buf());
                                canonical.as_ref() == Some(&fp) || Path::new(f) == path
                            })
                        })
                        .map(|e| e.id)
                    })
                })
                .unwrap_or_else(|| wire_model.to_string());
            if resolved.is_empty() {
                return None;
            }
            let entry_meta = catalog.entry(&resolved);
            Some(Resident {
                catalog_id: resolved,
                source: ResidentSource::Shim,
                endpoint,
                gguf_path: gguf.map(|p| p.display().to_string()),
                started_at: 0,
                last_stream_completed: entry_meta
                    .as_ref()
                    .map(|e| e.last_stream_completed)
                    .unwrap_or(0),
                state,
                spec_stale: false,
                engine: None,
            })
        })
        .collect()
}

fn gguf_from_cmd(cmd: &str) -> Option<std::path::PathBuf> {
    let tokens = shlex::split(cmd).unwrap_or_else(|| {
        cmd.split_whitespace().map(str::to_string).collect()
    });
    let mut iter = tokens.iter();
    while let Some(token) = iter.next() {
        if token == "--model" || token == "-m" {
            if let Some(path) = iter.next() {
                return Some(std::path::PathBuf::from(path));
            }
        }
    }
    None
}

/// Poll llama-swap `/running`. Network failure is reported, never fatal (the
/// shim is one source among four).
pub async fn poll_shim(
    client: &reqwest::Client,
    llama_swap_url: &str,
    catalog: &SwapCatalog,
) -> ResidencyView {
    let mut view = ResidencyView {
        observed_at: chrono::Utc::now().to_rfc3339(),
        ..ResidencyView::default()
    };
    match client
        .get(format!("{}/running", llama_swap_url.trim_end_matches('/')))
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
    {
        Ok(response) => match response.json::<serde_json::Value>().await {
            Ok(json) => view.residents = observe_shim(&json, catalog),
            Err(e) => view.shim_error = Some(format!("llama-swap /running decode: {e}")),
        },
        Err(e) => view.shim_error = Some(format!("llama-swap /running: {e}")),
    }
    view
}

/// Container residents from the existing server-mode serving-identity registry.
pub fn observe_containers(identities: &[ServingIdentityView]) -> Vec<Resident> {
    identities
        .iter()
        .filter(|view| view.container_running)
        .map(|view| Resident {
            catalog_id: view.identity.served_model.clone(),
            source: ResidentSource::Container,
            endpoint: view.identity.endpoint.trim_end_matches('/').to_string(),
            gguf_path: None,
            started_at: view.identity.registered_at.timestamp_millis() as u64,
            last_stream_completed: 0,
            state: "ready".into(),
            spec_stale: false,
            engine: None,
        })
        .collect()
}

/// The external-engine slot (L5 observer feeds this; L4 ships the registry).
#[derive(Default)]
pub struct ExternalResidencyRegistry {
    inner: std::sync::Mutex<Vec<Resident>>,
}

impl ExternalResidencyRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn replace(&self, residents: Vec<Resident>) {
        *self.inner.lock().unwrap_or_else(|e| e.into_inner()) = residents;
    }

    pub fn snapshot(&self) -> Vec<Resident> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

/// Compose one view from all sources. `native` is provided by the scheduler
/// (its own registry), containers by the caller, externals by the L5 slot.
pub fn compose(
    shim: ResidencyView,
    native: Vec<Resident>,
    containers: Vec<Resident>,
    external: Vec<Resident>,
) -> ResidencyView {
    let mut view = ResidencyView {
        observed_at: if shim.observed_at.is_empty() {
            chrono::Utc::now().to_rfc3339()
        } else {
            shim.observed_at
        },
        shim_error: shim.shim_error,
        residents: Vec::new(),
    };
    view.residents.extend(shim.residents);
    view.residents.extend(native);
    view.residents.extend(containers);
    view.residents.extend(external);
    view
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swap_catalog::{import_llama_swap_config, SwapCatalog};
    use serde_json::json;

    fn catalog() -> SwapCatalog {
        let path = std::env::temp_dir().join(format!(
            "brainrouter-residency-catalog-{}.json",
            uuid::Uuid::new_v4()
        ));
        let catalog = SwapCatalog::load(path).unwrap();
        catalog
            .apply_import(
                &import_llama_swap_config(
                    r#"
models:
  model-a:
    cmd: |
      /usr/bin/llama-server --port ${PORT} --model /models/a.gguf
"#,
                )
                .unwrap(),
            )
            .unwrap();
        catalog
    }

    #[test]
    fn shim_resident_answers_ready_without_a_native_spawn() {
        let catalog = catalog();
        let running = json!({"running": [{
            "model": "model-a",
            "state": "ready",
            "proxy": "http://127.0.0.1:5801",
            "cmd": "/usr/bin/llama-server --port 5801 --model /models/a.gguf"
        }]});
        let view = observe_shim(&running, &catalog);
        assert_eq!(view.len(), 1);
        assert_eq!(view[0].source, ResidentSource::Shim);
        assert_eq!(view[0].endpoint, "http://127.0.0.1:5801");
        assert_eq!(
            view[0].gguf_path.as_deref(),
            Some("/models/a.gguf")
        );
        let composed = compose(
            ResidencyView {
                residents: view,
                ..ResidencyView::default()
            },
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        // Residency-before-budget: this answer is what makes ensure attach.
        assert!(composed
            .ready_for("model-a", Some(Path::new("/models/a.gguf")), Some("model-a"))
            .is_some());
    }

    #[test]
    fn every_running_entry_is_observed_past_running_zero() {
        let catalog = catalog();
        let running = json!({"running": [
            {"model": "model-a", "state": "ready", "proxy": "http://127.0.0.1:1", "cmd": "llama-server --model /models/a.gguf"},
            {"model": "other", "state": "starting", "proxy": "http://127.0.0.1:2", "cmd": "llama-server --model /models/b.gguf"}
        ]});
        let view = observe_shim(&running, &catalog);
        assert_eq!(view.len(), 2);
        assert_eq!(view[1].catalog_id, "other");
        assert_eq!(view[1].state, "starting");
        assert!(!view[1].is_ready());
        // Loading residents are visible but do not answer "ready".
        let composed = ResidencyView {
            residents: view,
            ..ResidencyView::default()
        };
        assert!(composed.ready_for("other", None, None).is_none());
    }

    #[test]
    fn gguf_match_resolves_an_unimported_wire_alias() {
        let catalog = catalog();
        let running = json!({"running": [{
            "model": "a-alias-on-the-wire",
            "state": "ready",
            "proxy": "http://127.0.0.1:3",
            "cmd": "llama-server --model /models/a.gguf"
        }]});
        let view = observe_shim(&running, &catalog);
        assert_eq!(view[0].catalog_id, "model-a");
    }

    #[test]
    fn containers_only_count_when_running() {
        use crate::serving_identity::ServingIdentity;
        let identity = ServingIdentity {
            toolbox_backend: "gufo",
            compute_api: "rocm".into(),
            runtime_profile_id: "p".into(),
            served_model: "gufo-model".into(),
            endpoint: "http://127.0.0.1:8002".into(),
            openai_compatible: true,
            registered_at: chrono::Utc::now(),
        };
        let running = ServingIdentityView {
            container_name: "c".into(),
            identity: identity.clone(),
            container_running: true,
        };
        let stopped = ServingIdentityView {
            container_name: "c2".into(),
            identity,
            container_running: false,
        };
        let residents = observe_containers(&[running, stopped]);
        assert_eq!(residents.len(), 1);
        assert_eq!(residents[0].source, ResidentSource::Container);
        assert_eq!(residents[0].endpoint, "http://127.0.0.1:8002");
    }

    #[test]
    fn external_slot_round_trips() {
        let registry = ExternalResidencyRegistry::new();
        registry.replace(vec![Resident {
            catalog_id: "hipfire-model".into(),
            source: ResidentSource::External,
            endpoint: "http://127.0.0.1:9000".into(),
            gguf_path: None,
            started_at: 0,
            last_stream_completed: 0,
            state: "ready".into(),
            spec_stale: false,
            engine: Some("hipfire".into()),
        }]);
        assert_eq!(registry.snapshot().len(), 1);
        assert_eq!(registry.snapshot()[0].catalog_id, "hipfire-model");
    }
}
