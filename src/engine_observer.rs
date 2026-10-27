//! External-engine Residency observer (v2 L5).
//!
//! Icon-launched `strix_server` / `hipfire` servers must be VISIBLE to the
//! supervisor: attach-not-spawn (Ready-at-external), budget-counted, and
//! structurally un-killable by a cutover. The observer is read-only:
//!
//! * `/proc` scan for the two engine process names (a pid source seam; never
//!   a signal target);
//! * **classification precedence**: a pid present in the native adoption
//!   markers (or our own registry) is native, never double-listed external
//!   (an adopted resident listed external would be un-swappable);
//! * endpoint from the process argv `--port` (hipfire defaults to 8080);
//! * `/v1/models` probed for the served stem set, then the gguf/stem matched
//!   against the swap catalog so the external resident lands under the same
//!   catalog id the router uses.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;

use crate::residency::{Resident, ResidentSource};
use crate::swap_catalog::SwapCatalog;

/// Recognized engine process names (argv or comm substring).
pub const ENGINE_PROCESSES: &[(&str, &str)] = &[("strix_server", "strixite"), ("hipfire", "hipfire")];

#[derive(Debug, Clone)]
pub struct ProcInfo {
    pub pid: u32,
    pub cmdline: String,
    pub comm: String,
}

/// `/proc`-backed pid source. Injectable so the observer never needs a real
/// icon server on the fixture tier.
pub trait ProcSource: Send + Sync {
    fn scan(&self) -> Vec<ProcInfo>;
}

pub struct ProcFsSource;

impl ProcSource for ProcFsSource {
    fn scan(&self) -> Vec<ProcInfo> {
        let mut out = Vec::new();
        let Ok(entries) = std::fs::read_dir("/proc") else {
            return out;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(pid) = name.to_str().and_then(|n| n.parse::<u32>().ok()) else {
                continue;
            };
            let base = entry.path();
            let cmdline = std::fs::read(base.join("cmdline"))
                .map(|raw| {
                    raw.split(|b| *b == 0)
                        .map(|part| String::from_utf8_lossy(part).into_owned())
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_default();
            let comm = std::fs::read_to_string(base.join("comm"))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            out.push(ProcInfo { pid, cmdline, comm });
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ObservedEngine {
    pub pid: u32,
    pub engine: String,
    pub endpoint: String,
    pub port: u16,
    pub stems: Vec<String>,
    pub catalog_id: Option<String>,
    pub gguf: Option<String>,
    /// True when the pid is in the native adoption registry (excluded from
    /// the external list; present here only for diagnostics).
    pub native: bool,
}

/// Classify one process as a known engine, if it is one.
pub fn classify(info: &ProcInfo) -> Option<(String, u16, Option<String>)> {
    let haystack = format!("{} {}", info.comm, info.cmdline);
    for (needle, engine) in ENGINE_PROCESSES {
        if !haystack.contains(needle) {
            continue;
        }
        // Never classify the daemon itself (its argv could contain a path that
        // mentions the engine).
        if info.pid == std::process::id() {
            return None;
        }
        let default_port = if *engine == "hipfire" { 8080 } else { 0 };
        let port = parse_port(&info.cmdline).unwrap_or(default_port);
        let weights = parse_flag_value(&info.cmdline, &["--weights", "--model", "-m"]);
        return Some((engine.to_string(), port, weights));
    }
    None
}

pub fn parse_port(cmdline: &str) -> Option<u16> {
    parse_flag_value(cmdline, &["--port"])
        .and_then(|value| value.parse::<u16>().ok())
        .or_else(|| {
            cmdline
                .split_whitespace()
                .find_map(|token| token.strip_prefix("--port="))
                .and_then(|value| value.parse::<u16>().ok())
        })
}

pub fn parse_flag_value(cmdline: &str, flags: &[&str]) -> Option<String> {
    let tokens: Vec<&str> = cmdline.split_whitespace().collect();
    for (index, token) in tokens.iter().enumerate() {
        if flags.contains(token) {
            if let Some(value) = tokens.get(index + 1) {
                if !value.starts_with('-') {
                    return Some(value.trim_matches('"').to_string());
                }
            }
        }
        for flag in flags {
            if let Some(value) = token.strip_prefix(&format!("{flag}=")) {
                if !value.is_empty() {
                    return Some(value.trim_matches('"').to_string());
                }
            }
        }
    }
    None
}

/// Pids already owned by the supervisor (adoption markers). Classification
/// precedence: these are native, never external.
pub fn marker_pids(marker_dir: &Path) -> BTreeSet<u32> {
    let mut pids = BTreeSet::new();
    let Ok(entries) = std::fs::read_dir(marker_dir) else {
        return pids;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        if let Some(pid) = json.get("pid").and_then(|p| p.as_u64()) {
            if pid > 0 {
                pids.insert(pid as u32);
            }
        }
    }
    pids
}

/// Injectable `/v1/models` probe (endpoint -> served stems).
pub type ProbeFn = Arc<dyn Fn(&str) -> Option<Vec<String>> + Send + Sync>;

/// Probe `/v1/models` on an endpoint (sync, short timeout).
pub fn probe_models(endpoint: &str) -> Option<Vec<String>> {
    crate::engine_process::probe_stems(endpoint, Duration::from_millis(600))
}

/// The observer: proc source + probe + catalog + marker dir.
pub struct ExternalObserver {
    source: Arc<dyn ProcSource>,
    marker_dir: std::path::PathBuf,
    catalog: Arc<SwapCatalog>,
    probe: ProbeFn,
    /// Pids the scheduler's own registry knows (belt and braces beside the
    /// markers).
    native_pids: std::sync::Mutex<BTreeSet<u32>>,
}

impl ExternalObserver {
    pub fn new(
        source: Arc<dyn ProcSource>,
        marker_dir: std::path::PathBuf,
        catalog: Arc<SwapCatalog>,
    ) -> Self {
        Self {
            source,
            marker_dir,
            catalog,
            probe: Arc::new(probe_models),
            native_pids: std::sync::Mutex::new(BTreeSet::new()),
        }
    }

    pub fn with_probe(
        mut self,
        probe: ProbeFn,
    ) -> Self {
        self.probe = probe;
        self
    }

    /// Scheduler-owned pids (its live runtime handles), merged with markers.
    pub fn set_native_pids(&self, pids: impl IntoIterator<Item = u32>) {
        *self
            .native_pids
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = pids.into_iter().collect();
    }

    /// Read-only observation. External residents only (native pids filtered).
    pub fn observe(&self) -> Vec<ObservedEngine> {
        let markers = marker_pids(&self.marker_dir);
        let scheduler_pids = self
            .native_pids
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let data = self.catalog.snapshot();
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        for info in self.source.scan() {
            if !seen.insert(info.pid) {
                continue;
            }
            let Some((engine, port, weights)) = classify(&info) else {
                continue;
            };
            let native = markers.contains(&info.pid) || scheduler_pids.contains(&info.pid);
            if native {
                continue;
            }
            let endpoint = format!("http://127.0.0.1:{port}");
            let stems = if port == 0 {
                Vec::new()
            } else {
                (self.probe)(&endpoint).unwrap_or_default()
            };
            let catalog_id = resolve_catalog_id(&data, weights.as_deref(), &stems);
            out.push(ObservedEngine {
                pid: info.pid,
                engine,
                endpoint,
                port,
                stems,
                catalog_id,
                gguf: weights,
                native: false,
            });
        }
        out
    }

    /// The external slot of the Residency view.
    pub fn residents(&self, observed: &[ObservedEngine]) -> Vec<Resident> {
        observed
            .iter()
            .map(|engine| Resident {
                catalog_id: engine
                    .catalog_id
                    .clone()
                    .or_else(|| engine.stems.first().cloned())
                    .unwrap_or_else(|| format!("{}:pid{}", engine.engine, engine.pid)),
                source: ResidentSource::External,
                endpoint: engine.endpoint.clone(),
                gguf_path: engine.gguf.clone(),
                started_at: 0,
                last_stream_completed: 0,
                state: "ready".into(),
                spec_stale: false,
                engine: Some(engine.engine.clone()),
            })
            .collect()
    }

    /// Refresh the scheduler's external slot + detail rows in one call shape.
    pub fn snapshot(&self) -> (Vec<Resident>, Vec<ObservedEngine>) {
        let observed = self.observe();
        let residents = self.residents(&observed);
        (residents, observed)
    }
}

/// Resolve an external process to a catalog id via its weights path or its
/// served stems (entry id, served alias, alias list).
pub fn resolve_catalog_id(
    data: &crate::swap_catalog::SwapCatalogData,
    weights: Option<&str>,
    stems: &[String],
) -> Option<String> {
    let weights_base = weights.and_then(|w| Path::new(w).file_name()).map(|n| n.to_string_lossy().into_owned());
    for entry in data.entries.values() {
        if let Some(base) = &weights_base {
            if entry
                .files
                .iter()
                .any(|file| Path::new(file).file_name().map(|n| n.to_string_lossy().into_owned()) == Some(base.clone()))
            {
                return Some(entry.id.clone());
            }
            if entry
                .canonical_gguf()
                .and_then(|path| path.file_name().map(|n| n.to_string_lossy().into_owned()))
                .as_ref()
                == Some(base)
            {
                return Some(entry.id.clone());
            }
        }
        if stems.iter().any(|stem| {
            stem == &entry.id
                || stem == &entry.served_alias()
                || entry.aliases.iter().any(|alias| alias == stem)
        }) {
            return Some(entry.id.clone());
        }
    }
    None
}

/// Group observed processes by engine (Engines tab).
pub fn by_engine(observed: &[ObservedEngine]) -> BTreeMap<String, Vec<u32>> {
    let mut grouped: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    for engine in observed {
        grouped
            .entry(engine.engine.clone())
            .or_default()
            .push(engine.pid);
    }
    grouped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swap_catalog::{import_llama_swap_config, SwapCatalog};

    struct FixtureProcs(Vec<ProcInfo>);
    impl ProcSource for FixtureProcs {
        fn scan(&self) -> Vec<ProcInfo> {
            self.0.clone()
        }
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "brainrouter-observer-{name}-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn catalog() -> Arc<SwapCatalog> {
        let path = temp_dir("catalog").join("swap_catalog.json");
        let catalog = SwapCatalog::load(path).unwrap();
        catalog
            .apply_import(
                &import_llama_swap_config(
                    r#"
models:
  swift-card:
    cmd: llama-server --port ${PORT} --model /models/swift.mq4
"#,
                )
                .unwrap(),
            )
            .unwrap();
        Arc::new(catalog)
    }

    #[test]
    fn classifies_engine_port_weights_and_skips_self() {
        let info = ProcInfo {
            pid: 999_001,
            cmdline: "/home/papa/.local/bin/strix_server --config /x.conf --weights /models/a.gguf --port 5941".into(),
            comm: "strix_server".into(),
        };
        let (engine, port, weights) = classify(&info).unwrap();
        assert_eq!(engine, "strixite");
        assert_eq!(port, 5941);
        assert_eq!(weights.as_deref(), Some("/models/a.gguf"));

        let hip = ProcInfo {
            pid: 999_002,
            cmdline: "/usr/bin/hipfire serve --model /models/swift.mq4".into(),
            comm: "hipfire".into(),
        };
        let (engine, port, weights) = classify(&hip).unwrap();
        assert_eq!(engine, "hipfire");
        assert_eq!(port, 8080, "hipfire single-owner default");
        assert_eq!(weights.as_deref(), Some("/models/swift.mq4"));

        let self_proc = ProcInfo {
            pid: std::process::id(),
            cmdline: "brainrouter serve --engine strix_server".into(),
            comm: "brainrouter".into(),
        };
        assert!(classify(&self_proc).is_none());
    }

    #[test]
    fn native_marker_pids_are_never_double_listed_as_external() {
        let dir = temp_dir("markers");
        std::fs::write(
            dir.join("adopted.json"),
            serde_json::json!({"model": "a", "pid": 424_242, "endpoint": "http://127.0.0.1:5941"}).to_string(),
        )
        .unwrap();
        let observer = ExternalObserver::new(
            Arc::new(FixtureProcs(vec![
                ProcInfo {
                    pid: 424_242,
                    cmdline: "strix_server --port 5941 --weights /models/a.gguf".into(),
                    comm: "strix_server".into(),
                },
                ProcInfo {
                    pid: 424_243,
                    cmdline: "strix_server --port 5942 --weights /models/swift.mq4".into(),
                    comm: "strix_server".into(),
                },
            ])),
            dir,
            catalog(),
        )
        .with_probe(Arc::new(|_| Some(vec!["swift-card".to_string()])));
        let observed = observer.observe();
        assert_eq!(observed.len(), 1, "{observed:?}");
        assert_eq!(observed[0].pid, 424_243);
        assert_eq!(observed[0].catalog_id.as_deref(), Some("swift-card"));
        let residents = observer.residents(&observed);
        assert_eq!(residents.len(), 1);
        assert_eq!(residents[0].source, ResidentSource::External);
        assert_eq!(residents[0].engine.as_deref(), Some("strixite"));
    }

    #[test]
    fn scheduler_owned_pids_are_also_filtered() {
        let dir = temp_dir("scheduler-pids");
        let observer = ExternalObserver::new(
            Arc::new(FixtureProcs(vec![ProcInfo {
                pid: 515_151,
                cmdline: "hipfire serve --port 8080".into(),
                comm: "hipfire".into(),
            }])),
            dir,
            catalog(),
        )
        .with_probe(Arc::new(|_| Some(vec!["swift-card".to_string()])));
        observer.set_native_pids([515_151]);
        assert!(observer.observe().is_empty());
    }

    #[test]
    fn catalog_id_resolves_from_weights_then_stems() {
        let data = catalog().snapshot();
        assert_eq!(
            resolve_catalog_id(&data, Some("/models/swift.mq4"), &[]).as_deref(),
            Some("swift-card")
        );
        assert_eq!(
            resolve_catalog_id(&data, None, &["swift-card".to_string()]).as_deref(),
            Some("swift-card")
        );
        assert!(resolve_catalog_id(&data, Some("/models/other.gguf"), &[]).is_none());
    }
}
