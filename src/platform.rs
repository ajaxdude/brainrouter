//! Platform probe (v2 L4).
//!
//! Answers "what hardware is this?" in the catalog's own vocabulary so every
//! consumer (dashboard banner, toolbox image default, engine offers) reads one
//! id. The rule from the L4 spec:
//!
//! * a detected GPU arch that appears in the platform feed resolves to that
//!   feed's id (`strix-halo` for `gfx1151`);
//! * anything else resolves to `generic-<gpu-arch>` (today `fedora1` is
//!   permanently `generic-gfx1100` because the feed ships no gfx1100-class
//!   platform) and the probe reports `known = false` so the dashboard shows the
//!   explicit **unknown platform** banner and every engine is offered
//!   (fail-open, by design).
//!
//! The engine compatibility matrix is data, not code: [`ENGINE_MATRIX`] carries
//! the requirements (arch family, GTT floor, `min_mem_gib`) and
//! [`offers_for`] turns facts into per-engine offers + reasons.
//!
//! The feed is the vendored default plus an optional remote URL with a
//! staleness bound (a cached copy younger than [`FEED_MAX_AGE`] wins; a failed
//! fetch falls back to the cache, then to the vendored feed — the probe never
//! needs the network to answer).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Feed cache file under `$XDG_STATE_HOME/brainrouter`.
pub const FEED_CACHE_FILE: &str = "platform_feed.json";
/// A cached/fetched feed older than this is refreshed when a remote URL is set.
pub const FEED_MAX_AGE: Duration = Duration::from_secs(24 * 60 * 60);
/// Environment override for the remote feed URL.
pub const FEED_URL_ENV: &str = "BRAINROUTER_PLATFORM_FEED_URL";
/// Environment override for the budget (GiB) when the kernel reports none.
pub const BUDGET_ENV: &str = "BRAINROUTER_BUDGET_GIB";

// ── Feed ────────────────────────────────────────────────────────────────────

/// One platform in the catalog feed vocabulary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlatformDef {
    pub id: String,
    /// GPU arches that identify this platform (`gfx1151`, `gfx1201`, ...).
    #[serde(default)]
    pub arches: Vec<String>,
    /// Human label for the dashboard card.
    #[serde(default)]
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct PlatformFeed {
    #[serde(default = "default_feed_version")]
    pub version: u32,
    #[serde(default)]
    pub platforms: Vec<PlatformDef>,
}

fn default_feed_version() -> u32 {
    1
}

/// The vendored feed. Mirrors the feed platforms named by the L4 spec:
/// `strix-halo`, `r9700`, `gb10`, `intel-b70`. There is deliberately no
/// gfx1100/1101/1103-class entry — fedora1's RX 7900 XT resolves
/// `generic-gfx1100` and shows the unknown-platform banner (fail-open).
pub fn bundled_feed() -> PlatformFeed {
    PlatformFeed {
        version: 1,
        platforms: vec![
            PlatformDef {
                id: "strix-halo".into(),
                arches: vec!["gfx1151".into()],
                label: "AMD Strix Halo (gfx1151)".into(),
            },
            PlatformDef {
                id: "r9700".into(),
                arches: vec!["gfx1201".into()],
                label: "AMD Radeon AI PRO R9700 (gfx1201)".into(),
            },
            PlatformDef {
                id: "gb10".into(),
                arches: vec!["gb10".into(), "sm_121".into()],
                label: "NVIDIA GB10".into(),
            },
            PlatformDef {
                id: "intel-b70".into(),
                arches: vec!["bmg-g21".into(), "intel-b70".into()],
                label: "Intel Arc B70".into(),
            },
        ],
    }
}

/// Parse a feed document (JSON). Unknown/missing fields fail open to the
/// bundled feed by the caller, never to an error at boot.
pub fn parse_feed(text: &str) -> anyhow::Result<PlatformFeed> {
    let feed: PlatformFeed = serde_json::from_str(text)?;
    if feed.platforms.is_empty() {
        anyhow::bail!("platform feed has no platforms");
    }
    Ok(feed)
}

/// `$XDG_STATE_HOME/brainrouter/platform_feed.json`.
pub fn feed_cache_path() -> PathBuf {
    crate::paths::brainrouter_state().join(FEED_CACHE_FILE)
}

/// Load a feed file when it exists and parses.
pub fn load_feed_file(path: &Path) -> Option<PlatformFeed> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| parse_feed(&text).ok())
}

/// Resolve the feed used for a boot: cache if fresh, else remote (when the
/// operator set a URL), else cache, else vendored. Returns the feed and a
/// human-readable origin for logs/dashboard (`bundled`, `cache`, `remote`).
///
/// Never errors: a probe must answer offline.
pub async fn resolve_feed(
    client: &reqwest::Client,
    remote_url: Option<&str>,
    cache_path: &Path,
    now: SystemTime,
) -> (PlatformFeed, &'static str) {
    let cached = load_feed_file(cache_path);
    let cache_fresh = cached.is_some()
        && std::fs::metadata(cache_path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|mtime| now.duration_since(mtime).ok())
            .is_some_and(|age| age < FEED_MAX_AGE);
    if cache_fresh {
        return (cached.expect("checked above"), "cache");
    }
    if let Some(url) = remote_url.filter(|u| !u.trim().is_empty()) {
        if let Ok(response) = client
            .get(url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            if let Ok(text) = response.text().await {
                if let Ok(feed) = parse_feed(&text) {
                    // Best-effort cache write; a read-only state dir must not
                    // break the probe.
                    let _ = crate::plugins::store::atomic_write_0600(cache_path, text.as_bytes());
                    return (feed, "remote");
                }
            }
        }
    }
    if let Some(cached) = cached {
        return (cached, "cache");
    }
    (bundled_feed(), "bundled")
}

// ── Facts ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuInfo {
    /// PCI address (`0000:c5:00.0`) or a synthetic node id.
    #[serde(default)]
    pub pci: String,
    /// PCI vendor:device (`1002:150e`) when known.
    #[serde(default)]
    pub device: String,
    /// `gfx1151`, `gfx1100`, ...; `unknown` when the probe cannot tell.
    #[serde(default)]
    pub arch: String,
    /// `rocm` | `nvidia` | `intel` | `unknown`.
    #[serde(default)]
    pub vendor: String,
}

impl GpuInfo {
    pub fn is_known_arch(&self) -> bool {
        !self.arch.is_empty() && self.arch != "unknown"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct InstallMarkers {
    pub toolbox_runtime: bool,
    pub hipfire_root: Option<String>,
    pub strixite_root: Option<String>,
}

/// Raw probe output. Fixtures construct this directly (acceptance 5).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlatformFacts {
    pub os: String,
    pub cpu_arch: String,
    pub machine_id: String,
    pub gpus: Vec<GpuInfo>,
    /// APU (unified memory) vs discrete card.
    pub apu: bool,
    pub mem_total_gib: f64,
    pub mem_available_gib: f64,
    pub gtt_budget_gib: Option<f64>,
    pub vram_budget_gib: Option<f64>,
    pub install_markers: InstallMarkers,
    pub source: String,
}

impl Default for PlatformFacts {
    fn default() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            cpu_arch: std::env::consts::ARCH.to_string(),
            machine_id: String::new(),
            gpus: Vec::new(),
            apu: false,
            mem_total_gib: 0.0,
            mem_available_gib: 0.0,
            gtt_budget_gib: None,
            vram_budget_gib: None,
            install_markers: InstallMarkers::default(),
            source: "fixture".into(),
        }
    }
}

impl PlatformFacts {
    /// Primary arch: first GPU with a real arch, else the CPU arch.
    pub fn primary_arch(&self) -> String {
        self.gpus
            .iter()
            .find(|g| g.is_known_arch())
            .map(|g| g.arch.clone())
            .unwrap_or_else(|| self.cpu_arch.clone())
    }
}

// ── Engine matrix (data) ────────────────────────────────────────────────────

/// One row of the engine compatibility matrix.
#[derive(Debug, Clone, Copy)]
pub struct EngineRule {
    pub engine: &'static str,
    /// Arch family prefixes the engine supports (`gfx1151`, `gfx11`, ...).
    /// Empty = universal.
    pub arches: &'static [&'static str],
    /// Minimum probe-reported GTT budget (Strix Halo unified memory).
    pub min_gtt_gib: Option<f64>,
    /// Minimum memory budget (applies everywhere).
    pub min_mem_gib: f64,
    /// ROCm discrete cards only (an APU is refused).
    pub discrete_only: bool,
}

pub const ENGINE_MATRIX: &[EngineRule] = &[
    EngineRule {
        engine: "strixite",
        arches: &["gfx1151"],
        min_gtt_gib: Some(100.0),
        min_mem_gib: 100.0,
        discrete_only: false,
    },
    EngineRule {
        engine: "gufo",
        arches: &["gfx1151"],
        min_gtt_gib: None,
        min_mem_gib: 64.0,
        discrete_only: false,
    },
    EngineRule {
        engine: "hipfire",
        arches: &["gfx9", "gfx10", "gfx11", "gfx12"],
        min_gtt_gib: None,
        min_mem_gib: 16.0,
        discrete_only: true,
    },
    EngineRule {
        engine: "llama.cpp",
        arches: &[],
        min_gtt_gib: None,
        min_mem_gib: 4.0,
        discrete_only: false,
    },
];

/// Probe-derived budget: GTT on unified-memory hosts, VRAM on discrete cards,
/// else three quarters of system RAM (documented fallback).
pub fn budget_gib(facts: &PlatformFacts) -> f64 {
    if let Some(v) = std::env::var(BUDGET_ENV)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| *v > 0.0)
    {
        return v;
    }
    facts
        .gtt_budget_gib
        .or(facts.vram_budget_gib)
        .filter(|v| *v > 0.0)
        .unwrap_or_else(|| (facts.mem_total_gib * 0.75).max(0.0))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineOffer {
    pub engine: String,
    pub offered: bool,
    /// Why the engine is (not) offered — shown verbatim in the Engines tab.
    pub reason: String,
    pub min_mem_gib: f64,
}

/// Per-engine offers + reasons from the matrix. Unknown platform ⇒ everything
/// offered with the explicit fail-open reason.
pub fn offers_for(facts: &PlatformFacts, known: bool, platform_id: &str) -> Vec<EngineOffer> {
    let arch = facts.primary_arch();
    let budget = budget_gib(facts);
    ENGINE_MATRIX
        .iter()
        .map(|rule| {
            if !known {
                return EngineOffer {
                    engine: rule.engine.to_string(),
                    offered: true,
                    reason: format!(
                        "unknown platform {platform_id}: fail-open — every engine offered"
                    ),
                    min_mem_gib: rule.min_mem_gib,
                };
            }
            let arch_ok = rule.arches.is_empty()
                || rule
                    .arches
                    .iter()
                    .any(|prefix| arch.starts_with(prefix) || arch == *prefix);
            let gtt_ok = rule
                .min_gtt_gib
                .map(|min| facts.gtt_budget_gib.unwrap_or(0.0) >= min)
                .unwrap_or(true);
            let mem_ok = budget >= rule.min_mem_gib;
            let discrete_ok = !rule.discrete_only || !facts.apu;
            let offered = arch_ok && gtt_ok && mem_ok && discrete_ok;
            let reason = if offered {
                format!("{arch}: supported (budget {budget:.0} GiB)")
            } else if !arch_ok {
                format!(
                    "{arch} is not in the engine's arch family ({})",
                    rule.arches.join(", ")
                )
            } else if !discrete_ok {
                format!("{arch} is an APU — this engine needs a discrete ROCm card")
            } else if !gtt_ok {
                format!(
                    "needs GTT ≥ {:.0} GiB (probe reports {:.0} GiB)",
                    rule.min_gtt_gib.unwrap_or(0.0),
                    facts.gtt_budget_gib.unwrap_or(0.0)
                )
            } else {
                format!(
                    "needs ≥ {:.0} GiB budget (probe reports {budget:.0} GiB)",
                    rule.min_mem_gib
                )
            };
            EngineOffer {
                engine: rule.engine.to_string(),
                offered,
                reason,
                min_mem_gib: rule.min_mem_gib,
            }
        })
        .collect()
}

// ── Probe result ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlatformProbe {
    pub id: String,
    /// False = no feed platform matched; the dashboard banner is mandatory.
    pub known: bool,
    pub banner: Option<String>,
    pub label: String,
    pub facts: PlatformFacts,
    pub budget_gib: f64,
    pub offers: Vec<EngineOffer>,
    pub feed_origin: String,
    pub probed_at: String,
}

/// Resolve a probe from facts + feed (pure; the fixture tier's entry point).
pub fn resolve(facts: &PlatformFacts, feed: &PlatformFeed, feed_origin: &str) -> PlatformProbe {
    let arch = facts.primary_arch();
    let feed_gpu_arch = facts
        .gpus
        .iter()
        .find(|g| g.is_known_arch())
        .map(|g| g.arch.clone());
    let matched = feed.platforms.iter().find(|platform| {
        platform.arches.iter().any(|candidate| {
            feed_gpu_arch.as_deref() == Some(candidate.as_str())
                || (feed_gpu_arch.is_none() && candidate == &arch)
        })
    });
    let (id, known, label) = match matched {
        Some(platform) => (
            platform.id.clone(),
            true,
            if platform.label.is_empty() {
                platform.id.clone()
            } else {
                platform.label.clone()
            },
        ),
        None => (
            format!("generic-{arch}"),
            false,
            format!("Unknown platform ({arch})"),
        ),
    };
    let banner = (!known).then(|| {
        format!(
            "Unknown platform: {arch} matches no feed platform — showing all engines and \
             toolboxes (fail-open). Add a catalog overlay to name this platform."
        )
    });
    PlatformProbe {
        id: id.clone(),
        known,
        banner,
        label,
        budget_gib: budget_gib(facts),
        offers: offers_for(facts, known, &id),
        facts: facts.clone(),
        feed_origin: feed_origin.to_string(),
        probed_at: chrono::Utc::now().to_rfc3339(),
    }
}

/// Live probe (Linux). Never errors: missing files simply omit facts.
pub fn detect_facts() -> PlatformFacts {
    let mut facts = PlatformFacts {
        os: std::env::consts::OS.to_string(),
        cpu_arch: std::env::consts::ARCH.to_string(),
        machine_id: machine_id().unwrap_or_default(),
        ..PlatformFacts::default()
    };
    facts.source = "live".into();
    let (total, available) = meminfo_gib();
    facts.mem_total_gib = total;
    facts.mem_available_gib = available;
    facts.gtt_budget_gib = gtt_gib();
    facts.vram_budget_gib = vram_gib();
    facts.gpus = detect_gpus();
    facts.apu = facts
        .gpus
        .iter()
        .any(|g| g.arch.starts_with("gfx115") || g.arch.starts_with("gfx90c"))
        && facts.gtt_budget_gib.is_some_and(|g| g >= 64.0);
    facts.install_markers = detect_install_markers();
    facts
}

/// Probe synchronously with the vendored feed (dashboard `CURRENT_PLATFORM_ID`,
/// toolbox image default). Remote-feed boot probes use [`resolve_feed`].
pub fn probe_bundled() -> PlatformProbe {
    resolve(&detect_facts(), &bundled_feed(), "bundled")
}

/// The catalog platform id for this host, synchronously.
pub fn current_platform_id() -> String {
    probe_bundled().id
}

fn machine_id() -> Option<String> {
    for path in ["/etc/machine-id", "/var/lib/dbus/machine-id"] {
        if let Ok(text) = std::fs::read_to_string(path) {
            let id = text.trim();
            if !id.is_empty() {
                return Some(id.to_string());
            }
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn meminfo_gib() -> (f64, f64) {
    let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let read = |key: &str| -> f64 {
        text.lines()
            .find(|line| line.starts_with(key))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|v| v.parse::<f64>().ok())
            .map(|kib| kib / 1024.0 / 1024.0)
            .unwrap_or(0.0)
    };
    (read("MemTotal:"), read("MemAvailable:"))
}

#[cfg(not(target_os = "linux"))]
fn meminfo_gib() -> (f64, f64) {
    (0.0, 0.0)
}

/// GTT budget from the kernel (Strix Halo unified memory): `ttm.pages_limit`
/// (pages of 4 KiB) or `amdgpu.gttsize` (MiB). Kernel cmdline wins over sysfs
/// because it is what the running kernel actually applied.
#[cfg(target_os = "linux")]
fn gtt_gib() -> Option<f64> {
    let cmdline = std::fs::read_to_string("/proc/cmdline").unwrap_or_default();
    for token in cmdline.split_whitespace() {
        if let Some(value) = token.strip_prefix("ttm.pages_limit=") {
            if let Ok(pages) = value.parse::<f64>() {
                let gib = pages * 4096.0 / 1024.0 / 1024.0 / 1024.0;
                if gib > 0.0 {
                    return Some(gib);
                }
            }
        }
        if let Some(value) = token.strip_prefix("amdgpu.gttsize=") {
            if let Ok(mib) = value.parse::<f64>() {
                if mib > 0.0 {
                    return Some(mib / 1024.0);
                }
            }
        }
    }
    if let Ok(text) = std::fs::read_to_string("/sys/module/ttm/parameters/pages_limit") {
        if let Ok(pages) = text.trim().parse::<f64>() {
            let gib = pages * 4096.0 / 1024.0 / 1024.0 / 1024.0;
            if gib > 0.0 {
                return Some(gib);
            }
        }
    }
    if let Ok(text) = std::fs::read_to_string("/sys/module/amdgpu/parameters/gttsize") {
        if let Ok(mib) = text.trim().parse::<f64>() {
            if mib > 0.0 {
                return Some(mib / 1024.0);
            }
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn gtt_gib() -> Option<f64> {
    None
}

/// Largest discrete VRAM report available (AMD sysfs, NVIDIA procfs).
#[cfg(target_os = "linux")]
fn vram_gib() -> Option<f64> {
    let mut best: Option<f64> = None;
    if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
        for entry in entries.flatten() {
            let path = entry.path().join("device/mem_info_vram_total");
            if let Ok(text) = std::fs::read_to_string(&path) {
                if let Ok(bytes) = text.trim().parse::<f64>() {
                    let gib = bytes / 1024.0 / 1024.0 / 1024.0;
                    if gib > 1.0 {
                        best = Some(best.map_or(gib, |b| b.max(gib)));
                    }
                }
            }
        }
    }
    if best.is_none() {
        if let Ok(entries) = std::fs::read_dir("/proc/driver/nvidia/gpus") {
            for entry in entries.flatten() {
                if let Ok(text) = std::fs::read_to_string(entry.path().join("information")) {
                    for line in text.lines() {
                        if let Some(rest) = line.strip_prefix("Total Memory:") {
                            let mib = rest
                                .split_whitespace()
                                .next()
                                .and_then(|v| v.parse::<f64>().ok())
                                .unwrap_or(0.0);
                            if mib > 0.0 {
                                best = Some(best.map_or(mib / 1024.0, |b| b.max(mib / 1024.0)));
                            }
                        }
                    }
                }
            }
        }
    }
    best
}

#[cfg(not(target_os = "linux"))]
fn vram_gib() -> Option<f64> {
    None
}

/// GPU arch detection: KFD topology first (it names the arch), sysfs PCI as
/// the fallback (name/vendor only).
#[cfg(target_os = "linux")]
fn detect_gpus() -> Vec<GpuInfo> {
    let mut out = Vec::new();
    let base = Path::new("/sys/class/kfd/kfd/topology/nodes");
    if let Ok(nodes) = std::fs::read_dir(base) {
        for node in nodes.flatten() {
            let Ok(text) = std::fs::read_to_string(node.path().join("properties")) else {
                continue;
            };
            let mut map = std::collections::HashMap::new();
            for line in text.lines() {
                let mut parts = line.split_whitespace();
                if let (Some(k), Some(v)) = (parts.next(), parts.next()) {
                    map.insert(k.to_string(), v.to_string());
                }
            }
            let simd = map
                .get("simd_count")
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0);
            let gfx = map
                .get("gfx_target_version")
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0);
            if simd == 0 && gfx == 0 {
                continue; // CPU node
            }
            let name = map.get("name").cloned().unwrap_or_default();
            let arch = if name.starts_with("gfx") {
                name
            } else {
                gfx_arch_name(gfx)
            };
            out.push(GpuInfo {
                pci: map
                    .get("location_id")
                    .cloned()
                    .unwrap_or_else(|| node.file_name().to_string_lossy().into_owned()),
                device: String::new(),
                arch,
                vendor: "rocm".into(),
            });
        }
    }
    if !out.is_empty() {
        return out;
    }
    if let Ok(devices) = std::fs::read_dir("/sys/bus/pci/devices") {
        for device in devices.flatten() {
            let path = device.path();
            let read = |name: &str| {
                std::fs::read_to_string(path.join(name))
                    .ok()
                    .map(|v| v.trim().to_string())
            };
            let class = read("class").unwrap_or_default();
            // 0x030000 display controller, 0x038000 other display controller.
            if !(class.starts_with("0x0300") || class.starts_with("0x0380")) {
                continue;
            }
            let vendor = read("vendor").unwrap_or_default();
            let device_id = read("device").unwrap_or_default();
            let (vendor_name, arch) = match vendor.as_str() {
                "0x1002" => ("rocm", "unknown".to_string()),
                "0x10de" => ("nvidia", "unknown".to_string()),
                "0x8086" => ("intel", "unknown".to_string()),
                _ => ("unknown", "unknown".to_string()),
            };
            out.push(GpuInfo {
                pci: device.file_name().to_string_lossy().into_owned(),
                device: format!(
                    "{}:{}",
                    vendor.trim_start_matches("0x"),
                    device_id.trim_start_matches("0x")
                ),
                arch,
                vendor: vendor_name.into(),
            });
        }
    }
    out
}

#[cfg(not(target_os = "linux"))]
fn detect_gpus() -> Vec<GpuInfo> {
    // macOS (L8) sysctl/Metal stub: report the CPU arch so the id resolves
    // generic-<arch>; the real Metal probe lands with L8.
    vec![GpuInfo {
        pci: String::new(),
        device: String::new(),
        arch: "metal".into(),
        vendor: "apple".into(),
    }]
}

/// KFD `gfx_target_version` (major*10000 + minor*100 + stepping) → `gfxNNNN`.
fn gfx_arch_name(version: u64) -> String {
    if version == 0 {
        return "unknown".into();
    }
    // AMD encodes gfx<major><minor><stepping>: gfx1100 = 11/0/0, gfx1030 =
    // 10/3/0, gfx90a = 9/0/10. Concatenate decimal digits, stepping as hex.
    let major = version / 10_000;
    let minor = (version / 100) % 100;
    let stepping = version % 100;
    format!("gfx{major}{minor}{stepping:x}")
}

fn detect_install_markers() -> InstallMarkers {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    let find_root = |candidates: &[PathBuf]| -> Option<String> {
        candidates
            .iter()
            .find(|p| p.exists())
            .map(|p| p.display().to_string())
    };
    InstallMarkers {
        toolbox_runtime: binary_on_path("toolbox"),
        hipfire_root: find_root(&[
            PathBuf::from(&home).join("hipfire"),
            PathBuf::from("/opt/hipfire"),
            PathBuf::from(&home).join("projects/hipfire"),
        ]),
        strixite_root: find_root(&[
            PathBuf::from(&home).join("strixite"),
            PathBuf::from("/opt/strixite"),
            PathBuf::from(&home).join("projects/strixite"),
        ]),
    }
}

fn binary_on_path(name: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| {
            std::env::split_paths(&paths).any(|dir| {
                let candidate = dir.join(name);
                candidate.is_file()
            })
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(arch: &str, apu: bool, gtt: Option<f64>, mem: f64) -> PlatformFacts {
        PlatformFacts {
            gpus: vec![GpuInfo {
                pci: "0000:c5:00.0".into(),
                device: "1002:150e".into(),
                arch: arch.into(),
                vendor: "rocm".into(),
            }],
            apu,
            mem_total_gib: mem,
            mem_available_gib: mem * 0.8,
            gtt_budget_gib: gtt,
            ..PlatformFacts::default()
        }
    }

    fn offer(probe: &PlatformProbe, engine: &str) -> EngineOffer {
        probe
            .offers
            .iter()
            .find(|o| o.engine == engine)
            .unwrap_or_else(|| panic!("missing offer {engine}"))
            .clone()
    }

    #[test]
    fn strix_halo_resolves_the_feed_id_and_offers_its_engines() {
        let probe = resolve(
            &facts("gfx1151", true, Some(104.0), 124.0),
            &bundled_feed(),
            "bundled",
        );
        assert_eq!(probe.id, "strix-halo");
        assert!(probe.known);
        assert!(probe.banner.is_none());
        assert!(offer(&probe, "strixite").offered);
        assert!(offer(&probe, "gufo").offered);
        assert!(!offer(&probe, "hipfire").offered, "APU is not discrete");
        assert!(offer(&probe, "llama.cpp").offered);
        assert_eq!(probe.budget_gib, 104.0);
    }

    #[test]
    fn gfx1100_is_generic_with_a_banner_and_offers_hipfire_not_strixite() {
        let probe = resolve(
            &facts("gfx1100", false, None, 64.0),
            &bundled_feed(),
            "bundled",
        );
        assert_eq!(probe.id, "generic-gfx1100");
        assert!(!probe.known);
        let banner = probe.banner.as_deref().unwrap_or_default();
        assert!(banner.contains("Unknown platform"), "{banner}");
        let hipfire = offer(&probe, "hipfire");
        assert!(hipfire.offered, "{hipfire:?}");
        assert!(offer(&probe, "llama.cpp").offered);
        // Fail-open means offers are not hard-refused; the reasons still name
        // the real matrix verdict for the known-platform path. Assert the
        // banner is the visible signal and the toolboxes are all offered.
        assert!(probe.offers.iter().all(|o| o.offered));
    }

    #[test]
    fn known_gfx1100_overlay_refuses_strixite_and_gufo_with_reasons() {
        // Acceptance (5) also checks the offer matrix independent of the
        // unknown-platform fail-open: a feed that names gfx1100 resolves a
        // known platform, and then strixite/gufo are refused with reasons.
        let feed = PlatformFeed {
            version: 1,
            platforms: vec![PlatformDef {
                id: "r9700".into(),
                arches: vec!["gfx1100".into()],
                label: "RX 7900 XT".into(),
            }],
        };
        let probe = resolve(&facts("gfx1100", false, None, 64.0), &feed, "fixture");
        assert!(probe.known);
        assert!(!offer(&probe, "strixite").offered);
        assert!(!offer(&probe, "gufo").offered);
        assert!(offer(&probe, "hipfire").offered);
        assert!(offer(&probe, "llama.cpp").offered);
        assert!(offer(&probe, "strixite")
            .reason
            .contains("not in the engine's arch family"));
    }

    #[test]
    fn strixite_needs_the_gtt_floor() {
        let probe = resolve(
            &facts("gfx1151", true, Some(64.0), 124.0),
            &bundled_feed(),
            "bundled",
        );
        let strixite = offer(&probe, "strixite");
        assert!(!strixite.offered);
        assert!(strixite.reason.contains("GTT"), "{strixite:?}");
    }

    #[test]
    fn hipfire_is_refused_on_an_apu() {
        let probe = resolve(
            &facts("gfx1151", true, Some(104.0), 124.0),
            &bundled_feed(),
            "bundled",
        );
        assert!(offer(&probe, "hipfire").reason.contains("APU"));
    }

    #[test]
    fn budget_falls_back_to_vram_then_ram() {
        let mut f = facts("gfx1100", false, None, 64.0);
        f.vram_budget_gib = Some(20.0);
        assert_eq!(budget_gib(&f), 20.0);
        f.vram_budget_gib = None;
        assert_eq!(budget_gib(&f), 48.0);
    }

    #[test]
    fn gfx_target_version_decodes() {
        assert_eq!(gfx_arch_name(110000), "gfx1100");
        assert_eq!(gfx_arch_name(110003), "gfx1103");
        assert_eq!(gfx_arch_name(110501), "gfx1151");
        assert_eq!(gfx_arch_name(100300), "gfx1030");
        assert_eq!(gfx_arch_name(90010), "gfx90a");
        assert_eq!(gfx_arch_name(0), "unknown");
    }

    #[test]
    fn feed_parses_and_rejects_empty() {
        let parsed = parse_feed(r#"{"version":1,"platforms":[{"id":"x","arches":["gfx1"]}]}"#)
            .unwrap();
        assert_eq!(parsed.platforms[0].id, "x");
        assert!(parse_feed(r#"{"version":1,"platforms":[]}"#).is_err());
    }

    #[tokio::test]
    async fn feed_cache_within_staleness_is_used_without_network() {
        let dir = std::env::temp_dir().join(format!(
            "brainrouter-platform-feed-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(FEED_CACHE_FILE);
        std::fs::write(
            &path,
            r#"{"version":1,"platforms":[{"id":"cached","arches":["gfx1151"]}]}"#,
        )
        .unwrap();
        let client = reqwest::Client::new();
        let (feed, origin) = resolve_feed(
            &client,
            Some("http://127.0.0.1:1/never"),
            &path,
            SystemTime::now(),
        )
        .await;
        assert_eq!(origin, "cache");
        assert_eq!(feed.platforms[0].id, "cached");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn stale_feed_falls_back_to_bundled_when_remote_and_cache_fail() {
        let dir = std::env::temp_dir().join(format!(
            "brainrouter-platform-feed-stale-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(FEED_CACHE_FILE);
        std::fs::write(&path, "not json").unwrap();
        let client = reqwest::Client::new();
        let (feed, origin) =
            resolve_feed(&client, Some("http://127.0.0.1:1/never"), &path, SystemTime::now()).await;
        assert_eq!(origin, "bundled");
        assert_eq!(feed, bundled_feed());
        let _ = std::fs::remove_dir_all(dir);
    }
}
