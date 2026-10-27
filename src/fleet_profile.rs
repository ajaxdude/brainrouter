//! Per-host fleet profiles (v2 L5).
//!
//! The supervisor's adapter set is data, not code: which engines this host
//! runs, their unit names, port scopes, memory floors and the Active-Consumer
//! grace all come from one profile. Built-ins mirror the L5 spec's wiring:
//!
//! * `strix-halo`: strixite (+ gufo + llama_swap_shim, sized pins);
//! * `fedora1` (gfx1100, unknown to the feed): hipfire (+ llama_cpp);
//! * `fedora2`: strixite prod (+ deepinfra in L6);
//! * unknown platform (fail-open): both adapters registered so a catalog
//!   entry can resolve; the platform matrix still drives what the UI offers.
//!
//! Every field has an environment override (`BRAINROUTER_*`) so fixtures and
//! operators can adjust a host without a rebuild; `UPGRADE.md` documents the
//! per-host procedures that use these names.

use crate::engine_hipfire::HipfireProfile;
use crate::engine_strixite::StrixiteProfile;
use crate::platform::PlatformProbe;

pub const DEFAULT_ACTIVE_CONSUMER_GRACE_S: u64 = 120;

#[derive(Debug, Clone)]
pub struct FleetProfile {
    pub platform_id: String,
    /// Engines this host runs (adapter registration order is irrelevant).
    pub engines: Vec<String>,
    pub strixite: Option<StrixiteProfile>,
    pub hipfire: Option<HipfireProfile>,
    /// Idle grace after the last routed stream before an eviction is legal.
    pub active_consumer_grace_s: u64,
    /// Gate poll interval (memory floor re-check).
    pub gate_poll_ms: u64,
    /// Gate timeout (spec: 10 minutes; env override for fixtures).
    pub gate_timeout_ms: u64,
    /// Post-pid-gone settle sleep before the memory gate.
    pub settle_ms: u64,
    /// Bounded single ACS re-check at the SIGTERM point.
    pub cutover_recheck_ms: u64,
}

impl FleetProfile {
    /// Built-in profile for a probe. `BRAINROUTER_FLEET_ENGINES` is a hard
    /// override (`strixite,hipfire`), useful on the fixture tier.
    pub fn for_probe(probe: &PlatformProbe) -> Self {
        let offered = |engine: &str| {
            probe
                .offers
                .iter()
                .find(|offer| offer.engine == engine)
                .map(|offer| offer.offered)
                .unwrap_or(false)
        };
        let mut engines = Vec::new();
        if offered("strixite") {
            engines.push("strixite".to_string());
        }
        if offered("hipfire") {
            engines.push("hipfire".to_string());
        }
        if let Ok(forced) = std::env::var("BRAINROUTER_FLEET_ENGINES") {
            for engine in forced.split(',').map(|e| e.trim()).filter(|e| !e.is_empty()) {
                if !engines.iter().any(|e| e == engine) {
                    engines.push(engine.to_string());
                }
            }
        }
        let strixite = engines
            .iter()
            .any(|e| e == "strixite")
            .then(StrixiteProfile::from_env);
        let hipfire = engines
            .iter()
            .any(|e| e == "hipfire")
            .then(HipfireProfile::from_env);
        Self {
            platform_id: probe.id.clone(),
            engines,
            strixite,
            hipfire,
            active_consumer_grace_s: std::env::var("BRAINROUTER_ACTIVE_CONSUMER_GRACE_S")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_ACTIVE_CONSUMER_GRACE_S),
            gate_poll_ms: std::env::var("BRAINROUTER_GATE_POLL_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1_000),
            gate_timeout_ms: std::env::var("BRAINROUTER_GATE_TIMEOUT_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(600_000),
            settle_ms: std::env::var("BRAINROUTER_CUTOVER_SETTLE_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2_000),
            cutover_recheck_ms: std::env::var("BRAINROUTER_CUTOVER_RECHECK_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5_000),
        }
    }

    pub fn has_engine(&self, engine: &str) -> bool {
        self.engines.iter().any(|e| e == engine)
    }
}

/// Built-in wiring table (documentation + tests): platform id → engines.
pub fn builtin_engines(platform_id: &str) -> &'static [&'static str] {
    match platform_id {
        "strix-halo" => &["strixite"],
        "fedora2" => &["strixite"],
        "fedora1" => &["hipfire"],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::{EngineOffer, PlatformFacts, PlatformProbe};

    fn probe(id: &str, known: bool, offers: &[(&str, bool)]) -> PlatformProbe {
        PlatformProbe {
            id: id.into(),
            known,
            banner: None,
            label: id.into(),
            facts: PlatformFacts::default(),
            budget_gib: 128.0,
            offers: offers
                .iter()
                .map(|(engine, offered)| EngineOffer {
                    engine: (*engine).to_string(),
                    offered: *offered,
                    reason: "fixture".into(),
                    min_mem_gib: if *engine == "strixite" { 100.0 } else { 16.0 },
                })
                .collect(),
            feed_origin: "fixture".into(),
            probed_at: String::new(),
        }
    }

    #[test]
    fn strix_halo_gets_strixite_and_fedora1_gets_hipfire() {
        let halo = FleetProfile::for_probe(&probe(
            "strix-halo",
            true,
            &[("strixite", true), ("hipfire", false)],
        ));
        assert_eq!(halo.engines, vec!["strixite".to_string()]);
        assert!(halo.strixite.is_some());
        assert!(halo.hipfire.is_none());

        let fedora1 = FleetProfile::for_probe(&probe(
            "generic-gfx1100",
            false,
            &[("strixite", true), ("hipfire", true)],
        ));
        assert!(fedora1.has_engine("hipfire"));
        assert!(fedora1.hipfire.is_some());
        assert!(!fedora1.has_engine("strixite") || fedora1.strixite.is_some());
    }

    #[test]
    fn builtin_table_matches_the_spec_wiring() {
        assert_eq!(builtin_engines("strix-halo"), &["strixite"]);
        assert_eq!(builtin_engines("fedora1"), &["hipfire"]);
        assert_eq!(builtin_engines("fedora2"), &["strixite"]);
        assert!(builtin_engines("generic-x").is_empty());
    }

    #[test]
    fn grace_defaults_to_the_spec_value() {
        let profile = FleetProfile::for_probe(&probe(
            "strix-halo",
            true,
            &[("strixite", true)],
        ));
        assert_eq!(profile.active_consumer_grace_s, 120);
        assert_eq!(profile.gate_timeout_ms, 600_000);
    }
}
