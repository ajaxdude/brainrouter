//! Memory-gated admission for the local code reviewer (design H1/H2, v6 table).
//!
//! Pure decision logic (`decide_backend`) is fully unit-testable; the async
//! wrapper in the review service reads `/proc/meminfo` and the admission
//! permit and calls it. The gate is **best-effort**: `MemAvailable` is an
//! estimate and non-brainrouter memory consumers are not coordinated (see the
//! design's Risk R-mem), so `system_reserve_mb` absorbs one concurrent
//! main-agent model and the gate never *guesses* a budget — a model with no
//! measured budget is not admitted locally.

use std::collections::HashMap;

/// The backend a review was admitted to run on (fixed before dispatch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmittedBackend {
    Cloud,
    Local,
}

impl AdmittedBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            AdmittedBackend::Cloud => "cloud",
            AdmittedBackend::Local => "local",
        }
    }
}

/// Outcome of the admission decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendDecision {
    /// Run on cloud. `reason` is `Some` only when a local run was wanted but
    /// declined (recorded for the dashboard/ledger).
    Cloud { reason: Option<String> },
    /// Run on the local reviewer model.
    Local,
    /// Cannot run at all; `reason` becomes the `blocked:<reason>` wire status.
    Blocked { reason: String },
}

/// Inputs to the pure admission decision.
pub struct AdmissionInputs<'a> {
    /// `"cloud" | "local" | "auto"` (unknown treated as `auto`).
    pub configured_backend: &'a str,
    pub configured_model: Option<&'a str>,
    pub fallback_model: &'a str,
    pub manifest_enabled: bool,
    pub budgets: &'a HashMap<String, u64>,
    pub system_reserve_mb: u64,
    /// `None` when `/proc/meminfo` is unreadable (non-Linux, or macOS).
    pub mem_available_mb: Option<u64>,
    /// Whether a local-review permit could be acquired.
    pub permit_available: bool,
}

/// Decide the reviewer backend per the v6 admission table. Pure.
pub fn decide_backend(i: &AdmissionInputs) -> BackendDecision {
    let cloud_available = i.manifest_enabled;

    // Resolve the intended backend from the configured choice.
    let want_local = match i.configured_backend {
        "cloud" => return BackendDecision::Cloud { reason: None },
        "local" => true,
        // `auto` (and any unknown value): cloud iff Manifest is enabled, else local.
        _ => !i.manifest_enabled,
    };
    if !want_local {
        return BackendDecision::Cloud { reason: None };
    }

    // Local path — apply the memory gate. `cloud_or_block` encodes "fall back
    // to cloud when it exists, otherwise block" so we never silently run an
    // unbudgeted or over-budget local model.
    let cloud_or_block = |reason: &str| -> BackendDecision {
        if cloud_available {
            BackendDecision::Cloud { reason: Some(reason.to_string()) }
        } else {
            BackendDecision::Blocked { reason: "budget_unavailable".to_string() }
        }
    };

    let model_key = i.configured_model.unwrap_or(i.fallback_model);
    let Some(&budget) = i.budgets.get(model_key) else {
        return cloud_or_block("budget_unavailable");
    };
    if !i.permit_available {
        return cloud_or_block("permit_busy");
    }
    let Some(avail) = i.mem_available_mb else {
        // No /proc/meminfo (non-Linux / macOS): can't verify headroom.
        return if cloud_available {
            BackendDecision::Cloud { reason: Some("meminfo_unavailable".to_string()) }
        } else {
            BackendDecision::Blocked { reason: "platform_unsupported".to_string() }
        };
    };
    let required = budget.saturating_add(i.system_reserve_mb);
    if avail >= required {
        BackendDecision::Local
    } else {
        cloud_or_block("insufficient_headroom")
    }
}

/// Read `MemAvailable` from `/proc/meminfo`, in MiB. `None` on any platform
/// without a readable/parseable `/proc/meminfo` (e.g. macOS).
pub fn read_meminfo_available_mb() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb / 1024);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budgets(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn inputs<'a>(
        configured: &'a str,
        manifest: bool,
        b: &'a HashMap<String, u64>,
        mem: Option<u64>,
        permit: bool,
    ) -> AdmissionInputs<'a> {
        AdmissionInputs {
            configured_backend: configured,
            configured_model: None,
            fallback_model: "local-model",
            manifest_enabled: manifest,
            budgets: b,
            system_reserve_mb: 4096,
            mem_available_mb: mem,
            permit_available: permit,
        }
    }

    #[test]
    fn explicit_cloud_is_cloud_without_touching_memory() {
        let b = budgets(&[]);
        assert_eq!(
            decide_backend(&inputs("cloud", false, &b, None, false)),
            BackendDecision::Cloud { reason: None }
        );
    }

    #[test]
    fn auto_prefers_cloud_when_manifest_enabled() {
        let b = budgets(&[]);
        assert_eq!(
            decide_backend(&inputs("auto", true, &b, None, true)),
            BackendDecision::Cloud { reason: None }
        );
    }

    #[test]
    fn auto_without_cloud_and_no_budget_blocks() {
        // Fresh-install contract (v6): auto -> local, no measured budget, no
        // cloud -> blocked:budget_unavailable (never guesses).
        let b = budgets(&[]);
        assert_eq!(
            decide_backend(&inputs("auto", false, &b, Some(999_999), true)),
            BackendDecision::Blocked { reason: "budget_unavailable".to_string() }
        );
    }

    #[test]
    fn local_with_budget_and_headroom_admits_local() {
        let b = budgets(&[("local-model", 8000)]);
        // 8000 + 4096 reserve = 12096 required; 20000 available.
        assert_eq!(
            decide_backend(&inputs("local", false, &b, Some(20_000), true)),
            BackendDecision::Local
        );
    }

    #[test]
    fn local_tight_memory_falls_back_to_cloud_when_available() {
        let b = budgets(&[("local-model", 8000)]);
        // required 12096; only 10000 available.
        assert_eq!(
            decide_backend(&inputs("local", true, &b, Some(10_000), true)),
            BackendDecision::Cloud { reason: Some("insufficient_headroom".to_string()) }
        );
    }

    #[test]
    fn local_tight_memory_without_cloud_blocks() {
        let b = budgets(&[("local-model", 8000)]);
        assert_eq!(
            decide_backend(&inputs("local", false, &b, Some(10_000), true)),
            BackendDecision::Blocked { reason: "budget_unavailable".to_string() }
        );
    }

    #[test]
    fn permit_busy_falls_back_to_cloud() {
        let b = budgets(&[("local-model", 8000)]);
        assert_eq!(
            decide_backend(&inputs("local", true, &b, Some(99_999), false)),
            BackendDecision::Cloud { reason: Some("permit_busy".to_string()) }
        );
    }

    #[test]
    fn no_meminfo_without_cloud_is_platform_unsupported() {
        let b = budgets(&[("local-model", 8000)]);
        assert_eq!(
            decide_backend(&inputs("local", false, &b, None, true)),
            BackendDecision::Blocked { reason: "platform_unsupported".to_string() }
        );
    }

    #[test]
    fn no_meminfo_with_cloud_falls_back() {
        let b = budgets(&[("local-model", 8000)]);
        assert_eq!(
            decide_backend(&inputs("local", true, &b, None, true)),
            BackendDecision::Cloud { reason: Some("meminfo_unavailable".to_string()) }
        );
    }
}
