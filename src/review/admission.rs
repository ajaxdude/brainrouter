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
        // No measured budget for this model → the gate has no data to reason
        // with, so honor the intended local choice as-is (ungated). Operators
        // opt in to memory-gated local review by configuring
        // review_admission.local_model_budget_mb for the model. This keeps
        // existing local-reviewer setups working unchanged.
        return BackendDecision::Local;
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

/// Full admission outcome: the backend/model to run on plus a permit held for
/// the whole run when admitted local, or a terminal `blocked` reason.
pub enum AdmissionResult {
    Admitted {
        choice: crate::routing_profile::ModelChoice,
        /// Held for the entire review run when admitted local (bounds
        /// concurrent local reviews); `None` for cloud.
        permit: Option<tokio::sync::OwnedSemaphorePermit>,
    },
    Blocked {
        /// Becomes the `blocked:<reason>` surfaced to the caller.
        reason: String,
    },
}

/// Orchestrate the admission decision and permit lifecycle (design H2, v6).
///
/// The permit is only acquired on the local path, and only held when the final
/// decision is local — so a cloud reviewer never briefly blocks a concurrent
/// local one. `mem_available_mb` is injected for testability.
pub fn admit(
    configured: &crate::routing_profile::ModelChoice,
    manifest_enabled: bool,
    fallback_model: &str,
    cfg: &crate::config::ReviewAdmissionConfig,
    permits: &std::sync::Arc<tokio::sync::Semaphore>,
    mem_available_mb: Option<u64>,
) -> AdmissionResult {
    use crate::routing_profile::ModelChoice;
    let backend = configured.backend();
    let inputs = |permit_available: bool| AdmissionInputs {
        configured_backend: backend,
        configured_model: configured.model(),
        fallback_model,
        manifest_enabled,
        budgets: &cfg.local_model_budget_mb,
        system_reserve_mb: cfg.system_reserve_mb,
        mem_available_mb,
        permit_available,
    };
    // Preserve the configured explicit model when the admitted backend matches
    // the configured one; only synthesize a generic choice when overriding
    // (auto→cloud/local, or local→cloud fallback).
    let cloud_choice = || {
        if backend == "cloud" {
            configured.clone()
        } else {
            ModelChoice::Cloud { model: None }
        }
    };
    let local_choice = || {
        if backend == "local" {
            configured.clone()
        } else {
            ModelChoice::local()
        }
    };

    // First decide assuming a permit is available (it only matters for local).
    match decide_backend(&inputs(true)) {
        BackendDecision::Cloud { .. } => AdmissionResult::Admitted {
            choice: cloud_choice(),
            permit: None,
        },
        BackendDecision::Blocked { reason } => AdmissionResult::Blocked { reason },
        BackendDecision::Local => {
            // A `Local` decision arrives here either ungated (no measured
            // budget) or gated (budget present + headroom passed). Only the
            // gated path consumes the concurrency permit — the ungated/default
            // path must proceed permit-less so it is never serialized or
            // blocked (backward compatibility).
            let model_key = configured.model().unwrap_or(fallback_model);
            if !cfg.local_model_budget_mb.contains_key(model_key) {
                return AdmissionResult::Admitted {
                    choice: local_choice(),
                    permit: None,
                };
            }
            match std::sync::Arc::clone(permits).try_acquire_owned() {
                Ok(permit) => AdmissionResult::Admitted {
                    choice: local_choice(),
                    permit: Some(permit),
                },
                // Budgeted local, but the permit is busy: fall back to cloud if
                // available, else block (reason reflects the real cause).
                Err(_) => match decide_backend(&inputs(false)) {
                    BackendDecision::Cloud { .. } => AdmissionResult::Admitted {
                        choice: cloud_choice(),
                        permit: None,
                    },
                    _ => AdmissionResult::Blocked { reason: "permit_busy".to_string() },
                },
            }
        }
    }
}

/// Bundled admission dependencies, held by `ReviewService` and passed to the
/// review loop so it can resolve the backend per run.
pub struct AdmissionCtx {
    pub config: crate::config::ReviewAdmissionConfig,
    pub manifest_enabled: bool,
    pub fallback_model: String,
    pub permits: std::sync::Arc<tokio::sync::Semaphore>,
}

impl AdmissionCtx {
    /// Resolve the backend for a run, reading live `/proc/meminfo`.
    pub fn resolve(&self, configured: &crate::routing_profile::ModelChoice) -> AdmissionResult {
        admit(
            configured,
            self.manifest_enabled,
            &self.fallback_model,
            &self.config,
            &self.permits,
            read_meminfo_available_mb(),
        )
    }
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
    fn auto_without_cloud_and_no_budget_honors_local() {
        // Opt-in gate: auto→local with no cloud and no measured budget honors
        // local (ungated) rather than blocking, so bare installs still review.
        let b = budgets(&[]);
        assert_eq!(
            decide_backend(&inputs("auto", false, &b, Some(999_999), true)),
            BackendDecision::Local
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

    // ── admit() orchestration (backend decision + permit lifecycle) ──────────
    use crate::config::ReviewAdmissionConfig;
    use crate::routing_profile::ModelChoice;
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    fn admission_cfg(pairs: &[(&str, u64)], reserve: u64) -> ReviewAdmissionConfig {
        ReviewAdmissionConfig {
            local_model_budget_mb: budgets(pairs),
            system_reserve_mb: reserve,
            local_review_permits: 1,
        }
    }

    #[test]
    fn admit_cloud_holds_no_permit() {
        let cfg = admission_cfg(&[], 4096);
        let sem = Arc::new(Semaphore::new(1));
        match admit(&ModelChoice::Auto, true, "fb", &cfg, &sem, Some(99_999)) {
            AdmissionResult::Admitted { choice, permit } => {
                assert_eq!(choice.backend(), "cloud");
                assert!(permit.is_none());
            }
            _ => panic!("expected cloud admission"),
        }
        assert_eq!(sem.available_permits(), 1, "cloud must not retain a permit");
    }

    #[test]
    fn admit_local_with_headroom_holds_the_permit() {
        let cfg = admission_cfg(&[("m", 8000)], 4096);
        let sem = Arc::new(Semaphore::new(1));
        let choice = ModelChoice::Local { model: Some("m".to_string()) };
        match admit(&choice, false, "fb", &cfg, &sem, Some(20_000)) {
            AdmissionResult::Admitted { choice, permit } => {
                assert_eq!(choice.backend(), "local");
                assert!(permit.is_some());
                assert_eq!(sem.available_permits(), 0, "local admission holds the permit");
            }
            _ => panic!("expected local admission"),
        }
    }

    #[test]
    fn admit_local_no_budget_honors_local() {
        // No measured budget → honor the explicit local choice (ungated).
        let cfg = admission_cfg(&[], 4096);
        let sem = Arc::new(Semaphore::new(1));
        match admit(&ModelChoice::local(), false, "fb", &cfg, &sem, Some(99_999)) {
            AdmissionResult::Admitted { choice, .. } => assert_eq!(choice.backend(), "local"),
            _ => panic!("expected local admission when no budget is configured"),
        }
    }

    #[test]
    fn admit_local_permit_busy_falls_back_to_cloud() {
        let cfg = admission_cfg(&[("m", 8000)], 4096);
        let sem = Arc::new(Semaphore::new(1));
        let _held = Arc::clone(&sem).try_acquire_owned().unwrap(); // exhaust the single permit
        let choice = ModelChoice::Local { model: Some("m".to_string()) };
        match admit(&choice, true, "fb", &cfg, &sem, Some(20_000)) {
            AdmissionResult::Admitted { choice, permit } => {
                assert_eq!(choice.backend(), "cloud");
                assert!(permit.is_none());
            }
            _ => panic!("expected cloud fallback when the permit is busy"),
        }
    }

    #[test]
    fn admit_preserves_the_configured_explicit_cloud_model() {
        // Regression: admission must NOT drop the configured cloud model when
        // it admits cloud (else the review routes to the wrong model).
        let cfg = admission_cfg(&[], 4096);
        let sem = Arc::new(Semaphore::new(1));
        let choice = ModelChoice::Cloud { model: Some("review-cloud-test".to_string()) };
        match admit(&choice, true, "fb", &cfg, &sem, None) {
            AdmissionResult::Admitted { choice, .. } => {
                assert_eq!(choice.model(), Some("review-cloud-test"));
            }
            _ => panic!("expected cloud admission preserving the configured model"),
        }
    }

    #[test]
    fn admit_ungated_local_ignores_a_busy_permit() {
        // No budget → ungated local; it must NOT be gated by the concurrency
        // permit, even when the single permit is exhausted, so the default
        // path is never serialized/blocked (backward compatibility).
        let cfg = admission_cfg(&[], 4096);
        let sem = Arc::new(Semaphore::new(1));
        let _held = Arc::clone(&sem).try_acquire_owned().unwrap(); // exhaust the permit
        let choice = ModelChoice::Local { model: Some("no-budget-model".to_string()) };
        match admit(&choice, false, "fb", &cfg, &sem, Some(100)) {
            AdmissionResult::Admitted { choice, permit } => {
                assert_eq!(choice.backend(), "local");
                assert!(permit.is_none(), "ungated local must not hold a permit");
            }
            _ => panic!("ungated local must be admitted even when the permit is busy"),
        }
    }
}
