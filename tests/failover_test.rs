//! Failover and circuit-breaker tests.
//!
//! These tests do NOT spin up the Bonsai classifier (it would require loading
//! the 8B GGUF model). Instead, they verify the primitives used by the router:
//! the `HealthTracker` circuit breaker.

use brainrouter::health::HealthTracker;

#[test]
fn circuit_breaker_opens_after_threshold_failures() {
    let tracker = HealthTracker::new();
    let key = "manifest";

    // Initially healthy
    assert!(tracker.is_healthy(key));

    // Report the threshold number of failures (3 per implementation)
    tracker.report_failure(key);
    tracker.report_failure(key);
    assert!(tracker.is_healthy(key), "still healthy below threshold");
    tracker.report_failure(key);

    // Now the circuit should be open
    assert!(!tracker.is_healthy(key), "circuit should be open after 3 failures");
}

#[test]
fn reporting_success_resets_circuit() {
    let tracker = HealthTracker::new();
    let key = "manifest";

    // Trip the breaker
    tracker.report_failure(key);
    tracker.report_failure(key);
    tracker.report_failure(key);
    assert!(!tracker.is_healthy(key));

    // Success reports should reset it
    tracker.report_success(key);
    assert!(tracker.is_healthy(key));
}

#[test]
fn independent_providers_have_independent_state() {
    let tracker = HealthTracker::new();

    // Break one provider
    tracker.report_failure("manifest");
    tracker.report_failure("manifest");
    tracker.report_failure("manifest");
    assert!(!tracker.is_healthy("manifest"));

    // The other provider is unaffected
    assert!(tracker.is_healthy("llama-swap"));
}
