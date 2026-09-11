//! In-memory circular buffer recording every routing decision.
//!
//! two-provider architecture (Manifest ↔ llama-swap). Each successful or
//! failed chat completion call records a snapshot of what happened, enabling
//! the dashboard to show a timeline of model selections over time.
//!
//! Events and completed stream measurements each retain at most 500 entries;
//! oldest entries are dropped independently via pop_front when full.

use chrono::Utc;
use serde::Serialize;
use std::collections::VecDeque;

/// Maximum number of events retained before dropping the oldest.
const MAX_EVENTS: usize = 500;

/// Which branch of the router produced this event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    CloudPrimary,   // First attempt → Manifest
    CloudFallback,  // Manifest failed/circuit-open → llama-swap
    LocalPrimary,   // Bonsai→Local, first llama-swap attempt
    LocalFallback,  // Primary llama-swap failed → fallback_model retry
}

/// A single routing-decision event, recorded after every successful or failed
/// chat-completion call.
#[derive(Debug, Clone, Serialize)]
pub struct RouteEvent {
    /// Monotonically increasing counter — unique within process lifetime.
    pub id: u64,
    /// ISO-8601 timestamp of the event.
    pub timestamp: String,
    /// Last user message content, truncated to 200 chars.
    pub prompt_excerpt: String,
    /// Model name requested by the caller ("auto" or explicit).
    pub requested_model: String,
    /// Actual backend provider that served the response, if known.
    /// None when the route errored before reaching any provider.
    pub effective_provider: Option<String>,
    /// The specific model key used on llama-swap (e.g. "qwen3.6-35b-a3b").
    /// Empty string when routed through Manifest.
    pub model_key: String,
    /// Milliseconds from routing-body entry until provider response/error return.
    /// This does not include consuming the response stream.
    pub latency_ms: u64,
    /// Which code path was taken.
    pub stage: Stage,
    /// Whether routing returned a provider response, not stream completion.
    pub success: bool,
    /// Error description (empty on success).
    pub error: String,
    /// Internal routing tag used by aggregate statistics and stage derivation.
    pub bonsai_decision: &'static str,
    /// Human-readable reason the route was selected, shown in the dashboard.
    pub routing_class: &'static str,
    /// Working directory of the OMP process that sent this request.
    /// Empty string when the cwd cannot be resolved (e.g. UDS connections).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub cwd: String,
    /// Session ID if this route call was for a review loop iteration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// User-Agent header of the calling harness ("" when internal/unknown).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub user_agent: String,
    /// Stable conversation fingerprint (hash of system prompt + first user
    /// message). Lets the dashboard group every turn of one harness
    /// conversation into a single card when the client sends no session id.
    /// Empty for review-loop events (they carry `session_id` instead).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub conv_id: String,
    /// Legacy prompt throughput. Stream measurements leave this at zero:
    /// router-observed TTFT is not a measurement of provider prompt processing.
    #[serde(skip_serializing_if = "is_zero_f64")]
    pub pp_tps: f64,
    /// Observed generation throughput, backfilled by event ID on valid completion.
    /// Zero means unavailable; see CompletedStreamMeasurement for its definition.
    #[serde(skip_serializing_if = "is_zero_f64")]
    pub tg_tps: f64,
}

fn is_zero_f64(f: &f64) -> bool {
    *f == 0.0
}

/// Shared description for consumers displaying completed-stream trends.
pub const STREAM_MEASUREMENT_DEFINITION: &str = concat!(
    "Completed samples require [DONE] followed by clean EOF; errors, cancellation, ",
    "malformed or incomplete streams are excluded. TTFT is router-observed milliseconds ",
    "from routing-body entry (including in-body routing and retries) to the first ",
    "complete SSE frame with nonempty content, reasoning, or tool-call arguments. Work ",
    "before routing-body entry, including outer profile resolution, is excluded; this ",
    "is not client end-to-end latency. Generation TPS estimates ",
    "(provider completion_tokens - 1) / seconds from first to last generated-output ",
    "frame, only with >1 reported completion tokens and distinct output observation ",
    "times. Usage/DONE tail is excluded. SSE deltas may batch tokens, and usage may ",
    "include hidden reasoning tokens outside the observed interval. Stream duration is ",
    "routing-body entry through [DONE], excluding the wait for confirming EOF. Missing, ",
    "malformed, or conflicting usage remains unknown; prompt-processing TPS is not ",
    "inferred. Local polling, buffering, and backpressure affect these measurements: ",
    "same-model trends are indicative, not controlled benchmark regressions.",
);

/// A completed OpenAI-compatible SSE stream, correlated to one routing request.
///
/// Recorded only after an explicit `[DONE]` frame followed by clean EOF.
/// Cancellation/drop, transport or SSE errors, malformed JSON, oversized frames,
/// and incomplete streams produce no record. Measurements reflect local polling
/// and buffering, not provider kernel timings or client end-to-end latency.
#[derive(Debug, Clone, Serialize)]
pub struct CompletedStreamMeasurement {
    /// ID returned by RoutingEvents::emit for the successful provider attempt.
    pub event_id: u64,
    /// RFC3339 UTC timestamp when clean EOF confirms completion.
    pub completed_at: String,
    pub model_key: String,
    pub effective_provider: Option<String>,
    /// Milliseconds from routing-body entry (including in-body routing/retries) to
    /// observing the first complete SSE frame with nonempty generated content,
    /// reasoning, or tool-call arguments. None if no generated output is seen.
    /// Excludes outer profile resolution and other pre-body work, and may include
    /// downstream backpressure.
    pub measured_ttft_ms: Option<f64>,
    /// Observed estimate: (completion_tokens - 1) / seconds between the first and
    /// last generated-output frames. Available only with >1 reported completion
    /// tokens and distinct output observation times; usage/DONE tail is excluded.
    /// Subtracts one token for the first arrival, but SSE deltas may batch tokens.
    /// Provider usage may include hidden reasoning tokens outside that observed
    /// interval, so this is not a pure decode benchmark or visible-token rate.
    pub generation_tps: Option<f64>,
    /// Provider-reported usage; absent, malformed, or conflicting counts are None.
    pub prompt_tokens: Option<u64>,
    /// Provider-reported completion usage, potentially including reasoning tokens.
    pub completion_tokens: Option<u64>,
    /// Milliseconds from routing-body entry through the complete `[DONE]` frame.
    /// Includes pre-stream routing/waiting, but not the wait for confirming EOF.
    pub stream_duration_ms: f64,
}

/// Thread-safe in-memory circular buffer.
pub struct RoutingEvents {
    inner: std::sync::Mutex<Inner>,
}

struct Inner {
    events: VecDeque<RouteEvent>,
    measurements: VecDeque<CompletedStreamMeasurement>,
    counter: u64,
}

/// JSON wrapper for the /api/routing-events endpoint.
#[derive(Serialize)]
pub struct RoutingEventsResponse {
    pub events: Vec<RouteEvent>,
}

impl Default for RoutingEvents {
    fn default() -> Self {
        Self::new()
    }
}

impl RoutingEvents {
    pub fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(Inner {
                events: VecDeque::with_capacity(MAX_EVENTS),
                measurements: VecDeque::with_capacity(MAX_EVENTS),
                counter: 0,
            }),
        }
    }

    /// Emit a new event and return its unique ID. Oldest entries are dropped
    /// when full; a stream must retain this ID rather than a conversation ID.
    pub fn emit(&self, mut event: RouteEvent) -> u64 {
        let mut inner = self.inner.lock().unwrap();
        inner.counter += 1;
        event.id = inner.counter;
        event.timestamp = Utc::now().to_rfc3339();
        inner.events.push_back(event);
        while inner.events.len() > MAX_EVENTS {
            inner.events.pop_front();
        }
        inner.counter
    }

    /// Legacy conversation-based API retained for compatibility. Production
    /// stream capture uses record_measurement and exact event IDs instead.
    pub fn update_tps(&self, conv_id: &str, pp_tps: f64, tg_tps: f64) -> bool {
        let mut inner = self.inner.lock().unwrap();
        for e in inner.events.iter_mut().rev() {
            if e.conv_id == conv_id && e.success {
                e.pp_tps = pp_tps;
                e.tg_tps = tg_tps;
                return true;
            }
        }
        false
    }

    /// Retain a confirmed stream sample and backfill only its own route event.
    /// The sample survives even if its route event was evicted while streaming.
    pub(crate) fn record_measurement(&self, measurement: CompletedStreamMeasurement) -> bool {
        let mut inner = self.inner.lock().unwrap();
        if measurement.event_id == 0
            || measurement.event_id > inner.counter
            || inner.measurements.iter().any(|m| m.event_id == measurement.event_id)
        {
            return false;
        }
        if let Some(event) = inner.events.iter_mut().find(|e| e.id == measurement.event_id) {
            if !event.success {
                return false;
            }
            event.pp_tps = 0.0;
            event.tg_tps = measurement.generation_tps.unwrap_or(0.0);
        }
        inner.measurements.push_back(measurement);
        while inner.measurements.len() > MAX_EVENTS {
            inner.measurements.pop_front();
        }
        true
    }

    /// Completed samples, newest completion first, independently capped at 500.
    pub fn get_measurements(&self) -> Vec<CompletedStreamMeasurement> {
        let inner = self.inner.lock().unwrap();
        inner.measurements.iter().rev().cloned().collect()
    }

    /// Return all events newest-first for dashboard rendering.
    ///
    /// Events are appended under a single Mutex, so insertion order matches
    /// wall-clock order. Reversing the deque yields newest-first without sorting.
    pub fn get_all(&self) -> Vec<RouteEvent> {
        let inner = self.inner.lock().unwrap();
        inner.events.iter().rev().cloned().collect()
    }

    /// Wrap events in the HTTP response envelope.
    pub fn get_all_as_response(&self) -> RoutingEventsResponse {
        RoutingEventsResponse { events: self.get_all() }
    }

    /// Aggregate statistics over all events — used by the stat-cards row.
    pub fn get_stats(&self) -> EventStats {
        let inner = self.inner.lock().unwrap();
        let events = &inner.events;
        if events.is_empty() {
            return EventStats::default();
        }

        let total = events.len();
        let successes: usize = events.iter().filter(|e| e.success).count();
        let failures = total - successes;
        let fallbacks: usize = events
            .iter()
            .filter(|e| matches!(e.stage, Stage::CloudFallback | Stage::LocalFallback))
            .count();
        let avg_latency: u64 = events
            .iter()
            .map(|e| e.latency_ms)
            .sum::<u64>()
            .checked_div(total as u64)
            .unwrap_or(0);
        let cloud_count: usize = events.iter().filter(|e| e.bonsai_decision == "cloud" || e.bonsai_decision == "cloud-direct").count();
        let local_count: usize = events.iter().filter(|e| e.bonsai_decision == "local" || e.bonsai_decision == "local-direct").count();
        let direct_local_count: usize = events.iter().filter(|e| e.bonsai_decision == "local-direct").count();
        let direct_cloud_count: usize = events.iter().filter(|e| e.bonsai_decision == "cloud-direct").count();
        let manifest_count: usize = events
            .iter()
            .filter(|e| e.success && e.effective_provider.as_deref() == Some("manifest"))
            .count();
        let llama_count: usize = events
            .iter()
            .filter(|e| e.success && e.effective_provider.as_deref() == Some("llama-swap"))
            .count();

        EventStats {
            total,
            successes,
            failures,
            fallbacks,
            avg_latency,
            cloud_count,
            local_count,
            direct_local_count,
            direct_cloud_count,
            manifest_count,
            llama_count,
        }
    }
}

/// Aggregated stats for the dashboard stat-cards row.
#[derive(Default, Debug, Clone, Serialize)]
pub struct EventStats {
    pub total: usize,
    pub successes: usize,
    pub failures: usize,
    pub fallbacks: usize,
    pub avg_latency: u64,
    pub cloud_count: usize,
    pub local_count: usize,
    /// Requests routed via model="local" (direct, no Bonsai).
    pub direct_local_count: usize,
    /// Requests routed via model="cloud" (direct, no Bonsai).
    pub direct_cloud_count: usize,
    pub manifest_count: usize,
    pub llama_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event() -> RouteEvent {
        RouteEvent {
            id: 0,
            timestamp: String::new(),
            prompt_excerpt: String::new(),
            requested_model: "auto".into(),
            effective_provider: Some("llama-swap".into()),
            model_key: "model".into(),
            latency_ms: 0,
            stage: Stage::LocalPrimary,
            success: true,
            error: String::new(),
            bonsai_decision: "local",
            routing_class: "bonsai → local",
            cwd: String::new(),
            session_id: None,
            user_agent: String::new(),
            conv_id: "same-conversation".into(),
            pp_tps: 0.0,
            tg_tps: 0.0,
        }
    }

    fn sample(event_id: u64) -> CompletedStreamMeasurement {
        CompletedStreamMeasurement {
            event_id,
            completed_at: Utc::now().to_rfc3339(),
            model_key: "model".into(),
            effective_provider: Some("llama-swap".into()),
            measured_ttft_ms: Some(100.0),
            generation_tps: Some(20.0),
            prompt_tokens: Some(100),
            completion_tokens: Some(5),
            stream_duration_ms: 400.0,
        }
    }

    #[test]
    fn measurements_are_independently_bounded_and_newest_first() {
        let events = RoutingEvents::new();
        for expected in 1..=MAX_EVENTS as u64 + 1 {
            let id = events.emit(event());
            assert_eq!(id, expected);
            assert!(events.record_measurement(sample(id)));
        }
        let measurements = events.get_measurements();
        assert_eq!(measurements.len(), MAX_EVENTS);
        assert_eq!(measurements[0].event_id, MAX_EVENTS as u64 + 1);
        assert_eq!(measurements.last().unwrap().event_id, 2);
        assert_eq!(events.get_all().len(), MAX_EVENTS);
        for _ in 0..MAX_EVENTS {
            events.emit(event());
        }
        assert_eq!(events.get_measurements().len(), MAX_EVENTS);
        assert_eq!(events.get_measurements()[0].event_id, MAX_EVENTS as u64 + 1);
    }

    #[test]
    fn measurements_survive_eviction_and_do_not_update_newer_conversation_event() {
        let events = RoutingEvents::new();
        let old_id = events.emit(event());
        for _ in 0..MAX_EVENTS {
            events.emit(event());
        }
        assert!(events.record_measurement(sample(old_id)));
        assert_eq!(events.get_measurements()[0].event_id, old_id);
        assert!(events.get_all().iter().all(|event| event.tg_tps == 0.0));
        assert!(!events.record_measurement(sample(old_id)));
        assert_eq!(events.get_measurements().len(), 1);
    }

    #[test]
    fn measurements_reject_failed_unknown_duplicate_ids_and_serialize_unknowns() {
        let events = RoutingEvents::new();
        let mut failed = event();
        failed.success = false;
        let failed_id = events.emit(failed);
        assert!(!events.record_measurement(sample(failed_id)));
        assert!(!events.record_measurement(sample(0)));
        assert!(!events.record_measurement(sample(failed_id + 1)));
        let id = events.emit(event());
        let mut measurement = sample(id);
        measurement.measured_ttft_ms = None;
        measurement.generation_tps = None;
        measurement.prompt_tokens = None;
        measurement.completion_tokens = None;
        assert!(events.record_measurement(measurement));
        assert!(!events.record_measurement(sample(id)));
        assert_eq!(events.get_all()[0].pp_tps, 0.0);
        assert_eq!(events.get_all()[0].tg_tps, 0.0);
        let json = serde_json::to_value(&events.get_measurements()[0]).unwrap();
        for field in ["measured_ttft_ms", "generation_tps", "prompt_tokens", "completion_tokens"] {
            assert!(json.get(field).unwrap().is_null());
        }
    }
}
