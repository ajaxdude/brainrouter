//! In-memory circular buffer recording every routing decision.
//!
//! two-provider architecture (Manifest ↔ llama-swap). Each successful or
//! failed chat completion call records a snapshot of what happened, enabling
//! the dashboard to show a timeline of model selections over time.
//!
//! MaxEvents = 500 entries; oldest are dropped via shift when full.

use chrono::Utc;
use serde::Serialize;

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
    /// Elapsed wall-clock time from route() entry to final result.
    pub latency_ms: u64,
    /// Which code path was taken.
    pub stage: Stage,
    /// Whether the request succeeded end-to-end.
    pub success: bool,
    /// Error description (empty on success).
    pub error: String,
    /// "cloud" or "local" — what Bonsai decided for this request.
    pub bonsai_decision: &'static str,
    /// Working directory of the OMP process that sent this request.
    /// Empty string when the cwd cannot be resolved (e.g. UDS connections).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub cwd: String,
    /// Session ID if this route call was for a review loop iteration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

/// Thread-safe in-memory circular buffer.
pub struct RoutingEvents {
    events: std::sync::Mutex<Vec<RouteEvent>>,
    counter: std::sync::Mutex<u64>,
}

/// JSON wrapper for the /api/routing-events endpoint.
#[derive(Serialize)]
pub struct RoutingEventsResponse {
    pub events: Vec<RouteEvent>,
}

impl RoutingEvents {
    pub fn new() -> Self {
        Self {
            events: std::sync::Mutex::new(Vec::with_capacity(MAX_EVENTS)),
            counter: std::sync::Mutex::new(0),
        }
    }

    /// Emit a new event into the buffer. Oldest entries are dropped when full.
    pub fn emit(&self, mut event: RouteEvent) {
        let mut counter = self.counter.lock().unwrap();
        *counter += 1;
        event.id = *counter;
        event.timestamp = Utc::now().to_rfc3339();

        let mut events = self.events.lock().unwrap();
        events.push(event);
        while events.len() > MAX_EVENTS {
            events.remove(0);
        }
    }

    /// Return all events sorted newest-first for dashboard rendering.
    pub fn get_all(&self) -> Vec<RouteEvent> {
        let events = self.events.lock().unwrap();
        let mut list: Vec<_> = events.iter().cloned().collect();
        list.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        list
    }

    /// Wrap events in the HTTP response envelope.
    pub fn get_all_as_response(&self) -> RoutingEventsResponse {
        RoutingEventsResponse { events: self.get_all() }
    }

    /// Aggregate statistics over all events — used by the stat-cards row.
    pub fn get_stats(&self) -> EventStats {
        let events = self.events.lock().unwrap();
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
#[derive(Debug, Clone, Serialize)]
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

impl Default for EventStats {
    fn default() -> Self {
        Self {
            total: 0,
            successes: 0,
            failures: 0,
            fallbacks: 0,
            avg_latency: 0,
            cloud_count: 0,
            local_count: 0,
            direct_local_count: 0,
            direct_cloud_count: 0,
            manifest_count: 0,
            llama_count: 0,
        }
    }
}
