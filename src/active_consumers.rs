//! Active-Consumer Set (v2 L5): the scheduler's protection input.
//!
//! An endpoint is a *protected consumer* while any live routed stream targets
//! it (proxy in-flight rows are consulted by the scheduler separately; this
//! registry carries the router-internal, in-process streams the proxy never
//! sees, e.g. the review loop) **plus an idle grace window** after the last
//! stream ends. The grace bridges tool-execution gaps so a live session is
//! never mid-swap.
//!
//! Keys are deliberately opaque strings: the scheduler asks about the catalog
//! id AND the resident endpoint (a span started against one must protect the
//! other). Proxy rows complete through [`ActiveConsumers::note_completion`]
//! (the in-flight completion hook); in-process streams use [`enter`] /
//! [`leave`] via [`wrap_with_active_consumer`].
//!
//! Override is one action: `swap ensure <model> --override` bypasses the span
//! for that operation only (audit-logged by the caller). Block reason:
//! `blocked:active-consumer`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::provider::ProviderResponse;

/// Default idle grace after the last stream ends (spec: `active_consumer_grace_s`).
pub const DEFAULT_GRACE_S: u64 = 120;

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[derive(Debug, Clone, Default)]
struct Span {
    active: u64,
    last_end_ms: u64,
    /// Highest stream count ever seen (diagnostics: distinguishes "never used"
    /// from "used once, grace expired").
    total: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConsumerSpan {
    pub key: String,
    /// Live in-process streams right now.
    pub active: u64,
    pub total: u64,
    pub last_end_ms: u64,
    /// `0` while active; otherwise the ms at which grace expires.
    pub protected_until_ms: u64,
    pub protected: bool,
}

/// The per-endpoint (and per-catalog-id) consumer registry. Cheap, lock-based,
/// in-process only (no schema, no state file).
pub struct ActiveConsumers {
    grace_ms: AtomicU64,
    spans: Mutex<BTreeMap<String, Span>>,
}

impl Default for ActiveConsumers {
    fn default() -> Self {
        Self::new(Duration::from_secs(
            std::env::var("BRAINROUTER_ACTIVE_CONSUMER_GRACE_S")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(DEFAULT_GRACE_S),
        ))
    }
}

impl ActiveConsumers {
    pub fn new(grace: Duration) -> Self {
        Self {
            grace_ms: AtomicU64::new(grace.as_millis() as u64),
            spans: Mutex::new(BTreeMap::new()),
        }
    }

    /// Env-first construction (`BRAINROUTER_ACTIVE_CONSUMER_GRACE_S`, or
    /// `_MS` for fixtures that need sub-second grace), else `fallback`.
    pub fn from_env_or(fallback: Duration) -> Self {
        if let Some(ms) = std::env::var("BRAINROUTER_ACTIVE_CONSUMER_GRACE_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
        {
            return Self::new(Duration::from_millis(ms));
        }
        if let Some(secs) = std::env::var("BRAINROUTER_ACTIVE_CONSUMER_GRACE_S")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
        {
            return Self::new(Duration::from_secs(secs));
        }
        Self::new(fallback)
    }

    pub fn grace(&self) -> Duration {
        Duration::from_millis(self.grace_ms.load(Ordering::SeqCst))
    }

    pub fn set_grace(&self, grace: Duration) {
        self.grace_ms
            .store(grace.as_millis() as u64, Ordering::SeqCst);
    }

    /// A stream started (proxy route time or an in-process wrapper).
    pub fn enter(&self, key: &str) {
        if key.is_empty() {
            return;
        }
        let mut spans = self.spans.lock().unwrap_or_else(|e| e.into_inner());
        let span = spans.entry(key.to_string()).or_default();
        span.active = span.active.saturating_add(1);
        span.total = span.total.saturating_add(1);
    }

    /// A stream ended: decrement the live count and start the grace window.
    pub fn leave(&self, key: &str) {
        if key.is_empty() {
            return;
        }
        let mut spans = self.spans.lock().unwrap_or_else(|e| e.into_inner());
        let span = spans.entry(key.to_string()).or_default();
        span.active = span.active.saturating_sub(1);
        span.last_end_ms = now_ms();
    }

    /// A proxy stream completed (its row already covers the live span; the
    /// completion hook only needs to open the grace window).
    pub fn note_completion(&self, key: &str) {
        self.leave(key);
    }

    pub fn active(&self, key: &str) -> u64 {
        self.spans
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)
            .map(|span| span.active)
            .unwrap_or(0)
    }

    pub fn protected(&self, key: &str) -> bool {
        self.protected_at(key, now_ms())
    }

    pub fn protected_at(&self, key: &str, now: u64) -> bool {
        if key.is_empty() {
            return false;
        }
        let grace = self.grace_ms.load(Ordering::SeqCst);
        let spans = self.spans.lock().unwrap_or_else(|e| e.into_inner());
        let Some(span) = spans.get(key) else {
            return false;
        };
        if span.active > 0 {
            return true;
        }
        span.last_end_ms > 0 && now.saturating_sub(span.last_end_ms) < grace
    }

    /// Milliseconds until this key leaves the protection span (`0` = not
    /// protected). Used by the Engines tab and the cutover re-check log.
    pub fn span_remaining_ms(&self, key: &str) -> u64 {
        self.span_remaining_at(key, now_ms())
    }

    pub fn span_remaining_at(&self, key: &str, now: u64) -> u64 {
        let grace = self.grace_ms.load(Ordering::SeqCst);
        let spans = self.spans.lock().unwrap_or_else(|e| e.into_inner());
        let Some(span) = spans.get(key) else {
            return 0;
        };
        if span.active > 0 {
            return u64::MAX;
        }
        if span.last_end_ms == 0 {
            return 0;
        }
        grace.saturating_sub(now.saturating_sub(span.last_end_ms))
    }

    pub fn snapshot(&self) -> Vec<ConsumerSpan> {
        let now = now_ms();
        let grace = self.grace_ms.load(Ordering::SeqCst);
        let spans = self.spans.lock().unwrap_or_else(|e| e.into_inner());
        spans
            .iter()
            .map(|(key, span)| {
                let protected = span.active > 0
                    || (span.last_end_ms > 0 && now.saturating_sub(span.last_end_ms) < grace);
                let protected_until_ms = if span.active > 0 || span.last_end_ms == 0 {
                    0
                } else {
                    span.last_end_ms.saturating_add(grace)
                };
                ConsumerSpan {
                    key: key.clone(),
                    active: span.active,
                    total: span.total,
                    last_end_ms: span.last_end_ms,
                    protected_until_ms,
                    protected,
                }
            })
            .collect()
    }
}

/// Wrap an in-process stream so the key set is entered now and left on drop
/// (clean EOF, error or client abort all release the span).
pub fn wrap_with_active_consumer(
    resp: ProviderResponse,
    consumers: Arc<ActiveConsumers>,
    keys: Vec<String>,
) -> ProviderResponse {
    let keys: Vec<String> = keys.into_iter().filter(|k| !k.is_empty()).collect();
    for key in &keys {
        consumers.enter(key);
    }
    wrap_held_with_active_consumer(resp, consumers, keys)
}

/// Like [`wrap_with_active_consumer`] for a caller that already entered the
/// span (the router enters before the upstream call so a long prefill cannot
/// race an eviction; it must still get the drop-time release).
pub fn wrap_held_with_active_consumer(
    resp: ProviderResponse,
    consumers: Arc<ActiveConsumers>,
    keys: Vec<String>,
) -> ProviderResponse {
    let keys: Vec<String> = keys.into_iter().filter(|k| !k.is_empty()).collect();
    match resp {
        ProviderResponse::Stream(stream) => {
            ProviderResponse::Stream(Box::pin(ActiveConsumerStream {
                stream,
                consumers,
                keys,
            }))
        }
    }
}

struct ActiveConsumerStream {
    stream: crate::provider::SseStream,
    consumers: Arc<ActiveConsumers>,
    keys: Vec<String>,
}

impl futures_util::Stream for ActiveConsumerStream {
    type Item = <crate::provider::SseStream as futures_util::Stream>::Item;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.stream.as_mut().poll_next(cx)
    }
}

impl Drop for ActiveConsumerStream {
    fn drop(&mut self) {
        for key in &self.keys {
            self.consumers.leave(key);
        }
    }
}

/// The keys a scheduler model is protected under: the catalog id, its alias,
/// and every endpoint a resident serves it at.
pub fn protection_keys(model: &str, endpoint: Option<&str>) -> Vec<String> {
    let mut keys = vec![model.to_string()];
    if let Some(endpoint) = endpoint.filter(|e| !e.is_empty()) {
        keys.push(endpoint.trim_end_matches('/').to_string());
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_streams_protect_and_grace_bridges_the_gap() {
        let consumers = ActiveConsumers::new(Duration::from_millis(60));
        assert!(!consumers.protected("m"));

        consumers.enter("m");
        assert!(consumers.protected("m"));
        assert_eq!(consumers.span_remaining_ms("m"), u64::MAX);

        consumers.leave("m");
        assert!(consumers.protected("m"), "grace window protects immediately after the end");
        assert!(consumers.span_remaining_ms("m") > 0);

        std::thread::sleep(Duration::from_millis(90));
        assert!(!consumers.protected("m"), "grace expired");
        assert_eq!(consumers.span_remaining_ms("m"), 0);
    }

    #[test]
    fn nested_enters_need_matching_leaves() {
        let consumers = ActiveConsumers::new(Duration::from_millis(0));
        consumers.enter("ep");
        consumers.enter("ep");
        consumers.leave("ep");
        assert_eq!(consumers.active("ep"), 1);
        assert!(consumers.protected("ep"));
        consumers.leave("ep");
        assert!(!consumers.protected("ep"), "zero grace: no span once the last stream ends");
    }

    #[test]
    fn snapshot_reports_the_span_and_keys_are_independent() {
        let consumers = ActiveConsumers::new(Duration::from_millis(1_000));
        consumers.enter("http://127.0.0.1:5901");
        let rows = consumers.snapshot();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].protected);
        assert_eq!(rows[0].key, "http://127.0.0.1:5901");
        assert!(!consumers.protected("model-a"));
        assert!(protection_keys("model-a", Some("http://127.0.0.1:5901/"))
            .contains(&"http://127.0.0.1:5901".to_string()));
    }
}
