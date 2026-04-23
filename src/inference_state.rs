//! Tracks the current inference request lifecycle for the dashboard progress bar.
//!
//! The router updates this at each stage of request processing. The dashboard
//! polls `/api/inference-status` which reads the current state. Only one
//! request is tracked at a time (brainrouter runs single-slot).

use serde::Serialize;
use std::sync::Mutex;
use std::time::Instant;

/// Current phase of a request moving through brainrouter.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// No active request.
    Idle,
    /// Bonsai classifier is deciding cloud vs local.
    Classifying,
    /// Request sent to Manifest, waiting for first token.
    CloudWaiting,
    /// Manifest is streaming tokens back.
    CloudStreaming,
    /// Request sent to llama-swap, model may be loading or processing prompt.
    LocalWaiting,
    /// llama-swap is streaming tokens back.
    LocalStreaming,
}

/// Snapshot of the current inference state, returned to the dashboard.
#[derive(Debug, Clone, Serialize)]
pub struct InferenceSnapshot {
    pub phase: Phase,
    /// Model name (set once routing decision is made).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Human-readable provider (e.g. "Manifest", "llama-swap").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Elapsed milliseconds since the request started.
    pub elapsed_ms: u64,
}

/// Thread-safe inference state tracker. Updated by the router, read by the
/// dashboard endpoint. Interior mutability via Mutex — contention is negligible
/// at the access pattern here (one writer, one poller per second).
pub struct InferenceTracker {
    inner: Mutex<Inner>,
}

struct Inner {
    phase: Phase,
    model: Option<String>,
    provider: Option<String>,
    started_at: Option<Instant>,
}

impl InferenceTracker {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                phase: Phase::Idle,
                model: None,
                provider: None,
                started_at: None,
            }),
        }
    }

    /// Transition to a new phase, optionally setting model/provider.
    pub fn set(&self, phase: Phase, model: Option<String>, provider: Option<String>) {
        let mut inner = self.inner.lock().unwrap();
        // Start the timer on first non-idle transition.
        if inner.phase == Phase::Idle && phase != Phase::Idle {
            inner.started_at = Some(Instant::now());
        }
        inner.phase = phase;
        if model.is_some() {
            inner.model = model;
        }
        if provider.is_some() {
            inner.provider = provider;
        }
    }

    /// Reset to idle. Called when the request completes or errors.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.phase = Phase::Idle;
        inner.model = None;
        inner.provider = None;
        inner.started_at = None;
    }

    /// Take a snapshot of the current state for the dashboard.
    pub fn snapshot(&self) -> InferenceSnapshot {
        let inner = self.inner.lock().unwrap();
        let elapsed_ms = inner
            .started_at
            .map(|s| s.elapsed().as_millis() as u64)
            .unwrap_or(0);
        InferenceSnapshot {
            phase: inner.phase.clone(),
            model: inner.model.clone(),
            provider: inner.provider.clone(),
            elapsed_ms,
        }
    }
}
