//! Append-only review-outcome ledger (design G4/H7, pragmatic form).
//!
//! A bounded, durable audit trail of code-review outcomes, surfaced by
//! `GET /api/review/status`. Each completed review contributes exactly one
//! terminal event (status + escalation reason + reviewer + iteration count),
//! so the spec's multi-event-per-run machinery (run_id streams, Admitted-before-
//! dispatch, startup reconciliation of orphaned runs) collapses: every event is
//! already terminal, and a design-gate block is captured by its `escalation_reason`
//! (`design_not_approved` / `design_unavailable`). The richer per-iteration stream
//! remains a future extension.
//!
//! Storage: a JSON array in `<config_dir>/review_ledger.json`, capped at
//! `MAX_EVENTS` (oldest dropped first), written disk-first atomically (temp file
//! + rename + parent-dir fsync). Absent/corrupt on load ⇒ empty (logged), never
//! blocks startup. Paths/hashes are not stored here (audit fields only), so the
//! response can be surfaced without redaction concerns.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const FILE_NAME: &str = "review_ledger.json";
const MAX_EVENTS: usize = 500;

/// Serializes the ledger read-modify-write so two concurrent reviews cannot lose
/// an event to a last-writer-wins race. The atomic rename already prevents
/// corruption; this makes appends lossless within the process.
static LEDGER_LOCK: Mutex<()> = Mutex::new(());

/// One terminal review-outcome event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEvent {
    pub event_id: String,
    pub session_id: String,
    /// RFC 3339 completion timestamp.
    pub ts: String,
    /// Terminal `ReviewStatus` wire string (e.g. `approved`, `escalated`).
    pub status: String,
    /// `EscalationReason` wire string when escalated, else `None`.
    #[serde(default)]
    pub escalation_reason: Option<String>,
    /// `ReviewerType` wire string (`llm` / `human`).
    pub reviewer_type: String,
    pub iteration_count: u32,
}

/// Canonical ledger path (beside `review_runtime_state.json`).
pub fn ledger_path() -> PathBuf {
    crate::config::default_config_path().with_file_name(FILE_NAME)
}

/// Load the ledger. Absent, unreadable, or corrupt ⇒ empty (logged); never panics.
pub fn load(path: &Path) -> Vec<LedgerEvent> {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            tracing::warn!(path = %path.display(), error = %e, "Ignoring corrupt review_ledger.json");
            Vec::new()
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "Could not read review_ledger.json");
            Vec::new()
        }
    }
}

/// The most recent `n` events, newest last (chronological order preserved).
pub fn recent(path: &Path, n: usize) -> Vec<LedgerEvent> {
    let all = load(path);
    let start = all.len().saturating_sub(n);
    all[start..].to_vec()
}

/// Append one event, evicting the oldest to stay within `MAX_EVENTS`, then write
/// the whole ledger back atomically. A write failure is returned so callers can
/// log it; ledger recording is best-effort and never blocks a review.
pub fn append(path: &Path, event: LedgerEvent) -> std::io::Result<()> {
    let _guard = LEDGER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut events = load(path);
    events.push(event);
    let overflow = events.len().saturating_sub(MAX_EVENTS);
    if overflow > 0 {
        events.drain(0..overflow);
    }
    write_atomic(path, &events)
}

fn write_atomic(path: &Path, events: &[LedgerEvent]) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;

    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "ledger path needs a parent directory",
        )
    })?;
    std::fs::create_dir_all(parent)?;
    let bytes = serde_json::to_vec_pretty(events)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let tmp = parent.join(format!(".review_ledger-{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        std::fs::rename(&tmp, path)?;
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
        Ok::<_, std::io::Error>(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// Build a `LedgerEvent` from a finished review result.
pub fn event_from_result(result: &crate::review::review_loop::ReviewResult) -> LedgerEvent {
    LedgerEvent {
        event_id: uuid::Uuid::new_v4().to_string(),
        session_id: result.session_id.clone(),
        ts: chrono::Utc::now().to_rfc3339(),
        status: result.status.as_str().to_string(),
        escalation_reason: result.escalation_reason.as_ref().map(|r| r.as_str().to_string()),
        reviewer_type: result.reviewer_type.as_str().to_string(),
        iteration_count: result.iteration_count,
    }
}

/// Best-effort: append a review outcome to the ledger, logging a write failure.
/// Ledger recording never blocks or fails a review.
pub fn record(result: &crate::review::review_loop::ReviewResult) {
    record_event(event_from_result(result));
}

/// Append an already-built event, best-effort (logs a write failure).
pub fn record_event(event: LedgerEvent) {
    if let Err(e) = append(&ledger_path(), event) {
        tracing::warn!(error = %e, "Failed to append review ledger event");
    }
}

/// Build a human-resolution ledger event (a person approved / requested changes
/// on an escalated review).
pub fn human_event(session_id: &str, status: &str) -> LedgerEvent {
    LedgerEvent {
        event_id: uuid::Uuid::new_v4().to_string(),
        session_id: session_id.to_string(),
        ts: chrono::Utc::now().to_rfc3339(),
        status: status.to_string(),
        escalation_reason: None,
        reviewer_type: "human".to_string(),
        iteration_count: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(session: &str, status: &str) -> LedgerEvent {
        LedgerEvent {
            event_id: uuid::Uuid::new_v4().to_string(),
            session_id: session.into(),
            ts: "2026-01-01T00:00:00Z".into(),
            status: status.into(),
            escalation_reason: None,
            reviewer_type: "llm".into(),
            iteration_count: 1,
        }
    }

    fn tmp_path() -> PathBuf {
        std::env::temp_dir()
            .join(format!("br-ledger-{}", uuid::Uuid::new_v4()))
            .join(FILE_NAME)
    }

    #[test]
    fn append_load_recent_roundtrip() {
        let path = tmp_path();
        assert!(load(&path).is_empty());
        append(&path, ev("s1", "approved")).unwrap();
        append(&path, ev("s2", "escalated")).unwrap();
        let all = load(&path);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].session_id, "s1");
        let last = recent(&path, 1);
        assert_eq!(last.len(), 1);
        assert_eq!(last[0].session_id, "s2");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn eviction_keeps_the_newest_up_to_the_cap() {
        let path = tmp_path();
        for i in 0..(MAX_EVENTS + 25) {
            append(&path, ev(&format!("s{i}"), "approved")).unwrap();
        }
        let all = load(&path);
        assert_eq!(all.len(), MAX_EVENTS);
        // The oldest 25 were evicted; the first retained is s25.
        assert_eq!(all[0].session_id, "s25");
        assert_eq!(all.last().unwrap().session_id, format!("s{}", MAX_EVENTS + 24));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn corrupt_ledger_loads_empty() {
        let path = tmp_path();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not json at all").unwrap();
        assert!(load(&path).is_empty());
        // And a subsequent append still works (starts fresh from empty).
        append(&path, ev("s1", "approved")).unwrap();
        assert_eq!(load(&path).len(), 1);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn concurrent_appends_are_lossless() {
        let path = std::sync::Arc::new(tmp_path());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut handles = Vec::new();
        for i in 0..24 {
            let p = std::sync::Arc::clone(&path);
            handles.push(std::thread::spawn(move || {
                append(&p, ev(&format!("s{i}"), "approved")).unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        // The LEDGER_LOCK serializes read-modify-write, so every event survives.
        assert_eq!(load(&path).len(), 24);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn human_event_is_marked_human() {
        let e = human_event("sess-1", "approved");
        assert_eq!(e.reviewer_type, "human");
        assert_eq!(e.status, "approved");
        assert_eq!(e.session_id, "sess-1");
    }
}
