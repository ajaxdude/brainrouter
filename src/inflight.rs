//! In-flight request registry backing the dashboard's In-Flight Requests panel.
//!
//! Every streaming chat request (`/v1/chat/completions`, `/v1/messages`)
//! registers an [`InflightEntry`] in an [`InflightRegistry`] when its body
//! starts streaming. The entry tracks elapsed time, byte count, and a sticky
//! activity code (tool calling / reasoning / asking / generating) sniffed from
//! the raw SSE chunks we stream ourselves — like llama-swap's own dashboard.
//!
//! ## Cancellation is best-effort
//!
//! [`InflightRegistry::cancel`] sets the entry's cancel flag. The next chunk
//! the stream polls while the flag is set ends the stream (the client job
//! tears the request down; the registry row goes with it). A client that
//! never polls again leaves a stale row until the registry's next sweep.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering as AtomicOrdering};
use std::task::{Context, Poll};
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use serde::Serialize;
use serde_json::json;
use tokio::sync::watch;
use tokio_stream::Stream;
use tracing::debug;

/// Activity codes; the dashboard mirrors them in `activityPill()`.
pub const ACT_NONE: u8 = 0;
/// Streaming output tokens (decode).
pub const ACT_GENERATING: u8 = 1;
/// Processing the prompt (prefill).
pub const ACT_PREFILLING: u8 = 2;
/// Emitting reasoning_content / thinking.
pub const ACT_REASONING: u8 = 3;
/// Emitting a tool_call / tool_calls fragment.
pub const ACT_TOOL: u8 = 4;
/// Emitting an ask_user question.
pub const ACT_ASKING: u8 = 5;

pub fn activity_label(code: u8) -> &'static str {
    match code {
        ACT_GENERATING => "generating",
        ACT_PREFILLING => "prefilling",
        ACT_REASONING => "reasoning",
        ACT_TOOL => "tool calling",
        ACT_ASKING => "asking a multiple choice question",
        _ => "",
    }
}

/// Priority rank for sticky activity: tool > reasoning > asking > generating.
/// A sticky label is never downgraded once set (sniffing is first-hit-wins).
fn activity_rank(code: u8) -> u8 {
    match code {
        ACT_TOOL => 3,
        ACT_REASONING => 2,
        ACT_ASKING => 1,
        _ => 0,
    }
}

/// One row of the in-flight panel (JSON contract shared with the dashboard).
#[derive(Clone, Debug, Serialize)]
pub struct InflightRow {
    pub id: u64,
    pub elapsed_ms: u64,
    pub method_path: String,
    pub model: String,
    pub user_agent: String,
    pub peer_addr: String,
    pub conv_id: String,
    pub session_id: String,
    pub bytes_received: u64,
    pub activity: &'static str,
    pub pp_progress: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct InflightView {
    pub requests: Vec<InflightRow>,
}

/// One in-flight request. `Arc`-owned so the sniffing stream adapter and the
/// registry can both hold a reference.
pub struct InflightEntry {
    pub id: u64,
    pub method_path: String,
    pub model: std::sync::Mutex<String>,
    pub user_agent: String,
    pub peer_addr: String,
    pub session_id: String,
    pub conv_id: String,
    pub bytes_received: AtomicU64,
    pub activity: AtomicU8,
    /// PP progress fraction (0.0..=1.0) stored as f64 bits.
    pub pp_progress: AtomicU64,
    cancelled: AtomicBool,
    started_ms: AtomicU64,
}

/// Handle to an in-flight request for the sniffing stream: byte/activity/
/// progress counters while the stream is alive, plus `Drop → registry.remove`
/// and `poll → if cancelled { end stream }`.
pub struct InflightHandle {
    entry: Arc<InflightEntry>,
    registry: Arc<InflightRegistry>,
    /// Kept alive so `cancel()` can notify even after the entry's own
    /// Arc-ness is gone; the entry itself is shared with the registry.
    _tx: watch::Sender<bool>,
}

impl InflightHandle {
    pub fn id(&self) -> u64 {
        self.entry.id
    }

    pub fn is_cancelled(&self) -> bool {
        self.entry.cancelled.load(AtomicOrdering::SeqCst)
    }

    /// Set the activity code, but never downgrade a sticky label already set:
    /// tool > reasoning > asking > generating. First strong hit wins.
    pub fn set_activity(&self, code: u8) {
        let cur = self.entry.activity.load(AtomicOrdering::Relaxed);
        if activity_rank(code) >= activity_rank(cur) {
            self.entry.activity.store(code, AtomicOrdering::Relaxed);
        }
    }

    pub fn set_model(&self, model: String) {
        *self.entry.model.lock().unwrap_or_else(|e| e.into_inner()) = model;
    }

    pub fn add_bytes(&self, n: u64) {
        self.entry.bytes_received.fetch_add(n, AtomicOrdering::Relaxed);
    }

    pub fn set_pp_progress(&self, frac: f64) {
        self.entry.pp_progress.store(frac.to_bits(), AtomicOrdering::Relaxed);
    }

    pub fn cancel(&self) {
        self.entry.cancelled.store(true, AtomicOrdering::SeqCst);
        debug!(id = self.entry.id, "in-flight request cancelled from dashboard");
    }
}

impl std::fmt::Debug for InflightHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InflightHandle")
            .field("id", &self.entry.id)
            .finish()
    }
}

impl Drop for InflightHandle {
    fn drop(&mut self) {
        self.registry.remove(self.entry.id);
    }
}

pub struct InflightRegistry {
    entries: std::sync::Mutex<Vec<Arc<InflightEntry>>>,
    next_id: AtomicU64,
}

impl Default for InflightRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Sweep entries older than this (a client that vanished without ever
/// completing its stream leaves a stale row otherwise).
const STALE_MS: u64 = 30 * 60 * 1000;

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

impl InflightRegistry {
    pub fn new() -> Self {
        Self {
            entries: std::sync::Mutex::new(Vec::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Snapshot for the dashboard (elapsed recomputed at read time). Stale
    /// entries (> 30 min) are swept here too, so the list self-heals.
    pub fn snapshot(&self) -> Vec<InflightRow> {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let now = now_ms();
        entries.retain(|e| now.saturating_sub(e.started_ms.load(AtomicOrdering::Relaxed)) <= STALE_MS);
        entries
            .iter()
            .map(|e| InflightRow {
                id: e.id,
                elapsed_ms: now.saturating_sub(e.started_ms.load(AtomicOrdering::Relaxed)),
                method_path: e.method_path.clone(),
                model: e.model.lock().unwrap_or_else(|e| e.into_inner()).clone(),
                user_agent: e.user_agent.clone(),
                peer_addr: e.peer_addr.clone(),
                conv_id: e.conv_id.clone(),
                session_id: e.session_id.clone(),
                bytes_received: e.bytes_received.load(AtomicOrdering::Relaxed),
                activity: activity_label(e.activity.load(AtomicOrdering::Relaxed)),
                pp_progress: f64::from_bits(e.pp_progress.load(AtomicOrdering::Relaxed)),
            })
            .collect()
    }

    /// Mark a live entry cancelled. Returns false when the id is unknown
    /// (already gone / never existed) so the HTTP handler answers 404.
    pub fn cancel(&self, id: u64) -> bool {
        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(e) = entries.iter().find(|e| e.id == id) {
            e.cancelled.store(true, AtomicOrdering::SeqCst);
            true
        } else {
            false
        }
    }

    /// Feed llama-server /slots PP progress onto every in-flight row whose
    /// resolved model matches `model`. Called by a daemon poller; no-op when
    /// the active model doesn't match or the build exposes no /slots.
    pub fn set_pp_progress_for_model(&self, model: &str, frac: f64) {
        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let frac = frac.clamp(0.0, 1.0);
        for e in entries.iter() {
            let m = e.model.lock().unwrap_or_else(|e| e.into_inner());
            if m.as_str() == model {
                drop(m);
                e.pp_progress.store(frac.to_bits(), AtomicOrdering::Relaxed);
            }
        }
    }

    pub fn json(&self) -> serde_json::Value {
        json!({ "requests": self.snapshot() })
    }

    /// Register a live request. The handle shares the entry Arc with the
    /// registry; dropping the last handle removes the row.
    pub fn register(
        self: &Arc<Self>,
        method_path: String,
        model: String,
        user_agent: String,
        peer_addr: String,
        session_id: String,
        conv_id: String,
        bytes_received: u64,
    ) -> Arc<InflightHandle> {
        let id = self.next_id.fetch_add(1, AtomicOrdering::Relaxed);
        let (tx, _rx) = watch::channel(false);
        let entry = Arc::new(InflightEntry {
            id,
            method_path,
            model: std::sync::Mutex::new(model),
            user_agent,
            peer_addr,
            session_id,
            conv_id,
            bytes_received: AtomicU64::new(bytes_received),
            activity: AtomicU8::new(ACT_PREFILLING),
            pp_progress: AtomicU64::new(0f64.to_bits()),
            cancelled: AtomicBool::new(false),
            started_ms: AtomicU64::new(now_ms()),
        });
        let handle = Arc::new(InflightHandle {
            entry: Arc::clone(&entry),
            registry: Arc::clone(self),
            _tx: tx,
        });
        {
            let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            entries.retain(|e| now_ms().saturating_sub(e.started_ms.load(AtomicOrdering::Relaxed)) <= STALE_MS);
            entries.push(Arc::clone(&handle.entry));
        }
        debug!(id, path = %handle.entry.method_path, "in-flight request registered");
        handle
    }

    fn remove(&self, id: u64) {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries.retain(|e| e.id != id);
    }
}


/// A chunk stream that ends the response when the request was cancelled from
/// the dashboard, counts bytes received, and derives a sticky activity label
/// (tool calling > reasoning > asking > generating) from the raw SSE bytes.
/// Windows never span chunks; a window is exactly one chunk. Items are the
/// provider's `anyhow::Result<Bytes>`; `Err` chunks pass through untouched.
#[derive(Debug)]
pub struct SniffStream<S> {
    inner: S,
    handle: Arc<InflightHandle>,
    window: Vec<u8>,
    finished: bool,
}

/// Bytes scanned per chunk before sticky detection; larger chunks are
/// passed through unsniffed (bytes still counted).
const SNIFF_WINDOW: usize = 4096;

impl<S> SniffStream<S>
where
    S: Stream<Item = anyhow::Result<Bytes>> + Unpin,
{
    pub fn new(inner: S, handle: Arc<InflightHandle>) -> Self {
        Self { inner, handle, window: Vec::new(), finished: false }
    }

    pub fn handle(&self) -> &Arc<InflightHandle> {
        &self.handle
    }

    /// Scan the retained head-window for activity markers; sticky so a strong
    /// signal (tool > reasoning > asking) is never overwritten by a weaker one.
    fn sniff(&mut self) {
        let w = &self.window;
        let has = |pat: &[u8]| w.windows(pat.len()).any(|win| win == pat);
        if has(b"\"tool_call\"") || has(b"\"tool_calls\"") || has(b"\"toolUse\"") {
            self.handle.set_activity(ACT_TOOL);
        } else if has(b"\"reasoning_content\"") || has(b"\"thinking\"") || has(b"\"redacted_thinking\"") {
            self.handle.set_activity(ACT_REASONING);
        } else if has(b"\"Which\"") || has(b"\"which \"") || has(b"\"multiple choice\"") {
            self.handle.set_activity(ACT_ASKING);
        } else {
            self.handle.set_activity(ACT_GENERATING);
        }
        self.window.clear();
    }
}

impl<S> Stream for SniffStream<S>
where
    S: Stream<Item = anyhow::Result<Bytes>> + Unpin,
{
    type Item = anyhow::Result<Bytes>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let s = self.as_mut().get_mut();
        if s.finished {
            return Poll::Ready(None);
        }
        match Pin::new(&mut s.inner).poll_next(cx) {
            Poll::Ready(Some(item)) => {
                if s.handle.is_cancelled() {
                    // The cancel sentinel: end the stream so the client job
                    // tears the request down and the row is dropped.
                    s.finished = true;
                    return Poll::Ready(None);
                }
                if let Ok(bytes) = &item {
                    s.handle.add_bytes(bytes.len() as u64);
                    if bytes.len() <= SNIFF_WINDOW {
                        s.window.clear();
                        s.window.extend_from_slice(bytes);
                        s.sniff();
                    } else if s.window.len() < SNIFF_WINDOW {
                        s.handle.set_activity(ACT_GENERATING);
                    }
                }
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                s.finished = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_stream::StreamExt;
    use tokio_stream::wrappers::UnboundedReceiverStream;

    fn entry(h: &Arc<InflightHandle>) -> &Arc<InflightEntry> {
        &h.entry
    }

    #[tokio::test]
    async fn registry_lists_and_cancels_rows() {
        let reg = Arc::new(InflightRegistry::new());
        let h1 = reg.register("POST /v1/chat/completions".into(), "qwen".into(), "ua".into(), "127.0.0.1:1234".into(), "sess".into(), "conv".into(), 1234);
        let h2 = reg.register("POST /v1/messages".into(), "qwen".into(), "ua".into(), "::1".into(), "sess2".into(), "conv2".into(), 7);
        assert!(reg.cancel(h1.id()));
        let snap = reg.snapshot();
        assert_eq!(snap.len(), 2);
        let rows = reg.json()["requests"].as_array().unwrap().clone();
        let ids: Vec<u64> = rows.iter().map(|r| r["id"].as_u64().unwrap()).collect();
        assert_eq!(ids, vec![h1.id(), h2.id()]);
        assert_eq!(rows[0]["bytes_received"].as_u64().unwrap(), 1234);
        // cancelling an unknown id fails
        assert!(!reg.cancel(9999));
    }

    #[tokio::test]
    async fn handle_updates_are_visible_in_snapshot() {
        let reg = Arc::new(InflightRegistry::new());
        let h = reg.register("POST /x".into(), "m".into(), "ua".into(), "127.0.0.1:1".into(), "s".into(), "c".into(), 0);
        h.set_activity(ACT_TOOL);
        h.add_bytes(12345);
        h.set_pp_progress(0.25);
        let rows = reg.json()["requests"].as_array().unwrap().clone();
        assert_eq!(rows[0]["activity"].as_str().unwrap(), "tool calling");
        assert_eq!(rows[0]["bytes_received"].as_u64().unwrap(), 12345);
        assert_eq!(rows[0]["pp_progress"].as_f64().unwrap(), 0.25);
    }

    #[tokio::test]
    async fn cancel_ended_stream_stops_stream() {
        let reg = Arc::new(InflightRegistry::new());
        let h = reg.register("POST /x".into(), "m".into(), "ua".into(), "127.0.0.1:1".into(), "s".into(), "c".into(), 0);
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<anyhow::Result<Bytes>>();
        let mut s = SniffStream::new(UnboundedReceiverStream::new(rx), Arc::clone(&h));
        tx.send(Ok(Bytes::from("data: hello\n\n"))).unwrap();
        let got = s.next().await.unwrap().unwrap();
        assert!(got.starts_with(b"data:"));
        assert_eq!(s.handle().entry.activity.load(AtomicOrdering::Relaxed), ACT_GENERATING);
        // Cancel: next poll ends the stream.
        h.cancel();
        tx.send(Ok(Bytes::from("data: late\n\n"))).unwrap();
        assert!(s.next().await.is_none());
        // Row stays listed but flagged.
        assert!(entry(&h).cancelled.load(AtomicOrdering::SeqCst));
    }

    #[tokio::test]
    async fn sniff_detects_tool_reasoning_asking_with_sticky_priority() {
        let reg = Arc::new(InflightRegistry::new());
        let h = reg.register("POST /x".into(), "m".into(), "ua".into(), "127.0.0.1:1".into(), "s".into(), "c".into(), 0);
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<anyhow::Result<Bytes>>();
        let mut s = SniffStream::new(UnboundedReceiverStream::new(rx), Arc::clone(&h));
        tx.send(Ok(Bytes::from("{\"tool_calls\":[]}"))).unwrap();
        s.next().await.unwrap().unwrap();
        assert_eq!(s.handle().entry.activity.load(AtomicOrdering::Relaxed), ACT_TOOL);
        // weaker signals never downgrade a sticky tool calling
        tx.send(Ok(Bytes::from("which one? multiple choice"))).unwrap();
        s.next().await.unwrap().unwrap();
        assert_eq!(s.handle().entry.activity.load(AtomicOrdering::Relaxed), ACT_TOOL);
    }

    #[tokio::test]
    async fn stale_entries_are_swept_on_read() {
        let reg = Arc::new(InflightRegistry::new());
        let h1 = reg.register("POST /a".into(), "m".into(), "ua".into(), "1".into(), "s".into(), "c".into(), 1);
        let h2 = reg.register("POST /b".into(), "m".into(), "ua".into(), "1".into(), "s".into(), "c".into(), 1);
        // Age the first entry past the sweep threshold and re-read: the
        // stale row must be swept on the next read, leaving h2.
        h1.entry.started_ms.store(0, AtomicOrdering::Relaxed);
        let snap = reg.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].id, h2.id());
    }

    #[tokio::test]
    async fn dropping_handle_removes_row() {
        let reg = Arc::new(InflightRegistry::new());
        let h = reg.register("POST /a".into(), "m".into(), "ua".into(), "1".into(), "s".into(), "c".into(), 1);
        assert_eq!(reg.snapshot().len(), 1);
        drop(h);
        assert_eq!(reg.snapshot().len(), 0);
    }
}
