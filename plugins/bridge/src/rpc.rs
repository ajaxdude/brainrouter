//! Minimal JSON-RPC 2.0 client over the daemon-assigned plugin UDS.
//!
//! Newline-delimited framing (one JSON object per line, `\n` terminated).
//! Responses are matched to requests by `id`; push notifications from the
//! daemon (frames with a `method` and no `id`, e.g. `config.changed`) are
//! forwarded on an unbounded channel.  Keying by id is mandatory because a
//! long-running `chat.completions` response can interleave with pushes.
//!
//! Self-contained on purpose: `tokio::net::UnixStream` + `serde_json`, no
//! heavyweight JSON-RPC framework.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, oneshot, Mutex};
use tracing::warn;

/// Per-request response timeout (the daemon's LLM path can legitimately take
/// minutes, so long calls pass their own budget).
pub const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_secs(30);
/// Write-side timeout for a single frame.
const WRITE_TIMEOUT: Duration = Duration::from_secs(10);

/// JSON-RPC error object returned by the daemon.
#[derive(Debug, Clone)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rpc error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for RpcError {}

struct Pending {
    pending: Mutex<HashMap<u64, oneshot::Sender<std::result::Result<Value, RpcError>>>>,
}

/// Connected JSON-RPC client.  Cheap to share via `Arc`; every call is
/// serialized on the single daemon connection by an internal write mutex.
pub struct RpcClient {
    shared: Arc<Pending>,
    write: Mutex<BufWriter<tokio::io::WriteHalf<UnixStream>>>,
    next_id: AtomicU64,
}

impl RpcClient {
    /// Connect to the daemon socket and spawn the reader loop.
    /// Returns the client plus the receiver for daemon→plugin push frames.
    pub async fn connect(path: &str) -> Result<(Self, mpsc::UnboundedReceiver<Value>)> {
        let stream = UnixStream::connect(path)
            .await
            .with_context(|| format!("connecting to daemon socket {path}"))?;
        let (read_half, write_half) = tokio::io::split(stream);

        let shared = Arc::new(Pending {
            pending: Mutex::new(HashMap::new()),
        });
        let (push_tx, push_rx) = mpsc::unbounded_channel();

        let reader_shared = Arc::clone(&shared);
        tokio::spawn(async move {
            let shared = reader_shared;
            let mut reader = BufReader::new(read_half);
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line).await {
                    Ok(0) => {
                        warn!("daemon closed the RPC connection");
                        break;
                    }
                    Ok(_) => {}
                    Err(e) => {
                        warn!("daemon RPC read failed: {e}");
                        break;
                    }
                }
                let trimmed = line.trim_end_matches(['\n', '\r']);
                if trimmed.trim().is_empty() {
                    continue;
                }
                let value: Value = match serde_json::from_str(trimmed) {
                    Ok(v) => v,
                    Err(e) => {
                        warn!("ignoring non-JSON daemon frame: {e}");
                        continue;
                    }
                };
                // Response: carries an id matching a pending request.
                if let Some(id) = value.get("id").and_then(Value::as_u64) {
                    let tx = shared.pending.lock().await.remove(&id);
                    match tx {
                        Some(tx) => {
                            let out = match value.get("error") {
                                Some(err) if !err.is_null() => Err(RpcError {
                                    code: err.get("code").and_then(Value::as_i64).unwrap_or(-1),
                                    message: err
                                        .get("message")
                                        .and_then(Value::as_str)
                                        .unwrap_or("unknown error")
                                        .to_string(),
                                }),
                                _ => Ok(value.get("result").cloned().unwrap_or(Value::Null)),
                            };
                            let _ = tx.send(out);
                        }
                        None => {
                            // Late response for a timed-out request — drop it.
                            warn!("discarding response for unknown id {id}");
                        }
                    }
                    continue;
                }
                // Push notification (no id): forward to the plugin loop.
                if value.get("method").is_some() {
                    let _ = push_tx.send(value);
                    continue;
                }
                warn!("ignoring unmatched daemon frame (no id, no method)");
            }
            // Connection gone: fail everything still pending.
            let mut pending = shared.pending.lock().await;
            for (_, tx) in pending.drain() {
                let _ = tx.send(Err(RpcError {
                    code: -32000,
                    message: "daemon connection closed".to_string(),
                }));
            }
        });

        Ok((
            Self {
                shared,
                write: Mutex::new(BufWriter::new(write_half)),
                next_id: AtomicU64::new(1),
            },
            push_rx,
        ))
    }

    /// One JSON-RPC call. `timeout` bounds the wait for the matching response.
    pub async fn call_with_timeout(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.shared.pending.lock().await.insert(id, tx);

        let frame = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let write = {
            let mut w = self.write.lock().await;
            tokio::time::timeout(
                WRITE_TIMEOUT,
                async {
                    w.write_all(serde_json::to_string(&frame)?.as_bytes())
                        .await?;
                    w.write_all(b"\n").await?;
                    w.flush().await?;
                    Ok::<(), anyhow::Error>(())
                },
            )
            .await
        };
        let write: Result<()> = match write {
            Ok(inner) => inner,
            Err(_) => Err(anyhow!("RPC write timed out")),
        };
        if let Err(e) = write {
            self.shared.pending.lock().await.remove(&id);
            return Err(e).context("writing RPC request");
        }

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(out)) => out.map_err(|e| anyhow!(e)),
            Ok(Err(_)) => {
                self.shared.pending.lock().await.remove(&id);
                Err(anyhow!("RPC response channel dropped for {method}"))
            }
            Err(_) => {
                self.shared.pending.lock().await.remove(&id);
                Err(anyhow!("RPC call {method} (id {id}) timed out"))
            }
        }
    }

    pub async fn call(&self, method: &str, params: Value) -> Result<Value> {
        self.call_with_timeout(method, params, DEFAULT_CALL_TIMEOUT)
            .await
    }

    // -----------------------------------------------------------------
    // Typed wrappers for the v1 daemon RPC surface (all grant-gated).
    // -----------------------------------------------------------------

    /// Mandatory first call: authenticates this connection against the
    /// minted token and pins the protocol version.
    pub async fn initialize(&self, plugin_id: &str, name: &str, token: &str) -> Result<Value> {
        self.call(
            "initialize",
            json!({
                "id": plugin_id,
                "plugin_id": plugin_id,
                "name": name,
                "token": token,
            }),
        )
        .await
    }

    /// `config.get {"key":"*"}` → the plugin's full config object.
    pub async fn config_get(&self, key: &str) -> Result<Value> {
        self.call("config.get", json!({ "key": key })).await
    }

    pub async fn config_set(&self, key: &str, value: Value) -> Result<Value> {
        self.call("config.set", json!({ "key": key, "value": value }))
            .await
    }

    /// `secrets.get` — the only path to the Discord token.  The daemon
    /// accepts any of the id aliases; send all of them for compatibility.
    pub async fn secrets_get(&self, credential_id: &str) -> Result<Value> {
        self.call(
            "secrets.get",
            json!({
                "id": credential_id,
                "credential_id": credential_id,
                "key": credential_id,
                "name": credential_id,
            }),
        )
        .await
    }

    pub async fn kv_get(&self, key: &str) -> Result<Value> {
        self.call("kv.get", json!({ "key": key })).await
    }

    pub async fn kv_set(&self, key: &str, value: Value) -> Result<Value> {
        self.call("kv.set", json!({ "key": key, "value": value })).await
    }

    pub async fn models_list(&self) -> Result<Value> {
        self.call("models.list", json!({})).await
    }

    /// Grant present for manifest completeness; the bridge LLM path is the
    /// omp subprocess → proxy (v1 parity), so the plugin makes zero calls.
    pub async fn chat_completions(&self, request: Value) -> Result<Value> {
        self.call_with_timeout("chat.completions", request, Duration::from_secs(3600))
            .await
    }

    pub async fn routing_events_tail(&self, since: u64, limit: u64) -> Result<Value> {
        self.call(
            "routing.events.tail",
            json!({ "since": since, "limit": limit }),
        )
        .await
    }

    pub async fn inflight_list(&self) -> Result<Value> {
        self.call("inflight.list", json!({})).await
    }

    pub async fn notify(&self, text: &str, level: &str) -> Result<Value> {
        self.call("notify", json!({ "text": text, "level": level }))
            .await
    }

    pub async fn set_status(&self, text: &str) -> Result<Value> {
        self.call("set_status", json!({ "text": text })).await
    }
}
