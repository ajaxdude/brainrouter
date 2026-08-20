//! Minimal HTTP client for talking to the brainrouter daemon.
//!
//! Used by `brainrouter mcp` and `brainrouter cli`. Both are thin clients:
//! they forward commands to the running daemon over its Unix domain socket
//! (preferred) or its TCP listener, and never load Bonsai or run the review
//! loop themselves. All heavy lifting stays in the daemon.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::PathBuf;

/// How to reach the daemon.
#[derive(Debug, Clone)]
pub enum DaemonEndpoint {
    /// Unix domain socket (preferred: no TCP port, local-only).
    Socket(PathBuf),
    /// TCP URL, e.g. http://127.0.0.1:9099.
    Url(String),
}

/// Raw HTTP response from the daemon.
#[derive(Debug)]
pub struct DaemonResponse {
    pub status: u16,
    pub body: String,
}

impl DaemonResponse {
    /// Parse the body as JSON, or fail with a status-tagged error.
    pub fn json(&self) -> Result<Value> {
        if self.body.is_empty() {
            bail!("Daemon returned empty body (status {})", self.status);
        }
        serde_json::from_str(&self.body)
            .with_context(|| format!("Failed to parse daemon JSON response (status {})", self.status))
    }
}

/// Thin HTTP client for daemon commands.
#[derive(Debug, Clone)]
pub struct DaemonClient {
    endpoint: DaemonEndpoint,
}

impl DaemonClient {
    pub fn new(endpoint: DaemonEndpoint) -> Self {
        Self { endpoint }
    }

    /// Send a JSON request. `body` is serialized as JSON for non-GET methods.
    pub async fn request(&self, method: &str, path: &str, body: Option<Value>) -> Result<DaemonResponse> {
        let raw = match &body {
            Some(v) => serde_json::to_vec(v)?,
            None => Vec::new(),
        };
        self.request_raw(method, path, raw).await
    }

    /// Send a request with a raw byte body (used for YAML config uploads).
    pub async fn request_raw(&self, method: &str, path: &str, body: Vec<u8>) -> Result<DaemonResponse> {
        match &self.endpoint {
            DaemonEndpoint::Socket(socket_path) => {
                uds_request(socket_path, method, path, body).await
            }
            DaemonEndpoint::Url(url) => tcp_request(url, method, path, body).await,
        }
    }

    /// Convenience: GET and parse JSON.
    pub async fn get_json(&self, path: &str) -> Result<Value> {
        let resp = self.request_raw("GET", path, Vec::new()).await?;
        check_ok(&resp)?;
        resp.json()
    }

    /// Convenience: POST a JSON body and parse the JSON response.
    pub async fn post_json(&self, path: &str, body: Value) -> Result<Value> {
        let resp = self.request("POST", path, Some(body)).await?;
        check_ok(&resp)?;
        resp.json()
    }

    /// Convenience: GET raw text (used for YAML config bodies).
    pub async fn get_raw(&self, path: &str) -> Result<String> {
        let resp = self.request_raw("GET", path, Vec::new()).await?;
        check_ok(&resp)?;
        Ok(resp.body)
    }

    /// Convenience: POST raw text (used for YAML config uploads).
    pub async fn post_raw(&self, path: &str, body: &str) -> Result<String> {
        let resp = self.request_raw("POST", path, body.as_bytes().to_vec()).await?;
        check_ok(&resp)?;
        Ok(resp.body)
    }
}

fn check_ok(resp: &DaemonResponse) -> Result<()> {
    if (200..300).contains(&resp.status) {
        Ok(())
    } else {
        let preview: String = resp.body.chars().take(500).collect();
        bail!("Daemon returned HTTP {} — {}", resp.status, preview)
    }
}

/// Make an HTTP request over the Unix domain socket.
async fn uds_request(
    socket_path: &PathBuf,
    method: &str,
    path: &str,
    body_bytes: Vec<u8>,
) -> Result<DaemonResponse> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixStream;

    // Bound the connect so a wedged daemon can't stall cli/mcp forever. A
    // nonexistent/refused socket fails fast anyway; this catches the hung-
    // accept case. The read below is bounded to match the TCP path's 3600s
    // (the legacy blocking /review/api/request can legitimately take that long).
    let stream = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        UnixStream::connect(socket_path),
    )
    .await
    .with_context(|| format!("Cannot connect to daemon socket at {}", socket_path.display()))?
    .with_context(|| "Timed out connecting to the daemon socket — is `brainrouter serve` running?")?;

    let request = if method == "GET" {
        format!("{} {} HTTP/1.0\r\nHost: localhost\r\n\r\n", method, path)
    } else {
        format!(
            "{} {} HTTP/1.0\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            method, path, body_bytes.len()
        )
    };

    let (mut rx, mut tx) = stream.into_split();
    tx.write_all(request.as_bytes()).await?;
    if method != "GET" && !body_bytes.is_empty() {
        tx.write_all(&body_bytes).await?;
    }
    // Keep tx alive until we finish reading — dropping it before read_to_end
    // closes the underlying socket (sends FIN) which Hyper treats as a reset.
    let mut response_bytes = Vec::new();
    let read = rx.read_to_end(&mut response_bytes);
    tokio::time::timeout(std::time::Duration::from_secs(3600), read)
        .await
        .with_context(|| "Timed out waiting for the daemon to respond (3600s)")??;
    drop(tx);

    parse_http_response(&response_bytes)
}

/// Make an HTTP request over TCP via reqwest (handles chunked/keep-alive).
async fn tcp_request(url: &str, method: &str, path: &str, body_bytes: Vec<u8>) -> Result<DaemonResponse> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()
        .context("Failed to build HTTP client")?;

    let full_url = format!("{}{}", url.trim_end_matches('/'), path);
    let method = reqwest::Method::from_bytes(method.as_bytes())
        .unwrap_or(reqwest::Method::GET);

    let mut req = client.request(method, &full_url);
    if !body_bytes.is_empty() {
        req = req
            .header("content-type", "application/json")
            .body(body_bytes);
    }

    let resp = req.send().await.with_context(|| format!("Failed to reach daemon at {}", full_url))?;
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap_or_default();
    Ok(DaemonResponse { status, body })
}

/// Parse a raw HTTP/1.x response into status + body.
fn parse_http_response(response_bytes: &[u8]) -> Result<DaemonResponse> {
    let response_str = String::from_utf8_lossy(response_bytes);
    let body_start = response_str
        .find("\r\n\r\n")
        .map(|i| i + 4)
        .unwrap_or(0);

    let status_line = response_str.lines().next().unwrap_or("");
    let mut status: u16 = 0;
    if status_line.starts_with("HTTP/") {
        status = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(0);
    }

    let body = String::from_utf8_lossy(&response_bytes[body_start..]).to_string();
    Ok(DaemonResponse { status, body })
}
