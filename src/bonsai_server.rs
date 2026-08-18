//! External Bonsai llama-server process management.
//!
//! [`BonsaiControl`] owns the classifier's llama-server lifecycle: it is
//! started at daemon boot and can be stopped/restarted at runtime from the
//! dashboard (`POST /api/bonsai/toggle`) to free VRAM. While stopped, the
//! classifier's `enabled` flag is off and routing falls back to the safe
//! Cloud default.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use tokio::process::Command;
use tracing::{debug, info, warn};

/// Shared HTTP client for /health probes.
static HEALTH_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(reqwest::Client::new);

/// One live Bonsai llama-server process.
pub struct BonsaiServer {
    /// The HTTP base URL the classifier should call (e.g. http://127.0.0.1:9200).
    url: String,
    /// OS pid of the llama-server process (kill target).
    pid: u32,
    /// Wait task — completes when the child process is reaped.
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl BonsaiServer {
    /// Launch the external llama-server binary and wait for it to be healthy.
    /// If the health poll fails, the just-spawned process is killed so it is
    /// never orphaned.
    pub async fn start(
        fork_path: PathBuf,
        model_path: PathBuf,
        server_port: u16,
    ) -> Result<Self> {
        let url = format!("http://127.0.0.1:{}", server_port);

        info!(
            fork = %fork_path.display(),
            model = %model_path.display(),
            port = server_port,
            "Starting Bonsai llama-server"
        );

        if !fork_path.exists() {
            anyhow::bail!(
                "Bonsai fork binary not found at {}; download it first.",
                fork_path.display()
            );
        }

        // Spawn synchronously so the pid is known before the wait task takes
        // ownership of the child handle.
        let mut child = Command::new(&fork_path)
            .arg("--model")
            .arg(model_path.as_os_str())
            .arg("--port")
            .arg(server_port.to_string())
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--threads")
            .arg(num_cpus::get().to_string())
            .spawn()
            .with_context(|| format!("Failed to spawn llama-server at {}", fork_path.display()))?;
        let pid = child.id().unwrap_or(0);

        let handle = tokio::spawn(async move {
            let result = child.wait().await;
            match result {
                Ok(status) => info!(pid, status = %status, "Bonsai llama-server exited"),
                Err(e) => warn!(pid, error = %e, "Bonsai llama-server wait failed"),
            }
        });

        let server = Self { url, pid, handle: Some(handle) };

        // Wait for health endpoint; kill the process we just spawned if it
        // never comes up.
        if let Err(e) = Self::poll_health(&server.url).await {
            server.stop().await;
            return Err(e);
        }

        Ok(server)
    }

    /// Poll the /health endpoint until the server is ready (up to 60s).
    async fn poll_health(url: &str) -> Result<()> {
        let health_url = format!("{}/health", url);
        let deadline = Instant::now() + Duration::from_secs(60);

        loop {
            match HEALTH_CLIENT.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    info!("Bonsai llama-server is healthy");
                    return Ok(());
                }
                Ok(resp) => {
                    let status = resp.status();
                    debug!("Bonsai health returned {}: {}", status, status.canonical_reason().unwrap_or(""));
                }
                Err(e) => debug!("Bonsai health check failed (still starting): {}", e),
            }

            if Instant::now() >= deadline {
                anyhow::bail!("Bonsai llama-server did not become healthy within 60 seconds");
            }

            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }

    /// Shut down the server process: SIGTERM, escalating to SIGKILL after 5 s.
    /// Blocks until the process is reaped.
    pub async fn stop(mut self) {
        info!(pid = self.pid, "Stopping Bonsai llama-server (SIGTERM)");
        kill_pid(self.pid, libc::SIGTERM);
        let deadline = Instant::now() + Duration::from_secs(5);
        while pid_alive(self.pid) && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        if pid_alive(self.pid) {
            warn!(pid = self.pid, "Bonsai llama-server ignored SIGTERM; sending SIGKILL");
            kill_pid(self.pid, libc::SIGKILL);
            // Give the wait task a moment to reap the child.
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }
}

impl Drop for BonsaiServer {
    fn drop(&mut self) {
        // Best-effort: don't orphan the process when dropped without stop()
        // (e.g. daemon shutdown).
        kill_pid(self.pid, libc::SIGTERM);
    }
}

fn kill_pid(pid: u32, sig: i32) {
    if pid == 0 {
        return;
    }
    unsafe {
        libc::kill(pid as i32, sig);
    }
}

/// Signal-0 existence check.
fn pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

/// Runtime-controllable Bonsai llama-server lifecycle.
///
/// Held by `AppState` behind an `Arc`. The dashboard can stop it to free VRAM
/// and start it again later. The classifier reads [`Self::enabled`] before
/// every request; when off it skips HTTP and returns the safe Cloud default.
pub struct BonsaiControl {
    fork_path: PathBuf,
    model_path: PathBuf,
    port: u16,
    server: Mutex<Option<BonsaiServer>>,
    /// Serializes start/stop so concurrent toggles cannot double-spawn.
    op_lock: tokio::sync::Mutex<()>,
    enabled: Arc<AtomicBool>,
}

impl BonsaiControl {
    /// Build the control and start the server (initial state: on).
    pub async fn start(fork_path: PathBuf, model_path: PathBuf, port: u16) -> Result<Self> {
        let control = Self {
            fork_path,
            model_path,
            port,
            server: Mutex::new(None),
            op_lock: tokio::sync::Mutex::new(()),
            enabled: Arc::new(AtomicBool::new(false)),
        };
        control.start_server().await?;
        Ok(control)
    }

    /// HTTP base URL for the classifier to call (constant for this daemon).
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// True while the llama-server process is running.
    pub fn is_running(&self) -> bool {
        self.server.lock().map(|s| s.is_some()).unwrap_or(false)
    }

    /// Shared enabled flag — read by the classifier before every request.
    pub fn enabled(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.enabled)
    }

    /// Start the llama-server if it is not already running.
    pub async fn start_server(&self) -> Result<()> {
        let _guard = self.op_lock.lock().await;
        if self.is_running() {
            return Ok(());
        }
        let server = BonsaiServer::start(
            self.fork_path.clone(),
            self.model_path.clone(),
            self.port,
        )
        .await?;
        *self.server.lock().map_err(|_| anyhow::anyhow!("Bonsai control mutex poisoned"))? = Some(server);
        self.enabled.store(true, Ordering::SeqCst);
        info!(port = self.port, "Bonsai classifier server started");
        Ok(())
    }

    /// Stop the llama-server (frees VRAM). No-op if already stopped.
    pub async fn stop_server(&self) -> Result<()> {
        let Some(server) = self.server.lock().map_err(|_| anyhow::anyhow!("Bonsai control mutex poisoned"))?.take()
        else {
            self.enabled.store(false, Ordering::SeqCst);
            return Ok(());
        };
        // Disable classification before killing so in-flight requests don't
        // wait on a dying server.
        self.enabled.store(false, Ordering::SeqCst);
        server.stop().await;
        info!(port = self.port, "Bonsai classifier server stopped");
        Ok(())
    }

    /// Toggle the server. Returns the new running state.
    pub async fn toggle(&self) -> Result<bool> {
        if self.is_running() {
            self.stop_server().await?;
            Ok(false)
        } else {
            self.start_server().await?;
            Ok(true)
        }
    }

    /// Probe /health with a short timeout. False when stopped or unreachable.
    pub async fn healthy(&self) -> bool {
        if !self.is_running() {
            return false;
        }
        HEALTH_CLIENT
            .get(format!("{}/health", self.url()))
            .timeout(Duration::from_secs(1))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}
