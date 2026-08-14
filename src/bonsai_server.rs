//! External Bonsai llama-server process management.
//!
//! Launches the PrismML fork binary, waits for readiness, and provides
//! the HTTP endpoint for the classifier to call.

use anyhow::Result;
use std::{path::PathBuf, time::Duration};
use tokio::process::Command;
use tracing::{debug, info, warn};

/// Manages the external Bonsai llama-server process.
pub struct BonsaiServer {
    /// The HTTP base URL the classifier should call (e.g. http://127.0.0.1:9200).
    url: String,
    /// Tokio task handle — kept alive to keep the task (and child process) running.
    #[allow(dead_code)]
    handle: tokio::task::JoinHandle<()>,
    /// Tokio task abort handle — used to kill the child process.
    abort: tokio::task::AbortHandle,
}

impl BonsaiServer {
    /// Launch the external llama-server binary and wait for it to be healthy.
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

        let server_port = server_port as i32;

        let handle = tokio::spawn(async move {
            let mut child = Command::new(fork_path)
                .arg("--model")
                .arg(model_path.as_os_str())
                .arg("--port")
                .arg(server_port.to_string())
                .arg("--host")
                .arg("127.0.0.1")
                .arg("--threads")
                .arg(num_cpus::get().to_string())
                .spawn()
                .expect("Failed to spawn llama-server");

            let result = child.wait().await;
            if let Err(e) = result {
                warn!("llama-server exited with error: {}", e);
            } else {
                info!("llama-server exited normally");
            }
        });

        let abort = handle.abort_handle();

        // Wait for health endpoint
        Self::poll_health(&url).await?;

        Ok(Self { url, handle, abort })
    }

    /// Poll the /health endpoint until the server is ready (up to 60s).
    async fn poll_health(url: &str) -> Result<()> {
        let http = reqwest::Client::new();
        let health_url = format!("{}/health", url);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(60);

        loop {
            match http.get(&health_url).send().await {
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

            if tokio::time::Instant::now() >= deadline {
                anyhow::bail!("Bonsai llama-server did not become healthy within 60 seconds");
            }

            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }

    /// HTTP base URL for the classifier to call.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Shut down the server process.
    pub async fn stop(self) {
        self.abort.abort();
        // Drop the struct (Drop will also try to abort, but that's harmless)
    }
}

impl Drop for BonsaiServer {
    fn drop(&mut self) {
        self.abort.abort();
        // Non-async: can't await, but the task will eventually clean up
    }
}
