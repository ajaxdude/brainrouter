//! Plugin supervisor (design `docs/design/v2-streamlining.L2.md`).
//!
//! One task per plugin. Spawns the enabled plugin's child, monitors it, and
//! applies the contract:
//! - crash (nonzero/signal exit) → exponential backoff, capped at 60 s;
//! - 5 consecutive crashes → terminal `failed` state: no supervision until an
//!   explicit `plugin enable`;
//! - clean exit (status 0) → `stopped`, no restart;
//! - stop = SIGTERM with grace, then SIGKILL by PID (tokio `Child` drop does
//!   NOT kill — this repo's rule — so the kill is explicit);
//! - an `initialize` auth failure on the RPC socket is NOT a crash (the
//!   supervisor never sees it: it is connection-level, handled in `rpc`).

use super::registry::PluginEntry;
use super::{Inner, SupCmd};
use crate::plugins::{registry::PluginState, store};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::process::ExitStatusExt;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{info, warn};

/// Consecutive crashes before the terminal `failed` state.
pub const CRASH_LIMIT: u32 = 5;
/// Base backoff; doubles per consecutive crash, capped at 60 s (design).
pub const BACKOFF_BASE: Duration = Duration::from_secs(1);
pub const BACKOFF_CAP: Duration = Duration::from_secs(60);
/// SIGTERM grace before SIGKILL.
pub const STOP_GRACE: Duration = Duration::from_secs(5);
/// A child that lives this long resets the consecutive-crash counter.
pub const STABLE_UPTIME: Duration = Duration::from_secs(60);

pub(crate) async fn supervisor_loop(inner: Arc<Inner>, id: String, mut rx: mpsc::UnboundedReceiver<SupCmd>) {
    // Boot/enable entrypoint: read the registry decision fresh.
    let (enabled, failed, mut consecutive) = {
        let reg = inner.registry.lock().unwrap();
        match reg.plugins.get(&id) {
            Some(e) => (e.enabled, e.state == PluginState::Failed, e.consecutive_crashes),
            None => (false, false, 0),
        }
    };
    if !enabled || failed {
        finish(&inner, &id);
        return;
    }

    loop {
        let mut child = match spawn_child(&inner, &id).await {
            Ok(child) => child,
            Err(e) => {
                // A spawn failure counts like a crash (a missing binary is a
                // permanent one → the same terminal path bounds the loop).
                consecutive += 1;
                warn!(plugin = %id, error = %e, consecutive_crashes = consecutive,
                    "failed to spawn plugin child");
                if consecutive >= CRASH_LIMIT {
                    set_failed(&inner, &id, &e.to_string(), consecutive);
                    warn!(plugin = %id,
                        "plugin reached terminal `failed` state after {CRASH_LIMIT} consecutive failures; \
                         no further supervision until `brainrouter plugin enable {id}`");
                    finish(&inner, &id);
                    return;
                }
                if !sleep_backoff(&mut rx, &inner, &id, consecutive).await {
                    return;
                }
                continue;
            }
        };
        let started = tokio::time::Instant::now();
        set_running(&inner, &id, child.id());
        // Monitor: child exit vs stop command.
        let outcome: Result<std::process::ExitStatus, StopReason> = loop {
            tokio::select! {
                status = child.wait() => break Ok(status.unwrap_or_else(|e| {
                    warn!(plugin = %id, error = %e, "failed to reap plugin child");
                    clean_exit_status()
                })),
                cmd = rx.recv() => match cmd {
                    Some(SupCmd::Stop) => break Err(StopReason::Disable),
                    Some(SupCmd::Shutdown) | None => break Err(StopReason::Shutdown),
                    Some(SupCmd::Start) => {}
                },
            }
        };
        let status = match outcome {
            Err(reason) => {
                stop_child(&mut child).await;
                info!(plugin = %id, ?reason, "plugin child stopped by supervisor");
                set_stopped(&inner, &id);
                finish(&inner, &id);
                return;
            }
            Ok(status) => status,
        };
        let exit_text = describe_exit(&status);
        if status.success() {
            info!(plugin = %id, "{exit_text}: plugin exited cleanly, no restart");
            set_stopped_clean(&inner, &id, &exit_text);
            finish(&inner, &id);
            return;
        }
        if started.elapsed() >= STABLE_UPTIME {
            consecutive = 0;
        }
        consecutive += 1;
        warn!(plugin = %id, exit = %exit_text, consecutive_crashes = consecutive,
            "plugin child crashed");
        if consecutive >= CRASH_LIMIT {
            set_failed(&inner, &id, &exit_text, consecutive);
            warn!(plugin = %id,
                "plugin reached terminal `failed` state after {CRASH_LIMIT} consecutive crashes; \
                 no further supervision until `brainrouter plugin enable {id}`");
            finish(&inner, &id);
            return;
        }
        set_crashed(&inner, &id, &exit_text, consecutive);
        if !sleep_backoff(&mut rx, &inner, &id, consecutive).await {
            return;
        }
    }
}

#[derive(Debug)]
enum StopReason {
    Disable,
    Shutdown,
}

/// `true` = keep supervising, `false` = a stop command arrived.
async fn sleep_backoff(
    rx: &mut mpsc::UnboundedReceiver<SupCmd>,
    inner: &Arc<Inner>,
    id: &str,
    consecutive: u32,
) -> bool {
    let delay = backoff(consecutive);
    info!(plugin = %id, restart_in = ?delay, "restarting plugin after crash backoff");
    tokio::select! {
        _ = tokio::time::sleep(delay) => true,
        cmd = rx.recv() => match cmd {
            Some(SupCmd::Stop) | Some(SupCmd::Shutdown) | None => {
                set_stopped(inner, id);
                finish(inner, id);
                false
            }
            Some(SupCmd::Start) => true,
        },
    }
}

/// SIGTERM with grace, then SIGKILL by PID (design: "stop = SIGTERM with
/// grace, then kill (tokio-Child PID rule)").
async fn stop_child(child: &mut tokio::process::Child) {
    if let Some(pid) = child.id() {
        unsafe {
            libc::kill(pid as i32, libc::SIGTERM);
        }
    }
    if tokio::time::timeout(STOP_GRACE, child.wait()).await.is_err() {
        if let Some(pid) = child.id() {
            warn!(pid, "plugin child ignored SIGTERM; sending SIGKILL");
        }
        let _ = child.start_kill();
        let _ = tokio::time::timeout(STOP_GRACE, child.wait()).await;
    }
}

fn backoff(consecutive: u32) -> Duration {
    let ms = BACKOFF_BASE.as_millis() as u64 * 2u64.pow(consecutive.saturating_sub(1).min(16));
    Duration::from_millis(ms).min(BACKOFF_CAP)
}

async fn spawn_child(inner: &Arc<Inner>, id: &str) -> anyhow::Result<tokio::process::Child> {
    let (binary, token, state_dir) = {
        let reg = inner.registry.lock().unwrap();
        let entry = reg
            .plugins
            .get(id)
            .ok_or_else(|| anyhow::anyhow!("plugin '{id}' is not installed"))?;
        let dir = store::plugin_state_dir(&inner.state_root, id);
        (dir.join(&entry.binary), dir.join("token"), dir)
    };
    let token_text = std::fs::read_to_string(&token)
        .map(|t| t.trim().to_string())
        .unwrap_or_default();
    if token_text.is_empty() {
        anyhow::bail!("plugin '{id}' has no minted token (reinstall it)");
    }
    let log_path = state_dir.join("plugin.log");
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open(&log_path)
        .map_err(|e| anyhow::anyhow!("opening plugin log {}: {e}", log_path.display()))?;
    let stdout_log = log.try_clone()?;

    let mut cmd = tokio::process::Command::new(&binary);
    cmd.current_dir(&state_dir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(stdout_log))
        .stderr(std::process::Stdio::from(log))
        // Design: the token reaches the child ONLY via this env var.
        .env("BRAINROUTER_PLUGIN_TOKEN", token_text)
        .env("BRAINROUTER_PLUGIN_ID", id)
        .env(
            "BRAINROUTER_PLUGIN_SOCKET",
            store::plugin_socket_path(&inner.runtime_root, id),
        )
        .env("BRAINROUTER_PLUGIN_STATE", &state_dir)
        .env("BRAINROUTER_DAEMON_SOCKET", daemon_socket_hint());
    let child = cmd
        .spawn()
        .map_err(|e| anyhow::anyhow!("spawning {}: {e}", binary.display()))?;
    info!(plugin = %id, pid = ?child.id(), binary = %binary.display(), "plugin child spawned");
    Ok(child)
}

fn daemon_socket_hint() -> String {
    crate::config::default_socket_path().display().to_string()
}

/// Synthetic success status for the reap-failure path (treated as clean
/// exit, which stops the loop — the safe direction).
fn clean_exit_status() -> std::process::ExitStatus {
    use std::os::unix::process::ExitStatusExt;
    std::process::ExitStatus::from_raw(0)
}

fn describe_exit(status: &std::process::ExitStatus) -> String {
    match (status.code(), status.signal()) {
        (Some(code), _) => format!("exit code {code}"),
        (None, Some(sig)) => format!("killed by signal {sig}"),
        (None, None) => "unknown exit".to_string(),
    }
}

// ── Registry persistence helpers ────────────────────────────────────────────

fn mutate<F: FnOnce(&mut PluginEntry)>(inner: &Arc<Inner>, id: &str, f: F) {
    let mut reg = inner.registry.lock().unwrap();
    if let Some(entry) = reg.plugins.get_mut(id) {
        f(entry);
        if let Err(e) = reg.save(&inner.state_root) {
            warn!(plugin = %id, error = %e, "failed to persist plugin registry");
        }
    }
}

fn set_running(inner: &Arc<Inner>, id: &str, pid: Option<u32>) {
    mutate(inner, id, |e| {
        e.state = PluginState::Running;
        e.pid = pid;
        e.last_exit = None;
    });
}

fn set_crashed(inner: &Arc<Inner>, id: &str, exit: &str, consecutive: u32) {
    mutate(inner, id, |e| {
        e.state = PluginState::Running; // supervised: restarting
        e.consecutive_crashes = consecutive;
        e.total_restarts += 1;
        e.last_exit = Some(exit.to_string());
        e.pid = None;
    });
}

fn set_failed(inner: &Arc<Inner>, id: &str, exit: &str, consecutive: u32) {
    mutate(inner, id, |e| {
        e.state = PluginState::Failed;
        e.consecutive_crashes = consecutive;
        e.last_exit = Some(exit.to_string());
        e.pid = None;
        e.status_line = format!("failed after {consecutive} consecutive crashes ({exit})");
    });
}

fn set_stopped(inner: &Arc<Inner>, id: &str) {
    mutate(inner, id, |e| {
        e.state = PluginState::Stopped;
        e.consecutive_crashes = 0;
        e.pid = None;
    });
}

fn set_stopped_clean(inner: &Arc<Inner>, id: &str, exit: &str) {
    mutate(inner, id, |e| {
        e.state = PluginState::Stopped;
        e.consecutive_crashes = 0;
        e.last_exit = Some(exit.to_string());
        e.pid = None;
    });
}

fn finish(inner: &Arc<Inner>, id: &str) {
    // Supervisors self-remove so `shutdown` can bound its wait on this map.
    inner.supervisors.lock().unwrap().remove(id);
    // Sessions die with the child; drop them so config.changed stops queueing.
    inner.sessions.lock().unwrap().remove(id);
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_caps_at_60s() {
        assert_eq!(backoff(1), Duration::from_secs(1));
        assert_eq!(backoff(2), Duration::from_secs(2));
        assert_eq!(backoff(6), Duration::from_secs(32));
        assert_eq!(backoff(7), BACKOFF_CAP);
        assert_eq!(backoff(50), BACKOFF_CAP);
    }

    #[test]
    fn crash_limit_is_five() {
        assert_eq!(CRASH_LIMIT, 5);
    }
}
