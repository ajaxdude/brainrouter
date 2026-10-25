//! Shared engine-process plumbing (v2 L5).
//!
//! The L4 adapter (`engine_native`) and the L5 strixite/hipfire adapters all
//! need the same three invariants: a transient systemd user scope around the
//! child (cgroup escape, the podman-scope precedent), the child-held GGUF
//! flock (inherited fd), and a SIGTERM → grace → SIGKILL stop. This module is
//! the single copy of the spawn/stop half so the adapters stay data + argv.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

use crate::swap_scheduler::GgufLock;

/// The exact `(command, args)` for an engine spawn. `use_scope` wraps the
/// program in `systemd-run --user --scope` (the L4 cgroup-escape binding);
/// `scope_name` becomes `--unit` so stop can `systemctl --user stop` the scope.
pub fn scoped_argv(
    program: &str,
    args: &[String],
    scope_name: Option<&str>,
    use_scope: bool,
) -> (String, Vec<String>) {
    if !use_scope {
        return (program.to_string(), args.to_vec());
    }
    let mut out = vec![
        "--user".to_string(),
        "--scope".to_string(),
        "--quiet".to_string(),
        "--collect".to_string(),
        "-p".to_string(),
        "KillMode=control-group".to_string(),
    ];
    if let Some(scope) = scope_name {
        out.push("--unit".to_string());
        out.push(scope.to_string());
    }
    out.push(program.to_string());
    out.extend(args.iter().cloned());
    ("systemd-run".to_string(), out)
}

pub struct EngineProcessRequest<'a> {
    pub program: &'a str,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub log_path: &'a Path,
    pub scope_name: Option<&'a str>,
    pub use_scope: bool,
    pub lock: Option<&'a GgufLock>,
}

/// Spawn the engine, append stdout+stderr to `log_path`, keep the child held
/// (so `try_wait` detects an exit), and hand the flock fd across exec.
pub fn spawn(request: EngineProcessRequest<'_>) -> Result<std::process::Child> {
    if let Some(parent) = request.log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let stdout = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(request.log_path)
        .with_context(|| format!("opening engine log {}", request.log_path.display()))?;
    let stderr = stdout.try_clone().context("cloning engine log handle")?;

    let (command, args) = scoped_argv(
        request.program,
        &request.args,
        request.scope_name,
        request.use_scope,
    );
    let mut cmd = std::process::Command::new(&command);
    cmd.args(&args);
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::from(stdout));
    cmd.stderr(std::process::Stdio::from(stderr));
    for (key, value) in &request.env {
        cmd.env(key, value);
    }
    if let Some(lock) = request.lock {
        lock.inherit_fd_in(&mut cmd);
    }
    if !request.use_scope {
        // Plain setsid fallback: the child becomes a session leader so a
        // daemon restart does not signal it (adoption best-effort).
        unsafe {
            use std::os::unix::process::CommandExt;
            cmd.pre_exec(|| {
                if libc::setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
    cmd.spawn()
        .with_context(|| format!("spawning `{command}` ({})", request.program))
}

/// SIGTERM → bounded `try_wait` → SIGKILL for an owned child, then reap.
/// (The L4 `Child::kill` is SIGKILL-only; the icon-era escalation window is
/// the spec's hung-start recovery.)
pub fn stop_child(child: &mut std::process::Child, grace: Duration) {
    unsafe {
        libc::kill(child.id() as i32, libc::SIGTERM);
    }
    let deadline = std::time::Instant::now() + grace;
    loop {
        if matches!(child.try_wait(), Ok(Some(_))) {
            return;
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// SIGTERM → poll → SIGKILL after `grace`.
pub fn terminate(pid: u32, grace: Duration) {
    if pid == 0 {
        return;
    }
    unsafe {
        libc::kill(pid as i32, libc::SIGTERM);
    }
    let deadline = std::time::Instant::now() + grace;
    while std::time::Instant::now() < deadline {
        if unsafe { libc::kill(pid as i32, 0) != 0 } {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    unsafe {
        libc::kill(pid as i32, libc::SIGKILL);
    }
}

/// Wait (bounded) until `pid` is gone. Returns the elapsed milliseconds.
pub fn wait_pid_gone(pid: u32, timeout: Duration) -> Option<u64> {
    let started = std::time::Instant::now();
    while started.elapsed() < timeout {
        if pid == 0 || unsafe { libc::kill(pid as i32, 0) != 0 } {
            return Some(started.elapsed().as_millis() as u64);
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    if pid == 0 || unsafe { libc::kill(pid as i32, 0) != 0 } {
        Some(started.elapsed().as_millis() as u64)
    } else {
        None
    }
}

/// Environment knobs shared by the L5 adapters (`BRAINROUTER_*`).
pub fn env_bool(key: &str) -> Option<bool> {
    std::env::var(key)
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
}

pub fn env_u64(key: &str) -> Option<u64> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

pub fn env_f64(key: &str) -> Option<f64> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

pub fn write_atomic_0600(path: &Path, bytes: &[u8]) -> Result<()> {
    crate::plugins::store::atomic_write_0600(path, bytes)
        .with_context(|| format!("writing {}", path.display()))
}

/// Append one timestamped line to an adapter log file (never fails a spawn).
pub fn append_log(path: &Path, line: &str) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(file, "{} {}", chrono::Utc::now().to_rfc3339(), line);
    }
}

/// Probe an endpoint's `/v1/models` stems (sync HTTP, short timeout).
pub fn probe_stems(endpoint: &str, timeout: Duration) -> Option<Vec<String>> {
    let (status, body) = crate::engine_native::http_get(endpoint, "/v1/models", timeout)?;
    if status != 200 {
        return None;
    }
    let json: serde_json::Value = serde_json::from_str(&body).ok()?;
    Some(
        json.get("data")?
            .as_array()?
            .iter()
            .filter_map(|m| m.get("id").and_then(|i| i.as_str()).map(String::from))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_argv_wraps_and_plain_argv_does_not() {
        let args = vec!["--config".to_string(), "/tmp/x.conf".to_string()];
        let (program, scoped) =
            scoped_argv("strix_server", &args, Some("brainrouter-swap-x"), true);
        assert_eq!(program, "systemd-run");
        assert!(scoped.contains(&"--unit".to_string()));
        assert!(scoped.contains(&"strix_server".to_string()));

        let (program, plain) = scoped_argv("strix_server", &args, None, false);
        assert_eq!(program, "strix_server");
        assert_eq!(plain, args);
    }
}
