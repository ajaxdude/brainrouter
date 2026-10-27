//! Native llama.cpp engine adapter (v2 L4).
//!
//! Spawns a direct `llama-server` (the toolbox wrapper precedent) as a
//! **transient systemd user scope** — the podman-scope pattern — so the child
//! escapes the daemon's own cgroup. Under the service's
//! `KillMode=control-group`, a directly attached child would die with the
//! brainrouter unit, which makes adoption impossible. Where systemd is absent
//! (development boxes) the adapter falls back to `setsid`; adoption is then
//! best-effort and the Engines tab says so.
//!
//! The child holds the per-GGUF `flock` (inherited fd, close-on-exec cleared;
//! see [`crate::swap_scheduler::GgufLock`]). Stop is SIGTERM → grace → KILL,
//! plus `systemctl --user stop <scope>` so the whole scope dies.
//!
//! Health is synchronous HTTP/1.1 over the assigned loopback port:
//! `/health` says ready (`200`) or loading (`503`/connection refused while the
//! process is alive), `/v1/models` supplies the stem SET for the spawn-time
//! assertion. llama-server offers no load-progress endpoint, so progress is
//! indeterminate + elapsed (ETA comes from the registry's EWMA).

use anyhow::{Context, Result};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::swap_scheduler::{
    EngineAdapter, EngineHandle, GgufLock, HealthReport, LoadProgress, SpawnRequest,
};

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Whether `systemd-run --user --scope` works here. Cached for the process.
static SYSTEMD_SCOPE_AVAILABLE: LazyLock<bool> = LazyLock::new(|| {
    if std::env::var("BRAINROUTER_SWAP_NO_SYSTEMD").as_deref() == Ok("1") {
        return false;
    }
    std::process::Command::new("systemd-run")
        .args(["--user", "--scope", "--quiet", "true"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
});

pub fn systemd_scope_available() -> bool {
    *SYSTEMD_SCOPE_AVAILABLE
}

/// The exact argv the adapter would run, exposed for tests and the Engines tab.
pub fn build_command_line(
    request: &SpawnRequest,
    use_scope: bool,
) -> (String, Vec<String>) {
    let args = request.resolved_args();
    if use_scope {
        let mut out = vec![
            "--user".to_string(),
            "--scope".to_string(),
            "--quiet".to_string(),
            "--collect".to_string(),
            "-p".to_string(),
            "KillMode=control-group".to_string(),
        ];
        if let Some(scope) = &request.scope_name {
            out.push("--unit".to_string());
            out.push(scope.clone());
        }
        out.push(request.spec.program.clone());
        out.extend(args);
        ("systemd-run".to_string(), out)
    } else {
        (request.spec.program.clone(), args)
    }
}

/// Resolve the scope's ENGINE pid. `systemd-run --scope` has two behaviors
/// across systemd versions: it may `exec` the command (the pid we spawned IS
/// the engine), or fork and wait (the spawned pid is a `systemd-run` wrapper
/// and the engine is its child). Track either correctly — killing/health-
/// checking a transient wrapper would leave the engine and its GGUF flock
/// behind. Resolution order:
///   1. the scope's cgroup (`cgroup.procs`), skipping the wrapper;
///   2. an exec'd wrapper (its argv is no longer `systemd-run`);
///   3. `/proc/<wrapper>/children`.
pub fn resolve_scope_engine_pid(
    scope: &str,
    wrapper_pid: u32,
    timeout: Duration,
) -> Option<u32> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if let Some(pids) = scope_pids(scope) {
            if let Some(engine) = pids
                .iter()
                .copied()
                .find(|pid| *pid != wrapper_pid && pid_alive(*pid))
            {
                return Some(engine);
            }
            if pids.contains(&wrapper_pid) && wrapper_is_exec(wrapper_pid) {
                return Some(wrapper_pid);
            }
        }
        // A scope that has already exec'd: the spawned pid is the engine.
        if wrapper_is_exec(wrapper_pid) {
            return Some(wrapper_pid);
        }
        if let Some(pid) = proc_children(wrapper_pid).into_iter().next() {
            return Some(pid);
        }
        if std::time::Instant::now() > deadline {
            return None;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// The unit name systemd actually created (`systemd-run --unit foo` creates
/// `foo.scope`; `systemctl` without the suffix looks for a service).
pub fn scope_unit_name(scope: &str) -> String {
    if scope.ends_with(".scope") || scope.ends_with(".service") {
        scope.to_string()
    } else {
        format!("{scope}.scope")
    }
}

fn wrapper_is_exec(pid: u32) -> bool {
    let args = crate::swap_scheduler::process_args(pid);
    match args.first() {
        Some(first) => !first
            .rsplit('/')
            .next()
            .unwrap_or(first)
            .starts_with("systemd-run"),
        None => false,
    }
}

/// PIDs in the transient scope's cgroup (cgroup v2 path under /sys/fs/cgroup).
fn scope_pids(scope: &str) -> Option<Vec<u32>> {
    let unit = scope_unit_name(scope);
    let output = std::process::Command::new("systemctl")
        .args(["--user", "show", "-p", "ControlGroup", "--value", &unit])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    let group = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if group.is_empty() {
        return None;
    }
    let path = if group.starts_with('/') {
        format!("/sys/fs/cgroup{group}")
    } else {
        format!("/sys/fs/cgroup/{group}")
    };
    let text = std::fs::read_to_string(format!("{path}/cgroup.procs")).ok()?;
    Some(
        text.split_whitespace()
            .filter_map(|p| p.parse::<u32>().ok())
            .collect(),
    )
}

fn proc_children(pid: u32) -> Vec<u32> {
    // A multi-threaded parent forks from any thread: scan every task, not
    // just the thread-group leader (the daemon spawns from tokio workers).
    let mut children = Vec::new();
    let Ok(tasks) = std::fs::read_dir(format!("/proc/{pid}/task")) else {
        return children;
    };
    for task in tasks.flatten() {
        if let Ok(text) = std::fs::read_to_string(task.path().join("children")) {
            children.extend(
                text.split_whitespace().filter_map(|p| p.parse::<u32>().ok()),
            );
        }
    }
    children
}

pub struct NativeLlamaCppAdapter {
    /// Path to the llama-server binary when the catalog entry leaves
    /// `launch_spec.program` empty (config/env default).
    default_program: String,
    /// Owned child handles, keyed by endpoint, so exit detection uses
    /// `try_wait` (a zombie still answers `kill(pid, 0)` and would otherwise
    /// look alive forever).
    children: std::sync::Mutex<std::collections::HashMap<String, std::process::Child>>,
}

impl Default for NativeLlamaCppAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeLlamaCppAdapter {
    pub fn new() -> Self {
        Self {
            default_program: std::env::var("BRAINROUTER_LLAMA_SERVER")
                .unwrap_or_else(|_| "llama-server".to_string()),
            children: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    pub fn with_program(program: impl Into<String>) -> Self {
        Self {
            default_program: program.into(),
            children: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// `try_wait` for the owned child: `None` = still running (or unknown /
    /// adopted child), `Some(())` = exited (and reaped).
    fn reaped(&self, endpoint: &str) -> Option<std::process::ExitStatus> {
        let mut children = self
            .children
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match children.get_mut(endpoint) {
            Some(child) => match child.try_wait() {
                Ok(Some(status)) => Some(status),
                _ => None,
            },
            None => None,
        }
    }
}

impl Drop for NativeLlamaCppAdapter {
    fn drop(&mut self) {
        // Deliberately NO kill: a daemon shutdown/upgrade must leave the
        // engine serving (that is what adoption relies on). The wrapper is
        // reaped by init; the engine lives in its own scope or session.
        let _ = self
            .children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .drain();
    }
}

impl EngineAdapter for NativeLlamaCppAdapter {
    fn name(&self) -> &'static str {
        "llama_cpp"
    }

    fn probe(&self) -> Result<()> {
        let program = &self.default_program;
        let status = std::process::Command::new(program)
            .arg("--version")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        match status {
            Ok(_) => Ok(()),
            Err(e) => anyhow::bail!("llama-server binary `{program}` unavailable: {e}"),
        }
    }

    fn spawn(&self, request: &SpawnRequest) -> Result<EngineHandle> {
        let mut program = request.spec.program.clone();
        if program.is_empty() {
            program = self.default_program.clone();
        }
        let mut effective = request.clone();
        effective.spec.program = program;

        let lock = match &effective.gguf {
            Some(gguf) => match GgufLock::try_acquire(gguf) {
                Ok(Some(lock)) => Some(lock),
                Ok(None) => anyhow::bail!(
                    "GGUF `{}` is locked by another running engine",
                    gguf.display()
                ),
                Err(e) => anyhow::bail!("failed to lock `{}`: {e}", gguf.display()),
            },
            None => None,
        };

        let use_scope = systemd_scope_available();
        let (command, args) = build_command_line(&effective, use_scope);
        if let Some(parent) = effective.log_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let stdout = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&effective.log_path)
            .with_context(|| format!("opening engine log {}", effective.log_path.display()))?;
        let stderr = stdout.try_clone().context("cloning engine log handle")?;

        let mut cmd = std::process::Command::new(&command);
        cmd.args(&args);
        cmd.stdin(std::process::Stdio::null());
        cmd.stdout(std::process::Stdio::from(stdout));
        cmd.stderr(std::process::Stdio::from(stderr));
        for (key, value) in &effective.spec.env {
            cmd.env(key, value);
        }
        for (key, value) in &effective.env {
            cmd.env(key, value);
        }
        if let Some(lock) = &lock {
            lock.inherit_fd_in(&mut cmd);
        }
        if !use_scope {
            // Plain setsid fallback: the child becomes a session leader, so a
            // daemon restart does not signal it. Adoption is best-effort here
            // (the unit's cgroup can still reap it on a full systemd restart).
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
        let child = cmd
            .spawn()
            .with_context(|| format!("spawning `{command}` for {}", effective.model_id))?;
        let wrapper_pid = child.id();
        // The tracked pid is the ENGINE, never the systemd-run wrapper.
        let pid = if use_scope {
            match effective.scope_name.as_deref().and_then(|scope| {
                resolve_scope_engine_pid(scope, wrapper_pid, Duration::from_secs(3))
            }) {
                Some(pid) => pid,
                None => {
                    if let Some(scope) = &effective.scope_name {
                        let unit = scope_unit_name(scope);
                        let _ = std::process::Command::new("systemctl")
                            .args(["--user", "stop", &unit])
                            .stdin(std::process::Stdio::null())
                            .stdout(std::process::Stdio::null())
                            .stderr(std::process::Stdio::null())
                            .status();
                    }
                    let mut child = child;
                    let _ = child.kill();
                    let _ = child.wait();
                    anyhow::bail!(
                        "systemd scope `{}` started but its MainPID could not be resolved; refusing to track the wrapper pid (adoption/stop would target the wrong process)",
                        effective.scope_name.as_deref().unwrap_or("<unnamed>")
                    );
                }
            }
        } else {
            wrapper_pid
        };
        self.children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(effective.endpoint.clone(), child);
        // The parent's copy goes away; the locked fd lives in the child.
        drop(lock);
        Ok(EngineHandle {
            pid,
            endpoint: effective.endpoint.clone(),
            started_at: now_ms(),
            scope_name: effective.scope_name.clone(),
        })
    }

    fn stop(&self, handle: &EngineHandle) -> Result<()> {
        // Preferred: group TERM through the scope (KillMode=control-group).
        // `--no-block` keeps the call bounded; we then apply the spec's
        // SIGTERM → grace → SIGKILL to the ENGINE pid and reap the wrapper.
        if let Some(scope) = &handle.scope_name {
            let unit = scope_unit_name(scope);
            let _ = std::process::Command::new("systemctl")
                .args(["--user", "stop", "--no-block", &unit])
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
        }
        terminate(handle.pid, Duration::from_secs(10));
        if pid_alive(handle.pid) {
            anyhow::bail!("engine pid {} survived SIGTERM+grace+SIGKILL", handle.pid);
        }
        // Reap our wrapper (scope: systemd-run; setsid: the engine itself).
        if let Some(mut child) = self
            .children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&handle.endpoint)
        {
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => break,
                    _ if std::time::Instant::now() > deadline => {
                        let _ = child.kill();
                        let _ = child.wait();
                        break;
                    }
                    _ => std::thread::sleep(Duration::from_millis(20)),
                }
            }
        }
        Ok(())
    }

    fn health(&self, endpoint: &str, handle: &EngineHandle) -> Result<HealthReport> {
        // A reaped own-child is authoritative (setsid fallback: pid == child).
        // In scope mode the tracked pid is the ENGINE, so probe it directly;
        // the wrapper's exit alone does not prove the engine is gone.
        let owned_reaped = self.reaped(endpoint).is_some()
            && self
                .children
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(endpoint)
                .map(|child| child.id() == handle.pid)
                .unwrap_or(false);
        let alive = !owned_reaped && unsafe { libc::kill(handle.pid as i32, 0) == 0 };
        let Some((status, body)) = http_get(endpoint, "/health", Duration::from_millis(800)) else {
            if alive {
                // Boot window: the process is up but not accepting yet.
                return Ok(HealthReport::loading());
            }
            anyhow::bail!("engine pid {} is not running", handle.pid);
        };
        if status != 200 {
            if alive {
                return Ok(HealthReport::loading());
            }
            anyhow::bail!("engine pid {} is not running", handle.pid);
        }
        let _ = body;
        let mut stems = std::collections::BTreeSet::new();
        if let Some((status, body)) = http_get(endpoint, "/v1/models", Duration::from_millis(800)) {
            if status == 200 {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                    if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
                        for model in data {
                            if let Some(id) = model.get("id").and_then(|i| i.as_str()) {
                                stems.insert(id.to_string());
                            }
                        }
                    }
                }
            }
        }
        Ok(HealthReport { ready: true, stems })
    }

    fn progress(&self, _endpoint: &str, handle: &EngineHandle) -> LoadProgress {
        // llama-server exposes no load-progress endpoint.
        LoadProgress {
            fraction: None,
            elapsed_ms: now_ms().saturating_sub(handle.started_at),
            eta_ms: None,
        }
    }
}

fn pid_alive(pid: u32) -> bool {
    pid > 0 && unsafe { libc::kill(pid as i32, 0) == 0 }
}

fn terminate(pid: u32, grace: Duration) {
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

/// Minimal HTTP/1.1 GET (no async runtime, no reqwest blocking feature).
/// Returns `(status, body)`; `None` when the connection cannot be made.
pub fn http_get(base: &str, path: &str, timeout: Duration) -> Option<(u16, String)> {
    let rest = base
        .strip_prefix("http://")
        .unwrap_or(base)
        .trim_end_matches('/');
    let (host, port) = rest.rsplit_once(':')?;
    let port: u16 = port.parse().ok()?;
    let address = (host, port).to_socket_addrs().ok()?.next()?;
    let mut stream = TcpStream::connect_timeout(&address, timeout).ok()?;
    let _ = stream.set_read_timeout(Some(timeout));
    let _ = stream.set_write_timeout(Some(timeout));
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nAccept: application/json\r\n\r\n"
    )
    .ok()?;
    let mut raw = Vec::new();
    let _ = stream.read_to_end(&mut raw);
    let text = String::from_utf8_lossy(&raw);
    let status_line = text.lines().next()?;
    let status = status_line.split_whitespace().nth(1)?.parse().ok()?;
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, body)| body.to_string())
        .unwrap_or_default();
    Some((status, body))
}

/// Convenience for tests/tools: is `program` runnable on this host?
pub fn program_exists(program: &str) -> bool {
    if program.contains('/') {
        return PathBuf::from(program).is_file();
    }
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|dir| dir.join(program).is_file()))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swap_catalog::LaunchSpec;
    use std::collections::BTreeMap;

    fn request() -> SpawnRequest {
        SpawnRequest {
            model_id: "m".into(),
            spec: LaunchSpec {
                program: "/usr/bin/llama-server".into(),
                args: vec![
                    "--port".into(),
                    "${PORT}".into(),
                    "--model".into(),
                    "/models/m.gguf".into(),
                ],
                served_model_alias: "m".into(),
                ..LaunchSpec::default()
            },
            gguf: None,
            port: 5911,
            endpoint: "http://127.0.0.1:5911".into(),
            log_path: std::env::temp_dir().join("brainrouter-native-test.log"),
            scope_name: Some("brainrouter-swap-m-abc".into()),
            env: BTreeMap::new(),
        }
    }

    #[test]
    fn scope_command_line_uses_systemd_run_quiet_and_the_scope_unit() {
        let (program, args) = build_command_line(&request(), true);
        assert_eq!(program, "systemd-run");
        assert!(args.contains(&"--user".to_string()));
        assert!(args.contains(&"--scope".to_string()));
        assert!(args.contains(&"--quiet".to_string()));
        assert!(args.windows(2).any(|w| w == ["--unit", "brainrouter-swap-m-abc"]));
        assert!(args.contains(&"/usr/bin/llama-server".to_string()));
        // ${PORT} resolved.
        assert!(args.contains(&"5911".to_string()));
        assert!(!args.iter().any(|a| a.contains("${PORT}")));
    }

    #[test]
    fn fallback_command_line_is_plain_setsid_argv() {
        let (program, args) = build_command_line(&request(), false);
        assert_eq!(program, "/usr/bin/llama-server");
        assert_eq!(&args[..2], &["--port".to_string(), "5911".to_string()]);
    }

    #[test]
    fn port_flag_is_appended_when_the_spec_has_none() {
        let mut request = request();
        request.spec.args = vec!["--model".into(), "/models/m.gguf".into()];
        let args = request.resolved_args();
        assert_eq!(&args[args.len() - 2..], &["--port".to_string(), "5911".to_string()]);
    }

    #[test]
    fn scope_unit_name_has_the_scope_suffix() {
        assert_eq!(scope_unit_name("brainrouter-swap-m-abc"), "brainrouter-swap-m-abc.scope");
        assert_eq!(scope_unit_name("x.scope"), "x.scope");
        assert_eq!(scope_unit_name("x.service"), "x.service");
    }

    #[test]
    fn proc_children_lists_a_spawned_child() {
        let mut child = std::process::Command::new("sleep")
            .arg("5")
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let mut found = false;
        for _ in 0..100 {
            if proc_children(std::process::id()).contains(&child.id()) {
                found = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let _ = child.kill();
        let _ = child.wait();
        assert!(found, "the child pid must appear in /proc/<self>/children");
    }

    #[test]
    fn http_get_reports_failure_for_a_dead_port() {
        assert!(http_get("http://127.0.0.1:1", "/health", Duration::from_millis(100)).is_none());
    }

    #[test]
    fn http_get_parses_a_minimal_response() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 512];
                let _ = stream.read(&mut buf);
                let body = r#"{"data":[{"id":"m"}]}"#;
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        let (status, body) =
            http_get(&format!("http://127.0.0.1:{port}"), "/v1/models", Duration::from_secs(1))
                .unwrap();
        assert_eq!(status, 200);
        assert!(
            body.contains("\"id\":\"m\""),
            "parsed body: {body:?}"
        );
    }
}
