//! Acceptance tests for leg L7 (llama-swap absorbed: `swap:` section, the
//! native flip gate, flip-mode consumer pinning) of
//! `docs/design/v2-streamlining.md` — spec: `docs/design/v2-streamlining.L7.md`
//! (L7-v3), log: `docs/design/v2-streamlining.L7.log.md`.
//!
//! AUTHORSHIP DISCLOSURE (I8 process): the designated suite author (Ray-L7c)
//! was quota-blocked across four consecutive attempts (429, synthetic provider
//! exhaustion, 2026-10-26). These rows were therefore WRITTEN BY THE BUILDER
//! from the leg spec's promises, stand ALONE-NEW (no frozen L1–L6 file was
//! touched), and are submitted for author ratification exactly as the L6
//! `test_root` amendment was. Ratification is owed; the rows themselves cite
//! the spec text they pin.
//!
//! Black-box only: live daemon over loopback, the CLI binary, and documented
//! state files. No `src/` Rust API is referenced.
//!
//! Rows (spec promises → test):
//! - Flip gate (`enforce_native_flip_gate`, L7 §"boot refuses"):
//!   * `l7_flip_gate_refuses_boot_with_loaded_shim` — shim holds a model:
//!     boot exits non-zero naming it; nothing listens.
//!   * `l7_flip_gate_unknown_state_refuses` — non-2xx / unparseable / timeout
//!     are UNKNOWN, not "empty": boot refuses and quotes the probe failure.
//!   * `l7_flip_gate_boots_on_empty_and_unreachable_shim` — `{"running":[]}`
//!     and a dead port ("nothing is listening = nothing is resident") boot.
//! - `v2.0-p7` migration (L7 §config surface): textual rename, comments and
//!   untouched lines byte-identical, snapshot + restore:
//!   * `l7_p7_textual_rename_snapshot_and_restore`
//! - Group gate (L7 §groups deferred to owner decision):
//!   * `l7_groups_gate_native` — spawn of a group member refused with
//!     `blocked:groups-pending`; group names are not routable; ordinary
//!     models unaffected.
//! - Mode pinning (L7 §"native never dials the shim after the boot gate"):
//!   * `l7_native_never_dials_shim_after_boot`
//! - Completeness reporting (recovery review I5):
//!   * `l7_flush_and_restart_report_completeness`
//!
//! GAPS this file declares instead of silently asserting:
//! - Groups SEMANTICS (swap/exclusive/persistent scheduling) are out of scope
//!   by spec — only the REFUSAL contract is pinned.
//! - Crash-restart bounds, `shutdown_all` cleanliness, and adoption across a
//!   real daemon restart are soak-tier (systemd + real engines): see
//!   `docs/design/v2-streamlining.L7.soak.md`; a fixture daemon here cannot
//!   model SIGTERM grace against a real llama-server.
//! - The I4 engine-hint event field is unit-tier (`routing_events::tests`);
//!   no fixture-native stream exists to emit one end-to-end (the scripted
//!   `fake` adapter never answers chat).

use serde_json::{json, Value};
use std::fs;
use std::future::Future;
use std::net::TcpListener;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

static SUITE_GATE: std::sync::LazyLock<tokio::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));

async fn gate() -> Option<tokio::sync::MutexGuard<'static, ()>> {
    if std::env::var("RAY_L7_NO_SERIAL_GATE").is_ok() {
        return None;
    }
    Some(SUITE_GATE.lock().await)
}

fn unique_suffix() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        SEQ.fetch_add(1, Ordering::SeqCst)
    )
}

/// Short, shallow test root (UDS `SUN_LEN` budget) — the L6-ratified pattern:
/// a readable unique name hashed (FNV-1a) to a 16-hex leaf under the on-disk
/// `CARGO_TARGET_TMPDIR`, surfaced through a `/tmp/br<pid>-<n>` symlink so the
/// daemon's socket path stays well under 108 bytes.
fn test_root(tag: &str) -> PathBuf {
    let readable = format!("brainrouter-l7-{tag}-{}", unique_suffix());
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in readable.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let leaf = format!("{hash:016x}");
    let base = match option_env!("CARGO_TARGET_TMPDIR") {
        Some(dir) => PathBuf::from(dir),
        None => std::env::temp_dir(),
    };
    let real = base.join(&leaf);
    fs::create_dir_all(&real).unwrap();
    static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let link = std::env::temp_dir().join(format!(
        "br{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    match std::os::unix::fs::symlink(&real, &link) {
        Ok(()) => link,
        Err(_) => real,
    }
}

fn remove_test_root(root: &Path) {
    let target = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    if !target.exists() {
        let _ = fs::remove_file(root);
        return;
    }
    if fs::remove_dir_all(&target).is_err() {
        let _ = Command::new("podman")
            .arg("unshare")
            .args(["rm", "-rf"])
            .arg(&target)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    if root.is_symlink() {
        let _ = fs::remove_file(root);
    }
}

struct TestEnv {
    root: PathBuf,
    owned: bool,
}

impl TestEnv {
    fn new(tag: &str) -> Self {
        let root = test_root(tag);
        for dir in ["home", "config", "data", "state", "run"] {
            fs::create_dir_all(root.join(dir)).unwrap();
        }
        fs::set_permissions(root.join("run"), fs::Permissions::from_mode(0o700)).unwrap();
        fs::create_dir_all(root.join("state/brainrouter")).unwrap();
        Self { root, owned: true }
    }

    fn apply(&self, cmd: &mut Command, extra: &[(String, String)]) {
        cmd.env_clear()
            .env("HOME", self.root.join("home"))
            .env("XDG_CONFIG_HOME", self.root.join("config"))
            .env("XDG_DATA_HOME", self.root.join("data"))
            .env("XDG_STATE_HOME", self.root.join("state"))
            .env("XDG_RUNTIME_DIR", self.root.join("run"))
            .env("PATH", "/usr/local/bin:/usr/bin:/bin")
            .env("RUST_LOG", "debug")
            .envs(extra.iter().cloned());
    }

    fn state_dir(&self) -> PathBuf {
        self.root.join("state/brainrouter")
    }

    fn config_path(&self) -> PathBuf {
        self.root.join("config.yaml")
    }
}

impl Drop for TestEnv {
    fn drop(&mut self) {
        if self.owned {
            remove_test_root(&self.root);
        }
    }
}

// ── Shim mock ───────────────────────────────────────────────────────────────

/// llama-swap stand-in. `/running` responds with a scriptable status/body and
/// an optional pre-response delay; every request path is recorded.
struct ShimMock {
    url: String,
    reply: Arc<Mutex<MockReply>>,
    hits: Arc<Mutex<Vec<String>>>,
}

#[derive(Clone)]
struct MockReply {
    status: u16,
    body: String,
    delay_ms: u64,
}

impl ShimMock {
    #[allow(dead_code)]
    fn set(&self, reply: MockReply) {
        *self.reply.lock().unwrap() = reply;
    }

    fn hit_count(&self) -> usize {
        self.hits.lock().unwrap().len()
    }

    /// Paths hit since the returned snapshot (drains the log).
    fn take_hits(&self) -> Vec<String> {
        std::mem::take(&mut *self.hits.lock().unwrap())
    }
}

impl Default for ShimMock {
    fn default() -> Self {
        Self::with_reply(MockReply {
            status: 200,
            body: json!({"running": []}).to_string(),
            delay_ms: 0,
        })
    }
}

impl ShimMock {
    fn with_reply(reply: MockReply) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let reply = Arc::new(Mutex::new(reply));
        let hits = Arc::new(Mutex::new(Vec::new()));
        let (r2, h2) = (reply.clone(), hits.clone());
        // PLAIN OS THREADS, not tokio: the gate probes happen while the test
        // thread blocks synchronously on the booting child, so a task-based
        // mock on the test's current-thread runtime would never get polled
        // (connections would hang -> the very UNKNOWN state we are testing).
        std::thread::spawn(move || loop {
            let Ok((mut sock, _)) = listener.accept() else {
                continue;
            };
            let (r, h) = (r2.clone(), h2.clone());
            std::thread::spawn(move || {
                use std::io::{Read, Write};
                let _ = sock.set_read_timeout(Some(Duration::from_secs(15)));
                let mut buf = [0u8; 4096];
                let mut seen = String::new();
                let mut path = String::new();
                loop {
                    let Ok(n) = sock.read(&mut buf) else { return };
                    if n == 0 {
                        break;
                    }
                    seen.push_str(&String::from_utf8_lossy(&buf[..n]));
                    if let Some(line) = seen.lines().next() {
                        path = line.split_whitespace().nth(1).unwrap_or("").to_string();
                    }
                    if seen.contains("\r\n\r\n") {
                        break;
                    }
                }
                if path.is_empty() {
                    return;
                }
                h.lock().unwrap().push(path.clone());
                let MockReply {
                    status,
                    body,
                    delay_ms,
                } = r.lock().unwrap().clone();
                if delay_ms > 0 {
                    std::thread::sleep(Duration::from_millis(delay_ms));
                }
                if path.starts_with("/running") {
                    let reason = if status == 200 { "OK" } else { "Error" };
                    let _ = write!(
                        sock,
                        "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                } else {
                    let _ = write!(
                        sock,
                        "HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                    );
                }
            });
        });
        Self {
            url: format!("http://127.0.0.1:{port}"),
            reply,
            hits,
        }
    }
}

// ── Daemon harness ──────────────────────────────────────────────────────────

struct Daemon {
    child: std::process::Child,
    url: String,
    log: PathBuf,
}

impl Daemon {
    fn start(env: &TestEnv, config_yaml: &str) -> Self {
        fs::write(env.config_path(), config_yaml).unwrap();
        Self::boot(env)
    }

    fn boot(env: &TestEnv) -> Self {
        let mut attempt = 0u8;
        let (address, mut child) = loop {
            attempt += 1;
            let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = reservation.local_addr().unwrap();
            drop(reservation);
            let log = env.root.join("daemon.log");
            let log_file = fs::File::create(&log).unwrap();
            let mut cmd = Command::new(env!("CARGO_BIN_EXE_brainrouter"));
            cmd.args(["serve", "--config"])
                .arg(env.config_path())
                .args(["--tcp-addr", address.to_string().as_str(), "--socket"])
                .arg(env.root.join("d.sock"))
                .current_dir(&env.root)
                .stdin(Stdio::null())
                .stdout(Stdio::from(log_file.try_clone().unwrap()))
                .stderr(Stdio::from(log_file));
            env.apply(&mut cmd, &[]);
            let mut child = cmd.spawn().unwrap();
            std::thread::sleep(Duration::from_millis(150));
            if child.try_wait().unwrap().is_none() || attempt == 3 {
                break (address, child);
            }
            let text = fs::read_to_string(&log).unwrap_or_default();
            if !text.to_lowercase().contains("address already in use") {
                break (address, child);
            }
            let _ = child.kill();
            let _ = child.wait();
        };
        let log = env.root.join("daemon.log");
        Self {
            child,
            url: format!("http://{address}"),
            log,
        }
    }

    /// Spawn `serve` expecting a boot REFUSAL: returns (exit code, log text).
    /// Panics if the daemon comes up healthy instead.
    fn boot_expect_exit(env: &TestEnv, deadline: Instant) -> (Option<i32>, String) {
        let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = reservation.local_addr().unwrap();
        drop(reservation);
        let log = env.root.join("daemon.log");
        let log_file = fs::File::create(&log).unwrap();
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_brainrouter"));
        cmd.args(["serve", "--config"])
            .arg(env.config_path())
            .args(["--tcp-addr", address.to_string().as_str(), "--socket"])
            .arg(env.root.join("d.sock"))
            .current_dir(&env.root)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log_file.try_clone().unwrap()))
            .stderr(Stdio::from(log_file));
        env.apply(&mut cmd, &[]);
        let mut child = cmd.spawn().unwrap();
        let started = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            // A healthy listener means the boot was ACCEPTED — not what this
            // helper is for.
            if let Ok(probe) = std::net::TcpStream::connect(address) {
                let _ = probe.set_nonblocking(true);
                drop(probe);
                let _ = child.kill();
                let _ = child.wait();
                panic!(
                    "daemon booted despite the gate; log:\n{}",
                    fs::read_to_string(&log).unwrap_or_default()
                );
            }
            assert!(started < deadline, "daemon neither exited nor answered: boot should have refused within {deadline:?}: {}", fs::read_to_string(&log).unwrap_or_default());
            std::thread::sleep(Duration::from_millis(50));
        };
        let text = fs::read_to_string(&log).unwrap_or_default();
        drop(child);
        (status.code(), text)
    }

    async fn ready(&mut self) {
        let client = http();
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                panic!("daemon exited with {status}: {}", self.log_tail());
            }
            if let Ok(r) = client.get(format!("{}/health", self.url)).send().await {
                if r.status().is_success()
                    && r.json::<Value>().await.unwrap_or(Value::Null) == json!({"status": "ok"})
                {
                    return;
                }
            }
            assert!(
                Instant::now() < deadline,
                "daemon not healthy: {}",
                self.log_tail()
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    fn log_tail(&self) -> String {
        let text = fs::read_to_string(&self.log).unwrap_or_default();
        text.lines()
            .rev()
            .take(60)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}{path}", self.url)
    }

    async fn get_json(&self, path: &str) -> Value {
        let (status, text) = self.get_text(path).await;
        assert_eq!(status, 200, "GET {path} -> {status}: {text}");
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("GET {path} not JSON ({e}): {text}"))
    }

    async fn get_text(&self, path: &str) -> (u16, String) {
        let response = http()
            .get(self.endpoint(path))
            .send()
            .await
            .unwrap_or_else(|e| panic!("GET {path}: {e}"));
        let status = response.status().as_u16();
        (status, response.text().await.unwrap_or_default())
    }

    async fn post_json(&self, path: &str, body: Value) -> (u16, String) {
        let response = http()
            .post(self.endpoint(path))
            .json(&body)
            .send()
            .await
            .unwrap_or_else(|e| panic!("POST {path}: {e}"));
        let status = response.status().as_u16();
        (status, response.text().await.unwrap_or_default())
    }

    fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        self.stop();
    }
}

fn http() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(60))
        .build()
        .unwrap()
}

async fn poll_until<T, F, Fut>(what: &str, secs: u64, mut f: F) -> T
where
    F: FnMut() -> Fut + Send,
    Fut: Future<Output = Option<T>> + Send,
{
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        if let Some(value) = f().await {
            return value;
        }
        assert!(Instant::now() < deadline, "timeout waiting for {what}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

// ── Fixtures ────────────────────────────────────────────────────────────────

/// Native-mode config. `shim_url` is the (deprecated, fixture-essential)
/// `base_url` the flip gate probes — honored for non-default endpoints.
fn native_config(shim_url: &str) -> String {
    serde_yaml::to_string(&json!({
        "manifest": { "enabled": false, "base_url": "http://127.0.0.1:1/cloud/v1" },
        "swap": {
            "mode": "native",
            "base_url": format!("{shim_url}/v1"),
            "fallback_model": "ray-l7-solo",
            "local_models": [],
        },
        "bonsai": { "enabled": false, "fork_path": "/nonexistent/ray-l7-no-bonsai" },
        "review": {},
    }))
    .unwrap()
}

fn gguf(env: &TestEnv, name: &str) -> PathBuf {
    let path = env.root.join("models").join(name);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    if !path.exists() {
        fs::write(&path, b"GGUF-ray-l7-fixture\n").unwrap();
    }
    path
}

fn fake_entry(id: &str, groups: &[&str]) -> Value {
    json!({
        "id": id,
        "engine": "fake",
        "files": [],
        "ctx": 4096,
        "quant": Value::Null,
        "family": Value::Null,
        "tools": false,
        "vram_estimate_gib": 0.001,
        "launch_spec": {
            "program": "fake",
            "args": [],
            "env": {},
            "port_policy": "managed",
            "served_model_alias": id,
        },
        "aliases": [],
        "groups": groups,
        "name": Value::Null,
        "provenance": {"source": "cli", "imported_at": Value::Null, "spec_hash": ""},
        "last_stream_completed": 0,
        "ewma_load_ms": Value::Null,
        "fingerprints": [],
    })
}

async fn add_entry(d: &Daemon, entry: Value) {
    let id = entry["id"].as_str().unwrap().to_string();
    let (status, text) = d.post_json("/api/engines/catalog/add", entry).await;
    assert_eq!(status, 200, "catalog add `{id}` -> {status}: {text}");
}

async fn engines(d: &Daemon) -> Value {
    d.get_json("/api/engines").await
}

async fn wait_model_state(d: &Daemon, id: &str, state: &str) {
    poll_until(&format!("model {id} state={state}"), 30, || async {
        let e = engines(d).await;
        let hit = e["models"]
            .as_array()
            .map(|rows| {
                rows.iter()
                    .any(|m| m["id"] == json!(id) && m["state"] == json!(state))
            })
            .unwrap_or(false);
        if hit {
            Some(())
        } else {
            None
        }
    })
    .await;
}

async fn ensure(d: &Daemon, id: &str) -> (u16, String) {
    d.post_json("/api/engines/ensure", json!({"model": id}))
        .await
}

// ── Flip gate ───────────────────────────────────────────────────────────────

/// L7 §boot gate: "native mode refuses to boot while the shim still reports
/// loaded models — the operator flushes first, so no model is double-resident."
#[tokio::test]
async fn l7_flip_gate_refuses_boot_with_loaded_shim() {
    let _guard = gate().await;
    let env = TestEnv::new("gate-held");
    let mock = ShimMock::with_reply(MockReply {
        status: 200,
        body: json!({"running": [{"model": "ray-l7-held-model", "proxy": "http://127.0.0.1:1"}]})
            .to_string(),
        delay_ms: 0,
    });
    fs::write(env.config_path(), native_config(&mock.url)).unwrap();
    let (code, log) = Daemon::boot_expect_exit(&env, Instant::now() + Duration::from_secs(30));
    assert_ne!(
        code.unwrap_or(-1),
        0,
        "native boot must exit non-zero; log:\n{log}"
    );
    assert!(log.contains("refuses to boot"), "{log}");
    assert!(
        log.contains("ray-l7-held-model"),
        "the held model must be named; log:\n{log}"
    );
    assert!(
        log.contains("Flush"),
        "the refusal must carry the operator instruction; log:\n{log}"
    );
    // The gate dialed `/running` — and the boot never bound its port.
    assert!(mock.hit_count() >= 1);
}

/// L7 §boot gate tri-state (recovery review I1): only a CONNECTION-level
/// failure means "nothing resident". Timeout, non-2xx, and an unparseable
/// body are UNKNOWN state and must refuse boot with the probe error quoted.
#[tokio::test]
async fn l7_flip_gate_unknown_state_refuses() {
    let _guard = gate().await;
    // (a) non-2xx.
    let env = TestEnv::new("gate-500");
    let mock = ShimMock::with_reply(MockReply {
        status: 500,
        body: "boom".into(),
        delay_ms: 0,
    });
    fs::write(env.config_path(), native_config(&mock.url)).unwrap();
    let (_, log) = Daemon::boot_expect_exit(&env, Instant::now() + Duration::from_secs(30));
    assert!(
        log.contains("state is unknown"),
        "HTTP 500 must refuse as UNKNOWN; log:\n{log}"
    );
    assert!(
        log.contains("HTTP 500"),
        "the refusal must quote the status; log:\n{log}"
    );

    // (b) 200 with a body that is not the documented envelope.
    let env = TestEnv::new("gate-garbage");
    let mock = ShimMock::with_reply(MockReply {
        status: 200,
        body: "this is not json".into(),
        delay_ms: 0,
    });
    fs::write(env.config_path(), native_config(&mock.url)).unwrap();
    let (_, log) = Daemon::boot_expect_exit(&env, Instant::now() + Duration::from_secs(30));
    assert!(
        log.contains("state is unknown"),
        "an unparseable /running must refuse as UNKNOWN; log:\n{log}"
    );

    // (c) timeout: the gate client times out at 5 s — answer at 7 s.
    let env = TestEnv::new("gate-slow");
    let mock = ShimMock::with_reply(MockReply {
        status: 200,
        body: json!({"running": []}).to_string(),
        delay_ms: 7_000,
    });
    fs::write(env.config_path(), native_config(&mock.url)).unwrap();
    let (_, log) = Daemon::boot_expect_exit(&env, Instant::now() + Duration::from_secs(60));
    assert!(
        log.contains("state is unknown"),
        "a timed-out /running must refuse as UNKNOWN (an empty answer that \
         arrived late would have meant boot); log:\n{log}"
    );
}

/// L7 §boot gate: empty shim + unreachable shim both boot — "an unreachable
/// shim holds nothing; boot proceeds."
#[tokio::test]
async fn l7_flip_gate_boots_on_empty_and_unreachable_shim() {
    let _guard = gate().await;
    // (a) reachable shim reporting nothing resident.
    let env = TestEnv::new("gate-empty");
    let mock = ShimMock::default();
    fs::write(env.config_path(), native_config(&mock.url)).unwrap();
    let mut d = Daemon::start(&env, &fs::read_to_string(env.config_path()).unwrap());
    d.ready().await;
    let health = d.get_json("/api/service-health").await;
    assert_eq!(health["swap_mode"], json!("native"), "{}", health);
    drop(d);

    // (b) no listener at all: connection-level failure = nothing resident.
    let env = TestEnv::new("gate-dead");
    let dead = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().to_string()
    };
    let mut d = Daemon::start(&env, &native_config(&format!("http://{dead}")));
    d.ready().await;
    let engines = engines(&d).await;
    assert_eq!(engines["swap"]["mode"], json!("native"), "{engines}");
}

// ── p7 migration ────────────────────────────────────────────────────────────

/// L7 §migration: the rename "moves every kept key verbatim" — and (recovery
/// review I6) it is TEXTUAL: operator comments, quoting, and untouched lines
/// survive byte-for-byte; `base_url` (default endpoint) is dropped; snapshot +
/// `migration restore v2.0-p7` put the old spelling back.
#[tokio::test]
async fn l7_p7_textual_rename_snapshot_and_restore() {
    let _guard = gate().await;
    let env = TestEnv::new("p7");
    // A file only p7 acts on: no `manifest:` (p6), no `bonsai:`/routing keys
    // (p2/p3) — those phases re-serialize by design, and their churn would
    // mask the promise under review: THE P7 RENAME IS TEXTUAL.
    // `decision_model:` is pre-seeded so p3 (which bootstrap-rewrites any
    // config lacking it) is a no-op; with only p7 firing, snapshot == bytes.
    let original = "\
# ray-l7 operator comment — must survive
llama_swap:
  # nested operator comment — must survive
  base_url: \"http://localhost:8081/v1\"
  fallback_model: ray-l7-v1-model
  local_models:
    - ray-l7-v1-model
decision_model:
  enabled: false
  kind: heuristic
";
    let mut daemon = Daemon::start(&env, original);
    daemon.ready().await;
    daemon.stop();
    let after = fs::read_to_string(env.config_path()).unwrap();
    assert!(
        after.contains("swap:"),
        "section must be renamed; file:\n{after}"
    );
    assert!(
        !after.contains("llama_swap:"),
        "old spelling must be gone; file:\n{after}"
    );
    assert!(
        after.contains("# ray-l7 operator comment — must survive"),
        "top comment wiped; file:\n{after}"
    );
    assert!(
        after.contains("# nested operator comment — must survive"),
        "nested comment wiped; file:\n{after}"
    );
    assert!(
        !after.contains("base_url: \"http://localhost:8081/v1\""),
        "default base_url must be dropped; file:\n{after}"
    );
    assert!(
        after.contains("fallback_model: ray-l7-v1-model"),
        "kept key must move verbatim; file:\n{after}"
    );
    assert!(
        after.contains("  - ray-l7-v1-model"),
        "list value must move verbatim; file:\n{after}"
    );

    // Snapshot holds the pre-rename bytes (with the OLD spelling).
    let snapshot = env.state_dir().join("migrations/v2.0-p7/config.yaml");
    let snapshot_text = fs::read_to_string(&snapshot)
        .unwrap_or_else(|e| panic!("snapshot {}: {e}", snapshot.display()));
    assert_eq!(
        snapshot_text, original,
        "only p7 fires — snapshot must be the pristine pre-migration bytes"
    );

    // Restore puts the old section back and removes the new one. The
    // standalone verb takes explicit --config/--state (defaults are the
    // user's real XDG tree — never aim there from a fixture).
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_brainrouter"));
    cmd.args(["migration", "--config"])
        .arg(env.config_path())
        .arg("--state")
        .arg(env.state_dir())
        .args(["restore", "v2.0-p7"])
        .current_dir(&env.root)
        .stdin(Stdio::null());
    env.apply(&mut cmd, &[]);
    let out = cmd.output().unwrap();
    assert_eq!(
        out.status.code(),
        Some(0),
        "restore failed: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let restored = fs::read_to_string(env.config_path()).unwrap();
    assert!(
        restored.contains("llama_swap:"),
        "restore must bring the old section back; file:\n{restored}"
    );
    assert!(
        !restored.contains("\nswap:"),
        "restored file must not keep the new section; file:\n{restored}"
    );
    assert_eq!(
        restored, snapshot_text,
        "restore is byte-exact against the snapshot"
    );
}

// ── Group gate ──────────────────────────────────────────────────────────────

/// L7 §groups: group semantics are deferred to the owner decision, so the
/// supervisor REFUSES native spawns of group members rather than guessing at
/// slot sharing — while ordinary models keep working (`blocked:groups-pending`).
#[tokio::test]
async fn l7_groups_gate_native() {
    let _guard = gate().await;
    let env = TestEnv::new("groups");
    let mut d = Daemon::start(&env, &native_config("http://127.0.0.1:1"));
    d.ready().await;
    for id in ["ray-l7-ga", "ray-l7-gb"] {
        let mut entry = fake_entry(id, &[]);
        entry["groups"] = json!(["ray-l7-pair"]);
        add_entry(&d, entry).await;
    }
    add_entry(&d, fake_entry("ray-l7-solo", &[])).await;

    let (status, text) = ensure(&d, "ray-l7-ga").await;
    assert_eq!(
        status, 503,
        "group-member spawn must be a transient refusal: {text}"
    );
    assert!(text.contains("blocked:groups-pending"), "{text}");
    assert!(
        text.contains("ray-l7-pair"),
        "the refusal must name the group: {text}"
    );

    // A group NAME is not a routable model key.
    let (status, text) = ensure(&d, "ray-l7-pair").await;
    assert_ne!(
        status, 200,
        "the group name must not spawn anything: {text}"
    );
    assert!(text.contains("not in the swap catalog"), "{text}");

    // Control: a non-member ensures and reaches ready.
    let (status, text) = ensure(&d, "ray-l7-solo").await;
    assert_eq!(status, 200, "ensure solo -> {status}: {text}");
    wait_model_state(&d, "ray-l7-solo", "ready").await;
    let engines = engines(&d).await;
    for id in ["ray-l7-ga", "ray-l7-gb"] {
        let row = engines["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == json!(id))
            .unwrap_or_else(|| panic!("catalog rows: {}", engines["models"]));
        assert_eq!(
            row["state"],
            json!("stopped"),
            "group member was spawned: {row}"
        );
        assert!(
            row["endpoint"].is_null(),
            "group member got an endpoint: {row}"
        );
    }
}

// ── Mode pinning ────────────────────────────────────────────────────────────

/// L7 §"native never dials :8081 after the boot gate": every local-serving
/// consumer reads the supervisor, and the shim receives zero traffic.
#[tokio::test]
async fn l7_native_never_dials_shim_after_boot() {
    let _guard = gate().await;
    let env = TestEnv::new("nopin");
    let mock = ShimMock::default();
    fs::write(env.config_path(), native_config(&mock.url)).unwrap();
    let mut d = Daemon::start(&env, &fs::read_to_string(env.config_path()).unwrap());
    d.ready().await;
    add_entry(&d, fake_entry("ray-l7-solo", &[])).await;
    // Drain the boot gate's dial(s); only later traffic matters.
    d.get_json("/api/engines").await;
    mock.take_hits();

    let models = d.get_json("/v1/models").await;
    let ids: Vec<&str> = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["id"].as_str())
        .collect();
    assert!(
        ids.contains(&"ray-l7-solo"),
        "catalog key missing from /v1/models: {models}"
    );
    let solo = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == json!("ray-l7-solo"))
        .unwrap();
    assert_eq!(solo["owned_by"], json!("swap"), "{solo}");

    let llm_models = d.get_json("/api/models/llama-swap").await;
    assert!(
        llm_models["data"]
            .as_array()
            .map(|rows| rows
                .iter()
                .any(|r| r["id"] == json!("ray-l7-solo") && r["owned_by"] == json!("swap")))
            .unwrap_or(false),
        "catalog key missing from /api/models/llama-swap: {llm_models}"
    );

    let status = d.get_json("/api/inference-status").await;
    assert_eq!(status["state"], json!("idle"), "{status}");

    let health = d.get_json("/api/service-health").await;
    assert_eq!(health["swap_mode"], json!("native"), "{health}");

    let hits = mock.take_hits();
    assert!(
        hits.is_empty(),
        "native mode dialed the shim post-boot: {hits:?}"
    );
}

// ── Completeness reporting ──────────────────────────────────────────────────

/// Recovery review I5: flush and restart answer with EVERY outcome — what
/// stopped, what was already gone, what failed — never a bare "ok".
#[tokio::test]
async fn l7_flush_and_restart_report_completeness() {
    let _guard = gate().await;
    let env = TestEnv::new("flush");
    let mut d = Daemon::start(&env, &native_config("http://127.0.0.1:1"));
    d.ready().await;
    add_entry(&d, fake_entry("ray-l7-flush-a", &[])).await;
    add_entry(&d, fake_entry("ray-l7-flush-b", &[])).await;
    for id in ["ray-l7-flush-a", "ray-l7-flush-b"] {
        let (status, text) = ensure(&d, id).await;
        assert_eq!(status, 200, "ensure {id} -> {status}: {text}");
        wait_model_state(&d, id, "ready").await;
    }

    let (status, text) = d.post_json("/api/models/flush", json!({})).await;
    assert_eq!(status, 200, "flush -> {status}: {text}");
    let body: Value = serde_json::from_str(&text).unwrap();
    let mut unloaded = body["unloaded"].as_array().unwrap().clone();
    unloaded.sort_by(|a, b| a.as_str().cmp(&b.as_str()));
    assert_eq!(
        unloaded,
        vec![json!("ray-l7-flush-a"), json!("ray-l7-flush-b")],
        "{body}"
    );
    assert_eq!(body["already_stopped"], json!([]), "fresh flush: {body}");
    assert_eq!(body["reload_failed"], json!([]), "{body}");
    for id in ["ray-l7-flush-a", "ray-l7-flush-b"] {
        wait_model_state(&d, id, "stopped").await;
    }

    // Second flush: nothing resident — honest empty arrays, still 200/ok.
    let (status, text) = d.post_json("/api/models/flush", json!({})).await;
    assert_eq!(status, 200, "second flush -> {status}: {text}");
    let body: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(body["status"], json!("ok"), "{body}");
    assert_eq!(body["unloaded"], json!([]), "{body}");
    assert!(
        body.get("already_stopped").is_some(),
        "completeness arrays must always be present: {body}"
    );

    // Restart: resident engine is reported under `restarted`, status ok.
    let (status, text) = ensure(&d, "ray-l7-flush-a").await;
    assert_eq!(status, 200, "ensure -> {status}: {text}");
    wait_model_state(&d, "ray-l7-flush-a", "ready").await;
    let (status, text) = d.post_json("/api/restart/llama-swap", json!({})).await;
    assert_eq!(status, 200, "restart -> {status}: {text}");
    let body: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(body["service"], json!("swap"), "{body}");
    assert_eq!(body["status"], json!("ok"), "{body}");
    assert_eq!(body["restarted"], json!(["ray-l7-flush-a"]), "{body}");
    assert!(body.get("skipped").is_some(), "{body}");
    wait_model_state(&d, "ray-l7-flush-a", "stopped").await;
}
