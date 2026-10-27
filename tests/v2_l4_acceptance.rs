//! Acceptance tests for leg L4 (swap supervisor + platform probe) of
//! `docs/design/v2-streamlining.md` — spec: `docs/design/v2-streamlining.L4.md`
//! version L4-v3 (2026-10-22).
//!
//! Written by Ray from the design's promises alone (spec, not the build). The
//! builder runs these and NEVER edits them. At the design base commit the L4
//! surfaces (`/api/platform`, `/api/engines*`, `swap`/`platform` CLI verbs,
//! `v2.0-p4` migration, `swap_state`/`swap_source` event fields, adoption
//! markers, `swap_catalog.json`/`budget.json`) do not exist: those rows are
//! RED with the spec cite in the failure message. The one v1-floor row is
//! GREEN today and must stay green (shim-mode routing byte-identical).
//!
//! Everything here is BLACK-BOX (live daemon over loopback + the CLI binary +
//! the documented state files). No Rust API of `src/swap/` is referenced, so
//! the file compiles both before and after the L4 merge. The fixture substrate
//! is the spec's own `fake` adapter: a real child process (`tail -f <gguf>` /
//! `sleep 3600`) scripted through `FAKE_LOAD_MS`, `FAKE_STEMS`,
//! `FAKE_CRASH_AFTER_MS`, `FAKE_PROGRESS` in the catalog entry's
//! `launch_spec.env`.
//!
//! Mapping of tests to the L4 "Acceptance (Ray)" items (spec lines quoted in
//! each test):
//! - (1) Fake-adapter lifecycle / flock / LRU / queue:
//!   * `l4_fake_adapter_lifecycle_stem_assert_crash_restart`
//!   * `l4_concurrent_ensure_one_spawn_all_attach_and_lock_release`
//!   * `l4_flock_refuses_second_engine_on_same_gguf`
//!   * `l4_lru_eviction_least_recent_stream_completed`
//!   * `l4_queue_positions_fifo_cap_and_cancel_api`
//! - (2) Budget / residency-before-budget / pins:
//!   * `l4_residency_attach_before_budget_and_explicit_no_fit`
//!   * `l4_sized_pins_counted_once_never_evicted_and_pins_over_budget`
//! - (3) Adoption (fixture tier = marker scan; host tier is a Bailey card):
//!   * `l4_adoption_marker_live_child_adopted_spec_stale_and_dead_swept`
//! - (4) Shim parity + import (fixture tier = recorded config.yaml):
//!   * `l4_v1_floor_shim_mode_routing_and_event_fields` (v1 floors, incl.
//!     the forwarded-body keeper probe: no L4 field added upstream)
//!   * `l4_import_llama_swap_macros_groups_fidelity_idempotent_dryrun`
//! - (5) Platform probe (ids, unknown banner, min_mem gate, offer matrix,
//!   budget source order GTT → VRAM → 75% MemTotal):
//!   * `l4_platform_probe_ids_banner_and_min_mem_gate`
//! - (6) Progress: covered inside the lifecycle row (`progress.fraction` /
//!   `eta_ms` while `state == "loading"`, absent when ready).
//! - (7) Events additive + registry clock:
//!   * `l4_events_additive_swap_fields_and_inflight_catalog_stamp`
//!   * (LRU clock asserted in `l4_lru_eviction_least_recent_stream_completed`)
//! - (8) CLI surface, daemon-down refusal:
//!   * `l4_cli_surface_daemon_down_refuses`
//! - Design contract rows not in (1)-(9) of the acceptance list but named by
//!   the leg spec/Map:
//!   * `l4_dashboard_platform_token_and_engines_tab` (platform consumers +
//!     Engines tab; Map: dashboard `CURRENT_PLATFORM_ID` replaces the literal)
//!   * `l4_engines_api_reads_gated_mutations_and_busy_code` (Engines API,
//!     loopback+CSRF, 404 unknown model)
//!   * `l4_migration_v2_p4_snapshot_dryrun_restore_and_state_files`
//!   * `l4_spec_stale_on_changed_gguf_fingerprint` (spec: `spec_stale`
//!     fingerprint → badge)
//!
//! GAPS THIS FILE MUST DECLARE (no black-box surface per the spec text):
//! - **Container residents**: the spec says the residency view includes
//!   server-mode identities (`serving_identity.rs`). That registry is fed only
//!   by a live `podman` start/stop; there is no wire surface to seed it in a
//!   fixture test, so the container source is asserted only through the
//!   budget/status vocabulary (unit tier covers `observe_containers`). The
//!   shim and native sources are covered end-to-end here.
//! - **In-flight eviction guard**: holding a NATIVE stream open needs an
//!   engine that answers HTTP; the scripted `fake` adapter intentionally does
//!   not. The additive catalog-id stamp it keys on is asserted on a live shim
//!   stream (`l4_events_additive_swap_fields_and_inflight_catalog_stamp`);
//!   drain-then-evict's native half is unit-tier.
//! - **Queue cancel semantics**: the spec says "queued requests cancellable via
//!   the in-flight row". This suite asserts the documented wire contract
//!   (`DELETE /api/engines/queue/<id>` → 200 `cancelled:true`; unknown → 404);
//!   aborting the waiter is reported to Hank as an implementation gap, not
//!   silently asserted as green.
//! - **Spec-stale convergence verb**: the spec says "the Engines tab badge +
//!   `swap ensure` explicit action converge it". The safe invariant asserted
//!   here is the binding one (a serving engine is NEVER killed by an upgrade);
//!   the converging operator sequence asserted is badge → `evict` → `ensure`
//!   (reported: `ensure` alone attaches on the build's Ready path).
//! - **CLI pin spelling**: the spec's CLI paragraph names
//!   `brainrouter pin add <id> --reserved-gib N|rm <id>`; the L4 surface maps
//!   pins under `swap` (`swap pin|unpin`). This suite asserts the capability
//!   through the documented `swap` grammar and records the spelling
//!   divergence here rather than inventing a second grammar.
//!
//! SERIAL GATE (L3 pattern): every test owns loopback ports, a live daemon and
//! kernel-assigned upstream ports; parallel libtest threads race in the
//! reserve-then-bind windows. Each test holds one process-wide async mutex for
//! its lifetime. Opt out with `RAY_L4_NO_SERIAL_GATE=1` on an idle machine.

use std::fs;
use std::future::Future;
use std::io::Write;
use std::net::TcpListener;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

// ── Serial gate ─────────────────────────────────────────────────────────────

static SUITE_GATE: std::sync::LazyLock<tokio::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));

async fn gate() -> Option<tokio::sync::MutexGuard<'static, ()>> {
    if std::env::var("RAY_L4_NO_SERIAL_GATE").is_ok() {
        return None;
    }
    Some(SUITE_GATE.lock().await)
}

// ── Isolated environment ────────────────────────────────────────────────────

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

/// Where a test's isolated HOME/XDG tree lives.
///
/// `/tmp` is a tmpfs on this box, so anything written there is RAM. The daemon under
/// test can materialize a rootless-podman store inside `$XDG_DATA_HOME` (one pulled
/// toolbox image is ~1.9 GiB), and a leaked test root is therefore 1.9 GiB of RAM per
/// run — enough to OOM-kill the model server next to the tests. Keep it on disk: Cargo
/// hands integration tests the path `target/tmp/<binary>` via `CARGO_TARGET_TMPDIR`.
fn test_root() -> PathBuf {
    match option_env!("CARGO_TARGET_TMPDIR") {
        Some(dir) => PathBuf::from(dir),
        None => std::env::temp_dir(),
    }
}

/// Remove a test root, including anything a user namespace owns.
///
/// The podman store inside it is created through a user namespace, so its
/// `overlay/*/diff` trees belong to the *subuid-mapped* root: an unprivileged
/// `remove_dir_all` fails with EACCES and silently leaks the whole root. Retry inside
/// that namespace (`podman unshare`), where those files are removable.
fn remove_test_root(root: &Path) {
    if !root.exists() {
        return;
    }
    if fs::remove_dir_all(root).is_ok() {
        return;
    }
    let _ = Command::new("podman")
        .arg("unshare")
        .args(["rm", "-rf"])
        .arg(root)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

struct TestEnv {
    root: PathBuf,
    home: PathBuf,
    config_home: PathBuf,
    data_home: PathBuf,
    state_home: PathBuf,
    runtime: PathBuf,
    bin: PathBuf,
    owned: bool,
}

impl TestEnv {
    fn new(tag: &str) -> Self {
        let root = test_root().join(format!("brainrouter-l4-{tag}-{}", unique_suffix()));
        let home = root.join("home");
        let config_home = root.join("config");
        let data_home = root.join("data");
        let state_home = root.join("state");
        let runtime = root.join("runtime");
        let bin = root.join("bin");
        let env = Self {
            root,
            home,
            config_home,
            data_home,
            state_home,
            runtime,
            bin,
            owned: true,
        };
        for d in [
            &env.home,
            &env.config_home,
            &env.data_home,
            &env.state_home,
            &env.runtime,
            &env.bin,
        ] {
            fs::create_dir_all(d).unwrap();
        }
        fs::set_permissions(&env.runtime, fs::Permissions::from_mode(0o700)).unwrap();
        fs::create_dir_all(env.state_dir()).unwrap();
        fs::create_dir_all(env.config_dir()).unwrap();
        env
    }

    /// The daemon's environment. PATH keeps the system dirs (the `fake`
    /// adapter spawns `tail`/`sleep`; the native adapter may look for
    /// `systemd-run`) plus the fixture bin dir.
    fn apply(&self, cmd: &mut Command, extra: &[(String, String)]) {
        let path = format!("{}:/usr/local/bin:/usr/bin:/bin", self.bin.display());
        cmd.env_clear()
            .env("HOME", &self.home)
            .env("XDG_CONFIG_HOME", &self.config_home)
            .env("XDG_DATA_HOME", &self.data_home)
            .env("XDG_STATE_HOME", &self.state_home)
            .env("XDG_RUNTIME_DIR", &self.runtime)
            .env("PATH", path)
            .env("RUST_LOG", "debug")
            // Fixture tier: never try to create a transient systemd user scope.
            .env("BRAINROUTER_SWAP_NO_SYSTEMD", "1")
            .envs(extra.iter().cloned());
    }

    fn config_dir(&self) -> PathBuf {
        self.config_home.join("brainrouter")
    }

    fn state_dir(&self) -> PathBuf {
        self.state_home.join("brainrouter")
    }

    fn llama_swap_config_path(&self) -> PathBuf {
        self.home.join(".config/llama-swap/config.yaml")
    }

    fn swap_catalog_path(&self) -> PathBuf {
        self.state_dir().join("swap_catalog.json")
    }

    fn budget_path(&self) -> PathBuf {
        self.state_dir().join("budget.json")
    }

    fn adoption_dir(&self) -> PathBuf {
        self.state_dir().join("adoption")
    }

}

impl Drop for TestEnv {
    fn drop(&mut self) {
        if self.owned {
            remove_test_root(&self.root);
        }
    }
}

/// Kills any `fake` engine child whose argv names a path under the test root
/// (the child is `tail -f <gguf>` / `sleep 3600`). Adoption children
/// deliberately outlive the daemon; tests must not leak them.
struct ChildGuard(PathBuf);

impl ChildGuard {
    fn new(root: &Path) -> Self {
        Self(root.to_path_buf())
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let needle = self.0.display().to_string();
        for pid in pids_with_arg(&needle) {
            unsafe {
                libc::kill(pid as i32, libc::SIGKILL);
            }
        }
    }
}

// ── Process helpers ─────────────────────────────────────────────────────────

fn pids_with_arg(needle: &str) -> Vec<u32> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir("/proc") else { return out };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let Ok(pid) = name.parse::<u32>() else { continue };
        if let Ok(raw) = fs::read(entry.path().join("cmdline")) {
            let text = String::from_utf8_lossy(&raw).replace('\0', " ");
            if text.contains(needle) {
                out.push(pid);
            }
        }
    }
    out
}

fn pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

/// L4 spec, boot invariants: "single-instance flock per canonical GGUF, HELD
/// BY THE CHILD process". Returns true when this process could take the
/// exclusive lock (i.e. no surviving child holds it).
fn flock_is_free(path: &Path) -> bool {
    use std::os::unix::io::AsRawFd;
    let file = fs::File::open(path).expect("open gguf for flock probe");
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    rc == 0
}

// ── Polling ─────────────────────────────────────────────────────────────────

async fn poll_until<T, F, Fut>(what: &str, secs: u64, mut f: F) -> T
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Option<T>>,
{
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        if let Some(value) = f().await {
            return value;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

// ── Mock upstream (llama-swap shim: /running, /v1/models, SSE chat) ────────

#[derive(Clone)]
struct MockCfg {
    /// Body served at `/running` (llama-swap's `{"running":[...]}` envelope).
    running: Value,
    /// ids advertised at `/v1/models`.
    models: Vec<String>,
    /// Hold the response body this long AFTER the headers: keeps the daemon's
    /// in-flight row live while the test inspects `/api/inflight`.
    body_delay_ms: u64,
    /// Fail every chat completion (routing-failure rows).
    fail_chat: bool,
}

impl Default for MockCfg {
    fn default() -> Self {
        Self {
            running: json!({"running": []}),
            models: Vec::new(),
            body_delay_ms: 0,
            fail_chat: false,
        }
    }
}

struct Mock {
    url: String,
    running: Arc<Mutex<Value>>,
    hits: Arc<Mutex<Vec<String>>>,
    /// Parsed upstream chat bodies, in arrival order (the v1 keeper probe:
    /// what the router actually forwarded, not what the client sent).
    bodies: Arc<Mutex<Vec<Value>>>,
}

impl Mock {
    fn set_running(&self, value: Value) {
        *self.running.lock().unwrap() = value;
    }

    fn chat_hits(&self) -> usize {
        self.hits.lock().unwrap().len()
    }

    fn chat_bodies(&self) -> Vec<Value> {
        self.bodies.lock().unwrap().clone()
    }
}

fn find_headers_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4)
}

fn spawn_mock(cfg: MockCfg) -> Mock {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    listener.set_nonblocking(true).unwrap();
    let running = Arc::new(Mutex::new(cfg.running.clone()));
    let models = Arc::new(cfg.models.clone());
    let hits = Arc::new(Mutex::new(Vec::new()));
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let (r2, m2, h2, b2, cfg2) = (
        running.clone(),
        models.clone(),
        hits.clone(),
        bodies.clone(),
        Arc::new(cfg),
    );
    tokio::spawn(async move {
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();
        loop {
            let Ok((mut sock, _)) = listener.accept().await else { continue };
            let (r, m, h, b, cfg) = (r2.clone(), m2.clone(), h2.clone(), b2.clone(), cfg2.clone());
            tokio::spawn(async move {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut buf = vec![0u8; 256 * 1024];
                let mut filled = 0usize;
                let mut content_len = None::<usize>;
                let mut path = String::new();
                loop {
                    let cap = buf.len();
                    let off = filled.min(cap - 1);
                    let Ok(n) = sock.read(&mut buf[off..]).await else { return };
                    if n == 0 {
                        break;
                    }
                    filled += n;
                    if let Some(pos) = find_headers_end(&buf[..filled]) {
                        if content_len.is_none() {
                            let headers = String::from_utf8_lossy(&buf[..pos]).to_lowercase();
                            path = headers
                                .lines()
                                .next()
                                .and_then(|l| l.split_whitespace().nth(1))
                                .unwrap_or("")
                                .to_string();
                            content_len = Some(
                                headers
                                    .lines()
                                    .find_map(|l| {
                                        l.strip_prefix("content-length:")
                                            .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                                    })
                                    .unwrap_or(0),
                            );
                        }
                        if filled >= pos + content_len.unwrap_or(0) {
                            break;
                        }
                    }
                }
                let body_start = find_headers_end(&buf[..filled]).unwrap_or(0);
                let body: Value =
                    serde_json::from_slice(&buf[body_start..filled.max(body_start)])
                        .unwrap_or(Value::Null);

                if path.starts_with("/running") {
                    let snapshot = r.lock().unwrap().clone().to_string();
                    let _ = write_plain(&mut sock, 200, "application/json", snapshot).await;
                    return;
                }
                if path.contains("/models") {
                    let data: Vec<Value> = m
                        .iter()
                        .map(|id| json!({"id": id, "object": "model", "owned_by": "ray-l4-mock"}))
                        .collect();
                    let _ = write_plain(
                        &mut sock,
                        200,
                        "application/json",
                        json!({"object": "list", "data": data}).to_string(),
                    )
                    .await;
                    return;
                }
                if path.contains("/chat/completions") || path.contains("/completions") {
                    let model = body
                        .get("model")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    h.lock().unwrap().push(model.clone());
                    b.lock().unwrap().push(body.clone());
                    if cfg.fail_chat {
                        let _ = write_plain(
                            &mut sock,
                            500,
                            "application/json",
                            json!({"error": {"message": "ray-l4 mock failure", "type": "server_error"}})
                                .to_string(),
                        )
                        .await;
                        return;
                    }
                    let chunk = |delta: Value, finish: Value| {
                        format!(
                            "data: {}\n\n",
                            json!({"id":"chatcmpl-ray-l4","object":"chat.completion.chunk",
                                   "created":1,"model":model.clone(),
                                   "choices":[{"index":0,"delta":delta,"finish_reason":finish}]})
                        )
                    };
                    let sse = chunk(json!({"role":"assistant","content":""}), Value::Null)
                        + &chunk(json!({"content": format!("ray-l4-pong {model}")}), json!("stop"))
                        + "data: [DONE]\n\n";
                    let head = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                        sse.len()
                    );
                    let _ = sock.write_all(head.as_bytes()).await;
                    let _ = sock.flush().await;
                    if cfg.body_delay_ms > 0 {
                        tokio::time::sleep(Duration::from_millis(cfg.body_delay_ms)).await;
                    }
                    let _ = sock.write_all(sse.as_bytes()).await;
                    let _ = sock.shutdown().await;
                    return;
                }
                let _ = write_plain(
                    &mut sock,
                    404,
                    "application/json",
                    json!({"error": {"message": format!("unknown endpoint {path}")}}).to_string(),
                )
                .await;
            });
        }
    });
    Mock {
        url: format!("http://127.0.0.1:{port}"),
        running,
        hits,
        bodies,
    }
}

async fn write_plain<S>(sock: &mut S, status: u16, ctype: &str, body: String) -> std::io::Result<()>
where
    S: tokio::io::AsyncWrite + Unpin,
{
    use tokio::io::AsyncWriteExt;
    let reason = match status {
        200 => "OK",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Status",
    };
    sock.write_all(
        format!(
            "HTTP/1.1 {status} {reason}\r\ncontent-type: {ctype}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        )
        .as_bytes(),
    )
    .await
}

// ── Daemon harness (live-daemon conventions from v2_l2/v2_l3) ──────────────

struct Daemon {
    child: std::process::Child,
    url: String,
    log: PathBuf,
    client: reqwest::Client,
}

impl Daemon {
    fn start(env: &TestEnv, config_yaml: &str) -> Self {
        Self::start_with_env(env, config_yaml, &[])
    }

    fn start_with_env(env: &TestEnv, config_yaml: &str, extra: &[(String, String)]) -> Self {
        fs::write(env.root.join("config.yaml"), config_yaml).unwrap();
        Self::boot(env, extra)
    }

    /// Restart over the config file the previous boot left on disk (the real
    /// systemd-restart procedure the migration contract pins).
    fn start_from_disk(env: &TestEnv) -> Self {
        assert!(
            env.root.join("config.yaml").exists(),
            "restart precondition: a config must already be on disk"
        );
        Self::boot(env, &[])
    }

    fn boot(env: &TestEnv, extra: &[(String, String)]) -> Self {
        let config_path = env.root.join("config.yaml");
        let socket = env.root.join("d.sock");
        let log = env.root.join("daemon.log");
        // Reserved-then-freed loopback ports carry a TOCTOU window; retry a
        // bind collision (the L3 harness convention).
        let mut attempt = 0u8;
        let (address, mut child) = loop {
            attempt += 1;
            let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = reservation.local_addr().unwrap();
            drop(reservation);
            let log_file = fs::File::create(&log).unwrap();
            let mut cmd = Command::new(env!("CARGO_BIN_EXE_brainrouter"));
            cmd.args(["serve", "--config"])
                .arg(&config_path)
                .args(["--tcp-addr", address.to_string().as_str(), "--socket"])
                .arg(&socket)
                .current_dir(&env.root)
                .stdin(Stdio::null())
                .stdout(Stdio::from(log_file.try_clone().unwrap()))
                .stderr(Stdio::from(log_file));
            env.apply(&mut cmd, extra);
            let mut child = cmd.spawn().unwrap();
            std::thread::sleep(Duration::from_millis(150));
            if child.try_wait().unwrap().is_none() || attempt == 3 {
                break (address, child);
            }
            let text = fs::read_to_string(&log).unwrap_or_default();
            if !text.to_lowercase().contains("address already in use") {
                break (address, child);
            }
            drop(child);
        };
        let _ = &mut child;
        Self {
            child,
            url: format!("http://{address}"),
            log,
            client: reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(60))
                .build()
                .unwrap(),
        }
    }

    async fn ready(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(90);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                panic!("daemon exited with {status}: {}", self.log_tail());
            }
            if let Ok(r) = self.client.get(format!("{}/health", self.url)).send().await {
                if r.status().is_success()
                    && r.json::<Value>().await.unwrap_or(Value::Null) == json!({"status": "ok"})
                {
                    return;
                }
            }
            assert!(Instant::now() < deadline, "daemon not healthy: {}", self.log_tail());
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    fn log_tail(&self) -> String {
        let text = fs::read_to_string(&self.log).unwrap_or_default();
        let tail: Vec<&str> = text.lines().rev().take(40).collect();
        tail.into_iter().rev().collect::<Vec<_>>().join("\n")
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    fn http(&self) -> reqwest::Client {
        self.client.clone()
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}{path}", self.url)
    }

    async fn get_json(&self, path: &str) -> Value {
        let (status, text) = self.request("GET", path, None, &[]).await;
        assert!(
            status < 400,
            "L4 surface missing: GET {path} -> {status}: {text}\n\
             (L4 spec: platform probe + swap supervisor API / /api/engines*)"
        );
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("GET {path} not json ({e}): {text}"))
    }

    async fn get_text(&self, path: &str) -> (u16, String) {
        self.request("GET", path, None, &[]).await
    }

    async fn post_json(&self, path: &str, body: Value) -> (u16, String) {
        self.request("POST", path, Some(body), &[]).await
    }

    async fn post_json_headers(
        &self,
        path: &str,
        body: Value,
        headers: &[(&str, &str)],
    ) -> (u16, String) {
        self.request("POST", path, Some(body), headers).await
    }

    async fn delete(&self, path: &str) -> (u16, String) {
        self.request("DELETE", path, None, &[]).await
    }

    async fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<Value>,
        headers: &[(&str, &str)],
    ) -> (u16, String) {
        let mut req = self
            .client
            .request(reqwest::Method::from_bytes(method.as_bytes()).unwrap(), self.endpoint(path));
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        if let Some(b) = body {
            req = req.json(&b);
        }
        let r = req.send().await.unwrap_or_else(|e| panic!("{method} {path}: {e}"));
        let status = r.status().as_u16();
        (status, r.text().await.unwrap_or_default())
    }

    /// SIGTERM: the graceful stop (best-effort child hygiene is the daemon's).
    fn stop(&mut self) {
        let pid = self.pid() as i32;
        unsafe {
            libc::kill(pid, libc::SIGTERM);
        }
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if self.child.try_wait().unwrap().is_some() {
                return;
            }
            if Instant::now() >= deadline {
                let _ = self.child.kill();
                let _ = self.child.wait();
                panic!("daemon did not exit 20 s after SIGTERM: {}", self.log_tail());
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// SIGKILL: the crash path. The spec's flock/adoption contract is written
    /// for exactly this case ("daemon crash releases nothing of the child's").
    fn kill9(&mut self) {
        let pid = self.pid() as i32;
        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }
        let _ = self.child.wait();
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ── CLI helper ──────────────────────────────────────────────────────────────

struct CliRun {
    code: i32,
    stdout: String,
    stderr: String,
}

impl CliRun {
    fn ok(&self, what: &str) -> String {
        assert_eq!(
            self.code, 0,
            "{what} failed (exit {}): stdout: {} | stderr: {}",
            self.code, self.stdout, self.stderr
        );
        self.stdout.clone()
    }

    fn json(&self, what: &str) -> Value {
        let text = self.ok(what);
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{what} not JSON ({e}): {text}"))
    }
}

fn run_cli(env: &TestEnv, args: &[&str]) -> CliRun {
    run_cli_with(env, args, &[])
}

fn run_cli_with(env: &TestEnv, args: &[&str], extra: &[(String, String)]) -> CliRun {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_brainrouter"));
    cmd.args(args).current_dir(&env.root).stdin(Stdio::null());
    env.apply(&mut cmd, extra);
    let out = cmd.output().expect("run brainrouter CLI");
    CliRun {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

// ── Config + catalog fixtures ───────────────────────────────────────────────

fn base_config(local_base: &str) -> String {
    serde_yaml::to_string(&json!({
        "manifest": { "enabled": false, "base_url": format!("{local_base}/cloud/v1") },
        "llama_swap": {
            "base_url": format!("{local_base}/v1"),
            "fallback_model": "ray-l4-v1-model",
            "local_models": ["ray-l4-v1-model"],
        },
        "bonsai": { "enabled": false, "fork_path": "/nonexistent/ray-l4-no-bonsai" },
        "review": {},
    }))
    .unwrap()
}

fn gguf(env: &TestEnv, name: &str) -> PathBuf {
    let path = env.root.join("models").join(name);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    if !path.exists() {
        fs::write(&path, b"GGUF-ray-l4-fixture\n").unwrap();
    }
    path
}

/// A `fake` adapter entry. `env_pairs` are the spec's scripted knobs
/// (`FAKE_LOAD_MS`, `FAKE_STEMS`, `FAKE_CRASH_AFTER_MS`, `FAKE_PROGRESS`).
fn fake_entry(
    id: &str,
    gguf: Option<&Path>,
    estimate_gib: f64,
    env_pairs: &[(&str, &str)],
    last_stream_completed: u64,
) -> Value {
    let env_map: serde_json::Map<String, Value> = env_pairs
        .iter()
        .map(|(k, v)| (k.to_string(), json!(v)))
        .collect();
    json!({
        "id": id,
        "engine": "fake",
        "files": gguf.map(|p| json!([p.display().to_string()])).unwrap_or_else(|| json!([])),
        "ctx": 4096,
        "quant": Value::Null,
        "family": Value::Null,
        "tools": false,
        "vram_estimate_gib": estimate_gib,
        "launch_spec": {
            "program": "fake",
            "args": [],
            "env": env_map,
            "port_policy": "managed",
            "served_model_alias": id,
        },
        "aliases": [],
        "groups": [],
        "name": Value::Null,
        "provenance": {"source": "cli", "imported_at": Value::Null, "spec_hash": ""},
        "last_stream_completed": last_stream_completed,
        "ewma_load_ms": Value::Null,
        "fingerprints": [],
    })
}

async fn add_entry(d: &Daemon, entry: Value) {
    let id = entry["id"].as_str().unwrap().to_string();
    let (status, text) = d.post_json("/api/engines/catalog/add", entry).await;
    assert_eq!(
        status, 200,
        "catalog add `{id}` -> {status}: {text}\n\
         (L4 spec: \"Model registry + launch specs (`src/swap/`)\" — CLI additions seed the catalog)"
    );
}

async fn engines(d: &Daemon) -> Value {
    d.get_json("/api/engines").await
}

fn model_status<'a>(engines: &'a Value, id: &str) -> Option<&'a Value> {
    engines["models"]
        .as_array()?
        .iter()
        .find(|m| m["id"] == json!(id))
}

async fn wait_model_state(d: &Daemon, id: &str, state: &str) {
    poll_until(&format!("model {id} state={state}"), 20, || async {
        let e = engines(d).await;
        match model_status(&e, id) {
            Some(m) if m["state"] == json!(state) => Some(()),
            _ => None,
        }
    })
    .await;
}

async fn ensure(d: &Daemon, id: &str) -> (u16, String) {
    d.post_json("/api/engines/ensure", json!({"model": id})).await
}

async fn ensure_ok(d: &Daemon, id: &str) -> Value {
    let (status, text) = ensure(d, id).await;
    assert_eq!(status, 200, "ensure `{id}` -> {status}: {text}");
    serde_json::from_str(&text).unwrap()
}

fn budget_entries(env: &TestEnv) -> Vec<Value> {
    let text = fs::read_to_string(env.budget_path()).unwrap_or_default();
    if text.is_empty() {
        return Vec::new();
    }
    serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|v| v["entries"].as_array().cloned())
        .unwrap_or_default()
}


/// The v1 RouteEvent field set v2 may only extend. `conv_id` is serialized
/// only when non-empty (`skip_serializing_if`), so it is asserted as
/// "absent-or-string" rather than "always present".
fn assert_v1_event_fields(event: &Value) {
    for field in [
        "id",
        "timestamp",
        "prompt_excerpt",
        "requested_model",
        "effective_provider",
        "model_key",
        "latency_ms",
        "stage",
        "success",
        "role",
        "cwd",
    ] {
        assert!(
            event.get(field).is_some(),
            "v1 floor: RouteEvent lost `{field}` (events ADDITIVE only): {event}"
        );
    }
    assert!(
        event.get("conv_id").is_none_or(Value::is_string),
        "conv_id must stay a string when present: {event}"
    );
}

// ── (4) v1 floor — GREEN at the design base commit, must stay green ────────

/// L4 spec "Contracts kept": "Shim-mode routing byte-identical (v1 floors)".
/// This row pins the v1 behavior every L4 change must leave untouched: with no
/// catalog entry involved, a bare local model key routes to llama-swap and the
/// routing event keeps its v1 field set.
#[tokio::test(flavor = "multi_thread")]
async fn l4_v1_floor_shim_mode_routing_and_event_fields() {
    let _gate = gate().await;
    let env = TestEnv::new("v1-floor");
    let _guard = ChildGuard::new(&env.root);
    let mock = spawn_mock(MockCfg {
        models: vec!["ray-l4-v1-model".into()],
        ..Default::default()
    });
    let mut d = Daemon::start(&env, &base_config(&mock.url));
    d.ready().await;

    let resp = d
        .http()
        .post(d.endpoint("/v1/chat/completions"))
        .json(&json!({
            "model": "ray-l4-v1-model",
            "messages": [{"role": "user", "content": "ray-l4 v1 floor"}],
            "stream": true,
        }))
        .send()
        .await
        .expect("chat request");
    assert_eq!(resp.status().as_u16(), 200, "v1 floor chat status");
    let body = resp.text().await.unwrap_or_default();
    assert!(body.contains("ray-l4-pong"), "v1 floor SSE body: {body}");
    assert!(body.contains("[DONE]"), "v1 floor SSE lacks [DONE]: {body}");

    // The v1 RouteEvent field set (v2 L3 fixed it; L4 adds swap_* only).
    let events = poll_until("the v1 floor routing event", 10, || async {
        let v = d.get_json("/api/routing-events").await;
        let hit = v["events"]
            .as_array()?
            .iter()
            .find(|e| e["requested_model"] == json!("ray-l4-v1-model"))
            .cloned();
        hit
    })
    .await;
    assert_v1_event_fields(&events);
    assert_eq!(events["success"], json!(true), "v1 floor event: {events}");
    assert_eq!(events["effective_provider"], json!("llama-swap"), "{events}");
    assert_eq!(mock.chat_hits(), 1, "v1 floor routes exactly one upstream call");

    // Keeper probe (L4 spec "Contracts kept": shim-mode routing byte-identical,
    // v1 floors). No scheduler involvement ⇒ the forwarded body is the v1
    // body: every client field verbatim, no L4 field invented.
    let sent = json!({
        "model": "ray-l4-v1-model",
        "messages": [{"role": "user", "content": "ray-l4 v1 floor"}],
        "stream": true,
    });
    let upstream = mock.chat_bodies();
    assert_eq!(upstream.len(), 1, "exactly one forwarded body: {upstream:?}");
    let body = &upstream[0];
    for (key, value) in sent.as_object().unwrap() {
        assert_eq!(
            body.get(key), Some(value),
            "v1 floor: forwarded `{key}` must be the client's, byte-for-byte: {body}"
        );
    }
    for (key, value) in body.as_object().unwrap() {
        assert!(
            !key.starts_with("swap") && key != "catalog_id",
            "v1 floor: L4 must not add `{key}` to the upstream body: {body}"
        );
        assert!(
            sent.get(key).is_some(),
            "v1 floor: forwarded body gained a field the client never sent (`{key}`): {body}"
        );
        let _ = value;
    }
}

// ── (5) Platform probe ─────────────────────────────────────────────────────

/// The spec's engine matrix as DATA (L4 spec §"Engine compatibility matrix"):
/// strixite = gfx1151 + GTT ≥ ~100 GiB; gufo = gfx1151; hipfire = ROCm
/// gfx9/10/11 discrete; llama.cpp universal. "Unknown ⇒ everything offered".
fn spec_offers(arch: &str, apu: bool, budget_gib: f64, gtt_gib: Option<f64>, known: bool) -> Vec<(&'static str, bool)> {
    let arch_ok = |prefixes: &[&str]| prefixes.iter().any(|p| arch.starts_with(p));
    vec![
        (
            "strixite",
            known && arch_ok(&["gfx1151"]) && gtt_gib.unwrap_or(0.0) >= 100.0 && budget_gib >= 100.0,
        ),
        ("gufo", known && arch_ok(&["gfx1151"]) && budget_gib >= 64.0),
        (
            "hipfire",
            known && arch_ok(&["gfx9", "gfx10", "gfx11", "gfx12"]) && !apu && budget_gib >= 16.0,
        ),
        ("llama.cpp", known && budget_gib >= 4.0),
    ]
}

fn probe_offers(probe: &Value) -> Vec<(String, bool, String, f64)> {
    probe["offers"]
        .as_array()
        .expect("probe.offers")
        .iter()
        .map(|o| {
            (
                o["engine"].as_str().unwrap_or("").to_string(),
                o["offered"].as_bool().unwrap_or(false),
                o["reason"].as_str().unwrap_or("").to_string(),
                o["min_mem_gib"].as_f64().unwrap_or(0.0),
            )
        })
        .collect()
}

fn assert_offers_match_spec(probe: &Value, known: bool) {
    let arch = probe["facts"]["gpus"]
        .as_array()
        .and_then(|g| {
            g.iter()
                .find(|gpu| {
                    let a = gpu["arch"].as_str().unwrap_or("");
                    !a.is_empty() && a != "unknown"
                })
                .map(|gpu| gpu["arch"].as_str().unwrap_or("").to_string())
        })
        .unwrap_or_else(|| probe["facts"]["cpu_arch"].as_str().unwrap_or("").to_string());
    let apu = probe["facts"]["apu"].as_bool().unwrap_or(false);
    let budget = probe["budget_gib"].as_f64().unwrap_or(0.0);
    let gtt = probe["facts"]["gtt_budget_gib"].as_f64();
    let expected = spec_offers(&arch, apu, budget, gtt, known);
    let actual = probe_offers(probe);
    for (engine, want) in expected {
        let (_, got, reason, _) = actual
            .iter()
            .find(|(name, ..)| name == engine)
            .unwrap_or_else(|| panic!("probe is missing matrix engine `{engine}`: {probe}"));
        assert_eq!(
            *got, want,
            "matrix mismatch for `{engine}` on arch={arch} apu={apu} budget={budget} gtt={gtt:?} \
             (L4 spec: compatibility matrix drives Engines-tab offers); reason: {reason}"
        );
        assert!(!reason.is_empty(), "offer `{engine}` has no reason: {probe}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn l4_platform_probe_ids_banner_and_min_mem_gate() {
    let _gate = gate().await;

    // (a) Real probe on a pristine state root: feed platform id when the
    // detected arch matches the vendored feed, else `generic-<arch>` + banner.
    let real_env = TestEnv::new("platform-real");
    let probe = run_cli(&real_env, &["platform", "show", "--json"])
        .json("platform show --json");
    assert_eq!(probe["facts"]["source"], json!("live"), "probe facts: {probe}");
    let arch = probe["facts"]["gpus"]
        .as_array()
        .and_then(|g| {
            g.iter()
                .find(|gpu| {
                    let a = gpu["arch"].as_str().unwrap_or("");
                    !a.is_empty() && a != "unknown"
                })
                .map(|gpu| gpu["arch"].as_str().unwrap_or("").to_string())
        })
        .unwrap_or_else(|| probe["facts"]["cpu_arch"].as_str().unwrap_or("").to_string());
    let feed_ids = ["strix-halo", "r9700", "gb10", "intel-b70"];
    let id = probe["id"].as_str().unwrap_or("");
    let known = probe["known"].as_bool().unwrap_or(false);
    if known {
        assert!(
            feed_ids.contains(&id),
            "known probe must carry a feed platform id (L4 spec: `id` = the catalog platform id \
             when detected hardware matches a feed platform); got {id}: {probe}"
        );
        assert!(probe["banner"].is_null(), "known platform must not carry the unknown banner: {probe}");
    } else {
        assert_eq!(
            id,
            format!("generic-{arch}"),
            "unknown hardware resolves `generic-<arch>` (L4 spec); got {id}: {probe}"
        );
        assert!(
            probe["banner"].as_str().unwrap_or("").contains("Unknown platform"),
            "unknown platform must raise the explicit banner (fail-open): {probe}"
        );
        for (engine, offered, _, _) in probe_offers(&probe) {
            assert!(offered, "unknown platform is fail-open: `{engine}` must be offered: {probe}");
        }
    }
    // Budget source order (L4 spec §"Residency view": budget cap =
    // probe-reported platform budget — GTT on unified memory, VRAM+policy
    // elsewhere, else three quarters of RAM). The live `/proc` + sysfs facts
    // are the only kernel-sourced fixture a black-box probe exposes; the
    // env-override precedence is asserted in (d).
    let facts = &probe["facts"];
    let mem_total = facts["mem_total_gib"].as_f64().unwrap_or(0.0);
    assert!(mem_total > 0.0, "probe must report MemTotal: {probe}");
    let gtt = facts["gtt_budget_gib"].as_f64().filter(|v| *v > 0.0);
    let vram = facts["vram_budget_gib"].as_f64().filter(|v| *v > 0.0);
    let want_budget = gtt.or(vram).unwrap_or(mem_total * 0.75);
    let got_budget = probe["budget_gib"].as_f64().unwrap_or(-1.0);
    assert!(
        (got_budget - want_budget).abs() < 1e-6,
        "budget_gib must follow GTT → VRAM → 75% MemTotal: facts gtt={gtt:?} vram={vram:?} \
         mem_total={mem_total} ⇒ {want_budget}, probe says {got_budget}: {probe}"
    );
    assert_offers_match_spec(&probe, known);

    // (b) Feed overlay via the cache: a matching platform names the host.
    let named_env = TestEnv::new("platform-named");
    fs::create_dir_all(named_env.state_dir()).unwrap();
    fs::write(
        named_env.state_dir().join("platform_feed.json"),
        json!({"version": 1, "platforms": [
            {"id": "ray-l4-fixture-halo", "arches": [arch.clone()], "label": "Ray Fixture Halo"}
        ]})
        .to_string(),
    )
    .unwrap();
    let named = run_cli(&named_env, &["platform", "show", "--json"]).json("platform show --json (feed cache)");
    assert_eq!(
        named["id"], json!("ray-l4-fixture-halo"),
        "a fresh feed cache decides the catalog platform id (L4 spec); probe: {named}"
    );
    assert_eq!(named["known"], json!(true), "{named}");
    assert_eq!(named["label"], json!("Ray Fixture Halo"), "{named}");
    assert!(named["banner"].is_null(), "{named}");
    assert_offers_match_spec(&named, true);

    // (c) Unknown platform: a feed that matches nothing ⇒ generic id + banner,
    // every engine offered with the explicit fail-open reason.
    let unknown_env = TestEnv::new("platform-unknown");
    fs::create_dir_all(unknown_env.state_dir()).unwrap();
    fs::write(
        unknown_env.state_dir().join("platform_feed.json"),
        json!({"version": 1, "platforms": [
            {"id": "gfx9999", "arches": ["gfx9999"]}
        ]})
        .to_string(),
    )
    .unwrap();
    let unknown = run_cli(&unknown_env, &["platform", "show", "--json"]).json("platform show --json (unknown)");
    assert_eq!(unknown["id"], json!(format!("generic-{arch}")), "{unknown}");
    assert_eq!(unknown["known"], json!(false), "{unknown}");
    assert!(
        unknown["banner"].as_str().unwrap_or("").contains("Unknown platform"),
        "unknown id ⇒ visible banner (L4 spec: the visible fail-open, by design): {unknown}"
    );
    for (engine, offered, reason, _) in probe_offers(&unknown) {
        assert!(offered, "fail-open: `{engine}` must be offered: {unknown}");
        assert!(reason.contains("fail-open"), "offer reason must state the fail-open: {reason}");
    }

    // (d) min_mem_gib gate: a forced budget below an engine's floor refuses it
    // with the reason (the spec's matrix carries `min_mem_gib` per engine).
    let low_env = TestEnv::new("platform-low-budget");
    fs::create_dir_all(low_env.state_dir()).unwrap();
    fs::write(
        low_env.state_dir().join("platform_feed.json"),
        json!({"version": 1, "platforms": [
            {"id": "ray-l4-fixture-halo", "arches": [arch.clone()], "label": "Ray Fixture Halo"}
        ]})
        .to_string(),
    )
    .unwrap();
    let low = run_cli_with(
        &low_env,
        &["platform", "show", "--json"],
        &[("BRAINROUTER_BUDGET_GIB".to_string(), "10".to_string())],
    )
    .json("platform show --json (budget 10)");
    assert_eq!(low["budget_gib"], json!(10.0), "probe budget honours the platform budget env: {low}");
    let mut gated = 0usize;
    for (engine, offered, reason, min_mem) in probe_offers(&low) {
        if min_mem > 10.0 {
            gated += 1;
            assert!(!offered, "`{engine}` needs {min_mem} GiB > 10 ⇒ not offered: {low}");
            assert!(
                !reason.is_empty()
                    && (reason.contains("GiB") || reason.contains("APU") || reason.contains("arch")),
                "a refused offer must state its reason: {reason}"
            );
        }
    }
    assert!(gated > 0, "the matrix must carry at least one min_mem floor above 10 GiB: {low}");
    assert_offers_match_spec(&low, true);
}

// ── Engines tab / platform token ───────────────────────────────────────────

/// L4 spec §"Platform probe": "Consumers: dashboard `CURRENT_PLATFORM_ID`
/// (replaces the hardcoded literal)" and §"Engines tab". The dashboard must
/// render the probe's id and expose the Engines tab.
#[tokio::test(flavor = "multi_thread")]
async fn l4_dashboard_platform_token_and_engines_tab() {
    let _gate = gate().await;
    let env = TestEnv::new("dashboard");
    let mock = spawn_mock(MockCfg::default());
    let mut d = Daemon::start(&env, &base_config(&mock.url));
    d.ready().await;

    let platform = d.get_json("/api/platform").await;
    let id = platform["id"]
        .as_str()
        .expect("L4 spec: GET /api/platform carries the probe id")
        .to_string();

    let (status, html) = d.get_text("/dashboard").await;
    assert_eq!(status, 200, "GET /dashboard: {html}");
    assert!(
        !html.contains("__CURRENT_PLATFORM_ID__"),
        "dashboard platform token was not rendered (L4 spec: CURRENT_PLATFORM_ID replaces the hardcoded literal)"
    );
    assert!(
        html.contains(&format!("CURRENT_PLATFORM_ID = '{id}'"))
            || html.contains(&format!("CURRENT_PLATFORM_ID=\"{id}\"")),
        "dashboard does not carry the probed platform id `{id}` (L4 spec); body excerpt: {}",
        &html[html.find("CURRENT_PLATFORM_ID").unwrap_or(0)..][..200.min(html.len().saturating_sub(html.find("CURRENT_PLATFORM_ID").unwrap_or(0)))]
    );
    assert!(
        html.contains("nav-engines") && html.contains("engines-view"),
        "Engines tab missing from the dashboard (L4 spec §Routing + dashboard: Engines tab)"
    );
}

// ── Engines API: reads, gating, busy code ──────────────────────────────────

/// Contract surface from the leg spec §CLI/§Routing+dashboard and acceptance
/// (1)/(8): engine reads are loopback; mutations are loopback + CSRF gated;
/// unknown models are 404; capacity failures are 503 with the shared `-32005`
/// busy code ("explicit no-fit", "queue full").
#[tokio::test(flavor = "multi_thread")]
async fn l4_engines_api_reads_gated_mutations_and_busy_code() {
    let _gate = gate().await;
    let env = TestEnv::new("engines-api");
    let _guard = ChildGuard::new(&env.root);
    let mock = spawn_mock(MockCfg::default());
    let mut d = Daemon::start_with_env(
        &env,
        &base_config(&mock.url),
        &[("BRAINROUTER_BUDGET_GIB".into(), "10".into())],
    );
    d.ready().await;

    let e = engines(&d).await;
    for key in ["models", "queue", "budget", "adoption", "adapters"] {
        assert!(e.get(key).is_some(), "GET /api/engines missing `{key}`: {e}");
    }
    assert!(e["models"].is_array(), "{e}");
    assert!(e["queue"].is_array(), "{e}");
    assert!(e["budget"]["cap_gib"].is_number(), "budget gauge carries the cap: {e}");
    let adapters = e["adapters"].as_array().unwrap();
    for wanted in ["fake", "llama_cpp", "external"] {
        assert!(
            adapters.iter().any(|a| a == &json!(wanted)),
            "adapter `{wanted}` not registered (L4 spec §Engine adapters: llama_swap_shim, \
             llama_cpp native, fake); adapters: {adapters:?}"
        );
    }

    // Reads are not gated; mutations are loopback + CSRF gated.
    let (status, _) = d.get_text("/api/engines/queue").await;
    assert_eq!(status, 200, "GET /api/engines/queue");

    let entry = fake_entry("ray-l4-csrf", Some(&gguf(&env, "csrf.gguf")), 1.0, &[], 0);
    add_entry(&d, entry).await;
    let (status, text) = d
        .post_json_headers(
            "/api/engines/pin",
            json!({"model": "ray-l4-csrf", "reserved_gib": 1.0}),
            &[("Origin", "http://evil.example")],
        )
        .await;
    assert_eq!(
        status, 403,
        "swap mutations must be CSRF-gated (the existing loopback+CSRF boundary): {text}"
    );
    let (status, text) = d
        .post_json_headers(
            "/api/engines/pin",
            json!({"model": "ray-l4-csrf", "reserved_gib": 1.0}),
            &[("Origin", "http://127.0.0.1:9099")],
        )
        .await;
    assert_eq!(status, 200, "loopback Origin must pass the gate: {text}");
    let (status, text) = d
        .post_json("/api/engines/unpin", json!({"model": "ray-l4-csrf"}))
        .await;
    assert_eq!(status, 200, "unpin: {text}");

    // Unknown model: 404 (never a spawn attempt).
    let (status, text) = ensure(&d, "ray-l4-does-not-exist").await;
    assert_eq!(status, 404, "unknown model ensure must be 404: {text}");

    // Explicit no-fit: a request larger than the whole budget ⇒ 503 -32005.
    add_entry(
        &d,
        fake_entry("ray-l4-too-big", Some(&gguf(&env, "too-big.gguf")), 11.0, &[], 0),
    )
    .await;
    let (status, text) = ensure(&d, "ray-l4-too-big").await;
    assert_eq!(
        status, 503,
        "over-budget ensure must be an explicit 503, not a generic 502: {text}"
    );
    let body: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    assert_eq!(body["code"], json!(-32005), "busy code contract (-32005): {text}");
    assert!(
        text.contains("no-fit"),
        "the failure must name the explicit no-fit (L4 spec §LRU): {text}"
    );
}

// ── (1)/(6) Fake-adapter lifecycle, stem assertion, crash/restart ─────────

#[tokio::test(flavor = "multi_thread")]
async fn l4_fake_adapter_lifecycle_stem_assert_crash_restart() {
    let _gate = gate().await;
    let env = TestEnv::new("lifecycle");
    let _guard = ChildGuard::new(&env.root);
    let mock = spawn_mock(MockCfg::default());
    let mut d = Daemon::start_with_env(
        &env,
        &base_config(&mock.url),
        &[
            ("BRAINROUTER_BUDGET_GIB".into(), "10".into()),
            ("BRAINROUTER_SWAP_WAIT_MS".into(), "8000".into()),
        ],
    );
    d.ready().await;

    // ensure spawns (scripted load time); loading exposes progress; ready
    // forces the stem set.
    let gguf_a = gguf(&env, "a.gguf");
    add_entry(
        &d,
        fake_entry(
            "ray-l4-a",
            Some(&gguf_a),
            1.0,
            &[("FAKE_LOAD_MS", "700"), ("FAKE_PROGRESS", "1")],
            0,
        ),
    )
    .await;
    // ensure() returns only once Ready, so the Loading row is observed while
    // the ensure request is still in flight.
    let client = d.http();
    let ensure_url = d.endpoint("/api/engines/ensure");
    let inflight = tokio::spawn(async move {
        let r = client
            .post(&ensure_url)
            .json(&json!({"model": "ray-l4-a"}))
            .send()
            .await
            .expect("ensure send");
        (r.status().as_u16(), r.text().await.unwrap_or_default())
    });
    // Scripted progress is visible while Loading (L4 spec §Load progress).
    let loading = poll_until("a loading row with progress", 10, || async {
        let e = engines(&d).await;
        let m = model_status(&e, "ray-l4-a")?.clone();
        (m["state"] == json!("loading")).then_some(m)
    })
    .await;
    let progress = &loading["progress"];
    assert!(
        progress["fraction"].as_f64().is_some_and(|f| (0.0..=1.0).contains(&f)),
        "fake progress must render a fraction (L4 spec: fake = scripted): {loading}"
    );
    assert!(progress["eta_ms"].is_number(), "scripted ETA: {loading}");

    let (status, text) = inflight.await.unwrap();
    assert_eq!(status, 200, "ensure spawn: {text}");
    let out: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(out["spawned"], json!(true), "first ensure spawns: {text}");
    let endpoint = out["endpoint"].as_str().unwrap().to_string();
    assert!(endpoint.starts_with("http://127.0.0.1:"), "managed port policy endpoint: {endpoint}");

    wait_model_state(&d, "ray-l4-a", "ready").await;
    let e = engines(&d).await;
    let m = model_status(&e, "ray-l4-a").unwrap();
    assert_eq!(m["source"], json!("native"), "native registry resident: {m}");
    assert_eq!(m["resident"], json!(true), "{m}");
    assert!(m["ewma_load_ms"].is_number(), "measured load lands in the registry (ETA source): {m}");
    assert_eq!(
        pids_with_arg(&gguf_a.display().to_string()).len(),
        1,
        "exactly one fake child (the fake adapter is a REAL child, L4 spec §Engine adapters)"
    );

    // Ready → attach (no second process).
    let out = ensure_ok(&d, "ray-l4-a").await;
    assert_eq!(out["spawned"], json!(false), "Ready must attach: {out}");
    assert_eq!(out["endpoint"], json!(endpoint), "attach keeps the endpoint: {out}");

    // Stop = SIGTERM the child and release the lock (spec: start-ownership
    // "own child cleaned").
    let (status, text) = d.post_json("/api/engines/evict", json!({"model": "ray-l4-a"})).await;
    assert_eq!(status, 200, "evict: {text}");
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["stopped"], json!(true), "{text}");
    poll_until("the evicted child to exit", 10, || async {
        pids_with_arg(&gguf_a.display().to_string()).is_empty().then_some(())
    })
    .await;
    assert!(flock_is_free(&gguf_a), "evict must release the child-held GGUF flock");

    // Stem assertion failure ⇒ Failed (spawn validation scope).
    let gguf_b = gguf(&env, "b.gguf");
    add_entry(
        &d,
        fake_entry(
            "ray-l4-b",
            Some(&gguf_b),
            1.0,
            &[("FAKE_LOAD_MS", "50"), ("FAKE_STEMS", "someone-else")],
            0,
        ),
    )
    .await;
    let (status, text) = ensure(&d, "ray-l4-b").await;
    assert_eq!(status, 503, "stem mismatch must fail the spawn: {text}");
    assert!(
        text.contains("stem assertion"),
        "failure must name the stem assertion (L4 spec §Boot invariants): {text}"
    );
    let e = engines(&d).await;
    let m = model_status(&e, "ray-l4-b").unwrap();
    assert_eq!(m["state"], json!("failed"), "stem failure ⇒ Failed: {m}");
    poll_until("the stem-failed child to be cleaned", 10, || async {
        pids_with_arg(&gguf_b.display().to_string()).is_empty().then_some(())
    })
    .await;

    // Crash during load ⇒ bounded restart, then Failed.
    let gguf_c = gguf(&env, "c.gguf");
    add_entry(
        &d,
        fake_entry(
            "ray-l4-c",
            Some(&gguf_c),
            1.0,
            &[("FAKE_LOAD_MS", "5000"), ("FAKE_CRASH_AFTER_MS", "60")],
            0,
        ),
    )
    .await;
    let (status, text) = ensure(&d, "ray-l4-c").await;
    assert_eq!(status, 503, "a crash-looping engine ends Failed: {text}");
    let e = engines(&d).await;
    assert_eq!(
        model_status(&e, "ray-l4-c").unwrap()["state"],
        json!("failed"),
        "bounded restart must terminate in Failed (L4 spec: plugin-supervisor pattern)"
    );
    let spawns = budget_entries(&env)
        .iter()
        .filter(|e| e["kind"] == json!("spawn") && e["model"] == json!("ray-l4-c"))
        .count();
    assert!(
        (2..=4).contains(&spawns),
        "crash ⇒ bounded restart (initial spawn + ≤3 retries), got {spawns} spawn records"
    );
}

// ── (1) Concurrent ensure ⇒ one spawn, all attach (child-held flock) ───────

#[tokio::test(flavor = "multi_thread")]
async fn l4_concurrent_ensure_one_spawn_all_attach_and_lock_release() {
    let _gate = gate().await;
    let env = TestEnv::new("concurrent");
    let _guard = ChildGuard::new(&env.root);
    let mock = spawn_mock(MockCfg::default());
    let mut d = Daemon::start_with_env(
        &env,
        &base_config(&mock.url),
        &[
            ("BRAINROUTER_BUDGET_GIB".into(), "10".into()),
            ("BRAINROUTER_SWAP_WAIT_MS".into(), "15000".into()),
        ],
    );
    d.ready().await;

    let path = gguf(&env, "conc.gguf");
    add_entry(
        &d,
        fake_entry("ray-l4-conc", Some(&path), 1.0, &[("FAKE_LOAD_MS", "1200")], 0),
    )
    .await;

    // N concurrent ensure(X): exactly one spawn, every caller attaches to the
    // same endpoint (L4 spec §Lifecycle: "concurrent ensure(X) = exactly one
    // spawn, all attach").
    let client = d.http();
    let url = d.endpoint("/api/engines/ensure");
    let mut handles = Vec::new();
    for _ in 0..5 {
        let client = client.clone();
        let url = url.clone();
        handles.push(tokio::spawn(async move {
            let r = client
                .post(&url)
                .json(&json!({"model": "ray-l4-conc"}))
                .send()
                .await
                .expect("concurrent ensure send");
            let status = r.status().as_u16();
            let text = r.text().await.unwrap_or_default();
            (status, text)
        }));
    }
    let mut spawned = 0usize;
    let mut endpoints = Vec::new();
    for h in handles {
        let (status, text) = h.await.unwrap();
        assert_eq!(status, 200, "concurrent ensure: {text}");
        let v: Value = serde_json::from_str(&text).unwrap();
        if v["spawned"] == json!(true) {
            spawned += 1;
        }
        endpoints.push(v["endpoint"].as_str().unwrap_or("").to_string());
    }
    assert_eq!(spawned, 1, "exactly one concurrent ensure may spawn: {endpoints:?}");
    let first = endpoints[0].clone();
    assert!(
        endpoints.iter().all(|e| *e == first),
        "every concurrent ensure attaches to the one resident: {endpoints:?}"
    );
    assert_eq!(
        pids_with_arg(&path.display().to_string()).len(),
        1,
        "one child process for N concurrent ensures"
    );

    // The CHILD holds the flock: a foreign lock attempt must fail while it
    // lives (L4 spec §Boot invariants: "HELD BY THE CHILD").
    assert!(!flock_is_free(&path), "the child holds the GGUF flock while serving");

    // Child exit releases the lock; the next ensure takes it again.
    d.post_json("/api/engines/evict", json!({"model": "ray-l4-conc"})).await;
    poll_until("the evicted child's lock to release", 15, || async {
        (pids_with_arg(&path.display().to_string()).is_empty() && flock_is_free(&path)).then_some(())
    })
    .await;
    let out = ensure_ok(&d, "ray-l4-conc").await;
    assert_eq!(out["spawned"], json!(true), "a fresh child after the stop: {out}");
    assert!(!flock_is_free(&path), "the fresh child inherits the lock");
}

// ── (1) Flock: second engine on the same canonical GGUF is refused ────────

#[tokio::test(flavor = "multi_thread")]
async fn l4_flock_refuses_second_engine_on_same_gguf() {
    let _gate = gate().await;
    let env = TestEnv::new("flock");
    let _guard = ChildGuard::new(&env.root);
    let mock = spawn_mock(MockCfg::default());
    let mut d = Daemon::start_with_env(
        &env,
        &base_config(&mock.url),
        &[
            ("BRAINROUTER_BUDGET_GIB".into(), "10".into()),
            ("BRAINROUTER_SWAP_WAIT_MS".into(), "8000".into()),
        ],
    );
    d.ready().await;

    let path = gguf(&env, "shared.gguf");
    add_entry(
        &d,
        fake_entry("ray-l4-lock-a", Some(&path), 1.0, &[("FAKE_LOAD_MS", "100")], 0),
    )
    .await;
    add_entry(
        &d,
        fake_entry("ray-l4-lock-b", Some(&path), 1.0, &[("FAKE_LOAD_MS", "100")], 0),
    )
    .await;

    ensure_ok(&d, "ray-l4-lock-a").await;
    wait_model_state(&d, "ray-l4-lock-a", "ready").await;
    assert!(!flock_is_free(&path), "first engine's child holds the lock");

    // Single residency: X's canonical GGUF resident anywhere ⇒ X is
    // Ready-at-that-endpoint, no second process (L4 spec §Residency view).
    let attached = ensure_ok(&d, "ray-l4-lock-b").await;
    assert_eq!(attached["spawned"], json!(false), "same GGUF ⇒ attach, no spawn: {attached}");
    assert_eq!(
        pids_with_arg(&path.display().to_string()).len(),
        1,
        "single residency: one process for the shared GGUF"
    );

    // The flock itself refuses a native start while a foreign process holds
    // it (the native-vs-native race guard, L4 spec §Boot invariants).
    d.post_json("/api/engines/evict", json!({"model": "ray-l4-lock-a"}))
        .await;
    poll_until("the lock holder to exit", 10, || async {
        (pids_with_arg(&path.display().to_string()).is_empty()).then_some(())
    })
    .await;
    let foreign = fs::File::open(&path).unwrap();
    use std::os::unix::io::AsRawFd;
    let rc = unsafe { libc::flock(foreign.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(rc, 0, "test could not take the fixture lock");
    let (status, text) = ensure(&d, "ray-l4-lock-b").await;
    assert_eq!(
        status, 503,
        "the per-GGUF lock must refuse a native start (L4 spec §Boot invariants): {text}"
    );
    assert!(
        text.contains("locked by another running engine"),
        "refusal must name the child-held lock: {text}"
    );
    drop(foreign);

    // Releasing the holder lets the model start.
    let out = ensure_ok(&d, "ray-l4-lock-b").await;
    assert_eq!(out["spawned"], json!(true), "{out}");
    wait_model_state(&d, "ray-l4-lock-b", "ready").await;
    assert!(!flock_is_free(&path), "the new child now holds the lock");
}

// ── (2) Residency before budget; attach-not-spawn; explicit no-fit ────────

#[tokio::test(flavor = "multi_thread")]
async fn l4_residency_attach_before_budget_and_explicit_no_fit() {
    let _gate = gate().await;
    let env = TestEnv::new("residency");
    let _guard = ChildGuard::new(&env.root);
    // llama-swap shim: one model resident with a (fake) proxy endpoint and a
    // cmd naming a GGUF, enough for the residency observer.
    let mock = spawn_mock(MockCfg {
        models: vec!["ray-l4-shim-s".into()],
        ..Default::default()
    });
    mock.set_running(json!({"running": [
        {"model": "ray-l4-shim-s", "state": "ready", "proxy": mock.url, "cmd": "--model /models/shim.gguf"}
    ]}));

    let mut d = Daemon::start_with_env(
        &env,
        &base_config(&mock.url),
        &[
            // Budget far below the shim resident's estimate: Residency must
            // precede budget for the attach.
            ("BRAINROUTER_BUDGET_GIB".into(), "10".into()),
            ("BRAINROUTER_SWAP_WAIT_MS".into(), "5000".into()),
        ],
    );
    d.ready().await;

    // Shim-sourced catalog entry (external engine: managed outside the
    // supervisor). Files empty ⇒ residency matches on the wire model key.
    add_entry(
        &d,
        json!({
            "id": "ray-l4-shim-s",
            "engine": "external",
            "files": [],
            "ctx": Value::Null,
            "quant": Value::Null,
            "family": Value::Null,
            "tools": Value::Null,
            "vram_estimate_gib": 50.0,
            "launch_spec": {"program": "", "args": [], "env": {}, "port_policy": "managed",
                            "served_model_alias": "ray-l4-shim-s"},
            "aliases": [], "groups": [], "name": Value::Null,
            "provenance": {"source": "cli", "imported_at": Value::Null, "spec_hash": ""},
            "last_stream_completed": 0, "ewma_load_ms": Value::Null, "fingerprints": [],
        }),
    )
    .await;

    // Residency observer (2 s cadence): the shim resident appears with
    // source=shim.
    poll_until("the shim resident in the Residency view", 20, || async {
        let e = engines(&d).await;
        let m = model_status(&e, "ray-l4-shim-s")?.clone();
        (m["resident"] == json!(true) && m["source"] == json!("shim")).then_some(())
    })
    .await;

    // Sized pin on the resident: counted ONCE (max(estimate, reserved)).
    let (status, text) = d
        .post_json("/api/engines/pin", json!({"model": "ray-l4-shim-s", "reserved_gib": 50.0}))
        .await;
    assert_eq!(status, 200, "pin: {text}");

    // Residency FIRST: budget is already exceeded, yet the ensure ATTACHES.
    let out = ensure_ok(&d, "ray-l4-shim-s").await;
    assert_eq!(out["spawned"], json!(false), "resident ⇒ attach, no native spawn: {out}");
    assert_eq!(
        out["endpoint"], json!(mock.url),
        "attach returns the resident endpoint (Ready-at-that-endpoint): {out}"
    );

    let e = engines(&d).await;
    let budget = &e["budget"];
    let resident_line = budget["residents"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["model"] == json!("ray-l4-shim-s"))
        .unwrap_or_else(|| panic!("shim resident missing from the budget gauge: {budget}"));
    assert_eq!(resident_line["source"], json!("shim"), "{resident_line}");
    assert_eq!(
        resident_line["gib"], json!(50.0),
        "each model id counted ONCE: max(estimate, reserved) = 50, never estimate+pin: {resident_line}"
    );
    assert!(
        !budget["pins"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["model"] == json!("ray-l4-shim-s")),
        "an observed resident must not ALSO count as an unmatched pin: {budget}"
    );
    assert_eq!(budget["used_gib"], json!(50.0), "used = 50 (cap 10): {budget}");

    // Budget full + NOT resident ⇒ explicit no-fit (never an unbudgeted spawn).
    let gguf_n = gguf(&env, "n.gguf");
    add_entry(
        &d,
        fake_entry("ray-l4-elsewhere", Some(&gguf_n), 8.0, &[("FAKE_LOAD_MS", "50")], 0),
    )
    .await;
    let (status, text) = ensure(&d, "ray-l4-elsewhere").await;
    assert_eq!(status, 503, "spawn over a full budget ⇒ explicit no-fit: {text}");
    assert!(
        text.contains("no-fit"),
        "the named no-fit event/message (L4 spec: explicit no-fit after candidates exhausted): {text}"
    );
    assert_eq!(
        serde_json::from_str::<Value>(&text).unwrap()["code"],
        json!(-32005),
        "{text}"
    );
    assert_eq!(
        pids_with_arg(&gguf_n.display().to_string()).len(),
        0,
        "a denied ensure must not leave a child"
    );
}

// ── (2) Sized pins: counted once, never evicted, pins-over-budget ─────────

#[tokio::test(flavor = "multi_thread")]
async fn l4_sized_pins_counted_once_never_evicted_and_pins_over_budget() {
    let _gate = gate().await;
    let env = TestEnv::new("pins");
    let _guard = ChildGuard::new(&env.root);
    let mock = spawn_mock(MockCfg::default());
    let mut d = Daemon::start_with_env(
        &env,
        &base_config(&mock.url),
        &[
            ("BRAINROUTER_BUDGET_GIB".into(), "20".into()),
            ("BRAINROUTER_SWAP_WAIT_MS".into(), "8000".into()),
        ],
    );
    d.ready().await;

    for (id, estimate) in [
        ("ray-l4-pa", 4.0),
        ("ray-l4-pb", 4.0),
        ("ray-l4-pc", 4.0),
        ("ray-l4-pd", 1.0),
        ("ray-l4-pe", 1.0),
    ] {
        add_entry(
            &d,
            fake_entry(id, Some(&gguf(&env, &format!("{id}.gguf"))), estimate, &[("FAKE_LOAD_MS", "100")], 0),
        )
        .await;
    }
    for (id, gib) in [("ray-l4-pa", 6.0), ("ray-l4-pb", 4.0), ("ray-l4-pd", 2.0)] {
        let (status, text) = d
            .post_json("/api/engines/pin", json!({"model": id, "reserved_gib": gib}))
            .await;
        assert_eq!(status, 200, "pin {id}: {text}");
    }

    // Unmatched pins count their FULL reserved_gib and are labelled
    // source=pin (L4 spec: "unmatched pin ⇒ full reserved_gib").
    let e = engines(&d).await;
    let budget = &e["budget"];
    assert_eq!(budget["cap_gib"], json!(20.0), "{budget}");
    assert_eq!(
        budget["used_gib"], json!(12.0),
        "unmatched pins reserve 6 + 4 + 2: {budget}"
    );
    for (id, gib) in [("ray-l4-pa", 6.0), ("ray-l4-pb", 4.0), ("ray-l4-pd", 2.0)] {
        let line = budget["pins"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["model"] == json!(id))
            .unwrap_or_else(|| panic!("pin {id} missing from the gauge: {budget}"));
        assert_eq!(line["source"], json!("pin"), "{line}");
        assert_eq!(line["gib"], json!(gib), "{line}");
    }

    // Resident with a sized pin: counted ONCE as max(estimate, reserved).
    ensure_ok(&d, "ray-l4-pa").await;
    wait_model_state(&d, "ray-l4-pa", "ready").await;
    let e = engines(&d).await;
    let budget = &e["budget"];
    let pa = budget["residents"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["model"] == json!("ray-l4-pa"))
        .unwrap()
        .clone();
    assert_eq!(pa["gib"], json!(6.0), "sized pin counted once, not estimate+pin=10: {pa}");
    assert!(
        !budget["pins"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["model"] == json!("ray-l4-pa")),
        "an observed resident must not ALSO count as an unmatched pin: {budget}"
    );
    assert_eq!(
        budget["used_gib"], json!(12.0),
        "used = 6 (PA) + 4 (PB pin) + 2 (PD pin); PA converted, never double-counted: {budget}"
    );

    // A resident attaches even when the pins already reserve past the cap.
    let out = ensure_ok(&d, "ray-l4-pa").await;
    assert_eq!(out["spawned"], json!(false), "resident attaches unaffected: {out}");

    // pins-over-budget ⇒ every ensure THAT WOULD SPAWN is an explicit no-fit
    // (L4 spec: Residency precedes budget; pins are in the arithmetic).
    let (status, text) = d
        .post_json("/api/engines/pin", json!({"model": "ray-l4-pd", "reserved_gib": 14.0}))
        .await;
    assert_eq!(status, 200, "re-pin: {text}");
    let e = engines(&d).await;
    assert_eq!(
        e["budget"]["used_gib"], json!(24.0),
        "6 (PA resident) + 4 (PB pin) + 14 (PD pin) > cap 20: {}",
        e["budget"]
    );
    let (status, text) = ensure(&d, "ray-l4-pc").await;
    assert_eq!(status, 503, "pins-over-budget spawn must be no-fit: {text}");
    assert!(text.contains("no-fit"), "{text}");
    assert_eq!(
        model_status(&engines(&d).await, "ray-l4-pc").unwrap()["resident"],
        json!(false),
        "the denied model must not be resident"
    );
    // Pinned residents are never eviction candidates (no-fit, not a silent
    // eviction of PA/PB).
    let e = engines(&d).await;
    for id in ["ray-l4-pa"] {
        assert_eq!(
            model_status(&e, id).unwrap()["resident"],
            json!(true),
            "pinned resident `{id}` must never be evicted: {e}"
        );
    }
    assert_eq!(e["budget"]["evictions_today"], json!(0), "no eviction may happen: {}", e["budget"]);

    // Removing the oversized unmatched pin restores admission.
    d.post_json("/api/engines/unpin", json!({"model": "ray-l4-pd"})).await;
    ensure_ok(&d, "ray-l4-pc").await;
    wait_model_state(&d, "ray-l4-pc", "ready").await;
    let e = engines(&d).await;
    assert_eq!(
        e["budget"]["used_gib"], json!(14.0),
        "6 (PA) + 4 (PB pin) + 4 (PC): {}",
        e["budget"]
    );
    assert!(
        !e["budget"]["pins"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["model"] == json!("ray-l4-pd")),
        "unpin must drop the reservation: {}",
        e["budget"]
    );
    ensure_ok(&d, "ray-l4-pd").await;
    wait_model_state(&d, "ray-l4-pd", "ready").await;

    // LRU under pressure still refuses to touch pinned residents: raising PA's
    // pin fills the cap, and the new model evicts the least-recent UNPINNED
    // resident (PC, clock 0, id order before PD).
    let (status, text) = d
        .post_json("/api/engines/pin", json!({"model": "ray-l4-pa", "reserved_gib": 12.0}))
        .await;
    assert_eq!(status, 200, "size PA up: {text}");
    let e = engines(&d).await;
    assert_eq!(
        e["budget"]["used_gib"], json!(21.0),
        "12 (PA) + 4 (PB pin) + 4 (PC) + 1 (PD): {}",
        e["budget"]
    );
    ensure_ok(&d, "ray-l4-pe").await;
    wait_model_state(&d, "ray-l4-pe", "ready").await;
    let e = engines(&d).await;
    assert_eq!(
        model_status(&e, "ray-l4-pc").unwrap()["resident"],
        json!(false),
        "the unpinned LRU resident is the eviction candidate: {e}"
    );
    assert_eq!(
        model_status(&e, "ray-l4-pa").unwrap()["resident"],
        json!(true),
        "the pinned resident survives: {e}"
    );
    assert!(
        e["budget"]["evictions_today"].as_u64().unwrap_or(0) >= 1,
        "the evictions-today counter feeds the Engines tab: {}",
        e["budget"]
    );
}

// ── (1)/(7) LRU eviction: least recent STREAM COMPLETION ──────────────────

#[tokio::test(flavor = "multi_thread")]
async fn l4_lru_eviction_least_recent_stream_completed() {
    let _gate = gate().await;
    let env = TestEnv::new("lru");
    let _guard = ChildGuard::new(&env.root);
    let mock = spawn_mock(MockCfg::default());
    let mut d = Daemon::start_with_env(
        &env,
        &base_config(&mock.url),
        &[
            ("BRAINROUTER_BUDGET_GIB".into(), "10".into()),
            ("BRAINROUTER_SWAP_WAIT_MS".into(), "8000".into()),
        ],
    );
    d.ready().await;

    // Registry clocks seeded per model (the LRU key is the catalog id; the
    // clock is last STREAM COMPLETION, L4 spec §LRU).
    add_entry(&d, fake_entry("ray-l4-lru-a", Some(&gguf(&env, "a.gguf")), 4.0, &[("FAKE_LOAD_MS", "100")], 2_000)).await;
    add_entry(&d, fake_entry("ray-l4-lru-b", Some(&gguf(&env, "b.gguf")), 4.0, &[("FAKE_LOAD_MS", "100")], 1_000)).await;
    add_entry(&d, fake_entry("ray-l4-lru-c", Some(&gguf(&env, "c.gguf")), 4.0, &[("FAKE_LOAD_MS", "100")], 3_000)).await;

    ensure_ok(&d, "ray-l4-lru-a").await;
    ensure_ok(&d, "ray-l4-lru-b").await;
    wait_model_state(&d, "ray-l4-lru-a", "ready").await;
    wait_model_state(&d, "ray-l4-lru-b", "ready").await;

    // C needs 4 GiB; used = 8. Evict the least-recently-completed (B: 1000).
    let out = ensure_ok(&d, "ray-l4-lru-c").await;
    assert_eq!(out["spawned"], json!(true), "{out}");
    wait_model_state(&d, "ray-l4-lru-c", "ready").await;

    let e = engines(&d).await;
    assert_eq!(
        model_status(&e, "ray-l4-lru-b").unwrap()["resident"],
        json!(false),
        "LRU must evict the least recent STREAM COMPLETION (B=1000): {e}"
    );
    for survivor in ["ray-l4-lru-a", "ray-l4-lru-c"] {
        assert_eq!(
            model_status(&e, survivor).unwrap()["resident"],
            json!(true),
            "{survivor} must survive (clock newer than B): {e}"
        );
    }
    assert_eq!(model_status(&e, "ray-l4-lru-a").unwrap()["last_stream_completed"], json!(2000));
    assert_eq!(model_status(&e, "ray-l4-lru-c").unwrap()["last_stream_completed"], json!(3000));
    let evict_ledger: Vec<Value> = budget_entries(&env)
        .into_iter()
        .filter(|entry| entry["kind"] == json!("evict"))
        .collect();
    assert!(
        evict_ledger.iter().any(|entry| entry["model"] == json!("ray-l4-lru-b")),
        "eviction sequence must be recorded (budget.json): {evict_ledger:?}"
    );
}

// ── (1) Queue: FIFO positions, cap (503 -32005), cancel API ───────────────

#[tokio::test(flavor = "multi_thread")]
async fn l4_queue_positions_fifo_cap_and_cancel_api() {
    let _gate = gate().await;
    let env = TestEnv::new("queue");
    let _guard = ChildGuard::new(&env.root);
    let mock = spawn_mock(MockCfg::default());
    let mut d = Daemon::start_with_env(
        &env,
        &base_config(&mock.url),
        &[
            ("BRAINROUTER_BUDGET_GIB".into(), "10".into()),
            ("BRAINROUTER_SWAP_WAIT_MS".into(), "15000".into()),
            ("BRAINROUTER_SWAP_QUEUE_CAP".into(), "3".into()),
        ],
    );
    d.ready().await;

    let path = gguf(&env, "slow.gguf");
    add_entry(&d, fake_entry("ray-l4-q", Some(&path), 1.0, &[("FAKE_LOAD_MS", "2500")], 0)).await;

    // Spawner first: it owns queue position 1.
    let spawner = {
        let client = d.http();
        let url = d.endpoint("/api/engines/ensure");
        tokio::spawn(async move {
            let r = client.post(&url).json(&json!({"model": "ray-l4-q"})).send().await.unwrap();
            (r.status().as_u16(), r.text().await.unwrap_or_default())
        })
    };
    poll_until("the spawner's loading row", 20, || async {
        let e = engines(&d).await;
        model_status(&e, "ray-l4-q")
            .filter(|m| m["state"] == json!("loading"))
            .map(|_| ())
    })
    .await;

    // Three more waiters: cap 3 ⇒ two queue, one is refused with -32005.
    let mut waiters = Vec::new();
    for _ in 0..3 {
        let client = d.http();
        let url = d.endpoint("/api/engines/ensure");
        waiters.push(tokio::spawn(async move {
            let r = client.post(&url).json(&json!({"model": "ray-l4-q"})).send().await.unwrap();
            (r.status().as_u16(), r.text().await.unwrap_or_default())
        }));
    }

    // FIFO: per-request position, in arrival order (L4 spec §Lifecycle:
    // "Loading → queue (cap, per-request position)").
    let queue = poll_until("three queued rows", 10, || async {
        let e = engines(&d).await;
        let q = e["queue"].as_array()?.clone();
        (q.len() == 3).then_some(q)
    })
    .await;
    let positions: Vec<u64> = queue.iter().filter_map(|q| q["position"].as_u64()).collect();
    assert_eq!(positions, vec![1, 2, 3], "queue positions are 1-based FIFO: {queue:?}");
    let ids: Vec<u64> = queue.iter().filter_map(|q| q["id"].as_u64()).collect();
    assert!(ids.windows(2).all(|w| w[0] < w[1]), "FIFO ids ascend: {queue:?}");
    assert!(queue.iter().all(|q| q["model"] == json!("ray-l4-q")), "{queue:?}");

    // Cancel API contract (documented wire surface).
    let (status, text) = d.delete(&format!("/api/engines/queue/{}", ids[1])).await;
    assert_eq!(status, 200, "cancel queued request: {text}");
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["cancelled"], json!(true), "{text}");
    let (status, _) = d.delete("/api/engines/queue/999999").await;
    assert_eq!(status, 404, "unknown queue id must be 404");

    // One waiter is over the cap ⇒ 503 with the shared busy code.
    let mut ok = 0usize;
    let mut refused = 0usize;
    let mut spawned = 0usize;
    for h in waiters {
        let (status, text) = h.await.unwrap();
        match status {
            200 => {
                ok += 1;
                let v: Value = serde_json::from_str(&text).unwrap();
                if v["spawned"] == json!(true) {
                    spawned += 1;
                }
            }
            503 => {
                refused += 1;
                assert!(text.contains("-32005"), "queue-full uses the shared busy code: {text}");
                assert!(text.contains("queue"), "queue-full message: {text}");
            }
            other => panic!("queued ensure -> {other}: {text}"),
        }
    }
    let (status, text) = spawner.await.unwrap();
    assert_eq!(status, 200, "the spawner completes: {text}");
    if serde_json::from_str::<Value>(&text).unwrap()["spawned"] == json!(true) {
        spawned += 1;
    }
    assert_eq!(ok, 2, "cap 3 = spawner row + 2 queued; the third waiter is refused (got {ok} ok)");
    assert_eq!(refused, 1, "exactly one queue-full refusal");
    assert_eq!(spawned, 1, "one spawn for the whole queue");
}

// ── (4) llama-swap import: macros, groups, fidelity, idempotent, dry-run ──

#[tokio::test(flavor = "multi_thread")]
async fn l4_import_llama_swap_macros_groups_fidelity_idempotent_dryrun() {
    let _gate = gate().await;
    let env = TestEnv::new("import");
    let mock = spawn_mock(MockCfg::default());
    let mut d = Daemon::start(&env, &base_config(&mock.url));
    d.ready().await;

    let gguf_a = gguf(&env, "ray-l4-imp-a-Q4_K_M.gguf");
    let gguf_b = gguf(&env, "ray-l4-imp-b-Q8_0.gguf");
    let fixture_dir = env.home.join("fixtures");
    fs::create_dir_all(&fixture_dir).unwrap();
    let fixture = fixture_dir.join("llama-swap.yaml");
    let yaml = format!(
        r#"
startPort: 5800
macros:
  "ls": "/usr/bin/llama-server"
  "common": >-
    --jinja -t 8 -ngl 999
    --ctx-size ${{ctx}}
  "ctx": "16384"
groups:
  pair:
    swap: false
    exclusive: false
    members:
      - ray-l4-imp-a
      - ray-l4-imp-b
  solo:
    members: [ray-l4-imp-b]
models:
  ray-l4-imp-a:
    name: "Model A"
    aliases: ["a-alias"]
    env:
      - "LLAMA_CONTAINER=a"
    macros:
      "ctx": "32768"
    cmd: |
      ${{ls}}
      --port ${{PORT}}
      ${{common}}
      --model {}
  ray-l4-imp-b:
    cmd: |
      ${{ls}} --port ${{PORT}} --ctx-size ${{ctx}} --model {} --jinja
"#,
        gguf_a.display(),
        gguf_b.display()
    );
    fs::write(&fixture, &yaml).unwrap();

    // Dry-run via the API preview: stable and inert.
    let query = format!("/api/engines/import/preview?path={}", fixture.display());
    let (s1, first) = d.get_text(&query).await;
    let (s2, second) = d.get_text(&query).await;
    assert_eq!(s1, 200, "import preview: {first}");
    assert_eq!(s2, 200, "import preview: {second}");
    assert_eq!(first, second, "two dry-runs must be byte-identical (L4 spec: --dry-run stable)");
    let preview: Value = serde_json::from_str(&first).unwrap();
    assert_eq!(preview["would_import"], json!(2), "{preview}");
    assert!(
        !engines(&d).await["models"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["id"] == json!("ray-l4-imp-a")),
        "dry-run must not write the catalog (L4 spec: --dry-run stable/inert)"
    );
    assert!(!env.swap_catalog_path().exists(), "preview wrote swap_catalog.json");

    // Apply: idempotent merge into the catalog.
    let (status, text) = d
        .post_json("/api/engines/import/apply", json!({"path": fixture.display().to_string()}))
        .await;
    assert_eq!(status, 200, "import apply: {text}");
    let outcome: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(outcome["added"], json!(2), "{outcome}");
    assert_eq!(
        outcome["skipped"].as_array().map(|v| v.len()),
        Some(0),
        "CLI-owned entries would be skipped: {outcome}"
    );

    // Fidelity: every field of the recorded entry, macros fully expanded.
    let catalog: Value = serde_json::from_str(&fs::read_to_string(env.swap_catalog_path()).unwrap()).unwrap();
    let a = &catalog["entries"]["ray-l4-imp-a"];
    assert_eq!(a["engine"], json!("llama_cpp"), "{a}");
    assert_eq!(a["files"], json!([gguf_a.display().to_string()]), "{a}");
    assert_eq!(
        a["ctx"], json!(32768),
        "per-model macro shadowing must win over the global ctx (L4 spec: macro chain fully expanded): {a}"
    );
    assert_eq!(a["quant"], json!("Q4_K_M"), "{a}");
    assert_eq!(a["tools"], json!(true), "\"--jinja\" ⇒ tools: {a}");
    assert_eq!(a["name"], json!("Model A"), "{a}");
    assert_eq!(a["aliases"], json!(["a-alias"]), "{a}");
    assert_eq!(a["groups"], json!(["pair"]), "{a}");
    assert_eq!(a["launch_spec"]["program"], json!("/usr/bin/llama-server"), "{a}");
    assert_eq!(
        a["launch_spec"]["args"],
        json!([
            "--port", "${PORT}",
            "--jinja", "-t", "8", "-ngl", "999", "--ctx-size", "32768",
            "--model", gguf_a.display().to_string()
        ]),
        "every entry records fully-resolved args (only ${{PORT}} survives): {a}"
    );
    assert_eq!(
        a["launch_spec"]["served_model_alias"], json!("ray-l4-imp-a"),
        "served alias = catalog id: {a}"
    );
    assert_eq!(a["launch_spec"]["env"]["LLAMA_CONTAINER"], json!("a"), "{a}");
    assert_eq!(a["launch_spec"]["port_policy"], json!("managed"), "{a}");
    assert_eq!(a["provenance"]["source"], json!("llama-swap-import"), "{a}");
    assert_eq!(
        a["fingerprints"].as_array().map(|f| f.len()),
        Some(1),
        "imported entries track their GGUF fingerprint (spec_stale input): {a}"
    );
    assert!(
        a["vram_estimate_gib"].as_f64().is_some_and(|v| v > 0.0),
        "measured estimate from the file size: {a}"
    );

    let b = &catalog["entries"]["ray-l4-imp-b"];
    assert_eq!(b["ctx"], json!(16384), "non-shadowed model keeps the global ctx: {b}");
    assert_eq!(b["quant"], json!("Q8_0"), "{b}");
    assert_eq!(b["tools"], json!(true), "{b}");
    assert_eq!(b["groups"], json!(["pair", "solo"]), "{b}");

    // Groups: recorded as metadata (L7 slot-sharing input).
    assert_eq!(catalog["groups"]["pair"]["members"], json!(["ray-l4-imp-a", "ray-l4-imp-b"]));
    assert_eq!(catalog["groups"]["pair"]["swap"], json!(false));
    assert_eq!(catalog["groups"]["pair"]["exclusive"], json!(false));
    assert_eq!(catalog["groups"]["solo"]["swap"], json!(true), "llama-swap default swap=true");
    assert_eq!(catalog["groups"]["solo"]["exclusive"], json!(true), "default exclusive=true");
    assert_eq!(catalog["settings"]["startPort"], json!(5800), "llama-swap settings captured: {catalog}");

    // Idempotent re-import: nothing added/updated.
    let (status, text) = d
        .post_json("/api/engines/import/apply", json!({"path": fixture.display().to_string()}))
        .await;
    assert_eq!(status, 200, "re-import: {text}");
    let second_outcome: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(second_outcome["added"], json!(0), "{second_outcome}");
    assert_eq!(second_outcome["updated"], json!(0), "{second_outcome}");
    assert_eq!(second_outcome["unchanged"], json!(2), "{second_outcome}");

    // CLI dry-run: stable output and inert (a separate state root).
    let cli_state = env.root.join("cli-state");
    fs::create_dir_all(&cli_state).unwrap();
    let args = [
        "swap",
        "--state",
        cli_state.to_str().unwrap(),
        "import-llamaswap",
        "--dry-run",
        "--config",
        fixture.to_str().unwrap(),
    ];
    let first = run_cli(&env, &args).ok("swap import-llamaswap --dry-run");
    let second = run_cli(&env, &args).ok("swap import-llamaswap --dry-run");
    assert_eq!(first, second, "repeated CLI dry-runs are byte-identical");
    assert!(first.contains("would import 2 model"), "{first}");
    assert!(
        !cli_state.join("swap_catalog.json").exists(),
        "CLI dry-run wrote swap_catalog.json"
    );
}

// ── Migration v2.0-p4: snapshot, dry-run, restore, re-run no-op ───────────

#[tokio::test(flavor = "multi_thread")]
async fn l4_migration_v2_p4_snapshot_dryrun_restore_and_state_files() {
    let _gate = gate().await;

    // ── (a) fresh state: boot-time import creates the catalog ─────────────
    let env = TestEnv::new("mig-create");
    let _guard = ChildGuard::new(&env.root);
    let mock = spawn_mock(MockCfg::default());
    let gguf_m = gguf(&env, "m1.gguf");
    fs::create_dir_all(env.llama_swap_config_path().parent().unwrap()).unwrap();
    fs::write(
        env.llama_swap_config_path(),
        format!(
            "macros:\n  \"ls\": \"/usr/bin/llama-server\"\nmodels:\n  ray-l4-m1:\n    cmd: |\n      ${{ls}} --port ${{PORT}} --model {}\n",
            gguf_m.display()
        ),
    )
    .unwrap();
    let cfg = base_config(&mock.url);

    // --dry-run first: inert, and it previews the p4 phase.
    let config_path = env.root.join("config.yaml");
    fs::write(&config_path, &cfg).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_brainrouter"));
    cmd.args(["serve", "--config"])
        .arg(&config_path)
        .arg("--dry-run")
        .current_dir(&env.root)
        .stdin(Stdio::null());
    env.apply(&mut cmd, &[]);
    let out = cmd.output().expect("serve --dry-run");
    assert_eq!(out.status.code(), Some(0), "serve --dry-run must exit 0: {:?}", String::from_utf8_lossy(&out.stderr));
    let dry_stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        dry_stdout.contains("v2.0-p4") && dry_stdout.contains("would import"),
        "dry-run must preview the p4 llama-swap import: {dry_stdout}"
    );
    assert!(
        !env.swap_catalog_path().exists(),
        "--dry-run wrote swap_catalog.json (snapshot + dry-run contract)"
    );
    assert!(
        !env.state_dir().join("migrations").join("v2.0-p4").exists(),
        "--dry-run wrote the v2.0-p4 snapshot"
    );

    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    let catalog: Value = poll_until("the p4-imported catalog", 10, || async {
        let text = fs::read_to_string(env.swap_catalog_path()).ok()?;
        let v: Value = serde_json::from_str(&text).ok()?;
        v["entries"].get("ray-l4-m1")?;
        Some(v)
    })
    .await;
    assert_eq!(catalog["entries"]["ray-l4-m1"]["engine"], json!("llama_cpp"), "{catalog}");
    assert!(
        catalog["migrations"].get("v2.0-p4").is_some(),
        "one-shot phase marker: {catalog}"
    );
    assert!(
        env.state_dir().join("migrations").join("v2.0-p4").exists(),
        "v2.0-p4 snapshot directory (state-snapshot contract) missing"
    );

    // `budget.json` is a named L4 state file; it appears on the first ledger
    // write (pin/spawn). Assert it is created by a pin and lives beside the
    // catalog.
    let (status, text) = d
        .post_json("/api/engines/pin", json!({"model": "ray-l4-m1", "reserved_gib": 1.0}))
        .await;
    assert_eq!(status, 200, "pin creates the ledger: {text}");
    assert!(env.budget_path().exists(), "budget.json must be created in the state dir");
    d.stop();

    // Re-run no-op: a second boot over the migrated state changes no bytes.
    let after_first = fs::read_to_string(env.swap_catalog_path()).unwrap();
    let mut d2 = Daemon::start_from_disk(&env);
    d2.ready().await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        fs::read_to_string(env.swap_catalog_path()).unwrap(),
        after_first,
        "a second boot over the v2.0-p4 state must be a no-op (one-shot migration)"
    );
    d2.stop();

    // ── (b) snapshot + restore byte-round-trip ────────────────────────────
    let env = TestEnv::new("mig-restore");
    let _guard = ChildGuard::new(&env.root);
    let mock = spawn_mock(MockCfg::default());
    let gguf_r = gguf(&env, "m1.gguf");
    fs::create_dir_all(env.llama_swap_config_path().parent().unwrap()).unwrap();
    fs::write(
        env.llama_swap_config_path(),
        format!(
            "macros:\n  \"ls\": \"/usr/bin/llama-server\"\nmodels:\n  ray-l4-r1:\n    cmd: |\n      ${{ls}} --port ${{PORT}} --model {}\n",
            gguf_r.display()
        ),
    )
    .unwrap();
    // Pre-migration state: a CLI-owned catalog + budget ledger with known
    // bytes (the snapshot must restore THEM verbatim).
    let pre_catalog = "{\"version\":1,\"entries\":{},\"pins\":{},\"groups\":{},\"settings\":{},\"migrations\":{}}";
    let pre_budget = "{\"entries\":[]}";
    fs::write(env.swap_catalog_path(), pre_catalog).unwrap();
    fs::write(env.budget_path(), pre_budget).unwrap();

    let cfg = base_config(&mock.url);
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    let snapshot_dir = env.state_dir().join("migrations").join("v2.0-p4");
    assert_eq!(
        fs::read_to_string(snapshot_dir.join("swap_catalog.json")).unwrap(),
        pre_catalog,
        "the p4 snapshot must capture the pre-migration catalog byte-for-byte"
    );
    assert_eq!(
        fs::read_to_string(snapshot_dir.join("budget.json")).unwrap(),
        pre_budget,
        "the p4 snapshot must capture the pre-migration budget ledger"
    );
    assert!(fs::read_to_string(env.swap_catalog_path()).unwrap().contains("ray-l4-r1"));

    // Mutate after migration: a new entry + a ledger record.
    add_entry(
        &d,
        fake_entry("ray-l4-post", Some(&gguf(&env, "post.gguf")), 1.0, &[("FAKE_LOAD_MS", "50")], 0),
    )
    .await;
    let (status, text) = d
        .post_json("/api/engines/pin", json!({"model": "ray-l4-post", "reserved_gib": 1.0}))
        .await;
    assert_eq!(status, 200, "pin: {text}");
    d.stop();

    // v2.0-p4's snapshot holds SWAP STATE ONLY. Restoring it must leave the
    // unrelated stores byte-for-byte untouched (config, credentials, cards).
    let config_path = env.root.join("config.yaml");
    let credentials_path = env.state_dir().join("credentials.json");
    let cards_path = env.state_dir().join("role_cards.json");
    fs::write(&credentials_path, r#"{"credentials":{}}"#).unwrap();
    fs::write(
        &cards_path,
        r#"{"ray-l4-sentinel":"role cards must survive a p4 restore"}"#,
    )
    .unwrap();
    let config_before = fs::read(&config_path).unwrap();
    let credentials_before = fs::read(&credentials_path).unwrap();
    let cards_before = fs::read(&cards_path).unwrap();
    assert!(
        String::from_utf8_lossy(&config_before).contains("llama_swap"),
        "precondition: the boot config is on disk"
    );

    let run = run_cli(
        &env,
        &[
            "migration",
            "--config",
            env.root.join("config.yaml").to_str().unwrap(),
            "restore",
            "v2.0-p4",
        ],
    );
    run.ok("migration restore v2.0-p4");
    assert_eq!(
        fs::read(&config_path).unwrap(),
        config_before,
        "restore v2.0-p4 must not touch the config YAML (p4 snapshots swap state only)"
    );
    assert_eq!(
        fs::read(&credentials_path).unwrap(),
        credentials_before,
        "restore v2.0-p4 must not touch the credentials store"
    );
    assert_eq!(
        fs::read(&cards_path).unwrap(),
        cards_before,
        "restore v2.0-p4 must not touch the role-cards store"
    );
    assert_eq!(
        fs::read_to_string(env.swap_catalog_path()).unwrap(),
        pre_catalog,
        "restore v2.0-p4 must round-trip the pre-migration catalog byte-for-byte"
    );
    assert_eq!(
        fs::read_to_string(env.budget_path()).unwrap(),
        pre_budget,
        "restore v2.0-p4 must round-trip the pre-migration budget ledger"
    );
}

// ── (3) Adoption markers: live child adopt, spec_stale, dead sweep ────────

fn read_marker(env: &TestEnv, id: &str) -> Value {
    let path = env.adoption_dir().join(format!("{id}.json"));
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("adoption marker {} unreadable ({e}); L4 spec §Boot invariants: \
             state-dir file per model {{pid, gguf, start_time, spec_hash, scope_name?}}", path.display()));
    serde_json::from_str(&text).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn l4_adoption_marker_live_child_adopted_spec_stale_and_dead_swept() {
    let _gate = gate().await;
    let env = TestEnv::new("adoption");
    let _guard = ChildGuard::new(&env.root);
    let mock = spawn_mock(MockCfg::default());
    let cfg = base_config(&mock.url);

    // Boot 1: spawn a fake native child; the marker is written BEFORE spawn
    // and completed with the pid/start_time (atomic marker contract).
    let mut d = Daemon::start_with_env(
        &env,
        &cfg,
        &[("BRAINROUTER_BUDGET_GIB".into(), "10".into()), ("BRAINROUTER_SWAP_WAIT_MS".into(), "8000".into())],
    );
    d.ready().await;
    let path = gguf(&env, "adopt.gguf");
    add_entry(&d, fake_entry("ray-l4-adopt", Some(&path), 1.0, &[("FAKE_LOAD_MS", "200")], 0)).await;
    ensure_ok(&d, "ray-l4-adopt").await;
    wait_model_state(&d, "ray-l4-adopt", "ready").await;

    let marker = read_marker(&env, "ray-l4-adopt");
    let pid = marker["pid"].as_u64().expect("marker pid") as u32;
    assert_eq!(marker["model"], json!("ray-l4-adopt"), "{marker}");
    assert_eq!(
        marker["gguf"].as_str().unwrap_or(""),
        fs::canonicalize(&path).unwrap().display().to_string(),
        "marker records the canonical GGUF identity: {marker}"
    );
    assert_eq!(
        marker["spec_hash"].as_str().map(str::len),
        Some(64),
        "marker records the resolved launch-spec hash (sha256): {marker}"
    );
    assert!(marker["endpoint"].as_str().unwrap_or("").starts_with("http://127.0.0.1:"), "{marker}");
    assert!(marker.get("start_time").is_some(), "{marker}");
    assert!(marker.get("scope_name").is_some(), "marker carries the cgroup scope name: {marker}");
    assert!(pid_alive(pid), "the fake child must be alive");
    assert_eq!(marker["pid"].as_u64().unwrap() as u32, pid);

    // Daemon CRASH (SIGKILL): the child survives and keeps holding the flock
    // (L4 spec: "daemon crash releases nothing of the child's").
    let daemon_pid = d.pid();
    d.kill9();
    poll_until("the daemon process to die", 10, || async {
        (!pid_alive(daemon_pid)).then_some(()) // the engine child stays alive
    })
    .await;
    assert!(pid_alive(pid), "the child must survive a daemon crash (adoption substrate)");
    assert!(!flock_is_free(&path), "the surviving child keeps the child-held flock");

    // Boot 2: ADOPT — live pid + gguf match + spec_hash match ⇒ no kill, no
    // respawn, registry shows the native resident.
    let mut d2 = Daemon::start_from_disk(&env);
    d2.ready().await;
    let report = poll_until("the adoption report", 10, || async {
        let e = engines(&d2).await;
        let adopted = e["adoption"]["adopted"].as_array()?.clone();
        adopted.iter().any(|m| m == &json!("ray-l4-adopt")).then_some(e)
    })
    .await;
    assert_eq!(
        model_status(&report, "ray-l4-adopt").unwrap()["source"],
        json!("native"),
        "adopted child is a native resident: {report}"
    );
    assert!(pid_alive(pid), "adoption must NOT kill the serving engine");
    let out = ensure_ok(&d2, "ray-l4-adopt").await;
    assert_eq!(out["spawned"], json!(false), "adoption ⇒ attach, no respawn: {out}");
    assert_eq!(
        pids_with_arg(&path.display().to_string()),
        vec![pid],
        "still exactly the adopted child (no second process)"
    );

    // spec_hash MISMATCH: an upgrade changed the resolved args. Boot 3 must
    // adopt + mark spec_stale and NEVER kill it (L4 spec).
    d2.kill9();
    let mut disk: Value = serde_json::from_str(&fs::read_to_string(env.swap_catalog_path()).unwrap()).unwrap();
    disk["entries"]["ray-l4-adopt"]["ctx"] = json!(8192);
    fs::write(env.swap_catalog_path(), serde_json::to_string_pretty(&disk).unwrap()).unwrap();
    let mut d3 = Daemon::start_from_disk(&env);
    d3.ready().await;
    let e = poll_until("the spec_stale badge", 10, || async {
        let e = engines(&d3).await;
        let stale = e["adoption"]["spec_stale"].as_array()?.clone();
        stale.iter().any(|m| m == &json!("ray-l4-adopt")).then_some(e)
    })
    .await;
    assert_eq!(
        model_status(&e, "ray-l4-adopt").unwrap()["spec_stale"],
        json!(true),
        "spec_hash mismatch ⇒ adopt + spec_stale (never kill): {e}"
    );
    assert!(pid_alive(pid), "a serving engine is never killed by an upgrade");

    // The converging operator sequence: explicit evict (stop only what we
    // started / adopted) → ensure spawns the new spec.
    d3.post_json("/api/engines/evict", json!({"model": "ray-l4-adopt"})).await;
    poll_until("the stale child to stop", 10, || async {
        (!pid_alive(pid)).then_some(())
    })
    .await;
    let out = ensure_ok(&d3, "ray-l4-adopt").await;
    assert_eq!(out["spawned"], json!(true), "post-evict ensure spawns the converged spec: {out}");
    wait_model_state(&d3, "ray-l4-adopt", "ready").await;
    let new_pid = pids_with_arg(&path.display().to_string());
    assert_eq!(new_pid.len(), 1, "one converged child: {new_pid:?}");
    assert_ne!(new_pid[0], pid, "a new pid serves the converged spec");
    assert_eq!(
        model_status(&engines(&d3).await, "ray-l4-adopt").unwrap()["spec_stale"],
        json!(false),
        "convergence clears the badge"
    );
    d3.stop();

    // Dead marker ⇒ swept on boot (no resident, marker removed, recorded in
    // the adoption report). A dead pid is the crash-between-write-and-spawn
    // case the marker ordering is designed for.
    let mut dead_child = Command::new("sh").arg("-c").arg("exit 0").spawn().unwrap();
    let dead = dead_child.id();
    dead_child.wait().unwrap();
    fs::create_dir_all(env.adoption_dir()).unwrap();
    let marker_path = env.adoption_dir().join("ray-l4-dead.json");
    fs::write(
        &marker_path,
        json!({
            "model": "ray-l4-dead", "pid": dead, "gguf": "",
            "start_time": 0, "spec_hash": "0".repeat(64),
            "scope_name": null, "endpoint": "http://127.0.0.1:1"
        })
        .to_string(),
    )
    .unwrap();
    let mut d4 = Daemon::start_from_disk(&env);
    d4.ready().await;
    poll_until("the dead marker sweep", 10, || async {
        let e = engines(&d4).await;
        let cleaned = e["adoption"]["cleaned"].as_array()?.clone();
        (cleaned.iter().any(|m| m == &json!("ray-l4-dead")) && !marker_path.exists()).then_some(())
    })
    .await;
    assert!(
        model_status(&engines(&d4).await, "ray-l4-dead").is_none(),
        "a swept marker must not fabricate a resident"
    );
}

// ── spec_stale from a changed GGUF fingerprint ────────────────────────────

/// L4 spec §Model registry: the catalog carries `files`; a tracked GGUF whose
/// mtime/size moved is the `spec_stale` signal (badge, never a kill).
#[tokio::test(flavor = "multi_thread")]
async fn l4_spec_stale_on_changed_gguf_fingerprint() {
    let _gate = gate().await;
    let env = TestEnv::new("fingerprint");
    let _guard = ChildGuard::new(&env.root);
    let mock = spawn_mock(MockCfg::default());
    let mut d = Daemon::start_with_env(
        &env,
        &base_config(&mock.url),
        &[("BRAINROUTER_BUDGET_GIB".into(), "10".into()), ("BRAINROUTER_SWAP_WAIT_MS".into(), "8000".into())],
    );
    d.ready().await;

    let path = gguf(&env, "fp.gguf");
    let meta = fs::metadata(&path).unwrap();
    let mtime = meta.modified().unwrap().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
    let mut entry = fake_entry("ray-l4-fp", Some(&path), 1.0, &[("FAKE_LOAD_MS", "100")], 0);
    entry["fingerprints"] = json!([{
        "path": path.display().to_string(), "mtime": mtime, "size": meta.len()
    }]);
    add_entry(&d, entry).await;
    ensure_ok(&d, "ray-l4-fp").await;
    wait_model_state(&d, "ray-l4-fp", "ready").await;
    assert_eq!(
        model_status(&engines(&d).await, "ray-l4-fp").unwrap()["spec_stale"],
        json!(false),
        "matching fingerprint ⇒ no stale badge"
    );

    // Change the tracked file (append a byte): size/mtime drift.
    let mut f = fs::OpenOptions::new().append(true).open(&path).unwrap();
    f.write_all(b"changed\n").unwrap();
    let stale = poll_until("the spec_stale badge after the GGUF changed", 10, || async {
        let e = engines(&d).await;
        let m = model_status(&e, "ray-l4-fp")?.clone();
        (m["spec_stale"] == json!(true)).then_some(m)
    })
    .await;
    assert_eq!(stale["resident"], json!(true), "a serving engine keeps serving (never killed): {stale}");
}

// ── (7) Events additive + inflight catalog-id stamp + LRU clock ───────────

#[tokio::test(flavor = "multi_thread")]
async fn l4_events_additive_swap_fields_and_inflight_catalog_stamp() {
    let _gate = gate().await;
    let env = TestEnv::new("events");
    let _guard = ChildGuard::new(&env.root);
    let mock = spawn_mock(MockCfg {
        models: vec!["ray-l4-shim-e".into(), "ray-l4-bare-e".into()],
        body_delay_ms: 900,
        ..Default::default()
    });
    mock.set_running(json!({"running": [
        {"model": "ray-l4-shim-e", "state": "ready", "proxy": mock.url, "cmd": "--model /models/e.gguf"}
    ]}));
    let mut d = Daemon::start_with_env(
        &env,
        &base_config(&mock.url),
        &[("BRAINROUTER_BUDGET_GIB".into(), "10".into())],
    );
    d.ready().await;

    add_entry(
        &d,
        json!({
            "id": "ray-l4-shim-e",
            "engine": "external",
            "files": [],
            "ctx": Value::Null, "quant": Value::Null, "family": Value::Null, "tools": Value::Null,
            "vram_estimate_gib": 1.0,
            "launch_spec": {"program": "", "args": [], "env": {}, "port_policy": "managed",
                            "served_model_alias": "ray-l4-shim-e"},
            "aliases": [], "groups": [], "name": Value::Null,
            "provenance": {"source": "cli", "imported_at": Value::Null, "spec_hash": ""},
            "last_stream_completed": 0, "ewma_load_ms": Value::Null, "fingerprints": [],
        }),
    )
    .await;
    poll_until("the shim resident", 20, || async {
        let e = engines(&d).await;
        let m = model_status(&e, "ray-l4-shim-e")?.clone();
        (m["resident"] == json!(true)).then_some(())
    })
    .await;

    // Stream through the daemon; the mock holds the body so the in-flight row
    // is observable. The additive catalog-id stamp is what LRU matches on.
    let resp = d
        .http()
        .post(d.endpoint("/v1/chat/completions"))
        .json(&json!({
            "model": "ray-l4-shim-e",
            "messages": [{"role": "user", "content": "ray-l4 events"}],
            "stream": true,
        }))
        .send()
        .await
        .expect("streaming chat");
    assert_eq!(resp.status().as_u16(), 200, "stream status");
    let row = poll_until("the in-flight row with catalog_id", 5, || async {
        let v = d.get_json("/api/inflight").await;
        v["requests"]
            .as_array()?
            .iter()
            .find(|r| r["catalog_id"] == json!("ray-l4-shim-e"))
            .cloned()
    })
    .await;
    assert_eq!(row["model"], json!("ray-l4-shim-e"), "v1 row fields unchanged: {row}");
    let body = resp.text().await.unwrap_or_default();
    assert!(body.contains("[DONE]"), "stream body: {body}");

    let event = poll_until("the routed event with swap fields", 10, || async {
        let v = d.get_json("/api/routing-events").await;
        v["events"]
            .as_array()?
            .iter()
            .find(|e| e["requested_model"] == json!("ray-l4-shim-e"))
            .cloned()
    })
    .await;
    assert_v1_event_fields(&event);
    assert_eq!(event["success"], json!(true), "{event}");
    assert_eq!(
        event["swap_state"], json!("ready"),
        "L4 additive swap_state (carried, not rendered): {event}"
    );
    assert_eq!(
        event["swap_source"], json!("shim"),
        "L4 additive swap_source = shim (residency source): {event}"
    );

    // The SniffStream drop hook writes the LRU clock on clean completion.
    let clock = poll_until("the registry LRU clock", 10, || async {
        let e = engines(&d).await;
        let m = model_status(&e, "ray-l4-shim-e")?.clone();
        (m["last_stream_completed"].as_u64().unwrap_or(0) > 0).then_some(m)
    })
    .await;
    assert!(clock["last_stream_completed"].as_u64().unwrap() > 0, "{clock}");

    // A non-catalogued model keeps v1 semantics: no swap_* values invented.
    let _ = d
        .http()
        .post(d.endpoint("/v1/chat/completions"))
        .json(&json!({
            "model": "ray-l4-bare-e",
            "messages": [{"role": "user", "content": "ray-l4 bare"}],
            "stream": true,
        }))
        .send()
        .await
        .expect("bare chat")
        .text()
        .await;
    let bare = poll_until("the bare-model event", 10, || async {
        let v = d.get_json("/api/routing-events").await;
        v["events"]
            .as_array()?
            .iter()
            .find(|e| e["requested_model"] == json!("ray-l4-bare-e"))
            .cloned()
    })
    .await;
    assert_eq!(bare["success"], json!(true), "{bare}");
    assert!(
        bare.get("swap_state").is_none_or(Value::is_null),
        "uncatalogued routes carry no swap_state: {bare}"
    );
}

// ── (8) CLI surface: reads direct, mutations refuse when the daemon is down ─

#[tokio::test(flavor = "multi_thread")]
async fn l4_cli_surface_daemon_down_refuses() {
    let _gate = gate().await;
    let env = TestEnv::new("cli");
    let state = env.root.join("cli-state");
    fs::create_dir_all(&state).unwrap();
    let socket = env.root.join("no-daemon.sock");
    let socket_arg = socket.to_str().unwrap();
    let state_arg = state.to_str().unwrap();

    // Reads work without a daemon.
    let status = run_cli(&env, &["swap", "--state", state_arg, "status"]).json("swap status");
    assert_eq!(
        status["source"], json!("state files (daemon down)"),
        "reads are direct state-file reads (L4 spec §CLI): {status}"
    );
    let catalog = run_cli(&env, &["swap", "--state", state_arg, "catalog", "list"]).json("swap catalog list");
    assert!(catalog.get("entries").is_some(), "{catalog}");
    assert!(catalog.get("groups").is_some(), "{catalog}");
    assert!(catalog.get("pins").is_some(), "{catalog}");

    let platform = run_cli(&env, &["platform", "show"]).ok("platform show");
    assert!(platform.contains("platform:"), "{platform}");
    let platform_json = run_cli(&env, &["platform", "show", "--json"]).json("platform show --json");
    assert!(platform_json["id"].is_string(), "{platform_json}");
    assert!(platform_json["facts"].is_object(), "{platform_json}");
    assert!(platform_json["offers"].is_array(), "{platform_json}");
    assert!(platform_json["budget_gib"].is_number(), "{platform_json}");

    // Mutations go through the daemon and refuse cleanly when it is down.
    for args in [
        vec!["swap", "--socket", socket_arg, "ensure", "ray-l4-x"],
        vec!["swap", "--socket", socket_arg, "evict", "ray-l4-x"],
        vec!["swap", "--socket", socket_arg, "pin", "ray-l4-x", "--reserved-gib", "1"],
        vec!["swap", "--socket", socket_arg, "unpin", "ray-l4-x"],
    ] {
        let run = run_cli(&env, &args);
        let combined = format!("{}{}", run.stdout, run.stderr);
        assert_ne!(run.code, 0, "`{}` must refuse with the daemon down: {combined}", args.join(" "));
        assert!(
            combined.contains("not reachable") || combined.contains("daemon is not reachable"),
            "daemon-down refusal must be explicit (L4 spec §CLI: mutations via daemon): {combined}"
        );
    }
}
