//! Acceptance tests for leg L5 (strixite + hipfire adapters, Active-Consumer
//! protection, takeover, external residents) of `docs/design/v2-streamlining.md`
//! — spec: `docs/design/v2-streamlining.L5.md` version L5-v3 (2026-10-23).
//!
//! Written by Ray from the design's promises alone (spec, not the build). The
//! builder runs these and NEVER edits them.
//!
//! Everything here is BLACK-BOX (live daemon over loopback + the CLI binary +
//! the documented state files). No Rust API of `src/swap/`, `src/engine_*` or
//! `src/active_consumers` is referenced, so the file compiles both before and
//! after the L5 merge; a missing surface fails RED with the spec cite in the
//! message.
//!
//! The fixture substrate is a scripted mock engine (`strix_server` / `hipfire`
//! copies of one Python HTTP server, the L4 `fake`-adapter pattern extended to
//! answer `/v1/models`, `/slots`, `/health` and a holdable SSE chat), an
//! injectable `systemctl` wrapper (`BRAINROUTER_SYSTEMCTL`), a read-only
//! `/proc`-visible external engine (argv names the process, `--port` carries
//! the endpoint), and the documented env seams
//! (`BRAINROUTER_FLEET_ENGINES`, `BRAINROUTER_STRIXITE_*`,
//! `BRAINROUTER_HIPFIRE_*`, `BRAINROUTER_ACTIVE_CONSUMER_GRACE_MS`,
//! `BRAINROUTER_GATE_*`, `BRAINROUTER_READINESS_TIMEOUT_MS`,
//! `BRAINROUTER_BUDGET_GIB`).
//!
//! Mapping of tests to the L5 "Acceptance (Ray)" fixture-tier items:
//!
//! (1) cutover sequence / (2) exclusivity:
//!   * `l5_strixite_adapter_spawn_conf_port_progress_stem_and_stop`
//!     (conf-generated port, slots-equivalent progress, stem assert, stop)
//!   * `l5_strixite_stem_assert_conf_owned_flags_and_hung_start_kill9`
//!   * `l5_exclusive_cutover_evict_before_spawn_single_resident_and_one_spawn`
//! (3) ACS: proxy-stream pin, in-process review stream pin, grace span,
//!     SIGTERM-point re-check, `--override` + audit, `blocked:active-consumer`:
//!   * `l5_acs_proxy_stream_pin_evict_skip_override_and_grace`
//!   * `l5_acs_in_process_review_stream_pin_without_inflight_row`
//! (4) external residents: observed external, attach-not-spawn,
//!     budget-counted, cutover never signals an unregistered external,
//!     `swap adopt`, squat detection:
//!   * `l5_external_observer_ready_at_endpoint_attach_adopt_and_ownership`
//!   * `l5_hipfire_cards_sidecar_squat_and_port_free_gate`
//!     (`l5_takeover_unit_attach_refusal_single_override_and_pid_scope` also
//!     covers the unit-running attach-not-spawn shape)
//! (5) hipfire version-keyed schema:
//!   * `l5_hipfire_version_keyed_schema_from_engine_marker` — SPEC ROW, kept
//!     strict on purpose: the L5 worktree build never calls the adapter's
//!     version probe (`probed_version`/`refresh_schema` have no caller), so
//!     `adapter_detail.schema.version` is permanently `"unknown"` and a v1
//!     engine's documented `--flash-attn` is rejected. Reported as a build
//!     gap; do not weaken this row.
//! (6) Engines tab + `/api/engines` + events additive + floors:
//!   * `l5_engines_tab_columns_events_additive_and_v1_floor`
//!   * `l5_cli_surface_override_force_gate_takeover_adopt`
//!
//! GAPS THIS FILE MUST DECLARE (no black-box surface per spec text):
//! - **`swap takeover` single-action audit identity**: the ledger entry carries
//!   `ts`/`kind`/`model`/`reason`; there is no actor field anywhere in the
//!   swap surface, so "audit-logged (who, when, target)" is asserted as
//!   when+target+operation only. Reported, not invented.
//! - **"Explicit operator action required before swapping onto a device with a
//!   live external server"**: the spec names no wire reason for the refusal
//!   (unlike `blocked:active-consumer`), so this suite asserts the *binding*
//!   safety invariant instead (SIGTERM only ever targets supervisor-owned
//!   marker pids; adopting transfers ownership and then the pid becomes
//!   signalable). Whether `ensure` must refuse to co-reside with an
//!   unregistered external before adoption is reported, not asserted.
//! - **`swap_catalog.json` port allocation vs llama-swap dynamic range**: the
//!   static range check (`avoidance_from_catalog`) is unit-tier; the fixture
//!   asserts the observable part (allocated port inside the profile range and
//!   written into the conf, uniqueness across two models).
//! - **Queue cap / cancel** are L4 rows and are not duplicated here.
//! - **Container source** has no wire seed surface (L4 row, unchanged).
//!
//! SERIAL GATE (L4/L3 pattern): every test owns loopback ports, a live daemon
//! and scripted child processes; parallel libtest threads race in the
//! reserve-then-bind windows. Each test holds one process-wide async mutex for
//! its lifetime. Opt out with `RAY_L5_NO_SERIAL_GATE=1` on an idle machine.

use std::fs;
use std::future::Future;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Child, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

// ── Serial gate ─────────────────────────────────────────────────────────────

static SUITE_GATE: std::sync::LazyLock<tokio::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));

async fn gate() -> Option<tokio::sync::MutexGuard<'static, ()>> {
    if std::env::var("RAY_L5_NO_SERIAL_GATE").is_ok() {
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

struct TestEnv {
    root: PathBuf,
    home: PathBuf,
    config_home: PathBuf,
    data_home: PathBuf,
    state_home: PathBuf,
    runtime: PathBuf,
    bin: PathBuf,
}

impl TestEnv {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!("brainrouter-l5-{tag}-{}", unique_suffix()));
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
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&env.runtime, fs::Permissions::from_mode(0o700)).unwrap();
        fs::create_dir_all(env.state_dir()).unwrap();
        fs::create_dir_all(env.config_dir()).unwrap();
        env
    }

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

    fn adoption_dir(&self) -> PathBuf {
        self.state_dir().join("adoption")
    }

    fn conf_dir(&self) -> PathBuf {
        self.root.join("strixite-confs")
    }
}

impl Drop for TestEnv {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

/// Kills every scripted engine child whose argv names a path under the test
/// root (supervisor children are setsid'd and outlive a SIGKILLed daemon).
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

/// A test-spawned mock's parent is this test process, so a signalled child is
/// a zombie (still `kill(pid, 0)`-alive) until reaped: read the state from
/// `/proc/<pid>/stat` for the "is it really gone" assertions.
fn pid_running(pid: u32) -> bool {
    if !pid_alive(pid) {
        return false;
    }
    let Ok(stat) = fs::read_to_string(format!("/proc/{pid}/stat")) else {
        return false;
    };
    match stat.rfind(')') {
        Some(close) => stat[close + 1..].trim_start().chars().next() != Some('Z'),
        None => true,
    }
}

fn cmdline_of(pid: u32) -> String {
    fs::read(format!("/proc/{pid}/cmdline"))
        .map(|raw| String::from_utf8_lossy(&raw).replace('\0', " "))
        .unwrap_or_default()
}

/// L5 spec, start-ownership: the child-held GGUF flock must be released when
/// the failed/hung child is killed (no leaked holder).
fn flock_is_free(path: &Path) -> bool {
    use std::os::unix::io::AsRawFd;
    let file = fs::File::open(path).expect("open gguf for flock probe");
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    rc == 0
}

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

fn port_bindable(port: u16) -> bool {
    TcpListener::bind(("127.0.0.1", port)).is_ok()
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

// ── Scripted mock engine (strix_server / hipfire) ──────────────────────────

/// One Python HTTP server, copied to `<bin>/strix_server` and `<bin>/hipfire`
/// so the external observer's pid-scoped classification sees a real process.
/// Knobs (all via the catalog entry's `launch_spec.env`):
/// `MOCK_STEM` / `MOCK_STEMS`, `MOCK_READY_DELAY_MS`, `MOCK_HOLD_MS`,
/// `MOCK_SLOTS_FRACTION`, `MOCK_VERDICT_JSON`, `MOCK_VERSION`,
/// `MOCK_IGNORE_TERM`, `MOCK_CRASH_AFTER_MS`.
const MOCK_ENGINE_PY: &str = r#"#!/usr/bin/env python3
import json, os, signal, sys, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = sys.argv[1:]

def flag(name):
    for i, a in enumerate(ARGS):
        if a == name and i + 1 < len(ARGS):
            return ARGS[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None

if "--version" in ARGS:
    print(os.environ.get("MOCK_VERSION", "hipfire 1.4.2"))
    sys.exit(0)
if "--print-config" in ARGS:
    sys.exit(0)

def env(name, default=None):
    return os.environ.get(name, default)

port = flag("--port")
conf = flag("--config")
if not port and conf:
    try:
        with open(conf) as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("port") and "=" in s:
                    port = s.split("=", 1)[1].strip()
    except OSError:
        pass
port = int(port or env("BRAINROUTER_SWAP_PORT") or env("PORT") or "8080")

stems = [s for s in (env("MOCK_STEMS") or env("MOCK_STEM") or "mock").split(",") if s]
ready_delay = int(env("MOCK_READY_DELAY_MS", "0") or 0) / 1000.0
hold = int(env("MOCK_HOLD_MS", "0") or 0) / 1000.0
slots = env("MOCK_SLOTS_FRACTION")
verdict = env("MOCK_VERDICT_JSON")
if env("MOCK_IGNORE_TERM") == "1":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
crash_after = env("MOCK_CRASH_AFTER_MS")
if crash_after:
    def _crash():
        time.sleep(int(crash_after) / 1000.0)
        os._exit(3)
    threading.Thread(target=_crash, daemon=True).start()
ready_at = time.time() + ready_delay

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a):
        pass
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass
    def do_GET(self):
        p = self.path.split("?")[0]
        if p.startswith("/v1/models"):
            if time.time() < ready_at:
                return self._json(503, {"error": {"message": "loading"}})
            return self._json(200, {"object": "list",
                                    "data": [{"id": s, "object": "model"} for s in stems]})
        if p.startswith("/slots"):
            if slots is None:
                return self._json(404, {"error": {"message": "unknown endpoint"}})
            f = float(slots)
            return self._json(200, [{"is_processing": True, "n_prompt_tokens": 100,
                                     "n_prompt_processed": f * 100,
                                     "next_token": {"n_decoded": 0}}])
        if p.startswith("/health"):
            return self._json(200, {"status": "ok"})
        return self._json(404, {"error": {"message": "unknown endpoint " + p}})
    def do_POST(self):
        length = int(self.headers.get("content-length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except Exception:
            body = {}
        p = self.path.split("?")[0]
        if not (p.startswith("/v1/chat/completions") or p.startswith("/v1/completions")):
            return self._json(404, {"error": {"message": "unknown endpoint " + p}})
        model = str(body.get("model", "mock"))
        content = verdict if verdict else ("mock-pong " + model)
        if not body.get("stream"):
            if hold:
                time.sleep(hold)
            return self._json(200, {"id": "chatcmpl-mock", "object": "chat.completion",
                                    "model": model,
                                    "choices": [{"index": 0,
                                                 "message": {"role": "assistant", "content": content},
                                                 "finish_reason": "stop"}]})
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        def chunk(delta):
            payload = {"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                       "created": 1, "model": model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            return ("data: " + json.dumps(payload) + "\n\n").encode()
        try:
            self.wfile.write(chunk({"role": "assistant", "content": ""}))
            self.wfile.flush()
            if hold:
                time.sleep(hold)
            self.wfile.write(chunk({"content": content}))
            self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception:
            pass
        self.close_connection = True

srv = ThreadingHTTPServer(("127.0.0.1", port), H)
srv.daemon_threads = True
srv.serve_forever()
"#;

fn write_executable(path: &Path, body: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, body).unwrap();
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

/// Install the mock engine twice (both observer-recognized names) and return
/// the strixite program path.
fn install_mock_engines(env: &TestEnv) -> PathBuf {
    let strix = env.bin.join("strix_server");
    write_executable(&strix, MOCK_ENGINE_PY);
    write_executable(&env.bin.join("hipfire"), MOCK_ENGINE_PY);
    strix
}

/// A test-launched external engine (icon-era shape): argv names the engine and
/// carries `--port`, so the read-only `/proc` observer sees it.
struct ExternalEngine {
    child: Child,
    pid: u32,
    url: String,
}

impl ExternalEngine {
    fn spawn(env: &TestEnv, program: &str, port: u16, weights: &Path, extra_env: &[(&str, &str)]) -> Self {
        let program_path = env.bin.join(program);
        let mut cmd = Command::new(&program_path);
        cmd.arg("--port")
            .arg(port.to_string())
            .arg("--weights")
            .arg(weights)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        for (key, value) in extra_env {
            cmd.env(key, value);
        }
        let child = cmd.spawn().expect("spawn external mock engine");
        let pid = child.id();
        let url = format!("http://127.0.0.1:{port}");
        Self { child, pid, url }
    }

    fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for ExternalEngine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

async fn wait_mock_ready(engine: &ExternalEngine) {
    poll_until(&format!("mock engine {}", engine.url), 10, || async {
        let ok = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client")
            .get(format!("{}/v1/models", engine.url))
            .timeout(Duration::from_millis(500))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false);
        ok.then_some(())
    })
    .await;
}

// ── Scripted systemctl (takeover unit fixture) ─────────────────────────────

/// `BRAINROUTER_SYSTEMCTL` wrapper: `show` answers from
/// `$BRAINROUTER_MOCK_UNIT_DIR/<unit>.{active,pid}`; `stop` appends to
/// `stops.log` and flips the unit inactive (the operator's system unit).
const SYSTEMCTL_SHIM: &str = r#"#!/bin/bash
set -u
dir="${BRAINROUTER_MOCK_UNIT_DIR:?}"
args=()
for a in "$@"; do
  if [ "$a" != "--user" ]; then args+=("$a"); fi
done
cmd="${args[0]:-}"
unit="${args[1]:-}"
case "$cmd" in
  show)
    active="inactive"; pid="0"
    [ -f "$dir/$unit.active" ] && active="$(cat "$dir/$unit.active")"
    [ -f "$dir/$unit.pid" ] && pid="$(cat "$dir/$unit.pid")"
    echo "ActiveState=$active"
    echo "MainPID=$pid"
    echo "ControlGroup="
    ;;
  stop)
    echo "$unit" >> "$dir/stops.log"
    echo "inactive" > "$dir/$unit.active"
    ;;
  *)
    exit 0
    ;;
esac
"#;

fn write_systemctl_shim(env: &TestEnv) -> (PathBuf, PathBuf) {
    let dir = env.root.join("units");
    fs::create_dir_all(&dir).unwrap();
    let path = env.bin.join("mock-systemctl");
    write_executable(&path, SYSTEMCTL_SHIM);
    (path, dir)
}

fn set_unit(dir: &Path, unit: &str, active: bool, pid: u32) {
    fs::write(
        dir.join(format!("{unit}.active")),
        if active { "active" } else { "inactive" },
    )
    .unwrap();
    fs::write(dir.join(format!("{unit}.pid")), pid.to_string()).unwrap();
}

fn unit_stops(dir: &Path) -> Vec<String> {
    fs::read_to_string(dir.join("stops.log"))
        .unwrap_or_default()
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

// ── Mock llama-swap upstream (v1 floor) ────────────────────────────────────

#[derive(Clone, Default)]
struct ShimCfg {
    models: Vec<String>,
    body_delay_ms: u64,
}

struct Shim {
    url: String,
    hits: Arc<Mutex<Vec<Value>>>,
}

impl Shim {
    fn chat_bodies(&self) -> Vec<Value> {
        self.hits.lock().unwrap().clone()
    }
}

fn spawn_shim(cfg: ShimCfg) -> Shim {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    listener.set_nonblocking(true).unwrap();
    let models = Arc::new(cfg.models.clone());
    let hits = Arc::new(Mutex::new(Vec::new()));
    let (m2, h2, cfg2) = (models.clone(), hits.clone(), Arc::new(cfg));
    tokio::spawn(async move {
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();
        loop {
            let Ok((mut sock, _)) = listener.accept().await else { continue };
            let (m, h, cfg) = (m2.clone(), h2.clone(), cfg2.clone());
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
                let body: Value = serde_json::from_slice(&buf[body_start..filled.max(body_start)])
                    .unwrap_or(Value::Null);
                if path.starts_with("/running") {
                    let _ = write_plain(&mut sock, 200, "application/json", json!({"running": []}).to_string()).await;
                    return;
                }
                if path.contains("/models") {
                    let data: Vec<Value> = m
                        .iter()
                        .map(|id| json!({"id": id, "object": "model", "owned_by": "ray-l5-mock"}))
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
                    let model = body.get("model").and_then(Value::as_str).unwrap_or("").to_string();
                    h.lock().unwrap().push(body.clone());
                    let chunk = |delta: Value, finish: Value| {
                        format!(
                            "data: {}\n\n",
                            json!({"id":"chatcmpl-ray-l5","object":"chat.completion.chunk",
                                   "created":1,"model":model.clone(),
                                   "choices":[{"index":0,"delta":delta,"finish_reason":finish}]})
                        )
                    };
                    let sse = chunk(json!({"role":"assistant","content":""}), Value::Null)
                        + &chunk(json!({"content": format!("ray-l5-pong {model}")}), json!("stop"))
                        + "data: [DONE]\n\n";
                    let head = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                        sse.len()
                    );
                    let _ = sock.write_all(head.as_bytes()).await;
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
    Shim { url: format!("http://127.0.0.1:{port}"), hits }
}

fn find_headers_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4)
}

async fn write_plain<S>(sock: &mut S, status: u16, ctype: &str, body: String) -> std::io::Result<()>
where
    S: tokio::io::AsyncWrite + Unpin,
{
    use tokio::io::AsyncWriteExt;
    sock.write_all(
        format!(
            "HTTP/1.1 {status} OK\r\ncontent-type: {ctype}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        )
        .as_bytes(),
    )
    .await
}

// ── Daemon harness ─────────────────────────────────────────────────────────

struct Daemon {
    child: Child,
    url: String,
    log: PathBuf,
    client: reqwest::Client,
}

impl Daemon {
    fn start(env: &TestEnv, config_yaml: &str, extra: &[(String, String)]) -> Self {
        fs::write(env.root.join("config.yaml"), config_yaml).unwrap();
        let config_path = env.root.join("config.yaml");
        let socket = env.root.join("d.sock");
        let log = env.root.join("daemon.log");
        let mut attempt = 0u8;
        let (address, child) = loop {
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
        };
        Self {
            child,
            url: format!("http://{address}"),
            log,
            client: reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(120))
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
        let tail: Vec<&str> = text.lines().rev().take(60).collect();
        tail.into_iter().rev().collect::<Vec<_>>().join("\n")
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}{path}", self.url)
    }

    async fn get_json(&self, path: &str) -> Value {
        let (status, text) = self.request("GET", path, None, &[]).await;
        assert!(
            status < 400,
            "L5 surface missing: GET {path} -> {status}: {text}\n\
             (L5 spec: external observers / takeover / ACS / Engines tab API)"
        );
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("GET {path} not json ({e}): {text}"))
    }

    async fn post_json(&self, path: &str, body: Value) -> (u16, String) {
        self.request("POST", path, Some(body), &[]).await
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
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ── CLI helper ──────────────────────────────────────────────────────────────

#[derive(Debug)]
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
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_brainrouter"));
    cmd.args(args).current_dir(&env.root).stdin(Stdio::null());
    env.apply(&mut cmd, &[]);
    let out = cmd.output().expect("run brainrouter CLI");
    CliRun {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

// ── Config + catalog fixtures ──────────────────────────────────────────────

fn base_config(shim_base: &str) -> String {
    serde_yaml::to_string(&json!({
        "manifest": { "enabled": false, "base_url": format!("{shim_base}/cloud/v1") },
        "llama_swap": {
            "base_url": format!("{shim_base}/v1"),
            "fallback_model": "ray-l5-v1-model",
            "local_models": ["ray-l5-v1-model"],
        },
        "bonsai": { "enabled": false, "fork_path": "/nonexistent/ray-l5-no-bonsai" },
        "review": {},
    }))
    .unwrap()
}

fn gguf(env: &TestEnv, name: &str) -> PathBuf {
    let path = env.root.join("models").join(name);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    if !path.exists() {
        fs::write(&path, b"GGUF-ray-l5-fixture\n").unwrap();
    }
    path
}

fn catalog_entry(
    id: &str,
    engine: &str,
    files: &[&Path],
    estimate_gib: f64,
    args: &[&str],
    env_pairs: &[(&str, &str)],
) -> Value {
    let env_map: serde_json::Map<String, Value> = env_pairs
        .iter()
        .map(|(k, v)| (k.to_string(), json!(v)))
        .collect();
    json!({
        "id": id,
        "engine": engine,
        "files": files.iter().map(|p| json!(p.display().to_string())).collect::<Vec<_>>(),
        "ctx": 8192,
        "quant": Value::Null,
        "family": Value::Null,
        "tools": false,
        "vram_estimate_gib": estimate_gib,
        "launch_spec": {
            "program": "",
            "args": args,
            "env": env_map,
            "port_policy": "managed",
            "served_model_alias": id,
        },
        "aliases": [],
        "groups": [],
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
    assert_eq!(
        status, 200,
        "catalog add `{id}` -> {status}: {text}\n\
         (L5 spec: adapters are trait impls + registry data)"
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
    poll_until(&format!("model {id} state={state}"), 30, || async {
        let e = engines(d).await;
        match model_status(&e, id) {
            Some(m) if m["state"] == json!(state) => Some(()),
            _ => None,
        }
    })
    .await;
}

async fn ensure_with(d: &Daemon, id: &str, extra: Value) -> (u16, Value) {
    let mut body = json!({ "model": id });
    if let (Some(obj), Some(extras)) = (body.as_object_mut(), extra.as_object()) {
        for (k, v) in extras {
            obj.insert(k.clone(), v.clone());
        }
    }
    let (status, text) = d.post_json("/api/engines/ensure", body).await;
    let parsed = serde_json::from_str::<Value>(&text).unwrap_or(Value::Null);
    (status, parsed)
}

async fn ensure_ok(d: &Daemon, id: &str) -> Value {
    let (status, body) = ensure_with(d, id, json!({})).await;
    assert_eq!(status, 200, "ensure `{id}` -> {status}: {body}");
    body
}

/// The budget ledger doubles as the L5 audit/event surface
/// (`GET /api/engines/events`, spec: `blocked:active-consumer` + `--override`
/// audit). Returns entries oldest → newest.
async fn swap_events(d: &Daemon) -> Vec<Value> {
    let v = d.get_json("/api/engines/events").await;
    v["events"].as_array().cloned().unwrap_or_default()
}

async fn events_for(d: &Daemon, model: &str) -> Vec<Value> {
    swap_events(d)
        .await
        .into_iter()
        .filter(|e| e["model"] == json!(model))
        .collect()
}

fn kinds(events: &[Value]) -> Vec<String> {
    events
        .iter()
        .filter_map(|e| e["kind"].as_str().map(String::from))
        .collect()
}

fn event_index(events: &[Value], kind: &str, model: &str) -> Option<usize> {
    events
        .iter()
        .position(|e| e["kind"] == json!(kind) && e["model"] == json!(model))
}

async fn consumers(d: &Daemon) -> Vec<Value> {
    let v = d.get_json("/api/engines/consumers").await;
    v["active_consumers"].as_array().cloned().unwrap_or_default()
}

async fn external_rows(d: &Daemon) -> Vec<Value> {
    let v = d.get_json("/api/engines/external").await;
    v["external"].as_array().cloned().unwrap_or_default()
}

// ═══════════════════════════════════════════════════════════════════════════
// (1) strixite adapter — spawn, conf-owned port, slots progress, stem, stop
// ═══════════════════════════════════════════════════════════════════════════

/// L5 spec §strixite adapter: "launching `strix_server --config <conf>
/// --weights <weights>` takes the port FROM THE CONF — per-model ports =
/// per-model GENERATED confs"; "Load progress: readiness poll only ⇒
/// indeterminate + EWMA ETA"; stem assertion; `Waiting(gate)` is the memory
/// floor row (separate test).
#[tokio::test(flavor = "multi_thread")]
async fn l5_strixite_adapter_spawn_conf_port_progress_stem_and_stop() {
    let _gate = gate().await;
    let env = TestEnv::new("strixite-spawn");
    let _guard = ChildGuard::new(&env.root);
    let program = install_mock_engines(&env);
    let shim = spawn_shim(ShimCfg::default());
    let port_min = free_port();
    let port_max = port_min.saturating_add(1);
    let program_s = program.display().to_string();
    let conf_dir = env.conf_dir();
    let conf_dir_s = conf_dir.display().to_string();
    let range = format!("{port_min}-{port_max}");
    let extra = vec![
        ("BRAINROUTER_FLEET_ENGINES".to_string(), "strixite".to_string()),
        ("BRAINROUTER_STRIXITE_BIN".to_string(), program_s),
        ("BRAINROUTER_STRIXITE_CONF_DIR".to_string(), conf_dir_s),
        ("BRAINROUTER_STRIXITE_PORT_RANGE".to_string(), range),
        ("BRAINROUTER_STRIXITE_MIN_MEM_GIB".to_string(), "0".to_string()),
        ("BRAINROUTER_READINESS_TIMEOUT_MS".to_string(), "20000".to_string()),
    ];
    let mut d = Daemon::start(&env, &base_config(&shim.url), &extra);
    d.ready().await;

    let file = gguf(&env, "strix-a.gguf");
    let id = "ray-l5-strix-a";
    add_entry(
        &d,
        catalog_entry(
            id,
            "strixite",
            &[&file],
            6.0,
            &[],
            &[
                ("MOCK_STEM", id),
                ("MOCK_READY_DELAY_MS", "2500"),
                ("MOCK_SLOTS_FRACTION", "0.5"),
            ],
        ),
    )
    .await;

    // Ensure concurrently so the `loading` row (and its adapter progress) is
    // observable — the spec's "slots-equivalent" status fields.
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(60))
        .build()
        .unwrap();
    let url = d.url.clone();
    let ensure_task = tokio::spawn(async move {
        let r = client
            .post(format!("{url}/api/engines/ensure"))
            .json(&json!({ "model": "ray-l5-strix-a" }))
            .send()
            .await
            .expect("ensure request");
        (r.status().as_u16(), r.text().await.unwrap_or_default())
    });

    let loading = poll_until(
        "strixite loading row with `/slots`-equivalent progress",
        20,
        || async {
            let e = engines(&d).await;
            let m = model_status(&e, id)?;
            if m["state"] != json!("loading") {
                return None;
            }
            if m["progress"]["fraction"].as_f64() != Some(0.5) {
                return None;
            }
            Some(m.clone())
        },
    )
    .await;
    let progress = &loading["progress"];
    assert!(
        progress["elapsed_ms"].is_number(),
        "L5 spec: loading progress carries elapsed_ms: {loading}"
    );
    assert_eq!(
        progress["fraction"].as_f64(),
        Some(0.5),
        "L5 spec strixite: `/slots`-equivalent progress is surfaced while loading \
         (mock /slots reports n_prompt_processed/n_prompt_tokens = 0.5): {loading}"
    );

    let port = loading["port"].as_u64().expect("allocated port in status") as u16;
    assert!(
        (port_min..=port_max).contains(&port),
        "L5 spec: supervisor-allocated port inside the registry-managed range \
         ({port_min}-{port_max}), got {port}"
    );
    let conf_path = PathBuf::from(
        loading["conf_path"]
            .as_str()
            .expect("generated conf path in status"),
    );
    assert!(
        conf_path.starts_with(&conf_dir),
        "L5 spec: per-model GENERATED conf lives in the profile conf dir: {conf_path:?}"
    );

    let (status, body) = ensure_task.await.unwrap();
    assert_eq!(status, 200, "ensure {id} -> {status}: {body}");
    let outcome: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(outcome["spawned"], json!(true), "{outcome}");
    assert_eq!(
        outcome["endpoint"],
        json!(format!("http://127.0.0.1:{port}")),
        "the endpoint's port is the one written into the conf: {outcome}"
    );

    // The conf is the engine's source of truth for the port (spec binding).
    let conf = fs::read_to_string(&conf_path).expect("generated conf exists");
    assert!(
        conf.contains(&format!("port = {port}"))
            || conf.contains(&format!("port={port}")),
        "L5 spec: the port must live in the generated conf: {conf}"
    );
    assert!(
        conf.contains(file.file_name().unwrap().to_str().unwrap()),
        "L5 spec: the generated conf names the weights: {conf}"
    );

    wait_model_state(&d, id, "ready").await;
    let ready = engines(&d).await;
    let row = model_status(&ready, id).expect("ready row");
    assert_eq!(row["engine"], json!("strixite"), "{row}");
    assert_eq!(row["source"], json!("native"), "{row}");

    // `--config <conf> --weights <weights>` spawned with the mock program.
    let pid = pids_with_arg(conf_path.to_str().unwrap());
    assert_eq!(
        pid.len(),
        1,
        "L5 spec: one `strix_server --config <conf> --weights <weights>` child per model: {pid:?}"
    );
    let argv = cmdline_of(pid[0]);
    assert!(
        argv.contains("--config") && argv.contains("--weights"),
        "L5 spec strixite argv: {argv}"
    );

    // Adapter detail badge (Engines tab) exists with the profile data.
    let detail = ready["adapter_detail"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let strix_detail = detail
        .iter()
        .find(|a| a["engine"] == json!("strixite"))
        .unwrap_or_else(|| panic!("L5 spec: Engines tab adapter detail for strixite: {ready}"));
    assert_eq!(strix_detail["conf_dir"], json!(conf_dir.display().to_string()));

    // Stop: SIGTERM the owned child, release the flock, keep the conf.
    let (status, text) = d
        .post_json("/api/engines/evict", json!({"model": id}))
        .await;
    assert_eq!(status, 200, "evict {id} -> {status}: {text}");
    wait_model_state(&d, id, "stopped").await;
    assert!(!pid_alive(pid[0]), "evict must stop the supervised child");
    assert!(
        flock_is_free(&file),
        "L5/L4 binding: the child-held flock is released when the child exits"
    );
    assert!(conf_path.exists(), "the generated conf is retained across stops");
}

// ═══════════════════════════════════════════════════════════════════════════
// (1) strixite — stem assertion, engine-validated flags, hung-start recovery
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn l5_strixite_stem_assert_conf_owned_flags_and_hung_start_kill9() {
    let _gate = gate().await;
    let env = TestEnv::new("strixite-negatives");
    let _guard = ChildGuard::new(&env.root);
    let program = install_mock_engines(&env);
    let shim = spawn_shim(ShimCfg::default());
    let port_min = free_port();
    let range = format!("{port_min}-{}", port_min.saturating_add(2));
    let extra = vec![
        ("BRAINROUTER_FLEET_ENGINES".to_string(), "strixite".to_string()),
        ("BRAINROUTER_STRIXITE_BIN".to_string(), program.display().to_string()),
        ("BRAINROUTER_STRIXITE_CONF_DIR".to_string(), env.conf_dir().display().to_string()),
        ("BRAINROUTER_STRIXITE_PORT_RANGE".to_string(), range),
        ("BRAINROUTER_STRIXITE_MIN_MEM_GIB".to_string(), "0".to_string()),
        ("BRAINROUTER_SWAP_STOP_GRACE_MS".to_string(), "400".to_string()),
        // Generous enough for a cold Python start under load (otherwise a slow
        // first probe turns the stem-assertion case into a readiness timeout);
        // the hung-start case never becomes ready at all.
        ("BRAINROUTER_READINESS_TIMEOUT_MS".to_string(), "5000".to_string()),
    ];
    let mut d = Daemon::start(&env, &base_config(&shim.url), &extra);
    d.ready().await;

    // (a) Stem assertion: the engine answers `/v1/models` with a foreign stem.
    let wrong = gguf(&env, "strix-wrong.gguf");
    let wrong_id = "ray-l5-strix-wrong";
    add_entry(
        &d,
        catalog_entry(
            wrong_id,
            "strixite",
            &[&wrong],
            1.0,
            &[],
            &[("MOCK_STEM", "someone-elses-model")],
        ),
    )
    .await;
    let (status, body) = ensure_with(&d, wrong_id, json!({})).await;
    assert_eq!(
        status, 503,
        "L5 spec: stem assertion failure is a spawn failure (503): {status} {body}"
    );
    let text = body["error"].as_str().unwrap_or("");
    assert!(
        text.contains("stem assertion failed"),
        "L5 spec: the failure must name the stem assertion, got: {text}"
    );
    let e = engines(&d).await;
    let row = model_status(&e, wrong_id).expect("failed row");
    assert_eq!(row["state"], json!("failed"), "{row}");
    assert!(
        row["last_error"].as_str().unwrap_or("").contains("stem assertion failed"),
        "{row}"
    );
    assert!(
        flock_is_free(&wrong),
        "L5/L4 binding: a failed start releases the child-held flock"
    );

    // (b) Engine-validated launch args: the generated conf owns the port, and
    // no catalog spec may smuggle its own `--port`.
    let flags = gguf(&env, "strix-flags.gguf");
    let flags_id = "ray-l5-strix-flags";
    add_entry(
        &d,
        catalog_entry(
            flags_id,
            "strixite",
            &[&flags],
            1.0,
            &["--port", "5999"],
            &[("MOCK_STEM", flags_id)],
        ),
    )
    .await;
    let (status, body) = ensure_with(&d, flags_id, json!({})).await;
    assert_eq!(
        status, 503,
        "L5 spec: a conf-owned flag in the launch spec is refused: {status} {body}"
    );
    assert!(
        body["error"]
            .as_str()
            .unwrap_or("")
            .contains("the generated conf owns it"),
        "L5 spec: refuse rather than forward a second --port: {body}"
    );
    assert!(
        !kinds(&events_for(&d, flags_id).await).contains(&"spawn".to_string()),
        "L5 spec: a refused spec never spawns: {:?}",
        events_for(&d, flags_id).await
    );

    // (c) Hung start: the child ignores SIGTERM and never becomes ready ⇒
    // SIGTERM, escalation window, kill -9, flock released, state Failed/timeout.
    let hung = gguf(&env, "strix-hung.gguf");
    let hung_id = "ray-l5-strix-hung";
    add_entry(
        &d,
        catalog_entry(
            hung_id,
            "strixite",
            &[&hung],
            1.0,
            &[],
            &[
                ("MOCK_STEM", hung_id),
                ("MOCK_IGNORE_TERM", "1"),
                ("MOCK_READY_DELAY_MS", "600000"),
            ],
        ),
    )
    .await;
    let (status, body) = ensure_with(&d, hung_id, json!({})).await;
    assert_eq!(
        status, 504,
        "L5 spec: a hung start fails `ensure` after the readiness timeout: {status} {body}"
    );
    let leftover = pids_with_arg(hung.to_str().unwrap());
    assert!(
        leftover.is_empty(),
        "L5 spec hung-start: SIGTERM ⇒ kill -9 after the escalation window; \
         the SIGTERM-ignoring child must be gone: {leftover:?} {:?}",
        leftover.iter().map(|p| cmdline_of(*p)).collect::<Vec<_>>()
    );
    assert!(
        flock_is_free(&hung),
        "L5 spec hung-start: kill -9 releases the flock before the gate re-run"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// (2) Exclusivity + serialized swap lock
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn l5_exclusive_cutover_evict_before_spawn_single_resident_and_one_spawn() {
    let _gate = gate().await;
    let env = TestEnv::new("exclusive");
    let _guard = ChildGuard::new(&env.root);
    let program = install_mock_engines(&env);
    let shim = spawn_shim(ShimCfg::default());
    let port_min = free_port();
    let range = format!("{port_min}-{}", port_min.saturating_add(1));
    let extra = vec![
        ("BRAINROUTER_FLEET_ENGINES".to_string(), "strixite".to_string()),
        ("BRAINROUTER_STRIXITE_BIN".to_string(), program.display().to_string()),
        ("BRAINROUTER_STRIXITE_CONF_DIR".to_string(), env.conf_dir().display().to_string()),
        ("BRAINROUTER_STRIXITE_PORT_RANGE".to_string(), range),
        ("BRAINROUTER_STRIXITE_MIN_MEM_GIB".to_string(), "0".to_string()),
        ("BRAINROUTER_CUTOVER_SETTLE_MS".to_string(), "0".to_string()),
        ("BRAINROUTER_CUTOVER_RECHECK_MS".to_string(), "50".to_string()),
    ];
    let mut d = Daemon::start(&env, &base_config(&shim.url), &extra);
    d.ready().await;

    let a_file = gguf(&env, "excl-a.gguf");
    let b_file = gguf(&env, "excl-b.gguf");
    let a = "ray-l5-excl-a";
    let b = "ray-l5-excl-b";
    for (id, file) in [(a, &a_file), (b, &b_file)] {
        add_entry(
            &d,
            catalog_entry(id, "strixite", &[file], 4.0, &[], &[("MOCK_STEM", id)]),
        )
        .await;
    }

    ensure_ok(&d, a).await;
    wait_model_state(&d, a, "ready").await;
    let a_pid = pids_with_arg(a_file.to_str().unwrap());
    assert_eq!(a_pid.len(), 1, "one supervised child for A: {a_pid:?}");

    // ensure(second model) = evict-then-spawn swap, never co-resident.
    let outcome = ensure_ok(&d, b).await;
    assert_eq!(outcome["spawned"], json!(true), "{outcome}");
    assert!(
        !pid_alive(a_pid[0]),
        "L5 spec: one resident per device (`exclusive: true`) — A drained before B spawned"
    );
    wait_model_state(&d, b, "ready").await;

    let events = swap_events(&d).await;
    let evict_a = event_index(&events, "evict", a).expect("A eviction event");
    let spawn_b = event_index(&events, "spawn", b).expect("B spawn event");
    assert!(
        evict_a < spawn_b,
        "L5 spec cutover order: SIGTERM → pid gone → settle → gate → spawn \
         (evict of A must land before the spawn of B): {events:?}"
    );

    // Never co-resident: at most one native strixite resident at a time.
    let status = engines(&d).await;
    let ready_native: Vec<&Value> = status["models"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["engine"] == json!("strixite") && m["state"] == json!("ready"))
        .collect();
    assert_eq!(
        ready_native.len(),
        1,
        "L5 spec: exactly one ready strixite resident: {ready_native:?}"
    );

    // Serialized per-device swap lock: concurrent ensure(X) ⇒ exactly one spawn.
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(60))
        .build()
        .unwrap();
    let mut handles = Vec::new();
    for _ in 0..2 {
        let client = client.clone();
        let url = d.url.clone();
        handles.push(tokio::spawn(async move {
            let r = client
                .post(format!("{url}/api/engines/ensure"))
                .json(&json!({ "model": "ray-l5-excl-a" }))
                .send()
                .await
                .expect("concurrent ensure");
            let status = r.status().as_u16();
            let body: Value = r.json().await.unwrap_or(Value::Null);
            (status, body)
        }));
    }
    let mut spawned = 0;
    for handle in handles {
        let (status, body) = handle.await.unwrap();
        assert_eq!(status, 200, "concurrent ensure -> {status}: {body}");
        if body["spawned"] == json!(true) {
            spawned += 1;
        }
    }
    assert_eq!(
        spawned, 1,
        "L5 spec: serialized per-device swap lock — exactly one of two concurrent \
         ensures of the same target spawns; the other attaches"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// (3) ACS — proxy-stream pin, evict skip, override, grace span
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn l5_acs_proxy_stream_pin_evict_skip_override_and_grace() {
    let _gate = gate().await;
    let env = TestEnv::new("acs-proxy");
    let _guard = ChildGuard::new(&env.root);
    let program = install_mock_engines(&env);
    let shim = spawn_shim(ShimCfg::default());
    let port_min = free_port();
    let range = format!("{port_min}-{}", port_min.saturating_add(2));
    let extra = vec![
        ("BRAINROUTER_FLEET_ENGINES".to_string(), "strixite".to_string()),
        ("BRAINROUTER_STRIXITE_BIN".to_string(), program.display().to_string()),
        ("BRAINROUTER_STRIXITE_CONF_DIR".to_string(), env.conf_dir().display().to_string()),
        ("BRAINROUTER_STRIXITE_PORT_RANGE".to_string(), range),
        ("BRAINROUTER_STRIXITE_MIN_MEM_GIB".to_string(), "0".to_string()),
        ("BRAINROUTER_CUTOVER_SETTLE_MS".to_string(), "0".to_string()),
        ("BRAINROUTER_CUTOVER_RECHECK_MS".to_string(), "100".to_string()),
        // Idle grace (`active_consumer_grace_s`, fixture form in ms).
        ("BRAINROUTER_ACTIVE_CONSUMER_GRACE_MS".to_string(), "2500".to_string()),
    ];
    let mut d = Daemon::start(&env, &base_config(&shim.url), &extra);
    d.ready().await;

    let a_file = gguf(&env, "acs-a.gguf");
    let b_file = gguf(&env, "acs-b.gguf");
    let a = "ray-l5-acs-a";
    let b = "ray-l5-acs-b";
    add_entry(
        &d,
        catalog_entry(
            a,
            "strixite",
            &[&a_file],
            4.0,
            &[],
            &[("MOCK_STEM", a), ("MOCK_HOLD_MS", "6000")],
        ),
    )
    .await;
    add_entry(
        &d,
        catalog_entry(b, "strixite", &[&b_file], 4.0, &[], &[("MOCK_STEM", b)]),
    )
    .await;

    ensure_ok(&d, a).await;
    wait_model_state(&d, a, "ready").await;
    let a_pid = pids_with_arg(a_file.to_str().unwrap());
    assert_eq!(a_pid.len(), 1);

    let url = d.url.clone();
    let stream_task = tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap();
        let resp = client
            .post(format!("{url}/v1/chat/completions"))
            .json(&json!({
                "model": "ray-l5-acs-a",
                "messages": [{"role": "user", "content": "acs proxy pin"}],
                "stream": true,
            }))
            .send()
            .await
            .expect("acs stream");
        assert_eq!(resp.status().as_u16(), 200);
        resp.text().await.unwrap_or_default()
    });

    // Live in-flight row (L4's authorized additive registration) + the ACS span.
    poll_until("in-flight row for the native stream", 20, || async {
        let v = d.get_json("/api/inflight").await;
        let hit = v["requests"]
            .as_array()?
            .iter()
            .any(|r| r["catalog_id"] == json!(a));
        hit.then_some(())
    })
    .await;

    let span = poll_until("ACS span for the live stream", 20, || async {
        consumers(&d)
            .await
            .into_iter()
            .find(|c| (c["key"] == json!(a) || c["key"] == json!("ray-l5-acs-a")) && c["active"].as_u64().unwrap_or(0) > 0)
    })
    .await;
    assert_eq!(span["protected"], json!(true), "{span}");
    let model_row = model_status(&engines(&d).await, a).cloned().unwrap_or(Value::Null);
    assert_eq!(
        model_row["active_consumer"]["protected"], json!(true),
        "L5 spec: the Engines tab card exposes the Active-Consumer span: {model_row}"
    );

    // Explicit evict is refused with the honest reason, and the model is skipped.
    let (status, text) = d
        .post_json("/api/engines/evict", json!({"model": a}))
        .await;
    assert_eq!(
        status, 503,
        "L5 spec: a protected consumer blocks eviction (503): {text}"
    );
    let body: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    assert!(
        body["error"]
            .as_str()
            .unwrap_or("")
            .contains("blocked:active-consumer"),
        "L5 spec block reason `blocked:active-consumer`: {body}"
    );
    assert!(pid_alive(a_pid[0]), "a skipped eviction keeps the engine running");

    // Cutover of the exclusive device: the protected predecessor re-checks at
    // the SIGTERM point and re-queues/fails with the same named reason.
    let (status, body) = ensure_with(&d, b, json!({})).await;
    assert_eq!(
        status, 503,
        "L5 spec: cutover of a protected predecessor is blocked: {status} {body}"
    );
    assert!(
        body["error"]
            .as_str()
            .unwrap_or("")
            .contains("blocked:active-consumer"),
        "{body}"
    );
    let a_events = events_for(&d, a).await;
    assert!(
        a_events
            .iter()
            .any(|e| e["kind"] == json!("blocked")
                && e["reason"].as_str().unwrap_or("").contains("blocked:active-consumer")),
        "L5 spec: `blocked:active-consumer` is an event (audit/Engines tab): {a_events:?}"
    );

    // The one-action override bypasses the span for that operation (audited).
    let (status, text) = d
        .post_json(
            "/api/engines/evict",
            json!({"model": a, "override": true}),
        )
        .await;
    assert_eq!(status, 200, "--override evict -> {status}: {text}");
    let override_body: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(override_body["stopped"], json!(true), "{override_body}");
    let events = events_for(&d, a).await;
    assert!(
        kinds(&events).contains(&"override".to_string()),
        "L5 spec: the single-action override is audit-logged: {events:?}"
    );
    let audit = events
        .iter()
        .find(|e| e["kind"] == json!("override"))
        .expect("override event");
    assert!(
        audit["ts"].as_str().is_some_and(|ts| !ts.is_empty()) && audit["model"] == json!(a),
        "audit entry carries when+target: {audit}"
    );
    poll_until("override eviction to stop the child", 20, || async {
        (!pid_alive(a_pid[0])).then_some(())
    })
    .await;
    let _ = stream_task.await;

    // Grace span: after the last stream ends the model stays protected for
    // `active_consumer_grace_s` (fixture: 2500 ms), bridging tool gaps.
    ensure_ok(&d, a).await;
    wait_model_state(&d, a, "ready").await;
    let url = d.url.clone();
    let stream_task = tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap();
        client
            .post(format!("{url}/v1/chat/completions"))
            .json(&json!({
                "model": "ray-l5-acs-a",
                "messages": [{"role": "user", "content": "acs grace probe"}],
                "stream": true,
            }))
            .send()
            .await
            .expect("grace stream")
            .text()
            .await
            .unwrap_or_default()
    });

    poll_until("live span for the grace probe", 20, || async {
        consumers(&d)
            .await
            .into_iter()
            .find(|c| c["key"] == json!(a) && c["active"].as_u64().unwrap_or(0) > 0)
    })
    .await;
    // Let the stream finish (6 s hold) and then observe the grace window.
    let _ = stream_task.await;
    let grace_span = poll_until("idle grace span after stream end", 20, || async {
        consumers(&d)
            .await
            .into_iter()
            .find(|c| c["key"] == json!(a) && c["active"].as_u64().unwrap_or(0) == 0 && c["protected"] == json!(true))
    })
    .await;
    assert!(
        grace_span["protected_until_ms"].as_u64().unwrap_or(0) > 0,
        "L5 spec: the span records when grace expires: {grace_span}"
    );
    let (status, body) = d
        .post_json("/api/engines/evict", json!({"model": a}))
        .await;
    assert_eq!(
        status, 503,
        "L5 spec: within the grace window the model is still protected: {status} {body}"
    );

    // Once grace expires the model is evictable.
    poll_until("grace expiry", 15, || async {
        let spans = consumers(&d).await;
        let still = spans
            .iter()
            .any(|c| c["key"] == json!(a) && c["protected"] == json!(true));
        (!still).then_some(())
    })
    .await;
    let (status, text) = d
        .post_json("/api/engines/evict", json!({"model": a}))
        .await;
    assert_eq!(
        status, 200,
        "L5 spec: after the grace window the model is evictable: {status} {text}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// (3) ACS — in-process (review loop) stream pin without an in-flight row
// ═══════════════════════════════════════════════════════════════════════════

/// L5 spec: "in-process router calls (review loop `route_with_choice` — the
/// router's stream wrapper increments/decrements an endpoint-keyed counter;
/// today this traffic creates no rows)". The review start API is the only
/// in-process route surface: `POST /review/api/request-async` with the review
/// card pinned to a catalog model.
#[tokio::test(flavor = "multi_thread")]
async fn l5_acs_in_process_review_stream_pin_without_inflight_row() {
    let _gate = gate().await;
    let env = TestEnv::new("acs-review");
    let _guard = ChildGuard::new(&env.root);
    let program = install_mock_engines(&env);
    let shim = spawn_shim(ShimCfg::default());
    let port_min = free_port();
    let range = format!("{port_min}-{}", port_min.saturating_add(2));
    let extra = vec![
        ("BRAINROUTER_FLEET_ENGINES".to_string(), "strixite".to_string()),
        ("BRAINROUTER_STRIXITE_BIN".to_string(), program.display().to_string()),
        ("BRAINROUTER_STRIXITE_CONF_DIR".to_string(), env.conf_dir().display().to_string()),
        ("BRAINROUTER_STRIXITE_PORT_RANGE".to_string(), range),
        ("BRAINROUTER_STRIXITE_MIN_MEM_GIB".to_string(), "0".to_string()),
        ("BRAINROUTER_CUTOVER_SETTLE_MS".to_string(), "0".to_string()),
        ("BRAINROUTER_CUTOVER_RECHECK_MS".to_string(), "100".to_string()),
        ("BRAINROUTER_ACTIVE_CONSUMER_GRACE_MS".to_string(), "2000".to_string()),
    ];
    let mut d = Daemon::start(&env, &base_config(&shim.url), &extra);
    d.ready().await;

    let review_file = gguf(&env, "review-model.gguf");
    let other_file = gguf(&env, "review-other.gguf");
    let review_model = "ray-l5-review-model";
    let other = "ray-l5-review-other";
    add_entry(
        &d,
        catalog_entry(
            review_model,
            "strixite",
            &[&review_file],
            4.0,
            &[],
            &[
                ("MOCK_STEM", review_model),
                ("MOCK_HOLD_MS", "6000"),
                (
                    "MOCK_VERDICT_JSON",
                    "{\"status\":\"approved\",\"feedback\":\"ray-l5 mock approval\"}",
                ),
            ],
        ),
    )
    .await;
    add_entry(
        &d,
        catalog_entry(other, "strixite", &[&other_file], 4.0, &[], &[("MOCK_STEM", other)]),
    )
    .await;

    // The review card is the single reviewer source of truth (L3): pin it to
    // the catalog model so the loop's in-process route targets our engine.
    let (status, text) = d
        .post_json(
            "/api/roles/set",
            json!({"role": "review", "assignment": review_model}),
        )
        .await;
    assert_eq!(status, 200, "pin the review card -> {status}: {text}");

    // A real project shape for the context gatherer.
    let project = env.root.join("review-project");
    fs::create_dir_all(&project).unwrap();
    fs::write(project.join("README.md"), "# ray-l5 review fixture\n").unwrap();
    fs::write(project.join("code.txt"), "fn main() {}\n").unwrap();
    for args in [
        vec!["init", "-q"],
        vec!["config", "user.email", "ray@example.invalid"],
        vec!["config", "user.name", "Ray"],
        vec!["add", "."],
        vec!["commit", "-q", "-m", "fixture"],
    ] {
        let out = Command::new("git")
            .args(&args)
            .current_dir(&project)
            .output()
            .expect("git");
        assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    }
    fs::write(project.join("code.txt"), "fn main() { println!(\"diff\"); }\n").unwrap();

    let (status, text) = d
        .post_json(
            "/review/api/request-async",
            json!({
                "taskId": "ray-l5-review",
                "summary": "in-process ACS fixture",
                "details": "assert the review route consumes without a proxy row",
                "conversationHistory": [],
                "cwd": project.display().to_string(),
            }),
        )
        .await;
    assert_eq!(status, 200, "review start -> {status}: {text}");
    let started: Value = serde_json::from_str(&text).unwrap();
    let session_id = started["sessionId"]
        .as_str()
        .expect("review session id")
        .to_string();

    // The in-process stream enters the ACS with NO proxy in-flight row.
    let span = poll_until("in-process ACS span from the review loop", 40, || async {
        let spans = consumers(&d).await;
        let inflight = d.get_json("/api/inflight").await;
        let rows = inflight["requests"].as_array().cloned().unwrap_or_default();
        if !rows.is_empty() {
            return None;
        }
        spans
            .into_iter()
            .find(|c| c["key"] == json!(review_model) && c["active"].as_u64().unwrap_or(0) > 0)
    })
    .await;
    assert_eq!(span["protected"], json!(true), "{span}");

    // Eviction / cutover respects the in-process consumer.
    let (status, body) = d
        .post_json("/api/engines/evict", json!({"model": review_model}))
        .await;
    assert_eq!(
        status, 503,
        "L5 spec: an in-process review stream blocks eviction: {status} {body}"
    );
    let (status, body) = ensure_with(&d, other, json!({})).await;
    assert_eq!(
        status, 503,
        "L5 spec: the review stream blocks the exclusive cutover: {status} {body}"
    );
    assert!(
        body["error"]
            .as_str()
            .unwrap_or("")
            .contains("blocked:active-consumer"),
        "{body}"
    );
    let events = events_for(&d, review_model).await;
    assert!(
        events
            .iter()
            .any(|e| e["reason"].as_str().unwrap_or("").contains("in-process stream")),
        "L5 spec: the block detail names the in-process consumer: {events:?}"
    );

    // The review completes (the mock verdict), then the grace window applies.
    poll_until("review session to reach a terminal status", 40, || async {
        let v = d.get_json(&format!("/review/api/sessions/{session_id}")).await;
        matches!(
            v["status"].as_str(),
            Some("approved") | Some("needs_revision") | Some("escalated") | Some("failed")
        )
        .then_some(())
    })
    .await;
    let grace = poll_until("idle grace after the review stream", 30, || async {
        consumers(&d)
            .await
            .into_iter()
            .find(|c| c["key"] == json!(review_model) && c["active"].as_u64() == Some(0) && c["protected"] == json!(true))
    })
    .await;
    assert!(grace["protected_until_ms"].as_u64().unwrap_or(0) > 0, "{grace}");
    let (status, body) = ensure_with(&d, other, json!({})).await;
    assert_eq!(
        status, 503,
        "L5 spec: grace continues to protect the endpoint after the stream ends: {status} {body}"
    );
    poll_until("grace expiry after the review", 20, || async {
        let still = consumers(&d)
            .await
            .iter()
            .any(|c| c["key"] == json!(review_model) && c["protected"] == json!(true));
        (!still).then_some(())
    })
    .await;
    let outcome = ensure_ok(&d, other).await;
    assert_eq!(outcome["spawned"], json!(true), "{outcome}");
}

// ═══════════════════════════════════════════════════════════════════════════
// (4) External residents — observed, attach-not-spawn, adopt, ownership
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn l5_external_observer_ready_at_endpoint_attach_adopt_and_ownership() {
    let _gate = gate().await;
    let env = TestEnv::new("external");
    let _guard = ChildGuard::new(&env.root);
    let program = install_mock_engines(&env);
    let shim = spawn_shim(ShimCfg::default());
    let port_min = free_port();
    let range = format!("{port_min}-{}", port_min.saturating_add(2));
    let extra = vec![
        ("BRAINROUTER_FLEET_ENGINES".to_string(), "strixite".to_string()),
        ("BRAINROUTER_STRIXITE_BIN".to_string(), program.display().to_string()),
        ("BRAINROUTER_STRIXITE_CONF_DIR".to_string(), env.conf_dir().display().to_string()),
        ("BRAINROUTER_STRIXITE_PORT_RANGE".to_string(), range),
        ("BRAINROUTER_STRIXITE_MIN_MEM_GIB".to_string(), "0".to_string()),
        ("BRAINROUTER_CUTOVER_SETTLE_MS".to_string(), "0".to_string()),
    ];
    let mut d = Daemon::start(&env, &base_config(&shim.url), &extra);
    d.ready().await;

    let icon_file = gguf(&env, "icon-a.gguf");
    let second_file = gguf(&env, "native-b.gguf");
    let icon = "ray-l5-icon-a";
    let second = "ray-l5-native-b";
    add_entry(
        &d,
        catalog_entry(icon, "strixite", &[&icon_file], 4.0, &[], &[("MOCK_STEM", icon)]),
    )
    .await;
    add_entry(
        &d,
        catalog_entry(second, "strixite", &[&second_file], 4.0, &[], &[("MOCK_STEM", second)]),
    )
    .await;

    // Icon-launched engine: real pid, argv names `strix_server`, `--port`,
    // `--weights`; `/v1/models` advertises the catalog stem.
    let external_port = free_port();
    let mut external = ExternalEngine::spawn(
        &env,
        "strix_server",
        external_port,
        &icon_file,
        &[("MOCK_STEM", icon)],
    );
    wait_mock_ready(&external).await;

    // Read-only observation: source=external + engine field, stems probed.
    let row = poll_until("observed external resident", 20, || async {
        external_rows(&d)
            .await
            .into_iter()
            .find(|r| r["pid"] == json!(external.pid))
    })
    .await;
    assert_eq!(row["engine"], json!("strixite"), "{row}");
    assert_eq!(row["catalog_id"], json!(icon), "{row}");
    assert_eq!(row["stems"], json!([icon]), "{row}");
    assert_eq!(row["adopted"], json!(false), "{row}");
    assert_eq!(row["port"], json!(external_port), "{row}");

    let model_row = poll_until("resident model row", 20, || async {
        let e = engines(&d).await;
        let m = model_status(&e, icon)?.clone();
        (m["state"] == json!("ready")).then_some(m)
    })
    .await;
    assert_eq!(
        model_row["source"],
        json!("external"),
        "L5 spec: Residency source=external for icon-launched servers: {model_row}"
    );

    // Residency-first: ensure attaches at the external endpoint, never spawns.
    let outcome = ensure_ok(&d, icon).await;
    assert_eq!(outcome["spawned"], json!(false), "{outcome}");
    assert_eq!(
        outcome["endpoint"],
        json!(format!("http://127.0.0.1:{external_port}")),
        "L5 spec: Ready-at-external (attach-not-spawn): {outcome}"
    );
    let spawn_events = events_for(&d, icon).await;
    assert!(
        !kinds(&spawn_events).contains(&"spawn".to_string()),
        "L5 spec: attach-not-spawn must not spawn: {spawn_events:?}"
    );
    // Budget-counted (external resident is not free VRAM).
    let status = engines(&d).await;
    let counted = status["budget"]["residents"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .any(|r| r["model"] == json!(icon) && r["source"] == json!("external"));
    assert!(
        counted,
        "L5 spec: external residents are budget-counted: {}",
        status["budget"]
    );

    // Cutover SIGTERM targets ONLY supervisor-owned marker pids: an
    // unregistered external is never signalled (start-ownership, L4 binding).
    let outcome = ensure_ok(&d, second).await;
    assert_eq!(outcome["spawned"], json!(true), "{outcome}");
    assert!(
        pid_alive(external.pid),
        "L5 spec: cutover SIGTERM targets ONLY supervisor-spawned/adopted marker pids — \
         never an unregistered external"
    );

    // `swap adopt` registers the external pid as a native resident (marker,
    // port-less spec hash) and classification precedence flips to native.
    let (status, text) = d
        .post_json(
            "/api/engines/adopt",
            json!({"model": icon, "pid": external.pid}),
        )
        .await;
    assert_eq!(status, 200, "adopt -> {status}: {text}");
    let report: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(report["pid"], json!(external.pid), "{report}");
    assert_eq!(report["endpoint"], json!(format!("http://127.0.0.1:{external_port}")), "{report}");
    assert!(
        report["spec_hash"].as_str().is_some_and(|h| !h.is_empty()),
        "L5 spec: adoption records a marker `spec_hash` (port-less): {report}"
    );
    let marker_path = env.adoption_dir().join(format!("{icon}.json"));
    let marker: Value =
        serde_json::from_str(&fs::read_to_string(&marker_path).expect("adoption marker")).unwrap();
    assert_eq!(marker["pid"], json!(external.pid), "{marker}");
    assert_eq!(marker["model"], json!(icon), "{marker}");
    poll_until("adopted pid leaves the external list (native precedence)", 20, || async {
        let gone = !external_rows(&d)
            .await
            .iter()
            .any(|r| r["pid"] == json!(external.pid));
        gone.then_some(())
    })
    .await;
    assert!(
        kinds(&events_for(&d, icon).await).contains(&"adopt".to_string()),
        "L5 spec: adoption is audit-logged: {:?}",
        events_for(&d, icon).await
    );

    // Ownership transferred: a later cutover may now signal that pid.
    let (status, text) = d
        .post_json("/api/engines/evict", json!({"model": second}))
        .await;
    assert_eq!(status, 200, "evict the native sibling -> {status}: {text}");
    wait_model_state(&d, second, "stopped").await;
    let outcome = ensure_ok(&d, second).await;
    assert_eq!(outcome["spawned"], json!(true), "{outcome}");
    poll_until("adopted pid drained by the cutover", 20, || async {
        (!pid_running(external.pid)).then_some(())
    })
    .await;
    let _ = external.child.try_wait();
}

// ═══════════════════════════════════════════════════════════════════════════
// (4) Takeover — unit attach, refusal, single --override, pid scope, CSRF
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn l5_takeover_unit_attach_refusal_single_override_and_pid_scope() {
    let _gate = gate().await;
    let env = TestEnv::new("takeover");
    let _guard = ChildGuard::new(&env.root);
    let program = install_mock_engines(&env);
    let shim = spawn_shim(ShimCfg::default());
    let (systemctl, unit_dir) = write_systemctl_shim(&env);
    let port_min = free_port();
    let range = format!("{port_min}-{}", port_min.saturating_add(2));
    let unit = "strix_server.service";
    let extra = vec![
        ("BRAINROUTER_FLEET_ENGINES".to_string(), "strixite".to_string()),
        ("BRAINROUTER_STRIXITE_BIN".to_string(), program.display().to_string()),
        ("BRAINROUTER_STRIXITE_CONF_DIR".to_string(), env.conf_dir().display().to_string()),
        ("BRAINROUTER_STRIXITE_PORT_RANGE".to_string(), range),
        ("BRAINROUTER_STRIXITE_MIN_MEM_GIB".to_string(), "0".to_string()),
        ("BRAINROUTER_SYSTEMCTL".to_string(), systemctl.display().to_string()),
        ("BRAINROUTER_MOCK_UNIT_DIR".to_string(), unit_dir.display().to_string()),
        ("BRAINROUTER_SWAP_STOP_GRACE_MS".to_string(), "800".to_string()),
        ("BRAINROUTER_CUTOVER_SETTLE_MS".to_string(), "0".to_string()),
    ];
    let mut d = Daemon::start(&env, &base_config(&shim.url), &extra);
    d.ready().await;

    let served_file = gguf(&env, "unit-served.gguf");
    let wanted_file = gguf(&env, "unit-wanted.gguf");
    let served = "ray-l5-unit-served";
    let wanted = "ray-l5-unit-wanted";
    add_entry(
        &d,
        catalog_entry(served, "strixite", &[&served_file], 4.0, &[], &[("MOCK_STEM", served)]),
    )
    .await;
    add_entry(
        &d,
        catalog_entry(wanted, "strixite", &[&wanted_file], 4.0, &[], &[("MOCK_STEM", wanted)]),
    )
    .await;

    // The operator's unit runs an icon-launched engine that serves `served`.
    let served_port = free_port();
    let mut unit_engine = ExternalEngine::spawn(
        &env,
        "strix_server",
        served_port,
        &served_file,
        &[("MOCK_STEM", served)],
    );
    wait_mock_ready(&unit_engine).await;
    set_unit(&unit_dir, unit, true, unit_engine.pid);
    // Unrelated foreign engine (unregistered weights + stem): never a
    // takeover target, never signalled.
    let foreign_file = gguf(&env, "unit-foreign.gguf");
    let foreign_port = free_port();
    let mut foreign = ExternalEngine::spawn(
        &env,
        "strix_server",
        foreign_port,
        &foreign_file,
        &[("MOCK_STEM", "ray-l5-foreign")],
    );
    wait_mock_ready(&foreign).await;

    // Units are visible on the Engines tab surface with their state.
    let units = poll_until("configured unit with its live state", 20, || async {
        let v = d.get_json("/api/engines/external").await;
        v["units"]
            .as_array()?
            .iter()
            .find(|u| u["unit"] == json!(unit) && u["active"] == json!(true))
            .cloned()
    })
    .await;
    assert_eq!(units["main_pid"], json!(unit_engine.pid), "{units}");

    // The observer must have classified the unit's pid as an external resident
    // before residency-first admission can attach (2 s refresh loop).
    let observed = poll_until("unit engine observed as an external resident", 20, || async {
        external_rows(&d)
            .await
            .into_iter()
            .find(|r| r["pid"] == json!(unit_engine.pid) && r["catalog_id"] == json!(served))
    })
    .await;
    assert_eq!(observed["engine"], json!("strixite"), "{observed}");
    assert_eq!(observed["adopted"], json!(false), "{observed}");

    // Unit-running mock ⇒ `ensure` ATTACHES to the registered external
    // endpoint, no spawn (Residency-first).
    let outcome = ensure_ok(&d, served).await;
    assert_eq!(outcome["spawned"], json!(false), "{outcome}");
    assert_eq!(
        outcome["endpoint"],
        json!(format!("http://127.0.0.1:{served_port}")),
        "{outcome}"
    );

    // `takeover` of the unit that serves the model: attach + adopt its pid.
    let (status, text) = d
        .post_json(
            "/api/engines/takeover",
            json!({"model": served, "unit": unit}),
        )
        .await;
    assert_eq!(status, 200, "takeover(served) -> {status}: {text}");
    let report: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(report["attached"], json!(true), "{report}");
    assert_eq!(report["spawned"], json!(false), "{report}");
    assert_eq!(
        report["endpoint"],
        json!(format!("http://127.0.0.1:{served_port}")),
        "L5 spec: Ready-at-external — takeover attaches at the serving endpoint: {report}"
    );
    assert_eq!(report["stopped_unit"], Value::Null, "{report}");

    // A unit that does NOT serve the requested model is refused without the
    // single-action `--override` (blocked:unit-owned, pid untouched).
    let (status, text) = d
        .post_json(
            "/api/engines/takeover",
            json!({"model": wanted, "unit": unit}),
        )
        .await;
    assert_eq!(
        status, 409,
        "L5 spec: takeover of an operator unit is refused without --override: {status} {text}"
    );
    let body: Value = serde_json::from_str(&text).unwrap();
    assert!(
        body["error"].as_str().unwrap_or("").contains("blocked:unit-owned"),
        "{body}"
    );
    assert!(pid_alive(unit_engine.pid), "a refused takeover must not signal the unit");
    assert!(unit_stops(&unit_dir).is_empty(), "no `systemctl stop` before the override");

    // One `--override` action: pid-scoped SIGTERM of the unit's pids, unit
    // stopped as cleanup, then the normal admission spawns the target.
    let (status, text) = d
        .post_json(
            "/api/engines/takeover",
            json!({"model": wanted, "unit": unit, "override": true}),
        )
        .await;
    assert_eq!(status, 200, "takeover --override -> {status}: {text}");
    let report: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(report["spawned"], json!(true), "{report}");
    assert_eq!(report["stopped_unit"], json!(unit), "{report}");
    poll_until("pid-scoped unit termination", 20, || async {
        (!pid_running(unit_engine.pid)).then_some(())
    })
    .await;
    assert!(
        pid_alive(foreign.pid),
        "L5 spec: pid-scoped takeover — a foreign engine is never signalled"
    );
    assert_eq!(
        unit_stops(&unit_dir),
        vec![unit.to_string()],
        "the operator unit is stopped exactly once"
    );

    // A second takeover call is an attach (single-action semantics; no second
    // stop, no reset of the now-supervised resident).
    let (status, text) = d
        .post_json(
            "/api/engines/takeover",
            json!({"model": wanted, "unit": unit}),
        )
        .await;
    assert_eq!(status, 200, "second takeover -> {status}: {text}");
    let report: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(report["spawned"], json!(false), "{report}");
    assert_eq!(report["attached"], json!(true), "{report}");
    assert_eq!(
        unit_stops(&unit_dir).len(),
        1,
        "L5 spec: override is ONE action — the second call must not repeat it"
    );

    // Mutations are loopback+CSRF gated (same pattern as every v2 write path).
    let (status, text) = d
        .request(
            "POST",
            "/api/engines/takeover",
            Some(json!({"model": wanted})),
            &[("Origin", "http://evil.example")],
        )
        .await;
    assert_eq!(
        status, 403,
        "L5 spec: takeover mutations keep the loopback+CSRF gate: {status} {text}"
    );
    let (status, text) = d
        .request(
            "POST",
            "/api/engines/adopt",
            Some(json!({"model": wanted, "pid": foreign.pid})),
            &[("Origin", "http://evil.example")],
        )
        .await;
    assert_eq!(status, 403, "adopt keeps the CSRF gate: {status} {text}");

    foreign.kill();
    let _ = unit_engine.child.try_wait();
}

// ═══════════════════════════════════════════════════════════════════════════
// (5) hipfire — cards/sidecar, squat, port-free gate
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn l5_hipfire_cards_sidecar_squat_and_port_free_gate() {
    let _gate = gate().await;
    let env = TestEnv::new("hipfire");
    let _guard = ChildGuard::new(&env.root);
    let program = install_mock_engines(&env);
    let shim = spawn_shim(ShimCfg::default());
    let models_dir = env.root.join("hipfire-models");
    fs::create_dir_all(&models_dir).unwrap();
    fs::write(models_dir.join("ray-l5-alpha.mq4"), b"MQ4-alpha").unwrap();
    fs::write(models_dir.join("ray-l5-alpha.mtp"), b"MTP-alpha").unwrap();
    fs::write(models_dir.join("ray-l5-beta.mq4"), b"MQ4-beta").unwrap();
    let hip_port = free_port();
    let extra = vec![
        ("BRAINROUTER_FLEET_ENGINES".to_string(), "hipfire".to_string()),
        ("BRAINROUTER_HIPFIRE_BIN".to_string(), program.display().to_string()),
        ("BRAINROUTER_HIPFIRE_MODELS_DIR".to_string(), models_dir.display().to_string()),
        ("BRAINROUTER_HIPFIRE_PORT".to_string(), hip_port.to_string()),
        ("BRAINROUTER_HIPFIRE_MIN_MEM_GIB".to_string(), "0".to_string()),
        ("BRAINROUTER_HIPFIRE_MAX_SEQ".to_string(), "8192".to_string()),
        ("BRAINROUTER_HIPFIRE_PORT_FREE_MS".to_string(), "3000".to_string()),
        ("BRAINROUTER_READINESS_TIMEOUT_MS".to_string(), "20000".to_string()),
        ("BRAINROUTER_SWAP_STOP_GRACE_MS".to_string(), "1000".to_string()),
    ];
    let mut d = Daemon::start(&env, &base_config(&shim.url), &extra);
    d.ready().await;

    // Card registry: `*.mq4` + `.mtp` sidecar recorded; missing sidecar = warn.
    let status = engines(&d).await;
    let detail = status["adapter_detail"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .find(|a| a["engine"] == json!("hipfire"))
        .unwrap_or_else(|| panic!("L5 spec: Engines tab hipfire adapter detail: {status}"));
    let cards = detail["cards"].as_array().cloned().unwrap_or_default();
    let alpha = cards
        .iter()
        .find(|c| c["stem"] == json!("ray-l5-alpha"))
        .expect("alpha card");
    assert_eq!(alpha["sidecar_present"], json!(true), "{alpha}");
    let beta = cards
        .iter()
        .find(|c| c["stem"] == json!("ray-l5-beta"))
        .expect("beta card");
    assert_eq!(beta["sidecar_present"], json!(false), "{beta}");
    let warnings = detail["warnings"].as_array().cloned().unwrap_or_default();
    assert!(
        warnings.iter().any(|w| w.as_str().unwrap_or("").contains("ray-l5-beta")),
        "L5 spec: a missing sidecar is a warning, never an error: {warnings:?}"
    );

    // Unknown/unsupported flag rejected at spec build: hipfire has no `--mtp`.
    let mtp_id = "ray-l5-hipfire-mtp";
    add_entry(
        &d,
        catalog_entry(
            mtp_id,
            "hipfire",
            &[&models_dir.join("ray-l5-alpha.mq4")],
            4.0,
            &["--mtp"],
            &[("MOCK_STEM", mtp_id)],
        ),
    )
    .await;
    let (status, body) = ensure_with(&d, mtp_id, json!({})).await;
    assert_eq!(
        status, 503,
        "L5 spec: the engine-validated schema rejects `--mtp` at spec build: {status} {body}"
    );
    assert!(
        body["error"].as_str().unwrap_or("").contains("--mtp"),
        "{body}"
    );

    // The card-backed catalog entry (no `files`: the card supplies weights).
    let alpha_id = "ray-l5-alpha";
    add_entry(
        &d,
        catalog_entry(
            alpha_id,
            "hipfire",
            &[],
            4.0,
            &[],
            &[("MOCK_STEM", alpha_id)],
        ),
    )
    .await;

    // Squat detection: a foreign server owns the single hipfire port ⇒ Failed
    // with the mismatch named (losing a suite to a squat is structurally
    // impossible), and the foreign server is untouched.
    let squatter = ExternalEngine::spawn(
        &env,
        "hipfire",
        hip_port,
        &models_dir.join("ray-l5-alpha.mq4"),
        &[("MOCK_STEM", "ray-l5-squatter")],
    );
    wait_mock_ready(&squatter).await;
    assert!(
        cards.iter().any(|c| c["stem"] == json!("ray-l5-alpha")),
        "fixture precondition: the card stem is the catalog id"
    );
    let (status, body) = ensure_with(&d, alpha_id, json!({})).await;
    assert_eq!(
        status, 503,
        "L5 spec: an expected-stem mismatch at the endpoint is Failed: {status} {body}"
    );
    let text = body["error"].as_str().unwrap_or("");
    assert!(
        text.contains("port is owned by a foreign server") || text.contains("foreign server"),
        "L5 spec squat detection names the mismatch: {text}"
    );
    assert!(pid_alive(squatter.pid), "squat detection never kills the foreign owner");
    let row = model_status(&engines(&d).await, alpha_id).cloned().unwrap_or(Value::Null);
    assert_eq!(row["state"], json!("failed"), "the Engines tab badge row: {row}");
    assert!(
        row["last_error"].as_str().unwrap_or("").contains("foreign server"),
        "{row}"
    );

    // Port-free gate: stop the squatter, spawn, stop, and prove the port is
    // free before the next spawn can happen.
    let mut squatter = squatter;
    squatter.kill();
    poll_until("hipfire port free", 10, || async {
        port_bindable(hip_port).then_some(())
    })
    .await;
    let outcome = ensure_ok(&d, alpha_id).await;
    assert_eq!(outcome["spawned"], json!(true), "{outcome}");
    assert_eq!(
        outcome["endpoint"],
        json!(format!("http://127.0.0.1:{hip_port}")),
        "L5 spec: single-owner port on :8080 (fixture port): {outcome}"
    );
    wait_model_state(&d, alpha_id, "ready").await;
    let (status, text) = d
        .post_json("/api/engines/evict", json!({"model": alpha_id}))
        .await;
    assert_eq!(status, 200, "hipfire evict -> {status}: {text}");
    poll_until("hipfire port free after stop", 15, || async {
        port_bindable(hip_port).then_some(())
    })
    .await;
    let health_fails = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("client")
        .get(format!("http://127.0.0.1:{hip_port}/health"))
        .timeout(Duration::from_millis(400))
        .send()
        .await
        .is_err();
    assert!(
        health_fails,
        "L5 spec cutover: stop = SIGTERM + port-free gate (`/health` must FAIL before next spawn)"
    );
    let outcome = ensure_ok(&d, alpha_id).await;
    assert_eq!(outcome["spawned"], json!(true), "the port-free gate allows the next spawn");
}

// ═══════════════════════════════════════════════════════════════════════════
// (5) hipfire — version-keyed schema (probe discovers the version marker)
// ═══════════════════════════════════════════════════════════════════════════

/// L5 spec §hipfire adapter: "engine-validated flag schema keyed on the engine
/// version marker (probe discovers install root + version): re-derived on
/// version change, schema drift = Engines-tab badge (an engine upgrade can't
/// silently poison the stored schema)". The mock reports `hipfire 1.4.2`, so a
/// v1+ documented flag (`--flash-attn`) must be legal.
#[tokio::test(flavor = "multi_thread")]
async fn l5_hipfire_version_keyed_schema_from_engine_marker() {
    let _gate = gate().await;
    let env = TestEnv::new("hipfire-schema");
    let _guard = ChildGuard::new(&env.root);
    install_mock_engines(&env);
    let shim = spawn_shim(ShimCfg::default());
    let models_dir = env.root.join("hipfire-models");
    fs::create_dir_all(&models_dir).unwrap();
    fs::write(models_dir.join("ray-l5-schema.mq4"), b"MQ4-schema").unwrap();
    let hip_port = free_port();
    let extra = vec![
        ("BRAINROUTER_FLEET_ENGINES".to_string(), "hipfire".to_string()),
        (
            "BRAINROUTER_HIPFIRE_BIN".to_string(),
            env.bin.join("hipfire").display().to_string(),
        ),
        ("BRAINROUTER_HIPFIRE_MODELS_DIR".to_string(), models_dir.display().to_string()),
        ("BRAINROUTER_HIPFIRE_PORT".to_string(), hip_port.to_string()),
        ("BRAINROUTER_HIPFIRE_MIN_MEM_GIB".to_string(), "0".to_string()),
        ("BRAINROUTER_READINESS_TIMEOUT_MS".to_string(), "20000".to_string()),
    ];
    let mut d = Daemon::start(&env, &base_config(&shim.url), &extra);
    d.ready().await;

    let status = engines(&d).await;
    let detail = status["adapter_detail"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .find(|a| a["engine"] == json!("hipfire"))
        .expect("hipfire adapter detail");
    let schema_version = detail["schema"]["version"].as_str().unwrap_or("").to_string();
    assert_eq!(
        schema_version, "1.4.2",
        "L5 spec: the schema is keyed on the engine version marker probed from the \
         installed engine (`<program> --version` reports `hipfire 1.4.2`); the Engines \
         tab renders it. Adapter detail: {detail}"
    );

    // A v1+ documented flag is legal under that derived schema.
    let model_id = "ray-l5-schema-model";
    add_entry(
        &d,
        catalog_entry(
            model_id,
            "hipfire",
            &[&models_dir.join("ray-l5-schema.mq4")],
            4.0,
            &["--flash-attn"],
            &[("MOCK_STEM", model_id)],
        ),
    )
    .await;
    let (status, body) = ensure_with(&d, model_id, json!({})).await;
    assert_eq!(
        status, 200,
        "L5 spec: under a version-keyed schema derived from engine 1.4.2 the documented \
         `--flash-attn` flag is legal (re-derivation on version change): {status} {body}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// (6) Engines tab, events additive, v1/L4 floors
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn l5_engines_tab_columns_events_additive_and_v1_floor() {
    let _gate = gate().await;
    let env = TestEnv::new("engines-tab");
    let _guard = ChildGuard::new(&env.root);
    let _ = install_mock_engines(&env);
    let shim = spawn_shim(ShimCfg {
        models: vec!["ray-l5-v1-model".into()],
        ..Default::default()
    });
    let mut d = Daemon::start(&env, &base_config(&shim.url), &[]);
    d.ready().await;

    // Engines tab HTML carries the L5 surfaces (external residents, takeover
    // units, ACS, the one-action override, gate + consumer state).
    let (status, html) = d.request("GET", "/dashboard", None, &[]).await;
    assert_eq!(status, 200, "dashboard -> {status}");
    for token in [
        "engines-external",
        "engines-units",
        "engines-consumers",
        "data-eng-action=\"ensure-override\"",
        "data-eng-action=\"takeover\"",
        "data-eng-action=\"adopt\"",
        "waiting(gate)",
        "active consumer",
        "adapter_detail",
    ] {
        assert!(
            html.contains(token),
            "L5 spec: the Engines tab renders `{token}` (adapter states, Waiting(gate), \
             takeover/adopt/swap-anyway, ACS)"
        );
    }

    // `/api/engines` keeps the L4 payload and adds the L5 sections additively.
    let status = engines(&d).await;
    for key in [
        "platform", "models", "queue", "budget", "adoption", "adapters", "groups", "pins",
        "adapter_detail", "external", "units", "active_consumers",
    ] {
        assert!(
            status.get(key).is_some(),
            "L5 spec: `/api/engines` is additive — `{key}` must still be present: {status}"
        );
    }
    assert!(status["external"].is_array(), "{status}");
    assert!(status["units"].is_array(), "{status}");
    assert!(status["active_consumers"].is_array(), "{status}");
    // No L5 engine is engaged in this fixture: no foreign residents, no spans.
    assert_eq!(status["external"], json!([]), "{}", status["external"]);
    assert_eq!(status["active_consumers"], json!([]), "{}", status["active_consumers"]);

    // v1 floor: no L5 catalog entry involved ⇒ byte-identical shim routing and
    // the v1 RouteEvent field set (events additive only).
    let sent = json!({
        "model": "ray-l5-v1-model",
        "messages": [{"role": "user", "content": "ray-l5 v1 floor"}],
        "stream": true,
    });
    let resp = d
        .client
        .post(d.endpoint("/v1/chat/completions"))
        .json(&sent)
        .send()
        .await
        .expect("v1 floor chat");
    assert_eq!(resp.status().as_u16(), 200);
    let body = resp.text().await.unwrap_or_default();
    assert!(body.contains("ray-l5-pong"), "{body}");
    let forwarded = shim.chat_bodies();
    assert_eq!(forwarded.len(), 1, "{forwarded:?}");
    for (key, value) in sent.as_object().unwrap() {
        assert_eq!(
            forwarded[0].get(key),
            Some(value),
            "L5 floors: shim-mode forwarded body is v1's byte-for-byte (`{key}`): {}",
            forwarded[0]
        );
    }
    for key in forwarded[0].as_object().unwrap().keys() {
        assert!(
            sent.get(key).is_some(),
            "L5 floors: no L5 field may be invented upstream (`{key}`)"
        );
    }

    let event = poll_until("v1 floor routing event", 10, || async {
        let v = d.get_json("/api/routing-events").await;
        v["events"]
            .as_array()?
            .iter()
            .find(|e| e["requested_model"] == json!("ray-l5-v1-model"))
            .cloned()
    })
    .await;
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
            "L5 floors: RouteEvent is additive — lost `{field}`: {event}"
        );
    }
    assert_eq!(event["effective_provider"], json!("llama-swap"), "{event}");
    // L4's additive swap fields stay absent/None for a non-swap route.
    assert!(
        event.get("swap_state").is_none_or(Value::is_null),
        "L5 floors: non-swap route carries no swap_state: {event}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// (6) CLI surface — one-action override, force-gate, takeover, adopt
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn l5_cli_surface_override_force_gate_takeover_adopt() {
    let _gate = gate().await;
    let env = TestEnv::new("cli");
    let _guard = ChildGuard::new(&env.root);
    let program = install_mock_engines(&env);
    let shim = spawn_shim(ShimCfg::default());
    let port_min = free_port();
    let range = format!("{port_min}-{}", port_min.saturating_add(1));
    let extra = vec![
        ("BRAINROUTER_FLEET_ENGINES".to_string(), "strixite".to_string()),
        ("BRAINROUTER_STRIXITE_BIN".to_string(), program.display().to_string()),
        ("BRAINROUTER_STRIXITE_CONF_DIR".to_string(), env.conf_dir().display().to_string()),
        ("BRAINROUTER_STRIXITE_PORT_RANGE".to_string(), range),
        ("BRAINROUTER_STRIXITE_MIN_MEM_GIB".to_string(), "0".to_string()),
    ];
    let mut d = Daemon::start(&env, &base_config(&shim.url), &extra);
    d.ready().await;

    // Help text pins the spec's exact spellings (CLI is the operator surface).
    let help = run_cli(&env, &["swap", "ensure", "--help"]).ok("swap ensure --help");
    for flag in ["--override", "--force-gate", "--attach-only"] {
        assert!(
            help.contains(flag),
            "L5 spec: `swap ensure <model> --override` is THE one action; missing `{flag}`:\n{help}"
        );
    }
    let help = run_cli(&env, &["swap", "takeover", "--help"]).ok("swap takeover --help");
    for flag in ["--unit", "--override", "--force-gate"] {
        assert!(help.contains(flag), "L5 spec takeover flags (`{flag}`):\n{help}");
    }
    let help = run_cli(&env, &["swap", "adopt", "--help"]).ok("swap adopt --help");
    assert!(help.contains("--pid"), "L5 spec `swap adopt` takes an optional pid:\n{help}");

    // Mutations go through the daemon and reach the documented wire surface.
    let a_file = gguf(&env, "cli-a.gguf");
    let a = "ray-l5-cli-a";
    add_entry(
        &d,
        catalog_entry(a, "strixite", &[&a_file], 4.0, &[], &[("MOCK_STEM", a)]),
    )
    .await;
    let url = d.url.clone();
    let ensured = run_cli(&env, &["swap", "--url", &url, "ensure", a]).json("swap ensure");
    assert_eq!(ensured["spawned"], json!(true), "{ensured}");
    let evicted = run_cli(
        &env,
        &["swap", "--url", &url, "evict", a, "--override"],
    )
    .json("swap evict --override");
    assert_eq!(evicted["stopped"], json!(true), "{evicted}");
    let unknown = run_cli(&env, &["swap", "--url", &url, "ensure", "ray-l5-nope"]);
    assert_ne!(unknown.code, 0, "unknown model must fail: {unknown:?}",);
    assert!(
        format!("{}{}", unknown.stdout, unknown.stderr).contains("not in the swap catalog"),
        "{}{}",
        unknown.stdout,
        unknown.stderr
    );
}
