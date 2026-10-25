//! v2 L3 integration coverage for the plugin decision kind (assignment
//! requirement: "integration coverage: mock plugin exercising
//! `decision.classify` + initialize protocol negotiation — Ray's suite
//! doesn't cover these").
//!
//! Covers, against a live daemon + supervisor-spawned python fixtures:
//! 1. initialize protocol negotiation: v2 plugin ⇒ `protocol_version: 2` +
//!    `server_methods` includes `decision.classify`; v1 plugin ⇒ negotiated 1
//!    (the server-initiated surface is not offered).
//! 2. Server-initiated `decision.classify` over the plugin RPC: request-id
//!    correlation, `timeout_ms` honored, features JSON (no excerpt for the
//!    webhook rule doesn't apply here — plugin is a model kind, the excerpt
//!    rides), and the daemon routes by the plugin's verdict.
//! 3. Failures: v1-only plugin (no v2 session) ⇒ heuristic + `unavailable`;
//!    plugin that never answers ⇒ the per-request timeout row.
//! 4. Review session card-snapshot continuity (spec: "a session snapshots
//!    its card record at start, never re-resolves") through the live daemon
//!    review endpoints.

use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    net::TcpListener,
    os::unix::fs::PermissionsExt,
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

const L3_LOCAL_MODELS: &str = "rayl3p-local";

// ── fixtures ─────────────────────────────────────────────────────────────────

/// Protocol-v2-capable decision fixture. Modes:
/// - `answer`: negotiates v2, answers `decision.classify` with the configured
///   verdict, echoes correlation fields into report.json.
/// - `v1`: initializes with `protocol_version: 1` (no decision surface).
/// - `silent`: negotiates v2 but never answers (timeout row).
const DECISION_FIXTURE_PY: &str = r#"#!/usr/bin/env python3
import json, os, socket, time

here = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(here, "fixture.json")))
pid = cfg["id"]
mode = cfg.get("mode", "answer")
runtime = os.environ["XDG_RUNTIME_DIR"]
token = os.environ.get("BRAINROUTER_PLUGIN_TOKEN", "")
state = os.path.join(os.environ.get("XDG_STATE_HOME", os.path.join(os.environ["HOME"], ".local", "state")),
                     "brainrouter", "plugins", pid)
os.makedirs(state, exist_ok=True)
report_path = os.path.join(state, "report.json")

def flush(report):
    tmp = report_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f)
    os.replace(tmp, report_path)

sock_path = os.path.join(runtime, "brainrouter", "plugins", pid + ".sock")
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(60.0)
s.connect(sock_path)

def line(o):
    return (json.dumps(o) + "\n").encode()

n = [0]
report = {"id": pid, "mode": mode, "requests": []}

def send(o):
    s.sendall(line(o))

def read():
    buf = b""
    while True:
        if b"\n" in buf:
            head, rest = buf.split(b"\n", 1)
            buf = rest
            return json.loads(head.decode())
        chunk = s.recv(65536)
        if not chunk:
            return None
        buf += chunk

def call(method, params):
    n[0] += 1
    send({"jsonrpc": "2.0", "id": n[0], "method": method, "params": params})

# initialize: declare protocol v2 capability (v1 mode declares 1).
call("initialize", {"id": pid, "token": token, "protocol_version": 1 if mode == "v1" else 2})
buf = b""
def read_frame():
    global buf
    while True:
        if b"\n" in buf:
            head, buf2 = buf.split(b"\n", 1)
            globals()["buf"] = buf2
            return json.loads(head.decode())
        chunk = s.recv(65536)
        if not chunk:
            return None
        buf += chunk

def is_result(frame, want_id):
    return (frame is not None and frame.get("id") == want_id
            and ("result" in frame or "error" in frame))

init = read_frame()
while init is not None and not is_result(init, 1):
    init = read_frame()
report["initialize"] = init
flush(report)

# serve loop: answer daemon-initiated requests (responses to our own calls
# are consumed and ignored).
while True:
    frame = read_frame()
    if frame is None:
        break
    if frame.get("method") is None:
        continue
    method = frame.get("method")
    if method == "decision.classify":
        params = frame.get("params", {})
        entry = {
            "request_id": frame.get("id"),
            "params_request_id": params.get("request_id"),
            "timeout_ms": params.get("timeout_ms"),
            "has_features": "features" in params,
            "features_keys": sorted((params.get("features") or {}).keys()),
        }
        report["requests"].append(entry)
        flush(report)
        if mode == "silent":
            continue
        verdict = cfg.get("verdict", {"route": "cloud", "tier": "balanced", "confidence": 0.9, "rationale": "plugin-decided"})
        send({"jsonrpc": "2.0", "id": frame.get("id"), "result": verdict})
    elif "result" in frame or "error" in frame:
        pass
"#;

fn manifest_toml(id: &str, grants: &[&str]) -> String {
    let grants = grants
        .iter()
        .map(|g| format!("\"{g}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "id = \"{id}\"\nname = \"Ray L3 decision fixture {id}\"\nversion = \"0.1.0\"\nbinary = \"fixture.py\"\nmin_daemon_version = \"1.0.0\"\ngrants = [{grants}]\n"
    )
}

// ── mock upstreams ───────────────────────────────────────────────────────────

#[derive(Clone, Default)]
struct MockState {
    hits: Arc<Mutex<Vec<(String, String)>>>,
}

#[derive(Clone)]
struct MockServer {
    url: String,
    state: MockState,
}

impl MockServer {
    fn local(tag: &str, models: &[&str]) -> MockServer {
        Self::start(tag, models.iter().map(|m| m.to_string()).collect(), false)
    }
    fn cloud(tag: &str) -> MockServer {
        Self::start(tag, vec!["vendor/cloud-2026".into()], true)
    }
    fn start(tag: &str, models: Vec<String>, cloud: bool) -> MockServer {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let state = MockState::default();
        let shared = state.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let state = shared.clone();
                let models = models.clone();
                std::thread::spawn(move || {
                    handle(stream, state, models, cloud);
                });
            }
        });
        MockServer {
            url: format!("http://127.0.0.1:{port}"),
            state,
        }
    }
    fn hits(&self) -> Vec<(String, String)> {
        self.state.hits.lock().unwrap().clone()
    }
}

fn read_request(stream: &mut std::net::TcpStream) -> Option<(String, String, Value)> {
    use std::io::Read;
    let mut buf = vec![0u8; 262144];
    let mut text = String::new();
    loop {
        let n = stream.read(&mut buf).ok()?;
        if n == 0 {
            return None;
        }
        text.push_str(&String::from_utf8_lossy(&buf[..n]));
        if text.contains("\r\n\r\n") {
            break;
        }
    }
    let head = text.split("\r\n\r\n").next()?.to_string();
    let first = head.lines().next()?.to_string();
    let mut header_lines = head.lines().skip(1);
    let mut content_length = 0usize;
    for l in header_lines.by_ref() {
        if l.to_ascii_lowercase().starts_with("content-length:") {
            content_length = l.split(':').nth(1)?.trim().parse().unwrap_or(0);
        }
    }
    let mut body = text.split_once("\r\n\r\n").map(|x| x.1.to_string()).unwrap_or_default();
    let mut tmp = [0u8; 8192];
    while body.len() < content_length {
        let n = stream.read(&mut tmp).ok()?;
        if n == 0 {
            break;
        }
        body.push_str(&String::from_utf8_lossy(&tmp[..n]));
    }
    let mut parts = first.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    let json: Value = serde_json::from_str(&body).unwrap_or(json!({}));
    Some((method, path, json))
}

fn write_json(stream: &mut std::net::TcpStream, status: &str, body: &str) {
    let resp = format!(
        "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(resp.as_bytes());
    let _ = stream.flush();
}

fn handle(mut stream: std::net::TcpStream, state: MockState, models: Vec<String>, cloud: bool) {
    let Some((method, path, body)) = read_request(&mut stream) else {
        return;
    };
    if method == "GET" && path.ends_with("/models") {
        let data: Vec<Value> = models.iter().map(|m| json!({"id": m})).collect();
        write_json(&mut stream, "200 OK", &json!({"data": data}).to_string());
        return;
    }
    if method == "POST" && path.ends_with("/chat/completions") {
        let model = body["model"].as_str().unwrap_or("").to_string();
        state.hits.lock().unwrap().push((path, model.clone()));
        let excerpt = body["messages"]
            .as_array()
            .map(|a| {
                a.iter()
                    .rev()
                    .find(|m| m["role"] == "user")
                    .and_then(|m| m["content"].as_str())
                    .unwrap_or("")
                    .to_string()
            })
            .unwrap_or_default();
        let content = if cloud {
            format!("rayl3p-cloud-answer for {excerpt}")
        } else {
            format!("rayl3p-pong {model} for {excerpt}")
        };
        let resp = json!({
            "id": "rayl3p",
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        });
        write_json(&mut stream, "200 OK", &resp.to_string());
        return;
    }
    write_json(&mut stream, "404 Not Found", "{}");
}

// ── daemon harness (L2/L3 suite conventions) ────────────────────────────────

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
        let base = std::env::temp_dir().join(format!("brl3p-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let root = base.join(format!("{:08x}", rand_u32()));
        let env = Self {
            home: root.join("home"),
            config_home: root.join("config"),
            data_home: root.join("data"),
            state_home: root.join("state"),
            runtime: root.join("run"),
            bin: root.join("bin"),
            root,
            owned: true,
        };
        for d in [&env.home, &env.config_home, &env.data_home, &env.state_home, &env.runtime, &env.bin] {
            fs::create_dir_all(d).unwrap();
            if d == &env.runtime {
                fs::set_permissions(d, fs::Permissions::from_mode(0o700)).unwrap();
            }
        }
        link_interpreter(&env.bin, "python3");
        env
    }
    fn apply<I, S>(&self, cmd: &mut Command, extra: I)
    where
        I: IntoIterator<Item = (S, S)>,
        S: AsRef<std::ffi::OsStr>,
    {
        cmd.env_clear()
            .env("HOME", &self.home)
            .env("XDG_CONFIG_HOME", &self.config_home)
            .env("XDG_DATA_HOME", &self.data_home)
            .env("XDG_STATE_HOME", &self.state_home)
            .env("XDG_RUNTIME_DIR", &self.runtime)
            .env("PATH", &self.bin)
            .env("RUST_LOG", "debug")
            .envs(extra);
    }
    fn state_dir(&self) -> PathBuf {
        self.state_home.join("brainrouter")
    }
    fn config_dir(&self) -> PathBuf {
        self.config_home.join("brainrouter")
    }
}

impl Drop for TestEnv {
    fn drop(&mut self) {
        if self.owned {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}

fn rand_u32() -> u32 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (std::time::SystemTime::now().elapsed().ok(), std::process::id()).hash(&mut hasher);
    hasher.finish() as u32
}

fn link_interpreter(dir: &PathBuf, name: &str) {
    let which = Command::new("/usr/bin/env").arg("which").arg(name).output().ok();
    let path = which.and_then(|o| String::from_utf8(o.stdout).ok()).map(|s| s.trim().to_string());
    let Some(p) = path.filter(|p| !p.is_empty()) else { return };
    let _ = std::os::unix::fs::symlink(std::fs::canonicalize(&p).unwrap(), dir.join(name));
}

fn config_yaml(local: &str, cloud: &str, decision: &str) -> String {
    format!(
        r#"manifest:
  base_url: {cloud}/v1
  enabled: true
llama_swap:
  base_url: {local}/v1
  fallback_model: rayl3p-fallback
  local_models:
    - rayl3p-local
bonsai:
  enabled: false
  fork_path: /nonexistent/rayl3p
{decision}review: {{}}
"#
    )
}

struct Daemon {
    child: Child,
    url: String,
    socket: PathBuf,
    log: PathBuf,
    env: TestEnv,
}

impl Daemon {
    fn start(env: TestEnv, config_yaml: &str) -> Self {
        let config_path = env.root.join("config.yaml");
        fs::write(&config_path, config_yaml).unwrap();
        let socket = env.root.join("d.sock");
        let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = reservation.local_addr().unwrap();
        drop(reservation);
        let log = env.root.join("daemon.log");
        let log_file = fs::File::create(&log).unwrap();
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_brainrouter"));
        cmd.args(["serve", "--config"])
            .arg(&config_path)
            .args(["--tcp-addr", &address.to_string(), "--socket"])
            .arg(&socket)
            .current_dir(&env.root)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log_file.try_clone().unwrap()))
            .stderr(Stdio::from(log_file));
        env.apply(&mut cmd, std::iter::empty::<(String, String)>());
        let child = cmd.spawn().unwrap();
        Self { child, url: format!("http://{address}"), socket, log, env }
    }
    async fn ready(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                panic!("daemon exited with {status}: {}", fs::read_to_string(&self.log).unwrap());
            }
            if let Ok(r) = client().get(format!("{}/health", self.url)).send().await {
                if r.status().is_success() {
                    return;
                }
            }
            assert!(Instant::now() < deadline, "daemon not healthy: {}", fs::read_to_string(&self.log).unwrap());
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    fn cli(&self, args: &[&str]) -> std::process::Output {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_brainrouter"));
        cmd.args(["cli", "--socket", &self.socket.to_string_lossy().to_string()]).args(args);
        self.env.apply(&mut cmd, [("TMPDIR", self.env.root.as_path().to_str().unwrap())]);
        cmd.output().unwrap()
    }
    fn stop(&mut self) {
        let pid = self.child.id() as i32;
        unsafe { libc::kill(pid, libc::SIGTERM) };
        let deadline = Instant::now() + Duration::from_secs(8);
        loop {
            if self.child.try_wait().unwrap().is_some() {
                return;
            }
            if Instant::now() >= deadline {
                // Supervisor-spawned plugin children can keep the pipe open;
                // the documented escalation is kill after the grace period.
                unsafe { libc::kill(pid, libc::SIGKILL) };
                let _ = self.child.wait();
                return;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        self.stop();
    }
}

async fn chat(daemon: &Daemon, model: &str, prompt: &str) -> (u16, String) {
    let response = client()
        .post(format!("{}/v1/chat/completions", daemon.url))
        .json(&json!({"model": model, "stream": true, "messages": [{"role": "user", "content": prompt}]}))
        .send()
        .await
        .unwrap();
    let status = response.status().as_u16();
    (status, response.text().await.unwrap())
}

async fn events(daemon: &Daemon) -> Vec<Value> {
    let v = daemon.get_json("/api/routing-events").await;
    if let Some(list) = v.as_array() {
        return list.clone();
    }
    v["events"].as_array().cloned().unwrap_or_default()
}

impl Daemon {
    async fn get_json(&self, path: &str) -> Value {
        client().get(format!("{}{path}", self.url)).send().await
            .unwrap_or_else(|e| panic!("GET {path}: {e}"))
            .json().await
            .unwrap_or_else(|e| panic!("GET {path} json: {e}"))
    }
}

fn client() -> reqwest::Client {
    reqwest::Client::builder().timeout(Duration::from_secs(15)).no_proxy().build().unwrap()
}

async fn event_for(daemon: &Daemon, prompt: &str) -> Value {
    for _ in 0..40 {
        let found = events(daemon)
            .await
            .into_iter()
            .find(|e| e["prompt_excerpt"].as_str().unwrap_or("") == prompt);
        if let Some(e) = found {
            return e;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("no event for {prompt}");
}

fn write_plugin(env: &TestEnv, id: &str, mode: &str, verdict: Value) -> PathBuf {
    let dir = env.root.join("fixtures").join(id);
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("plugin.toml"), manifest_toml(id, &["chat.completions"])).unwrap();
    let script = dir.join("fixture.py");
    fs::write(&script, DECISION_FIXTURE_PY).unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    fs::write(
        dir.join("fixture.json"),
        json!({"id": id, "mode": mode, "verdict": verdict}).to_string(),
    )
    .unwrap();
    dir
}

fn poll_report(env: &TestEnv, id: &str, timeout: Duration, pred: impl Fn(&Value) -> bool) -> Value {
    let path = env.state_dir().join("plugins").join(id).join("report.json");
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(text) = fs::read_to_string(&path) {
            if let Ok(v) = serde_json::from_str::<Value>(&text) {
                if pred(&v) {
                    return v;
                }
            }
        }
        assert!(Instant::now() < deadline, "no qualifying report at {}: {:?}", path.display(), fs::read_to_string(&path));
        std::thread::sleep(Duration::from_millis(100));
    }
}

// ── tests ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn plugin_decision_negotiates_v2_and_routes_by_verdict() {
    let env = TestEnv::new("plugv2");
    let local = MockServer::local("plugv2-local", &[L3_LOCAL_MODELS]);
    let cloud = MockServer::cloud("plugv2-cloud");
    let mut daemon = Daemon::start(
        env,
        &config_yaml(&local.url, &cloud.url, "decision_model:\n  enabled: true\n  kind: plugin\n  plugin_id: rayl3plug\n  timeout_ms: 4000\n"),
    );
    daemon.ready().await;
    let dir = {
        let e = &daemon.env;
        write_plugin(e, "rayl3plug", "answer", json!({"route": "cloud", "tier": "deep", "confidence": 0.8, "rationale": "plugin-decided"}))
    };
    let out = daemon.cli(&["plugin", "install", dir.to_str().unwrap()]);
    assert!(out.status.success(), "install: {}", String::from_utf8_lossy(&out.stderr));
    let out = daemon.cli(&["plugin", "enable", "rayl3plug"]);
    assert!(out.status.success(), "enable: {}", String::from_utf8_lossy(&out.stderr));

    // Negotiation: initialize declares v2; the daemon answers protocol_version
    // 2 + advertises the server-initiated surface.
    let report = poll_report(&daemon.env, "rayl3plug", Duration::from_secs(20), |v| {
        v.get("initialize").is_some()
    });
    let init = &report["initialize"]["result"];
    assert_eq!(init["protocol_version"], json!(2), "protocol negotiated at initialize (v2)");
    assert!(
        init["server_methods"].as_array().map(|m| m.iter().any(|x| x == "decision.classify")).unwrap_or(false),
        "v2 must advertise decision.classify; got {init}"
    );

    // Server-initiated decision: plugin says cloud → the request lands on
    // cloud with tier deep carried unchanged; the event carries the plugin
    // rationale, no fallback.
    let (status, text) = chat(&daemon, "brainrouter/auto", "r3p-plugin-cloud").await;
    assert_eq!(status, 200, "{text}");
    assert!(text.contains("rayl3p-cloud-answer"), "cloud served per plugin verdict: {text}");
    assert_eq!(cloud.hits().last().unwrap().1, "auto", "cloud decisions carry the managed model");
    assert_eq!(local.hits().len(), 0, "plugin said cloud: nothing local");

    // Request-id correlation + timeout_ms + features (model kind: excerpt
    // allowed) observed by the plugin.
    let report = poll_report(&daemon.env, "rayl3plug", Duration::from_secs(10), |v| {
        v["requests"].as_array().map(|r| !r.is_empty()).unwrap_or(false)
    });
    let req = &report["requests"][0];
    assert_eq!(req["request_id"], req["params_request_id"], "request-id correlation");
    assert_eq!(req["timeout_ms"], json!(4000));
    assert_eq!(req["has_features"], json!(true));
    let keys: Vec<&str> = req["features_keys"].as_array().unwrap().iter().map(|k| k.as_str().unwrap()).collect();
    for k in ["role", "privacy", "quality", "msg_count", "approx_prompt_tokens", "has_tools", "code_density"] {
        assert!(keys.contains(&k), "features must carry {k}; got {keys:?}");
    }

    let e = event_for(&daemon, "r3p-plugin-cloud").await;
    assert_eq!(e["role"], json!("main"));
    assert_eq!(e["decision"]["route"], json!("cloud"));
    assert_eq!(e["decision"]["tier"], json!("deep"), "plugin tier carried unchanged");
    assert!(e["decision"]["confidence"].is_number());
    assert!(e["rationale"].as_str().unwrap().contains("plugin-decided"), "rationale carries the plugin verdict: {}", e["rationale"]);
    assert!(e["decision_fallback"].is_null() || e.get("decision_fallback").is_none());
    daemon.stop();
}

#[tokio::test]
async fn plugin_decision_v1_only_plugin_falls_back_unavailable() {
    let env = TestEnv::new("plugv1");
    let local = MockServer::local("plugv1-local", &[L3_LOCAL_MODELS]);
    let cloud = MockServer::cloud("plugv1-cloud");
    let mut daemon = Daemon::start(
        env,
        &config_yaml(&local.url, &cloud.url, "decision_model:\n  enabled: true\n  kind: plugin\n  plugin_id: rayl3plug1\n  timeout_ms: 4000\n"),
    );
    daemon.ready().await;
    let dir = {
        let e = &daemon.env;
        write_plugin(e, "rayl3plug1", "v1", Value::Null)
    };
    daemon.cli(&["plugin", "install", dir.to_str().unwrap()]);
    daemon.cli(&["plugin", "enable", "rayl3plug1"]);
    let report = poll_report(&daemon.env, "rayl3plug1", Duration::from_secs(20), |v| {
        v.get("initialize").is_some()
    });
    assert_eq!(report["initialize"]["result"]["protocol_version"], json!(1), "v1 declares 1 ⇒ negotiated 1");
    assert!(report["initialize"]["result"].get("server_methods").is_none());

    // A v1-only session speaks no decision.classify ⇒ the fresh-config rule:
    // heuristic + fallback=unavailable.
    let (status, text) = chat(&daemon, "brainrouter/auto", "r3p-plugin-v1").await;
    assert_eq!(status, 200, "{text}");
    assert!(text.contains("rayl3p-pong rayl3p-fallback"), "heuristic parity local: {text}");
    assert_eq!(cloud.hits().len(), 0);
    let e = event_for(&daemon, "r3p-plugin-v1").await;
    assert_eq!(e["decision"]["route"], json!("local"));
    assert_eq!(e["decision_fallback"], json!("unavailable"));
    daemon.stop();
}

#[tokio::test]
async fn plugin_decision_timeout_is_unavailable_fallback() {
    let env = TestEnv::new("plugto");
    let local = MockServer::local("plugto-local", &[L3_LOCAL_MODELS]);
    let cloud = MockServer::cloud("plugto-cloud");
    let mut daemon = Daemon::start(
        env,
        &config_yaml(&local.url, &cloud.url, "decision_model:\n  enabled: true\n  kind: plugin\n  plugin_id: rayl3plugto\n  timeout_ms: 1200\n"),
    );
    daemon.ready().await;
    let dir = {
        let e = &daemon.env;
        write_plugin(e, "rayl3plugto", "silent", Value::Null)
    };
    daemon.cli(&["plugin", "install", dir.to_str().unwrap()]);
    daemon.cli(&["plugin", "enable", "rayl3plugto"]);
    poll_report(&daemon.env, "rayl3plugto", Duration::from_secs(20), |v| {
        v.get("initialize").is_some()
    });
    let start = Instant::now();
    let (status, text) = chat(&daemon, "brainrouter/auto", "r3p-plugin-timeout").await;
    assert_eq!(status, 200, "{text}");
    assert!(text.contains("rayl3p-pong rayl3p-fallback"), "timeout ⇒ heuristic local: {text}");
    // timeout_ms (1200) + 500 ms slack bounds the request; bounded, not hung.
    assert!(start.elapsed() < Duration::from_secs(10), "decision request must time out, not hang");
    let e = event_for(&daemon, "r3p-plugin-timeout").await;
    assert_eq!(e["decision"]["route"], json!("local"));
    assert_eq!(e["decision_fallback"], json!("unavailable"));
    daemon.stop();
}

#[tokio::test]
async fn review_session_snapshots_its_card_and_never_re_resolves() {
    // Spec: "a session snapshots its card record at start, never
    // re-resolves". Live-daemon path: configure the review card cloud→
    // manifest, start a review, rotate the card to a different cloud model,
    // continue the session, and prove the ORIGINAL card served both passes.
    let env = TestEnv::new("cardsnap");
    let local = MockServer::local("cardsnap-local", &[L3_LOCAL_MODELS]);
    let cloud = MockServer::cloud("cardsnap-cloud");
    let mut daemon = Daemon::start(env, &config_yaml(&local.url, &cloud.url, "decision_model:\n  enabled: false\n  kind: heuristic\n"));
    daemon.ready().await;
    // Review card: explicit cloud id.
    let out = daemon.cli(&["roles", "set", "review", "cloud/vendor/review-snap-1"]);
    assert!(out.status.success(), "roles set: {}", String::from_utf8_lossy(&out.stderr));

    let summary = "card snapshot continuity review".to_string();
    let body = json!({
        "taskId": "cardsnap-1", "summary": summary,
        "details": "The fixture answers approved.",
        "conversationHistory": ["ctx"],
        "cwd": "/tmp",
    });
    let started: Value = client()
        .post(format!("{}/review/api/request-async", daemon.url))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let session_id = started["sessionId"].as_str().unwrap().to_string();
    for _ in 0..80 {
        let status = daemon.get_json(&format!("/review/api/sessions/{session_id}")).await;
        if status["status"].as_str().map(|s| s == "approved").unwrap_or(false) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let cloud_hits = cloud.hits();
    assert_eq!(cloud_hits.last().unwrap().1, "vendor/review-snap-1", "first pass on the card");

    // Rotate the card; the running session must NOT re-resolve.
    daemon.cli(&["roles", "set", "review", "cloud/vendor/review-snap-2"]);
    // The stored session snapshot keeps the ORIGINAL card selector after the
    // rotation (session detail exposes review_selector); a NEW review picks up
    // the rotated card. Continuation reuses the snapshot (service-level
    // continuity: routing_profiles_test::project_reviewer_pin_enters_the_
    // snapshot_and_continuations_keep_it).
    // Poll: the snapshot persists when the session reaches its terminal
    // state (the mock's unparseable replies escalate fast).
    let mut detail = json!({});
    for _ in 0..80 {
        detail = daemon
            .get_json(&format!("/review/api/sessions/{session_id}"))
            .await;
        if detail.get("review_selector").is_some() || detail.get("reviewSelector").is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(
        detail["reviewSelector"].as_str().or(detail["review_selector"].as_str()),
        Some("cloud/vendor/review-snap-1"),
        "stored card snapshot survives rotation: {detail}"
    );
    assert!(
        detail["review_model"].as_str().unwrap_or_default().contains("vendor/review-snap-1"),
        "the running session stayed on the snapshotted card: {detail}"
    );
    let before = cloud.hits().len();
    let second: Value = client()
        .post(format!("{}/review/api/request-async", daemon.url))
        .json(&json!({
            "taskId": "cardsnap-2", "summary": "second review",
            "details": "post-rotation", "conversationHistory": [], "cwd": "/tmp",
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let second_id = second["sessionId"].as_str().unwrap().to_string();
    for _ in 0..150 {
        if cloud.hits().len() > before {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(
        cloud.hits().last().unwrap().1,
        "vendor/review-snap-2",
        "the NEW review resolves the rotated card"
    );
    let _ = second_id;
    daemon.stop();
}

// ── L3 boot-window contract (startup/shutdown race regressions) ─────────────

/// The listeners are bound BEFORE the boot work (migration, plugin
/// registry, model resolution): the instant TCP `/health` answers, the UDS
/// socket file must exist, so a CLI invoked in the startup window is
/// served, not refused ("Daemon socket not found" flakes under load).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn daemon_serves_cli_verb_the_instant_health_answers() {
    let env = TestEnv::new("bootwindow");
    let local = MockServer::local("boot-local", &[L3_LOCAL_MODELS]);
    let cloud = MockServer::cloud("boot-cloud");
    let cfg = config_yaml(
        &local.url,
        &cloud.url,
        "decision_model:\n  enabled: true\n  kind: heuristic\n",
    );
    for _ in 0..5 {
        let mut daemon = Daemon::start(env_borrow(&env), &cfg);
        daemon.ready().await;
        // NO settle sleep: this is the boot window.
        let out = daemon.cli(&["roles", "get"]);
        assert!(
            out.status.success(),
            "CLI in the boot window failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        daemon.stop();
    }
}

/// SIGTERM during boot (migration running, supervisors mid-spawn): the
/// child must not escape as an orphan (ppid 1) — the shutdown sweep
/// reaps pids registered after the supervisor enumeration.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sigterm_during_boot_does_not_orphan_plugin_children() {
    let _guard = BOOT_SIGTERM_LOCK.lock().unwrap();
    let env = TestEnv::new("bootsigterm");
    let local = MockServer::local("bs-local", &[L3_LOCAL_MODELS]);
    let cloud = MockServer::cloud("bs-cloud");
    for _i in 0..3 {
        // Round 1 installs+enables; rounds 2-3 boot with the plugin
        // already enabled (registry persists) — the boot spawn path.
        let mut daemon = Daemon::start(env_borrow(&env), &config_yaml(&local.url, &cloud.url, "decision_model:\n  enabled: true\n  kind: plugin\n  plugin_id: raybsplug\n  timeout_ms: 2000\n"));
        daemon.ready().await;
        if _i == 0 {
            let dir = env.root.join("fixtures/raybsplug");
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("plugin.toml"), manifest_toml("raybsplug", &["chat.completions"])).unwrap();
            let script = dir.join("fixture.py");
            fs::write(&script, DECISION_FIXTURE_PY).unwrap();
            fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
            fs::write(
                dir.join("fixture.json"),
                json!({"id": "raybsplug", "mode": "v2", "verdict": null, "idle": true}).to_string(),
            )
            .unwrap();
            let run = daemon.cli(&["plugin", "install", dir.to_str().unwrap()]);
            assert!(run.status.success(), "install: {}", String::from_utf8_lossy(&run.stderr));
            let run = daemon.cli(&["plugin", "enable", "raybsplug"]);
            assert!(run.status.success(), "enable: {}", String::from_utf8_lossy(&run.stderr));
        }
        daemon.stop(); // SIGTERM: the enabled plugin supervisor is mid-spawn

        // The fixture child (pid file heartbeats) must be gone shortly
        // after daemon exit; scan for live children of the dead daemon.
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let orphans = alive_plugin_pids();
            if orphans.is_empty() {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "plugin children survived daemon shutdown: {orphans:?}"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

static BOOT_SIGTERM_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Pids of running plugin fixture processes (the python decision fixture
/// under this suite's state roots).
fn alive_plugin_pids() -> Vec<u32> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/proc") {
        for e in entries.flatten() {
            let name = e.file_name();
            let pid: u32 = match name.to_str().and_then(|s| s.parse().ok()) {
                Some(p) => p,
                None => continue,
            };
            let Ok(cmd) = std::fs::read(format!("/proc/{pid}/cmdline")) else {
                continue;
            };
            let cmd = String::from_utf8_lossy(&cmd).replace('\0', " ");
            if cmd.contains("raybsplug") {
                out.push(pid);
            }
        }
    }
    out
}

fn env_borrow(env: &TestEnv) -> TestEnv {
    TestEnv {
        root: env.root.clone(),
        home: env.home.clone(),
        config_home: env.config_home.clone(),
        data_home: env.data_home.clone(),
        state_home: env.state_home.clone(),
        runtime: env.runtime.clone(),
        bin: env.bin.clone(),
        owned: false,
    }
}

// ── Step 9 B3: the bridge !br review command's daemon contract ─────────────

/// `!br review <mode>` (bridge) speaks: POST /api/roles/set
/// {role: review, assignment: <mode>} | POST /api/roles/remove
/// {role: review} (auto), readback GET /api/roles → assignments
/// [{scope, role, assignment}]. This test pins that round-trip against the
/// live daemon (the bridge binary itself is exercised by Ray's suites).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn br_review_command_roles_round_trip() {
    let env = TestEnv::new("brreview");
    let local = MockServer::local("brv-local", &[L3_LOCAL_MODELS]);
    let cloud = MockServer::cloud("brv-cloud");
    let mut daemon = Daemon::start(
        env_borrow(&env),
        &config_yaml(&local.url, &cloud.url, "decision_model:\n  enabled: true\n  kind: heuristic\n"),
    );
    daemon.ready().await;
    let base = daemon.url.clone();
    let client = client();

    async fn read_review(client: &reqwest::Client, base: &str) -> Value {
        client
            .get(format!("{base}/api/roles"))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap()
    }

    // Default: no review card -> the readback renders "auto" (absent).
    let roles = read_review(&client, &base).await;
    let review_of = |roles: &Value| -> String {
        roles
            .get("assignments")
            .and_then(|a| a.as_array())
            .and_then(|arr| {
                arr.iter().find(|c| {
                    c.get("role").and_then(|r| r.as_str()) == Some("review")
                        && c.get("scope").and_then(|s| s.as_str()) == Some("global")
                })
            })
            .and_then(|c| c.get("assignment").and_then(|a| a.as_str()))
            .unwrap_or("auto")
            .to_string()
    };
    assert_eq!(review_of(&roles), "auto", "fresh daemon: no review card");

    for mode in ["local", "cloud", "cloud/vendor/review-bridge-1"] {
        let r = client
            .post(format!("{base}/api/roles/set"))
            .json(&json!({"role": "review", "assignment": mode}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "!br review {mode}");
        let roles = read_review(&client, &base).await;
        assert_eq!(review_of(&roles), mode, "readback after set {mode}");
    }

    // `auto` = remove (the card default).
    let r = client
        .post(format!("{base}/api/roles/remove"))
        .json(&json!({"role": "review"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "!br review auto");
    let roles = read_review(&client, &base).await;
    assert_eq!(review_of(&roles), "auto", "readback after remove");
    daemon.stop();
}
