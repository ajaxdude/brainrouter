//! Acceptance tests for leg L2 (Plugins + bridge pilot + minimal credentials
//! store) of `docs/design/v2-streamlining.md` — spec:
//! `docs/design/v2-streamlining.L2.md` at commit 64ca696.
//!
//! Written by Ray from the design's promises alone. The builder runs these and
//! NEVER edits them. Expectation: every test is RED at the design base commit
//! (no plugin subsystem exists) and GREEN once L2 lands.
//!
//! Mapping of tests to the L2 "Acceptance (Ray)" items:
//! - (1) RPC contract over a real UDS:
//!   * `rpc_initialize_good_token_returns_grants` — initialize + grants returned
//!   * `rpc_bad_token_connect_refused_not_counted_as_crash` — bad token closed
//!     + logged, not counted toward crash restarts
//!   * `rpc_config_kv_namespace_isolation_across_plugins` — method-grant matrix
//!     part 1: namespaced config/kv (A cannot read B, and vice versa)
//!   * `rpc_secrets_get_grant_matrix` — method-grant matrix part 2: ungranted
//!     `secrets.get` refused (no method grant, and credential not granting the
//!     plugin), granting credential returned
//!   * `rpc_oversized_body_rejected` — body cap 1 MiB
//!   * `rpc_unknown_method_minus_32601` — unknown method -> -32601
//!   * `rpc_inflight_cap_six_concurrent_chat_completions` — per-plugin
//!     in-flight cap: 6 rapid concurrent chat.completions, minimal boundary
//!     (>=2 succeed; excess queued-served or JSON-RPC limit error per spec
//!     wording); at least: no daemon crash, all responses JSON-RPC-shaped
//!   * `rpc_socket_layout_and_permissions` — UDS dir 0700 / socket 0600, token
//!     file 0600 (Plugin core + RPC v1 surface contracts)
//! - (2) Supervisor:
//!   * `supervisor_crash_loop_reaches_terminal_failed` — repeated nonzero exit
//!     -> terminal `failed` (status shows failed), spawn stops (log/spawn line
//!     count ceiling), daemon keeps serving /health throughout
//!   * `supervisor_stops_plugin_children_on_daemon_shutdown` — SIGTERM-then-kill
//!     stop contract: daemon shutdown stops the plugin child
//! - (3) Credentials:
//!   * `credentials_v1_migration_store_perms_grants_and_masking` — boot with v1
//!     YAML `bridge.discord.token` -> `credentials.json` 0600 with a migrated
//!     credential granting `bridge`; token absent from `config show`, API
//!     responses, plugin config files, and daemon logs; logged migration
//!   * `credentials_plugin_secrets_get_round_trip` — plugin fixture resolves
//!     the migrated token via `secrets.get`, value equal to the YAML value
//!   * `credentials_dry_run_preview_leaves_state_untouched` — `--dry-run`
//!     preview shows the migration, state untouched
//! - (4) Bridge plugin command matrix: SKIPPED — requires the shipped bridge
//!   plugin binary; reported to Hank as builder-integration coverage (the
//!   fixtures here cover the RPC/supervisor/credentials contracts with the
//!   plugin id `bridge` standing in for the real plugin's grant wiring).
//! - (5) Cargo tree:
//!   * `core_build_has_no_serenity` — run-free text assertion on Cargo.toml +
//!     Cargo.lock: no bridge features in the default set, serenity (and the
//!     other bridge-only deps named in "Core removals") gone.
//!
//! SPEC GAPS (assumptions this file must make to be runnable; all reported to
//! Hank — the design fixes the promises, not these wire shapes):
//! - RPC framing: the spec says "JSON-RPC 2.0" over UDS with no framing rule.
//!   These tests speak newline-delimited JSON (one object per line), the
//!   conventional stream framing. If L2 ships length-prefixed frames, that is a
//!   design amendment, not a test bug.
//! - `plugin.toml` key spellings for the manifest fields the design names in
//!   prose (`id`, `name`, `version`, `binary`, `min_daemon_version`, "RPC
//!   grants", "config-schema path"): `grants` (array of RPC method names) is
//!   used; the config-schema path is omitted (fixtures assert no schema).
//! - Params of `config.set`/`config.get`/`kv.set`/`kv.get`/`secrets.get`:
//!   `{"key"/"name", "value"}` and `{"id"/"credential_id"/"key"/"name"}`
//!   (all aliases sent; extra params members are inert).
//! - The credential id that the Discord-token migration creates is not named
//!   in the design; the fixture probes a candidate list and the test fails
//!   with an explicit "id unresolvable" message if none matches.
//! - CLI verb location: v1's daemon-client CLI lives under `brainrouter cli
//!   ...`; the design writes `brainrouter plugin install <dir>`. Tests try
//!   `cli --socket S plugin ...` first, then the bare `plugin ...` form.
//! - Peer-uid rejection cannot be exercised by an unprivileged test (needs a
//!   foreign-uid peer); not covered here — reported to Hank.

use serde_json::{json, Value};
use std::fs;
use std::io::Write as _;
use std::net::TcpListener;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

// ── Fixture constants ───────────────────────────────────────────────────────

/// Unique-ish sentinel used as the v1 Discord token. Every "value absent"
/// assertion greps for this exact string.
const DISCORD_TOKEN: &str = "ray-L2-s3cr3t-discord-tok-9f3c21";

/// Credential ids the secrets.get fixture probes, in order. The design names
/// the migration ("v1 plaintext bridge.discord.token -> store credential
/// granted to bridge") but not the resulting credential id; a resolvable id
/// under any of these spellings satisfies the promise.
const CREDENTIAL_ID_CANDIDATES: &[&str] = &[
    "bridge.discord.token",
    "discord",
    "discord-token",
    "bridge-discord",
    "discord_token",
    "bridge.discord",
    "bridge",
    "token",
];

/// Long-lived RPC fixture plugin. Spawned by the supervisor with
/// BRAINROUTER_PLUGIN_TOKEN + the daemon's environment (the test sets
/// XDG_RUNTIME_DIR / XDG_STATE_HOME). Reads a `fixture.json` sidecar next to
/// the script for its id/mode, connects to the per-plugin UDS socket, speaks
/// `initialize`, writes `<state>/report.json`, then stays resident.
const RPC_FIXTURE_PY: &str = r#"#!/usr/bin/env python3
import json, os, socket, sys, time

here = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(here, "fixture.json")))
pid = cfg["id"]
mode = cfg.get("mode", "basic")
runtime = os.environ["XDG_RUNTIME_DIR"]
token = os.environ.get("BRAINROUTER_PLUGIN_TOKEN", "")
state = os.path.join(os.environ.get("XDG_STATE_HOME", os.path.join(os.environ["HOME"], ".local", "state")),
                     "brainrouter", "plugins", pid)
os.makedirs(state, exist_ok=True)

def line(o):
    return (json.dumps(o) + "\n").encode()

if mode == "crash":
    # Supervisor crash-loop fixture: count spawns in the state dir, exit
    # nonzero immediately. Runs before any RPC so a slow socket bind cannot
    # mask the crash.
    with open(os.path.join(state, "spawns"), "a") as f:
        f.write("%f %d\n" % (time.time(), os.getpid()))
    sys.exit(3)

sock_path = os.path.join(runtime, "brainrouter", "plugins", pid + ".sock")
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(15.0)
s.connect(sock_path)

n = [0]
def call(method, params):
    n[0] += 1
    s.sendall(line({"jsonrpc": "2.0", "id": n[0], "method": method, "params": params}))
    buf = b""
    while b"\n" not in buf:
        chunk = s.recv(65536)
        if not chunk:
            return None
        buf += chunk
    return json.loads(buf.split(b"\n", 1)[0].decode())

report = {"id": pid, "mode": mode, "pid": os.getpid(), "env_token_present": bool(token)}
try:
    report["initialize"] = call("initialize",
                                {"id": pid, "plugin_id": pid, "name": pid, "token": token})
    if mode == "secrets":
        found = {"found_id": None, "value": None, "tried": {}}
        for c in cfg.get("credential_ids", []):
            r = call("secrets.get", {"id": c, "credential_id": c, "key": c, "name": c})
            found["tried"][c] = r
            if r and "result" in r:
                txt = json.dumps(r["result"])
                if txt not in ("", "null", '""', "[]", "{}"):
                    found["found_id"] = c
                    found["value"] = txt
                    break
        report["secrets"] = found
except Exception as e:  # noqa: BLE001 - fixture: record and stay resident
    report["error"] = repr(e)

with open(os.path.join(state, "report.json"), "w") as f:
    json.dump(report, f)

# Resident: the supervisor stop tests need a live child; a SIGTERM kills the
# default-ignored sleep loop.
while True:
    time.sleep(0.5)
"#;

/// Crash-loop fixture: counts spawns, exits nonzero immediately.
const CRASH_FIXTURE_PY: &str = r#"#!/usr/bin/env python3
import json, os, sys, time

here = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(here, "fixture.json")))
pid = cfg["id"]
state = os.path.join(os.environ.get("XDG_STATE_HOME", os.path.join(os.environ["HOME"], ".local", "state")),
                     "brainrouter", "plugins", pid)
os.makedirs(state, exist_ok=True)
with open(os.path.join(state, "spawns"), "a") as f:
    f.write("%f %d\n" % (time.time(), os.getpid()))
sys.exit(3)
"#;

/// Long-lived plain child (no RPC at all) for the shutdown contract.
const ALIVE_FIXTURE_PY: &str = r#"#!/usr/bin/env python3
import os, time

here = os.path.dirname(os.path.abspath(__file__))
cfg = __import__("json").load(open(os.path.join(here, "fixture.json")))
pid = cfg["id"]
state = os.path.join(os.environ.get("XDG_STATE_HOME", os.path.join(os.environ["HOME"], ".local", "state")),
                     "brainrouter", "plugins", pid)
os.makedirs(state, exist_ok=True)
# Heartbeat: rewrite the pid file every tick so a test can prove THIS
# (current boot's) supervisor spawned the live child, not a stale file.
while True:
    with open(os.path.join(state, "pid"), "w") as f:
        f.write(str(os.getpid()))
    time.sleep(0.5)
"#;

fn manifest_toml(id: &str, binary: &str, grants: &[&str]) -> String {
    let grants = grants
        .iter()
        .map(|g| format!("\"{g}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "id = \"{id}\"\nname = \"Ray L2 fixture {id}\"\nversion = \"0.1.0\"\nbinary = \"{binary}\"\nmin_daemon_version = \"1.0.0\"\ngrants = [{grants}]\n"
    )
}

const ALL_GRANTS: &[&str] = &[
    "chat.completions",
    "models.list",
    "routing.events.tail",
    "inflight.list",
    "config.get",
    "config.set",
    "kv.get",
    "kv.set",
    "secrets.get",
    "notify",
    "set_status",
];
/// rayB: config/kv only — no `secrets.get` method grant.
const CONFIG_KV_GRANTS: &[&str] = &["config.get", "config.set", "kv.get", "kv.set"];
/// The `bridge` fixture: secrets only — the migrated credential grants to
/// plugin id `bridge`, which is exactly the surface it needs.
const SECRETS_GRANTS: &[&str] = &["secrets.get"];
const STATUS_GRANTS: &[&str] = &["set_status"];

// ── Test environment harness ────────────────────────────────────────────────

/// Isolated test root: HOME / XDG_CONFIG / XDG_DATA / XDG_STATE /
/// XDG_RUNTIME + a PATH bin dir containing symlinks for the fixture
/// interpreters (mirrors the empty-PATH convention of
/// `tests/daemon_availability_test.rs` / `v2_l1_acceptance.rs` so no podman /
/// omp / llama-swap is reachable).
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
        let base = std::env::temp_dir().join(format!("brl2-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let root = base.join(unique_suffix());
        let env = Self {
            home: root.join("home"),
            config_home: root.join("config"),
            data_home: root.join("data"),
            state_home: root.join("state"),
            runtime: root.join("run"),
            bin: root.join("bin"),
            root: root.clone(),
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

    /// `$XDG_STATE_HOME/brainrouter/plugins/<id>` (design: plugin state dir).
    fn plugin_state(&self, id: &str) -> PathBuf {
        self.state_home.join("brainrouter").join("plugins").join(id)
    }

    /// `$XDG_RUNTIME_DIR/brainrouter/plugins/<id>.sock` (design: per-plugin
    /// UDS server).
    fn plugin_socket(&self, id: &str) -> PathBuf {
        self.runtime.join("brainrouter").join("plugins").join(format!("{id}.sock"))
    }
}

impl Drop for TestEnv {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn unique_suffix() -> String {
    use std::sync::atomic::AtomicU64;
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

fn link_interpreter(bin: &Path, name: &str) {
    for cand in [
        format!("/usr/bin/{name}"),
        format!("/bin/{name}"),
        format!("/usr/local/bin/{name}"),
    ] {
        let p = Path::new(&cand);
        if p.exists() {
            std::os::unix::fs::symlink(p, bin.join(name)).expect("symlink interpreter");
            return;
        }
    }
    panic!("no {name} interpreter found on the test host; the plugin fixtures cannot run");
}

/// v1 llama-swap URL: an unused loopback port (connection refused = "dead"),
/// mirroring `dead_upstream_url()` in v2_l1_acceptance.rs.
fn dead_upstream_url() -> String {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let a = l.local_addr().unwrap();
    drop(l);
    format!("http://{a}")
}

fn base_config_yaml(llama_swap_base_url: &str) -> String {
    serde_yaml::to_string(&json!({
        "manifest": { "base_url": format!("{llama_swap_base_url}/cloud/v1"), "enabled": false },
        "llama_swap": { "base_url": format!("{llama_swap_base_url}/v1"), "fallback_model": "test-only-model" },
        "bonsai": { "enabled": false, "fork_path": "/nonexistent/brainrouter-test-no-server" },
        "review": {},
    }))
    .unwrap()
}

/// v1 YAML: the base config plus the plaintext Discord token key that L2 must
/// migrate out of the config (`bridge.discord.token`, `src/config.rs`).
fn v1_yaml_with_discord_token(llama_swap_base_url: &str) -> String {
    serde_yaml::to_string(&json!({
        "manifest": { "base_url": format!("{llama_swap_base_url}/cloud/v1"), "enabled": false },
        "llama_swap": { "base_url": format!("{llama_swap_base_url}/v1"), "fallback_model": "test-only-model" },
        "bonsai": { "enabled": false, "fork_path": "/nonexistent/brainrouter-test-no-server" },
        "review": {},
        "bridge": {
            "omp_path": "/nonexistent/omp",
            "discord": { "enabled": false, "token": DISCORD_TOKEN },
            "signal": { "enabled": false },
        },
    }))
    .unwrap()
}

// ── Daemon harness (live-daemon conventions from daemon_availability_test /
//    v2_l1_acceptance) ──────────────────────────────────────────────────────

struct Daemon {
    child: Child,
    url: String,
    log: PathBuf,
    socket: PathBuf,
    client: reqwest::Client,
    env: TestEnv,
}

impl Daemon {
    fn start(env: &TestEnv, config_yaml: &str) -> Self {
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
            .args(["--tcp-addr", address.to_string().as_str(), "--socket"])
            .arg(&socket)
            .current_dir(&env.root)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log_file.try_clone().unwrap()))
            .stderr(Stdio::from(log_file));
        env.apply(&mut cmd, std::iter::empty::<(String, String)>());
        let child = cmd.spawn().unwrap();
        Self {
            child,
            url: format!("http://{address}"),
            log,
            socket,
            client: reqwest::Client::builder().no_proxy().timeout(Duration::from_secs(10)).build().unwrap(),
            env: TestEnv::borrow(env),
        }
    }

    fn borrow(env: &TestEnv) -> TestEnv {
        // Cheap clone of path fields (no fs work; the real root is owned by the
        // test's binding, not the harness, so Drop of the borrow is a no-op).
        TestEnv {
            root: env.root.clone(),
            home: env.home.clone(),
            config_home: env.config_home.clone(),
            data_home: env.data_home.clone(),
            state_home: env.state_home.clone(),
            runtime: env.runtime.clone(),
            bin: env.bin.clone(),
        }
    }

    async fn ready(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                panic!("daemon exited with {status}: {}", self.log_tail());
            }
            if let Ok(r) = self.client.get(format!("{}/health", self.url)).send().await {
                if r.status().is_success()
                    && r.json::<Value>().await.unwrap() == json!({"status": "ok"})
                {
                    return;
                }
            }
            assert!(Instant::now() < deadline, "daemon not healthy: {}", self.log_tail());
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    fn log_tail(&self) -> String {
        fs::read_to_string(&self.log).unwrap_or_default()
    }

    async fn health_ok(&self) -> bool {
        matches!(
            self.client.get(format!("{}/health", self.url)).send().await,
            Ok(r) if r.status().is_success()
        )
    }

    async fn get_json(&self, path: &str) -> Value {
        self.client
            .get(format!("{}{path}", self.url))
            .send()
            .await
            .unwrap_or_else(|e| panic!("GET {path}: {e}"))
            .json::<Value>()
            .await
            .unwrap_or_else(|e| panic!("GET {path} json: {e}"))
    }

    /// Graceful stop: SIGTERM (the daemon's documented stop path), 15 s grace,
    /// then kill. Returns after the child is reaped.
    fn stop(&mut self) {
        let pid = self.child.id() as i32;
        unsafe {
            libc::kill(pid, libc::SIGTERM);
        }
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if self.child.try_wait().unwrap().is_some() {
                return;
            }
            if Instant::now() >= deadline {
                let _ = self.child.kill();
                let _ = self.child.wait();
                panic!("daemon did not exit 15 s after SIGTERM: {}", self.log_tail());
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// `brainrouter cli --socket S <args>` with the isolated environment, with
    /// fallback to the bare `brainrouter <args>` spelling (spec-gap note).
    fn cli(&self, args: &[&str]) -> CliRun {
        self.cli_env(args, std::iter::empty::<(String, String)>())
    }

    fn cli_env<I, S>(&self, args: &[&str], extra: I) -> CliRun
    where
        I: IntoIterator<Item = (S, S)> + Clone,
        S: AsRef<std::ffi::OsStr>,
    {
        let socket = self.socket.display().to_string();
        let forms: Vec<Vec<String>> = vec![
            std::iter::once("cli".to_string())
                .chain([
                    "--socket".to_string(),
                    socket.clone(),
                ])
                .chain(args.iter().map(|a| a.to_string()))
                .collect(),
            args.iter()
                .map(|a| a.to_string())
                .chain([
                    "--socket".to_string(),
                    socket.clone(),
                    "--config".to_string(),
                    self.env.root.join("config.yaml").display().to_string(),
                ])
                .collect(),
        ];
        let mut runs = Vec::new();
        for form in forms {
            let mut cmd = Command::new(env!("CARGO_BIN_EXE_brainrouter"));
            cmd.args(&form)
                .current_dir(&self.env.root)
                .stdin(Stdio::null());
            self.env.apply(&mut cmd, extra.clone());
            let out = cmd.output().expect("run CLI");
            let run = CliRun {
                args: form.join(" "),
                code: out.status.code().unwrap_or(-1),
                stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            };
            let ok = run.code == 0;
            runs.push(run);
            if ok {
                return runs.into_iter().next().unwrap();
            }
        }
        let combined = runs
            .iter()
            .map(|r| format!("{} => exit {} (stdout: {} | stderr: {})", r.args, r.code, r.stdout, r.stderr))
            .collect::<Vec<_>>()
            .join("\n");
        CliRun {
            args: "<all forms failed>".into(),
            code: -1,
            stdout: combined.clone(),
            stderr: combined,
        }
    }

    fn plugin_cli(&self, args: &[&str]) -> CliRun {
        let mut full = vec!["plugin"];
        full.extend_from_slice(args);
        self.cli(&full)
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct CliRun {
    #[allow(dead_code)]
    args: String,
    code: i32,
    stdout: String,
    stderr: String,
}

impl CliRun {
    #[allow(dead_code)]
    fn ok(&self) -> bool {
        self.code == 0
    }
    fn out(&self) -> String {
        format!("{}{}", self.stdout, self.stderr)
    }
    fn expect_ok(&self, what: &str) -> &Self {
        assert!(
            self.code == 0,
            "CLI `{what}` failed (exit {}): {}",
            self.code,
            self.out()
        );
        self
    }
}

/// Install + enable fixtures through the real CLI verbs, then restart the
/// daemon so the supervisor spawns them at boot (design: "Supervisor: spawn
/// enabled plugins at boot"). Fixtures are built on disk first; every CLI
/// verb (install/list/enable) is exercised.
async fn boot_with_plugins(env: &TestEnv, config_yaml: &str, fixtures: &[&Fixture]) -> Daemon {
    let mut daemon = Daemon::start(env, config_yaml);
    daemon.ready().await;
    for fx in fixtures {
        let dir = write_fixture(env, fx);
        let run = daemon.plugin_cli(&["install", dir.to_str().unwrap()]);
        assert!(
            run.code == 0,
            "plugin install {} failed (exit {}):\n{}",
            fx.id,
            run.code,
            run.out()
        );
    }
    let list = daemon.plugin_cli(&["list"]);
    list.expect_ok("plugin list");
    for fx in fixtures {
        assert!(
            list.out().contains(fx.id),
            "`plugin list` must show installed plugin {}\nCLI: {}\nstdout: {}\nstderr: {}",
            fx.id,
            list.args,
            list.stdout,
            list.stderr
        );
        let run = daemon.plugin_cli(&["enable", fx.id]);
        assert!(
            run.code == 0,
            "plugin enable {} failed (exit {}):\n{}",
            fx.id,
            run.code,
            run.out()
        );
    }
    daemon.stop();
    let mut daemon = Daemon::start(env, config_yaml);
    daemon.ready().await;
    daemon
}

struct Fixture {
    id: &'static str,
    script: &'static str,
    binary: &'static str,
    grants: &'static [&'static str],
    mode: &'static str,
    /// Extra fixture.json members (secrets candidates etc.).
    credential_ids: bool,
}

fn fixture_rpc(id: &'static str, mode: &'static str, grants: &'static [&'static str]) -> Fixture {
    Fixture { id, script: RPC_FIXTURE_PY, binary: "fixture.py", grants, mode, credential_ids: false }
}

fn write_fixture(env: &TestEnv, fx: &Fixture) -> PathBuf {
    let dir = env.root.join("fixtures").join(fx.id);
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("plugin.toml"),
        manifest_toml(fx.id, fx.binary, fx.grants),
    )
    .unwrap();
    let script_path = dir.join(fx.binary);
    let mut f = fs::File::create(&script_path).unwrap();
    f.write_all(fx.script.as_bytes()).unwrap();
    fs::set_permissions(&script_path, fs::Permissions::from_mode(0o755)).unwrap();
    let mut sidecar = json!({"id": fx.id, "mode": fx.mode});
    if fx.credential_ids {
        sidecar["credential_ids"] = json!(CREDENTIAL_ID_CANDIDATES);
    }
    fs::write(dir.join("fixture.json"), sidecar.to_string()).unwrap();
    dir
}

fn poll_read_json(path: &Path, timeout: Duration) -> Value {
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(text) = fs::read_to_string(path) {
            if let Ok(v) = serde_json::from_str::<Value>(&text) {
                return v;
            }
        }
        assert!(Instant::now() < deadline, "no JSON at {}: {:?}", path.display(), fs::read_to_string(path));
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn mode_bits(p: &Path) -> u32 {
    fs::metadata(p)
        .unwrap_or_else(|e| panic!("stat {}: {e}", p.display()))
        .permissions()
        .mode()
        & 0o777
}

// ── Minimal JSON-RPC client over the plugin UDS socket ─────────────────────

struct RpcConn {
    w: tokio::io::WriteHalf<UnixStream>,
    r: BufReader<tokio::io::ReadHalf<UnixStream>>,
    next_id: u64,
}

impl RpcConn {
    async fn connect(path: &Path) -> Self {
        let stream = UnixStream::connect(path)
            .await
            .unwrap_or_else(|e| panic!("connect plugin socket {}: {e}", path.display()));
        let (r, w) = tokio::io::split(stream);
        Self { w, r: BufReader::new(r), next_id: 0 }
    }

    async fn call(&mut self, method: &str, params: Value, timeout: Duration) -> Option<Value> {
        self.next_id += 1;
        let msg = json!({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params});
        if !self.write_line(&msg).await {
            // Write failed: the daemon closed the connection mid-request
            // (e.g. body-cap breach). EOF is an answer, not a test failure.
            return None;
        }
        self.read_msg(timeout).await
    }

    /// Returns false when the write failed (peer closed / cap breach).
    async fn write_line(&mut self, msg: &Value) -> bool {
        let mut bytes = msg.to_string().into_bytes();
        bytes.push(b'\n');
        let r = tokio::time::timeout(Duration::from_secs(10), self.w.write_all(&bytes)).await;
        if !matches!(r, Ok(Ok(()))) {
            return false;
        }
        let _ = tokio::time::timeout(Duration::from_secs(10), self.w.flush()).await;
        true
    }

    async fn read_msg(&mut self, timeout: Duration) -> Option<Value> {
        match tokio::time::timeout(timeout, self.r.lines().next_line()).await {
            Err(_) => panic!("RPC read timed out after {timeout:?}"),
            Ok(Ok(None)) => None, // EOF
            Ok(Ok(Some(line))) => Some(
                serde_json::from_str(&line)
                    .unwrap_or_else(|e| panic!("non-JSON RPC line from daemon: {e}: {line}")),
            ),
            Ok(Err(e)) => panic!("RPC read error: {e}"),
        }
    }

    /// `initialize` with `token`; panics with a design-anchored message on a
    /// refused handshake.
    async fn initialize(&mut self, plugin_id: &str, token: &str) -> Value {
        let resp = self
            .call(
                "initialize",
                json!({"id": plugin_id, "plugin_id": plugin_id, "name": plugin_id, "token": token}),
                Duration::from_secs(10),
            )
            .await;
        match resp {
            Some(v) if v.get("result").is_some() => {
                let result = v.get("result").unwrap().clone();
                assert!(
                    !result.get("grants").map(Value::is_null).unwrap_or(false),
                    "initialize result must carry the plugin's grants (design: 'returns grants + protocol version'); got {result}"
                );
                result
            }
            other => panic!("initialize with good token must be accepted; got {other:?}"),
        }
    }
}

fn plugin_token(env: &TestEnv, id: &str) -> String {
    let p = env.plugin_state(id).join("token");
    let text = fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("token file {} missing (design: plugins/<id>/token): {e}", p.display()));
    text.trim().to_string()
}

fn is_jsonrpc(msg: &Value) -> bool {
    msg.get("jsonrpc").and_then(Value::as_str) == Some("2.0")
        && ((msg.get("result").is_some() as u8) ^ (msg.get("error").is_some() as u8)) == 1
}

// ── Mock upstream (slow llama-swap stand-in for the in-flight cap) ──────────

struct MockUpstream {
    port: u16,
    hits: Arc<AtomicUsize>,
}

/// Loopback OpenAI-ish upstream: answers `/v1/chat/completions` with a plain
/// completion after `delay`, so requests stay in-flight on the daemon side.
fn spawn_mock_upstream(delay: Duration) -> MockUpstream {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    listener.set_nonblocking(true).unwrap();
    let hits = Arc::new(AtomicUsize::new(0));
    let hits2 = hits.clone();
    tokio::spawn(async move {
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();
        loop {
            let Ok((mut sock, _)) = listener.accept().await else { continue };
            let hits = hits2.clone();
            tokio::spawn(async move {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut buf = vec![0u8; 256 * 1024];
                let mut filled = 0usize;
                let mut content_len = None::<usize>;
                loop {
                    let Ok(n) = sock.read(&mut buf[filled.min(buf.len() - 1)..]).await else { return };
                    if n == 0 {
                        break;
                    }
                    filled += n;
                    if let Some(cl) = content_len {
                        if let Some(pos) = find_headers_end(&buf[..filled]) {
                            if filled >= pos + cl {
                                break;
                            }
                        }
                    } else if let Some(pos) = find_headers_end(&buf[..filled]) {
                        let headers = String::from_utf8_lossy(&buf[..pos]).to_lowercase();
                        content_len = headers
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0)));
                        if content_len == Some(0) {
                            break;
                        }
                    }
                }
                hits.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(delay).await;
                let body = json!({
                    "id": "chatcmpl-ray-l2",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test-only-model",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ray-pong"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                })
                .to_string();
                let _ = sock
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await;
                let _ = sock.shutdown().await;
            });
        }
    });
    MockUpstream { port, hits }
}

fn find_headers_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4)
}

// ── (1) RPC contract over a real UDS ───────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rpc_initialize_good_token_returns_grants() {
    let env = TestEnv::new("init");
    let fx = fixture_rpc("rayA", "basic", ALL_GRANTS);
    let daemon = boot_with_plugins(&env, &base_config_yaml(&dead_upstream_url()), &[&fx]).await;

    // The supervisor-spawned child completes `initialize` with the
    // daemon-minted token delivered via BRAINROUTER_PLUGIN_TOKEN (env),
    // proving mint -> store -> env -> handshake end to end.
    let report = poll_read_json(&env.plugin_state("rayA").join("report.json"), Duration::from_secs(20));
    assert_eq!(report["env_token_present"], json!(true), "child must receive BRAINROUTER_PLUGIN_TOKEN (design: passed to the spawned child via env)");
    assert!(report["initialize"].get("result").is_some(), "spawned plugin initialize must be accepted: {report}");

    // Direct handshake with the stored token returns grants + protocol version.
    let token = plugin_token(&env, "rayA");
    let mut conn = RpcConn::connect(&env.plugin_socket("rayA")).await;
    let result = conn.initialize("rayA", &token).await;
    assert!(
        result.get("grants").map(Value::as_array).map(|a| !a.is_empty()).unwrap_or_else(|| {
            // Some spellings (e.g. object-valued grants) are acceptable so
            // long as a grants member exists and is non-null.
            result.get("grants").is_some()
        }),
        "grants must be returned at connect: {result}"
    );
    let lower = result.to_string().to_lowercase();
    assert!(
        lower.contains("grant"),
        "initialize result must expose grants (design): {result}"
    );
    assert!(
        lower.contains("version") || lower.contains("protocol"),
        "initialize result must expose the protocol version (design: 'returns grants + protocol version'): {result}"
    );
    assert!(daemon.health_ok().await);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rpc_bad_token_connect_refused_not_counted_as_crash() {
    let env = TestEnv::new("badtok");
    let fx = fixture_rpc("rayA", "basic", ALL_GRANTS);
    let daemon = boot_with_plugins(&env, &base_config_yaml(&dead_upstream_url()), &[&fx]).await;
    let report_path = env.plugin_state("rayA").join("report.json");
    let report0 = poll_read_json(&report_path, Duration::from_secs(20));
    assert!(report0["initialize"]["result"].is_some(), "healthy baseline: spawned plugin must initialize: {report0}");

    // Repeated bad-token handshakes over the same socket: the connection must
    // be closed (EOF / error, never a grants payload).
    for i in 0..3 {
        let mut conn = RpcConn::connect(&env.plugin_socket("rayA")).await;
        let resp = conn
            .call(
                "initialize",
                json!({"id": "rayA", "plugin_id": "rayA", "token": format!("wrong-token-{i}")}),
                Duration::from_secs(10),
            )
            .await;
        match resp {
            None => {} // connection closed
            Some(v) => assert!(
                v.get("result").is_none(),
                "bad token must never yield a grants result: {v}"
            ),
        }
    }

    // NOT counted toward crash restarts: the plugin child is untouched (one
    // spawn, same report file, status not `failed`), and the daemon is healthy.
    let report1 = poll_read_json(&report_path, Duration::from_secs(2));
    assert_eq!(
        report1["pid"], report0["pid"],
        "plugin must not have been restarted by bad-token auth failures (restarts are not consumed)"
    );
    let status = daemon.plugin_cli(&["status", "rayA"]);
    let status = status.expect_ok("plugin status");
    assert!(
        !status.out().to_lowercase().contains("failed"),
        "3 bad-token connects must not drive the plugin toward terminal `failed` (design: 'an initialize auth failure ... is NOT counted as a crash'); status: {}",
        status.out()
    );
    assert!(daemon.health_ok().await);

    // Refusals are logged.
    let log = daemon.log_tail().to_lowercase();
    assert!(
        log.contains("raya")
            && log.contains("token")
            && ["refus", "denied", "auth", "invalid", "reject", "handshake"]
                .iter()
                .any(|w| log.contains(w)),
        "daemon log must record the bad-token refusals (design: 'closes the connection and logs')\nlog tail: {}",
        daemon.log_tail()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rpc_config_kv_namespace_isolation_across_plugins() {
    let env = TestEnv::new("nsmatrix");
    let fa = fixture_rpc("rayA", "basic", ALL_GRANTS);
    let fb = fixture_rpc("rayB", "basic", CONFIG_KV_GRANTS);
    let daemon = boot_with_plugins(&env, &base_config_yaml(&dead_upstream_url()), &[&fa, &fb]).await;

    let mut a = RpcConn::connect(&env.plugin_socket("rayA")).await;
    a.initialize("rayA", &plugin_token(&env, "rayA")).await;
    let mut b = RpcConn::connect(&env.plugin_socket("rayB")).await;
    b.initialize("rayB", &plugin_token(&env, "rayB")).await;

    // A writes; A reads back.
    let r = a.call("config.set", json!({"key": "k1", "name": "k1", "value": "alpha-A"}), Duration::from_secs(10)).await;
    assert!(r.map(|v| v.get("result").is_some()).unwrap_or(false), "A config.set must succeed: {r:?}");
    let r = a.call("config.get", json!({"key": "k1", "name": "k1"}), Duration::from_secs(10)).await;
    assert!(
        r.as_ref().map(|v| v.to_string().contains("alpha-A")).unwrap_or(false),
        "A must read its own config value back: {r:?}"
    );
    let r = a.call("kv.set", json!({"key": "k1", "name": "k1", "value": "kv-alpha-A"}), Duration::from_secs(10)).await;
    assert!(r.map(|v| v.get("result").is_some()).unwrap_or(false), "A kv.set must succeed: {r:?}");
    let r = a.call("kv.get", json!({"key": "k1", "name": "k1"}), Duration::from_secs(10)).await;
    assert!(
        r.as_ref().map(|v| v.to_string().contains("kv-alpha-A")).unwrap_or(false),
        "A must read its own kv value back: {r:?}"
    );

    // B cannot read A's namespaced values (config/kv are namespaced per plugin,
    // design: "config.get/set + kv.get/set (namespaced, state dir)").
    let r = b.call("config.get", json!({"key": "k1", "name": "k1"}), Duration::from_secs(10)).await;
    assert!(
        !r.as_ref().map(|v| v.to_string().contains("alpha-A")).unwrap_or(false),
        "B must NOT read A's config value (namespace isolation): {r:?}"
    );
    let r = b.call("kv.get", json!({"key": "k1", "name": "k1"}), Duration::from_secs(10)).await;
    assert!(
        !r.as_ref().map(|v| v.to_string().contains("kv-alpha-A")).unwrap_or(false),
        "B must NOT read A's kv value (namespace isolation): {r:?}"
    );

    // Same key in B's namespace is a distinct value.
    let r = b.call("config.set", json!({"key": "k1", "name": "k1", "value": "bravo-B"}), Duration::from_secs(10)).await;
    assert!(r.map(|v| v.get("result").is_some()).unwrap_or(false), "B config.set must succeed: {r:?}");
    let r = a.call("config.get", json!({"key": "k1", "name": "k1"}), Duration::from_secs(10)).await;
    assert!(
        r.as_ref().map(|v| v.to_string().contains("alpha-A")).unwrap_or(false),
        "B's write must not clobber A's namespace: {r:?}"
    );

    // Method-grant matrix: B (config/kv grants only) is refused `secrets.get`
    // (design: "token + per-method grant gating").
    let r = b.call("secrets.get", json!({"id": "anything", "credential_id": "anything", "key": "anything", "name": "anything"}), Duration::from_secs(10)).await;
    assert!(
        r.is_none() || r.as_ref().unwrap().get("error").is_some(),
        "B lacks the secrets.get grant and must be refused: {r:?}"
    );

    // models.list is granted to A and must not be an unknown-method (-32601).
    let r = a.call("models.list", json!({}), Duration::from_secs(15)).await;
    let r = r.expect("models.list answered");
    let code = r.get("error").and_then(|e| e.get("code")).and_then(Value::as_i64);
    assert_ne!(code, Some(-32601), "models.list is part of the RPC v1 surface: {r}");
    assert!(daemon.health_ok().await);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rpc_secrets_get_grant_matrix() {
    let env = TestEnv::new("secmatrix");
    // v1 YAML with the plaintext Discord token; the migration's credential
    // grants to plugin id `bridge`, which the `bridge` fixture occupies.
    let fx_a = fixture_rpc("rayA", "basic", ALL_GRANTS);
    let fx_bridge = Fixture {
        id: "bridge",
        script: RPC_FIXTURE_PY,
        binary: "fixture.py",
        grants: SECRETS_GRANTS,
        mode: "secrets",
        credential_ids: true,
    };
    let daemon = boot_with_plugins(&env, &v1_yaml_with_discord_token(&dead_upstream_url()), &[&fx_a, &fx_bridge]).await;

    // Positive leg (also acceptance item 3's round-trip): the bridge-granted
    // plugin resolves the migrated credential via secrets.get; value == YAML.
    let report = poll_read_json(&env.plugin_state("bridge").join("report.json"), Duration::from_secs(25));
    let found = report
        .pointer("/secrets/found_id")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!(
            "migrated Discord credential not resolvable by plugin id `bridge` under any conventional id {CREDENTIAL_ID_CANDIDATES:?} (design: 'v1 plaintext bridge.discord.token -> store credential granted to bridge'); report: {report}"
        ))
        .to_string();
    let value = report.pointer("/secrets/value").and_then(Value::as_str).unwrap();
    assert!(
        value.contains(DISCORD_TOKEN),
        "secrets.get must return the migrated token value (round-trip store -> plugin); got {value}"
    );

    // Negative leg: the credential does not grant rayA; A HAS the secrets.get
    // method grant, so this is the credential-grant axis of the matrix.
    let mut a = RpcConn::connect(&env.plugin_socket("rayA")).await;
    a.initialize("rayA", &plugin_token(&env, "rayA")).await;
    let r = a.call("secrets.get", json!({"id": found, "credential_id": found, "key": found, "name": found}), Duration::from_secs(10)).await;
    match &r {
        Some(v) if v.get("result").is_some() => panic!(
            "rayA is NOT granted the credential '{found}' and must be refused; got result containing token: {}",
            v.get("result").unwrap().to_string().contains(DISCORD_TOKEN)
        ),
        Some(_) => {}
        None => {}
    }
    assert!(daemon.health_ok().await);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rpc_oversized_body_rejected() {
    let env = TestEnv::new("size");
    let fx = fixture_rpc("rayA", "basic", ALL_GRANTS);
    let daemon = boot_with_plugins(&env, &base_config_yaml(&dead_upstream_url()), &[&fx]).await;
    let mut a = RpcConn::connect(&env.plugin_socket("rayA")).await;
    a.initialize("rayA", &plugin_token(&env, "rayA")).await;

    // Body cap 1 MiB: a ~1.5 MiB request must be rejected (JSON-RPC error)
    // with the daemon and the plugin service intact.
    let big = "x".repeat(1_600_000);
    let r = a.call("config.set", json!({"key": "big", "name": "big", "value": big}), Duration::from_secs(15)).await;
    match r {
        None => {} // connection closed on cap breach is a refusal
        Some(v) => assert!(
            v.get("error").is_some(),
            "a body > 1 MiB must not be accepted as success (design: 'body cap 1 MiB'): result present in {v}"
        ),
    }
    assert!(daemon.health_ok().await, "daemon must survive an over-cap body");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rpc_unknown_method_minus_32601() {
    let env = TestEnv::new("32601");
    let fx = fixture_rpc("rayA", "basic", ALL_GRANTS);
    let daemon = boot_with_plugins(&env, &base_config_yaml(&dead_upstream_url()), &[&fx]).await;
    let mut a = RpcConn::connect(&env.plugin_socket("rayA")).await;
    a.initialize("rayA", &plugin_token(&env, "rayA")).await;
    let r = a.call("ray.does-not-exist", json!({}), Duration::from_secs(10)).await;
    let r = r.expect("unknown method must be answered with a JSON-RPC error");
    assert_eq!(
        r.pointer("/error/code").and_then(Value::as_i64),
        Some(-32601),
        "design: 'Unknown method -> -32601'; got {r}"
    );
    assert!(daemon.health_ok().await);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn rpc_inflight_cap_six_concurrent_chat_completions() {
    // Cap 4 + queue 16; the queue means 6 rapid requests SHOULD be served
    // (the design's "excess -> limit error" fires beyond cap+queue). Minimal
    // boundary per the leg's acceptance: no daemon crash, every response is
    // JSON-RPC-shaped, at least 2 succeed.
    let env = TestEnv::new("inflight");
    let mock = spawn_mock_upstream(Duration::from_millis(1200));
    let yaml = base_config_yaml(&format!("http://127.0.0.1:{}", mock.port));
    let fx = fixture_rpc("rayA", "basic", ALL_GRANTS);
    let daemon = boot_with_plugins(&env, &yaml, &[&fx]).await;
    let token = plugin_token(&env, "rayA");
    let sock = env.plugin_socket("rayA");

    let mut tasks = Vec::new();
    for i in 0..6 {
        let sock = sock.clone();
        let token = token.clone();
        tasks.push(tokio::spawn(async move {
            let mut c = RpcConn::connect(&sock).await;
            c.initialize("rayA", &token).await;
            let resp = c
                .call(
                    "chat.completions",
                    json!({"model": "test-only-model", "messages": [{"role": "user", "content": format!("ray probe {i}")}]}),
                    Duration::from_secs(45),
                )
                .await;
            (i, resp)
        }));
    }
    let mut shaped = 0usize;
    let mut succeeded = 0usize;
    for t in tasks {
        let (i, resp) = t.await.unwrap();
        match resp {
            Some(v) if is_jsonrpc(&v) => {
                shaped += 1;
                if v.get("result").is_some() {
                    succeeded += 1;
                }
            }
            other => panic!("request {i} must get a JSON-RPC-shaped response (limit error is an acceptable excess answer; silence/crash is not): {other:?}"),
        }
    }
    assert_eq!(shaped, 6, "all six concurrent chat.completions answered in JSON-RPC shape");
    assert!(
        succeeded >= 2,
        "at least 2 of 6 rapid concurrent chat.completions must succeed (cap 4 in-flight; queue absorbs the rest): got {succeeded} succeeded, upstream hits {}",
        mock.hits.load(Ordering::SeqCst)
    );
    assert!(daemon.health_ok().await, "daemon must survive the burst");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rpc_socket_layout_and_permissions() {
    let env = TestEnv::new("perms");
    let fx = fixture_rpc("rayA", "basic", ALL_GRANTS);
    let daemon = boot_with_plugins(&env, &base_config_yaml(&dead_upstream_url()), &[&fx]).await;
    // The spawned fixture initialized over the socket; check layout + modes.
    poll_read_json(&env.plugin_state("rayA").join("report.json"), Duration::from_secs(20));
    let sock = env.plugin_socket("rayA");
    let plugins_dir = env.runtime.join("brainrouter").join("plugins");
    assert_eq!(mode_bits(&plugins_dir), 0o700, "plugin socket dir must be 0700 (design: 'dir 0700, socket 0600')");
    assert_eq!(mode_bits(&sock), 0o600, "plugin socket must be 0600 (design)");
    let token_path = env.plugin_state("rayA").join("token");
    assert_eq!(mode_bits(&token_path), 0o600, "minted token file must be 0600 (design: 'stored plugins/<id>/token 0600')");
    assert!(daemon.health_ok().await);
}

// ── (2) Supervisor ─────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supervisor_crash_loop_reaches_terminal_failed() {
    let env = TestEnv::new("crash");
    let fx = Fixture {
        id: "raycrash",
        script: CRASH_FIXTURE_PY,
        binary: "crash.py",
        grants: STATUS_GRANTS,
        mode: "crash",
        credential_ids: false,
    };
    let daemon = boot_with_plugins(&env, &base_config_yaml(&dead_upstream_url()), &[&fx]).await;
    let spawns_path = env.plugin_state("raycrash").join("spawns");

    // 5 consecutive crashes -> terminal `failed` (backoff is exponential,
    // capped at 60 s; generous ceiling covers builder base values).
    let deadline = Instant::now() + Duration::from_secs(180);
    let mut saw_failed = false;
    while Instant::now() < deadline {
        assert!(daemon.health_ok().await, "daemon must keep serving /health throughout the crash loop");
        let status = daemon.plugin_cli(&["status", "raycrash"]);
        if status.out().to_lowercase().contains("failed") {
            saw_failed = true;
            break;
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    assert!(saw_failed, "crash loop must reach terminal `failed` (status shows failed)\nlast daemon log: {}", daemon.log_tail());

    // Spawn stops: the spawn count is at the 5-crash terminal boundary and
    // stops growing.
    let count0 = spawn_lines(&spawns_path);
    assert!(
        (5..=7).contains(&count0),
        "terminal state after 5 consecutive crashes (1 initial + 4 restarts, +1 grace for racing restarts): observed {count0} spawns"
    );
    tokio::time::sleep(Duration::from_secs(10)).await;
    let count1 = spawn_lines(&spawns_path);
    assert_eq!(count1, count0, "spawn must stop at terminal `failed` (backoff restarts continue -> supervision not stopped)");

    // Log-line ceiling: the daemon's supervision log lines for this plugin
    // stop growing at the same point (spawn stops => no new spawn log traffic).
    let log = daemon.log_tail();
    let mentions = log.lines().filter(|l| l.to_lowercase().contains("raycrash")).count();
    assert!(
        mentions <= 60,
        "daemon log must not contain runaway supervision spam for the failed plugin ({mentions} lines mentioning raycrash)"
    );

    // No supervision until explicit `enable` (design): a disable/enable is the
    // documented recovery lever; here just confirm status persists as failed
    // while /health keeps answering.
    assert!(daemon.health_ok().await);
}

fn spawn_lines(spawns_path: &Path) -> usize {
    fs::read_to_string(spawns_path)
        .map(|t| t.lines().filter(|l| !l.trim().is_empty()).count())
        .unwrap_or(0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supervisor_stops_plugin_children_on_daemon_shutdown() {
    let env = TestEnv::new("stop");
    let fx = Fixture {
        id: "rayalive",
        script: ALIVE_FIXTURE_PY,
        binary: "alive.py",
        grants: STATUS_GRANTS,
        mode: "alive",
        credential_ids: false,
    };
    let mut daemon = boot_with_plugins(&env, &base_config_yaml(&dead_upstream_url()), &[&fx]).await;

    // Child started (pid file written by the fixture itself).
    let pid_path = env.plugin_state("rayalive").join("pid");
    // Drop any file left by the install daemon's child; the running child
    // rewrites it every 500 ms (see fixture heartbeat).
    let _ = fs::remove_file(&pid_path);
    let deadline = Instant::now() + Duration::from_secs(25);
    let child_pid: u32 = loop {
        if let Ok(text) = fs::read_to_string(&pid_path) {
            if let Ok(pid) = text.trim().parse::<u32>() {
                break pid;
            }
        }
        assert!(Instant::now() < deadline, "plugin child never wrote its pid file: {}", daemon.log_tail());
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    assert!(process_alive(child_pid), "fixture child {child_pid} must be running");

    // Graceful daemon shutdown (SIGTERM, the systemd stop path).
    daemon.stop();

    // "stop = SIGTERM with grace, then kill": the plugin child must be gone
    // shortly after the daemon exits.
    let deadline = Instant::now() + Duration::from_secs(10);
    while process_alive(child_pid) {
        assert!(
            Instant::now() < deadline,
            "daemon shutdown must stop its plugin child (pid {child_pid} still alive 10 s after daemon exit)"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn process_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

// ── (3) Credentials ────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn credentials_v1_migration_store_perms_grants_and_masking() {
    let env = TestEnv::new("cred");
    let daemon = Daemon::start(&env, &v1_yaml_with_discord_token(&dead_upstream_url()));
    let mut daemon = daemon;
    daemon.ready().await;

    // Store lands at boot (design: v1 YAML boots every v2 phase via
    // auto-migration). Locate `credentials.json` under the XDG state/data
    // trees; the design pins the file name + mode, not the directory.
    let store = poll_credentials_json(&env, Duration::from_secs(20));
    assert_eq!(
        mode_bits(&store),
        0o600,
        "credentials.json must be 0600 (design: 'credentials.json (0600, atomic owner-only writes)') at {}",
        store.display()
    );
    let store_text = fs::read_to_string(&store).unwrap();
    assert!(
        store_text.contains("bridge"),
        "the migrated Discord credential must grant to plugin `bridge` (design: 'v1 plaintext bridge.discord.token -> store credential granted to bridge'): {store_text}"
    );

    // Migration is logged (snapshot-first + logged, design).
    let log = daemon.log_tail().to_lowercase();
    assert!(
        log.contains("discord") || log.contains("credential"),
        "the migration must be logged (design: 'snapshot first, --dry-run preview, logged')\nlog: {log}"
    );
    // Snapshot: the pre-migration state is snapshotted (contract: "every leg
    // that migrates ... first snapshots the pre-migration state").
    let snapshot = find_file_containing(&env.root, &["snapshot", "backup", "pre-migration"], DISCORD_TOKEN, 6);
    assert!(
        snapshot,
        "a versioned pre-migration snapshot must exist (config migration framework contract: snapshot first); root {}: nothing under a snapshot-named path carries the pre-migration YAML",
        env.root.display()
    );

    // The token VALUE is masked everywhere it must never appear:
    // `config show` (CLI over the UDS), the HTTP config APIs, the plugin
    // state/config files, and the daemon log.
    let show = daemon.cli(&["config", "show"]);
    show.expect_ok("config show");
    assert!(
        !show.out().contains(DISCORD_TOKEN),
        "`config show` must not leak the migrated token value (design: 'values masked in all APIs/logs/config show')\n{}",
        show.out()
    );
    for path in ["/api/config", "/api/config-files", "/api/service-health"] {
        let body = daemon.get_json(path).await.to_string();
        assert!(!body.contains(DISCORD_TOKEN), "GET {path} must not leak the token value; body: {body}");
    }
    let plugin_bridge_dir = env.plugin_state("bridge");
    if plugin_bridge_dir.is_dir() {
        for entry in walk_files(plugin_bridge_dir) {
            if entry.file_name().map(|n| n == "report.json").unwrap_or(false) {
                continue; // the fixture's own report is not a config file
            }
            let text = fs::read_to_string(&entry).unwrap_or_default();
            assert!(
                !text.contains(DISCORD_TOKEN),
                "token value must be absent from plugin config files (R3 acceptance): {}",
                entry.display()
            );
        }
    }
    assert!(
        !daemon.log_tail().contains(DISCORD_TOKEN),
        "token value must never appear in daemon logs (masked everywhere)"
    );
    assert!(daemon.health_ok().await);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn credentials_plugin_secrets_get_round_trip() {
    // Acceptance item 3's round-trip: the bridge-granted plugin resolves the
    // migrated token via secrets.get and the value equals the v1 YAML value.
    let env = TestEnv::new("roundtrip");
    let fx_bridge = Fixture {
        id: "bridge",
        script: RPC_FIXTURE_PY,
        binary: "fixture.py",
        grants: SECRETS_GRANTS,
        mode: "secrets",
        credential_ids: true,
    };
    let daemon = boot_with_plugins(&env, &v1_yaml_with_discord_token(&dead_upstream_url()), &[&fx_bridge]).await;
    let report = poll_read_json(&env.plugin_state("bridge").join("report.json"), Duration::from_secs(25));
    let value = report
        .pointer("/secrets/value")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!(
            "secrets.get round-trip failed: credential not resolvable by the bridge-granted plugin under {CREDENTIAL_ID_CANDIDATES:?}; report: {report}"
        ));
    assert!(
        value.contains(DISCORD_TOKEN),
        "resolved credential value must equal the v1 YAML bridge.discord.token value (round-trip): got {value}"
    );
    // R3: the token value must be absent from the plugin's config files —
    // it reaches the plugin ONLY over secrets.get, never on disk.
    for entry in walk_files(env.plugin_state("bridge")) {
        if entry.file_name().map(|n| n == "report.json").unwrap_or(false) {
            continue; // the fixture's own round-trip report, not a config file
        }
        let text = fs::read_to_string(&entry).unwrap_or_default();
        assert!(
            !text.contains(DISCORD_TOKEN),
            "token value must be absent from plugin config files (R3): {}",
            entry.display()
        );
    }
    assert!(daemon.health_ok().await);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn credentials_dry_run_preview_leaves_state_untouched() {
    let env = TestEnv::new("dryrun");
    let yaml = v1_yaml_with_discord_token(&dead_upstream_url());
    let config_path = env.root.join("config.yaml");
    fs::write(&config_path, &yaml).unwrap();
    let address = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let a = l.local_addr().unwrap();
        drop(l);
        a.to_string()
    };

    // `--dry-run` preview (R9: "v1 YAML boots every v2 phase via
    // auto-migration with --dry-run preview").
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_brainrouter"));
    cmd.args(["serve", "--config"])
        .arg(&config_path)
        .args(["--tcp-addr", address.as_str(), "--socket", env.root.join("dry.sock").to_str().unwrap()])
        .arg("--dry-run")
        .current_dir(&env.root)
        .stdin(Stdio::null());
    env_apply_no_daemon(&env, &mut cmd);
    let out = cmd.output().expect("serve --dry-run must run and exit");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        out.status.code(),
        Some(0),
        "`serve --dry-run` must exit 0 with a migration preview (exit {:?}):\n{text}",
        out.status.code()
    );
    let lower = text.to_lowercase();
    assert!(
        (lower.contains("discord") || lower.contains("bridge") || lower.contains("credential"))
            && (lower.contains("migrat") || lower.contains("dry") || lower.contains("preview")),
        "--dry-run must PREVIEW the Discord-token migration (design: 'snapshot first, --dry-run preview, logged'); output:\n{text}"
    );

    // State untouched: no store, no plugin dirs, no bound daemon socket.
    assert!(
        find_credentials_json(&env).is_none(),
        "--dry-run must create no credentials.json (state untouched)"
    );
    assert!(
        !env.state_home.join("brainrouter").join("plugins").exists()
            && !env.runtime.join("brainrouter").join("plugins").exists(),
        "--dry-run must create no plugin state (state untouched)"
    );
    assert!(
        !env.root.join("dry.sock").exists(),
        "--dry-run must not bind the daemon socket (state untouched)"
    );
    assert!(
        !text.contains(DISCORD_TOKEN),
        "the preview itself must mask the token value (masked in every preview): {text}"
    );
}

fn env_apply_no_daemon(env: &TestEnv, cmd: &mut Command) {
    env.apply(cmd, std::iter::empty::<(String, String)>());
}

fn poll_credentials_json(env: &TestEnv, timeout: Duration) -> PathBuf {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(p) = find_credentials_json(env) {
            return p;
        }
        assert!(Instant::now() < deadline, "credentials.json never appeared under {}", env.root.display());
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn find_credentials_json(env: &TestEnv) -> Option<PathBuf> {
    for root in [&env.state_home, &env.data_home, &env.config_home] {
        if !root.exists() {
            continue;
        }
        for f in walk_files(root) {
            if f.file_name().map(|n| n == "credentials.json").unwrap_or(false) {
                return Some(f);
            }
        }
    }
    None
}

/// True when some file under `root` whose path contains any of `path_words`
/// contains `needle` (used for the migration snapshot, whose exact directory
/// the design does not pin).
fn find_file_containing(root: &Path, path_words: &[&str], needle: &str, max_depth: usize) -> bool {
    fn walk(dir: &Path, depth: usize, path_words: &[&str], needle: &str) -> bool {
        if depth == 0 {
            return false;
        }
        let Ok(entries) = fs::read_dir(dir) else { return false };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                if walk(&p, depth - 1, path_words, needle) {
                    return true;
                }
            } else if path_words
                .iter()
                .any(|w| p.to_string_lossy().to_lowercase().contains(w))
                && fs::read_to_string(&p).map(|t| t.contains(needle)).unwrap_or(false)
            {
                return true;
            }
        }
        false
    }
    walk(root, max_depth, path_words, needle)
}

fn walk_files(root: PathBuf) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out
}

// ── (5) Cargo tree: no serenity in the default core build ──────────────────

#[test]
fn core_build_has_no_serenity() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let toml = fs::read_to_string(root.join("Cargo.toml")).unwrap();

    // [features].default must not carry the bridge features (design "Core
    // removals": cargo features bridge-discord/bridge-signal gone).
    let features = toml
        .split("[features]")
        .nth(1)
        .and_then(|rest| rest.split("[features").next())
        .expect("Cargo.toml has a [features] table");
    let default_line = features
        .lines()
        .find(|l| l.trim_start().starts_with("default"))
        .expect("[features] declares the default set");
    assert!(
        !default_line.contains("bridge-discord") && !default_line.contains("bridge-signal"),
        "default build must not include the bridge features (design: R3 'bridge plugin moves out of the core binary'): {default_line}"
    );

    // The bridge-only deps named in "Core removals" are gone entirely.
    for dep in ["serenity", "async-trait", "serde_yml"] {
        let present = toml
            .lines()
            .any(|l| l.trim_start().starts_with(&format!("{dep} ")) || l.trim_start().starts_with(&format!("{dep}=")));
        assert!(
            !present,
            "`{dep}` must be removed from Cargo.toml (design 'Core removals': deps serenity/async-trait/serde_yml)"
        );
    }

    // Cargo.lock agrees: no serenity in the dependency graph at all.
    let lock = fs::read_to_string(root.join("Cargo.lock")).unwrap();
    assert!(
        !lock.lines().any(|l| l.trim() == format!("name = \"{}\"", "serenity")),
        "Cargo.lock must contain no `serenity` package: R3 'cargo tree core shows no serenity'"
    );
}

// ── (4) Bridge plugin behavior ─────────────────────────────────────────────
// Intentionally omitted: the v1 Discord/Signal command matrix needs the shipped
// bridge plugin binary (plugins/bridge -> brainrouter-bridge), which lives in
// the builder's crate, not in fixture form. Reported to Hank as
// builder-integration coverage; the grant/secrets wiring it depends on
// (credential granted to `bridge`) is covered by
// `rpc_secrets_get_grant_matrix` + `credentials_plugin_secrets_get_round_trip`
// with the fixture plugin under the real plugin id `bridge`.
