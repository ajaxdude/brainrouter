//! Step 9 fix-pass regression tests (Builder-L2; separate from Ray's ratified
//! v2_l2_acceptance suite). Covers the blocking/integration findings with
//! per-item asserts:
//!   B2 — /api/plugins* id traversal -> 4xx, registry untouched
//!   I1 — daemon-less `plugin enable` persists state, spawns nothing
//!   I3 — corrupt plugin registry / credentials vault boot (quarantine)

use serde_json::{json, Value};
use std::fs;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Env {
    root: PathBuf,
}

impl Env {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "brl2fix-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        for d in ["home", "config", "data", "state", "run", "bin"] {
            fs::create_dir_all(root.join(d)).unwrap();
        }
        use std::os::unix::fs::PermissionsExt as _;
        fs::set_permissions(root.join("run"), fs::Permissions::from_mode(0o700)).unwrap();
        for cand in ["/usr/bin/python3", "/bin/python3", "/usr/local/bin/python3"] {
            if Path::new(cand).exists() {
                std::os::unix::fs::symlink(cand, root.join("bin").join("python3")).unwrap();
                break;
            }
        }
        Env { root }
    }
    fn apply(&self, cmd: &mut Command) {
        cmd.env_clear()
            .env("HOME", self.root.join("home"))
            .env("XDG_CONFIG_HOME", self.root.join("config"))
            .env("XDG_DATA_HOME", self.root.join("data"))
            .env("XDG_STATE_HOME", self.root.join("state"))
            .env("XDG_RUNTIME_DIR", self.root.join("run"))
            .env("PATH", self.root.join("bin"))
            .env("RUST_LOG", "debug");
    }
    fn state_root(&self) -> PathBuf {
        self.root.join("state/brainrouter")
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

struct Daemon {
    child: Child,
    url: String,
    socket: PathBuf,
    log: PathBuf,
}

impl Daemon {
    fn start(env: &Env) -> Self {
        let config_path = env.root.join("config.yaml");
        if !config_path.exists() {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            let a = l.local_addr().unwrap();
            drop(l);
            fs::write(
                &config_path,
                format!(
                    "manifest:\n  base_url: http://{a}/cloud/v1\n  enabled: false\nllama_swap:\n  base_url: http://{a}/v1\n  fallback_model: test-only-model\nbonsai:\n  enabled: false\n  fork_path: /nonexistent\nreview: {{}}\n"
                ),
            )
            .unwrap();
        }
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
        env.apply(&mut cmd);
        let child = cmd.spawn().unwrap();
        Self {
            child,
            url: format!("http://{address}"),
            socket,
            log,
        }
    }

    async fn ready(&mut self) {
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                panic!(
                    "daemon exited with {status}: {}",
                    fs::read_to_string(&self.log).unwrap_or_default()
                );
            }
            if let Ok(r) = client.get(format!("{}/health", self.url)).send().await {
                if r.status().is_success() {
                    return;
                }
            }
            assert!(Instant::now() < deadline, "daemon not healthy");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    async fn get(&self, path: &str) -> (u16, String) {
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let r = client.get(format!("{}{}", self.url, path)).send().await.unwrap();
        (r.status().as_u16(), r.text().await.unwrap())
    }

    async fn post(&self, path: &str, body: Value) -> (u16, String) {
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let r = client
            .post(format!("{}{}", self.url, path))
            .json(&body)
            .send()
            .await
            .unwrap();
        (r.status().as_u16(), r.text().await.unwrap())
    }

    fn cli(&self, env: &Env, args: &[&str]) -> (i32, String) {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_brainrouter"));
        cmd.args(["cli", "--socket", &self.socket.display().to_string()])
            .args(args)
            .current_dir(&env.root);
        env.apply(&mut cmd);
        let out = cmd.output().expect("cli");
        (
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout).into_owned(),
        )
    }

    fn stop(&mut self) {
        let pid = self.child.id() as i32;
        unsafe { libc::kill(pid, libc::SIGTERM) };
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if self.child.try_wait().unwrap().is_some() {
                return;
            }
            assert!(Instant::now() < deadline, "daemon did not stop");
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

fn write_plugin_fixture(env: &Env, id: &str) -> PathBuf {
    let dir = env.root.join(format!("fx-{id}"));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("plugin.toml"),
        format!(
            "id = \"{id}\"\nname = \"fix\"\nversion = \"0.1.0\"\nbinary = \"sleeper.py\"\nmin_daemon_version = \"1.0.0\"\ngrants = [\"set_status\"]\n"
        ),
    )
    .unwrap();
    fs::write(
        dir.join("sleeper.py"),
        "#!/usr/bin/env python3\nimport time\nwhile True: time.sleep(1)\n",
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt as _;
    fs::set_permissions(dir.join("sleeper.py"), fs::Permissions::from_mode(0o755)).unwrap();
    dir
}

fn pgrep(marker: &str) -> Vec<u32> {
    // The caller's marker is always `sleeper.py`; the bracket pattern keeps
    // the grep from matching its own `sh -c` command line.
    let _ = marker;
    let out = Command::new("/bin/sh")
        .args(["-c", "grep -l 'sleeper[.]py' /proc/[0-9]*/cmdline 2>/dev/null"])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            l.split("/proc/").nth(1)
                .and_then(|s| s.split('/').next())
                .and_then(|p| p.parse::<u32>().ok())
        })
        .filter(|p| *p != std::process::id())
        .collect()
}

/// B2: every /api/plugins* handler must reject traversal ids with 4xx and
/// leave the registry + vault untouched.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn b2_http_plugin_endpoints_reject_traversal_ids() {
    let env = Env::new("b2");
    let mut daemon = Daemon::start(&env);
    daemon.ready().await;

    let registry_before = daemon.get("/api/plugins").await.1;
    let creds = env.state_root().join("credentials.json");
    fs::create_dir_all(env.state_root()).unwrap();
    fs::write(
        &creds,
        r#"{"credentials":{"keeper":{"id":"keeper","kind":"store","value":"topsecret","grants":["x"],"created_at":"","updated_at":""}}}"#,
    )
    .unwrap();

    for evil in ["..", "a/b", "/etc/passwd", ""] {
        for verb in ["enable", "disable", "remove", "config"] {
            let body = if verb == "config" {
                json!({ "id": evil, "key": "k", "value": 1 })
            } else {
                json!({ "id": evil })
            };
            let (code, text) = daemon.post(&format!("/api/plugins/{verb}"), body).await;
            assert!(
                (400..500).contains(&code),
                "POST {verb} id={evil:?} => {code} {text}"
            );
        }
        let (code, text) = daemon
            .get(&format!("/api/plugins/logs?id={}", urlencoding(evil)))
            .await;
        assert!(
            (400..500).contains(&code),
            "GET logs id={evil:?} => {code} {text}"
        );
    }

    // The vault and the registry survived the remove attempts (the exploit
    // target of B2: `remove("..")` ran remove_dir_all before validation).
    let after = daemon.get("/api/plugins").await.1;
    assert_eq!(
        serde_json::from_str::<Value>(&after).unwrap()["plugins"],
        serde_json::from_str::<Value>(&registry_before).unwrap()["plugins"],
        "registry changed via traversal"
    );
    assert!(creds.exists(), "credentials vault removed by traversal id");
    daemon.stop();
}

fn urlencoding(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'/' => "%2F".to_string(),
            _ if (b as char).is_ascii_alphanumeric() || b == b'-' || b == b'_' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// I1: with no daemon running, `plugin enable` must persist the registry
/// flag and spawn NOTHING (the daemon spawns at next boot).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn i1_daemonless_enable_persists_without_spawning() {
    let env = Env::new("i1");
    let fx = write_plugin_fixture(&env, "ghost");
    let bin = env!("CARGO_BIN_EXE_brainrouter");
    let sock = env.root.join("no-daemon.sock");

    let mut cmd = Command::new(bin);
    cmd.args(["plugin", "install", fx.to_str().unwrap(), "--socket", sock.to_str().unwrap()])
        .current_dir(&env.root);
    env.apply(&mut cmd);
    let out = cmd.output().unwrap();
    assert_eq!(out.status.code(), Some(0), "install: {}", String::from_utf8_lossy(&out.stderr));

    let mut cmd = Command::new(bin);
    cmd.args(["plugin", "enable", "ghost", "--socket", sock.to_str().unwrap()])
        .current_dir(&env.root);
    env.apply(&mut cmd);
    let out = cmd.output().unwrap();
    assert_eq!(out.status.code(), Some(0), "enable: {}", String::from_utf8_lossy(&out.stderr));
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("spawn_deferred"),
        "daemon-less enable must defer spawn: {}",
        String::from_utf8_lossy(&out.stdout)
    );

    std::thread::sleep(Duration::from_millis(500));
    let strays = pgrep("sleeper.py");
    assert!(strays.is_empty(), "daemon-less enable spawned {strays:?}");

    let registry = env.state_root().join("plugins/plugins.json");
    let reg: Value = serde_json::from_str(&fs::read_to_string(registry).unwrap()).unwrap();
    assert_eq!(reg["plugins"]["ghost"]["enabled"], json!(true));
    assert_eq!(reg["plugins"]["ghost"]["state"], json!("stopped"));
}

/// I3: corrupt plugins.json and credentials.json must boot (quarantine +
/// empty state), not brick the daemon.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn i3_corrupt_registry_and_vault_boot_quarantined() {
    let env = Env::new("i3");
    let plugins_dir = env.state_root().join("plugins");
    fs::create_dir_all(&plugins_dir).unwrap();
    fs::write(plugins_dir.join("plugins.json"), b"{\"plugins\": [corrupt").unwrap();
    fs::write(env.state_root().join("credentials.json"), b"NOT JSON AT ALL").unwrap();

    let mut daemon = Daemon::start(&env);
    daemon.ready().await; // panics with the log on hard-fail

    // Registry boots empty.
    let (code, text) = daemon.get("/api/plugins").await;
    assert_eq!(code, 200);
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["plugins"].as_array().unwrap().len(), 0);

    let (code, _) = daemon.get("/health").await;
    assert_eq!(code, 200, "daemon alive with corrupt state");

    // Quarantine files exist (forensics preserved) at the original paths' side.
    let quarantined = fs::read_dir(plugins_dir)
        .unwrap()
        .flatten()
        .any(|e| e.file_name().to_string_lossy().contains(".corrupt-"));
    assert!(quarantined, "plugins.json not quarantined");
    let vault_quarantined = fs::read_dir(env.state_root())
        .unwrap()
        .flatten()
        .any(|e| e.file_name().to_string_lossy().contains("credentials.json.corrupt-"));
    assert!(vault_quarantined, "credentials.json not quarantined");

    // A plugin installed on the clean boot resolves nothing -> secrets.get
    // fails cleanly (no panic), daemon stays alive.
    let fx = write_plugin_fixture(&env, "probe");
    let (code, _) = daemon
        .post("/api/plugins/install", json!({ "dir": fx.to_str().unwrap() }))
        .await;
    assert_eq!(code, 200, "install after quarantine boot");
    daemon.stop();

    let log = fs::read_to_string(&daemon.log).unwrap();
    assert!(log.contains("quarantin"), "no quarantine log:\n{}", &log[log.len().min(4000)..]);
}
