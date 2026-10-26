//! Acceptance tests for leg L6 (cloud providers, Manifest absorbed, LAN
//! federation) of `docs/design/v2-streamlining.md`. Spec:
//! `docs/design/v2-streamlining.L6.md` version L6-v5 (2026-10-23); Map
//! requirements R7 (credentials + cloud + federation-lite) and R8 (Manifest
//! retirement), the cross-leg contracts, and Map section "Security, privacy,
//! reliability, and operations".
//!
//! Written by Ray from the design's promises alone (spec, not the build). The
//! builder runs these and NEVER edits them.
//!
//! RED BY DESIGN: at the design base commit none of the L6 surfaces exist:
//! `providers.json` + the `provider` CLI + the Providers API, the generalized
//! openai-compat adapter and the NEW Anthropic outbound adapter, per-provider
//! breaker `state()`, the `cloud` resolution matrix (a/b/c/d), the v2.0-p6
//! migration, the LAN second listener (auth/rate/expose/peer-info), role-header
//! forwarding, the monthly budget ledger, and `credential export --env-line`.
//! Every such row fails naming the missing surface plus the spec bullet it
//! cites. Three rows are GREEN at the base commit by design, as parity
//! anchors the L6 build must NOT break:
//! - `l6_v1_floor_manifest_cloud_and_reviewer_pinned` (the v1 floor the Map
//!   requires to survive: cloud card -> model "auto", cloud/<id> passthrough,
//!   reviewer strictness).
//! - `l6_resolution_qualified_miss_cascade_and_cloud_prefix_never_local` (the
//!   v1 half of rule (d): an (a)-miss cascades to the passthrough and a
//!   `cloud/` prefix never escapes to llama-swap; the spec calls this
//!   byte-identical v1 behavior).
//! - `l6_parity_corpus_recording_fixture_contract` (the recorded corpus is
//!   data; its shape is asserted, not the build).
//!
//! Mapping of tests to the assignment contract items (1)-(10):
//! - (1) Provider CRUD, Test per kind, breakers, dual-auto, 0600, masked audit:
//!   * `l6_provider_crud_and_test_per_kind_auth_headers`
//!   * `l6_provider_breaker_trip_state_and_heal`
//!   * `l6_provider_dual_auto_manifest_refused`
//!   * `l6_provider_state_file_0600_and_masked_everywhere`
//! - (2) Resolution matrix:
//!   * `l6_resolution_qualified_provider_dispatch_and_peer` (a)
//!   * `l6_resolution_deepinfra_selector_passthrough_then_qualified` (b, d)
//!   * `l6_resolution_bare_cloud_manifest_first_and_provider_pool_tiebreak` (c)
//!   * `l6_resolution_qualified_miss_cascade_and_cloud_prefix_never_local` (d)
//!   * `l6_resolution_multi_match_4xx_and_validator_accepts_qualified` (c)
//!   * `l6_resolution_peer_pool_excluded_from_cloud_route` (peer pooling rule)
//! - (3) Parity corpus:
//!   * `l6_parity_corpus_recording_fixture_contract` (fixture set, GREEN)
//!   * `l6_parity_corpus_replay_v1_behaviors` (replay through v2.0-p6, RED)
//! - (4) v2.0-p6 migration:
//!   * `l6_migration_v2_p6_manifest_to_providers_snapshot_dryrun_restore`
//! - (5) LAN second listener:
//!   * `l6_lan_second_listener_auth_rotation_rate_and_admin_404`
//!   * `l6_lan_bearer_constant_time_compare_source`
//!   * `l6_lan_expose_double_enforcement_raw_and_resolved`
//!   * `l6_lan_peer_info_readonly_shape`
//!   * `l6_lan_primary_unchanged_and_off_listener_absent`
//! - (6) Role-header forwarding:
//!   * `l6_role_header_forwarded_to_lan_peer_only`
//! - (7) Budget guard:
//!   * `l6_budget_warn_event_badge_and_estimator_accrual`
//!   * `l6_budget_stop_primary_failure_persistence_and_rollover`
//! - (8) Credential env-line export:
//!   * `l6_credential_export_env_line_0600_gates`
//! - (9) Events additive:
//!   * `l6_events_additive_hop_and_v1_shape`
//! - (10) v1 floors:
//!   * `l6_v1_floor_manifest_cloud_and_reviewer_pinned` (GREEN)
//! - Distributed-role fixture (spec "Distributed-role acceptance flows",
//!   fixture tier):
//!   * `l6_distributed_role_fixture_plan_dev_review`
//!
//! PARITY CORPUS RECORDING PROCEDURE (assignment item 3). The corpus below is
//! the recorded triple set from the spec's "Parity corpus" bullet: request,
//! mock-gateway SSE script, and expected [normalized outbound request, event
//! sequence, terminal outcome, breaker transitions]. Recording procedure,
//! repeated verbatim by the next recorder:
//!   1. `git checkout v1-classic` (branch `v1`, tag `v1-classic`).
//!   2. Run the v1 binary with `manifest.base_url` pointing at the SAME scripted
//!      mock gateway this file embeds (`MockGateway`, openai-sse mode) and
//!      `llama_swap.base_url` at the local mock. No live LLM is involved: model
//!      outputs are excluded from comparison.
//!   3. For each of the seven cases send the request recorded in `request`,
//!      then capture: (i) the normalized outbound body at the mock (model +
//!      stream flag only, transport fields excluded), (ii) `GET
//!      /api/routing-events` filtered to the case prompt tag, normalized to
//!      `{stage, success, effective_provider, model_key}`, (iii) the client
//!      terminal (HTTP status class + `served`/`error`), (iv) the breaker state
//!      after the case (the v1 key is the literal `manifest`).
//!   4. Paste the tuple into `PARITY_CORPUS_JSON`. Never edit an expectation to
//!      match a v2 build: a mismatch is a parity failure.
//! The fixture contract row validates the set (seven behaviors, unique ids,
//! four expected elements per tuple, the v1 pinned rule for each outbound
//! model). The replay row runs the corpus through the L6 build (which first
//! runs v2.0-p6) and byte-compares the normalized tuples.
//!
//! GAPS THIS FILE MUST DECLARE (wire shapes the design does not pin; each row
//! probes candidates and fails with the probe list rather than inventing one):
//! - Providers API routes: the design names the Providers section verbs, not
//!   paths. Probes: POST /api/providers[/add|/upsert], PUT /api/providers/{id};
//!   GET /api/providers[/list]; DELETE /api/providers/{id}, POST
//!   /api/providers/remove; POST /api/providers/{id}/test,
//!   POST /api/providers/test.
//! - Provider record `price` spelling: the spec says "$/1K in,out"; the fixture
//!   uses `{"in": N, "out": N}` (alternate `{"input","output"}` reported).
//! - Breaker state shape: the spec pins the `state()` API, not its JSON. Rows
//!   search the provider record for a breaker/health/state value in the
//!   open/closed vocabulary instead of pinning a key.
//! - Budget ledger file: L4 already owns `budget.json` for swap reservations;
//!   L6 names the monthly provider accumulator `budget.json` too. Rows treat
//!   the ledger as shape-agnostic (any state-dir file holding the provider id
//!   and the current `YYYY-MM` key) and report the collision as a spec gap.
//! - `credential export --env-line` spelling: spec text is
//!   `brainrouter credential export --env-line <file>`; the L3-era nesting is
//!   `brainrouter cli credential ...`. Both are probed. The env-line NAME is
//!   unpinned; only `NAME=VALUE` with the store value is asserted.
//! - LAN bind field: spec says `lan: {enabled, bind, token_credential, expose,
//!   rate}`, so `bind` is used verbatim.
//! - Budget warn badge: credential/provider JSON badges are unpinned; rows
//!   assert the serialized payload names the budget state.
//! - Reviewer escalation mapping: the review-loop escalation on a strict
//!   failure is L3 behavior, already pinned by `tests/v2_l3_acceptance.rs`;
//!   this file pins the L6 precondition (strict failure, no local hop).
//! - `credential export` "audited event": the design says the owner command is
//!   audited; no audit surface is pinned, so the row asserts the file contract
//!   and the loopback/LAN absence of an export route only (reported to Hank).
//! - LAN expose addressability: `peer-info` advertises `models[].key`, and the
//!   resolved-escape row requests an exposed ALIAS whose resolved provider
//!   model is hidden. That assumes exposed provider model ids/aliases are
//!   addressable as bare keys on the LAN listener (the peer-info feed is the
//!   peer's menu). The unambiguous raw==resolved positive case uses a local
//!   key, so the row does not hang on that assumption alone.
//! - Breaker heal: black-box, the only heal path is the production cooldown
//!   (`health.rs` COOLDOWN_PERIOD = 60 s). The breaker row waits it out once
//!   (61 s) instead of inventing a reset API the spec does not name.
//! - Peer-info on the primary socket: asserted 404 per the assignment
//!   contract ("LAN inference paths 404 on primary"); the Map only pins the
//!   LAN listener's route table.
//!
//! SERIAL GATE (L3/L4 pattern): every row owns loopback ports, a live daemon
//! and kernel-assigned mock ports; parallel libtest threads race in the
//! reserve-then-bind windows. Each test holds one process-wide async mutex for
//! its lifetime. Opt out with `RAY_L6_NO_SERIAL_GATE=1` on an idle machine.

use std::collections::BTreeMap;
use std::fs;
use std::future::Future;
use std::net::{IpAddr, SocketAddr, TcpListener};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

// ── Constants ───────────────────────────────────────────────────────────────

/// Sentinels. Every "value absent" assertion greps for these.
const L6_SECRET: &str = "ray-L6-provider-key-9c31";
const L6_SECRET_ROTATED: &str = "ray-L6-provider-key-ROTATED-51fe";
const L6_MANIFEST_KEY: &str = "ray-L6-manifest-env-key-77ad";
const L6_LAN_TOKEN: &str = "ray-L6-lan-bearer-token-a4d2";
const L6_LAN_TOKEN_ROTATED: &str = "ray-L6-lan-bearer-ROTATED-90bc";
const L6_EXPORT_VALUE: &str = "ray-L6-exported-secret-3b7e";

/// v1 RouteEvent fields (src/routing_events.rs; the additive contract keeps
/// every one of them). `conv_id`/`pp_tps`/`tg_tps` are skip-when-empty, so the
/// row asserts them as "absent or the right type" instead.
const V1_EVENT_FIELDS: &[&str] = &[
    "id",
    "timestamp",
    "prompt_excerpt",
    "requested_model",
    "effective_provider",
    "model_key",
    "latency_ms",
    "stage",
    "success",
    "error",
    "bonsai_decision",
    "routing_class",
    "cwd",
];

/// A recorded v1 event (shape only; values are irrelevant). The additive
/// contract means this JSON must still parse as an event shape: v1 consumers
/// read these keys.
const V1_EVENT_FIXTURE: &str = r#"{"id":41,"timestamp":"2026-10-20T10:00:00Z","prompt_excerpt":"v1 shape","requested_model":"cloud","effective_provider":"manifest","model_key":"auto","latency_ms":12,"stage":"cloud_primary","success":true,"error":"","bonsai_decision":"cloud-direct","routing_class":"cloud","cwd":"/tmp","user_agent":"omp/18"}"#;

/// The recorded v1 parity tuple for the slash-bearing migrated selector
/// (spec rule (d): `cloud/deepinfra/deepseek-ai/DeepSeek-V4-Flash` has no
/// enabled provider `deepinfra`, so the WHOLE remainder passes through
/// verbatim).
const V1_DEEPINFRA_PASSTHROUGH: &str = "deepinfra/deepseek-ai/DeepSeek-V4-Flash";

/// The seven recorded parity behaviors (recording procedure in the header).
const PARITY_CORPUS_JSON: &str = r#"[
  {
    "id": "auto-send",
    "spec": "model:\"auto\" send",
    "role": "main",
    "selector": "cloud",
    "requests": 1,
    "script": {"kind": "sse"},
    "expected": {
      "outbound": {"model": "auto", "stream": true},
      "events": [{"stage": "cloud_primary", "success": true, "effective_provider": "manifest", "model_key": "auto"}],
      "terminal": "served",
      "breaker": {"manifest": "healthy"}
    }
  },
  {
    "id": "cloud-id-passthrough-empty-models",
    "spec": "cloud/<id> passthrough with an EMPTY migrated models list (the B1 case)",
    "role": "main",
    "selector": "cloud/deepseek-ai/DeepSeek-V4-Flash",
    "requests": 1,
    "script": {"kind": "sse"},
    "expected": {
      "outbound": {"model": "deepseek-ai/DeepSeek-V4-Flash", "stream": true},
      "events": [{"stage": "cloud_primary", "success": true, "effective_provider": "manifest", "model_key": "deepseek-ai/DeepSeek-V4-Flash"}],
      "terminal": "served",
      "breaker": {"manifest": "healthy"}
    }
  },
  {
    "id": "stream-peek-actual-model",
    "spec": "stream-peek actual-model (64 KiB/TTFT window, byte-preserving reassembly)",
    "role": "main",
    "selector": "cloud",
    "requests": 1,
    "script": {"kind": "sse", "first_frame_model": "DeepSeek-V4-Flash-actual"},
    "expected": {
      "outbound": {"model": "auto", "stream": true},
      "events": [{"stage": "cloud_primary", "success": true, "effective_provider": "manifest", "model_key": "DeepSeek-V4-Flash-actual"}],
      "terminal": "served",
      "breaker": {"manifest": "healthy"}
    }
  },
  {
    "id": "pseudo-success-sentinel",
    "spec": "pseudo-success sentinel (model==\"manifest\" first frame => fallback WITHOUT breaker failure report; empty stream => failure)",
    "role": "main",
    "selector": "cloud",
    "requests": 1,
    "script": {"kind": "sse", "pseudo_success": true},
    "expected": {
      "outbound": {"model": "auto", "stream": true},
      "events": [
        {"stage": "cloud_primary", "success": false, "effective_provider": "manifest"},
        {"stage": "cloud_fallback", "success": true, "effective_provider": "llama-swap"}
      ],
      "terminal": "served",
      "breaker": {"manifest": "healthy"}
    },
    "variant": {
      "script": {"kind": "empty_stream"},
      "expected": {
        "outbound": {"model": "auto", "stream": true},
        "events": [
          {"stage": "cloud_primary", "success": false, "effective_provider": "manifest"},
          {"stage": "cloud_fallback", "success": true, "effective_provider": "llama-swap"}
        ],
        "terminal": "served",
        "breaker": {"manifest": "degraded"}
      }
    }
  },
  {
    "id": "breaker-key-manifest-429-is-fault",
    "spec": "breaker key `manifest` + 429-is-fault trip",
    "role": "main",
    "selector": "cloud",
    "requests": 3,
    "script": {"kind": "fail", "status": 429},
    "expected": {
      "outbound": {"model": "auto", "stream": true},
      "events": [{"stage": "cloud_primary", "success": false, "effective_provider": "manifest"}],
      "terminal": "served",
      "breaker": {"manifest": "broken"}
    }
  },
  {
    "id": "cloud-to-fallback-stop",
    "spec": "cloud->fallback_model->stop (one FailedAttempt, stop on second failure)",
    "role": "main",
    "selector": "cloud",
    "requests": 1,
    "script": {"kind": "fail", "status": 500, "also_fail_local": true},
    "expected": {
      "outbound": {"model": "auto", "stream": true},
      "events": [
        {"stage": "cloud_primary", "success": false}
      ],
      "terminal": "error",
      "breaker": {"manifest": "degraded"}
    }
  },
  {
    "id": "reviewer-strict",
    "spec": "reviewer strict (no local hop, escalation on failure)",
    "role": "review",
    "selector": "cloud/vendor/review-2026",
    "requests": 1,
    "script": {"kind": "fail", "status": 500},
    "expected": {
      "outbound": {"model": "vendor/review-2026", "stream": true},
      "events": [{"stage": "cloud_primary", "success": false, "role": "review"}],
      "terminal": "error",
      "breaker": {"manifest": "degraded"}
    }
  }
]"#;

// ── Serial gate ─────────────────────────────────────────────────────────────

static SUITE_GATE: std::sync::LazyLock<tokio::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));

async fn gate() -> Option<tokio::sync::MutexGuard<'static, ()>> {
    if std::env::var("RAY_L6_NO_SERIAL_GATE").is_ok() {
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

/// The isolated XDG tree, cloneable so a Daemon can spawn CLI children with
/// exactly the environment its own process was booted with.
#[derive(Clone)]
struct EnvPaths {
    home: PathBuf,
    config_home: PathBuf,
    data_home: PathBuf,
    state_home: PathBuf,
    runtime: PathBuf,
}

impl EnvPaths {
    fn of(env: &TestEnv) -> Self {
        Self {
            home: env.home.clone(),
            config_home: env.config_home.clone(),
            data_home: env.data_home.clone(),
            state_home: env.state_home.clone(),
            runtime: env.runtime.clone(),
        }
    }

    fn apply(&self, cmd: &mut Command, extra: &[(String, String)]) {
        cmd.env_clear()
            .env("HOME", &self.home)
            .env("XDG_CONFIG_HOME", &self.config_home)
            .env("XDG_DATA_HOME", &self.data_home)
            .env("XDG_STATE_HOME", &self.state_home)
            .env("XDG_RUNTIME_DIR", &self.runtime)
            .env("PATH", "/usr/local/bin:/usr/bin:/bin")
            .env("RUST_LOG", "debug")
            .envs(extra.iter().cloned());
    }
}

struct TestEnv {
    root: PathBuf,
    home: PathBuf,
    config_home: PathBuf,
    data_home: PathBuf,
    state_home: PathBuf,
    runtime: PathBuf,
    owned: bool,
}
/// Where a test's isolated HOME/XDG tree lives: on DISK (not tmpfs — a
/// leaked rootless-podman store is ~1.9 GiB of RAM per run), but reached
/// through a SHORT path. The readable fixture name (`brainrouter-l6-{tag}-…`,
/// up to ~90 bytes) plus `/d.sock` overruns the `SUN_LEN` cap once the base
/// is deeper than `/tmp` — and `CARGO_TARGET_TMPDIR` is far deeper. So the
/// readable name is hashed into a short deterministic leaf on disk, exposed
/// through a per-boot `/tmp` symlink: every UDS/config path the daemon binds
/// stays short while the payload bytes stay on disk.
fn test_root(tag: &str) -> PathBuf {
    let readable = format!("brainrouter-l6-{tag}-{}", unique_suffix());
    // FNV-1a 64 over the unique readable name.
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
        // The caller treats the returned path as the root directly (it does
        // NOT append a fixture name), so the symlink path is the short root.
        Ok(()) => link,
        Err(_) => real,
    }
}

/// Remove a test root, including anything a user namespace owns.
///
/// The podman store inside it is created through a user namespace, so its
/// `overlay/*/diff` trees belong to the *subuid-mapped* root: an unprivileged
/// `remove_dir_all` fails with EACCES and silently leaks the whole root. Retry inside
/// that namespace (`podman unshare`), where those files are removable.
fn remove_test_root(root: &Path) {
    // The helper may hand out a `/tmp` symlink: remove the REAL tree on
    // disk first, then the link itself.
    let target = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    if !target.exists() {
        let _ = fs::remove_file(root);
        return;
    }
    let removed = fs::remove_dir_all(&target).is_ok();
    if !removed {
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

impl TestEnv {
    fn new(tag: &str) -> Self {
        let root = test_root(tag);
        let env = Self {
            home: root.join("home"),
            config_home: root.join("config"),
            data_home: root.join("data"),
            state_home: root.join("state"),
            runtime: root.join("run"),
            root,
            owned: true,
        };
        for d in [
            &env.home,
            &env.config_home,
            &env.data_home,
            &env.state_home,
            &env.runtime,
        ] {
            fs::create_dir_all(d).unwrap();
        }
        fs::set_permissions(&env.runtime, fs::Permissions::from_mode(0o700)).unwrap();
        fs::create_dir_all(env.state_dir()).unwrap();
        env
    }

    fn apply(&self, cmd: &mut Command, extra: &[(String, String)]) {
        EnvPaths::of(self).apply(cmd, extra);
    }

    fn state_dir(&self) -> PathBuf {
        self.state_home.join("brainrouter")
    }

    fn config_path(&self) -> PathBuf {
        self.root.join("config.yaml")
    }

    fn providers_path(&self) -> PathBuf {
        self.state_dir().join("providers.json")
    }

    fn credentials_path(&self) -> PathBuf {
        self.state_dir().join("credentials.json")
    }

    fn daemon_log(&self) -> String {
        fs::read_to_string(self.root.join("daemon.log")).unwrap_or_default()
    }

    /// Candidate monthly-provider-ledger files (spec gap: L6 names `budget.json`,
    /// which L4 already owns for swap reservations).
    fn ledger_paths(&self) -> Vec<PathBuf> {
        [
            "budget.json",
            "provider_budget.json",
            "providers_budget.json",
            "budget_providers.json",
        ]
        .iter()
        .map(|name| self.state_dir().join(name))
        .collect()
    }
}

impl Drop for TestEnv {
    fn drop(&mut self) {
        if self.owned {
            remove_test_root(&self.root);
        }
    }
}

fn mode_of(path: &Path) -> u32 {
    fs::metadata(path)
        .unwrap_or_else(|e| panic!("stat {}: {e}", path.display()))
        .permissions()
        .mode()
        & 0o777
}

fn month_key() -> String {
    chrono::Utc::now().format("%Y-%m").to_string()
}

fn seed_store_credential(env: &TestEnv, id: &str, value: &str) {
    let path = env.credentials_path();
    let mut root: Value = fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_else(|| json!({"credentials": {}}));
    root["credentials"][id] = json!({
        "id": id,
        "description": "ray-l6 fixture",
        "kind": "store",
        "value": value,
        "grants": [],
        "created_at": "2026-10-24T00:00:00Z",
        "updated_at": "2026-10-24T00:00:00Z",
    });
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, serde_json::to_vec_pretty(&root).unwrap()).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
}

/// Seed non-store credential kinds (export comments).
fn seed_env_credential(env: &TestEnv, id: &str, var: &str) {
    let path = env.credentials_path();
    let mut root: Value = fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_else(|| json!({"credentials": {}}));
    root["credentials"][id] = json!({
        "id": id, "description": "ray-l6 env ref", "kind": "env", "var": var, "grants": [],
    });
    fs::write(&path, serde_json::to_vec_pretty(&root).unwrap()).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
}

fn seed_file_credential(env: &TestEnv, id: &str, file: &Path) {
    let path = env.credentials_path();
    let mut root: Value = fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_else(|| json!({"credentials": {}}));
    root["credentials"][id] = json!({
        "id": id, "description": "ray-l6 file ref", "kind": "file",
        "path": file.display().to_string(), "grants": [],
    });
    fs::write(&path, serde_json::to_vec_pretty(&root).unwrap()).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
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

/// A non-loopback local IPv4 for the hop-event rows (any provider base_url on
/// it must carry `hop: {host, endpoint}`). Falls back to 127.0.0.2, which is
/// still a non-`127.0.0.1` host string; the row records that fallback.
fn non_loopback_ip() -> IpAddr {
    let sock = std::net::UdpSocket::bind("0.0.0.0:0");
    if let Ok(sock) = sock {
        if sock.connect("8.8.8.8:80").is_ok() {
            if let Ok(addr) = sock.local_addr() {
                if !addr.ip().is_loopback() {
                    return addr.ip();
                }
            }
        }
    }
    IpAddr::from([127, 0, 0, 2])
}

// ── Mock gateway (OpenAI-compatible + Anthropic wire) ───────────────────────

#[derive(Clone)]
struct MockCfg {
    /// ids advertised at GET /v1/models.
    models: Vec<String>,
    /// every completion fails with this status (backend fault for 5xx/429).
    fail_status: Option<u16>,
    /// the local (llama-swap) gateway fails too (cloud->fallback->stop).
    fail_completions: bool,
    /// first SSE frame reports model=="manifest" (v1 pseudo-success sentinel).
    pseudo_success: bool,
    /// first SSE frame reports this actual model (stream-peek case).
    actual_model: Option<String>,
    /// 200 with an empty body (peek failure).
    empty_stream: bool,
    /// hold the body this long after the headers.
    body_delay_ms: u64,
}

impl Default for MockCfg {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            fail_status: None,
            fail_completions: false,
            pseudo_success: false,
            actual_model: None,
            empty_stream: false,
            body_delay_ms: 0,
        }
    }
}

#[derive(Clone)]
struct MockReq {
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
    body: Value,
}

impl MockReq {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }

    fn is_completion(&self) -> bool {
        self.path.contains("/chat/completions") || self.path.contains("/messages")
    }
}

struct MockGateway {
    url: String,
    requests: Arc<Mutex<Vec<MockReq>>>,
    cfg: Arc<Mutex<MockCfg>>,
}

impl MockGateway {
    fn spawn(cfg: MockCfg) -> Self {
        Self::spawn_on(IpAddr::from([127, 0, 0, 1]), cfg)
    }

    fn spawn_on(ip: IpAddr, cfg: MockCfg) -> Self {
        let listener = std::net::TcpListener::bind(SocketAddr::new(ip, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let cfg = Arc::new(Mutex::new(cfg));
        let (r, c) = (Arc::clone(&requests), Arc::clone(&cfg));
        tokio::spawn(async move {
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    continue;
                };
                let (r, c) = (Arc::clone(&r), Arc::clone(&c));
                tokio::spawn(async move {
                    use tokio::io::AsyncReadExt;
                    let mut buf = vec![0u8; 512 * 1024];
                    let mut filled = 0usize;
                    let mut content_len = None::<usize>;
                    loop {
                        let off = filled.min(buf.len() - 1);
                        let Ok(n) = sock.read(&mut buf[off..]).await else {
                            return;
                        };
                        if n == 0 {
                            break;
                        }
                        filled += n;
                        if let Some(pos) = find_headers_end(&buf[..filled]) {
                            if content_len.is_none() {
                                content_len = Some(
                                    String::from_utf8_lossy(&buf[..pos])
                                        .lines()
                                        .filter_map(|l| {
                                            l.to_ascii_lowercase()
                                                .strip_prefix("content-length:")
                                                .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                                        })
                                        .next()
                                        .unwrap_or(0),
                                );
                            }
                            if filled >= pos + content_len.unwrap_or(0) {
                                break;
                            }
                        }
                    }
                    let head_end = find_headers_end(&buf[..filled]).unwrap_or(0);
                    let head_text = String::from_utf8_lossy(&buf[..head_end]).to_string();
                    let mut lines = head_text.lines();
                    let request_line = lines.next().unwrap_or("").to_string();
                    let mut parts = request_line.split_whitespace();
                    let method = parts.next().unwrap_or("").to_string();
                    let path = parts.next().unwrap_or("").to_string();
                    let mut headers = BTreeMap::new();
                    for line in lines {
                        if let Some((k, v)) = line.split_once(':') {
                            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
                        }
                    }
                    let body: Value = serde_json::from_slice(&buf[head_end..filled.max(head_end)])
                        .unwrap_or(Value::Null);
                    let request = MockReq {
                        method: method.clone(),
                        path: path.clone(),
                        headers,
                        body: body.clone(),
                    };
                    r.lock().unwrap().push(request);
                    let cfg = c.lock().unwrap().clone();
                    respond(&mut sock, &method, &path, &body, &cfg).await;
                });
            }
        });
        Self {
            url: format!("http://{ip}:{port}"),
            requests,
            cfg,
        }
    }

    fn reset(&self, cfg: MockCfg) {
        *self.cfg.lock().unwrap() = cfg;
    }

    fn requests(&self) -> Vec<MockReq> {
        self.requests.lock().unwrap().clone()
    }

    fn clear(&self) {
        self.requests.lock().unwrap().clear();
    }

    fn completions(&self) -> Vec<MockReq> {
        self.requests()
            .into_iter()
            .filter(MockReq::is_completion)
            .collect()
    }

    fn models_seen(&self) -> Vec<String> {
        self.completions()
            .iter()
            .filter_map(|r| {
                r.body
                    .get("model")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect()
    }

    fn count(&self) -> usize {
        self.completions().len()
    }
}

fn find_headers_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4)
}

async fn respond<S>(sock: &mut S, method: &str, path: &str, body: &Value, cfg: &MockCfg)
where
    S: tokio::io::AsyncWrite + Unpin,
{
    if method == "GET" && path.contains("/models") {
        let data: Vec<Value> = cfg
            .models
            .iter()
            .map(|id| json!({"id": id, "object": "model", "owned_by": "ray-l6-mock"}))
            .collect();
        write_http(
            sock,
            200,
            "application/json",
            &json!({"object": "list", "data": data}).to_string(),
        )
        .await;
        return;
    }
    let is_anthropic = path.contains("/messages");
    let is_completion = path.contains("/chat/completions") || is_anthropic;
    if !is_completion {
        // Provider Test may probe any cheap path; answer a valid JSON body.
        write_http(sock, 200, "application/json", "{\"ok\":true}").await;
        return;
    }
    if cfg.fail_completions || cfg.fail_status.is_some() {
        let status = cfg.fail_status.unwrap_or(500);
        write_http(
            sock,
            status,
            "application/json",
            &json!({"error": {"message": "ray-l6 mock failure", "type": "server_error"}})
                .to_string(),
        )
        .await;
        return;
    }
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if is_anthropic {
        if stream {
            let actual = cfg.actual_model.clone().unwrap_or_else(|| model.clone());
            let sse = anthropic_sse(&actual);
            write_http(sock, 200, "text/event-stream", &sse).await;
        } else {
            write_http(
                sock,
                200,
                "application/json",
                &json!({
                    "id": "msg_ray_l6", "type": "message", "role": "assistant",
                    "model": model,
                    "content": [{"type": "text", "text": "ray-l6-pong"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 3, "output_tokens": 2}
                })
                .to_string(),
            )
            .await;
        }
        return;
    }
    if cfg.empty_stream {
        write_http(sock, 200, "text/event-stream", "").await;
        return;
    }
    if cfg.pseudo_success {
        let sse = openai_chunk("manifest", "", Value::Null)
            + &openai_chunk("manifest", "ray-l6-pseudo", json!("stop"))
            + "data: [DONE]\n\n";
        write_http(sock, 200, "text/event-stream", &sse).await;
        return;
    }
    let actual = cfg.actual_model.clone().unwrap_or_else(|| model.clone());
    if stream {
        let sse = openai_chunk(&actual, "", Value::Null)
            + &openai_chunk(&actual, &format!("ray-l6-pong {actual}"), json!("stop"))
            + "data: [DONE]\n\n";
        if cfg.body_delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(cfg.body_delay_ms)).await;
        }
        write_http(sock, 200, "text/event-stream", &sse).await;
    } else {
        write_http(
            sock,
            200,
            "application/json",
            &json!({
                "id": "chatcmpl-ray-l6", "object": "chat.completion", "created": 1,
                "model": actual,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ray-l6-pong"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
            })
            .to_string(),
        )
        .await;
    }
}

fn openai_chunk(model: &str, content: &str, finish: Value) -> String {
    format!(
        "data: {}\n\n",
        json!({
            "id": "chatcmpl-ray-l6", "object": "chat.completion.chunk", "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}]
        })
    )
}

fn anthropic_sse(model: &str) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "event: message_start\ndata: {}\n\n",
        json!({"type": "message_start", "message": {"id": "msg_ray_l6", "type": "message",
            "role": "assistant", "model": model, "content": [],
            "usage": {"input_tokens": 3, "output_tokens": 0}}})
    ));
    out.push_str(&format!(
        "event: content_block_start\ndata: {}\n\n",
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
    ));
    out.push_str(&format!(
        "event: content_block_delta\ndata: {}\n\n",
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ray-l6-pong"}})
    ));
    out.push_str(&format!(
        "event: content_block_stop\ndata: {}\n\n",
        json!({"type": "content_block_stop", "index": 0})
    ));
    out.push_str(&format!(
        "event: message_delta\ndata: {}\n\n",
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}})
    ));
    out.push_str(&format!(
        "event: message_stop\ndata: {}\n\n",
        json!({"type": "message_stop"})
    ));
    out
}

async fn write_http<S>(sock: &mut S, status: u16, ctype: &str, body: &str)
where
    S: tokio::io::AsyncWrite + Unpin,
{
    use tokio::io::AsyncWriteExt;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Status",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: {ctype}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    );
    let _ = sock.write_all(head.as_bytes()).await;
    let _ = sock.write_all(body.as_bytes()).await;
    let _ = sock.flush().await;
    let _ = sock.shutdown().await;
}

// ── Daemon harness (live-daemon conventions from v2_l3/v2_l4) ──────────────

struct Daemon {
    child: std::process::Child,
    url: String,
    log: PathBuf,
    socket: PathBuf,
    client: reqwest::Client,
    paths: EnvPaths,
}

impl Daemon {
    fn start(env: &TestEnv, config_yaml: &str) -> Self {
        Self::start_with_env(env, config_yaml, &[])
    }

    fn start_with_env(env: &TestEnv, config_yaml: &str, extra: &[(String, String)]) -> Self {
        fs::write(env.config_path(), config_yaml).unwrap();
        Self::boot(env, extra)
    }

    fn start_from_disk(env: &TestEnv) -> Self {
        assert!(
            env.config_path().exists(),
            "restart precondition: config on disk"
        );
        Self::boot(env, &[])
    }

    fn boot(env: &TestEnv, extra: &[(String, String)]) -> Self {
        let config_path = env.config_path();
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
            socket,
            client: reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(90))
                .build()
                .unwrap(),
            paths: EnvPaths::of(env),
        }
    }

    async fn ready(&mut self) {
        self.ready_cited("the daemon never became healthy").await;
    }

    async fn ready_cited(&mut self, cite: &str) {
        let deadline = Instant::now() + Duration::from_secs(90);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                panic!("daemon exited with {status} ({cite})\n{}", self.log_tail());
            }
            if let Ok(r) = self.client.get(format!("{}/health", self.url)).send().await {
                if r.status().is_success()
                    && r.json::<Value>().await.unwrap_or(Value::Null) == json!({"status": "ok"})
                {
                    return;
                }
            }
            assert!(
                Instant::now() < deadline,
                "daemon not healthy ({cite}): {}",
                self.log_tail()
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    fn log_tail(&self) -> String {
        let text = fs::read_to_string(&self.log).unwrap_or_default();
        let tail: Vec<&str> = text.lines().rev().take(30).collect();
        tail.into_iter().rev().collect::<Vec<_>>().join("\n")
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}{path}", self.url)
    }

    async fn get_json(&self, path: &str) -> Value {
        let (status, text) = self.request("GET", path, None, &[]).await;
        assert!(status < 400, "GET {path} -> {status}: {text}");
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("GET {path} not json ({e}): {text}"))
    }

    async fn get_text(&self, path: &str) -> (u16, String) {
        self.request("GET", path, None, &[]).await
    }

    async fn post_json(&self, path: &str, body: Value) -> (u16, String) {
        self.request("POST", path, Some(body), &[]).await
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
        let mut req = self.client.request(
            reqwest::Method::from_bytes(method.as_bytes()).unwrap(),
            self.endpoint(path),
        );
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        if let Some(b) = body {
            req = req.json(&b);
        }
        let r = req
            .send()
            .await
            .unwrap_or_else(|e| panic!("{method} {path}: {e}"));
        let status = r.status().as_u16();
        (status, r.text().await.unwrap_or_default())
    }

    /// `brainrouter cli --socket S <args>` with the L3/L4 fallback spellings
    /// (top-level verbs take their own global `--socket`).
    fn cli(&self, args: &[&str]) -> CliRun {
        let socket = self.socket.display().to_string();
        let mut forms: Vec<Vec<String>> = Vec::new();
        forms.push(
            std::iter::once("cli".to_string())
                .chain(["--socket".to_string(), socket.clone()])
                .chain(args.iter().map(|a| a.to_string()))
                .collect(),
        );
        forms.push(
            args.iter()
                .map(|a| a.to_string())
                .chain(["--socket".to_string(), socket.clone()])
                .collect(),
        );
        if !args.is_empty() {
            forms.push(
                std::iter::once(args[0].to_string())
                    .chain(["--socket".to_string(), socket.clone()])
                    .chain(args[1..].iter().map(|a| a.to_string()))
                    .collect(),
            );
        }
        run_cli_forms(&self.paths, forms)
    }

    fn stop(&mut self) {
        let pid = self.child.id() as i32;
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
                panic!(
                    "daemon did not exit 20 s after SIGTERM: {}",
                    self.log_tail()
                );
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct CliRun {
    code: i32,
    stdout: String,
    stderr: String,
}

impl CliRun {
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

/// Run CLI forms in order, returning the first exit-0 run (otherwise the last
/// run with every attempt's output folded in). Children run with the isolated
/// XDG tree so the CLI resolves the same state root as the daemon.
fn run_cli_forms(paths: &EnvPaths, forms: Vec<Vec<String>>) -> CliRun {
    let mut attempts = Vec::new();
    for form in forms {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_brainrouter"));
        cmd.args(&form).stdin(Stdio::null());
        paths.apply(&mut cmd, &[]);
        let out = cmd.output().expect("run brainrouter CLI");
        let run = CliRun {
            code: out.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        };
        if run.code == 0 {
            return run;
        }
        attempts.push(format!(
            "brainrouter {} => exit {} ({})",
            form.join(" "),
            run.code,
            run.out()
        ));
    }
    CliRun {
        code: -1,
        stdout: String::new(),
        stderr: attempts.join("\n"),
    }
}

fn run_cli(env: &TestEnv, args: &[&str], extra: &[(String, String)]) -> CliRun {
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

// The `Daemon::cli` helper spawns CLI children with the daemon's own EnvPaths
// (see run_cli_forms), so every CLI call reads the isolated state root.

fn parse_json_from(text: &str) -> Option<Value> {
    if let Ok(v) = serde_json::from_str::<Value>(text) {
        return Some(v);
    }
    for (i, ch) in text.char_indices() {
        if ch == '{' || ch == '[' {
            if let Ok(v) = serde_json::from_str::<Value>(&text[i..]) {
                return Some(v);
            }
        }
    }
    None
}

// ── Chat / LAN request helpers ─────────────────────────────────────────────

async fn chat_tagged(
    d: &Daemon,
    model: &str,
    role: Option<&str>,
    tag: &str,
) -> (u16, Result<String, String>) {
    let mut req = d
        .client
        .post(d.endpoint("/v1/chat/completions"))
        .json(&json!({
            "model": model,
            "messages": [{"role": "user", "content": format!("ray-l6 {tag}")}],
            "stream": true,
            "max_tokens": 32,
        }));
    if let Some(role) = role {
        req = req.header("x-brainrouter-role", role);
    }
    let resp = req.send().await.expect("chat request reaches daemon");
    let status = resp.status().as_u16();
    (status, resp.text().await.map_err(|e| e.to_string()))
}

fn assert_served(result: &(u16, Result<String, String>), label: &str) {
    let (status, text) = result;
    let text = text
        .as_ref()
        .unwrap_or_else(|e| panic!("{label}: stream failed: {e}"));
    assert_eq!(*status, 200, "{label}: non-200: {text}");
    assert!(
        text.contains("ray-l6-pong"),
        "{label}: body lacks the mock payload: {text}"
    );
}

async fn events(d: &Daemon) -> Vec<Value> {
    let v = d.get_json("/api/routing-events").await;
    v["events"].as_array().cloned().unwrap_or_default()
}

async fn event_for(d: &Daemon, tag: &str) -> Value {
    poll_until(&format!("the routing event for `{tag}`"), 15, || async {
        events(d).await.into_iter().find(|e| {
            e["prompt_excerpt"]
                .as_str()
                .map(|p| p.contains(tag))
                .unwrap_or(false)
        })
    })
    .await
}

fn assert_v1_event_fields(event: &Value) {
    for field in V1_EVENT_FIELDS {
        assert!(
            event.get(*field).is_some(),
            "v1 floor: RouteEvent lost `{field}` (events ADDITIVE only): {event}"
        );
    }
    // skip-when-empty v2 additive fields: absent or the recorded type.
    for field in ["conv_id", "user_agent"] {
        assert!(
            event.get(field).is_none_or(Value::is_string),
            "{field} must stay a string when present: {event}"
        );
    }
}

async fn lan_request(
    client: &reqwest::Client,
    url: &str,
    method: &str,
    path: &str,
    token: Option<&str>,
    body: Option<Value>,
    role: Option<&str>,
) -> (u16, String) {
    let mut req = client.request(
        reqwest::Method::from_bytes(method.as_bytes()).unwrap(),
        format!("{url}{path}"),
    );
    if let Some(token) = token {
        req = req.header("Authorization", format!("Bearer {token}"));
    }
    if let Some(role) = role {
        req = req.header("x-brainrouter-role", role);
    }
    if let Some(b) = body {
        req = req.json(&b);
    }
    match req.send().await {
        Ok(r) => {
            let status = r.status().as_u16();
            (status, r.text().await.unwrap_or_default())
        }
        Err(e) => (0, format!("connect error: {e}")),
    }
}

// ── Provider API (routes unpinned; probes per the header) ──────────────────

struct ProviderApi;

impl ProviderApi {
    fn add_candidates(record: &Value, id: &str) -> Vec<(String, String, Value)> {
        vec![
            ("POST".into(), "/api/providers/add".into(), record.clone()),
            ("POST".into(), "/api/providers".into(), record.clone()),
            (
                "POST".into(),
                "/api/providers/upsert".into(),
                record.clone(),
            ),
            (
                "POST".into(),
                "/api/providers/add".into(),
                json!({"provider": record.clone()}),
            ),
            ("PUT".into(), format!("/api/providers/{id}"), record.clone()),
        ]
    }

    /// Returns the FIRST non-404/405 answer (so a 4xx refusal is observable,
    /// e.g. the dual-auto rule), preferring a 2xx.
    async fn add(d: &Daemon, record: Value) -> (u16, String) {
        let id = record["id"].as_str().unwrap_or("").to_string();
        let mut probes = Vec::new();
        let mut present: Option<(u16, String)> = None;
        for (method, path, body) in Self::add_candidates(&record, &id) {
            let (status, text) = d.request(&method, &path, Some(body), &[]).await;
            if (200..300).contains(&status) {
                return (status, text);
            }
            if status != 404 && status != 405 {
                probes.push(format!("{method} {path} -> {status}: {text}"));
                if present.is_none() {
                    present = Some((status, text));
                }
            } else {
                probes.push(format!("{method} {path} -> {status}"));
            }
        }
        if let Some(answer) = present {
            return answer;
        }
        panic!(
            "MISSING SURFACE: no provider-add route is registered (L6 spec \"Providers\": \
             `provider add|list|remove|test`, dashboard Providers section; the record speaks \
             `{{id, kind, base_url, credential_id?, auto, models, lan_peer, enabled}}`). \
             Probes: {}",
            probes.join("; ")
        )
    }

    async fn add_ok(d: &Daemon, record: Value) {
        let id = record["id"].as_str().unwrap().to_string();
        let (status, text) = Self::add(d, record).await;
        assert!(
            (200..300).contains(&status),
            "provider add `{id}` -> {status}: {text}\n\
             (L6 spec \"Providers\": CLI `provider add|list|remove|test <id>`, mutations via \
             the daemon)"
        );
    }

    async fn list(d: &Daemon) -> Value {
        let mut probes = Vec::new();
        for path in ["/api/providers", "/api/providers/list"] {
            let (status, text) = d.request("GET", path, None, &[]).await;
            if status == 200 {
                if let Ok(v) = serde_json::from_str::<Value>(&text) {
                    return v;
                }
            }
            probes.push(format!("GET {path} -> {status}: {text}"));
        }
        let run = d.cli(&["provider", "list"]);
        if run.code == 0 {
            if let Some(v) = parse_json_from(&run.stdout) {
                return v;
            }
        }
        probes.push(format!(
            "cli provider list => exit {} ({})",
            run.code,
            run.out()
        ));
        panic!(
            "MISSING SURFACE: no provider-list surface answered (L6 spec \"Providers\": \
             `provider list`, dashboard Providers section, live per-provider breaker state). \
             Probes:\n{}",
            probes.join("\n")
        )
    }

    async fn remove(d: &Daemon, id: &str) -> (u16, String) {
        let mut probes = Vec::new();
        let mut present = None;
        let candidates = vec![
            ("DELETE".to_string(), format!("/api/providers/{id}"), None),
            (
                "POST".to_string(),
                "/api/providers/remove".to_string(),
                Some(json!({"id": id})),
            ),
            (
                "POST".to_string(),
                format!("/api/providers/{id}/remove"),
                None,
            ),
        ];
        for (method, path, body) in candidates {
            let (status, text) = if method == "DELETE" {
                d.delete(&path).await
            } else {
                d.request(&method, &path, body, &[]).await
            };
            if (200..300).contains(&status) {
                return (status, text);
            }
            if status != 404 && status != 405 {
                if present.is_none() {
                    present = Some((status, text.clone()));
                }
            }
            probes.push(format!("{method} {path} -> {status}"));
        }
        if let Some(answer) = present {
            return answer;
        }
        panic!(
            "MISSING SURFACE: no provider-remove route answered (L6 spec \"Providers\": \
             `provider remove <id>` via the daemon). Probes: {}",
            probes.join("; ")
        )
    }

    async fn test(d: &Daemon, id: &str) -> (u16, String) {
        let mut probes = Vec::new();
        let mut present = None;
        let candidates = vec![
            (
                "POST".to_string(),
                format!("/api/providers/{id}/test"),
                None,
            ),
            (
                "POST".to_string(),
                "/api/providers/test".to_string(),
                Some(json!({"id": id})),
            ),
        ];
        for (method, path, body) in candidates {
            let (status, text) = d.request(&method, &path, body, &[]).await;
            if (200..300).contains(&status) {
                return (status, text);
            }
            if status != 404 && status != 405 {
                if present.is_none() {
                    present = Some((status, text.clone()));
                }
            }
            probes.push(format!("{method} {path} -> {status}"));
        }
        if let Some(answer) = present {
            return answer;
        }
        panic!(
            "MISSING SURFACE: no provider-test route answered (L6 spec \"Providers\": \
             `provider test <id>` = cheap completion probe; auth per kind). Probes: {}",
            probes.join("; ")
        )
    }
}

/// Depth-first search for provider-ish records (objects carrying an `id` plus a
/// provider marker), tolerant of array/map wrapping.
fn provider_records(v: &Value, out: &mut Vec<Value>) {
    match v {
        Value::Object(map) => {
            let has_id = map.get("id").and_then(Value::as_str).is_some();
            let providerish = map.contains_key("kind")
                || map.contains_key("base_url")
                || map.contains_key("lan_peer")
                || map.contains_key("models");
            if has_id && providerish {
                out.push(v.clone());
            }
            for child in map.values() {
                provider_records(child, out);
            }
        }
        Value::Array(items) => {
            for child in items {
                provider_records(child, out);
            }
        }
        _ => {}
    }
}

fn find_provider(list: &Value, id: &str) -> Option<Value> {
    let mut records = Vec::new();
    provider_records(list, &mut records);
    records.into_iter().find(|r| r["id"] == json!(id))
}

async fn provider_added(d: &Daemon, record: Value) -> Value {
    let id = record["id"].as_str().unwrap().to_string();
    ProviderApi::add_ok(d, record.clone()).await;
    let list = ProviderApi::list(d).await;
    find_provider(&list, &id).unwrap_or_else(|| {
        panic!("provider `{id}` was accepted but never appears in the provider list: {list}")
    })
}

/// Breaker state, shape-agnostic (spec gap: only `HealthTracker::state()` is
/// pinned). Returns the first string value in the record from either
/// vocabulary, searching nested objects too.
fn breaker_state(record: &Value) -> Option<String> {
    fn walk(v: &Value, out: &mut Option<String>) {
        match v {
            Value::Object(map) => {
                for (k, child) in map {
                    if let Some(s) = child.as_str() {
                        let lower = s.to_ascii_lowercase();
                        let keyish = k.to_ascii_lowercase();
                        if keyish.contains("breaker")
                            || keyish.contains("health")
                            || keyish.contains("circuit")
                            || keyish.contains("state")
                        {
                            if matches!(
                                lower.as_str(),
                                "open" | "broken" | "closed" | "healthy" | "degraded" | "tripped"
                            ) {
                                *out = Some(lower);
                                return;
                            }
                        }
                    }
                    walk(child, out);
                }
            }
            Value::Array(items) => {
                for child in items {
                    walk(child, out);
                }
            }
            _ => {}
        }
    }
    let mut found = None;
    walk(record, &mut found);
    found
}

fn is_open(state: &str) -> bool {
    matches!(state, "open" | "broken" | "tripped")
}

// ── Config fixtures ────────────────────────────────────────────────────────

fn base_config(
    local_base: &str,
    manifest_base: &str,
    manifest_enabled: bool,
    extra: &Value,
) -> String {
    let mut root = json!({
        "manifest": {
            "enabled": manifest_enabled,
            "base_url": format!("{manifest_base}/cloud/v1"),
        },
        "llama_swap": {
            "base_url": format!("{local_base}/v1"),
            "fallback_model": "ray-l6-fallback",
            "local_models": ["ray-l6-local"],
        },
        "review": {},
        "decision_model": {"enabled": false, "kind": "heuristic"},
    });
    if let Value::Object(map) = extra {
        for (k, v) in map {
            root[k] = v.clone();
        }
    }
    serde_yaml::to_string(&root).unwrap()
}

fn provider(
    id: &str,
    kind: &str,
    base_url: &str,
    credential: Option<&str>,
    auto: bool,
    models: Vec<Value>,
    lan_peer: bool,
    enabled: bool,
) -> Value {
    let mut record = json!({
        "id": id,
        "kind": kind,
        "base_url": format!("{base_url}/v1"),
        "auto": auto,
        "models": models,
        "lan_peer": lan_peer,
        "enabled": enabled,
        "budget_mode": "warn",
    });
    if let Some(credential) = credential {
        record["credential_id"] = json!(credential);
    }
    record
}

fn model(id: &str, tier: &str) -> Value {
    json!({"id": id, "ctx": 131072, "tools": true, "tier": tier})
}

fn model_with(id: &str, alias: Option<&str>, ctx: u64, tools: bool, tier: &str) -> Value {
    let mut m = json!({"id": id, "ctx": ctx, "tools": tools, "tier": tier});
    if let Some(alias) = alias {
        m["alias"] = json!(alias);
    }
    m
}

/// A fixture: isolated env, two mock gateways (manifest + llama-swap), one
/// live daemon booted from a v1-shaped config.
struct Fixture {
    env: TestEnv,
    local: MockGateway,
    manifest: MockGateway,
    d: Daemon,
}

async fn fixture(tag: &str, manifest_enabled: bool, extra: Value) -> Fixture {
    let env = TestEnv::new(tag);
    let local = MockGateway::spawn(MockCfg {
        models: vec!["ray-l6-local".into(), "ray-l6-fallback".into()],
        ..Default::default()
    });
    let manifest = MockGateway::spawn(MockCfg::default());
    let cfg = base_config(&local.url, &manifest.url, manifest_enabled, &extra);
    let mut d = Daemon::start(&env, &cfg);
    d.ready_cited("L6 fixture daemon did not boot").await;
    Fixture {
        env,
        local,
        manifest,
        d,
    }
}

// ── (10) v1 floor: GREEN at the design base commit ─────────────────────────

/// L6 spec "Contracts kept": "v1 configs boot" and Map R7's reference
/// deployment ("manifest->deepinfra(deepseek) reviewer binding"). This is the
/// owner's live path: a `cloud` card sends `model:"auto"` to Manifest; an
/// explicit `cloud/<id>` passes the ID through verbatim; a reviewer cloud card
/// never hops to the local fallback. It is GREEN now and must stay green
/// through the L6 cutover (where the same path is served by the migrated
/// `manifest` provider).
#[tokio::test(flavor = "multi_thread")]
async fn l6_v1_floor_manifest_cloud_and_reviewer_pinned() {
    let _gate = gate().await;
    let mut fx = fixture("v1-floor", true, json!({})).await;

    // main = cloud, v1 parity: model `auto` to Manifest.
    let (status, text) =
        fx.d.post_json(
            "/api/roles/set",
            json!({"role": "main", "assignment": "cloud"}),
        )
        .await;
    assert_eq!(status, 200, "roles set main cloud: {text}");
    let r = chat_tagged(&fx.d, "brainrouter/auto", None, "r6-floor-auto").await;
    assert_served(&r, "v1 floor: cloud card");
    assert_eq!(
        fx.manifest.models_seen(),
        vec!["auto".to_string()],
        "v1 floor: a `cloud` card must send model \"auto\" to Manifest (L6 spec, parity case 1)"
    );

    let event = event_for(&fx.d, "r6-floor-auto").await;
    assert_v1_event_fields(&event);
    assert_eq!(event["effective_provider"], json!("manifest"));
    assert_eq!(event["stage"], json!("cloud_primary"));
    assert_eq!(event["success"], json!(true));

    // Explicit cloud/<id> passthrough, verbatim.
    fx.manifest.clear();
    let r = chat_tagged(
        &fx.d,
        "cloud/deepseek-ai/DeepSeek-V4-Flash",
        None,
        "r6-floor-pass",
    )
    .await;
    assert_served(&r, "v1 floor: cloud/<id> passthrough");
    assert_eq!(
        fx.manifest.models_seen(),
        vec!["deepseek-ai/DeepSeek-V4-Flash".to_string()],
        "v1 floor: cloud/<id> must pass the id verbatim (L6 spec: \"accepted without discovery\")"
    );

    // Reviewer strict: cloud failure surfaces, never a local hop.
    let (status, text) =
        fx.d.post_json(
            "/api/roles/set",
            json!({"role": "review", "assignment": "cloud/vendor/review-2026"}),
        )
        .await;
    assert_eq!(status, 200, "roles set review: {text}");
    fx.manifest.reset(MockCfg {
        fail_status: Some(500),
        ..Default::default()
    });
    fx.local.clear();
    let before_local = fx.local.count();
    let (status, _body) =
        chat_tagged(&fx.d, "brainrouter/auto", Some("review"), "r6-floor-review").await;
    assert!(
        status >= 400,
        "v1 floor: a reviewer cloud failure must surface (escalation source), got {status}"
    );
    assert_eq!(
        fx.local.count(),
        before_local,
        "v1 floor: role `review` on a cloud selector must NEVER hop to the local fallback"
    );
    let event = event_for(&fx.d, "r6-floor-review").await;
    assert_eq!(event["role"], json!("review"));
    assert_eq!(event["success"], json!(false));
    fx.d.stop();
}

// ── (1) Provider CRUD + Test per kind ──────────────────────────────────────

/// L6 spec "Providers": record shape, `provider add|list|remove|test <id>`,
/// "Test = cheap completion probe, auth per kind", and the outbound adapters:
/// "openai-compat generalization (Bearer)"; "NEW Anthropic outbound adapter
/// (auth = x-api-key + anthropic-version headers, NOT Bearer)"; "absent =
/// auth-less local gateway".
#[tokio::test(flavor = "multi_thread")]
async fn l6_provider_crud_and_test_per_kind_auth_headers() {
    let _gate = gate().await;
    let mut fx = fixture("crud", false, json!({})).await;
    seed_store_credential(&fx.env, "ray-openai-key", L6_SECRET);
    seed_store_credential(&fx.env, "ray-anthropic-key", L6_SECRET);

    let openai_mock = MockGateway::spawn(MockCfg::default());
    let anthropic_mock = MockGateway::spawn(MockCfg::default());
    let anon_mock = MockGateway::spawn(MockCfg::default());

    provider_added(
        &fx.d,
        provider(
            "ray-openai",
            "openai-compatible",
            &openai_mock.url,
            Some("ray-openai-key"),
            false,
            vec![model("openai-model", "balanced")],
            false,
            true,
        ),
    )
    .await;
    provider_added(
        &fx.d,
        provider(
            "ray-anthropic",
            "anthropic",
            &anthropic_mock.url,
            Some("ray-anthropic-key"),
            false,
            vec![model("claude-model", "balanced")],
            false,
            true,
        ),
    )
    .await;
    provider_added(
        &fx.d,
        provider(
            "ray-anon",
            "openai-compatible",
            &anon_mock.url,
            None,
            false,
            vec![model("anon-model", "fast")],
            false,
            true,
        ),
    )
    .await;

    // Test per kind: one cheap completion probe each.
    let (status, text) = ProviderApi::test(&fx.d, "ray-openai").await;
    assert_eq!(
        status, 200,
        "provider test (openai-compat) -> {status}: {text}"
    );
    let probe = openai_mock
        .requests()
        .into_iter()
        .find(|r| r.method == "POST")
        .unwrap_or_else(|| panic!("provider test (openai-compat) sent no completion probe"));
    assert_eq!(
        probe.header("authorization").map(str::to_string),
        Some(format!("Bearer {L6_SECRET}")),
        "L6 spec: OpenAI-compat outbound uses Bearer from credential resolution"
    );

    let (status, text) = ProviderApi::test(&fx.d, "ray-anthropic").await;
    assert_eq!(status, 200, "provider test (anthropic) -> {status}: {text}");
    let probe = anthropic_mock
        .requests()
        .into_iter()
        .find(|r| r.method == "POST")
        .unwrap_or_else(|| panic!("provider test (anthropic) sent no completion probe"));
    assert_eq!(
        probe.header("x-api-key"),
        Some(L6_SECRET),
        "L6 spec: Anthropic auth is `x-api-key`, not Bearer: {:?}",
        probe.headers
    );
    assert!(
        probe
            .header("anthropic-version")
            .is_some_and(|v| !v.is_empty()),
        "L6 spec: Anthropic outbound must send `anthropic-version`: {:?}",
        probe.headers
    );
    assert!(
        probe.header("authorization").is_none(),
        "L6 spec: Anthropic adapter must NOT use Bearer: {:?}",
        probe.headers
    );

    let (status, text) = ProviderApi::test(&fx.d, "ray-anon").await;
    assert_eq!(status, 200, "provider test (auth-less) -> {status}: {text}");
    let probe = anon_mock
        .requests()
        .into_iter()
        .find(|r| r.method == "POST")
        .unwrap_or_else(|| panic!("provider test (auth-less) sent no completion probe"));
    assert!(
        probe.header("authorization").is_none(),
        "L6 spec: a provider record without credential_id is an auth-less local gateway: {:?}",
        probe.headers
    );

    // Remove + list.
    let (status, text) = ProviderApi::remove(&fx.d, "ray-anon").await;
    assert!(
        (200..300).contains(&status),
        "provider remove -> {status}: {text}"
    );
    let list = ProviderApi::list(&fx.d).await;
    assert!(
        find_provider(&list, "ray-anon").is_none(),
        "removed provider still listed: {list}"
    );
    assert!(
        find_provider(&list, "ray-openai").is_some(),
        "provider list lost an untouched provider: {list}"
    );
    fx.d.stop();
}

/// L6 spec "Providers": "live per-provider breaker state (new additive
/// `HealthTracker::state()` read API)" and "Per-provider circuit breakers via
/// `health.rs` (provider-name keyed)". Trip: three backend faults. Heal: the
/// production cooldown (health.rs COOLDOWN_PERIOD = 60 s) is the only heal
/// path this file can observe black-box, so this row waits it out once.
#[tokio::test(flavor = "multi_thread")]
async fn l6_provider_breaker_trip_state_and_heal() {
    let _gate = gate().await;
    let mut fx = fixture("breaker", false, json!({})).await;
    let mock = MockGateway::spawn(MockCfg {
        fail_status: Some(500),
        ..Default::default()
    });
    provider_added(
        &fx.d,
        provider(
            "ray-brk",
            "openai-compatible",
            &mock.url,
            None,
            false,
            vec![model("brk-model", "balanced")],
            false,
            true,
        ),
    )
    .await;
    let (status, text) =
        fx.d.post_json(
            "/api/roles/set",
            json!({"role": "main", "assignment": "cloud/ray-brk/brk-model"}),
        )
        .await;
    assert_eq!(status, 200, "roles set to the tripping provider: {text}");

    // Three backend faults trip the per-provider breaker.
    for i in 0..3 {
        let _ = chat_tagged(&fx.d, "brainrouter/auto", None, &format!("r6-brk-{i}")).await;
    }
    let list = ProviderApi::list(&fx.d).await;
    let record = find_provider(&list, "ray-brk").expect("provider record");
    let state = breaker_state(&record).unwrap_or_else(|| {
        panic!(
            "MISSING SURFACE: provider record carries no breaker/health state \
             (L6 spec \"Providers\": live per-provider breaker state via `HealthTracker::state()`): {record}"
        )
    });
    assert!(
        is_open(&state),
        "three 500s must trip the per-provider breaker, state={state}: {record}"
    );
    let hits_before = mock.count();

    // Open breaker: skipped, the managed hop still serves from fallback_model.
    let before_local = fx.local.count();
    let r = chat_tagged(&fx.d, "brainrouter/auto", None, "r6-brk-open").await;
    assert_served(&r, "open breaker: managed fallback");
    assert_eq!(
        mock.count(),
        hits_before,
        "an OPEN breaker must skip the provider (no upstream hit)"
    );
    assert_eq!(
        fx.local.count(),
        before_local + 1,
        "fallback_model serves the managed hop"
    );

    // Heal: the 60 s cooldown. The provider is healthy again; the next request
    // is attempted and a success resets the state.
    mock.reset(MockCfg::default());
    tokio::time::sleep(Duration::from_secs(61)).await;
    let r = chat_tagged(&fx.d, "brainrouter/auto", None, "r6-brk-heal").await;
    assert_served(&r, "healed breaker");
    assert_eq!(
        mock.count(),
        hits_before + 1,
        "after the cooldown the provider must be attempted again"
    );
    let list = ProviderApi::list(&fx.d).await;
    let record = find_provider(&list, "ray-brk").expect("provider record");
    let state = breaker_state(&record).unwrap_or_default();
    assert!(
        !is_open(&state),
        "a successful probe must heal the breaker, state={state}: {record}"
    );
    fx.d.stop();
}

/// L6 spec "Selector resolution (c)": "two enabled `auto: true` manifest
/// providers is likewise a 4xx configuration error at provider-add time (rule
/// (b) requires uniqueness)".
#[tokio::test(flavor = "multi_thread")]
async fn l6_provider_dual_auto_manifest_refused() {
    let _gate = gate().await;
    let mut fx = fixture("dual-auto", false, json!({})).await;
    let a = MockGateway::spawn(MockCfg::default());
    let b = MockGateway::spawn(MockCfg::default());

    provider_added(
        &fx.d,
        provider(
            "manifest-a",
            "manifest",
            &a.url,
            None,
            true,
            vec![],
            false,
            true,
        ),
    )
    .await;
    let (status, text) = ProviderApi::add(
        &fx.d,
        provider(
            "manifest-b",
            "manifest",
            &b.url,
            None,
            true,
            vec![],
            false,
            true,
        ),
    )
    .await;
    assert!(
        (400..500).contains(&status),
        "L6 spec: a SECOND enabled `auto: true` manifest provider must be a 4xx at add time \
         (rule (b) needs uniqueness); got {status}: {text}"
    );
    let visible = format!("{text} {:?}", ProviderApi::list(&fx.d).await);
    assert!(
        visible.contains("manifest-a") || visible.contains("manifest"),
        "the refusal should name the conflicting providers: {visible}"
    );
    fx.d.stop();
}

/// L6 spec "Providers": `providers.json` state file "0600 atomic", keys from
/// the store, and Map R7 "masked-value audit (no secret in any API/log/
/// config-show)". The vault file itself is the one place a value lives.
#[tokio::test(flavor = "multi_thread")]
async fn l6_provider_state_file_0600_and_masked_everywhere() {
    let _gate = gate().await;
    let mut fx = fixture("masked", false, json!({})).await;
    seed_store_credential(&fx.env, "ray-masked-key", L6_SECRET);
    let mock = MockGateway::spawn(MockCfg::default());
    provider_added(
        &fx.d,
        provider(
            "ray-masked",
            "openai-compatible",
            &mock.url,
            Some("ray-masked-key"),
            false,
            vec![model("masked-model", "balanced")],
            false,
            true,
        ),
    )
    .await;

    // providers.json: 0600, and it stores the credential ID, never the value.
    let path = fx.env.providers_path();
    assert!(
        path.exists(),
        "MISSING SURFACE: providers.json must live in the state dir (L6 spec \"Providers\"): {}",
        path.display()
    );
    assert_eq!(
        mode_of(&path),
        0o600,
        "providers.json must be 0600 (owner-only)"
    );
    let raw = fs::read_to_string(&path).unwrap();
    assert!(
        !raw.contains(L6_SECRET),
        "providers.json must bind `credential_id`, never the key value"
    );
    assert!(
        raw.contains("ray-masked") && raw.contains("ray-masked-key"),
        "providers.json must record the provider id and its credential id: {raw}"
    );

    // The provider actually uses the value (so the audit is not vacuous).
    let (status, text) = ProviderApi::test(&fx.d, "ray-masked").await;
    assert_eq!(status, 200, "provider test: {text}");
    assert_eq!(
        mock.requests()
            .iter()
            .find(|r| r.method == "POST")
            .and_then(|r| r.header("authorization").map(str::to_string)),
        Some(format!("Bearer {L6_SECRET}")),
        "the bound credential value must reach the provider"
    );

    // Rotation: the new value takes over immediately, the old one never leaks
    // into a rendering path either.
    let (status, text) = fx
        .d
        .post_json(
            "/api/credentials/add",
            json!({"id": "ray-masked-key", "description": "rotated", "kind": "store", "value": L6_SECRET_ROTATED}),
        )
        .await;
    assert_eq!(status, 200, "rotate the provider credential: {text}");
    mock.clear();
    let (status, text) = ProviderApi::test(&fx.d, "ray-masked").await;
    assert_eq!(status, 200, "provider test after rotation: {text}");
    assert_eq!(
        mock.requests()
            .iter()
            .find(|r| r.method == "POST")
            .and_then(|r| r.header("authorization").map(str::to_string)),
        Some(format!("Bearer {L6_SECRET_ROTATED}")),
        "rotation must reach the provider immediately"
    );

    // Masked everywhere: APIs, CLI list, config-show, the daemon log.
    let mut surfaces: Vec<(String, String)> = Vec::new();
    for path in [
        "/api/providers",
        "/api/providers/list",
        "/api/credentials",
        "/api/credentials/audit",
        "/api/roles",
        "/api/status",
        "/api/inflight",
    ] {
        let (status, text) = fx.d.get_text(path).await;
        if status < 400 {
            surfaces.push((format!("GET {path}"), text));
        }
    }
    for args in [
        vec!["provider", "list"],
        vec!["credential", "list"],
        vec!["config", "show"],
    ] {
        let run = fx.d.cli(&args);
        surfaces.push((format!("cli {}", args.join(" ")), run.out()));
    }
    surfaces.push(("daemon log".into(), fx.env.daemon_log()));
    for (where_, body) in &surfaces {
        assert!(
            !body.contains(L6_SECRET) && !body.contains(L6_SECRET_ROTATED),
            "MASKED-VALUE AUDIT FAILED: key material appeared in {where_} (Map R7): {body}"
        );
    }
    // The masked list still names the credential.
    let (_, creds) = fx.d.get_text("/api/credentials").await;
    assert!(
        creds.contains("***") || creds.contains("masked"),
        "masked credential rendering expected: {creds}"
    );
    fx.d.stop();
}

// ── (2) Resolution matrix ──────────────────────────────────────────────────

/// L6 spec rule (a): "qualified `cloud/<provider>/<model>` is taken ONLY when
/// the first segment names an ENABLED PROVIDER" and the qualified form is the
/// federation form ("`cloud/fedora1/deepseek`"). Also "dispatch precedence:
/// qualified -> 2-part -> bare, checked BEFORE the named-model catch-all".
#[tokio::test(flavor = "multi_thread")]
async fn l6_resolution_qualified_provider_dispatch_and_peer() {
    let _gate = gate().await;
    let mut fx = fixture("qualified", false, json!({})).await;
    let peer = MockGateway::spawn(MockCfg::default());
    provider_added(
        &fx.d,
        provider(
            "fedora1",
            "openai-compatible",
            &peer.url,
            None,
            false,
            vec![model("deepseek", "balanced")],
            true,
            true,
        ),
    )
    .await;

    let (status, text) =
        fx.d.post_json(
            "/api/roles/set",
            json!({"role": "review", "assignment": "cloud/fedora1/deepseek"}),
        )
        .await;
    assert_eq!(status, 200, "roles set cloud/fedora1/deepseek: {text}");

    let r = chat_tagged(&fx.d, "brainrouter/auto", Some("review"), "r6-qualified").await;
    assert_served(&r, "qualified dispatch");
    assert_eq!(
        peer.models_seen(),
        vec!["deepseek".to_string()],
        "L6 rule (a): `cloud/fedora1/deepseek` must dispatch the EXACT model id to provider fedora1"
    );
    assert_eq!(
        fx.manifest.count(),
        0,
        "a qualified dispatch must never touch Manifest"
    );

    // Direct selector form (dispatch precedence before the named-model
    // catch-all).
    peer.clear();
    let r = chat_tagged(&fx.d, "cloud/fedora1/deepseek", None, "r6-qualified-direct").await;
    assert_served(&r, "qualified direct selector");
    assert_eq!(peer.models_seen(), vec!["deepseek".to_string()]);
    fx.d.stop();
}

/// L6 spec rule (b)+(d): `cloud/<id>` with NO unique model match AND exactly
/// one enabled `auto: true` manifest provider passes through verbatim; and the
/// qualified form is taken ONLY when the first segment is an enabled provider,
/// so `cloud/deepinfra/deepseek-ai/DeepSeek-V4-Flash` (no provider `deepinfra`)
/// cascades to the v1-parity passthrough, byte-identical to the recorded v1
/// fixture. Adding provider `deepinfra` flips the same selector to qualified
/// dispatch without touching the manifest leg.
#[tokio::test(flavor = "multi_thread")]
async fn l6_resolution_deepinfra_selector_passthrough_then_qualified() {
    let _gate = gate().await;
    let mut fx = fixture("passthrough", true, json!({})).await;

    // (b) v1 parity: whole remainder verbatim to the manifest provider.
    let r = chat_tagged(
        &fx.d,
        "cloud/deepinfra/deepseek-ai/DeepSeek-V4-Flash",
        None,
        "r6-deepinfra-pass",
    )
    .await;
    assert_served(&r, "deepinfra passthrough");
    assert_eq!(
        fx.manifest.models_seen(),
        vec![V1_DEEPINFRA_PASSTHROUGH.to_string()],
        "L6 rule (d): no enabled provider `deepinfra`, so the WHOLE remainder \
         `{}` must pass through verbatim (byte-identical v1 fixture)",
        V1_DEEPINFRA_PASSTHROUGH
    );
    let event = event_for(&fx.d, "r6-deepinfra-pass").await;
    assert_eq!(event["model_key"], json!(V1_DEEPINFRA_PASSTHROUGH));
    assert_eq!(event["stage"], json!("cloud_primary"));

    // (a) The same selector, now that provider `deepinfra` EXISTS, dispatches
    // to that provider with the exact model id.
    let provider_mock = MockGateway::spawn(MockCfg::default());
    provider_added(
        &fx.d,
        provider(
            "deepinfra",
            "openai-compatible",
            &provider_mock.url,
            None,
            false,
            vec![model("deepseek-ai/DeepSeek-V4-Flash", "deep")],
            false,
            true,
        ),
    )
    .await;
    let manifest_before = fx.manifest.count();
    let r = chat_tagged(
        &fx.d,
        "cloud/deepinfra/deepseek-ai/DeepSeek-V4-Flash",
        None,
        "r6-deepinfra-qualified",
    )
    .await;
    assert_served(&r, "deepinfra qualified");
    assert_eq!(
        provider_mock.models_seen(),
        vec!["deepseek-ai/DeepSeek-V4-Flash".to_string()],
        "L6 rule (a): with provider `deepinfra` enabled the selector dispatches to it"
    );
    assert_eq!(
        fx.manifest.count(),
        manifest_before,
        "qualified dispatch must not also hit the manifest provider"
    );
    fx.d.stop();
}

/// L6 spec rule (c) + pool rules: bare `cloud` = "the enabled manifest-kind
/// provider with `model:\"auto\"` when one exists (v1 parity), else the auto
/// pool's top cloud candidate"; the pool is "filtered by ctx/tools, ranked by
/// tier fit, deterministic tie-break = provider record order, then model list
/// order".
#[tokio::test(flavor = "multi_thread")]
async fn l6_resolution_bare_cloud_manifest_first_and_provider_pool_tiebreak() {
    let _gate = gate().await;
    let mut fx = fixture("pool", true, json!({})).await;
    let p1 = MockGateway::spawn(MockCfg::default());
    let p2 = MockGateway::spawn(MockCfg::default());

    // v1 parity first: enabled manifest provider present -> "auto".
    provider_added(
        &fx.d,
        provider(
            "pool-one",
            "openai-compatible",
            &p1.url,
            None,
            false,
            vec![model("p1-m1", "balanced"), model("p1-m2", "balanced")],
            false,
            true,
        ),
    )
    .await;
    provider_added(
        &fx.d,
        provider(
            "pool-two",
            "openai-compatible",
            &p2.url,
            None,
            false,
            vec![model("p2-m1", "balanced")],
            false,
            true,
        ),
    )
    .await;
    let (status, text) =
        fx.d.post_json(
            "/api/roles/set",
            json!({"role": "main", "assignment": "cloud"}),
        )
        .await;
    assert_eq!(status, 200, "roles set main cloud: {text}");
    let r = chat_tagged(&fx.d, "brainrouter/auto", None, "r6-pool-manifest").await;
    assert_served(&r, "bare cloud with a manifest provider");
    assert_eq!(
        fx.manifest.models_seen(),
        vec!["auto".to_string()],
        "L6 rule (c): bare `cloud` prefers the enabled manifest-kind auto provider (v1 parity)"
    );
    assert_eq!(
        p1.count() + p2.count(),
        0,
        "the pool must not be consulted while the manifest provider answers"
    );

    // Remove the migrated manifest provider: the pool's top candidate is
    // provider record order, then model list order (p1, then p1-m1).
    let (status, text) = ProviderApi::remove(&fx.d, "manifest").await;
    assert!(
        (200..300).contains(&status),
        "remove the migrated manifest provider: {status}: {text}"
    );
    fx.manifest.clear();
    let r = chat_tagged(&fx.d, "brainrouter/auto", None, "r6-pool-tie1").await;
    assert_served(&r, "bare cloud pool top");
    assert_eq!(
        p1.models_seen(),
        vec!["p1-m1".to_string()],
        "L6 rule (c): tie-break = provider record order, then model list order"
    );
    assert_eq!(
        p2.count(),
        0,
        "a lower-ranked provider must not be tried first"
    );

    // Remove p1: p2 serves.
    let (status, text) = ProviderApi::remove(&fx.d, "pool-one").await;
    assert!(
        (200..300).contains(&status),
        "remove pool-one: {status}: {text}"
    );
    fx.manifest.clear();
    let r = chat_tagged(&fx.d, "brainrouter/auto", None, "r6-pool-tie2").await;
    assert_served(&r, "bare cloud pool next");
    assert_eq!(
        p2.models_seen(),
        vec!["p2-m1".to_string()],
        "disabling the top provider must fall to the next in record order"
    );

    // tools filter: a provider model with tools=false is excluded when the
    // request carries tool definitions (v1 pool rule: filtered by ctx/tools).
    let (status, text) = ProviderApi::remove(&fx.d, "pool-two").await;
    assert!(
        (200..300).contains(&status),
        "remove pool-two: {status}: {text}"
    );
    let p3 = MockGateway::spawn(MockCfg::default());
    provider_added(
        &fx.d,
        provider(
            "pool-notools",
            "openai-compatible",
            &p3.url,
            None,
            false,
            vec![model_with(
                "no-tools-model",
                None,
                131072,
                false,
                "balanced",
            )],
            false,
            true,
        ),
    )
    .await;
    p3.clear();
    let mut req = fx
        .d
        .client
        .post(fx.d.endpoint("/v1/chat/completions"))
        .json(&json!({
            "model": "brainrouter/auto",
            "messages": [{"role": "user", "content": "ray-l6 r6-pool-tools"}],
            "stream": true,
            "tools": [{"type": "function", "function": {"name": "probe", "parameters": {"type": "object"}}}],
        }));
    req = req.header("x-brainrouter-role", "main");
    let _ = req.send().await.expect("tools request").text().await;
    assert_eq!(
        p3.count(),
        0,
        "a tools=false provider model must be excluded when the request carries tools"
    );
    fx.d.stop();
}

/// L6 spec rules (b)/(d) and resolution hardening: an (a)-miss with a
/// non-unique/no manifest provider cascades, and a `cloud/<local-key>` selector
/// can never resolve to the local model (a cloud prefix must never escape to
/// llama-swap).
#[tokio::test(flavor = "multi_thread")]
async fn l6_resolution_qualified_miss_cascade_and_cloud_prefix_never_local() {
    let _gate = gate().await;
    let mut fx = fixture("cascade", true, json!({})).await;

    // `cloud/<local-key>`: the local key exists on llama-swap, but the cloud
    // prefix must NEVER route locally.
    fx.manifest.clear();
    fx.local.clear();
    let r = chat_tagged(&fx.d, "cloud/ray-l6-local", None, "r6-cloud-local").await;
    assert_served(&r, "cloud prefix on a local key");
    assert_eq!(
        fx.local.count(),
        0,
        "`cloud/<local-key>` must be impossible locally: the cloud prefix can never escape to llama-swap"
    );
    assert_eq!(
        fx.manifest.models_seen(),
        vec!["ray-l6-local".to_string()],
        "an (a)-miss cascades to the manifest passthrough with the id verbatim"
    );

    // Cascade is not an error: a 2-part id that matches nothing anywhere still
    // reaches the manifest provider.
    fx.manifest.clear();
    let r = chat_tagged(
        &fx.d,
        "cloud/not-a-provider-not-a-model",
        None,
        "r6-cascade",
    )
    .await;
    assert_served(&r, "qualified miss cascade");
    assert_eq!(
        fx.manifest.models_seen(),
        vec!["not-a-provider-not-a-model".to_string()],
        "L6 rule (d): an (a)-miss cascades to (b) passthrough, never an error"
    );
    fx.d.stop();
}

/// L6 spec rule (c) 4xx: "multi-match across providers => explicit 4xx naming
/// the providers, no silent pick". And "`validate_model_id` accepts the 3-part
/// qualified form".
#[tokio::test(flavor = "multi_thread")]
async fn l6_resolution_multi_match_4xx_and_validator_accepts_qualified() {
    let _gate = gate().await;
    let mut fx = fixture("multi", false, json!({})).await;
    let a = MockGateway::spawn(MockCfg::default());
    let b = MockGateway::spawn(MockCfg::default());
    provider_added(
        &fx.d,
        provider(
            "multi-a",
            "openai-compatible",
            &a.url,
            None,
            false,
            vec![model("shared-model", "balanced")],
            false,
            true,
        ),
    )
    .await;
    provider_added(
        &fx.d,
        provider(
            "multi-b",
            "openai-compatible",
            &b.url,
            None,
            false,
            vec![model("shared-model", "balanced")],
            false,
            true,
        ),
    )
    .await;

    // Validator: a 3-part qualified selector is a legal card assignment.
    let (status, text) =
        fx.d.post_json(
            "/api/roles/set",
            json!({"role": "main", "assignment": "cloud/multi-a/shared-model"}),
        )
        .await;
    assert_eq!(
        status, 200,
        "validate_model_id must accept the 3-part qualified form (L6 rule d): {text}"
    );
    let roles = fx.d.get_json("/api/roles").await;
    assert!(
        serde_json::to_string(&roles)
            .unwrap()
            .contains("cloud/multi-a/shared-model"),
        "the qualified assignment must round-trip: {roles}"
    );

    // Multi-match: explicit 4xx naming both providers, zero upstream calls.
    a.clear();
    b.clear();
    let (status, text) = chat_tagged(&fx.d, "cloud/shared-model", None, "r6-multi").await;
    let text = text.unwrap_or_else(|e| panic!("multi-match request failed to read a body: {e}"));
    assert!(
        (400..500).contains(&status),
        "L6 rule (c): a model id matching two enabled providers is an explicit 4xx, got {status}: {text}"
    );
    assert!(
        text.contains("multi-a") && text.contains("multi-b"),
        "the 4xx must name both providers (no silent pick): {text}"
    );
    assert_eq!(
        a.count() + b.count(),
        0,
        "a multi-match must not dispatch anywhere"
    );
    fx.d.stop();
}

/// L6 spec "Peer pooling rule": "`route=cloud` pools never include `lan_peer`
/// models"; peer providers count as peers, not cloud (`cloud_available` = at
/// least one enabled non-`lan_peer` provider).
#[tokio::test(flavor = "multi_thread")]
async fn l6_resolution_peer_pool_excluded_from_cloud_route() {
    let _gate = gate().await;
    // No manifest provider (manifest disabled), only a lan_peer provider.
    let mut fx = fixture("peer-pool", false, json!({})).await;
    let peer = MockGateway::spawn(MockCfg::default());
    provider_added(
        &fx.d,
        provider(
            "peer-only",
            "openai-compatible",
            &peer.url,
            None,
            false,
            vec![model("peer-model", "deep")],
            true,
            true,
        ),
    )
    .await;
    let (status, text) =
        fx.d.post_json(
            "/api/roles/set",
            json!({"role": "main", "assignment": "cloud"}),
        )
        .await;
    assert_eq!(status, 200, "roles set main cloud: {text}");

    let r = chat_tagged(&fx.d, "brainrouter/auto", None, "r6-peer-pool").await;
    let _ = r;
    assert_eq!(
        peer.count(),
        0,
        "L6 peer pooling rule: `route=cloud` pools NEVER include lan_peer models"
    );
    // With no cloud candidate the managed shape falls back locally (or errors);
    // either way the peer is never selected. The v1 floor's local fallback
    // must still be the only local hop.
    assert!(
        fx.manifest.count() == 0,
        "a disabled manifest provider must not serve either"
    );
    fx.d.stop();
}

// ── (3) Parity corpus ──────────────────────────────────────────────────────

/// The recorded corpus as DATA (GREEN by design). Validates the seven
/// behaviors, the unique ids, the four recorded elements per tuple, and the
/// v1 pinned rule for each outbound model. Recording procedure: file header.
#[tokio::test(flavor = "multi_thread")]
async fn l6_parity_corpus_recording_fixture_contract() {
    let corpus: Value = serde_json::from_str(PARITY_CORPUS_JSON)
        .expect("PARITY_CORPUS_JSON must be valid JSON (recording artifact)");
    let cases = corpus.as_array().expect("corpus must be an array");
    assert_eq!(
        cases.len(),
        7,
        "spec lists seven v1 cloud behaviors; recorded {}",
        cases.len()
    );
    let ids: Vec<&str> = cases.iter().filter_map(|c| c["id"].as_str()).collect();
    let mut unique = ids.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), 7, "corpus case ids must be unique: {ids:?}");
    for case in cases {
        let id = case["id"].as_str().unwrap();
        for field in ["spec", "role", "selector", "script", "expected"] {
            assert!(case.get(field).is_some(), "case `{id}` lacks `{field}`");
        }
        for element in ["outbound", "events", "terminal", "breaker"] {
            assert!(
                case["expected"].get(element).is_some(),
                "case `{id}` lacks the recorded element `expected.{element}`"
            );
        }
        let outbound = &case["expected"]["outbound"];
        assert!(outbound["model"].is_string(), "case `{id}`: outbound model");
        assert_eq!(
            outbound["stream"],
            json!(true),
            "case `{id}`: v1 forces stream"
        );
        let events = case["expected"]["events"].as_array().unwrap();
        assert!(!events.is_empty(), "case `{id}`: event sequence");
        for event in events {
            let stage = event["stage"].as_str().unwrap();
            assert!(
                ["cloud_primary", "cloud_fallback"].contains(&stage),
                "case `{id}`: v1 stage vocabulary, got `{stage}`"
            );
            assert!(event["success"].is_boolean(), "case `{id}`: success flag");
        }
        let terminal = case["expected"]["terminal"].as_str().unwrap();
        assert!(
            ["served", "error"].contains(&terminal),
            "case `{id}`: terminal outcome vocabulary"
        );
        assert!(
            case["expected"]["breaker"]["manifest"].is_string(),
            "case `{id}`: breaker transitions are keyed `manifest` (v1 literal)"
        );
    }

    // v1 pinned rules per behavior (read from v1-classic src/router.rs; see
    // tests/v2_l3_acceptance.rs header for the line cites).
    let by_id = |id: &str| cases.iter().find(|c| c["id"] == json!(id)).unwrap();
    assert_eq!(
        by_id("auto-send")["expected"]["outbound"]["model"],
        json!("auto")
    );
    assert_eq!(
        by_id("cloud-id-passthrough-empty-models")["expected"]["outbound"]["model"],
        json!("deepseek-ai/DeepSeek-V4-Flash")
    );
    let peek = by_id("stream-peek-actual-model");
    assert_eq!(
        peek["script"]["first_frame_model"],
        json!("DeepSeek-V4-Flash-actual")
    );
    assert_eq!(
        peek["expected"]["events"][0]["model_key"],
        json!("DeepSeek-V4-Flash-actual"),
        "stream-peek: the event carries the ACTUAL model from the first frame"
    );
    let pseudo = by_id("pseudo-success-sentinel");
    assert_eq!(
        pseudo["expected"]["breaker"]["manifest"],
        json!("healthy"),
        "pseudo-success must NOT report a breaker failure (v1: reachable, cannot fulfill)"
    );
    assert!(
        pseudo["variant"]["expected"]["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["stage"] == json!("cloud_primary") && e["success"] == json!(false)),
        "the empty-stream variant is a cloud_primary failure"
    );
    let breaker = by_id("breaker-key-manifest-429-is-fault");
    assert_eq!(breaker["script"]["status"], json!(429));
    assert_eq!(
        breaker["expected"]["breaker"]["manifest"],
        json!("broken"),
        "429 is a backend fault and trips the breaker after 3 reports"
    );
    let stop = by_id("cloud-to-fallback-stop");
    assert_eq!(stop["expected"]["terminal"], json!("error"));
    assert_eq!(
        stop["expected"]["events"].as_array().unwrap().len(),
        1,
        "v1 managed shape: one FailedAttempt event for the failed cloud->fallback chain, \
         then stop (src/router.rs error branch: Stage::CloudPrimary, provider None)"
    );
    let reviewer = by_id("reviewer-strict");
    assert_eq!(reviewer["expected"]["terminal"], json!("error"));
    assert_eq!(reviewer["expected"]["events"][0]["role"], json!("review"));
    assert!(
        !reviewer["expected"]["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["stage"] == json!("cloud_fallback")),
        "reviewer strict: never a local hop"
    );
}

/// Replays the recorded corpus through the L6 build. Requires v2.0-p6 to have
/// migrated the v1 config into a `manifest` provider first (spec: "Parity
/// corpus ... through v2.0-p6"), so this row is RED until the build lands.
/// Every case runs against a FRESH daemon: the recorded breaker transitions
/// were captured per-case by the v1 recorder, so replaying them on one shared
/// breaker would accumulate state the recording never saw.
#[tokio::test(flavor = "multi_thread")]
async fn l6_parity_corpus_replay_v1_behaviors() {
    let _gate = gate().await;
    let corpus: Value = serde_json::from_str(PARITY_CORPUS_JSON).unwrap();
    for case in corpus.as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        replay_parity_case(
            id,
            case["role"].as_str().unwrap(),
            case["selector"].as_str().unwrap(),
            case["requests"].as_u64().unwrap_or(1) as usize,
            &case["script"],
            &case["expected"],
            "",
        )
        .await;
        if let Some(variant) = case.get("variant") {
            replay_parity_case(
                id,
                case["role"].as_str().unwrap(),
                case["selector"].as_str().unwrap(),
                1,
                &variant["script"],
                &variant["expected"],
                "variant-empty-stream",
            )
            .await;
        }
    }
}

/// One recorded tuple: script the mocks, send the request(s), compare the
/// normalized [outbound, event sequence, terminal, breaker] tuple.
async fn replay_parity_case(
    case_id: &str,
    role: &str,
    selector: &str,
    requests: usize,
    script: &Value,
    expected: &Value,
    flavor: &str,
) {
    let tag = if flavor.is_empty() {
        format!("parity-{case_id}")
    } else {
        format!("parity-{case_id}-{flavor}")
    };
    let case = format!(
        "{case_id}{}",
        if flavor.is_empty() { "" } else { "/variant" }
    );
    let mut fx = fixture(&tag, true, json!({})).await;
    assert!(
        fx.env.providers_path().exists(),
        "MISSING SURFACE: v2.0-p6 must migrate manifest.* into providers.json before the \
         parity corpus can replay (L6 spec \"v1 cloud-parity migration\"); {} is absent",
        fx.env.providers_path().display()
    );
    let list = ProviderApi::list(&fx.d).await;
    assert!(
        find_provider(&list, "manifest").is_some(),
        "MISSING SURFACE: the migrated manifest provider must be registered (id `manifest`): {list}"
    );

    // Reset both gateways for the case script.
    fx.manifest.reset(mock_cfg_from_script(script, false));
    fx.local.reset(mock_cfg_from_script(script, true));
    fx.manifest.clear();
    fx.local.clear();

    let (status, text) =
        fx.d.post_json(
            "/api/roles/set",
            json!({"role": role, "assignment": selector}),
        )
        .await;
    assert_eq!(
        status, 200,
        "case `{case}`: roles set {role} {selector}: {text}"
    );

    let mut last_status = 0u16;
    let mut last_body = String::new();
    for i in 0..requests {
        let prompt_tag = format!("r6-parity-{case_id}-{i}");
        let (status, body) = chat_tagged(&fx.d, "brainrouter/auto", Some(role), &prompt_tag).await;
        last_status = status;
        last_body = body.unwrap_or_else(|e| format!("stream error: {e}"));
    }

    // Normalized outbound: model + stream at the provider gateway.
    let expected_outbound = &expected["outbound"];
    let manifest_hits = fx.manifest.completions();
    assert!(
        !manifest_hits.is_empty(),
        "case `{case}`: the manifest provider was never called (v2.0-p6 provider dispatch)"
    );
    let first = &manifest_hits[0].body;
    assert_eq!(
        first["model"], expected_outbound["model"],
        "case `{case}`: normalized outbound model"
    );
    assert_eq!(
        first["stream"], expected_outbound["stream"],
        "case `{case}`: v1 forces stream:true outbound"
    );

    // Terminal outcome.
    let expected_terminal = expected["terminal"].as_str().unwrap();
    if expected_terminal == "served" {
        assert_eq!(
            last_status, 200,
            "case `{case}`: expected served, body: {last_body}"
        );
        assert!(
            last_body.contains("ray-l6-pong"),
            "case `{case}`: served body: {last_body}"
        );
    } else {
        assert!(
            last_status >= 400,
            "case `{case}`: expected an error terminal, got {last_status}: {last_body}"
        );
    }

    // Event sequence (normalized) for this case.
    let observed: Vec<Value> = events(&fx.d)
        .await
        .into_iter()
        .filter(|e| {
            e["prompt_excerpt"]
                .as_str()
                .map(|p| p.contains(&format!("r6-parity-{case_id}-")))
                .unwrap_or(false)
        })
        .map(|e| {
            json!({
                "stage": e["stage"],
                "success": e["success"],
                "effective_provider": e["effective_provider"],
                "model_key": e["model_key"],
                "role": e["role"],
            })
        })
        .collect();
    for expected_event in expected["events"].as_array().unwrap() {
        let stage = expected_event["stage"].as_str().unwrap();
        let hit = observed
            .iter()
            .find(|e| e["stage"] == json!(stage) && e["success"] == expected_event["success"]);
        assert!(
            hit.is_some(),
            "case `{case}`: recorded event {} not observed; observed: {observed:?}",
            serde_json::to_string(expected_event).unwrap()
        );
        let hit = hit.unwrap();
        if let Some(provider) = expected_event.get("effective_provider") {
            assert_eq!(
                hit["effective_provider"], *provider,
                "case `{case}`: provider"
            );
        }
        if let Some(model_key) = expected_event.get("model_key") {
            assert_eq!(hit["model_key"], *model_key, "case `{case}`: model_key");
        }
        if let Some(role) = expected_event.get("role") {
            assert_eq!(hit["role"], *role, "case `{case}`: role");
        }
    }

    // Breaker transition.
    let expected_breaker = expected["breaker"]["manifest"].as_str().unwrap();
    let list = ProviderApi::list(&fx.d).await;
    let record = find_provider(&list, "manifest").expect("manifest provider");
    let state = breaker_state(&record).unwrap_or_else(|| {
        panic!("MISSING SURFACE: no breaker state on the migrated provider: {record}")
    });
    match expected_breaker {
        "broken" => assert!(
            is_open(&state),
            "case `{case}`: 429-is-fault must trip the `manifest` breaker, state={state}"
        ),
        "degraded" => assert!(
            state == "degraded",
            "case `{case}`: a single fault leaves the breaker degraded, state={state}"
        ),
        "healthy" => assert!(
            !is_open(&state) && state != "degraded",
            "case `{case}`: no reported failure must leave the breaker healthy, state={state}"
        ),
        other => panic!("corpus breaker vocabulary `{other}`"),
    }
    fx.d.stop();
}

/// Mock script for a corpus case (`local` selects the fallback gateway).
fn mock_cfg_from_script(script: &Value, local: bool) -> MockCfg {
    let kind = script["kind"].as_str().unwrap_or("sse");
    let mut cfg = MockCfg::default();
    cfg.models = vec!["ray-l6-local".into(), "ray-l6-fallback".into()];
    if local {
        cfg.fail_completions = script["also_fail_local"].as_bool().unwrap_or(false);
        cfg.fail_status = if cfg.fail_completions {
            script["status"].as_u64().map(|s| s as u16)
        } else {
            None
        };
        return cfg;
    }
    match kind {
        "fail" => {
            cfg.fail_completions = true;
            cfg.fail_status = script["status"].as_u64().map(|s| s as u16);
        }
        "empty_stream" => cfg.empty_stream = true,
        _ => {
            cfg.pseudo_success = script["pseudo_success"].as_bool().unwrap_or(false);
            cfg.actual_model = script["first_frame_model"].as_str().map(str::to_string);
        }
    }
    cfg
}

// ── (4) v2.0-p6 migration ─────────────────────────────────────────────────

/// L6 spec "v1 cloud-parity migration": exact v1 keys (there is NO
/// `manifest.model`), snapshot + `--dry-run` + restore `v2.0-p6`, one-shot,
/// "`manifest.*` YAML removed post-migration", credential from
/// `manifest.api_key_env`, backup. Map contract: "every leg that migrates or
/// creates a state file first snapshots the pre-migration state ... restore is
/// a named command per leg".
#[tokio::test(flavor = "multi_thread")]
async fn l6_migration_v2_p6_manifest_to_providers_snapshot_dryrun_restore() {
    let _gate = gate().await;
    let env = TestEnv::new("mig-p6");
    let local = MockGateway::spawn(MockCfg {
        models: vec!["ray-l6-local".into(), "ray-l6-fallback".into()],
        ..Default::default()
    });
    let manifest = MockGateway::spawn(MockCfg::default());
    let extra_env = [(
        "RAY_L6_MANIFEST_KEY".to_string(),
        L6_MANIFEST_KEY.to_string(),
    )];
    let v1_yaml = base_config(&local.url, &manifest.url, true, &json!({}));
    // The v1 key set is EXACTLY enabled + base_url + api_key_env.
    let mut v1_root: Value = serde_yaml::from_str(&v1_yaml).unwrap();
    v1_root["manifest"]["api_key_env"] = json!("RAY_L6_MANIFEST_KEY");
    let v1_yaml = serde_yaml::to_string(&v1_root).unwrap();
    let v1_bytes = v1_yaml.clone().into_bytes();
    fs::write(env.config_path(), &v1_yaml).unwrap();

    // --dry-run: previews v2.0-p6, writes nothing.
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_brainrouter"));
    cmd.args(["serve", "--config"])
        .arg(env.config_path())
        .arg("--dry-run")
        .current_dir(&env.root)
        .stdin(Stdio::null());
    env.apply(&mut cmd, &extra_env);
    let out = cmd.output().expect("serve --dry-run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        out.status.code(),
        Some(0),
        "serve --dry-run must exit 0: {stdout}"
    );
    assert!(
        stdout.contains("v2.0-p6"),
        "MISSING SURFACE: --dry-run must preview the v2.0-p6 manifest migration \
         (L6 spec \"v1 cloud-parity migration\": snapshot, --dry-run, restore v2.0-p6): {stdout}"
    );
    assert!(
        !env.providers_path().exists(),
        "--dry-run wrote providers.json (snapshot + dry-run contract)"
    );
    assert!(
        !env.state_dir().join("migrations").join("v2.0-p6").exists(),
        "--dry-run wrote the v2.0-p6 snapshot"
    );

    // Boot: migrate.
    let mut d = Daemon::start_with_env(&env, &v1_yaml, &extra_env);
    d.ready_cited("L6 spec \"v1 cloud-parity migration\": a v1 config must boot and migrate to providers.json")
        .await;
    let providers_text = poll_until("the v2.0-p6 providers.json", 15, || async {
        fs::read_to_string(env.providers_path()).ok()
    })
    .await;
    assert_eq!(
        mode_of(&env.providers_path()),
        0o600,
        "providers.json must be 0600"
    );
    let providers: Value = serde_json::from_str(&providers_text)
        .unwrap_or_else(|e| panic!("providers.json is not JSON ({e}): {providers_text}"));
    let record = find_provider(&providers, "manifest").unwrap_or_else(|| {
        panic!(
            "L6 migration table: `manifest.enabled/base_url/api_key_env` must produce provider \
             {{id: \"manifest\", kind: manifest, auto: true, models: []}}: {providers}"
        )
    });
    assert_eq!(record["kind"], json!("manifest"), "{record}");
    assert_eq!(
        record["base_url"],
        json!(format!("{}/cloud/v1", manifest.url)),
        "{record}"
    );
    assert_eq!(record["auto"], json!(true), "{record}");
    assert_eq!(
        record["models"],
        json!([]),
        "migrated models list is EMPTY (the B1 case): {record}"
    );
    let credential = record["credential_id"].as_str().unwrap_or("");
    assert!(
        credential.contains("RAY_L6_MANIFEST_KEY"),
        "manifest.api_key_env must migrate to a credential reference: {record}"
    );
    assert!(
        record.get("model").is_none(),
        "there is NO `manifest.model` key in v1; the record must not invent one: {record}"
    );

    // Budget accumulator created by the migration (state file), 0600.
    let ledger = env
        .ledger_paths()
        .into_iter()
        .find(|p| p.exists())
        .unwrap_or_else(|| {
            panic!(
                "MISSING SURFACE: v2.0-p6 must create the monthly provider budget accumulator \
                 (L6 spec \"Budget guard\": budget.json in the state dir, 0600): {:?}",
                env.ledger_paths()
            )
        });
    assert_eq!(mode_of(&ledger), 0o600, "{} must be 0600", ledger.display());

    // YAML keys stripped, backup keeps the original bytes, snapshot keeps the
    // pre-migration bytes.
    let migrated_yaml = fs::read_to_string(env.config_path()).unwrap();
    assert!(
        !migrated_yaml.contains("manifest:"),
        "L6 spec: `manifest.*` YAML is REMOVED post-migration: {migrated_yaml}"
    );
    let backup = fs::read(env.root.join("config.yaml.bak")).unwrap_or_else(|e| {
        panic!("MISSING SURFACE: migration must leave a `.bak` backup (existing framework): {e}")
    });
    assert_eq!(
        backup, v1_bytes,
        "the `.bak` must be the pre-migration YAML byte-for-byte"
    );
    let snapshot_dir = env.state_dir().join("migrations").join("v2.0-p6");
    let snapshot_yaml = fs::read(snapshot_dir.join("config.yaml")).unwrap_or_else(|e| {
        panic!(
            "MISSING SURFACE: v2.0-p6 snapshot directory (state-snapshot contract) at {}: {e}",
            snapshot_dir.display()
        )
    });
    assert_eq!(
        snapshot_yaml, v1_bytes,
        "the snapshot must capture the pre-migration YAML verbatim"
    );

    // The migrated v1 config still routes cloud end-to-end (v1 floor through
    // the manifest PROVIDER, models=[] passthrough).
    let (status, text) = d
        .post_json(
            "/api/roles/set",
            json!({"role": "main", "assignment": "cloud"}),
        )
        .await;
    assert_eq!(status, 200, "roles set: {text}");
    let r = chat_tagged(&d, "brainrouter/auto", None, "r6-p6-cloud").await;
    assert_served(&r, "post-migration cloud");
    assert_eq!(
        manifest.models_seen(),
        vec!["auto".to_string()],
        "the migrated manifest provider must serve bare `cloud` with model auto (v1 parity)"
    );
    d.stop();

    // One-shot: a second boot over the migrated state writes no bytes.
    let providers_before = fs::read(env.providers_path()).unwrap();
    let yaml_before = fs::read(env.config_path()).unwrap();
    let mut d2 = Daemon::start_from_disk(&env);
    d2.ready_cited("v1 configs boot after the v2.0-p6 migration")
        .await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        fs::read(env.providers_path()).unwrap(),
        providers_before,
        "a second boot must be a v2.0-p6 no-op (one-shot)"
    );
    assert_eq!(
        fs::read(env.config_path()).unwrap(),
        yaml_before,
        "a second boot must not rewrite the migrated YAML"
    );
    d2.stop();

    // Restore: byte round-trip to the v1 config, providers.json back to
    // pre-migration absence.
    let config_arg = env.config_path().display().to_string();
    let run = run_cli(
        &env,
        &["migration", "--config", &config_arg, "restore", "v2.0-p6"],
        &extra_env,
    );
    run.expect_ok("migration restore v2.0-p6");
    assert_eq!(
        fs::read(env.config_path()).unwrap(),
        v1_bytes,
        "restore v2.0-p6 must round-trip the v1 YAML byte-for-byte"
    );
    assert!(
        !env.providers_path().exists(),
        "restore v2.0-p6 must return providers.json to its pre-migration (absent) state"
    );
}

// ── (5) LAN federation ─────────────────────────────────────────────────────

fn lan_config(
    local_base: &str,
    manifest_base: &str,
    manifest_enabled: bool,
    bind: &str,
    token_credential: &str,
    expose: Vec<&str>,
    burst: u64,
    per_sec: u64,
) -> String {
    base_config(
        local_base,
        manifest_base,
        manifest_enabled,
        &json!({
            "lan": {
                "enabled": true,
                "bind": bind,
                "token_credential": token_credential,
                "expose": expose,
                "rate": {"burst": burst, "per_sec": per_sec},
            }
        }),
    )
}

fn reserved_addr() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    addr.to_string()
}

async fn wait_lan_ready(client: &reqwest::Client, url: &str, token: &str) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let (status, _) = lan_request(
            client,
            url,
            "GET",
            "/api/peer-info",
            Some(token),
            None,
            None,
        )
        .await;
        if status == 200 {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "MISSING SURFACE: the LAN second listener never answered on {url} (L6 spec \"LAN \
             federation\": `lan: {{enabled, bind, token_credential, expose, rate}}`, route table \
             /v1/chat/completions, /v1/messages, /v1/models, /api/peer-info). Last probe: {status}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// L6 spec "LAN federation": second listener, bearer constant-time against the
/// store credential, 401 otherwise, per-token rate limit (defaults burst 20 /
/// per_sec 4), admin paths NOT registered on the LAN socket (404 at listener
/// level), rotation immediate.
#[tokio::test(flavor = "multi_thread")]
async fn l6_lan_second_listener_auth_rotation_rate_and_admin_404() {
    let _gate = gate().await;
    let env = TestEnv::new("lan-auth");
    seed_store_credential(&env, "lan-token", L6_LAN_TOKEN);
    let local = MockGateway::spawn(MockCfg {
        models: vec!["ray-l6-local".into(), "ray-l6-fallback".into()],
        ..Default::default()
    });
    let manifest = MockGateway::spawn(MockCfg::default());
    let lan_addr = reserved_addr();
    let cfg = lan_config(
        &local.url,
        &manifest.url,
        false,
        &lan_addr,
        "lan-token",
        vec!["ray-l6-local"],
        2,
        1,
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready_cited("L6 LAN listener fixture").await;
    let lan_url = format!("http://{lan_addr}");
    wait_lan_ready(&d.client, &lan_url, L6_LAN_TOKEN).await;

    // Missing and wrong bearers.
    let (status, text) = lan_request(
        &d.client,
        &lan_url,
        "GET",
        "/api/peer-info",
        None,
        None,
        None,
    )
    .await;
    assert_eq!(status, 401, "LAN without a bearer must be 401: {text}");
    let (status, text) = lan_request(
        &d.client,
        &lan_url,
        "GET",
        "/api/peer-info",
        Some("nope"),
        None,
        None,
    )
    .await;
    assert_eq!(status, 401, "LAN with a bad bearer must be 401: {text}");

    // Rotation is immediate (constant-time compare against the RESOLVED store
    // value on every request, no restart).
    let (status, text) = d
        .post_json(
            "/api/credentials/add",
            json!({"id": "lan-token", "description": "rotated", "kind": "store", "value": L6_LAN_TOKEN_ROTATED}),
        )
        .await;
    assert_eq!(status, 200, "rotate the LAN token: {text}");
    let (status, _) = lan_request(
        &d.client,
        &lan_url,
        "GET",
        "/api/peer-info",
        Some(L6_LAN_TOKEN),
        None,
        None,
    )
    .await;
    assert_eq!(
        status, 401,
        "the OLD token must stop working immediately after rotation"
    );
    let (status, text) = lan_request(
        &d.client,
        &lan_url,
        "GET",
        "/api/peer-info",
        Some(L6_LAN_TOKEN_ROTATED),
        None,
        None,
    )
    .await;
    assert_eq!(status, 200, "the NEW token must work immediately: {text}");

    // Admin/dashboard/credentials/roles endpoints are not registered at all.
    for (method, path) in [
        ("GET", "/api/roles"),
        ("GET", "/api/credentials"),
        ("GET", "/api/providers"),
        ("GET", "/api/engines"),
        ("GET", "/dashboard"),
        ("POST", "/api/roles/set"),
        ("GET", "/api/credentials/export"),
    ] {
        let body = if method == "POST" {
            Some(json!({"role": "main", "assignment": "cloud"}))
        } else {
            None
        };
        let (status, text) = lan_request(
            &d.client,
            &lan_url,
            method,
            path,
            Some(L6_LAN_TOKEN_ROTATED),
            body,
            None,
        )
        .await;
        assert_eq!(
            status, 404,
            "L6 spec: admin/dashboard/credentials endpoints are NOT registered on the LAN \
             listener (listener-level 404); {method} {path} -> {status}: {text}"
        );
    }

    // Per-token token bucket: burst 2, per_sec 1. The peer-info readiness probe
    // above consumed part of the bucket, so drain with a fresh bucket token by
    // waiting one second first.
    tokio::time::sleep(Duration::from_millis(1100)).await;
    let mut statuses = Vec::new();
    for _ in 0..6 {
        let (status, text) = lan_request(
            &d.client,
            &lan_url,
            "GET",
            "/v1/models",
            Some(L6_LAN_TOKEN_ROTATED),
            None,
            None,
        )
        .await;
        statuses.push((status, text));
    }
    let rate_limited = statuses
        .iter()
        .find(|(status, _)| *status == 429 || *status == 503);
    let (status, text) = rate_limited.unwrap_or_else(|| {
        panic!(
            "MISSING SURFACE: no rate limit on the LAN listener (L6 spec: per-token bucket \
             `lan.rate: {{burst, per_sec}}`, excess rejected). Statuses: {:?}",
            statuses.iter().map(|(s, _)| *s).collect::<Vec<_>>()
        )
    });
    if *status == 503 {
        assert!(
            text.contains("-32005"),
            "a 503 rate-limit refusal must carry the shared overloaded code -32005: {text}"
        );
    }
    // The allowed requests are the first two of the burst.
    assert_eq!(
        statuses.iter().filter(|(s, _)| *s == 200).count(),
        2,
        "burst: 2 must allow exactly two then rate-limit; statuses: {:?}",
        statuses.iter().map(|(s, _)| *s).collect::<Vec<_>>()
    );
    d.stop();
}

/// L6 spec "LAN federation" auth: "compared constant-time against the
/// resolved `token_credential`". Constant-time comparison is unobservable
/// black-box, so this row pins the primitive in source (Map section Security:
/// constant-time compare). RED until the LAN implementation exists.
#[tokio::test(flavor = "multi_thread")]
async fn l6_lan_bearer_constant_time_compare_source() {
    let _gate = gate().await;
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = manifest_dir.join("src");
    let mut surfaces = Vec::new();
    walk_rs(&src, &mut surfaces);
    let lan_sources: Vec<(PathBuf, String)> = surfaces
        .into_iter()
        .filter_map(|path| {
            let text = fs::read_to_string(&path).ok()?;
            let mentions_lan = path
                .file_name()
                .map(|n| n.to_string_lossy().to_ascii_lowercase().contains("lan"))
                .unwrap_or(false)
                || text.contains("peer-info")
                || text.contains("peer_info")
                || text.contains("token_credential");
            mentions_lan.then_some((path, text))
        })
        .collect();
    assert!(
        !lan_sources.is_empty(),
        "MISSING SURFACE: no LAN listener/auth implementation in src (L6 spec \"LAN federation\": \
         second listener, bearer compared constant-time against the store credential)"
    );
    let joined = lan_sources
        .iter()
        .map(|(p, t)| format!("{}\n{t}", p.display()))
        .collect::<String>();
    let has_constant_time = joined.contains("ConstantTimeEq")
        || joined.contains("ct_eq")
        || joined.contains("constant_time")
        || joined.contains("subtle::");
    assert!(
        has_constant_time,
        "L6 spec: the bearer comparison must be constant-time (Map Security). No \
         ConstantTimeEq/ct_eq/constant_time primitive found in the LAN sources: {}",
        lan_sources
            .iter()
            .map(|(p, _)| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
}

fn walk_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_rs(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// L6 spec: "Expose double-enforcement: the RAW requested model string must be
/// in `expose`, AND the RESOLVED model (no alias/qualified escape into
/// unexposed cloud spend)."
#[tokio::test(flavor = "multi_thread")]
async fn l6_lan_expose_double_enforcement_raw_and_resolved() {
    let _gate = gate().await;
    let env = TestEnv::new("lan-expose");
    seed_store_credential(&env, "lan-token", L6_LAN_TOKEN);
    let local = MockGateway::spawn(MockCfg {
        models: vec!["ray-l6-local".into(), "ray-l6-fallback".into()],
        ..Default::default()
    });
    let manifest = MockGateway::spawn(MockCfg::default());
    let upstream = MockGateway::spawn(MockCfg::default());
    let lan_addr = reserved_addr();
    // Expose: the local key (unambiguous positive case), one provider model id,
    // and one alias whose RESOLVED target is hidden.
    let cfg = lan_config(
        &local.url,
        &manifest.url,
        true,
        &lan_addr,
        "lan-token",
        vec!["ray-l6-local", "plain-model", "allowed-alias"],
        20,
        4,
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready_cited("L6 LAN listener fixture").await;
    let lan_url = format!("http://{lan_addr}");
    wait_lan_ready(&d.client, &lan_url, L6_LAN_TOKEN).await;

    provider_added(
        &d,
        provider(
            "expose-cloud",
            "openai-compatible",
            &upstream.url,
            None,
            false,
            vec![
                model("plain-model", "balanced"),
                model_with("costly-model", Some("allowed-alias"), 131072, true, "deep"),
            ],
            false,
            true,
        ),
    )
    .await;

    // RAW not exposed.
    upstream.clear();
    let body = json!({"model": "not-exposed", "messages": [{"role": "user", "content": "ray-l6 lan raw"}], "stream": true});
    let (status, text) = lan_request(
        &d.client,
        &lan_url,
        "POST",
        "/v1/chat/completions",
        Some(L6_LAN_TOKEN),
        Some(body),
        None,
    )
    .await;
    assert!(
        (400..500).contains(&status),
        "L6 expose: a RAW model outside `expose` must be refused, got {status}: {text}"
    );
    assert_eq!(
        upstream.count(),
        0,
        "no upstream spend for an unexposed raw model"
    );

    // RESOLVED escape: raw alias IS exposed, but it resolves to an unexposed
    // cloud model.
    upstream.clear();
    let body = json!({"model": "allowed-alias", "messages": [{"role": "user", "content": "ray-l6 lan alias"}], "stream": true});
    let (status, text) = lan_request(
        &d.client,
        &lan_url,
        "POST",
        "/v1/chat/completions",
        Some(L6_LAN_TOKEN),
        Some(body),
        None,
    )
    .await;
    assert!(
        (400..500).contains(&status),
        "L6 expose: an exposed alias resolving to an UNEXPOSED model must be refused (no \
         alias escape into unexposed cloud spend), got {status}: {text}"
    );
    assert_eq!(
        upstream.count(),
        0,
        "no upstream spend for an unexposed resolved model"
    );

    // Both raw and resolved exposed: served (the local key, where raw ==
    // resolved, so the positive case does not depend on provider-name
    // addressability).
    let (status, text) = lan_request(
        &d.client,
        &lan_url,
        "POST",
        "/v1/chat/completions",
        Some(L6_LAN_TOKEN),
        Some(json!({"model": "ray-l6-local", "messages": [{"role": "user", "content": "ray-l6 lan ok"}], "stream": true})),
        None,
    )
    .await;
    assert_eq!(
        status, 200,
        "an exposed raw+resolved model must be served: {status}: {text}"
    );
    assert!(text.contains("ray-l6-pong"), "LAN served body: {text}");
    assert_eq!(
        local.completions().len(),
        1,
        "the exposed local key is routed to llama-swap"
    );
    d.stop();
}

/// L6 spec `peer-info`: "`{host_id, version, models: [{key, ctx, tools, tier,
/// state}], serving: [...]}`", read-only, "models" filtered by `expose`.
#[tokio::test(flavor = "multi_thread")]
async fn l6_lan_peer_info_readonly_shape() {
    let _gate = gate().await;
    let env = TestEnv::new("lan-peerinfo");
    seed_store_credential(&env, "lan-token", L6_LAN_TOKEN);
    let local = MockGateway::spawn(MockCfg {
        models: vec!["ray-l6-local".into(), "ray-l6-fallback".into()],
        ..Default::default()
    });
    let manifest = MockGateway::spawn(MockCfg::default());
    let lan_addr = reserved_addr();
    let cfg = lan_config(
        &local.url,
        &manifest.url,
        false,
        &lan_addr,
        "lan-token",
        vec!["ray-l6-local"],
        20,
        4,
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready_cited("L6 LAN listener fixture").await;
    let lan_url = format!("http://{lan_addr}");
    wait_lan_ready(&d.client, &lan_url, L6_LAN_TOKEN).await;

    let (status, text) = lan_request(
        &d.client,
        &lan_url,
        "GET",
        "/api/peer-info",
        Some(L6_LAN_TOKEN),
        None,
        None,
    )
    .await;
    assert_eq!(status, 200, "peer-info: {text}");
    let info: Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("peer-info is not JSON ({e}): {text}"));
    assert!(
        info["host_id"].is_string(),
        "peer-info must carry `host_id` (L6 spec): {info}"
    );
    assert!(
        info["version"].is_string(),
        "peer-info must carry `version`: {info}"
    );
    let models = info["models"]
        .as_array()
        .unwrap_or_else(|| panic!("peer-info must carry a `models` array: {info}"));
    assert!(
        !models.is_empty(),
        "peer-info must enumerate the exposed models: {info}"
    );
    for entry in models {
        for field in ["key", "ctx", "tools", "tier", "state"] {
            assert!(
                entry.get(field).is_some(),
                "peer-info model entry must carry `{field}` (L6 spec shape): {entry}"
            );
        }
        assert!(
            entry["tier"]
                .as_str()
                .is_some_and(|t| ["fast", "balanced", "deep"].contains(&t)),
            "peer-info tier vocabulary is fast|balanced|deep: {entry}"
        );
    }
    let keys: Vec<&str> = models.iter().filter_map(|m| m["key"].as_str()).collect();
    assert!(
        keys.iter().all(|k| *k == "ray-l6-local"),
        "peer-info must be filtered by `lan.expose` (model enumeration to token holders is \
         accepted, but only for exposed keys): {keys:?}"
    );
    assert!(
        info["serving"].is_array(),
        "peer-info must carry `serving`: {info}"
    );

    // Read-only: no mutations.
    for method in ["POST", "DELETE", "PUT"] {
        let (status, text) = lan_request(
            &d.client,
            &lan_url,
            method,
            "/api/peer-info",
            Some(L6_LAN_TOKEN),
            Some(json!({})),
            None,
        )
        .await;
        assert!(
            status == 404 || status == 405,
            "peer-info is read-only; {method} -> {status}: {text}"
        );
    }
    d.stop();
}

/// L6 spec: "The primary `--tcp_addr` (loopback) is NEVER affected"; "LAN
/// default-off = v1 attack surface byte-for-byte (listener not created)". The
/// assignment contract adds: LAN inference paths 404 on primary, admin paths
/// 404 on LAN (asserted above).
#[tokio::test(flavor = "multi_thread")]
async fn l6_lan_primary_unchanged_and_off_listener_absent() {
    let _gate = gate().await;
    let env = TestEnv::new("lan-primary");
    seed_store_credential(&env, "lan-token", L6_LAN_TOKEN);
    let local = MockGateway::spawn(MockCfg {
        models: vec!["ray-l6-local".into(), "ray-l6-fallback".into()],
        ..Default::default()
    });
    let manifest = MockGateway::spawn(MockCfg::default());
    let lan_addr = reserved_addr();
    let lan_url = format!("http://{lan_addr}");

    // Boot A: LAN off (v1 config). The configured port must stay closed.
    let off_cfg = base_config(&local.url, &manifest.url, false, &json!({}));
    let mut a = Daemon::start(&env, &off_cfg);
    a.ready().await;
    assert!(
        std::net::TcpStream::connect(&lan_addr).is_err(),
        "L6 spec: LAN default-off must NOT create a listener on {}",
        lan_addr
    );
    let (status, text) = a
        .post_json(
            "/api/roles/set",
            json!({"role": "main", "assignment": "cloud"}),
        )
        .await;
    assert_eq!(status, 200, "roles set: {text}");
    let health_a = a.get_text("/health").await;
    let models_a = a.get_text("/v1/models").await;
    let roles_a = a.get_text("/api/roles").await;
    a.stop();

    // Boot B: LAN on, same state. Primary surfaces byte-identical.
    let on_cfg = lan_config(
        &local.url,
        &manifest.url,
        true,
        &lan_addr,
        "lan-token",
        vec!["ray-l6-local"],
        20,
        4,
    );
    let mut b = Daemon::start(&env, &on_cfg);
    b.ready_cited("L6 LAN listener fixture").await;
    let health_b = b.get_text("/health").await;
    let models_b = b.get_text("/v1/models").await;
    let roles_b = b.get_text("/api/roles").await;
    assert_eq!(
        health_a, health_b,
        "the primary /health surface must be byte-unchanged"
    );
    assert_eq!(
        models_a, models_b,
        "the primary /v1/models surface must be byte-unchanged"
    );
    assert_eq!(
        roles_a, roles_b,
        "the primary /api/roles surface must be byte-unchanged"
    );

    // LAN-only discovery endpoint is not on the primary surface.
    let (status, text) = b.get_text("/api/peer-info").await;
    assert_eq!(
        status, 404,
        "L6 assignment contract: LAN inference/discovery paths are not on the primary socket; \
         GET /api/peer-info -> {status}: {text}"
    );
    // And the LAN socket is live.
    wait_lan_ready(&b.client, &lan_url, L6_LAN_TOKEN).await;
    b.stop();
}

// ── (6) Role-header forwarding ─────────────────────────────────────────────

/// L6 spec: "The adapter forwards the request's effective
/// `x-brainrouter-role` header verbatim to `lan_peer` providers" and a LAN
/// inbound request resolves the PEER's role cards (Map section Security).
/// Non-peer providers never receive the header.
#[tokio::test(flavor = "multi_thread")]
async fn l6_role_header_forwarded_to_lan_peer_only() {
    let _gate = gate().await;
    let mut fx = fixture("role-fwd", true, json!({})).await;
    let peer = MockGateway::spawn(MockCfg::default());
    let cloud = MockGateway::spawn(MockCfg::default());
    provider_added(
        &fx.d,
        provider(
            "peer-fedora1",
            "openai-compatible",
            &peer.url,
            None,
            false,
            vec![
                model("peer-main", "balanced"),
                model("peer-sub", "balanced"),
                model("peer-review", "balanced"),
            ],
            true,
            true,
        ),
    )
    .await;
    provider_added(
        &fx.d,
        provider(
            "plain-cloud",
            "openai-compatible",
            &cloud.url,
            None,
            false,
            vec![model("cloud-model", "balanced")],
            false,
            true,
        ),
    )
    .await;

    // Role cards pinned at the peer, one per role.
    for (role, target) in [
        ("main", "cloud/peer-fedora1/peer-main"),
        ("subagent", "cloud/peer-fedora1/peer-sub"),
        ("review", "cloud/peer-fedora1/peer-review"),
    ] {
        let (status, text) =
            fx.d.post_json(
                "/api/roles/set",
                json!({"role": role, "assignment": target}),
            )
            .await;
        assert_eq!(status, 200, "roles set {role} {target}: {text}");
    }

    for role in ["main", "subagent", "review"] {
        peer.clear();
        let r = chat_tagged(
            &fx.d,
            "brainrouter/auto",
            Some(role),
            &format!("r6-role-{role}"),
        )
        .await;
        assert_served(&r, &format!("peer forward {role}"));
        let hit = peer
            .completions()
            .into_iter()
            .next()
            .unwrap_or_else(|| panic!("role `{role}` never reached the peer provider"));
        assert_eq!(
            hit.header("x-brainrouter-role"),
            Some(role),
            "L6 spec: the effective role header must be forwarded VERBATIM to a lan_peer provider \
             (role={role}); headers: {:?}",
            hit.headers
        );
    }

    // No header on the wire -> effective role is main, forwarded as main.
    peer.clear();
    let r = chat_tagged(&fx.d, "brainrouter/auto", None, "r6-role-default").await;
    assert_served(&r, "peer forward default");
    assert_eq!(
        peer.completions()[0].header("x-brainrouter-role"),
        Some("main"),
        "an unlabelled harness request is effective role `main`"
    );

    // Non-peer providers never see the header.
    let (status, text) =
        fx.d.post_json(
            "/api/roles/set",
            json!({"role": "main", "assignment": "cloud/plain-cloud/cloud-model"}),
        )
        .await;
    assert_eq!(status, 200, "roles set main plain-cloud: {text}");
    cloud.clear();
    let r = chat_tagged(&fx.d, "brainrouter/auto", Some("main"), "r6-role-nonpeer").await;
    assert_served(&r, "non-peer dispatch");
    let hit = cloud
        .completions()
        .into_iter()
        .next()
        .expect("non-peer hit");
    assert!(
        hit.header("x-brainrouter-role").is_none(),
        "L6 spec: ONLY `lan_peer` providers receive the forwarded role header; headers: {:?}",
        hit.headers
    );

    // Manifest (non-peer) likewise.
    fx.manifest.clear();
    let (status, text) =
        fx.d.post_json(
            "/api/roles/set",
            json!({"role": "main", "assignment": "cloud"}),
        )
        .await;
    assert_eq!(status, 200, "roles set main cloud: {text}");
    let r = chat_tagged(&fx.d, "brainrouter/auto", Some("main"), "r6-role-manifest").await;
    assert_served(&r, "manifest dispatch");
    let hit = fx
        .manifest
        .completions()
        .into_iter()
        .next()
        .expect("manifest hit");
    assert!(
        hit.header("x-brainrouter-role").is_none(),
        "the migrated manifest provider is not a `lan_peer` and must not receive the role header"
    );
    fx.d.stop();
}

/// L6 spec "Distributed-role acceptance flows (fixture tier)": plan -> peer
/// (role header forwarded, hop event), dev -> local, review -> cloud (a
/// non-peer provider). One session: the events share one conversation
/// fingerprint (cards grouping unchanged, "conversation fingerprints come from
/// request content, not origin").
#[tokio::test(flavor = "multi_thread")]
async fn l6_distributed_role_fixture_plan_dev_review() {
    let _gate = gate().await;
    let mut fx = fixture("distributed", false, json!({})).await;
    let plan_peer = MockGateway::spawn(MockCfg::default());
    let review_cloud = MockGateway::spawn(MockCfg::default());
    provider_added(
        &fx.d,
        provider(
            "peer-fedora1",
            "openai-compatible",
            &plan_peer.url,
            None,
            false,
            vec![model("hipfire-swift", "deep")],
            true,
            true,
        ),
    )
    .await;
    provider_added(
        &fx.d,
        provider(
            "deepinfra",
            "openai-compatible",
            &review_cloud.url,
            None,
            false,
            vec![model("deepseek-v4-flash", "deep")],
            false,
            true,
        ),
    )
    .await;

    let prompt = "ray-l6 r6-distributed one session";

    // plan: explicit peer selector (main-role flow name).
    let resp = fx
        .d
        .client
        .post(fx.d.endpoint("/v1/chat/completions"))
        .json(&json!({"model": "cloud/peer-fedora1/hipfire-swift", "messages": [{"role": "user", "content": prompt}], "stream": true}))
        .header("x-brainrouter-role", "main")
        .send()
        .await
        .expect("plan request");
    let (status, body) = status_body(resp).await;
    assert_eq!(status, 200, "plan request: {body}");
    assert_eq!(
        plan_peer.completions()[0].header("x-brainrouter-role"),
        Some("main"),
        "plan -> peer: role header forwarded"
    );
    assert_eq!(plan_peer.models_seen(), vec!["hipfire-swift".to_string()]);

    // dev: local selector.
    let resp = fx
        .d
        .client
        .post(fx.d.endpoint("/v1/chat/completions"))
        .json(&json!({"model": "local", "messages": [{"role": "user", "content": prompt}], "stream": true}))
        .header("x-brainrouter-role", "main")
        .send()
        .await
        .expect("dev request");
    let (status, body) = status_body(resp).await;
    assert_eq!(status, 200, "dev request: {body}");
    assert_eq!(
        fx.local.models_seen(),
        vec!["ray-l6-fallback".to_string()],
        "dev -> local fallback model"
    );

    // review: cloud provider, review role.
    let resp = fx
        .d
        .client
        .post(fx.d.endpoint("/v1/chat/completions"))
        .json(&json!({"model": "cloud/deepinfra/deepseek-v4-flash", "messages": [{"role": "user", "content": prompt}], "stream": true}))
        .header("x-brainrouter-role", "review")
        .send()
        .await
        .expect("review request");
    let (status, body) = status_body(resp).await;
    assert_eq!(status, 200, "review request: {body}");
    assert_eq!(
        review_cloud.models_seen(),
        vec!["deepseek-v4-flash".to_string()]
    );
    assert!(
        review_cloud.completions()[0]
            .header("x-brainrouter-role")
            .is_none(),
        "review -> cloud is not a lan_peer: no forwarded role header"
    );

    // One session: identical conversation fingerprint across the three events.
    let evs = poll_until("three distributed-role events", 15, || async {
        let evs: Vec<Value> = events(&fx.d)
            .await
            .into_iter()
            .filter(|e| {
                e["prompt_excerpt"]
                    .as_str()
                    .is_some_and(|p| p.contains("r6-distributed one session"))
            })
            .collect();
        (evs.len() >= 3).then_some(evs)
    })
    .await;
    let convs: Vec<String> = evs
        .iter()
        .filter_map(|e| e["conv_id"].as_str().map(str::to_string))
        .collect();
    assert_eq!(
        convs.len(),
        evs.len(),
        "every event carries conv_id: {evs:?}"
    );
    assert!(
        convs.iter().all(|c| !c.is_empty() && c == &convs[0]),
        "one session: all three route events share one conversation fingerprint: {convs:?}"
    );
    fx.d.stop();
}

/// (status, body) for the one-off explicit-selector requests above.
async fn status_body(resp: reqwest::Response) -> (u16, String) {
    let status = resp.status().as_u16();
    (status, resp.text().await.unwrap_or_default())
}

// ── (7) Budget guard ───────────────────────────────────────────────────────

fn ledger_text(env: &TestEnv) -> String {
    env.ledger_paths()
        .into_iter()
        .filter_map(|p| fs::read_to_string(p).ok())
        .collect::<Vec<_>>()
        .join("\n")
}

fn ledger_existing(env: &TestEnv) -> Vec<PathBuf> {
    env.ledger_paths()
        .into_iter()
        .filter(|p| p.exists())
        .collect()
}

fn sum_numbers(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        Value::Object(map) => map.values().map(sum_numbers).sum(),
        Value::Array(items) => items.iter().map(sum_numbers).sum(),
        _ => 0.0,
    }
}

fn ledger_sum(env: &TestEnv) -> f64 {
    env.ledger_paths()
        .into_iter()
        .filter_map(|p| fs::read_to_string(p).ok())
        .filter_map(|t| serde_json::from_str::<Value>(&t).ok())
        .map(|v| sum_numbers(&v))
        .sum()
}

fn budgeted_provider(id: &str, base_url: &str, mode: &str, budget: f64) -> Value {
    let mut record = provider(
        id,
        "openai-compatible",
        base_url,
        None,
        false,
        vec![json!({
            "id": format!("{id}-model"),
            "ctx": 131072,
            "tools": true,
            "tier": "balanced",
            "price": {"in": 1000.0, "out": 1000.0},
        })],
        false,
        true,
    );
    record["monthly_budget_usd"] = json!(budget);
    record["budget_mode"] = json!(mode);
    record
}

async fn set_main_card(d: &Daemon, selector: &str) {
    let (status, text) = d
        .post_json(
            "/api/roles/set",
            json!({"role": "main", "assignment": selector}),
        )
        .await;
    assert_eq!(status, 200, "roles set main {selector}: {text}");
}

/// L6 spec "Budget guard": per-provider `monthly_budget_usd?` + `budget_mode`
/// (default `warn`); the accumulator is a 0600 state file keyed `YYYY-MM`; cost
/// = price x usage tokens, absent usage => `(bytes+3)/4` estimator flagged
/// `estimated: true`; `warn` => event + Credentials-page badge, never blocks.
#[tokio::test(flavor = "multi_thread")]
async fn l6_budget_warn_event_badge_and_estimator_accrual() {
    let _gate = gate().await;
    let mut fx = fixture("budget-warn", false, json!({})).await;
    let mock = MockGateway::spawn(MockCfg::default());
    provider_added(
        &fx.d,
        budgeted_provider("warn-provider", &mock.url, "warn", 0.000001),
    )
    .await;
    set_main_card(&fx.d, "cloud/warn-provider/warn-provider-model").await;

    // Two requests: warn never blocks.
    let r = chat_tagged(&fx.d, "brainrouter/auto", None, "r6-budget-warn-1").await;
    assert_served(&r, "warn-mode request 1");
    assert_eq!(mock.count(), 1, "the provider must be attempted");

    let files = ledger_existing(&fx.env);
    assert!(
        !files.is_empty(),
        "MISSING SURFACE: the budget accumulator must exist in the state dir (L6 spec \"Budget \
         guard\": budget.json, 0600, keyed YYYY-MM); tried {:?}",
        fx.env.ledger_paths()
    );
    for file in &files {
        assert_eq!(mode_of(file), 0o600, "{} must be 0600", file.display());
    }
    let text = ledger_text(&fx.env);
    let month = month_key();
    assert!(
        text.contains(&month),
        "the ledger must be keyed by the calendar month `{month}`: {text}"
    );
    assert!(
        text.contains("warn-provider"),
        "the ledger must be per-provider: {text}"
    );
    assert!(
        text.contains("estimated"),
        "absent usage (the mock sends no usage chunk) must be flagged `estimated: true`: {text}"
    );
    let after_one = ledger_sum(&fx.env);
    assert!(
        after_one > 0.0,
        "the estimator must accrue a positive cost: {text}"
    );

    let r = chat_tagged(&fx.d, "brainrouter/auto", None, "r6-budget-warn-2").await;
    assert_served(&r, "warn-mode request 2 (never blocks)");
    assert_eq!(mock.count(), 2, "warn mode must never block the provider");
    let after_two = ledger_sum(&fx.env);
    assert!(
        after_two > after_one,
        "the accumulator must grow per request: {after_one} -> {after_two}\n{}",
        ledger_text(&fx.env)
    );

    // Event + badge.
    let evs = serde_json::to_string(&events(&fx.d).await).unwrap();
    assert!(
        evs.contains("budget"),
        "L6 spec: a warn-mode over-budget request emits an event naming the budget: {evs}"
    );
    let mut badge = String::new();
    for path in ["/api/credentials", "/api/providers"] {
        let (status, body) = fx.d.get_text(path).await;
        if status < 400 {
            badge.push_str(&body);
        }
    }
    assert!(
        badge.contains("budget"),
        "L6 spec: warn mode surfaces a Credentials-page badge (provider payload names the budget): {badge}"
    );
    fx.d.stop();
}

/// L6 spec "Budget guard": "`stop` => refusal is a PRIMARY failure of the
/// managed two-step shape (triggers the `fallback_model` hop; review =>
/// escalation), not a terminal 402"; "daemon restart cannot reset it";
/// "calendar rollover on key change"; "A `stop`-mode over-budget provider is
/// EXCLUDED from the auto cloud pool".
#[tokio::test(flavor = "multi_thread")]
async fn l6_budget_stop_primary_failure_persistence_and_rollover() {
    let _gate = gate().await;
    let fx = fixture("budget-stop", false, json!({})).await;
    let mock = MockGateway::spawn(MockCfg::default());
    provider_added(
        &fx.d,
        budgeted_provider("stop-provider", &mock.url, "stop", 0.000001),
    )
    .await;
    set_main_card(&fx.d, "cloud/stop-provider/stop-provider-model").await;

    // Request 1 spends past the budget.
    let r = chat_tagged(&fx.d, "brainrouter/auto", None, "r6-budget-stop-1").await;
    assert_served(&r, "stop-mode request 1 (under budget on entry)");
    assert_eq!(mock.count(), 1);

    // Request 2: refused as a PRIMARY failure -> managed fallback_model hop.
    let before_local = fx.local.count();
    let r = chat_tagged(&fx.d, "brainrouter/auto", None, "r6-budget-stop-2").await;
    assert_served(&r, "stop-mode request 2: managed fallback");
    assert_eq!(
        mock.count(),
        1,
        "an over-budget stop provider must not be called"
    );
    assert_eq!(
        fx.local.count(),
        before_local + 1,
        "the fallback_model hop must serve"
    );
    let event = event_for(&fx.d, "r6-budget-stop-2").await;
    let chain = serde_json::to_string(&events(&fx.d).await).unwrap();
    assert!(
        chain.contains("budget")
            || event["error"]
                .as_str()
                .is_some_and(|e| e.to_ascii_lowercase().contains("budget")),
        "the stop refusal must be a cloud_primary failure naming the budget: {chain}"
    );

    // Review card on the same provider: strict, no local hop (escalation
    // precondition, L3 review-loop mapping).
    let (status, text) =
        fx.d.post_json(
            "/api/roles/set",
            json!({"role": "review", "assignment": "cloud/stop-provider/stop-provider-model"}),
        )
        .await;
    assert_eq!(status, 200, "roles set review: {text}");
    let local_before_review = fx.local.count();
    let (status, _) = chat_tagged(
        &fx.d,
        "brainrouter/auto",
        Some("review"),
        "r6-budget-stop-review",
    )
    .await;
    assert!(
        status >= 400,
        "an over-budget stop provider on a review card must surface (escalation), got {status}"
    );
    assert_eq!(
        fx.local.count(),
        local_before_review,
        "review never hops locally"
    );

    // Restart: the accumulator survives.
    let mut d = fx.d;
    d.stop();
    let mut d = Daemon::start_from_disk(&fx.env);
    d.ready_cited("L6 budget ledger restart fixture").await;
    set_main_card(&d, "cloud/stop-provider/stop-provider-model").await;
    let before_local = fx.local.count();
    let r = chat_tagged(&d, "brainrouter/auto", None, "r6-budget-stop-restart").await;
    assert_served(&r, "restart: still over budget -> fallback");
    assert_eq!(
        mock.count(),
        1,
        "daemon restart must not reset the monthly accumulator"
    );
    assert_eq!(fx.local.count(), before_local + 1);

    // Calendar rollover: move the current-month key to a past month (fixture
    // clock, shape-agnostic rewrite) and restart; the provider is allowed.
    let month = month_key();
    for file in ledger_existing(&fx.env) {
        let text = fs::read_to_string(&file).unwrap();
        assert!(
            text.contains(&month),
            "ledger `{}` must hold the current month key `{month}`: {text}",
            file.display()
        );
        fs::write(&file, text.replace(&month, "1999-01")).unwrap();
        fs::set_permissions(&file, fs::Permissions::from_mode(0o600)).unwrap();
    }
    d.stop();
    let mut d = Daemon::start_from_disk(&fx.env);
    d.ready_cited("L6 budget rollover fixture").await;
    set_main_card(&d, "cloud/stop-provider/stop-provider-model").await;
    let r = chat_tagged(&d, "brainrouter/auto", None, "r6-budget-rollover").await;
    assert_served(&r, "calendar rollover re-opens the budget window");
    assert_eq!(
        mock.count(),
        2,
        "after a month-key change the provider is attempted again (calendar rollover)"
    );
    assert!(
        ledger_text(&fx.env).contains(&month),
        "the new month's key must be written on the rollover request: {}",
        ledger_text(&fx.env)
    );
    d.stop();

    // Stop-opened provider excluded from the auto cloud pool (no manifest
    // provider; only the stopped one plus a healthy sibling).
    let mut fx2 = fixture("budget-pool", false, json!({})).await;
    let stopped = MockGateway::spawn(MockCfg::default());
    let healthy = MockGateway::spawn(MockCfg::default());
    provider_added(
        &fx2.d,
        budgeted_provider("pool-stopped", &stopped.url, "stop", 0.000001),
    )
    .await;
    provider_added(
        &fx2.d,
        provider(
            "pool-healthy",
            "openai-compatible",
            &healthy.url,
            None,
            false,
            vec![model("healthy-model", "balanced")],
            false,
            true,
        ),
    )
    .await;
    set_main_card(&fx2.d, "cloud/pool-stopped/pool-stopped-model").await;
    let _ = chat_tagged(&fx2.d, "brainrouter/auto", None, "r6-budget-pool-spend").await;
    set_main_card(&fx2.d, "cloud").await;
    stopped.clear();
    healthy.clear();
    let _ = chat_tagged(&fx2.d, "brainrouter/auto", None, "r6-budget-pool").await;
    assert_eq!(
        stopped.count(),
        0,
        "L6 spec: a stop-mode over-budget provider is EXCLUDED from the auto cloud pool"
    );
    assert_eq!(
        healthy.count(),
        1,
        "the pool must pick the healthy provider"
    );
    fx2.d.stop();
}

// ── (8) Credential env-line export ─────────────────────────────────────────

/// L6 spec "Credential env-line export": `brainrouter credential export
/// --env-line <file>`, 0600, atomic, refuse-if-exists without `--force`,
/// `NAME=VALUE` for store-kind credentials, `env`/`file` kinds pass their
/// references through as comments, masked lists unchanged.
#[tokio::test(flavor = "multi_thread")]
async fn l6_credential_export_env_line_0600_gates() {
    let _gate = gate().await;
    let env = TestEnv::new("cred-export");
    seed_store_credential(&env, "ray-exported", L6_EXPORT_VALUE);
    seed_env_credential(&env, "ray-env-ref", "RAY_L6_ENV_VAR");
    let file_secret = env.root.join("secret.txt");
    fs::write(&file_secret, "ray-l6-file-secret\n").unwrap();
    seed_file_credential(&env, "ray-file-ref", &file_secret);
    let out_path = env.root.join("exported.env");

    // No daemon: the owner command reads the vault directly. Both the spec
    // spelling and the L3-era nesting are probed.
    let forms: Vec<Vec<&str>> = vec![
        vec![
            "credential",
            "export",
            "--env-line",
            out_path.to_str().unwrap(),
        ],
        vec![
            "cli",
            "credential",
            "export",
            "--env-line",
            out_path.to_str().unwrap(),
        ],
    ];
    let mut attempts = Vec::new();
    let mut working_form: Option<Vec<&str>> = None;
    for form in &forms {
        let run = run_cli(&env, form, &[]);
        if run.code == 0 {
            working_form = Some(form.clone());
            break;
        }
        attempts.push(format!(
            "brainrouter {} => exit {} ({})",
            form.join(" "),
            run.code,
            run.out()
        ));
    }
    let working_form = working_form.unwrap_or_else(|| {
        panic!(
            "MISSING SURFACE: `brainrouter credential export --env-line <file>` is not \
             available (L6 spec \"Credential env-line export\", Map R7 reverse path). \
             Attempts:\n{}",
            attempts.join("\n")
        )
    });

    assert!(
        out_path.exists(),
        "export file was not written: {}",
        out_path.display()
    );
    assert_eq!(
        mode_of(&out_path),
        0o600,
        "the env-line export must be 0600"
    );
    let text = fs::read_to_string(&out_path).unwrap();
    let value_line = text
        .lines()
        .find(|l| l.contains(L6_EXPORT_VALUE))
        .unwrap_or_else(|| {
            panic!("the store-kind credential value must be exported as NAME=VALUE: {text}")
        });
    let (name, value) = value_line
        .split_once('=')
        .unwrap_or_else(|| panic!("the exported line must be NAME=VALUE: {value_line}"));
    assert!(
        !name.trim().is_empty()
            && name
                .trim()
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_'),
        "the env-line name must be a valid shell identifier: {value_line}"
    );
    assert_eq!(
        value, L6_EXPORT_VALUE,
        "the exported value must be the vault value"
    );
    let env_comment = text
        .lines()
        .find(|l| l.contains("RAY_L6_ENV_VAR"))
        .unwrap_or_else(|| panic!("env-kind references must pass through as comments: {text}"));
    assert!(
        env_comment.trim_start().starts_with('#'),
        "env-kind references must be COMMENTS, not live assignments: {env_comment}"
    );
    let file_comment = text
        .lines()
        .find(|l| l.contains(&file_secret.display().to_string()))
        .unwrap_or_else(|| panic!("file-kind references must pass through as comments: {text}"));
    assert!(
        file_comment.trim_start().starts_with('#'),
        "file-kind references must be COMMENTS: {file_comment}"
    );
    assert!(
        !text.contains("ray-l6-file-secret"),
        "a file-kind reference must never inline the file's secret: {text}"
    );

    // Refuse-if-exists without --force: the SAME spelling that worked.
    let before = fs::read(&out_path).unwrap();
    let refused = run_cli(&env, &working_form, &[]);
    assert_ne!(
        refused.code,
        0,
        "L6 spec: export must refuse an existing file without --force: {}",
        refused.out()
    );
    assert_eq!(
        fs::read(&out_path).unwrap(),
        before,
        "a refused export must not modify the existing file"
    );

    // --force overwrites.
    let mut forced_form = working_form.clone();
    forced_form.push("--force");
    let forced = run_cli(&env, &forced_form, &[]);
    assert_eq!(
        forced.code,
        0,
        "L6 spec: `--force` must allow overwriting an existing env-line file: {}",
        forced.out()
    );
    assert_eq!(
        fs::read(&out_path).unwrap(),
        before,
        "--force rewrites the same content (the vault did not change)"
    );

    // Masked lists unchanged: the vault value never appears on any API render.
    let raw_store = fs::read_to_string(env.credentials_path()).unwrap();
    assert!(
        raw_store.contains(L6_EXPORT_VALUE),
        "precondition: the vault (0600) is the one place the value lives"
    );
    let mut daemon = fixture_with_env(env, "cred-export-api", true).await;
    let (status, creds) = daemon.d.get_text("/api/credentials").await;
    assert_eq!(status, 200, "credentials list: {creds}");
    assert!(
        !creds.contains(L6_EXPORT_VALUE),
        "export must not change the masked list rendering: {creds}"
    );
    daemon.d.stop();
}

/// Boot a fixture over an existing env (the export row seeds the vault first).
async fn fixture_with_env(env: TestEnv, _tag: &str, manifest_enabled: bool) -> Fixture {
    let local = MockGateway::spawn(MockCfg {
        models: vec!["ray-l6-local".into(), "ray-l6-fallback".into()],
        ..Default::default()
    });
    let manifest = MockGateway::spawn(MockCfg::default());
    let cfg = base_config(&local.url, &manifest.url, manifest_enabled, &json!({}));
    let mut d = Daemon::start(&env, &cfg);
    d.ready_cited("L6 fixture daemon did not boot").await;
    Fixture {
        env,
        local,
        manifest,
        d,
    }
}

// ── (9) Events additive ────────────────────────────────────────────────────

/// L6 spec "Events": "additive `hop: {host, endpoint}` on provider routes with
/// non-loopback base_url"; Map contract "Event schema: additive fields only
/// (role, decision, rationale, hop)". Also the v1 event shape must still
/// parse (a v1 consumer keeps reading these keys).
#[tokio::test(flavor = "multi_thread")]
async fn l6_events_additive_hop_and_v1_shape() {
    let _gate = gate().await;

    // The recorded v1 shape parses as before (additive-only contract).
    let v1_event: Value = serde_json::from_str(V1_EVENT_FIXTURE).expect("v1 event fixture JSON");
    assert_v1_event_fields(&v1_event);

    let mut fx = fixture("events-hop", false, json!({})).await;
    let ip = non_loopback_ip();
    let remote = MockGateway::spawn_on(ip, MockCfg::default());
    provider_added(
        &fx.d,
        provider(
            "remote-provider",
            "openai-compatible",
            &remote.url,
            None,
            false,
            vec![model("remote-model", "balanced")],
            false,
            true,
        ),
    )
    .await;
    set_main_card(&fx.d, "cloud/remote-provider/remote-model").await;

    let r = chat_tagged(&fx.d, "brainrouter/auto", None, "r6-hop-event").await;
    assert_served(&r, "remote provider dispatch");
    let event = event_for(&fx.d, "r6-hop-event").await;
    assert_v1_event_fields(&event);

    // Provider identity (the spec's provider-name keyed breaker/dispatch id):
    // either a dedicated additive field or the existing effective_provider.
    let provider_id = event
        .get("provider")
        .and_then(Value::as_str)
        .or_else(|| event.get("provider_id").and_then(Value::as_str))
        .or_else(|| event.get("effective_provider").and_then(Value::as_str))
        .unwrap_or("");
    assert_eq!(
        provider_id, "remote-provider",
        "a provider route event must name the provider id (L6 spec: per-provider breakers/hops): {event}"
    );

    let hop = event.get("hop").unwrap_or_else(|| {
        panic!(
            "MISSING SURFACE: a provider route with a non-loopback base_url must carry the \
             additive `hop: {{host, endpoint}}` (L6 spec \"Events\"): {event}"
        )
    });
    assert!(
        hop["host"].is_string() && !hop["host"].as_str().unwrap().is_empty(),
        "hop.host must be the remote host: {hop}"
    );
    if ip.to_string() != "127.0.0.2" {
        assert_eq!(
            hop["host"],
            json!(ip.to_string()),
            "hop.host must be the provider base_url host: {hop}"
        );
    }
    assert!(
        hop["endpoint"].is_string() && !hop["endpoint"].as_str().unwrap().is_empty(),
        "hop.endpoint must be the provider endpoint: {hop}"
    );

    // A loopback provider route carries no hop (the field is for LAN hops).
    let loopback = MockGateway::spawn(MockCfg::default());
    provider_added(
        &fx.d,
        provider(
            "loopback-provider",
            "openai-compatible",
            &loopback.url,
            None,
            false,
            vec![model("loopback-model", "balanced")],
            false,
            true,
        ),
    )
    .await;
    set_main_card(&fx.d, "cloud/loopback-provider/loopback-model").await;
    let r = chat_tagged(&fx.d, "brainrouter/auto", None, "r6-hop-loopback").await;
    assert_served(&r, "loopback provider dispatch");
    let event = event_for(&fx.d, "r6-hop-loopback").await;
    assert!(
        event.get("hop").is_none() || event["hop"].is_null(),
        "a loopback base_url is not a LAN hop; `hop` stays absent: {event}"
    );
    fx.d.stop();
}
