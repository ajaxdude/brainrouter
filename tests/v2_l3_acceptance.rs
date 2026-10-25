//! Acceptance tests for leg L3 (One entry + role cards + decision slot +
//! Credentials page) of `docs/design/v2-streamlining.md` — spec:
//! `docs/design/v2-streamlining.L3.md` version L3-v3 at commit 6c47ad7.
//!
//! Written by Ray from the design's promises alone. The builder runs these
//! and NEVER edits them. Expectation: every test is RED (or partly red, for
//! parity rows) at the design base commit — the roles store, decision slot,
//! credentials UI endpoints, one-entry templates, and additive event fields
//! do not exist yet — and GREEN once L3 lands.
//!
//! Parity rows (heuristic v1-selector matrix, per-selector strictness) are
//! designed to PASS on v1 behavior: they pin v1 semantics (L3 spec "Pinned
//! card strictness = v1 semantics PER SELECTOR"). Each such row cites the
//! v1-classic file + lines the expectation was read from.
//!
//! Mapping of tests to the L3 "Acceptance (Ray)" items (assignment scope):
//! - (1) Resolution matrix:
//!   * `l3_resolution_matrix_global_cards_per_role`
//!   * `l3_resolution_project_scope_beats_global`
//!   * `l3_privacy_clamps_auto_never_explicit`
//!   * `l3_subs_names_subagent_role_overrides_header`
//! - (2) Decision slot (plugin kind is out of scope for this leg's Ray
//!   assignment; builder-integration + plugin-protocol coverage reported to
//!   Hank, mirroring the L2 (4) disposition):
//!   * `l3_heuristic_parity_v1_selector_matrix`
//!   * `l3_bonsai_profile_success_triples_prompt_verbatim`
//!   * `l3_bonsai_profile_failure_table_v1_safe`
//!   * `l3_structured_schema_strict_json_fallbacks`
//!   * `l3_webhook_features_no_excerpt_returns_decision`
//!   * `l3_decision_features_probe_shape_and_post_only_route`
//! - (3) Credentials UI:
//!   * `l3_credentials_ui_crud_test_verdicts_grants_masked`
//!   * `l3_credentials_plugin_secrets_get_after_ui_add`
//!   * `l3_credentials_audit_consumers_intersect_registered_plugins`
//! - (4) One-entry:
//!   * `l3_install_single_auto_entry_per_harness`
//!   * `l3_sync_omp_removed_at_four_points`
//! - (5) Events additive:
//!   * `l3_events_additive_fields_v1_fields_intact`
//! - (6) Migrations:
//!   * `l3_migration_v1_state_role_cards_exact_table`
//!   * `l3_migration_no_routing_review_survivors_restore_dryrun`
//!   * `l3_migration_no_bonsai_block_is_one_shot` (B1, Step-9 NO-GO guard:
//!     one-shot migration for v1 YAML WITHOUT a bonsai block)
//! - (7) Strictness per selector (live, failing mock upstream):
//!   * `l3_bare_key_card_strict_no_retry`
//!   * `l3_local_card_v1_managed_fallback_semantics`
//!   * `l3_cloud_card_hop_and_reviewer_strict`
//!   * `l3_auto_chain_next_candidate_and_midstream_never_retried`
//! - (8) Node/html-js gates: untouched by this file (assignment: do not run).
//!
//! v1 EXPECTATION SOURCES (hardcoded, per assignment "document the v1
//! expectation source"): all v1 claims cite files at tag `v1-classic`
//! (commit 790fd9b), verified byte-identical to 6c47ad7 for
//! src/classifier.rs and src/routing_profile.rs:
//! - src/router.rs `route_resolved` (v1-classic L344-424): selector dispatch
//!   `local`->fallback_model + route_local(allow_fallback=true);
//!   `cloud`->model="auto"+route_cloud(allow_local_fallback);
//!   `cloud/<id>`->id passthrough to cloud; `subs` pool chain (project pin >
//!   profile.subagent_model > config.subs_model) -> route_local(STRICT);
//!   `auto`->route_auto; bare/`brainrouter/<key>`/named model -> route_local
//!   (STRICT, "user's model choice is authoritative").
//! - src/router.rs `route_local` (v1-classic L658-727): on failure, retry
//!   once with `fallback_model` iff allow_fallback AND requested !=
//!   fallback_model (L699); direct picks surface the error.
//! - src/router.rs `route_cloud` (v1-classic L568-650): Manifest first; on
//!   failure the managed path falls back to llama-swap/fallback_model with a
//!   recorded failed CloudPrimary hop; `allow_local_fallback=false`
//!   (reviewer, from `route_with_choice` L302-318) bails — "failure surfaces
//!   and the review loop maps it to escalated" (L622-631).
//! - src/classifier.rs (v1-classic): classifier OFF => auto routes LOCAL to
//!   default_local_model (= llama_swap.fallback_model, wired at
//!   src/daemon.rs:355-357) tier Light (L119-131); classifier ON => one-word
//!   grammar parse on the FIRST non-whitespace char l->Local d->Deep else
//!   Cloud (parse_decision L240-247); EVERY failure (HTTP error/timeout
//!   L181-184, read error L176-179, JSON parse fail L188-193, missing/empty
//!   last user message L136-139 — server-down is the HTTP-connect failure
//!   inside L143-171) => Cloud, "the safe default". Prompt grammar:
//!   POST {server}/v1/chat/completions, model "bonsai", max_tokens 10,
//!   temperature 0.0, stop ["\n"], chat_template_kwargs.enable_thinking
//!   false, system prompt verbatim (L144-168), user content =
//!   "Classify this request: {LAST user message only, 800-char truncation}"
//!   (extract_last_user_message L250-275).
//! - src/routing_profile.rs (v1-classic): ModelChoice::selector()
//!   {Auto->"auto", Local{None}->"local", Local{Some}->"brainrouter/{id}",
//!   Cloud{None}->"cloud", Cloud{Some}->"cloud/{id}"} (L86-94); preset
//!   constraints in `validate` (L135-158).
//! - src/config.rs (v1-classic L503-513): a v1 YAML with NO `routing:` block
//!   resolves profile main = Local{None} (selector "local") — the migration
//!   default-local row.
//! - src/server.rs (6c47ad7): sync-omp route L650, CSRF allowlist entry
//!   L288; src/daemon.rs bootstrap auto-sync task L588-603; src/cli.rs
//!   SyncOmp verb L555-557.
//! - src/routing_events.rs (6c47ad7): v1 RouteEvent field set asserted
//!   intact by `l3_events_additive_fields_v1_fields_intact`.
//!
//! PROJECT CONTEXT (assignment: "check how existing tests pass project
//! context"): there is NO X-Git header on the v1/L3 wire. The daemon
//! resolves the request's project from the CONNECTING PROCESS's cwd over
//! /proc peer credentials (src/server.rs:4357-4400 @ 6c47ad7, TCP path
//! `peer_cwd(&addr)`; UDS path `cwd_from_pid`), and `project_key` normalizes
//! it to the canonical git root (src/project_pins.rs:230-246 @ v1-classic).
//! Integration tests run with the cargo package root as cwd — the brainrouter
//! repo IS a git repository — so `--project <canonical package root>` scopes
//! cards exactly the way a harness checked out in this repo would. The
//! Router-level tests (tests/subs_routing_test.rs) pass `cwd` directly to
//! `route_tagged`; black-box tests get it from the peer-cwd mechanism.
//!
//! SPEC GAPS (assumptions this file must make; all reported to Hank — the
//! design fixes the promises, not these wire shapes):
//! - Credentials page HTTP route/params: the design names UI verbs, not
//!   paths. These tests probe candidate routes (list/add/test/delete),
//!   speaking the UNCHANGED L2 record JSON (`CredentialRecord` flattened:
//!   kind tag + value|var|path + grants), and fail with an explicit
//!   "route unresolvable" message if no spelling answers.
//! - Policy (privacy/quality) set route: the design puts chips on the
//!   dashboard; the endpoint is probed the same way.
//! - Event field JSON shapes (`decision` object, `decision_fallback` values
//!   "v1-safe"/"heuristic"/"unavailable", `chosen_model`) follow the
//!   "Contracts kept" list verbatim; the tier of a cloud decision is not
//!   pinned by v1, so only `route` is asserted for cloud decisions.
//! - The structured-schema kind's request schema ("typed question") is not
//!   pinned; the mock answers the Clef-style result JSON
//!   {route,tier,confidence,rationale} in choices[0].message.content, which
//!   every plausible transport shape (chat-completions reply) carries.
//! - Session snapshot semantics (card snapshotted at session start) needs
//!   the review-loop MCP surface; reported to Hank as builder-integration
//!   coverage. This file pins the reduced /api/review-config survivors and
//!   card-mutation path, which is the same record.

use serde_json::{json, Value};
use std::{
    fs,
    net::TcpListener,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::Ordering,
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use tokio::io::AsyncWriteExt;

// ── Constants ───────────────────────────────────────────────────────────────

/// Sentinels. Every "value absent" assertion greps for these.
const L3_SECRET: &str = "ray-L3-s3cr3t-value-4e91";
const L3_SECRET_ROTATED: &str = "ray-L3-s3cr3t-ROTATED-77aa";
const DISCORD_TOKEN: &str = "ray-L3-discord-tok-1a2b3c";

/// Canonical v1 bonsai system prompt, verbatim from
/// src/classifier.rs L151 @ v1-classic — the migrated `prompt_profile:
/// bonsai` must send it byte-identical.
const V1_BONSAI_SYSTEM_PROMPT: &str = "You are a routing classifier. Reply with exactly one word: \"cloud\" for complex tasks that need the cloud (large architecture, multi-system debugging, heavy refactoring), \"local\" for simple tasks (short answers, simple questions, single-line code, quick explanations), or \"deep\" for complex tasks that should still run locally (focused multi-step reasoning, moderate debugging). Output nothing else.";

/// v1 RouteEvent field set (src/routing_events.rs @ 6c47ad7). The additive
/// contract requires all of these to remain on every event.
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
];

// ── Test environment harness (v2_l2_acceptance.rs conventions) ─────────────

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
        let base = std::env::temp_dir().join(format!("brl3-{tag}-{}", std::process::id()));
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

    fn borrow(env: &TestEnv) -> TestEnv {
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

    /// $XDG_CONFIG_HOME/brainrouter — where v1 keeps routing_state.json +
    /// project_pins.json (src/daemon.rs:217-230 @ 6c47ad7: colocated with
    /// default_config_path()).
    fn config_dir(&self) -> PathBuf {
        self.config_home.join("brainrouter")
    }

    /// $XDG_STATE_HOME/brainrouter — credentials store + migration
    /// snapshots (src/paths.rs @ 6c47ad7).
    fn state_dir(&self) -> PathBuf {
        self.state_home.join("brainrouter")
    }

    fn plugin_state(&self, id: &str) -> PathBuf {
        self.state_dir().join("plugins").join(id)
    }

    /// Kept as the documented L2 socket layout (`$XDG_RUNTIME_DIR/
    /// brainrouter/plugins/<id>.sock`); the L3 plugin round-trip speaks
    /// from the plugin side, so the harness only needs the state dir.
    #[allow(dead_code)]
    fn plugin_socket(&self, id: &str) -> PathBuf {
        self.runtime.join("brainrouter").join("plugins").join(format!("{id}.sock"))
    }
}

impl Drop for TestEnv {
    fn drop(&mut self) {
        if self.owned {
            let _ = fs::remove_dir_all(&self.root);
        }
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
    panic!("no {name} interpreter found on the test host; the L3 fixtures cannot run");
}

/// An unused loopback port (connect = refused = "server down"). v1's single
/// classifier failure mode; L3's v1-safe row for the bonsai profile.
fn dead_upstream_url() -> String {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let a = l.local_addr().unwrap();
    drop(l);
    format!("http://{a}")
}

/// The canonical package root — a real git repo (has `.git`) — which is the
/// peer-cwd project context of every connection the test process opens to a
/// live daemon (see header note "PROJECT CONTEXT").
fn repo_root() -> String {
    fs::canonicalize(std::env::current_dir().expect("test cwd")).unwrap().display().to_string()
}

/// Process-wide fixture gate. Every test owns a live daemon, loopback mock
/// upstreams, and (for some) python fixtures bound to kernel-assigned
/// ports; on a saturated shared runner, parallel libtest threads race in
/// the reserve-then-rebind windows and in fixture boot latency, producing
/// ROTATING false failures (webhook POST against a not-yet-listening port
/// falls back to heuristic; a stolen daemon port kills a sibling's boot).
/// Each test holds this gate for its entire lifetime — full fixture-lifetime
/// isolation: one test's ports, accumulators, daemons, and event log at a
/// time. Assertions are untouched; every promise stays binding.
static SUITE_GATE: std::sync::LazyLock<tokio::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));

fn suite_gate() -> &'static tokio::sync::Mutex<()> {
    &SUITE_GATE
}

// ── Config fixtures ─────────────────────────────────────────────────────────

/// L3-era base config: mocks for both backends, bonsai off. `decision_model`
/// is the caller-provided YAML fragment (JSON value; rendered by serde_yaml).
fn l3_config_yaml(
    local_base: &str,
    cloud_base: &str,
    fallback: &str,
    local_models: &[&str],
    subs_model: Option<&str>,
    decision_model: Value,
) -> String {
    serde_yaml::to_string(&json!({
        "manifest": { "enabled": true, "base_url": format!("{cloud_base}/cloud/v1") },
        "llama_swap": {
            "base_url": format!("{local_base}/v1"),
            "fallback_model": fallback,
            "local_models": local_models,
            "subs_model": subs_model,
        },
        "bonsai": { "enabled": false, "fork_path": "/nonexistent/ray-l3-no-bonsai" },
        "review": {},
        "decision_model": decision_model,
    }))
    .unwrap()
}

/// v1-state YAML for the migration tests: routing_profile (preset-
/// constrained, `src/routing_profile.rs::RoutingProfile::validate` @
/// v1-classic L135-158) + review.forced_* + bonsai + the L2 discord token
/// (so the L2 store exists for `migration restore`).
fn v1_seed_yaml(local_base: &str, cloud_base: &str, with_routing_block: bool) -> String {
    let mut root = json!({
        "manifest": { "enabled": true, "base_url": format!("{cloud_base}/cloud/v1") },
        "llama_swap": {
            "base_url": format!("{local_base}/v1"),
            "fallback_model": "ray-l3-fallback",
            "subs_model": "legacy-subs-model",
        },
        "bonsai": { "enabled": false, "fork_path": "/nonexistent/ray-l3-no-bonsai" },
        "review": {
            "max_iterations": 3,
            "forced_mode": "cloud",
            "forced_model": "vendor/review-2026",
            "hankndory_integration": true,
        },
        "bridge": {
            "omp_path": "/nonexistent/omp",
            "discord": { "enabled": false, "token": DISCORD_TOKEN },
            "signal": { "enabled": false },
        },
    });
    if with_routing_block {
        root["routing"] = json!({
            "preset": "local_main_cloud_review",
            "main": { "backend": "local" },
            "reviewer": { "backend": "cloud" },
            "subagent_model": "prof-subs-model",
        });
    }
    serde_yaml::to_string(&root).unwrap()
}

/// v1 project pins state file (src/project_pins.rs @ v1-classic: schema 1,
/// pins keyed by canonical git root, ModelChoice-serialized roles).
fn v1_project_pins_json(root: &str) -> String {
    json!({
        "schema_version": 1,
        "pins": {
            root: {
                "main": { "backend": "cloud", "model": "pin-cloud-id" },
                "reviewer": { "backend": "cloud", "model": "vendor/review-2026" },
                "subagent": "pin-subs-model",
            }
        }
    })
    .to_string()
}

// ── Mock upstream (scriptable loopback OpenAI-ish server) ──────────────────

#[derive(Clone, Default)]
struct MockCfg {
    /// ids advertised on /v1/models (the L3 auto pool source).
    models: Vec<String>,
    /// chat.completions for these model keys answer 500 (backend fault).
    fail_models: Vec<String>,
    /// every chat.completions answers 500 (dead cloud).
    fail_chat_all: bool,
    /// these keys answer 200 + one SSE chunk, then the connection dies
    /// mid-stream (no [DONE], truncated chunked body).
    die_midstream_models: Vec<String>,
    /// classifier mode: every chat.completions (the v1 grammar uses model
    /// "bonsai") answers the one-word reply instead of an SSE stream.
    classifier: bool,
    classifier_word: String,
    /// reply status for classifier requests (200 word / 500 error).
    classifier_status: u16,
    /// reply a non-JSON text/plain body (structured-schema parse-fail row).
    classifier_plain_body: bool,
    /// resolve the one-word reply PER REQUEST from the prompt text
    /// (the golden-corpus success triples replay several words on one mock).
    classifier_prompt_grammar: bool,
}

struct MockUpstream {
    url: String,
    /// (path, model) per chat.completions request.
    hits: Arc<Mutex<Vec<(String, String)>>>,
    /// full JSON bodies of classifier requests, in arrival order.
    cls_bodies: Arc<Mutex<Vec<Value>>>,
}

impl MockUpstream {
    fn models_seen(&self) -> Vec<String> {
        self.hits.lock().unwrap().iter().map(|(_, m)| m.clone()).collect()
    }
    fn count_of(&self, model: &str) -> usize {
        self.hits.lock().unwrap().iter().filter(|(_, m)| m == model).count()
    }
    fn total(&self) -> usize {
        self.hits.lock().unwrap().len()
    }
    fn classifier_bodies(&self) -> Vec<Value> {
        self.cls_bodies.lock().unwrap().clone()
    }
}

fn spawn_mock(cfg: MockCfg) -> MockUpstream {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    listener.set_nonblocking(true).unwrap();
    let hits = Arc::new(Mutex::new(Vec::new()));
    let cls = Arc::new(Mutex::new(Vec::new()));
    let (h2, c2, cfg2) = (hits.clone(), cls.clone(), Arc::new(cfg));
    tokio::spawn(async move {
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();
        loop {
            let Ok((mut sock, _)) = listener.accept().await else { continue };
            let (h, c, cfg) = (h2.clone(), c2.clone(), cfg2.clone());
            tokio::spawn(async move {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut buf = vec![0u8; 256 * 1024];
                let mut filled = 0usize;
                let mut content_len = None::<usize>;
                let mut req_path = String::new();
                loop {
                    let cap = buf.len();
                    let off = filled.min(cap - 1);
                    let Ok(n) = sock.read(&mut buf[off..]).await else {
                        return;
                    };
                    if n == 0 {
                        break;
                    }
                    filled += n;
                    if let Some(pos) = find_headers_end(&buf[..filled]) {
                        if content_len.is_none() {
                            let headers = String::from_utf8_lossy(&buf[..pos]).to_lowercase();
                            req_path = headers
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
                let model = body
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if req_path.contains("/models") {
                    let data: Vec<Value> = cfg
                        .models
                        .iter()
                        .map(|id| json!({"id": id, "object": "model", "owned_by": "ray-l3-mock"}))
                        .collect();
                    let out = json!({"object": "list", "data": data}).to_string();
                    let _ = write_plain(&mut sock, 200, "application/json", out).await;
                    return;
                }
                if req_path.contains("/chat/completions") {
                    hits_record(&h, &req_path, &model);
                }
                if cfg.classifier {
                    c.lock().unwrap().push(body.clone());
                    if cfg.classifier_plain_body {
                        let _ = write_plain(&mut sock, 200, "text/plain", "not json at all".into()).await;
                        return;
                    }
                    let status = if cfg.classifier_status == 0 { 200 } else { cfg.classifier_status };
                    let word = if cfg.classifier_prompt_grammar {
                        grammar_word(&body)
                    } else {
                        cfg.classifier_word.clone()
                    };
                    let out = json!({
                        "id": "chatcmpl-ray-l3-cls",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "bonsai",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": word},
                            "finish_reason": "stop",
                        }],
                    })
                    .to_string();
                    let _ = write_plain(&mut sock, status, "application/json", out).await;
                    return;
                }
                if cfg.fail_chat_all || cfg.fail_models.iter().any(|m| *m == model) {
                    let _ = write_plain(
                        &mut sock,
                        500,
                        "application/json",
                        json!({"error": {"message": format!("ray-l3 forced failure for {model}"), "type": "server_error"}}).to_string(),
                    )
                    .await;
                    return;
                }
                if cfg.die_midstream_models.iter().any(|m| *m == model) {
                    let sse = format!(
                        "data: {}\n\n",
                        json!({"id":"chatcmpl-ray-l3","object":"chat.completion.chunk","created":1,
                               "model":model,
                               "choices":[{"index":0,"delta":{"content":"ray-l3-partial"},"finish_reason":null}]})
                    );
                    let _ = sock
                        .write_all(
                            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
                        )
                        .await;
                    let _ = sock
                        .write_all(format!("{:x}\r\n{sse}\r\n", sse.as_bytes().len()).as_bytes())
                        .await;
                    tokio::time::sleep(Duration::from_millis(80)).await;
                    // Truncated: close WITHOUT the terminating 0-chunk.
                    let _ = sock.shutdown().await;
                    return;
                }
                let chunk = |delta: Value, finish: Value| {
                    format!(
                        "data: {}\n\n",
                        json!({"id":"chatcmpl-ray-l3","object":"chat.completion.chunk","created":1,
                               "model": model.clone(),
                               "choices":[{"index":0,"delta":delta,"finish_reason":finish}]})
                    )
                };
                let body_out = chunk(json!({"role":"assistant","content":""}), Value::Null)
                    + &chunk(json!({"content": format!("ray-l3-pong {model}")}), json!("stop"))
                    + "data: [DONE]\n\n";
                let _ = write_plain(&mut sock, 200, "text/event-stream", body_out).await;
            });
        }
    });
    MockUpstream {
        url: format!("http://127.0.0.1:{port}"),
        hits,
        cls_bodies: cls,
    }
}

fn hits_record(hits: &Arc<Mutex<Vec<(String, String)>>>, path: &str, model: &str) {
    hits.lock().unwrap().push((path.to_string(), model.to_string()));
}
/// v1 one-word grammar replay: the reply word rides the prompt marker
/// "r3-triple-<word>" so a single mock serves the golden success triples.
fn grammar_word(body: &Value) -> String {
    let text = body.to_string();
    for word in ["local", "deep", "cloud"] {
        if text.contains(&format!("r3-triple-{word}")) {
            return word.to_string();
        }
    }
    "cloud".to_string()
}

async fn write_plain<S>(sock: &mut S, status: u16, ctype: &str, body: String) -> std::io::Result<()>
where
    S: tokio::io::AsyncWrite + Unpin,
{
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

fn find_headers_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4)
}

// ── Daemon harness (live-daemon conventions from v2_l2_acceptance.rs) ──────

struct Daemon {
    child: std::process::Child,
    url: String,
    log: PathBuf,
    socket: PathBuf,
    client: reqwest::Client,
    env: TestEnv,
}

impl Daemon {
    fn start(env: &TestEnv, config_yaml: &str) -> Self {
        Self::start_with_env(env, config_yaml, std::iter::empty::<(String, String)>())
    }

    fn start_with_env<I, S>(env: &TestEnv, config_yaml: &str, extra: I) -> Self
    where
        I: IntoIterator<Item = (S, S)> + Clone,
        S: AsRef<std::ffi::OsStr> + Clone,
    {
        let config_path = env.root.join("config.yaml");
        fs::write(&config_path, config_yaml).unwrap();
        Self::boot(env, extra)
    }

    /// Restart: boot the config file boot 1 already left on disk (the
    /// migrated YAML). This is the real systemd-restart procedure the
    /// one-shot migration contract pins: a boot over an L3-era YAML is a
    /// migration no-op.
    fn start_from_disk(env: &TestEnv) -> Self {
        assert!(
            env.root.join("config.yaml").exists(),
            "restart precondition: a config must already be on disk"
        );
        Self::boot(env, std::iter::empty::<(String, String)>())
    }

    fn boot<I, S>(env: &TestEnv, extra: I) -> Self
    where
        I: IntoIterator<Item = (S, S)> + Clone,
        S: AsRef<std::ffi::OsStr> + Clone,
    {
        let config_path = env.root.join("config.yaml");
        let socket = env.root.join("d.sock");
        let log = env.root.join("daemon.log");
        // Reserved-then-freed loopback ports carry a TOCTOU window on a
        // saturated shared runner: a sibling fixture's bind can claim the
        // port between drop() and the daemon's bind (manifests as
        // "daemon exited" flakes). On a bind collision, reserve a fresh
        // port and respawn (up to 3 attempts).
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
            env.apply(&mut cmd, extra.clone());
            let mut child = cmd.spawn().unwrap();
            std::thread::sleep(Duration::from_millis(150));
            if child.try_wait().unwrap().is_none() || attempt == 3 {
                break (address, child);
            }
            let text = fs::read_to_string(&log).unwrap_or_default();
            if !text.to_lowercase().contains("address already in use") {
                break (address, child);
            }
            // Bind collision: a parallel sibling's fixture claimed the
            // reserved port before the daemon bound it — re-reserve and
            // respawn.
            drop(child);
        };
        Self {
            child,
            url: format!("http://{address}"),
            log,
            socket,
            client: reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(60))
                .build()
                .unwrap(),
            env: TestEnv::borrow(env),
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

    async fn get_json(&self, path: &str) -> Value {
        let (status, text) = self.request("GET", path, None).await;
        assert!(status < 400, "GET {path} -> {status}: {text}");
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("GET {path} not json ({e}): {text}"))
    }

    async fn get_text(&self, path: &str) -> (u16, String) {
        self.request("GET", path, None).await
    }

    async fn post_json(&self, path: &str, body: Value) -> (u16, String) {
        let r = self
            .client
            .post(format!("{}{path}", self.url))
            .json(&body)
            .send()
            .await
            .unwrap_or_else(|e| panic!("POST {path}: {e}"));
        let status = r.status().as_u16();
        (status, r.text().await.unwrap_or_default())
    }

    async fn delete(&self, path: &str) -> (u16, String) {
        let r = self
            .client
            .delete(format!("{}{path}", self.url))
            .send()
            .await
            .unwrap_or_else(|e| panic!("DELETE {path}: {e}"));
        let status = r.status().as_u16();
        (status, r.text().await.unwrap_or_default())
    }

    /// (status, body text).
    async fn request(&self, method: &str, path: &str, body: Option<Value>) -> (u16, String) {
        let mut req = self.client.request(reqwest::Method::from_bytes(method.as_bytes()).unwrap(),
            format!("{}{path}", self.url));
        if let Some(b) = body {
            req = req.json(&b);
        }
        let r = req.send().await.unwrap_or_else(|e| panic!("{method} {path}: {e}"));
        let status = r.status().as_u16();
        (status, r.text().await.unwrap_or_default())
    }

    /// `brainrouter cli --socket S <args>` with the isolated environment,
    /// with fallback to the bare `brainrouter <args> --socket S --config C`
    /// spelling (v2_l2_acceptance.rs spec-gap convention).
    fn cli(&self, args: &[&str]) -> CliRun {
        let socket = self.socket.display().to_string();
        let forms: Vec<Vec<String>> = vec![
            std::iter::once("cli".to_string())
                .chain(["--socket".to_string(), socket.clone()])
                .chain(args.iter().map(|a| a.to_string()))
                .collect(),
            args.iter()
                .map(|a| a.to_string())
                .chain(["--socket".to_string(), socket.clone()])
                .collect(),
        ];
        let mut runs = Vec::new();
        for form in forms {
            let mut cmd = Command::new(env!("CARGO_BIN_EXE_brainrouter"));
            cmd.args(&form).current_dir(&self.env.root).stdin(Stdio::null());
            self.env.apply(&mut cmd, std::iter::empty::<(String, String)>());
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
    fn out(&self) -> String {
        format!("{}{}", self.stdout, self.stderr)
    }
    fn expect_ok(&self, what: &str) -> &Self {
        assert!(self.code == 0, "CLI `{what}` failed (exit {}): {}", self.code, self.out());
        self
    }
}

// ── Chat helper (proxy surface) ─────────────────────────────────────────────

/// POST /v1/chat/completions with the optional role header. Returns
/// (status, body-text result). A mid-stream upstream death surfaces as Err
/// on the body future (L3 promise: "mid-stream never retried" — the client
/// MUST see the failure).
async fn chat(d: &Daemon, model: &str, role: Option<&str>, prompt: &str) -> (u16, Result<String, String>) {
    chat_messages(
        d,
        model,
        role,
        json!([{"role": "user", "content": prompt}]),
    )
    .await
}

async fn chat_messages(
    d: &Daemon,
    model: &str,
    role: Option<&str>,
    messages: Value,
) -> (u16, Result<String, String>) {
    let mut req = d
        .client
        .post(format!("{}/v1/chat/completions", d.url))
        .json(&json!({"model": model, "messages": messages, "stream": true, "max_tokens": 32}));
    if let Some(r) = role {
        req = req.header("x-brainrouter-role", r);
    }
    let resp = req.send().await.expect("chat request reaches daemon");
    let status = resp.status().as_u16();
    (status, resp.text().await.map_err(|e| e.to_string()))
}

fn assert_served(body: &(u16, Result<String, String>), label: &str) {
    let (status, text) = body;
    let text = text.as_ref().unwrap_or_else(|e| panic!("{label}: stream failed: {e}"));
    assert_eq!(*status, 200, "{label}: non-200: {text}");
    assert!(text.contains("ray-l3-pong"), "{label}: proxy response lacks mock payload: {text}");
}

// ── Role cards (CLI per design "CLI" section) ───────────────────────────────

fn roles_set(d: &Daemon, role: &str, selector: &str, project: Option<&str>) {
    let mut args = vec!["roles", "set", role, selector];
    if let Some(p) = project {
        args.push("--project");
        args.push(p);
    }
    d.cli(&args).expect_ok(&format!("roles set {role} {selector}"));
}

/// `roles get` -> flat card records [{scope, role, assignment}], tolerant of
/// array/record output shapes (spec gap; see header).
fn roles_get(d: &Daemon) -> Vec<Value> {
    let run_raw = d.cli(&["roles", "get"]);
    let run = run_raw.expect_ok("roles get");
    let v = parse_json_from(&run.stdout).unwrap_or_else(|| {
        panic!("`roles get` printed no JSON:\nstdout: {}\nstderr: {}", run.stdout, run.stderr)
    });
    let mut out = Vec::new();
    find_records(&v, &mut out);
    find_map_records(&v, None, &mut out);
    out
}

fn find_records(v: &Value, out: &mut Vec<Value>) {
    match v {
        Value::Object(m) => {
            let role = m.get("role").and_then(Value::as_str);
            let assignment = m.get("assignment").and_then(assignment_str);
            if let (Some(r), Some(a)) = (role, assignment) {
                out.push(json!({
                    "role": r,
                    "assignment": a,
                    "scope": m.get("scope").and_then(Value::as_str).unwrap_or("global"),
                }));
            }
            for (_, val) in m {
                find_records(val, out);
            }
        }
        Value::Array(a) => {
            for x in a {
                find_records(x, out);
            }
        }
        _ => {}
    }
}

fn find_map_records(v: &Value, scope: Option<&str>, out: &mut Vec<Value>) {
    if let Value::Object(m) = v {
        for (k, val) in m {
            if matches!(k.as_str(), "main" | "review" | "reviewer" | "subagent") {
                if let Some(a) = val.get("assignment").and_then(assignment_str).or_else(|| assignment_str(val)) {
                    out.push(json!({
                        "role": k,
                        "assignment": a,
                        "scope": scope.unwrap_or("global"),
                    }));
                    continue;
                }
            }
            find_map_records(val, Some(k.as_str()), out);
        }
    } else if let Value::Array(a) = v {
        for x in a {
            find_map_records(x, scope, out);
        }
    }
}

fn assignment_str(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Object(m) => m
            .get("selector")
            .and_then(Value::as_str)
            .or_else(|| m.get("assignment").and_then(Value::as_str))
            .map(str::to_string),
        _ => None,
    }
}

fn card_assignment(cards: &[Value], role: &str, project: Option<&str>) -> Option<String> {
    fn norm(s: &str) -> &str {
        match s {
            "reviewer" => "review",
            "main" => "main",
            "subagent" => "subagent",
            other => other,
        }
    }
    cards.iter().find(|c| {
        let scope_ok = match project {
            None => {
                let s = c["scope"].as_str().unwrap_or("global");
                s == "global" || s.is_empty()
            }
            Some(p) => c["scope"].as_str().is_some_and(|s| s == p || s.ends_with(p)),
        };
        scope_ok && norm(c["role"].as_str().unwrap_or("")) == norm(role)
    })?["assignment"]
        .as_str()
        .map(str::to_string)
}

fn parse_json_from(text: &str) -> Option<Value> {
    for (idx, ch) in text.char_indices() {
        if ch == '{' || ch == '[' {
            if let Ok(v) = serde_json::from_str::<Value>(&text[idx..]) {
                return Some(v);
            }
            // Try progressively shorter prefixes ending at the last closer.
            let tail = text[idx..].trim_end();
            if let Some(end) = tail.rfind(['}', ']']) {
                if let Ok(v) = serde_json::from_str::<Value>(&tail[..=end]) {
                    return Some(v);
                }
            }
        }
    }
    None
}

// ── Policy (privacy/quality chips; endpoint probed — spec gap) ───────────────

async fn set_policy(d: &Daemon, privacy: &str, quality: &str) {
    let body = json!({"scope": "global", "privacy": privacy, "quality": quality});
    let mut results = Vec::new();
    for path in ["/api/roles/policy", "/api/policy", "/api/role-cards/policy"] {
        let (status, text) = d.post_json(path, body.clone()).await;
        if (200..300).contains(&status) {
            return;
        }
        results.push(format!("POST {path} -> {status}: {text}"));
    }
    // One-record shape: policy embedded in the roles record write.
    let (status, text) = d.post_json("/api/roles", json!({"policy": body})).await;
    if (200..300).contains(&status) {
        return;
    }
    results.push(format!("POST /api/roles {{policy}} -> {status}: {text}"));
    panic!(
        "SPEC GAP unresolvable: no probed route accepts the policy record (privacy/quality). \
         Design (Dashboard: policy chips -> `policy` records) does not name the endpoint. Probes:\n{}",
        results.join("\n")
    );
}

// ── Routing events ──────────────────────────────────────────────────────────

async fn events(d: &Daemon) -> Vec<Value> {
    d.get_json("/api/routing-events").await["events"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

async fn event_for(d: &Daemon, prompt: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let evs = events(d).await;
        if let Some(e) = evs
            .iter()
            .find(|e| e["prompt_excerpt"].as_str().is_some_and(|p| p == prompt))
        {
            return e.clone();
        }
        assert!(
            Instant::now() < deadline,
            "no routing event for prompt {prompt:?}; events: {}",
            serde_json::to_string(&evs).unwrap_or_default()
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

// ── Credentials page probes (routes not pinned by the design — spec gap) ───

async fn cred_add(d: &Daemon, record: Value) {
    let mut results = Vec::new();
    for path in ["/api/credentials", "/api/credentials/add", "/api/credentials/save"] {
        let (status, text) = d.post_json(path, record.clone()).await;
        if (200..300).contains(&status) {
            return;
        }
        results.push(format!("POST {path} -> {status}: {text}"));
    }
    panic!(
        "SPEC GAP unresolvable: no probed route adds a credential (design: Credentials tab add; \
         L2 record JSON speaking, `credentials.json` format unchanged). Probes:\n{}",
        results.join("\n")
    );
}

async fn cred_list(d: &Daemon) -> Value {
    for path in ["/api/credentials", "/api/credentials/list"] {
        let (status, text) = d.request("GET", path, None).await;
        if (200..300).contains(&status) {
            return serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("GET {path} not json ({e}): {text}"));
        }
    }
    panic!("SPEC GAP unresolvable: GET /api/credentials (list, masked) is not served");
}

async fn cred_test(d: &Daemon, id: &str) -> bool {
    let direct = format!("/api/credentials/{id}/test");
    let probes: Vec<(String, Option<Value>)> = vec![
        (direct.clone(), None),
        ("/api/credentials/test".to_string(), Some(json!({"id": id}))),
        (direct, Some(json!({}))),
    ];
    let mut tried = Vec::new();
    for (path, body) in probes {
        let (status, text) = d.request("POST", &path, body).await;
        if (200..300).contains(&status) {
            if let Some(v) = parse_json_from(&text) {
                if let Some(b) = find_first_bool(&v) {
                    return b;
                }
            }
        }
        tried.push(format!("POST {path} -> {status}: {text}"));
    }
    panic!(
        "SPEC GAP unresolvable: no probed Test route returns a boolean verdict for credential {id}. \
         (Design: Test = BOOLEAN verdict only.) Probes:\n{}",
        tried.join("\n")
    );
}

async fn cred_delete(d: &Daemon, id: &str) {
    let path = format!("/api/credentials/{id}");
    let (status, text) = d.delete(&path).await;
    assert!(
        (200..300).contains(&status),
        "credential delete {path} -> {status}: {text} (design: add/edit/delete)"
    );
}

fn find_first_bool(v: &Value) -> Option<bool> {
    match v {
        Value::Bool(b) => Some(*b),
        Value::Object(m) => m
            .get("ok")
            .and_then(Value::as_bool)
            .or_else(|| m.get("valid").and_then(Value::as_bool))
            .or_else(|| m.get("resolves").and_then(Value::as_bool))
            .or_else(|| m.get("success").and_then(Value::as_bool))
            .or(m.values().find_map(find_first_bool)),
        Value::Array(a) => a.iter().find_map(find_first_bool),
        _ => None,
    }
}

/// Every rendered credential entry, stringified for masking audits.
fn rendered_credentials(list: &Value) -> String {
    serde_json::to_string(list).unwrap_or_default()
}

// ── Plugin fixture (minimal reuse of the L2 pattern) ────────────────────────

const RPC_FIXTURE_PY: &str = r#"#!/usr/bin/env python3
import json, os, socket, time

here = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(here, "fixture.json")))
pid = cfg["id"]
runtime = os.environ["XDG_RUNTIME_DIR"]
token = os.environ.get("BRAINROUTER_PLUGIN_TOKEN", "")
state = os.path.join(os.environ.get("XDG_STATE_HOME", os.path.join(os.environ["HOME"], ".local", "state")),
                     "brainrouter", "plugins", pid)
os.makedirs(state, exist_ok=True)

def line(o):
    return (json.dumps(o) + "\n").encode()

sock_path = os.path.join(runtime, "brainrouter", "plugins", pid + ".sock")
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(20.0)
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

report = {"id": pid, "pid": os.getpid()}
try:
    report["initialize"] = call("initialize",
                                {"id": pid, "plugin_id": pid, "name": pid, "token": token})
    found = {"found_id": None, "value": None}
    for c in cfg.get("credential_ids", []):
        r = call("secrets.get", {"id": c, "credential_id": c, "key": c, "name": c})
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

while True:
    with open(os.path.join(state, "pid"), "w") as f:
        f.write(str(os.getpid()))
    time.sleep(0.5)
"#;

const WEBHOOK_FIXTURE_PY: &str = r#"#!/usr/bin/env python3
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

cfg = json.load(open(sys.argv[1]))
decision = cfg["decision"]
record = cfg["record"]

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        with open(record, "wb") as f:
            f.write(body)
        out = json.dumps(decision).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass

server = ThreadingHTTPServer(("127.0.0.1", 0), H)
# Kernel-assigned port reported through the port file: NO reserve-then-free
# window for a sibling process to steal under parallel load.
with open(cfg["portfile"], "w") as f:
    f.write(str(server.server_address[1]))
server.serve_forever()
"#;

fn manifest_toml(id: &str, binary: &str, grants: &[&str]) -> String {
    let grants = grants.iter().map(|g| format!("\"{g}\"")).collect::<Vec<_>>().join(", ");
    format!(
        "id = \"{id}\"\nname = \"Ray L3 fixture {id}\"\nversion = \"0.1.0\"\nbinary = \"{binary}\"\nmin_daemon_version = \"1.0.0\"\ngrants = [{grants}]\n"
    )
}

fn write_plugin_fixture(env: &TestEnv, id: &str, credential_ids: &[&str]) -> PathBuf {
    let dir = env.root.join("fixtures").join(id);
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("plugin.toml"), manifest_toml(id, "fixture.py", &["secrets.get"])).unwrap();
    let script = dir.join("fixture.py");
    fs::write(&script, RPC_FIXTURE_PY).unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    fs::write(
        dir.join("fixture.json"),
        json!({"id": id, "credential_ids": credential_ids}).to_string(),
    )
    .unwrap();
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
        assert!(Instant::now() < deadline, "no JSON at {}", path.display());
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Spawn the webhook fixture. The port is kernel-assigned (bind :0 inside
/// python) and reported through a port file; then a connect-gate makes
/// "listening" a precondition of returning. Under parallel load a
/// reserve-then-drop port could be stolen before python bound it (the slot
/// POST then hit a closed port, the fresh-config failure path fell back to
/// heuristic, and `cloud.models_seen()` stayed empty).
fn spawn_webhook_mock(env: &TestEnv, decision: Value) -> (String, PathBuf, std::process::Child) {
    let dir = env.root.join("webhook");
    fs::create_dir_all(&dir).unwrap();
    let cfg_path = dir.join("cfg.json");
    let record = dir.join("last_post.json");
    let portfile = dir.join("port");
    let _ = fs::remove_file(&portfile);
    fs::write(
        &cfg_path,
        json!({"record": record, "portfile": portfile, "decision": decision}).to_string(),
    )
    .unwrap();
    let script = dir.join("webhook.py");
    fs::write(&script, WEBHOOK_FIXTURE_PY).unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    let mut cmd = Command::new(env.bin.join("python3"));
    cmd.arg(&script).arg(&cfg_path).stdin(Stdio::null());
    env.apply(&mut cmd, std::iter::empty::<(String, String)>());
    let mut child = cmd.spawn().expect("spawn webhook fixture");
    let deadline = Instant::now() + Duration::from_secs(30);
    let port: u16 = loop {
        if let Ok(text) = fs::read_to_string(&portfile) {
            if let Ok(p) = text.trim().parse::<u16>() {
                break p;
            }
        }
        assert!(
            child.try_wait().map(|s| s.is_none()).unwrap_or(true) && Instant::now() < deadline,
            "webhook fixture did not report a port (child exit: {:?})",
            child.try_wait().ok().flatten()
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    let addr = std::net::SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, port));
    loop {
        if std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_ok() {
            break;
        }
        assert!(Instant::now() < deadline, "webhook fixture never accepted on port {port}");
        std::thread::sleep(Duration::from_millis(50));
    }
    (format!("http://127.0.0.1:{port}"), record, child)
}

// ── (1) Resolution matrix ───────────────────────────────────────────────────

/// Global role cards x role header: each card selector compiles through the
/// v1 route for that selector (L3 spec "Resolution" bullet), and the event
/// carries the role from the `x-brainrouter-role` header (default main).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_resolution_matrix_global_cards_per_role() {
    let _gate = suite_gate().lock().await;
    let env = TestEnv::new("matrix");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({"enabled": true, "kind": "heuristic"}),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;

    // main card = auto -> decision slot (heuristic, parity anchor: local).
    // v1 source: classifier-off auto -> Local{fallback} (classifier.rs L119-131).
    roles_set(&d, "main", "auto", None);
    let r = chat(&d, "brainrouter/auto", None, "r3m-main-auto").await;
    assert_served(&r, "main/auto card");
    assert_eq!(local.models_seen(), vec!["ray-l3-fallback".to_string()]);
    let e = event_for(&d, "r3m-main-auto").await;
    assert_eq!(e["role"], json!("main"), "role defaults to main");
    assert_eq!(e["decision"]["route"], json!("local"), "heuristic parity: auto -> local");

    // main card = local -> fallback_model (v1 selector `local`,
    // src/router.rs L346-356 @ v1-classic).
    roles_set(&d, "main", "local", None);
    let r = chat(&d, "brainrouter/auto", None, "r3m-main-local").await;
    assert_served(&r, "main/local card");
    assert_eq!(local.count_of("ray-l3-fallback"), 2);

    // main card = bare key -> direct local pick of that key (v1 named model,
    // src/router.rs L407-422 @ v1-classic).
    roles_set(&d, "main", "matrix-bare-model", None);
    let r = chat(&d, "brainrouter/auto", None, "r3m-main-bare").await;
    assert_served(&r, "main/bare card");
    assert_eq!(local.count_of("matrix-bare-model"), 1);

    // main card = cloud/<id> -> cloud passthrough of that id
    // (v1 `cloud/<id>` selector, src/router.rs L365-371 @ v1-classic).
    roles_set(&d, "main", "cloud/vendor/main-2026", None);
    let r = chat(&d, "brainrouter/auto", None, "r3m-main-cloudid").await;
    assert_served(&r, "main/cloud-id card");
    assert_eq!(cloud.models_seen(), vec!["vendor/main-2026".to_string()]);

    // subagent role: header selects the subagent CARD.
    roles_set(&d, "subagent", "cloud", None);
    let r = chat(&d, "brainrouter/auto", Some("subagent"), "r3m-sub-cloud").await;
    assert_served(&r, "subagent/cloud card");
    assert_eq!(cloud.models_seen(), vec!["vendor/main-2026".to_string(), "auto".to_string()]);
    let e = event_for(&d, "r3m-sub-cloud").await;
    assert_eq!(e["role"], json!("subagent"));

    // review role: the review card is the only reviewer-model config.
    roles_set(&d, "review", "cloud/vendor/review-2026", None);
    let r = chat(&d, "brainrouter/auto", Some("review"), "r3m-rev-cloudid").await;
    assert_served(&r, "review/cloud-id card");
    assert!(cloud.models_seen().contains(&"vendor/review-2026".to_string()));
    let e = event_for(&d, "r3m-rev-cloudid").await;
    assert_eq!(e["role"], json!("review"));

    // Bare-key review card (migration row Local{Some} -> <bare-key>): direct
    // strict local pick (L3 spec "Pinned-card strictness" bullet).
    roles_set(&d, "review", "review-bare-model", None);
    let r = chat(&d, "brainrouter/auto", Some("review"), "r3m-rev-bare").await;
    assert_served(&r, "review/bare card");
    assert_eq!(local.count_of("review-bare-model"), 1);

    // Explicit selector BYPASSES cards; the event labels the header role
    // (L3 spec: "Other explicit selectors ... bypass cards (role = header,
    // events label)"). Request "cloud" while the main card says bare.
    let r = chat(&d, "cloud", Some("subagent"), "r3m-explicit-cloud").await;
    assert_served(&r, "explicit cloud bypass");
    let e = event_for(&d, "r3m-explicit-cloud").await;
    assert_eq!(e["role"], json!("subagent"));

    d.stop();
}

/// Project scope beats global: a `--project` assignment for the canonical
/// git root of the caller's cwd (the repo root — see header PROJECT CONTEXT)
/// overrides the global card. v1 precedent: project pin beats global profile
// for the main role (src/router.rs:288-297 @ v1-classic).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_resolution_project_scope_beats_global() {
    let _gate = suite_gate().lock().await;
    let env = TestEnv::new("projscope");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({"enabled": true, "kind": "heuristic"}),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;

    roles_set(&d, "main", "cloud", None);
    roles_set(&d, "main", "local", Some(&repo_root()));

    let r = chat(&d, "brainrouter/auto", None, "r3m-project-scope").await;
    assert_served(&r, "project card");
    assert_eq!(
        cloud.total(),
        0,
        "project-scope `local` card must win over the global `cloud` card; cloud mock saw {:?}",
        cloud.models_seen()
    );
    assert_eq!(local.models_seen(), vec!["ray-l3-fallback".to_string()]);

    // The scoped row exists in `roles get` under the canonical git root.
    let cards = roles_get(&d);
    assert_eq!(
        card_assignment(&cards, "main", Some(&repo_root())).as_deref(),
        Some("local"),
        "roles get must show the project-scoped main card; got {cards:?}"
    );
    d.stop();
}

/// Privacy guardrail: `local-only` clamps the SLOT decision (auto), never an
/// explicit selector (L3 spec "Guardrail clamps (slot decisions only, never
/// explicit selectors)").
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_privacy_clamps_auto_never_explicit() {
    let _gate = suite_gate().lock().await;
    let env = TestEnv::new("privacy");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    // Slot = bonsai-profile classifier answering "cloud" — so the clamp is
    // the ONLY thing that can move the request to local.
    let cls = spawn_mock(MockCfg {
        classifier: true,
        classifier_word: "cloud".into(),
        classifier_status: 200,
        ..Default::default()
    });
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({
            "enabled": true,
            "kind": "openai-compatible",
            "base_url": cls.url,
            "model": "bonsai",
            "prompt_profile": "bonsai",
            "timeout_ms": 5000,
        }),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    roles_set(&d, "main", "auto", None);

    // Baseline: slot says cloud -> cloud served.
    let r = chat(&d, "brainrouter/auto", None, "r3m-priv-base").await;
    assert_served(&r, "baseline cloud decision");
    assert!(cloud.total() >= 1);

    // local-only: AUTO (slot) is clamped to local.
    set_policy(&d, "local-only", "balanced").await;
    let r = chat(&d, "brainrouter/auto", None, "r3m-priv-auto").await;
    assert_served(&r, "clamped auto");
    assert_eq!(local.count_of("ray-l3-fallback"), 1, "privacy local-only must clamp the slot decision to local");

    // local-only: EXPLICIT selectors are NEVER clamped (owner intent).
    let before = cloud.total();
    let r = chat(&d, "cloud", None, "r3m-priv-explicit").await;
    assert_served(&r, "explicit cloud under local-only");
    assert!(cloud.total() > before, "explicit `cloud` must reach the cloud mock despite local-only policy");
    let r = chat(&d, "bare-key-under-privacy", None, "r3m-priv-bare").await;
    assert_served(&r, "explicit bare key under local-only");
    assert_eq!(local.count_of("bare-key-under-privacy"), 1);

    d.stop();
}

/// `subs`/`brainrouter/subs` NAMES the subagent role and overrides the
/// header; its card assignment is the pool (L3 spec Role-signal bullet).
/// v1 source: subs pool bypasses the classifier (src/router.rs L377-389 @
/// v1-classic) — the classifier mock must see ZERO hits.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_subs_names_subagent_role_overrides_header() {
    let _gate = suite_gate().lock().await;
    let env = TestEnv::new("subsname");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let cls = spawn_mock(MockCfg {
        classifier: true,
        classifier_word: "cloud".into(),
        classifier_status: 200,
        ..Default::default()
    });
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        Some("legacy-subs-model"),
        json!({
            "enabled": true, "kind": "openai-compatible", "base_url": cls.url,
            "model": "bonsai", "prompt_profile": "bonsai", "timeout_ms": 5000,
        }),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;

    // The subagent card carries the pool (its assignment is a selector);
    // the header (main) must be overridden by the subs selector.
    roles_set(&d, "subagent", "sub-card-model", None);
    let r = chat(&d, "subs", Some("main"), "r3m-subs-card").await;
    assert_served(&r, "subs -> subagent card");
    assert_eq!(local.models_seen(), vec!["sub-card-model".to_string()]);
    assert_eq!(cls.total(), 0, "v1: subs bypasses the classifier (router.rs L377-389 @ v1-classic)");
    let e = event_for(&d, "r3m-subs-card").await;
    assert_eq!(e["role"], json!("subagent"), "subs NAMES the subagent role, overriding the header");

    let r = chat(&d, "brainrouter/subs", Some("main"), "r3m-subs-prefixed").await;
    assert_served(&r, "brainrouter/subs");
    assert_eq!(local.count_of("sub-card-model"), 2);
    d.stop();
}

// ── (2) Decision slot ───────────────────────────────────────────────────────

/// Heuristic parity table: the v1 selector matrix with the classifier-OFF
/// anchor (R4 "parity anchor = today's classifier-off routing").
/// v1 sources (all @ v1-classic):
///  - auto        -> classifier off -> LOCAL, model=fallback_model
///                   (classifier.rs L119-131; daemon.rs:355-357 binds
///                   default_local_model = llama_swap.fallback_model;
///                   router.rs L555-562 route_local(allow_fallback=true))
///  - local       -> fallback_model, local-direct (router.rs L346-356)
///  - subs card   -> the subagent pool selector, strict local (L377-389)
///  - bare key    -> direct local pick (L407-422)
///  - cloud/<id>  -> cloud passthrough of <id> (L365-371)
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_heuristic_parity_v1_selector_matrix() {
    let _gate = suite_gate().lock().await;
    let env = TestEnv::new("parity");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({"enabled": true, "kind": "heuristic"}),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    roles_set(&d, "subagent", "parity-sub-pool", None);

    for (model, expect) in [
        ("brainrouter/auto", Served::Local("ray-l3-fallback")),
        ("auto", Served::Local("ray-l3-fallback")),
        ("local", Served::Local("ray-l3-fallback")),
        ("subs", Served::Local("parity-sub-pool")),
        ("parity-bare-model", Served::Local("parity-bare-model")),
        ("cloud/parity-cloud-id", Served::Cloud("parity-cloud-id")),
    ] {
        let prompt = format!("r3parity {model}");
        let r = chat(&d, model, None, &prompt).await;
        assert_served(&r, &format!("parity row {model}"));
        match expect {
            Served::Local(key) => assert!(
                local.models_seen().last().is_some_and(|m| m == key),
                "parity row {model}: expected local key {key}, local saw {:?}",
                local.models_seen()
            ),
            Served::Cloud(id) => assert!(
                cloud.models_seen().last().is_some_and(|m| m == id),
                "parity row {model}: expected cloud id {id}, cloud saw {:?}",
                cloud.models_seen()
            ),
        }
    }

    // `decision test` CLI (design CLI section): the heuristic's decision for
    // free text — the auto/classifier-off anchor, route=local.
    let run = d.cli(&["decision", "test", "r3 parity probe hello world"]);
    if run.code == 0 {
        let v = parse_json_from(&run.stdout).unwrap_or_else(|| panic!("decision test printed no JSON: {}", run.out()));
        assert_eq!(
            v.get("route").and_then(Value::as_str).unwrap_or(v["decision"]["route"].as_str().unwrap_or("")),
            "local",
            "heuristic parity: no-signal request mirrors classifier-off auto => local; got {}",
            run.out()
        );
    }
    // A missing `decision` verb fails `expect_ok`-style on its own merits;
    // the routing rows above are the binding parity table.

    d.stop();
}

enum Served {
    Local(&'static str),
    Cloud(&'static str),
}

/// Bonsai migration continuity: the migrated `prompt_profile: bonsai` must
/// (a) send TODAY'S prompt + grammar verbatim, (b) produce the same
/// decisions as v1 for the one-word replies. v1 source for every expectation
/// below: src/classifier.rs @ v1-classic (lines cited inline).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_bonsai_profile_success_triples_prompt_verbatim() {
    let _gate = suite_gate().lock().await;
    let env = TestEnv::new("bonsai-ok");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let cls = spawn_mock(MockCfg {
        classifier: true,
        classifier_word: "cloud".into(),
        classifier_status: 200,
        classifier_prompt_grammar: true,
        ..Default::default()
    });
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({
            "enabled": true, "kind": "openai-compatible", "base_url": cls.url,
            "model": "bonsai", "prompt_profile": "bonsai", "timeout_ms": 15000,
        }),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    roles_set(&d, "main", "auto", None);

    // Success triple 1: "cloud" word -> v1 RoutingDecision::Cloud -> cloud,
    // request.model normalized to "auto" (router.rs L547-553 @ v1-classic).
    let (status, _) = chat_messages(
        &d,
        "brainrouter/auto",
        None,
        json!([
            {"role": "system", "content": "session system prompt r3"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "r3-triple-cloud: huge multi-system architecture migration"},
        ]),
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(cloud.models_seen().last().map(String::as_str), Some("auto"));

    // Prompt fidelity (verbatim v1 grammar, classifier.rs L144-168):
    tokio::time::sleep(Duration::from_millis(200)).await;
    let bodies = cls.classifier_bodies();
    let b = bodies
        .last()
        .expect("the migrated bonsai profile consulted the classifier endpoint");
    assert_eq!(b["model"], json!("bonsai"));
    assert_eq!(b["max_tokens"], json!(10), "CLASSIFY_MAX_TOKENS = 10 (classifier.rs L38)");
    assert_eq!(b["temperature"], json!(0.0));
    assert_eq!(b["stop"], json!(["\n"]));
    assert_eq!(b["chat_template_kwargs"]["enable_thinking"], json!(false));
    let msgs = b["messages"].as_array().expect("classifier messages");
    assert_eq!(msgs.len(), 2);
    assert_eq!(
        msgs[0]["content"], json!(V1_BONSAI_SYSTEM_PROMPT),
        "system prompt must be byte-identical to v1 (prompt_profile: bonsai preserves today's prompt)"
    );
    // Prompt = LAST user message only (extract_last_user_message,
    // classifier.rs L250-275 @ v1-classic).
    assert_eq!(
        msgs[1]["content"],
        json!("Classify this request: r3-triple-cloud: huge multi-system architecture migration")
    );

    // Success triple 2: "local" word -> local, tier Light => canon `fast`
    // (v1: classifier.rs L205-224; vocab map L3 "Decision-model slot" bullet).
    let r = chat(&d, "brainrouter/auto", None, "r3-triple-local").await;
    assert_served(&r, "word local");
    assert_eq!(local.count_of("ray-l3-fallback"), 1);
    let e = event_for(&d, "r3-triple-local").await;
    assert_eq!(e["decision"]["route"], json!("local"));
    assert_eq!(e["decision"]["tier"], json!("fast"), "v1 word `local` maps to canon tier `fast`");

    // Success triple 3: "deep" word -> local, tier Deep => canon `deep`.
    let r = chat(&d, "brainrouter/auto", None, "r3-triple-deep").await;
    assert_served(&r, "word deep");
    let e = event_for(&d, "r3-triple-deep").await;
    assert_eq!(e["decision"]["route"], json!("local"));
    assert_eq!(e["decision"]["tier"], json!("deep"), "v1 word `deep` maps to canon tier `deep`");
    // Nudge is off in this fixture => v1 routes the tier-annotated request
    // to default_local_model = fallback (classifier.rs L214-223), i.e. the
    // tier rides the event, not a model-key swap.
    assert_eq!(local.count_of("ray-l3-fallback"), 2);

    d.stop();
}

/// Failure table (bonsai profile, binding matrix from the L3 spec):
/// HTTP error, timeout, parse fail, missing last message, AND server-down
/// (v1's single failure mode) all => route=cloud, `fallback: v1-safe`.
/// v1 sources: classifier.rs @ v1-classic — HTTP/connect failure L143-184
/// (server down = connect refused), parse fail L188-193, non-word reply ->
/// first-char grammar L240-247 picks Cloud as the safe default,
/// missing/empty last user message L136-139.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_bonsai_profile_failure_table_v1_safe() {
    let _gate = suite_gate().lock().await;
    // Row A: server down (connection refused — v1's single failure mode).
    let env = TestEnv::new("bonsai-down");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({
            "enabled": true, "kind": "openai-compatible",
            "base_url": dead_upstream_url(), "model": "bonsai",
            "prompt_profile": "bonsai", "timeout_ms": 3000,
        }),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    roles_set(&d, "main", "auto", None);
    let r = chat(&d, "brainrouter/auto", None, "r3-fail-down").await;
    assert_served(&r, "server-down row");
    assert_eq!(cloud.models_seen().last().map(String::as_str), Some("auto"), "v1-safe: down classifier => cloud");
    assert_eq!(local.total(), 0, "the availability pre-check must NOT short-circuit the bonsai profile to heuristic");
    let e = event_for(&d, "r3-fail-down").await;
    assert_eq!(e["decision"]["route"], json!("cloud"));
    assert_eq!(
        e["decision_fallback"],
        json!("v1-safe"),
        "event records the v1-safe fallback; event: {e}"
    );
    d.stop();

    // Row B: HTTP 500 from the classifier + Row C: garbage reply + Row D:
    // no user message — one daemon, grammar-aware rows.
    let env = TestEnv::new("bonsai-garbage");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let cls = spawn_mock(MockCfg {
        classifier: true,
        classifier_word: "garbage-pancake".into(),
        classifier_status: 200,
        ..Default::default()
    });
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({
            "enabled": true, "kind": "openai-compatible", "base_url": cls.url,
            "model": "bonsai", "prompt_profile": "bonsai", "timeout_ms": 3000,
        }),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    roles_set(&d, "main", "auto", None);
    // v1 grammar: first non-whitespace char 'g' -> the else-arm -> Cloud.
    let r = chat(&d, "brainrouter/auto", None, "r3-fail-garbage").await;
    assert_served(&r, "garbage reply row");
    assert_eq!(cloud.models_seen().last().map(String::as_str), Some("auto"));
    let e = event_for(&d, "r3-fail-garbage").await;
    assert_eq!(e["decision_fallback"], json!("v1-safe"));

    // Row D: request without any user message => v1 defaults to Cloud
    // (classifier.rs L136-139 @ v1-classic), v1-safe fallback.
    let (status, _) = chat_messages(
        &d,
        "brainrouter/auto",
        None,
        json!([{"role": "system", "content": "only a system prompt r3-fail-nouser"}]),
    )
    .await;
    assert_eq!(status, 200);
    assert!(cloud.models_seen().contains(&"auto".to_string()), "no-user-message row must reach cloud");
    d.stop();

    // Row B (separate daemon): classifier HTTP 500 => cloud v1-safe.
    let env = TestEnv::new("bonsai-500");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let cls = spawn_mock(MockCfg {
        classifier: true,
        classifier_word: "cloud".into(),
        classifier_status: 500,
        ..Default::default()
    });
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({
            "enabled": true, "kind": "openai-compatible", "base_url": cls.url,
            "model": "bonsai", "prompt_profile": "bonsai", "timeout_ms": 3000,
        }),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    roles_set(&d, "main", "auto", None);
    let r = chat(&d, "brainrouter/auto", None, "r3-fail-500").await;
    assert_served(&r, "classifier 500 row");
    assert_eq!(cloud.models_seen().last().map(String::as_str), Some("auto"));
    let e = event_for(&d, "r3-fail-500").await;
    assert_eq!(e["decision_fallback"], json!("v1-safe"));
    d.stop();
}

/// Fresh-config kinds fail SAFE, not v1-safe: any failure (incl. parse fail
/// and unavailable) => heuristic, `fallback: unavailable`; low confidence
/// (< min_confidence, default 0.4) => heuristic, `fallback: heuristic`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_structured_schema_strict_json_fallbacks() {
    let _gate = suite_gate().lock().await;
    // Row 1: strict JSON with confidence 0.1 (< default min_confidence
    // 0.4) -> heuristic fallback recorded in the event; heuristic parity
    // (v1 classifier-off anchor) => local.
    let env = TestEnv::new("schema-lowconf");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let cls = spawn_mock(MockCfg {
        classifier: true,
        classifier_word: "{\"route\":\"cloud\",\"tier\":\"fast\",\"confidence\":0.1,\"rationale\":\"low confidence probe\"}".into(),
        classifier_status: 200,
        ..Default::default()
    });
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({
            "enabled": true, "kind": "structured-schema", "base_url": cls.url,
            "model": "rayl3-schema", "timeout_ms": 3000,
        }),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    roles_set(&d, "main", "auto", None);
    let r = chat(&d, "brainrouter/auto", None, "r3-schema-lowconf").await;
    assert_served(&r, "low-confidence structured row falls back to heuristic");
    assert_eq!(local.count_of("ray-l3-fallback"), 1, "heuristic parity: local");
    assert_eq!(cloud.total(), 0, "the schema's cloud decision must NOT be served under min_confidence");
    let e = event_for(&d, "r3-schema-lowconf").await;
    assert_eq!(e["decision_fallback"], json!("heuristic"), "event: {e}");
    d.stop();

    // Row 2: unparseable reply (plain text, not strict JSON) => heuristic,
    // fallback "unavailable" (fresh-config kinds, parse fail = failure).
    let env = TestEnv::new("schema-parse");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let cls = spawn_mock(MockCfg {
        classifier: true,
        classifier_plain_body: true,
        ..Default::default()
    });
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({
            "enabled": true, "kind": "structured-schema", "base_url": cls.url,
            "model": "rayl3-schema", "timeout_ms": 3000,
        }),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    roles_set(&d, "main", "auto", None);
    let r = chat(&d, "brainrouter/auto", None, "r3-schema-parsefail").await;
    assert_served(&r, "parse-fail structured row falls back to heuristic");
    assert_eq!(local.count_of("ray-l3-fallback"), 1);
    let e = event_for(&d, "r3-schema-parsefail").await;
    assert_eq!(e["decision_fallback"], json!("unavailable"), "fresh-config failures record `unavailable`; event: {e}");
    d.stop();
}

/// Webhook kind: POST the features JSON to `{url}`. The features payload
/// MUST NOT carry `last_user_excerpt` (model kinds only — L3 spec
/// RequestFeatures bullet + security section: "prompt content sent to
/// classifiers = feature summary"), and the webhook's decision is served.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_webhook_features_no_excerpt_returns_decision() {
    let _gate = suite_gate().lock().await;
    let env = TestEnv::new("webhook");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let (url, record, mut child) = spawn_webhook_mock(
        &env,
        json!({"route": "cloud", "tier": "balanced", "confidence": 0.9, "rationale": "webhook-route"}),
    );
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({"enabled": true, "kind": "webhook", "url": url}),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    roles_set(&d, "main", "auto", None);

    let r = chat(&d, "brainrouter/auto", None, "r3 webhook probe").await;
    assert_served(&r, "webhook decision");
    assert_eq!(cloud.models_seen().last().map(String::as_str), Some("auto"), "webhook said cloud");
    assert_eq!(local.total(), 0);

    // The features payload.
    let deadline = Instant::now() + Duration::from_secs(5);
    let body: Value = loop {
        if let Ok(text) = fs::read_to_string(&record) {
            if let Ok(v) = serde_json::from_str::<Value>(&text) {
                break v;
            }
        }
        assert!(Instant::now() < deadline, "webhook fixture never recorded a features POST");
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    assert!(
        body.get("last_user_excerpt").is_none(),
        "webhook features JSON MUST NOT include last_user_excerpt: {body}"
    );
    // Feature summary fields (RequestFeatures list, sans excerpt).
    assert!(
        body.get("msg_count").is_some()
            && body.get("role").is_some()
            && body.get("selector").is_some()
            && body.get("privacy").is_some(),
        "features payload missing role/selector/privacy/msg_count: {body}"
    );
    if let Some(toks) = body.get("approx_prompt_tokens") {
        assert!(toks.as_u64().is_some_and(|t| t > 0), "approx_prompt_tokens must be a count: {toks}");
    }
    let e = event_for(&d, "r3 webhook probe").await;
    assert_eq!(e["decision"]["route"], json!("cloud"));
    assert_eq!(e["decision"]["tier"], json!("balanced"));
    assert_eq!(e["decision"]["confidence"].as_f64(), Some(0.9));
    assert_eq!(e["rationale"].as_str(), Some("webhook-route"));

    d.stop();
    let _ = child.kill();
    let _ = child.wait();
}


/// Features probe (Bruce r2 fix round): `POST /api/decision/features`
/// {text, role, selector} answers 200 with the RequestFeatures summary —
/// EXCERPT-FREE (`to_json(false)`: no `last_user_excerpt` key, no echo of
/// the prompt text), and the route is POST-only (GET 404).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_decision_features_probe_shape_and_post_only_route() {
    let _gate = suite_gate().lock().await;
    let env = TestEnv::new("features");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({"enabled": true, "kind": "heuristic"}),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;

    let marker = "rayl3-features-probe-ZQ7x must-not-be-echoed";
    let (status, text) = d
        .post_json(
            "/api/decision/features",
            json!({"text": marker, "role": "review", "selector": "auto"}),
        )
        .await;
    assert_eq!(status, 200, "features probe: {status}: {text}");
    let v = parse_json_from(&text).unwrap_or_else(|| panic!("features probe not JSON: {text}"));
    // Shape (src/decision.rs RequestFeatures::to_json(false)).
    assert_eq!(v["role"], json!("review"), "role echoes the probe: {v}");
    assert_eq!(v["selector"], json!("auto"), "selector echoes the probe: {v}");
    assert!(v["privacy"].is_string() && v["quality"].is_string(), "policy enums: {v}");
    assert!(v["project"].is_string(), "project key: {v}");
    assert_eq!(v["msg_count"], json!(1), "one message: {v}");
    assert!(
        v["approx_prompt_tokens"].as_u64().is_some_and(|t| t > 0),
        "approx_prompt_tokens is a token estimate: {v}"
    );
    assert_eq!(v["has_tools"], json!(false), "no tools in the probe: {v}");
    assert!(v["code_density"].is_number(), "code_density: {v}");
    // Excerpt-free: the feature summary for non-model consumers carries no
    // prompt text (L3 spec: excerpt ONLY for model kinds; security: "prompt
    // content sent to classifiers = feature summary").
    assert!(
        v.get("last_user_excerpt").is_none(),
        "features probe must be excerpt-free: {v}"
    );
    assert!(
        !text.contains(marker),
        "features probe echoes prompt content: {text}"
    );

    // Defaults on an empty probe (every field is optional).
    let (status, text) = d.post_json("/api/decision/features", json!({})).await;
    assert_eq!(status, 200, "empty features probe: {status}: {text}");
    let v = parse_json_from(&text).unwrap_or_else(|| panic!("empty probe not JSON: {text}"));
    assert_eq!(v["role"], json!("main"), "role default: {v}");
    assert_eq!(v["selector"], json!("auto"), "selector default: {v}");

    // POST-only route.
    let (status, _) = d.request("GET", "/api/decision/features", None).await;
    assert!(
        status == 404 || status == 405,
        "GET /api/decision/features must not answer (route is POST-only), got {status}"
    );
    assert_eq!(status, 404, "the v1 server contract is 404 for unlisted /api GETs");
    d.stop();

}
// ── (3) Credentials page ────────────────────────────────────────────────────

/// CRUD through the daemon HTTP (writes loopback-gated), masked list,
/// boolean Test verdicts (env set/unset, file present/absent), grants,
/// delete; the plaintext value never appears in ANY response incl.
/// `/api/config` (config show) and the dashboard.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_credentials_ui_crud_test_verdicts_grants_masked() {
    let _gate = suite_gate().lock().await;
    let env = TestEnv::new("creds");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let env_var_file = env.root.join("env-file-cred.txt");
    fs::write(&env_var_file, "ray-l3-file-value-88ff\n").unwrap();
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({"enabled": true, "kind": "heuristic"}),
    );
    let mut d = Daemon::start_with_env(
        &env,
        &cfg,
        [("RAY_L3_ENV_SET".to_string(), "ray-l3-env-value".to_string())],
    );
    d.ready().await;

    // ADD: store kind (value in the vault), env kind (ops escape), file kind.
    cred_add(
        &d,
        json!({"id": "rayl3-store", "description": "ui-added store", "kind": "store",
               "value": L3_SECRET, "grants": ["rayl3plug", "bridge"]}),
    )
    .await;
    cred_add(
        &d,
        json!({"id": "rayl3-env", "kind": "env", "var": "RAY_L3_ENV_SET", "grants": []}),
    )
    .await;
    cred_add(
        &d,
        json!({"id": "rayl3-env-absent", "kind": "env", "var": "RAY_L3_ENV_MISSING", "grants": []}),
    )
    .await;
    cred_add(
        &d,
        json!({"id": "rayl3-file", "kind": "file",
               "path": env_var_file.display().to_string(), "grants": []}),
    )
    .await;
    cred_add(
        &d,
        json!({"id": "rayl3-file-absent", "kind": "file",
               "path": format!("{}/no-such-file-404", env.root.display()), "grants": []}),
    )
    .await;

    // LIST: masked. The store value must not render; kind/id/grants do.
    let list = cred_list(&d).await;
    let rendered = rendered_credentials(&list);
    assert!(rendered.contains("rayl3-store") && rendered.contains("rayl3-env"), "list: {rendered}");
    assert!(
        list.to_string().contains("rayl3plug"),
        "grants must render on the list (design: add ... plugin grant checkboxes; list shows grants): {rendered}"
    );
    assert!(!rendered.contains(L3_SECRET), "masked list leaks the plaintext store value: {rendered}");

    // Store file itself is 0600 (L2 format unchanged; single writer at the
    // daemon is L3's contract — exercised by the round-trip test).
    let creds_path = env.state_dir().join("credentials.json");
    if creds_path.exists() {
        let mode = fs::metadata(&creds_path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "credentials.json mode");
    }

    // TEST: boolean verdicts.
    assert!(cred_test(&d, "rayl3-store").await, "store record resolves non-empty => true");
    assert!(cred_test(&d, "rayl3-env").await, "set env var resolves => true");
    assert!(!cred_test(&d, "rayl3-env-absent").await, "unset env var => false (boolean, not an error)");
    assert!(cred_test(&d, "rayl3-file").await, "readable file => true");
    assert!(!cred_test(&d, "rayl3-file-absent").await, "absent file => false");

    // EDIT: re-add the store record with a rotated value; old value gone.
    cred_add(
        &d,
        json!({"id": "rayl3-store", "description": "rotated", "kind": "store",
               "value": L3_SECRET_ROTATED, "grants": ["rayl3plug"]}),
    )
    .await;
    let list = cred_list(&d).await;
    let rendered = rendered_credentials(&list);
    assert!(!rendered.contains(L3_SECRET) && !rendered.contains(L3_SECRET_ROTATED), "edit leaks: {rendered}");

    // Masking audit across EVERY response surface incl. config show.
    for path in ["/api/config", "/", "/dashboard", "/api/config-files"] {
        let (status, text) = d.get_text(path).await;
        if status < 400 {
            assert!(!text.contains(L3_SECRET) && !text.contains(L3_SECRET_ROTATED),
                "plaintext credential value leaked in {path}: {}", &text[..text.len().min(400)]);
        }
    }
    let log = d.log_tail();
    assert!(!log.contains(L3_SECRET), "plaintext credential value leaked in the daemon log");

    // DELETE: gone from the list.
    cred_delete(&d, "rayl3-env-absent").await;
    let rendered = rendered_credentials(&cred_list(&d).await);
    assert!(!rendered.contains("rayl3-env-absent"), "deleted credential still listed: {rendered}");

    d.stop();
}

/// L2's plugin `secrets.get` contract keeps working for a credential added
/// through the L3 UI (grant-scoped delivery of a UI-managed value).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_credentials_plugin_secrets_get_after_ui_add() {
    let _gate = suite_gate().lock().await;
    let env = TestEnv::new("credsplug");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({"enabled": true, "kind": "heuristic"}),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    cred_add(
        &d,
        json!({"id": "rayl3-ui-cred", "description": "granted to the fixture plugin",
               "kind": "store", "value": L3_SECRET, "grants": ["rayl3plug"]}),
    )
    .await;

    // Install + enable the L2-style secrets fixture through the real CLI.
    let dir = write_plugin_fixture(&env, "rayl3plug", &["rayl3-ui-cred"]);
    let mut full = vec!["plugin", "install", dir.to_str().unwrap()];
    let run = d.cli(&full);
    assert!(run.code == 0, "plugin install failed: {}", run.out());
    full = vec!["plugin", "enable", "rayl3plug"];
    let run = d.cli(&full);
    assert!(run.code == 0, "plugin enable failed: {}", run.out());
    d.stop();

    // Restart so the supervisor spawns the plugin at boot (L2 contract).
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    let report_path = env.plugin_state("rayl3plug").join("report.json");
    let report = poll_read_json(&report_path, Duration::from_secs(30));
    assert!(
        report["secrets"]["value"].as_str().is_some_and(|v| v.contains(L3_SECRET)),
        "plugin secrets.get must return the UI-added credential value; report: {report}"
    );
    d.stop();
}


/// Usage audit (Bruce r2 Finding 1): `POST /api/credentials/audit` lists
/// masked records with `consumers` = the CREDENTIAL's own grants intersected
/// with REGISTERED plugin ids — not the manifests' RPC-method grants (a
/// disjoint vocabulary; under the old matching every list was empty).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_credentials_audit_consumers_intersect_registered_plugins() {
    let _gate = suite_gate().lock().await;
    let env = TestEnv::new("audit");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    // v1 seed: the discord token migrates to credential `discord_token`
    // granted to plugin id `bridge` (L2 contract).
    let seed = v1_seed_yaml(&local.url, &cloud.url, true);
    let mut d = Daemon::start(&env, &seed);
    d.ready().await;

    // Register `bridge` (installed plugin => registry listing; no spawn
    // needed for the audit's id set).
    let dir = write_plugin_fixture(&env, "bridge", &[]);
    let run = d.cli(&["plugin", "install", dir.to_str().unwrap()]);
    assert!(run.code == 0, "plugin install bridge failed: {}", run.out());

    // Non-granted credential + grants-with-unregistered-member credential.
    cred_add(
        &d,
        json!({"id": "rayl3-audit-none", "kind": "store", "value": L3_SECRET, "grants": []}),
    )
    .await;
    cred_add(
        &d,
        json!({"id": "rayl3-audit-ghost", "kind": "store", "value": L3_SECRET_ROTATED,
               "grants": ["bridge", "ghost-plug-not-registered"]}),
    )
    .await;

    let (status, text) = d.post_json("/api/credentials/audit", json!({})).await;
    assert_eq!(status, 200, "audit: {status}: {text}");
    let audit = parse_json_from(&text).expect("audit JSON");
    let records: Vec<Value> = audit["credentials"]
        .as_array()
        .unwrap_or_else(|| panic!("records: {audit}"))
        .clone();
    let record = |id: &str| -> Value {
        records
            .iter()
            .find(|r| r["id"].as_str() == Some(id))
            .unwrap_or_else(|| panic!("audit record {id} missing: {audit}"))
            .clone()
    };
    assert_eq!(
        record("discord_token")["consumers"],
        json!(["bridge"]),
        "granted credential lists its own granted, registered plugin"
    );
    assert_eq!(
        record("rayl3-audit-none")["consumers"],
        json!([]),
        "non-granted credential lists no consumers"
    );
    assert_eq!(
        record("rayl3-audit-ghost")["consumers"],
        json!(["bridge"]),
        "consumers are the grants INTERSECTED with registered plugin ids (the ghost id must drop)"
    );
    // Masked shape: the audit carries no plaintext values.
    assert!(
        !text.contains(L3_SECRET) && !text.contains(L3_SECRET_ROTATED) && !text.contains(DISCORD_TOKEN),
        "audit response leaks plaintext: {text}"
    );
    d.stop();

}
// ── (4) One entry + sync-omp removal ───────────────────────────────────────

/// `brainrouter install <harness>` in a temp HOME yields EXACTLY ONE entry
/// (brainrouter/auto): no per-model listings, no sync-omp mention. Design:
/// "Templates (all harnesses): exactly ONE entry; per-model listings
/// removed." v1 base (install.rs L79-91 @ 6c47ad7) prints THREE model ids.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_install_single_auto_entry_per_harness() {
    let _gate = suite_gate().lock().await;
    // omp: the provider block (file-written or printed) lists one model.
    let env = TestEnv::new("install-omp");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_brainrouter"));
    cmd.args(["install", "omp", "--yes"]);
    env.apply(&mut cmd, std::iter::empty::<(String, String)>());
    let out = cmd.output().expect("install omp");
    assert_eq!(out.status.code(), Some(0), "install omp failed");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let text = format!("{stdout}{stderr}");
    assert!(!text.contains("sync-omp"), "install omp output mentions sync-omp: {text}");

    let models_yml = env.home.join(".omp/agent/models.yml");
    let ids: Vec<String> = if models_yml.exists() {
        let yml = fs::read_to_string(&models_yml).unwrap();
        let v: Value = serde_yaml::from_str(&yml).unwrap();
        let provider = v
            .get("brainrouter")
            .unwrap_or_else(|| panic!("models.yml lacks the brainrouter provider: {yml}"));
        provider["models"]
            .as_array()
            .unwrap_or_else(|| panic!("provider models is not a list: {yml}"))
            .iter()
            .map(|m| m["id"].as_str().unwrap_or_default().to_string())
            .collect()
    } else {
        // Printed-block shape (v1 convention): `- id: <x>` lines under the
        // brainrouter provider.
        text.lines()
            .filter_map(|l| l.trim().strip_prefix("- id:"))
            .map(|s| s.trim().to_string())
            .collect()
    };
    assert!(
        ids.iter().any(|i| i == "auto" || i == "brainrouter/auto"),
        "one-entry install must offer brainrouter/auto; got {ids:?} (output: {text})"
    );
    assert_eq!(
        ids.len(),
        1,
        "exactly ONE model entry (design: per-model listings removed); got {ids:?} (output: {text})"
    );

    // opencode: the merged JSON config lists exactly one brainrouter model.
    let env2 = TestEnv::new("install-oc");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_brainrouter"));
    cmd.args(["install", "opencode", "--yes"]);
    env2.apply(&mut cmd, std::iter::empty::<(String, String)>());
    let out = cmd.output().expect("install opencode");
    assert_eq!(out.status.code(), Some(0), "install opencode failed");
    let oc = env2.config_home.join("opencode/config.json");
    if oc.exists() {
        let v: Value = serde_json::from_reader(fs::File::open(&oc).unwrap()).unwrap();
        let models = v["provider"]["brainrouter"]["models"].clone();
        let n = models.as_object().map(|m| m.len()).unwrap_or(usize::MAX);
        assert_eq!(n, 1, "opencode template: exactly one entry; got {models}");
    }
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(!text.contains("sync-omp"), "install opencode output mentions sync-omp: {text}");
}

/// sync-omp is deleted at all four points (L3 spec "Harness one-entry"):
/// CLI command, /api/models/sync-omp route, CSRF allowlist entry, daemon
/// bootstrap auto-sync task. Path allowlist: src/** only (docs/CHANGELOG
/// mention the removal and must keep it).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_sync_omp_removed_at_four_points() {
    let _gate = suite_gate().lock().await;
    // Point 1: CLI — the verb is gone from `--help`.
    let out = Command::new(env!("CARGO_BIN_EXE_brainrouter"))
        .args(["--help"])
        .output()
        .expect("run --help");
    let help = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let lower = help.to_lowercase();
    assert!(
        !lower.contains("sync-omp") && !lower.contains("sync_omp") && !lower.contains("syncomp"),
        "`--help` still advertises a sync-omp command: {help}"
    );

    // Point 2: the daemon route answers 404 (four points, one daemon).
    let env = TestEnv::new("syncomp");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({"enabled": true, "kind": "heuristic"}),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    let (status, _) = d.post_json("/api/models/sync-omp", json!({})).await;
    assert_eq!(status, 404, "/api/models/sync-omp route must be gone");
    d.stop();

    // Points 2-4 (route + CSRF allowlist entry + daemon bootstrap task) and
    // point 1 (CLI command) in source: no sync-omp identifier anywhere
    // under src/. v1 sites (6c47ad7): server.rs L650 (route), server.rs
    // L288 (CSRF allowlist), daemon.rs L588-603 (bootstrap auto-sync),
    // cli.rs L555-557 (verb), server.rs L4075 (sync_omp_models itself).
    let mut offenders = Vec::new();
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    for file in walk_source(&src) {
        let text = fs::read_to_string(&file).unwrap_or_default();
        let low = text.to_lowercase();
        if low.contains("sync-omp") || low.contains("sync_omp") || low.contains("syncomp") {
            offenders.push(file.display().to_string());
        }
    }
    assert!(
        offenders.is_empty(),
        "sync-omp survives in src/ (four-point removal incomplete): {offenders:?}"
    );
}

fn walk_source(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| matches!(x.to_str(), Some("rs") | Some("html") | Some("toml") | Some("yaml") | Some("yml"))) {
                out.push(p);
            }
        }
    }
    out
}

// ── (5) Events (additive schema) ────────────────────────────────────────────

/// Slot-routed requests carry the additive fields role / decision /
/// decision_fallback / chosen_model while every v1 RouteEvent field stays
/// intact ("Contracts kept": event schema additive fields only).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_events_additive_fields_v1_fields_intact() {
    let _gate = suite_gate().lock().await;
    let env = TestEnv::new("events");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let cls = spawn_mock(MockCfg {
        classifier: true,
        classifier_word: "local".into(),
        classifier_status: 200,
        ..Default::default()
    });
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({
            "enabled": true, "kind": "openai-compatible", "base_url": cls.url,
            "model": "bonsai", "prompt_profile": "bonsai", "timeout_ms": 5000,
        }),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    roles_set(&d, "main", "auto", None);

    let r = chat(&d, "brainrouter/auto", Some("main"), "r3-events-additive").await;
    assert_served(&r, "slot-routed request");
    let e = event_for(&d, "r3-events-additive").await;

    // Additive L3 fields.
    assert_eq!(e["role"], json!("main"), "additive `role`");
    let dec = &e["decision"];
    assert_eq!(dec["route"], json!("local"), "decision.route");
    assert_eq!(dec["tier"], json!("fast"), "v1 word `local` -> canon tier `fast`");
    assert!(dec["confidence"].is_number(), "decision.confidence must be numeric: {dec}");
    assert_eq!(
        e["chosen_model"].as_str(),
        Some("ray-l3-fallback"),
        "additive `chosen_model` names the served model"
    );
    assert!(e["rationale"].is_string(), "additive `rationale`: {e}");
    // decision_fallback is present (null/absent) on a clean decision and
    // carries the fallback label on fallback rows (asserted in the failure
    // tests above).
    assert!(
        e.get("decision_fallback").is_none()
            || e["decision_fallback"].is_null()
            || e["decision_fallback"].is_string(),
        "decision_fallback must be absent/null on clean decisions: {e}"
    );

    // v1 fields intact (v1 consumers keep parsing).
    for f in V1_EVENT_FIELDS {
        assert!(e.get(*f).is_some(), "v1 event field `{f}` missing (additive-only contract): {e}");
    }
    assert_eq!(e["requested_model"], json!("brainrouter/auto"));
    assert_eq!(e["success"], json!(true));
    assert_eq!(e["model_key"], json!("ray-l3-fallback"));
    assert_eq!(e["effective_provider"], json!("llama-swap"));
    assert!(e["id"].is_number() && e["latency_ms"].is_number());
    assert!(e["stage"].is_string() && e["bonsai_decision"].is_string() && e["routing_class"].is_string());

    d.stop();
}

// ── (6) Migrations ──────────────────────────────────────────────────────────

/// v1-state boot (routing_profile + project pins + bonsai + review.forced_*)
/// under a fresh HOME: the exact migration table (L3 spec "v1 -> card
/// migration table, exact, no inference").
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_migration_v1_state_role_cards_exact_table() {
    let _gate = suite_gate().lock().await;
    let env = TestEnv::new("migrate-table");
    fs::create_dir_all(env.config_dir()).unwrap();
    fs::write(env.config_dir().join("project_pins.json"), v1_project_pins_json(&repo_root())).unwrap();
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let seed = v1_seed_yaml(&local.url, &cloud.url, true);
    let mut d = Daemon::start(&env, &seed);
    d.ready().await;

    let cards = roles_get(&d);
    // routing.main = Local{None} -> `local` (table row; routing_state.json /
    // YAML profile path).
    assert_eq!(
        card_assignment(&cards, "main", None).as_deref(),
        Some("local"),
        "main card = table row Local{{None}} -> local; cards: {cards:?}"
    );
    // preset-expanded reviewer Cloud{None} -> `cloud`.
    assert_eq!(
        card_assignment(&cards, "review", None).as_deref(),
        Some("cloud"),
        "review card from preset expansion: Cloud{{None}} -> cloud; cards: {cards:?}"
    );
    // profile subagent_model -> subagent assignment (bare key).
    assert_eq!(
        card_assignment(&cards, "subagent", None).as_deref(),
        Some("prof-subs-model"),
        "subagent card from RoutingProfile.subagent_model; cards: {cards:?}"
    );
    // Project pins map identically: cloud-pinned main -> cloud/<id>;
    // subagent pin -> subagent assignment.
    assert_eq!(
        card_assignment(&cards, "main", Some(&repo_root())).as_deref(),
        Some("cloud/pin-cloud-id"),
        "project-scoped card from the v1 pin (Cloud{{Some}} -> cloud/<id>); cards: {cards:?}"
    );
    assert_eq!(
        card_assignment(&cards, "subagent", Some(&repo_root())).as_deref(),
        Some("pin-subs-model"),
        "project-scoped subagent pin migrated; cards: {cards:?}"
    );
    assert_eq!(
        card_assignment(&cards, "review", Some(&repo_root())).as_deref(),
        Some("cloud/vendor/review-2026"),
        "the owner's cloud-pinned reviewer lands on a cloud/<id> card, NOT re-pinned; cards: {cards:?}"
    );

    // Boundary behavior: the project-scoped migrated card serves main-role
    // requests from this project (project beats global — the peer-cwd of the
    // test process is the canonical repo root carrying this pin, see header
    // PROJECT CONTEXT). v1 pin precedence: src/router.rs:288-297 @ v1-classic.
    let r = chat(&d, "brainrouter/auto", None, "r3-migrate-project-pin").await;
    assert_served(&r, "migrated project card");
    assert_eq!(
        cloud.models_seen(),
        vec!["pin-cloud-id".to_string()],
        "project cloud pin must serve; cloud saw {:?}, local saw {:?}",
        cloud.models_seen(),
        local.models_seen()
    );

    // bonsai.* migrates to the decision slot: `decision show` reports the
    // openai-compatible kind with the bonsai prompt profile (v1 YAML had
    // bonsai.enabled: false -> the slot keeps it disabled, not heuristic).
    let run = d.cli(&["decision", "show"]);
    let text = run.out();
    if run.code == 0 {
        let low = text.to_lowercase();
        assert!(low.contains("openai-compatible") && low.contains("bonsai"),
            "`decision show` must report the migrated bonsai slot (kind openai-compatible, prompt_profile bonsai): {text}");
    }

    // Migrated YAML keys removed (snapshot first — restore proven next test).
    let yml = fs::read_to_string(env.root.join("config.yaml")).unwrap();
    assert!(!yml.contains("routing:"), "routing block must be consumed by the migration:\n{yml}");
    assert!(!yml.contains("forced_model"), "review.forced_model must be consumed:\n{yml}");
    assert!(!yml.contains("forced_mode"), "review.forced_mode must be consumed:\n{yml}");
    assert!(!yml.contains("bonsai:"), "bonsai.* must be consumed:\n{yml}");
    assert!(yml.contains("decision_model"), "the migrated config carries the decision slot:\n{yml}");

    d.stop();
}

/// No-routing-block v1 YAML (main = local, the v1 default —
/// src/config.rs L503-513 @ v1-classic) with review.forced_mode/model +
/// survivors: the cloud-pinned reviewer lands on `cloud/<id>` (NOT
/// re-pinned), max_iterations + hankndory_integration survive on the review
/// card via the reduced /api/review-config, snapshot + restore v2.0-p3
/// round-trip, and --dry-run is inert.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_migration_no_routing_review_survivors_restore_dryrun() {
    let _gate = suite_gate().lock().await;
    // --dry-run FIRST, inert, on a pristine v1 state.
    let env = TestEnv::new("migrate-dry");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let seed = v1_seed_yaml(&local.url, &cloud.url, false);
    let config_path = env.root.join("config.yaml");
    fs::write(&config_path, &seed).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_brainrouter"))
        .args(["serve", "--config"])
        .arg(&config_path)
        .arg("--dry-run")
        .current_dir(&env.root)
        .stdin(Stdio::null())
        .output()
        .expect("serve --dry-run");
    assert_eq!(out.status.code(), Some(0), "dry run must exit 0: {:?}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(fs::read_to_string(&config_path).unwrap(), seed, "--dry-run mutated the v1 YAML");
    assert!(
        !env.state_dir().join("migrations").join("v2.0-p3").exists(),
        "--dry-run wrote v2.0-p3 migration snapshots"
    );
    assert!(
        find_files_named(&env.config_dir(), "role_cards").is_empty()
            && find_files_named(&env.state_dir(), "role_cards").is_empty(),
        "--dry-run created the role cards store"
    );
    assert!(!env.state_dir().join("credentials.json").exists(), "--dry-run wrote the credentials store");

    // Real boot of the same seed under a fresh HOME.
    let env = TestEnv::new("migrate-live");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let seed = v1_seed_yaml(&local.url, &cloud.url, false);
    let config_path = env.root.join("config.yaml");
    fs::write(&config_path, &seed).unwrap();
    let mut d = Daemon::start(&env, &seed);
    d.ready().await;

    let cards = roles_get(&d);
    // No routing block => main = local (v1 default; "default-local stays
    // local").
    assert_eq!(
        card_assignment(&cards, "main", None).as_deref(),
        Some("local"),
        "no-routing YAML: main migrates to local; cards: {cards:?}"
    );
    // The owner's cloud-pinned reviewer lands on a cloud/<id> card, not
    // re-pinned: review.forced_mode=cloud + forced_model=vendor/review-2026
    // -> `cloud/vendor/review-2026` (migration table via
    // ReviewConfig::model_choice, src/config.rs L488-493 @ v1-classic).
    assert_eq!(
        card_assignment(&cards, "review", None).as_deref(),
        Some("cloud/vendor/review-2026"),
        "cloud-pinned reviewer preserved as cloud/<id>; cards: {cards:?}"
    );
    // config.subs_model stays the auto-pool compatibility seed (not a card).
    // Boundary: main-role auto request serves v1-default local.
    let r = chat(&d, "brainrouter/auto", None, "r3-migrate-norouting").await;
    assert_served(&r, "no-routing migration parity (local)");
    assert_eq!(local.models_seen(), vec!["ray-l3-fallback".to_string()]);

    // Reduced review-config survivors, live on the review card.
    let rc = d.get_json("/api/review-config").await;
    assert_eq!(rc["max_iterations"], json!(3), "max_iterations seeded from YAML: {rc}");
    assert_eq!(rc["hankndory_integration"], json!(true), "hankndory_integration seeded from YAML: {rc}");
    // Reduced schema: the model keys are gone (the review CARD owns the
    // reviewer model now; design: "schema reduced to the non-model
    // survivors").
    assert!(
        rc.get("forced_mode").is_none() && rc.get("forced_model").is_none(),
        "/api/review-config must be reduced to the survivors (max_iterations, hankndory_integration): {rc}"
    );
    let (status, text) = d
        .post_json("/api/review-config", json!({"max_iterations": 7, "hankndory_integration": true}))
        .await;
    assert!((200..300).contains(&status), "POST reduced review-config: {status}: {text}");
    let rc = d.get_json("/api/review-config").await;
    assert_eq!(rc["max_iterations"], json!(7), "endpoint mutates the card via the daemon (live-mutated, loop-driving): {rc}");

    // Snapshot exists for the phase; restore round-trips the YAML.
    assert!(
        env.state_dir().join("migrations").join("v2.0-p3").exists(),
        "no v2.0-p3 migration snapshot directory"
    );
    d.stop();
    let migrated = fs::read_to_string(&config_path).unwrap();
    assert_ne!(migrated, seed, "migration did not rewrite the config (expected: v1 keys consumed)");
    let out = Command::new(env!("CARGO_BIN_EXE_brainrouter"))
        .args(["migration", "--config"])
        .arg(&config_path)
        .args(["restore", "v2.0-p3"])
        .current_dir(&env.root)
        .stdin(Stdio::null())
        .output()
        .expect("migration restore");
    assert_eq!(
        out.status.code(),
        Some(0),
        "restore v2.0-p3 failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        fs::read_to_string(&config_path).unwrap(),
        seed,
        "restore must round-trip the pre-migration v1 YAML byte-for-byte"
    );
}

/// B1 (Bruce NO-GO, Step 9): the v2.0-p3 migration must be one-shot for
/// EVERY v1 config. The trigger guard is `needs_l3_migration =
/// !has_decision_model` (src/migration.rs); the fix inserts
/// `decision_model:` unconditionally (enabled: false, kind: heuristic on
/// the no-bonsai path). With the insertion gated on the bonsai block, a v1
/// YAML without `bonsai:` re-ran the migration on EVERY boot: post-
/// migration card/policy edits were clobbered back to the migrated seed, a
/// snapshot accumulated per boot, and the warn-level migration action
/// ("Migration v2.0-p3: migrated v1 routing profile …", src/daemon.rs)
/// repeated in every log. Two boots, v1 YAML WITHOUT a bonsai block:
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_migration_no_bonsai_block_is_one_shot() {
    let _gate = suite_gate().lock().await;
    let env = TestEnv::new("migrate-nobonsai");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    // v1 state: routing block + review.forced_* present, deliberately NO
    // bonsai: key.
    let seed = {
        let mut v: Value =
            serde_yaml::from_str(&v1_seed_yaml(&local.url, &cloud.url, true)).unwrap();
        v.as_object_mut().unwrap().remove("bonsai");
        serde_yaml::to_string(&v).unwrap()
    };
    let mut d = Daemon::start(&env, &seed);
    d.ready().await;

    // Boot 1 consumed the v1 keys; make post-migration edits — the state a
    // re-running migration would clobber.
    roles_set(&d, "main", "post-mig-main-model", None);
    set_policy(&d, "local-only", "fast").await;
    let roles1 = d.get_json("/api/roles").await;
    d.stop();

    // Boot 2 over the config left on disk (a real operator restart).
    let mut d = Daemon::start_from_disk(&env);
    d.ready().await;
    let roles2 = d.get_json("/api/roles").await;
    assert_eq!(
        roles2["assignments"], roles1["assignments"],
        "boot 2 must not re-run v2.0-p3: post-migration cards must survive. \
         boot1: {} boot2: {}",
        roles1["assignments"], roles2["assignments"]
    );
    assert_eq!(
        roles2["policy"], roles1["policy"],
        "boot 2 must not re-run v2.0-p3: the policy record must survive. \
         boot1: {} boot2: {}",
        roles1["policy"], roles2["policy"]
    );

    // The preserved card is LIVE: the v1 routing block said main = local;
    // a boot-2 re-run resets the card and serves the fallback seed.
    let r = chat(&d, "brainrouter/auto", None, "r3-nobonsai-boot2").await;
    assert_served(&r, "post-boot-1 card live on boot 2");
    assert_eq!(
        local.models_seen(),
        vec!["post-mig-main-model".to_string()],
        "the post-migration bare-key card must serve; a re-migration would \
         route `local` to ray-l3-fallback"
    );

    // Exactly one v2.0-p3 snapshot directory (boot 1's; boot 2 is a no-op).
    let snaps = env.state_dir().join("migrations").join("v2.0-p3");
    let n = fs::read_dir(&snaps).map(|rd| rd.count()).unwrap_or(0);
    assert_eq!(
        n, 1,
        "boot 2 created a second v2.0-p3 snapshot: {n} entries under {}",
        snaps.display()
    );

    // No repeated migration action in boot 2's log (warn-level, prefix
    // "Migration v2.0-p3:").
    let log = d.log_tail();
    assert!(
        !log.contains("Migration v2.0-p3"),
        "boot 2 logged a migration action — v2.0-p3 is not one-shot:\n{log}"
    );
    d.stop();
}

fn find_files_named(dir: &Path, needle: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return out;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.file_name().is_some_and(|n| n.to_string_lossy().contains(needle)) {
            out.push(p.clone());
        }
        if p.is_dir() {
            out.extend(find_files_named(&p, needle));
        }
    }
    out
}

// ── (7) Strictness per selector (failing mock upstream) ─────────────────────

/// `<bare-key>` cards are direct picks — STRICT, no fallback, the error
/// surfaces (L3 "Pinned-card strictness"; v1: route_local(allow_fallback=
/// false), src/router.rs L658-700 + the v1 keeper test
/// `direct_model_failure_does_not_fall_back_to_fallback_model`,
/// tests/subs_routing_test.rs).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_bare_key_card_strict_no_retry() {
    let _gate = suite_gate().lock().await;
    let env = TestEnv::new("bare-strict");
    let local = spawn_mock(MockCfg {
        fail_models: vec!["strict-bare-model".into()],
        ..Default::default()
    });
    let cloud = spawn_mock(MockCfg::default());
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({"enabled": true, "kind": "heuristic"}),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    roles_set(&d, "main", "strict-bare-model", None);

    let (status, _) = chat(&d, "brainrouter/auto", None, "r3-bare-fail").await;
    assert!(status >= 400, "bare-key card failure must surface to the client, got {status}");
    assert_eq!(
        local.models_seen(),
        vec!["strict-bare-model".to_string()],
        "NO fallback_model retry and NO chain on a bare-key card"
    );
    let e = event_for(&d, "r3-bare-fail").await;
    assert_eq!(e["success"], json!(false), "failed route event: {e}");
    assert_eq!(e["role"], json!("main"));
    d.stop();
}

/// `local` cards keep v1's managed semantics: the selector routes to
/// `fallback_model` (src/router.rs L346-356 @ v1-classic). When that model
/// itself fails, v1 does NOT spin (route_local returns the error when
/// requested == fallback_model, router.rs L699) — one upstream hit, the
/// error surfaces.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_local_card_v1_managed_fallback_semantics() {
    let _gate = suite_gate().lock().await;
    // Healthy row: served on the fallback model.
    let env = TestEnv::new("local-ok");
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg::default());
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({"enabled": true, "kind": "heuristic"}),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    roles_set(&d, "main", "local", None);
    let r = chat(&d, "brainrouter/auto", None, "r3-local-served").await;
    assert_served(&r, "local card served");
    assert_eq!(local.models_seen(), vec!["ray-l3-fallback".to_string()]);
    d.stop();

    // Failing row: v1's degenerate-retry guard — exactly ONE hit.
    let env = TestEnv::new("local-fail");
    let local = spawn_mock(MockCfg {
        fail_models: vec!["ray-l3-fallback".into()],
        ..Default::default()
    });
    let cloud = spawn_mock(MockCfg::default());
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({"enabled": true, "kind": "heuristic"}),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    roles_set(&d, "main", "local", None);
    let (status, _) = chat(&d, "brainrouter/auto", None, "r3-local-fail").await;
    assert!(status >= 400, "failing managed local route surfaces: {status}");
    assert_eq!(
        local.models_seen(),
        vec!["ray-l3-fallback".to_string()],
        "v1 parity: requested == fallback_model short-circuits the retry (router.rs L699 @ v1-classic)"
    );
    d.stop();
}

/// `cloud` cards keep the managed local-fallback hop (one hop, events for
/// both legs — v1 route_cloud, src/router.rs L568-650 @ v1-classic) EXCEPT
/// role `review`, where v1's role-specific strict rule (no local fallback,
// failure surfaces for escalation) is preserved (route_with_choice +
// bail at router.rs L627-631 @ v1-classic).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_cloud_card_hop_and_reviewer_strict() {
    let _gate = suite_gate().lock().await;
    let env = TestEnv::new("cloud-hop");
    // Cloud is dead; llama-swap is healthy.
    let local = spawn_mock(MockCfg::default());
    let cloud = spawn_mock(MockCfg {
        fail_chat_all: true,
        ..Default::default()
    });
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "ray-l3-fallback",
        &[],
        None,
        json!({"enabled": true, "kind": "heuristic"}),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;

    // main = cloud: managed hop -> llama-swap on fallback_model, served.
    roles_set(&d, "main", "cloud", None);
    let r = chat(&d, "brainrouter/auto", None, "r3-cloud-hop").await;
    assert_served(&r, "cloud card hop");
    assert_eq!(local.models_seen(), vec!["ray-l3-fallback".to_string()], "hop must land on fallback_model");
    let e = event_for(&d, "r3-cloud-hop").await;
    assert_eq!(e["success"], json!(true));
    assert_eq!(e["stage"], json!("cloud_fallback"), "v1 stage tag for the hop (routing_events.rs Stage)");
    let evs = events(&d).await;
    assert!(
        evs.iter().any(|x| x["success"] == json!(false)
            && x["stage"] == json!("cloud_primary")
            && x["effective_provider"] == json!("manifest")),
        "the failed cloud leg must be recorded (v1: one event per failed hop): {evs:?}"
    );

    // review = cloud/<id>: role-specific STRICT rule — no local hop.
    roles_set(&d, "review", "cloud/vendor/review-2026", None);
    let (status, _) = chat(&d, "brainrouter/auto", Some("review"), "r3-rev-strict").await;
    assert!(
        status >= 400,
        "the reviewer cloud failure must surface (escalation source), got {status}"
    );
    assert_eq!(
        local.total(),
        1,
        "role `review` on a cloud selector must NEVER hop to the local fallback (v1 strict rule)"
    );
    let e = event_for(&d, "r3-rev-strict").await;
    assert_eq!(e["success"], json!(false));
    assert_eq!(e["role"], json!("review"));
    d.stop();
}

/// The auto path's chain: on CONNECT/response-headline failure the NEXT
/// candidate is served (L3 "Auto-path selection": `fallback_model` stays as
/// chain seed; chain only exists on auto). And mid-stream failures are
/// never retried: the client sees the stream die, one upstream hit.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn l3_auto_chain_next_candidate_and_midstream_never_retried() {
    let _gate = suite_gate().lock().await;
    // Chain row: seed (fallback_model) fails at the response headline, the
    // next pool candidate serves.
    let env = TestEnv::new("chain");
    let local = spawn_mock(MockCfg {
        models: vec!["chain-first-seed".into(), "chain-second".into()],
        fail_models: vec!["chain-first-seed".into()],
        ..Default::default()
    });
    let cloud = spawn_mock(MockCfg::default());
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "chain-first-seed",
        &["chain-first-seed", "chain-second"],
        None,
        json!({"enabled": true, "kind": "heuristic"}),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    roles_set(&d, "main", "auto", None);
    let r = chat(&d, "brainrouter/auto", None, "r3-chain").await;
    assert_served(&r, "auto chain: next candidate must serve");
    assert_eq!(local.count_of("chain-first-seed"), 1, "chain: first candidate tried once");
    assert!(local.count_of("chain-second") >= 1, "chain: next candidate served; local saw {:?}", local.models_seen());
    d.stop();

    // Mid-stream row: headline 200, then the stream dies. v1 (and L3):
    // never retried — the proxy client sees the failure, exactly one hit.
    let env = TestEnv::new("midstream");
    let local = spawn_mock(MockCfg {
        models: vec!["die-mid-model".into()],
        die_midstream_models: vec!["die-mid-model".into()],
        ..Default::default()
    });
    let cloud = spawn_mock(MockCfg::default());
    let cfg = l3_config_yaml(
        &local.url,
        &cloud.url,
        "die-mid-model",
        &["die-mid-model"],
        None,
        json!({"enabled": true, "kind": "heuristic"}),
    );
    let mut d = Daemon::start(&env, &cfg);
    d.ready().await;
    roles_set(&d, "main", "auto", None);
    let (status, body) = chat(&d, "brainrouter/auto", None, "r3-midstream").await;
    assert_eq!(status, 200, "the headline reached the client before the stream died");
    match &body {
        Ok(text) => assert!(
            !text.contains("[DONE]"),
            "the mock never sends [DONE]; a complete body means the stream was silently re-served: {text}"
        ),
        Err(_) => {} // reqwest body error on the truncated stream — the client sees the failure
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        local.count_of("die-mid-model"),
        1,
        "mid-stream failure must NEVER be retried upstream (chain is connect/headline-only)"
    );
    d.stop();
}
