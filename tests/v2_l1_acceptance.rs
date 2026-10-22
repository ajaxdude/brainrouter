//! Acceptance tests for leg L1 (Purge) of `docs/design/v2-streamlining.md`.
//!
//! Written by Ray from the design's promises alone (Requirements R1/R2, L1
//! Contracts kept, L1 Error behavior, L1 Acceptance grep gate). The builder
//! runs these and NEVER edits them.
//!
//! Mapping of tests to the design's L1 acceptance bullets:
//! - Bullet 1 (grep gate): `grep_gate_no_purged_backend_names_outside_exceptions`
//! - Bullet 2 (R1 browse-only): `r1_vendored_feed_parses_and_removed_backends_are_browse_only`
//! - Bullet 3 (R1 kept behavior, routes): `r1_removed_server_mode_routes_answer_404`,
//!   `r1_kept_server_mode_routes_stay_registered`
//! - Bullet 4 (R2 benchmarks feature gate):
//!   `r2_benchmarks_feature_is_declared_and_non_default`,
//!   `r2_benchmarks_route_serves_documented_degradation_contract`
//!
//! Expectation: the purge-specific tests are RED on the design base commit
//! (halogen/vllm/r9v still present) and GREEN once L1 lands; the keeper tests
//! (kept routes, kept-behavior asserts, degradation contract) are green at both
//! points and guard against over-purge.

use brainrouter::benchmark;
use brainrouter::toolbox_catalog::{self, schema_validate, CatalogBackendId, SupportedServingBackend};
use bytes::Bytes;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use std::fs;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// The three backends leg L1 purges (design "Decision log": server-mode
/// backends = ds4+gufo only).
const REMOVED_BACKENDS: [&str; 3] = ["halogen", "vllm", "r9v"];
/// Serving backends L1 must keep actionable (R1: "routes llama-swap/Manifest/
/// gufo/ds4 traffic identically"; llama_cpp is the catalog GGUF path).
const KEPT_SERVING_BACKENDS: [&str; 3] = ["llama_cpp", "ds4", "gufo"];

// ── Bullet 1: grep gate ──────────────────────────────────────────────────────
//
// Design (L1 Acceptance): no `halogen|vllm|r9v` (case-insensitive) outside the
// enumerated exceptions. The exception list here mirrors the design, expressed
// as concrete paths: the `CatalogBackendId` parsing + feed-vocabulary code
// constants kept so the vendored feed parses (`src/toolbox_catalog/types.rs`,
// `src/toolbox_catalog/models.rs`, `src/toolbox_catalog/schema_validate.rs`
// KNOWN_BACKEND_IDS, `src/toolbox_catalog.rs`), the keeper/parent-module
// doc-comment exception (`src/routing_events.rs`, `src/toolbox_catalog.rs`),
// the applied migration (`migrations/`, frozen, outside the walk scope below),
// the feed asset (`assets/cockpit-catalog/`, outside the walk scope),
// historical docs (PRD.md / README.md / docs/ / CHANGELOG.md, outside the walk
// scope), and this test file itself.
//
// The walk scope is `src/`, `scripts/`, `tests/` (minus this file), and
// `templates/` if present — kept simple and deterministic: sorted traversal,
// a fixed text-extension allowlist, and the enumerated exception list.
const GREP_GATE_EXCEPTIONS: &[&str] = &[
    "src/toolbox_catalog/types.rs",
    "src/toolbox_catalog/models.rs",
    "src/toolbox_catalog/schema_validate.rs",
    "src/toolbox_catalog.rs",
    "src/routing_events.rs",
    "tests/v2_l1_acceptance.rs",
];

/// Text file extensions the gate reads (everything else in the walk scope is
/// binary/data and cannot contain a source-level reference the gate targets).
const GREP_GATE_EXTENSIONS: &[&str] = &[
    "rs", "toml", "md", "html", "htm", "sh", "bash", "cjs", "mjs", "js", "json", "jsonl", "yaml",
    "yml", "txt", "sql", "css",
];

/// Directories walked by the gate, relative to the manifest root. `templates/`
/// is included where present (it is absent at the base commit).
const GREP_GATE_ROOTS: &[&str] = &["src", "scripts", "tests", "templates"];

fn grep_gate_violations(root: &Path) -> Vec<String> {
    let mut violations = Vec::new();
    for top in GREP_GATE_ROOTS.iter() {
        let dir = root.join(top);
        if dir.is_dir() {
            grep_gate_walk(top, &dir, &mut violations);
        }
    }
    violations
}

fn grep_gate_walk(rel_dir: &str, dir: &Path, violations: &mut Vec<String>) {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("grep gate cannot read {dir:?}: {e}"))
        .map(|entry| entry.expect("grep gate readdir entry"))
        .collect();
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();
        if name == ".git" || name == "target" || name == "node_modules" {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            grep_gate_walk(&format!("{rel_dir}/{name}"), &path, violations);
            continue;
        }
        let rel = format!("{rel_dir}/{name}");
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default();
        if !GREP_GATE_EXTENSIONS.contains(&extension) {
            continue;
        }
        if GREP_GATE_EXCEPTIONS.contains(&rel.as_str()) {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else {
            continue; // non-UTF-8 content: not source text the gate targets
        };
        for (index, line) in text.lines().enumerate() {
            let lower = line.to_lowercase();
            for term in REMOVED_BACKENDS {
                if lower.contains(term) {
                    violations.push(format!(
                        "{rel}:{}: {term}: {}",
                        index + 1,
                        line.trim()
                    ));
                }
            }
        }
    }
}

#[test]
fn grep_gate_no_purged_backend_names_outside_exceptions() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let violations = grep_gate_violations(&root);
    assert!(
        violations.is_empty(),
        "L1 grep gate: `halogen|vllm|r9v` (case-insensitive) must not appear in \
         src/, scripts/, tests/, templates/ outside the design's enumerated \
         exception paths. {} violation(s):\n{}",
        violations.len(),
        violations.join("\n")
    );
}

// ── Bullet 2: R1 browse-only catalog feed ────────────────────────────────────

#[test]
fn r1_vendored_feed_parses_and_removed_backends_are_browse_only() {
    // The vendored feed must parse, structurally validate clean, and survive
    // the typed layer ("vendored catalog feed parses unchanged", Contracts
    // kept) — this is green at base and must stay green through L1.
    let vendored = toolbox_catalog::load_vendored_catalog();
    assert!(
        vendored.report.errors.is_empty(),
        "vendored feed must not produce validation errors: {:?}",
        vendored.report.errors
    );
    assert!(
        vendored.toolbox_count().unwrap_or(0) > 0,
        "vendored toolboxes.json must parse into entries"
    );
    let (toolboxes, models) = vendored
        .typed()
        .expect("typed vendored feed (ToolboxCatalog + ModelCatalog) must parse");

    // Guard against a vacuous pass: the feed really contains the purged
    // backends' entries (they are data the purge keeps parsing, not deletes).
    for name in REMOVED_BACKENDS {
        assert!(
            toolboxes
                .toolboxes
                .iter()
                .any(|toolbox| toolbox.backend.as_str() == name),
            "sanity: vendored feed must contain at least one `{name}` toolbox entry"
        );
        let section = models
            .backends
            .iter()
            .find(|section| section.backend.as_str() == name);
        assert!(
            section.is_some_and(|section| !section.entries.is_empty()),
            "sanity: vendored feed must contain `{name}` model entries"
        );
    }

    for name in REMOVED_BACKENDS {
        // Feed vocabulary: the backend id keeps parsing losslessly into
        // CatalogBackendId and round-trips...
        let id: CatalogBackendId = serde_json::from_value(json!(name))
            .unwrap_or_else(|e| panic!("feed-vocabulary id `{name}` must keep parsing: {e}"));
        assert_eq!(id.as_str(), name);
        assert_eq!(
            serde_json::to_value(&id).expect("CatalogBackendId serializes"),
            json!(name),
            "feed-vocabulary id `{name}` must round-trip"
        );

        // ...but it converts to NO supported serving backend (browse-only, the
        // L1 "parses as browse-only" promise; Error behavior: "not supported
        // by this host").
        assert!(
            SupportedServingBackend::try_from(&id).is_err(),
            "L1 R1: feed id `{name}` must convert to a CatalogBackendId only, \
             not to a SupportedServingBackend (browse-only after the purge)"
        );
        assert!(
            serde_json::from_value::<SupportedServingBackend>(json!(name)).is_err(),
            "L1 R1: `{name}` must no longer deserialize as a SupportedServingBackend"
        );

        // Feed entries keep the data (id + raw) and gain no actionable payload.
        for toolbox in toolboxes
            .toolboxes
            .iter()
            .filter(|toolbox| toolbox.backend.as_str() == name)
        {
            assert!(
                toolbox.supported_backend().is_none(),
                "L1 R1: feed toolbox `{}` (backend `{name}`) must be browse-only",
                toolbox.id
            );
        }
        for section in models
            .backends
            .iter()
            .filter(|section| section.backend.as_str() == name)
        {
            for entry in &section.entries {
                assert!(
                    entry.payload.is_none(),
                    "L1 R1: feed model entry `{}` (backend `{name}`) must be \
                     browse-only (no actionable payload)",
                    entry.id
                );
            }
        }
    }

    // Kept serving backends stay actionable (guard against over-purge).
    for name in KEPT_SERVING_BACKENDS {
        let id: CatalogBackendId = serde_json::from_value(json!(name))
            .unwrap_or_else(|e| panic!("kept feed id `{name}` must parse: {e}"));
        assert!(
            SupportedServingBackend::try_from(&id).is_ok(),
            "R1 kept behavior: `{name}` must remain a supported serving backend"
        );
    }

    let all: Vec<&str> = SupportedServingBackend::ALL
        .iter()
        .map(SupportedServingBackend::as_str)
        .collect();
    for name in REMOVED_BACKENDS {
        assert!(
            !all.contains(&name),
            "L1 R1: SupportedServingBackend::ALL must not contain `{name}`; found {all:?}"
        );
    }
    for name in KEPT_SERVING_BACKENDS {
        assert!(
            all.contains(&name),
            "R1 kept behavior: SupportedServingBackend::ALL must contain `{name}`; found {all:?}"
        );
    }

    // The feed-vocabulary CODE constant is kept (design v3.1) so the vendored
    // feed keeps parsing without new "unrecognized backend" noise.
    for name in REMOVED_BACKENDS
        .iter()
        .chain(KEPT_SERVING_BACKENDS.iter())
        .chain(["comfyui"].iter())
    {
        assert!(
            schema_validate::KNOWN_BACKEND_IDS.contains(name),
            "L1 keeps `schema_validate::KNOWN_BACKEND_IDS` containing `{name}` \
             so the vendored feed parses unchanged; found {:?}",
            schema_validate::KNOWN_BACKEND_IDS
        );
    }

    // The effective catalog (vendored feed + gufo overlay) keeps loading
    // fail-closed-ok through the purge.
    let effective = toolbox_catalog::load_effective_typed_catalog();
    assert!(
        effective.is_ok(),
        "R1 kept behavior: the effective catalog (feed + gufo overlay) must keep \
         parsing after the purge: {effective:?}"
    );
}

// ── Bullets 3 + 4: daemon-served route contracts ────────────────────────────

fn dead_upstream_url() -> String {
    // Bind-then-drop: a port with nothing listening. The routes under test
    // never touch the upstream; if they did, requests would fail loudly.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{address}")
}

/// Minimal daemon harness following `tests/daemon_availability_test.rs`
/// conventions: isolated temp HOME/XDG dirs, empty PATH (no podman), config
/// with manifest disabled and a dead llama-swap port.
struct L1Daemon {
    child: Child,
    url: String,
    log: PathBuf,
    client: reqwest::Client,
}

impl L1Daemon {
    fn start(directory: &Path, benchmarks_database: &Path) -> Self {
        let config_path = directory.join("config.yaml");
        let upstream = dead_upstream_url();
        fs::write(
            &config_path,
            serde_yaml::to_string(&json!({
                "manifest": {
                    "base_url": format!("{upstream}/cloud/v1"),
                    "enabled": false,
                },
                "llama_swap": {
                    "base_url": format!("{upstream}/v1"),
                    "fallback_model": "test-only-model",
                },
                "bonsai": {
                    "enabled": false,
                    "fork_path": directory.join("no-model-server"),
                },
                "benchmarks": { "database_path": benchmarks_database },
                "review": {},
            }))
            .unwrap(),
        )
        .unwrap();
        let socket = directory.join("d.sock");
        let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = reservation.local_addr().unwrap();
        drop(reservation);
        let log = directory.join("daemon.log");
        let log_file = fs::File::create(&log).unwrap();
        let empty_bin = directory.join("empty-bin");
        fs::create_dir_all(&empty_bin).unwrap();
        let child = Command::new(env!("CARGO_BIN_EXE_brainrouter"))
            .args(["serve", "--config"])
            .arg(&config_path)
            .args(["--tcp-addr", &address.to_string(), "--socket"])
            .arg(&socket)
            .current_dir(directory)
            .env_clear()
            .env("HOME", directory)
            .env("XDG_CONFIG_HOME", directory.join("config"))
            .env("XDG_DATA_HOME", directory.join("data"))
            .env("PATH", &empty_bin)
            .env("RUST_LOG", "warn")
            .stdin(Stdio::null())
            .stdout(Stdio::from(log_file.try_clone().unwrap()))
            .stderr(Stdio::from(log_file))
            .spawn()
            .unwrap();
        Self {
            child,
            url: format!("http://{address}"),
            log,
            client: reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap(),
        }
    }

    async fn wait_until_ready(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                panic!(
                    "daemon exited with {status}: {}",
                    fs::read_to_string(&self.log).unwrap()
                );
            }
            if let Ok(response) = self.client.get(format!("{}/health", self.url)).send().await {
                if response.status().is_success()
                    && response.json::<Value>().await.unwrap() == json!({"status": "ok"})
                {
                    return;
                }
            }
            assert!(
                Instant::now() < deadline,
                "daemon did not become healthy: {}",
                fs::read_to_string(&self.log).unwrap()
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    async fn get(&self, path: &str) -> reqwest::Response {
        self.client
            .get(format!("{}{path}", self.url))
            .send()
            .await
            .unwrap_or_else(|e| panic!("GET {path} failed: {e}"))
    }

    /// POST a raw JSON body. No Origin header: loopback non-browser clients
    /// pass the CSRF gate by design (`is_destructive` + no-Origin allowance).
    async fn post(&self, path: &str, body: &Value) -> reqwest::Response {
        self.client
            .post(format!("{}{path}", self.url))
            .json(body)
            .send()
            .await
            .unwrap_or_else(|e| panic!("POST {path} failed: {e}"))
    }
}

impl Drop for L1Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

async fn read_json_body<B>(response: hyper::Response<B>) -> Value
where
    B: hyper::body::Body<Data = Bytes> + Unpin,
    B::Error: std::fmt::Debug,
{
    let (parts, body) = response.into_parts();
    assert_eq!(
        parts.status,
        hyper::StatusCode::SERVICE_UNAVAILABLE,
        "degraded benchmark responses are documented as 503"
    );
    let bytes = body.collect().await.expect("body collects").to_bytes();
    serde_json::from_slice(&bytes).expect("degraded benchmark response is JSON")
}

/// Bullet 3, purge side (L1 Error behavior: "requests to removed routes → 404").
/// Red at base (the routes are registered: start answers 400 on an invalid
/// body, stop/status run the podman path), green once the route blocks are
/// removed and dispatch falls through to the generic 404.
#[tokio::test]
async fn r1_removed_server_mode_routes_answer_404() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut daemon = L1Daemon::start(
        tmp.path(),
        &tmp.path().join("benchmarks.sqlite3"),
    );
    daemon.wait_until_ready().await;

    for name in REMOVED_BACKENDS {
        let status = daemon
            .get(&format!("/api/server-mode/{name}/status"))
            .await;
        assert_eq!(
            status.status(),
            404,
            "L1 R1: GET /api/server-mode/{name}/status must 404 after the purge"
        );

        for (path, body) in [
            (format!("/api/server-mode/{name}/start"), json!([])),
            (format!("/api/server-mode/{name}/stop"), json!({})),
        ] {
            let response = daemon.post(&path, &body).await;
            assert_eq!(
                response.status(),
                404,
                "L1 R1: POST {path} must 404 after the purge (removed route → 404)"
            );
            let json: Value = response
                .json()
                .await
                .expect("route-table 404 answers with the JSON error envelope");
            assert!(
                json.get("error").is_some_and(Value::is_string),
                "L1 R1: POST {path} 404 body must carry the JSON `error` field, got {json}"
            );
        }
    }

    // Backend-specific extra routes of the purged backends, same 404 promise.
    for path in ["/api/server-mode/vllm/cache-paths", "/api/server-mode/r9v/paths"] {
        let response = daemon.post(path, &json!({})).await;
        assert_eq!(
            response.status(),
            404,
            "L1 R1: POST {path} must 404 after the purge"
        );
    }
}

/// Bullet 3, keeper side (Contracts kept: "/api/server-mode/{ds4,gufo}/*
/// unchanged"). Green at base and after L1. Registration is asserted by
/// dispatching an obviously-invalid request body (a JSON array where the
/// handlers require a typed object): a registered route answers 400 from the
/// body parser; a removed route can only 404 at dispatch.
#[tokio::test]
async fn r1_kept_server_mode_routes_stay_registered() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut daemon = L1Daemon::start(
        tmp.path(),
        &tmp.path().join("benchmarks.sqlite3"),
    );
    daemon.wait_until_ready().await;

    for name in ["ds4", "gufo"] {
        let status = daemon
            .get(&format!("/api/server-mode/{name}/status"))
            .await;
        assert_ne!(
            status.status(),
            404,
            "R1 kept behavior: GET /api/server-mode/{name}/status must stay registered"
        );

        let response = daemon
            .post(&format!("/api/server-mode/{name}/start"), &json!([]))
            .await;
        let code = response.status();
        assert!(
            code.is_client_error(),
            "R1 kept behavior: POST /api/server-mode/{name}/start must be registered \
             (an invalid body answers 4xx from the handler); got {code}"
        );
        assert_ne!(
            code, 404,
            "R1 kept behavior: /api/server-mode/{name}/start must not have been removed"
        );
        let json: Value = response
            .json()
            .await
            .expect("start-handler body-parse rejection is a JSON error");
        assert!(
            json.get("error").is_some_and(Value::is_string),
            "R1 kept behavior: {name} start rejection carries the JSON `error` field, got {json}"
        );
    }
}

// ── Bullet 4: R2 benchmarks feature gate ────────────────────────────────────

/// Extract the `key = ...` entries of the [features] table from a Cargo.toml.
/// Deliberately text-level (no TOML crate dependency): the assertions below
/// only need key names and the default list's content.
fn cargo_feature_entries(text: &str) -> Vec<String> {
    let mut entries = Vec::new();
    let mut in_features = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            in_features = trimmed == "[features]";
            continue;
        }
        if in_features && trimmed.contains('=') && !trimmed.starts_with('#') {
            entries.push(trimmed.to_string());
        }
    }
    entries
}

#[test]
fn r2_benchmarks_feature_is_declared_and_non_default() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let text = fs::read_to_string(&manifest).expect("root Cargo.toml is readable");
    let features = cargo_feature_entries(&text);
    assert!(
        !features.is_empty(),
        "R2: Cargo.toml must keep a [features] table (bridges are feature-gated today)"
    );

    let has_benchmarks = features
        .iter()
        .any(|entry| entry.split('=').next().unwrap_or_default().trim() == "benchmarks");
    assert!(
        has_benchmarks,
        "R2: Cargo.toml [features] must declare a `benchmarks` feature; found: {features:?}"
    );

    let default = features
        .iter()
        .find(|entry| entry.split('=').next().unwrap_or_default().trim() == "default")
        .expect("R2: the existing `default` features list must stay declared");
    assert!(
        !default.contains("benchmarks"),
        "R2: `benchmarks` must NOT be part of `default` features (default build \
         excludes the explorer); default = {default}"
    );
}

/// Bullet 4, degradation contract: the explorer route answers with exactly the
/// documented unavailable envelope (`benchmark::unavailable_response`, the same
/// 503 contract R2 pins: "explorer routes return 503-when-off as today's
/// unavailable-storage path"). Built here via a daemon whose benchmark storage
/// cannot be opened, the existing production trigger for that response; the
/// feature-gated default build must serve the same envelope from the same
/// route. Green at base, and it is the contract the L1 feature gate must
/// preserve.
#[tokio::test]
async fn r2_benchmarks_route_serves_documented_degradation_contract() {
    // 1. The documented response itself: 503 + the single-key JSON envelope.
    let probe = benchmark::unavailable_response("acceptance-probe-reason");
    let probe_body = read_json_body(probe).await;
    let probe_error = probe_body
        .get("error")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("unavailable_response body is {{\"error\": string}}, got {probe_body}"));
    assert!(
        probe_error.starts_with("Benchmark explorer unavailable: acceptance-probe-reason."),
        "documented degradation message must lead with the reason: {probe_error}"
    );
    assert!(
        probe_error.contains("Core routing is still available."),
        "documented degradation message must promise core routing stays available: {probe_error}"
    );

    // 2. The route answers that contract, produced by the server-side path.
    let tmp = tempfile::TempDir::new().unwrap();
    let blocker = tmp.path().join("not-a-directory");
    fs::write(&blocker, b"keep this file").unwrap();
    let database = blocker.join("benchmarks.sqlite3");
    let mut daemon = L1Daemon::start(tmp.path(), &database);
    daemon.wait_until_ready().await;

    for path in ["/benchmarks", "/api/benchmarks/runs"] {
        let response = daemon.get(path).await;
        assert_eq!(
            response.status(),
            503,
            "R2: {path} must answer the documented 503 degradation when the \
             explorer is off/unavailable"
        );
        let body: Value = response.json().await.expect("degradation body is JSON");
        let error = body
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("{path} degradation body is {{\"error\": string}}, got {body}"));
        assert!(
            error.starts_with("Benchmark explorer unavailable: "),
            "R2: {path} degradation message must match the unavailable_response \
             envelope (same prefix + reason), got: {error}"
        );
        assert!(
            error.contains("Core routing is still available."),
            "R2: {path} degradation message must match the unavailable_response \
             envelope (core-routing promise), got: {error}"
        );
    }
}
