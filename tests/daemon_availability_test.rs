//! Exercise the real daemon with isolated storage and a synthetic SSE upstream.
//! No model servers, user configuration, or external network access are needed.

use brainrouter::{
    benchmark::BenchmarkStore,
    daemon_client::{DaemonClient, DaemonEndpoint},
};
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{body::Incoming, server::conn::http1, service::service_fn, Request, Response};
use hyper_util::rt::TokioIo;
use rusqlite::Connection;
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    fs,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::{net::TcpListener as AsyncTcpListener, task::JoinHandle};

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("br-{}", uuid::Uuid::new_v4().simple()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove isolated daemon test directory");
    }
}

#[cfg(unix)]
fn write_fake_bwrap(path: &Path) {
    use std::os::unix::fs::PermissionsExt;

    fs::write(
        path,
        r#"#!/bin/sh
omp=
python=
work=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --ro-bind)
      if [ "$3" = "/opt/omp" ]; then omp="$2"; fi
      if [ "$3" = "/opt/python" ]; then python="$2"; fi
      shift 3
      ;;
    --bind)
      if [ "$3" = "/work" ]; then work="$2"; fi
      shift 3
      ;;
    --)
      shift
      cd "$work" || exit 126
      if [ "$1" = "/opt/brainrouter" ]; then
        while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do shift; done
        [ "$#" -gt 0 ] && shift
        [ "$1" = "/opt/omp" ] && shift
        exec "$omp" "$@"
      fi
      if [ "$1" = "/opt/python" ]; then
        shift
        exec "$python" "$@"
      fi
      exit 127
      ;;
    *)
      shift
      ;;
  esac
done
exit 127
"#,
    )
    .unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

struct MockUpstream {
    url: String,
    requests: Arc<Mutex<Vec<Value>>>,
    task: JoinHandle<()>,
}

impl MockUpstream {
    async fn start() -> Self {
        let listener = AsyncTcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        let task = tokio::spawn(async move {
            let mut connections = tokio::task::JoinSet::new();
            loop {
                tokio::select! {
                    connection = listener.accept() => {
                        let (stream, _) = connection.unwrap();
                        let requests = Arc::clone(&captured);
                        connections.spawn(async move {
                            let service = service_fn(move |req: Request<Incoming>| {
                                let requests = Arc::clone(&requests);
                                async move {
                                    let response = if req.method() == "POST"
                                        && matches!(req.uri().path(),
                                            "/v1/chat/completions" | "/cloud/v1/chat/completions")
                                    {
                                        let path = req.uri().path().to_string();
                                        let body = req.collect().await.unwrap().to_bytes();
                                        let mut payload: Value = serde_json::from_slice(&body).unwrap();
                                        let model = payload["model"].clone();
                                        let content = if payload["model"].as_str().unwrap().contains("review") {
                                            json!({"status": "approved", "feedback": "synthetic review"}).to_string()
                                        } else {
                                            "routing-ok".to_string()
                                        };
                                        payload["_test_path"] = json!(path);
                                        requests.lock().unwrap().push(payload);
                                        let frames = [
                                            json!({"id": "stub", "model": model, "choices": [{
                                                "index": 0, "delta": {"role": "assistant"}, "finish_reason": null,
                                            }]}),
                                            json!({"id": "stub", "model": model, "choices": [{
                                                "index": 0, "delta": {"content": content}, "finish_reason": null,
                                            }]}),
                                            json!({"id": "stub", "model": model, "choices": [{
                                                "index": 0, "delta": {}, "finish_reason": "stop",
                                            }]}),
                                            json!({"id": "stub", "model": model, "choices": [],
                                                "usage": {"prompt_tokens": 64, "completion_tokens": 8, "total_tokens": 72},
                                            }),
                                        ];
                                        let mut body = frames.iter()
                                            .map(|frame| format!("data: {frame}\n\n"))
                                            .collect::<String>();
                                        body.push_str("data: [DONE]\n\n");
                                        Response::builder()
                                            .header("content-type", "text/event-stream")
                                            .body(Full::new(Bytes::from(body)))
                                            .unwrap()
                                    } else {
                                        // Also rejects CONNECT requests from version-check clients
                                        // configured to use this mock as their external proxy.
                                        Response::builder()
                                            .status(404)
                                            .body(Full::new(Bytes::from_static(b"{}")))
                                            .unwrap()
                                    };
                                    Ok::<_, Infallible>(response)
                                }
                            });
                            http1::Builder::new()
                                .serve_connection(TokioIo::new(stream), service)
                                .await
                        });
                    }
                    Some(result) = connections.join_next() => {
                        result.expect("mock connection task panicked").expect("mock HTTP connection");
                    }
                }
            }
        });
        Self {
            url,
            requests,
            task,
        }
    }
}

impl Drop for MockUpstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

struct TestDaemon {
    child: Child,
    url: String,
    socket: PathBuf,
    log: PathBuf,
    client: reqwest::Client,
}

impl TestDaemon {
    fn start(directory: &Path, database: &Path, upstream: &str) -> Self {
        Self::start_with_cloud(directory, database, upstream, false)
    }

    fn start_with_cloud(
        directory: &Path,
        database: &Path,
        upstream: &str,
        cloud_enabled: bool,
    ) -> Self {
        Self::start_with_review(directory, database, upstream, cloud_enabled, json!({}))
    }

    fn start_with_review(
        directory: &Path,
        database: &Path,
        upstream: &str,
        cloud_enabled: bool,
        review: Value,
    ) -> Self {
        Self::start_with_review_and_lab(directory, database, upstream, cloud_enabled, review, None)
    }

    fn start_with_lab(directory: &Path, database: &Path, upstream: &str, lab: Value) -> Self {
        Self::start_with_review_and_lab(directory, database, upstream, false, json!({}), Some(lab))
    }

    fn start_with_review_and_lab(
        directory: &Path,
        database: &Path,
        upstream: &str,
        cloud_enabled: bool,
        review: Value,
        lab: Option<Value>,
    ) -> Self {
        let config = directory.join("config.yaml");
        let mut benchmarks = json!({"database_path": database});
        if let Some(lab) = lab {
            benchmarks["lab"] = lab;
        }
        fs::write(
            &config,
            serde_yaml::to_string(&json!({
                "manifest": {
                    "base_url": format!("{upstream}/cloud/v1"),
                    "enabled": cloud_enabled,
                },
                "llama_swap": {
                    "base_url": format!("{upstream}/v1"),
                    "fallback_model": "test-only-model",
                },
                "bonsai": {
                    "enabled": false,
                    "fork_path": directory.join("no-model-server"),
                },
                "benchmarks": benchmarks,
                "review": review,
            }))
            .unwrap(),
        )
        .unwrap();
        let socket = directory.join("d.sock");
        let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = reservation.local_addr().unwrap();
        let log = directory.join("daemon.log");
        let log_file = fs::File::create(&log).unwrap();
        let empty_bin = directory.join("empty-bin");
        fs::create_dir_all(&empty_bin).unwrap();
        drop(reservation);
        let child = Command::new(env!("CARGO_BIN_EXE_brainrouter"))
            .args(["serve", "--config"])
            .arg(config)
            .args(["--tcp-addr", &address.to_string(), "--socket"])
            .arg(&socket)
            .current_dir(directory)
            .env_clear()
            .env("HOME", directory)
            .env("XDG_CONFIG_HOME", directory.join("config"))
            .env("XDG_DATA_HOME", directory.join("data"))
            .env("PATH", empty_bin)
            .env("RUST_LOG", "warn")
            .env("HTTP_PROXY", upstream)
            .env("HTTPS_PROXY", upstream)
            .env("ALL_PROXY", upstream)
            .env("NO_PROXY", "127.0.0.1,localhost,::1")
            .stdin(Stdio::null())
            .stdout(Stdio::from(log_file.try_clone().unwrap()))
            .stderr(Stdio::from(log_file))
            .spawn()
            .unwrap();
        Self {
            child,
            url: format!("http://{address}"),
            socket,
            log,
            client: reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap(),
        }
    }

    async fn wait_until_ready(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                panic!(
                    "daemon exited with {status}: {}",
                    fs::read_to_string(&self.log).unwrap()
                );
            }
            if let Ok(response) = self.client.get(format!("{}/health", self.url)).send().await {
                if response.status().is_success() {
                    assert_eq!(
                        response.json::<Value>().await.unwrap(),
                        json!({"status": "ok"})
                    );
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

    async fn get_json(&self, path: &str) -> Value {
        self.client
            .get(format!("{}{path}", self.url))
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .await
            .unwrap()
    }

    async fn post_json(&self, path: &str, body: &Value) -> Value {
        self.client
            .post(format!("{}{path}", self.url))
            .header("Origin", "http://localhost:8080")
            .json(body)
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .await
            .unwrap()
    }

    async fn synthetic_chat(&self, model: &str) {
        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.url))
            .json(&json!({
                "model": model, "stream": true,
                "messages": [{"role": "user", "content": "synthetic request"}],
            }))
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(response.contains("routing-ok"), "{response}");
        assert!(response.contains("[DONE]"), "{response}");
    }

    async fn synthetic_anthropic_chat(&self, model: &str) {
        let response = self
            .client
            .post(format!("{}/v1/messages", self.url))
            .json(&json!({
                "model": model, "stream": true, "max_tokens": 64,
                "messages": [{"role": "user", "content": "synthetic Anthropic request"}],
            }))
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(response.contains("routing-ok"), "{response}");
        assert!(response.contains("message_stop"), "{response}");
    }

    async fn synthetic_review(&self, directory: &Path) {
        let created = self
            .post_json(
                "/review/api/request-async",
                &json!({
                    "taskId": "synthetic-review", "summary": "synthetic review", "cwd": directory,
                }),
            )
            .await;
        let session_id = created["sessionId"].as_str().unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let session = self
                .get_json(&format!("/review/api/sessions/{session_id}"))
                .await;
            if session["status"] == "approved" {
                return;
            }
            assert_eq!(session["status"], "pending", "{session}");
            assert!(
                Instant::now() < deadline,
                "review did not finish: {session}"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    async fn assert_completed_observations(&self, model_key: &str, expected_count: usize) {
        let deadline = Instant::now() + Duration::from_secs(12);
        loop {
            let snapshot = self.get_json("/api/observability/models").await;
            let samples = snapshot["models"]
                .as_array()
                .unwrap()
                .iter()
                .find(|model| model["model_key"] == model_key)
                .and_then(|model| model["recent_measurements"].as_array());
            if let Some(samples) = samples.filter(|samples| samples.len() == expected_count) {
                let mut event_ids = std::collections::BTreeSet::new();
                for sample in samples {
                    assert!(event_ids.insert(sample["event_id"].as_u64().unwrap()));
                    assert!(sample["measured_ttft_ms"].as_f64().unwrap() >= 0.0);
                    assert_eq!(sample["prompt_tokens"], 64);
                    assert_eq!(sample["completion_tokens"], 8);
                }
                return;
            }
            assert!(
                Instant::now() < deadline,
                "completed measurements missing: {snapshot}"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    async fn assert_core_routing(&self, upstream: &MockUpstream) {
        let uds = DaemonClient::new(DaemonEndpoint::Socket(self.socket.clone()));
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), uds.get_json("/health"))
                .await
                .unwrap()
                .unwrap(),
            json!({"status": "ok"})
        );
        let bonsai = self
            .client
            .get(format!("{}/api/bonsai", self.url))
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(bonsai["enabled"], false);
        let routed = self
            .client
            .post(format!("{}/v1/chat/completions", self.url))
            .json(&json!({
                "model": "auto", "stream": true,
                "messages": [{"role": "user", "content": "synthetic test request"}],
            }))
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(routed.contains("routing-ok"), "{routed}");
        assert!(routed.contains("[DONE]"), "{routed}");
        let requests = upstream.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["model"], "test-only-model");
    }

    async fn assert_explorer_unavailable(&self) {
        for (method, path) in [
            ("GET", "/benchmarks"),
            ("GET", "/benchmarks/"),
            ("GET", "/api/benchmarks/runs"),
            ("GET", "/api/benchmarks/runs/example"),
            ("GET", "/api/benchmarks/filters"),
            ("GET", "/api/benchmarks/export?format=csv"),
            ("POST", "/api/benchmarks/ingest"),
            ("POST", "/api/benchmarks/ingest/llama-bench"),
            ("POST", "/api/benchmarks/plan"),
        ] {
            let response = self
                .client
                .request(method.parse().unwrap(), format!("{}{path}", self.url))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 503, "{method} {path}");
            assert_eq!(response.headers()["cache-control"], "no-store");
            let error = response.json::<Value>().await.unwrap();
            assert!(error["error"]
                .as_str()
                .unwrap()
                .contains("Benchmark explorer unavailable"));
        }
        let uds = DaemonClient::new(DaemonEndpoint::Socket(self.socket.clone()));
        assert_eq!(
            uds.request("GET", "/api/benchmarks/filters", None)
                .await
                .unwrap()
                .status,
            503
        );
        let log = fs::read_to_string(&self.log).unwrap();
        assert!(log.contains("Benchmark explorer unavailable"), "{log}");
    }
}

impl Drop for TestDaemon {
    fn drop(&mut self) {
        if self
            .child
            .try_wait()
            .expect("inspect test daemon")
            .is_none()
        {
            self.child.kill().expect("stop test daemon");
        }
        self.child.wait().expect("reap test daemon");
    }
}

#[tokio::test]
async fn corrupt_benchmark_database_does_not_stop_core_routing() {
    let directory = TestDirectory::new();
    let database = directory.0.join("benchmarks.sqlite3");
    fs::write(&database, b"not a SQLite database").unwrap();
    assert!(BenchmarkStore::open(&database).is_err());
    let upstream = MockUpstream::start().await;
    let mut daemon = TestDaemon::start(&directory.0, &database, &upstream.url);
    daemon.wait_until_ready().await;
    daemon.assert_explorer_unavailable().await;
    daemon.assert_core_routing(&upstream).await;
    assert_eq!(fs::read(&database).unwrap(), b"not a SQLite database");
}

#[tokio::test]
async fn forward_benchmark_schema_does_not_stop_core_routing() {
    let directory = TestDirectory::new();
    let database = directory.0.join("benchmarks.sqlite3");
    BenchmarkStore::open(&database).unwrap();
    let connection = Connection::open(&database).unwrap();
    connection
        .execute(
            "INSERT INTO schema_migrations(version,name) VALUES(3,'future_schema')",
            [],
        )
        .unwrap();
    drop(connection);
    assert!(BenchmarkStore::open(&database).is_err());
    let upstream = MockUpstream::start().await;
    let mut daemon = TestDaemon::start(&directory.0, &database, &upstream.url);
    daemon.wait_until_ready().await;
    daemon.assert_explorer_unavailable().await;
    daemon.assert_core_routing(&upstream).await;
    let connection = Connection::open(&database).unwrap();
    let version: i64 = connection
        .query_row("SELECT MAX(version) FROM schema_migrations", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(
        version, 3,
        "startup must not downgrade an unsupported database"
    );
}

#[tokio::test]
async fn native_riddllr_job_routes_grades_persists_and_preserves_sources() {
    let directory = TestDirectory::new();
    let database = directory.0.join("benchmarks.sqlite3");
    let riddllr = directory.0.join("riddllr");
    let prompts = riddllr.join("prompts");
    let solutions = riddllr.join("solutions");
    fs::create_dir_all(&prompts).unwrap();
    fs::create_dir_all(&solutions).unwrap();
    fs::write(
        prompts.join("smoke.txt"),
        "Reply with exactly routing-ok.\n",
    )
    .unwrap();
    fs::write(solutions.join("smoke-solution.txt"), "routing-ok\n").unwrap();
    let original_prompt = fs::read(prompts.join("smoke.txt")).unwrap();
    let original_solution = fs::read(solutions.join("smoke-solution.txt")).unwrap();

    let upstream = MockUpstream::start().await;
    let mut daemon = TestDaemon::start_with_lab(
        &directory.0,
        &database,
        &upstream.url,
        json!({
            "enabled": true,
            "riddllr_root": riddllr,
            "workspace_path": directory.0.join("lab-work"),
            "omp_bin": directory.0.join("missing-omp"),
            "python_bin": "python3",
            "max_job_seconds": 10,
            "riddllr_max_tokens": 64,
        }),
    );
    daemon.wait_until_ready().await;

    let suites = daemon
        .client
        .get(format!("{}/api/benchmarks/lab/suites", daemon.url))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(suites["suites"][0]["cases"][0]["id"], "smoke");

    let response = daemon
        .client
        .post(format!("{}/api/benchmarks/lab/jobs", daemon.url))
        .json(&json!({
            "suite": "riddllr",
            "case_id": "smoke",
            "model": "test-only-model",
            "repetition": 0,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 202);
    let queued = response.json::<Value>().await.unwrap();
    let job_id = queued["id"].as_str().unwrap();

    let deadline = Instant::now() + Duration::from_secs(10);
    let completed = loop {
        let job = daemon
            .client
            .get(format!("{}/api/benchmarks/lab/jobs/{job_id}", daemon.url))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        if !matches!(job["status"].as_str(), Some("queued" | "running")) {
            break job;
        }
        assert!(Instant::now() < deadline, "benchmark job did not complete");
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert_eq!(completed["status"], "succeeded", "{completed}");
    assert_eq!(completed["result"]["passed"], true, "{completed}");
    let run_id = completed["result"]["run_id"].as_str().unwrap();

    let detail = daemon
        .client
        .get(format!(
            "{}/api/benchmarks/runs/{}",
            daemon.url,
            url::form_urlencoded::byte_serialize(run_id.as_bytes()).collect::<String>()
        ))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(detail["run_record"]["status"], "succeeded");
    assert_eq!(detail["quality_results"][0]["metric_name"], "pass@1");
    assert_eq!(detail["quality_results"][0]["metric_value"], 1.0);

    let repeated = daemon
        .client
        .post(format!("{}/api/benchmarks/lab/jobs", daemon.url))
        .json(&json!({
            "suite": "riddllr",
            "case_id": "smoke",
            "model": "test-only-model",
            "repetition": 1,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(repeated.status(), 202);
    let repeated_id = repeated.json::<Value>().await.unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();
    let repeated_deadline = Instant::now() + Duration::from_secs(10);
    let repeated = loop {
        let job = daemon
            .client
            .get(format!(
                "{}/api/benchmarks/lab/jobs/{repeated_id}",
                daemon.url
            ))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        if !matches!(job["status"].as_str(), Some("queued" | "running")) {
            break job;
        }
        assert!(
            Instant::now() < repeated_deadline,
            "repeated benchmark job did not complete"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert_eq!(repeated["status"], "succeeded", "{repeated}");
    let connection = Connection::open(&database).unwrap();
    let mut statement = connection
        .prepare("SELECT repetition,experiment_id FROM runs ORDER BY repetition")
        .unwrap();
    let runs = statement
        .query_map([], |row| {
            Ok((row.get::<_, u64>(0)?, row.get::<_, String>(1)?))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0].0, 0);
    assert_eq!(runs[1].0, 1);
    assert_eq!(runs[0].1, runs[1].1);

    assert_eq!(
        fs::read(prompts.join("smoke.txt")).unwrap(),
        original_prompt
    );
    assert_eq!(
        fs::read(solutions.join("smoke-solution.txt")).unwrap(),
        original_solution
    );
}

#[cfg(unix)]
#[tokio::test]
async fn native_plumebench_job_hides_tests_runs_tools_and_persists_quality() {
    use std::os::unix::fs::PermissionsExt;

    let directory = TestDirectory::new();
    let database = directory.0.join("benchmarks.sqlite3");
    let plumebench = directory.0.join("plumebench");
    let task = plumebench.join("tasks/t1_smoke");
    fs::create_dir_all(task.join("starter")).unwrap();
    fs::create_dir_all(task.join("tests_hidden")).unwrap();
    fs::create_dir_all(task.join("reference")).unwrap();
    fs::write(task.join("task.md"), "Create generated.py.\n").unwrap();
    fs::write(task.join("starter/base.py"), "VALUE = 1\n").unwrap();
    fs::write(
        task.join("tests_hidden/test_generated.py"),
        "def test_generated(): assert True\n",
    )
    .unwrap();
    fs::write(task.join("reference/generated.py"), "GENERATED = True\n").unwrap();

    let fake_omp = directory.0.join("fake-omp");
    fs::write(
        &fake_omp,
        "#!/bin/sh\nif [ \"$1\" = \"--help\" ]; then printf '%s\\n' '--model --mode --max-time --thinking --auto-approve --no-session --no-extensions --no-skills --no-rules --cwd'; exit 0; fi\nprintf 'GENERATED = True\\n' > generated.py\nprintf '%s\\n' '{\"type\":\"session\",\"id\":\"fake\"}' '{\"type\":\"turn_start\"}' '{\"type\":\"turn_end\",\"stopReason\":\"stop\",\"usage\":{\"input\":5,\"output\":7}}'\n",
    )
    .unwrap();
    fs::set_permissions(&fake_omp, fs::Permissions::from_mode(0o755)).unwrap();
    let fake_bwrap = directory.0.join("fake-bwrap");
    write_fake_bwrap(&fake_bwrap);
    let fake_python = directory.0.join("fake-python");
    fs::write(
        &fake_python,
        "#!/bin/sh\nprintf '%s\\n' '2 passed in 0.01s'\n",
    )
    .unwrap();
    fs::set_permissions(&fake_python, fs::Permissions::from_mode(0o755)).unwrap();

    let upstream = MockUpstream::start().await;
    let mut daemon = TestDaemon::start_with_lab(
        &directory.0,
        &database,
        &upstream.url,
        json!({
            "enabled": true,
            "plumebench_root": plumebench,
            "workspace_path": directory.0.join("lab-work"),
            "omp_bin": fake_omp,
            "plumebench_sandbox_bin": fake_bwrap,
            "python_bin": fake_python,
            "max_job_seconds": 10,
            "plumebench_max_turns": 3,
            "plumebench_thinking": "minimal",
        }),
    );
    daemon.wait_until_ready().await;

    let response = daemon
        .client
        .post(format!("{}/api/benchmarks/lab/jobs", daemon.url))
        .json(&json!({
            "suite": "plumebench",
            "case_id": "t1_smoke",
            "model": "test-only-model",
            "repetition": 0,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 202);
    let queued = response.json::<Value>().await.unwrap();
    let job_id = queued["id"].as_str().unwrap();

    let deadline = Instant::now() + Duration::from_secs(10);
    let completed = loop {
        let job = daemon
            .client
            .get(format!("{}/api/benchmarks/lab/jobs/{job_id}", daemon.url))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        if !matches!(job["status"].as_str(), Some("queued" | "running")) {
            break job;
        }
        assert!(Instant::now() < deadline, "benchmark job did not complete");
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert_eq!(completed["status"], "succeeded", "{completed}");
    assert_eq!(completed["result"]["passed"], true, "{completed}");
    let run_id = completed["result"]["run_id"].as_str().unwrap();

    let detail = daemon
        .client
        .get(format!(
            "{}/api/benchmarks/runs/{}",
            daemon.url,
            url::form_urlencoded::byte_serialize(run_id.as_bytes()).collect::<String>()
        ))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(detail["run_record"]["status"], "succeeded");
    assert_eq!(detail["quality_results"][0]["metric_name"], "pass@1");
    assert_eq!(detail["quality_results"][0]["tests_passed"], 2);
    assert_eq!(detail["quality_results"][0]["tests_total"], 2);
    assert_eq!(detail["quality_results"][0]["generated_tokens"], 7);
    assert!(!task.join("starter/generated.py").exists());
    assert!(!task.join("starter/tests_hidden").exists());
}

#[cfg(unix)]
#[tokio::test]
async fn native_plumebench_cancellation_terminates_the_active_process_group() {
    use std::os::unix::fs::PermissionsExt;

    let directory = TestDirectory::new();
    let database = directory.0.join("benchmarks.sqlite3");
    let plumebench = directory.0.join("plumebench");
    let task = plumebench.join("tasks/t1_cancel");
    fs::create_dir_all(task.join("starter")).unwrap();
    fs::create_dir_all(task.join("tests_hidden")).unwrap();
    fs::create_dir_all(task.join("reference")).unwrap();
    fs::write(task.join("task.md"), "Wait until cancelled.\n").unwrap();
    fs::write(task.join("starter/base.py"), "VALUE = 1\n").unwrap();
    fs::write(
        task.join("tests_hidden/test_never.py"),
        "def test_never(): assert False\n",
    )
    .unwrap();
    fs::write(task.join("reference/base.py"), "VALUE = 1\n").unwrap();

    let fake_omp = directory.0.join("slow-omp");
    fs::write(
        &fake_omp,
        "#!/bin/sh\nif [ \"$1\" = \"--help\" ]; then printf '%s\\n' '--model --mode --max-time --thinking --auto-approve --no-session --no-extensions --no-skills --no-rules --cwd'; exit 0; fi\ntrap 'exit 0' TERM INT\n/bin/sh -c 'trap \"\" TERM INT; while :; do /bin/sleep 1; done' &\nwait\n",
    )
    .unwrap();
    fs::set_permissions(&fake_omp, fs::Permissions::from_mode(0o755)).unwrap();
    let fake_bwrap = directory.0.join("fake-bwrap");
    write_fake_bwrap(&fake_bwrap);
    let fake_python = directory.0.join("fake-python");
    fs::write(&fake_python, "#!/bin/sh\nexit 99\n").unwrap();
    fs::set_permissions(&fake_python, fs::Permissions::from_mode(0o755)).unwrap();

    let upstream = MockUpstream::start().await;
    let mut daemon = TestDaemon::start_with_lab(
        &directory.0,
        &database,
        &upstream.url,
        json!({
            "enabled": true,
            "plumebench_root": plumebench,
            "workspace_path": directory.0.join("lab-work"),
            "omp_bin": fake_omp,
            "plumebench_sandbox_bin": fake_bwrap,
            "python_bin": fake_python,
            "max_job_seconds": 30,
            "plumebench_max_turns": 3,
            "plumebench_thinking": "minimal",
        }),
    );
    daemon.wait_until_ready().await;

    let queued = daemon
        .client
        .post(format!("{}/api/benchmarks/lab/jobs", daemon.url))
        .json(&json!({
            "suite": "plumebench",
            "case_id": "t1_cancel",
            "model": "test-only-model",
            "repetition": 0,
        }))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();
    let job_id = queued["id"].as_str().unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let job = daemon
            .client
            .get(format!("{}/api/benchmarks/lab/jobs/{job_id}", daemon.url))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        if job["status"] == "running" {
            break;
        }
        assert!(Instant::now() < deadline, "benchmark job never started");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let cancel_started = Instant::now();
    let response = daemon
        .client
        .post(format!(
            "{}/api/benchmarks/lab/jobs/{job_id}/cancel",
            daemon.url
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 202);
    let completed = loop {
        let job = daemon
            .client
            .get(format!("{}/api/benchmarks/lab/jobs/{job_id}", daemon.url))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        if !matches!(job["status"].as_str(), Some("queued" | "running")) {
            break job;
        }
        assert!(
            Instant::now() < deadline,
            "cancelled benchmark job did not stop"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert_eq!(completed["status"], "cancelled", "{completed}");
    assert!(
        cancel_started.elapsed() < Duration::from_secs(5),
        "process group cancellation exceeded the grace window"
    );
    assert_eq!(completed["result"]["passed"], Value::Null);
}

#[tokio::test]
async fn uncreatable_benchmark_path_does_not_stop_core_routing() {
    let directory = TestDirectory::new();
    let parent_file = directory.0.join("not-a-directory");
    fs::write(&parent_file, b"keep this file").unwrap();
    let database = parent_file.join("benchmarks.sqlite3");
    assert!(BenchmarkStore::open(&database).is_err());
    let upstream = MockUpstream::start().await;
    let mut daemon = TestDaemon::start(&directory.0, &database, &upstream.url);
    daemon.wait_until_ready().await;
    daemon.assert_explorer_unavailable().await;
    daemon.assert_core_routing(&upstream).await;
    assert_eq!(fs::read(parent_file).unwrap(), b"keep this file");
}

#[tokio::test]
async fn remapped_loopback_origins_and_referers_can_mutate_but_untrusted_origins_cannot() {
    let directory = TestDirectory::new();
    let database = directory.0.join("benchmarks.sqlite3");
    let upstream = MockUpstream::start().await;
    let mut daemon = TestDaemon::start(&directory.0, &database, &upstream.url);
    daemon.wait_until_ready().await;
    let origin_port = if daemon.url.ends_with(":8080") {
        8082
    } else {
        8080
    };
    for header in ["Origin", "Referer"] {
        for host in ["localhost", "127.0.0.1", "[::1]"] {
            let value = format!("http://{host}:{origin_port}");
            let value = if header == "Referer" {
                format!("{value}/dashboard")
            } else {
                value
            };
            let response = daemon
                .client
                .post(format!("{}/api/nudge", daemon.url))
                .header(header, &value)
                .json(&json!({"enabled": true}))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200, "{header}: {value}");
            assert_eq!(response.json::<Value>().await.unwrap()["enabled"], true);
        }
        for value in [
            "null",
            "https://localhost:8080",
            "http://example.com:8080",
            "http://localhost.example.com:8080",
            "http://192.0.2.1:8080",
            "http://[2001:db8::1]:8080",
            "http://localhost@evil.example:8080",
            "http://evil.example@localhost:8080",
        ] {
            let response = daemon
                .client
                .post(format!("{}/api/nudge", daemon.url))
                .header(header, value)
                .json(&json!({"enabled": false}))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 403, "{header}: {value}");
        }
    }
    let nudge = daemon
        .client
        .get(format!("{}/api/nudge", daemon.url))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(
        nudge["enabled"], true,
        "rejected mutations must not change state"
    );
    let page = daemon
        .client
        .get(format!("{}/api/benchmarks/runs", daemon.url))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(page["total"], 0, "healthy benchmark storage still works");
}

#[tokio::test]
async fn profile_and_model_page_remain_available_without_benchmark_storage() {
    let directory = TestDirectory::new();
    let database = directory.0.join("benchmarks.sqlite3");
    fs::write(&database, b"not a SQLite database").unwrap();
    let upstream = MockUpstream::start().await;
    let mut daemon = TestDaemon::start(&directory.0, &database, &upstream.url);
    daemon.wait_until_ready().await;

    for path in ["/dashboard", "/models", "/api/routing-profile"] {
        let response = daemon
            .client
            .get(format!("{}{path}", daemon.url))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200, "{path}");
    }
    let catalog = daemon.get_json("/api/routing-models").await;
    assert_eq!(catalog["cloud_enabled"], false);
    assert!(catalog["cloud"]["models"].as_array().unwrap().is_empty());
    let models = daemon.get_json("/api/observability/models").await;
    assert!(models["models"].is_array());
    let baseline = daemon
        .get_json("/api/observability/baseline?model_key=test-only-model")
        .await;
    assert_eq!(baseline["status"], "not_selected");
    let reference = daemon
        .client
        .get(format!(
            "{}/api/observability/reference?run_id=example",
            daemon.url,
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(reference.status(), 503);
    let settings = daemon.get_json("/api/observability/settings").await;
    let clear = json!({
        "revision": settings["revision"],
        "model_key": "test-only-model",
        "run_id": null,
        "expected_experiment_hash": null,
        "note": "Synthetic test: clear reference without a database",
    });
    let denied = daemon
        .client
        .post(format!("{}/api/observability/baseline", daemon.url))
        .header("Origin", "null")
        .json(&clear)
        .send()
        .await
        .unwrap();
    assert_eq!(denied.status(), 403);
    daemon
        .post_json("/api/observability/baseline", &clear)
        .await;
    assert!(upstream.requests.lock().unwrap().is_empty());
    daemon.assert_explorer_unavailable().await;
    daemon.assert_core_routing(&upstream).await;
}

#[tokio::test]
async fn benchmark_previews_check_csrf_before_storage_availability() {
    let directory = TestDirectory::new();
    let database = directory.0.join("benchmarks.sqlite3");
    fs::write(&database, b"not a SQLite database").unwrap();
    let upstream = MockUpstream::start().await;
    let mut daemon = TestDaemon::start(&directory.0, &database, &upstream.url);
    daemon.wait_until_ready().await;

    for path in [
        "/api/benchmarks/prepare",
        "/api/benchmarks/validate",
        "/api/benchmarks/validate/llama-bench",
    ] {
        for origin in ["null", "http://example.com:8080"] {
            let response = daemon
                .client
                .post(format!("{}{path}", daemon.url))
                .header("Origin", origin)
                .json(&json!({}))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 403, "{path}, Origin: {origin}");
        }
        let response = daemon
            .client
            .post(format!("{}{path}", daemon.url))
            .header("Origin", "http://localhost:8080")
            .json(&json!({}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 503, "{path}");
    }
    assert!(upstream.requests.lock().unwrap().is_empty());
    daemon.assert_core_routing(&upstream).await;
}

#[tokio::test]
async fn hybrid_roles_are_independent_and_persist_without_benchmark_storage() {
    let directory = TestDirectory::new();
    let database = directory.0.join("benchmarks.sqlite3");
    fs::write(&database, b"not a SQLite database").unwrap();
    let upstream = MockUpstream::start().await;
    let mut daemon = TestDaemon::start_with_cloud(&directory.0, &database, &upstream.url, true);
    daemon.wait_until_ready().await;
    let mut profile = json!({
        "preset": "local_main_cloud_review",
        "main": {"backend": "local", "model": "main-local-test"},
        "reviewer": {"backend": "cloud", "model": "review-cloud-test"},
        "subagent_model": "subagent-local-test",
    });
    let saved = daemon.post_json("/api/routing-profile", &profile).await;
    assert_eq!(saved["profile"], profile);
    daemon.synthetic_chat("auto").await;
    daemon.synthetic_anthropic_chat("auto").await;
    daemon.synthetic_chat("subs").await;
    daemon.synthetic_chat("brainrouter/pinned-test-model").await;
    daemon.synthetic_review(&directory.0).await;
    daemon
        .assert_completed_observations("main-local-test", 2)
        .await;

    profile["preset"] = json!("cloud_main_local_review");
    profile["main"] = json!({"backend": "cloud", "model": "main-cloud-test"});
    profile["reviewer"] = json!({"backend": "local", "model": "review-local-test"});
    let saved = daemon.post_json("/api/routing-profile", &profile).await;
    assert_eq!(saved["profile"], profile);
    daemon.synthetic_chat("auto").await;
    daemon.synthetic_review(&directory.0).await;
    {
        let captured = upstream.requests.lock().unwrap();
        let choices: Vec<_> = captured
            .iter()
            .map(|request| {
                (
                    request["_test_path"].as_str().unwrap(),
                    request["model"].as_str().unwrap(),
                )
            })
            .collect();
        assert_eq!(
            choices,
            [
                ("/v1/chat/completions", "main-local-test"),
                ("/v1/chat/completions", "main-local-test"),
                ("/v1/chat/completions", "subagent-local-test"),
                ("/v1/chat/completions", "pinned-test-model"),
                ("/cloud/v1/chat/completions", "review-cloud-test"),
                ("/cloud/v1/chat/completions", "main-cloud-test"),
                ("/v1/chat/completions", "review-local-test"),
            ]
        );
    }
    daemon.assert_explorer_unavailable().await;
    drop(daemon);

    let mut restarted = TestDaemon::start_with_cloud(&directory.0, &database, &upstream.url, true);
    restarted.wait_until_ready().await;
    let restored = restarted.get_json("/api/routing-profile").await;
    assert_eq!(restored["profile"], profile);
    assert_eq!(restored["cloud_enabled"], true);
    restarted.synthetic_chat("auto").await;
    restarted.synthetic_anthropic_chat("auto").await;
    assert_eq!(
        upstream.requests.lock().unwrap().last().unwrap()["model"],
        "main-cloud-test"
    );
}

#[tokio::test]
async fn synthetic_example_can_be_prepared_validated_and_imported_without_inference() {
    let directory = TestDirectory::new();
    let database = directory.0.join("benchmarks.sqlite3");
    let upstream = MockUpstream::start().await;
    let mut daemon = TestDaemon::start(&directory.0, &database, &upstream.url);
    daemon.wait_until_ready().await;
    let profile_before = daemon.get_json("/api/routing-profile").await;
    let template = daemon
        .get_json("/api/benchmarks/examples/template.json")
        .await;
    let matrix = daemon
        .get_json("/api/benchmarks/examples/matrix.json")
        .await;
    let llama_bench = daemon
        .get_json("/api/benchmarks/examples/llama-bench.json")
        .await;
    let plan = daemon.post_json("/api/benchmarks/plan", &matrix).await;
    let experiment = plan["experiments"]
        .as_array()
        .unwrap()
        .first()
        .unwrap()
        .clone();
    let now = chrono::Utc::now().to_rfc3339();
    let preview = daemon
        .post_json(
            "/api/benchmarks/prepare",
            &json!({
                "template": template,
                "experiment": experiment,
                "repetition": 0,
                "status": "succeeded",
                "exact_command": "synthetic example; no process executed",
                "started_at": now,
                "ended_at": now,
                "llama_bench": llama_bench,
            }),
        )
        .await;
    assert_eq!(preview["valid"], true);
    assert_eq!(preview["persisted"], false);
    let bundle = &preview["bundle"];
    assert_eq!(bundle["performance_metrics"]["generation_tps"], 50.0);
    assert_eq!(bundle["experiment"]["id"], plan["experiments"][0]["id"]);
    let validation = daemon.post_json("/api/benchmarks/validate", bundle).await;
    assert_eq!(validation["valid"], true);
    assert_eq!(validation["persisted"], false);
    assert_eq!(daemon.get_json("/api/benchmarks/runs").await["total"], 0);

    let imported = daemon.post_json("/api/benchmarks/ingest", bundle).await;
    assert_eq!(imported["run_id"], bundle["run"]["id"]);
    let page = daemon.get_json("/api/benchmarks/runs").await;
    assert_eq!(page["total"], 1);
    assert_eq!(page["items"][0]["run_id"], imported["run_id"]);
    let run_id = imported["run_id"].as_str().unwrap();
    let reference = daemon
        .client
        .get(format!("{}/api/observability/reference", daemon.url))
        .query(&[("run_id", run_id)])
        .send()
        .await
        .unwrap();
    assert_eq!(reference.status(), 200);
    let profile_after = daemon.get_json("/api/routing-profile").await;
    assert_eq!(profile_after["profile"], profile_before["profile"]);
    assert!(upstream.requests.lock().unwrap().is_empty());
    daemon.assert_core_routing(&upstream).await;
}

#[tokio::test]
async fn legacy_auto_yaml_survives_startup_but_new_auto_model_writes_are_rejected() {
    let directory = TestDirectory::new();
    let database = directory.0.join("benchmarks.sqlite3");
    let upstream = MockUpstream::start().await;
    let legacy_review = json!({
        "forced_mode":"auto","forced_model":"my-model","max_iterations":5,
    });
    let mut daemon = TestDaemon::start_with_review(
        &directory.0,
        &database,
        &upstream.url,
        false,
        legacy_review.clone(),
    );
    daemon.wait_until_ready().await;
    daemon.assert_core_routing(&upstream).await;
    let before = daemon.get_json("/api/routing-profile").await;
    assert_eq!(before["profile"]["reviewer"], json!({"backend":"auto"}));
    let review = daemon.get_json("/api/review-config").await;
    assert_eq!(review["forced_mode"], "auto");
    assert!(review["forced_model"].is_null());
    let config_path = directory.0.join("config.yaml");
    let original_yaml = fs::read(&config_path).unwrap();
    let mut invalid_profile = before["profile"].clone();
    invalid_profile["reviewer"] = json!({"backend":"auto","model":"new-model"});
    for (path, body) in [
        (
            "/api/review-config",
            serde_json::to_vec(&legacy_review).unwrap(),
        ),
        (
            "/api/routing-profile",
            serde_json::to_vec(&invalid_profile).unwrap(),
        ),
        ("/api/config", original_yaml.clone()),
    ] {
        let response = daemon
            .client
            .post(format!("{}{path}", daemon.url))
            .header("Origin", "http://localhost:8080")
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400, "{path}");
        let error = response.json::<Value>().await.unwrap();
        assert!(
            error["error"]
                .as_str()
                .unwrap()
                .contains("auto cannot specify a model"),
            "{error}"
        );
    }
    assert_eq!(
        daemon.get_json("/api/routing-profile").await["profile"],
        before["profile"]
    );
    assert_eq!(fs::read(&config_path).unwrap(), original_yaml);
    let log = fs::read_to_string(&daemon.log).unwrap();
    assert_eq!(
        log.matches("Ignoring legacy forced_model").count(),
        1,
        "{log}"
    );
    assert!(log.contains(config_path.to_str().unwrap()), "{log}");
    assert!(log.contains("remove forced_model"), "{log}");
}

#[tokio::test]
async fn legacy_auto_saved_state_survives_startup_and_is_migrated_once() {
    let directory = TestDirectory::new();
    let database = directory.0.join("benchmarks.sqlite3");
    let config_directory = directory.0.join("config/brainrouter");
    fs::create_dir_all(&config_directory).unwrap();
    let legacy_path = config_directory.join("review_state.json");
    let legacy = r#"{"forced_mode":"auto","forced_model":"leftover-reviewer","max_iterations":9}"#;
    fs::write(&legacy_path, legacy).unwrap();
    let upstream = MockUpstream::start().await;
    let mut daemon = TestDaemon::start(&directory.0, &database, &upstream.url);
    daemon.wait_until_ready().await;
    daemon.assert_core_routing(&upstream).await;
    let profile = daemon.get_json("/api/routing-profile").await["profile"].clone();
    assert_eq!(profile["reviewer"], json!({"backend":"auto"}));
    assert_eq!(
        daemon.get_json("/api/review-config").await["max_iterations"],
        5
    );
    let state_path = config_directory.join("routing_state.json");
    let persisted: Value = serde_json::from_slice(&fs::read(state_path).unwrap()).unwrap();
    assert_eq!(persisted, profile);
    assert_eq!(fs::read_to_string(&legacy_path).unwrap(), legacy);
    let log = fs::read_to_string(&daemon.log).unwrap();
    assert_eq!(
        log.matches("Ignoring legacy forced_model").count(),
        1,
        "{log}"
    );
    assert!(log.contains(legacy_path.to_str().unwrap()), "{log}");
    drop(daemon);
    fs::write(&legacy_path, "invalid legacy state; already migrated").unwrap();
    let mut restarted = TestDaemon::start(&directory.0, &database, &upstream.url);
    restarted.wait_until_ready().await;
    assert_eq!(
        restarted.get_json("/api/routing-profile").await["profile"],
        profile
    );
    upstream.requests.lock().unwrap().clear();
    restarted.assert_core_routing(&upstream).await;
    let log = fs::read_to_string(&restarted.log).unwrap();
    assert!(!log.contains("Ignoring legacy forced_model"), "{log}");
}
