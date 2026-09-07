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
                                        && req.uri().path() == "/v1/chat/completions"
                                    {
                                        let body = req.collect().await.unwrap().to_bytes();
                                        requests.lock().unwrap().push(
                                            serde_json::from_slice(&body).unwrap(),
                                        );
                                        Response::builder()
                                            .header("content-type", "text/event-stream")
                                            .body(Full::new(Bytes::from_static(
                                                b"data: {\"id\":\"stub\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"routing-ok\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
                                            )))
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
        let config = directory.join("config.yaml");
        fs::write(
            &config,
            serde_yaml::to_string(&json!({
                "manifest": {"base_url": format!("{upstream}/v1"), "enabled": false},
                "llama_swap": {
                    "base_url": format!("{upstream}/v1"),
                    "fallback_model": "test-only-model",
                },
                "bonsai": {
                    "enabled": false,
                    "fork_path": directory.join("no-model-server"),
                },
                "benchmarks": {"database_path": database},
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
        fs::create_dir(&empty_bin).unwrap();
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
            "INSERT INTO schema_migrations(version,name) VALUES(2,'future_schema')",
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
        version, 2,
        "startup must not downgrade an unsupported database"
    );
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
