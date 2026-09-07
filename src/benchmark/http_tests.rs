use super::*;
use crate::benchmark::tests::{example_bundle, test_store};

async fn response_json(response: HttpResponse) -> Value {
    serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap()
}

#[tokio::test]
async fn http_preview_and_ingest_revalidate_without_persisting_invalid_llama_rows() {
    let test = test_store();
    let mut sample = example_bundle();
    sample.performance_metrics = None;
    let invalid_outputs = [
        json!([{"test":"tg128","avg_ts":50}, {"test":"tg128","avg_ts":60}]),
        json!({"test":"tg128","avg_ts":"50"}),
        json!({"test":"tg128","avg_ts":50,"tps":50}),
        json!({"test":"tg128","avg_ts":-1}),
        json!({"test":"tg128","n_gen":256,"avg_ts":50}),
        json!({"test":"tg256","avg_ts":50}),
        json!({"test":"pp512","n_gen":128,"avg_ts":50}),
        json!({"test":"pp512+tg128","avg_ts":50}),
        json!([{"test":"tg128","avg_ts":50}, {"unrecognized":true}]),
        json!({"results":"not an array","generation_tps":50}),
        json!({"test":"tg128","avg_ts":50,"ttft_ms":null}),
        json!({"avg_ts":50,"ttft_ms":1}),
    ];
    for output in invalid_outputs {
        for path in [
            "/api/benchmarks/ingest/llama-bench",
            "/api/benchmarks/validate/llama-bench",
        ] {
            let bytes =
                serde_json::to_vec(&json!({"bundle":sample, "llama_bench":output})).unwrap();
            let result = dispatch(&test.store, "POST", path, None, false, &bytes);
            assert!(
                matches!(result, Err(BenchmarkError::Validation(_))),
                "{path}: {output}"
            );
            assert_eq!(
                test.store
                    .query_runs(&RunQuery::parse(None).unwrap())
                    .unwrap()
                    .total,
                0
            );
        }
    }
    let bytes = serde_json::to_vec(&json!({
        "bundle":sample, "llama_bench":{"test":"tg128","avg_ts":50}
    }))
    .unwrap();
    let preview = dispatch(
        &test.store,
        "POST",
        "/api/benchmarks/validate/llama-bench",
        None,
        false,
        &bytes,
    )
    .unwrap();
    assert_eq!(preview.status(), StatusCode::OK);
    assert_eq!(preview.headers()["cache-control"], "no-store");
    let preview = response_json(preview).await;
    assert_eq!(preview["persisted"], false);
    assert_eq!(
        preview["bundle"]["run"]["raw_result"]["llama_bench"]["avg_ts"],
        50
    );
    let prepared = serde_json::to_vec(&preview["bundle"]).unwrap();
    let response = dispatch(
        &test.store,
        "POST",
        "/api/benchmarks/ingest",
        None,
        false,
        &prepared,
    )
    .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(response_json(response).await["run_id"], sample.run.id);
    let conflict = dispatch(
        &test.store,
        "POST",
        "/api/benchmarks/ingest",
        None,
        false,
        &prepared,
    )
    .unwrap_err();
    assert_eq!(error_response(conflict).status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn response_buffers_keep_admission_until_dropped() {
    let test = test_store();
    let permit = test.store.acquire_worker().unwrap();
    let response = test
        .store
        .execute_blocking(permit, |_, permit| {
            Ok(json_response(StatusCode::OK, &json!({"example":true}))
                .map(|inner| AdmittedBody::new(inner, permit).boxed_unsync()))
        })
        .await
        .unwrap();
    assert_eq!(
        test.store.http_workers.available_permits(),
        MAX_HTTP_WORKERS - 1
    );
    drop(response);
    assert_eq!(
        test.store.http_workers.available_permits(),
        MAX_HTTP_WORKERS
    );
    let response = error_response(BenchmarkError::Busy("busy".into()));
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(response.headers()["retry-after"], "1");
    assert_eq!(response.headers()["cache-control"], "no-store");
}

#[tokio::test]
async fn emitted_response_frames_retain_their_permits_after_body_drop() {
    let test = test_store();
    let mut retained_frames = Vec::new();
    for _ in 0..MAX_HTTP_WORKERS {
        let permit = test.store.acquire_worker().unwrap();
        let mut body = AdmittedBody::new(
            json_response(StatusCode::OK, &"x".repeat(64 * 1024)).into_body(),
            permit,
        );
        while let Some(frame) = body.frame().await {
            let bytes = frame.unwrap().into_data().unwrap();
            assert!(bytes.len() <= 16 * 1024);
            retained_frames.push(bytes);
        }
        drop(body);
    }
    assert_eq!(test.store.http_workers.available_permits(), 0);
    assert!(matches!(
        test.store.run_blocking(|_| Ok(())).await,
        Err(BenchmarkError::Busy(_))
    ));
    retained_frames.clear();
    assert_eq!(
        test.store.http_workers.available_permits(),
        MAX_HTTP_WORKERS
    );
}

#[tokio::test]
async fn backpressured_hyper_connection_keeps_admission() {
    use hyper::{server::conn::http1, service::service_fn};
    use hyper_util::rt::TokioIo;
    use tokio::io::AsyncWriteExt;
    let test = test_store();
    let store = test.store.clone();
    let (mut client, server) = tokio::io::duplex(64);
    let started = Arc::new(tokio::sync::Notify::new());
    let started_signal = Arc::clone(&started);
    let connection = tokio::spawn(async move {
        let service = service_fn(move |_| {
            let response =
                json_response(StatusCode::OK, &"x".repeat(2 * 1024 * 1024)).map(|body| {
                    AdmittedBody::new(body, store.acquire_worker().unwrap()).boxed_unsync()
                });
            started_signal.notify_one();
            async move { Ok::<_, Infallible>(response) }
        });
        http1::Builder::new()
            .serve_connection(TokioIo::new(server), service)
            .await
    });
    client
        .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .unwrap();
    started.notified().await;
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(
        test.store.http_workers.available_permits(),
        MAX_HTTP_WORKERS - 1
    );
    drop(client);
    let _closed = connection.await.unwrap();
    assert_eq!(
        test.store.http_workers.available_permits(),
        MAX_HTTP_WORKERS
    );
}

#[tokio::test]
async fn byte_budgets_report_413_instead_of_partial_success() {
    let test = test_store();
    let sample = example_bundle();
    test.store.ingest(&sample).unwrap();
    let connection = test.store.connect().unwrap();
    connection
        .execute(
            "UPDATE models SET metadata_json=?1 WHERE id=?2",
            params![
                format!("{{\"large\":\"{}\"}}", "x".repeat(MAX_ROW_BYTES)),
                sample.model.id
            ],
        )
        .unwrap();
    let error = test
        .store
        .run_blocking(move |store| store.run_detail(&sample.run.id))
        .await
        .unwrap_err();
    assert_eq!(
        error_response(error).status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );
    let response = json_response(StatusCode::OK, &"x".repeat(MAX_RESPONSE_BYTES));
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let body = response_json(response).await;
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("Nothing was truncated"));
}

#[tokio::test(flavor = "current_thread")]
async fn http_health_responds_while_sqlite_writers_wait_and_admission_is_full() {
    use hyper::{server::conn::http1, service::service_fn};
    use hyper_util::rt::TokioIo;
    let test = test_store();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let store = test.store.clone();
    let server = tokio::spawn(async move {
        let mut connections = tokio::task::JoinSet::new();
        loop {
            tokio::select! {
                incoming = listener.accept() => {
                    let (stream, _) = incoming.unwrap();
                    let store = store.clone();
                    connections.spawn(async move {
                        let service = service_fn(move |req: Request<Incoming>| {
                            let store = store.clone();
                            async move {
                                if req.uri().path() == "/health" {
                                    Ok(json_response(StatusCode::OK, &json!({"status":"ok"})))
                                } else {
                                    handle_request(req, &store).await
                                }
                            }
                        });
                        http1::Builder::new().serve_connection(TokioIo::new(stream), service).await.unwrap();
                    });
                }
                Some(result) = connections.join_next() => { result.unwrap(); }
            }
        }
    });
    let lock = test.store.connect().unwrap();
    lock.execute_batch("BEGIN IMMEDIATE").unwrap();
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap();
    let mut requests = Vec::new();
    for _ in 0..MAX_HTTP_WORKERS {
        let client = client.clone();
        requests.push(tokio::spawn(async move {
            client
                .post(format!("http://{address}/api/benchmarks/validate"))
                .json(&example_bundle())
                .send()
                .await
                .unwrap()
        }));
    }
    tokio::time::timeout(Duration::from_secs(1), async {
        while test.store.http_workers.available_permits() > 0 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    let health = tokio::time::timeout(
        Duration::from_millis(500),
        client.get(format!("http://{address}/health")).send(),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(
        health.json::<Value>().await.unwrap(),
        json!({"status":"ok"})
    );
    let saturated = client
        .get(format!("http://{address}/api/benchmarks/runs"))
        .send()
        .await
        .unwrap();
    assert_eq!(saturated.status(), 503);
    assert_eq!(saturated.headers()["retry-after"], "1");
    assert_eq!(saturated.headers()["cache-control"], "no-store");
    lock.execute_batch("ROLLBACK").unwrap();
    for request in requests {
        let response = request.await.unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(response.json::<Value>().await.unwrap()["persisted"], false);
    }
    assert_eq!(
        test.store
            .query_runs(&RunQuery::parse(None).unwrap())
            .unwrap()
            .total,
        0
    );
    server.abort();
    assert!(server.await.unwrap_err().is_cancelled());
}
