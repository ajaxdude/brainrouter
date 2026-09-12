//! HTTP admission, blocking work, and response buffers share the same budget.

use super::*;
use hyper::body::{Body, Frame, SizeHint};
use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};
use tokio::sync::OwnedSemaphorePermit;

type HttpResponse = Response<UnsyncBoxBody<Bytes, anyhow::Error>>;

impl BenchmarkStore {
    fn acquire_worker(&self) -> BenchmarkResult<OwnedSemaphorePermit> {
        Arc::clone(&self.http_workers).try_acquire_owned().map_err(|_| {
            BenchmarkError::Busy(format!(
                "all {MAX_HTTP_WORKERS} benchmark workers are occupied; retry later (core routing remains available)"
            ))
        })
    }

    async fn execute_blocking<T, F>(
        &self,
        permit: OwnedSemaphorePermit,
        work: F,
    ) -> BenchmarkResult<T>
    where
        T: Send + 'static,
        F: FnOnce(&BenchmarkStore, OwnedSemaphorePermit) -> BenchmarkResult<T> + Send + 'static,
    {
        let mut store = self.clone();
        store.http_limits = true;
        tokio::task::spawn_blocking(move || work(&store, permit))
            .await
            .map_err(|error| {
                BenchmarkError::Io(std::io::Error::other(format!(
                    "benchmark worker failed: {error}"
                )))
            })?
    }

    /// Share bounded HTTP worker capacity with other asynchronous consumers.
    /// Cancellation stops waiting, not SQLite: the permit is owned by the actual
    /// blocking task until it exits. Reads use the documented HTTP resource limits.
    pub async fn run_blocking<T, F>(&self, work: F) -> BenchmarkResult<T>
    where
        T: Send + 'static,
        F: FnOnce(&BenchmarkStore) -> BenchmarkResult<T> + Send + 'static,
    {
        let permit = self.acquire_worker()?;
        self.execute_blocking(permit, move |store, permit| {
            let result = work(store);
            drop(permit);
            result
        })
        .await
    }

    /// Run an internal lifecycle operation without the fail-fast HTTP admission
    /// pool. Native benchmark execution is already serialized separately, and
    /// terminal job persistence must not be stranded by unrelated explorer
    /// requests occupying the two HTTP workers.
    pub async fn run_critical<T, F>(&self, work: F) -> BenchmarkResult<T>
    where
        T: Send + 'static,
        F: FnOnce(&BenchmarkStore) -> BenchmarkResult<T> + Send + 'static,
    {
        let store = self.clone();
        tokio::task::spawn_blocking(move || work(&store))
            .await
            .map_err(|error| {
                BenchmarkError::Io(std::io::Error::other(format!(
                    "critical benchmark worker failed: {error}"
                )))
            })?
    }
}

struct AdmittedBody {
    inner: UnsyncBoxBody<Bytes, anyhow::Error>,
    pending: Bytes,
    permit: Arc<OwnedSemaphorePermit>,
}

struct AdmittedBytes {
    bytes: Bytes,
    _permit: Arc<OwnedSemaphorePermit>,
}

impl AsRef<[u8]> for AdmittedBytes {
    fn as_ref(&self) -> &[u8] {
        self.bytes.as_ref()
    }
}

impl AdmittedBody {
    fn new(inner: UnsyncBoxBody<Bytes, anyhow::Error>, permit: OwnedSemaphorePermit) -> Self {
        Self {
            inner,
            pending: Bytes::new(),
            permit: Arc::new(permit),
        }
    }

    fn next_chunk(&mut self) -> Poll<Option<Result<Frame<Bytes>, anyhow::Error>>> {
        Poll::Ready(Some(Ok(Frame::data(
            self.pending.split_to(self.pending.len().min(16 * 1024)),
        ))))
    }
}

impl Body for AdmittedBody {
    type Data = Bytes;
    type Error = anyhow::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        if !self.pending.is_empty() {
            return self.next_chunk();
        }
        match Pin::new(&mut self.inner).poll_frame(cx) {
            Poll::Ready(Some(Ok(frame))) => match frame.into_data() {
                Ok(bytes) => {
                    // Hyper may drop a completed body before flushing its Bytes.
                    // The allocation itself owns admission; sliced/cloned frames
                    // retain it too. Small frames also bound transport-side copies.
                    self.pending = Bytes::from_owner(AdmittedBytes {
                        bytes,
                        _permit: Arc::clone(&self.permit),
                    });
                    self.next_chunk()
                }
                Err(frame) => Poll::Ready(Some(Ok(frame))),
            },
            other => other,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.pending.is_empty() && self.inner.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        let mut hint = self.inner.size_hint();
        let pending = self.pending.len() as u64;
        if let Some(upper) = hint.upper() {
            hint.set_upper(upper + pending);
        }
        hint.set_lower(hint.lower() + pending);
        hint
    }
}

pub async fn handle_request(
    req: Request<Incoming>,
    store: &BenchmarkStore,
) -> Result<HttpResponse, Infallible> {
    let method = req.method().as_str().to_owned();
    let path = req.uri().path().to_owned();
    if method == "GET" && matches!(path.as_str(), "/benchmarks" | "/benchmarks/") {
        return Ok(html_response(EXPLORER_HTML));
    }
    let permit = match store.acquire_worker() {
        Ok(permit) => permit,
        Err(error) => return Ok(error_response(error)),
    };
    if req.uri().to_string().len() > 8192 {
        return Ok(error_response(BenchmarkError::Limit(
            "benchmark URI exceeds 8192 bytes".into(),
        )));
    }
    let query = req.uri().query().map(str::to_owned);
    let is_yaml = req
        .headers()
        .get(hyper::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.contains("yaml"));
    let bytes = if method == "POST" {
        if req
            .headers()
            .get(hyper::header::CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<usize>().ok())
            .is_some_and(|length| length > MAX_INGEST_BYTES)
        {
            return Ok(body_error_response(BodyReadError::TooLarge, "benchmark"));
        }
        match tokio::time::timeout(Duration::from_secs(15), collect_body(req)).await {
            Ok(Ok(bytes)) => bytes,
            Ok(Err(error)) => return Ok(body_error_response(error, "benchmark")),
            Err(_) => {
                return Ok(json_response(
                    StatusCode::REQUEST_TIMEOUT,
                    &json!({
                        "error": "benchmark request body timed out after 15 seconds; retry the upload"
                    }),
                ))
            }
        }
    } else {
        Bytes::new()
    };
    let response = store
        .execute_blocking(permit, move |store, permit| {
            let response = match dispatch(store, &method, &path, query.as_deref(), is_yaml, &bytes)
            {
                Ok(response) => response,
                Err(error) => error_response(error),
            };
            // Slow readers retain their bounded response allocation and its permit.
            Ok(response.map(|inner| AdmittedBody::new(inner, permit).boxed_unsync()))
        })
        .await;
    Ok(response.unwrap_or_else(error_response))
}

fn decode_json<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> BenchmarkResult<T> {
    serde_json::from_slice(bytes)
        .map_err(|error| BenchmarkError::Validation(format!("invalid benchmark payload: {error}")))
}

fn dispatch(
    store: &BenchmarkStore,
    method: &str,
    path: &str,
    query: Option<&str>,
    is_yaml: bool,
    bytes: &[u8],
) -> BenchmarkResult<HttpResponse> {
    match (method, path) {
        ("GET", "/api/benchmarks/filters") => {
            Ok(json_response(StatusCode::OK, &store.filter_options()?))
        }
        ("GET", "/api/benchmarks/runs") => Ok(json_response(
            StatusCode::OK,
            &store.query_runs(&RunQuery::parse(query)?)?,
        )),
        ("GET", "/api/benchmarks/export") => {
            let (query, format) = parse_export_query(query)?;
            Ok(download_response(
                &format,
                store.export_runs(query, &format)?,
            ))
        }
        ("GET", path) if path.starts_with("/api/benchmarks/runs/") => {
            let encoded = path.trim_start_matches("/api/benchmarks/runs/");
            // A path '+' is literal; form-urlencoding would otherwise turn it into a space.
            let encoded = encoded.replace('+', "%2B").replace('&', "%26");
            let run_id = url::form_urlencoded::parse(format!("id={encoded}").as_bytes())
                .next()
                .map(|(_, value)| value.into_owned())
                .ok_or_else(|| BenchmarkError::Validation("invalid run id".into()))?;
            Ok(json_response(StatusCode::OK, &store.run_detail(&run_id)?))
        }
        ("GET", path) if path.starts_with("/api/benchmarks/examples/") => {
            let (body, content_type, name) = imports::example(path)?;
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header("content-type", content_type)
                .header(
                    "content-disposition",
                    format!("attachment; filename=\"{name}\""),
                )
                .header("cache-control", "no-store")
                .body(
                    Full::new(Bytes::from(body))
                        .map_err(|error: Infallible| match error {})
                        .boxed_unsync(),
                )
                .expect("benchmark example response"))
        }
        ("POST", "/api/benchmarks/plan") => {
            let matrix: ExperimentMatrix = if is_yaml {
                serde_yaml::from_slice(bytes).map_err(|error| {
                    BenchmarkError::Validation(format!("invalid experiment matrix YAML: {error}"))
                })?
            } else {
                decode_json(bytes)?
            };
            Ok(json_response(StatusCode::OK, &matrix.expand()?))
        }
        ("POST", "/api/benchmarks/prepare") => {
            let bundle = store.prepare_import(decode_json(bytes)?)?;
            preview_response(store, bundle)
        }
        ("POST", "/api/benchmarks/ingest")
        | ("POST", "/api/benchmarks/validate")
        | ("POST", "/api/benchmarks/ingest/llama-bench")
        | ("POST", "/api/benchmarks/validate/llama-bench") => {
            let bundle = if path.ends_with("/llama-bench") {
                let mut ingest: LlamaBenchIngest = decode_json(bytes)?;
                apply_llama_bench_metrics(&mut ingest.bundle, &ingest.llama_bench)?;
                ingest.bundle
            } else {
                decode_json(bytes)?
            };
            if path.starts_with("/api/benchmarks/validate") {
                preview_response(store, bundle)
            } else {
                Ok(json_response(
                    StatusCode::CREATED,
                    &json!({"run_id": store.ingest(&bundle)?}),
                ))
            }
        }
        _ => Err(BenchmarkError::NotFound("not found".into())),
    }
}

fn preview_response(store: &BenchmarkStore, bundle: IngestBundle) -> BenchmarkResult<HttpResponse> {
    store.ingest_checked(&bundle, false)?;
    Ok(json_response(
        StatusCode::OK,
        &json!({
            "bundle": bundle,
            "valid": true,
            "persisted": false,
            "warnings": [
                "Metadata and hashes are user-supplied declarations, not verified against artifact files.",
                "Preview reserves no IDs. Confirmation revalidates; successful runs and registry definitions are immutable.",
                "No command, path, URL, or benchmark is executed or opened."
            ]
        }),
    ))
}

#[cfg(test)]
#[path = "http_tests.rs"]
mod tests;
