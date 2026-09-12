//! Narrow Unix-socket proxy used by isolated benchmark workers.
//!
//! The proxy exposes only model discovery and chat completions. Benchmark
//! sandboxes receive this socket instead of the daemon's full administrative
//! socket, preventing generated code from reaching configuration or control
//! endpoints.

use std::{
    convert::Infallible,
    ffi::OsString,
    net::SocketAddr,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result};
use bytes::Bytes;
use http_body_util::{combinators::UnsyncBoxBody, BodyExt, Full};
use hyper::{
    body::Incoming, client::conn::http1, service::service_fn, Request, Response, StatusCode,
};
use hyper_util::rt::TokioIo;
use tokio::{
    net::{TcpListener, UnixListener, UnixStream},
    process::Command,
    task::JoinHandle,
};
use tracing::warn;

type ProxyBody = UnsyncBoxBody<Bytes, anyhow::Error>;

pub struct InferenceProxy {
    socket_path: PathBuf,
    task: JoinHandle<()>,
}

impl Drop for InferenceProxy {
    fn drop(&mut self) {
        self.task.abort();
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

pub async fn start(socket_path: PathBuf, target_socket: PathBuf) -> Result<InferenceProxy> {
    if let Some(parent) = socket_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    match tokio::fs::remove_file(&socket_path).await {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("failed to bind inference proxy {}", socket_path.display()))?;
    let cleanup_path = socket_path.clone();
    let task = tokio::spawn(async move {
        loop {
            let (stream, _) = match listener.accept().await {
                Ok(connection) => connection,
                Err(error) => {
                    warn!(error = %error, "Benchmark inference proxy accept failed");
                    break;
                }
            };
            let target_socket = target_socket.clone();
            tokio::spawn(async move {
                let service =
                    service_fn(move |request| proxy_request(request, target_socket.clone()));
                if let Err(error) = hyper::server::conn::http1::Builder::new()
                    .serve_connection(TokioIo::new(stream), service)
                    .await
                {
                    warn!(error = %error, "Benchmark inference proxy connection failed");
                }
            });
        }
        let _ = tokio::fs::remove_file(cleanup_path).await;
    });
    Ok(InferenceProxy { socket_path, task })
}

/// Run one command behind a TCP-to-Unix inference-only proxy.
///
/// This is invoked inside the benchmark network namespace. The only mounted
/// host socket is already filtered by [`start`], so neither the child nor other
/// processes in the namespace can reach daemon administration endpoints.
pub async fn run_sandbox_command(
    listen: SocketAddr,
    target_socket: PathBuf,
    command: Vec<OsString>,
) -> Result<()> {
    let (program, args) = command
        .split_first()
        .context("benchmark sandbox command is empty")?;
    let listener = TcpListener::bind(listen)
        .await
        .with_context(|| format!("failed to bind benchmark inference proxy at {listen}"))?;
    let proxy_task = tokio::spawn(async move {
        loop {
            let (stream, _) = match listener.accept().await {
                Ok(connection) => connection,
                Err(error) => {
                    warn!(error = %error, "Sandbox inference proxy accept failed");
                    break;
                }
            };
            let target_socket = target_socket.clone();
            tokio::spawn(async move {
                let service =
                    service_fn(move |request| proxy_request(request, target_socket.clone()));
                if let Err(error) = hyper::server::conn::http1::Builder::new()
                    .serve_connection(TokioIo::new(stream), service)
                    .await
                {
                    warn!(error = %error, "Sandbox inference proxy connection failed");
                }
            });
        }
    });

    let status = Command::new(program)
        .args(args)
        .kill_on_drop(true)
        .status()
        .await
        .with_context(|| format!("failed to start {}", Path::new(program).display()))?;
    proxy_task.abort();
    if !status.success() {
        anyhow::bail!("sandbox command exited with status {status}");
    }
    Ok(())
}

async fn proxy_request(
    request: Request<Incoming>,
    target_socket: PathBuf,
) -> Result<Response<ProxyBody>, Infallible> {
    if !request_allowed(request.method().as_str(), request.uri().path()) {
        return Ok(text_response(
            StatusCode::FORBIDDEN,
            "Benchmark proxy allows inference endpoints only",
        ));
    }

    Ok(match forward(request, &target_socket).await {
        Ok(response) => response,
        Err(error) => {
            warn!(error = %error, "Benchmark inference proxy upstream failed");
            text_response(StatusCode::BAD_GATEWAY, "Inference proxy upstream failed")
        }
    })
}

fn request_allowed(method: &str, path: &str) -> bool {
    matches!(
        (method, path),
        ("POST", "/v1/chat/completions") | ("GET", "/v1/models")
    )
}

async fn forward(request: Request<Incoming>, target_socket: &Path) -> Result<Response<ProxyBody>> {
    let stream = UnixStream::connect(target_socket)
        .await
        .with_context(|| format!("failed to connect to {}", target_socket.display()))?;
    let (mut sender, connection) = http1::handshake(TokioIo::new(stream)).await?;
    tokio::spawn(async move {
        if let Err(error) = connection.await {
            warn!(error = %error, "Benchmark inference proxy upstream connection failed");
        }
    });
    let response = sender.send_request(request).await?;
    Ok(response.map(|body| body.map_err(anyhow::Error::new).boxed_unsync()))
}

fn text_response(status: StatusCode, message: &'static str) -> Response<ProxyBody> {
    Response::builder()
        .status(status)
        .header("content-type", "text/plain; charset=utf-8")
        .body(
            Full::new(Bytes::from_static(message.as_bytes()))
                .map_err(|error: Infallible| match error {})
                .boxed_unsync(),
        )
        .expect("static proxy response is valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxy_allows_only_inference_endpoints() {
        assert!(request_allowed("POST", "/v1/chat/completions"));
        assert!(request_allowed("GET", "/v1/models"));
        assert!(!request_allowed("POST", "/api/config"));
        assert!(!request_allowed("POST", "/api/benchmarks/lab/jobs"));
        assert!(!request_allowed("GET", "/health"));
    }
}
