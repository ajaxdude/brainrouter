use anyhow::Result;
use bytes::Bytes;
use http_body_util::Full;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{body::Incoming, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde::Serialize;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::{TcpListener, UnixListener};
use tracing::{debug, error, info};

/// Shared state passed to all request handlers
pub struct AppState {
    // Will hold router, config, etc. in later phases
    // For now, empty
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

#[derive(Serialize)]
struct ModelListResponse {
    object: &'static str,
    data: Vec<ModelObject>,
}

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

/// Create a JSON response with the given status code
fn json_response<T: Serialize>(status: StatusCode, body: &T) -> Response<Full<Bytes>> {
    let json = serde_json::to_vec(body).unwrap_or_else(|e| {
        error!("Failed to serialize response: {}", e);
        br#"{"error":"internal serialization error"}"#.to_vec()
    });

    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(json)))
        .expect("Failed to build response")
}

/// Handle incoming HTTP requests
async fn handle_request(
    req: Request<Incoming>,
    _state: Arc<AppState>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let method = req.method();
    let path = req.uri().path();

    debug!("Request: {} {}", method, path);

    let response = match (method.as_str(), path) {
        ("GET", "/health") => {
            json_response(StatusCode::OK, &HealthResponse { status: "ok" })
        }
        ("GET", "/v1/models") => {
            let models = ModelListResponse {
                object: "list",
                data: vec![],
            };
            json_response(StatusCode::OK, &models)
        }
        ("POST", "/v1/chat/completions") => {
            json_response(
                StatusCode::NOT_IMPLEMENTED,
                &ErrorResponse {
                    error: "not yet implemented".to_string(),
                },
            )
        }
        _ => {
            json_response(
                StatusCode::NOT_FOUND,
                &ErrorResponse {
                    error: format!("Not found: {} {}", method, path),
                },
            )
        }
    };

    Ok(response)
}

/// Run the HTTP server with dual listeners (TCP + Unix domain socket)
pub async fn run(
    tcp_addr: SocketAddr,
    uds_path: PathBuf,
    state: Arc<AppState>,
) -> Result<()> {
    // Remove existing UDS socket if it exists
    if uds_path.exists() {
        info!("Removing existing Unix socket at {:?}", uds_path);
        std::fs::remove_file(&uds_path)?;
    }

    // Bind TCP listener
    let tcp_listener = TcpListener::bind(tcp_addr).await?;
    info!("TCP listener bound to {}", tcp_addr);

    // Bind Unix domain socket listener
    let uds_listener = UnixListener::bind(&uds_path)?;
    info!("Unix socket listener bound to {:?}", uds_path);

    // Clone state for both tasks
    let tcp_state = state.clone();
    let uds_state = state;

    // Spawn TCP listener task
    let tcp_task = tokio::spawn(async move {
        loop {
            match tcp_listener.accept().await {
                Ok((stream, addr)) => {
                    info!("New TCP connection from {}", addr);
                    let io = TokioIo::new(stream);
                    let state = tcp_state.clone();

                    tokio::spawn(async move {
                        if let Err(e) = http1::Builder::new()
                            .serve_connection(
                                io,
                                service_fn(move |req| handle_request(req, state.clone())),
                            )
                            .await
                        {
                            error!("Error serving TCP connection: {}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to accept TCP connection: {}", e);
                }
            }
        }
    });

    // Spawn UDS listener task
    let uds_path_for_cleanup = uds_path.clone();
    let uds_task = tokio::spawn(async move {
        loop {
            match uds_listener.accept().await {
                Ok((stream, _addr)) => {
                    info!("New Unix socket connection");
                    let io = TokioIo::new(stream);
                    let state = uds_state.clone();

                    tokio::spawn(async move {
                        if let Err(e) = http1::Builder::new()
                            .serve_connection(
                                io,
                                service_fn(move |req| handle_request(req, state.clone())),
                            )
                            .await
                        {
                            error!("Error serving Unix socket connection: {}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to accept Unix socket connection: {}", e);
                }
            }
        }
    });

    // Wait for both tasks (they run forever unless cancelled)
    tokio::select! {
        _ = tcp_task => {
            info!("TCP listener task ended");
        }
        _ = uds_task => {
            info!("Unix socket listener task ended");
        }
    }

    // Clean up UDS socket on shutdown
    if uds_path_for_cleanup.exists() {
        info!("Cleaning up Unix socket at {:?}", uds_path_for_cleanup);
        let _ = std::fs::remove_file(&uds_path_for_cleanup);
    }

    Ok(())
}
