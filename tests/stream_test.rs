//! Unit tests for stream wrappers.

use anyhow::Result;
use brainrouter::stream::{KeepaliveStream, KEEPALIVE_INTERVAL};
use bytes::Bytes;
use futures_util::{pin_mut, stream, StreamExt};
use tokio::time;

/// A keepalive ping is a `: ping\n\n` SSE comment line.
const PING: &[u8] = b": ping\n\n";

/// Advancing past an interval on a permanently-idle inner stream must produce
/// at least one keepalive ping.
#[tokio::test]
async fn keepalive_emits_ping_when_inner_is_idle() {
    time::pause();

    // Inner stream never yields anything (simulates a slow TTFT indefinitely).
    let inner = stream::pending::<Result<Bytes>>();
    let ka = KeepaliveStream::new(inner, KEEPALIVE_INTERVAL);
    pin_mut!(ka);

    // Advance past one interval. The keepalive sleep should fire.
    time::advance(KEEPALIVE_INTERVAL + std::time::Duration::from_millis(1)).await;

    let chunk = ka.next().await.expect("expected Some").expect("expected Ok");
    assert_eq!(chunk.as_ref(), PING, "expected keepalive comment line");
}

/// When the inner stream yields data without any delay, no ping should be
/// emitted before the first data chunk.
#[tokio::test]
async fn keepalive_transparent_when_fast() {
    let items: Vec<Result<Bytes>> = vec![
        Ok(Bytes::from_static(b"a")),
        Ok(Bytes::from_static(b"b")),
    ];
    let inner = stream::iter(items);
    let ka = KeepaliveStream::new(inner, KEEPALIVE_INTERVAL);
    pin_mut!(ka);

    let first = ka.next().await.unwrap().unwrap();
    assert_eq!(first, Bytes::from_static(b"a"));

    let second = ka.next().await.unwrap().unwrap();
    assert_eq!(second, Bytes::from_static(b"b"));

    assert!(ka.next().await.is_none());
}

/// Errors from the inner stream are passed through unchanged.
#[tokio::test]
async fn keepalive_passes_errors() {
    let inner = stream::once(async { Err::<Bytes, _>(anyhow::anyhow!("backend error")) });
    let ka = KeepaliveStream::new(inner, KEEPALIVE_INTERVAL);
    pin_mut!(ka);

    let err = ka.next().await.unwrap().unwrap_err();
    assert_eq!(err.to_string(), "backend error");
}