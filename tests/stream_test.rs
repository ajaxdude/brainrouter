//! Unit tests for stream wrappers.

use anyhow::Result;
use brainrouter::stream::{KeepaliveStream, StreamFormat, KEEPALIVE_INTERVAL};
use bytes::Bytes;
use futures_util::{pin_mut, stream, StreamExt};
use tokio::time;

/// OpenAI keepalive is a real empty-delta `data:` SSE frame so the SDK yields
/// it to the application-level iterator (SDK silently drops comment lines).
const KEEPALIVE_OPENAI: &[u8] =
    b"data: {\"id\":\"\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"finish_reason\":null}]}\n\n";

/// Anthropic keepalive is an SSE comment line — Anthropic SDK and OMP treat
/// comment lines as ignorable heartbeats.
const KEEPALIVE_ANTHROPIC: &[u8] = b": ping\n\n";

/// Advancing past an interval on a permanently-idle inner stream must produce
/// at least one OpenAI-format keepalive frame.
#[tokio::test]
async fn keepalive_emits_openai_frame_when_idle() {
    time::pause();

    let inner = stream::pending::<Result<Bytes>>();
    let ka = KeepaliveStream::new(inner, KEEPALIVE_INTERVAL, StreamFormat::OpenAi);
    pin_mut!(ka);

    time::advance(KEEPALIVE_INTERVAL + std::time::Duration::from_millis(1)).await;

    let chunk = ka.next().await.expect("expected Some").expect("expected Ok");
    assert_eq!(chunk.as_ref(), KEEPALIVE_OPENAI, "expected OpenAI keepalive frame");
}

/// Anthropic-format idle stream emits SSE comment.
#[tokio::test]
async fn keepalive_emits_anthropic_comment_when_idle() {
    time::pause();

    let inner = stream::pending::<Result<Bytes>>();
    let ka = KeepaliveStream::new(inner, KEEPALIVE_INTERVAL, StreamFormat::Anthropic);
    pin_mut!(ka);

    time::advance(KEEPALIVE_INTERVAL + std::time::Duration::from_millis(1)).await;

    let chunk = ka.next().await.expect("expected Some").expect("expected Ok");
    assert_eq!(chunk.as_ref(), KEEPALIVE_ANTHROPIC, "expected Anthropic keepalive comment");
}

/// When the inner stream yields data without any delay, no keepalive is emitted.
#[tokio::test]
async fn keepalive_transparent_when_fast() {
    let items: Vec<Result<Bytes>> = vec![
        Ok(Bytes::from_static(b"a")),
        Ok(Bytes::from_static(b"b")),
    ];
    let inner = stream::iter(items);
    let ka = KeepaliveStream::new(inner, KEEPALIVE_INTERVAL, StreamFormat::OpenAi);
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
    let ka = KeepaliveStream::new(inner, KEEPALIVE_INTERVAL, StreamFormat::OpenAi);
    pin_mut!(ka);

    let err = ka.next().await.unwrap().unwrap_err();
    assert_eq!(err.to_string(), "backend error");
}
