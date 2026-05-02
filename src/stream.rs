use std::future::Future;
use anyhow::Result;
use bytes::Bytes;
use futures_util::Stream;
use pin_project::pin_project;
use serde_json;
use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};
use tokio::time::{sleep, Sleep};

#[derive(Clone, Copy, Debug)]
pub enum StreamFormat {
    OpenAi,
    Anthropic,
}

/// A stream wrapper that catches errors and yields a final SSE error chunk
/// before ending gracefully, avoiding "unexpected socket closure" errors.
///
/// Intended to be the outer wrapper for streaming responses:
/// `SafeStream::new(KeepaliveStream::new(TimeoutStream::new(raw_stream, ...), ...), format)`
#[pin_project]
pub struct SafeStream<S: Stream> {
    #[pin]
    inner: S,
    format: StreamFormat,
    error_sent: bool,
}

impl<S: Stream> SafeStream<S> {
    pub fn new(inner: S, format: StreamFormat) -> Self {
        Self {
            inner,
            format,
            error_sent: false,
        }
    }
}

impl<S: Stream> Stream for SafeStream<S>
where
    S: Stream<Item = Result<Bytes>>,
{
    type Item = Result<Bytes>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();

        if *this.error_sent {
            return Poll::Ready(None);
        }

        match this.inner.poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => Poll::Ready(Some(Ok(bytes))),
            Poll::Ready(Some(Err(e))) => {
                *this.error_sent = true;
                let message = e.to_string();
                let chunk = match this.format {
                    StreamFormat::OpenAi => {
                        let payload = serde_json::json!({
                            "error": {
                                "message": message,
                                "type": "brainrouter_error"
                            }
                        });
                        format!("data: {}\n\ndata: [DONE]\n\n", payload)
                    }
                    StreamFormat::Anthropic => {
                        let payload = serde_json::json!({
                            "type": "error",
                            "error": {
                                "type": "overloaded_error",
                                "message": message
                            }
                        });
                        format!("data: {}\n\n", payload)
                    }
                };
                Poll::Ready(Some(Ok(Bytes::from(chunk))))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// A stream that applies a per-chunk timeout.
/// If no item arrives within `duration`, the stream yields a "stalled" error.
///
/// The sleep deadline is deferred until the first `Pending` poll, so the
/// timeout counts from the first actual wait — not from construction.
#[pin_project]
pub struct TimeoutStream<S: Stream> {
    #[pin]
    stream: S,
    #[pin]
    sleep: Sleep,
    duration: Duration,
    timed_out: bool,
    started: bool,
}

impl<S: Stream> TimeoutStream<S>
where
    S: Stream<Item = Result<Bytes>>,
{
    pub fn new(stream: S, duration: Duration) -> Self {
        Self {
            stream,
            sleep: sleep(duration),
            duration,
            timed_out: false,
            started: false,
        }
    }
}

impl<S: Stream> Stream for TimeoutStream<S>
where
    S: Stream<Item = Result<Bytes>>,
{
    type Item = Result<Bytes>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        if *this.timed_out {
            return Poll::Ready(None);
        }

        // 1. Try to get the next item from the inner stream
        match this.stream.as_mut().poll_next(cx) {
            Poll::Ready(Some(item)) => {
                // Item received! Reset the timeout for the next chunk.
                this.sleep.reset(tokio::time::Instant::now() + *this.duration);
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                // Stream ended normally
                Poll::Ready(None)
            }
            Poll::Pending => {
                // Lazily arm the timer on first pending poll so the
                // deadline counts from now, not from construction.
                if !*this.started {
                    *this.started = true;
                    this.sleep.as_mut().reset(tokio::time::Instant::now() + *this.duration);
                }
                // Check if the timeout has elapsed
                match this.sleep.poll(cx) {
                    Poll::Ready(_) => {
                        // Timeout elapsed!
                        *this.timed_out = true;
                        Poll::Ready(Some(Err(anyhow::anyhow!("Stream stalled"))))
                    }
                    Poll::Pending => {
                        // Still waiting for either an item or the timeout
                        Poll::Pending
                    }
                }
            }
        }
    }
}

/// Interval at which keepalive frames are emitted when the inner stream is
/// idle. Chosen to be well below any reasonable client-side stream-idle watchdog
/// (OMP defaults to 120 s; we ping every 15 s).
pub const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(15);

/// An SSE comment used as a keepalive for Anthropic-format streams.
/// Anthropic SDK and OMP both treat comment lines as ignorable heartbeats.
const KEEPALIVE_ANTHROPIC: &[u8] = b": ping\n\n";

/// An empty-delta OpenAI SSE chunk used as a keepalive for OpenAI-format streams.
///
/// OMP's `iterateWithIdleTimeout` races `iterator.next()` against a wall-clock
/// timer. The openai SDK's SSE parser silently discards comment lines and never
/// yields them to the async iterator — so SSE comments cannot reset that timer.
/// We must emit a valid `data:` frame that the SDK will parse into a
/// `ChatCompletionChunk` and yield, which resets OMP's idle watchdog.
///
/// An empty string delta is safe: the provider code skips zero-length content.
const KEEPALIVE_OPENAI: &[u8] =
    b"data: {\"id\":\"\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"finish_reason\":null}]}\n\n";

/// A stream wrapper that emits a periodic keepalive frame while the inner stream
/// is idle (`Poll::Pending`). Once the inner stream yields data or terminates this
/// wrapper is fully transparent.
///
/// **Format matters:** for OpenAI streams, keepalives must be real `data:` frames
/// because the SDK parser discards SSE comment lines before they reach the
/// application-level async iterator. For Anthropic streams, SSE comments suffice.
///
/// Place this **outside** `TimeoutStream` so keepalive frames do not interfere
/// with the stall detector — if the backend is truly hung for 180 s the stall
/// error still fires.
#[pin_project]
pub struct KeepaliveStream<S: Stream> {
    #[pin]
    inner: S,
    #[pin]
    sleep: Sleep,
    interval: Duration,
    format: StreamFormat,
}

impl<S> KeepaliveStream<S>
where
    S: Stream<Item = Result<Bytes>>,
{
    pub fn new(inner: S, interval: Duration, format: StreamFormat) -> Self {
        Self {
            inner,
            sleep: sleep(interval),
            interval,
            format,
        }
    }
}

impl<S> Stream for KeepaliveStream<S>
where
    S: Stream<Item = Result<Bytes>>,
{
    type Item = Result<Bytes>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        // 1. Try to get the next item from the inner stream.
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(item) => {
                // Data or end — reset the keepalive clock and pass through.
                this.sleep.reset(tokio::time::Instant::now() + *this.interval);
                return Poll::Ready(item);
            }
            Poll::Pending => {}
        }

        // 2. Inner is idle. Check whether a keepalive interval has elapsed.
        match this.sleep.as_mut().poll(cx) {
            Poll::Ready(_) => {
                // Emit a keepalive and restart the clock.
                this.sleep.reset(tokio::time::Instant::now() + *this.interval);
                let bytes = match this.format {
                    StreamFormat::OpenAi => Bytes::from_static(KEEPALIVE_OPENAI),
                    StreamFormat::Anthropic => Bytes::from_static(KEEPALIVE_ANTHROPIC),
                };
                Poll::Ready(Some(Ok(bytes)))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}