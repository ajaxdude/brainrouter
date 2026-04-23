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
/// `SafeStream::new(TimeoutStream::new(raw_stream, ...), format)`
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
#[pin_project]
pub struct TimeoutStream<S: Stream> {
    #[pin]
    stream: S,
    #[pin]
    sleep: Sleep,
    duration: Duration,
    timed_out: bool,
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
                // Item received! Reset the timeout for the next chunk
                this.sleep.reset(tokio::time::Instant::now() + *this.duration);
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                // Stream ended normally
                Poll::Ready(None)
            }
            Poll::Pending => {
                // No item yet, check if the timeout has elapsed
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
