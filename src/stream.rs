use anyhow::Result;
use bytes::Bytes;
use futures_util::{Future, Stream, StreamExt};
use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};
use tokio::time::timeout;
use tokio_stream::StreamMap;

/// A stream that wraps another stream and applies a timeout to each item.
#[pin_project::pin_project]
pub struct TimeoutStream<S: Stream> {
    #[pin]
    stream: S,
    duration: Duration,
}

impl<S: Stream> TimeoutStream<S> {
    pub fn new(stream: S, duration: Duration) -> Self {
        Self { stream, duration }
    }
}

impl<S: Stream> Stream for TimeoutStream<S>
where
    S: Stream<Item = Result<Bytes>>,
{
    type Item = Result<Bytes>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        let duration = *this.duration;
        let mut stream = this.stream;

        // Create a future that wraps the stream's next item in a timeout
        let future = timeout(duration, stream.next());

        // We need to poll this future. Since we're in a poll function, we can't just .await it.
        // We need to create a new future on the heap that we can poll.
        let mut future = Box::pin(future);
        
        match future.as_mut().poll(cx) {
            Poll::Ready(Ok(Some(item))) => Poll::Ready(Some(item)),
            Poll::Ready(Ok(None)) => Poll::Ready(None), // Stream ended
            Poll::Ready(Err(_)) => {
                // Timeout elapsed
                Poll::Ready(Some(Err(anyhow::anyhow!("Stream stalled"))))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
