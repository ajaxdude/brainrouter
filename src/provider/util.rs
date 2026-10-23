//! Small shared HTTP helpers for non-provider call sites (decision slot).
use anyhow::{Context, Result};
use std::time::Duration;

/// POST a JSON body and return the response text. Errors on connect/timeout/
/// non-2xx (the decision slot maps every such error to its failure row).
static CLIENT: std::sync::LazyLock<reqwest::Client> = std::sync::LazyLock::new(|| {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .build()
        .expect("decision HTTP client")
});

pub async fn post_json_body(url: &str, body: &serde_json::Value, timeout: Duration) -> Result<String> {
    let client = &*CLIENT;
    let resp = client
        .post(url)
        .timeout(timeout)
        .json(body)
        .send()
        .await
        .with_context(|| format!("POST {url}"))?
        .error_for_status()
        .with_context(|| format!("POST {url} non-2xx"))?;
    resp.text().await.context("reading response body")
}
