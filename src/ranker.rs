use crate::config::ModelConfig;
use crate::provider::embedded::EmbeddedProvider;
use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::info;

#[derive(Debug, Deserialize)]
struct RankedModels {
    ranked: Vec<String>,
}

pub fn rank_models(
    models: &[ModelConfig],
    bonsai_provider: &EmbeddedProvider,
) -> Result<Vec<String>> {
    info!("Ranking models with Bonsai...");

    let model_names: Vec<String> = models.iter().map(|m| m.name.clone()).collect();
    let prompt = format!(
        "Rank these LLM models for software engineering tasks (coding, debugging, refactoring). \
        Return a JSON array ordered best to worst. Only rank what I give you: {:?}",
        model_names
    );

    let request = crate::types::ChatCompletionRequest {
        model: "bonsai".to_string(),
        messages: vec![crate::types::ChatMessage {
            role: "user".to_string(),
            content: Some(serde_json::Value::String(prompt)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        stream: Some(false),
        temperature: Some(0.0),
        max_tokens: Some(1024),
        top_p: None,
        stop: None,
        extra: Default::default(),
    };

    let response = bonsai_provider.chat_completion_blocking(request)?;

    let choice = response
        .choices
        .get(0)
        .context("Bonsai returned no choices")?;
    let content = choice
        .message
        .content
        .as_ref()
        .and_then(|c| c.as_str())
        .context("Bonsai response content is not a string")?;

    // Extract JSON from the response. It might be in a code block.
    let json_str = if let Some(start) = content.find('{') {
        if let Some(end) = content.rfind('}') {
            &content[start..=end]
        } else {
            content
        }
    } else {
        content
    };
    
    let ranked_models: RankedModels =
        serde_json::from_str(json_str).context("Failed to parse Bonsai's JSON response")?;

    info!("Bonsai ranking complete: {:?}", ranked_models.ranked);
    Ok(ranked_models.ranked)
}
