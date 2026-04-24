//! Review loop — iterates up to `max_iterations` calling the LLM via the Router.
//!
//! The router applies the same Bonsai classification and provider failover that
//! chat-completion requests go through, so review calls benefit from the same
//! circuit breaker and stall detection.

use anyhow::Result;
use serde::Deserialize;
use std::sync::Arc;
use tracing::{info, warn};

use crate::{
    router::Router,
    session::{EscalationReason, ReviewStatus, ReviewerType, SessionManager, SessionUpdate},
    types::{ChatCompletionRequest, ChatMessage},
};

use super::{
    context,
    prompt::build_review_prompt,
};

/// Structured response the LLM is instructed to return.
#[derive(Debug, Deserialize)]
struct LlmReviewResponse {
    status: String,
    feedback: String,
}

/// Terminal result of a review run.
pub struct ReviewResult {
    pub status: ReviewStatus,
    pub feedback: String,
    pub session_id: String,
    pub iteration_count: u32,
    pub reviewer_type: ReviewerType,
    pub escalation_reason: Option<EscalationReason>,
}

/// Run the review loop for an already-created session.
///
/// Iterates up to `max_iterations`. On each iteration:
/// 1. Gather context (PRD, git diff, AGENTS).
/// 2. Build prompt.
/// 3. Send through the Router (Bonsai classify → Manifest or llama-swap).
/// 4. Parse the JSON response.
/// 5. Update the session.
/// 6. Break on "approved" or "escalated"; otherwise continue.
///
/// If all iterations exhaust without approval, escalates with `max_iterations` reason.
pub async fn run_loop(
    session_id: &str,
    task_id: &str,
    summary: &str,
    details: Option<&str>,
    router: &Arc<Router>,
    sessions: &Arc<SessionManager>,
    config: &crate::config::ReviewConfig,
    project_dir: &str,
) -> Result<ReviewResult> {
    let mut iteration_count: u32 = 0;
    let mut status = ReviewStatus::Pending;
    let mut feedback = String::new();
    let mut escalation_reason: Option<EscalationReason> = None;
    let mut session_history: Vec<String> = Vec::new();

    while iteration_count < config.max_iterations {
        iteration_count += 1;
        info!(session_id, iteration = iteration_count, max_iterations = config.max_iterations, "Review iteration");

        // Gather context fresh each iteration (git diff may change).
        // context::gather runs blocking I/O (git diff, fs reads); run off the async executor.
        let project_dir_owned = project_dir.to_string();
        let ctx = tokio::task::spawn_blocking(move || context::gather(&project_dir_owned))
            .await
            .unwrap_or_else(|_| context::ReviewContext { prd: None, git_diff: String::new(), agents_content: None });

        let prompt = build_review_prompt(&ctx, task_id, summary, details, &session_history);

        // Determine the model for this review call
        let requested_model = match config.forced_mode.as_str() {
            "cloud" => "cloud".to_string(),
            "local" => config.forced_model.clone().unwrap_or_else(|| "local".to_string()),
            _ => "auto".to_string(),
        };

        // Route through the same Router used by the HTTP proxy, tagging the event
        // with this session_id so the dashboard can correlate review calls.
        let result = call_llm_for_review(router, prompt.clone(), session_id, requested_model, project_dir).await;

        match result {
            Err(e) => {
                warn!(session_id, error = %e, "LLM call failed during review");
                status = ReviewStatus::Escalated;
                escalation_reason = Some(EscalationReason::LlmError);
                feedback = format!("LLM error: {}", e);

                sessions.update_session(
                    session_id,
                    SessionUpdate {
                        status: Some(ReviewStatus::Escalated),
                        feedback: Some(feedback.clone()),
                        reviewer_type: Some(ReviewerType::Llm),
                        escalation_reason: Some(EscalationReason::LlmError),
                        review_model: None,
                    },
                );
                break;
            }
            Ok((raw_text, route_info)) => {
                // Record which model handled this review (first iteration sets it; later
                // iterations are no-ops because update_session only sets it once).
                let review_model = route_info.display();

                // Parse the JSON response from the LLM
                match parse_llm_response(&raw_text) {
                    Ok(parsed) => {
                        status = map_status(&parsed.status);
                        feedback = parsed.feedback.clone();

                        info!(
                            session_id,
                            iteration = iteration_count,
                            status = %status,
                            review_model = %review_model,
                            "LLM returned review decision"
                        );

                        sessions.update_session(
                            session_id,
                            SessionUpdate {
                                status: Some(status.clone()),
                                feedback: Some(feedback.clone()),
                                reviewer_type: Some(ReviewerType::Llm),
                                escalation_reason: if status == ReviewStatus::Escalated {
                                    Some(EscalationReason::LlmError)
                                } else {
                                    None
                                },
                                review_model: Some(review_model),
                            },
                        );

                        session_history.push(format!(
                            "Iteration {}:\nStatus: {}\nFeedback: {}",
                            iteration_count, status, feedback
                        ));

                        if matches!(status, ReviewStatus::Approved | ReviewStatus::Escalated) {
                            break;
                        }

                        // needs_revision — increment and continue
                        sessions.increment_iteration(session_id);
                    }
                    Err(e) => {
                        warn!(session_id, error = %e, raw = %raw_text, "Failed to parse LLM review response");
                        // Treat parse failure as LLM error on last iteration, otherwise retry
                        if iteration_count >= config.max_iterations {
                            status = ReviewStatus::Escalated;
                            escalation_reason = Some(EscalationReason::LlmError);
                            feedback = format!("Failed to parse LLM response: {}", e);

                            sessions.update_session(
                                session_id,
                                SessionUpdate {
                                    status: Some(ReviewStatus::Escalated),
                                    feedback: Some(feedback.clone()),
                                    reviewer_type: Some(ReviewerType::Llm),
                                    escalation_reason: Some(EscalationReason::LlmError),
                                    review_model: Some(review_model),
                                },
                            );
                        } else {
                            sessions.increment_iteration(session_id);
                        }
                    }
                }
            }
        }
    }

    // All iterations exhausted without approval — escalate
    if !matches!(status, ReviewStatus::Approved | ReviewStatus::Escalated) {
        status = ReviewStatus::Escalated;
        escalation_reason = Some(EscalationReason::MaxIterations);
        feedback = format!(
            "Review did not converge after {} iterations. Last feedback: {}",
            iteration_count, feedback
        );
        sessions.update_session(
            session_id,
            SessionUpdate {
                status: Some(ReviewStatus::Escalated),
                feedback: Some(feedback.clone()),
                reviewer_type: Some(ReviewerType::Llm),
                escalation_reason: Some(EscalationReason::MaxIterations),
                review_model: None,
            },
        );
    }

    Ok(ReviewResult {
        status,
        feedback,
        session_id: session_id.to_string(),
        iteration_count,
        reviewer_type: ReviewerType::Llm,
        escalation_reason,
    })
}

/// Send the review prompt through the Router, tagged with the session_id.
/// Returns the full collected text response and routing metadata.
async fn call_llm_for_review(
    router: &Arc<Router>,
    prompt: String,
    session_id: &str,
    model: String,
    project_dir: &str,
) -> Result<(String, crate::router::RouteInfo)> {
    let request = ChatCompletionRequest {
        model,
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: Some(serde_json::Value::String(
                    "You are a code review expert. Review the provided code changes carefully and respond with a JSON object as specified.".to_string()
                )),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some(serde_json::Value::String(prompt)),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        stream: Some(true),
        temperature: Some(0.1),
        max_tokens: Some(2048),
        top_p: None,
        stop: None,
        extra: serde_json::Value::Object(serde_json::Map::new()),
    };


    let (provider_response, route_info) = router
        .route_tagged(request, Some(session_id.to_string()), project_dir.to_string())
        .await?;

    // Collect the SSE stream into a full text response
    use crate::provider::ProviderResponse;
    use futures_util::StreamExt;

    let ProviderResponse::Stream(mut stream) = provider_response;
    let mut collected = String::new();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        // Each chunk is a bytes::Bytes SSE line like "data: {...}\n\n"
        let text = std::str::from_utf8(&chunk)?;
        for line in text.lines() {
            let line = line.trim();
            if let Some(json_str) = line.strip_prefix("data: ") {
                if json_str == "[DONE]" {
                    break;
                }
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if let Some(content) = parsed
                        .get("choices")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("delta"))
                        .and_then(|d| d.get("content"))
                        .and_then(|c| c.as_str())
                    {
                        collected.push_str(content);
                    }
                }
            }
        }
    }

    Ok((collected, route_info))
}

/// Extract JSON from LLM response text (may be wrapped in markdown code fences).
fn parse_llm_response(text: &str) -> Result<LlmReviewResponse> {
    // Try to find the first occurrence of a JSON block
    let json_str = if let Some(start) = text.find("```json") {
        let after = &text[start + 7..];
        if let Some(end) = after.find("```") {
            after[..end].trim()
        } else {
            after.trim()
        }
    } else if let Some(start) = text.find("```") {
        let after = &text[start + 3..];
        if let Some(end) = after.find("```") {
            after[..end].trim()
        } else {
            after.trim()
        }
    } else if let Some(start) = text.find('{') {
        // Find the last closing brace
        if let Some(end) = text.rfind('}') {
            &text[start..=end]
        } else {
            text.trim()
        }
    } else {
        text.trim()
    };

    // Robust parsing: if it's truncated or has trailing garbage, 
    // some JSON parsers might fail. We attempt to parse what we have.
    match serde_json::from_str::<LlmReviewResponse>(json_str) {
        Ok(parsed) => Ok(parsed),
        Err(e) => {
             // Fallback: If status and feedback are present but slightly malformed, 
             // we could try a manual regex extraction, but let's stick to strict JSON first.
             Err(anyhow::anyhow!("Could not parse JSON from LLM response: {}. Raw: {}", e, text))
        }
    }
}

/// Map the LLM's string status to our enum.
fn map_status(s: &str) -> ReviewStatus {
    match s {
        "approved" => ReviewStatus::Approved,
        "needs_revision" => ReviewStatus::NeedsRevision,
        _ => ReviewStatus::Escalated,
    }
}
