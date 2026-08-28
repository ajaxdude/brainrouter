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
    initial_history: &[String],
    router: &Arc<Router>,
    sessions: &Arc<SessionManager>,
    config: &crate::config::ReviewConfig,
    project_dir: &str,
) -> Result<ReviewResult> {
    let mut iteration_count: u32 = 0;
    let mut status = ReviewStatus::Pending;
    let mut feedback = String::new();
    let mut escalation_reason: Option<EscalationReason> = None;
    // Seed with any prior turns so a "continue" run keeps the review's context.
    let mut session_history: Vec<String> = initial_history.to_vec();

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
                        llm_turns: Some(session_history.clone()),
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
                                    Some(EscalationReason::LlmEscalated)
                                } else {
                                    None
                                },
                                review_model: Some(review_model),
                                llm_turns: Some(session_history.clone()),
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
                                    llm_turns: Some(session_history.clone()),
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
                llm_turns: Some(session_history.clone()),
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
        max_tokens: Some(16384),
        top_p: None,
        stop: None,
        extra: serde_json::Value::Object(serde_json::Map::new()),
    };


    let (provider_response, route_info) = router
        .route_tagged(request, Some(session_id.to_string()), project_dir.to_string(), String::new())
        .await?;

    // Collect the SSE stream into a full text response
    use crate::provider::ProviderResponse;
    use futures_util::StreamExt;

    match provider_response {
        ProviderResponse::Stream(mut stream) => {
            let mut collected = String::new();
            let mut byte_buf = Vec::new();

            'outer: while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result?;
                byte_buf.extend_from_slice(&chunk);

                // Process all complete lines from the buffer
                while let Some(newline_pos) = byte_buf.iter().position(|&b| b == b'\n') {
                    let line_bytes = byte_buf.drain(..=newline_pos).collect::<Vec<_>>();
                    let line = String::from_utf8_lossy(&line_bytes);
                    let line = line.trim();
                    if let Some(json_str) = line.strip_prefix("data: ") {
                        if json_str == "[DONE]" {
                            byte_buf.clear();
                            break 'outer;
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

            // Process any remaining bytes in the buffer
            if !byte_buf.is_empty() {
                let line = String::from_utf8_lossy(&byte_buf);
                let line = line.trim();
                if let Some(json_str) = line.strip_prefix("data: ") {
                    if json_str != "[DONE]" {
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
    }
}

/// Extract JSON from LLM response text (may be wrapped in markdown code fences).
/// Handles truncated output from token limits by attempting to repair the JSON.
fn parse_llm_response(text: &str) -> Result<LlmReviewResponse> {
    // Strip markdown code fences if present
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
        if let Some(end) = text.rfind('}') {
            &text[start..=end]
        } else {
            // No closing brace — truncated JSON. Take from the opening brace onward.
            &text[start..]
        }
    } else {
        text.trim()
    };

    // First try: parse as-is
    if let Ok(parsed) = serde_json::from_str::<LlmReviewResponse>(json_str) {
        return Ok(parsed);
    }

    // Second try: the JSON was likely truncated by the token limit.
    // Common truncation: the "feedback" string is cut mid-sentence.
    // Attempt repair by closing any open string and braces.
    let repaired = repair_truncated_json(json_str);
    if let Ok(parsed) = serde_json::from_str::<LlmReviewResponse>(&repaired) {
        tracing::warn!("Parsed review response after repairing truncated JSON (token limit likely hit)");
        return Ok(parsed);
    }

    // Third try: extract status and feedback fields manually with string search.
    // This handles cases where the JSON structure is too damaged for serde but
    // the key fields are present.
    if let Some(resp) = extract_fields_manually(json_str) {
        tracing::warn!("Extracted review fields via manual parsing (JSON was malformed)");
        return Ok(resp);
    }

    Err(anyhow::anyhow!(
        "Could not parse JSON from LLM response: truncated or malformed. First 500 chars: {}",
        &text[..text.len().min(500)]
    ))
}

/// Attempt to repair truncated JSON by closing open strings and braces.
/// Handles the common case where max_tokens cuts off mid-string-value.
fn repair_truncated_json(s: &str) -> String {
    let mut result = s.to_string();
    // Trim trailing whitespace and incomplete escape sequences
    result = result.trim_end().to_string();
    if result.ends_with('\\') {
        result.push('n'); // close incomplete escape like \n
    }
    // Count unmatched quotes (ignoring escaped ones)
    let mut in_string = false;
    let mut prev = ' ';
    for c in result.chars() {
        if c == '"' && prev != '\\' {
            in_string = !in_string;
        }
        prev = c;
    }
    // If we're inside a string, close it
    if in_string {
        // Trim trailing backslash that would escape our closing quote
        if result.ends_with('\\') {
            result.pop();
        }
        result.push('"');
    }
    // Close any open braces/brackets
    let open_braces = result.chars().filter(|&c| c == '{').count();
    let close_braces = result.chars().filter(|&c| c == '}').count();
    for _ in 0..(open_braces.saturating_sub(close_braces)) {
        // Ensure the last token before } is valid JSON (not a trailing comma)
        let trimmed = result.trim_end();
        if trimmed.ends_with(',') {
            result = trimmed.trim_end_matches(',').to_string();
        }
        result.push('}');
    }
    result
}

/// Last-resort field extraction using simple string matching.
fn extract_fields_manually(text: &str) -> Option<LlmReviewResponse> {
    // Look for "status": "..." pattern
    let status = extract_json_string_field(text, "status")?;
    // feedback may be truncated — extract whatever we have
    let feedback = extract_json_string_field(text, "feedback")
        .unwrap_or_else(|| "[feedback truncated by token limit]".to_string());
    Some(LlmReviewResponse { status, feedback })
}

/// Extract a JSON string field value by name using simple pattern matching.
fn extract_json_string_field(text: &str, field: &str) -> Option<String> {
    let pattern = format!("\"{}\":", field);
    let start = text.find(&pattern)?;
    let after = &text[start + pattern.len()..];
    let after = after.trim_start();
    if !after.starts_with('"') {
        return None;
    }
    let after = &after[1..]; // skip opening quote
    // Find the closing quote (respecting escapes)
    let mut result = String::new();
    let mut chars = after.chars();
    loop {
        match chars.next() {
            Some('\\') => {
                // Escaped character — take the next one literally
                if let Some(escaped) = chars.next() {
                    match escaped {
                        'n' => result.push('\n'),
                        't' => result.push('\t'),
                        '"' => result.push('"'),
                        '\\' => result.push('\\'),
                        other => { result.push('\\'); result.push(other); }
                    }
                } else {
                    break; // truncated escape at end
                }
            }
            Some('"') => return Some(result), // clean end
            Some(c) => result.push(c),
            None => {
                // Truncated string — return what we have
                if !result.is_empty() {
                    return Some(result);
                }
                return None;
            }
        }
    }
    if !result.is_empty() { Some(result) } else { None }
}

/// Map the LLM's string status to our enum.
fn map_status(s: &str) -> ReviewStatus {
    match s {
        "approved" => ReviewStatus::Approved,
        "needs_revision" => ReviewStatus::NeedsRevision,
        _ => ReviewStatus::Escalated,
    }
}
