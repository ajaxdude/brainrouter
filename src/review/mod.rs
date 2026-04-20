//! Review service — orchestrates the code review loop.
//!
//! `ReviewService` is the public face. Callers (HTTP handler, MCP handler)
//! call `start_review` and receive a result once the loop completes or escalates.

pub mod context;
pub mod prompt;
pub mod review_loop;

use anyhow::Result;
use std::sync::Arc;
use tracing::info;

use crate::{
    config::ReviewConfig,
    router::Router,
    session::{ReviewStatus, ReviewerType, SessionManager, SessionUpdate},
};

use self::review_loop::run_loop;

/// Result returned to the MCP caller or HTTP client after a review completes.
pub struct RequestReviewResult {
    pub status: ReviewStatus,
    pub feedback: String,
    pub session_id: String,
    pub iteration_count: u32,
    pub reviewer_type: ReviewerType,
}

/// Orchestrates review sessions. Shared via Arc in AppState.
pub struct ReviewService {
    router: Arc<Router>,
    sessions: Arc<SessionManager>,
    config: ReviewConfig,
}

impl ReviewService {
    pub fn new(
        router: Arc<Router>,
        sessions: Arc<SessionManager>,
        config: ReviewConfig,
    ) -> Self {
        ReviewService {
            router,
            sessions,
            config,
        }
    }

    /// Create a session and run the review loop to completion.
    /// Blocks the calling async task until the review is done.
    /// The session is visible in the dashboard throughout.
    pub async fn start_review(
        &self,
        task_id: String,
        summary: String,
        details: Option<String>,
        conversation_history: Vec<String>,
    ) -> Result<RequestReviewResult> {
        // Create the session first — the agent gets the ID in the response regardless
        // of whether the review loop succeeds or fails.
        let session = self.sessions.create_session(
            task_id.clone(),
            summary.clone(),
            details.clone(),
            conversation_history,
        );
        info!(session_id = %session.id, task_id = %task_id, "Created review session");

        let result = run_loop(
            &session.id,
            &task_id,
            &summary,
            details.as_deref(),
            &self.router,
            &self.sessions,
            self.config.max_iterations,
        )
        .await?;

        Ok(RequestReviewResult {
            status: result.status,
            feedback: result.feedback,
            session_id: result.session_id,
            iteration_count: result.iteration_count,
            reviewer_type: result.reviewer_type,
        })
    }

    /// Resolve a session with human feedback (from the escalation UI).
    pub fn resolve_session(&self, session_id: &str, feedback: String) -> Result<()> {


        let _session = self
            .sessions
            .get_session(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;

        // Determine approval status from feedback text
        let is_approval = {
            let lower = feedback.to_lowercase();
            lower.starts_with("ok")
                || lower.starts_with("lgtm")
                || lower.starts_with("approved")
                || lower.starts_with("looks good")
                || lower.starts_with("ship it")
        };

        let new_status = if is_approval {
            ReviewStatus::Approved
        } else {
            ReviewStatus::NeedsRevision
        };

        self.sessions.update_session(
            session_id,
            SessionUpdate {
                status: Some(new_status),
                feedback: Some(feedback),
                reviewer_type: Some(ReviewerType::Human),
                escalation_reason: None,
                review_model: None,
            },
        );

        Ok(())
    }

    pub fn session_manager(&self) -> &Arc<SessionManager> {
        &self.sessions
    }
}
