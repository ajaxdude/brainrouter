//! Review service — orchestrates the code review loop.
//!
//! `ReviewService` is the public face. Callers (HTTP handler, MCP handler)
//! call `start_review` and receive a result once the loop completes or escalates.

pub mod context;
pub mod prompt;
pub mod review_loop;

use anyhow::Result;
use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};
use tracing::{info, warn};

use crate::{
    config::ReviewConfig,
    router::Router,
    routing_profile::{ModelChoice, ProfileStore, RoutingPreset, RoutingProfile},
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
    preferences: Arc<ProfileStore>,
    active_reviews: Arc<Mutex<HashSet<String>>>,
}

struct ActiveReviewGuard {
    session_id: String,
    active_reviews: Arc<Mutex<HashSet<String>>>,
}

impl Drop for ActiveReviewGuard {
    fn drop(&mut self) {
        self.active_reviews.lock().unwrap().remove(&self.session_id);
    }
}

impl ReviewService {
    pub fn new(
        router: Arc<Router>,
        sessions: Arc<SessionManager>,
        config: ReviewConfig,
    ) -> Self {
        let preferences = router.profiles().cloned().unwrap_or_else(|| {
            Arc::new(ProfileStore::memory(RoutingProfile {
                preset: RoutingPreset::Custom,
                main: ModelChoice::Auto,
                reviewer: config.model_choice().expect("validated review configuration"),
                subagent_model: None,
            }, config.max_iterations).expect("validated review configuration"))
        });

        ReviewService {
            router,
            sessions,
            preferences,
            active_reviews: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    fn begin_review(&self, session_id: &str) -> Result<ActiveReviewGuard> {
        let mut active = self.active_reviews.lock().unwrap();
        if !active.insert(session_id.to_string()) {
            anyhow::bail!("Review session {session_id} is already running");
        }
        Ok(ActiveReviewGuard {
            session_id: session_id.to_string(),
            active_reviews: Arc::clone(&self.active_reviews),
        })
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
        cwd: String,
    ) -> Result<RequestReviewResult> {
        // Create the session first — the agent gets the ID in the response regardless
        // of whether the review loop succeeds or fails.
        let config_snapshot = self.get_config();
        let session = self.sessions.create_session_with_config(
            task_id.clone(),
            summary.clone(),
            details.clone(),
            conversation_history,
            cwd.clone(), // clone needed: cwd is moved into Session, but also needed for run_loop below
            Some(config_snapshot.clone()),
        );
        info!(session_id = %session.id, task_id = %task_id, "Created review session");
        let active_review = self.begin_review(&session.id)?;

        // Register a notifier *before* the loop so we never miss a notification
        // that arrives between the loop returning Escalated and our first wait.
        let notifier = self.sessions.register_notifier(&session.id);

        let result = run_loop(
            &session.id,
            &task_id,
            &summary,
            details.as_deref(),
            &session.conversation_history,
            &self.router,
            &self.sessions,
            &config_snapshot,
            &cwd,
        )
        .await;
        drop(active_review);
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                self.sessions.remove_notifier(&session.id);
                return Err(error);
            }
        };

        let (loop_status, loop_feedback, loop_reviewer_type, iteration_count, loop_session_id) = (
            result.status,
            result.feedback,
            result.reviewer_type,
            result.iteration_count,
            result.session_id,
        );

        let (final_status, final_feedback, reviewer_type) =
            if loop_status == ReviewStatus::Escalated {
                info!(session_id = %session.id, "Review escalated — waiting for human resolution");
                loop {
                    // Register the notified() future BEFORE checking state so we
                    // cannot miss a notification that fires between the check and
                    // the await.
                    let next = notifier.notified();
                    tokio::pin!(next);
                    next.as_mut().enable();
                    // Re-read session to get the latest status.
                    if let Some(s) = self.sessions.get_session(&session.id) {
                        match s.status {
                            ReviewStatus::Escalated => {
                                // Still escalated — wait for the next signal.
                                next.await;
                            }
                            ReviewStatus::Approved => {
                                info!(session_id = %session.id, "Human approved the review");
                                break (
                                    ReviewStatus::Approved,
                                    s.human_feedback.unwrap_or_else(|| "lgtm".to_string()),
                                    ReviewerType::Human,
                                );
                            }
                            ReviewStatus::NeedsRevision => {
                                info!(session_id = %session.id, "Human requested revision");
                                break (
                                    ReviewStatus::NeedsRevision,
                                    s.human_feedback.unwrap_or_default(),
                                    ReviewerType::Human,
                                );
                            }
                            ReviewStatus::Pending => {
                                // Should not normally happen; wait for the next signal.
                                next.await;
                            }
                        }
                    } else {
                        // Session was deleted — treat as aborted.
                        warn!(session_id = %session.id, "Session disappeared while waiting for human");
                        break (
                            ReviewStatus::Escalated,
                            "Session was deleted before human resolved it.".to_string(),
                            ReviewerType::Human,
                        );
                    }
                }
            } else {
                (loop_status, loop_feedback, loop_reviewer_type)
            };

        // Clean up the notifier regardless of outcome.
        self.sessions.remove_notifier(&session.id);

        // loop_session_id and session.id should match; use session.id as primary.
        let _ = loop_session_id;

        Ok(RequestReviewResult {
            status: final_status,
            feedback: final_feedback,
            session_id: session.id,
            iteration_count,
            reviewer_type,
        })
    }
    /// Get a copy of the current review configuration.
    pub fn get_config(&self) -> ReviewConfig {
        self.preferences.review_config()
    }

    pub fn preferences(&self) -> &Arc<ProfileStore> { &self.preferences }

    /// A successful response means preferences reached disk before runtime mutation.
    pub async fn update_config(&self, new_config: ReviewConfig) -> Result<()> {
        new_config.validate()?;
        let preferences = Arc::clone(&self.preferences);
        tokio::task::spawn_blocking(move || preferences.update_review(new_config)).await?
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
                llm_turns: None,
            },
        );

        Ok(())
    }

    /// Continue iterating on an existing session with additional LLM review rounds.
    /// Resets the session to Pending and runs the review loop for `extra_iterations` more rounds.
    pub async fn continue_review(
        &self,
        session_id: &str,
        extra_iterations: u32,
    ) -> Result<RequestReviewResult> {
        let session = self
            .sessions
            .get_session(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;
        let active_review = self.begin_review(session_id)?;

        // Continuations retain the original reviewer, even after profile changes.
        let mut config_snapshot = session.review_config.clone().unwrap_or_else(|| self.get_config());
        config_snapshot.max_iterations = extra_iterations;
        config_snapshot.validate()?;

        // Reset session to pending so the loop can run.
        // llm_turns is preserved — the continuation seeds from it.
        self.sessions.update_session(
            session_id,
            SessionUpdate {
                status: Some(ReviewStatus::Pending),
                feedback: None,
                reviewer_type: None,
                escalation_reason: None,
                review_model: None,
                llm_turns: None,
            },
        );

        info!(session_id, extra_iterations, "Continuing review with additional iterations");

        // Register the notifier BEFORE running the loop so we cannot miss a
        // human resolution that lands between the loop returning Escalated
        // and our first wait (same race-avoidance as start_review).
        let notifier = self.sessions.register_notifier(session_id);

        let history = if session
            .llm_turns
            .starts_with(&session.conversation_history)
        {
            session.llm_turns.clone()
        } else {
            let mut history = session.conversation_history.clone();
            history.extend(session.llm_turns.clone());
            history
        };
        let result = run_loop(
            session_id,
            &session.task_id,
            &session.summary,
            session.details.as_deref(),
            &history,
            &self.router,
            &self.sessions,
            &config_snapshot,
            &session.cwd,
        )
        .await;
        drop(active_review);
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                self.sessions.remove_notifier(session_id);
                return Err(error);
            }
        };

        let (status, feedback, reviewer_type) = if result.status == ReviewStatus::Escalated {
            info!(session_id = %session_id, "Continuation escalated — waiting for human resolution");
            loop {
                // Register the notified() future BEFORE checking state so we
                // cannot miss a notification that fires between check and await.
                let next = notifier.notified();
                tokio::pin!(next);
                next.as_mut().enable();
                if let Some(s) = self.sessions.get_session(session_id) {
                    match s.status {
                        ReviewStatus::Escalated => {
                            // Still escalated — wait for the next signal.
                            next.await;
                        }
                        ReviewStatus::Approved => {
                            info!(session_id = %session_id, "Human approved the continuation");
                            break (
                                ReviewStatus::Approved,
                                s.human_feedback.unwrap_or_else(|| "lgtm".to_string()),
                                ReviewerType::Human,
                            );
                        }
                        ReviewStatus::NeedsRevision => {
                            info!(session_id = %session_id, "Human requested revision");
                            break (
                                ReviewStatus::NeedsRevision,
                                s.human_feedback.unwrap_or_default(),
                                ReviewerType::Human,
                            );
                        }
                        ReviewStatus::Pending => {
                            // Should not normally happen; wait for the next signal.
                            next.await;
                        }
                    }
                } else {
                    // Session was deleted — treat as aborted.
                    warn!(session_id = %session_id, "Session disappeared while waiting for human");
                    break (
                        ReviewStatus::Escalated,
                        "Session was deleted before human resolved it.".to_string(),
                        ReviewerType::Human,
                    );
                }
            }
        } else {
            (result.status, result.feedback, result.reviewer_type)
        };

        // Clean up the notifier regardless of outcome.
        self.sessions.remove_notifier(session_id);

        Ok(RequestReviewResult {
            status,
            feedback,
            session_id: result.session_id,
            iteration_count: result.iteration_count,
            reviewer_type,
        })
    }

    /// Spawn the review loop in the background and return the session ID immediately.
    ///
    /// Unlike [`start_review`], this method does **not** wait for the review to
    /// finish — it returns as soon as the session is created so the caller can
    /// poll [`SessionManager::get_session`] (or GET /review/api/sessions/:id)
    /// until the status reaches a terminal state.
    ///
    /// This is the preferred entry point for MCP callers where a long-lived
    /// blocking HTTP request would exceed the client-side MCP timeout.
    pub fn start_review_async(
        self: &Arc<Self>,
        task_id: String,
        summary: String,
        details: Option<String>,
        conversation_history: Vec<String>,
        cwd: String,
    ) -> String {
        // Create the session synchronously so the caller has an ID to poll.
        let config_snapshot = self.get_config();
        let session = self.sessions.create_session_with_config(
            task_id.clone(),
            summary.clone(),
            details.clone(),
            conversation_history,
            cwd.clone(),
            Some(config_snapshot.clone()),
        );
        let session_id = session.id.clone();
        info!(session_id = %session_id, task_id = %task_id, "Created review session (async mode)");
        let active_review = self
            .begin_review(&session_id)
            .expect("new review session cannot already be active");

        // Spawn the review loop directly on the already-created session.
        // We do NOT call start_review() here — that would create a second session.
        let router = Arc::clone(&self.router);
        let sessions = Arc::clone(&self.sessions);
        let sid = session_id.clone();
        let tid = task_id.clone();
        let summ = summary.clone();
        let det = details.clone();
        let cwd2 = cwd.clone();
        let history = session.conversation_history.clone();
        let notifier = self.sessions.register_notifier(&session_id);

        tokio::spawn(async move {
            let result = run_loop(
                &sid,
                &tid,
                &summ,
                det.as_deref(),
                &history,
                &router,
                &sessions,
                &config_snapshot,
                &cwd2,
            )
            .await;
            drop(active_review);

            match result {
                Err(e) => {
                    warn!(session_id = %sid, error = %e, "Background review task failed");
                    sessions.remove_notifier(&sid);
                }
                Ok(r) if r.status == ReviewStatus::Escalated => {
                    // Wait for human resolution, same as start_review.
                    loop {
                        let next = notifier.notified();
                        tokio::pin!(next);
                        next.as_mut().enable();
                        if let Some(s) = sessions.get_session(&sid) {
                            match s.status {
                                ReviewStatus::Escalated => { next.await; }
                                ReviewStatus::Approved | ReviewStatus::NeedsRevision => break,
                                ReviewStatus::Pending => { next.await; }
                            }
                        } else {
                            break;
                        }
                    }
                    sessions.remove_notifier(&sid);
                }
                Ok(_) => {
                    sessions.remove_notifier(&sid);
                }
            }
        });

        session_id
    }

    pub fn session_manager(&self) -> &Arc<SessionManager> {
        &self.sessions
    }
}
