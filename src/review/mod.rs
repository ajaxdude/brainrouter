//! Review service — orchestrates the code review loop.
//!
//! `ReviewService` is the public face. Callers (HTTP handler, MCP handler)
//! call `start_review` and receive a result once the loop completes or escalates.

pub mod context;
pub mod prompt;
pub mod review_loop;

use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn};

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
    config: std::sync::Mutex<ReviewConfig>,
    state_path: PathBuf,
}

impl ReviewService {
    pub fn new(
        router: Arc<Router>,
        sessions: Arc<SessionManager>,
        mut config: ReviewConfig,
    ) -> Self {
        // Resolve state path: respect XDG_CONFIG_HOME, fallback to ~/.config/brainrouter/
        let state_path = std::env::var("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                std::env::var("HOME")
                    .map(|h| PathBuf::from(h).join(".config"))
                    .unwrap_or_else(|_| {
                        warn!("$HOME not set, falling back to current directory for state");
                        PathBuf::from(".")
                    })
            })
            .join("brainrouter/review_state.json");

        // Load existing state if available
        if let Ok(content) = std::fs::read_to_string(&state_path) {
            match serde_json::from_str::<ReviewConfig>(&content) {
                Ok(saved) => {
                    info!(path = %state_path.display(), "Loaded saved review configuration");
                    // Intentional partial restore: only UI-driven overrides (forced_mode,
                    // forced_model) are persisted and reloaded. max_iterations is always
                    // taken from brainrouter.yaml; persisting it would let a runtime
                    // change silently shadow the config file value across restarts.
                    config.forced_mode = saved.forced_mode;
                    config.forced_model = saved.forced_model;
                }
                Err(e) => warn!(error = %e, path = %state_path.display(), "Failed to parse saved review configuration, using defaults"),
            }
        }

        ReviewService {
            router,
            sessions,
            config: std::sync::Mutex::new(config),
            state_path,
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
        cwd: String,
    ) -> Result<RequestReviewResult> {
        // Create the session first — the agent gets the ID in the response regardless
        // of whether the review loop succeeds or fails.
        let session = self.sessions.create_session(
            task_id.clone(),
            summary.clone(),
            details.clone(),
            conversation_history,
            cwd.clone(), // clone needed: cwd is moved into Session, but also needed for run_loop below
        );
        info!(session_id = %session.id, task_id = %task_id, "Created review session");

        // Clone current config snapshot for this review run
        let config_snapshot = self.get_config();

        let result = run_loop(
            &session.id,
            &task_id,
            &summary,
            details.as_deref(),
            &self.router,
            &self.sessions,
            &config_snapshot,
            &cwd,
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

    /// Get a copy of the current review configuration.
    pub fn get_config(&self) -> ReviewConfig {
        self.config.lock().unwrap().clone()
    }

    /// Update the review configuration globally for new sessions and persist to disk.
    pub async fn update_config(&self, new_config: ReviewConfig) {
        {
            let mut config = self.config.lock().unwrap();
            *config = new_config.clone();
        }
        
        // Persist to disk asynchronously to avoid blocking the executor
        let state_path = self.state_path.clone();
        // Fire-and-forget: the JoinHandle is intentionally dropped. The task
        // is detached and continues to completion. Panics inside the closure
        // are swallowed by Tokio's default panic handler (logged, not fatal).
        // This is acceptable for a best-effort disk persist — in-memory config
        // is already updated above and that is the authoritative state.
        let _handle = tokio::task::spawn_blocking(move || {
            if let Some(parent) = state_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            
            match serde_json::to_string_pretty(&new_config) {
                Ok(json) => {
                    // Atomic write: write to temp then rename
                    let tmp_path = state_path.with_extension("json.tmp");
                    if let Err(e) = std::fs::write(&tmp_path, json) {
                        warn!(error = %e, path = %tmp_path.display(), "Failed to write temporary review configuration");
                        return;
                    }
                    if let Err(e) = std::fs::rename(&tmp_path, &state_path) {
                        warn!(error = %e, from = %tmp_path.display(), to = %state_path.display(), "Failed to rename temporary review configuration");
                    }
                }
                Err(e) => warn!(error = %e, "Failed to serialize review configuration"),
            }
        });
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
