//! In-memory session manager for code review sessions.
//!
//! Sessions are ephemeral — they live only as long as the daemon process.
//! Each `request_review` MCP call creates exactly one session; the review loop
//! updates it in-place. The escalation UI reads sessions from this store.

use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use uuid::Uuid;

/// Status of a review session lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewStatus {
    Pending,
    Approved,
    NeedsRevision,
    Escalated,
}

impl ReviewStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            ReviewStatus::Pending => "pending",
            ReviewStatus::Approved => "approved",
            ReviewStatus::NeedsRevision => "needs_revision",
            ReviewStatus::Escalated => "escalated",
        }
    }
}

impl std::fmt::Display for ReviewStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why a session was escalated to a human.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EscalationReason {
    MaxIterations,
    LlmError,
    ConnectionFailed,
}

/// Who produced the last feedback.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewerType {
    Llm,
    Human,
}

/// A single code-review session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub task_id: String,
    pub status: ReviewStatus,
    pub summary: String,
    pub details: Option<String>,
    pub conversation_history: Vec<String>,
    pub llm_feedback: Option<String>,
    pub human_feedback: Option<String>,
    pub escalation_reason: Option<EscalationReason>,
    pub iteration_count: u32,
    pub reviewer_type: Option<ReviewerType>,
    /// Which model/provider handled the review LLM calls (e.g. "cloud via manifest").
    pub review_model: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    /// Working directory of the process that started the review.
    pub cwd: String,
}

impl Session {
    fn new(task_id: String, summary: String, details: Option<String>, conversation_history: Vec<String>, cwd: String) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Session {
            id: Uuid::new_v4().to_string(),
            task_id,
            status: ReviewStatus::Pending,
            summary,
            details,
            conversation_history,
            llm_feedback: None,
            human_feedback: None,
            escalation_reason: None,
            iteration_count: 0,
            reviewer_type: None,
            review_model: None,
            created_at: now.clone(),
            updated_at: now,
            cwd,
        }
    }
}

/// Update applied to a session after a review iteration.
pub struct SessionUpdate {
    pub status: Option<ReviewStatus>,
    pub feedback: Option<String>,
    pub reviewer_type: Option<ReviewerType>,
    pub escalation_reason: Option<EscalationReason>,
    /// Model/provider string to record on the session (set once on first review iteration).
    pub review_model: Option<String>,
}

/// Thread-safe in-memory session store.
pub struct SessionManager {
    sessions: Mutex<HashMap<String, Session>>,
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionManager {
    pub fn new() -> Self {
        SessionManager {
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Create and store a new session, returning a clone.
    pub fn create_session(
        &self,
        task_id: String,
        summary: String,
        details: Option<String>,
        conversation_history: Vec<String>,
        cwd: String,
    ) -> Session {
        let session = Session::new(task_id, summary, details, conversation_history, cwd);
        let mut map = self.sessions.lock().unwrap();
        map.insert(session.id.clone(), session.clone());
        session
    }

    /// Retrieve a session by ID.
    pub fn get_session(&self, id: &str) -> Option<Session> {
        self.sessions.lock().unwrap().get(id).cloned()
    }

    /// Apply an update to a session. No-op if session not found.
    pub fn update_session(&self, id: &str, update: SessionUpdate) {
        let mut map = self.sessions.lock().unwrap();
        let session = match map.get_mut(id) {
            Some(s) => s,
            None => return,
        };

        if let Some(status) = update.status {
            session.status = status;
        }
        if let Some(rt) = update.reviewer_type {
            session.reviewer_type = Some(rt.clone());
            if let Some(feedback) = update.feedback {
                match rt {
                    ReviewerType::Human => session.human_feedback = Some(feedback),
                    ReviewerType::Llm => session.llm_feedback = Some(feedback),
                }
            }
        } else if let Some(feedback) = update.feedback {
            // No reviewer type specified — store as LLM feedback by default
            session.llm_feedback = Some(feedback);
        }
        if let Some(reason) = update.escalation_reason {
            session.escalation_reason = Some(reason);
        }
        // Only set review_model if provided and not already set (first iteration wins).
        if let Some(rm) = update.review_model {
            if session.review_model.is_none() {
                session.review_model = Some(rm);
            }
        }
        session.updated_at = chrono::Utc::now().to_rfc3339();
    }

    /// Increment the iteration counter for a session.
    pub fn increment_iteration(&self, id: &str) {
        let mut map = self.sessions.lock().unwrap();
        if let Some(session) = map.get_mut(id) {
            session.iteration_count += 1;
            session.updated_at = chrono::Utc::now().to_rfc3339();
        }
    }

    /// List all sessions (snapshot).
    pub fn list_sessions(&self) -> Vec<Session> {
        self.sessions.lock().unwrap().values().cloned().collect()
    }

    /// Delete a session by ID. Used by the human-resolve flow.
    pub fn delete_session(&self, id: &str) {
        self.sessions.lock().unwrap().remove(id);
    }
}

/// Shared handle passed around by the daemon.
pub type SharedSessionManager = Arc<SessionManager>;
