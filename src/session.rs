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
use tokio::sync::Notify;
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
    /// LLM voluntarily returned "escalated" status (not an error).
    LlmEscalated,
    ConnectionFailed,
}

impl EscalationReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            EscalationReason::MaxIterations => "max_iterations",
            EscalationReason::LlmError => "llm_error",
            EscalationReason::LlmEscalated => "llm_escalated",
            EscalationReason::ConnectionFailed => "connection_failed",
        }
    }
}

/// Who produced the last feedback.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewerType {
    Llm,
    Human,
}

impl ReviewerType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ReviewerType::Llm => "llm",
            ReviewerType::Human => "human",
        }
    }
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
    /// Accumulated review-loop turns (prompt/response pairs), persisted so a
    /// "continue" run can seed its history instead of starting from scratch.
    pub llm_turns: Vec<String>,
    pub llm_feedback: Option<String>,
    pub human_feedback: Option<String>,
    pub escalation_reason: Option<EscalationReason>,
    pub iteration_count: u32,
    pub reviewer_type: Option<ReviewerType>,
    /// Which model/provider handled the review LLM calls (e.g. "cloud via manifest").
    pub review_model: Option<String>,
    /// Immutable reviewer policy for this session, including continuations.
    #[serde(default)]
    pub review_config: Option<crate::config::ReviewConfig>,
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
            llm_turns: Vec::new(),
            llm_feedback: None,
            human_feedback: None,
            escalation_reason: None,
            iteration_count: 0,
            reviewer_type: None,
            review_model: None,
            review_config: None,
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
    /// Latest model/provider string, including requested policy and actual fallback.
    pub review_model: Option<String>,
    /// Full accumulated review-loop history (replaces the session's copy).
    pub llm_turns: Option<Vec<String>>,
}

/// Thread-safe in-memory session store.
pub struct SessionManager {
    sessions: Mutex<HashMap<String, Session>>,
    /// Per-session notifiers: fired when a session's status changes (e.g. human resolves).
    notifiers: Mutex<HashMap<String, (Arc<Notify>, usize)>>,
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
            notifiers: Mutex::new(HashMap::new()),
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
        self.create_session_with_config(task_id, summary, details, conversation_history, cwd, None)
    }

    pub fn create_session_with_config(
        &self,
        task_id: String,
        summary: String,
        details: Option<String>,
        conversation_history: Vec<String>,
        cwd: String,
        review_config: Option<crate::config::ReviewConfig>,
    ) -> Session {
        let mut session = Session::new(task_id, summary, details, conversation_history, cwd);
        session.review_config = review_config;
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
        // Keep the latest actual route visible, including a later fallback.
        if let Some(rm) = update.review_model {
            session.review_model = Some(rm);
        }
        if let Some(turns) = update.llm_turns {
            session.llm_turns = turns;
        }
        session.updated_at = chrono::Utc::now().to_rfc3339();
        // Notify any waiter (e.g. start_review blocked on human escalation).
        let notifier = self.notifiers.lock().unwrap().get(id).cloned();
        if let Some((notifier, _)) = notifier {
            notifier.notify_waiters();
        }
    }

    /// Increment the iteration counter for a session.
    pub fn increment_iteration(&self, id: &str) {
        let mut map = self.sessions.lock().unwrap();
        if let Some(session) = map.get_mut(id) {
            session.iteration_count += 1;
            session.updated_at = chrono::Utc::now().to_rfc3339();
        }
    }
    /// Register a `Notify` for a session so that `start_review` can be woken
    /// when human feedback arrives via `update_session`.
    pub fn register_notifier(&self, session_id: &str) -> Arc<Notify> {
        let mut notifiers = self.notifiers.lock().unwrap();
        let (notifier, registrations) = notifiers
            .entry(session_id.to_string())
            .or_insert_with(|| (Arc::new(Notify::new()), 0));
        *registrations += 1;
        Arc::clone(notifier)
    }

    /// Release one notifier registration, removing the entry after the last waiter.
    pub fn remove_notifier(&self, session_id: &str) {
        let mut notifiers = self.notifiers.lock().unwrap();
        let should_remove = match notifiers.get_mut(session_id) {
            Some((_, registrations)) if *registrations > 1 => {
                *registrations -= 1;
                false
            }
            Some(_) => true,
            None => false,
        };
        if should_remove {
            notifiers.remove(session_id);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn notifier_registrations_share_wakeups_until_last_waiter_leaves() {
        let manager = Arc::new(SessionManager::new());
        let session = manager.create_session(
            "task".into(),
            "summary".into(),
            None,
            Vec::new(),
            ".".into(),
        );
        let first = manager.register_notifier(&session.id);
        let second = manager.register_notifier(&session.id);
        assert!(Arc::ptr_eq(&first, &second));

        let first_wait = {
            let notifier = Arc::clone(&first);
            tokio::spawn(async move { notifier.notified().await })
        };
        let second_wait = {
            let notifier = Arc::clone(&second);
            tokio::spawn(async move { notifier.notified().await })
        };
        tokio::task::yield_now().await;
        manager.update_session(
            &session.id,
            SessionUpdate {
                status: Some(ReviewStatus::Approved),
                feedback: Some("lgtm".into()),
                reviewer_type: Some(ReviewerType::Human),
                escalation_reason: None,
                review_model: None,
                llm_turns: None,
            },
        );
        tokio::time::timeout(std::time::Duration::from_secs(1), first_wait)
            .await
            .expect("first waiter should wake")
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), second_wait)
            .await
            .expect("second waiter should wake")
            .unwrap();

        manager.remove_notifier(&session.id);
        let still_registered = manager.register_notifier(&session.id);
        assert!(Arc::ptr_eq(&first, &still_registered));
        manager.remove_notifier(&session.id);
        manager.remove_notifier(&session.id);
        let replacement = manager.register_notifier(&session.id);
        assert!(!Arc::ptr_eq(&first, &replacement));
    }
}
