//! Tests for the session manager and review service.
//!
//! The review loop requires a live Router (which needs Bonsai + providers),
//! so we test the session manager and the supporting logic in isolation.

use brainrouter::session::{
    EscalationReason, ReviewStatus, ReviewerType, SessionManager, SessionUpdate,
};
use std::sync::Arc;

#[test]
fn session_create_and_retrieve() {
    let manager = SessionManager::new();

    let session = manager.create_session(
        "task-001".to_string(),
        "Implement feature X".to_string(),
        Some("Details here".to_string()),
        vec!["history line 1".to_string()],
    );

    assert!(!session.id.is_empty());
    assert_eq!(session.task_id, "task-001");
    assert_eq!(session.summary, "Implement feature X");
    assert_eq!(session.details.as_deref(), Some("Details here"));
    assert_eq!(session.status, ReviewStatus::Pending);
    assert_eq!(session.iteration_count, 0);

    let retrieved = manager.get_session(&session.id).expect("session should exist");
    assert_eq!(retrieved.id, session.id);
    assert_eq!(retrieved.task_id, "task-001");
}

#[test]
fn session_update_status_and_feedback() {
    let manager = SessionManager::new();

    let session = manager.create_session(
        "task-002".to_string(),
        "Fix bug Y".to_string(),
        None,
        vec![],
    );

    manager.update_session(
        &session.id,
        SessionUpdate {
            status: Some(ReviewStatus::NeedsRevision),
            feedback: Some("Add more tests".to_string()),
            reviewer_type: Some(ReviewerType::Llm),
            escalation_reason: None,
        },
    );

    let updated = manager.get_session(&session.id).unwrap();
    assert_eq!(updated.status, ReviewStatus::NeedsRevision);
    assert_eq!(updated.llm_feedback.as_deref(), Some("Add more tests"));
    assert!(updated.human_feedback.is_none());
}

#[test]
fn session_human_feedback_goes_to_correct_field() {
    let manager = SessionManager::new();

    let session = manager.create_session(
        "task-003".to_string(),
        "Deploy Z".to_string(),
        None,
        vec![],
    );

    manager.update_session(
        &session.id,
        SessionUpdate {
            status: Some(ReviewStatus::Approved),
            feedback: Some("LGTM".to_string()),
            reviewer_type: Some(ReviewerType::Human),
            escalation_reason: None,
        },
    );

    let updated = manager.get_session(&session.id).unwrap();
    assert_eq!(updated.status, ReviewStatus::Approved);
    assert_eq!(updated.human_feedback.as_deref(), Some("LGTM"));
    assert!(updated.llm_feedback.is_none());
}

#[test]
fn session_escalation_reason_stored() {
    let manager = SessionManager::new();

    let session = manager.create_session(
        "task-004".to_string(),
        "Refactor".to_string(),
        None,
        vec![],
    );

    manager.update_session(
        &session.id,
        SessionUpdate {
            status: Some(ReviewStatus::Escalated),
            feedback: Some("Too many failures".to_string()),
            reviewer_type: Some(ReviewerType::Llm),
            escalation_reason: Some(EscalationReason::MaxIterations),
        },
    );

    let updated = manager.get_session(&session.id).unwrap();
    assert_eq!(updated.status, ReviewStatus::Escalated);
    assert_eq!(
        updated.escalation_reason,
        Some(EscalationReason::MaxIterations)
    );
}

#[test]
fn increment_iteration_counter() {
    let manager = SessionManager::new();

    let session = manager.create_session(
        "task-005".to_string(),
        "Iterate".to_string(),
        None,
        vec![],
    );

    assert_eq!(session.iteration_count, 0);

    manager.increment_iteration(&session.id);
    manager.increment_iteration(&session.id);

    let updated = manager.get_session(&session.id).unwrap();
    assert_eq!(updated.iteration_count, 2);
}

#[test]
fn list_sessions_returns_all() {
    let manager = Arc::new(SessionManager::new());

    let s1 = manager.create_session("t1".to_string(), "S1".to_string(), None, vec![]);
    let s2 = manager.create_session("t2".to_string(), "S2".to_string(), None, vec![]);

    let sessions = manager.list_sessions();
    assert_eq!(sessions.len(), 2);

    let ids: Vec<_> = sessions.iter().map(|s| s.id.clone()).collect();
    assert!(ids.contains(&s1.id));
    assert!(ids.contains(&s2.id));
}

#[test]
fn get_nonexistent_session_returns_none() {
    let manager = SessionManager::new();
    assert!(manager.get_session("does-not-exist").is_none());
}

#[test]
fn update_nonexistent_session_is_noop() {
    let manager = SessionManager::new();
    // Should not panic
    manager.update_session(
        "nonexistent",
        SessionUpdate {
            status: Some(ReviewStatus::Approved),
            feedback: None,
            reviewer_type: None,
            escalation_reason: None,
        },
    );
}

#[test]
fn delete_session_removes_it() {
    let manager = SessionManager::new();

    let session = manager.create_session("t".to_string(), "s".to_string(), None, vec![]);
    assert!(manager.get_session(&session.id).is_some());

    manager.delete_session(&session.id);
    assert!(manager.get_session(&session.id).is_none());
}
