//! `review.fallback_selector` — infra reviewer failover.
//!
//! Contract (config.rs `ReviewConfig::fallback_selector`): when the primary
//! reviewer backend fails, the review retries ONCE on the configured cloud
//! selector; a second failure escalates; a healthy primary never dials the
//! fallback; a local review never migrates. Driven through the real
//! ReviewService → review_loop → Router path against synthetic providers.

use brainrouter::{
    classifier::Classifier,
    config::{DecisionKind, DecisionModelConfig, NudgeBudgets, ReviewConfig},
    health::HealthTracker,
    inference_state::InferenceTracker,
    providers::{BudgetMode, ProviderKind, ProviderRecord, ProviderStore},
    review::ReviewService,
    roles::{ModelChoice, Role, RoleCardStore, GLOBAL_SCOPE},
    router::{Router, RouterArgs},
    routing_events::RoutingEvents,
    session::{ReviewStatus, SessionManager},
};
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{body::Incoming, server::conn::http1, service::service_fn, Request, Response};
use hyper_util::rt::TokioIo;
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    fs,
    sync::{
        atomic::{AtomicBool, AtomicU8},
        Arc, Mutex,
    },
};
use tokio::{net::TcpListener, task::JoinHandle};

/// Canned reviewer answer the review loop parses as a verdict.
fn approved_sse(model: &Value) -> String {
    let chunk = json!({
        "id":"fb","model":model,
        "choices":[{"index":0,"delta":{"content":"{\"status\":\"approved\",\"feedback\":\"failover approved\"}"},"finish_reason":null}]
    });
    format!("data: {chunk}\n\ndata: [DONE]\n\n")
}

/// Synthetic provider fleet: any path containing `/down` fails with 503;
/// everything else returns the approved-review SSE.
struct MockFleet {
    url: String,
    seen: Arc<Mutex<Vec<(String, String)>>>,
    task: JoinHandle<()>,
}

impl MockFleet {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&seen);
        let task = tokio::spawn(async move {
            let mut connections = tokio::task::JoinSet::new();
            loop {
                tokio::select! {
                    connection = listener.accept() => {
                        let (stream, _) = connection.unwrap();
                        let captured = Arc::clone(&captured);
                        connections.spawn(async move {
                            let service = service_fn(move |request: Request<Incoming>| {
                                let captured = Arc::clone(&captured);
                                async move {
                                    let path = request.uri().path().to_string();
                                    let bytes = request.collect().await.unwrap().to_bytes();
                                    let body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
                                    captured.lock().unwrap().push((
                                        path.clone(),
                                        body["model"].as_str().unwrap_or_default().to_string(),
                                    ));
                                    let (status, text) = if path.contains("/down") {
                                        (503, "{}".to_string())
                                    } else {
                                        (200, approved_sse(&body["model"]))
                                    };
                                    Ok::<_, Infallible>(
                                        Response::builder().status(status)
                                            .body(Full::new(Bytes::from(text))).unwrap(),
                                    )
                                }
                            });
                            http1::Builder::new().serve_connection(TokioIo::new(stream), service).await.unwrap();
                        });
                    }
                    Some(result) = connections.join_next() => { result.unwrap(); }
                }
            }
        });
        Self { url, seen, task }
    }

    fn calls(&self) -> Vec<(String, String)> {
        self.seen
            .lock()
            .unwrap()
            .iter()
            .filter(|(path, _)| path.ends_with("/chat/completions"))
            .cloned()
            .collect()
    }
}

impl Drop for MockFleet {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn record(id: &str, url: String) -> ProviderRecord {
    ProviderRecord {
        id: id.into(),
        kind: ProviderKind::OpenaiCompatible,
        base_url: url,
        credential_id: None,
        auto: false,
        models: Vec::new(),
        lan_peer: false,
        enabled: true,
        monthly_budget_usd: None,
        budget_mode: BudgetMode::Warn,
    }
}

fn fleet(url: &str) -> Arc<ProviderStore> {
    Arc::new(ProviderStore::in_memory_with(vec![
        record("syn", format!("{url}/syn/v1")),
        record("down", format!("{url}/down/v1")),
        record("deep", format!("{url}/deep/v1")),
    ]))
}

fn router(fleet_url: &str, cards: Arc<RoleCardStore>) -> Arc<Router> {
    let classifier = Arc::new(Classifier::new(
        format!("{fleet_url}/classifier"),
        "default-local".into(),
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
        None,
    ));
    let slot = Arc::new(brainrouter::decision::DecisionSlot::new(
        DecisionModelConfig {
            enabled: false,
            kind: DecisionKind::OpenaiCompatible,
            base_url: None,
            model: None,
            prompt_profile: None,
            timeout_ms: None,
            ..Default::default()
        },
        classifier,
        None,
    ));
    Arc::new(Router::new(RouterArgs {
        slot,
        cards,
        providers: fleet(fleet_url),
        credentials: Arc::new(brainrouter::credentials::CredentialStore::in_memory()),
        budget: Arc::new(brainrouter::provider_budget::ProviderBudget::in_memory()),
        llama_swap: Arc::new(brainrouter::provider::openai::OpenAiProvider::new(
            "llama-swap".into(),
            format!("{fleet_url}/local/v1"),
            None,
        )),
        fallback_model: "default-local".into(),
        local_models: vec!["known-local".into()],
        subs_model: None,
        health: Arc::new(HealthTracker::new()),
        routing_events: Arc::new(RoutingEvents::new()),
        local_system_prompt: None,
        inference_tracker: Arc::new(InferenceTracker::new()),
        nudge_budgets: NudgeBudgets::default(),
        nudge_enabled: Arc::new(AtomicBool::new(false)),
        nudge_tier: Arc::new(AtomicU8::new(0)),
        nudge_model_key: None,
        prompt_rewrite: Arc::new(AtomicBool::new(false)),
        llama_swap_config_path: None,
        serving_identities: None,
    }))
}

struct Dir(std::path::PathBuf);

impl Dir {
    fn new() -> (Self, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!("br-fb-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&path).unwrap();
        (Self(path.clone()), path)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        // Best effort — a leftover temp dir must never fail a test.
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// `human_resolves`: escalated reviews park until a human resolves the
/// session (the MCP wait-for-human contract), so the escalation tests run a
/// background "human" that resolves as soon as the session escalates.
async fn run_review(
    fleet_url: &str,
    review_selector: &str,
    config: ReviewConfig,
    human_resolves: bool,
) -> (
    brainrouter::review::RequestReviewResult,
    Arc<SessionManager>,
) {
    let (_dir, cwd) = Dir::new();
    let cards = Arc::new(RoleCardStore::memory(Default::default()));
    cards
        .set_assignment(GLOBAL_SCOPE, Role::Review, review_selector)
        .unwrap();
    let sessions = Arc::new(SessionManager::new());
    let service = Arc::new(ReviewService::new(
        router(fleet_url, cards.clone()),
        sessions.clone(),
        cards,
        config,
        Default::default(),
        fleet(fleet_url),
        "default-local".to_string(),
    ));
    if human_resolves {
        let svc = Arc::clone(&service);
        let watchers = Arc::clone(&sessions);
        tokio::spawn(async move {
            for _ in 0..5000 {
                let escalated = watchers
                    .list_sessions()
                    .into_iter()
                    .find(|s| s.task_id == "fb-test" && s.status == ReviewStatus::Escalated);
                if let Some(s) = escalated {
                    svc.resolve_session(&s.id, "test human: ok".into()).unwrap();
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            }
        });
    }
    let result = service
        .start_review(
            "fb-test".into(),
            "fallback selector coverage".into(),
            None,
            vec![],
            cwd.to_str().unwrap().into(),
        )
        .await
        .unwrap();
    (result, sessions)
}

fn config_with_fallback(fallback: Option<&str>) -> ReviewConfig {
    ReviewConfig {
        fallback_selector: fallback.map(str::to_string),
        ..Default::default()
    }
}

#[tokio::test]
async fn dead_primary_retries_once_on_the_fallback_selector() {
    let fleet = MockFleet::start().await;
    let (result, sessions) = run_review(
        &fleet.url,
        "cloud/down/dead-reviewer",
        config_with_fallback(Some("cloud/deep/model-b")),
        false,
    )
    .await;
    assert_eq!(
        result.status,
        ReviewStatus::Approved,
        "fallback verdict must complete the review: {:?}",
        result.feedback
    );
    assert_eq!(
        fleet.calls(),
        vec![
            (
                "/down/v1/chat/completions".to_string(),
                "dead-reviewer".into()
            ),
            ("/deep/v1/chat/completions".to_string(), "model-b".into()),
        ],
        "exactly one primary attempt then one fallback hop"
    );
    let provenance = sessions
        .get_session(&result.session_id)
        .unwrap()
        .review_model
        .expect("review_model recorded");
    assert!(
        provenance.contains("fell back to cloud/deep/model-b"),
        "session must record the failover hop: {provenance}"
    );
}

#[tokio::test]
async fn healthy_primary_never_dials_the_fallback() {
    let fleet = MockFleet::start().await;
    let (result, sessions) = run_review(
        &fleet.url,
        "cloud/syn/healthy-reviewer",
        config_with_fallback(Some("cloud/deep/model-b")),
        false,
    )
    .await;
    assert_eq!(result.status, ReviewStatus::Approved);
    assert_eq!(
        fleet.calls(),
        vec![(
            "/syn/v1/chat/completions".to_string(),
            "healthy-reviewer".into()
        )],
        "fallback selector must be inert while the primary is healthy"
    );
    assert!(
        !sessions
            .get_session(&result.session_id)
            .unwrap()
            .review_model
            .unwrap()
            .contains("fell back"),
        "no failover note without a failure"
    );
}

#[tokio::test]
async fn double_failure_escalates_naming_both_backends() {
    let fleet = MockFleet::start().await;
    let (result, sessions) = run_review(
        &fleet.url,
        "cloud/down/dead-reviewer",
        config_with_fallback(Some("cloud/down/model-b")),
        true,
    )
    .await;
    let session = sessions.get_session(&result.session_id).unwrap();
    assert_eq!(
        session
            .escalation_reason
            .map(|r| format!("{r:?}"))
            .as_deref(),
        Some("LlmError"),
        "double backend failure must land as an LLM-error escalation"
    );
    let feedback = session.llm_feedback.unwrap_or_default();
    assert!(
        feedback.contains("fallback `cloud/down/model-b` failed too"),
        "escalation must report the fallback attempt: {feedback}"
    );
    assert_eq!(
        fleet
            .calls()
            .iter()
            .filter(|c| c.0.ends_with("/chat/completions"))
            .count(),
        2,
        "one primary + one fallback attempt, then stop"
    );
}

#[tokio::test]
async fn no_fallback_selector_keeps_the_single_attempt_contract() {
    let fleet = MockFleet::start().await;
    let (result, sessions) = run_review(
        &fleet.url,
        "cloud/down/dead-reviewer",
        ReviewConfig::default(),
        true,
    )
    .await;
    let session = sessions.get_session(&result.session_id).unwrap();
    assert!(
        session
            .llm_feedback
            .unwrap_or_default()
            .contains("LLM error"),
        "terminal single-attempt escalation with no fallback configured"
    );
    assert_eq!(
        fleet.calls(),
        vec![(
            "/down/v1/chat/completions".to_string(),
            "dead-reviewer".into()
        )],
        "unset fallback_selector: today's terminal single attempt"
    );
}

#[test]
fn fallback_selector_validation() {
    // Qualified and plain cloud selectors are accepted; anything that could
    // migrate a local review is rejected.
    assert!(
        config_with_fallback(Some("cloud/deepseek-ai/DeepSeek-V4-Flash"))
            .validate()
            .is_ok()
    );
    assert!(config_with_fallback(Some("cloud/deep/model-b"))
        .validate()
        .is_ok());
    let bare = config_with_fallback(Some("some-local-model"))
        .validate()
        .unwrap_err();
    assert!(bare.to_string().contains("cloud"), "{bare}");
    let local = config_with_fallback(Some("local/model"))
        .validate()
        .unwrap_err();
    assert!(local.to_string().contains("cloud"), "{local}");
    assert!(config_with_fallback(None).validate().is_ok());
}

#[test]
fn fallback_selector_defaults_off_and_parses_from_yaml() {
    let plain: ReviewConfig = serde_yaml::from_str("max_iterations: 4\n").unwrap();
    assert!(plain.fallback_selector.is_none());
    let with: ReviewConfig =
        serde_yaml::from_str("fallback_selector: cloud/deepseek-ai/DeepSeek-V4-Flash\n").unwrap();
    assert_eq!(
        with.fallback_selector.as_deref(),
        Some("cloud/deepseek-ai/DeepSeek-V4-Flash")
    );
    assert!(with.validate().is_ok());
    let _ = ModelChoice::from_selector("cloud/x").unwrap(); // selector path sanity
}
