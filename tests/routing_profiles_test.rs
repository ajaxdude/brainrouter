//! v1 selector/reviewer test floor, v2 L3-retargeted.
//!
//! Every test keeps its v1 name and observable semantics; the machinery it
//! drove (ProfileStore/RoutingProfile/pins/presets/forced_*) was removed by
//! L3 ("Removals: routing_profile.rs + presets + CLI `routing-profile` +
//! dashboard role-chooser/preset widgets; project_pins store (migrated);
//! review.forced_mode/forced_model YAML keys (migrated)" — docs/design/
//! v2-streamlining.L3.md), so each test now drives the RoleCardStore
//! equivalents:
//!
//! - global profile ⇒ global-scope cards; `update_main` ⇒ set_assignment(global, main)
//! - project pins ⇒ project-scope cards ("per-project pins cards (project scope)")
//! - reviewer pin snapshot ⇒ review-card selector snapshotted at session start
//!   ("a session snapshots its card record at start, never re-resolves")
//! - presets ⇒ explicit per-card assignment rows ("the card assignment
//!   compiles through the SAME selector path")
//! - CLI `routing-profile` ⇒ CLI `roles set` ("mutations via daemon HTTP")
//! - legacy review-state migration ⇒ the v1 → card migration table
//!
//! Synthetic providers only: no model processes, credentials or external traffic.

use brainrouter::{
    classifier::Classifier,
    config::{self, DecisionKind, DecisionModelConfig, NudgeBudgets, ReviewConfig},
    health::HealthTracker,
    inference_state::InferenceTracker,
    provider::{openai::OpenAiProvider, ProviderResponse},
    review::ReviewService,
    migration::read_v1_state,
    roles::{migrate_v1, LegacyPreset, LegacyPin, LegacyRoutingProfile, ModelChoice, Role, RoleCardStore, GLOBAL_SCOPE},
    router::{Router, RouterArgs},
    routing_events::RoutingEvents,
    session::{ReviewStatus, SessionManager},
    types::ChatCompletionRequest,
};
use bytes::Bytes;
use futures_util::StreamExt;
use http_body_util::{BodyExt, Full};
use hyper::{body::Incoming, server::conn::http1, service::service_fn, Request, Response};
use hyper_util::rt::TokioIo;
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    fs,
    os::unix::fs::PermissionsExt,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU8},
        Arc, Mutex,
    },
};
use tokio::{net::TcpListener, task::JoinHandle};

struct Directory(PathBuf);

impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("br-cards-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[derive(Clone, Debug)]
struct Seen {
    path: String,
    body: Value,
    authorization: Option<String>,
}

struct SyntheticProviders {
    url: String,
    seen: Arc<Mutex<Vec<Seen>>>,
    task: JoinHandle<()>,
}

impl SyntheticProviders {
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
                                    let authorization = request.headers().get("authorization")
                                        .map(|v| v.to_str().unwrap().to_string());
                                    let bytes = request.collect().await.unwrap().to_bytes();
                                    let body = if bytes.is_empty() { Value::Null } else { serde_json::from_slice(&bytes).unwrap() };
                                    captured.lock().unwrap().push(Seen { path: path.clone(), body: body.clone(), authorization });
                                    let (status, text) = if path == "/api/review-config" {
                                        (200, serde_json::to_string(&ReviewConfig::default()).unwrap())
                                    } else if path.starts_with("/api/roles")
                                        || path.starts_with("/api/policy")
                                        || path.starts_with("/api/credentials")
                                    {
                                        (200, json!({"ok": true}).to_string())
                                    } else if path.starts_with("/broken") {
                                        (503, "{}".into())
                                    } else if path.ends_with("/models") {
                                        (200, json!({"data":[{"id":"model-b"},{"id":"model-a"},{"id":"model-a"}]}).to_string())
                                    } else if body["model"] == "missing-model" {
                                        (404, json!({"error":"unknown model"}).to_string())
                                    } else if path.starts_with("/classifier") {
                                        (200, json!({"choices":[{"message":{"content":"cloud"}}]}).to_string())
                                    } else {
                                        let chunk = json!({
                                            "id":"synthetic", "model":body["model"],
                                            "choices":[{"index":0,"delta":{"content":"{\"status\":\"approved\",\"feedback\":\"synthetic approval\"}"},"finish_reason":null}]
                                        });
                                        (200, format!("data: {chunk}\n\ndata: [DONE]\n\n"))
                                    };
                                    Ok::<_, Infallible>(Response::builder().status(status)
                                        .body(Full::new(Bytes::from(text))).unwrap())
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
            .filter(|r| r.path.contains("/v1/chat/completions"))
            .map(|r| (r.path.clone(), r.body["model"].as_str().unwrap_or_default().to_string()))
            .collect()
    }
}

impl Drop for SyntheticProviders {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn local(id: &str) -> ModelChoice {
    ModelChoice::Local { model: Some(id.into()) }
}
fn cloud(id: &str) -> ModelChoice {
    ModelChoice::Cloud { model: Some(id.into()) }
}

/// The v1 `profile()` floor as global-scope cards: main = local main-local,
/// reviewer = cloud vendor/reviewer-2026, subagent pool = subagent-pool.
fn cards() -> Arc<RoleCardStore> {
    let store = Arc::new(RoleCardStore::memory(Default::default()));
    store.set_assignment(GLOBAL_SCOPE, Role::Main, "main-local").unwrap();
    store.set_assignment(GLOBAL_SCOPE, Role::Review, "cloud/vendor/reviewer-2026").unwrap();
    store.set_assignment(GLOBAL_SCOPE, Role::Subagent, "subagent-pool").unwrap();
    store
}

fn provider_store(enabled: bool) -> Arc<brainrouter::providers::ProviderStore> {
    Arc::new(brainrouter::providers::ProviderStore::in_memory_with(vec![
        brainrouter::providers::ProviderRecord {
            id: "manifest".into(),
            kind: brainrouter::providers::ProviderKind::Manifest,
            base_url: "http://127.0.0.1:1/cloud/v1".into(),
            credential_id: None,
            auto: true,
            models: Vec::new(),
            lan_peer: false,
            enabled,
            monthly_budget_usd: None,
            budget_mode: brainrouter::providers::BudgetMode::Warn,
        },
    ]))
}

fn router_args(upstream: &SyntheticProviders, enabled: bool, classify: bool) -> RouterArgs {
    let classifier = Arc::new(Classifier::new(
        format!("{}/classifier", upstream.url),
        "default-local".into(),
        Arc::new(AtomicBool::new(classify)),
        Arc::new(AtomicBool::new(false)),
        None,
    ));
    let slot = Arc::new(brainrouter::decision::DecisionSlot::new(
        DecisionModelConfig {
            enabled: classify,
            kind: DecisionKind::OpenaiCompatible,
            base_url: Some(format!("{}/classifier", upstream.url)),
            model: Some("bonsai".into()),
            prompt_profile: Some("bonsai".into()),
            timeout_ms: Some(5000),
            ..Default::default()
        },
        classifier,
        None,
    ));
    RouterArgs {
        slot,
        cards: Arc::new(RoleCardStore::memory(Default::default())),
        providers: Arc::new(brainrouter::providers::ProviderStore::in_memory_with(vec![
            brainrouter::providers::ProviderRecord {
                id: "manifest".into(),
                kind: brainrouter::providers::ProviderKind::Manifest,
                base_url: format!("{}/cloud/v1", upstream.url),
                credential_id: Some("synthetic-test-key".into()),
                auto: true,
                models: Vec::new(),
                lan_peer: false,
                enabled,
                monthly_budget_usd: None,
                budget_mode: brainrouter::providers::BudgetMode::Warn,
            },
        ])),
        credentials: {
            let store = brainrouter::credentials::CredentialStore::in_memory();
            store
                .put(brainrouter::credentials::CredentialRecord {
                    id: "synthetic-test-key".into(),
                    description: String::new(),
                    kind: brainrouter::credentials::CredentialKind::Store {
                        value: brainrouter::credentials::SecretString::new("synthetic-test-key"),
                    },
                    grants: Vec::new(),
                    created_at: String::new(),
                    updated_at: String::new(),
                })
                .unwrap();
            Arc::new(store)
        },
        budget: Arc::new(brainrouter::provider_budget::ProviderBudget::in_memory()),
        llama_swap: Arc::new(OpenAiProvider::new(
            "llama-swap".into(),
            format!("{}/local/v1", upstream.url),
            None,
        )),
        fallback_model: "default-local".into(),
        local_models: vec!["known-local".into()],
        subs_model: Some("legacy-pool".into()),
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
    }
}

fn router(
    upstream: &SyntheticProviders,
    store: Arc<RoleCardStore>,
    enabled: bool,
    classify: bool,
) -> Arc<Router> {
    Arc::new(Router::new(router_args(upstream, enabled, classify)).with_cards(store))
}

fn request(model: &str) -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model":model, "messages":[{"role":"user","content":"synthetic test"}], "stream":true,
    }))
    .unwrap()
}

async fn drain(response: ProviderResponse) {
    let ProviderResponse::Stream(mut stream) = response;
    while let Some(chunk) = stream.next().await {
        chunk.unwrap();
    }
}

#[tokio::test]
async fn role_choices_and_client_models_are_independent() {
    let upstream = SyntheticProviders::start().await;
    let store = cards();
    let router = router(&upstream, store.clone(), true, false);
    for (alias, expected) in [
        ("auto", "main-local"),
        ("brainrouter/auto", "main-local"),
        ("", "main-local"),
        ("subs", "subagent-pool"),
        ("brainrouter/subs", "subagent-pool"),
        ("known-local", "known-local"),
        ("manual-local", "manual-local"),
        ("brainrouter/explicit-local", "explicit-local"),
        ("local", "default-local"),
        ("brainrouter/local", "default-local"),
        ("cloud", "auto"),
        ("brainrouter/cloud", "auto"),
        ("cloud/vendor/client-model", "vendor/client-model"),
    ] {
        let anthropic = brainrouter::anthropic::anthropic_to_openai(
            serde_json::from_value(json!({
                "model": alias, "max_tokens": 32, "stream": true,
                "messages": [{"role":"user","content":"synthetic test"}],
            }))
            .unwrap(),
        );
        for request in [request(alias), anthropic] {
            let (response, info) = router
                .route_tagged(request, None, String::new(), String::new())
                .await
                .unwrap();
            assert_eq!(info.model_key, expected, "{alias}");
            drain(response).await;
        }
    }
    store.set_assignment(GLOBAL_SCOPE, Role::Main, "cloud/author-cloud").unwrap();
    for (choice, expected_provider, expected_model) in [
        (local("review-local"), "llama-swap", "review-local"),
        (cloud("review-cloud"), "manifest", "review-cloud"),
        (ModelChoice::Auto, "llama-swap", "default-local"),
    ] {
        let (response, info) = router
            .route_with_choice(request("auto"), &choice, None, String::new(), String::new(), true)
            .await
            .unwrap();
        assert_eq!(info.effective_provider.as_deref(), Some(expected_provider));
        assert_eq!(info.model_key, expected_model);
        drain(response).await;
    }
    assert_eq!(upstream.calls().last().unwrap().1, "default-local");
    assert_eq!(store.assignment(Role::Subagent, None), "subagent-pool");
}

#[tokio::test]
async fn project_main_pin_overrides_global_and_unpinned_follows_global() {
    // Main precedence: a project-scope card overrides the global card; an
    // unpinned cwd follows the live global card.
    let upstream = SyntheticProviders::start().await;
    let store = cards(); // global main = main-local
    let repo = std::env::temp_dir().join(format!("br-cards-pin-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(repo.join(".git")).unwrap();
    store
        .set_assignment(repo.to_str().unwrap(), Role::Main, "project-pinned-main")
        .unwrap();
    let router = Arc::new(
        Router::new(router_args(&upstream, true, false)).with_cards(store),
    );

    let (response, info) = router
        .route_tagged(request("auto"), None, repo.to_string_lossy().into_owned(), String::new())
        .await
        .unwrap();
    assert_eq!(info.model_key, "project-pinned-main");
    drain(response).await;

    let (response, info) = router
        .route_tagged(request("auto"), None, String::new(), String::new())
        .await
        .unwrap();
    assert_eq!(info.model_key, "main-local");
    drain(response).await;

    fs::remove_dir_all(&repo).ok();
}

#[tokio::test]
async fn project_reviewer_pin_enters_the_snapshot_and_continuations_keep_it() {
    // Reviewer precedence: the project-scoped review card is the configured
    // choice in the fresh-review snapshot (admission still governs); an
    // unpinned cwd uses the global review card; a continuation reuses the
    // stored snapshot even after the global card changes.
    let upstream = SyntheticProviders::start().await;
    let store = cards(); // global review card = cloud/vendor/reviewer-2026
    let repo = std::env::temp_dir().join(format!("br-cards-review-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(repo.join(".git")).unwrap();
    store
        .set_assignment(repo.to_str().unwrap(), Role::Review, "cloud/project-reviewer")
        .unwrap();
    let router = Arc::new(
        Router::new(router_args(&upstream, true, false)).with_cards(store.clone()),
    );
    let sessions = Arc::new(SessionManager::new());
    let service = Arc::new(ReviewService::new(
        router,
        sessions.clone(),
        store.clone(),
        ReviewConfig::default(),
        Default::default(),
        provider_store(true),
        "default-local".to_string(),
    ));

    let pinned = service
        .start_review("t".into(), "pinned review".into(), None, vec![], repo.to_str().unwrap().into())
        .await
        .unwrap();
    assert_eq!(pinned.status, ReviewStatus::Approved);
    assert_eq!(
        upstream.calls().last().unwrap(),
        &("/cloud/v1/chat/completions".into(), "project-reviewer".into())
    );
    // The stored snapshot carries the pinned card selector, so continuations
    // keep it ("a session snapshots its card record at start").
    let snapshot = sessions.get_session(&pinned.session_id).unwrap().review_selector;
    assert_eq!(snapshot.as_deref(), Some("cloud/project-reviewer"));

    let global = service
        .start_review("t2".into(), "global review".into(), None, vec![], String::new())
        .await
        .unwrap();
    assert_eq!(global.status, ReviewStatus::Approved);
    assert_eq!(
        upstream.calls().last().unwrap(),
        &("/cloud/v1/chat/completions".into(), "vendor/reviewer-2026".into())
    );

    // Continuation keeps its snapshot even after the global card changes.
    store
        .set_assignment(GLOBAL_SCOPE, Role::Review, "cloud/changed-global")
        .unwrap();
    service.continue_review(&pinned.session_id, 1).await.unwrap();
    assert_eq!(
        upstream.calls().last().unwrap(),
        &("/cloud/v1/chat/completions".into(), "project-reviewer".into())
    );

    fs::remove_dir_all(&repo).ok();
}

#[tokio::test]
async fn cloud_ids_survive_review_and_continuation_snapshots() {
    let directory = Directory::new();
    let upstream = SyntheticProviders::start().await;
    let store = cards();
    let router = router(&upstream, store.clone(), true, false);
    let sessions = Arc::new(SessionManager::new());
    let service = Arc::new(ReviewService::new(
        router,
        sessions.clone(),
        store.clone(),
        ReviewConfig::default(),
        Default::default(),
        provider_store(false),
        "default-local".to_string(),
    ));
    let id = service.start_review_async(
        "test".into(),
        "synthetic review".into(),
        None,
        vec![],
        directory.0.to_str().unwrap().into(),
    );
    // v1 forced_mode=local + forced_model=new-reviewer ⇒ the migration-table
    // spelling is a bare-key card (Local{Some(id)} → `<bare-key>`).
    store
        .set_assignment(GLOBAL_SCOPE, Role::Review, "new-reviewer")
        .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while sessions.get_session(&id).unwrap().status != ReviewStatus::Approved {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    service.continue_review(&id, 1).await.unwrap();
    assert_eq!(
        upstream.calls(),
        vec![
            (
                "/cloud/v1/chat/completions".into(),
                "vendor/reviewer-2026".into()
            ),
            (
                "/cloud/v1/chat/completions".into(),
                "vendor/reviewer-2026".into()
            ),
        ]
    );
    let session = sessions.get_session(&id).unwrap();
    assert_eq!(
        session.review_selector.as_deref(),
        Some("cloud/vendor/reviewer-2026")
    );
    assert!(session
        .review_model
        .unwrap()
        .contains("requested cloud/vendor/reviewer-2026; actual"));
    let local_session = service
        .start_review(
            "next".into(),
            "synthetic local review".into(),
            None,
            vec![],
            directory.0.to_str().unwrap().into(),
        )
        .await
        .unwrap();
    assert_eq!(local_session.status, ReviewStatus::Approved);
    assert_eq!(
        upstream.calls().last().unwrap(),
        &("/local/v1/chat/completions".into(), "new-reviewer".into())
    );
    let before = sessions.get_session(&id).unwrap().status;
    assert!(service.continue_review(&id, 0).await.is_err());
    assert_eq!(sessions.get_session(&id).unwrap().status, before);
}

#[tokio::test]
async fn auto_reviewer_ignores_cloud_main_but_still_classifies() {
    let upstream = SyntheticProviders::start().await;
    let store = cards();
    store.set_assignment(GLOBAL_SCOPE, Role::Main, "local/explicit-main").unwrap();
    let router = router(&upstream, store, true, true);
    let (response, info) = router
        .route_with_choice(
            request("auto"),
            &ModelChoice::Auto,
            None,
            String::new(),
            String::new(),
            true,
        )
        .await
        .unwrap();
    drain(response).await;
    assert_eq!(info.model_key, "auto");
    assert_eq!(info.effective_provider.as_deref(), Some("manifest"));
    assert_eq!(upstream.calls().last().unwrap().1, "auto");
    assert!(upstream.calls()[0].0.starts_with("/classifier"));
}

#[tokio::test]
async fn strict_reviewer_cloud_route_errors_instead_of_local_fallback() {
    // H3 (preserved per-selector strictness, L3): the review loop passes
    // allow_local_fallback=false. With Manifest disabled, a cloud-admitted
    // reviewer must ERROR (so the loop escalates) rather than silently run
    // on a local model.
    let upstream = SyntheticProviders::start().await;
    let store = cards();
    let disabled = router(&upstream, store, false, false);
    let result = disabled
        .route_with_choice(
            request("auto"),
            &cloud("explicit-cloud"),
            None,
            String::new(),
            String::new(),
            false,
        )
        .await;
    assert!(result.is_err(), "strict cloud reviewer must error, not fall back to local");
    assert!(
        upstream.calls().iter().all(|call| !call.0.starts_with("/local")),
        "no local upstream call should happen under strict routing"
    );
}

#[tokio::test]
async fn disabled_cloud_and_cloud_failure_expose_requested_and_actual() {
    let upstream = SyntheticProviders::start().await;
    let store = cards();
    let disabled = router(&upstream, store.clone(), false, false);
    let (response, info) = disabled
        .route_with_choice(
            request("auto"),
            &cloud("explicit-cloud"),
            None,
            String::new(),
            String::new(),
            true,
        )
        .await
        .unwrap();
    drain(response).await;
    assert_eq!(info.model_key, "default-local");
    assert_eq!(info.effective_provider.as_deref(), Some("llama-swap"));
    assert_eq!(info.failed_attempts[0].model_key, "explicit-cloud");
    assert!(upstream
        .calls()
        .iter()
        .all(|call| call.0.starts_with("/local")));
    let catalog = disabled.model_catalog().await;
    assert_eq!(catalog["cloud_enabled"], false);
    assert!(catalog["cloud"]["error"]
        .as_str()
        .unwrap()
        .contains("disabled"));
    assert!(upstream
        .seen
        .lock()
        .unwrap()
        .iter()
        .all(|r| !r.path.starts_with("/cloud")));

    let enabled = router(&upstream, store, true, false);
    let (response, info) = enabled
        .route_with_choice(
            request("auto"),
            &cloud("missing-model"),
            None,
            String::new(),
            String::new(),
            true,
        )
        .await
        .unwrap();
    drain(response).await;
    assert_eq!(info.failed_attempts[0].model_key, "missing-model");
    assert_eq!(info.model_key, "default-local");
    let count = upstream.calls().len();
    assert!(enabled
        .route_with_choice(
            request("auto"),
            &local("missing-model"),
            None,
            String::new(),
            String::new(),
            true,
        )
        .await
        .is_err());
    assert_eq!(
        upstream.calls().len(),
        count + 1,
        "explicit local failure must not retry another model"
    );
}

#[tokio::test]
async fn discovery_is_authenticated_metadata_only_and_failure_is_explicit() {
    let upstream = SyntheticProviders::start().await;
    let store = cards();
    let router = router(&upstream, store, true, false);
    let catalog = router.model_catalog().await;
    assert_eq!(catalog["local"]["models"], json!(["model-a", "model-b"]));
    assert_eq!(catalog["cloud"]["models"], json!(["model-a", "model-b"]));
    assert_eq!(catalog["cloud"]["error"], Value::Null);
    assert!(upstream.calls().is_empty());
    let seen = upstream.seen.lock().unwrap().clone();
    assert_eq!(
        seen.iter()
            .find(|r| r.path.starts_with("/cloud"))
            .unwrap()
            .authorization
            .as_deref(),
        Some("Bearer synthetic-test-key")
    );
    let broken = OpenAiProvider::new("broken".into(), format!("{}/broken", upstream.url), None);
    assert!(broken
        .list_models()
        .await
        .unwrap_err()
        .to_string()
        .contains("503"));
}

#[tokio::test]
async fn presets_route_every_role_without_changing_the_pool() {
    // v1 presets are removed; the SAME routing table is exercised by setting
    // the card assignments each preset expanded to.
    let upstream = SyntheticProviders::start().await;
    for (label, main, reviewer, main_provider, main_model, reviewer_provider, reviewer_model) in [
        ("auto", "auto", "auto", "llama-swap", "default-local", "llama-swap", "default-local"),
        ("cloud", "cloud", "cloud", "manifest", "auto", "manifest", "auto"),
        ("local-main-sub", "local", "local", "llama-swap", "default-local", "llama-swap", "default-local"),
        ("local-custom", "main-local", "local", "llama-swap", "main-local", "llama-swap", "default-local"),
        ("cloud-main-local-review", "cloud", "local", "manifest", "auto", "llama-swap", "default-local"),
        ("local-main-cloud-review", "local", "cloud/vendor/reviewer-2026", "llama-swap", "default-local", "manifest", "vendor/reviewer-2026"),
    ] {
        let store = cards();
        store.set_assignment(GLOBAL_SCOPE, Role::Main, main).unwrap();
        store.set_assignment(GLOBAL_SCOPE, Role::Review, reviewer).unwrap();
        let router = router(&upstream, store, true, false);
        let (response, info) = router
            .route_tagged(request("auto"), None, String::new(), String::new())
            .await
            .unwrap();
        drain(response).await;
        assert_eq!(
            info.effective_provider.as_deref(),
            Some(main_provider),
            "{label}"
        );
        assert_eq!(info.model_key, main_model);
        let choice = ModelChoice::from_selector(reviewer).unwrap();
        let (response, info) = router
            .route_with_choice(
                request("auto"),
                &choice,
                None,
                String::new(),
                String::new(),
                true,
            )
            .await
            .unwrap();
        drain(response).await;
        assert_eq!(info.effective_provider.as_deref(), Some(reviewer_provider));
        assert_eq!(info.model_key, reviewer_model);
        let (response, info) = router
            .route_tagged(request("subs"), None, String::new(), String::new())
            .await
            .unwrap();
        drain(response).await;
        assert_eq!(info.model_key, "subagent-pool");
    }
}

#[tokio::test]
async fn clearing_pool_uses_legacy_auto_not_the_main_profile() {
    let upstream = SyntheticProviders::start().await;
    // v2 L3 retarget (spec: "config.subs_model remains the auto-pool
    // COMPATIBILITY SEED"): an unassigned subagent card = `auto` → decision
    // slot → local chain, seeded from subs_model — so the pool now follows
    // the legacy pool key (v1 routed it direct; L3 runs it through the
    // slot, landing on the same configured pool member).
    let upstream = SyntheticProviders::start().await;
    let store = cards();
    store.set_assignment(GLOBAL_SCOPE, Role::Main, "cloud/author").unwrap();
    store.remove_assignment(GLOBAL_SCOPE, Role::Subagent).unwrap(); // ⇒ default auto
    let router = router(&upstream, store, true, false);
    let (response, info) = router
        .route_tagged(request("subs"), None, String::new(), String::new())
        .await
        .unwrap();
    drain(response).await;
    assert_eq!(info.model_key, "legacy-pool");
    assert_eq!(
        upstream.calls(),
        vec![("/local/v1/chat/completions".into(), "legacy-pool".into())]
    );
}

#[tokio::test]
async fn cli_uses_the_same_typed_cards_and_reduced_review_api() {
    let upstream = SyntheticProviders::start().await;
    let cases: &[(&[&str], &str, Value)] = &[
        (
            &["roles", "set", "review", "cloud/cli-reviewer"],
            "/api/roles/set",
            json!({"role":"review","assignment":"cloud/cli-reviewer","scope":"global"}),
        ),
        (
            &["roles", "set", "main", "cloud", "--project", "/srv/repo"],
            "/api/roles/set",
            json!({"role":"main","assignment":"cloud","scope":"/srv/repo"}),
        ),
        (
            &["credential", "add", "api-key", "--kind", "env", "--env", "MY_KEY", "--grants", "bridge"],
            "/api/credentials/add",
            json!({"id":"api-key","kind":"env","grants":["bridge"],"var":"MY_KEY"}),
        ),
        (
            &["review-config", "update", "--max-iterations", "7"],
            "/api/review-config",
            json!({"max_iterations":7,"hankndory_integration":false,"design_doc_path":null}),
        ),
    ];
    for (args, path, expected) in cases {
        let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_brainrouter"))
            .args(["cli", "--url", &upstream.url])
            .args(*args)
            .output()
            .await
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let last = upstream.seen.lock().unwrap().last().unwrap().clone();
        assert_eq!(&last.path, path);
        assert_eq!(&last.body, expected);
    }
    let count = upstream.seen.lock().unwrap().len();
    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_brainrouter"))
        .args(["cli", "--url", &upstream.url, "roles", "set", "typo-role", "cloud"])
        .output()
        .await
        .unwrap();
    assert!(!output.status.success());
    assert_eq!(upstream.seen.lock().unwrap().len(), count);
}

#[test]
fn invalid_policy_input_cannot_mutate_preferences() {
    let store = cards();
    for input in [
        json!({"backend":"invalid"}),
        json!({"backend":"auto","model":"ignored"}),
        json!({"backend":"local","unknown_model_policy":"fallback"}),
    ] {
        assert!(serde_json::from_value::<ModelChoice>(input).is_err());
    }
    // v2 L3 retarget (spec selector vocabulary): reserved-as-BARE keys are
    // gone — `cloud/foo`/`brainrouter/foo` are canonical selectors now. The
    // bare-key smuggle list shrinks to non-grammatical ids; `subs` stays
    // refused as a card assignment.
    for id in ["", " ", "id with space", "\n", "subs", "cloud/", "brainrouter/"] {
        assert!(store.set_assignment(GLOBAL_SCOPE, Role::Main, id).is_err(), "{id:?}");
        assert_eq!(store.assignment(Role::Main, None), "main-local");
    }
    store.set_assignment(GLOBAL_SCOPE, Role::Main, "cloud/foo").unwrap();
    assert_eq!(store.assignment(Role::Main, None), "cloud/foo");
    store.set_assignment(GLOBAL_SCOPE, Role::Main, "main-local").unwrap();
    for mode in ["unknown", "AUTO", "subs"] {
        assert!(ModelChoice::from_legacy(mode, None).is_err());
    }
    assert!(store.set_assignment(GLOBAL_SCOPE, Role::Main, "subs").is_err());
    assert!(store.set_review_extras(brainrouter::roles::ReviewExtras {
        max_iterations: 0,
        hankndory_integration: false,
        design_doc_path: None,
    })
    .is_err());
    assert_eq!(store.assignment(Role::Main, None), "main-local");
    // Discovery is optional: an unlisted, well-formed explicit ID is provider-validated.
    store.set_assignment(GLOBAL_SCOPE, Role::Main, "cloud/not-in-discovery").unwrap();
}

#[test]
fn cards_persist_atomically_and_survive_restart() {
    let directory = Directory::new();
    let path = directory.0.join("role_cards.json");
    let store = RoleCardStore::load(path.clone());
    store.set_assignment(GLOBAL_SCOPE, Role::Main, "cloud/saved-author").unwrap();
    store
        .set_review_extras(brainrouter::roles::ReviewExtras {
            max_iterations: 9,
            hankndory_integration: true,
            design_doc_path: None,
        })
        .unwrap();
    assert_eq!(fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
    let reloaded = RoleCardStore::load(path.clone());
    assert_eq!(reloaded.assignment(Role::Main, None), "cloud/saved-author");
    assert_eq!(reloaded.review_extras().max_iterations, 9);
    assert!(reloaded.review_extras().hankndory_integration);
    assert_eq!(
        fs::read_dir(&directory.0).unwrap().count(),
        1,
        "no temporary files remain"
    );
    // Disk failure must not mutate runtime.
    fs::remove_file(&path).unwrap();
    fs::create_dir(&path).unwrap();
    assert!(reloaded.set_assignment(GLOBAL_SCOPE, Role::Main, "local/must-not-apply").is_err());
    assert_eq!(reloaded.assignment(Role::Main, None), "cloud/saved-author");
}

#[test]
fn l3_configuration_is_validated() {
    let directory = Directory::new();
    let config_path = directory.0.join("config.yaml");
    let good = json!({
        "manifest":{"base_url":"http://127.0.0.1:1/v1"},
        "llama_swap":{"base_url":"http://127.0.0.1:1/v1","fallback_model":"fallback","subs_model":"legacy-subs"},
        "decision_model":{"enabled":false,"kind":"heuristic"},
    });
    fs::write(&config_path, serde_yaml::to_string(&good).unwrap()).unwrap();
    let loaded = config::load(&config_path).unwrap();
    assert!(!loaded.manifest.enabled);
    // v2 L7 `v2.0-p7`: the section is `swap:` (the `llama_swap:` spelling
    // still parses through the serde alias).
    assert_eq!(loaded.swap.subs_model.as_deref(), Some("legacy-subs"));
    assert!(!loaded.decision_model.enabled);

    // A v1-era review block (forced_mode) is NOT an L3 config any more:
    // the reduced ReviewConfig rejects the key (it must arrive migrated).
    let mut legacy = good.clone();
    legacy["review"] = json!({"forced_mode":"cloud","forced_model":"exact-cloud"});
    fs::write(&config_path, serde_yaml::to_string(&legacy).unwrap()).unwrap();
    assert!(config::load(&config_path).is_err());

    // Kinds requiring an endpoint fail validation without it.
    let mut openai = good.clone();
    openai["decision_model"] = json!({"enabled":true,"kind":"openai-compatible","model":"m"});
    fs::write(&config_path, serde_yaml::to_string(&openai).unwrap()).unwrap();
    assert!(config::load(&config_path).is_err(), "openai-compatible without base_url");
    let mut webhook = good.clone();
    webhook["decision_model"] = json!({"enabled":true,"kind":"webhook"});
    fs::write(&config_path, serde_yaml::to_string(&webhook).unwrap()).unwrap();
    assert!(config::load(&config_path).is_err(), "webhook without url");
}

#[test]
fn legacy_auto_yaml_ignores_the_leftover_model_only_on_read() {
    // The v1 read-parity row of the migration: forced_mode auto + a leftover
    // forced_model normalizes to Auto (never a validation error).
    let directory = Directory::new();
    let path = directory.0.join("brainrouter.yaml");
    let yaml = serde_yaml::to_string(&json!({
        "manifest": {"base_url": "http://127.0.0.1:1/v1"},
        "llama_swap": {"base_url": "http://127.0.0.1:1/v1", "fallback_model": "main-local"},
        "bonsai": {"enabled": false},
        "review": {"forced_mode": "auto", "forced_model": "my-model", "max_iterations": 5},
    }))
    .unwrap();
    fs::write(&path, &yaml).unwrap();
    let state = brainrouter::migration::read_v1_state(&path).unwrap();
    assert!(state.needs_l3_migration());
    let profile = state.effective_profile().unwrap();
    assert_eq!(profile.reviewer, ModelChoice::Auto);
    assert_eq!(profile.main, ModelChoice::local());
    assert_eq!(fs::read_to_string(&path).unwrap(), yaml);
}

#[test]
fn legacy_state_migrates_once_into_cards() {
    let directory = Directory::new();
    let path = directory.0.join("brainrouter.yaml");
    fs::write(
        &path,
        serde_yaml::to_string(&json!({
            "manifest": {"base_url": "http://127.0.0.1:1/v1"},
            "llama_swap": {"base_url": "http://127.0.0.1:1/v1", "fallback_model": "m"},
            "routing": {
                "preset": "custom",
                "main": {"backend": "cloud", "model": null},
                "reviewer": {"backend": "local", "model": "rev-local"},
                "subagent_model": "pool-x"
            },
            "review": {"max_iterations": 3},
        }))
        .unwrap(),
    )
    .unwrap();
    let state = read_v1_state(&path).unwrap();
    let profile = state.effective_profile().unwrap();
    let pins: std::collections::BTreeMap<String, LegacyPin> = [(
        "/srv/project".to_string(),
        LegacyPin {
            main: Some(ModelChoice::Cloud { model: Some("pinned-cloud".into()) }),
            reviewer: None,
            subagent: Some("pinned-subs".into()),
        },
    )]
    .into_iter()
    .collect();
    let out = migrate_v1(&profile, &pins, state.review.clone());
    let store = RoleCardStore::memory(Default::default());
    store
        .replace_from_migration(brainrouter::roles::StoreDataForMigration {
            assignments: out.cards,
            review_extras: out.review_extras,
        })
        .unwrap();
    assert_eq!(store.assignment(Role::Main, None), "cloud");
    assert_eq!(store.assignment(Role::Review, None), "rev-local");
    assert_eq!(store.assignment(Role::Subagent, None), "pool-x");
    assert_eq!(store.assignment(Role::Main, Some("/srv/project")), "cloud/pinned-cloud");
    assert_eq!(store.assignment(Role::Subagent, Some("/srv/project")), "pinned-subs");
    // Unpinned project follows the global card.
    assert_eq!(store.assignment(Role::Review, Some("/other/repo")), "rev-local");
    assert_eq!(store.review_extras().max_iterations, 3);
}

#[test]
fn migration_validation_failures_report_the_source_path_and_cause() {
    let directory = Directory::new();
    for (review, cause) in [
        (json!({"forced_mode":"unknown"}), "unknown routing mode"),
        (
            json!({"forced_mode":"local","forced_model":"bad model"}),
            "model ID",
        ),
        (
            json!({"forced_mode":"cloud","forced_model":"cloud/alias"}),
            "model ID",
        ),
    ] {
        let path = directory.0.join(format!("brainrouter-{}.yaml", cause.replace(' ', "_")));
        fs::write(
            &path,
            serde_yaml::to_string(&json!({
                "manifest": {"base_url": "http://127.0.0.1:1/v1"},
                "llama_swap": {"base_url": "http://127.0.0.1:1/v1", "fallback_model": "m"},
                "review": review,
            }))
            .unwrap(),
        )
        .unwrap();
        let state = read_v1_state(&path).unwrap();
        let error = state.effective_profile().unwrap_err().to_string();
        assert!(error.contains(cause), "{review} -> {error}");
    }
    // Invalid card extras report the iterations rule.
    let store = RoleCardStore::memory(Default::default());
    assert!(store
        .set_review_extras(brainrouter::roles::ReviewExtras {
            max_iterations: 0,
            hankndory_integration: false,
            design_doc_path: None,
        })
        .is_err());
}

#[test]
fn migration_table_is_exact() {
    let pins: std::collections::BTreeMap<String, LegacyPin> =
        std::collections::BTreeMap::new();
    let rows = [
        (
            LegacyRoutingProfile {
                preset: LegacyPreset::Custom,
                main: ModelChoice::Auto,
                reviewer: ModelChoice::Local { model: Some("k".into()) },
                subagent_model: None,
            },
            ("auto", "k"),
        ),
        (
            LegacyRoutingProfile {
                preset: LegacyPreset::Custom,
                main: ModelChoice::local(),
                reviewer: ModelChoice::Cloud { model: None },
                subagent_model: None,
            },
            ("local", "cloud"),
        ),
        (
            LegacyRoutingProfile {
                preset: LegacyPreset::Custom,
                main: ModelChoice::Local { model: None },
                reviewer: ModelChoice::Cloud { model: Some("vendor/x".into()) },
                subagent_model: Some("pool".into()),
            },
            ("local", "cloud/vendor/x"),
        ),
    ];
    for (profile, (main, reviewer)) in rows {
        let out = migrate_v1(&profile, &pins, None);
        let get = |role| {
            out.cards
                .iter()
                .find(|c| c.role == role && c.scope == GLOBAL_SCOPE)
                .unwrap()
                .assignment
                .clone()
        };
        assert_eq!(get(Role::Main), main);
        assert_eq!(get(Role::Review), reviewer);
    }
    // No routing block ⇒ main migrates to `local` (v1 default).
    let out = migrate_v1(
        &LegacyRoutingProfile {
            preset: LegacyPreset::Custom,
            main: ModelChoice::local(),
            reviewer: ModelChoice::Auto,
            subagent_model: None,
        },
        &pins,
        None,
    );
    assert_eq!(out.cards[0].assignment, "local");
}
