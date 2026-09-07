//! Synthetic providers only: no model processes, credentials or external traffic.

use brainrouter::{
    classifier::Classifier,
    config::{self, NudgeBudgets, ReviewConfig},
    health::HealthTracker,
    inference_state::InferenceTracker,
    provider::{openai::OpenAiProvider, ProviderResponse},
    review::ReviewService,
    router::{Router, RouterArgs},
    routing_events::RoutingEvents,
    routing_profile::{ModelChoice, ProfileStore, RoutingPreset, RoutingProfile},
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
        let path = std::env::temp_dir().join(format!("br-profile-{}", uuid::Uuid::new_v4()));
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
                                    let (status, text) = if path == "/api/routing-profile" {
                                        (200, json!({"profile": if body.is_null() { serde_json::to_value(profile()).unwrap() } else { body.clone() }}).to_string())
                                    } else if path == "/api/review-config" {
                                        (200, serde_json::to_string(&ReviewConfig::default()).unwrap())
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
            .filter(|r| r.path.ends_with("/chat/completions"))
            .map(|r| (r.path.clone(), r.body["model"].as_str().unwrap().into()))
            .collect()
    }
}

impl Drop for SyntheticProviders {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn local(id: &str) -> ModelChoice {
    ModelChoice::Local {
        model: Some(id.into()),
    }
}
fn cloud(id: &str) -> ModelChoice {
    ModelChoice::Cloud {
        model: Some(id.into()),
    }
}

fn profile() -> RoutingProfile {
    RoutingProfile {
        preset: RoutingPreset::Custom,
        main: local("main-local"),
        reviewer: cloud("vendor/reviewer-2026"),
        subagent_model: Some("subagent-pool".into()),
    }
}

fn router(
    upstream: &SyntheticProviders,
    store: Arc<ProfileStore>,
    enabled: bool,
    classify: bool,
) -> Arc<Router> {
    Arc::new(
        Router::new(RouterArgs {
            classifier: Arc::new(Classifier::new(
                format!("{}/classifier", upstream.url),
                "default-local".into(),
                Arc::new(AtomicBool::new(classify)),
                Arc::new(AtomicBool::new(false)),
                None,
            )),
            manifest: Arc::new(OpenAiProvider::new(
                "manifest".into(),
                format!("{}/cloud/v1", upstream.url),
                Some("synthetic-test-key".into()),
            )),
            manifest_enabled: enabled,
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
            prompt_rewrite: Arc::new(AtomicBool::new(false)),
        })
        .with_profiles(store),
    )
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
    let store = Arc::new(ProfileStore::memory(profile(), 1).unwrap());
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
    store.update_main(cloud("author-cloud")).unwrap();
    for (choice, expected_provider, expected_model) in [
        (local("review-local"), "llama-swap", "review-local"),
        (cloud("review-cloud"), "manifest", "review-cloud"),
        (ModelChoice::Auto, "llama-swap", "default-local"),
    ] {
        let (response, info) = router
            .route_with_choice(request("auto"), &choice, None, String::new(), String::new())
            .await
            .unwrap();
        assert_eq!(info.effective_provider.as_deref(), Some(expected_provider));
        assert_eq!(info.model_key, expected_model);
        drain(response).await;
    }
    assert_eq!(upstream.calls().last().unwrap().1, "default-local");
    assert_eq!(
        store.profile().subagent_model.as_deref(),
        Some("subagent-pool")
    );
}

#[tokio::test]
async fn cloud_ids_survive_review_and_continuation_snapshots() {
    let directory = Directory::new();
    let upstream = SyntheticProviders::start().await;
    let store = Arc::new(ProfileStore::memory(profile(), 1).unwrap());
    let router = router(&upstream, store.clone(), true, false);
    let sessions = Arc::new(SessionManager::new());
    let service = Arc::new(ReviewService::new(
        router,
        sessions.clone(),
        ReviewConfig::default(),
    ));
    let id = service.start_review_async(
        "test".into(),
        "synthetic review".into(),
        None,
        vec![],
        directory.0.to_str().unwrap().into(),
    );
    store
        .update_review(ReviewConfig {
            max_iterations: 1,
            forced_mode: "local".into(),
            forced_model: Some("new-reviewer".into()),
        })
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
        session.review_config.unwrap().forced_model.as_deref(),
        Some("vendor/reviewer-2026")
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
    let store = Arc::new(ProfileStore::memory(profile(), 1).unwrap());
    store.update_main(local("explicit-main")).unwrap();
    let router = router(&upstream, store, true, true);
    let (response, info) = router
        .route_with_choice(
            request("auto"),
            &ModelChoice::Auto,
            None,
            String::new(),
            String::new(),
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
async fn disabled_cloud_and_cloud_failure_expose_requested_and_actual() {
    let upstream = SyntheticProviders::start().await;
    let store = Arc::new(ProfileStore::memory(profile(), 1).unwrap());
    let disabled = router(&upstream, store.clone(), false, false);
    let (response, info) = disabled
        .route_with_choice(
            request("auto"),
            &cloud("explicit-cloud"),
            None,
            String::new(),
            String::new(),
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
    let store = Arc::new(ProfileStore::memory(profile(), 1).unwrap());
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
    let upstream = SyntheticProviders::start().await;
    for (preset, main_provider, main_model, reviewer_provider, reviewer_model) in [
        (
            RoutingPreset::Auto,
            "llama-swap",
            "default-local",
            "llama-swap",
            "default-local",
        ),
        (RoutingPreset::Cloud, "manifest", "auto", "manifest", "auto"),
        (
            RoutingPreset::LocalMainSub,
            "llama-swap",
            "default-local",
            "llama-swap",
            "default-local",
        ),
        (
            RoutingPreset::LocalCustom,
            "llama-swap",
            "main-local",
            "llama-swap",
            "default-local",
        ),
        (
            RoutingPreset::CloudMainLocalReview,
            "manifest",
            "auto",
            "llama-swap",
            "default-local",
        ),
        (
            RoutingPreset::LocalMainCloudReview,
            "llama-swap",
            "default-local",
            "manifest",
            "auto",
        ),
    ] {
        let mut profile = profile();
        profile.apply_preset(preset).unwrap();
        let store = Arc::new(ProfileStore::memory(profile.clone(), 1).unwrap());
        let router = router(&upstream, store, true, false);
        let (response, info) = router
            .route_tagged(request("auto"), None, String::new(), String::new())
            .await
            .unwrap();
        drain(response).await;
        assert_eq!(
            info.effective_provider.as_deref(),
            Some(main_provider),
            "{preset:?}"
        );
        assert_eq!(info.model_key, main_model);
        let (response, info) = router
            .route_with_choice(
                request("auto"),
                &profile.reviewer,
                None,
                String::new(),
                String::new(),
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
    let mut profile = profile();
    profile.main = cloud("author");
    profile.subagent_model = None;
    let store = Arc::new(ProfileStore::memory(profile, 1).unwrap());
    let router = router(&upstream, store, true, false);
    let (response, info) = router
        .route_tagged(request("subs"), None, String::new(), String::new())
        .await
        .unwrap();
    drain(response).await;
    assert_eq!(info.model_key, "default-local");
    assert_eq!(
        upstream.calls(),
        vec![("/local/v1/chat/completions".into(), "default-local".into())]
    );
}

#[tokio::test]
async fn cli_uses_the_same_typed_profile_and_legacy_review_api() {
    let upstream = SyntheticProviders::start().await;
    let cases: &[(&[&str], &str, Value)] = &[
        (
            &[
                "routing-profile",
                "choose",
                "reviewer",
                "cloud",
                "--model",
                "cli-reviewer",
            ],
            "/api/routing-profile",
            json!({"preset":"custom","main":{"backend":"local","model":"main-local"},"reviewer":{"backend":"cloud","model":"cli-reviewer"},"subagent_model":"subagent-pool"}),
        ),
        (
            &[
                "routing-profile",
                "preset",
                "local_custom",
                "--main-model",
                "cli-main",
            ],
            "/api/routing-profile",
            json!({"preset":"local_custom","main":{"backend":"local","model":"cli-main"},"reviewer":{"backend":"local","model":null},"subagent_model":"subagent-pool"}),
        ),
        (
            &["routing-profile", "pool", "cli-pool"],
            "/api/routing-profile",
            json!({"preset":"custom","main":{"backend":"local","model":"main-local"},"reviewer":{"backend":"cloud","model":"vendor/reviewer-2026"},"subagent_model":"cli-pool"}),
        ),
        (
            &[
                "review-config",
                "update",
                "--forced-mode",
                "cloud",
                "--forced-model",
                "cli-reviewer",
            ],
            "/api/review-config",
            json!({"max_iterations":5,"forced_mode":"cloud","forced_model":"cli-reviewer"}),
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
        .args([
            "cli",
            "--url",
            &upstream.url,
            "routing-profile",
            "choose",
            "reviewer",
            "typo",
        ])
        .output()
        .await
        .unwrap();
    assert!(!output.status.success());
    assert_eq!(upstream.seen.lock().unwrap().len(), count);
}

#[test]
fn invalid_policy_input_cannot_mutate_preferences() {
    let store = ProfileStore::memory(profile(), 1).unwrap();
    for input in [
        json!({"backend":"invalid"}),
        json!({"backend":"auto","model":"ignored"}),
        json!({"backend":"local","unknown_model_policy":"fallback"}),
    ] {
        assert!(serde_json::from_value::<ModelChoice>(input).is_err());
    }
    for id in [
        "",
        " ",
        "id with space",
        "auto",
        "subs",
        "brainrouter/foo",
        "cloud/foo",
        "\n",
    ] {
        assert!(store.update_main(local(id)).is_err(), "{id:?}");
        assert_eq!(store.profile(), profile());
    }
    for mode in ["unknown", "AUTO", "subs"] {
        assert!(ModelChoice::from_legacy(mode, None).is_err());
    }
    let mut invalid = profile();
    invalid.preset = RoutingPreset::Cloud;
    assert!(store.update_profile(invalid).is_err());
    let mut serialized = serde_json::to_value(profile()).unwrap();
    serialized["preset"] = json!("typo");
    assert!(serde_json::from_value::<RoutingProfile>(serialized).is_err());
    assert!(store
        .update_review(ReviewConfig {
            max_iterations: 0,
            ..ReviewConfig::default()
        })
        .is_err());
    assert_eq!(store.profile(), profile());
    // Discovery is optional: an unlisted, well-formed explicit ID is provider-validated.
    store.update_main(cloud("not-in-discovery")).unwrap();
}

#[test]
fn profiles_persist_atomically_and_migrate_valid_legacy_review_preferences() {
    let directory = Directory::new();
    let path = directory.0.join("routing_state.json");
    fs::write(
        directory.0.join("review_state.json"),
        r#"{"forced_mode":"cloud","forced_model":"legacy-cloud","max_iterations":9}"#,
    )
    .unwrap();
    let store = ProfileStore::load(path.clone(), profile(), &ReviewConfig::default()).unwrap();
    assert_eq!(store.profile().reviewer, cloud("legacy-cloud"));
    assert_eq!(
        store.review_config().max_iterations,
        5,
        "YAML still owns iterations after restart"
    );
    let mut updated = profile();
    updated.main = cloud("saved-author");
    store.update_profile(updated.clone()).unwrap();
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    let reloaded = ProfileStore::load(path.clone(), profile(), &ReviewConfig::default()).unwrap();
    assert_eq!(reloaded.profile(), updated);
    assert_eq!(
        serde_json::from_slice::<RoutingProfile>(&fs::read(&path).unwrap()).unwrap(),
        updated
    );
    assert_eq!(
        fs::read_dir(&directory.0).unwrap().count(),
        2,
        "no temporary files remain"
    );
    fs::remove_file(&path).unwrap();
    fs::create_dir(&path).unwrap();
    assert!(reloaded.update_main(local("must-not-apply")).is_err());
    assert_eq!(
        reloaded.profile(),
        updated,
        "disk failure must not mutate runtime"
    );
}

#[test]
fn malformed_saved_state_and_legacy_configuration_are_validated() {
    let directory = Directory::new();
    let path = directory.0.join("routing_state.json");
    fs::write(&path, r#"{"preset":"wat"}"#).unwrap();
    assert!(ProfileStore::load(path.clone(), profile(), &ReviewConfig::default()).is_err());
    fs::remove_file(&path).unwrap();
    fs::write(
        directory.0.join("review_state.json"),
        r#"{"forced_mode":"wat"}"#,
    )
    .unwrap();
    assert!(ProfileStore::load(path, profile(), &ReviewConfig::default()).is_err());

    let config_path = directory.0.join("config.yaml");
    let legacy = json!({
        "manifest":{"base_url":"http://127.0.0.1:1/v1"},
        "llama_swap":{"base_url":"http://127.0.0.1:1/v1","fallback_model":"fallback","subs_model":"legacy-subs"},
        "bonsai":{},
        "review":{"forced_mode":"cloud","forced_model":"exact-cloud"},
    });
    fs::write(&config_path, serde_yaml::to_string(&legacy).unwrap()).unwrap();
    let loaded = config::load(&config_path).unwrap();
    assert!(!loaded.manifest.enabled);
    assert!(!loaded.bonsai.enabled);
    assert_eq!(loaded.routing_profile().unwrap().main, ModelChoice::local());
    assert_eq!(
        loaded.routing_profile().unwrap().reviewer,
        cloud("exact-cloud")
    );
    assert_eq!(
        loaded.routing_profile().unwrap().subagent_model.as_deref(),
        Some("legacy-subs")
    );
    let mut invalid = legacy;
    invalid["review"]["forced_mode"] = json!("typo");
    fs::write(&config_path, serde_yaml::to_string(&invalid).unwrap()).unwrap();
    assert!(config::load(&config_path).is_err());
}

#[test]
fn legacy_auto_yaml_ignores_the_leftover_model_only_on_read() {
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
    let loaded = config::load(&path).unwrap();
    assert_eq!(loaded.review.forced_mode, "auto");
    assert_eq!(loaded.review.forced_model, None);
    assert_eq!(
        loaded.routing_profile().unwrap().reviewer,
        ModelChoice::Auto
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), yaml);

    let new_write: config::BrainrouterConfig = serde_yaml::from_str(&yaml).unwrap();
    assert!(new_write.review.validate().is_err());
    assert!(new_write.routing_profile().is_err());
}

#[test]
fn legacy_auto_state_migrates_once_and_persists_normalized_preferences() {
    let directory = Directory::new();
    let legacy = directory.0.join("review_state.json");
    let path = directory.0.join("routing_state.json");
    let original =
        r#"{"forced_mode":"auto","forced_model":"ignored old model","max_iterations":9}"#;
    fs::write(&legacy, original).unwrap();
    let review = ReviewConfig {
        max_iterations: 3,
        ..ReviewConfig::default()
    };
    let store = ProfileStore::load(path.clone(), profile(), &review).unwrap();
    assert_eq!(store.profile().reviewer, ModelChoice::Auto);
    assert_eq!(store.profile().main, profile().main);
    assert_eq!(store.profile().subagent_model, profile().subagent_model);
    assert_eq!(store.review_config().max_iterations, 3);
    assert_eq!(store.review_config().forced_model, None);
    let normalized: RoutingProfile = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    assert_eq!(normalized, store.profile());
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert_eq!(fs::read_to_string(&legacy).unwrap(), original);
    fs::write(
        &legacy,
        "malformed legacy state that must no longer be read",
    )
    .unwrap();
    let reloaded = ProfileStore::load(path, profile(), &review).unwrap();
    assert_eq!(reloaded.profile(), normalized);
    assert!(store
        .update_review(ReviewConfig {
            forced_mode: "auto".into(),
            forced_model: Some("new-invalid-model".into()),
            ..ReviewConfig::default()
        })
        .is_err());
    assert_eq!(store.profile(), normalized);
}

#[test]
fn migration_validation_failures_report_the_source_path_and_cause() {
    let directory = Directory::new();
    let legacy = directory.0.join("review_state.json");
    let path = directory.0.join("routing_state.json");
    for (saved, cause) in [
        (json!({"forced_mode":"unknown"}), "unknown routing mode"),
        (
            json!({"forced_mode":"local","forced_model":"bad model"}),
            "model ID",
        ),
        (
            json!({"forced_mode":"cloud","forced_model":"cloud/alias"}),
            "model ID",
        ),
        (
            json!({"forced_mode":"auto","forced_model":"ignored","max_iterations":0}),
            "max_iterations",
        ),
    ] {
        fs::write(&legacy, serde_json::to_vec(&saved).unwrap()).unwrap();
        let error = ProfileStore::load(path.clone(), profile(), &ReviewConfig::default())
            .err()
            .expect("invalid migration must fail");
        let message = format!("{error:#}");
        assert!(message.contains(legacy.to_str().unwrap()), "{message}");
        assert!(message.contains(cause), "{message}");
        assert!(!path.exists(), "failed migration must not write new state");
    }

    let invalid_profile = json!({
        "preset":"local_main_sub","main":{"backend":"cloud","model":"cloud-model"},
        "reviewer":{"backend":"local","model":null},"subagent_model":null,
    });
    fs::write(&path, serde_json::to_vec(&invalid_profile).unwrap()).unwrap();
    let error = ProfileStore::load(path.clone(), profile(), &ReviewConfig::default())
        .err()
        .expect("invalid saved profile must fail");
    let message = format!("{error:#}");
    assert!(message.contains(path.to_str().unwrap()), "{message}");
    assert!(message.contains("preset"), "{message}");
}
