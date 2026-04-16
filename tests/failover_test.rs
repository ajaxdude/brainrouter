use brainrouter::{
    config::{ModelConfig, ProviderConfig, ProviderType},
    health::HealthTracker,
    provider::{Provider, ProviderResponse, SseStream},
    router::Router,
    types::ChatCompletionRequest,
};
use anyhow::Result;
use bytes::Bytes;
use futures_util::stream;
use std::{
    collections::HashMap,
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
};

// A mock provider that can be configured to succeed or fail.
struct MockProvider {
    name: String,
    should_fail: bool,
    called: Arc<Mutex<bool>>,
}

impl Provider for MockProvider {
    fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ProviderResponse>> + Send + '_>> {
        *self.called.lock().unwrap() = true;
        if self.should_fail {
            Box::pin(async { Err(anyhow::anyhow!("Provider failed as configured")) })
        } else {
            let stream: SseStream = Box::pin(stream::once(async { Ok(Bytes::from("data: ok\\n\\n")) }));
            Box::pin(async { Ok(ProviderResponse::Stream(stream)) })
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[tokio::test]
async fn test_failover_logic() {
    let health_tracker = Arc::new(HealthTracker::new());
    let mut providers = HashMap::new();

    let called1 = Arc::new(Mutex::new(false));
    let provider1 = Arc::new(MockProvider {
        name: "provider1".to_string(),
        should_fail: true,
        called: called1.clone(),
    });
    providers.insert("provider1".to_string(), provider1 as Arc<dyn Provider>);

    let called2 = Arc::new(Mutex::new(false));
    let provider2 = Arc::new(MockProvider {
        name: "provider2".to_string(),
        should_fail: false,
        called: called2.clone(),
    });
    providers.insert("provider2".to_string(), provider2 as Arc<dyn Provider>);

    let model_configs = vec![
        ModelConfig {
            name: "model1".to_string(),
            providers: vec!["provider1".to_string(), "provider2".to_string()],
        },
    ];

    let ranked_models = vec!["model1".to_string()];

    let router = Router::new(
        providers,
        ranked_models,
        &model_configs,
        health_tracker,
    );

    let request = ChatCompletionRequest {
        model: "model1".to_string(),
        messages: vec![],
        stream: Some(true),
        temperature: None,
        max_tokens: None,
        top_p: None,
        stop: None,
        extra: Default::default(),
    };

    let response = router.route(request).await;

    assert!(response.is_ok());
    assert!(*called1.lock().unwrap());
    assert!(*called2.lock().unwrap());
}
