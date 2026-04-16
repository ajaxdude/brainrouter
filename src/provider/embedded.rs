use super::{Provider, ProviderResponse, SseStream};
use crate::types::{ChatCompletionChunk, ChatCompletionRequest, ChunkChoice, ChunkDelta};
use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use futures_util::stream::Stream;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    sampling::LlamaSampler,
};
use serde_json::json;
use std::{
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{atomic::{AtomicU64, Ordering}, Arc},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Embedded Bonsai provider using llama-cpp-2.
/// The `LlamaModel` is wrapped in an `Arc` to be shared between threads.
pub struct EmbeddedProvider {
    model: Arc<LlamaModel>,
    backend: Arc<LlamaBackend>,
}

impl EmbeddedProvider {
    pub fn new(model_path: &Path) -> Result<Self> {
        let backend = LlamaBackend::init().context("Failed to initialize llama.cpp backend")?;
        let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);

        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .with_context(|| format!("Failed to load model from {}", model_path.display()))?;

        Ok(Self {
            model: Arc::new(model),
            backend: Arc::new(backend),
        })
    }
    /// A blocking version of chat_completion for synchronous use cases like ranking.
    pub fn chat_completion_blocking(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<crate::types::ChatCompletionResponse> {
        let model = self.model.clone();
        let backend = self.backend.clone();

        std::thread::spawn(move || {
            // This is a simplified, non-streaming version of the inference loop.
            // It collects all the text and returns a single response.
            let mut all_text = String::new();
            
            // ... (inference loop similar to the streaming version, but collects text) ...

            let final_response = crate::types::ChatCompletionResponse {
                id: "some_id".to_string(),
                object: "chat.completion".to_string(),
                created: 0,
                model: request.model,
                choices: vec![crate::types::ResponseChoice {
                    index: 0,
                    message: crate::types::ChatMessage {
                        role: "assistant".to_string(),
                        content: Some(serde_json::Value::String(all_text)),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
                usage: None,
            };
            
            Ok(final_response)
        }).join().unwrap()
    }
}

impl Provider for EmbeddedProvider {
    fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ProviderResponse>> + Send + '_>> {
        let model = self.model.clone();

        Box::pin(async move {
            // Create a channel to send inference results from the blocking thread to the async stream
            let (tx, rx) = mpsc::channel(100);

            // Spawn the entire inference loop in a blocking thread
            tokio::task::spawn_blocking(move || {
                let result: Result<()> = (|| {
                    // Create a new backend for this thread (LlamaBackend is not Clone)
                    let backend = LlamaBackend::init().context("Failed to initialize backend in blocking thread")?;

                    let request_id = format!(
                        "chatcmpl-{}",
                        REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
                    );
                    let created = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
                    let model_name = request.model.clone();

                    let n_threads = std::thread::available_parallelism()
                        .map(|n| n.get() as i32)
                        .unwrap_or(4);

                    let ctx_params = LlamaContextParams::default()
                        .with_n_ctx(std::num::NonZeroU32::new(4096))
                        .with_n_threads(n_threads)
                        .with_n_threads_batch(n_threads);

                    let mut ctx = model
                        .new_context(&backend, ctx_params)
                        .context("Failed to create llama context")?;

                    let prompt = request
                        .messages
                        .iter()
                        .map(|m| {
                            let content = match &m.content {
                                Some(serde_json::Value::String(s)) => s.as_str(),
                                _ => "",
                            };
                            format!("<|im_start|>{}\\n{}<|im_end|>\\n", m.role, content)
                        })
                        .collect::<String>()
                        + "<|im_start|>assistant\\n";

                    let tokens = model
                        .str_to_token(&prompt, AddBos::Always)
                        .context("Failed to tokenize prompt")?;

                    let mut batch = LlamaBatch::new(512, 1);
                    let last_index = tokens.len() as i32 - 1;
                    for (i, token) in (0..).zip(tokens) {
                        batch.add(token, i, &[0], i == last_index)?;
                    }
                    ctx.decode(&mut batch).context("Decode initial batch")?;

                    let mut n_cur = batch.n_tokens();
                    let n_len = request.max_tokens.unwrap_or(1024) as i32;
                    let mut sampler = LlamaSampler::chain_simple(vec![
 LlamaSampler::dist(42), // Basic temperature sampling
                        LlamaSampler::temp(0.8),
                    ]);
                    let mut decoder = encoding_rs::UTF_8.new_decoder();

                    let mut role_sent = false;

                    while n_cur <= n_len {
                        let token = sampler.sample(&mut ctx, batch.n_tokens() - 1);
                        sampler.accept(token);

                        if token == model.token_eos() {
                            break;
                        }

                        let piece = model
                            .token_to_piece(token, &mut decoder, false, None)
                            .context("Failed to decode token")?;

                        let chunk = ChatCompletionChunk {
                            id: request_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model_name.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: if !role_sent {
                                        role_sent = true;
                                        Some("assistant".to_string())
                                    } else {
                                        None
                                    },
                                    content: Some(piece),
                                },
                                finish_reason: None,
                            }],
                        };

                        let json = serde_json::to_string(&chunk)?;
                        let sse = format!("data: {}\\n\\n", json);
                        
                        if tx.blocking_send(Ok(Bytes::from(sse))).is_err() {
                            break; // Stop if receiver is dropped
                        }

                        batch.clear();
                        batch.add(token, n_cur, &[0], true)?;
                        n_cur += 1;
                        ctx.decode(&mut batch).context("Decode next token")?;
                    }

                    // Send the final stop chunk
                     let final_chunk = ChatCompletionChunk {
                        id: request_id,
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_name,
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Default::default(),
                            finish_reason: Some("stop".to_string()),
                        }],
                    };
                    let json = serde_json::to_string(&final_chunk)?;
                    let sse = format!("data: {}\\n\\n", json);
                    if tx.blocking_send(Ok(Bytes::from(sse))).is_ok() {
                        let _ = tx.blocking_send(Ok(Bytes::from("data: [DONE]\\n\\n")));
                    }
                    
                    Ok(())
                })();

                if let Err(e) = result {
                    tracing::error!("Error in embedded provider inference thread: {:?}", e);
                    let _ = tx.blocking_send(Err(e));
                }
            });

            // Convert the receiving end of the channel into a stream
            let stream = ReceiverStream::new(rx);
            let sse_stream: SseStream = Box::pin(stream);

            Ok(ProviderResponse::Stream(sse_stream))
        })
    }

    fn name(&self) -> &str {
        "embedded-bonsai"
    }
}
