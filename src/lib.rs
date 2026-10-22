pub mod bonsai_server;
pub mod anthropic;
// Benchmark registry core (store schema, ingestion types, and the documented
// `unavailable_response` degradation contract) is always compiled; the
// explorer HTTP surface is gated inside the module. The native lab ships
// only with the `benchmarks` feature (v2 R2).
pub mod benchmark;
#[cfg(feature = "benchmarks")]
pub mod benchmark_lab;
pub mod daemon_client;
pub mod classifier;
pub mod paths;
pub mod credentials;
pub mod migration;
pub mod plugins;
pub mod cockpit_config;
pub mod config;
pub mod escalation;
pub mod inference_state;
pub mod health;
pub mod managed_toolboxes;
pub mod peer_cwd;
pub mod prompt_rewriter;
pub mod project_pins;
pub mod provider;
pub mod review;
pub mod router;
pub mod routing_profile;
pub mod server;
pub mod session;
pub mod stream;
pub mod toolbox_catalog;
pub mod types;
pub mod routing_events;
pub mod inflight;
pub mod inference_proxy;
pub mod llama_swap_register;
pub mod model_downloads;
pub mod observability;
pub mod server_mode;
pub mod serving_identity;
