//! brainrouter bridge plugin: Discord + Signal chat transports that shell
//! out to `omp` (v1 `src/bridge/`, moved out of the daemon verbatim — design
//! R3, behavior-unchanged).
//!
//! Runs as a supervised child of the brainrouter daemon; talks to the daemon
//! over the plugin UDS with newline-delimited JSON-RPC 2.0 ([`rpc`]).

pub mod bridge;
pub mod rpc;
