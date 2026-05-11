//! OMP Bridge — chat transports (Discord, Signal) that shell out to `omp`.

pub mod core;
pub mod persist;

#[cfg(feature = "bridge-discord")]
pub mod discord;
#[cfg(feature = "bridge-signal")]
pub mod signal;

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Top-level bridge configuration, deserialized from the brainrouter config.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct BridgeConfig {
    /// Path to the `omp` binary. Default: `"omp"` (relies on $PATH).
    pub omp_path: Option<String>,
    /// Working directory for omp invocations. Default: `$HOME`.
    pub work_dir: Option<String>,
    /// Path to the YAML file containing model aliases.
    /// Default: `"~/.config/omp-bridge/config.yaml"`.
    pub aliases_config: Option<String>,
    /// Timeout in seconds for a single omp invocation. Default: 600.
    pub timeout_secs: Option<u64>,
    /// Model name to use when the user doesn't specify one.
    /// Default: `"brainrouter/auto"`.
    pub default_model: Option<String>,

    /// Discord transport configuration. Absent = disabled.
    #[cfg(feature = "bridge-discord")]
    pub discord: Option<DiscordConfig>,

    /// Signal transport configuration. Absent = disabled.
    #[cfg(feature = "bridge-signal")]
    pub signal: Option<SignalConfig>,
}

impl BridgeConfig {
    pub fn omp_path(&self) -> &str {
        self.omp_path.as_deref().unwrap_or("omp")
    }

    pub fn work_dir(&self) -> String {
        self.work_dir
            .clone()
            .unwrap_or_else(|| std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string()))
    }

    pub fn aliases_config(&self) -> &str {
        self.aliases_config
            .as_deref()
            .unwrap_or("~/.config/omp-bridge/config.yaml")
    }

    pub fn timeout_secs(&self) -> u64 {
        self.timeout_secs.unwrap_or(600)
    }

    pub fn default_model(&self) -> &str {
        self.default_model.as_deref().unwrap_or("brainrouter/auto")
    }
}

// Re-export transport configs so callers can use them from `bridge::`.
#[cfg(feature = "bridge-discord")]
pub use discord::DiscordConfig;
#[cfg(feature = "bridge-signal")]
pub use signal::SignalConfig;

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

/// Status snapshot for a single transport.
#[derive(Debug, Clone, Serialize)]
pub struct TransportStatus {
    pub enabled: bool,
    pub connected: bool,
    pub uptime_secs: u64,
    pub messages_handled: u64,
    pub last_error: Option<String>,
}

/// Aggregated bridge status across all transports.
#[derive(Debug, Clone, Serialize)]
pub struct BridgeStatus {
    pub discord: Option<TransportStatus>,
    pub signal: Option<TransportStatus>,
}

// ---------------------------------------------------------------------------
// BridgeManager
// ---------------------------------------------------------------------------

/// Shared runtime state for the bridge, held behind an `Arc` so transports
/// and the status endpoint can both reference it.
pub struct BridgeManager {
    start_time: Instant,
    discord_messages: AtomicU64,
    signal_messages: AtomicU64,
    discord_connected: std::sync::atomic::AtomicBool,
    signal_connected: std::sync::atomic::AtomicBool,
    discord_last_error: std::sync::Mutex<Option<String>>,
    signal_last_error: std::sync::Mutex<Option<String>>,
    /// Dashboard-controlled pause flag. When false the transport skips dispatch.
    pub discord_enabled: std::sync::atomic::AtomicBool,
    /// Dashboard-controlled pause flag. When false the transport skips dispatch.
    pub signal_enabled: std::sync::atomic::AtomicBool,
}

impl BridgeManager {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            discord_messages: AtomicU64::new(0),
            signal_messages: AtomicU64::new(0),
            discord_connected: std::sync::atomic::AtomicBool::new(false),
            signal_connected: std::sync::atomic::AtomicBool::new(false),
            discord_last_error: std::sync::Mutex::new(None),
            signal_last_error: std::sync::Mutex::new(None),
            discord_enabled: std::sync::atomic::AtomicBool::new(true),
            signal_enabled: std::sync::atomic::AtomicBool::new(true),
        }
    }

    pub fn record_discord_message(&self) {
        self.discord_messages.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_signal_message(&self) {
        self.signal_messages.fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_discord_connected(&self, connected: bool) {
        self.discord_connected
            .store(connected, Ordering::Relaxed);
    }

    pub fn set_signal_connected(&self, connected: bool) {
        self.signal_connected
            .store(connected, Ordering::Relaxed);
    }

    pub fn set_discord_error(&self, err: String) {
        *self.discord_last_error.lock().unwrap() = Some(err);
    }

    pub fn set_signal_error(&self, err: String) {
        *self.signal_last_error.lock().unwrap() = Some(err);
    }

    pub fn status(&self) -> BridgeStatus {
        let uptime = self.start_time.elapsed().as_secs();

        let discord = {
            let enabled = self.discord_enabled.load(Ordering::Relaxed);
            let connected = self.discord_connected.load(Ordering::Relaxed);
            let messages = self.discord_messages.load(Ordering::Relaxed);
            let last_error = self.discord_last_error.lock().unwrap().clone();
            // Report if ever active OR if the enabled flag has been toggled.
            if connected || messages > 0 || last_error.is_some() || !enabled {
                Some(TransportStatus {
                    enabled,
                    connected,
                    uptime_secs: uptime,
                    messages_handled: messages,
                    last_error,
                })
            } else {
                None
            }
        };

        let signal = {
            let enabled = self.signal_enabled.load(Ordering::Relaxed);
            let connected = self.signal_connected.load(Ordering::Relaxed);
            let messages = self.signal_messages.load(Ordering::Relaxed);
            let last_error = self.signal_last_error.lock().unwrap().clone();
            if connected || messages > 0 || last_error.is_some() || !enabled {
                Some(TransportStatus {
                    enabled,
                    connected,
                    uptime_secs: uptime,
                    messages_handled: messages,
                    last_error,
                })
            } else {
                None
            }
        };

        BridgeStatus { discord, signal }
    }
}

impl Default for BridgeManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// Start enabled bridge transports. Spawns each as an independent tokio task;
/// errors are logged, not propagated, so one crashing transport doesn't take
/// down the server.
pub async fn start(config: BridgeConfig, manager: Arc<BridgeManager>) {
    let mut any_started = false;

    // Move fields out of config to get owned values without borrow conflicts.
    let omp_path = config.omp_path.unwrap_or_else(|| "omp".to_string());
    let work_dir = config.work_dir.unwrap_or_else(|| {
        std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string())
    });
    let aliases = config.aliases_config.unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        format!("{}/.config/omp-bridge/config.yaml", home)
    });
    let timeout = config.timeout_secs.unwrap_or(600);
    let default_model = config.default_model.unwrap_or_else(|| "brainrouter/auto".to_string());

    #[cfg(feature = "bridge-discord")]
    let discord_cfg = config.discord;
    #[cfg(feature = "bridge-signal")]
    let signal_cfg = config.signal;
    #[cfg(feature = "bridge-discord")]
    {
        if let Some(dcfg) = discord_cfg {
            if dcfg.is_enabled() {
                let omp_path = omp_path.clone();
                let work_dir = work_dir.clone();
                let aliases = aliases.clone();
                let default_model = default_model.clone();
                let mgr = Arc::clone(&manager);
                tokio::spawn(async move {
                    info!("starting Discord bridge transport");
                    mgr.set_discord_connected(true);
                    if let Err(e) = discord::start(&dcfg, &omp_path, &work_dir, &aliases, timeout, &default_model, Arc::clone(&mgr)).await {
                        mgr.set_discord_connected(false);
                        mgr.set_discord_error(e.to_string());
                        warn!("Discord bridge exited with error: {e}");
                    }
                });
                any_started = true;
            }
        }
    }

    #[cfg(feature = "bridge-signal")]
    {
        if let Some(scfg) = signal_cfg {
            if scfg.is_enabled() {
                let omp_path = omp_path.clone();
                let work_dir = work_dir.clone();
                let aliases = aliases.clone();
                let default_model = default_model.clone();
                let mgr = Arc::clone(&manager);
                tokio::spawn(async move {
                    info!("starting Signal bridge transport");
                    mgr.set_signal_connected(true);
                    if let Err(e) = signal::start(&scfg, &omp_path, &work_dir, &aliases, timeout, &default_model, mgr.clone()).await {
                        mgr.set_signal_connected(false);
                        mgr.set_signal_error(e.to_string());
                        warn!("Signal bridge exited with error: {e}");
                    }
                });
                any_started = true;
            }
        }
    }

    if !any_started {
        info!("no bridge transports configured; bridge idle");
    }
}
