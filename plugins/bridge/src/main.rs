//! brainrouter-bridge — plugin entry point (v2 L2).
//!
//! Startup sequence (design `docs/design/v2-streamlining.L2.md`, bridge leg):
//!   1. read supervisor env (token / plugin id / socket / state dir)
//!   2. connect the plugin UDS and `initialize` (failure ⇒ exit nonzero)
//!   3. `config.get {"key":"*"}` + `secrets.get("discord_token")`
//!   4. first-start persistence copy from the v1 data dir (originals kept)
//!   5. start enabled transports exactly as v1 `bridge::start()`
//!   6. push `set_status` every 30 s
//!   7. `config.changed` pushes ⇒ re-read config, flip transport enable
//!      flags (v1 dashboard-toggle parity) / restart transports when
//!      non-LLM fields (timeout, aliases, default_model, …) change

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use tracing::{error, info, warn};

use brainrouter_bridge::bridge::{self, persist, BridgeConfig, BridgeManager};
use brainrouter_bridge::rpc::RpcClient;

/// v1 persistence files (`~/.local/share/omp-bridge/`), copied once into
/// the plugin state dir on first start.
const V1_STATE_FILES: [&str; 6] = [
    "discord-sessions.json",
    "discord-channel-models.json",
    "discord-work-dirs.json",
    "signal-sessions.json",
    "signal-channel-models.json",
    "signal-work-dirs.json",
];

const STATUS_PUSH_INTERVAL: Duration = Duration::from_secs(30);

fn main() -> std::process::ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    match tokio::runtime::Runtime::new() {
        Ok(rt) => match rt.block_on(run()) {
            Ok(()) => std::process::ExitCode::SUCCESS,
            Err(e) => {
                error!("bridge plugin exiting: {e:#}");
                std::process::ExitCode::FAILURE
            }
        },
        Err(e) => {
            error!("cannot start tokio runtime: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<()> {
    // ---- 1. supervisor environment -------------------------------------
    let token =
        std::env::var("BRAINROUTER_PLUGIN_TOKEN").context("BRAINROUTER_PLUGIN_TOKEN missing")?;
    let plugin_id =
        std::env::var("BRAINROUTER_PLUGIN_ID").unwrap_or_else(|_| "bridge".to_string());
    let socket =
        std::env::var("BRAINROUTER_PLUGIN_SOCKET").context("BRAINROUTER_PLUGIN_SOCKET missing")?;
    let state_dir =
        std::env::var("BRAINROUTER_PLUGIN_STATE").context("BRAINROUTER_PLUGIN_STATE missing")?;
    if let Ok(daemon) = std::env::var("BRAINROUTER_DAEMON_SOCKET") {
        info!("daemon socket (informational): {daemon}");
    }

    // The state dir is created 0700 by the daemon; self-heal for robustness
    // (persist.rs writes all runtime state here).
    std::fs::create_dir_all(&state_dir)
        .with_context(|| format!("creating plugin state dir {state_dir}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&state_dir, std::fs::Permissions::from_mode(0o700));
    }
    std::env::set_var("BRAINROUTER_PLUGIN_STATE", &state_dir);
    let state_dir = PathBuf::from(state_dir);

    // ---- 2. connect + handshake ----------------------------------------
    let (rpc, pushes) = RpcClient::connect(&socket)
        .await
        .context("connecting to the daemon plugin socket")?;
    let rpc = Arc::new(rpc);

    let init = rpc
        .initialize("bridge", "bridge", &token)
        .await
        .context("initialize handshake failed")?;
    let got_id = init.get("plugin_id").and_then(Value::as_str).unwrap_or("");
    if got_id != plugin_id {
        return Err(anyhow!("initialize returned plugin_id {got_id:?}, expected {plugin_id:?}"));
    }
    let proto = init.get("protocol_version").and_then(Value::as_u64);
    if proto != Some(1) {
        return Err(anyhow!("unsupported daemon protocol_version {proto:?} (expected 1)"));
    }
    info!(
        daemon_version = %init.get("daemon_version").and_then(|v| v.as_str()).unwrap_or("?"),
        grants = ?init.get("grants").and_then(|v| v.as_array()).map(|v| v.len()).unwrap_or(0),
        "handshake ok"
    );

    // ---- 3. config + credentials ---------------------------------------
    let mut config = fetch_config(&rpc).await.context("startup config.get failed")?;
    let mut spawned = apply_token(&rpc, &mut config, true)
        .await
        .context("startup credential resolution failed")?;

    // ---- 4. first-start persistence copy (v1 dir → plugin state) --------
    migrate_v1_state(&state_dir);

    // ---- 5. transports (v1 lifecycle) -----------------------------------
    let manager = Arc::new(BridgeManager::new());
    apply_enabled_flags(&manager, &config);
    let mut handles = bridge::spawn(config.clone(), Arc::clone(&manager)).await;
    info!("bridge plugin running (state dir {})", state_dir.display());

    // ---- 6. periodic status push ---------------------------------------
    {
        let rpc = Arc::clone(&rpc);
        let manager = Arc::clone(&manager);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(STATUS_PUSH_INTERVAL).await;
                let text = status_line(&manager);
                if let Err(e) = rpc.set_status(&text).await {
                    warn!("set_status failed: {e}");
                }
            }
        });
    }

    // ---- 7. config watcher (daemon push notifications) ------------------
    let mut pushes = pushes;
    loop {
        let Some(frame) = pushes.recv().await else {
            // Daemon closed the connection; exit nonzero so the supervisor
            // crash-loops (re-connects) us.
            return Err(anyhow!("daemon push channel closed; exiting for supervisor restart"));
        };
        let method = frame.get("method").and_then(Value::as_str).unwrap_or("");
        if method != "config.changed" {
            info!("ignoring daemon push {method:?}");
            continue;
        }
        info!(
            key = %frame.get("params").and_then(|p| p.get("key")).and_then(|v| v.as_str()).unwrap_or("*"),
            "config.changed: reloading bridge config"
        );
        match reload_and_apply(&rpc, &mut config, &manager, &mut handles, &mut spawned).await {
            Ok(()) => {}
            Err(e) => warn!("config reload failed, keeping previous config: {e:#}"),
        }
    }
}

/// `config.get {"key":"*"}` → the v1 `BridgeConfig` (bridge.* shape minus
/// `discord.token`, which arrives via `secrets.get`).
async fn fetch_config(rpc: &Arc<RpcClient>) -> Result<BridgeConfig> {
    let res = rpc
        .config_get("*")
        .await
        .context("config.get {\"key\":\"*\"}")?;
    let value = res
        .get("value")
        .cloned()
        .ok_or_else(|| anyhow!("config.get response has no `value` field: {res}"))?;
    let cfg: BridgeConfig =
        serde_json::from_value(value).context("parsing bridge config from daemon")?;
    info!(
        "bridge config: discord={} signal={}",
        transport_enabled_discord(&cfg),
        transport_enabled_signal(&cfg)
    );
    Ok(cfg)
}

/// Fill `discord.token` from `secrets.get("discord_token")`.
/// Fatal at boot (credentials contract); best-effort on reload.
/// Returns the set of transports the config spawns.
async fn apply_token(
    rpc: &Arc<RpcClient>,
    config: &mut BridgeConfig,
    fatal: bool,
) -> Result<SpawnedState> {
    #[cfg(feature = "bridge-discord")]
    if let Some(d) = config.discord.as_mut() {
        if d.is_enabled() {
            match rpc.secrets_get("discord_token").await {
                Ok(v) => {
                    let tok = v.get("value").and_then(Value::as_str).unwrap_or("");
                    if tok.is_empty() {
                        let msg = "secrets.get(discord_token) returned an empty value";
                        if fatal {
                            report_fatal(rpc, msg).await;
                            return Err(anyhow!(msg));
                        }
                        return Err(anyhow!(msg));
                    }
                    d.token = Some(tok.to_string());
                    info!("discord token resolved from credential store");
                }
                Err(e) => {
                    let msg = format!("secrets.get(discord_token) failed: {e:#}");
                    if fatal {
                        report_fatal(rpc, &msg).await;
                        return Err(anyhow!(msg));
                    }
                    return Err(anyhow!(msg));
                }
            }
        }
    }
    Ok(SpawnedState {
        discord: discord_enabled(config),
        signal: signal_enabled(config),
    })
}

async fn report_fatal(rpc: &Arc<RpcClient>, msg: &str) {
    error!("{msg}");
    let _ = rpc
        .set_status(&format!("bridge: fatal at startup — {msg}"))
        .await;
}

/// Re-read config and apply it: v1-toggle parity for `enabled` flips
/// (AtomicBool dispatch gates, transports stay up); anything else
/// (timeout, aliases, default_model, omp_path, work_dir, transport fields)
/// restarts the transports with the new config.
async fn reload_and_apply(
    rpc: &Arc<RpcClient>,
    current: &mut BridgeConfig,
    manager: &Arc<BridgeManager>,
    handles: &mut Vec<tokio::task::JoinHandle<()>>,
    spawned: &mut SpawnedState,
) -> Result<()> {
    let mut fresh = fetch_config(rpc).await?;
    // Keep the previously resolved token if the credential fetch fails at
    // runtime (v1: transports keep serving; the flag path never needs it).
    let old_token = config_discord_token(current);
    // Inject the token without boot-fatality (same code, fatal=false); on a
    // credential failure fall back to the token the transports are running
    // with, so the toggle path never needs to respawn without a token.
    let new_spawned = match apply_token(rpc, &mut fresh, false).await {
        Ok(s) => s,
        Err(e) => {
            warn!("token refresh failed on reload: {e:#}; keeping running transports unchanged");
            match old_token {
                Some(t) => {
                    set_discord_token(&mut fresh, Some(t));
                    SpawnedState {
                        discord: discord_enabled(&fresh),
                        signal: signal_enabled(&fresh),
                    }
                }
                None => return Err(e),
            }
        }
    };

    let flags_only = config_equal_ignoring_enabled(current, &fresh);
    let enabling_new =
        (new_spawned.discord && !spawned.discord) || (new_spawned.signal && !spawned.signal);

    if flags_only && !enabling_new {
        info!("config change is enabled-flag-only; toggling dispatch gates (v1 toggle parity)");
        apply_enabled_flags(manager, &fresh);
        *spawned = new_spawned;
        // persist for future comparisons
        // (current is replaced by the caller-side swap below)
        swap_config(current, &fresh);
        return Ok(());
    }

    info!("config change requires a transport restart; respawning transports");
    for h in handles.drain(..) {
        h.abort();
    }
    apply_enabled_flags(manager, &fresh);
    *handles = bridge::spawn(fresh.clone(), Arc::clone(manager)).await;
    *spawned = new_spawned;
    swap_config(current, &fresh);
    Ok(())
}

fn swap_config(current: &mut BridgeConfig, fresh: &BridgeConfig) {
    *current = fresh.clone();
}

/// Flip the v1 dashboard pause flags from the config `enabled` fields.
fn apply_enabled_flags(manager: &BridgeManager, config: &BridgeConfig) {
    #[cfg(feature = "bridge-discord")]
    if let Some(d) = config.discord.as_ref() {
        manager
            .discord_enabled
            .store(d.is_enabled(), Ordering::Relaxed);
    }
    #[cfg(feature = "bridge-signal")]
    if let Some(s) = config.signal.as_ref() {
        manager
            .signal_enabled
            .store(s.is_enabled(), Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SpawnedState {
    discord: bool,
    signal: bool,
}

fn discord_enabled(config: &BridgeConfig) -> bool {
    #[cfg(feature = "bridge-discord")]
    {
        config.discord.as_ref().map(|d| d.is_enabled()).unwrap_or(false)
    }
    #[cfg(not(feature = "bridge-discord"))]
    {
        let _ = config;
        false
    }
}

fn signal_enabled(config: &BridgeConfig) -> bool {
    #[cfg(feature = "bridge-signal")]
    {
        config.signal.as_ref().map(|s| s.is_enabled()).unwrap_or(false)
    }
    #[cfg(not(feature = "bridge-signal"))]
    {
        let _ = config;
        false
    }
}

fn transport_enabled_discord(config: &BridgeConfig) -> &'static str {
    match config_enabled_discord(config) {
        Some(v) => {
            if v {
                "enabled"
            } else {
                "disabled"
            }
        }
        None => "absent",
    }
}

fn transport_enabled_signal(config: &BridgeConfig) -> &'static str {
    match config_enabled_signal(config) {
        Some(v) => {
            if v {
                "enabled"
            } else {
                "disabled"
            }
        }
        None => "absent",
    }
}

fn config_enabled_discord(config: &BridgeConfig) -> Option<bool> {
    #[cfg(feature = "bridge-discord")]
    {
        config.discord.as_ref().map(|d| d.is_enabled())
    }
    #[cfg(not(feature = "bridge-discord"))]
    {
        let _ = config;
        None
    }
}

fn config_enabled_signal(config: &BridgeConfig) -> Option<bool> {
    #[cfg(feature = "bridge-signal")]
    {
        config.signal.as_ref().map(|s| s.is_enabled())
    }
    #[cfg(not(feature = "bridge-signal"))]
    {
        let _ = config;
        None
    }
}

fn config_discord_token(config: &BridgeConfig) -> Option<String> {
    #[cfg(feature = "bridge-discord")]
    {
        config.discord.as_ref().and_then(|d| d.token.clone())
    }
    #[cfg(not(feature = "bridge-discord"))]
    {
        let _ = config;
        None
    }
}

fn set_discord_token(config: &mut BridgeConfig, token: Option<String>) {
    #[cfg(feature = "bridge-discord")]
    if let Some(d) = config.discord.as_mut() {
        d.token = token;
    }
    #[cfg(not(feature = "bridge-discord"))]
    {
        let _ = (config, token);
    }
}

/// True when the two configs differ only in `discord.enabled` /
/// `signal.enabled` (the token is plugin-side and never compared).
fn config_equal_ignoring_enabled(a: &BridgeConfig, b: &BridgeConfig) -> bool {
    fn strip(cfg: &BridgeConfig) -> Value {
        let mut v = serde_json::to_value(cfg).unwrap_or(Value::Null);
        if let Some(obj) = v.as_object_mut() {
            obj.remove("discord").map(remove_keys);
            obj.remove("signal").map(remove_keys);
        }
        fn remove_keys(mut t: Value) -> Value {
            if let Some(o) = t.as_object_mut() {
                o.remove("enabled");
                o.remove("token");
            }
            t
        }
        // Re-insert the transports with enabled/token stripped so absent vs
        // present-empty sections still compare structurally.
        if let Some(d) = cfg.discord.as_ref().and_then(json_of) {
            obj_discord(&mut v, d);
        }
        if let Some(s) = cfg.signal.as_ref().and_then(json_of) {
            obj_signal(&mut v, s);
        }
        fn obj_discord(v: &mut Value, mut d: Value) {
            if let (Some(o), Some(dm)) = (v.as_object_mut(), d.as_object_mut()) {
                dm.remove("enabled");
                dm.remove("token");
                o.insert("discord".into(), Value::Object(dm.clone()));
            }
        }
        fn obj_signal(v: &mut Value, mut s: Value) {
            if let (Some(o), Some(so)) = (v.as_object_mut(), s.as_object_mut()) {
                so.remove("enabled");
                o.insert("signal".into(), Value::Object(so.clone()));
            }
        }
        v
    }
    strip(a) == strip(b)
}

fn json_of<T: serde::Serialize>(v: &T) -> Option<Value> {
    serde_json::to_value(v).ok()
}

/// One-line status for `set_status` (v1 BridgeStatus snapshot).
fn status_line(manager: &BridgeManager) -> String {
    let status = manager.status();
    let mut parts: Vec<String> = Vec::new();
    if let Some(d) = status.discord {
        parts.push(format!(
            "discord: {}, {} msgs",
            if !d.enabled {
                "disabled"
            } else if d.connected {
                "connected"
            } else {
                "disconnected"
            },
            d.messages_handled
        ));
    }
    if let Some(s) = status.signal {
        parts.push(format!(
            "signal: {}, {} msgs",
            if !s.enabled {
                "disabled"
            } else if s.connected {
                "connected"
            } else {
                "disconnected"
            },
            s.messages_handled
        ));
    }
    if parts.is_empty() {
        return "bridge idle (no transports configured)".to_string();
    }
    parts.join("; ")
}

/// One-time first-start copy of the six v1 persistence files into the
/// plugin state dir. v1 originals are LEFT in place (revert path).
fn migrate_v1_state(state_dir: &Path) {
    let v1 = persist::v1_data_dir();
    if v1 == state_dir {
        return; // standalone run pointed at the v1 dir: nothing to do
    }
    let mut copied = 0usize;
    for name in V1_STATE_FILES {
        let src = v1.join(name);
        let dst = state_dir.join(name);
        if !src.is_file() || dst.exists() {
            continue;
        }
        match std::fs::copy(&src, &dst) {
            Ok(n) => {
                info!("migrated v1 bridge state file {name} ({n} bytes)");
                copied += 1;
            }
            Err(e) => warn!("could not copy v1 bridge state file {name}: {e}"),
        }
    }
    if copied > 0 {
        info!("persisted bridge state migrated ({copied} files copied, originals kept)");
    }
}
