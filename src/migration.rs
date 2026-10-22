//! Config migration framework (v2 L2).
//!
//! Cross-leg contract: every leg that migrates state FIRST snapshots the
//! pre-migration state (owner-only), supports `--dry-run` preview, logs what
//! it did, and ships a named restore command. Boot-time migrations run before
//! config validation so v1 YAML always boots.
//!
//! Snapshot layout:
//! `$XDG_STATE_HOME/brainrouter/migrations/<phase-tag>/<UTC timestamp>/`
//! containing a `snapshot/` directory with the pre-migration config YAML and
//! `credentials.json` (whichever exist). `restore` reads the newest snapshot
//! for a phase.

use crate::credentials::{CredentialKind, CredentialRecord, CredentialStore};
use anyhow::{Context, Result};
use std::os::unix::fs::PermissionsExt;
use serde_yaml::Value as YamlValue;
use std::path::{Path, PathBuf};

/// The L2 phase tag: `brainrouter migration restore v2.0-p2`.
pub const PHASE_V2_P2: &str = "v2.0-p2";

/// Credential id the Discord-token migration creates (stable, conventional
/// id; the plugin resolves it via `secrets.get`).
pub const DISCORD_TOKEN_CREDENTIAL_ID: &str = "discord_token";

/// The plugin id the migrated credential grants.
pub const BRIDGE_PLUGIN_ID: &str = "bridge";

/// The `v2.0-p2` migration: move the v1 plaintext `bridge.discord.token` into
/// the credentials store (granted to plugin `bridge`), snapshot first, then
/// strip the key from the YAML (with a `.bak` sidecar). Idempotent.
///
/// `state_root` is `$XDG_STATE_HOME/brainrouter` (snapshots live there).
/// Returns the human-readable actions taken (worded for logs and CLI output,
/// token value masked). Empty = nothing to do.
pub fn run_v2_p2(
    config_path: &Path,
    store: &CredentialStore,
    state_root: &Path,
) -> Result<Vec<String>> {
    let Some(token) = read_discord_token(config_path)? else {
        return Ok(Vec::new());
    };
    let snapshot_dir = snapshot(config_path, store, state_root, PHASE_V2_P2)?;

    store.put(CredentialRecord {
        id: DISCORD_TOKEN_CREDENTIAL_ID.to_string(),
        description: "Discord bot token (migrated from v1 bridge.discord.token)".to_string(),
        kind: CredentialKind::Store {
            value: crate::credentials::SecretString::new(token),
        },
        grants: vec![BRIDGE_PLUGIN_ID.to_string()],
        created_at: String::new(),
        updated_at: String::new(),
    })?;

    strip_discord_token(config_path)?;

    Ok(vec![format!(
        "migrated v1 bridge.discord.token into the credentials store as credential \
         '{DISCORD_TOKEN_CREDENTIAL_ID}' (kind store, grants: {BRIDGE_PLUGIN_ID}); \
         key removed from {} (backup: {}.bak); pre-migration snapshot: {}",
        config_path.display(),
        config_path.display(),
        snapshot_dir.display()
    )])
}

/// `--dry-run` preview: describe exactly what the `v2.0-p2` migration would
/// do, without touching any state. Token values are never rendered.
pub fn preview_v2_p2(
    config_path: &Path,
    store_path: &Path,
    state_root: &Path,
) -> Result<Vec<String>> {
    let mut lines = Vec::new();
    lines.push(format!("dry-run preview: phase {PHASE_V2_P2} (state untouched)"));
    match read_discord_token(config_path)? {
        Some(_token) => {
            lines.push(format!(
                "  would migrate bridge.discord.token (value masked) from {} into credential \
                 '{DISCORD_TOKEN_CREDENTIAL_ID}' (kind store, grants: {BRIDGE_PLUGIN_ID})",
                config_path.display()
            ));
            lines.push(format!(
                "  would snapshot the pre-migration config + credentials under {}",
                snapshot_root(state_root, PHASE_V2_P2).display()
            ));
            lines.push(format!(
                "  would remove bridge.discord.token from {} (backup: {}.bak)",
                config_path.display(),
                config_path.display()
            ));
        }
        None => lines.push(format!(
            "  no bridge.discord.token in {} — migration {PHASE_V2_P2} has nothing to do",
            config_path.display()
        )),
    }
    if store_path.exists() {
        lines.push(format!("  credentials store present: {}", store_path.display()));
    } else {
        lines.push(format!(
            "  credentials store would be created at {}",
            store_path.display()
        ));
    }
    Ok(lines)
}

/// `brainrouter migration restore <phase-tag>`: restore the newest snapshot
/// for the phase — config YAML and credentials store files — and with them
/// the pre-migration YAML keys (the Discord token) and grants.
pub fn restore(
    phase: &str,
    config_path: &Path,
    store: &CredentialStore,
    state_root: &Path,
) -> Result<String> {
    let latest = newest_snapshot(phase, state_root)?.with_context(|| {
        format!(
            "no {phase} migration snapshot found under {}",
            snapshot_root(state_root, phase).display()
        )
    })?;
    let snap_dir = latest.clone();
    let mut restored = Vec::new();

    let snap_config = snap_dir.join(SNAPSHOT_CONFIG_FILE);
    if snap_config.exists() {
        let text = std::fs::read_to_string(&snap_config)
            .with_context(|| format!("reading snapshot {}", snap_config.display()))?;
        std::fs::write(config_path, &text)
            .with_context(|| format!("restoring config to {}", config_path.display()))?;
        restored.push("config YAML (v1 keys incl. bridge.discord.token)");
    }
    let snap_creds = snap_dir.join(SNAPSHOT_CREDENTIALS_FILE);
    if snap_creds.exists() {
        // Restore the pre-migration credential set verbatim (this drops the
        // migrated credential + its grants, i.e. the v1 state).
        let text = std::fs::read_to_string(&snap_creds)
            .with_context(|| format!("reading snapshot {}", snap_creds.display()))?;
        let records = extract_records(&text)?;
        store.replace_all(records)?;
        restored.push("credentials store (pre-migration values + grants)");
    }
    if restored.is_empty() {
        anyhow::bail!("snapshot {} contains no restorable files", snap_dir.display());
    }
    Ok(format!(
        "restored {phase} from {} ({})",
        latest.display(),
        restored.join(", ")
    ))
}

fn extract_records(creds_json: &str) -> Result<Vec<CredentialRecord>> {
    let v: serde_json::Value =
        serde_json::from_str(creds_json).context("parsing snapshot credentials.json")?;
    let creds = v
        .get("credentials")
        .and_then(|c| c.as_array())
        .cloned()
        .unwrap_or_default();
    let mut out = Vec::new();
    for c in creds {
        out.push(serde_json::from_value(c).context("decoding snapshot credential")?);
    }
    Ok(out)
}

pub fn snapshot_root(state_root: &Path, phase: &str) -> PathBuf {
    state_root.join("migrations").join(phase)
}

/// Snapshot files (directly under the timestamp directory).
pub const SNAPSHOT_CONFIG_FILE: &str = "snapshot-config.yaml";
pub const SNAPSHOT_CREDENTIALS_FILE: &str = "snapshot-credentials.json";

pub fn newest_snapshot(phase: &str, state_root: &Path) -> Result<Option<PathBuf>> {
    let root = snapshot_root(state_root, phase);
    let mut dirs: Vec<PathBuf> = match std::fs::read_dir(&root) {
        Ok(rd) => rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).context("listing migration snapshots"),
    };
    // Timestamps are fixed-width `YYYYMMDDTHHMMSSZ`, so lexical order is time order.
    dirs.sort();
    Ok(dirs.pop())
}

/// Copy the current config YAML + credentials.json (whichever exist) into a
/// fresh `snapshot/` directory for the phase. Returns the snapshot dir.
pub fn snapshot(
    config_path: &Path,
    store: &CredentialStore,
    state_root: &Path,
    phase: &str,
) -> Result<PathBuf> {
    let ts = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let migrations = state_root.join("migrations");
    let dir = snapshot_root(state_root, phase).join(&ts);
    std::fs::create_dir_all(&dir).context("creating migration snapshot directory")?;
    for d in [state_root, &migrations, &snapshot_root(state_root, phase)] {
        std::fs::set_permissions(d, std::fs::Permissions::from_mode(0o700)).ok();
    }

    if config_path.exists() {
        let text = std::fs::read_to_string(config_path)
            .with_context(|| format!("reading {} for snapshot", config_path.display()))?;
        std::fs::write(dir.join(SNAPSHOT_CONFIG_FILE), text)?;
    }
    if store.path().exists() {
        let text = std::fs::read_to_string(store.path())
            .with_context(|| format!("reading {} for snapshot", store.path().display()))?;
        std::fs::write(dir.join(SNAPSHOT_CREDENTIALS_FILE), text)?;
    }
    Ok(dir)
}

/// The plaintext `bridge.discord.token` value from raw YAML, if present.
pub fn read_discord_token(config_path: &Path) -> Result<Option<String>> {
    let text = match std::fs::read_to_string(config_path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).context("reading config for migration"),
    };
    let root: YamlValue = serde_yaml::from_str(&text)
        .with_context(|| format!("parsing {} for migration", config_path.display()))?;
    let token = root
        .get("bridge")
        .and_then(|b| b.get("discord"))
        .and_then(|d| d.get("token"))
        .and_then(|t| t.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    Ok(token)
}

/// Rewrite the config YAML with `bridge.discord.token` removed, leaving a
/// `.bak` copy of the original.
fn strip_discord_token(config_path: &Path) -> Result<()> {
    let text = std::fs::read_to_string(config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    std::fs::write(bak_path(config_path), &text).context("writing config .bak backup")?;
    let mut root: YamlValue = serde_yaml::from_str(&text).context("re-parsing config YAML")?;
    if let Some(map) = root.as_mapping_mut() {
        let bridge_key = YamlValue::String("bridge".into());
        if let Some(bridge) = map.get_mut(&bridge_key) {
            if let Some(b) = bridge.as_mapping_mut() {
                let discord_key = YamlValue::String("discord".into());
                if let Some(discord) = b.get_mut(&discord_key) {
                    if let Some(d) = discord.as_mapping_mut() {
                        d.remove(YamlValue::String("token".into()));
                    }
                }
            }
        }
    }
    let out = serde_yaml::to_string(&root).context("serializing migrated config YAML")?;
    std::fs::write(config_path, out).context("writing migrated config YAML")?;
    Ok(())
}

fn bak_path(config_path: &Path) -> PathBuf {
    let mut p = config_path.as_os_str().to_os_string();
    p.push(".bak");
    PathBuf::from(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v1_yaml(token: &str) -> String {
        format!(
            "manifest:\n  base_url: http://127.0.0.1:1/cloud/v1\n  enabled: false\n\
             llama_swap:\n  base_url: http://127.0.0.1:1/v1\n  fallback_model: m\n\
             bonsai:\n  enabled: false\n  fork_path: /nonexistent\n\
             bridge:\n  omp_path: /nonexistent/omp\n  discord:\n    enabled: false\n    token: {token}\n  signal:\n    enabled: false\n"
        )
    }

    #[test]
    fn migration_moves_the_token_out_of_yaml() {
        let dir = tempfile::tempdir().unwrap();
        let state = dir.path().join("state");
        let config = dir.path().join("config.yaml");
        std::fs::write(&config, v1_yaml("PLAINTEXT-TOK-1234")).unwrap();
        let store = CredentialStore::load(dir.path().join("credentials.json")).unwrap();

        let actions = run_v2_p2(&config, &store, &state).unwrap();
        assert_eq!(actions.len(), 1);
        assert!(!actions[0].contains("PLAINTEXT-TOK-1234"), "actions are log-safe: {}", actions[0]);

        // Value lives in the store, granted to `bridge`.
        assert_eq!(store.resolve("discord_token").unwrap().expose(), "PLAINTEXT-TOK-1234");
        assert!(store.get_masked("discord_token").unwrap()["grants"]
            .to_string()
            .contains("bridge"));

        // YAML no longer has the key; .bak keeps the original.
        let after = std::fs::read_to_string(&config).unwrap();
        assert!(!after.contains("PLAINTEXT-TOK-1234"));
        assert!(after.contains("bridge:"), "remaining bridge keys survive: {after}");
        let bak = std::fs::read_to_string(dir.path().join("config.yaml.bak")).unwrap();
        assert!(bak.contains("PLAINTEXT-TOK-1234"));

        // Snapshot under the snapshot-named path carries the pre-migration YAML.
        let snap = newest_snapshot(PHASE_V2_P2, &state).unwrap().expect("snapshot exists");
        let found = std::fs::read_to_string(snap.join(SNAPSHOT_CONFIG_FILE)).unwrap();
        assert!(found.contains("PLAINTEXT-TOK-1234"));

        // Idempotent: a second run sees no token key.
        assert!(run_v2_p2(&config, &store, &state).unwrap().is_empty());

        // Restore puts the YAML key (and value) back from the snapshot.
        let msg = restore(PHASE_V2_P2, &config, &store, &state).unwrap();
        assert!(msg.contains("restored v2.0-p2"));
        assert!(std::fs::read_to_string(&config).unwrap().contains("PLAINTEXT-TOK-1234"));
        assert!(store.exists("discord_token").unwrap());
    }

    #[test]
    fn dry_run_previews_without_writing() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.yaml");
        std::fs::write(&config, v1_yaml("TOK-DRY")).unwrap();
        let store_path = dir.path().join("credentials.json");
        let lines = preview_v2_p2(&config, &store_path, &dir.path().join("state")).unwrap();
        let text = lines.join("\n").to_lowercase();
        assert!(text.contains("discord") && text.contains("migrat") && text.contains("dry"));
        assert!(!text.contains("tok-dry"), "preview masks the value");
        assert!(!store_path.exists());
        assert!(!dir.path().join("state").exists());
        assert!(std::fs::read_to_string(&config).unwrap().contains("TOK-DRY"));
    }
}
