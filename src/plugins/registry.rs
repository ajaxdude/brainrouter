//! Installed-plugin registry (`$XDG_STATE_HOME/brainrouter/plugins/plugins.json`).
//!
//! Persisted entries are the supervisor's source of truth across restarts:
//! a terminal `failed` state survives daemon restarts (no supervision until
//! an explicit `enable`).

use super::manifest::PluginManifest;
use crate::plugins::store;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginState {
    /// Enabled and supervised (child running).
    Running,
    /// Not supervised: disabled, cleanly exited, or installed-but-idle.
    Stopped,
    /// 5 consecutive crashes: terminal. No supervision until explicit enable.
    Failed,
}

impl PluginState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Stopped => "stopped",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginEntry {
    pub id: String,
    pub name: String,
    pub version: String,
    pub binary: String,
    pub min_daemon_version: String,
    pub grants: Vec<String>,
    #[serde(default)]
    pub config_schema: Option<String>,
    /// `plugin enable` / `plugin disable`.
    pub enabled: bool,
    pub state: PluginState,
    /// Consecutive crashes (reset on a long-running child, on enable, and on
    /// a clean exit).
    #[serde(default)]
    pub consecutive_crashes: u32,
    #[serde(default)]
    pub total_restarts: u64,
    #[serde(default)]
    pub last_exit: Option<String>,
    /// Latest `set_status` line from the plugin.
    #[serde(default)]
    pub status_line: String,
    #[serde(default)]
    pub pid: Option<u32>,
    pub installed_at: String,
}

impl PluginEntry {
    pub fn from_manifest(manifest: &PluginManifest) -> Self {
        Self {
            id: manifest.id.clone(),
            name: manifest.name.clone(),
            version: manifest.version.clone(),
            binary: manifest.binary.clone(),
            min_daemon_version: manifest.min_daemon_version.clone(),
            grants: manifest.grants.clone(),
            config_schema: manifest.config_schema.clone(),
            enabled: false,
            state: PluginState::Stopped,
            consecutive_crashes: 0,
            total_restarts: 0,
            last_exit: None,
            status_line: String::new(),
            pid: None,
            installed_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    pub fn masked_json(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "binary": self.binary,
            "min_daemon_version": self.min_daemon_version,
            "grants": self.grants,
            "enabled": self.enabled,
            "state": self.state.as_str(),
            "consecutive_crashes": self.consecutive_crashes,
            "total_restarts": self.total_restarts,
            "last_exit": self.last_exit,
            "status_line": self.status_line,
            "pid": self.pid,
            "installed_at": self.installed_at,
        })
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Registry {
    #[serde(default)]
    pub plugins: BTreeMap<String, PluginEntry>,
}

impl Registry {
    pub fn path(state_root: &Path) -> PathBuf {
        store::plugins_state_root(state_root).join("plugins.json")
    }

    pub fn load(state_root: &Path) -> Result<Self> {
        let path = Self::path(state_root);
        match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text)
                .with_context(|| format!("parsing plugin registry {}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e).context("reading plugin registry"),
        }
    }

    pub fn save(&self, state_root: &Path) -> Result<()> {
        let path = Self::path(state_root);
        let text = serde_json::to_string_pretty(self).context("serializing plugin registry")?;
        store::atomic_write_0600(&path, text.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_roundtrip_persists_terminal_failed() {
        let dir = tempfile::tempdir().unwrap();
        let mut reg = Registry::load(dir.path()).unwrap();
        assert!(reg.plugins.is_empty());
        let mut e = PluginEntry::from_manifest(&PluginManifest {
            id: "bridge".into(),
            name: "Bridge".into(),
            version: "0.1.0".into(),
            binary: "brainrouter-bridge".into(),
            min_daemon_version: "1.0.0".into(),
            grants: vec!["secrets.get".into()],
            config_schema: None,
        });
        e.enabled = true;
        e.state = PluginState::Failed;
        e.consecutive_crashes = 5;
        reg.plugins.insert("bridge".into(), e);
        reg.save(dir.path()).unwrap();

        let reg2 = Registry::load(dir.path()).unwrap();
        let e2 = &reg2.plugins["bridge"];
        assert_eq!(e2.state, PluginState::Failed);
        assert!(e2.enabled);
        assert_eq!(e2.consecutive_crashes, 5);
        // State file is owner-only.
        assert_eq!(
            std::os::unix::fs::PermissionsExt::mode(
                &std::fs::metadata(Registry::path(dir.path()))
                    .unwrap()
                    .permissions()
            ) & 0o777,
            0o600
        );
    }
}
