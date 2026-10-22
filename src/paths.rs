//! XDG base-directory helpers shared by the credentials store, the plugin
//! core, and the migration framework (v2 L2). Every subsystem used to inline
//! its own `$XDG_*`-with-`$HOME`-fallback chain; these are the single
//! implementation so the plugin state dir, the credentials store, and the
//! migration snapshots agree on one root.

use std::path::PathBuf;

fn env_dir(var: &str, _fallback: &str, home_fallback: &str) -> PathBuf {
    if let Ok(v) = std::env::var(var) {
        let p = PathBuf::from(&v);
        if v.starts_with('/') {
            return p;
        }
        // Relative $XDG_* values are invalid per the spec; treat as unset.
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    PathBuf::from(home).join(home_fallback)
}

/// `$XDG_STATE_HOME` (default `~/.local/state`).
pub fn state_home() -> PathBuf {
    env_dir("XDG_STATE_HOME", "", ".local/state")
}

/// `$XDG_RUNTIME_DIR` (default `/tmp` — the systemd unit always sets it; the
/// fallback keeps development runs working).
pub fn runtime_dir() -> PathBuf {
    if let Ok(v) = std::env::var("XDG_RUNTIME_DIR") {
        let p = PathBuf::from(&v);
        if v.starts_with('/') {
            return p;
        }
    }
    PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string()))
}

/// `$XDG_CONFIG_HOME` (default `~/.config`).
pub fn config_home() -> PathBuf {
    env_dir("XDG_CONFIG_HOME", "", ".config")
}

/// `$XDG_DATA_HOME` (default `~/.local/share`).
pub fn data_home() -> PathBuf {
    env_dir("XDG_DATA_HOME", "", ".local/share")
}

/// `$XDG_STATE_HOME/brainrouter` — the daemon's state root (plugins,
/// credentials, migration snapshots).
pub fn brainrouter_state() -> PathBuf {
    state_home().join("brainrouter")
}

/// `$XDG_RUNTIME_DIR/brainrouter` — the daemon's runtime root (plugin sockets).
pub fn brainrouter_runtime() -> PathBuf {
    runtime_dir().join("brainrouter")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_helpers_are_absolute() {
        assert!(state_home().is_absolute());
        assert!(runtime_dir().is_absolute());
        assert!(config_home().is_absolute());
        assert!(data_home().is_absolute());
        assert_eq!(brainrouter_state(), state_home().join("brainrouter"));
    }
}
