//! Persistence helpers for session IDs, channel models, and working directories.
//!
//! All three maps are stored as JSON files under `~/.local/share/omp-bridge/`.
//! Writes are best-effort: failures are logged but never fatal.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

fn data_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home).join(".local/share/omp-bridge")
}

// ---------------------------------------------------------------------------
// Generic load/save
// ---------------------------------------------------------------------------

fn load_json_map(path: &Path) -> HashMap<String, String> {
    match std::fs::read_to_string(path) {
        Ok(content) => match serde_json::from_str::<HashMap<String, String>>(&content) {
            Ok(map) => {
                info!("Loaded {} entry/entries from {}", map.len(), path.display());
                map
            }
            Err(e) => {
                warn!("Could not parse {}: {}", path.display(), e);
                HashMap::new()
            }
        },
        Err(_) => HashMap::new(),
    }
}

fn save_json_map(path: &Path, map: &HashMap<String, String>) {
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            warn!("Could not create dir {}: {}", parent.display(), e);
            return;
        }
    }
    match serde_json::to_string(map) {
        Ok(json) => {
            if let Err(e) = std::fs::write(path, json) {
                warn!("Could not write {}: {}", path.display(), e);
            }
        }
        Err(e) => warn!("Could not serialize map for {}: {}", path.display(), e),
    }
}

// ---------------------------------------------------------------------------
// Sessions  (conversation-key → OMP session ID)
// ---------------------------------------------------------------------------

/// `transport` is a short tag like `"discord"` or `"signal"` that namespaces
/// the file so both transports can run concurrently without collisions.
pub fn sessions_path(transport: &str) -> PathBuf {
    data_dir().join(format!("{}-sessions.json", transport))
}

pub fn load_sessions(transport: &str) -> HashMap<String, String> {
    load_json_map(&sessions_path(transport))
}

pub fn save_sessions(transport: &str, sessions: &HashMap<String, String>) {
    save_json_map(&sessions_path(transport), sessions);
}

// ---------------------------------------------------------------------------
// Channel / conversation model preferences
// ---------------------------------------------------------------------------

pub fn channel_models_path(transport: &str) -> PathBuf {
    data_dir().join(format!("{}-channel-models.json", transport))
}

pub fn load_channel_models(transport: &str) -> HashMap<String, String> {
    load_json_map(&channel_models_path(transport))
}

pub fn save_channel_models(transport: &str, models: &HashMap<String, String>) {
    save_json_map(&channel_models_path(transport), models);
}

// ---------------------------------------------------------------------------
// Working directories
// ---------------------------------------------------------------------------

pub fn work_dirs_path(transport: &str) -> PathBuf {
    data_dir().join(format!("{}-work-dirs.json", transport))
}

pub fn load_work_dirs(transport: &str) -> HashMap<String, String> {
    load_json_map(&work_dirs_path(transport))
}

pub fn save_work_dirs(transport: &str, dirs: &HashMap<String, String>) {
    save_json_map(&work_dirs_path(transport), dirs);
}

// ---------------------------------------------------------------------------
// Display helper
// ---------------------------------------------------------------------------

/// Format a path as a virtual path rooted at `/`, relative to `root`.
///
/// Used in user-facing messages so the full host path is never exposed.
pub fn display_path(path: &Path, root: &Path) -> String {
    match path.strip_prefix(root) {
        Ok(rel) if rel == Path::new("") => "/".to_string(),
        Ok(rel) => format!("/{}", rel.display()),
        // Should not happen — sandbox enforces containment — fall back to absolute.
        Err(_) => path.display().to_string(),
    }
}
