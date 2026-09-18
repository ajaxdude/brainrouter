//! Persisted runtime state for the code-review master switch (design FR-A).
//!
//! A tiny JSON file kept next to `routing_state.json` recording whether
//! Brainrouter's code reviewer is enabled. The reviewer is **on by default**;
//! an absent, unreadable, or corrupt file falls back to on and never blocks
//! startup. Writes are atomic (temp file in the same directory + rename), the
//! same durability contract as the existing `routing_state.json` writer.
//!
//! Deliberately minimal: this first release persists only the single
//! `code_review_enabled` flag. The deferred hardening track (see
//! `docs/design/hankndory-brainrouter-integration.md`) layers a richer runtime
//! state and an append-only ledger on top of this file's `schema_version`.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const FILE_NAME: &str = "review_runtime_state.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReviewRuntimeState {
    #[serde(default = "default_schema_version")]
    schema_version: u32,
    #[serde(default = "default_true")]
    code_review_enabled: bool,
}

fn default_schema_version() -> u32 {
    1
}
fn default_true() -> bool {
    true
}

impl Default for ReviewRuntimeState {
    fn default() -> Self {
        ReviewRuntimeState {
            schema_version: 1,
            code_review_enabled: true,
        }
    }
}

/// Canonical path for the review runtime-state file (beside `routing_state.json`).
pub fn state_path() -> PathBuf {
    crate::config::default_config_path().with_file_name(FILE_NAME)
}

/// Read the persisted `code_review_enabled` flag. Absent, unreadable, or
/// corrupt ⇒ `true` (reviewer on by default). Never panics; never blocks
/// startup.
pub fn load_enabled(path: &Path) -> bool {
    match std::fs::read(path) {
        Ok(bytes) => match serde_json::from_slice::<ReviewRuntimeState>(&bytes) {
            Ok(state) => state.code_review_enabled,
            Err(e) => {
                tracing::warn!(
                    path = %path.display(), error = %e,
                    "Ignoring corrupt review_runtime_state.json; defaulting code review to on"
                );
                true
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => true,
        Err(e) => {
            tracing::warn!(
                path = %path.display(), error = %e,
                "Could not read review_runtime_state.json; defaulting code review to on"
            );
            true
        }
    }
}

/// Persist the `code_review_enabled` flag atomically (temp file + rename in the
/// same directory).
pub fn save_enabled(path: &Path, enabled: bool) -> std::io::Result<()> {
    let state = ReviewRuntimeState {
        schema_version: 1,
        code_review_enabled: enabled,
    };
    let bytes = serde_json::to_vec_pretty(&state)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path() -> PathBuf {
        std::env::temp_dir().join(format!("brainrouter-review-state-{}.json", uuid::Uuid::new_v4()))
    }

    #[test]
    fn absent_file_defaults_to_enabled() {
        let path = tmp_path();
        assert!(load_enabled(&path), "missing file must default to on");
    }

    #[test]
    fn corrupt_file_defaults_to_enabled() {
        let path = tmp_path();
        std::fs::write(&path, b"{ this is not json").unwrap();
        assert!(load_enabled(&path), "corrupt file must default to on");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn round_trips_disabled_and_enabled() {
        let path = tmp_path();
        save_enabled(&path, false).unwrap();
        assert!(!load_enabled(&path));
        save_enabled(&path, true).unwrap();
        assert!(load_enabled(&path));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn unknown_extra_fields_do_not_break_load() {
        // Forward-compatibility: the hardening track adds fields to this file.
        let path = tmp_path();
        std::fs::write(
            &path,
            br#"{"schema_version":2,"code_review_enabled":false,"future_field":123}"#,
        )
        .unwrap();
        assert!(!load_enabled(&path));
        let _ = std::fs::remove_file(&path);
    }
}
