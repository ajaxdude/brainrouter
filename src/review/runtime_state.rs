//! Persisted runtime state for the code-review master switch (design FR-A).
//!
//! A tiny JSON file kept next to `routing_state.json` recording whether
//! Brainrouter's code reviewer is enabled. The reviewer is **on by default**;
//! an absent, unreadable, or corrupt file falls back to on and never blocks
//! startup. Writes are atomic (temp file in the same directory + rename), the
//! same durability contract as the existing `routing_state.json` writer.
//!
//! Deliberately minimal: this file persists the `code_review_enabled` flag
//! (FR-A), the `pr_guidelines_enabled` flag (FR-D), and the
//! `hankndory_integration_enabled` runtime override (Phase-1b; `Option` so a
//! runtime toggle supersedes the YAML seed without regressing a YAML `true`).
//! Writes take the current values from the live sources and rewrite the full
//! snapshot under a lock, so a toggle of one flag never clobbers the others.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const FILE_NAME: &str = "review_runtime_state.json";

/// Serializes the whole read-free write (both flags are supplied by the caller
/// from the live atomics, so the writer never re-reads a possibly-corrupt file)
/// so concurrent toggles cannot race on the temp path or lose an update.
static WRITE_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReviewRuntimeState {
    #[serde(default = "default_schema_version")]
    schema_version: u32,
    #[serde(default = "default_true")]
    code_review_enabled: bool,
    /// FR-D: inject the PR-generation guideline into agent proxy requests.
    /// Opt-in, default off.
    #[serde(default)]
    pr_guidelines_enabled: bool,
    /// Phase-1b: runtime override for design-aware (HankNDory) review. `None`
    /// = never toggled at runtime ⇒ fall back to the YAML `review.hankndory_integration`
    /// seed; `Some(b)` = the dashboard toggle is authoritative.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hankndory_integration_enabled: Option<bool>,
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
            pr_guidelines_enabled: false,
            hankndory_integration_enabled: None,
        }
    }
}

/// Canonical path for the review runtime-state file (beside `routing_state.json`).
pub fn state_path() -> PathBuf {
    crate::config::default_config_path().with_file_name(FILE_NAME)
}

/// Read the persisted runtime flags. Absent, unreadable, or corrupt ⇒ defaults
/// `(code_review_enabled = true, pr_guidelines_enabled = false,
/// hankndory_integration_enabled = None)`. The third is an override: `None`
/// means "not toggled at runtime, use the YAML seed." Never panics.
pub fn load_state(path: &Path) -> (bool, bool, Option<bool>) {
    match std::fs::read(path) {
        Ok(bytes) => match serde_json::from_slice::<ReviewRuntimeState>(&bytes) {
            Ok(state) => (
                state.code_review_enabled,
                state.pr_guidelines_enabled,
                state.hankndory_integration_enabled,
            ),
            Err(e) => {
                tracing::warn!(
                    path = %path.display(), error = %e,
                    "Ignoring corrupt review_runtime_state.json; using defaults"
                );
                (true, false, None)
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (true, false, None),
        Err(e) => {
            tracing::warn!(
                path = %path.display(), error = %e,
                "Could not read review_runtime_state.json; using defaults"
            );
            (true, false, None)
        }
    }
}

/// Read the persisted `code_review_enabled` flag (default on).
pub fn load_enabled(path: &Path) -> bool {
    load_state(path).0
}

/// Read the persisted `pr_guidelines_enabled` flag (default off).
pub fn load_pr_guidelines(path: &Path) -> bool {
    load_state(path).1
}

/// Read the runtime HankNDory override (`None` ⇒ use the YAML seed).
pub fn load_hankndory_override(path: &Path) -> Option<bool> {
    load_state(path).2
}

/// Persist all runtime flags atomically. The caller passes the current values
/// (from the live atomics / ProfileStore), so this never re-reads the file and
/// can never reset a sibling flag. `hankndory_integration_enabled` is written as
/// `Some(..)` so a runtime toggle becomes authoritative over the YAML seed.
/// Holds `WRITE_LOCK` across serialize → unique temp write → `sync_all` →
/// rename → parent-dir fsync. Single-writer daemon assumed.
///
/// Note: each caller samples the sibling flags just before calling this, outside
/// `WRITE_LOCK`, so two *different* review toggles flipped within a sub-
/// millisecond window could persist a stale sibling value. This is benign on the
/// single-user dashboard (manual clicks can't race that tightly), live in-memory
/// state is always correct, design-aware review fails *safe* if stale (it
/// degrades to normal review), and the file self-heals on the next toggle.
pub fn save_state(
    path: &Path,
    code_review_enabled: bool,
    pr_guidelines_enabled: bool,
    hankndory_integration_enabled: bool,
) -> std::io::Result<()> {
    use std::io::Write;

    let _guard = WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let state = ReviewRuntimeState {
        schema_version: 1,
        code_review_enabled,
        pr_guidelines_enabled,
        hankndory_integration_enabled: Some(hankndory_integration_enabled),
    };
    let bytes = serde_json::to_vec_pretty(&state)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "runtime-state path needs a parent directory",
        )
    })?;
    std::fs::create_dir_all(parent)?;
    // Best-effort sweep of orphaned temp files from a prior crashed write. Safe
    // under WRITE_LOCK: no other writer holds an in-flight temp right now.
    if let Ok(entries) = std::fs::read_dir(parent) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(".review_runtime_state-") && name.ends_with(".tmp") {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
    let tmp = parent.join(format!(".review_runtime_state-{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        std::fs::rename(&tmp, path)?;
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
        Ok::<_, std::io::Error>(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path() -> PathBuf {
        std::env::temp_dir().join(format!("brainrouter-review-state-{}.json", uuid::Uuid::new_v4()))
    }

    #[test]
    fn absent_file_defaults() {
        let path = tmp_path();
        assert_eq!(load_state(&path), (true, false, None), "missing file defaults");
        assert!(load_enabled(&path));
        assert!(!load_pr_guidelines(&path));
        assert_eq!(load_hankndory_override(&path), None);
    }

    #[test]
    fn corrupt_file_defaults() {
        let path = tmp_path();
        std::fs::write(&path, b"{ this is not json").unwrap();
        assert_eq!(load_state(&path), (true, false, None), "corrupt file: defaults");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn round_trips_flags() {
        let path = tmp_path();
        save_state(&path, false, true, true).unwrap();
        assert_eq!(load_state(&path), (false, true, Some(true)));
        save_state(&path, true, false, false).unwrap();
        assert_eq!(load_state(&path), (true, false, Some(false)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn toggling_one_flag_preserves_the_others() {
        // The endpoints pass all live values, so a full-snapshot write can never
        // clobber a sibling flag.
        let path = tmp_path();
        save_state(&path, true, true, true).unwrap();
        // Flip only code-review off, carrying pr + hankndory current values.
        let (_code, pr, hank) = load_state(&path);
        save_state(&path, false, pr, hank.unwrap_or(false)).unwrap();
        assert_eq!(load_state(&path), (false, true, Some(true)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn hankndory_override_is_none_until_written_then_authoritative() {
        let path = tmp_path();
        // Absent field ⇒ None (caller falls back to the YAML seed).
        std::fs::write(
            &path,
            br#"{"schema_version":1,"code_review_enabled":true,"pr_guidelines_enabled":false}"#,
        )
        .unwrap();
        assert_eq!(load_hankndory_override(&path), None);
        // Once toggled, it is Some and authoritative.
        save_state(&path, true, false, true).unwrap();
        assert_eq!(load_hankndory_override(&path), Some(true));
        save_state(&path, true, false, false).unwrap();
        assert_eq!(load_hankndory_override(&path), Some(false));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn unknown_extra_fields_do_not_break_load() {
        let path = tmp_path();
        std::fs::write(
            &path,
            br#"{"schema_version":2,"code_review_enabled":false,"pr_guidelines_enabled":true,"hankndory_integration_enabled":true,"future_field":123}"#,
        )
        .unwrap();
        assert_eq!(load_state(&path), (false, true, Some(true)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn save_state_sweeps_orphaned_temp_files() {
        // Own subdirectory so the sweep can't touch sibling tests' temp dir.
        let dir = std::env::temp_dir().join(format!("br-rtstate-sweep-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(FILE_NAME);
        let orphan = dir.join(".review_runtime_state-deadbeef.tmp");
        std::fs::write(&orphan, b"stale").unwrap();
        save_state(&path, true, true, false).unwrap();
        assert!(!orphan.exists(), "a prior crashed write's temp must be swept");
        assert_eq!(load_state(&path), (true, true, Some(false)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_writes_do_not_corrupt_the_file() {
        let path = std::sync::Arc::new(tmp_path());
        let mut handles = Vec::new();
        for i in 0..16 {
            let p = std::sync::Arc::clone(&path);
            handles.push(std::thread::spawn(move || {
                // Alternate values; the lock serializes and each write is a full
                // valid snapshot (never a torn file).
                save_state(&p, i % 2 == 0, i % 3 == 0, i % 5 == 0).unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        // Whatever landed last, the file parses to a valid struct — not a
        // torn/corrupt read that would fall back to defaults on a valid file.
        let bytes = std::fs::read(path.as_ref()).unwrap();
        assert!(serde_json::from_slice::<ReviewRuntimeState>(&bytes).is_ok());
        let _ = std::fs::remove_file(path.as_ref());
    }
}
