//! Persistence helpers for session IDs, channel models, and working directories.
//!
//! All three maps are stored as JSON files under `~/.local/share/omp-bridge/`.
//! Writes are best-effort: failures are logged but never fatal.

use std::{
    collections::HashMap,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
};
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

static NEXT_SAVE_SEQUENCE: AtomicU64 = AtomicU64::new(1);
static LATEST_SAVE_SEQUENCES: OnceLock<Mutex<HashMap<PathBuf, u64>>> = OnceLock::new();
static SAVE_GATES: OnceLock<Mutex<HashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>>> = OnceLock::new();

fn save_json_map(path: &Path, map: &HashMap<String, String>) {
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            warn!("Could not create dir {}: {}", parent.display(), e);
            return;
        }
    }
    let data = match serde_json::to_vec(map) {
        Ok(data) => data,
        Err(e) => {
            warn!("Could not serialize map for {}: {}", path.display(), e);
            return;
        }
    };
    let sequence = NEXT_SAVE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let tmp_path = path.with_extension(format!("json.tmp.{}.{}", std::process::id(), sequence));
    let result = (|| -> std::io::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)?;
        file.write_all(&data)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp_path, path)?;
        #[cfg(unix)]
        if let Some(parent) = path.parent() {
            std::fs::File::open(parent)?.sync_all()?;
        }
        Ok(())
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp_path);
        warn!("Could not atomically write {}: {}", path.display(), e);
    }
}

fn schedule_json_map_save(path: PathBuf, map: HashMap<String, String>) {
    let sequence = NEXT_SAVE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    LATEST_SAVE_SEQUENCES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap()
        .insert(path.clone(), sequence);
    let gate = {
        let mut gates = SAVE_GATES
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap();
        Arc::clone(
            gates
                .entry(path.clone())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
        )
    };

    tokio::spawn(async move {
        let _guard = gate.lock().await;
        let is_latest = LATEST_SAVE_SEQUENCES
            .get()
            .and_then(|latest| latest.lock().ok()?.get(&path).copied())
            == Some(sequence);
        if !is_latest {
            return;
        }

        let save_path = path.clone();
        if let Err(error) =
            tokio::task::spawn_blocking(move || save_json_map(&save_path, &map)).await
        {
            warn!("Persistence task failed for {}: {}", path.display(), error);
        }
    });
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
    schedule_json_map_save(sessions_path(transport), sessions.clone());
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
    schedule_json_map_save(channel_models_path(transport), models.clone());
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
    schedule_json_map_save(work_dirs_path(transport), dirs.clone());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn scheduled_saves_cannot_revert_newer_state() {
        let root = std::env::temp_dir().join(format!(
            "brainrouter-persist-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("state.json");

        for version in 0..100 {
            schedule_json_map_save(
                path.clone(),
                HashMap::from([("version".to_string(), version.to_string())]),
            );
        }

        let mut observed = None;
        for _ in 0..100 {
            if let Ok(content) = std::fs::read_to_string(&path) {
                let map: HashMap<String, String> = serde_json::from_str(&content).unwrap();
                if map.get("version").map(String::as_str) == Some("99") {
                    observed = Some(map);
                    break;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert_eq!(
            observed
                .as_ref()
                .and_then(|map| map.get("version"))
                .map(String::as_str),
            Some("99")
        );

        let _ = std::fs::remove_dir_all(root);
    }
}
