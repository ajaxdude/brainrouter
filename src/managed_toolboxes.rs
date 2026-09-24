use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

static WRITE_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnershipEntry {
    pub container_id: String,
    pub image: String,
    pub catalog_id: String,
    pub catalog_revision: String,
    pub created_at: String,
}

pub type OwnershipMap = BTreeMap<String, OwnershipEntry>;

pub fn load(path: &Path) -> OwnershipMap {
    match read_map(path) {
        Ok(map) => map,
        Err(error) => {
            tracing::warn!(
                path = %path.display(),
                error = %error,
                "Ignoring unreadable managed-toolboxes sidecar; treating all sidecar ownership as absent"
            );
            BTreeMap::new()
        }
    }
}

pub fn get(path: &Path, name: &str) -> Option<OwnershipEntry> {
    load(path).remove(name)
}

pub fn record(path: &Path, name: &str, entry: OwnershipEntry) -> std::io::Result<()> {
    let _guard = WRITE_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let mut map = read_map(path)?;
    map.insert(name.to_string(), entry);
    write_map(path, &map)
}

pub fn remove(path: &Path, name: &str) -> std::io::Result<()> {
    let _guard = WRITE_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let mut map = read_map(path)?;
    if map.remove(name).is_none() && path.exists() {
        return Ok(());
    }
    write_map(path, &map)
}

fn read_map(path: &Path) -> std::io::Result<OwnershipMap> {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice::<OwnershipMap>(&bytes)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(error) => Err(error),
    }
}

fn write_map(path: &Path, map: &OwnershipMap) -> std::io::Result<()> {
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "managed-toolboxes path needs a parent directory",
        )
    })?;
    std::fs::create_dir_all(parent)?;

    if let Ok(entries) = std::fs::read_dir(parent) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(".managed_toolboxes-") && name.ends_with(".tmp") {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }

    let tmp = parent.join(format!(".managed_toolboxes-{}.tmp", uuid::Uuid::new_v4()));
    let bytes = serde_json::to_vec_pretty(map)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    let result = (|| {
        #[cfg(unix)]
        let mut file = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&tmp)?
        };
        #[cfg(not(unix))]
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        }

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
    use std::sync::Arc;

    fn temp_dir() -> tempfile::TempDir {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test-tmp");
        std::fs::create_dir_all(&base).unwrap();
        tempfile::TempDir::new_in(base).unwrap()
    }

    fn entry(id: &str) -> OwnershipEntry {
        OwnershipEntry {
            container_id: id.to_string(),
            image: "docker.io/example/toolbox:latest".to_string(),
            catalog_id: "catalog-toolbox".to_string(),
            catalog_revision: "rev-1".to_string(),
            created_at: "2026-09-24T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn absent_file_loads_empty_and_get_is_none() {
        let dir = temp_dir();
        let path = dir.path().join("managed_toolboxes.json");
        assert!(load(&path).is_empty());
        assert_eq!(get(&path, "missing"), None);
    }

    #[test]
    fn record_get_and_remove_round_trip() {
        let dir = temp_dir();
        let path = dir.path().join("managed_toolboxes.json");
        record(&path, "box", entry("abc")).unwrap();
        assert_eq!(get(&path, "box").unwrap().container_id, "abc");
        remove(&path, "box").unwrap();
        assert_eq!(get(&path, "box"), None);
    }

    #[test]
    fn corrupt_file_is_fail_safe_for_reads_but_blocks_writes() {
        let dir = temp_dir();
        let path = dir.path().join("managed_toolboxes.json");
        std::fs::write(&path, b"{not-json").unwrap();
        assert!(load(&path).is_empty());
        assert_eq!(get(&path, "box"), None);
        assert_eq!(
            record(&path, "box", entry("abc")).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        assert_eq!(
            remove(&path, "box").unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn concurrent_writes_are_serialized_and_valid() {
        let dir = temp_dir();
        let path = Arc::new(dir.path().join("managed_toolboxes.json"));
        let mut handles = Vec::new();
        for i in 0..16 {
            let path = Arc::clone(&path);
            handles.push(std::thread::spawn(move || {
                record(&path, &format!("box-{i}"), entry(&format!("id-{i}"))).unwrap();
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        let map = load(&path);
        assert_eq!(map.len(), 16);
        assert_eq!(map["box-7"].container_id, "id-7");
    }

    #[cfg(unix)]
    #[test]
    fn persisted_file_permissions_are_0600() {
        use std::os::unix::fs::PermissionsExt;

        let dir = temp_dir();
        let path = dir.path().join("managed_toolboxes.json");
        record(&path, "box", entry("abc")).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }
}
