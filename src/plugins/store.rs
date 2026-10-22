//! Per-plugin state files and paths.
//!
//! Plugin state dir: `$XDG_STATE_HOME/brainrouter/plugins/<id>/` (0700;
//! holds the copied manifest/binary, token, config, kv, logs, persistence).
//! Sockets: `$XDG_RUNTIME_DIR/brainrouter/plugins/<id>.sock` (dir 0700,
//! socket 0600).

use anyhow::{Context, Result};
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

pub fn plugins_state_root(state_root: &Path) -> PathBuf {
    state_root.join("plugins")
}

pub fn plugin_state_dir(state_root: &Path, id: &str) -> PathBuf {
    plugins_state_root(state_root).join(id)
}

pub fn plugins_runtime_root(runtime_root: &Path) -> PathBuf {
    runtime_root.join("plugins")
}

pub fn plugin_socket_path(runtime_root: &Path, id: &str) -> PathBuf {
    plugins_runtime_root(runtime_root).join(format!("{id}.sock"))
}

/// Create the plugins runtime directory (0700).
pub fn ensure_plugins_runtime_dir(runtime_root: &Path) -> Result<PathBuf> {
    let dir = plugins_runtime_root(runtime_root);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating plugin runtime dir {}", dir.display()))?;
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))
        .with_context(|| format!("chmod 0700 {}", dir.display()))?;
    Ok(dir)
}

/// Create/ensure a plugin's state directory (0700).
pub fn ensure_plugin_state_dir(state_root: &Path, id: &str) -> Result<PathBuf> {
    let root = plugins_state_root(state_root);
    std::fs::create_dir_all(&root)?;
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).ok();
    let dir = root.join(id);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating plugin state dir {}", dir.display()))?;
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))
        .with_context(|| format!("chmod 0700 {}", dir.display()))?;
    Ok(dir)
}

/// Atomic 0600 write (temp file + rename), used for token/config/kv/status.
pub fn atomic_write_0600(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().context("path has no parent dir")?;
    std::fs::create_dir_all(dir)?;
    let mut tmp = crate::credentials::tempfile_in_dir(dir)?;
    tmp.as_file_mut().write_all(bytes)?;
    tmp.as_file_mut().flush()?;
    std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o600))
        .context("chmod 0600 on plugin temp file")?;
    std::fs::rename(tmp.path(), path)
        .with_context(|| format!("installing {}", path.display()))?;
    std::mem::forget(tmp);
    Ok(())
}

pub fn write_json_0600(path: &Path, value: &serde_json::Value) -> Result<()> {
    let text = serde_json::to_string_pretty(value).context("serializing plugin state")?;
    atomic_write_0600(path, text.as_bytes())
}

/// Read a JSON file; missing file = `None`.
pub fn read_json(path: &Path) -> Result<Option<serde_json::Value>> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(
            serde_json::from_str(&text)
                .with_context(|| format!("parsing {}", path.display()))?,
        )),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

/// Mint a 256-bit hex plugin token.
pub fn mint_token() -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(uuid::Uuid::new_v4().as_bytes());
    h.update(uuid::Uuid::new_v4().as_bytes());
    h.update(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos().to_le_bytes())
            .unwrap_or_default(),
    );
    h.update(std::process::id().to_le_bytes());
    let digest = h.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest.iter() {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Constant-time string equality (token checks must not leak via timing).
pub fn ct_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_mint_and_ct_eq() {
        let a = mint_token();
        let b = mint_token();
        assert_eq!(a.len(), 64);
        assert_ne!(a, b);
        assert!(ct_eq(&a, &a));
        assert!(!ct_eq(&a, &b));
        assert!(!ct_eq(&a, &a[..63]));
    }

    #[test]
    fn atomic_write_is_0600_and_content_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("sub").join("token");
        atomic_write_0600(&p, b"hello").unwrap();
        assert_eq!(
            std::fs::metadata(&p).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "hello");
        atomic_write_0600(&p, b"world").unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "world");
    }

    #[test]
    fn plugin_dirs_are_0700() {
        let dir = tempfile::tempdir().unwrap();
        let state = dir.path().join("state");
        let st = ensure_plugin_state_dir(&state, "bridge").unwrap();
        assert_eq!(std::fs::metadata(&st).unwrap().permissions().mode() & 0o777, 0o700);
        let rt = ensure_plugins_runtime_dir(&dir.path().join("run")).unwrap();
        assert_eq!(std::fs::metadata(&rt).unwrap().permissions().mode() & 0o700 & 0o777, 0o700);
        assert_eq!(plugin_socket_path(&dir.path().join("run"), "bridge"), rt.join("bridge.sock"));
    }
}
