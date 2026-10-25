//! Minimal credentials store (v2 L2, design `docs/design/v2-streamlining.L2.md`).
//!
//! The only place a secret VALUE lives: `credentials.json` (0600, atomic
//! owner-only writes). Values are masked in every API response, log line, and
//! `Debug` rendering — the single leak surface is [`CredentialStore::resolve`],
//! which is consumed by the daemon itself and by `secrets.get` for plugins
//! that a credential explicitly grants.
//!
//! Kinds:
//! - `store` — the value lives in this file (the migrated Discord token).
//! - `env`   — the value names an environment variable (ops escape).
//! - `file`  — the value names a file whose contents are the secret (ops escape).
//!
//! Grants live on the credential (credential → plugin ids). Removing a
//! plugin drops the grants naming it.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

/// Rendered in place of a secret value in every non-`resolve` surface.
pub const MASK: &str = "***";

/// A secret value that is Debug- and Display-safe: neither rendering can
/// print the value. The plaintext is reachable only through [`Self::expose`].
#[derive(Clone, PartialEq, Eq)]
pub struct SecretString(String);

impl SecretString {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// The plaintext. Callers must not log the returned value.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for SecretString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SecretString({MASK})")
    }
}

impl std::fmt::Display for SecretString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{MASK}")
    }
}

impl Serialize for SecretString {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(MASK)
    }
}

impl<'de> Deserialize<'de> for SecretString {
    /// Reads the plaintext value back from the (0600) store file.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer).map(SecretString)
    }
}

/// Where a credential's value lives.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CredentialKind {
    /// Value stored directly in `credentials.json` (0600). The file IS the
    /// vault: it persists the plaintext (the 0600/atomic discipline is the
    /// protection); every *rendering* path masks (Debug/Display/`masked()`).
    Store {
        #[serde(
            serialize_with = "serialize_secret_plaintext",
            deserialize_with = "deserialize_secret"
        )]
        value: SecretString,
    },
    /// Name of an environment variable holding the value (ops escape).
    Env { var: String },
    /// Path of a file whose trimmed contents are the value (ops escape).
    File { path: PathBuf },
}

impl CredentialKind {
    /// Masked rendering for list/show APIs.
    pub fn masked(&self) -> serde_json::Value {
        match self {
            Self::Store { .. } => serde_json::json!({ "kind": "store", "value": MASK }),
            Self::Env { var } => serde_json::json!({ "kind": "env", "var": var }),
            Self::File { path } => serde_json::json!({
                "kind": "file",
                "path": path.display().to_string(),
            }),
        }
    }
}

fn serialize_secret_plaintext<S: serde::Serializer>(
    secret: &SecretString,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&secret.0)
}

fn deserialize_secret<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<SecretString, D::Error> {
    String::deserialize(deserializer).map(SecretString)
}

/// One credential: an id, where its value lives, and the plugin ids it grants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialRecord {
    pub id: String,
    #[serde(default)]
    pub description: String,
    #[serde(flatten)]
    pub kind: CredentialKind,
    /// Plugin ids allowed to `secrets.get` this credential.
    #[serde(default)]
    pub grants: Vec<String>,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
}

impl CredentialRecord {
    /// Masked rendering (never carries the value; `env`/`file` carry their refs).
    pub fn masked(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "description": self.description,
            "kind": self.kind.masked(),
            "grants": self.grants,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        })
    }

    pub fn grants_plugin(&self, plugin_id: &str) -> bool {
        self.grants.iter().any(|g| g == plugin_id)
    }
}

#[derive(Default, Serialize, Deserialize)]
struct StoreData {
    #[serde(default)]
    credentials: BTreeMap<String, CredentialRecord>,
}

/// The `credentials.json` store. All reads/writes go through the in-memory
/// copy guarded by a mutex; every mutation persists atomically with mode 0600.
pub struct CredentialStore {
    path: PathBuf,
    data: Mutex<StoreData>,
}

impl CredentialStore {
    /// `$XDG_STATE_HOME/brainrouter/credentials.json`
    /// (falls back to `~/.local/state`, matching the plugin state root).
    pub fn default_path() -> PathBuf {
        crate::paths::state_home()
            .join("brainrouter")
            .join("credentials.json")
    }

    /// Load the store. A missing file is an empty store (created on first
    /// write). A corrupt file is an error, not a panic — callers decide
    /// whether that is fatal (it is fatal-at-boot for credential consumers).
    pub fn load(path: PathBuf) -> Result<Self> {
        let data = match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str::<StoreData>(&text)
                .with_context(|| format!("credentials store {} is corrupt", path.display()))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => StoreData::default(),
            Err(e) => {
                return Err(e)
                    .with_context(|| format!("reading credentials store {}", path.display()))
            }
        };
        Ok(Self {
            path,
            data: Mutex::new(data),
        })
    }

    /// Boot-tolerant load (I3): a corrupt vault is renamed aside
    /// (`.corrupt-<UTC ts>`, kept for forensics — never overwritten) and the
    /// daemon boots with an empty store; every `resolve` then fails cleanly.
    /// The original file is preserved byte-for-byte under the new name.
    pub fn load_quarantining(path: PathBuf) -> (Result<Self>, Option<PathBuf>) {
        match Self::load(path.clone()) {
            Ok(store) => (Ok(store), None),
            Err(e) => {
                let ts = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
                let mut name = path.as_os_str().to_os_string();
                name.push(format!(".corrupt-{ts}"));
                let moved = PathBuf::from(name);
                match std::fs::rename(&path, &moved) {
                    Ok(()) => {
                        tracing::warn!(path = %path.display(), quarantined = %moved.display(),
                            error = %e,
                            "credentials store was corrupt; quarantined, booting with an empty store");
                        (Self::load(path), Some(moved))
                    }
                    Err(ren) => (
                        Err(e.context(format!(
                            "credentials store unreadable and quarantine rename failed: {ren}"
                        ))),
                        None,
                    ),
                }
            }
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn lock(&self) -> Result<MutexGuard<'_, StoreData>> {
        self.data
            .lock()
            .map_err(|_| anyhow::anyhow!("credentials store lock poisoned"))
    }

    /// Atomic owner-only write: temp file in the same directory, then rename.
    /// `NamedTempFile` creates the file with mode 0600 on Unix.
    fn save_locked(&self, data: &StoreData) -> Result<()> {
        let dir = self
            .path
            .parent()
            .context("credentials path has no parent directory")?;
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating credentials directory {}", dir.display()))?;
        let mut tmp = tempfile_in_dir(dir)?;
        let text = serde_json::to_string_pretty(data).context("serializing credentials store")?;
        tmp.as_file_mut()
            .write_all(text.as_bytes())
            .context("writing credentials temp file")?;
        tmp.as_file_mut().flush()?;
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o600))
            .context("chmod 0600 on credentials temp file")?;
        std::fs::rename(tmp.path(), &self.path)
            .with_context(|| format!("installing {}", self.path.display()))?;
        // The temp path is gone after the rename; `keep` prevents the
        // destructor from trying (and silently failing) to delete it.
        std::mem::forget(tmp);
        Ok(())
    }

    /// The single resolve implementation. Never panics; every failure (missing
    /// id, unreadable env/file) is a `Result::Err` the consumer treats as
    /// fatal-at-boot.
    pub fn resolve(&self, credential_id: &str) -> Result<SecretString> {
        let data = self.lock()?;
        let record = data
            .credentials
            .get(credential_id)
            .with_context(|| format!("credential '{credential_id}' not found"))?;
        match &record.kind {
            // The Debug rendering masks the value, so read the inner field
            // directly (a `%record` log line would mask, by design).
            CredentialKind::Store { value } => Ok(SecretString::new(value.0.clone())),
            CredentialKind::Env { var } => std::env::var(var).map(SecretString::new).with_context(
                || format!("credential '{credential_id}' references env var {var}, which is not set"),
            ),
            CredentialKind::File { path } => {
                let text = std::fs::read_to_string(path).with_context(|| {
                    format!("credential '{credential_id}' references file {}", path.display())
                })?;
                Ok(SecretString::new(text.trim_end_matches(['\n', '\r']).to_string()))
            }
        }
    }

    /// Insert or update a credential (atomic persist). Used by migration and
    /// future credential APIs.
    pub fn put(&self, mut record: CredentialRecord) -> Result<()> {
        if record.id.trim().is_empty() {
            bail!("credential id must not be empty");
        }
        let mut data = self.lock()?;
        let now = chrono::Utc::now().to_rfc3339();
        let created = data
            .credentials
            .get(&record.id)
            .map(|r| r.created_at.clone())
            .filter(|c| !c.is_empty())
            .unwrap_or_else(|| now.clone());
        record.created_at = created;
        record.updated_at = now;
        let id = record.id.clone();
        data.credentials.insert(id, record);
        self.save_locked(&data)
    }

    pub fn remove(&self, credential_id: &str) -> Result<bool> {
        let mut data = self.lock()?;
        let removed = data.credentials.remove(credential_id).is_some();
        if removed {
            self.save_locked(&data)?;
        }
        Ok(removed)
    }

    /// Replace the whole store (used by `migration restore`).
    pub fn replace_all(&self, records: Vec<CredentialRecord>) -> Result<()> {
        let mut data = self.lock()?;
        data.credentials.clear();
        for r in records {
            data.credentials.insert(r.id.clone(), r);
        }
        self.save_locked(&data)
    }

    /// Masked list for APIs/CLI — values never included.
    pub fn list_masked(&self) -> Result<Vec<serde_json::Value>> {
        let data = self.lock()?;
        Ok(data.credentials.values().map(|r| r.masked()).collect())
    }

    pub fn get_masked(&self, credential_id: &str) -> Result<serde_json::Value> {
        let data = self.lock()?;
        data.credentials
            .get(credential_id)
            .map(|r| r.masked())
            .context(format!("credential '{credential_id}' not found"))
    }

    pub fn exists(&self, credential_id: &str) -> Result<bool> {
        let data = self.lock()?;
        Ok(data.credentials.contains_key(credential_id))
    }

    /// Drop every grant naming `plugin_id` (called on plugin removal).
    /// Returns the credential ids whose grants changed.
    pub fn drop_grants_for_plugin(&self, plugin_id: &str) -> Result<Vec<String>> {
        let mut data = self.lock()?;
        let mut changed = Vec::new();
        for record in data.credentials.values_mut() {
            let before = record.grants.len();
            record.grants.retain(|g| g != plugin_id);
            if record.grants.len() != before {
                record.updated_at = chrono::Utc::now().to_rfc3339();
                changed.push(record.id.clone());
            }
        }
        if !changed.is_empty() {
            self.save_locked(&data)?;
        }
        Ok(changed)
    }

    /// Raw records for the migration/restore round-trip.
    pub fn snapshot_records(&self) -> Result<Vec<CredentialRecord>> {
        let data = self.lock()?;
        Ok(data.credentials.values().cloned().collect())
    }
}

/// Create an owner-only temp file in `dir` (NamedTempFile is 0600 on Unix).
pub(crate) fn tempfile_in_dir(dir: &Path) -> Result<tempfile::NamedTempFile> {
    tempfile::NamedTempFile::new_in(dir)
        .with_context(|| format!("creating temp file in {}", dir.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(dir: &Path) -> CredentialStore {
        CredentialStore::load(dir.join("credentials.json")).unwrap()
    }

    fn rec(id: &str, value: &str, grants: &[&str]) -> CredentialRecord {
        CredentialRecord {
            id: id.to_string(),
            description: String::new(),
            kind: CredentialKind::Store {
                value: SecretString::new(value.to_string()),
            },
            grants: grants.iter().map(|s| s.to_string()).collect(),
            created_at: String::new(),
            updated_at: String::new(),
        }
    }

    #[test]
    fn debug_and_display_mask_the_value() {
        let s = SecretString::new("s3cr3t-tok");
        let rendered = format!("{s:?} {s} {}", s.expose());
        assert!(rendered.starts_with("SecretString(***) ***"));
        assert!(rendered.contains("s3cr3t-tok"), "expose() is the leak surface");
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, "\"***\"");
    }

    #[test]
    fn store_roundtrip_resolve_and_masking() {
        let dir = tempfile::tempdir().unwrap();
        let st = store(dir.path());
        st.put(rec("discord_token", "abc123", &["bridge"])).unwrap();
        assert_eq!(st.resolve("discord_token").unwrap().expose(), "abc123");
        // masked list never contains the value
        let listed = serde_json::to_string(&st.list_masked().unwrap()).unwrap();
        assert!(!listed.contains("abc123"));
        assert!(listed.contains("***"));
        // reload from disk: file is 0600 and value survives
        drop(st);
        let st2 = store(dir.path());
        assert_eq!(st2.resolve("discord_token").unwrap().expose(), "abc123");
        let mode = std::fs::metadata(dir.path().join("credentials.json"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600);
        // missing id is an error, not a panic
        assert!(st2.resolve("nope").is_err());
    }

    #[test]
    fn env_and_file_kinds_resolve() {
        let dir = tempfile::tempdir().unwrap();
        let st = store(dir.path());
        let var = "BRL2_TEST_CRED_VAR";
        std::env::set_var(var, "from-env");
        st.put(CredentialRecord {
            id: "e".into(),
            description: String::new(),
            kind: CredentialKind::Env { var: var.into() },
            grants: vec![],
            created_at: String::new(),
            updated_at: String::new(),
        })
        .unwrap();
        assert_eq!(st.resolve("e").unwrap().expose(), "from-env");
        let f = dir.path().join("key.txt");
        std::fs::write(&f, "from-file\n").unwrap();
        st.put(CredentialRecord {
            id: "f".into(),
            description: String::new(),
            kind: CredentialKind::File { path: f.clone() },
            grants: vec![],
            created_at: String::new(),
            updated_at: String::new(),
        })
        .unwrap();
        assert_eq!(st.resolve("f").unwrap().expose(), "from-file");
        std::env::remove_var(var);
        assert!(st.resolve("e").is_err());
        std::fs::remove_file(&f).unwrap();
        assert!(st.resolve("f").is_err());
    }

    #[test]
    fn drop_grants_for_plugin() {
        let dir = tempfile::tempdir().unwrap();
        let st = store(dir.path());
        st.put(rec("a", "1", &["bridge", "other"])).unwrap();
        st.put(rec("b", "2", &["bridge"])).unwrap();
        let changed = st.drop_grants_for_plugin("bridge").unwrap();
        assert_eq!(changed, vec!["a".to_string(), "b".to_string()]);
        let masked = st.get_masked("a").unwrap();
        assert_eq!(masked["grants"], serde_json::json!(["other"]));
        assert!(st.get_masked("b").unwrap()["grants"].as_array().unwrap().is_empty());
    }
}
