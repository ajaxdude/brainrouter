//! Read-only (Phase 1) integration with ai-toolbox-cockpit's shared
//! `~/.config/ai-toolbox-cockpit/config.json`, plus an explicit,
//! user-triggered single-write "apply" action for the specific keys
//! cockpit itself owns (`active_platform`,
//! `backends.<id>.default_toolboxes.<platform_id>`).
//!
//! See `docs/design/ai-toolbox-cockpit-integration.md` §4 for the full
//! design rationale, in particular why **automatic, continuous
//! bidirectional merge is out of scope for this rollout**: cockpit's own
//! `save_settings()` is atomic-write-only (temp file + `rename()`), not
//! atomic-read-modify-write, so two processes editing the same file at
//! any time with no lock/version signal is a real race. Phase 1 narrows
//! that exposure to one explicit, reviewable write per user action.
//!
//! Upstream reference (read-only, not vendored): `ai_toolbox_cockpit/settings.py`.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Resolves the effective config file path exactly as cockpit's own
/// `settings.py::config_path()` does: `$XDG_CONFIG_HOME` (default
/// `$HOME/.config`) / `ai-toolbox-cockpit` / `config.json`. `$HOME` falls
/// back to `/root` if unset, matching brainrouter's existing HOME-resolution
/// convention (`src/daemon.rs:412`, `src/config.rs::default_config_path()`)
/// rather than inventing a new one.
pub fn config_path() -> PathBuf {
    let root = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
            PathBuf::from(home).join(".config")
        });
    root.join("ai-toolbox-cockpit").join("config.json")
}

/// Everything brainrouter reads from/writes to cockpit's config, kept
/// close to the raw JSON shape rather than fully modeled — cockpit owns
/// this schema, not brainrouter. `extra` round-trips every key brainrouter
/// doesn't explicitly understand so an `apply_*` write never silently
/// drops state a human (or cockpit itself) put there.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CockpitConfig {
    #[serde(default)]
    pub active_platform: Option<String>,
    #[serde(default)]
    pub backends: BTreeMap<String, BackendSettings>,
    /// A top-level (not per-backend) saved Hugging Face token, mirroring
    /// upstream's `huggingface.py::get_hf_token()` (`get_setting`/
    /// `set_setting("hf_token", ...)`, verified live against upstream source
    /// during PR6's design pass — see design doc §10). Read-only for now:
    /// brainrouter never *writes* this field (no `apply_*` function touches
    /// it) — see §10's flagged open item on whether a save-token write path
    /// should ever be added.
    #[serde(default)]
    pub hf_token: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BackendSettings {
    #[serde(default)]
    pub default_toolboxes: BTreeMap<String, String>,
    /// Per-backend model download directory, mirroring upstream's
    /// `<backend>/model_manager.py::get_models_dir()` (`backends.<id>
    /// .models_dir` in config.json; falls back to the catalog's own
    /// `backends.<id>.storage.default` when absent — see design doc §10 and
    /// `src/model_downloads.rs`). Read-only: brainrouter does not currently
    /// expose a way to change this value (no `apply_*` writer), only to
    /// read whatever cockpit itself already saved.
    #[serde(default)]
    pub models_dir: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl CockpitConfig {
    /// Mirrors `settings.py::load_default_toolbox()` (without its
    /// `fallback` parameter — callers combine this with the catalog's own
    /// `Platform.defaults` fallback themselves, since that's a brainrouter-
    /// side concept, not cockpit's).
    pub fn default_toolbox(&self, backend_id: &str, platform_id: &str) -> Option<&str> {
        self.backends
            .get(backend_id)?
            .default_toolboxes
            .get(platform_id)
            .map(String::as_str)
    }

    /// The cockpit-saved model directory override for `backend_id`, if any
    /// (§10). `None` means "no override" — callers fall back to the
    /// catalog's own `storage.default` for that backend.
    pub fn models_dir(&self, backend_id: &str) -> Option<&str> {
        self.backends.get(backend_id)?.models_dir.as_deref()
    }
}

/// A snapshot of cockpit-config state suitable for direct API/dashboard
/// display. Always includes the resolved path (and owning UID, when
/// determinable) so a HOME/config-path mismatch between brainrouter and an
/// interactively-run cockpit is visible rather than silent (§4's
/// path-mismatch mitigation (a)).
#[derive(Debug, Clone, Serialize)]
pub struct CockpitConfigStatus {
    pub path: String,
    pub exists: bool,
    /// Owning UID of the config file, if the platform exposes one and the
    /// file exists. `None` means "not determined", not "no owner".
    pub owner_uid: Option<u32>,
    pub config: Option<CockpitConfig>,
    /// Set only when the file exists but failed to parse/read — distinct
    /// from "absent", which is a normal, expected, never-an-error state
    /// (§4, resolved Open question 2).
    pub error: Option<String>,
}

/// Loads cockpit's config.json if present. Absence is a normal, first-class
/// state (brainrouter never auto-creates the file or its parent directory)
/// — reflected as `exists: false`, `error: None`, never as a failure.
/// A read/parse failure on an *existing* file is reported via `error`
/// instead of silently falling back to `{}` the way cockpit's own
/// `load_settings()` does — brainrouter has no reason to also replicate
/// cockpit's legacy `.llama-cockpit.conf`/`.ds4-cockpit.conf` migration
/// fallback; that is cockpit's own concern, not brainrouter's.
pub fn load() -> CockpitConfigStatus {
    let path = config_path();
    let path_str = path.display().to_string();
    match fs::read_to_string(&path) {
        Ok(raw) => {
            let owner_uid = fs::metadata(&path).ok().map(|m| {
                use std::os::unix::fs::MetadataExt;
                m.uid()
            });
            match serde_json::from_str::<CockpitConfig>(&raw) {
                Ok(config) => CockpitConfigStatus {
                    path: path_str,
                    exists: true,
                    owner_uid,
                    config: Some(config),
                    error: None,
                },
                Err(e) => CockpitConfigStatus {
                    path: path_str,
                    exists: true,
                    owner_uid,
                    config: None,
                    error: Some(format!("failed to parse config.json: {e}")),
                },
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => CockpitConfigStatus {
            path: path_str,
            exists: false,
            owner_uid: None,
            config: None,
            error: None,
        },
        Err(e) => CockpitConfigStatus {
            path: path_str,
            exists: true,
            owner_uid: None,
            config: None,
            error: Some(format!("failed to read config.json: {e}")),
        },
    }
}

/// Why an explicit "apply" write did not happen.
#[derive(Debug)]
pub enum ApplyError {
    /// The `ai-toolbox-cockpit` config directory does not exist on this
    /// host. Per §4's resolved Open question 2, brainrouter never creates
    /// it — the write action is simply unavailable until cockpit itself
    /// initializes the directory by running at least once.
    NotAvailable,
    /// The file exists but couldn't be read/parsed before the write.
    Read(String),
    /// The read succeeded (or the file was absent) but the atomic write
    /// itself failed.
    Write(std::io::Error),
}

impl std::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAvailable => write!(
                f,
                "ai-toolbox-cockpit config directory not found at {}; cockpit has not been run on this host yet",
                config_path().display()
            ),
            Self::Read(e) => write!(f, "failed to read existing cockpit config.json before applying: {e}"),
            Self::Write(e) => write!(f, "failed to write cockpit config.json: {e}"),
        }
    }
}

impl std::error::Error for ApplyError {}

/// Applies a single default-toolbox choice for `(backend_id, platform_id)`:
/// reload-verify-write against the *current* on-disk state (re-read
/// immediately before mutating, rather than reusing a possibly-stale
/// earlier snapshot), touching only
/// `backends.<backend_id>.default_toolboxes.<platform_id>` — every other
/// key, including ones brainrouter doesn't understand, round-trips
/// untouched. This narrows, but does not eliminate, the race against a
/// concurrent cockpit write (§4) — true bidirectional live-merge is out of
/// scope for this rollout.
pub fn apply_default_toolbox(
    backend_id: &str,
    platform_id: &str,
    toolbox_id: &str,
) -> Result<(), ApplyError> {
    apply(|cfg| {
        let backend = cfg.backends.entry(backend_id.to_string()).or_default();
        backend
            .default_toolboxes
            .insert(platform_id.to_string(), toolbox_id.to_string());
    })
}

/// Applies a new `active_platform` value, same reload-verify-write
/// contract as [`apply_default_toolbox`].
pub fn apply_active_platform(platform_id: &str) -> Result<(), ApplyError> {
    apply(|cfg| {
        cfg.active_platform = Some(platform_id.to_string());
    })
}

/// Merges arbitrary `backends.<backend_id>` key/value pairs (PR8+), same
/// reload-verify-write contract as [`apply_default_toolbox`]. Every value
/// lands in [`BackendSettings::extra`] via serde's `#[serde(flatten)]` —
/// there is no per-backend Rust-typed field for these (e.g. halogen's
/// `host`/`port`/`context`/`pool`/`slots`/`prompt_cache`/`bundle_id`,
/// mirroring upstream's own `save_backend_settings(backend_id, {...})`
/// call in `*/server.py::_start_confirmed()`), unlike `default_toolboxes`/
/// `models_dir` which are cockpit-schema-stable enough to model explicitly.
/// Callers should pass [`Value::String`] for every value (even numeric
/// ones) to match upstream's own storage shape exactly — cockpit's own
/// `Input` widgets save `.value` as a plain string, never a JSON number.
pub fn apply_backend_setting_values(
    backend_id: &str,
    updates: Vec<(&str, Value)>,
) -> Result<(), ApplyError> {
    apply(|cfg| {
        let backend = cfg.backends.entry(backend_id.to_string()).or_default();
        for (key, value) in updates {
            backend.extra.insert(key.to_string(), value);
        }
    })
}

fn apply(mutate: impl FnOnce(&mut CockpitConfig)) -> Result<(), ApplyError> {
    let path = config_path();
    let dir = path
        .parent()
        .expect("config_path() always has a parent (.../ai-toolbox-cockpit/config.json)");
    if !dir.exists() {
        return Err(ApplyError::NotAvailable);
    }

    let mut cfg = match fs::read_to_string(&path) {
        Ok(raw) => serde_json::from_str::<CockpitConfig>(&raw)
            .map_err(|e| ApplyError::Read(format!("failed to parse config.json: {e}")))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => CockpitConfig::default(),
        Err(e) => return Err(ApplyError::Read(e.to_string())),
    };
    mutate(&mut cfg);
    write_atomic(&path, &cfg).map_err(ApplyError::Write)
}

/// Same atomicity contract as cockpit's own `save_settings()` (temp file in
/// the same directory, `chmod 0600`, then `rename()` — a rename onto an
/// existing path is atomic on POSIX filesystems), using brainrouter's own
/// established temp-file convention (UUID-suffixed name, matching
/// `src/routing_profile.rs`/`src/observability.rs`) rather than mimicking
/// cockpit's literal `config.tmp` name — only the final atomic `rename()`
/// onto the shared path matters for safety, not the intermediate name.
fn write_atomic(path: &Path, cfg: &CockpitConfig) -> std::io::Result<()> {
    let parent = path
        .parent()
        .expect("config_path() always has a parent (.../ai-toolbox-cockpit/config.json)");
    let temporary = parent.join(format!(".ai-toolbox-cockpit-config-{}.tmp", uuid::Uuid::new_v4()));
    let mut bytes = serde_json::to_vec_pretty(cfg)?;
    bytes.push(b'\n'); // mirrors cockpit's own trailing `target.write("\n")`.

    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        Ok::<_, std::io::Error>(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_path_ends_with_the_cockpit_relative_path() {
        // Mirrors src/config.rs's default_config_path test: mutating
        // XDG_CONFIG_HOME/HOME in-process is unsafe under parallel test
        // execution, so this asserts the path *shape*, not a specific
        // absolute value.
        let p = config_path();
        assert!(
            p.ends_with("ai-toolbox-cockpit/config.json"),
            "unexpected path: {}",
            p.display()
        );
    }

    #[test]
    fn cockpit_config_default_toolbox_reads_nested_path() {
        let json = r#"{
            "active_platform": "strix-halo",
            "backends": {
                "llama_cpp": {
                    "default_toolboxes": { "strix-halo": "strix-halo-llama-vulkan-radv" },
                    "some_other_setting": 42
                }
            },
            "a_key_brainrouter_does_not_model": true
        }"#;
        let cfg: CockpitConfig = serde_json::from_str(json).expect("parses");
        assert_eq!(cfg.active_platform.as_deref(), Some("strix-halo"));
        assert_eq!(
            cfg.default_toolbox("llama_cpp", "strix-halo"),
            Some("strix-halo-llama-vulkan-radv")
        );
        assert_eq!(cfg.default_toolbox("llama_cpp", "r9700"), None);
        assert_eq!(cfg.default_toolbox("ds4", "strix-halo"), None);
        // Unknown top-level and per-backend keys must round-trip, not be dropped.
        assert_eq!(cfg.extra.get("a_key_brainrouter_does_not_model"), Some(&Value::Bool(true)));
        assert_eq!(
            cfg.backends["llama_cpp"].extra.get("some_other_setting"),
            Some(&serde_json::json!(42))
        );
    }

    #[test]
    fn load_reports_absence_as_normal_not_an_error() {
        // config_path() in this sandboxed test environment resolves under a
        // HOME that (almost certainly) has no ai-toolbox-cockpit directory.
        // This directly exercises the "absence is never an error" contract
        // (§4, resolved Open question 2) against the real, unmodified
        // config_path() rather than an injected fake path.
        let status = load();
        if !status.exists {
            assert!(status.error.is_none());
            assert!(status.config.is_none());
        }
    }

    #[test]
    fn apply_default_toolbox_preserves_unrelated_keys_round_trip() {
        let dir = std::env::temp_dir().join(format!("brainrouter-cockpit-config-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("config.json");
        fs::write(
            &path,
            r#"{"active_platform":"strix-halo","backends":{"ds4":{"default_toolboxes":{"strix-halo":"strix-halo-ds4-rocm-10-0"}}},"unrelated_top_level_key":"keep-me"}"#,
        )
        .expect("seed file");

        // Exercise the mutate-and-write-atomic core directly against a
        // temp path (config_path() itself is not overridable without
        // mutating process-global env vars — see config_path_ends_with_
        // the_cockpit_relative_path's comment on why that's avoided here).
        let mut cfg: CockpitConfig =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        cfg.backends
            .entry("llama_cpp".to_string())
            .or_default()
            .default_toolboxes
            .insert("strix-halo".to_string(), "strix-halo-llama-vulkan-radv".to_string());
        write_atomic(&path, &cfg).expect("atomic write");

        let written: CockpitConfig = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(
            written.default_toolbox("llama_cpp", "strix-halo"),
            Some("strix-halo-llama-vulkan-radv")
        );
        // Pre-existing ds4 default and unrelated top-level key survive untouched.
        assert_eq!(
            written.default_toolbox("ds4", "strix-halo"),
            Some("strix-halo-ds4-rocm-10-0")
        );
        assert_eq!(
            written.extra.get("unrelated_top_level_key"),
            Some(&Value::String("keep-me".to_string()))
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_atomic_sets_owner_only_permissions() {
        let dir = std::env::temp_dir().join(format!("brainrouter-cockpit-config-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("config.json");
        write_atomic(&path, &CockpitConfig::default()).expect("atomic write");
        let mode = fs::metadata(&path).unwrap().permissions();
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(mode.mode() & 0o777, 0o600);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn hf_token_and_models_dir_round_trip_and_fall_back_to_none() {
        let json = r#"{
            "hf_token": "hf_abc123",
            "backends": {
                "ds4": { "models_dir": "/data/ds4-models" },
                "halogen": {}
            }
        }"#;
        let cfg: CockpitConfig = serde_json::from_str(json).expect("parses");
        assert_eq!(cfg.hf_token.as_deref(), Some("hf_abc123"));
        assert_eq!(cfg.models_dir("ds4"), Some("/data/ds4-models"));
        // Present backend with no models_dir set falls back to None (caller
        // then falls back to the catalog's own storage.default, per §10).
        assert_eq!(cfg.models_dir("halogen"), None);
        // Absent backend entirely also falls back to None, not a panic.
        assert_eq!(cfg.models_dir("vllm"), None);

        // Round-trip through write_atomic + reparse must preserve both new
        // fields exactly, alongside the existing extra/default_toolboxes
        // behavior already covered above.
        let dir = std::env::temp_dir().join(format!("brainrouter-cockpit-config-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("config.json");
        write_atomic(&path, &cfg).expect("atomic write");
        let written: CockpitConfig = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(written.hf_token.as_deref(), Some("hf_abc123"));
        assert_eq!(written.models_dir("ds4"), Some("/data/ds4-models"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_hf_token_and_models_dir_are_none_not_an_error() {
        let cfg: CockpitConfig = serde_json::from_str("{}").expect("parses");
        assert_eq!(cfg.hf_token, None);
        assert_eq!(cfg.models_dir("llama_cpp"), None);
    }

    #[test]
    fn backend_setting_values_merge_into_extra_and_round_trip() {
        // Same "exercise the mutate-and-write-atomic core directly" pattern
        // as apply_default_toolbox_preserves_unrelated_keys_round_trip
        // (apply()'s own config_path() lookup isn't overridable in-process)
        // — this asserts apply_backend_setting_values' actual merge
        // semantics: unknown key/value pairs land in BackendSettings::extra
        // (there's no typed `host`/`port`/etc. field) and never clobber
        // sibling backends or already-set typed fields on the same backend.
        let dir = std::env::temp_dir().join(format!("brainrouter-cockpit-config-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("config.json");
        fs::write(
            &path,
            r#"{"backends":{"halogen":{"models_dir":"/data/halogen-models"},"ds4":{"default_toolboxes":{"strix-halo":"strix-halo-ds4-rocm-10-0"}}}}"#,
        )
        .expect("seed file");

        let mut cfg: CockpitConfig = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let backend = cfg.backends.entry("halogen".to_string()).or_default();
        for (key, value) in [
            ("host", Value::String("127.0.0.1".to_string())),
            ("port", Value::String("8731".to_string())),
            ("bundle_id", Value::String("qwen38-flash-next-w4b-quality".to_string())),
        ] {
            backend.extra.insert(key.to_string(), value);
        }
        write_atomic(&path, &cfg).expect("atomic write");

        let written: CockpitConfig = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let halogen = &written.backends["halogen"];
        assert_eq!(halogen.extra.get("host"), Some(&Value::String("127.0.0.1".to_string())));
        assert_eq!(halogen.extra.get("port"), Some(&Value::String("8731".to_string())));
        assert_eq!(
            halogen.extra.get("bundle_id"),
            Some(&Value::String("qwen38-flash-next-w4b-quality".to_string()))
        );
        // Pre-existing typed field on the same backend and the sibling
        // backend's own settings both survive untouched.
        assert_eq!(halogen.models_dir.as_deref(), Some("/data/halogen-models"));
        assert_eq!(
            written.default_toolbox("ds4", "strix-halo"),
            Some("strix-halo-ds4-rocm-10-0")
        );

        fs::remove_dir_all(&dir).ok();
    }
}
