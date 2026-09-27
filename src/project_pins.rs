//! Per-project model pins (Phase 2).
//!
//! A project **inherits the global routing profile by default**; the user may
//! pin a model per role (main / reviewer / subagent) for a specific project,
//! keyed by the request `cwd` normalized to its **canonical git repository
//! root**. Routing resolution (see `router.rs`) consults this store at three
//! seams with role-specific precedence; the reviewer path applies its pin into
//! the per-run snapshot at review creation (`review/mod.rs`).
//!
//! Durability mirrors the existing local-state writers: `0600` + `create_new`
//! (from `routing_profile::atomic_write`) plus a `WRITE_LOCK`, orphaned-temp
//! sweep, and parent-directory fsync (from `review::runtime_state`). An absent,
//! unreadable, corrupt, or unsupported-schema file loads as **empty** so every
//! project inherits the global profile — the R9 fail-safe.
//!
//! A lock-free `has_pins` flag lets the routing hot path skip all filesystem
//! work (and the store mutex) when no pin exists.

use crate::routing_profile::{validate_model_id, ModelChoice};
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::Write,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

const SCHEMA_VERSION: u32 = 1;
const FILE_STEM: &str = ".project_pins-";

/// One project's optional per-role overrides. All-`None` means "no pin" and is
/// pruned from the store. `subagent` mirrors `RoutingProfile.subagent_model`
/// (a local subs-pool model key), not a full `ModelChoice`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectPin {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub main: Option<ModelChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reviewer: Option<ModelChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent: Option<String>,
}

impl ProjectPin {
    /// An all-`None` pin carries no override and must never be stored.
    pub fn is_empty(&self) -> bool {
        self.main.is_none() && self.reviewer.is_none() && self.subagent.is_none()
    }

    fn validate(&self) -> Result<()> {
        if let Some(main) = &self.main {
            main.validate().context("main")?;
        }
        if let Some(reviewer) = &self.reviewer {
            reviewer.validate().context("reviewer")?;
        }
        if let Some(subagent) = &self.subagent {
            validate_model_id(subagent).context("subagent")?;
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct OnDisk {
    #[serde(default = "default_schema_version")]
    schema_version: u32,
    #[serde(default)]
    pins: HashMap<String, ProjectPin>,
}

fn default_schema_version() -> u32 {
    SCHEMA_VERSION
}

/// Normalizer from a raw path to a canonical repository-root key. Injectable so
/// tests can count invocations (proving the R9 hot-path short-circuit) or fake
/// resolution without touching the filesystem.
pub type Normalizer = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// Serializes the read-free full-snapshot write so concurrent writers (e.g. two
/// store instances over the same path in tests) cannot race on the temp path or
/// lose an update. Mirrors `review::runtime_state::WRITE_LOCK`.
static WRITE_LOCK: Mutex<()> = Mutex::new(());

pub struct ProjectPinStore {
    state: Mutex<HashMap<String, ProjectPin>>,
    /// Lock-free gate read by the routing hot path — `false` skips the mutex
    /// and the normalizer entirely (R9). Published under the write lock.
    has_pins: AtomicBool,
    path: Option<PathBuf>,
    normalizer: Normalizer,
}

impl ProjectPinStore {
    /// In-memory store (no persistence), used by tests and profile-less fallbacks.
    pub fn memory() -> Self {
        Self::from_map(HashMap::new(), None, default_normalizer())
    }

    /// In-memory store seeded with pins, used by routing tests.
    pub fn memory_with_pins(pins: HashMap<String, ProjectPin>) -> Self {
        Self::from_map(pins, None, default_normalizer())
    }

    /// In-memory store with an injected normalizer, used to assert the hot-path
    /// short-circuit never invokes normalization for an empty store or for
    /// selectors that do not consult a pin.
    pub fn memory_with_normalizer(pins: HashMap<String, ProjectPin>, normalizer: Normalizer) -> Self {
        Self::from_map(pins, None, normalizer)
    }

    fn from_map(pins: HashMap<String, ProjectPin>, path: Option<PathBuf>, normalizer: Normalizer) -> Self {
        let has = !pins.is_empty();
        Self {
            state: Mutex::new(pins),
            has_pins: AtomicBool::new(has),
            path,
            normalizer,
        }
    }

    /// Load the persisted store. Absent / unreadable / corrupt / unsupported
    /// schema ⇒ empty (inherit-global fail-safe). Never panics, never blocks
    /// startup.
    pub fn load(path: PathBuf) -> Self {
        let pins = read_pins(&path);
        Self::from_map(pins, Some(path), default_normalizer())
    }

    /// Lock-free: has any project a pin? The routing hot path calls this first.
    pub fn has_pins(&self) -> bool {
        self.has_pins.load(Ordering::SeqCst)
    }

    /// Resolve the pin for a request `cwd`. Returns `None` — without touching
    /// the mutex or the normalizer — when the store is empty (R9). This is a
    /// synchronous method; the proxy caller wraps it in `spawn_blocking`, while
    /// reviewer creation calls it synchronously (both off the proxy hot path or
    /// gated on `has_pins`).
    pub fn resolve(&self, cwd: &str) -> Option<ProjectPin> {
        if !self.has_pins() {
            return None;
        }
        let key = (self.normalizer)(cwd)?;
        self.state.lock().unwrap().get(&key).cloned()
    }

    /// Normalize a raw path to its canonical repository-root key, **bypassing**
    /// the `has_pins` short-circuit so the first pin can be created in an empty
    /// store. Backs the `GET /api/project-pin?path=` resolver.
    pub fn resolve_key(&self, path: &str) -> Option<String> {
        (self.normalizer)(path)
    }

    /// The pin currently stored under an exact canonical key (or `None`).
    pub fn get(&self, key: &str) -> Option<ProjectPin> {
        self.state.lock().unwrap().get(key).cloned()
    }

    /// A copy of the whole pin table (for `GET /api/project-pin`).
    pub fn snapshot(&self) -> HashMap<String, ProjectPin> {
        self.state.lock().unwrap().clone()
    }

    /// Set (full replacement) the pin for the project containing `path`. An
    /// empty pin removes the entry. Returns the canonical key. Errors if `path`
    /// does not normalize to a repository root or the pin fails validation.
    pub fn set(&self, path: &str, pin: ProjectPin) -> Result<String> {
        let key = (self.normalizer)(path)
            .ok_or_else(|| anyhow!("path is not an absolute git repository directory"))?;
        pin.validate()?;
        self.mutate(|map| {
            if pin.is_empty() {
                map.remove(&key);
            } else {
                map.insert(key.clone(), pin.clone());
            }
        })?;
        Ok(key)
    }

    /// Remove a project's entry by its exact canonical key. Idempotent — a
    /// missing key succeeds — and never requires the path to still exist (the
    /// stale-entry cleanup path).
    pub fn remove(&self, key: &str) -> Result<()> {
        self.mutate(|map| {
            map.remove(key);
        })
    }

    /// Durable-before-visible mutation: clone the map under the lock, apply the
    /// change, prune all-`None` entries, persist the snapshot, then commit the
    /// clone and publish `has_pins`. A failed write leaves memory unchanged.
    fn mutate(&self, change: impl FnOnce(&mut HashMap<String, ProjectPin>)) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        let mut candidate = state.clone();
        change(&mut candidate);
        candidate.retain(|_, pin| !pin.is_empty());
        if let Some(path) = &self.path {
            persist(path, &candidate)?;
        }
        let has = !candidate.is_empty();
        *state = candidate;
        self.has_pins.store(has, Ordering::SeqCst);
        Ok(())
    }
}

fn default_normalizer() -> Normalizer {
    Arc::new(project_key)
}

/// Normalize a working directory to its canonical repository-root key, or
/// `None` (⇒ inherit the global profile). One function, used identically on the
/// read (routing) and write (endpoint) paths, so keys never skew.
///
/// Rules: empty or relative ⇒ `None` (a relative path must never resolve
/// against the daemon's own cwd); canonicalize (resolve symlinks + absolutize);
/// then walk up to the nearest ancestor containing a `.git` entry (a file — for
/// linked worktrees — or a directory) and return that directory. No `.git`
/// ancestor, or a non-UTF-8 canonical path, ⇒ `None`.
pub fn project_key(cwd: &str) -> Option<String> {
    if cwd.is_empty() {
        return None;
    }
    let raw = Path::new(cwd);
    if !raw.is_absolute() {
        return None;
    }
    let canonical = fs::canonicalize(raw).ok()?;
    let mut dir: &Path = canonical.as_path();
    loop {
        if dir.join(".git").exists() {
            return dir.to_str().map(str::to_owned);
        }
        dir = dir.parent()?;
    }
}

fn read_pins(path: &Path) -> HashMap<String, ProjectPin> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return HashMap::new(),
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "Could not read project_pins.json; inheriting global");
            return HashMap::new();
        }
    };
    let on_disk: OnDisk = match serde_json::from_slice(&bytes) {
        Ok(on_disk) => on_disk,
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "Ignoring corrupt project_pins.json; inheriting global");
            return HashMap::new();
        }
    };
    if on_disk.schema_version > SCHEMA_VERSION {
        tracing::warn!(
            path = %path.display(),
            schema_version = on_disk.schema_version,
            "Ignoring project_pins.json with an unsupported schema_version; inheriting global"
        );
        return HashMap::new();
    }
    // `ModelChoice` self-validates during deserialization, so main/reviewer are
    // already valid here; a bad `subagent` string (a bare `String`) is dropped,
    // and all-`None` entries are pruned.
    let mut out = HashMap::new();
    for (key, mut pin) in on_disk.pins {
        if let Some(subagent) = &pin.subagent {
            if validate_model_id(subagent).is_err() {
                tracing::warn!(project = %key, "Dropping invalid subagent pin from project_pins.json");
                pin.subagent = None;
            }
        }
        if !pin.is_empty() {
            out.insert(key, pin);
        }
    }
    out
}

fn persist(path: &Path, pins: &HashMap<String, ProjectPin>) -> Result<()> {
    let _guard = WRITE_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let on_disk = OnDisk {
        schema_version: SCHEMA_VERSION,
        pins: pins.clone(),
    };
    let bytes = serde_json::to_vec_pretty(&on_disk).context("serializing project pins")?;
    let parent = path
        .parent()
        .context("project pins path needs a parent directory")?;
    fs::create_dir_all(parent).context("creating project pins directory")?;
    // Sweep orphaned temp files from a prior crashed write (safe under the lock).
    if let Ok(entries) = fs::read_dir(parent) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(FILE_STEM) && name.ends_with(".tmp") {
                let _ = fs::remove_file(entry.path());
            }
        }
    }
    let temporary = parent.join(format!("{FILE_STEM}{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
        Ok::<_, std::io::Error>(())
    })();
    if result.is_err() {
        if let Err(error) = fs::remove_file(&temporary) {
            if error.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(%error, "Failed to remove temporary project pins file");
            }
        }
    }
    result.context("persisting project pins")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn tmp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("br-pins-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn git_repo() -> PathBuf {
        let dir = tmp_dir();
        fs::create_dir_all(dir.join(".git")).unwrap();
        dir
    }

    fn canon(path: &Path) -> String {
        fs::canonicalize(path).unwrap().to_str().unwrap().to_owned()
    }

    fn main_pin(id: &str) -> ProjectPin {
        ProjectPin {
            main: Some(ModelChoice::Local { model: Some(id.into()) }),
            ..Default::default()
        }
    }

    // ── project_key normalization matrix (R-Phase2 / I12) ────────────────────

    #[test]
    fn empty_and_relative_cwd_inherit_global() {
        assert_eq!(project_key(""), None);
        assert_eq!(project_key("relative/path"), None);
        assert_eq!(project_key("./x"), None);
    }

    #[test]
    fn repo_root_and_subdir_map_to_the_same_key() {
        let repo = git_repo();
        let sub = repo.join("src").join("deep");
        fs::create_dir_all(&sub).unwrap();
        let expected = canon(&repo);
        assert_eq!(project_key(repo.to_str().unwrap()).as_deref(), Some(expected.as_str()));
        assert_eq!(project_key(sub.to_str().unwrap()).as_deref(), Some(expected.as_str()));
        fs::remove_dir_all(&repo).ok();
    }

    #[test]
    fn symlinked_path_resolves_to_canonical_root() {
        let repo = git_repo();
        let link = std::env::temp_dir().join(format!("br-link-{}", uuid::Uuid::new_v4()));
        std::os::unix::fs::symlink(&repo, &link).unwrap();
        let expected = canon(&repo);
        assert_eq!(project_key(link.to_str().unwrap()).as_deref(), Some(expected.as_str()));
        fs::remove_file(&link).ok();
        fs::remove_dir_all(&repo).ok();
    }

    #[test]
    fn git_worktree_dot_git_file_is_a_root() {
        // A linked worktree has a `.git` *file*, not a directory.
        let worktree = tmp_dir();
        fs::write(worktree.join(".git"), b"gitdir: /somewhere/.git/worktrees/wt\n").unwrap();
        let expected = canon(&worktree);
        assert_eq!(project_key(worktree.to_str().unwrap()).as_deref(), Some(expected.as_str()));
        fs::remove_dir_all(&worktree).ok();
    }

    #[test]
    fn non_git_directory_inherits_global() {
        let dir = tmp_dir();
        assert_eq!(project_key(dir.to_str().unwrap()), None);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn vanished_path_inherits_global() {
        let dir = tmp_dir();
        let path = dir.to_str().unwrap().to_owned();
        fs::remove_dir_all(&dir).ok();
        assert_eq!(project_key(&path), None); // canonicalize fails ⇒ None
    }

    // ── R9: empty store does zero filesystem/normalizer work (I7) ────────────

    #[test]
    fn empty_store_never_invokes_the_normalizer() {
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        let normalizer: Normalizer = Arc::new(move |_p: &str| {
            seen.fetch_add(1, Ordering::SeqCst);
            Some("/whatever".to_string())
        });
        let store = ProjectPinStore::memory_with_normalizer(HashMap::new(), normalizer);
        assert!(!store.has_pins());
        assert_eq!(store.resolve("/any/path"), None);
        assert_eq!(store.resolve(""), None);
        assert_eq!(calls.load(Ordering::SeqCst), 0, "empty store must not normalize");
    }

    #[test]
    fn nonempty_store_resolves_via_the_normalizer() {
        let mut pins = HashMap::new();
        pins.insert("/canon/repo".to_string(), main_pin("m"));
        let normalizer: Normalizer = Arc::new(|_p: &str| Some("/canon/repo".to_string()));
        let store = ProjectPinStore::memory_with_normalizer(pins, normalizer);
        assert!(store.has_pins());
        assert_eq!(store.resolve("/x").unwrap().main, Some(ModelChoice::Local { model: Some("m".into()) }));
    }

    // ── set / remove / persistence ───────────────────────────────────────────

    #[test]
    fn set_normalizes_and_round_trips_atomically() {
        let repo = git_repo();
        let path = repo.join("state").join("project_pins.json");
        let store = ProjectPinStore::load(path.clone());
        assert!(!store.has_pins());
        let key = store.set(repo.to_str().unwrap(), main_pin("qwen")).unwrap();
        assert_eq!(key, canon(&repo));
        assert!(store.has_pins());
        // Reload from disk: the pin persists.
        let reloaded = ProjectPinStore::load(path);
        assert_eq!(reloaded.get(&key).unwrap().main, Some(ModelChoice::Local { model: Some("qwen".into()) }));
        fs::remove_dir_all(&repo).ok();
    }

    #[test]
    fn all_none_pin_removes_the_entry() {
        let repo = git_repo();
        let store = ProjectPinStore::load(repo.join("pins.json"));
        let key = store.set(repo.to_str().unwrap(), main_pin("m")).unwrap();
        assert!(store.get(&key).is_some());
        store.set(repo.to_str().unwrap(), ProjectPin::default()).unwrap();
        assert!(store.get(&key).is_none());
        assert!(!store.has_pins());
        fs::remove_dir_all(&repo).ok();
    }

    #[test]
    fn remove_by_key_is_idempotent_and_needs_no_path() {
        let store = ProjectPinStore::memory_with_pins({
            let mut m = HashMap::new();
            m.insert("/gone/repo".to_string(), main_pin("m"));
            m
        });
        store.remove("/gone/repo").unwrap();
        assert!(!store.has_pins());
        store.remove("/gone/repo").unwrap(); // idempotent
        store.remove("/never/existed").unwrap();
    }

    #[test]
    fn set_rejects_non_git_path() {
        let dir = tmp_dir();
        let store = ProjectPinStore::load(dir.join("pins.json"));
        assert!(store.set(dir.to_str().unwrap(), main_pin("m")).is_err());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn set_rejects_invalid_pin() {
        let repo = git_repo();
        let store = ProjectPinStore::load(repo.join("pins.json"));
        let bad = ProjectPin { subagent: Some("has space".into()), ..Default::default() };
        assert!(store.set(repo.to_str().unwrap(), bad).is_err());
        fs::remove_dir_all(&repo).ok();
    }

    // ── load fail-safes (I9) ─────────────────────────────────────────────────

    #[test]
    fn absent_and_corrupt_and_bad_schema_load_empty() {
        assert!(read_pins(Path::new("/no/such/project_pins.json")).is_empty());

        let dir = tmp_dir();
        let corrupt = dir.join("corrupt.json");
        fs::write(&corrupt, b"{ not json").unwrap();
        assert!(read_pins(&corrupt).is_empty());

        let future = dir.join("future.json");
        fs::write(&future, br#"{"schema_version":999,"pins":{"/r":{"main":{"backend":"local","model":"m"}}}}"#).unwrap();
        assert!(read_pins(&future).is_empty(), "unsupported schema ⇒ empty");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_drops_invalid_subagent_and_prunes_empty() {
        let dir = tmp_dir();
        let file = dir.join("pins.json");
        fs::write(
            &file,
            br#"{"schema_version":1,"pins":{
                "/r1":{"subagent":"bad id"},
                "/r2":{"main":{"backend":"local","model":"ok"}}
            }}"#,
        )
        .unwrap();
        let pins = read_pins(&file);
        assert!(!pins.contains_key("/r1"), "entry with only an invalid subagent is pruned");
        assert_eq!(pins.get("/r2").unwrap().main, Some(ModelChoice::Local { model: Some("ok".into()) }));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn concurrent_writes_on_shared_store_keep_every_key() {
        // Arc<same store>: the state mutex serializes; every distinct key must
        // survive (no lost update), and the file stays valid.
        let repo = git_repo();
        // Pre-create sibling repos as real git dirs so set() normalizes them.
        let keys: Vec<PathBuf> = (0..8)
            .map(|i| {
                let d = repo.join(format!("member{i}"));
                fs::create_dir_all(d.join(".git")).unwrap();
                d
            })
            .collect();
        let store = Arc::new(ProjectPinStore::load(repo.join("pins.json")));
        let mut handles = Vec::new();
        for (i, dir) in keys.iter().cloned().enumerate() {
            let store = Arc::clone(&store);
            handles.push(std::thread::spawn(move || {
                store.set(dir.to_str().unwrap(), main_pin(&format!("m{i}"))).unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let snapshot = store.snapshot();
        assert_eq!(snapshot.len(), 8, "all distinct-key writes retained");
        for dir in &keys {
            assert!(snapshot.contains_key(&canon(dir)));
        }
        fs::remove_dir_all(&repo).ok();
    }

    #[test]
    fn has_pins_tracks_first_add_and_last_remove() {
        let repo = git_repo();
        let store = ProjectPinStore::load(repo.join("pins.json"));
        assert!(!store.has_pins());
        let key = store.set(repo.to_str().unwrap(), main_pin("m")).unwrap();
        assert!(store.has_pins(), "first add publishes has_pins=true");
        store.remove(&key).unwrap();
        assert!(!store.has_pins(), "last remove publishes has_pins=false");
        fs::remove_dir_all(&repo).ok();
    }
}
