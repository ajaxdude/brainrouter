//! Role cards (v2 L3): the single human-facing model config.
//!
//! `role_cards.json` (0600, atomic — the durability machinery relocated from
//! `src/project_pins.rs`) replaces the v1 `routing_profile` store and the
//! per-project pins store. Records:
//!
//! - `assignments`: `{scope: "global" | <canonical-git-root>, role, assignment}`
//!   where `assignment` is a selector string in the SAME vocabulary the proxy
//!   surface keeps through L7: `auto | local | cloud | cloud/<id> | <bare-key>`.
//! - `policy`: `{scope → {privacy, quality}}` guardrail chips.
//! - `review_extras`: the review card's surviving non-model config
//!   (`max_iterations`, `hankndory_integration`, `design_doc_path`).
//!
//! Lookup precedence: project → global → default (`auto` for every role).
//!
//! The shared selector types (`ModelChoice`, `validate_model_id`,
//! `validate_iterations`) relocated here from `src/routing_profile.rs`, which
//! L3 removes; the `Selector` string grammar is the user-facing layer over
//! them (cards emit and consume selector strings).

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    sync::{Mutex, MutexGuard},
};

pub const SCHEMA_VERSION: u32 = 1;
pub const GLOBAL_SCOPE: &str = "global";

// ── Shared selector types (relocated from src/routing_profile.rs) ────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "backend", rename_all = "snake_case")]
pub enum ModelChoice {
    Auto,
    Local {
        #[serde(default)]
        model: Option<String>,
    },
    Cloud {
        #[serde(default)]
        model: Option<String>,
    },
}

impl<'de> Deserialize<'de> for ModelChoice {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Serde's internally tagged unit variants otherwise ignore extra fields.
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Choice {
            backend: String,
            #[serde(default)]
            model: Option<String>,
        }
        let choice = Choice::deserialize(deserializer)?;
        Self::from_legacy(&choice.backend, choice.model).map_err(serde::de::Error::custom)
    }
}

impl ModelChoice {
    pub fn local() -> Self {
        Self::Local { model: None }
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Local { model: Some(id) } | Self::Cloud { model: Some(id) } => {
                validate_model_id(id)
            }
            _ => Ok(()),
        }
    }

    pub fn from_legacy(mode: &str, model: Option<String>) -> Result<Self> {
        let choice = match mode {
            "auto" if model.is_none() => Self::Auto,
            "auto" => bail!("auto cannot specify a model; choose local or cloud"),
            "local" => Self::Local { model },
            "cloud" => Self::Cloud { model },
            _ => bail!("unknown routing mode {mode:?}; expected auto, local, or cloud"),
        };
        choice.validate()?;
        Ok(choice)
    }

    pub fn backend(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Local { .. } => "local",
            Self::Cloud { .. } => "cloud",
        }
    }

    pub fn model(&self) -> Option<&str> {
        match self {
            Self::Auto => None,
            Self::Local { model } | Self::Cloud { model } => model.as_deref(),
        }
    }

    /// Unambiguous selector used for routing and existing requested-model events.
    pub fn selector(&self) -> String {
        match self {
            Self::Auto => "auto".into(),
            Self::Local { model: None } => "local".into(),
            Self::Cloud { model: None } => "cloud".into(),
            Self::Local { model: Some(id) } => format!("brainrouter/{id}"),
            Self::Cloud { model: Some(id) } => format!("cloud/{id}"),
        }
    }

    /// The CARD spelling of this choice (L3 migration table, exact):
    /// Auto→`auto`, Local{None}→`local`, Local{Some}→`<bare-key>`,
    /// Cloud{None}→`cloud`, Cloud{Some}→`cloud/<id>`.
    pub fn card_selector(&self) -> String {
        match self {
            Self::Auto => "auto".into(),
            Self::Local { model: None } => "local".into(),
            Self::Cloud { model: None } => "cloud".into(),
            Self::Local { model: Some(id) } => id.clone(),
            Self::Cloud { model: Some(id) } => format!("cloud/{id}"),
        }
    }

    /// Compile a card selector string back into the routing choice.
    pub fn from_selector(selector: &str) -> Result<Self> {
        match parse_selector(selector) {
            Selector::Auto => Ok(Self::Auto),
            Selector::Subs => bail!("subs is not a card assignment"),
            Selector::Local => Ok(Self::local()),
            Selector::Cloud => Ok(Self::Cloud { model: None }),
            Selector::CloudId(id) => Ok(Self::Cloud { model: Some(id) }),
            Selector::Bare(key) => Ok(Self::Local { model: Some(key) }),
        }
    }
}

pub fn validate_model_id(id: &str) -> Result<()> {
    if id.is_empty() || id.len() > 512 || id.chars().any(|c| c.is_whitespace() || c.is_control()) {
        bail!("model ID must be 1-512 bytes with no whitespace or control characters");
    }
    if matches!(id, "auto" | "local" | "cloud" | "subs")
        || id.starts_with("brainrouter/")
        || id.starts_with("cloud/")
    {
        bail!("model ID must be an explicit provider ID, not a routing alias or selector");
    }
    Ok(())
}

pub fn validate_iterations(iterations: u32) -> Result<()> {
    if iterations == 0 {
        bail!("max_iterations must be greater than zero");
    }
    Ok(())
}

// ── Selector vocabulary (the card / proxy string grammar) ────────────────────

/// The selector grammar the proxy accepts and cards store (Selector-vocabulary
/// contract, preserved through L7): `auto | local | cloud | cloud/<id> |
/// <bare-key>`, plus the harness-side `subs` pool token and the
/// `brainrouter/…` prefixed spellings v1 accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selector {
    Auto,
    Subs,
    Local,
    Cloud,
    CloudId(String),
    Bare(String),
}

impl Selector {
    /// Explicit selectors bypass cards and are never policy-clamped.
    pub fn is_explicit(&self) -> bool {
        matches!(self, Self::Local | Self::Cloud | Self::CloudId(_) | Self::Bare(_))
    }

    /// The v1 route this selector compiles through.
    pub fn backend(&self) -> &'static str {
        match self {
            Self::Auto | Self::Subs => "auto",
            Self::Local | Self::Bare(_) => "local",
            Self::Cloud | Self::CloudId(_) => "cloud",
        }
    }
}

/// Parse a requested/card selector string. Never fails on shape; a card
/// assignment is validated with [`validate_card_assignment`] before storage.
pub fn parse_selector(raw: &str) -> Selector {
    let s = raw.trim();
    match s {
        "" | "auto" | "brainrouter/auto" => Selector::Auto,
        "subs" | "brainrouter/subs" => Selector::Subs,
        "local" | "brainrouter/local" => Selector::Local,
        "cloud" | "brainrouter/cloud" => Selector::Cloud,
        _ => {
            if let Some(rest) = s.strip_prefix("cloud/") {
                Selector::CloudId(rest.to_string())
            } else if let Some(rest) = s.strip_prefix("brainrouter/") {
                // `brainrouter/` with no key is an invalid bare id (rejected by
                // validation), never a silent Auto.
                Selector::Bare(rest.to_string())
            } else {
                Selector::Bare(s.to_string())
            }
        }
    }
}

/// Validate a card assignment string: exactly the vocabulary
/// `auto | local | cloud | cloud/<id> | <bare-key>`.
pub fn validate_card_assignment(raw: &str) -> Result<String> {
    let s = raw.trim();
    if s.is_empty() {
        bail!("card assignment must not be empty");
    }
    match parse_selector(s) {
        Selector::Auto | Selector::Local | Selector::Cloud => Ok(s.to_string()),
        Selector::CloudId(id) => {
            validate_model_id(&id).context("cloud/<id>")?;
            Ok(format!("cloud/{id}"))
        }
        Selector::Bare(key) => {
            validate_model_id(&key)?;
            Ok(key)
        }
        Selector::Subs => bail!("subs is a proxy pool token, not a card assignment"),
    }
}

// ── Roles ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Main,
    Review,
    Subagent,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Review => "review",
            Self::Subagent => "subagent",
        }
    }

    /// Accepts the v1 role spellings (`reviewer` ≡ `review`).
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "main" => Ok(Self::Main),
            "review" | "reviewer" => Ok(Self::Review),
            "subagent" | "subs" => Ok(Self::Subagent),
            other => bail!("unknown role {other:?}; expected main, review, or subagent"),
        }
    }
}

// ── Policy ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Privacy {
    CloudOk,
    LocalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Quality {
    Fast,
    Balanced,
    Maximum,
}

impl Privacy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CloudOk => "cloud-ok",
            Self::LocalOnly => "local-only",
        }
    }
}

impl Quality {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Balanced => "balanced",
            Self::Maximum => "maximum",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyRecord {
    pub scope: String,
    pub privacy: Privacy,
    pub quality: Quality,
}

impl PolicyRecord {
    /// The default policy raises no floor: quality `fast` (no tier raise) and
    /// privacy `cloud-ok` (today's behavior — "defaults keep today's
    /// behavior").
    pub const DEFAULT_PRIVACY: Privacy = Privacy::CloudOk;
    pub const DEFAULT_QUALITY: Quality = Quality::Fast;
}

impl Default for PolicyRecord {
    fn default() -> Self {
        Self {
            scope: GLOBAL_SCOPE.to_string(),
            privacy: Self::DEFAULT_PRIVACY,
            quality: Self::DEFAULT_QUALITY,
        }
    }
}

// ── Records ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CardRecord {
    pub scope: String,
    pub role: Role,
    pub assignment: String,
}

/// The review card's surviving non-model config (design: "schema reduced to
/// the non-model survivors").
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReviewExtras {
    pub max_iterations: u32,
    pub hankndory_integration: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub design_doc_path: Option<String>,
}

impl Default for ReviewExtras {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            hankndory_integration: false,
            design_doc_path: None,
        }
    }
}

/// The migration input: cards + review extras, seeded atomically into the
/// store by the `v2.0-p3` migration.
#[derive(Debug, Clone, Default)]
pub struct StoreDataForMigration {
    pub assignments: Vec<CardRecord>,
    pub review_extras: Option<ReviewExtras>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreData {
    #[serde(default = "default_schema_version")]
    schema_version: u32,
    #[serde(default)]
    assignments: Vec<CardRecord>,
    #[serde(default)]
    policy: Vec<PolicyRecord>,
    #[serde(default)]
    review_extras: Option<ReviewExtras>,
}

fn default_schema_version() -> u32 {
    SCHEMA_VERSION
}

impl Default for StoreData {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            assignments: Vec::new(),
            policy: Vec::new(),
            review_extras: None,
        }
    }
}

/// Serializes full-snapshot writes so concurrent writers cannot race the temp
/// path (the relocated `project_pins` machinery).
static WRITE_LOCK: Mutex<()> = Mutex::new(());

pub struct RoleCardStore {
    /// Cheap existence hint: a scoped card was seen at load/last set. The
    /// router skips the peer-cwd git-root walk entirely when false — and,
    /// critically, when true and the walk cannot resolve (a CLI peer that
    /// exec'd and has not republished /proc/<pid>/cwd under full-suite
    /// load), the request keeps the GLOBAL card: the project-scope answer
    /// is simply not knowable, and a never-resolving walk must not stall
    /// the request up to its grace window.
    pub(crate) has_scoped: std::sync::atomic::AtomicBool,
    state: Mutex<StoreData>,
    path: Option<PathBuf>,
}

impl RoleCardStore {
    pub fn has_scoped_cards(&self) -> bool {
        self.has_scoped.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// `$XDG_STATE_HOME/brainrouter/role_cards.json` (L3 contract pin).
    pub fn default_path() -> PathBuf {
        crate::paths::brainrouter_state().join("role_cards.json")
    }

    pub fn memory(data: StoreData) -> Self {
        let has_scoped = std::sync::atomic::AtomicBool::new(
            data.assignments.iter().any(|c| c.scope != GLOBAL_SCOPE),
        );
        Self {
            state: Mutex::new(data),
            path: None,
            has_scoped,
        }
    }

    /// Load; an absent/corrupt/unsupported file loads as the empty store
    /// (defaults everywhere) — the R9 fail-safe, relocated from project_pins.
    pub fn load(path: PathBuf) -> Self {
        let data = fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str::<StoreData>(&text).ok())
            .filter(|d| d.schema_version <= SCHEMA_VERSION)
            .unwrap_or_default();
        let has_scoped = std::sync::atomic::AtomicBool::new(
            data.assignments.iter().any(|c| c.scope != GLOBAL_SCOPE),
        );
        Self {
            state: Mutex::new(data),
            path: Some(path),
            has_scoped,
        }
    }

    fn lock(&self) -> MutexGuard<'_, StoreData> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn has_assignments(&self) -> bool {
        !self.lock().assignments.is_empty()
    }

    // ── Assignments ──────────────────────────────────────────────────────────

    /// Resolve a role's card assignment: project scope → global → `auto`.
    pub fn assignment(&self, role: Role, project: Option<&str>) -> String {
        let data = self.lock();
        for scope in [project, Some(GLOBAL_SCOPE)] {
            let Some(scope) = scope else { continue };
            if let Some(card) = data
                .assignments
                .iter()
                .find(|c| c.role == role && c.scope == scope)
            {
                return card.assignment.clone();
            }
        }
        "auto".to_string()
    }

    pub fn set_assignment(&self, scope: &str, role: Role, assignment: &str) -> Result<()> {
        if scope != GLOBAL_SCOPE {
            self.has_scoped.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let assignment = validate_card_assignment(assignment)?;
        let scope = normalize_scope(scope)?;
        let mut data = self.lock();
        let before = data.clone();
        if let Some(card) = data
            .assignments
            .iter_mut()
            .find(|c| c.role == role && c.scope == scope)
        {
            card.assignment = assignment;
        } else {
            data.assignments.push(CardRecord {
                scope: scope.clone(),
                role,
                assignment,
            });
        }
        // Disk failure must not mutate runtime (v1 ProfileStore keeper).
        self.persist(&data).inspect_err(|_| *data = before)
    }

    pub fn remove_assignment(&self, scope: &str, role: Role) -> Result<bool> {
        let scope = normalize_scope(scope)?;
        let mut data = self.lock();
        let before = data.assignments.len();
        data.assignments.retain(|c| !(c.role == role && c.scope == scope));
        let removed = data.assignments.len() != before;
        if removed {
            self.persist(&data)?;
        }
        Ok(removed)
    }

    pub fn assignments_snapshot(&self) -> Vec<CardRecord> {
        self.lock().assignments.clone()
    }

    // ── Policy ───────────────────────────────────────────────────────────────

    /// Resolve the guardrail policy: project scope → global → default.
    pub fn policy_for(&self, project: Option<&str>) -> PolicyRecord {
        let data = self.lock();
        for scope in [project, Some(GLOBAL_SCOPE)] {
            let Some(scope) = scope else { continue };
            if let Some(p) = data.policy.iter().find(|p| p.scope == scope) {
                return p.clone();
            }
        }
        PolicyRecord::default()
    }

    pub fn set_policy(&self, scope: &str, privacy: Privacy, quality: Quality) -> Result<()> {
        let scope = normalize_scope(scope)?;
        let mut data = self.lock();
        let before = data.clone();
        if let Some(p) = data.policy.iter_mut().find(|p| p.scope == scope) {
            p.privacy = privacy;
            p.quality = quality;
        } else {
            data.policy.push(PolicyRecord {
                scope,
                privacy,
                quality,
            });
        }
        self.persist(&data).inspect_err(|_| *data = before)
    }

    pub fn policy_snapshot(&self) -> Vec<PolicyRecord> {
        self.lock().policy.clone()
    }

    // ── Review card extras ───────────────────────────────────────────────────

    pub fn has_review_extras(&self) -> bool {
        self.lock().review_extras.is_some()
    }

    pub fn review_extras(&self) -> ReviewExtras {
        self.lock().review_extras.clone().unwrap_or_default()
    }

    pub fn set_review_extras(&self, extras: ReviewExtras) -> Result<()> {
        validate_iterations(extras.max_iterations)?;
        let mut data = self.lock();
        data.review_extras = Some(extras);
        self.persist(&data)
    }

    pub fn set_hankndory_integration(&self, enabled: bool) -> Result<()> {
        let mut data = self.lock();
        let mut extras = data.review_extras.clone().unwrap_or_default();
        extras.hankndory_integration = enabled;
        data.review_extras = Some(extras);
        self.persist(&data)
    }

    pub fn set_max_iterations(&self, iterations: u32) -> Result<()> {
        validate_iterations(iterations)?;
        let mut data = self.lock();
        let mut extras = data.review_extras.clone().unwrap_or_default();
        extras.max_iterations = iterations;
        data.review_extras = Some(extras);
        self.persist(&data)
    }

    pub fn set_design_doc_path(&self, path: Option<String>) -> Result<()> {
        let mut data = self.lock();
        let mut extras = data.review_extras.clone().unwrap_or_default();
        extras.design_doc_path = path;
        data.review_extras = Some(extras);
        self.persist(&data)
    }

    // ── Persistence ──────────────────────────────────────────────────────────

    fn persist(&self, data: &StoreData) -> Result<()> {
        if let Some(path) = &self.path {
            let bytes = serde_json::to_vec_pretty(data)?;
            atomic_write(path, &bytes)?;
        }
        Ok(())
    }

    /// Seed a full store payload (used by the v1 migration before first write).
    pub fn replace_all(&self, data: StoreData) -> Result<()> {
        let mut guard = self.lock();
        // The scoped-card hint (router fast path) must reflect the new
        // contents — the migration seed lands here with project cards.
        self.has_scoped.store(
            data.assignments.iter().any(|c| c.scope != GLOBAL_SCOPE),
            std::sync::atomic::Ordering::Relaxed,
        );
        *guard = data;
        self.persist(&guard)
    }

    /// Seed the store from the v1 migration (replaces everything).
    pub fn replace_from_migration(&self, data: StoreDataForMigration) -> Result<()> {
        self.replace_all(StoreData {
            schema_version: SCHEMA_VERSION,
            assignments: data.assignments,
            policy: Vec::new(),
            review_extras: data.review_extras,
        })
    }

    pub fn snapshot(&self) -> StoreData {
        let data = self.lock();
        StoreData {
            schema_version: data.schema_version,
            assignments: data.assignments.clone(),
            policy: data.policy.clone(),
            review_extras: data.review_extras.clone(),
        }
    }
}

/// Normalize a scope string: `global` (and empty) → `"global"`; anything else
/// canonicalizes to its git root via the relocated `project_key`.
pub fn normalize_scope(raw: &str) -> Result<String> {
    let raw = raw.trim();
    if raw.is_empty() || raw == GLOBAL_SCOPE {
        return Ok(GLOBAL_SCOPE.to_string());
    }
    project_key(raw)
        .map(|key| key.display().to_string())
        .with_context(|| format!("{raw:?} is not inside a git repository; scope cards by a git root (or 'global')"))
}

/// Normalize a working directory to its canonical repository-root key, or
/// `None`. Relocated from `src/project_pins.rs` — rules identical: empty or
/// relative ⇒ `None`; canonicalize; walk up to the nearest ancestor containing
/// a `.git` entry; non-UTF-8 ⇒ `None`.
pub fn project_key(cwd: &str) -> Option<PathBuf> {
    if cwd.is_empty() {
        return None;
    }
    let raw = PathBuf::from(cwd);
    if raw == PathBuf::from(GLOBAL_SCOPE) {
        return None;
    }
    if !raw.is_absolute() {
        return None;
    }
    let canonical = fs::canonicalize(&raw).ok()?;
    project_root_of(&canonical)
}

/// The git root of a canonicalized path (walk to the ancestor holding
/// `.git`). Split out so the router can probe synchronously on its task
/// (the default blocking pool saturates under full-suite load; this walk
/// is 2-3 syscalls on a hot dentry).
pub fn project_root_of(canonical: &Path) -> Option<PathBuf> {
    let mut cursor: &Path = canonical;
    loop {
        if cursor.join(".git").exists() {
            return Some(cursor.to_path_buf());
        }
        cursor = cursor.parent()?;
    }
}

/// 0600 + create_new + rename, relocated from `routing_profile::atomic_write`
/// via the project_pins durability pattern.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let _guard = WRITE_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let parent = path
        .parent()
        .context("role cards need a parent directory")?;
    fs::create_dir_all(parent).context("creating role cards directory")?;
    let temporary = parent.join(format!(".role-cards-{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        Ok::<_, std::io::Error>(())
    })();
    if result.is_err() {
        if let Err(error) = fs::remove_file(&temporary) {
            if error.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(%error, "Failed to remove temporary role cards");
            }
        }
    }
    result.context("persisting role cards")
}

// ── v1 → card migration (L3 spec: exact table, no inference) ─────────────────

/// The v1 profile record shape persisted in `routing_state.json` (the
/// relocated `RoutingProfile`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyRoutingProfile {
    #[serde(default)]
    pub preset: LegacyPreset,
    pub main: ModelChoice,
    pub reviewer: ModelChoice,
    #[serde(default)]
    pub subagent_model: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LegacyPreset {
    Auto,
    Cloud,
    LocalMainSub,
    LocalCustom,
    CloudMainLocalReview,
    LocalMainCloudReview,
    #[default]
    Custom,
}

/// One v1 project pin (project_pins.json, schema 1).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct LegacyPin {
    #[serde(default)]
    pub main: Option<ModelChoice>,
    #[serde(default)]
    pub reviewer: Option<ModelChoice>,
    #[serde(default)]
    pub subagent: Option<String>,
}

/// The cards computed by a v1 → v2.0-p3 migration.
#[derive(Debug, Clone, Default)]
pub struct MigrationOutput {
    pub cards: Vec<CardRecord>,
    pub review_extras: Option<ReviewExtras>,
}

/// Expand the v1 state into cards via the exact migration table.
///
/// `profile` — the effective v1 routing profile: `routing_state.json` when
/// present, else the YAML `routing:` block, else the v1 default (main Local
/// {None} = `local`, reviewer from `review.forced_*`, subagent from the profile
/// block ONLY — `config.subs_model` stays as the auto-pool compatibility seed
/// and never becomes a card).
/// `pins` — the v1 `project_pins.json` map (canonical root → roles).
/// `review` — the review extras seed (max_iterations, hankndory_integration,
/// design_doc_path) from the YAML review block.
pub fn migrate_v1(
    profile: &LegacyRoutingProfile,
    pins: &std::collections::BTreeMap<String, LegacyPin>,
    review: Option<ReviewExtras>,
) -> MigrationOutput {
    let mut cards = vec![
        CardRecord {
            scope: GLOBAL_SCOPE.to_string(),
            role: Role::Main,
            assignment: profile.main.card_selector(),
        },
        CardRecord {
            scope: GLOBAL_SCOPE.to_string(),
            role: Role::Review,
            assignment: profile.reviewer.card_selector(),
        },
    ];
    if let Some(sub) = &profile.subagent_model {
        cards.push(CardRecord {
            scope: GLOBAL_SCOPE.to_string(),
            role: Role::Subagent,
            assignment: sub.clone(),
        });
    }
    for (scope, pin) in pins {
        if let Some(main) = &pin.main {
            cards.push(CardRecord {
                scope: scope.clone(),
                role: Role::Main,
                assignment: main.card_selector(),
            });
        }
        if let Some(reviewer) = &pin.reviewer {
            cards.push(CardRecord {
                scope: scope.clone(),
                role: Role::Review,
                assignment: reviewer.card_selector(),
            });
        }
        if let Some(sub) = &pin.subagent {
            cards.push(CardRecord {
                scope: scope.clone(),
                role: Role::Subagent,
                assignment: sub.clone(),
            });
        }
    }
    MigrationOutput {
        cards,
        review_extras: review,
    }
}

/// Parse a v1 `project_pins.json` (schema 1) into the migration input.
pub fn read_legacy_pins(path: &Path) -> Result<std::collections::BTreeMap<String, LegacyPin>> {
    let text = fs::read_to_string(path)?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parsing {}", path.display()))?;
    let pins = value.get("pins").cloned().unwrap_or(serde_json::Value::Null);
    let mut out = std::collections::BTreeMap::new();
    match pins {
        serde_json::Value::Object(map) => {
            for (key, pin) in map {
                out.insert(
                    key,
                    serde_json::from_value(pin).context("decoding v1 project pin")?,
                );
            }
        }
        serde_json::Value::Null => {}
        other => bail!(
            "v1 pins must be an object, got {}",
            serde_json::to_string(&other).unwrap_or_default()
        ),
    }
    Ok(out)
}

/// Parse a v1 `routing_state.json` (the persisted RoutingProfile).
pub fn read_legacy_routing_state(path: &Path) -> Result<LegacyRoutingProfile> {
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)
        .with_context(|| format!("parsing {}", path.display()))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn selector_grammar_parses_the_vocabulary() {
        assert_eq!(parse_selector("auto"), Selector::Auto);
        assert_eq!(parse_selector("brainrouter/auto"), Selector::Auto);
        assert_eq!(parse_selector(""), Selector::Auto);
        assert_eq!(parse_selector("subs"), Selector::Subs);
        assert_eq!(parse_selector("brainrouter/subs"), Selector::Subs);
        assert_eq!(parse_selector("local"), Selector::Local);
        assert_eq!(parse_selector("cloud"), Selector::Cloud);
        assert_eq!(
            parse_selector("cloud/vendor/x-1"),
            Selector::CloudId("vendor/x-1".into())
        );
        assert_eq!(parse_selector("model-key"), Selector::Bare("model-key".into()));
        assert_eq!(
            parse_selector("brainrouter/model-key"),
            Selector::Bare("model-key".into())
        );
    }

    #[test]
    fn explicit_selectors_bypass_cards() {
        assert!(parse_selector("local").is_explicit());
        assert!(parse_selector("cloud").is_explicit());
        assert!(parse_selector("cloud/a").is_explicit());
        assert!(parse_selector("key").is_explicit());
        assert!(!parse_selector("auto").is_explicit());
        assert!(!parse_selector("subs").is_explicit());
    }

    #[test]
    fn card_assignment_validation() {
        assert_eq!(validate_card_assignment("auto").unwrap(), "auto");
        assert_eq!(validate_card_assignment("local").unwrap(), "local");
        assert_eq!(validate_card_assignment("cloud").unwrap(), "cloud");
        assert_eq!(
            validate_card_assignment("cloud/vendor/x").unwrap(),
            "cloud/vendor/x"
        );
        assert_eq!(validate_card_assignment("model-key").unwrap(), "model-key");
        assert!(validate_card_assignment("subs").is_err());
        assert!(validate_card_assignment("cloud/").is_err());
        assert!(validate_card_assignment("").is_err());
        assert!(validate_card_assignment("has space").is_err());
    }

    #[test]
    fn migration_table_rows_exact() {
        let pins = [("repo".to_string(), LegacyPin {
            main: Some(ModelChoice::Cloud { model: Some("pin-cloud-id".into()) }),
            reviewer: Some(ModelChoice::Cloud { model: Some("vendor/review-2026".into()) }),
            subagent: Some("pin-subs-model".into()),
        })]
        .into_iter()
        .collect();
        let profile = LegacyRoutingProfile {
            preset: LegacyPreset::Custom,
            main: ModelChoice::local(),
            reviewer: ModelChoice::Cloud { model: None },
            subagent_model: Some("prof-subs-model".into()),
        };
        let out = migrate_v1(&profile, &pins, None);
        let get = |role, scope| {
            out.cards
                .iter()
                .find(|c| c.role == role && c.scope == scope)
                .map(|c| c.assignment.clone())
        };
        assert_eq!(get(Role::Main, GLOBAL_SCOPE).unwrap(), "local");
        assert_eq!(get(Role::Review, GLOBAL_SCOPE).unwrap(), "cloud");
        assert_eq!(get(Role::Subagent, GLOBAL_SCOPE).unwrap(), "prof-subs-model");
        assert_eq!(get(Role::Main, "repo").unwrap(), "cloud/pin-cloud-id");
        assert_eq!(get(Role::Review, "repo").unwrap(), "cloud/vendor/review-2026");
        assert_eq!(get(Role::Subagent, "repo").unwrap(), "pin-subs-model");
    }

    #[test]
    fn card_selector_round_trips() {
        assert_eq!(ModelChoice::Auto.card_selector(), "auto");
        assert_eq!(ModelChoice::local().card_selector(), "local");
        assert_eq!(
            ModelChoice::Local { model: Some("k".into()) }.card_selector(),
            "k"
        );
        assert_eq!(ModelChoice::Cloud { model: None }.card_selector(), "cloud");
        assert_eq!(
            ModelChoice::Cloud { model: Some("v/x".into()) }.card_selector(),
            "cloud/v/x"
        );
        assert_eq!(
            ModelChoice::from_selector("k").unwrap(),
            ModelChoice::Local { model: Some("k".into()) }
        );
        assert_eq!(
            ModelChoice::from_selector("cloud/v/x").unwrap(),
            ModelChoice::Cloud { model: Some("v/x".into()) }
        );
    }

    #[test]
    fn store_lookup_project_beats_global_beats_default() {
        let store = RoleCardStore::memory(StoreData::default());
        assert_eq!(store.assignment(Role::Main, Some("/repo")), "auto");
        store.set_assignment(GLOBAL_SCOPE, Role::Main, "cloud").unwrap();
        assert_eq!(store.assignment(Role::Main, Some("/repo")), "cloud");
        // Project scope requires a git root; plain paths fall back to global.
        store.set_assignment(GLOBAL_SCOPE, Role::Main, "local").unwrap();
        assert_eq!(store.assignment(Role::Main, None), "local");
        assert_eq!(store.assignment(Role::Subagent, Some("/repo")), "auto");
    }

    #[test]
    fn policy_lookup_precedence() {
        let store = RoleCardStore::memory(StoreData::default());
        let p = store.policy_for(None);
        assert_eq!(p.privacy, Privacy::CloudOk);
        assert_eq!(p.quality, Quality::Fast, "default quality raises no floor");
        assert_eq!(store.policy_for(Some("/repo")).quality, Quality::Fast);
        store
            .set_policy(GLOBAL_SCOPE, Privacy::LocalOnly, Quality::Fast)
            .unwrap();
        assert_eq!(store.policy_for(Some("/repo")).privacy, Privacy::LocalOnly);
    }

    #[test]
    fn persistence_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("role_cards.json");
        {
            let store = RoleCardStore::load(path.clone());
            store.set_assignment(GLOBAL_SCOPE, Role::Main, "local").unwrap();
            store
                .set_review_extras(ReviewExtras {
                    max_iterations: 3,
                    hankndory_integration: true,
                    design_doc_path: None,
                })
                .unwrap();
        }
        let mode = fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
        let store = RoleCardStore::load(path);
        assert_eq!(store.assignment(Role::Main, None), "local");
        assert_eq!(store.review_extras().max_iterations, 3);
        assert!(store.review_extras().hankndory_integration);
    }

    #[test]
    fn role_parse_accepts_v1_spellings() {
        assert_eq!(Role::parse("reviewer").unwrap(), Role::Review);
        assert_eq!(Role::parse("review").unwrap(), Role::Review);
        assert_eq!(Role::parse("main").unwrap(), Role::Main);
        assert!(Role::parse("nope").is_err());
    }
}
