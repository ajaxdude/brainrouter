//! Validated role choices and atomic, per-user routing preferences.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    sync::Mutex,
};

use crate::config::ReviewConfig;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
#[clap(rename_all = "snake_case")]
pub enum RoutingPreset {
    Auto,
    Cloud,
    LocalMainSub,
    LocalCustom,
    CloudMainLocalReview,
    LocalMainCloudReview,
    Custom,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingProfile {
    pub preset: RoutingPreset,
    pub main: ModelChoice,
    pub reviewer: ModelChoice,
    /// Only `subs` requests use this local pool. None retains legacy auto routing.
    #[serde(default)]
    pub subagent_model: Option<String>,
}

impl RoutingProfile {
    pub fn validate(&self) -> Result<()> {
        self.main.validate().context("main")?;
        self.reviewer.validate().context("reviewer")?;
        if let Some(id) = &self.subagent_model {
            validate_model_id(id).context("subagent_model")?;
        }
        let backends = (self.main.backend(), self.reviewer.backend());
        let valid = match self.preset {
            RoutingPreset::Auto => backends == ("auto", "auto"),
            RoutingPreset::Cloud => backends == ("cloud", "cloud"),
            RoutingPreset::LocalMainSub => backends == ("local", "local"),
            RoutingPreset::LocalCustom => {
                backends == ("local", "local") && self.main.model().is_some()
            }
            RoutingPreset::CloudMainLocalReview => backends == ("cloud", "local"),
            RoutingPreset::LocalMainCloudReview => backends == ("local", "cloud"),
            RoutingPreset::Custom => true,
        };
        if !valid {
            bail!(
                "role choices do not match the selected preset; use custom for independent choices"
            );
        }
        Ok(())
    }

    /// Presets never change the independently selected subagent pool.
    pub fn apply_preset(&mut self, preset: RoutingPreset) -> Result<()> {
        let (main, reviewer) = match preset {
            RoutingPreset::Auto => (ModelChoice::Auto, ModelChoice::Auto),
            RoutingPreset::Cloud => (
                ModelChoice::Cloud { model: None },
                ModelChoice::Cloud { model: None },
            ),
            RoutingPreset::LocalMainSub => (ModelChoice::local(), ModelChoice::local()),
            RoutingPreset::LocalCustom => {
                if !matches!(self.main, ModelChoice::Local { model: Some(_) }) {
                    bail!("local_custom requires an explicit local main model");
                }
                (self.main.clone(), ModelChoice::local())
            }
            RoutingPreset::CloudMainLocalReview => {
                (ModelChoice::Cloud { model: None }, ModelChoice::local())
            }
            RoutingPreset::LocalMainCloudReview => {
                (ModelChoice::local(), ModelChoice::Cloud { model: None })
            }
            RoutingPreset::Custom => (self.main.clone(), self.reviewer.clone()),
        };
        self.main = main;
        self.reviewer = reviewer;
        self.preset = preset;
        self.validate()
    }
}

#[derive(Debug, Clone)]
struct Preferences {
    profile: RoutingProfile,
    max_iterations: u32,
}

pub struct ProfileStore {
    state: Mutex<Preferences>,
    path: Option<PathBuf>,
}

impl ProfileStore {
    pub fn memory(profile: RoutingProfile, max_iterations: u32) -> Result<Self> {
        profile.validate()?;
        validate_iterations(max_iterations)?;
        Ok(Self {
            state: Mutex::new(Preferences {
                profile,
                max_iterations,
            }),
            path: None,
        })
    }

    pub fn load(path: PathBuf, profile: RoutingProfile, review: &ReviewConfig) -> Result<Self> {
        let mut store = Self::memory(profile, review.max_iterations).with_context(|| {
            format!("invalid initial routing preferences for {}", path.display())
        })?;
        let mut migrated_from = None;
        match fs::read(&path) {
            Ok(bytes) => {
                let saved: RoutingProfile = serde_json::from_slice(&bytes)
                    .with_context(|| format!("invalid routing preferences: {}", path.display()))?;
                saved
                    .validate()
                    .with_context(|| format!("invalid routing preferences: {}", path.display()))?;
                store.state.get_mut().unwrap().profile = saved;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Migrate only the UI overrides, as the old review service did.
                let legacy = path.with_file_name("review_state.json");
                match fs::read(&legacy) {
                    Ok(bytes) => {
                        let mut saved: ReviewConfig =
                            serde_json::from_slice(&bytes).with_context(|| {
                                format!("invalid legacy review preferences: {}", legacy.display())
                            })?;
                        saved.normalize_legacy_read(&legacy);
                        saved.validate().with_context(|| {
                            format!("invalid legacy review preferences: {}", legacy.display())
                        })?;
                        let state = store.state.get_mut().unwrap();
                        state.profile.reviewer = saved.model_choice().with_context(|| {
                            format!(
                                "converting legacy reviewer choice from {}",
                                legacy.display()
                            )
                        })?;
                        state.profile.preset = RoutingPreset::Custom;
                        migrated_from = Some(legacy);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => {
                        return Err(e).with_context(|| {
                            format!("reading legacy review preferences: {}", legacy.display())
                        })
                    }
                }
            }
            Err(e) => {
                return Err(e)
                    .with_context(|| format!("reading routing preferences: {}", path.display()))
            }
        }
        if let Some(legacy) = migrated_from {
            let bytes = serde_json::to_vec_pretty(&store.profile()).with_context(|| {
                format!(
                    "serializing migrated routing preferences from {}",
                    legacy.display()
                )
            })?;
            atomic_write(&path, &bytes).with_context(|| {
                format!(
                    "migrating legacy review preferences from {} to {}; check the destination directory permissions",
                    legacy.display(),
                    path.display()
                )
            })?;
        }
        store.path = Some(path);
        Ok(store)
    }

    pub fn profile(&self) -> RoutingProfile {
        self.state.lock().unwrap().profile.clone()
    }

    pub fn review_config(&self) -> ReviewConfig {
        let state = self.state.lock().unwrap();
        ReviewConfig {
            max_iterations: state.max_iterations,
            forced_mode: state.profile.reviewer.backend().into(),
            forced_model: state.profile.reviewer.model().map(str::to_owned),
        }
    }

    pub fn update_profile(&self, profile: RoutingProfile) -> Result<()> {
        profile.validate()?;
        self.update(move |state| {
            state.profile = profile;
            Ok(())
        })
    }

    pub fn update_main(&self, choice: ModelChoice) -> Result<()> {
        choice.validate()?;
        self.update(move |state| {
            state.profile.main = choice;
            state.profile.preset = RoutingPreset::Custom;
            Ok(())
        })
    }

    pub fn update_review(&self, review: ReviewConfig) -> Result<()> {
        review.validate()?;
        self.update(move |state| {
            state.profile.reviewer = review.model_choice()?;
            state.profile.preset = RoutingPreset::Custom;
            state.max_iterations = review.max_iterations;
            Ok(())
        })
    }

    fn update(&self, change: impl FnOnce(&mut Preferences) -> Result<()>) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        let mut candidate = state.clone();
        change(&mut candidate)?;
        candidate.profile.validate()?;
        if let Some(path) = &self.path {
            atomic_write(path, &serde_json::to_vec_pretty(&candidate.profile)?)?;
        }
        // No runtime mutation until validation AND durable storage succeed.
        *state = candidate;
        Ok(())
    }
}

pub fn validate_iterations(iterations: u32) -> Result<()> {
    if iterations == 0 {
        bail!("max_iterations must be greater than zero");
    }
    Ok(())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .context("routing preferences need a parent directory")?;
    fs::create_dir_all(parent).context("creating routing preferences directory")?;
    let temporary = parent.join(format!(".routing-{}.tmp", uuid::Uuid::new_v4()));
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
                tracing::warn!(%error, "Failed to remove temporary routing preferences");
            }
        }
    }
    result.context("persisting routing preferences")
}
