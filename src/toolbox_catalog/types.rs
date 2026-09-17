//! Strongly-typed representation of `assets/cockpit-catalog/toolboxes.json`.
//!
//! Two backend-identity types exist deliberately (see
//! `docs/design/ai-toolbox-cockpit-integration.md` §2):
//!
//! - [`CatalogBackendId`] is **open and lossless**: every backend id the
//!   catalog can contain today (including `comfyui`) or ever adds in the
//!   future parses successfully. Nothing about *reading* the catalog can
//!   fail just because upstream added a backend brainrouter doesn't have
//!   code for yet.
//! - [`SupportedServingBackend`] is the **closed** set of backends
//!   brainrouter can actually *act on* (list/create/update/delete/serve).
//!   `comfyui` is deliberately excluded per the integration's explicit
//!   scope (requirement 1: "exclude comfyui... image-gen, not
//!   coding-relevant"), and any backend upstream adds before brainrouter has
//!   code for it also falls outside this set until a future PR adds it.

use std::collections::BTreeMap;

use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Backend identifier exactly as it appears in the catalog JSON.
///
/// `serde`'s built-in `#[serde(other)]` enum fallback discards the original
/// string on an unrecognized variant, which isn't good enough here — we need
/// the literal id back for display/debugging/future-proofing (e.g. showing
/// "unrecognized backend: foo_v2" in the dashboard rather than a generic
/// "other"). That's why this has a hand-written `Deserialize` impl instead
/// of a derive.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CatalogBackendId {
    LlamaCpp,
    Ds4,
    Halogen,
    Vllm,
    R9v,
    /// Parsed, but never surfaced in any brainrouter UI/API list per
    /// requirement 1 — image generation, not coding-relevant.
    Comfyui,
    /// Any backend id the catalog contains that brainrouter doesn't have a
    /// named variant for yet (upstream added it, or it's a local/test
    /// fixture). Round-trips losslessly.
    Other(String),
}

impl CatalogBackendId {
    /// The literal string this id serializes to / was parsed from.
    pub fn as_str(&self) -> &str {
        match self {
            Self::LlamaCpp => "llama_cpp",
            Self::Ds4 => "ds4",
            Self::Halogen => "halogen",
            Self::Vllm => "vllm",
            Self::R9v => "r9v",
            Self::Comfyui => "comfyui",
            Self::Other(raw) => raw,
        }
    }

    /// Parses a raw backend-id string (e.g. a `models.json` `backends` map
    /// key) into a `CatalogBackendId`, losslessly falling back to `Other`
    /// for anything unrecognized. `pub(crate)` because it's used by
    /// `models.rs` to interpret map keys, which don't go through serde's
    /// normal `Deserialize` (the map key type there is a plain `String`).
    pub(crate) fn from_str(raw: &str) -> Self {
        match raw {
            "llama_cpp" => Self::LlamaCpp,
            "ds4" => Self::Ds4,
            "halogen" => Self::Halogen,
            "vllm" => Self::Vllm,
            "r9v" => Self::R9v,
            "comfyui" => Self::Comfyui,
            other => Self::Other(other.to_string()),
        }
    }
}

impl std::fmt::Display for CatalogBackendId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for CatalogBackendId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for CatalogBackendId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Ok(Self::from_str(&raw))
    }
}

/// The narrower, closed set of backends brainrouter can actually *execute*
/// actions for. `Comfyui` and `Other` are excluded — they can still be
/// *listed* (via [`CatalogBackendId`]) but never acted on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SupportedServingBackend {
    LlamaCpp,
    Ds4,
    Halogen,
    Vllm,
    R9v,
}

impl SupportedServingBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::LlamaCpp => "llama_cpp",
            Self::Ds4 => "ds4",
            Self::Halogen => "halogen",
            Self::Vllm => "vllm",
            Self::R9v => "r9v",
        }
    }

    /// All five supported backends, in a stable display order (matches the
    /// order they're introduced in the design doc / user requirements).
    pub const ALL: [SupportedServingBackend; 5] = [
        SupportedServingBackend::LlamaCpp,
        SupportedServingBackend::Ds4,
        SupportedServingBackend::Halogen,
        SupportedServingBackend::Vllm,
        SupportedServingBackend::R9v,
    ];
}

impl std::fmt::Display for SupportedServingBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error returned when a [`CatalogBackendId`] isn't one brainrouter can act
/// on (currently `Comfyui` or `Other`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnsupportedBackend(pub CatalogBackendId);

impl std::fmt::Display for UnsupportedBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "backend {:?} is not a supported serving backend (comfyui is out of scope; \
             unrecognized ids need a brainrouter code change first)",
            self.0.as_str()
        )
    }
}

impl std::error::Error for UnsupportedBackend {}

impl TryFrom<&CatalogBackendId> for SupportedServingBackend {
    type Error = UnsupportedBackend;

    fn try_from(value: &CatalogBackendId) -> Result<Self, Self::Error> {
        match value {
            CatalogBackendId::LlamaCpp => Ok(Self::LlamaCpp),
            CatalogBackendId::Ds4 => Ok(Self::Ds4),
            CatalogBackendId::Halogen => Ok(Self::Halogen),
            CatalogBackendId::Vllm => Ok(Self::Vllm),
            CatalogBackendId::R9v => Ok(Self::R9v),
            CatalogBackendId::Comfyui | CatalogBackendId::Other(_) => {
                Err(UnsupportedBackend(value.clone()))
            }
        }
    }
}

impl From<SupportedServingBackend> for CatalogBackendId {
    fn from(value: SupportedServingBackend) -> Self {
        match value {
            SupportedServingBackend::LlamaCpp => CatalogBackendId::LlamaCpp,
            SupportedServingBackend::Ds4 => CatalogBackendId::Ds4,
            SupportedServingBackend::Halogen => CatalogBackendId::Halogen,
            SupportedServingBackend::Vllm => CatalogBackendId::Vllm,
            SupportedServingBackend::R9v => CatalogBackendId::R9v,
        }
    }
}

/// `toolboxes.json`'s lifecycle-maturity vocabulary. Closed sets (unlike
/// backend ids) because these are brainrouter/cockpit's own UI vocabulary,
/// not upstream identity that needs future-proofing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Channel {
    Stable,
    Development,
    Experimental,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Maturity {
    Stable,
    Experimental,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureState {
    Supported,
    Experimental,
    Unavailable,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolboxFeatures {
    pub interactive: FeatureState,
    pub server: FeatureState,
    pub models: FeatureState,
}

impl ToolboxFeatures {
    pub fn state(&self, feature: &str) -> FeatureState {
        match feature {
            "interactive" => self.interactive,
            "server" => self.server,
            "models" => self.models,
            _ => FeatureState::Unavailable,
        }
    }
}

/// One entry from `toolboxes.json`'s `runtime_profiles` map: the raw podman
/// flags (`--device`, `--group-add`, `--security-opt`, `--env`, …) that must
/// be threaded into `toolbox`/`podman create` for a toolbox using this
/// profile. See design doc §5 — whether `toolbox create` itself can forward
/// all of these is the PR10 R9V spike's open question; this type just holds
/// the data losslessly regardless of how it ends up invoked.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeProfile {
    pub id: String,
    #[serde(default)]
    pub engine_args: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct RuntimeProfileRaw {
    #[serde(default)]
    engine_args: Vec<String>,
}

/// One entry from `toolboxes.json`'s `toolboxes[]` array.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolboxDefinition {
    pub id: String,
    pub backend: CatalogBackendId,
    pub name: String,
    pub container_name: String,
    pub group: String,
    pub image: String,
    pub channel: Channel,
    pub maturity: Maturity,
    #[serde(default)]
    pub description: String,
    pub runtime_profile: String,
    #[serde(default)]
    pub supports_load_mode: bool,
    pub features: ToolboxFeatures,
    /// `backend_config` varies per backend (llama_cpp's `recommended_use`
    /// sidecar shape vs. others) and isn't modeled further here — kept loose
    /// deliberately, matching how `ModelPayload`'s per-backend fields
    /// balance strong typing for common fields against a raw fallback for
    /// backend-specific structure (see `models.rs`).
    #[serde(default)]
    pub backend_config: Option<Value>,
    #[serde(default = "default_toolbox_compatible")]
    pub toolbox_compatible: bool,
}

fn default_toolbox_compatible() -> bool {
    true
}

impl ToolboxDefinition {
    /// The backend this toolbox belongs to, narrowed to the closed set
    /// brainrouter can act on. `None` for `comfyui`/unrecognized backends.
    pub fn supported_backend(&self) -> Option<SupportedServingBackend> {
        SupportedServingBackend::try_from(&self.backend).ok()
    }
}

/// One entry from `toolboxes.json`'s `platforms[]` array — a named hardware
/// profile (e.g. "AMD Strix Halo") grouping which toolboxes apply to it and
/// which is the default per backend.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Platform {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub toolbox_ids: Vec<String>,
    /// backend id (raw string, as it appears in JSON) -> default toolbox id
    /// for that backend on this platform.
    #[serde(default)]
    pub defaults: BTreeMap<String, String>,
}

/// Error returned by [`ToolboxCatalog::parse`]. Distinct from
/// `schema_validate`'s [`super::schema_validate::ValidationReport`] —
/// this is a hard parse failure (the JSON doesn't even deserialize into the
/// expected shape), whereas the validator can report softer structural
/// issues alongside a value that still deserializes.
#[derive(Debug)]
pub struct CatalogParseError(pub String);

impl std::fmt::Display for CatalogParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "toolbox catalog parse error: {}", self.0)
    }
}

impl std::error::Error for CatalogParseError {}

/// The fully parsed, typed `toolboxes.json`.
#[derive(Clone, Debug)]
pub struct ToolboxCatalog {
    pub schema_version: i64,
    pub runtime_profiles: BTreeMap<String, RuntimeProfile>,
    pub toolboxes: Vec<ToolboxDefinition>,
    pub platforms: Vec<Platform>,
}

impl ToolboxCatalog {
    /// Parses a `toolboxes.json` document. Callers that also want the
    /// softer structural-validation report (warnings for unrecognized
    /// backend ids, etc.) should additionally call
    /// [`super::schema_validate::validate_toolboxes_json`] — the two are
    /// independent and both cheap to run.
    pub fn parse(value: &Value) -> Result<Self, CatalogParseError> {
        let root = value
            .as_object()
            .ok_or_else(|| CatalogParseError("root must be an object".to_string()))?;

        let schema_version = root
            .get("schema_version")
            .and_then(Value::as_i64)
            .ok_or_else(|| CatalogParseError("schema_version must be an integer".to_string()))?;

        let raw_profiles: BTreeMap<String, RuntimeProfileRaw> = root
            .get("runtime_profiles")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| CatalogParseError(format!("runtime_profiles: {e}")))?
            .unwrap_or_default();
        let runtime_profiles: BTreeMap<String, RuntimeProfile> = raw_profiles
            .into_iter()
            .map(|(id, raw)| {
                (
                    id.clone(),
                    RuntimeProfile {
                        id,
                        engine_args: raw.engine_args,
                    },
                )
            })
            .collect();

        let toolboxes: Vec<ToolboxDefinition> = root
            .get("toolboxes")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| CatalogParseError(format!("toolboxes: {e}")))?
            .unwrap_or_default();

        let platforms: Vec<Platform> = root
            .get("platforms")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| CatalogParseError(format!("platforms: {e}")))?
            .unwrap_or_default();

        Ok(ToolboxCatalog {
            schema_version,
            runtime_profiles,
            toolboxes,
            platforms,
        })
    }

    /// Toolboxes for a given supported backend, in catalog order.
    pub fn toolboxes_for(
        &self,
        backend: SupportedServingBackend,
    ) -> impl Iterator<Item = &ToolboxDefinition> {
        self.toolboxes
            .iter()
            .filter(move |t| t.supported_backend() == Some(backend))
    }

    pub fn toolbox_by_id(&self, id: &str) -> Option<&ToolboxDefinition> {
        self.toolboxes.iter().find(|t| t.id == id)
    }

    pub fn platform_by_id(&self, id: &str) -> Option<&Platform> {
        self.platforms.iter().find(|p| p.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vendored_toolboxes() -> Value {
        serde_json::from_str(include_str!("../../assets/cockpit-catalog/toolboxes.json"))
            .expect("vendored toolboxes.json must be valid JSON")
    }

    #[test]
    fn catalog_backend_id_round_trips_known_and_unknown_ids() {
        for (raw, expected) in [
            ("llama_cpp", CatalogBackendId::LlamaCpp),
            ("ds4", CatalogBackendId::Ds4),
            ("halogen", CatalogBackendId::Halogen),
            ("vllm", CatalogBackendId::Vllm),
            ("r9v", CatalogBackendId::R9v),
            ("comfyui", CatalogBackendId::Comfyui),
        ] {
            let parsed: CatalogBackendId =
                serde_json::from_value(Value::String(raw.to_string())).unwrap();
            assert_eq!(parsed, expected);
            assert_eq!(parsed.as_str(), raw);
        }

        let unknown: CatalogBackendId =
            serde_json::from_value(Value::String("brand_new_future_backend".to_string()))
                .unwrap();
        assert_eq!(
            unknown,
            CatalogBackendId::Other("brand_new_future_backend".to_string())
        );
        assert_eq!(unknown.as_str(), "brand_new_future_backend");
        // Round-trip through serialize too.
        let serialized = serde_json::to_value(&unknown).unwrap();
        assert_eq!(serialized, Value::String("brand_new_future_backend".to_string()));
    }

    #[test]
    fn supported_serving_backend_excludes_comfyui_and_other() {
        assert!(SupportedServingBackend::try_from(&CatalogBackendId::LlamaCpp).is_ok());
        assert!(SupportedServingBackend::try_from(&CatalogBackendId::Ds4).is_ok());
        assert!(SupportedServingBackend::try_from(&CatalogBackendId::Halogen).is_ok());
        assert!(SupportedServingBackend::try_from(&CatalogBackendId::Vllm).is_ok());
        assert!(SupportedServingBackend::try_from(&CatalogBackendId::R9v).is_ok());
        assert!(SupportedServingBackend::try_from(&CatalogBackendId::Comfyui).is_err());
        assert!(SupportedServingBackend::try_from(&CatalogBackendId::Other(
            "brand_new_future_backend".to_string()
        ))
        .is_err());
    }

    #[test]
    fn parses_the_real_vendored_toolboxes_json() {
        let doc = vendored_toolboxes();
        let catalog = ToolboxCatalog::parse(&doc).expect("must parse");
        assert_eq!(catalog.schema_version, 3);
        assert!(!catalog.runtime_profiles.is_empty());
        assert!(!catalog.toolboxes.is_empty());
        assert!(!catalog.platforms.is_empty());

        // comfyui toolboxes parse (backend id recognized) but are excluded
        // from the supported set.
        let comfyui_toolboxes: Vec<_> = catalog
            .toolboxes
            .iter()
            .filter(|t| t.backend == CatalogBackendId::Comfyui)
            .collect();
        assert!(!comfyui_toolboxes.is_empty(), "fixture must contain comfyui toolboxes");
        assert!(comfyui_toolboxes.iter().all(|t| t.supported_backend().is_none()));

        // Every supported backend has at least one toolbox in the real data.
        for backend in SupportedServingBackend::ALL {
            let count = catalog.toolboxes_for(backend).count();
            assert!(count > 0, "expected at least one toolbox for {backend}");
        }
    }

    #[test]
    fn parses_lossless_with_an_unrecognized_backend_id() {
        let mut doc = vendored_toolboxes();
        let root = doc.as_object_mut().unwrap();
        let toolboxes = root.get_mut("toolboxes").unwrap().as_array_mut().unwrap();
        let index = toolboxes
            .iter()
            .position(|t| t["id"] == "strix-halo-llama-hrx-staging")
            .unwrap();
        toolboxes[index]["backend"] = Value::String("brand_new_future_backend".to_string());

        let catalog = ToolboxCatalog::parse(&doc).expect("must still parse");
        let toolbox = catalog
            .toolbox_by_id("strix-halo-llama-hrx-staging")
            .unwrap();
        assert_eq!(
            toolbox.backend,
            CatalogBackendId::Other("brand_new_future_backend".to_string())
        );
        assert!(toolbox.supported_backend().is_none());
    }

    #[test]
    fn toolbox_by_id_and_platform_by_id_find_real_entries() {
        let catalog = ToolboxCatalog::parse(&vendored_toolboxes()).unwrap();
        let toolbox = catalog
            .toolbox_by_id("strix-halo-llama-rocm-10-0")
            .expect("known fixture toolbox id must resolve");
        assert_eq!(toolbox.backend, CatalogBackendId::LlamaCpp);
        assert_eq!(toolbox.supported_backend(), Some(SupportedServingBackend::LlamaCpp));

        let platform = catalog
            .platform_by_id("strix-halo")
            .expect("known fixture platform id must resolve");
        assert!(platform.toolbox_ids.contains(&"strix-halo-llama-rocm-10-0".to_string()));
        assert_eq!(
            platform.defaults.get("llama_cpp").map(String::as_str),
            Some("strix-halo-llama-rocm-10-0")
        );
    }
}
