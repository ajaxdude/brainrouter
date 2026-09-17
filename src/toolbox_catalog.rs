//! The vendored ai-toolbox-cockpit catalog (`assets/cockpit-catalog/`), its
//! structural validator, and the strongly-typed parsed representation.
//!
//! - [`schema_validate`] mirrors upstream's *structural* invariants as
//!   warnings/errors (permissive on unrecognized backend ids).
//! - [`types`] / [`models`] are the typed layer: [`CatalogBackendId`],
//!   [`SupportedServingBackend`], [`ToolboxCatalog`], [`ModelCatalog`],
//!   [`ModelPayload`] etc. — see `docs/design/ai-toolbox-cockpit-integration.md`
//!   §2 for the design rationale.
//!
//! Per-backend command-builder submodules (the actual `toolbox`/`podman`
//! invocation sequences for ds4/halogen/vllm/r9v) are Wave 2 territory
//! (PR7-PR11) and don't exist yet — this module is catalog data only.

pub mod models;
pub mod schema_validate;
pub mod types;

use serde_json::Value;

use schema_validate::ValidationReport;

pub use models::{
    CatalogModelEntry, Ds4Model, HalogenModel, LlamaCppModel, ModelBackendCatalog, ModelCatalog,
    ModelPayload, R9vModel, VllmModel,
};
pub use types::{
    CatalogBackendId, CatalogParseError, Channel, FeatureState, Maturity, Platform,
    RuntimeProfile, SupportedServingBackend, ToolboxCatalog, ToolboxDefinition, ToolboxFeatures,
    UnsupportedBackend,
};


/// Vendored, pinned snapshot of upstream's `toolboxes.json`. See
/// `assets/cockpit-catalog/SOURCE` for the exact upstream commit this was
/// synced from.
const VENDORED_TOOLBOXES_JSON: &str = include_str!("../assets/cockpit-catalog/toolboxes.json");
/// Vendored, pinned snapshot of upstream's `models.json`.
const VENDORED_MODELS_JSON: &str = include_str!("../assets/cockpit-catalog/models.json");

/// A short, stable identifier for exactly which vendored catalog content is
/// embedded in this binary — a content hash, not a database row. Used as the
/// `io.brainrouter.catalog_revision` container label (§5c): since PR2
/// deliberately defers `toolbox_catalog_snapshot` (no DB row to reference),
/// this is the cheapest thing that still lets a future reconciliation pass
/// notice "this container was created from an older catalog than what's
/// embedded now" without any persisted state at all.
pub fn vendored_catalog_revision() -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(VENDORED_TOOLBOXES_JSON.as_bytes());
    hasher.update(VENDORED_MODELS_JSON.as_bytes());
    let digest = hasher.finalize();
    // Short (12 hex chars, ~48 bits) is plenty for a label value that only
    // ever needs to answer "same or different from what's embedded now?",
    // mirroring git's convention of short-SHA-as-good-enough-identifier.
    format!("{:x}", digest).chars().take(12).collect()
}

/// Result of loading the vendored catalog at runtime: the parsed JSON for
/// each file plus the structural validation report. Parsing the vendored
/// files themselves is not expected to ever fail (they're checked in CI and
/// by the unit tests in [`schema_validate`]), so this never panics — it
/// treats a parse failure as just another validation error, in case the
/// embedded files are ever hand-edited or corrupted.
pub struct VendoredCatalog {
    pub toolboxes_json: Option<Value>,
    pub models_json: Option<Value>,
    pub report: ValidationReport,
}

impl VendoredCatalog {
    /// Number of toolbox entries, if `toolboxes.json` parsed at all.
    pub fn toolbox_count(&self) -> Option<usize> {
        self.toolboxes_json
            .as_ref()?
            .get("toolboxes")?
            .as_array()
            .map(Vec::len)
    }

    /// Number of distinct model backends declared in `models.json`, if it
    /// parsed at all.
    pub fn model_backend_count(&self) -> Option<usize> {
        self.models_json
            .as_ref()?
            .get("backends")?
            .as_object()
            .map(|m| m.len())
    }

    /// Parses both files into the strongly-typed [`ToolboxCatalog`] /
    /// [`ModelCatalog`] layer. Separate from the raw JSON load above because
    /// typed parsing can fail in ways raw JSON parsing can't (e.g. a field
    /// with the wrong type) — callers that only need diagnostics (the
    /// startup check, the sync script) can use the untyped fields instead.
    pub fn typed(&self) -> Result<(ToolboxCatalog, ModelCatalog), CatalogParseError> {
        let toolboxes = self
            .toolboxes_json
            .as_ref()
            .ok_or_else(|| CatalogParseError("toolboxes.json did not parse as JSON".to_string()))?;
        let models = self
            .models_json
            .as_ref()
            .ok_or_else(|| CatalogParseError("models.json did not parse as JSON".to_string()))?;
        Ok((ToolboxCatalog::parse(toolboxes)?, ModelCatalog::parse(models)?))
    }
}

/// Loads and structurally validates the two catalog files embedded in the
/// binary at compile time. Called defensively at daemon startup (log-only,
/// never fatal — see `src/daemon.rs`) and by the sync script's validation
/// binary (`src/bin/toolbox_catalog_check.rs`) against freshly fetched
/// candidate files instead, via [`schema_validate::validate_catalog`]
/// directly.
pub fn load_vendored_catalog() -> VendoredCatalog {
    let toolboxes_json: Result<Value, _> = serde_json::from_str(VENDORED_TOOLBOXES_JSON);
    let models_json: Result<Value, _> = serde_json::from_str(VENDORED_MODELS_JSON);

    let mut report = ValidationReport::default();
    let toolboxes_value = match toolboxes_json {
        Ok(value) => Some(value),
        Err(error) => {
            report
                .errors
                .push(format!("vendored toolboxes.json failed to parse: {error}"));
            None
        }
    };
    let models_value = match models_json {
        Ok(value) => Some(value),
        Err(error) => {
            report
                .errors
                .push(format!("vendored models.json failed to parse: {error}"));
            None
        }
    };

    if let (Some(toolboxes), Some(models)) = (&toolboxes_value, &models_value) {
        let validation = schema_validate::validate_catalog(toolboxes, models);
        report.errors.extend(validation.errors);
        report.warnings.extend(validation.warnings);
    }

    VendoredCatalog {
        toolboxes_json: toolboxes_value,
        models_json: models_value,
        report,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendored_catalog_loads_and_validates_cleanly() {
        let catalog = load_vendored_catalog();
        assert!(
            catalog.report.is_ok(),
            "vendored catalog has structural errors: {:#?}",
            catalog.report.errors
        );
        assert!(catalog.toolbox_count().unwrap_or(0) > 0);
        assert!(catalog.model_backend_count().unwrap_or(0) > 0);
    }

    #[test]
    fn vendored_catalog_parses_into_the_typed_layer() {
        let catalog = load_vendored_catalog();
        let (toolboxes, models) = catalog.typed().expect("typed parse must succeed");
        assert!(!toolboxes.toolboxes.is_empty());
        for backend in SupportedServingBackend::ALL {
            assert!(models.backend(backend).is_some(), "expected a {backend} models section");
        }
    }

    #[test]
    fn vendored_catalog_revision_is_stable_and_a_short_hex_string() {
        let a = vendored_catalog_revision();
        let b = vendored_catalog_revision();
        assert_eq!(a, b);
        assert_eq!(a.len(), 12);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
