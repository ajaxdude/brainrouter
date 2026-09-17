//! The vendored ai-toolbox-cockpit catalog (`assets/cockpit-catalog/`) and
//! the code that structurally validates it.
//!
//! **PR1 scope only.** This module currently exposes just the vendored raw
//! JSON (embedded at compile time) plus structural validation
//! ([`schema_validate`]) — enough for the sync script and a defensive
//! startup check to confirm the catalog still parses. The strongly-typed
//! `CatalogBackendId` / `SupportedServingBackend` / `ModelPayload` layer and
//! the generalized Toolboxes tab described in
//! `docs/design/ai-toolbox-cockpit-integration.md` §2 land in PR2, in this
//! same module.

pub mod schema_validate;

use serde_json::Value;

use schema_validate::ValidationReport;

/// Vendored, pinned snapshot of upstream's `toolboxes.json`. See
/// `assets/cockpit-catalog/SOURCE` for the exact upstream commit this was
/// synced from.
const VENDORED_TOOLBOXES_JSON: &str = include_str!("../assets/cockpit-catalog/toolboxes.json");
/// Vendored, pinned snapshot of upstream's `models.json`.
const VENDORED_MODELS_JSON: &str = include_str!("../assets/cockpit-catalog/models.json");

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
}
