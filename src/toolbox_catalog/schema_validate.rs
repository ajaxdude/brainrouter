//! Structural validation for the vendored ai-toolbox-cockpit catalog files
//! (`assets/cockpit-catalog/{toolboxes,models}.json`).
//!
//! This is intentionally **not** a byte-for-byte port of upstream's
//! `ai_toolbox_cockpit/catalog/schema.py`. Upstream's validator is *closed*:
//! it hard-fails (`CatalogError`) on any `backend` id outside its current
//! `BACKEND_IDS` set, and on any `models.json` backend key it doesn't have a
//! `MODEL_KINDS` entry for. brainrouter must keep loading the catalog when
//! upstream adds a new backend before brainrouter has code for it — see
//! `docs/design/ai-toolbox-cockpit-integration.md` requirement 1 ("ds4,
//! halogen, vllm, r9v, and future ones"). So this validator enforces the
//! same *structural* invariants upstream enforces (duplicate ids, dangling
//! references, required fields, closed enums for channel/maturity/feature
//! state, which are UI/lifecycle vocabulary rather than backend identity)
//! as hard errors, but demotes "backend id not yet known to brainrouter" to
//! a warning that the sync script/CI surfaces without failing the build.
//!
//! Reference (read-only, not vendored as code): upstream's
//! `ai_toolbox_cockpit/catalog/schema.py` as of the pinned commit recorded in
//! `assets/cockpit-catalog/SOURCE`.

use std::collections::BTreeSet;

use serde_json::Value;

/// Backend ids brainrouter/upstream currently know about. Mirrors upstream's
/// `BACKEND_IDS` in `catalog/schema.py`, plus brainrouter's own `gufo` overlay
/// backend (see `crate::toolbox_catalog::gufo_overlay`) so that validating the
/// *effective* (vendored + gufo) catalog does not warn on gufo. Used only to
/// decide whether an unrecognized `backend` id should be reported as a
/// warning — it is never used to reject an otherwise well-formed catalog entry.
pub const KNOWN_BACKEND_IDS: &[&str] =
    &["llama_cpp", "ds4", "halogen", "vllm", "r9v", "comfyui", "gufo"];

const FEATURE_IDS: &[&str] = &["interactive", "models", "server"];
const FEATURE_STATES: &[&str] = &["supported", "experimental", "unavailable"];
const CHANNELS: &[&str] = &["stable", "development", "experimental"];
const MATURITY_STATES: &[&str] = &["stable", "experimental"];

/// `models.json` uses `"bundles"` instead of `"models"` as the entries key
/// for exactly one backend (comfyui, per upstream `schema.py`). brainrouter
/// doesn't act on comfyui but still needs to parse it losslessly.
const MODEL_ENTRIES_KEY_COMFYUI: &str = "bundles";
const MODEL_ENTRIES_KEY_DEFAULT: &str = "models";

/// `toolboxes.json`'s expected `schema_version`, as of the pinned commit.
/// A mismatch is a hard error: it means upstream's *format*, not just its
/// content, has changed in a way this validator hasn't been updated for.
pub const TOOLBOXES_SCHEMA_VERSION: i64 = 3;
/// `models.json`'s expected `schema_version`, as of the pinned commit.
pub const MODELS_SCHEMA_VERSION: i64 = 2;

/// Result of validating one catalog file: hard structural errors (fail the
/// sync/CI) and soft warnings (e.g. an unrecognized backend id — surfaced,
/// never fail the build on their own).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationReport {
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    fn error(&mut self, message: impl Into<String>) {
        self.errors.push(message.into());
    }

    fn warning(&mut self, message: impl Into<String>) {
        self.warnings.push(message.into());
    }

    fn merge(&mut self, other: ValidationReport) {
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
    }
}

/// Reads a required string field, recording an error and returning `None` if
/// it is missing or not a string. Mirrors `_required_string()` in upstream's
/// `catalog/schema.py`.
fn required_string<'a>(
    report: &mut ValidationReport,
    object: &'a serde_json::Map<String, Value>,
    key: &str,
    context: &str,
) -> Option<&'a str> {
    match object.get(key).and_then(Value::as_str) {
        Some(value) if !value.is_empty() => Some(value),
        _ => {
            report.error(format!("{context}.{key} must be a non-empty string"));
            None
        }
    }
}

fn required_string_list(
    report: &mut ValidationReport,
    object: &serde_json::Map<String, Value>,
    key: &str,
    context: &str,
) -> Vec<String> {
    match object.get(key) {
        Some(Value::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            let mut ok = true;
            for item in items {
                match item.as_str() {
                    Some(s) => out.push(s.to_string()),
                    None => ok = false,
                }
            }
            if !ok {
                report.error(format!("{context}.{key} must contain only strings"));
            }
            out
        }
        Some(_) => {
            report.error(format!("{context}.{key} must be an array of strings"));
            Vec::new()
        }
        None => Vec::new(),
    }
}

/// Validates `assets/cockpit-catalog/toolboxes.json`'s structure. Mirrors
/// `ToolboxCatalog.from_dict()` in upstream `catalog/schema.py`, with the
/// backend-id-membership check demoted from error to warning (see module
/// docs).
pub fn validate_toolboxes_json(value: &Value) -> ValidationReport {
    let mut report = ValidationReport::default();

    let Some(root) = value.as_object() else {
        report.error("toolboxes.json root must be an object");
        return report;
    };

    match root.get("schema_version").and_then(Value::as_i64) {
        Some(v) if v == TOOLBOXES_SCHEMA_VERSION => {}
        Some(v) => report.error(format!(
            "toolboxes.json schema_version must be {TOOLBOXES_SCHEMA_VERSION}, got {v}"
        )),
        None => report.error("toolboxes.json schema_version must be an integer"),
    }

    // --- runtime_profiles ---
    let mut known_profiles: BTreeSet<String> = BTreeSet::new();
    match root.get("runtime_profiles").and_then(Value::as_object) {
        Some(profiles) => {
            for (profile_id, raw) in profiles {
                let context = format!("runtime_profiles.{profile_id}");
                let Some(profile_obj) = raw.as_object() else {
                    report.error(format!("{context} must be an object"));
                    continue;
                };
                required_string_list(&mut report, profile_obj, "engine_args", &context);
                known_profiles.insert(profile_id.clone());
            }
        }
        None => report.error("toolboxes.json runtime_profiles must be an object"),
    }

    // --- toolboxes[] ---
    let mut known_toolbox_ids: BTreeSet<String> = BTreeSet::new();
    let mut toolbox_backend_by_id: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();
    let mut container_names: BTreeSet<String> = BTreeSet::new();
    match root.get("toolboxes").and_then(Value::as_array) {
        Some(toolboxes) => {
            for (index, raw) in toolboxes.iter().enumerate() {
                let context = format!("toolboxes[{index}]");
                let Some(entry) = raw.as_object() else {
                    report.error(format!("{context} must be an object"));
                    continue;
                };

                let toolbox_id =
                    required_string(&mut report, entry, "id", &context).map(str::to_string);
                if let Some(id) = &toolbox_id {
                    if !known_toolbox_ids.insert(id.clone()) {
                        report.error(format!("duplicate toolbox id: {id}"));
                    }
                }

                if let Some(backend) = required_string(&mut report, entry, "backend", &context) {
                    if !KNOWN_BACKEND_IDS.contains(&backend) {
                        report.warning(format!(
                            "{context}: unrecognized backend id {backend:?} — brainrouter will \
                             preserve it losslessly (CatalogBackendId::Other) but cannot act on \
                             it until code is added for it"
                        ));
                    }
                    if let Some(id) = &toolbox_id {
                        toolbox_backend_by_id.insert(id.clone(), backend.to_string());
                    }
                }

                if let Some(image) = required_string(&mut report, entry, "image", &context) {
                    let has_registry_path = image.contains('/');
                    let has_tag_or_digest = image.contains(':') || image.contains('@');
                    if !has_registry_path || !has_tag_or_digest {
                        report.error(format!(
                            "{context}.image must be a complete tagged OCI reference, got {image:?}"
                        ));
                    }
                }

                if let Some(profile_id) =
                    required_string(&mut report, entry, "runtime_profile", &context)
                {
                    if !known_profiles.contains(profile_id) {
                        report.error(format!("{context}: unknown runtime_profile {profile_id:?}"));
                    }
                }

                match entry.get("features") {
                    Some(Value::Object(features)) => {
                        let feature_keys: BTreeSet<&str> =
                            features.keys().map(String::as_str).collect();
                        let expected: BTreeSet<&str> = FEATURE_IDS.iter().copied().collect();
                        if feature_keys != expected {
                            report.error(format!(
                                "{context}.features must declare exactly: {}",
                                FEATURE_IDS.join(", ")
                            ));
                        }
                        for (feature, state) in features {
                            match state.as_str() {
                                Some(s) if FEATURE_STATES.contains(&s) => {}
                                _ => report.error(format!(
                                    "{context}.features.{feature} must be one of: {}",
                                    FEATURE_STATES.join(", ")
                                )),
                            }
                        }
                    }
                    Some(_) => report.error(format!("{context}.features must be an object")),
                    None => report.error(format!("{context}.features is required")),
                }

                if let Some(container_name) =
                    required_string(&mut report, entry, "container_name", &context)
                {
                    if !container_names.insert(container_name.to_string()) {
                        report.error(format!("duplicate toolbox container_name: {container_name}"));
                    }
                }

                required_string(&mut report, entry, "name", &context);
                required_string(&mut report, entry, "group", &context);

                if let Some(channel) = required_string(&mut report, entry, "channel", &context) {
                    if !CHANNELS.contains(&channel) {
                        report.error(format!(
                            "{context}.channel must be one of: {}",
                            CHANNELS.join(", ")
                        ));
                    }
                }
                if let Some(maturity) = required_string(&mut report, entry, "maturity", &context) {
                    if !MATURITY_STATES.contains(&maturity) {
                        report.error(format!(
                            "{context}.maturity must be one of: {}",
                            MATURITY_STATES.join(", ")
                        ));
                    }
                }

                if let Some(backend_config) = entry.get("backend_config") {
                    if !backend_config.is_object() {
                        report.error(format!("{context}.backend_config must be an object"));
                    }
                }
            }
        }
        None => report.error("toolboxes.json toolboxes must be an array"),
    }

    // --- platforms[] ---
    match root.get("platforms").and_then(Value::as_array) {
        Some(platforms) => {
            let mut platform_ids: BTreeSet<String> = BTreeSet::new();
            let mut assigned_toolboxes: std::collections::BTreeMap<String, String> =
                std::collections::BTreeMap::new();
            for (index, raw) in platforms.iter().enumerate() {
                let context = format!("platforms[{index}]");
                let Some(entry) = raw.as_object() else {
                    report.error(format!("{context} must be an object"));
                    continue;
                };
                let platform_id =
                    required_string(&mut report, entry, "id", &context).map(str::to_string);
                if let Some(id) = &platform_id {
                    if !platform_ids.insert(id.clone()) {
                        report.error(format!("duplicate platform id: {id}"));
                    }
                }
                required_string(&mut report, entry, "name", &context);

                let toolbox_ids = required_string_list(&mut report, entry, "toolbox_ids", &context);
                let mut missing = Vec::new();
                for tid in &toolbox_ids {
                    if !known_toolbox_ids.contains(tid) {
                        missing.push(tid.clone());
                    }
                }
                if !missing.is_empty() {
                    report.error(format!(
                        "{context} references missing toolboxes: {}",
                        missing.join(", ")
                    ));
                }
                if let Some(id) = &platform_id {
                    for tid in &toolbox_ids {
                        if let Some(previous) = assigned_toolboxes.get(tid) {
                            report.error(format!(
                                "toolbox {tid:?} is assigned to both {previous:?} and {id:?}"
                            ));
                        } else {
                            assigned_toolboxes.insert(tid.clone(), id.clone());
                        }
                    }
                }

                match entry.get("defaults") {
                    Some(Value::Object(defaults)) => {
                        for (backend, default_toolbox_id) in defaults {
                            let Some(default_toolbox_id) = default_toolbox_id.as_str() else {
                                report.error(format!(
                                    "{context}.defaults.{backend} must be a string toolbox id"
                                ));
                                continue;
                            };
                            if !KNOWN_BACKEND_IDS.contains(&backend.as_str()) {
                                report.warning(format!(
                                    "{context}.defaults: unrecognized backend id {backend:?}"
                                ));
                            }
                            if !toolbox_ids.iter().any(|t| t == default_toolbox_id) {
                                report.error(format!(
                                    "{context}: default toolbox {default_toolbox_id:?} is not \
                                     assigned to this platform"
                                ));
                            } else if let Some(actual_backend) =
                                toolbox_backend_by_id.get(default_toolbox_id)
                            {
                                if actual_backend != backend {
                                    report.error(format!(
                                        "{context}: default toolbox {default_toolbox_id:?} does \
                                         not use backend {backend:?}"
                                    ));
                                }
                            }
                        }
                    }
                    Some(_) => report.error(format!("{context}.defaults must be an object")),
                    None => {}
                }
            }

            let unassigned: Vec<&String> = known_toolbox_ids
                .iter()
                .filter(|id| !assigned_toolboxes.contains_key(*id))
                .collect();
            if !unassigned.is_empty() {
                let ids: Vec<&str> = unassigned.iter().map(|s| s.as_str()).collect();
                report.error(format!(
                    "toolboxes are not assigned to a platform: {}",
                    ids.join(", ")
                ));
            }
        }
        None => report.error("toolboxes.json platforms must be an array"),
    }

    report
}

/// Validates `assets/cockpit-catalog/models.json`'s structure. Mirrors
/// `ModelCatalog.from_dict()` in upstream `catalog/schema.py`, with the
/// backend-id-membership check demoted from error to warning, and the
/// per-backend `MODEL_KINDS` closed mapping dropped entirely (upstream uses
/// it to hard-fail on any backend it doesn't recognize at all — brainrouter
/// instead just requires `kind` to be a non-empty string and leaves stronger
/// per-backend typing to `src/toolbox_catalog.rs`'s `ModelPayload`, added in
/// PR2).
pub fn validate_models_json(value: &Value) -> ValidationReport {
    let mut report = ValidationReport::default();

    let Some(root) = value.as_object() else {
        report.error("models.json root must be an object");
        return report;
    };

    match root.get("schema_version").and_then(Value::as_i64) {
        Some(v) if v == MODELS_SCHEMA_VERSION => {}
        Some(v) => report.error(format!(
            "models.json schema_version must be {MODELS_SCHEMA_VERSION}, got {v}"
        )),
        None => report.error("models.json schema_version must be an integer"),
    }

    match root.get("backends").and_then(Value::as_object) {
        Some(backends) => {
            for (backend_id, raw) in backends {
                let context = format!("backends.{backend_id}");
                if !KNOWN_BACKEND_IDS.contains(&backend_id.as_str()) {
                    report.warning(format!(
                        "{context}: unrecognized backend id — brainrouter will preserve its \
                         entries losslessly but cannot act on them until code is added for it"
                    ));
                }
                let Some(entry) = raw.as_object() else {
                    report.error(format!("{context} must be an object"));
                    continue;
                };

                required_string(&mut report, entry, "kind", &context);

                match entry.get("storage") {
                    Some(Value::Object(storage)) => {
                        required_string(
                            &mut report,
                            storage,
                            "config_key",
                            &format!("{context}.storage"),
                        );
                        required_string(&mut report, storage, "default", &format!("{context}.storage"));
                    }
                    Some(_) => report.error(format!("{context}.storage must be an object")),
                    None => report.error(format!("{context}.storage is required")),
                }

                if let Some(config) = entry.get("config") {
                    if !config.is_object() {
                        report.error(format!("{context}.config must be an object"));
                    }
                }

                let entries_key = if backend_id == "comfyui" {
                    MODEL_ENTRIES_KEY_COMFYUI
                } else {
                    MODEL_ENTRIES_KEY_DEFAULT
                };
                match entry.get(entries_key).and_then(Value::as_array) {
                    Some(entries) => {
                        let mut seen_ids: BTreeSet<String> = BTreeSet::new();
                        for (index, model_entry) in entries.iter().enumerate() {
                            let entry_context = format!("{context}.{entries_key}[{index}]");
                            let Some(model_obj) = model_entry.as_object() else {
                                report.error(format!("{entry_context} must be an object"));
                                continue;
                            };
                            required_string(&mut report, model_obj, "name", &entry_context);
                            if let Some(id) = model_obj.get("id").and_then(Value::as_str) {
                                if !seen_ids.insert(id.to_string()) {
                                    report.error(format!(
                                        "duplicate {backend_id} model/bundle id: {id}"
                                    ));
                                }
                            }
                        }
                    }
                    None => {
                        report.error(format!("{context}.{entries_key} must be an array of objects"))
                    }
                }
            }
        }
        None => report.error("models.json backends must be an object"),
    }

    report
}

/// Validates both catalog files together and merges the two reports, for
/// callers (the sync script binary, and defensive runtime load) that only
/// care about "did the whole catalog load cleanly".
pub fn validate_catalog(toolboxes_json: &Value, models_json: &Value) -> ValidationReport {
    let mut report = validate_toolboxes_json(toolboxes_json);
    report.merge(validate_models_json(models_json));
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vendored_toolboxes() -> Value {
        serde_json::from_str(include_str!("../../assets/cockpit-catalog/toolboxes.json"))
            .expect("vendored toolboxes.json must be valid JSON")
    }

    fn vendored_models() -> Value {
        serde_json::from_str(include_str!("../../assets/cockpit-catalog/models.json"))
            .expect("vendored models.json must be valid JSON")
    }

    #[test]
    fn vendored_toolboxes_json_is_structurally_valid() {
        let report = validate_toolboxes_json(&vendored_toolboxes());
        assert!(
            report.is_ok(),
            "expected no structural errors, got: {:#?}",
            report.errors
        );
        // The vendored snapshot's own backend ids must all be known ones as
        // of the pinned commit — a stray warning here would mean SOURCE's
        // pinned-commit bookkeeping and KNOWN_BACKEND_IDS have drifted.
        assert!(
            report.warnings.is_empty(),
            "unexpected warnings against vendored data: {:#?}",
            report.warnings
        );
    }

    #[test]
    fn vendored_models_json_is_structurally_valid() {
        let report = validate_models_json(&vendored_models());
        assert!(
            report.is_ok(),
            "expected no structural errors, got: {:#?}",
            report.errors
        );
        assert!(
            report.warnings.is_empty(),
            "unexpected warnings against vendored data: {:#?}",
            report.warnings
        );
    }

    #[test]
    fn unrecognized_backend_id_in_toolboxes_is_a_warning_not_an_error() {
        let mut doc = vendored_toolboxes();
        let root = doc.as_object_mut().unwrap();
        let toolboxes = root.get_mut("toolboxes").unwrap().as_array_mut().unwrap();
        // Mutate a toolbox entry's backend to a made-up id that doesn't
        // exist yet anywhere in this codebase or upstream, proving catalog
        // loading stays lossless (warning) rather than failing outright
        // (error) — this is the direct test for requirement 1 ("future
        // backends" must not break catalog loading). Picks a toolbox that
        // is not any platform's `defaults` entry, so the mutation doesn't
        // also (correctly) trip the unrelated "default toolbox does not use
        // backend X" check.
        let index = toolboxes
            .iter()
            .position(|t| t["id"] == "strix-halo-llama-hrx-staging")
            .expect("fixture toolbox id must exist in the vendored catalog");
        toolboxes[index]["backend"] = Value::String("brand_new_future_backend".to_string());

        let report = validate_toolboxes_json(&doc);
        assert!(
            report.is_ok(),
            "an unrecognized backend id must not be a structural error: {:#?}",
            report.errors
        );
        assert!(
            report
                .warnings
                .iter()
                .any(|w| w.contains("brand_new_future_backend")),
            "expected a warning naming the unrecognized backend id, got: {:#?}",
            report.warnings
        );
    }

    #[test]
    fn unrecognized_backend_id_in_models_is_a_warning_not_an_error() {
        let mut doc = vendored_models();
        let root = doc.as_object_mut().unwrap();
        let backends = root.get_mut("backends").unwrap().as_object_mut().unwrap();
        let ds4 = backends.remove("ds4").unwrap();
        backends.insert("brand_new_future_backend".to_string(), ds4);

        let report = validate_models_json(&doc);
        assert!(
            report.is_ok(),
            "an unrecognized backend id must not be a structural error: {:#?}",
            report.errors
        );
        assert!(
            report
                .warnings
                .iter()
                .any(|w| w.contains("brand_new_future_backend")),
            "expected a warning naming the unrecognized backend id, got: {:#?}",
            report.warnings
        );
    }

    #[test]
    fn missing_schema_version_is_a_structural_error() {
        let mut doc = vendored_toolboxes();
        doc.as_object_mut().unwrap().remove("schema_version");
        let report = validate_toolboxes_json(&doc);
        assert!(!report.is_ok());
        assert!(report.errors.iter().any(|e| e.contains("schema_version")));
    }

    #[test]
    fn duplicate_toolbox_id_is_a_structural_error() {
        let mut doc = vendored_toolboxes();
        let root = doc.as_object_mut().unwrap();
        let toolboxes = root.get_mut("toolboxes").unwrap().as_array_mut().unwrap();
        let first_id = toolboxes[0]["id"].clone();
        toolboxes[1]["id"] = first_id;
        let report = validate_toolboxes_json(&doc);
        assert!(!report.is_ok());
        assert!(report.errors.iter().any(|e| e.contains("duplicate toolbox id")));
    }

    #[test]
    fn dangling_runtime_profile_reference_is_a_structural_error() {
        let mut doc = vendored_toolboxes();
        let root = doc.as_object_mut().unwrap();
        let toolboxes = root.get_mut("toolboxes").unwrap().as_array_mut().unwrap();
        toolboxes[0]["runtime_profile"] = Value::String("no-such-profile".to_string());
        let report = validate_toolboxes_json(&doc);
        assert!(!report.is_ok());
        assert!(report
            .errors
            .iter()
            .any(|e| e.contains("unknown runtime_profile")));
    }

    #[test]
    fn duplicate_model_id_within_a_backend_is_a_structural_error() {
        let mut doc = vendored_models();
        let root = doc.as_object_mut().unwrap();
        let backends = root.get_mut("backends").unwrap().as_object_mut().unwrap();
        let ds4 = backends.get_mut("ds4").unwrap().as_object_mut().unwrap();
        let models = ds4.get_mut("models").unwrap().as_array_mut().unwrap();
        assert!(models.len() >= 2, "fixture needs at least 2 models");
        let first_id = models[0]["id"].clone();
        models[1]["id"] = first_id;
        let report = validate_models_json(&doc);
        assert!(!report.is_ok());
        assert!(report
            .errors
            .iter()
            .any(|e| e.contains("duplicate ds4 model/bundle id")));
    }

    #[test]
    fn validate_catalog_merges_both_reports() {
        let report = validate_catalog(&vendored_toolboxes(), &vendored_models());
        assert!(report.is_ok(), "{:#?}", report.errors);
    }
}
