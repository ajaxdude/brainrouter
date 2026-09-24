//! The brainrouter-owned `gufo` catalog overlay (design doc DI-2/DI-3).
//!
//! gufo is a 6th serving backend that is **not** in the vendored upstream
//! cockpit catalog (`assets/cockpit-catalog/`). Rather than editing those
//! byte-for-byte vendored files (which the weekly sync would clobber — see
//! `assets/cockpit-catalog/SOURCE`), this module keeps gufo's toolbox, runtime
//! profile, `strix-halo` platform attachment, and model entries in a separate
//! brainrouter-owned overlay (`assets/gufo-catalog/`) and merges them into a
//! *copy* of the vendored catalog at load time via
//! [`crate::toolbox_catalog::load_effective_typed_catalog`]. The vendored feed,
//! its validation, and its provenance hashes stay pure.

use serde_json::Value;

/// Brainrouter-owned gufo toolbox/runtime-profile overlay, embedded at compile
/// time and merged into a copy of the vendored `toolboxes.json`.
const GUFO_TOOLBOXES_JSON: &str = include_str!("../../assets/gufo-catalog/toolboxes.json");
/// Brainrouter-owned gufo model overlay, embedded at compile time and merged
/// into a copy of the vendored `models.json`.
const GUFO_MODELS_JSON: &str = include_str!("../../assets/gufo-catalog/models.json");

/// The catalog backend id this overlay owns.
pub const GUFO_BACKEND_ID: &str = "gufo";
/// The single gufo toolbox id this overlay injects.
pub const GUFO_TOOLBOX_ID: &str = "strix-halo-gufo-runtime";

fn parse_overlay(raw: &str, what: &str) -> Result<Value, String> {
    serde_json::from_str(raw).map_err(|e| format!("gufo overlay {what} is not valid JSON: {e}"))
}

/// Merges the gufo overlay into *copies* of the vendored `toolboxes`/`models`
/// documents, returning the merged pair. Transactional: it mutates local clones
/// and only returns them on full success, so a partial merge is never observed
/// by a caller. Returns `Err` (never a silent partial merge) if a required base
/// structure or the attachment platform is missing, or if any gufo id already
/// exists in the vendored base (every duplicate is an error).
pub fn merge(toolboxes: &Value, models: &Value) -> Result<(Value, Value), String> {
    let mut merged_toolboxes = toolboxes.clone();
    let mut merged_models = models.clone();

    let overlay_tb = parse_overlay(GUFO_TOOLBOXES_JSON, "toolboxes.json")?;
    let overlay_models = parse_overlay(GUFO_MODELS_JSON, "models.json")?;

    // --- runtime_profiles: insert gufo's profile(s) ---
    let base_profiles = merged_toolboxes
        .get_mut("runtime_profiles")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| "vendored toolboxes.json has no `runtime_profiles` object".to_string())?;
    if let Some(overlay_profiles) = overlay_tb.get("runtime_profiles").and_then(Value::as_object) {
        for (id, profile) in overlay_profiles {
            if base_profiles.contains_key(id) {
                return Err(format!(
                    "gufo overlay runtime_profile `{id}` already exists in the vendored catalog"
                ));
            }
            base_profiles.insert(id.clone(), profile.clone());
        }
    }

    // --- toolboxes[]: append gufo's toolbox(es) ---
    let base_toolboxes = merged_toolboxes
        .get_mut("toolboxes")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "vendored toolboxes.json has no `toolboxes` array".to_string())?;
    let existing_toolbox_ids: std::collections::BTreeSet<String> = base_toolboxes
        .iter()
        .filter_map(|t| t.get("id").and_then(Value::as_str).map(str::to_string))
        .collect();
    if let Some(overlay_list) = overlay_tb.get("toolboxes").and_then(Value::as_array) {
        for tb in overlay_list {
            if let Some(id) = tb.get("id").and_then(Value::as_str) {
                if existing_toolbox_ids.contains(id) {
                    return Err(format!(
                        "gufo overlay toolbox id `{id}` already exists in the vendored catalog"
                    ));
                }
            }
            base_toolboxes.push(tb.clone());
        }
    }

    // --- platform attachment: add the gufo toolbox to the named platform ---
    let attach_platform = overlay_tb
        .get("attach_platform")
        .and_then(Value::as_str)
        .ok_or_else(|| "gufo overlay is missing `attach_platform`".to_string())?
        .to_string();
    let platforms = merged_toolboxes
        .get_mut("platforms")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "vendored toolboxes.json has no `platforms` array".to_string())?;
    let platform = platforms
        .iter_mut()
        .find(|p| p.get("id").and_then(Value::as_str) == Some(attach_platform.as_str()))
        .ok_or_else(|| {
            format!("vendored toolboxes.json has no `{attach_platform}` platform to attach gufo to")
        })?;
    let platform_obj = platform
        .as_object_mut()
        .ok_or_else(|| format!("platform `{attach_platform}` is not an object"))?;
    let toolbox_ids = platform_obj
        .entry("toolbox_ids")
        .or_insert_with(|| Value::Array(Vec::new()))
        .as_array_mut()
        .ok_or_else(|| format!("platform `{attach_platform}`.toolbox_ids is not an array"))?;
    if toolbox_ids.iter().any(|v| v.as_str() == Some(GUFO_TOOLBOX_ID)) {
        return Err(format!(
            "gufo toolbox `{GUFO_TOOLBOX_ID}` is already attached to `{attach_platform}`"
        ));
    }
    toolbox_ids.push(Value::String(GUFO_TOOLBOX_ID.to_string()));
    let defaults = platform_obj
        .entry("defaults")
        .or_insert_with(|| Value::Object(serde_json::Map::new()))
        .as_object_mut()
        .ok_or_else(|| format!("platform `{attach_platform}`.defaults is not an object"))?;
    if defaults.contains_key(GUFO_BACKEND_ID) {
        return Err(format!(
            "platform `{attach_platform}`.defaults already has a `{GUFO_BACKEND_ID}` entry"
        ));
    }
    defaults.insert(
        GUFO_BACKEND_ID.to_string(),
        Value::String(GUFO_TOOLBOX_ID.to_string()),
    );

    // --- models.backends.gufo ---
    let base_backends = merged_models
        .get_mut("backends")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| "vendored models.json has no `backends` object".to_string())?;
    if let Some(overlay_backends) = overlay_models.get("backends").and_then(Value::as_object) {
        for (backend_id, section) in overlay_backends {
            if base_backends.contains_key(backend_id) {
                return Err(format!(
                    "gufo overlay models backend `{backend_id}` already exists in the vendored catalog"
                ));
            }
            base_backends.insert(backend_id.clone(), section.clone());
        }
    }

    Ok((merged_toolboxes, merged_models))
}

/// Validates the gufo-specific semantic invariants on a *merged* catalog
/// (design doc DI-3, I4/round-4). Returns a list of error strings (empty ⇒ ok).
/// This is in addition to [`crate::toolbox_catalog::schema_validate::validate_catalog`],
/// which covers the structural invariants gufo shares with every backend.
pub fn validate_merged(toolboxes: &Value, models: &Value) -> Vec<String> {
    let mut errors = Vec::new();

    // The gufo toolbox exists exactly once, with a resolvable runtime_profile.
    let gufo_toolboxes: Vec<&Value> = toolboxes
        .get("toolboxes")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter(|t| t.get("id").and_then(Value::as_str) == Some(GUFO_TOOLBOX_ID))
                .collect()
        })
        .unwrap_or_default();
    match gufo_toolboxes.len() {
        1 => {
            let profile = gufo_toolboxes[0].get("runtime_profile").and_then(Value::as_str);
            let known = toolboxes
                .get("runtime_profiles")
                .and_then(Value::as_object)
                .and_then(|p| profile.map(|id| p.contains_key(id)))
                .unwrap_or(false);
            if !known {
                errors.push(format!(
                    "gufo toolbox references unknown runtime_profile {profile:?}"
                ));
            }
        }
        n => errors.push(format!(
            "expected exactly one gufo toolbox `{GUFO_TOOLBOX_ID}`, found {n}"
        )),
    }

    // gufo attached to strix-halo (toolbox_ids + defaults.gufo).
    let strix = toolboxes.get("platforms").and_then(Value::as_array).and_then(|ps| {
        ps.iter()
            .find(|p| p.get("id").and_then(Value::as_str) == Some("strix-halo"))
    });
    match strix {
        Some(p) => {
            let in_ids = p
                .get("toolbox_ids")
                .and_then(Value::as_array)
                .map(|a| a.iter().any(|v| v.as_str() == Some(GUFO_TOOLBOX_ID)))
                .unwrap_or(false);
            if !in_ids {
                errors.push(format!(
                    "gufo toolbox `{GUFO_TOOLBOX_ID}` is not in strix-halo.toolbox_ids"
                ));
            }
            let default_ok = p
                .get("defaults")
                .and_then(Value::as_object)
                .and_then(|d| d.get(GUFO_BACKEND_ID))
                .and_then(Value::as_str)
                == Some(GUFO_TOOLBOX_ID);
            if !default_ok {
                errors.push("strix-halo.defaults.gufo must point to the gufo toolbox".to_string());
            }
        }
        None => errors.push("strix-halo platform is missing from the merged catalog".to_string()),
    }

    // gufo models section exists, with storage + valid entries.
    let Some(gufo_section) = models
        .get("backends")
        .and_then(Value::as_object)
        .and_then(|b| b.get(GUFO_BACKEND_ID))
    else {
        errors.push("gufo models backend section is missing from the merged catalog".to_string());
        return errors;
    };
    let storage = gufo_section.get("storage").and_then(Value::as_object);
    if storage
        .and_then(|s| s.get("config_key"))
        .and_then(Value::as_str)
        .map(str::is_empty)
        .unwrap_or(true)
    {
        errors.push("gufo storage.config_key must be a non-empty string".to_string());
    }
    if storage
        .and_then(|s| s.get("default"))
        .and_then(Value::as_str)
        .map(str::is_empty)
        .unwrap_or(true)
    {
        errors.push("gufo storage.default must be a non-empty string".to_string());
    }

    let entries: Vec<Value> = gufo_section
        .get("models")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut ids = std::collections::BTreeSet::new();
    let mut has_main = false;
    for e in &entries {
        let id = e.get("id").and_then(Value::as_str).unwrap_or("");
        let name = e.get("name").and_then(Value::as_str).unwrap_or("");
        let repo = e.get("repo").and_then(Value::as_str).unwrap_or("");
        let revision = e.get("revision").and_then(Value::as_str).unwrap_or("");
        let role = e.get("role").and_then(Value::as_str).unwrap_or("");
        if id.is_empty() || name.is_empty() || repo.is_empty() || revision.is_empty() {
            errors.push(format!(
                "gufo model entry `{id}` must have non-empty id/name/repo/revision"
            ));
        }
        if !id.is_empty() && !ids.insert(id.to_string()) {
            errors.push(format!("duplicate gufo model id `{id}`"));
        }
        match e.get("files").and_then(Value::as_array) {
            Some(files) if files.len() == 1 => {
                let f = &files[0];
                if f.get("path").and_then(Value::as_str).map(str::is_empty).unwrap_or(true) {
                    errors.push(format!("gufo model `{id}` file has an empty path"));
                }
                if f.get("size_bytes").and_then(Value::as_u64).unwrap_or(0) == 0 {
                    errors.push(format!("gufo model `{id}` file has a non-positive size_bytes"));
                }
            }
            _ => errors.push(format!("gufo model `{id}` must declare exactly one file")),
        }
        if let Some(v) = e.get("ctx_default") {
            if v.as_u64().map(|n| n == 0).unwrap_or(true) {
                errors.push(format!("gufo model `{id}` ctx_default must be a positive integer"));
            }
        }
        if let Some(v) = e.get("sessions_default") {
            if v.as_u64().map(|n| n == 0).unwrap_or(true) {
                errors.push(format!("gufo model `{id}` sessions_default must be a positive integer"));
            }
        }
        if role == "main" {
            has_main = true;
        }
    }
    if !has_main {
        errors.push("gufo models section must contain at least one role=main entry".to_string());
    }

    // Speculative cross-references: only on main; the draft exists, is
    // role=draft, differs from the main, and declares no nested speculative.
    for e in &entries {
        let id = e.get("id").and_then(Value::as_str).unwrap_or("");
        let role = e.get("role").and_then(Value::as_str).unwrap_or("");
        let Some(spec) = e.get("speculative") else { continue };
        if role != "main" {
            errors.push(format!("gufo model `{id}` declares `speculative` but is not role=main"));
            continue;
        }
        let mode = spec.get("mode").and_then(Value::as_str).unwrap_or("");
        if mode != "dflash2" {
            errors.push(format!(
                "gufo model `{id}` speculative.mode must be `dflash2` (v1), got {mode:?}"
            ));
        }
        let draft_id = spec.get("draft_model_id").and_then(Value::as_str).unwrap_or("");
        if draft_id == id {
            errors.push(format!(
                "gufo model `{id}` speculative.draft_model_id must not reference itself"
            ));
            continue;
        }
        match entries
            .iter()
            .find(|d| d.get("id").and_then(Value::as_str) == Some(draft_id))
        {
            None => errors.push(format!("gufo model `{id}` references missing draft `{draft_id}`")),
            Some(draft) => {
                if draft.get("role").and_then(Value::as_str) != Some("draft") {
                    errors.push(format!(
                        "gufo draft `{draft_id}` (referenced by `{id}`) must have role=draft"
                    ));
                }
                if draft.get("speculative").is_some() {
                    errors.push(format!(
                        "gufo draft `{draft_id}` must not itself declare `speculative`"
                    ));
                }
            }
        }
    }

    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vendored() -> (Value, Value) {
        let tb: Value = serde_json::from_str(include_str!(
            "../../assets/cockpit-catalog/toolboxes.json"
        ))
        .unwrap();
        let models: Value =
            serde_json::from_str(include_str!("../../assets/cockpit-catalog/models.json")).unwrap();
        (tb, models)
    }

    #[test]
    fn merge_injects_gufo_and_validates_clean() {
        let (tb, models) = vendored();
        let (mt, mm) = merge(&tb, &models).expect("merge must succeed");

        // gufo toolbox present.
        assert!(mt["toolboxes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["id"] == "strix-halo-gufo-runtime" && t["backend"] == "gufo"));
        // runtime profile present.
        assert!(mt["runtime_profiles"]
            .as_object()
            .unwrap()
            .contains_key("strix-halo-gufo-rocm"));
        // attached to strix-halo.
        let strix = mt["platforms"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["id"] == "strix-halo")
            .unwrap();
        assert!(strix["toolbox_ids"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "strix-halo-gufo-runtime"));
        assert_eq!(strix["defaults"]["gufo"], "strix-halo-gufo-runtime");
        // gufo models section present.
        assert!(mm["backends"].as_object().unwrap().contains_key("gufo"));
        // Both the structural validator and gufo semantics pass.
        assert!(validate_merged(&mt, &mm).is_empty(), "{:?}", validate_merged(&mt, &mm));

        // The merged toolboxes must NOT leak the overlay-only `attach_platform`.
        assert!(mt.get("attach_platform").is_none());
        // The vendored inputs were not mutated (transactional).
        assert!(!tb["toolboxes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["id"] == "strix-halo-gufo-runtime"));
    }

    #[test]
    fn merge_rejects_a_duplicate_gufo_id() {
        let (mut tb, models) = vendored();
        // Pre-insert a toolbox with the gufo id → merge must error.
        tb["toolboxes"].as_array_mut().unwrap().push(serde_json::json!({
            "id": "strix-halo-gufo-runtime", "backend": "gufo", "name": "dup",
            "container_name": "dup", "group": "g", "image": "x/y:z",
            "channel": "stable", "maturity": "experimental", "runtime_profile": "p",
            "features": {"interactive":"unavailable","models":"supported","server":"experimental"}
        }));
        let err = merge(&tb, &models).unwrap_err();
        assert!(err.contains("already exists"), "{err}");
    }

    #[test]
    fn merge_errors_without_the_target_platform() {
        let (mut tb, models) = vendored();
        // Remove all platforms → the strix-halo attachment can't happen.
        tb["platforms"] = Value::Array(Vec::new());
        let err = merge(&tb, &models).unwrap_err();
        assert!(err.contains("strix-halo"), "{err}");
    }

    #[test]
    fn validate_merged_catches_a_bad_draft_reference() {
        let (tb, models) = vendored();
        let (mt, mut mm) = merge(&tb, &models).unwrap();
        // Point a main model's draft ref at a non-existent id.
        let entries = mm["backends"]["gufo"]["models"].as_array_mut().unwrap();
        for e in entries.iter_mut() {
            if e["role"] == "main" {
                e["speculative"]["draft_model_id"] = Value::String("nope".to_string());
                break;
            }
        }
        let errs = validate_merged(&mt, &mm);
        assert!(errs.iter().any(|e| e.contains("missing draft `nope`")), "{errs:?}");
    }

    #[test]
    fn validate_merged_rejects_a_non_dflash2_mode() {
        let (tb, models) = vendored();
        let (mt, mut mm) = merge(&tb, &models).unwrap();
        let entries = mm["backends"]["gufo"]["models"].as_array_mut().unwrap();
        for e in entries.iter_mut() {
            if e["role"] == "main" {
                e["speculative"]["mode"] = Value::String("mtp".to_string());
                break;
            }
        }
        let errs = validate_merged(&mt, &mm);
        assert!(errs.iter().any(|e| e.contains("must be `dflash2`")), "{errs:?}");
    }

    #[test]
    fn validate_merged_requires_exactly_one_file() {
        let (tb, models) = vendored();
        let (mt, mut mm) = merge(&tb, &models).unwrap();
        let entries = mm["backends"]["gufo"]["models"].as_array_mut().unwrap();
        entries[0]["files"] = Value::Array(Vec::new());
        let errs = validate_merged(&mt, &mm);
        assert!(errs.iter().any(|e| e.contains("exactly one file")), "{errs:?}");
    }
}
