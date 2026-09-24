//! Strongly-typed representation of `assets/cockpit-catalog/models.json`.
//!
//! Unlike `toolboxes.json`, the five supported backends' model entries are
//! **structurally incompatible** with each other (confirmed by inspecting
//! every entry in the vendored fixture, not just the first one per
//! backend — see field-commonality notes on each struct below). There's no
//! single `ModelEntry` shape that fits all of them, hence [`ModelPayload`]:
//! one variant per supported backend, holding that backend's fields.
//!
//! Each per-backend struct types the fields that are present on *every*
//! entry for that backend in the real vendored data, and captures everything
//! else via `#[serde(flatten)] extra`, so nothing is lost even for the more
//! elaborate optional structures (llama_cpp's `inference_profiles`/`dspark`/
//! `mtp`/`auxiliary_downloads`, vllm's `extra_flags`/`env`, etc.) that this
//! PR doesn't need to act on yet. This mirrors `ToolboxDefinition`'s
//! `backend_config: Option<Value>` — strongly type what's proven common,
//! keep the long tail loose rather than guessing at shapes.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::types::{CatalogBackendId, CatalogParseError, SupportedServingBackend};

/// `llama_cpp` model entry. Common across all 30 vendored entries: `id`,
/// `name`, `repo`. Everything else (`toolbox_defaults`, `inference_profiles`,
/// `default_inference_profile`, `dspark`, `mtp`, `vision_projector`,
/// `auxiliary_downloads`, `compatible_toolboxes`, `no_jinja`) is optional and
/// backend-specific per entry; kept in `extra`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlamaCppModel {
    pub id: String,
    pub name: String,
    pub repo: String,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

/// `ds4` model entry. Common across all 19 vendored entries: `id`, `name`,
/// `repo`, `family`, `filename`, `size_gb`, `recommended`. `server_defaults`,
/// `sha256`, `artifact_role` are optional/backend-specific; kept in `extra`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ds4Model {
    pub id: String,
    pub name: String,
    pub repo: String,
    pub family: String,
    pub filename: String,
    pub size_gb: f64,
    #[serde(default)]
    pub recommended: bool,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

/// A single file within a `halogen` or `r9v` model's `files[]` manifest.
/// r9v additionally carries `role` and `sha256`; halogen's is just
/// `path`/`size_bytes`, so those two extra fields are optional here and
/// simply absent for halogen entries.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CatalogModelFile {
    pub path: String,
    pub size_bytes: u64,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub sha256: Option<String>,
}

/// `halogen` model entry ("HGN bundle"). Common across all 4 vendored
/// entries: `id`, `name`, `repo`, `revision`, `quant`, `checkpoint`,
/// `overlay`, `tokenizer_dir`, `recommended`, `files`. `vision_tower` is
/// optional; kept in `extra`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HalogenModel {
    pub id: String,
    pub name: String,
    pub repo: String,
    pub revision: String,
    pub quant: String,
    pub checkpoint: String,
    pub overlay: String,
    pub tokenizer_dir: String,
    #[serde(default)]
    pub recommended: bool,
    pub files: Vec<CatalogModelFile>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

/// `vllm` model entry (HF-repo-based). Common across all 15 vendored
/// entries: `id`, `name`, `repo`, `trust_remote`, `valid_tp`, `max_num_seqs`,
/// `max_tokens`. Note `max_num_seqs`/`max_tokens` are strings in the real
/// data (e.g. `"64"`), not numbers — typed as `String` deliberately rather
/// than guessing a numeric type. `extra_flags`, `env`, `ctx`,
/// `attention_backend*`, `enforce_eager` are optional; kept in `extra`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VllmModel {
    pub id: String,
    pub name: String,
    pub repo: String,
    #[serde(default)]
    pub trust_remote: bool,
    #[serde(default)]
    pub valid_tp: Vec<u32>,
    pub max_num_seqs: String,
    pub max_tokens: String,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

/// The single large extracted file R9V's "Prepare PLE" step produces
/// (`per_layer_token_embd.iq4_nl.bin`, ~26.8 GiB in the vendored fixture),
/// typed for PR11's Server Mode readiness gate (design doc §15 item 1;
/// previously left in [`R9vModel::extra`] as "not characterized beyond the
/// single sample"). Readiness is checked by `size_bytes` only, mirroring
/// upstream's own `model_manager.py::ple_ready()` exactly — `sha256` is
/// carried here for a possible future stronger check but is **not**
/// verified today (§15's flagged open item, not silently upgraded).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct R9vPleFile {
    pub filename: String,
    pub size_bytes: u64,
    pub sha256: String,
}

/// `r9v` model entry. Only 1 entry exists in the vendored fixture as of the
/// pin, so "common across all entries" is weak evidence here — every field
/// below except `id`/`name`/`repo`/`files`/`ple` is deliberately
/// `#[serde(default)]` as a safety margin against a second entry omitting
/// one. `ple` is `Option` (not required) for the same reason, even though
/// it's semantically required for the one entry that exists today — a
/// future no-PLE r9v variant shouldn't fail catalog parsing entirely.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct R9vModel {
    pub id: String,
    pub name: String,
    pub repo: String,
    #[serde(default)]
    pub revision: String,
    #[serde(default)]
    pub quant: String,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub platform_id: String,
    #[serde(default)]
    pub profile_note: String,
    #[serde(default)]
    pub recommended: bool,
    #[serde(default)]
    pub files: Vec<CatalogModelFile>,
    #[serde(default)]
    pub ple: Option<R9vPleFile>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

/// Whether a `gufo` model entry is a serveable main model or a speculative
/// draft. Closed (unlike [`CatalogModelFile::role`], a free string) so an
/// unrecognized role fails typed parsing and is surfaced by
/// `load_effective_typed_catalog` rather than silently accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GufoRole {
    Main,
    Draft,
}

/// The speculative-decoding mode a `gufo` main model uses. v1 is DFlash2-only;
/// a non-`dflash2` overlay entry fails typed parsing (design doc D14/N2). MTP
/// and DSpark are future variants, added only with their own download+serve
/// contracts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GufoSpeculativeMode {
    Dflash2,
}

/// A `gufo` main model's speculative-decoding configuration: which mode, and
/// which other catalog entry (a `role=draft` [`GufoModel`]) supplies the draft
/// GGUF. Downloaded and served as a separate entry (design doc D3/D7).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GufoSpeculative {
    pub mode: GufoSpeculativeMode,
    pub draft_model_id: String,
}

/// `gufo` model entry (brainrouter-owned overlay backend). Each entry is a
/// single-repo `hf download` (`repo`+`revision`+one `files[]` GGUF). A
/// `role=main` entry may declare a `speculative` draft (another entry with
/// `role=draft`); an entry with no `speculative` serves autoregressive. The
/// structural invariants (exactly one file, closed role/mode, a resolvable
/// non-recursive draft ref) are enforced by `gufo_overlay::validate_merged`,
/// not here.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GufoModel {
    pub id: String,
    pub name: String,
    pub repo: String,
    pub revision: String,
    pub role: GufoRole,
    pub files: Vec<CatalogModelFile>,
    #[serde(default)]
    pub recommended: bool,
    #[serde(default)]
    pub ctx_default: Option<u32>,
    #[serde(default)]
    pub sessions_default: Option<u32>,
    #[serde(default)]
    pub speculative: Option<GufoSpeculative>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

/// Per-backend model payload. `id`/`name` are duplicated onto
/// [`CatalogModelEntry`] itself (they're the two fields proven common across
/// literally every backend, including `comfyui`) so callers that only care
/// about "what models exist" don't need to match on this enum at all.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "backend", rename_all = "snake_case")]
pub enum ModelPayload {
    LlamaCpp(LlamaCppModel),
    Ds4(Ds4Model),
    Halogen(HalogenModel),
    Vllm(VllmModel),
    R9v(R9vModel),
    Gufo(GufoModel),
}

/// One model/bundle entry from `models.json`, tagged with which backend's
/// `models[]`/`bundles[]` array it came from.
#[derive(Clone, Debug, Serialize)]
pub struct CatalogModelEntry {
    pub id: String,
    pub name: String,
    pub backend: CatalogBackendId,
    /// `Some` for the five supported backends; `None` for `comfyui` (out of
    /// scope) or any unrecognized backend id — `raw` still holds the full
    /// entry either way.
    pub payload: Option<ModelPayload>,
    pub raw: Value,
}

/// One backend's section of `models.json`'s `backends` map.
#[derive(Clone, Debug, Serialize)]
pub struct ModelBackendCatalog {
    pub backend: CatalogBackendId,
    pub kind: String,
    /// `storage`/`config` vary too much per backend to model further here;
    /// kept as raw JSON (same rationale as `ToolboxDefinition::backend_config`).
    pub storage: Value,
    pub config: Value,
    pub entries: Vec<CatalogModelEntry>,
}

/// The fully parsed, typed `models.json`.
#[derive(Clone, Debug, Serialize)]
pub struct ModelCatalog {
    pub schema_version: i64,
    pub backends: Vec<ModelBackendCatalog>,
}

impl ModelCatalog {
    /// Parses a `models.json` document. Like [`super::types::ToolboxCatalog::parse`],
    /// this is independent of `schema_validate`'s structural-warning pass —
    /// run both if you want the warnings too.
    pub fn parse(value: &Value) -> Result<Self, CatalogParseError> {
        let root = value
            .as_object()
            .ok_or_else(|| CatalogParseError("root must be an object".to_string()))?;

        let schema_version = root
            .get("schema_version")
            .and_then(Value::as_i64)
            .ok_or_else(|| CatalogParseError("schema_version must be an integer".to_string()))?;

        let backends_obj = root
            .get("backends")
            .and_then(Value::as_object)
            .ok_or_else(|| CatalogParseError("backends must be an object".to_string()))?;

        let mut backends = Vec::with_capacity(backends_obj.len());
        for (backend_key, backend_value) in backends_obj {
            let backend = CatalogBackendId::from_str(backend_key);
            let entry_obj = backend_value.as_object().ok_or_else(|| {
                CatalogParseError(format!("backends.{backend_key} must be an object"))
            })?;

            let kind = entry_obj
                .get("kind")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let storage = entry_obj.get("storage").cloned().unwrap_or(Value::Null);
            let config = entry_obj.get("config").cloned().unwrap_or(Value::Null);

            // comfyui uniquely keys its list "bundles" instead of "models";
            // every other backend (including future/unrecognized ones) uses
            // "models". Try both so nothing is silently dropped.
            let raw_entries: &Vec<Value> = entry_obj
                .get("models")
                .or_else(|| entry_obj.get("bundles"))
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    CatalogParseError(format!(
                        "backends.{backend_key} must have a `models` or `bundles` array"
                    ))
                })?;

            let mut entries = Vec::with_capacity(raw_entries.len());
            for raw in raw_entries {
                let id = raw
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        CatalogParseError(format!("backends.{backend_key}: entry missing `id`"))
                    })?
                    .to_string();
                let name = raw
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        CatalogParseError(format!("backends.{backend_key}: entry missing `name`"))
                    })?
                    .to_string();

                let payload = parse_payload(&backend, raw)
                    .map_err(|e| CatalogParseError(format!("backends.{backend_key}.{id}: {e}")))?;

                entries.push(CatalogModelEntry {
                    id,
                    name,
                    backend: backend.clone(),
                    payload,
                    raw: raw.clone(),
                });
            }

            backends.push(ModelBackendCatalog {
                backend,
                kind,
                storage,
                config,
                entries,
            });
        }
        backends.sort_by(|a, b| a.backend.as_str().cmp(b.backend.as_str()));

        Ok(ModelCatalog {
            schema_version,
            backends,
        })
    }

    pub fn backend(&self, backend: SupportedServingBackend) -> Option<&ModelBackendCatalog> {
        self.backends
            .iter()
            .find(|b| b.backend.as_str() == backend.as_str())
    }
}

/// Deserializes `raw` into the [`ModelPayload`] variant matching `backend`.
/// `None` (not an error) for backends outside the supported set (`comfyui`,
/// unrecognized ids) — those entries are still fully preserved via `raw`.
fn parse_payload(
    backend: &CatalogBackendId,
    raw: &Value,
) -> Result<Option<ModelPayload>, serde_json::Error> {
    let supported = match SupportedServingBackend::try_from(backend) {
        Ok(b) => b,
        Err(_) => return Ok(None),
    };
    let payload = match supported {
        SupportedServingBackend::LlamaCpp => {
            ModelPayload::LlamaCpp(serde_json::from_value(raw.clone())?)
        }
        SupportedServingBackend::Ds4 => ModelPayload::Ds4(serde_json::from_value(raw.clone())?),
        SupportedServingBackend::Halogen => {
            ModelPayload::Halogen(serde_json::from_value(raw.clone())?)
        }
        SupportedServingBackend::Vllm => ModelPayload::Vllm(serde_json::from_value(raw.clone())?),
        SupportedServingBackend::R9v => ModelPayload::R9v(serde_json::from_value(raw.clone())?),
        SupportedServingBackend::Gufo => ModelPayload::Gufo(serde_json::from_value(raw.clone())?),
    };
    Ok(Some(payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vendored_models() -> Value {
        serde_json::from_str(include_str!("../../assets/cockpit-catalog/models.json"))
            .expect("vendored models.json must be valid JSON")
    }

    #[test]
    fn parses_the_real_vendored_models_json() {
        let doc = vendored_models();
        let catalog = ModelCatalog::parse(&doc).expect("must parse");
        assert_eq!(catalog.schema_version, 2);

        // Gufo is a brainrouter overlay backend, not part of the vendored
        // cockpit feed, so it is asserted against the effective catalog
        // (see `gufo_overlay`), not here.
        for backend in [
            SupportedServingBackend::LlamaCpp,
            SupportedServingBackend::Ds4,
            SupportedServingBackend::Halogen,
            SupportedServingBackend::Vllm,
            SupportedServingBackend::R9v,
        ] {
            let entry = catalog
                .backend(backend)
                .unwrap_or_else(|| panic!("expected a {backend} section"));
            assert!(!entry.entries.is_empty(), "{backend} should have model entries");
            assert!(
                entry.entries.iter().all(|e| e.payload.is_some()),
                "{backend} entries should all parse into a typed payload"
            );
        }
    }

    #[test]
    fn comfyui_entries_parse_but_have_no_payload() {
        let catalog = ModelCatalog::parse(&vendored_models()).unwrap();
        let comfyui = catalog
            .backends
            .iter()
            .find(|b| b.backend == CatalogBackendId::Comfyui)
            .expect("fixture must contain a comfyui backend section");
        assert!(!comfyui.entries.is_empty());
        assert!(comfyui.entries.iter().all(|e| e.payload.is_none()));
    }

    #[test]
    fn llama_cpp_entries_carry_the_expected_common_fields() {
        let catalog = ModelCatalog::parse(&vendored_models()).unwrap();
        let llama = catalog.backend(SupportedServingBackend::LlamaCpp).unwrap();
        for entry in &llama.entries {
            match &entry.payload {
                Some(ModelPayload::LlamaCpp(m)) => {
                    assert_eq!(&m.id, &entry.id);
                    assert_eq!(&m.name, &entry.name);
                    assert!(!m.repo.is_empty());
                }
                other => panic!("expected LlamaCpp payload, got {other:?}"),
            }
        }
    }

    #[test]
    fn ds4_vllm_halogen_r9v_entries_carry_their_typed_fields() {
        let catalog = ModelCatalog::parse(&vendored_models()).unwrap();

        let ds4 = catalog.backend(SupportedServingBackend::Ds4).unwrap();
        assert!(ds4.entries.iter().all(|e| matches!(
            &e.payload,
            Some(ModelPayload::Ds4(m)) if m.size_gb > 0.0 && !m.family.is_empty()
        )));

        let vllm = catalog.backend(SupportedServingBackend::Vllm).unwrap();
        assert!(vllm.entries.iter().all(|e| matches!(
            &e.payload,
            Some(ModelPayload::Vllm(m)) if !m.valid_tp.is_empty()
        )));

        let halogen = catalog.backend(SupportedServingBackend::Halogen).unwrap();
        assert!(halogen.entries.iter().all(|e| matches!(
            &e.payload,
            Some(ModelPayload::Halogen(m)) if !m.files.is_empty() && !m.checkpoint.is_empty()
        )));

        let r9v = catalog.backend(SupportedServingBackend::R9v).unwrap();
        assert!(r9v.entries.iter().all(|e| matches!(
            &e.payload,
            Some(ModelPayload::R9v(m)) if !m.files.is_empty() && m.ple.is_some()
        )));
    }

    #[test]
    fn unrecognized_backend_model_entries_still_round_trip_via_raw() {
        let mut doc = vendored_models();
        let root = doc.as_object_mut().unwrap();
        let backends = root.get_mut("backends").unwrap().as_object_mut().unwrap();
        let ds4 = backends.get("ds4").unwrap().clone();
        backends.insert("brand_new_future_backend".to_string(), ds4);

        let catalog = ModelCatalog::parse(&doc).expect("must still parse");
        let future = catalog
            .backends
            .iter()
            .find(|b| b.backend == CatalogBackendId::Other("brand_new_future_backend".to_string()))
            .expect("unrecognized backend section must still be present");
        assert!(!future.entries.is_empty());
        assert!(future.entries.iter().all(|e| e.payload.is_none()));
        // The raw entry is fully preserved even though there's no typed payload.
        assert!(future.entries[0].raw.get("filename").is_some());
    }

    #[test]
    fn gufo_model_entry_parses_from_overlay_shape() {
        // A main entry declaring a DFlash2 draft.
        let raw = serde_json::json!({
            "id": "gufo-qwen38-27b-q4kxl",
            "name": "Qwen3.8-27B UD-Q4_K_XL (DFlash2)",
            "repo": "unsloth/Qwen3.8-27B-GGUF",
            "revision": "4ca720788d1e01f1bff70c033e0d0028fd02e502",
            "role": "main",
            "recommended": true,
            "ctx_default": 32768,
            "sessions_default": 2,
            "speculative": { "mode": "dflash2", "draft_model_id": "gufo-qwen38-27b-dflash2-draft" },
            "files": [ { "path": "Qwen3.8-27B-UD-Q4_K_XL.gguf", "size_bytes": 17559178144u64 } ]
        });
        let payload = parse_payload(&CatalogBackendId::Gufo, &raw)
            .expect("gufo entry must parse")
            .expect("gufo is a supported backend");
        match payload {
            ModelPayload::Gufo(m) => {
                assert_eq!(m.id, "gufo-qwen38-27b-q4kxl");
                assert_eq!(m.role, GufoRole::Main);
                assert_eq!(m.files.len(), 1);
                assert_eq!(m.files[0].size_bytes, 17_559_178_144);
                let spec = m.speculative.expect("declares a draft");
                assert_eq!(spec.mode, GufoSpeculativeMode::Dflash2);
                assert_eq!(spec.draft_model_id, "gufo-qwen38-27b-dflash2-draft");
            }
            other => panic!("expected Gufo payload, got {other:?}"),
        }

        // A draft entry with no speculative parses as role=draft.
        let draft = serde_json::json!({
            "id": "gufo-qwen38-27b-dflash2-draft",
            "name": "Qwen3.8-27B DFlash2 draft (Q4_K_M)",
            "repo": "z-lab/Qwen3.8-27B-DFlash2-GGUF",
            "revision": "2d9571f8ce46e151f61c6499c99dee6079e1d610",
            "role": "draft",
            "files": [ { "path": "Qwen3.8-27B-DFlash2-Q4_K_M.gguf", "size_bytes": 1143006816u64 } ]
        });
        match parse_payload(&CatalogBackendId::Gufo, &draft).unwrap().unwrap() {
            ModelPayload::Gufo(m) => {
                assert_eq!(m.role, GufoRole::Draft);
                assert!(m.speculative.is_none());
            }
            other => panic!("expected Gufo payload, got {other:?}"),
        }

        // A non-dflash2 speculative mode fails typed parsing (v1 is dflash2-only).
        let bad = serde_json::json!({
            "id": "x", "name": "x", "repo": "r", "revision": "v", "role": "main",
            "speculative": { "mode": "mtp", "draft_model_id": "y" },
            "files": [ { "path": "a.gguf", "size_bytes": 1u64 } ]
        });
        assert!(parse_payload(&CatalogBackendId::Gufo, &bad).is_err());
    }
}
