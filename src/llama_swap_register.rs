//! Safe registration helpers for adding downloaded llama.cpp GGUFs to llama-swap.
//!
//! The public endpoint that uses these helpers mutates the hand-tuned
//! llama-swap `config.yaml`, so this module keeps the risky pieces small and
//! unit-testable: deterministic GGUF selection, baseline-entry construction,
//! and comment-preserving insertion with structural YAML validation.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde_yaml::{Mapping, Value};

const AUXILIARY_GGUF_MARKERS: &[&str] = &[
    "mmproj", "mtp", "draft", "vision", "dspark", "-spec", "speculat",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegisterError {
    BadRequest(String),
    Conflict(String),
    Internal(String),
}

impl RegisterError {
    pub fn status_code(&self) -> u16 {
        match self {
            Self::BadRequest(_) => 400,
            Self::Conflict(_) => 409,
            Self::Internal(_) => 500,
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Self::BadRequest(m) | Self::Conflict(m) | Self::Internal(m) => m,
        }
    }
}

impl std::fmt::Display for RegisterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for RegisterError {}

/// Resolve the primary GGUF that llama-server should load for a downloaded
/// llama.cpp catalog entry. The resolver deliberately refuses ambiguous or
/// partial states instead of guessing: writing a wrong `--model` path into
/// llama-swap is worse than asking the operator to register manually.
pub fn resolve_primary_gguf(
    destination: &Path,
    quant_pattern: &str,
) -> Result<PathBuf, RegisterError> {
    let all = collect_ggufs(destination).map_err(|e| {
        RegisterError::Internal(format!(
            "failed to scan `{}` for GGUF files: {e}",
            destination.display()
        ))
    })?;
    if all.is_empty() {
        return Err(RegisterError::Conflict(
            "no .gguf found (download incomplete?)".to_string(),
        ));
    }

    let mut scoped: Vec<PathBuf> = if quant_pattern.ends_with(".gguf") {
        let basename = Path::new(quant_pattern)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(quant_pattern);
        all.iter()
            .filter(|p| {
                p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|name| wildcard_match(basename, name))
            })
            .cloned()
            .collect()
    } else {
        let scope = destination.join(quant_pattern);
        all.iter()
            .filter(|p| p.starts_with(&scope))
            .cloned()
            .collect()
    };

    scoped.retain(|p| !is_auxiliary_gguf(p));
    scoped.sort();

    if scoped.is_empty() {
        return Err(RegisterError::Conflict(format!(
            "no primary .gguf found for quant pattern `{quant_pattern}` (download incomplete?)"
        )));
    }

    let first_shards: Vec<(PathBuf, String, usize)> = scoped
        .iter()
        .filter_map(|p| {
            shard_info(p)
                .filter(|(_, index, _)| *index == 1)
                .map(|(prefix, _, total)| (p.clone(), prefix, total))
        })
        .collect();
    if !first_shards.is_empty() {
        if first_shards.len() > 1 {
            return Err(RegisterError::Conflict(
                "ambiguous — multiple sharded GGUF groups matched; register manually".to_string(),
            ));
        }
        let (first, prefix, total) = &first_shards[0];
        let count = scoped
            .iter()
            .filter_map(|p| shard_info(p))
            .filter(|(candidate_prefix, _index, candidate_total)| {
                candidate_prefix == prefix && candidate_total == total
            })
            .count();
        if count != *total {
            return Err(RegisterError::Conflict(format!(
                "incomplete shard set for `{prefix}`: found {count} of {total} shards"
            )));
        }
        return canonicalize_selected(first);
    }

    if scoped.len() == 1 {
        return canonicalize_selected(&scoped[0]);
    }

    Err(RegisterError::Conflict(
        "ambiguous — multiple primary GGUFs matched; register manually".to_string(),
    ))
}

pub fn build_entry(id: &str, name: &str, model_path: &Path) -> String {
    format!(
        "  {}:\n    name: {}\n    cmd: |\n      ${{ls}}\n      --port ${{PORT}}\n      ${{common}}\n      --model {}\n",
        yaml_quoted(id),
        yaml_quoted(&format!("{name} (registered baseline)")),
        shell_quote(&model_path.display().to_string())
    )
}

/// Inserts a pre-built, two-space-indented model entry immediately after the
/// top-level `models:` line while preserving surrounding comments/macros. The
/// result is parsed again and checked structurally so a textual false-positive
/// (`models:` inside a block scalar, quoted duplicate keys, malformed block)
/// never reaches disk.
pub fn insert_model_entry(
    config_text: &str,
    id: &str,
    entry_block: &str,
) -> Result<String, RegisterError> {
    let parsed: Value = serde_yaml::from_str(config_text)
        .map_err(|e| RegisterError::BadRequest(format!("invalid llama-swap YAML: {e}")))?;
    let root = parsed.as_mapping().ok_or_else(|| {
        RegisterError::BadRequest("llama-swap config must be a top-level YAML mapping".to_string())
    })?;
    let models_key = Value::String("models".to_string());
    let models = root.get(&models_key).ok_or_else(|| {
        RegisterError::BadRequest(
            "llama-swap config must contain a top-level `models` mapping".to_string(),
        )
    })?;
    let models_mapping = models.as_mapping().ok_or_else(|| {
        RegisterError::BadRequest("top-level `models` must be a YAML mapping".to_string())
    })?;
    if mapping_contains_key(models_mapping, id) {
        return Err(RegisterError::Conflict(format!(
            "llama-swap model `{id}` is already registered"
        )));
    }
    let original_keys = mapping_string_keys(models_mapping);

    let Some((models_line_start, models_line_end, inline_value)) =
        find_top_level_models_line(config_text)
    else {
        return Err(RegisterError::BadRequest(
            "could not locate top-level `models:` line for safe insertion".to_string(),
        ));
    };

    let candidate = if inline_value.trim().is_empty() {
        let insert_at = models_line_end;
        let mut out = String::with_capacity(config_text.len() + entry_block.len());
        out.push_str(&config_text[..insert_at]);
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(entry_block);
        out.push_str(&config_text[insert_at..]);
        out
    } else if inline_value.trim() == "{}" {
        let mut out = String::with_capacity(config_text.len() + entry_block.len() + 16);
        out.push_str(&config_text[..models_line_start]);
        out.push_str("models:\n");
        out.push_str(entry_block);
        out.push_str(&config_text[models_line_end..]);
        out
    } else {
        return Err(RegisterError::BadRequest(
            "top-level `models` must use block style (or `models: {}`) for comment-preserving insertion".to_string(),
        ));
    };

    let reparsed: Value = serde_yaml::from_str(&candidate).map_err(|e| {
        RegisterError::Internal(format!(
            "generated llama-swap config failed YAML validation: {e}"
        ))
    })?;
    let new_models = reparsed
        .as_mapping()
        .and_then(|m| m.get(&models_key))
        .and_then(Value::as_mapping)
        .ok_or_else(|| {
            RegisterError::Internal(
                "generated llama-swap config lost the top-level `models` mapping".to_string(),
            )
        })?;
    let mut expected_keys = original_keys.clone();
    expected_keys.insert(id.to_string());
    let actual_keys = mapping_string_keys(new_models);
    if actual_keys != expected_keys {
        return Err(RegisterError::Internal(format!(
            "generated llama-swap config changed existing model keys; expected {:?}, got {:?}",
            expected_keys, actual_keys
        )));
    }
    Ok(candidate)
}

pub fn path_visibility_warning(path: &Path) -> Option<String> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let visible = path.starts_with("/mnt") || home.as_ref().is_some_and(|h| path.starts_with(h));
    (!visible).then(|| {
        "path may not be visible inside the llama-server toolbox container; verify before use"
            .to_string()
    })
}

fn collect_ggufs(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    collect_ggufs_inner(root, &mut out)?;
    out.sort();
    Ok(out)
}

fn collect_ggufs_inner(path: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if !path.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let p = entry.path();
        let ft = entry.file_type()?;
        if ft.is_dir() {
            collect_ggufs_inner(&p, out)?;
        } else if ft.is_file()
            && p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            out.push(p);
        }
    }
    Ok(())
}

fn is_auxiliary_gguf(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    AUXILIARY_GGUF_MARKERS
        .iter()
        .any(|marker| name.contains(marker))
}

fn shard_info(path: &Path) -> Option<(String, usize, usize)> {
    let name = path.file_name()?.to_str()?;
    let stem = name
        .strip_suffix(".gguf")
        .or_else(|| name.strip_suffix(".GGUF"))?;
    let (left, total_str) = stem.rsplit_once("-of-")?;
    let total = total_str.parse::<usize>().ok()?;
    if total == 0 || total_str.len() != 5 {
        return None;
    }
    let (prefix, index_str) = left.rsplit_once('-')?;
    let index = index_str.parse::<usize>().ok()?;
    if index == 0 || index > total || index_str.len() != 5 {
        return None;
    }
    Some((prefix.to_string(), index, total))
}

fn wildcard_match(pattern: &str, candidate: &str) -> bool {
    if !pattern.contains('*') {
        return pattern == candidate;
    }
    let mut rest = candidate;
    let mut first = true;
    for part in pattern.split('*') {
        if part.is_empty() {
            continue;
        }
        if first && !pattern.starts_with('*') {
            let Some(after) = rest.strip_prefix(part) else {
                return false;
            };
            rest = after;
        } else if let Some(idx) = rest.find(part) {
            rest = &rest[idx + part.len()..];
        } else {
            return false;
        }
        first = false;
    }
    pattern.ends_with('*') || rest.is_empty()
}

fn canonicalize_selected(path: &Path) -> Result<PathBuf, RegisterError> {
    std::fs::canonicalize(path).map_err(|e| {
        RegisterError::Conflict(format!(
            "failed to canonicalize selected GGUF `{}`: {e}",
            path.display()
        ))
    })
}

fn mapping_contains_key(mapping: &Mapping, id: &str) -> bool {
    mapping.keys().any(|k| k.as_str() == Some(id))
}

fn mapping_string_keys(mapping: &Mapping) -> BTreeSet<String> {
    mapping
        .keys()
        .filter_map(|k| k.as_str().map(str::to_string))
        .collect()
}

fn find_top_level_models_line(config_text: &str) -> Option<(usize, usize, &str)> {
    let mut offset = 0;
    for line in config_text.split_inclusive('\n') {
        let without_newline = line.trim_end_matches(['\r', '\n']);
        if !without_newline.starts_with(char::is_whitespace)
            && without_newline.starts_with("models:")
        {
            return Some((
                offset,
                offset + line.len(),
                &without_newline["models:".len()..],
            ));
        }
        offset += line.len();
    }
    None
}

fn yaml_quoted(s: &str) -> String {
    serde_json::to_string(s).expect("JSON string serialization cannot fail")
}

fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn test_dir(name: &str) -> PathBuf {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let p = std::env::current_dir()
            .unwrap()
            .join("target")
            .join("llama_swap_register_tests")
            .join(format!("{name}-{n}"));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"gguf").unwrap();
    }

    #[test]
    fn resolve_exact_filename() {
        let d = test_dir("exact");
        touch(&d.join("model.Q4_K_M.gguf"));
        touch(&d.join("model.mmproj.gguf"));
        let p = resolve_primary_gguf(&d, "model.Q4_K_M.gguf").unwrap();
        assert_eq!(
            p,
            std::fs::canonicalize(d.join("model.Q4_K_M.gguf")).unwrap()
        );
    }

    #[test]
    fn resolve_glob_filename() {
        let d = test_dir("glob");
        touch(&d.join("abc-Q5_K_M.gguf"));
        let p = resolve_primary_gguf(&d, "*Q5_K_M.gguf").unwrap();
        assert_eq!(p.file_name().unwrap(), "abc-Q5_K_M.gguf");
    }

    #[test]
    fn resolve_folder_pattern_scopes_to_subdir() {
        let d = test_dir("folder");
        touch(&d.join("Q4/model.gguf"));
        touch(&d.join("BF16/model.gguf"));
        let p = resolve_primary_gguf(&d, "BF16").unwrap();
        assert!(p.ends_with("BF16/model.gguf"));
    }

    #[test]
    fn resolve_rejects_auxiliary_only() {
        let d = test_dir("aux");
        touch(&d.join("Q4/model-mmproj.gguf"));
        let err = resolve_primary_gguf(&d, "Q4").unwrap_err();
        assert!(matches!(err, RegisterError::Conflict(_)));
        assert!(err.message().contains("no primary .gguf"));
    }

    #[test]
    fn resolve_complete_shard_group_selects_first() {
        let d = test_dir("shards");
        touch(&d.join("BF16/model-00001-of-00002.gguf"));
        touch(&d.join("BF16/model-00002-of-00002.gguf"));
        let p = resolve_primary_gguf(&d, "BF16").unwrap();
        assert!(p.ends_with("BF16/model-00001-of-00002.gguf"));
    }

    #[test]
    fn resolve_incomplete_shard_group_conflicts() {
        let d = test_dir("incomplete-shards");
        touch(&d.join("BF16/model-00001-of-00003.gguf"));
        touch(&d.join("BF16/model-00002-of-00003.gguf"));
        let err = resolve_primary_gguf(&d, "BF16").unwrap_err();
        assert!(err.message().contains("incomplete shard set"));
    }

    #[test]
    fn resolve_ambiguous_multiple_primary_conflicts() {
        let d = test_dir("ambiguous");
        touch(&d.join("BF16/a.gguf"));
        touch(&d.join("BF16/b.gguf"));
        let err = resolve_primary_gguf(&d, "BF16").unwrap_err();
        assert!(err.message().contains("ambiguous"));
    }

    #[test]
    fn resolve_no_gguf_conflicts() {
        let d = test_dir("none");
        std::fs::write(d.join("readme.txt"), b"x").unwrap();
        let err = resolve_primary_gguf(&d, "BF16").unwrap_err();
        assert!(err.message().contains("no .gguf found"));
    }

    #[test]
    fn insert_after_block_models_preserves_comments_and_reparses() {
        let cfg = "macros:\n  ls: llama\nmodels:\n  old:\n    cmd: old\nsettings:\n  x: y\n";
        let entry = build_entry("new", "New", Path::new("/mnt/models/new.gguf"));
        let out = insert_model_entry(cfg, "new", &entry).unwrap();
        assert!(out.contains("models:\n  \"new\":"));
        assert!(out.contains("  old:"));
        serde_yaml::from_str::<Value>(&out).unwrap();
    }

    #[test]
    fn insert_rejects_when_candidate_would_swallow_existing_four_space_key() {
        let cfg = "models:
    old:
      cmd: old
";
        let entry = build_entry("new", "New", Path::new("/mnt/models/new.gguf"));
        let err = insert_model_entry(cfg, "new", &entry).unwrap_err();
        assert!(matches!(err, RegisterError::Internal(_)));
        assert!(err.message().contains("changed existing model keys"));
    }

    #[test]
    fn insert_normal_two_space_config_adds_exactly_one_key() {
        let cfg = "models:
  old:
    cmd: old
";
        let entry = build_entry("new", "New", Path::new("/mnt/models/new.gguf"));
        let out = insert_model_entry(cfg, "new", &entry).unwrap();
        let parsed: Value = serde_yaml::from_str(&out).unwrap();
        let keys = mapping_string_keys(
            parsed
                .as_mapping()
                .unwrap()
                .get(Value::String("models".to_string()))
                .unwrap()
                .as_mapping()
                .unwrap(),
        );
        assert_eq!(keys, BTreeSet::from(["old".to_string(), "new".to_string()]));
    }

    #[test]
    fn insert_rejects_structural_duplicate_even_if_quoted() {
        let cfg = "models:\n  \"new\":\n    cmd: old\n";
        let entry = build_entry("new", "New", Path::new("/mnt/models/new.gguf"));
        let err = insert_model_entry(cfg, "new", &entry).unwrap_err();
        assert!(matches!(err, RegisterError::Conflict(_)));
    }

    #[test]
    fn insert_ignores_models_inside_block_scalar() {
        let cfg = "note: |\n  models:\nmodels: {}\n";
        let entry = build_entry("new", "New", Path::new("/mnt/models/new.gguf"));
        let out = insert_model_entry(cfg, "new", &entry).unwrap();
        assert!(out.contains("note: |\n  models:\nmodels:\n  \"new\":"));
    }

    #[test]
    fn insert_handles_empty_inline_models() {
        let cfg = "macros: {}\nmodels: {}\n";
        let entry = build_entry("new", "New", Path::new("/mnt/models/new.gguf"));
        let out = insert_model_entry(cfg, "new", &entry).unwrap();
        assert!(out.contains("models:\n  \"new\":"));
    }

    #[test]
    fn insert_rejects_nonempty_inline_models_clearly() {
        let cfg = "models: { old: { cmd: old } }\n";
        let entry = build_entry("new", "New", Path::new("/mnt/models/new.gguf"));
        let err = insert_model_entry(cfg, "new", &entry).unwrap_err();
        assert!(err.message().contains("block style"));
    }

    #[test]
    fn insert_invalid_base_config_errors() {
        let entry = build_entry("new", "New", Path::new("/mnt/models/new.gguf"));
        let err = insert_model_entry("models: [", "new", &entry).unwrap_err();
        assert!(matches!(err, RegisterError::BadRequest(_)));
    }

    #[test]
    fn build_entry_escapes_yaml_scalars() {
        let entry = build_entry(
            "id: tricky",
            "Name \" quoted",
            Path::new("/mnt/models/a.gguf"),
        );
        let cfg = format!("models:\n{entry}");
        serde_yaml::from_str::<Value>(&cfg).unwrap();
        assert!(entry.contains("registered baseline"));
        assert!(entry.contains("--model '/mnt/models/a.gguf'"));
    }

    #[test]
    fn wildcard_match_supports_basename_globs() {
        assert!(super::wildcard_match("*Q4*.gguf", "abc-Q4_K_M.gguf"));
        assert!(!super::wildcard_match("Q5*.gguf", "abc-Q5.gguf"));
    }

    #[test]
    fn path_visibility_warning_allows_mnt() {
        assert!(path_visibility_warning(Path::new("/mnt/models/a.gguf")).is_none());
        assert!(path_visibility_warning(Path::new("/opt/models/a.gguf")).is_some());
    }
}
