//! Tests for the install subcommand's JSON merge logic.
//!
//! We test the idempotent merge function used when patching harness configs.

use serde_json::{json, Value};

/// Mirror the merge_json logic from install.rs (accessible via the same logic).
/// Since install.rs is in the binary, we replicate the merge logic here for testing.
fn merge_json(mut base: Value, patch: Value) -> Value {
    if let (Some(base_obj), Some(patch_obj)) = (base.as_object_mut(), patch.as_object()) {
        for (key, val) in patch_obj {
            let existing = base_obj.entry(key.clone()).or_insert(Value::Null);
            if existing.is_object() && val.is_object() {
                *existing = merge_json(existing.clone(), val.clone());
            } else {
                *existing = val.clone();
            }
        }
        base
    } else {
        patch
    }
}

#[test]
fn merge_empty_base_with_patch() {
    let base = json!({});
    let patch = json!({ "mcpServers": { "brainrouter": { "type": "stdio" } } });
    let result = merge_json(base, patch);
    assert!(result["mcpServers"]["brainrouter"]["type"].as_str() == Some("stdio"));
}

#[test]
fn merge_adds_new_key_preserving_existing() {
    let base = json!({ "mcpServers": { "other-tool": { "command": "node" } } });
    let patch = json!({ "mcpServers": { "brainrouter": { "type": "stdio" } } });
    let result = merge_json(base, patch);
    // Both keys present
    assert!(result["mcpServers"]["other-tool"]["command"].as_str() == Some("node"));
    assert!(result["mcpServers"]["brainrouter"]["type"].as_str() == Some("stdio"));
}

#[test]
fn merge_updates_existing_brainrouter_entry() {
    let base = json!({
        "mcpServers": {
            "brainrouter": { "type": "stdio", "command": "old-brainrouter" }
        }
    });
    let patch = json!({
        "mcpServers": {
            "brainrouter": { "type": "stdio", "command": "new-brainrouter", "args": ["mcp"] }
        }
    });
    let result = merge_json(base, patch);
    assert_eq!(result["mcpServers"]["brainrouter"]["command"].as_str(), Some("new-brainrouter"));
    assert_eq!(result["mcpServers"]["brainrouter"]["type"].as_str(), Some("stdio"));
    assert!(result["mcpServers"]["brainrouter"]["args"].is_array());
}

#[test]
fn merge_idempotent_same_patch_twice() {
    let base = json!({});
    let patch = json!({ "mcpServers": { "brainrouter": { "command": "br" } } });
    let after_first = merge_json(base, patch.clone());
    let after_second = merge_json(after_first, patch);
    assert_eq!(
        after_second["mcpServers"]["brainrouter"]["command"].as_str(),
        Some("br")
    );
}

#[test]
fn merge_opencode_provider_and_mcp() {
    let base = json!({
        "provider": {
            "anthropic": { "apiKey": "sk-xxx" }
        }
    });
    let patch = json!({
        "provider": {
            "brainrouter": {
                "name": "Brainrouter",
                "options": { "baseURL": "http://127.0.0.1:9099/v1" }
            }
        },
        "mcp": {
            "brainrouter": { "type": "local", "command": ["brainrouter", "mcp"] }
        }
    });
    let result = merge_json(base, patch);
    // Existing anthropic provider preserved
    assert!(result["provider"]["anthropic"]["apiKey"].as_str() == Some("sk-xxx"));
    // New brainrouter provider added
    assert_eq!(
        result["provider"]["brainrouter"]["name"].as_str(),
        Some("Brainrouter")
    );
    // MCP section created
    assert!(result["mcp"]["brainrouter"]["command"].is_array());
}

#[test]
fn merge_non_object_patch_replaces_base() {
    let base = json!("old string");
    let patch = json!("new string");
    let result = merge_json(base, patch);
    assert_eq!(result.as_str(), Some("new string"));
}

#[test]
fn merge_array_in_patch_replaces_array_in_base() {
    let base = json!({ "args": ["old", "args"] });
    let patch = json!({ "args": ["mcp", "--socket", "/run/brainrouter.sock"] });
    let result = merge_json(base, patch);
    let args: Vec<_> = result["args"].as_array().unwrap().iter()
        .filter_map(Value::as_str)
        .collect();
    assert_eq!(args, vec!["mcp", "--socket", "/run/brainrouter.sock"]);
}

// ── FR-C: vendored HankNDory skill + installer step ──────────────────────────
// These assert the packaging is present and internally consistent without
// executing install.sh (which mutates a user's ~/.omp).

fn repo_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

#[test]
fn vendored_hankndory_skill_files_are_present() {
    for f in [
        "assets/skills/hankndory/SKILL.md",
        "assets/skills/hankndory/agents/ui_metadata.yaml",
        "assets/skills/hankndory/reference/design-doc-template.md",
        "assets/skills/hankndory/SOURCE",
        "assets/skills/hankndory/APPEND_SYSTEM.snippet.md",
    ] {
        assert!(repo_path(f).is_file(), "missing vendored skill file: {f}");
    }
}

#[test]
fn append_snippet_is_a_reconciled_managed_block() {
    let snippet = std::fs::read_to_string(repo_path("assets/skills/hankndory/APPEND_SYSTEM.snippet.md")).unwrap();
    assert!(snippet.contains("# >>> brainrouter-managed (hankndory) v1 >>>"), "missing start marker");
    assert!(snippet.contains("# <<< brainrouter-managed <<<"), "missing end marker");
    assert!(snippet.contains("Step 9"), "must frame review as HankNDory Step 9");
    // The reconciliation must NOT reintroduce the stale 'local LLM' wording.
    assert!(!snippet.to_lowercase().contains("local llm"), "stale 'local LLM' wording must be gone");
}

#[test]
fn installer_step_is_idempotent_and_uses_the_vendored_path() {
    let sh = std::fs::read_to_string(repo_path("install.sh")).unwrap();
    assert!(sh.contains("assets/skills/hankndory"), "installer must copy the vendored skill");
    assert!(sh.contains("# >>> brainrouter-managed (hankndory) v1 >>>"), "installer must key on the managed marker");
    // Idempotency guard: skip when the managed block is already present.
    assert!(sh.contains("already has the managed block"), "installer must be idempotent");
    // Safety: writes as the target user and refuses symlinked destinations.
    assert!(sh.contains("sudo -u \"$TARGET_USER\""), "installer must write as the target user");
    assert!(sh.contains("Refusing to install into symlinked path"), "installer must refuse symlinked destinations");
}
