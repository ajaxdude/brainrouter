//! OMP subprocess invocation and output parsing.
//!
//! This is the heart of the bridge: spawn `omp -p --mode json`, parse its
//! NDJSON event stream, and surface `(assistant_text, session_id, model_info)`.
//!
//! Also contains model alias resolution and alias-file loading, shared by
//! all transports.

use std::collections::HashMap;
use std::path::PathBuf;
use tracing::{info, warn};

// ---------------------------------------------------------------------------
// Model aliases
// ---------------------------------------------------------------------------

/// Load model aliases from a YAML file.
///
/// The file must contain a top-level `model_aliases` mapping:
/// ```yaml
/// model_aliases:
///   gemma: llama.cpp/gemma-4-31b-draft
///   qwen:  llama.cpp/qwen3-coder-next
/// ```
/// Keys are stored lowercased for case-insensitive matching at runtime.
/// A missing file or missing `model_aliases` key is silently treated as an
/// empty map.
pub fn load_model_aliases(path: &str) -> HashMap<String, String> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            warn!("Could not read aliases config {}: {}", path, e);
            return HashMap::new();
        }
    };

    let doc: serde_yml::Value = match serde_yml::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            warn!("Could not parse aliases config {}: {}", path, e);
            return HashMap::new();
        }
    };

    let mapping = match doc.get("model_aliases").and_then(|v| v.as_mapping()) {
        Some(m) => m,
        None => {
            warn!(
                "No 'model_aliases' key found in {}; model aliases disabled",
                path
            );
            return HashMap::new();
        }
    };

    let mut aliases = HashMap::new();
    for (k, v) in mapping {
        if let (Some(key), Some(val)) = (k.as_str(), v.as_str()) {
            aliases.insert(key.to_lowercase(), val.to_string());
        }
    }
    info!("Loaded {} model alias(es) from {}", aliases.len(), path);
    aliases
}

/// Resolve a user-supplied model alias to the canonical OMP model ID.
///
/// Resolution rules (first match wins):
/// 1. If the raw alias contains `/`, `.`, or `-` it is fully-qualified — pass through.
/// 2. Case-insensitive exact match against the alias map.
/// 3. Case-insensitive substring search; longest matching key wins.
/// 4. No match — return verbatim; OMP surfaces its own unknown-model error.
pub fn resolve_model(raw: &str, aliases: &HashMap<String, String>) -> String {
    let lower = raw.to_lowercase();

    // Already fully-qualified or a direct llama-swap model ID — pass through unchanged.
    // Hyphens signal a bare llama-swap model name (e.g. "gemma-4-26b-a4b") that must
    // never be fuzzy-matched against a shorter alias.
    if lower.contains('/') || lower.contains('.') || lower.contains('-') {
        return raw.to_string();
    }

    // Exact match first — avoids "qwen" accidentally shadowing "qwen35".
    if let Some(canonical) = aliases.get(&lower) {
        tracing::debug!("resolved alias {:?} -> {:?} (exact)", raw, canonical);
        return canonical.clone();
    }

    // Substring match — prefer the longest matching needle so that a more
    // specific alias ("qwen35") wins over a shorter prefix ("qwen").
    let mut best: Option<(&str, &str)> = None;
    for (needle, canonical) in aliases {
        if lower.contains(needle.as_str()) {
            let is_better = match best {
                None => true,
                Some((prev, _)) => needle.len() > prev.len(),
            };
            if is_better {
                best = Some((needle.as_str(), canonical.as_str()));
            }
        }
    }

    if let Some((_, canonical)) = best {
        tracing::debug!("resolved alias {:?} -> {:?} (substring)", raw, canonical);
        return canonical.to_string();
    }

    raw.to_string()
}

// ---------------------------------------------------------------------------
// OMP invocation
// ---------------------------------------------------------------------------

/// Invoke the OMP CLI and return `(assistant_text, session_id, model_info)`.
///
/// Flags always set:
/// - `-p`            run as a subprocess (no interactive TUI)
/// - `--mode json`   write NDJSON events to stdout
///
/// `model` is passed as `--model <id>` when `Some`.
/// `session_id` is passed as `--resume <id>` when `Some`.
///
/// The timeout is an **inactivity** timeout: the clock resets every time OMP
/// produces a line of output.  This lets long-running tool-call chains
/// complete as long as OMP keeps making progress, while still catching
/// genuine hangs.
///
/// Returns `Err` if OMP couldn't be spawned, hung silent, or if the parsed
/// output contained a model-level error with no text.
pub async fn invoke_omp(
    omp_path: &str,
    work_dir: &str,
    model: Option<&str>,
    query: &str,
    session_id: Option<&str>,
    timeout_secs: u64,
) -> Result<(String, Option<String>, Option<(String, String)>), String> {
    use std::process::Stdio;
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
    use tokio::process::Command;

    const MAX_STDOUT_BYTES: usize = 4 * 1024 * 1024;
    const MAX_STDERR_BYTES: u64 = 1024 * 1024;

    let mut cmd = Command::new(omp_path);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.current_dir(work_dir);
    cmd.arg("-p");
    cmd.arg("--mode");
    cmd.arg("json");

    // Disable MCP server discovery, skill loading, and built-in tools.
    // Bridge messages are conversational — the model must respond with text,
    // not start a tool-call loop. Without --no-tools the cloud model immediately
    // calls bash/read and never produces a text response.
    cmd.arg("--no-extensions");
    cmd.arg("--no-skills");
    cmd.arg("--no-tools");

    if let Some(sid) = session_id {
        cmd.arg("--resume");
        cmd.arg(sid);
    }

    if let Some(m) = model {
        cmd.arg("--model");
        cmd.arg(m);
    }

    // The eager-todo-prelude in OMP injects a <system-reminder> that forces the
    // model to call `todo_write` before responding.  Local models that cannot
    // handle tool calling hang indefinitely.  OMP skips the prelude when the
    // prompt ends with '?' or '!'.  Since bridge messages are conversational
    // (not agentic tasks), we always bypass the prelude.
    let query = if query.ends_with('?') || query.ends_with('!') {
        query.to_string()
    } else {
        format!("{}?", query)
    };

    cmd.arg(&query);
    cmd.kill_on_drop(true);

    let mut child = cmd.spawn()
        .map_err(|e| format!("OMP process I/O error: {}", e))?;

    let stdout = child.stdout.take()
        .ok_or_else(|| "failed to capture OMP stdout".to_string())?;
    let stderr = child.stderr.take()
        .ok_or_else(|| "failed to capture OMP stderr".to_string())?;
    let stderr_task = tokio::spawn(async move {
        let mut stderr = stderr;
        let mut bytes = Vec::new();
        let mut truncated = false;
        let mut chunk = [0u8; 8192];
        loop {
            let count = stderr.read(&mut chunk).await?;
            if count == 0 {
                break;
            }
            let remaining = (MAX_STDERR_BYTES as usize).saturating_sub(bytes.len());
            let retained = remaining.min(count);
            bytes.extend_from_slice(&chunk[..retained]);
            truncated |= retained < count;
        }
        Ok::<_, std::io::Error>((bytes, truncated))
    });
    let mut reader = BufReader::new(stdout).lines();
    let inactivity = std::time::Duration::from_secs(timeout_secs);
    let mut collected = Vec::new();
    let mut collected_bytes = 0usize;

    loop {
        match tokio::time::timeout(inactivity, reader.next_line()).await {
            Ok(Ok(Some(line))) => {
                collected_bytes = collected_bytes.saturating_add(line.len() + 1);
                if collected_bytes > MAX_STDOUT_BYTES {
                    drop(reader);
                    let _ = child.kill().await;
                    let _ = child.wait().await;
                    let _ = stderr_task.await;
                    return Err(format!(
                        "OMP output exceeded the {} MiB limit",
                        MAX_STDOUT_BYTES / (1024 * 1024)
                    ));
                }
                collected.push(line);
            }
            Ok(Ok(None)) => break,           // EOF — process closed stdout
            Ok(Err(e)) => {
                drop(reader);
                let _ = child.kill().await;
                let _ = child.wait().await;
                let _ = stderr_task.await;
                return Err(format!("OMP stdout read error: {e}"));
            }
            Err(_) => {
                // Inactivity timeout — kill the process and reap to avoid zombies.
                tracing::warn!("OMP silent for {}s, killing", timeout_secs);
                // Drop the stdout reader first to unblock the child if its pipe is full.
                drop(reader);
                let _ = child.kill().await;
                let _ = child.wait().await;
                let _ = stderr_task.await;
                return Err(format!(
                    "OMP timed out after {}s of inactivity",
                    timeout_secs
                ));
            }
        }
    }

    // Wait for the process to finish (should be instant after EOF).
    let status = child.wait().await
        .map_err(|e| format!("OMP wait error: {}", e))?;
    let (stderr_bytes, truncated) = stderr_task
        .await
        .map_err(|e| format!("OMP stderr task failed: {e}"))?
        .map_err(|e| format!("OMP stderr read error: {e}"))?;
    let mut stderr = String::from_utf8_lossy(&stderr_bytes)
        .trim()
        .to_string();
    if truncated {
        stderr.push_str("\n[stderr truncated]");
    }

    if !status.success() {
        return Err(if stderr.is_empty() {
            format!("OMP exited with status {}", status)
        } else {
            format!("OMP exited with status {}: {}", status, stderr)
        });
    }

    let ndjson = collected.join("\n");
    parse_omp_json_output(ndjson.as_bytes())
}

// ---------------------------------------------------------------------------
// NDJSON output parsing
// ---------------------------------------------------------------------------

/// Parse OMP's `--mode json` NDJSON output.
///
/// Returns `Ok((assistant_text, session_id, Option<(provider, model)>))` on
/// success, or `Err(error_message)` when OMP's assistant turn ended with
/// `stopReason: "error"` and produced no text.
///
/// Events consumed:
/// - `{"type":"session","id":"<id>"}` — active session ID (first event)
/// - `{"type":"message_end","message":{"role":"assistant",...}}` —
///   completed assistant turn; collects `{"type":"text","text":"..."}` items
///   and captures the first `provider`/`model` pair.
pub fn parse_omp_json_output(
    ndjson: &[u8],
) -> Result<(String, Option<String>, Option<(String, String)>), String> {
    let content = String::from_utf8_lossy(ndjson);
    let mut text_pieces: Vec<String> = Vec::new();
    let mut session_id: Option<String> = None;
    let mut model_info: Option<(String, String)> = None;
    // The first model-error we encounter; used only when no text was produced.
    let mut model_error: Option<String> = None;
    let mut saw_tool_use = false;
    let mut saw_any_event = false;
    let mut saw_completed_assistant = false;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(val) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let Some(event_type) = val.get("type").and_then(|t| t.as_str()) else {
            continue;
        };
        saw_any_event = true;

        match event_type {
            "session" => {
                if let Some(id) = val.get("id").and_then(|v| v.as_str()) {
                    session_id = Some(id.to_string());
                }
            }
            "message_end" => {
                let Some(msg) = val.get("message") else { continue };
                if msg.get("role").and_then(|r| r.as_str()) != Some("assistant") {
                    continue;
                }
                saw_completed_assistant = true;
                // Capture provider+model from the first assistant message_end.
                if model_info.is_none() {
                    if let (Some(p), Some(m)) = (
                        msg.get("provider").and_then(|v| v.as_str()),
                        msg.get("model").and_then(|v| v.as_str()),
                    ) {
                        model_info = Some((p.to_string(), m.to_string()));
                    }
                }
                let Some(content) = msg.get("content").and_then(|c| c.as_array()) else {
                    continue;
                };
                for item in content {
                    match item.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                                let trimmed = text.trim().to_string();
                                if !trimmed.is_empty() {
                                    text_pieces.push(trimmed);
                                }
                            }
                        }
                        Some("tool_use") => {
                            saw_tool_use = true;
                        }
                        _ => {}
                    }
                }
                // Record model-level errors only when content was empty.
                if content.is_empty()
                    && msg.get("stopReason").and_then(|v| v.as_str()) == Some("error") {
                        if let Some(err_msg) = msg.get("errorMessage").and_then(|v| v.as_str()) {
                            let first_line = err_msg.lines().next().unwrap_or(err_msg);
                            model_error.get_or_insert_with(|| first_line.to_string());
                        }
                    }
            }
            _ => {}
        }
    }

    if !saw_completed_assistant {
        return Err("OMP exited before completing an assistant response.".to_string());
    }

    // Return Err when OMP produced no text at all and signalled an error.
    if text_pieces.is_empty() {
        if let Some(err) = model_error {
            return Err(err);
        }
        // If OMP ran tool calls but never produced a text response, that's
        // unexpected for a conversational bridge query.  Return a diagnostic.
        if saw_tool_use {
            return Err("The model performed actions but produced no visible response. Try rephrasing.".to_string());
        }
        if !saw_any_event {
            return Err("No response received from OMP.".to_string());
        }
    }

    Ok((text_pieces.join("\n\n"), session_id, model_info))
}

// ---------------------------------------------------------------------------
// Sandbox helpers
// ---------------------------------------------------------------------------

/// Validate and resolve a candidate path within `root`.
///
/// Returns `Ok(resolved)` when the canonicalized path is within `root`,
/// or `Err(message)` when it escapes the sandbox or doesn't exist.
pub fn sandbox_resolve(candidate: PathBuf, root: &std::path::Path) -> Result<PathBuf, String> {
    let resolved = candidate
        .canonicalize()
        .map_err(|_| format!("`{}`: no such directory", candidate.display()))?;
    if !resolved.starts_with(root) {
        return Err("Cannot navigate outside the root directory.".to_string());
    }
    Ok(resolved)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn executable_script(contents: &str) -> (PathBuf, PathBuf) {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "brainrouter-bridge-process-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let script = root.join("omp-test");
        std::fs::write(&script, contents).unwrap();
        let mut permissions = std::fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&script, permissions).unwrap();
        (root, script)
    }

    fn aliases() -> HashMap<String, String> {
        [
            ("gemma", "llama.cpp/gemma-4-31b-draft"),
            ("qwen", "llama.cpp/qwen3-coder-next"),
            ("mistral", "llama.cpp/mistral-small-4"),
            ("qwen35", "llama.cpp/qwen3-coder-next"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    #[test]
    fn exact_match_wins_over_prefix() {
        assert_eq!(
            resolve_model("qwen35", &aliases()),
            "llama.cpp/qwen3-coder-next"
        );
    }

    #[test]
    fn exact_match_is_case_insensitive() {
        assert_eq!(resolve_model("Qwen35", &aliases()), "llama.cpp/qwen3-coder-next");
        assert_eq!(resolve_model("QWEN35", &aliases()), "llama.cpp/qwen3-coder-next");
    }

    #[test]
    fn substring_match_prefers_longest_needle() {
        let mut a = aliases();
        a.insert("qwen35coder".to_string(), "llama.cpp/specific-coder".to_string());
        assert_eq!(resolve_model("qwen35coder", &a), "llama.cpp/specific-coder");
    }

    #[test]
    fn fully_qualified_passes_through() {
        assert_eq!(
            resolve_model("llama.cpp/some-model", &aliases()),
            "llama.cpp/some-model"
        );
    }

    #[test]
    fn unknown_alias_passes_through() {
        assert_eq!(resolve_model("gpt4o", &aliases()), "gpt4o");
    }

    #[test]
    fn hyphenated_model_id_passes_through() {
        assert_eq!(resolve_model("gemma-4-26b-a4b", &aliases()), "gemma-4-26b-a4b");
        assert_eq!(resolve_model("qwen3.5-35b-a3b", &aliases()), "qwen3.5-35b-a3b");
    }

    #[test]
    fn gemma_resolves() {
        assert_eq!(resolve_model("Gemma", &aliases()), "llama.cpp/gemma-4-31b-draft");
    }

    #[test]
    fn omp_model_error_is_surfaced() {
        let ndjson = br#"{"type":"session","id":"abc123"}
{"type":"message_end","message":{"role":"assistant","content":[],"stopReason":"error","errorMessage":"400 \"could not find suitable inference handler for bad-model\"\nraw-http-request=/tmp/foo.json","provider":"llama.cpp","model":"bad-model"}}"#;
        let result = parse_omp_json_output(ndjson);
        assert!(result.is_err(), "expected Err, got: {:?}", result);
        let err = result.unwrap_err();
        assert!(err.contains("bad-model"), "error should mention model name, got: {:?}", err);
        assert!(!err.contains("raw-http-request"), "must not leak internal path");
    }

    #[test]
    fn omp_successful_response_is_ok() {
        let ndjson = br#"{"type":"session","id":"sess1"}
{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"I am Gemma."}],"stopReason":"stop","provider":"llama.cpp","model":"gemma-4-26b-a4b"}}"#;
        let result = parse_omp_json_output(ndjson);
        assert!(result.is_ok(), "expected Ok, got: {:?}", result);
        let (text, session_id, model_info) = result.unwrap();
        assert_eq!(text, "I am Gemma.");
        assert_eq!(session_id.as_deref(), Some("sess1"));
        assert_eq!(model_info.as_ref().map(|(p, _)| p.as_str()), Some("llama.cpp"));
    }

    #[test]
    fn omp_session_without_completed_assistant_is_rejected() {
        let result = parse_omp_json_output(br#"{"type":"session","id":"sess1"}"#);
        assert_eq!(
            result.unwrap_err(),
            "OMP exited before completing an assistant response."
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn invoke_omp_drains_stderr_without_deadlocking() {
        let (root, script) = executable_script(
            "#!/bin/sh\nhead -c 2097152 /dev/zero >&2\nprintf '%s\\n' '{\"type\":\"session\",\"id\":\"s\"}' '{\"type\":\"message_end\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"ok\"}],\"stopReason\":\"stop\"}}'\n",
        );
        let result = invoke_omp(
            script.to_str().unwrap(),
            root.to_str().unwrap(),
            None,
            "hello",
            None,
            5,
        )
        .await
        .unwrap();
        assert_eq!(result.0, "ok");
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn invoke_omp_rejects_nonzero_exit_even_with_valid_stdout() {
        let (root, script) = executable_script(
            "#!/bin/sh\nprintf '%s\\n' '{\"type\":\"session\",\"id\":\"s\"}' '{\"type\":\"message_end\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"partial\"}],\"stopReason\":\"stop\"}}'\necho failed >&2\nexit 7\n",
        );
        let error = invoke_omp(
            script.to_str().unwrap(),
            root.to_str().unwrap(),
            None,
            "hello",
            None,
            5,
        )
        .await
        .unwrap_err();
        assert!(error.contains("status"));
        assert!(error.contains("failed"));
        let _ = std::fs::remove_dir_all(root);
    }
}
