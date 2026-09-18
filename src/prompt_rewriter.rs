//! System prompt rewriter for local LLM routing.
//!
//! OMP sends a massive system prompt (15-20K tokens) with rules, contracts,
//! skills, and behavioral directives that overwhelm small local models and
//! cause tool-call loops. This module replaces those system messages with a
//! lean coding prompt while preserving tool schemas and non-system messages.

use crate::types::ChatMessage;
use tracing::debug;

/// Default lean system prompt for local coding agents.
const DEFAULT_LOCAL_PROMPT: &str = "\
You are a coding assistant running on a local LLM via llama-swap.

## Rules
- Read files before editing. Use `read` to read, `grep` to search, `find` to locate.
- Use `edit` for changes, `write` for new files, `bash` for commands (build/test/install only).
- Make complete changes in one pass. No TODOs, no placeholders.
- Do NOT call tools repeatedly on the same target. Use information you already have.
- Do NOT re-read files to verify your own edits unless the edit tool reported failure.
- When done, state what you changed and stop. Do not loop through verification steps.

## Anti-loop
- If you have already read a file, do not read it again.
- If you have already made an edit, do not verify it by reading the file back.
- If a tool call returns an error, try ONE alternative approach, then report the error.
- Never call the same tool with the same arguments twice in one turn.";

/// FR-B gate-block sentinels. A `v1` gate block is delimited by these exact
/// ASCII markers. Only `v1` is recognized; any other version string is a
/// different marker and is treated as "no v1 block".
const GATE_START: &str = "<!--HANKNDORY:GATE:START v1-->";
const GATE_END: &str = "<!--HANKNDORY:GATE:END-->";
/// Upper bound on a preserved gate block, guarding against a pathological
/// oversized block re-inflating the lean prompt.
const MAX_GATE_BYTES: usize = 4096;

/// Rewrite system messages in a request for local LLM consumption.
///
/// Single pass: replaces the first prose system message with the lean prompt,
/// preserves tool-schema system messages, drops remaining prose, passes
/// non-system messages through untouched.
pub fn rewrite_for_local(messages: Vec<ChatMessage>, custom_prompt: Option<&str>) -> Vec<ChatMessage> {
    let prompt = custom_prompt.unwrap_or(DEFAULT_LOCAL_PROMPT);
    let mut result = Vec::with_capacity(messages.len());
    let mut injected = false;
    // FR-B: preserve exactly one well-formed HankNDory gate block found in the
    // prose system messages, so a marked skill/gate survives the local rewrite
    // instead of being dropped. Fail closed: preservation happens only when the
    // v1 START/END markers each appear exactly once across ALL prose system
    // messages and delimit one well-formed block. Any extra or malformed marker
    // anywhere ⇒ preserve none. When no marker is present, behavior is
    // byte-identical to the pre-FR-B rewrite.
    let mut gate_block: Option<String> = None;
    let mut total_starts = 0usize;
    let mut total_ends = 0usize;
    // Index of the injected lean prompt, so the gate is re-emitted immediately
    // after it (not after an earlier tool-schema system message).
    let mut lean_idx: Option<usize> = None;

    for msg in messages {
        if msg.role != "system" {
            result.push(msg);
            continue;
        }

        if is_tool_schema(&msg) {
            result.push(msg);
            continue;
        }

        // Prose system message: tally markers and capture the first well-formed
        // block before replacing/dropping. The total-marker tally (not a
        // per-message flag) is what enforces the fail-closed contract, so that a
        // single message carrying two blocks cannot slip past a per-message
        // "exactly one" check and let another message's block win.
        if let Some(text) = msg.content.as_ref().and_then(|c| c.as_str()) {
            total_starts += text.matches(GATE_START).count();
            total_ends += text.matches(GATE_END).count();
            if gate_block.is_none() {
                gate_block = extract_gate_block(text);
            }
        }

        if !injected {
            debug!("Replacing OMP system prompt with local lean prompt");
            lean_idx = Some(result.len());
            result.push(ChatMessage {
                role: "system".to_string(),
                content: Some(serde_json::Value::String(prompt.to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });
            injected = true;
        } else {
            // Drop additional prose system messages (skills, context files, etc.)
            debug!("Dropping extra system message in local mode rewrite");
        }
    }

    if !injected {
        lean_idx = Some(0);
        result.insert(0, ChatMessage {
            role: "system".to_string(),
            content: Some(serde_json::Value::String(prompt.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // Re-emit the preserved gate block verbatim, as its own system message
    // immediately after the lean prompt — but only when the markers are
    // unambiguous across the whole message set (exactly one START and one END).
    if total_starts == 1 && total_ends == 1 {
        if let Some(block) = gate_block {
            // `gate_block` is Some only after a prose message was processed,
            // which guarantees a lean prompt was injected, so `lean_idx` is set.
            let pos = lean_idx.map(|i| i + 1).unwrap_or(result.len());
            result.insert(
                pos,
                ChatMessage {
                    role: "system".to_string(),
                    content: Some(serde_json::Value::String(block)),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
            );
        }
    }

    result
}

/// FR-B gate-block grammar. Extract the single well-formed
/// `<!--HANKNDORY:GATE:START v1-->…<!--HANKNDORY:GATE:END-->` block in `text`,
/// or `None`. Returns `None` (block not captured from this message) when the
/// message does not contain exactly one START and one END marker, when END
/// precedes START, or when the block exceeds `MAX_GATE_BYTES`. Whole-set
/// disambiguation (a duplicate block split across two messages) is enforced by
/// the caller's total-marker tally, not here.
fn extract_gate_block(text: &str) -> Option<String> {
    let starts: Vec<_> = text.match_indices(GATE_START).collect();
    let ends: Vec<_> = text.match_indices(GATE_END).collect();
    if starts.len() != 1 || ends.len() != 1 {
        return None; // zero, or duplicate/nested markers
    }
    let start = starts[0].0;
    let end = ends[0].0;
    if end <= start {
        return None; // END before START
    }
    let block = &text[start..end + GATE_END.len()];
    if block.len() > MAX_GATE_BYTES {
        return None;
    }
    Some(block.to_string())
}

/// Fast heuristic: is this system message a tool schema rather than prose?
///
/// Tool schemas contain JSON structure markers. OMP prose is markdown.
/// Non-string content (arrays, objects) is assumed to be structured data.
#[inline]
fn is_tool_schema(msg: &ChatMessage) -> bool {
    let text = match &msg.content {
        Some(serde_json::Value::String(s)) => s.as_str(),
        Some(serde_json::Value::Array(_) | serde_json::Value::Object(_)) => return true,
        _ => return false,
    };

    // Two independent checks — either indicates a tool definition.
    (text.contains("\"type\"") && text.contains("\"function\""))
        || (text.contains("\"parameters\"") && text.contains("\"properties\""))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: Some(serde_json::Value::String(content.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn replaces_prose_system_message() {
        let msgs = vec![
            msg("system", "You are a distinguished staff engineer operating inside Oh My Pi..."),
            msg("user", "Hello"),
        ];
        let result = rewrite_for_local(msgs, None);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].role, "system");
        let content = result[0].content.as_ref().unwrap().as_str().unwrap();
        assert!(content.contains("coding assistant running on a local LLM"));
        assert_eq!(result[1].role, "user");
    }

    #[test]
    fn preserves_tool_schema_messages() {
        let msgs = vec![
            msg("system", "Long OMP prose..."),
            msg("system", r#"{"type":"function","function":{"name":"read","parameters":{"properties":{}}}}"#),
            msg("user", "Hello"),
        ];
        let result = rewrite_for_local(msgs, None);
        assert_eq!(result.len(), 3);
        assert!(result[1].content.as_ref().unwrap().as_str().unwrap().contains("\"function\""));
    }

    #[test]
    fn drops_extra_prose_messages() {
        let msgs = vec![
            msg("system", "Main OMP system prompt..."),
            msg("system", "Skills injection block..."),
            msg("system", "APPEND_SYSTEM.md content..."),
            msg("user", "Hello"),
        ];
        let result = rewrite_for_local(msgs, None);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn custom_prompt_override() {
        let msgs = vec![msg("system", "OMP stuff"), msg("user", "Hi")];
        let result = rewrite_for_local(msgs, Some("Custom instructions"));
        assert_eq!(result[0].content.as_ref().unwrap().as_str().unwrap(), "Custom instructions");
    }

    #[test]
    fn no_system_messages_injects_prompt() {
        let msgs = vec![msg("user", "Hello")];
        let result = rewrite_for_local(msgs, None);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].role, "system");
    }

    #[test]
    fn prose_with_json_like_words_is_not_tool_schema() {
        // A prose message mentioning JSON-like words should NOT be preserved as a tool schema.
        // The heuristic requires quoted JSON keys ("type", "function"), not bare words.
        let msgs = vec![
            msg("system", "Set the type to function and configure parameters for properties."),
            msg("user", "Hello"),
        ];
        let result = rewrite_for_local(msgs, None);
        assert_eq!(result.len(), 2);
        assert!(result[0].content.as_ref().unwrap().as_str().unwrap().contains("coding assistant"));
    }

    fn gate(body: &str) -> String {
        format!("<!--HANKNDORY:GATE:START v1-->{body}<!--HANKNDORY:GATE:END-->")
    }

    #[test]
    fn preserves_gate_block_from_a_dropped_system_message() {
        let block = gate("keep me: design gate");
        let msgs = vec![
            msg("system", "Main OMP prompt"),
            msg("system", &format!("skills\n{block}\nmore")),
            msg("user", "Hi"),
        ];
        let result = rewrite_for_local(msgs, None);
        // lean prompt + preserved gate block + user
        assert_eq!(result.len(), 3);
        assert!(result[0].content.as_ref().unwrap().as_str().unwrap().contains("coding assistant"));
        assert_eq!(result[1].content.as_ref().unwrap().as_str().unwrap(), block);
        assert_eq!(result[2].role, "user");
    }

    #[test]
    fn preserves_gate_block_from_the_first_replaced_message() {
        let block = gate("gate in the first message");
        let msgs = vec![msg("system", &format!("intro {block} outro")), msg("user", "Hi")];
        let result = rewrite_for_local(msgs, None);
        assert_eq!(result.len(), 3);
        assert_eq!(result[1].content.as_ref().unwrap().as_str().unwrap(), block);
    }

    #[test]
    fn duplicate_gate_blocks_are_not_preserved() {
        let block = gate("dup");
        let msgs = vec![
            msg("system", &format!("a {block}")),
            msg("system", &format!("b {block}")),
            msg("user", "Hi"),
        ];
        let result = rewrite_for_local(msgs, None);
        // Ambiguous ⇒ preserve none: only lean prompt + user.
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn unterminated_gate_block_is_not_preserved() {
        let msgs = vec![
            msg("system", "x <!--HANKNDORY:GATE:START v1--> no end here"),
            msg("user", "Hi"),
        ];
        let result = rewrite_for_local(msgs, None);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn unknown_gate_version_is_not_preserved() {
        let msgs = vec![
            msg("system", "<!--HANKNDORY:GATE:START v2-->future<!--HANKNDORY:GATE:END-->"),
            msg("user", "Hi"),
        ];
        let result = rewrite_for_local(msgs, None);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn a_two_block_message_poisons_the_whole_set() {
        // Fail-closed contract: a message carrying two blocks must not be
        // silently ignored while a different single-block message's gate wins.
        let msgs = vec![
            msg("system", &format!("two here: {} {}", gate("one"), gate("two"))),
            msg("system", &gate("three")),
            msg("user", "Hi"),
        ];
        let result = rewrite_for_local(msgs, None);
        // Total START markers across the set is 3 ⇒ preserve none: lean + user.
        assert_eq!(result.len(), 2);
        assert!(result[0].content.as_ref().unwrap().as_str().unwrap().contains("coding assistant"));
        assert_eq!(result[1].role, "user");
    }

    #[test]
    fn gate_is_inserted_after_the_lean_prompt_not_a_leading_tool_schema() {
        let block = gate("design gate");
        let msgs = vec![
            msg("system", r#"{"type":"function","function":{"name":"read","parameters":{"properties":{}}}}"#),
            msg("system", &format!("skills\n{block}")),
            msg("user", "Hi"),
        ];
        let result = rewrite_for_local(msgs, None);
        // tool schema, lean prompt, gate block, user — the gate sits directly
        // after the lean prompt, not after the leading tool schema.
        assert_eq!(result.len(), 4);
        assert!(result[0].content.as_ref().unwrap().as_str().unwrap().contains("\"function\""));
        assert!(result[1].content.as_ref().unwrap().as_str().unwrap().contains("coding assistant"));
        assert_eq!(result[2].content.as_ref().unwrap().as_str().unwrap(), block);
        assert_eq!(result[3].role, "user");
    }
}
