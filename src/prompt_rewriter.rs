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

/// Rewrite system messages in a request for local LLM consumption.
///
/// Single pass: replaces the first prose system message with the lean prompt,
/// preserves tool-schema system messages, drops remaining prose, passes
/// non-system messages through untouched.
pub fn rewrite_for_local(messages: Vec<ChatMessage>, custom_prompt: Option<&str>) -> Vec<ChatMessage> {
    let prompt = custom_prompt.unwrap_or(DEFAULT_LOCAL_PROMPT);
    let mut result = Vec::with_capacity(messages.len());
    let mut injected = false;

    for msg in messages {
        if msg.role != "system" {
            result.push(msg);
            continue;
        }

        if is_tool_schema(&msg) {
            result.push(msg);
        } else if !injected {
            debug!("Replacing OMP system prompt with local lean prompt");
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
        result.insert(0, ChatMessage {
            role: "system".to_string(),
            content: Some(serde_json::Value::String(prompt.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    result
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
}
