//! Prompt template for the review loop.
//!
//! Builds the full prompt from gathered context sections joined by separators.
//! All string manipulation; no templating engine needed here.

use super::context::{truncate, MAX_SECTION_SIZE, ReviewContext};

/// Separator between sections.
const SEP: &str = "\n\n============================================================\n\n";

/// Build the complete LLM review prompt from context.
pub fn build_review_prompt(
    ctx: &ReviewContext,
    task_id: &str,
    summary: &str,
    details: Option<&str>,
    session_history: &[String],
) -> String {
    let mut sections: Vec<String> = Vec::new();

    // 1. PRD
    if let Some(prd) = &ctx.prd {
        let body = truncate(prd.clone(), MAX_SECTION_SIZE);
        sections.push(format!("# PROJECT REQUIREMENTS DOCUMENT (PRD)\n\n{}", body));
    }

    // 2. Git diff
    let diff = ctx.git_diff.trim();
    if !diff.is_empty() {
        let body = truncate(diff.to_string(), MAX_SECTION_SIZE);
        sections.push(format!("# GIT DIFF\n\n{}", body));
    }

    // 3. Agent contract
    if let Some(agents) = &ctx.agents_content {
        let body = truncate(agents.clone(), MAX_SECTION_SIZE);
        sections.push(format!("# AGENT CONTRACT (LLAMACPP.md)\n\n{}", body));
    }

    // 4. Task details
    {
        let mut task_section = format!(
            "# TASK DETAILS\n\n## Task ID\n{}\n\n## Summary\n{}",
            task_id, summary
        );
        if let Some(d) = details {
            task_section.push_str(&format!("\n\n## Details\n{}", d));
        }
        sections.push(task_section);
    }

    // 5. Session history
    if !session_history.is_empty() {
        let history = session_history.join("\n\n");
        let body = truncate(history, MAX_SECTION_SIZE);
        sections.push(format!("# SESSION HISTORY\n\n{}", body));
    }

    // 6. Review criteria (always last)
    sections.push(REVIEW_CRITERIA.to_string());

    sections.join(SEP)
}

const REVIEW_CRITERIA: &str = r#"# REVIEW CRITERIA

Please review the code changes and provide feedback. Your response MUST be a JSON object with the following structure:

{
  "status": "approved" | "needs_revision" | "escalated",
  "feedback": "Your detailed feedback here"
}

## Status Values

- "approved": Code changes are good to merge
- "needs_revision": Code needs improvements before merging
- "escalated": Issue requires human review

## Feedback Guidelines

Provide specific, actionable feedback when status is "needs_revision" or "escalated":
- Identify specific line numbers or code sections
- Explain why something needs to change
- Suggest concrete improvements
- Mention security, performance, or maintainability concerns"#;
