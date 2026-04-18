//! Context gatherer — collects PRD, git diff, and AGENTS file for review prompts.

use std::process::Command;

const MAX_FILE_SIZE: usize = 100 * 1024; // 100 KB
const MAX_SECTION_SIZE: usize = 25 * 1024; // 25 KB per section

/// Gathered context for a single review pass.
pub struct ReviewContext {
    pub prd: Option<String>,
    pub git_diff: String,
    pub agents_content: Option<String>,
}

/// Detect and read the PRD file from common paths relative to cwd.
pub fn load_prd() -> Option<String> {
    let candidates = ["docs/PRD.md", "PRD.md", "README.md"];
    for candidate in &candidates {
        if let Ok(content) = std::fs::read_to_string(candidate) {
            return Some(truncate(content, MAX_FILE_SIZE));
        }
    }
    None
}

/// Run `git diff HEAD` to collect unstaged + staged changes.
/// Returns empty string if not in a git repo or git is unavailable.
pub fn load_git_diff() -> String {
    let out = Command::new("git").args(["diff", "HEAD"]).output();
    match out {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout).to_string();
            truncate(text, MAX_FILE_SIZE)
        }
        _ => String::new(),
    }
}

/// Load the agent contract from `~/.omp/agent/LLAMACPP.md`.
pub fn load_agents() -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    let path = format!("{}/.omp/agent/LLAMACPP.md", home);
    std::fs::read_to_string(&path).ok()
}

/// Gather all context in one call.
pub fn gather() -> ReviewContext {
    ReviewContext {
        prd: load_prd(),
        git_diff: load_git_diff(),
        agents_content: load_agents(),
    }
}

/// Truncate `text` to `max` bytes, appending a warning note if truncated.
pub fn truncate(text: String, max: usize) -> String {
    if text.len() <= max {
        return text;
    }
    let original_kb = text.len() / 1024;
    let mut out = text;
    out.truncate(max);
    out.push_str(&format!(
        "\n\n[WARNING: truncated to {}KB; original was {}KB]",
        max / 1024,
        original_kb
    ));
    out
}

/// Per-section truncation limit.
pub fn section_max() -> usize {
    MAX_SECTION_SIZE
}
