//! Context gatherer — collects PRD, git diff, and AGENTS file for review prompts.

use std::process::Command;

const MAX_FILE_SIZE: usize = 200 * 1024; // 200 KB
pub(crate) const MAX_SECTION_SIZE: usize = 150 * 1024; // 150 KB per section

/// Gathered context for a single review pass.
pub struct ReviewContext {
    pub prd: Option<String>,
    pub git_diff: String,
    pub agents_content: Option<String>,
}

/// Detect and read the PRD file from common paths relative to project_dir.
fn load_prd(project_dir: &str) -> Option<String> {
    let candidates = ["docs/PRD.md", "PRD.md", "README.md"];
    for candidate in &candidates {
        let path = if project_dir.is_empty() {
            std::path::PathBuf::from(candidate)
        } else {
            std::path::Path::new(project_dir).join(candidate)
        };
        if let Ok(content) = std::fs::read_to_string(path) {
            return Some(truncate(content, MAX_FILE_SIZE));
        }
    }
    None
}

/// Run `git diff HEAD` to collect unstaged + staged changes.
/// Returns empty string if not in a git repo or git is unavailable.
fn load_git_diff(project_dir: &str) -> String {
    let mut cmd = Command::new("git");
    // Collect unstaged + staged changes. Exclude README.md because it is often large and 
    // crowds out actual code changes in the review prompt.
    cmd.args(["diff", "HEAD", "--", ".", ":(exclude)README.md"]);
    if !project_dir.is_empty() {
        cmd.current_dir(project_dir);
    }
    let out = cmd.output();
    match out {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout).to_string();
            truncate(text, MAX_FILE_SIZE)
        }
        _ => String::new(),
    }
}

/// Load the agent contract from `~/.omp/agent/LLAMACPP.md`.
/// 
/// Note: This does not take project_dir because the agent contract is 
/// user-wide configuration, not project-specific.
fn load_agents() -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    let path = format!("{}/.omp/agent/LLAMACPP.md", home);
    std::fs::read_to_string(&path).ok()
}

/// Gather all context in one call.
pub fn gather(project_dir: &str) -> ReviewContext {
    ReviewContext {
        prd: load_prd(project_dir),
        git_diff: load_git_diff(project_dir),
        agents_content: load_agents(),
    }
}

/// Truncate `text` to `max` bytes, appending a warning note if truncated.
pub(crate) fn truncate(text: String, max: usize) -> String {
    if text.len() <= max {
        return text;
    }
    let original_kb = text.len() / 1024;
    let warning = format!(
        "\n\n[WARNING: truncated to {}KB; original was {}KB]",
        max / 1024,
        original_kb
    );
    let mut new_len = max.saturating_sub(warning.len());
    while !text.is_char_boundary(new_len) && new_len > 0 {
        new_len -= 1;
    }
    let mut out = text;
    out.truncate(new_len);
    out.push_str(&warning);
    out
}
