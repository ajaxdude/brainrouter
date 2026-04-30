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

/// Collect the most relevant git diff for review.
///
/// Strategy:
/// 1. `git diff HEAD` — uncommitted changes (working tree + staged vs HEAD).
/// 2. If empty, `git diff HEAD~1..HEAD` — the last commit's changes.
/// 3. If still empty, return empty string.
///
/// This layered approach ensures the review sees something useful whether the
/// agent is mid-work (uncommitted changes) or has already committed (typical
/// end-of-task flow).
fn load_git_diff(project_dir: &str) -> String {
    // Try uncommitted changes first.
    let uncommitted = run_git_diff(project_dir, &["diff", "HEAD", "--", ".", ":(exclude)README.md"]);
    if !uncommitted.is_empty() {
        return truncate(uncommitted, MAX_FILE_SIZE);
    }

    // Uncommitted diff is empty — the agent likely committed already.
    // Show what the most recent commit changed.
    let last_commit = run_git_diff(project_dir, &["diff", "HEAD~1..HEAD", "--", ".", ":(exclude)README.md"]);
    if !last_commit.is_empty() {
        let header = "[Note: No uncommitted changes found. Showing diff from the most recent commit.]\n\n";
        return truncate(format!("{}{}", header, last_commit), MAX_FILE_SIZE);
    }

    tracing::debug!(project_dir, "No git diff found (neither uncommitted nor HEAD~1..HEAD)");
    String::new()
}

/// Run a git command and return its stdout, or empty string on failure.
fn run_git_diff(project_dir: &str, args: &[&str]) -> String {
    let mut cmd = Command::new("git");
    cmd.args(args);
    if !project_dir.is_empty() {
        cmd.current_dir(project_dir);
    }
    match cmd.output() {
        Ok(o) if o.status.success() => {
            String::from_utf8_lossy(&o.stdout).trim().to_string()
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
