//! Context gatherer — collects PRD, git diff, and AGENTS file for review prompts.

use std::path::{Path, PathBuf};
use std::process::Command;

const MAX_FILE_SIZE: usize = 200 * 1024; // 200 KB
pub(crate) const MAX_SECTION_SIZE: usize = 150 * 1024; // 150 KB per section
const MAX_UNTRACKED_FILES: usize = 100;
const MAX_UNTRACKED_DIFF: usize = 70 * 1024;
const MAX_DIRTY_DIFF: usize = 50 * 1024;
const MAX_COMMITTED_DIFF: usize = 25 * 1024;
const MAX_UNTRACKED_FILE_CONTENT: usize = 32 * 1024;

/// Gathered context for a single review pass.
pub struct ReviewContext {
    pub prd: Option<String>,
    pub git_diff: String,
    pub agents_content: Option<String>,
}

/// Detect and read the PRD file from common paths relative to the repository root.
fn load_prd(project_root: &Path) -> Option<String> {
    let candidates = ["docs/PRD.md", "PRD.md", "README.md"];
    for candidate in &candidates {
        let path = project_root.join(candidate);
        if let Ok(content) = std::fs::read_to_string(path) {
            return Some(truncate(content, MAX_FILE_SIZE));
        }
    }
    None
}

/// Collect the most relevant git diff for review.
///
fn load_git_diff(project_root: &Path) -> String {
    let mut sections = Vec::new();
    let untracked = load_untracked_files(project_root);
    if !untracked.is_empty() {
        sections.push(format!(
            "[Untracked files]\n\n{}",
            truncate(untracked, MAX_UNTRACKED_DIFF)
        ));
    }
    let uncommitted = run_git(project_root, &["diff", "--find-renames", "HEAD"]);
    if !uncommitted.is_empty() {
        sections.push(format!(
            "[Staged and unstaged tracked changes]\n\n{}",
            truncate(uncommitted, MAX_DIRTY_DIFF)
        ));
    }
    if let Some(base) = review_base(project_root) {
        let committed = run_git(project_root, &["diff", "--find-renames", &format!("{base}..HEAD")]);
        if !committed.is_empty() {
            sections.push(format!(
                "[Committed changes from merge base {base} to HEAD]\n\n{}",
                truncate(committed, MAX_COMMITTED_DIFF)
            ));
        }
    }
    if sections.is_empty() {
        let last_commit = run_git(project_root, &["diff", "--find-renames", "HEAD~1..HEAD"]);
        if !last_commit.is_empty() {
            sections.push(format!(
                "[No task base or working-tree changes found; showing the most recent commit]\n\n{last_commit}"
            ));
        }
    }
    if sections.is_empty() {
        tracing::debug!(project_root = %project_root.display(), "No git review diff found");
    }
    sections.join("\n\n")
}

fn review_base(project_root: &Path) -> Option<String> {
    if let Ok(base) = std::env::var("BRAINROUTER_REVIEW_BASE") {
        let base = base.trim();
        if !base.is_empty()
            && !run_git(project_root, &["rev-parse", "--verify", base]).is_empty()
        {
            return Some(base.to_string());
        }
    }
    for reference in ["@{upstream}", "origin/HEAD", "origin/main", "origin/master"] {
        let base = run_git(project_root, &["merge-base", "HEAD", reference]);
        if !base.is_empty() && base != run_git(project_root, &["rev-parse", "HEAD"]) {
            return Some(base);
        }
    }
    None
}

fn load_untracked_files(project_root: &Path) -> String {
    let output = run_git_raw(
        project_root,
        &["ls-files", "--others", "--exclude-standard", "-z"],
    );
    let paths = output
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .take(MAX_UNTRACKED_FILES)
        .filter_map(|path| std::str::from_utf8(path).ok().map(str::to_string))
        .collect::<Vec<_>>();
    if paths.is_empty() {
        return String::new();
    }
    let mut sections = vec![format!(
        "Files:\n{}",
        paths
            .iter()
            .map(|path| format!("- {path}"))
            .collect::<Vec<_>>()
            .join("\n")
    )];
    for relative in paths {
        let path = project_root.join(&relative);
        let Ok(metadata) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            continue;
        }
        if metadata.len() as usize > MAX_FILE_SIZE {
            sections.push(format!("--- /dev/null\n+++ b/{relative}\n[untracked file exceeds review limit]"));
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&path) else {
            sections.push(format!("--- /dev/null\n+++ b/{relative}\n[binary or unreadable untracked file]"));
            continue;
        };
        sections.push(format!(
            "--- /dev/null\n+++ b/{relative}\n{}",
            truncate(content, MAX_UNTRACKED_FILE_CONTENT)
        ));
    }
    sections.join("\n\n")
}

/// Run a git command and return its stdout, or empty string on failure.
fn run_git(project_dir: &Path, args: &[&str]) -> String {
    String::from_utf8_lossy(&run_git_raw(project_dir, args))
        .trim()
        .to_string()
}

fn run_git_raw(project_dir: &Path, args: &[&str]) -> Vec<u8> {
    let mut cmd = Command::new("git");
    cmd.args(args).current_dir(project_dir);
    match cmd.output() {
        Ok(output) if output.status.success() => output.stdout,
        _ => Vec::new(),
    }
}

fn resolve_project_root(project_dir: &str) -> PathBuf {
    let start = if project_dir.is_empty() {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    } else {
        PathBuf::from(project_dir)
    };
    let root = run_git(&start, &["rev-parse", "--show-toplevel"]);
    if root.is_empty() {
        start
    } else {
        PathBuf::from(root)
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
    let project_root = resolve_project_root(project_dir);
    ReviewContext {
        prd: load_prd(&project_root),
        git_diff: load_git_diff(&project_root),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn git(dir: &Path, args: &[&str]) {
        let status = Command::new("git")
            .args(args)
            .current_dir(dir)
            .status()
            .expect("git should run");
        assert!(status.success(), "git {:?} failed", args);
    }

    #[test]
    fn nested_project_dir_uses_repository_wide_diff_and_root_prd() {
        let root = std::env::temp_dir().join(format!(
            "brainrouter-review-context-{}",
            uuid::Uuid::new_v4()
        ));
        let nested = root.join("src/nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(root.join("README.md"), "initial readme\n").unwrap();
        std::fs::write(root.join("Cargo.toml"), "initial manifest\n").unwrap();
        std::fs::write(nested.join("inside.rs"), "initial source\n").unwrap();

        git(&root, &["init", "-q"]);
        git(&root, &["config", "user.email", "brainrouter@example.invalid"]);
        git(&root, &["config", "user.name", "Brainrouter Test"]);
        git(&root, &["add", "."]);
        git(&root, &["commit", "-q", "-m", "initial"]);

        std::fs::write(root.join("README.md"), "updated readme\n").unwrap();
        std::fs::write(root.join("Cargo.toml"), "updated manifest\n").unwrap();

        let ctx = gather(nested.to_str().unwrap());
        assert_eq!(ctx.prd.as_deref(), Some("updated readme\n"));
        assert!(ctx.git_diff.contains("README.md"));
        assert!(ctx.git_diff.contains("Cargo.toml"));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn includes_commits_since_merge_base_and_untracked_files() {
        let root = std::env::temp_dir().join(format!(
            "brainrouter-review-history-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("README.md"), "base\n").unwrap();

        git(&root, &["init", "-q"]);
        git(&root, &["config", "user.email", "brainrouter@example.invalid"]);
        git(&root, &["config", "user.name", "Brainrouter Test"]);
        git(&root, &["add", "."]);
        git(&root, &["commit", "-q", "-m", "base"]);
        git(&root, &["update-ref", "refs/remotes/origin/main", "HEAD"]);

        std::fs::write(root.join("committed.txt"), "committed change\n").unwrap();
        git(&root, &["add", "committed.txt"]);
        git(&root, &["commit", "-q", "-m", "feature"]);
        std::fs::write(root.join("untracked.txt"), "untracked change\n").unwrap();

        let ctx = gather(root.to_str().unwrap());
        assert!(ctx.git_diff.contains("committed.txt"));
        assert!(ctx.git_diff.contains("committed change"));
        assert!(ctx.git_diff.contains("untracked.txt"));
        assert!(ctx.git_diff.contains("untracked change"));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn untracked_inventory_survives_large_tracked_diff_budget() {
        let root = std::env::temp_dir().join(format!(
            "brainrouter-review-budget-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("tracked.txt"), "base\n").unwrap();
        git(&root, &["init", "-q"]);
        git(&root, &["config", "user.email", "brainrouter@example.invalid"]);
        git(&root, &["config", "user.name", "Brainrouter Test"]);
        git(&root, &["add", "."]);
        git(&root, &["commit", "-q", "-m", "base"]);

        std::fs::write(root.join("tracked.txt"), "x".repeat(200 * 1024)).unwrap();
        std::fs::write(root.join("important-new.rs"), "pub fn important() {}\n").unwrap();

        let prompt_sized = truncate(load_git_diff(&root), MAX_SECTION_SIZE);
        assert!(prompt_sized.contains("important-new.rs"));
        assert!(prompt_sized.contains("pub fn important()"));

        let _ = std::fs::remove_dir_all(root);
    }
}
