//! Safe loading and approval verification of a HankNDory design document
//! (design G1 / hardening H5). This module is the *integrity boundary* for the
//! design-aware reviewer: it decides whether a design document may be trusted as
//! the source of truth a code review is judged against.
//!
//! Two guarantees matter here and both are enforced by this module, never by the
//! prompt text:
//!
//! 1. **Path safety.** The document is opened through a TOCTOU-resistant path so
//!    a symlink, a `..` escape, or a swapped component cannot make the reviewer
//!    read a file outside `"<git-root>/docs/design/"`. On Linux this is a
//!    `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)` anchored at the git-root
//!    directory FD, operating on that FD only. On macOS it is `canonicalize` +
//!    a `docs/design/` prefix assertion + `O_NOFOLLOW` on the final open (a
//!    slightly weaker guarantee with a small canonicalize-then-open window,
//!    acceptable for a single-user local repo). Every other OS fails closed with
//!    [`DesignDocError::PlatformUnsupported`].
//!
//! 2. **Approval binding.** A design is "approved" only when the document's own
//!    grammar says so (`## Status` → `**Workflow state:** approved-for-implementation`
//!    plus a non-empty `## Human approval`) **and** a separate local
//!    approved-record ([`approvals_path`]) holds the exact SHA-256 of the current
//!    document. Editing the document changes its hash and silently invalidates
//!    approval until it is re-approved. This makes "approved" bind to *content*,
//!    not merely to a line of prose an author could add in the same commit as a
//!    divergence.
//!
//! This module is deliberately self-contained and side-effect free except for
//! the approved-record read/write helpers; it is wired into the review loop by a
//! separate change.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

/// Hard cap on a design document we will load into a review prompt.
const MAX_DESIGN_BYTES: u64 = 512 * 1024;
/// The only directory a design document may live in, relative to the git root.
const DESIGN_SUBDIR: [&str; 2] = ["docs", "design"];
/// Approved-record file name, kept beside the other review state files.
const APPROVALS_FILE: &str = "review_approvals.json";
/// The one accepted "no human reviewer was available" disposition (HankNDory
/// rule 7 lets a solo author record that human review was skipped by judgment).
const SKIP_PHRASE: &str = "human review skipped by user judgment";
/// The workflow-state token that authorizes implementation.
const APPROVED_STATE: &str = "approved-for-implementation";

/// Why a design document could not be loaded, or is not approved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DesignDocError {
    /// The host OS has no safe-open implementation (non-Linux / non-macOS), or
    /// the kernel lacks `openat2`. The reviewer must fail closed.
    PlatformUnsupported,
    /// No git root, missing/badly-located file, oversize, non-regular, or
    /// non-UTF-8. The design cannot be used; distinct from an I/O fault.
    Unavailable(String),
    /// An I/O fault while opening or reading through the safe path.
    Io(String),
}

impl std::fmt::Display for DesignDocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DesignDocError::PlatformUnsupported => {
                write!(f, "design-aware review is unsupported on this platform")
            }
            DesignDocError::Unavailable(m) => write!(f, "design document unavailable: {m}"),
            DesignDocError::Io(m) => write!(f, "design document I/O error: {m}"),
        }
    }
}

/// A design document loaded through the platform safe-open path, with its
/// content hash and the approval signals parsed from its own grammar.
#[derive(Debug, Clone)]
pub struct LoadedDesignDoc {
    /// Repo-relative path, forward-slash normalized (e.g. `docs/design/x.md`).
    pub repo_rel_path: String,
    /// Raw UTF-8 document content.
    pub content: String,
    /// Lowercase hex SHA-256 of `content`.
    pub sha256: String,
    /// The token after `**Workflow state:**` in the first `## Status` section.
    pub workflow_state: Option<String>,
    /// Whether the workflow state is exactly `approved-for-implementation`.
    pub approved_for_impl: bool,
    /// Whether the first `## Human approval` section carries an approval
    /// disposition (or the explicit skip-by-judgment phrase).
    pub human_approved: bool,
    /// Best-effort `vN` version parsed from the approval/status sections.
    pub approved_version: Option<String>,
}

impl LoadedDesignDoc {
    /// The document's *own* claim of approval, before binding to a hash record.
    pub fn doc_marks_approved(&self) -> bool {
        self.approved_for_impl && self.human_approved
    }
}

/// Outcome of binding a loaded document to the local approved-record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalOutcome {
    /// Document is approved and its current hash matches the approved-record.
    Approved { version: Option<String> },
    /// Not approved; the human-readable reason is safe to surface.
    NotApproved(String),
}

/// A single approved-record entry, binding a design path to the exact content
/// hash that was approved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalRecord {
    /// Lowercase hex SHA-256 the approval was granted for.
    pub sha256: String,
    #[serde(default)]
    pub approved_version: Option<String>,
    #[serde(default)]
    pub approver: String,
    #[serde(default)]
    pub approved_at: String,
}

/// Map of repo-relative design path → approved-record.
pub type ApprovalMap = BTreeMap<String, ApprovalRecord>;

/// Load and parse a design document by repo-relative path, through the platform
/// safe-open path. The path must be relative, symlink/`..`-free, under
/// `docs/design/`, a `.md` file, a regular UTF-8 file no larger than 512 KiB.
pub fn load_design_doc(project_dir: &str, rel_path: &str) -> Result<LoadedDesignDoc, DesignDocError> {
    let rel = validate_rel_path(rel_path)?;
    let root = git_root(project_dir).ok_or_else(|| {
        DesignDocError::Unavailable("no git repository root for the design lookup".into())
    })?;

    let file = open_beneath(&root, &rel)?;
    let meta = file.metadata().map_err(|e| DesignDocError::Io(e.to_string()))?;
    if !meta.file_type().is_file() {
        return Err(DesignDocError::Unavailable(
            "design path is not a regular file".into(),
        ));
    }
    if meta.len() > MAX_DESIGN_BYTES {
        return Err(DesignDocError::Unavailable(format!(
            "design document exceeds the {MAX_DESIGN_BYTES}-byte cap"
        )));
    }

    // Bound the read independently of the fstat size in case the file grows
    // between fstat and read.
    let mut bytes = Vec::new();
    (&file)
        .take(MAX_DESIGN_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| DesignDocError::Io(e.to_string()))?;
    if bytes.len() as u64 > MAX_DESIGN_BYTES {
        return Err(DesignDocError::Unavailable(
            "design document exceeds the size cap".into(),
        ));
    }
    let content = String::from_utf8(bytes)
        .map_err(|_| DesignDocError::Unavailable("design document is not valid UTF-8".into()))?;

    let sha256 = sha256_hex(content.as_bytes());
    let parsed = parse_approval(&content);
    Ok(LoadedDesignDoc {
        repo_rel_path: rel.to_string_lossy().replace('\\', "/"),
        content,
        sha256,
        workflow_state: parsed.workflow_state,
        approved_for_impl: parsed.approved_for_impl,
        human_approved: parsed.human_approved,
        approved_version: parsed.approved_version,
    })
}

/// Decide approval by combining the document's own grammar with the local
/// approved-record. Fail-closed: absent record, or a hash mismatch, is
/// `NotApproved`.
pub fn evaluate_approval(doc: &LoadedDesignDoc, record: Option<&ApprovalRecord>) -> ApprovalOutcome {
    if !doc.doc_marks_approved() {
        return ApprovalOutcome::NotApproved(
            "design is not marked approved-for-implementation with a human approval".into(),
        );
    }
    match record {
        None => ApprovalOutcome::NotApproved(
            "no approval record for this design; approve the current version first".into(),
        ),
        Some(r) if r.sha256 != doc.sha256 => ApprovalOutcome::NotApproved(
            "design changed since it was approved; re-approve the current version".into(),
        ),
        Some(r) => ApprovalOutcome::Approved {
            version: r.approved_version.clone(),
        },
    }
}

// ── Path validation and git-root discovery ──────────────────────────────────

/// Validate a caller-supplied repo-relative design path. Rejects absolute
/// paths, any non-normal component (`.`, `..`, root, prefix), anything not under
/// `docs/design/`, and non-`.md` files.
fn validate_rel_path(rel: &str) -> Result<PathBuf, DesignDocError> {
    let path = Path::new(rel);
    if path.is_absolute() {
        return Err(DesignDocError::Unavailable(
            "design path must be repo-relative".into(),
        ));
    }
    let comps: Vec<Component> = path.components().collect();
    for c in &comps {
        if !matches!(c, Component::Normal(_)) {
            return Err(DesignDocError::Unavailable(format!(
                "illegal component in design path '{rel}'"
            )));
        }
    }
    if comps.len() < 3
        || comps[0].as_os_str() != DESIGN_SUBDIR[0]
        || comps[1].as_os_str() != DESIGN_SUBDIR[1]
    {
        return Err(DesignDocError::Unavailable(
            "design document must live under docs/design/".into(),
        ));
    }
    if path.extension().and_then(|e| e.to_str()) != Some("md") {
        return Err(DesignDocError::Unavailable(
            "design document must be a .md file".into(),
        ));
    }
    Ok(path.to_path_buf())
}

/// Resolve the git top-level directory for `project_dir`. Returns `None` when
/// there is no real git root — the design loader requires one and never falls
/// back to an arbitrary cwd (unlike `review::context::resolve_project_root`).
fn git_root(project_dir: &str) -> Option<PathBuf> {
    let start = if project_dir.is_empty() {
        std::env::current_dir().ok()?
    } else {
        PathBuf::from(project_dir)
    };
    let out = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(&start)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let root = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if root.is_empty() {
        None
    } else {
        Some(PathBuf::from(root))
    }
}

// ── Platform safe-open ───────────────────────────────────────────────────────

/// Linux: anchor at the git-root directory FD, then `openat2` the relative path
/// with `RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS` so no component may be a symlink
/// and resolution can never escape the root. `O_NONBLOCK` avoids blocking on a
/// pathological FIFO; the caller rejects non-regular files after `fstat`.
#[cfg(target_os = "linux")]
fn open_beneath(root: &Path, rel: &Path) -> Result<std::fs::File, DesignDocError> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::io::{AsRawFd, FromRawFd};

    let dir = std::fs::File::open(root)
        .map_err(|e| DesignDocError::Unavailable(format!("cannot open repo root: {e}")))?;
    if !dir
        .metadata()
        .map_err(|e| DesignDocError::Io(e.to_string()))?
        .is_dir()
    {
        return Err(DesignDocError::Unavailable(
            "repo root is not a directory".into(),
        ));
    }

    let rel_c = CString::new(rel.as_os_str().as_bytes())
        .map_err(|_| DesignDocError::Unavailable("design path contains a NUL byte".into()))?;
    // `open_how` is `#[non_exhaustive]` in libc, so it must be zero-initialized
    // (which also zeroes padding and any future fields) and then filled in,
    // rather than built with a struct literal.
    let mut how: libc::open_how = unsafe { std::mem::zeroed() };
    how.flags = (libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NONBLOCK) as u64;
    how.resolve = libc::RESOLVE_BENEATH | libc::RESOLVE_NO_SYMLINKS;
    // SAFETY: `dir` owns a valid dir FD for the whole call; `rel_c` is a valid
    // NUL-terminated C string that outlives the syscall; `how` is a correctly
    // sized, zero-initialized `open_how`. We take ownership of any returned FD
    // immediately. Integer args are widened to the syscall register width.
    let fd = unsafe {
        libc::syscall(
            libc::SYS_openat2,
            dir.as_raw_fd() as libc::c_long,
            rel_c.as_ptr(),
            &how as *const libc::open_how,
            std::mem::size_of::<libc::open_how>() as libc::c_long,
        )
    };
    if fd < 0 {
        let err = std::io::Error::last_os_error();
        // Kernels older than 5.6 lack openat2 — fail closed as unsupported.
        if err.raw_os_error() == Some(libc::ENOSYS) {
            return Err(DesignDocError::PlatformUnsupported);
        }
        return Err(DesignDocError::Unavailable(format!(
            "safe-open of the design document was denied: {err}"
        )));
    }
    // SAFETY: `fd` is a fresh, owned, valid FD returned by openat2.
    Ok(unsafe { std::fs::File::from_raw_fd(fd as i32) })
}

/// macOS: canonicalize the root and the target, assert the target is under the
/// canonical `docs/design/`, then open with `O_NOFOLLOW` on the final
/// component. Weaker than Linux (a small canonicalize-then-open window) but safe
/// for a single-user local repo.
#[cfg(target_os = "macos")]
fn open_beneath(root: &Path, rel: &Path) -> Result<std::fs::File, DesignDocError> {
    use std::os::unix::fs::OpenOptionsExt;

    let canon_root = root
        .canonicalize()
        .map_err(|e| DesignDocError::Unavailable(format!("cannot canonicalize repo root: {e}")))?;
    let design_dir = canon_root
        .join(DESIGN_SUBDIR[0])
        .join(DESIGN_SUBDIR[1])
        .canonicalize()
        .map_err(|e| DesignDocError::Unavailable(format!("cannot canonicalize docs/design: {e}")))?;
    let canon_target = canon_root
        .join(rel)
        .canonicalize()
        .map_err(|e| DesignDocError::Unavailable(format!("cannot resolve design document: {e}")))?;
    if !canon_target.starts_with(&design_dir) {
        return Err(DesignDocError::Unavailable(
            "design document escapes docs/design/".into(),
        ));
    }
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&canon_target)
        .map_err(|e| DesignDocError::Io(e.to_string()))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn open_beneath(_root: &Path, _rel: &Path) -> Result<std::fs::File, DesignDocError> {
    Err(DesignDocError::PlatformUnsupported)
}

// ── Approval grammar parsing ─────────────────────────────────────────────────

struct ParsedApproval {
    workflow_state: Option<String>,
    approved_for_impl: bool,
    human_approved: bool,
    approved_version: Option<String>,
}

/// Parse the first `## Status` and first `## Human approval` H2 sections,
/// ignoring fenced code blocks and duplicate later sections so a code-fenced or
/// second "approved" heading cannot forge approval.
fn parse_approval(content: &str) -> ParsedApproval {
    let mut fence: Option<(char, usize)> = None;
    let mut section: Option<String> = None;
    let mut status: Vec<&str> = Vec::new();
    let mut human: Vec<&str> = Vec::new();
    let mut status_captured = false;
    let mut human_captured = false;

    for line in content.lines() {
        let trimmed = line.trim_start();
        // Track fenced code blocks by fence char + run length so a mismatched or
        // shorter inner fence (``` vs ~~~) cannot appear to close an open block
        // and expose a forged heading as live text.
        if let Some((ch, len)) = fence_marker(trimmed) {
            match fence {
                None => fence = Some((ch, len)),
                Some((open_ch, open_len)) if ch == open_ch && len >= open_len => fence = None,
                Some(_) => {}
            }
            continue;
        }
        if fence.is_some() {
            continue;
        }
        if let Some(heading) = trimmed.strip_prefix("## ") {
            // Leaving a section: freeze the first non-empty occurrence.
            match section.as_deref() {
                Some("status") if !status.is_empty() => status_captured = true,
                Some("human approval") if !human.is_empty() => human_captured = true,
                _ => {}
            }
            section = Some(heading.trim().to_lowercase());
            continue;
        }
        match section.as_deref() {
            Some("status") if !status_captured => status.push(line),
            Some("human approval") if !human_captured => human.push(line),
            _ => {}
        }
    }

    let status_text = status.join("\n");
    let human_text = human.join("\n");
    let workflow_state = extract_workflow_state(&status_text);
    let approved_for_impl = workflow_state.as_deref() == Some(APPROVED_STATE);
    let human_approved = has_approval_disposition(&human_text);
    let approved_version =
        extract_version(&human_text).or_else(|| extract_version(&status_text));

    ParsedApproval {
        workflow_state,
        approved_for_impl,
        human_approved,
        approved_version,
    }
}

/// A leading Markdown code fence (`≥3` backticks or tildes), as `(char, len)`.
fn fence_marker(trimmed: &str) -> Option<(char, usize)> {
    let ch = trimmed.chars().next()?;
    if ch != '`' && ch != '~' {
        return None;
    }
    let len = trimmed.chars().take_while(|&c| c == ch).count();
    if len >= 3 {
        Some((ch, len))
    } else {
        None
    }
}

/// Whether a `## Human approval` section carries an affirmative approval
/// disposition. Word-boundary aware so "not approved", "unapproved", and
/// "disapproved" do NOT count as approval (a bare substring test would).
fn has_approval_disposition(human_text: &str) -> bool {
    let lower = human_text.to_lowercase();
    if lower.contains(SKIP_PHRASE) {
        return true;
    }
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    for (i, tok) in tokens.iter().enumerate() {
        if *tok == "approved" {
            let prev = if i > 0 { tokens[i - 1] } else { "" };
            if !matches!(prev, "not" | "never" | "un" | "dis" | "yet") {
                return true;
            }
        }
    }
    false
}

/// Extract the token following `Workflow state:` (markdown emphasis stripped).
fn extract_workflow_state(status_text: &str) -> Option<String> {
    for line in status_text.lines() {
        if let Some(idx) = line.find("Workflow state:") {
            let rest = &line[idx + "Workflow state:".len()..];
            let cleaned = rest.replace('*', "");
            if let Some(token) = cleaned.split_whitespace().next() {
                let token = token.trim_matches(|c: char| c == '`' || c == '.' || c == ',');
                if !token.is_empty() {
                    return Some(token.to_string());
                }
            }
        }
    }
    None
}

/// First `vN` token (v followed by digits at a word boundary), best effort.
fn extract_version(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let is_v = bytes[i] == b'v' || bytes[i] == b'V';
        let boundary = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
        if is_v && boundary && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            return Some(format!("v{}", &text[i + 1..j]));
        }
        i += 1;
    }
    None
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(out, "{b:02x}");
    }
    out
}

// ── Approved-record store (review_approvals.json) ────────────────────────────

/// Canonical path for the approved-record file (beside `review_runtime_state.json`).
///
/// Note: entries are keyed by repo-relative design path only, so two distinct
/// repositories that both hold an identically-named, byte-identical, approved
/// design doc would share one record. The SHA-256 binding neutralizes the unsafe
/// case (different content ⇒ mismatch ⇒ fail closed); this is acceptable under
/// the single-primary-repo assumption of a local single-user host.
pub fn approvals_path() -> PathBuf {
    crate::config::default_config_path().with_file_name(APPROVALS_FILE)
}

/// Read the approved-record map. Absent, unreadable, or corrupt ⇒ empty map
/// (fail-closed: no bindings ⇒ nothing is approved). Never panics.
pub fn load_approvals(path: &Path) -> ApprovalMap {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            tracing::warn!(path = %path.display(), error = %e, "Ignoring corrupt review_approvals.json");
            ApprovalMap::new()
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => ApprovalMap::new(),
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "Could not read review_approvals.json");
            ApprovalMap::new()
        }
    }
}

/// Strict read for the read-modify-write path: absent ⇒ empty, but an
/// unreadable or corrupt file is an error so a rewrite cannot silently discard
/// existing approvals on a transient fault.
fn read_approvals_strict(path: &Path) -> std::io::Result<ApprovalMap> {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(ApprovalMap::new()),
        Err(e) => Err(e),
    }
}

/// Bind `repo_rel_path` to `sha256` in the approved-record (read-modify-write,
/// atomic replace, owner-only perms, parent fsync).
pub fn record_approval(
    path: &Path,
    repo_rel_path: &str,
    sha256: &str,
    approved_version: Option<String>,
    approver: &str,
) -> std::io::Result<()> {
    let mut map = read_approvals_strict(path)?;
    map.insert(
        repo_rel_path.to_string(),
        ApprovalRecord {
            sha256: sha256.to_string(),
            approved_version,
            approver: approver.to_string(),
            approved_at: chrono::Utc::now().to_rfc3339(),
        },
    );
    write_approvals(path, &map)
}

/// Remove any binding for `repo_rel_path` (revoke approval). No-op if absent.
pub fn remove_approval(path: &Path, repo_rel_path: &str) -> std::io::Result<()> {
    let mut map = read_approvals_strict(path)?;
    if map.remove(repo_rel_path).is_some() {
        write_approvals(path, &map)?;
    }
    Ok(())
}

fn write_approvals(path: &Path, map: &ApprovalMap) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;

    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "approvals path needs a parent directory",
        )
    })?;
    std::fs::create_dir_all(parent)?;
    let bytes = serde_json::to_vec_pretty(map)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let tmp = parent.join(format!(".review_approvals-{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        std::fs::rename(&tmp, path)?;
        // Best-effort durability of the rename in the directory entry.
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
        Ok::<_, std::io::Error>(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Path validation ─────────────────────────────────────────────────────

    #[test]
    fn rejects_absolute_and_traversal_and_wrong_location() {
        assert!(validate_rel_path("/etc/passwd").is_err());
        assert!(validate_rel_path("docs/design/../../etc/passwd").is_err());
        assert!(validate_rel_path("./docs/design/x.md").is_err());
        assert!(validate_rel_path("src/design/x.md").is_err());
        assert!(validate_rel_path("docs/notdesign/x.md").is_err());
        assert!(validate_rel_path("docs/design/x.txt").is_err());
        assert!(validate_rel_path("docs/design/x.md").is_ok());
        assert!(validate_rel_path("docs/design/sub/x.md").is_ok());
    }

    // ── Approval grammar ─────────────────────────────────────────────────────

    fn approved_doc() -> String {
        "# Feature\n\n## Status\n\n- **Workflow state:** approved-for-implementation\n\n\
         ## Human approval\n\nApproved by the owner 2026-01-01. Approved version: **v6**.\n"
            .to_string()
    }

    #[test]
    fn parses_an_approved_document() {
        let p = parse_approval(&approved_doc());
        assert_eq!(p.workflow_state.as_deref(), Some("approved-for-implementation"));
        assert!(p.approved_for_impl);
        assert!(p.human_approved);
        assert_eq!(p.approved_version.as_deref(), Some("v6"));
    }

    #[test]
    fn drafting_document_is_not_approved() {
        let doc = "## Status\n\n- **Workflow state:** drafting\n\n## Human approval\n\nPending.\n";
        let p = parse_approval(doc);
        assert!(!p.approved_for_impl);
        assert!(!p.human_approved);
    }

    #[test]
    fn skip_by_judgment_counts_as_human_approved() {
        let doc = "## Status\n\n**Workflow state:** approved-for-implementation\n\n\
                   ## Human approval\n\nHuman review skipped by user judgment.\n";
        let p = parse_approval(doc);
        assert!(p.approved_for_impl);
        assert!(p.human_approved);
    }

    #[test]
    fn fenced_fake_approval_is_ignored() {
        let doc = "## Intro\n\n```\n## Status\n**Workflow state:** approved-for-implementation\n\
                   ## Human approval\nApproved.\n```\n\nJust an example above.\n";
        let p = parse_approval(doc);
        assert!(!p.approved_for_impl);
        assert!(!p.human_approved);
    }

    #[test]
    fn only_the_first_status_section_wins() {
        let doc = "## Status\n\n**Workflow state:** drafting\n\n\
                   ## Body\n\ntext\n\n## Status\n\n**Workflow state:** approved-for-implementation\n";
        let p = parse_approval(doc);
        // The first Status section is authoritative → drafting, not approved.
        assert_eq!(p.workflow_state.as_deref(), Some("drafting"));
        assert!(!p.approved_for_impl);
    }

    #[test]
    fn negated_disposition_is_not_human_approved() {
        for body in ["Not approved yet.", "disapproved", "unapproved", "Status: not approved."] {
            let doc = format!(
                "## Status\n\n**Workflow state:** approved-for-implementation\n\n## Human approval\n\n{body}\n"
            );
            assert!(!parse_approval(&doc).human_approved, "must NOT approve: {body}");
        }
        for body in ["Approved by the owner.", "approved 2026-09-18", "Release: approved by delegation."] {
            let doc = format!(
                "## Status\n\n**Workflow state:** approved-for-implementation\n\n## Human approval\n\n{body}\n"
            );
            assert!(parse_approval(&doc).human_approved, "must approve: {body}");
        }
    }

    #[test]
    fn mixed_fence_does_not_expose_a_forged_approval() {
        // Opens with ```; an inner ~~~ must NOT close it, so the fake headings
        // inside stay inert and no approval is parsed.
        let doc = "## Intro\n\n```\nexample\n~~~\n## Status\n\
                   **Workflow state:** approved-for-implementation\n## Human approval\nApproved\n```\n\nreal\n";
        let p = parse_approval(doc);
        assert!(!p.approved_for_impl);
        assert!(!p.human_approved);
    }

    // ── evaluate_approval binding ────────────────────────────────────────────

    fn loaded(content: &str) -> LoadedDesignDoc {
        let parsed = parse_approval(content);
        LoadedDesignDoc {
            repo_rel_path: "docs/design/x.md".into(),
            content: content.to_string(),
            sha256: sha256_hex(content.as_bytes()),
            workflow_state: parsed.workflow_state,
            approved_for_impl: parsed.approved_for_impl,
            human_approved: parsed.human_approved,
            approved_version: parsed.approved_version,
        }
    }

    #[test]
    fn approval_requires_a_matching_hash_record() {
        let doc = loaded(&approved_doc());
        // No record ⇒ not approved.
        assert!(matches!(
            evaluate_approval(&doc, None),
            ApprovalOutcome::NotApproved(_)
        ));
        // Matching record ⇒ approved.
        let good = ApprovalRecord {
            sha256: doc.sha256.clone(),
            approved_version: Some("v6".into()),
            approver: "owner".into(),
            approved_at: "2026-01-01T00:00:00Z".into(),
        };
        assert_eq!(
            evaluate_approval(&doc, Some(&good)),
            ApprovalOutcome::Approved { version: Some("v6".into()) }
        );
        // Stale record (different hash) ⇒ not approved.
        let stale = ApprovalRecord {
            sha256: "deadbeef".into(),
            ..good.clone()
        };
        assert!(matches!(
            evaluate_approval(&doc, Some(&stale)),
            ApprovalOutcome::NotApproved(_)
        ));
    }

    #[test]
    fn unapproved_doc_is_never_approved_even_with_a_matching_record() {
        let doc = loaded("## Status\n\n**Workflow state:** drafting\n\n## Human approval\n\nno\n");
        let rec = ApprovalRecord {
            sha256: doc.sha256.clone(),
            approved_version: None,
            approver: "x".into(),
            approved_at: "t".into(),
        };
        assert!(matches!(
            evaluate_approval(&doc, Some(&rec)),
            ApprovalOutcome::NotApproved(_)
        ));
    }

    // ── Approved-record store round-trip ─────────────────────────────────────

    #[test]
    fn approval_record_round_trips_and_revokes() {
        let dir = std::env::temp_dir().join(format!("br-approvals-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(APPROVALS_FILE);

        assert!(load_approvals(&path).is_empty());
        record_approval(&path, "docs/design/x.md", "abc123", Some("v6".into()), "owner").unwrap();
        let map = load_approvals(&path);
        assert_eq!(map.get("docs/design/x.md").unwrap().sha256, "abc123");
        assert_eq!(
            map.get("docs/design/x.md").unwrap().approved_version.as_deref(),
            Some("v6")
        );

        remove_approval(&path, "docs/design/x.md").unwrap();
        assert!(!load_approvals(&path).contains_key("docs/design/x.md"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_approvals_file_reads_as_empty() {
        let dir = std::env::temp_dir().join(format!("br-approvals-bad-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(APPROVALS_FILE);
        std::fs::write(&path, b"{ not json").unwrap();
        assert!(load_approvals(&path).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn record_approval_refuses_to_overwrite_a_corrupt_store() {
        let dir = std::env::temp_dir().join(format!("br-approvals-corrupt-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(APPROVALS_FILE);
        std::fs::write(&path, b"{ not json").unwrap();
        // The RMW path must error rather than clobber existing (unreadable) data.
        assert!(record_approval(&path, "docs/design/x.md", "abc", None, "o").is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"{ not json");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Safe-open through a real temporary git repo ──────────────────────────

    #[cfg(unix)]
    fn init_repo() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("br-design-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("docs").join("design")).unwrap();
        let ok = Command::new("git")
            .args(["init", "-q"])
            .current_dir(&dir)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        assert!(ok, "git init failed in {}", dir.display());
        dir
    }

    #[cfg(unix)]
    #[test]
    fn loads_a_real_document_and_hashes_it() {
        let dir = init_repo();
        let body = approved_doc();
        std::fs::write(dir.join("docs/design/plan.md"), &body).unwrap();

        let doc = load_design_doc(dir.to_str().unwrap(), "docs/design/plan.md").unwrap();
        assert_eq!(doc.repo_rel_path, "docs/design/plan.md");
        assert!(doc.doc_marks_approved());
        assert_eq!(doc.sha256, sha256_hex(body.as_bytes()));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_a_symlinked_final_component_escaping_the_tree() {
        let dir = init_repo();
        // secret outside docs/design/
        let secret = dir.join("secret.txt");
        std::fs::write(&secret, b"TOP SECRET").unwrap();
        // docs/design/evil.md -> ../../secret.txt
        let link = dir.join("docs/design/evil.md");
        std::os::unix::fs::symlink("../../secret.txt", &link).unwrap();

        let result = load_design_doc(dir.to_str().unwrap(), "docs/design/evil.md");
        assert!(result.is_err(), "symlink escape must be refused: {result:?}");
        // Must not have read the secret.
        if let Ok(doc) = &result {
            assert!(!doc.content.contains("TOP SECRET"));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn missing_document_is_unavailable() {
        let dir = init_repo();
        let result = load_design_doc(dir.to_str().unwrap(), "docs/design/nope.md");
        assert!(matches!(result, Err(DesignDocError::Unavailable(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn oversize_document_is_rejected() {
        let dir = init_repo();
        let big = vec![b'a'; (MAX_DESIGN_BYTES + 10) as usize];
        std::fs::write(dir.join("docs/design/big.md"), &big).unwrap();
        let result = load_design_doc(dir.to_str().unwrap(), "docs/design/big.md");
        assert!(matches!(result, Err(DesignDocError::Unavailable(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn a_directory_is_not_a_regular_file() {
        let dir = init_repo();
        std::fs::create_dir_all(dir.join("docs/design/adir.md")).unwrap();
        let result = load_design_doc(dir.to_str().unwrap(), "docs/design/adir.md");
        assert!(matches!(result, Err(DesignDocError::Unavailable(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn non_git_directory_is_unavailable() {
        let dir = std::env::temp_dir().join(format!("br-nogit-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("docs").join("design")).unwrap();
        std::fs::write(dir.join("docs/design/x.md"), b"## Status\n").unwrap();
        // No `git init` here: git_root must return None.
        let result = load_design_doc(dir.to_str().unwrap(), "docs/design/x.md");
        assert!(matches!(result, Err(DesignDocError::Unavailable(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
