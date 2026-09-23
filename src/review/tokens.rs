//! Token-aware review-prompt budget (design `docs/design/token-aware-review-budget.md`).
//!
//! The review prompt was assembled from sections each byte-capped at 150 KB, with
//! **no** whole-prompt budget against the reviewer model's context window — so it
//! could far exceed a local model's context (8k–32k) and get truncated or rejected
//! at the model. This module budgets the *entire* prompt by an estimated token
//! count to fit `context − output_reserve − margin`, trimming lower-priority
//! evidence in a fixed order while never dropping the task or the review criteria,
//! and signalling when diff/design evidence was lost so the loop can refuse a
//! confident `approved` on truncated evidence.
//!
//! MVP scope (see the design doc): a conservative byte-based `HeuristicCounter`
//! and configured context-window defaults. Exact per-model `n_ctx` from `/props`,
//! a `review.budget` config block, and exact tokenizers (tiktoken-rs / llama.cpp
//! `/tokenize` / huggingface-tokenizers) are documented follow-ons behind the
//! `TokenCounter` trait.

/// Conservative bytes→tokens divisor. For typical ASCII/code (~3–4 bytes/token)
/// this over-estimates, so trimming errs toward *over*-truncation. It is not a
/// universal upper bound — token-dense content (CJK, base64, minified, long digit
/// runs) can fall below 3 bytes/token and be under-counted; the `MARGIN_PERCENT`
/// cushion plus the fact that an overflow surfaces as a model error/escalation
/// (never a confident false `approved`) bound the risk. Exact tokenizers (R6)
/// remove it entirely.
pub const DEFAULT_DIVISOR: usize = 3;
/// Conservative local context-window default when the exact `n_ctx` is unknown.
pub const DEFAULT_LOCAL_CTX: usize = 8192;
/// Cloud context-window default (Manifest models are large).
pub const DEFAULT_CLOUD_CTX: usize = 128_000;
/// Upper bound on the review's output reservation (the prior hardcoded value).
pub const DEFAULT_MAX_OUTPUT: usize = 16_384;
/// Safety margin as a percent of the context window.
pub const MARGIN_PERCENT: usize = 10;

/// Estimates the token count of a piece of text.
pub trait TokenCounter {
    fn count(&self, text: &str) -> usize;
    /// Approximate bytes-per-token this counter assumes, used to convert a token
    /// budget back into a byte budget when trimming. Derived from the counter so
    /// a future exact counter (R6) stays sound rather than assuming `3`.
    fn bytes_per_token_hint(&self) -> usize {
        DEFAULT_DIVISOR
    }
}

/// Conservative byte-based estimator: `ceil(bytes / divisor)`.
pub struct HeuristicCounter {
    pub divisor: usize,
}

impl Default for HeuristicCounter {
    fn default() -> Self {
        HeuristicCounter { divisor: DEFAULT_DIVISOR }
    }
}

impl TokenCounter for HeuristicCounter {
    fn count(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        let divisor = self.divisor.max(1);
        text.len().div_ceil(divisor)
    }
    fn bytes_per_token_hint(&self) -> usize {
        self.divisor.max(1)
    }
}

/// The reviewer's context window, from configured defaults (MVP; exact `/props`
/// `n_ctx` is a follow-on).
pub fn context_window(is_local: bool) -> usize {
    if is_local {
        DEFAULT_LOCAL_CTX
    } else {
        DEFAULT_CLOUD_CTX
    }
}

/// Output reservation for the review response, coupled to the context window so
/// input+output always fits: `min(DEFAULT_MAX_OUTPUT, ctx/4)`. Preserves the
/// prior generous cloud value (16384 on a 128k window) while making a small local
/// window feasible (2048 on 8192).
pub fn output_reserve(ctx: usize) -> usize {
    DEFAULT_MAX_OUTPUT.min(ctx / 4).max(256)
}

/// The kind of a review-prompt section — drives priority and protection, rather
/// than inferring them from rendered headings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionKind {
    Prd,
    ApprovedDesign,
    GitDiff,
    AgentContract,
    TaskDetails,
    SessionHistory,
    Criteria,
    DesignDivergence,
}

impl SectionKind {
    /// Never dropped or trimmed: the task and the criteria the verdict depends on.
    pub fn protected(self) -> bool {
        matches!(
            self,
            SectionKind::TaskDetails | SectionKind::Criteria | SectionKind::DesignDivergence
        )
    }

    /// Evidence whose loss must block a confident `approved` (R4).
    pub fn is_evidence(self) -> bool {
        matches!(self, SectionKind::GitDiff | SectionKind::ApprovedDesign)
    }

    /// Keep-priority: higher is kept longer (dropped last). The diff (the subject
    /// of review) and the approved design are protected above the PRD, the agent
    /// contract, and the session history.
    pub fn keep_priority(self) -> u8 {
        match self {
            SectionKind::GitDiff => 5,
            SectionKind::ApprovedDesign => 4,
            SectionKind::Prd => 3,
            SectionKind::AgentContract => 2,
            SectionKind::SessionHistory => 1,
            // Protected kinds never enter the droppable ranking.
            SectionKind::TaskDetails | SectionKind::Criteria | SectionKind::DesignDivergence => 255,
        }
    }

    /// Canonical render order (position in the final prompt), independent of
    /// keep-priority.
    pub fn render_order(self) -> u8 {
        match self {
            SectionKind::Prd => 0,
            SectionKind::ApprovedDesign => 1,
            SectionKind::GitDiff => 2,
            SectionKind::AgentContract => 3,
            SectionKind::TaskDetails => 4,
            SectionKind::SessionHistory => 5,
            SectionKind::Criteria => 6,
            SectionKind::DesignDivergence => 7,
        }
    }
}

/// One fully-rendered prompt section (`# HEADING\n\n{body}` or a criteria block).
#[derive(Debug, Clone)]
pub struct Section {
    pub kind: SectionKind,
    pub text: String,
    /// True when this section's body was already truncated upstream (the 150 KB
    /// per-section byte cap). Set explicitly at build time — not sniffed from the
    /// text — so a diff that legitimately contains a truncation-warning literal
    /// never false-positives.
    pub upstream_truncated: bool,
}

/// Outcome of budgeting a set of sections.
#[derive(Debug, Clone)]
pub struct BudgetPlan {
    /// Kept (possibly trimmed) sections, in canonical render order.
    pub sections: Vec<Section>,
    /// Diff/design evidence was trimmed, dropped, or already truncated upstream.
    pub truncated_required: bool,
    /// Protected content (task + criteria) alone exceeds the budget ⇒ the loop
    /// must not call the LLM and should terminally escalate.
    pub protected_overflow: bool,
    /// Output-token reservation to use for the request's `max_tokens`.
    pub max_output_tokens: usize,
}

/// Budget `sections` to fit `ctx`. `fixed_overhead` accounts for the system
/// message + inter-section separators the caller adds around these sections.
pub fn plan_budget(
    sections: Vec<Section>,
    ctx: usize,
    counter: &dyn TokenCounter,
    fixed_overhead: usize,
) -> BudgetPlan {
    let max_output = output_reserve(ctx);
    let margin = ctx * MARGIN_PERCENT / 100;
    let available = ctx
        .saturating_sub(max_output)
        .saturating_sub(margin)
        .saturating_sub(fixed_overhead);

    // Any evidence section already truncated upstream (context.rs byte caps).
    let mut truncated_required = sections
        .iter()
        .any(|s| s.kind.is_evidence() && s.upstream_truncated);

    let (protected, mut droppable): (Vec<Section>, Vec<Section>) =
        sections.into_iter().partition(|s| s.kind.protected());

    let protected_cost: usize = protected.iter().map(|s| counter.count(&s.text)).sum();

    // Totality: protected content alone can't fit ⇒ don't dispatch.
    if protected_cost > available {
        let mut kept = protected;
        kept.sort_by_key(|s| s.kind.render_order());
        return BudgetPlan {
            sections: kept,
            truncated_required: true,
            protected_overflow: true,
            max_output_tokens: max_output,
        };
    }

    let mut remaining = available - protected_cost;
    let mut kept = protected;

    // Add droppable sections in keep-priority order (highest first).
    droppable.sort_by(|a, b| b.kind.keep_priority().cmp(&a.kind.keep_priority()));
    for section in droppable {
        let cost = counter.count(&section.text);
        if cost <= remaining {
            remaining -= cost;
            kept.push(section);
            continue;
        }
        // Doesn't fit whole. Trim to the remaining budget if it's worth keeping.
        // Byte budget is derived from the counter, so a non-default counter (R6)
        // stays sound.
        let byte_budget = remaining.saturating_mul(counter.bytes_per_token_hint());
        if byte_budget >= MIN_USEFUL_BYTES {
            let trimmed = trim_section(&section, byte_budget);
            kept.push(Section {
                kind: section.kind,
                text: trimmed,
                upstream_truncated: true,
            });
        }
        if section.kind.is_evidence() {
            truncated_required = true;
        }
        remaining = 0;
        // Everything after this is dropped; still flag dropped evidence.
    }

    kept.sort_by_key(|s| s.kind.render_order());
    BudgetPlan {
        sections: kept,
        truncated_required,
        protected_overflow: false,
        max_output_tokens: max_output,
    }
}

/// Minimum trimmed section size worth keeping (below this, drop entirely).
const MIN_USEFUL_BYTES: usize = 512;

/// Trim a rendered section to ~`byte_budget` bytes. The git diff keeps head+tail
/// (the diff's start and end are both informative); other sections keep the head.
fn trim_section(section: &Section, byte_budget: usize) -> String {
    let marker = "\n\n…[budget-truncated]…\n\n";
    if section.text.len() <= byte_budget {
        return section.text.clone();
    }
    let budget = byte_budget.saturating_sub(marker.len());
    if section.kind == SectionKind::GitDiff && budget > 64 {
        let head_len = budget * 7 / 10;
        let tail_len = budget - head_len;
        let head_end = floor_char_boundary(&section.text, head_len);
        let tail_start = ceil_char_boundary(&section.text, section.text.len() - tail_len);
        let mut out = String::with_capacity(budget + marker.len());
        out.push_str(&section.text[..head_end]);
        out.push_str(marker);
        out.push_str(&section.text[tail_start..]);
        out
    } else {
        let head_end = floor_char_boundary(&section.text, budget);
        let mut out = section.text[..head_end].to_string();
        out.push_str(marker);
        out
    }
}

fn floor_char_boundary(s: &str, mut idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn ceil_char_boundary(s: &str, mut idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heur() -> HeuristicCounter {
        HeuristicCounter::default()
    }

    #[test]
    fn heuristic_is_conservative_and_nonzero() {
        let c = heur();
        assert_eq!(c.count(""), 0);
        assert_eq!(c.count("abc"), 1); // ceil(3/3)
        assert_eq!(c.count("abcd"), 2); // ceil(4/3)
        // Over-estimates vs a ~4 bytes/token reference (never under-counts).
        let text = "a".repeat(400);
        assert!(c.count(&text) >= 100, "should over-estimate 400 bytes");
    }

    #[test]
    fn output_reserve_couples_to_context() {
        assert_eq!(output_reserve(128_000), DEFAULT_MAX_OUTPUT); // cloud unchanged
        assert_eq!(output_reserve(8192), 2048); // local made feasible
        assert!(output_reserve(1000) >= 250); // never zero
    }

    fn sec(kind: SectionKind, body: &str) -> Section {
        Section { kind, text: format!("# {:?}\n\n{}", kind, body), upstream_truncated: false }
    }

    #[test]
    fn fits_unchanged_when_under_budget() {
        let sections = vec![
            sec(SectionKind::GitDiff, "small diff"),
            sec(SectionKind::TaskDetails, "task"),
            sec(SectionKind::Criteria, "criteria"),
        ];
        let plan = plan_budget(sections.clone(), 128_000, &heur(), 50);
        assert!(!plan.truncated_required);
        assert!(!plan.protected_overflow);
        assert_eq!(plan.sections.len(), 3);
    }

    #[test]
    fn drops_lowest_priority_first_and_protects_task_criteria() {
        // Small context forces dropping; history should go before the diff.
        let big = "x".repeat(20_000);
        let sections = vec![
            sec(SectionKind::SessionHistory, &big),
            sec(SectionKind::GitDiff, &big),
            sec(SectionKind::TaskDetails, "task"),
            sec(SectionKind::Criteria, "criteria"),
        ];
        let plan = plan_budget(sections, 8192, &heur(), 50);
        let kinds: Vec<_> = plan.sections.iter().map(|s| s.kind).collect();
        assert!(kinds.contains(&SectionKind::TaskDetails));
        assert!(kinds.contains(&SectionKind::Criteria));
        // History (lowest priority) dropped before the diff evidence.
        assert!(!kinds.contains(&SectionKind::SessionHistory));
        assert!(plan.truncated_required, "diff trimmed ⇒ evidence flag");
    }

    #[test]
    fn protected_overflow_when_task_criteria_alone_exceed_budget() {
        let huge = "x".repeat(100_000);
        let sections = vec![
            sec(SectionKind::TaskDetails, &huge),
            sec(SectionKind::Criteria, &huge),
        ];
        let plan = plan_budget(sections, 8192, &heur(), 50);
        assert!(plan.protected_overflow);
        assert!(plan.truncated_required);
    }

    #[test]
    fn upstream_truncation_flag_sets_the_evidence_flag() {
        let mut diff = sec(SectionKind::GitDiff, "some diff that mentions [WARNING: truncated in its own text");
        diff.upstream_truncated = true;
        let sections = vec![
            diff,
            sec(SectionKind::TaskDetails, "task"),
            sec(SectionKind::Criteria, "criteria"),
        ];
        let plan = plan_budget(sections, 128_000, &heur(), 50);
        assert!(plan.truncated_required, "upstream-truncated diff must flag");
        assert!(!plan.protected_overflow);
    }

    #[test]
    fn a_diff_literally_containing_the_marker_but_not_truncated_does_not_flag() {
        // Regression: string-sniffing would false-positive here; the explicit
        // flag must not.
        let diff = sec(SectionKind::GitDiff, "+ let warning = \"[WARNING: truncated to {}KB\";");
        let sections = vec![
            diff,
            sec(SectionKind::TaskDetails, "task"),
            sec(SectionKind::Criteria, "criteria"),
        ];
        let plan = plan_budget(sections, 128_000, &heur(), 50);
        assert!(!plan.truncated_required, "a non-truncated diff must not flag on literal text");
    }

    #[test]
    fn design_absence_alone_does_not_flag() {
        // No ApprovedDesign section present, integration off: not evidence loss.
        let sections = vec![
            sec(SectionKind::GitDiff, "diff"),
            sec(SectionKind::TaskDetails, "task"),
            sec(SectionKind::Criteria, "criteria"),
        ];
        let plan = plan_budget(sections, 128_000, &heur(), 50);
        assert!(!plan.truncated_required);
    }

    #[test]
    fn git_diff_trim_keeps_head_and_tail() {
        let body = format!("HEADSTART{}TAILEND", "m".repeat(40_000));
        let section = sec(SectionKind::GitDiff, &body);
        let trimmed = trim_section(&section, 4096);
        assert!(trimmed.len() <= 4096 + 64);
        assert!(trimmed.contains("HEADSTART"));
        assert!(trimmed.contains("TAILEND"));
        assert!(trimmed.contains("budget-truncated"));
    }
}
