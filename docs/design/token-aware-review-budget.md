# Token-aware review prompt budget

## Status

- **Workflow state:** readiness-blocked → v2 closes the round-1 Dory blockers below; implemented under the user's explicit "a and b" approval with the Step-9 mean code review + tests as the correctness gate. ("a and b" is feature approval, not the post-readiness gate.)
- **Change classification:** **standard.** Changes reviewer evidence selection and approval safety (a review can now terminally escalate on truncated evidence), couples the request's `max_tokens` to the budget, and adds a config block. Not trivial.
- **Supersedes H6.** The H6 spec in `hankndory-brainrouter-integration.md` carries two non-identical value sets (≈line 95: local ctx 8192 / reserve 2048 / bytes÷4; ≈line 122: 4096 / bytes÷3 / explicit quotas / `blocked:truncated_evidence`). **This doc is the single source of truth for the budget and supersedes both**, using the conservative **bytes÷3** estimate + the R1–R5 values. H6's *dynamic-language* half already shipped (`prompt.rs`, Dory Round 13); only the budget half was missing.
- **Revisions:**
  - `v1 — 2026-09-23 — initial draft` after the `huggingface/tokenizers` investigation; user approved "(a) design doc + (b) implement the scoped token-aware review budget."
  - `v2 — 2026-09-23 — round-1 Dory corrections`: total + typed planner (protected-overflow ⇒ terminal escalation, no LLM call); detect upstream truncation via the existing `[WARNING: truncated…]` marker; couple the request `max_tokens` to the plan's output reserve; conservative bytes÷3; nested `review.budget` config with validation; reorder priority so GIT DIFF + APPROVED DESIGN are protected above PRD/contract/history; corrected current-system claims (caps live in `build_review_prompt`, not `gather`; TASK/criteria uncapped). **Scope note:** the MVP resolves the reviewer context window from **configured defaults** (conservative local 8192 / cloud 128000); reading the *exact* per-model `n_ctx` from the observability `/props` snapshot is the documented **first follow-on** (needs a public accessor + `run_loop` plumbing) — deferred to keep this slice bounded, with the output-reserve coupling ensuring input+output fit the assumed window.

## Problem

The code reviewer assembles a prompt from several sections (PRD, git diff, agent contract, task details, session history, review criteria) and truncates **each section independently to a fixed 150 KB byte cap** (`src/review/context.rs:7` `MAX_SECTION_SIZE`, `:189` `truncate`, `:298`). This has two real defects:
1. **It never budgets the *whole* prompt against the reviewer model's context window.** With ~6 sections at up to 150 KB each, the assembled prompt can reach ~900 KB ≈ 225k–300k tokens — far beyond a local model's 8k–32k context, and beyond a 128k cloud context. The model then truncates or errors, or the review silently loses evidence.
2. **Bytes are a poor proxy for tokens** (≈3–4 bytes/token for English/code, but highly variable), so even the per-section cap is miscalibrated.

The fix: budget the assembled review prompt by **tokens** to fit the *reviewer's actual context window* minus an output reserve and safety margin, truncating sections in a defined priority order so the task and criteria are never dropped. This is exactly what H6 in `hankndory-brainrouter-integration.md` specified but which was never implemented (truncation is still byte-based).

## Goals and non-goals

**Goals.**
- Budget the whole review prompt to `context_window − output_reserve − margin`, per the reviewer backend/model.
- Count tokens with the **best available counter per backend**, behind one abstraction, with a safe always-available fallback.
- Truncate sections in a fixed priority order; never drop TASK DETAILS or REVIEW CRITERIA.
- When design/diff evidence is dropped to fit, flag it so the loop won't emit a confident `approved` on truncated evidence.

**Non-goals.**
- Exact tokenization of proprietary cloud models (Claude has no public tokenizer; approximate).
- A heavy tokenizer dependency in the MVP (the exact counters are documented follow-ons behind the seam).
- Changing routing, persistence, or any non-review behavior.

## Current system

- `src/review/context.rs`: `gather(project_dir) -> ReviewContext` loads PRD / git diff / agent contract, each `truncate`d to `MAX_SECTION_SIZE` (150 KB, `:7`). `truncate` (`:189`) is byte-based (`String::truncate` on a char boundary).
- `src/review/prompt.rs`: `build_review_prompt(ctx, task, summary, details, history, design_doc, local)` joins the sections + criteria (byte-capped upstream); no token accounting.
- `src/review/review_loop.rs`: `run_loop` gathers context fresh each iteration and builds the prompt; the admitted backend (`choice`) and `is_local` are known here.
- `src/observability.rs:547`: reads the local model's `n_ctx` from llama-swap `/props` (`default_generation_settings.n_ctx`) — an existing source for the local context window.
- Token counts today are **post-hoc only**: provider `usage.prompt_tokens` (`observability.rs`) and llama-bench `n_prompt`/`n_gen` (`benchmark.rs:1562`). No a-priori counting; no tokenizer dependency (`Cargo.toml`, 45 deps, none tokenizer-related).
- **Backend reality (bounds the design):** the reviewer defaults to **cloud** (Manifest) — Claude/GPT, whose tokenizers are proprietary. Local coding models are **GGUF** with the tokenizer *embedded* (no standalone `tokenizer.json`; strix has 11 `tokenizer.json` files, all whisper/voice/Qwen3-4B, not the coding GGUFs). llama.cpp exposes a **`/tokenize`** endpoint that counts with the GGUF's embedded tokenizer (exact, no sidecar).

## Requirements and acceptance criteria

- **R1 — `TokenCounter` + conservative heuristic.** `trait TokenCounter { fn count(&self, text: &str) -> usize; }`; `HeuristicCounter` = `ceil(bytes / 3)` (conservative — over-estimates tokens so the failure direction is *over*-truncation, never context overflow; bytes not chars so multibyte never under-counts). Non-empty ⇒ ≥1. *AC:* unit tests on known strings; a worst-case corpus check that it never under-counts vs a reference on ASCII/code.
- **R2 — Context window (configured defaults; exact `n_ctx` is a follow-on).** `context_window(is_local, cfg) -> usize` = local `cfg.local_default_ctx` (8192) or cloud `cfg.cloud_default_ctx` (128000). Reading the exact per-model `n_ctx` from `/props` is the documented first follow-on. *AC:* deterministic; conservative local default.
- **R3 — Total, typed planner.** A typed `Section { kind: SectionKind, text: String }` list (not priority-inferred from headings). `plan_budget` counts the **entire rendered prompt** — the system message, every section heading, the `SEP` separators, plus the criteria — against `available = ctx − output_reserve − ceil(margin·ctx)` (defaults: `output_reserve = min(cfg.output_reserve (2048), ctx/2)`; `margin = 0.10`). **Protected** = TASK DETAILS + REVIEW CRITERIA + (when design-aware) DESIGN-DIVERGENCE CRITERIA — never dropped. **Droppable**, in keep-priority (highest kept longest): **GIT DIFF** (head+tail trim, never fully dropped) > **APPROVED DESIGN** (keep Status/Goals/Requirements/Detailed-implementation/Risks/Human-approval subsections, then head-trim) > **PRD** > **AGENT CONTRACT** > **SESSION HISTORY** (dropped first). Every section has an exact op: whole-drop, head-trim (`truncate`), head+tail (new helper), or subsection-extract. **Totality:** if the protected set alone exceeds `available`, **do not call the LLM** — return a terminal escalation (R4). *AC:* fits-⇒-unchanged; over-budget ⇒ final rendered prompt ≤ `available` by the counter (or protected-overflow terminal escalation); priority order respected; task+criteria always present.
- **R4 — Truncated-evidence safety (single terminal result).** `plan_budget` returns `truncated_required: bool`, set when GIT DIFF or (design-aware) APPROVED DESIGN is trimmed/dropped **at the budget stage OR already upstream** (detected by the existing `[WARNING: truncated` marker that `context.rs::truncate` appends), or when the protected set overflows. `run_loop`, at the post-parse point (`review_loop.rs:~193-211`), converts any `approved`/`needs_revision` into an **immediate `Escalated`** (reason: truncated evidence) **before** the session/history update — not a `needs_revision` loop that would re-review the same truncated evidence to `max_iterations`. Design absence sets the flag **only when design-aware review is on** (not when integration is off). *AC:* unit test that truncation forces a terminal escalate; design-off never sets the flag.
- **R5 — Output-reserve coupling + default-safe.** The plan returns `max_output_tokens = output_reserve` (bounded to `available`'s complement); `run_loop` uses it for `ChatCompletionRequest.max_tokens` instead of the hardcoded 16384 (`review_loop.rs:~328`), so declared input+output fits the assumed window. No new dependency, no network, no model load (heuristic only; `/props` deferred). *AC:* the request's `max_tokens` equals the plan's value; builds with zero new crates; existing suite green.
- **R6 (follow-ons, documented, behind the trait — unchanged):** exact `n_ctx` from `/props` (a public observability accessor + `run_loop` plumbing) as the **first** follow-on; then `tiktoken-rs` (GPT-family), llama.cpp `/tokenize` (local exact, model-loaded only, cached + fallback), `huggingface/tokenizers` (HF-format/vLLM/benchmark, `default-features=false`). A `CounterChain` selects the best available per backend, falling back to `HeuristicCounter`.

## Config (R5) — deferred (NOT yet implemented)

**Not shipped in the MVP.** The budget currently uses module constants (`DEFAULT_LOCAL_CTX` 8192, `DEFAULT_CLOUD_CTX` 128000, `DEFAULT_MAX_OUTPUT` 16384, `MARGIN_PERCENT` 10, `DEFAULT_DIVISOR` 3 in `src/review/tokens.rs`). A `review.budget` config block is a follow-on. **Do not add a `review.budget:` key to the YAML until the struct field exists** — `ReviewConfig` is `#[serde(deny_unknown_fields)]` (`config.rs:429`), so an unknown block would make the daemon reject the whole config. When implemented: `{ local_default_ctx, cloud_default_ctx, output_reserve, margin_percent, heuristic_divisor }`, all `#[serde(default)]`, validated (`heuristic_divisor ≥ 1`, `margin_percent ≤ 90`, contexts ≥ 512), saturating arithmetic.

## Technical plan

New `src/review/tokens.rs`: the `TokenCounter` trait + `HeuristicCounter` + `context_window()` + the budget planner `plan_budget(sections, ctx, reserve, margin) -> (kept_sections, truncated_required)`. `context.rs`/`prompt.rs` call the planner instead of the fixed per-section byte cap; `review_loop.rs` passes the admitted backend + observability (for `n_ctx`) and honors `truncated_required`. The exact counters (R6) are separate modules added later behind the same trait; the MVP wires only `HeuristicCounter`.

Because the reviewer's `choice`/`is_local` and the section texts are already available in `review_loop.rs`/`context.rs`, no routing or request-plumbing changes are needed. The budget replaces the arbitrary 150 KB cap with a model-aware token budget.

## Architecture and flows

```mermaid
flowchart TB
  A[gather sections\nPRD·diff·agents·task·history] --> P[plan_budget]
  CW[context_window(backend, model, /props n_ctx)] --> P
  TC[TokenCounter\nHeuristic → (later) tiktoken / /tokenize / HF] --> P
  P -->|fits| BP[build_review_prompt]
  P -->|over budget| TR[truncate in priority order\ntask+criteria never dropped\nset truncated_required]
  TR --> BP
  BP --> RL[run_loop\napproved+truncated_required ⇒ escalate]
```

## Alternatives considered

- **Add `huggingface/tokenizers` now as the primary counter.** *Rejected for the MVP:* its exact benefit is bounded — cloud models are proprietary and local coding GGUFs have no sidecar `tokenizer.json` (only 11 exist on strix, none the coding GGUFs). It fits the HF-format/vLLM/benchmark niche (R6), not the primary review path. `default-features=false` keeps it lean when added.
- **llama.cpp `/tokenize` as the only counter.** *Rejected as MVP sole source:* exact for local but needs the model loaded + a network round-trip; doesn't help cloud reviewers. It's the right *local* exact backend (R6), layered on the heuristic.
- **`tiktoken-rs` for everything.** *Rejected:* exact only for OpenAI/GPT-family; wrong for Claude and local. Good as the GPT-family exact backend (R6).
- **Keep the byte cap.** *Rejected:* it's the bug — no whole-prompt budget, can exceed context.

## Detailed implementation

**MVP (b):**
- `src/review/tokens.rs` (new): `trait TokenCounter { fn count(&self, text: &str) -> usize; }`; `struct HeuristicCounter { divisor: f32 }` (default 3.6, min-1 result); `fn context_window(is_local, n_ctx_hint, cfg) -> usize`; `fn plan_budget(...)` implementing R3's order; unit tests.
- `src/review/context.rs`/`prompt.rs`: assemble sections, then apply `plan_budget` against the reviewer's context window rather than the fixed 150 KB per-section cap. Keep `truncate` as the head/tail helper for the diff.
- `src/review/review_loop.rs`: pass `is_local` + the observability `n_ctx` hint into the budget; thread `truncated_required` into the verdict handling (R4).
- `src/config.rs`: optional `review.budget` knobs (`local_default_ctx`, `cloud_default_ctx`, `output_reserve`, `margin`, `heuristic_divisor`) with the R2/R3 defaults.
- Tests: heuristic calibration; fits-unchanged; over-budget reduction; priority order; task+criteria preserved; truncated_required forces non-approve.

**Follow-ons (R6, separate PRs, each behind the trait + its own review):** `tiktoken-rs` GPT-family counter; `/tokenize` local counter (with cache + fallback); `huggingface/tokenizers` HF-format counter (`default-features=false`).

## Testing and evaluation

- MVP: Rust unit tests for the counter, `context_window`, and `plan_budget` (fits / over-budget / priority / never-drop-task-criteria / truncated_required). Full suite green; clippy 0; no new deps.
- Follow-ons: compare each exact counter against provider `usage`/`/tokenize` on sample prompts within tolerance; fallback-on-error tests.

## Security, privacy, reliability, and operations

- MVP adds no network, no dependency, no persisted state, no model load. The `/tokenize` follow-on calls the local llama-swap only (loopback) and only when a model is already loaded; it must time-box and fall back to the heuristic so a review never blocks on tokenization.

## Rollout, migration, and rollback

- MVP ships first (heuristic budget, default-safe, replaces the byte cap). Fully reversible. Exact counters are additive follow-ons behind the seam; absent them, the heuristic is used.

## Risks and mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Heuristic under-counts → prompt still slightly over context | Med | Conservative divisor + 10% margin + output reserve; the model's own truncation is the last resort, not the first |
| Heuristic over-counts → over-truncation | Low | Priority order protects TASK+CRITERIA; margin tunable via config |
| Unknown context window (cloud/local) | Low | Deterministic configured defaults (128k cloud / 8k local) |
| `/tokenize` latency/among-load (follow-on) | Med | Time-box + cache + heuristic fallback; only when model already loaded |
| Truncated evidence yields false `approved` | Med | `truncated_required` post-parse override (R4) |

## Open questions

- **Q-T1** Heuristic divisor: one global default (3.6) vs. per-language/per-model calibration? (Proposed: one conservative global default, config-overridable; refine with the exact counters.)
- **Q-T2** Should the MVP already wire the `/tokenize` local exact path, or is heuristic-only the right first cut? (Proposed: heuristic-only MVP; `/tokenize` as the first follow-on since it's the highest-value exact backend for local reviews.)

## Decision log

- **Implement H6's budget with a real token counter, not the byte cap.** The byte cap never budgeted the whole prompt against the context window. 2026-09-23.
- **MVP = heuristic counter, no new dependency; exact counters (tiktoken-rs / `/tokenize` / huggingface/tokenizers) are follow-ons behind a `TokenCounter` trait.** Grounded in the backend reality (cloud=proprietary, local=GGUF-embedded, sidecar `tokenizer.json` rare). 2026-09-23.
- **`huggingface/tokenizers` is adopted narrowly (HF-format/vLLM/benchmark), `default-features=false`.** Its exact benefit doesn't cover the primary cloud+GGUF review path. 2026-09-23.

## Referenced files

- `src/review/context.rs` — section assembly + the byte cap this replaces (`:7`, `:189`, `:298`).
- `src/review/prompt.rs` — `build_review_prompt`.
- `src/review/review_loop.rs` — `run_loop`; where the admitted backend/`is_local` and section texts are available and `truncated_required` is honored.
- `src/observability.rs` — `n_ctx` from `/props` (`:547`); post-hoc token usage.
- `src/config.rs` — `ReviewConfig`; new optional `review.budget` knobs.
- `docs/design/hankndory-brainrouter-integration.md` — the H6 spec this implements.

## Dory validation record

- **Round 1 — comprehension + critic + readiness (v1), isolated sub-agent.** Comprehension PASS; readiness **NOT READY**. Confirmed the core claim (no whole-prompt context budget today; prompt can far exceed a local context window). **Refuted/corrected:** the 150 KB caps live in `build_review_prompt` (`prompt.rs:26-68`), not `gather`; TASK DETAILS + criteria are **uncapped**; the observability `n_ctx` snapshot is **private** with no `run_loop` handle; `truncated_required` does not exist; H6's dynamic-language half already shipped. **4 blocking:** (1) planner not total (protected content can exceed budget — no terminal case; `truncate` is head-only not head+tail); (2) R4 can't see upstream truncation (`context.rs` truncates the diff before the planner; `ReviewContext` is strings-only); (3) `n_ctx`/output-reserve not wired (`max_tokens` hardcoded 16384 would exceed an 8192 ctx); (4) H6 has conflicting normative values. Plus important findings (heuristic not demonstrably conservative; design can be dropped while criteria assert it; small-diff no-op needs a golden test; config schema under `deny_unknown_fields`). **All addressed in v2:** total+typed planner with protected-overflow terminal escalation; marker-based upstream-truncation detection; output-reserve coupling (exact `/props` `n_ctx` explicitly deferred to the first follow-on, with conservative configured defaults meanwhile); bytes÷3; priority reorder (DIFF/DESIGN protected-high); "Supersedes H6"; nested validated config. Self-audit (sub-agent): relied only on the doc + repo.
- **Round 2 — Step-9 mean code review of the implementation (reviewer used per user request).** **No blocking/high.** Verified: `plan_budget` is total; the budget bound holds for the heuristic; the truncated→escalate and protected-overflow paths are correct (both persisted + returned reasons; no double-update); the small-diff path is byte-identical; design-aware is safe (`DesignDivergence` protected; a dropped `ApprovedDesign` forces escalate); no char-boundary panic; no `as u32` truncation. **Fixed** its findings: (Med) softened the over-conservative "never overflow" claim to match reality (bytes÷3 is conservative for ASCII, not universal; risk is reliability not a false-approve); (**Low, self-affecting**) replaced the `[WARNING: truncated` **string-sniffing** — which false-positived on this repo's own `context.rs`/`tokens.rs` literals — with an explicit per-`Section` `upstream_truncated` bool set at build time (+ a regression test); (Low) made the trim byte-budget derive from `counter.bytes_per_token_hint()` so a future non-÷3 counter (R6) stays sound; (nit) shared `REVIEW_SYSTEM_MESSAGE` across the sender + overhead estimate; documented that the `review.budget` config block is a deferred follow-on (not shipped — avoids the `deny_unknown_fields` trap) and that GIT DIFF may be fully dropped only when it can't get a useful minimum (then `truncated_required` escalates). Validation: full suite green (12 binaries) + clippy 0 + `check-html-js` OK.

## Human approval

**Approved to implement the MVP (b) + write this doc (a):** user, 2026-09-23 ("a and b"). This doc is drafted and will be Dory-reviewed; the MVP (R1–R5) is implemented under that approval. The exact-counter follow-ons (R6) remain separate, individually-reviewed PRs.
