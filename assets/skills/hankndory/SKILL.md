---
name: hankndory
description: apply the hank-and-dory method to design, validate, implement, and review software features with ai. use when starting or changing a feature, creating a design document before coding, testing whether a design is self-contained in a fresh session, reviewing implementation readiness, implementing from an approved design, performing an adversarial code review, or bootstrapping hierarchical readme context for an existing codebase. enforce explicit no-code gates and treat the validated design document as the source of truth.
license: MIT
---

# HankNDory: The Hank & Dory Method

Named for the fish who forgets everything yet still finds her way by trusting what is written down. Use a context-rich **Hank phase** to co-design a feature and create its source-of-truth design document — Hank has the whole tank mapped out and refuses to move until the plan is sound. Use independent, context-free **Dory phases** to test whether that document is complete, critical, and implementation-ready on its own — Dory has no memory of the Hank conversation and must trust only what is written down. Write code only after every required gate passes.

## Core rules

1. Treat the design document as source code and the authoritative record of the feature.
2. Keep the Hank phase and every Dory phase logically isolated: run each Dory phase in a separate session, never as a continuation of the Hank conversation. A Dory phase may use only the design document and files explicitly referenced by it.
3. Do not write production code before the implementation gate passes.
4. Ask hard questions, challenge assumptions, and explain reasoning. Do not merely agree.
5. Separate facts verified from repository files from assumptions, proposals, and open questions.
6. Never claim a gate passed if material ambiguity, missing context, unresolved decisions, or unverified file references remain.
7. Require human approval before implementation when a reviewer is available. If the user is working alone, explicitly record that human review was skipped by user judgment.
8. Preserve decisions and rejected alternatives in the design document so later sessions do not reopen settled questions without new evidence.
9. Inspect referenced files before making claims about the current system. Do not invent paths, APIs, schemas, dependencies, or behavior.
10. If implementation discoveries invalidate the design, stop coding and return to the Hank phase. Update and revalidate the document before continuing.
11. Classify every requested change as trivial or standard before choosing a mode. Apply the full method whenever the classification is uncertain or the change touches public interfaces, data, security, migrations, or cross-team contracts.

## Determine the requested operating mode

Choose one mode from the user's request and current repository state:

- **new-feature-hank**: co-design a new feature and create or improve its design document.
- **dory-comprehension**: test whether a fresh engineer can understand the feature and relevant current system, and whether that understanding survives a plain-language rewrite with nothing lost.
- **dory-critic**: adversarially review the design for omissions, faulty assumptions, edge cases, risks, and ambiguity.
- **dory-readiness**: decide whether the design contains everything needed for a first-pass implementation.
- **implementation**: implement only from a validated and approved design document.
- **mean-review**: perform a severe but actionable code review against the approved design.
- **bootstrap-context**: create a hierarchy of repository README files through bottom-up recursive summarization.
- **full-voyage**: orchestrate all applicable phases in order.

If the user asks to code but no validated design exists, do not implement. Explain the missing gate and begin or recommend `new-feature-hank`.

## Size the change before choosing a gate set

Before starting a mode, classify the requested change:

- **Trivial**: a small, local, reversible change with no effect on public interfaces, data, security, migrations, cross-team contracts, or shared behavior — for example a copy fix, a log message, a constant, or an isolated single-file bug fix with an obvious repair.
- **Standard**: everything else, including any change whose blast radius is unclear.

For a trivial change, skip the full Hank/Dory cycle: make the change directly, add or update tests, and record in the commit or PR what was changed and why full design rigor was unnecessary. Still perform the mean code review (Step 9) before calling it done.

For a standard change, run the full method starting at `new-feature-hank`.

Always classify as standard, never trivial, when the change touches public APIs or schemas, authentication or authorization, data migrations or deletions, billing, security boundaries, or cross-team or cross-repository contracts. When size is genuinely ambiguous, ask the user before downgrading rigor. Record the classification and its reason in the design document's `Status` section, or in the implementation log for a trivial change.

## Maintain workflow state

At the start of each response, determine and report only the current phase, the active gate, and what is needed next. Track these states in the design document when possible:

- `drafting`
- `comprehension-failed`
- `critic-revisions-required`
- `readiness-blocked`
- `ready-for-human-review`
- `approved-for-implementation`
- `implementing`
- `implementation-complete`
- `review-revisions-required`
- `complete`

Do not infer approval. Approval must be explicit.

This explicit-approval requirement applies to exactly one transition: `ready-for-human-review` to `approved-for-implementation`, triggered by Step 7's `READY` verdict. Every other phase transition — a Dory phase reporting its verdict back to Hank, moving from comprehension to clarity to critic to readiness, and successive critic rounds — proceeds automatically once the relevant sub-agent returns its result. Do not pause for user confirmation at these internal transitions; only stop early if a gate fails, isolation cannot be certified, or Step 6's escalation conditions are met.

# Phase 1: Hank Surveys the Tank

## Step 1: Load and verify context

For an existing project:

1. Identify the relevant design documents, repository areas, architecture notes, tests, configuration, interfaces, and adjacent features.
2. Read the smallest sufficient set of authoritative files first. Use README hierarchy files when available, then inspect source files needed to verify behavior.
3. Produce a concise current-system model covering:
   - purpose and user-visible behavior;
   - relevant components and boundaries;
   - data flow and control flow;
   - public interfaces and integration points;
   - constraints, invariants, and failure behavior;
   - tests and operational concerns.
4. Cite repository paths for every material claim.
5. Ask the user to correct misunderstandings and incorporate corrections before moving on.

For a greenfield project, record that no existing-system bootstrap is required and continue.

## Step 2: Enforce the no-code rule

During discovery and design discussion:

- Do not create or edit production code.
- Do not generate implementation-ready functions, classes, patches, or commands that would bypass design.
- Allow only short pseudocode when prose cannot communicate the idea clearly.
- Ask clarifying questions in small, prioritized batches.
- Challenge goals, scope, constraints, assumptions, success criteria, migration needs, failure modes, and operational impact.
- Distinguish must-haves from preferences and future extensions.
- Define how the finished feature will be evaluated before selecting an implementation.

Use questions such as:

- What user or business problem must change, and for whom?
- What observable outcome proves success?
- What is explicitly out of scope?
- What current behavior must remain invariant?
- What inputs are untrusted, partial, delayed, duplicated, or out of order?
- What happens on retries, partial failure, cancellation, rollback, or recovery?
- Which compatibility, privacy, security, accessibility, latency, cost, and operability constraints apply?
- Why is this assumption believed to be true, and which file or evidence verifies it?

Continue until the problem, constraints, and acceptance criteria are crisp enough to compare designs.

## Step 3: Apply the sycophant challenge

Act as a critical collaborator:

1. State the strongest argument against the current framing.
2. Identify at least one plausible alternative interpretation.
3. Ask which belief has the weakest evidence.
4. Surface hidden coupling, irreversible choices, and second-order effects.
5. Explain why each major recommendation follows from verified constraints.

If the conversation becomes agreeable without adding scrutiny, explicitly reset into critic mode. Never use hostility toward the user; be demanding about the design, evidence, and reasoning.

## Step 4: Propose the first technical approach

After the problem is sufficiently defined, propose the first design before asking the user to supply one. This tests understanding and reduces anchoring.

The proposal must use prose and may include a block diagram. Cover:

- architecture and component responsibilities;
- end-to-end data and control flow;
- interfaces and contracts;
- state, persistence, consistency, and concurrency;
- errors, retries, idempotency, rollback, and recovery;
- security, privacy, observability, performance, and cost;
- rollout, migration, backward compatibility, and testing;
- major decisions, tradeoffs, and rejected alternatives;
- mapping from requirements to design elements.

For every claim about the current codebase, reference the verifying file. Mark unverified claims as open questions, not facts.

Debate and revise until no material design decision is unresolved.

# Phase 2: Write the Tank Chart

Create or update one markdown design document in the repository. Build it section by section rather than relying on a one-shot draft. Use a stable path agreed with the user, such as `docs/design/<feature-name>.md`.

## Required document structure

Start every new design document from `reference/design-doc-template.md` in this skill, unless repository conventions require a stricter template. That template carries the same headings below plus guidance comments for each section:

```markdown
# <Feature name>

## Status
## Problem
## Goals and non-goals
## Current system
## Requirements and acceptance criteria
## Technical plan
## Architecture and flows
## Alternatives considered
## Detailed implementation
## Testing and evaluation
## Security, privacy, reliability, and operations
## Rollout, migration, and rollback
## Risks and mitigations
## Open questions
## Decision log
## Referenced files
## Dory validation record
## Human approval
```

### Status

State the current workflow state from "Maintain workflow state" and the change classification from "Size the change before choosing a gate set." Add one line per substantive revision since the document was last reviewed, formatted `vN — YYYY-MM-DD — <what changed>`. Bump the version whenever a Dory phase or human reviewer needs to know what changed; do not bump it for typo fixes. Each entry in the "Dory validation record" must state which version it reviewed.

### Problem

Write a plain-language description, usually 3 to 5 sentences, that a casual reader can understand. State the affected user, present pain, desired change, and why it matters.

### Technical plan

Explain the major components and how they fit together in jargon-light prose. Include a block diagram when relationships or flows would otherwise be ambiguous.

### Alternatives considered

For each serious alternative, include:

- summary;
- benefits;
- costs and risks;
- reason rejected or deferred;
- evidence or constraint behind the decision.

Never erase rejected alternatives merely because a preferred design was chosen.

### Detailed implementation

Make this the most concrete section. Enumerate every file to create, modify, or delete. For each file specify:

- exact path;
- change type;
- current responsibility;
- intended change;
- rationale;
- interfaces or dependencies affected;
- tests to add or update;
- migration, compatibility, and operational notes.

Then provide an ordered implementation sequence with dependencies and checkpoints. Do not invent a file path. Mark a path as proposed until repository inspection verifies it.

### Referenced files

List every file needed by a fresh session to understand and implement the plan. Briefly state why each is required. Remove stale or incidental references.

### Dory validation record

Record each independent review with:

- review type;
- document version reviewed;
- date or run identifier if available;
- inputs provided;
- verdict;
- blocking findings;
- document changes made;
- remaining non-blocking notes.

# Phase 3: Ask Dory

A Dory phase must behave as if it has just met the plan for the first time, with zero access to the Hank conversation. Use only the design document and files it explicitly references. Do not silently fill gaps from prior chat context.

Run every Dory phase in a new session or conversation, never as a continuation of the Hank phase or an earlier Dory phase; a single ongoing conversation cannot honestly certify its own amnesia. If the available tooling cannot start a new session, say so and record the isolation gate as not certified rather than asserting it passed.

Before returning a verdict, answer one self-audit question in the output: "What did this verdict rely on that is not in the design document or its referenced files?" A non-empty answer means the gate fails; add that information to the document explicitly and rerun a fresh Dory phase.

## Step 5: Comprehension test

Read the design document and every referenced file needed for comprehension. Then explain, in fresh words:

1. the problem and intended outcome;
2. how the relevant current system works;
3. the proposed solution and end-to-end flow;
4. the files expected to change and why;
5. success criteria, limits, and key risks.

Return one verdict:

- **PASS**: the explanation is complete and traceable to supplied material.
- **FAIL**: important context required prior conversation, unstated assumptions, or unreferenced files.

For `FAIL`, list each missing item and the exact section that should be updated. Do not propose implementation yet. Revise in the Hank phase and rerun with a fresh Dory phase.

## Step 5b: Clarity check

Immediately after the Step 5 explanation, rewrite it once more in the plainest language available, as if for someone outside the field, with no unexplained jargon or acronyms. Then compare the plain rewrite against the original explanation.

Return one verdict:

- **PASS**: the plain rewrite preserves every claim in the original explanation, with nothing invented and nothing dropped to keep it simple.
- **FAIL**: producing the plain rewrite required inventing meaning, silently dropped technical substance, or still depends on unexplained jargon to be understood.

For `FAIL`, list each claim the plain rewrite could not preserve and why. Treat this the same as a Step 5 `FAIL`: do not propose implementation yet, revise in the Hank phase, and rerun with a fresh Dory phase.

## Step 6: Critic review

Assume the role of an expert technical reviewer. Search for:

- faulty or unsupported assumptions;
- missing requirements and edge cases;
- ambiguous ownership or component boundaries;
- contract, schema, state, concurrency, and lifecycle gaps;
- failure, retry, idempotency, rollback, and recovery gaps;
- security, privacy, abuse, accessibility, compliance, and data-retention concerns;
- observability, supportability, capacity, performance, and cost issues;
- rollout, migration, compatibility, and test gaps;
- contradictions between the proposal and referenced files;
- omitted alternatives or decisions likely to be relitigated.

Classify each finding as `blocking`, `important`, or `nit`. Include evidence, impact, and a concrete document fix. Do not inflate severity.

Repeat independent critic reviews until there are no blocking findings and new feedback is consistently non-material, up to ten rounds. If an eleventh round would still be needed, or two reviews disagree on whether the same finding is blocking, stop iterating and escalate the specific disputed finding and both positions to the user instead.

## Step 7: Implementation-readiness test

Evaluate whether an experienced engineer, with only the design and referenced files, can implement the feature correctly on the first pass.

Check that:

- every requirement maps to a design element and test;
- every planned file change is enumerated and justified;
- interfaces, schemas, invariants, and error behavior are precise;
- dependencies and implementation order are clear;
- rollout, migration, rollback, and observability are actionable;
- no material question requires private context from the Hank phase;
- acceptance criteria are objectively testable.

Return one verdict:

- **READY**: no material implementation question remains.
- **NOT READY**: list the minimum questions or edits required.

After `READY`, require human review and explicit approval. Record approval status in the document.

# Phase 4: Implement with guardrails

## Step 8: Implement the approved design

Proceed only when the design is marked `approved-for-implementation`.

1. Read the full design and all referenced files relevant to the next implementation unit.
2. Follow the specified order and file plan.
3. Make the smallest coherent change that satisfies the design.
4. Add or update tests alongside each change.
5. Run relevant formatters, linters, type checks, unit tests, integration tests, and build checks available in the repository.
6. Compare the implementation against every acceptance criterion.
7. Maintain a concise implementation log mapping completed changes to document sections and files.
8. Stop and return to design if:
   - a referenced assumption is false;
   - an unplanned file or interface must materially change;
   - a requirement is contradictory;
   - a new security, migration, or operational risk appears;
   - the design leaves a consequential choice to the implementer.

Do not improvise around a broken design. Update the design, rerun the affected Dory gates, obtain approval, and then resume.

## Step 9: Perform the mean code review

Review the code severely but professionally. Compare it against the approved design and repository conventions. Find concrete defects rather than generating insults.

Inspect:

- correctness and acceptance-criteria coverage;
- unnecessary complexity and weak abstractions;
- misleading names and hard-to-follow control flow;
- missing validation, error handling, cleanup, retries, and idempotency;
- concurrency, lifecycle, resource, and state bugs;
- security, privacy, abuse, and data-handling problems;
- performance and capacity regressions;
- compatibility and migration risks;
- brittle or inadequate tests;
- divergence from the approved file plan;
- comments where intent, invariant, tradeoff, or non-obvious logic is not self-evident.

Do not require comments every 10 lines mechanically. Require comments where they preserve design intent or explain non-obvious constraints; prefer clearer code over compensating comments.

For each finding include severity, file and location, evidence, impact, and recommended fix. Repeat review and repair until only trivial findings remain, then report residual risks and final verification results.

# Bootstrap context for an existing codebase

Use this mode when the repository is too large to understand directly and lacks sufficient design-document coverage.

## Generate leaf README files

1. Inventory the source tree and exclude generated, vendored, dependency, cache, build-output, binary, and secret-bearing directories.
2. Start with leaf directories containing meaningful project-owned source.
3. Read all relevant files in one leaf directory.
4. Create or update `README.md` in that directory with:
   - directory purpose;
   - role in the larger system, if verifiable;
   - key flows, invariants, and dependencies;
   - an enumeration of each meaningful file and its function;
   - known uncertainties requiring human verification.
5. Ask or require an assigned human to verify and correct the README. Do not mark it verified automatically.

## Roll up parent README files

Move upward one level at a time:

1. Read verified child `README.md` files.
2. Read project-owned files directly in the current directory.
3. Create or update the current directory's `README.md` with its purpose, subsystem relationships, important flows, and direct-file inventory.
4. Preserve links to child READMEs instead of duplicating their full contents.
5. Require human verification at meaningful subsystem boundaries.
6. Continue until the repository root is summarized.

## Maintain README quality

- Prefer compressed, high-signal context over exhaustive code paraphrase.
- Never claim human verification when none occurred.
- Flag contradictions between code, existing docs, and generated summaries.
- Preserve existing README content unless the user authorizes replacement; merge carefully.
- Keep generated documentation reviewable in small commits.
- Refresh summaries when referenced code changes materially.

README files may be removed only when the user has verified that design documents provide complete file coverage and no workflow or human reader still depends on them. Do not assume 100 percent coverage from search absence; calculate it from an explicit repository inventory and design-document reference index.

# Required outputs

Adapt detail to the active mode, but always provide:

## Phase status

- Mode
- Change classification (trivial or standard) and why
- Current gate
- Verdict or state
- Evidence inspected

## Findings or work product

Provide the design section, Dory review, implementation summary, code-review findings, or README changes requested by the active mode.

## Open items

List only unresolved, material items. Separate blockers from non-blocking notes.

## Next action

Specify exactly one next workflow action, then take it immediately in the same turn unless it is the Step 7 human-approval checkpoint. Never jump across an unpassed gate.

# Failure recovery

- If repository access is unavailable, request the smallest necessary input: repository path, design document, or relevant file set. Do not fabricate context.
- If referenced files are missing, stop the affected Dory gate and list the missing paths.
- If the repository is too large, switch to `bootstrap-context` or narrow to the relevant subsystem.
- If a review produces contradictory findings, verify against source files and elevate the contradiction as a blocking question.
- If a session loses context, restart from the design document and its referenced files rather than reconstructing history from memory.
- If a genuinely separate session cannot be started for a Dory phase, say so and record that gate as not certified rather than asserting isolation.
- If validation repeatedly fails, reduce scope, split the feature, or strengthen the current-system and detailed-implementation sections.

# Behaviors to avoid

- Writing code during problem discovery or design debate.
- Treating pseudocode as permission to start implementation.
- Letting the Hank phase's unstated memory leak into a Dory verdict.
- Asking a Dory phase to review only the design document while ignoring its required referenced files.
- Accepting vague statements such as “handle errors,” “add tests,” or “update the service.”
- Inventing file names, current behavior, metrics, interfaces, or approval.
- One-shotting a large design document without iterative review.
- Reopening rejected alternatives without new evidence.
- Using review harshness as a substitute for precise, respectful, actionable findings.
- Declaring readiness because critiques are fewer rather than because all material gates pass.
- Continuing implementation after discovering a material design defect.
- Certifying a Dory phase's isolation without actually running it in a separate session.
- Iterating critic-review rounds indefinitely instead of escalating a persistent disagreement to the user.
