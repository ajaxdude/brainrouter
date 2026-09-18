# <Feature name>

<!-- Replace <Feature name> and every bracketed or commented placeholder below.
     Keep every heading even when a section is briefly "Not applicable" with a
     stated reason. Do not delete a heading to avoid writing it. -->

## Status

<!-- Current workflow state from "Maintain workflow state" in SKILL.md, and the
     change classification (trivial or standard) from "Size the change before
     choosing a gate set," with the reason for that classification.
     Add one line per substantive revision since the document was last
     reviewed, formatted `vN — YYYY-MM-DD — <what changed>`. Bump the version
     whenever a Dory phase or human reviewer needs to know what is new; do not
     bump it for typo fixes. -->

## Problem

<!-- 3-5 plain-language sentences: affected user, present pain, desired
     change, and why it matters. -->

## Goals and non-goals

<!-- Must-haves versus explicit non-goals. Keep future extensions separate
     from both. -->

## Current system

<!-- Purpose and user-visible behavior; relevant components and boundaries;
     data flow and control flow; public interfaces and integration points;
     constraints, invariants, and failure behavior; tests and operational
     concerns. Cite a repository path for every material claim. -->

## Requirements and acceptance criteria

<!-- Each requirement must map to a design element in "Technical plan" or
     "Detailed implementation" and a test in "Testing and evaluation."
     Acceptance criteria must be objectively testable. -->

## Technical plan

<!-- Jargon-light prose covering the major components and how they fit
     together. Include a block diagram if flows would otherwise be
     ambiguous. -->

## Architecture and flows

<!-- End-to-end data and control flow; interfaces and contracts; state,
     persistence, consistency, and concurrency; errors, retries, idempotency,
     rollback, and recovery. -->

## Alternatives considered

<!-- For each serious alternative: summary; benefits; costs and risks; reason
     rejected or deferred; evidence or constraint behind the decision. Never
     delete a rejected alternative once recorded, even after a preferred
     design is chosen. -->

## Detailed implementation

<!-- The most concrete section. For every file to create, modify, or delete:
     exact path; change type; current responsibility; intended change;
     rationale; interfaces or dependencies affected; tests to add or update;
     migration, compatibility, and operational notes.
     Then give an ordered implementation sequence with dependencies and
     checkpoints. Mark any unverified path "(proposed, unverified)" until
     repository inspection confirms it. -->

## Testing and evaluation

<!-- How each acceptance criterion will be exercised: unit, integration,
     manual, or evaluation-harness coverage. State what proves the feature
     works, not just that it runs. -->

## Security, privacy, reliability, and operations

<!-- Threats, data handling, access control, observability, capacity, and
     on-call/runbook impact. State "Not applicable" explicitly with a reason
     if genuinely out of scope. -->

## Rollout, migration, and rollback

<!-- Deployment sequencing, feature flags, data migration steps, and exactly
     how to revert if something goes wrong. -->

## Risks and mitigations

<!-- Known risks ranked by severity, each with a concrete mitigation or an
     explicit accepted-risk rationale. -->

## Open questions

<!-- Only unresolved, material items. Remove a question once it is answered
     instead of leaving it stale. -->

## Decision log

<!-- One entry per settled decision that could plausibly be relitigated:
     decision, status, reason, date. Never erase a superseded entry; mark it
     superseded and link the replacement. -->

## Referenced files

<!-- Every file a fresh Dory phase needs to understand and implement the
     plan, with a one-line reason for each. Remove stale or incidental
     references. -->

## Dory validation record

<!-- One entry per independent review:
     - review type (comprehension | critic | readiness)
     - document version reviewed (from "Status" above)
     - date or run identifier if available
     - inputs provided
     - verdict(s) — a comprehension entry records both the content verdict
       (Step 5) and the clarity verdict (Step 5b)
     - blocking findings
     - document changes made
     - remaining non-blocking notes -->

## Human approval

<!-- Approver, date, and exactly what was approved. If the user is working
     alone, state explicitly that human review was skipped by user
     judgment. -->
