# Brainrouter dashboard left-nav redesign

## Status

- **Workflow state:** ready-for-human-review. Three isolated Dory rounds ran; v4 closes the last two (round-3) findings with deterministic spec fixes. **No implementation** proceeds until the user approves and lifts the "discuss, not code yet" hold. Recommended first slice on approval: Phase 1 (global, UI-only).
- **Change classification:** **standard.** Phase 1 is a UI information-architecture change to a shared surface; **Phase 1b** adds one small runtime toggle endpoint (HankNDory) mirroring the existing FR-A/FR-D toggles; Phase 2 changes model-routing resolution (hot path) and adds a persisted per-project store. HankNDory rule 11 forbids treating any as trivial.
- **Revisions:**
  - `v1 — 2026-09-23 — initial draft` from the design discussion + the user's frequency/scope answers.
  - `v2 — 2026-09-23 — round-1 Dory corrections`: fixed three factual errors (the toolbox list is not Sankey-redundant → relocate intact; no HankNDory UI toggle exists → carve a Phase-1b backend increment; posture data spans four refresh paths, not just `refreshSlow`); reworded the R8 invariant (UI JS is allowed, only Rust/routing/persistence/API contracts are frozen); resolved the Models-nav model, quick-switch placement, and toolbox relocation; strengthened the validation gate; added accessibility acceptance and Phase-2 project-key safety.
  - `v3 — 2026-09-23 — round-2 Dory corrections`: fixed internal-consistency errors the round-2 readiness pass found — corrected the `renderPosture()` source map (reasoning←nudge, review←`/api/review/enabled`, HankNDory←`/api/review/status`, PR←`/api/review/pr-guidelines`; model is the quick-switch, not the posture line) with a concrete health-color mapping; fixed R-A11y so the HankNDory posture entry links to the Overview ledger (its Phase-1 home), not a nonexistent Config toggle; named a runnable harness (`scripts/test-nav-ui.cjs`, mirroring `test-benchmark-ui.cjs`) with structural + behavioral assertions and an honest manual-only a11y/viewport carve-out; enumerated the full routing surface to relocate; normalized the Models sub-tab labels.
  - `v4 — 2026-09-23 — round-3 Dory corrections`: health pill is driven by `/api/service-health` **only** (Bonsai removed — it isn't in that response; it's an auto-routing enabler, not a health input) with a full first-match color precedence for every returned field/state; pinned the reviewer/subagent routing controls to the strip expander (removed the "or Config-Quality" contradiction). Round-3 also independently ran the harness precedent (`test-benchmark-ui.cjs`: 38/38) confirming the test approach is viable.

## Problem

Brainrouter's dashboard left nav (`#sidebar`, a single 224px column) is overloaded: it stacks **ten sections doing four unrelated jobs** — navigate, monitor, configure, act — in one scroll. The features that are brainrouter's reason to exist (model choice, reasoning, reviews, HankNDory, PR) are *quality levers*, but they are scattered across four non-adjacent sections and wedged between infrastructure status above and restart buttons below. The user reports the nav is "very busy" and that they can't tell what some entries do — there are four separate model-ish destinations (Models, Downloads, Server Mode, Model activity) whose distinction is not legible. The goal is a nav that helps the user **observe** output/quality and **set** the key levers, with each control placed according to how often it is actually used.

## Goals and non-goals

**Goals.**
- Reduce the primary nav to a small set of stable destinations.
- Place each control by usage frequency: the frequently-changed lever (model) above the fold; set-and-forget levers one click away but glanceable; plumbing/ops out of the daily sightline.
- Keep every existing capability reachable (nothing deleted, only relocated/regrouped).
- Split "observe" surfaces (status, ledgers) from "set" surfaces (toggles).

**Non-goals.**
- Phase 1 does **not** change routing behavior, persistence, or any Rust logic — it is a DOM/CSS reorganization of `main_dashboard.html` plus reuse of existing endpoints.
- Not redesigning the standalone `/models` (model activity) or `/benchmarks` pages' internals.
- Per-project settings are **out of scope for Phase 1** (Phase 2 only, and scoped to model roles).
- No visual re-theming beyond what the reorganization requires.

## Current system

All line references are `src/escalation/templates/main_dashboard.html` unless noted. The sidebar is `<nav id="sidebar">` (`:180`), `overflow-y:auto`, containing top→bottom:

1. **Logo + version** (`:181-213`).
2. **Nav / view switcher** (`:215-244`): buttons `switchView('stream')` "Routing", `switchView('config')` "Config", `switchView('models')` "Models", `switchView('toolbox-models')` "Downloads", `switchView('server-mode')` "Server Mode"; plus anchors `/models` "Model activity" and `/benchmarks` "Benchmarks". `switchView` is defined at `:1092`.
3. **System Health** (`:247-297`): status rows for llama-swap, llama.cpp, **bonsai (with `toggleBonsai` switch)**, manifest, toolbox engine (with `upgradeToolboxAll()` "all ↑"); each shows a health dot + version + optional upgrade chip.
4. **All toolboxes** (`:299-302`): container list + `#cockpit-config-status`.
5. **Independent routing roles** (`:303-352`): `#routing-preset` + main/reviewer/subagent model selectors (`routingBackendChanged`, `selectRoutingPreset`, `saveRoutingProfile`, `loadRoutingModels`), and the **prompt-rewrite** toggle. Backed by `/api/routing-profile` + `/api/routing-models` (server.rs) and the global `ProfileStore` (`src/routing_profile.rs`).
6. **Code review master switch** (`:354-371`): `toggleCodeReview` (FR-A) + `togglePrGuidelines` (FR-D), backed by `/api/review/enabled` and `/api/review/pr-guidelines`.
7. **HankNDory design-aware** (`:373-382`): `loadReviewStatus()` status + verdict ledger panel, backed by `/api/review/status`.
8. **Nudge + reasoning tier** (`:383-402`): `toggleNudge` + `setNudgeTier('auto'|'light'|'deep')`.
9. **Bridges** (`:404-426`): Discord/Signal status.
10. **Tools** (`:427-435`): `restartService('llama-swap')` "Restart Local Stack", `restartService('llama-cpp')`, `restartService('manifest')`, `restartService('brainrouter')`, `syncModels()` "Sync Models → OMP", `flushModels()` "Flush Models". Fn defs: `flushModels` `:2337`, `restartService` `:3902`, `syncModels` `:3933`.

**Main column** (`:437+`): command bar (`:440`), Stream view (`:465`: Sankey panel `:476`, in-flight `:489`, KPI grid `:509`), Config view (`:517`: `brainrouter.yaml` + llama-swap config editors + agent files list).

**Sankey columns** (`SK_COLS`, `:1705-1712`): `harness · session · routing · backend · engine · model`. Backend and engine are already first-class, click-to-filter Sankey columns — so live observation of which backend/engine served traffic is already covered by the Overview, independent of the "All toolboxes" list.

**Per-project keying feasibility (for Phase 2):** both proxy handlers already receive and thread `cwd` into `Router::route_tagged` (`src/router.rs:243,253,262,271`), so a per-project model override can be keyed by the request's working directory without new request plumbing.

## Requirements and acceptance criteria

**Phase 1 — nav cleanup (global; UI-only, JS allowed).**
- **R1** Primary nav ≤ 4 destinations: **Overview · Models · Benchmarks · Config**. **Models** uses **internal sub-tabs** over the existing `switchView` views — **Local models** (`models`) · **Downloads** (`toolbox-models`, which already combines catalog+downloads, `:551-594`) · **Serving** (`server-mode`) — plus an explicit **external link** to the standalone `/models` "Model activity" page (a separate handler, `server.rs:338-347`, not a `switchView` target). *AC:* the Models nav item stays active across its sub-tabs; the existing Downloads polling gate (`activeView === 'toolbox-models'`) is preserved or updated to the new active-state rule; browser history/default-child behavior is specified in Detailed implementation; nothing becomes unreachable.
- **R2** An **always-visible control strip in the command bar** (`:440`, which does not scroll — the sidebar itself scrolls, `:180`, so the strip lives in the command bar, not "top of sidebar") holds: the **model quick-switch** (main role prominent; reviewer/subagent in an expander), the three frequent actions **Eject/Flush**, **Reload llama-swap** (Restart Local Stack), **Sync → OMP**, a **health pill**, and a **read-only quality-posture summary** (`reasoning · review · HankNDory · PR`). *AC:* none requires scrolling; the model can be changed here via the existing `saveRoutingProfile` path (explicit Save, preserving preset + all three roles by reading the existing form state — save semantics unchanged); posture/health render via a single `renderPosture()`/`renderHealthPill()` contract (R-Posture).
- **R3** Set-and-forget levers (reasoning tier, code-review toggle, PR toggle, prompt-rewrite, and — after Phase 1b — the HankNDory toggle) live in **Config**, grouped under a clear **"Quality"** heading distinct from plumbing. *AC:* grouped by intent, not interleaved with status/ops.
- **R4** The **"All toolboxes" management surface** (`#toolbox-list`, `#cockpit-config-status`, `:299-302`,`:3038-3106`) **moves intact into Config** ("Toolbox management", collapsible), **not deleted** — the Sankey `backend`/`engine` columns show only *traffic that was actually served* (`:1801-1817`), so they do **not** replace the list's inventory/status/versions/create/update/adopt/delete/upgrade capabilities. *AC:* every existing toolbox capability (list, default select, create, update, adopt, delete, `upgradeToolboxAll`, per-container upgrade, cockpit-config status) is reachable in Config; the sidebar no longer carries the always-open list.
- **R5** The **HankNDory verdict ledger/status** (`#review-ledger`, `loadReviewStatus`, `:373-381`) moves to **Overview** (observe). **Phase 1 does not add a HankNDory toggle** — none exists today (`/api/review/status` is GET-only, `server.rs:943-965`; `hankndory_integration` is YAML-seeded and unaffected by profile updates, `routing_profile.rs:195-199,303-316`). A runtime toggle is **Phase 1b** (below). *AC:* status is on Overview; the "set" half arrives only with Phase 1b.
- **R6** **Bridges** move into **Config** (kept, demoted). *AC:* Discord/Signal status reachable in Config.
- **R7** Of the Tools buttons, keep **Eject/Flush + Reload llama-swap + Sync → OMP** in the command-bar strip (R2); move **Restart llama.cpp / Manifest / brainrouter** to Config "Service controls". *AC:* the three rare restarts still reachable in Config. Because `flushModels`/`restartService`/`syncModels` read the implicit global `event.target` (`:2343-2355,:3902-3938`), moving them must **preserve each button's plain content** (or pass the button explicitly) so the spinner/disable target stays the button, not a nested icon.
- **R8 (invariant, corrected)** Phase 1 changes **no Rust, no routing, no persistence, and no API contract** — every control keeps its existing element `id`, handler, and endpoint. **New/adjusted UI JavaScript is expected** (the `renderPosture()` renderer, the Models parent/active-state logic, the quick-switch wiring). *AC:* the existing Rust suite stays green (regression guard); the structural/interaction checks in Testing pass.
- **R-Posture** Two renderers, no new endpoint. **`renderPosture()`** shows the four *set* values and maps each to its exact existing source: **reasoning** ← `/api/nudge` (tier); **review** ← `/api/review/enabled`; **HankNDory** ← `loadReviewStatus()`/`/api/review/status`; **PR** ← `/api/review/pr-guidelines`. (The *model* is shown by the quick-switch from `/api/routing-profile`, not the posture line.) **`renderHealthPill()`** ← `/api/service-health` (`server.rs:1819-1915`). Each renderer is invoked after its source(s) refresh (nudge/review/PR arrive via `refreshSlow`, `:4113-4135`; HankNDory via its own `loadReviewStatus`; health via `refreshHealth`, `:4105-4111`), with an explicit `loading`/`unavailable`/`stale` state per value when its source hasn't answered. **Health color mapping (from `/api/service-health` fields only — `llama_swap`, `manifest`, `llama_cpp`, `toolbox`, `cloud_fallback`; Bonsai is NOT a health input, it is an auto-routing enabler surfaced in the routing area):** precedence, first match wins — **grey/unavailable** = service-health has not answered; **red** = a required local service (`llama_swap` or `llama_cpp`) is down/unreachable; **amber** = any non-fatal degraded state — a service `loading`, `cloud_fallback = true` (serving on fallback), or Manifest disabled-by-config/unhealthy (cloud is optional); **green** = `llama_swap` + `llama_cpp` healthy with no amber condition. *AC:* every field/state `/api/service-health` can return maps to exactly one color via this precedence; the pill never depends on Bonsai.
- **R-A11y** Relocating levers is acceptable only with: visible focus, keyboard-operable sub-navigation, `aria-current`/tab semantics on Models sub-tabs, `role="switch"`+`aria-checked` (or native checkbox) on toggles, **text (not color-only)** posture indicators, and a link from each posture entry **to its own surface** — reasoning/review/PR → their Config-Quality anchors; **HankNDory → the Overview ledger** (its Phase-1 home; there is no Config toggle until Phase 1b). *AC:* keyboard-only walkthrough reaches every moved control; posture is legible without color; every posture entry deep-links somewhere real.

**Phase 1b — HankNDory runtime toggle (small backend increment; enables R5's "set" half).**
- **R8b** Add `GET/POST /api/review/hankndory {enabled}` mirroring FR-A/FR-D (persisted in `review_runtime_state.json`, loopback-guarded POST) and a Config-Quality toggle. This is **not** UI-only and ships as its own slice with unit tests + a `code-review` pass. *AC:* toggling persists across restart and never clobbers the sibling flags (reuses the FR-D locked full-snapshot writer).

**Phase 2 — per-project model pin (deferred; additive).**
- **R9** A project **inherits the global model profile by default** (zero per-project setup). *AC:* a never-configured project behaves exactly as today.
- **R10** The user may **pin a model (role) for a specific project**, keyed by project/`cwd`; resolution precedence: **explicit request model > project pin > global profile > auto**. *AC:* unit tests for each precedence rung; a pinned project overrides global, an unpinned one follows global live.
- **R11** The model quick-switch shows **inherited vs pinned** state and which scope a change affects. *AC:* changing the global vs pinning the project are visibly distinct actions.

## Technical plan

**Phase 1** is a restructure of `main_dashboard.html`: introduce a compact command-bar "control + posture" strip (the quick-switch reuses `/api/routing-profile`; `renderPosture()` reuses `/api/nudge` + `/api/review/enabled` + `/api/review/status` + `/api/review/pr-guidelines`; `renderHealthPill()` reuses `/api/service-health` via `refreshHealth` — **no new endpoint**), move the set-and-forget control blocks into the existing **Config** view under a new "Quality" group, relocate Bridges + the **toolbox-management surface intact** + the three rare restarts into Config, and move the HankNDory ledger panel into the Stream/Overview view. Relocated controls keep their existing ids/handlers/endpoints (only their DOM parent moves); the strip, the `renderPosture()`/`renderHealthPill()` renderers, and the Models parent/active-state logic are the **new UI JavaScript** R8 permits. The four model-ish nav entries become one "Models" destination: internal sub-tabs **Local models** (`models`) · **Downloads** (`toolbox-models`) · **Serving** (`server-mode`) + an external **Model activity** link to `/models`.

**Phase 2** adds a per-project model override: a small persisted store (local file, same durability contract as `routing_state.json`) mapping project key → optional pinned `ModelChoice` per role; `Router` model resolution consults it (precedence in R10) using the `cwd` it already receives; the dashboard gains a project-context indicator and an inherited/pinned affordance on the model switch. This is a separate PR with its own Dory pass because it touches the routing hot path.

## Architecture and flows

```mermaid
flowchart TB
  subgraph Top["Above the fold — always visible"]
    MQ[Model quick-switch\n(frequent lever)]
    ACT[⏏ Eject · ↻ Reload llama-swap · ⇄ Sync→OMP]
    POS[Posture (read-only): reasoning · review · HankNDory · PR + ● health]
  end
  subgraph Nav["Primary nav — 4 destinations"]
    OV[Overview\nSankey backend/engine · in-flight · KPIs · HankNDory ledger]
    MD[Models\nLocal models · Downloads · Serving · ↗ Activity]
    BM[Benchmarks]
    CFG[Config]
  end
  CFG --> Q[Quality\nreasoning · review · HankNDory toggle · PR · prompt-rewrite]
  CFG --> INT[Integrations\nBridges]
  CFG --> TB[Toolboxes & upgrades]
  CFG --> SC[Service controls\nrestart llama.cpp / manifest / brainrouter]
  CFG --> RAW[Raw config\nbrainrouter.yaml · llama-swap · agent files]
```

## Alternatives considered

- **A dedicated top-level "Tune" screen for all quality levers.** *Benefit:* one home for every lever. *Rejected:* the user changes only *model* often; reasoning/review are set-and-forget, so a whole destination is overkill. Elevating just the model (above the fold) + a read-only posture strip + a Config-Quality group achieves the intent with less surface. Preserved here because if usage shifts (frequent reasoning/review tuning) it becomes the right call again.
- **Full per-project profiles (all levers per project).** *Benefit:* maximum flexibility. *Rejected/deferred:* the user flagged per-project setup as too complex, and only model varies by project; scoping Phase 2 to model-only with inherit-by-default gives the value with near-zero setup.
- **Remove Bridges.** *Rejected:* the user wants them kept; move to Config rather than delete.
- **Keep the "All toolboxes" list.** *Rejected:* backend/engine are already Sankey columns (`SK_COLS`), so the list is redundant for observation; upgrades move to Config.

## Detailed implementation

**Phase 1 (UI-only; JS allowed) — `src/escalation/templates/main_dashboard.html`:**
- **Step 0 — normalize section wrappers first.** The routing wrapper opened at `:305` is not explicitly closed before the Reviewer section (`:354`), so sidebar sections 6–10 are currently nested rather than clean siblings. Wrap each of blocks 5–10 in an explicit, self-closing container **before** moving anything, so relocations are safe cut-and-paste.
- Add a **command-bar control strip** (`:440`): model quick-switch (reuse the existing `main-model`/`main-backend` controls + `saveRoutingProfile`, reading preset + all roles so save semantics are unchanged), the three action buttons (reuse `flushModels`/`restartService('llama-swap')`/`syncModels`, preserving plain button content for the `event.target` spinner), a `renderHealthPill()` dot, and a `renderPosture()` line. Both renderers read already-fetched data; add a small aggregator invoked from the existing `refreshSlow`/`refreshHealth`/profile/HankNDory load paths (no new endpoint).
- **Relocate the full routing surface, not just the model select.** The existing routing block (`:303-352`, `:2540-2759`) also owns `#routing-preset`, the reviewer/subagent selects + local-model datalists, "Refresh lists", the saved/error message spans, and routing hints. Place the **main** control (+ preset) prominently in the strip; move reviewer/subagent + Refresh + messages/hints into the **strip expander** (per R2 — not Config, so the whole routing surface stays together and "nothing deleted" is verifiable by `test-nav-ui.cjs`).
- Move blocks into **Config**: Quality group (reasoning, review toggle, PR toggle, prompt-rewrite; HankNDory toggle added by Phase 1b), Integrations (Bridges), **Toolbox management** (`#toolbox-list` + `#cockpit-config-status` intact), Service controls (the 3 rare restarts). Keep every `id`/`onclick`/endpoint identical.
- Move the **HankNDory ledger** panel (`#review-ledger`, `loadReviewStatus`) into the Overview/Stream view.
- **Models** becomes one nav item with internal sub-tabs (Local models=`models`, Downloads=`toolbox-models`, Serving=`server-mode`) + an external "Model activity" link to `/models`; extend `switchView` (`:1092`) so the Models parent stays `active` across its sub-tabs and the Downloads polling gate (`activeView === 'toolbox-models'`) still fires. Specify: default child = Local models; sub-tab switches do not push browser history; the external `/models` link opens the standalone page.
- *Validation:* see Testing (strengthened beyond `check-html-js`).

**Phase 1b (backend increment):** `GET/POST /api/review/hankndory` in `server.rs` (mirror the FR-D `pr-guidelines` handlers + the locked `save_state` writer, extended to a third flag), `AppState` field, `daemon.rs` load, Config-Quality toggle + init. Unit tests + `code-review` pass; deploy like FR-D.

**Phase 2 (deferred, separate PR + its own Dory pass):** per-project model override store (project-key normalization defined in R-Phase2 below), `Router` resolution change with precedence tests, and the reviewer seam (`route_with_choice` snapshot, `router.rs:256-271`) must be defined — a project pin has to enter the reviewer's per-run snapshot, not only `route_resolved`.

## Testing and evaluation

Phase 1 (UI) acceptance is **objective, not just a manual walkthrough**. A new **`scripts/test-nav-ui.cjs`** follows the existing `scripts/test-benchmark-ui.cjs` pattern (Node `node:test` + a synthetic DOM via `vm`, extracting the template `<script>` — no browser, no server, no deps), run with `node --test scripts/test-nav-ui.cjs`:
- **Structural assertions**: every required `id` occurs exactly once; the four destinations and all three Models sub-tab targets (`models`/`toolbox-models`/`server-mode`) exist; the external `/models` link exists; the relocated toolbox/bridge/restart/quality controls each retain their original `onclick`/id after the move; no duplicate/missing ids.
- **Behavioral assertions** (in the synthetic DOM): `switchView('models')` keeps the Models parent active across sub-tab switches and the Downloads-polling gate still fires; `renderPosture()` maps each source to the right value and shows `loading`/`unavailable` when a source is absent; `renderHealthPill()` colors match the R-Posture mapping.
- **Regression guard**: the existing Rust suite stays green (Phase 1 changes no Rust); `scripts/check-html-js.sh` still runs (note it is JS-syntax-only and silently passes when Node/Bun is absent, `:21-25` — `test-nav-ui.cjs` must fail loudly if Node is missing, since it *is* the structural gate).
- **Accessibility/viewport** cannot be covered by the synthetic DOM and are a **documented manual checklist** (keyboard-only reachability of every moved control; posture legible without color; command-bar strip at a narrow viewport) until/unless a browser harness (e.g. Playwright) is added — flagged honestly as the one non-automated gate.
- Phase 1b + Phase 2 get real Rust unit tests (toggle persistence/no-clobber; resolution precedence R10; project-key normalization).

The earlier "no new tests are meaningful" stance is withdrawn.

## Security, privacy, reliability, and operations

- Phase 1 adds no network surface and no new persisted state; it reuses existing loopback-guarded endpoints.
- Phase 1b adds one loopback-guarded toggle endpoint + one boolean in the existing runtime-state file (same contract as FR-A/FR-D).
- **R-Phase2 (project-key safety, define before the Phase-2 Dory pass):** the raw `cwd` is connection-scoped and may be empty (peer-lookup failure, `server.rs:4116-4153`), a subdirectory, a symlink, or a git-worktree path. Phase 2 must define: normalization to a **repository-root identity** (not a raw path), empty-`cwd` fallback (→ global), symlink/worktree handling, store cleanup, and the exact main/reviewer/subagent lookup seams — including how a pin enters the reviewer's `route_with_choice` snapshot (`router.rs:256-271`), which does not pass through `route_resolved`'s lookup point.

## Rollout, migration, and rollback

- **Phase 1** ships first (global, UI-only) — fully reversible (DOM revert), no migration.
- **Phase 2** follows as an additive, flag-guarded PR; absent any project pin, behavior is identical to Phase 1 (safe default).

## Risks and mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Moving controls breaks JS wiring (ids/handlers) | Med | Step-0 explicit section wrappers; keep every `id`/`onclick`/endpoint identical; structural id assertions + walkthrough |
| `event.target` spinner targets a nested icon after a move | Med | Preserve plain button content (or pass the button explicitly) for `flushModels`/`restartService`/`syncModels` |
| Models parent/child nav loses history or breaks Downloads polling | Med | Specify default child, active-parent rule, and the `activeView` gate update; sub-tabs don't push history; `/models` stays an external link |
| Posture strip shows a stale value (4 refresh sources) | Med | Single `renderPosture()` invoked after each source refresh; explicit loading/unavailable/stale states |
| Dropping toolbox management loses capability | High (avoided) | R4 relocates the full `#toolbox-list` surface intact into Config — nothing deleted |
| "Split HankNDory toggle" implies a control that doesn't exist | High (avoided) | Phase 1 keeps HankNDory status-only; the toggle is the explicit Phase-1b backend increment |
| Phase-1 validation too weak to catch structural breakage | Med | Strengthened gate (structural + walkthrough + a11y + viewport + Rust regression) |
| Phase 2 routing-resolution change on the hot path | High | Separate PR + own Dory pass + precedence tests; inherit-by-default; reviewer-snapshot seam defined |
| Phase 2 project key from raw `cwd` (empty/subdir/symlink/worktree) | High | R-Phase2 repository-root normalization + empty→global fallback, defined before implementation |

## Open questions

*(Q-N1–Q-N3 resolved in v2 — see Decision log.)*
- **Q-N4** Phase-1b: is a HankNDory runtime on/off toggle actually wanted, or is YAML-seeded config sufficient (making R5 status-only permanently)? Defer to the user; Phase 1 stands without it.

## Decision log

- **Model is the only frequently-changed lever → elevate it above the fold; reasoning/review are set-and-forget → Config "Quality" group.** User input 2026-09-23: "I do change model often but not reasoning or review setting." 2026-09-23.
- **No separate top-level "Tune" destination.** Follows from the frequency decision; the posture strip + Config-Quality replaces it. 2026-09-23.
- **Per-project = model-only, inherit-by-default, deferred to Phase 2.** User input: "a global with per project would be nice but wouldn't that be even more complex since each project would have its own setup?" Resolution: inheritance means zero per-project setup; scope to model only; ship global-first. 2026-09-23.
- **Bridges kept, moved to Config.** User input: "Keep bridges but move them in Config." 2026-09-23.
- **Split HankNDory status (Overview) from toggle (Config).** User input: "do split HankNDory status from toggle." 2026-09-23.
- **Drop the sidebar "All toolboxes" list (Sankey covers backend/engine); toolbox upgrades → Config.** User input: "not sure I need all the toolboxes listed in the left nav since they should be captured in the sankey. Maybe put upgrading them in Config." Verified: `SK_COLS` has backend+engine (`:1705`). 2026-09-23.
- **Above the fold: Eject/Flush + Reload llama-swap + Sync → OMP; other restarts → Config.** User input: "Eject a model is used all the time so maybe it should be above the fold. Reloading llama-swap config, and models sync to Harness are used often too. The other buttons not so much." 2026-09-23.
- **Q-N1 resolved → posture strip is read-only; changes happen in Config.** Frequency data (reasoning/review rarely changed) makes quick-toggle unnecessary. 2026-09-23 (v2).
- **Q-N2 resolved → quick-switch shows main prominently, reviewer/subagent in an expander**, reusing the existing `saveRoutingProfile` (explicit Save, all roles preserved). 2026-09-23 (v2).
- **Q-N3 resolved → Config is a grouped page with headings** (Quality / Integrations / Toolbox management / Service controls / Raw config); sub-tabs only if it grows. 2026-09-23 (v2).
- **Control strip lives in the command bar, not the sidebar** — the sidebar scrolls (`:180`), so "always visible" requires the non-scrolling command bar (`:440`). 2026-09-23 (v2, Dory round-1 §Blocking-4).
- **Toolbox list relocated intact to Config, not deleted; HankNDory stays status-only in Phase 1 with the toggle carved to Phase 1b.** 2026-09-23 (v2, Dory round-1 §Blocking-1/3, corrected the v1 factual errors).

## Referenced files

- `src/escalation/templates/main_dashboard.html` — the dashboard; all Phase 1 changes live here (sidebar `:180`, nav `:215`, health `:247`, toolbox list `:299`, routing roles `:303`, reviewer `:354`, hankndory `:373`, nudge `:383`, bridges `:404`, tools `:427`; `switchView` `:1092`; `SK_COLS` `:1705`; `flushModels` `:2337`, `restartService` `:3902`, `syncModels` `:3933`).
- `src/router.rs` — `route_tagged`/`route_resolved` receive `cwd` (`route_tagged` params `:243-253`; `route_resolved` `cwd` param `~:278`); the reviewer path `route_with_choice` (`:256-271`) uses a per-run snapshot and bypasses `route_resolved`'s lookup point (Phase-2 seam). The two proxy handlers are in `src/server.rs`, not `router.rs`.
- `src/routing_profile.rs` — global `ProfileStore` (model roles); Phase 2 per-project store home.
- `src/server.rs` — the reused endpoints (`/api/routing-profile`, `/api/routing-models`, `/api/review/*`, `/api/nudge`) and `AppState`.
- `docs/design/hankndory-brainrouter-integration.md` — the review/HankNDory/PR features whose controls this nav reorganizes.
- `scripts/test-benchmark-ui.cjs` — the Node `node:test` + synthetic-DOM harness this design's `scripts/test-nav-ui.cjs` (new, Phase 1) mirrors; `scripts/check-html-js.sh` — the existing JS-syntax lint (insufficient alone).

## Dory validation record

- **Round 1 — comprehension + critic + readiness (v1), isolated sub-agent** (fresh-context `rubber-duck`, given only the doc + referenced files; **sub-agent-isolated, not a separate top-level session**). **Comprehension/clarity: PASS. Readiness: NOT READY.** Verified the sidebar/`switchView`/`SK_COLS`/action-fn/`cwd`/global-profile claims. **Refuted three v1 claims:** (a) the Sankey does *not* make the toolbox list redundant (it shows served traffic only; the list owns inventory/status/versions/create/update/adopt/delete); (b) there is *no* HankNDory UI toggle to move (`/api/review/status` is GET-only; `hankndory_integration` is YAML-seeded); (c) the posture data is *not* all in `refreshSlow` (health/profile/HankNDory are separate paths), and the routing wrapper at `:305` is unclosed so sections 6–10 are nested (unsafe cut-and-paste). **Blocking:** R8 "DOM-move only" impossible as written; Models parent/child nav contract unresolved; toolbox management can't be replaced by the Sankey; quick-switch placement/behavior undecided. **Important:** posture freshness/health semantics; `event.target` couplings; weak validation gate; missing a11y acceptance; Phase-2 project-key safety. **All resolved in v2** (R1/R2 nav + strip decided; R4 relocate-intact; R5 status-only + Phase-1b toggle; R8 reworded; R-Posture/R-A11y/R-Phase2 added; validation gate strengthened; citations corrected). Self-audit (sub-agent): relied only on the doc + repo + standard HTML/DOM/a11y knowledge.
- **Round 2 — readiness pass (v2), isolated sub-agent.** **Verdict: NOT READY**, but confirmed **all four round-1 blockers resolved**. Found two new blockers — both *internal-consistency errors in the v2-added sections*, not design flaws: (1) R-Posture listed the wrong sources (named the routing profile instead of HankNDory's `/api/review/status`), left the health color-mapping undefined, and R-A11y pointed the HankNDory posture entry at a Config toggle Phase 1 doesn't have; (2) the strengthened Testing gate named outcomes but no runnable harness/file. Plus an important finding (the quick-switch relocation omitted `#routing-preset`, reviewer/subagent selects, Refresh, messages, hints). **All resolved in v3**: corrected the `renderPosture()` source→value map + health mapping; repointed the HankNDory posture link to the Overview ledger; named `scripts/test-nav-ui.cjs` (mirroring `test-benchmark-ui.cjs`) with structural+behavioral assertions and an honest manual-only a11y/viewport carve-out; enumerated the full routing surface; normalized the Models labels. Self-audit (sub-agent): relied only on the doc + repo.
- **Round 3 — readiness pass (v3), isolated sub-agent. NOT READY → resolved in v4.** Confirmed R-A11y and the testing gate resolved (and *ran* the harness precedent `test-benchmark-ui.cjs`: 38/38, corroborating the `test-nav-ui.cjs` approach). Two remaining blockers, both concrete spec fixes: (1) the health pill referenced "Bonsai off" but `/api/service-health` does not return Bonsai (it's `/api/bonsai`), and color precedence was undefined for `loading`/unhealthy-Manifest/`cloud_fallback`; (2) the routing-role destination was contradictory (R2 strip-expander vs. Detailed-impl "expander or Config"). **Both closed in v4**: health pill uses service-health fields only with a full first-match precedence (Bonsai removed); reviewer/subagent pinned to the strip expander. These were deterministic corrections, not design disputes, so per the method (stop when findings are specification-level and each has a concrete resolution) no round-4 critic was run; the doc is marked ready-for-human-review.
- **Isolation note:** all three rounds were fresh-context sub-agents (no conversation history), **sub-agent-isolated, not separate top-level sessions** — recorded as such rather than certified.

## Human approval

**Pending.** The user framed this as "discuss, not code yet" and was unavailable for the sequencing decision (instructed: "Work autonomously and make good decisions"). Per HankNDory rule 7, this design stops at the human-review gate: it is drafted and Dory-reviewed, but **no implementation** proceeds until the user approves and lifts the "not code yet" hold. Recommended first step on approval: Phase 1 (global, UI-only) as its own PR.
