This file is part of the design: the Map lives in `v2-streamlining.md`; this is leg L3's full spec.

# Leg L3 in full — One entry + role cards + decision slot + Credentials page

Goal (R4): a harness is configured ONCE to brainrouter; the app owns the model decision. Four subsystems land together: harness one-entry, role cards, the decision-model slot, the Credentials page.

## Role resolution (the new center)

- Store: `role_cards.json` (0600, atomic — `src/project_pins.rs` machinery is the precedent) replaces `routing_profile` + `project_pins` stores via migration (snapshot; `--dry-run`; restore `v2.0-p3`). Record: `{scope: global|<canonical-git-root>, role: main|review|subagent, assignment: "auto" | "<model-key>"}`. Lookup: project scope → global → default (`main: auto`, `review: auto`, `subagent: auto`).
- Role signal: `x-brainrouter-role: main|subagent|review` header (default `main`). Compatibility aliases ride existing grammar: `subs`/`brainrouter/subs` selector ⇒ subagent, review loop calls the router internally with role `review` (the strict-reviewer path keeps its semantics). Explicit selectors (`local`, `cloud`, `cloud/<id>`, bare model id) BYPASS role cards and keep v1 routing meaning (Selector-vocabulary contract; the R4 "explicit model ID preserved" acceptance).
- Resolution algorithm (every proxied request): explicit selector? → v1 path (no card). Else role = header/alias (default main) → card assignment: pinned model-key → route directly (chain fallback to `auto` on hard failure, logged); `auto` → decision slot.

## Decision-model slot (R4; replaces Bonsai special-casing)

- `src/decision.rs`: `trait DecisionModel { fn kind(&self) -> DecisionKind; fn available(&self) -> bool; async fn classify(&self, RequestFeatures) -> Result<Decision, DecisionError>; }`; `RequestFeatures = {role, selector, privacy, project, msg_count, approx_prompt_tokens, has_tools, code_density, last_user_excerpt}` (excerpt only for model kinds, never for webhooks/plugins); `Decision = {route: local|cloud, tier: fast|balanced|deep, confidence: 0..=1, rationale: String}`.
- Kinds (config `decision_model.kind`): `heuristic` (default, always available); `openai-compatible` (`{base_url, model, prompt_profile, timeout}` — any chat model; **Bonsai auto-migrates to exactly this** from `bonsai.*`, `prompt_profile: bonsai`, preserving the current classifier prompt); `structured-schema` (typed-question contract — Clef-style: request features + fixed JSON question schema in, strict JSON decision out; shares the transport of openai-compatible, differs in prompt_profile + response validation); `webhook` (`{url, method: POST}` — features JSON in, decision JSON out; no prompt content leaves the box); `plugin` (`{plugin_id}` — daemon→plugin `decision.classify` RPC; plugin must declare the grant).
- Guardrail clamps (heuristic-level, applied to EVERY decision including model decisions, in order): privacy `local-only` forces `route=local`; approx tokens > 0.9 × resident ctx forces `tier=deep` (and pool by ctx ≥ needed); explicit selector never reaches the slot.
- Fallback: classify timeout/parse-fail/low-confidence (<0.4) → `heuristic` decides, event records `fallback: heuristic`, dashboard row shows it; slot availability (server down) = same path with `fallback: unavailable`.
- Selection after the decision: candidate pool from model catalog + live backend state (loaded > loading > known; ctx ≥ features requirement; tool support if `has_tools`); ranked by tier fit (pool tiers from catalog metadata); chain fallback across candidates (replaces the single `fallback_model`; `fallback_model` config remains as a compatibility seed for the chain, logged as deprecated).
- Config: `decision_model: {enabled, kind, endpoint?, model?, prompt_profile?, timeout_ms?, plugin_id?, min_confidence}`; `bonsai.*` migrates (enabled+server+prompt profile preserved verbatim — routing behavior must not shift at the migration boundary). Invariant: prompt-rewrite ⇒ `decision_model.enabled` (enforced at the same handler; heuristic satisfies it).

## Harness one-entry (install)

- `brainrouter install` templates (all harnesses): exactly one model entry (`brainrouter/auto`); remove per-model listings + `sync-omp` command + endpoint. Bare model-id selectors still route (compat), invisible in templates.

## Credentials page (L2 store → UI; the owner's keys screen)

- Dashboard Credentials tab: list (id, kind, grants, created; masked value for `store` kind), add/edit/delete (id, kind, value | env var name | file path, plugin grant checkboxes), `Test` (store: length+charset sanity; env: var resolves non-empty; file: path readable, 0600 warn), usage audit ("granted to: bridge plugin / L6 provider bindings"). All writes loopback+CSRF-gated (existing pattern). `credentials.json` format unchanged from L2.

## Dashboard

- Command strip → policy chips: `privacy: local-only|cloud-ok`, `quality: fast|balanced|maximum` (floor on tier ranking). Role cards panel: 3 cards (Main/Reviewer/Subagent) × (Auto | pin model) + per-project scope list (migrated pins). Decisions panel (read-only): last N = role, features summary, decision, fallback flag, chosen model. Sankey ROUTING node: `role → route/tier → model`.

## CLI

- Replace `routing-profile`: `brainrouter roles get|set <role> auto|<model-key> [--project <path>]`; `brainrouter decision show|test <text>`; `brainrouter credential list|add|remove|test <id>`.

## Removals (one-way door)

- `routing_profile.rs` + presets + CLI `routing-profile` + dashboard role-chooser widgets + presets command; `sync-omp` + endpoint; per-model harness template listings; `project_pins` store (migrated). Proxy selectors, review loop, classifier server process management (`bonsai_server.rs` — kept: it is the openai-compatible launch backend for the migrated preset) stay.

## Contracts kept

Selector vocabulary (L7 range), event schema additive (`role`, `decision`, `decision_fallback`, `chosen_model` fields), v1 configs boot (routing_profile/pins/bonsai migrations, `--dry-run`, snapshot, restore `v2.0-p3`), credentials store unchanged, plugins RPC additive (`decision.classify` daemon→plugin with grant).

## Build order

role store + migration → decision.rs (trait + heuristic + openai-compatible + migration from bonsai) → router role resolution → structured-schema/webhook/plugin kinds → selection pipeline + chain fallback → CLI (roles/decision/credential) → Credentials tab → policy chips + role cards panel + Decisions panel + Sankey node → install templates one-entry + sync-omp removal → docs.

## Acceptance (Ray)

(1) Role resolution matrix: header×card×project-scope×explicit-selector (explicit bypasses cards; project beats global; pinned bypasses decision; subs alias). (2) Decision slot: heuristic parity table (R4's v1 selector matrix outputs), openai-compatible against mock classifier (Bonsai-migrated config yields decisions identical to v1 classifier behavior on fixture prompts), structured-schema validation (strict JSON; malformed → fallback event), webhook (features JSON out, decision in, no excerpt), plugin kind via `decision.classify` grant; guardrail clamps (privacy forces local; ctx forces deep); low-confidence fallback recorded + visible. (3) Credentials page: CRUD + grants + Test all kinds + masked-everywhere audit (extends L2 test surface to UI endpoints). (4) One-entry: install templates contain exactly the auto entry; `sync-omp` gone (grep + CLI help). (5) Events: additive fields present, v1 consumers parse; Sankey node renders decision. (6) Migrations: v1 `routing_profile`/`pins`/`bonsai` config boots via dry-run + live, `roles get` reflects migrated assignments, classifier behavior continuity on fixture corpus (v1 comparison harness). (7) `cargo test --workspace` + Node + html-js green.

Residual risks stated: selection-quality regressions are governed by the Map's replay-eval loop (routing events corpus includes decision fields from this leg — the eval harness itself lands with L4's supervisor where it becomes operational).
