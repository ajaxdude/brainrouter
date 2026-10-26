# v2 L6 implementation log: providers + Manifest absorbed + LAN federation

Design: `docs/design/v2-streamlining.L6.md` (L6-v5, spec binding);
Map: `docs/design/v2-streamlining.md` (R7/R8, cross-leg contracts, Security
amendment + budget decision); charter:
`docs/design/v2-streamlining.charter.md`. Branch `v2-p3-roles`, base
`dad658d` (L1-L5 integrated). No commit made (builder contract). Ray's suite
`tests/v2_l6_acceptance.rs` was run, never edited.

## Build order landed

### 1. Provider registry (`src/providers.rs`)

`providers.json` in the state dir, 0600 atomic (the `plugins::store` atomic
write). Record shape exactly as spec "Providers": `{id, kind:
openai-compatible | anthropic | manifest, base_url, credential_id?, auto,
models: [{id, alias?, ctx, tools, tier, price?}], lan_peer, enabled,
monthly_budget_usd?, budget_mode}`. Validation: id/base_url/model/tier/budget
sanity; `auto` is manifest-kind only; two ENABLED `auto: true` manifest
providers refused at add time with both ids named (rule (b) uniqueness).

API (spellings picked from Ray's declared probe list and kept stable):
`GET /api/providers`, `GET /api/providers/list`, `POST /api/providers`,
`POST /api/providers/add`, `POST /api/providers/upsert`,
`PUT /api/providers/{id}`, `DELETE /api/providers/{id}`,
`POST /api/providers/remove`, `POST /api/providers/{id}/remove`,
`POST /api/providers/{id}/test`, `POST /api/providers/test`. Mutations are
loopback+CSRF gated like every other destructive API. CLI: `provider
add|list|remove|test`. Dashboard: Providers section on the Credentials page
(add/update, Test, Remove, live breaker state, budget badge, declared models)
and per-provider model lists on the Models page.

New additive `HealthTracker::state()` (`healthy | degraded | open`, with the
production 60 s cooldown transition), keyed by provider id. The migrated
Manifest provider keeps the literal `manifest` key = today's breaker key.

Test `l6_provider_crud_and_test_per_kind_auth_headers`,
`l6_provider_breaker_trip_state_and_heal`,
`l6_provider_dual_auto_manifest_refused`,
`l6_provider_state_file_0600_and_masked_everywhere`.

### 2. Adapters

- `src/provider/openai.rs`: generalized to any OpenAI-compatible base_url;
  `chat_completion_with_role` forwards `x-brainrouter-role` (only used for
  `lan_peer`). Bearer from credential resolution; auth-less when
  `credential_id` is absent.
- `src/provider/anthropic.rs` (NEW): canonical OpenAI request → Messages body
  (`system` extraction, tools, stop_sequences, temperature/top_p), auth
  `x-api-key` + `anthropic-version: 2023-06-01` and NEVER Bearer; SSE
  translated back to OpenAI chunks (text, thinking → reasoning_content,
  input_json_delta → tool_calls, stop mapping, `message_stop` → `[DONE]`).
- `manifest` kind reuses the OpenAI adapter and keeps the v1 stream-peek
  (64 KiB/TTFT window, byte-preserving reassembly), pseudo-success sentinel
  (fallback WITHOUT a breaker failure), and empty-stream failure semantics.

### 3. Resolution matrix + pool (`src/providers.rs::resolve_cloud_target`)

- bare `cloud`: enabled manifest-auto provider (v1 parity) → else the auto
  pool's top candidate. Pool = enabled non-`lan_peer` providers, ctx/tools
  filtered, tier-fit ranked, stable tie-break on provider record order then
  model list order; `stop`-exhausted providers excluded.
- `cloud/<id>`: (d) qualified `cloud/<provider>/<model>` only when the first
  segment names an ENABLED provider, checked before (a); (a) unique
  id/alias match; (c) multi-match = 4xx naming every provider (server maps
  `CloudSelectorError` to 400, no upstream call); (b) no match + exactly one
  enabled manifest-auto provider = verbatim passthrough of the whole
  remainder (the `cloud/deepinfra/deepseek-ai/...` case).
- A `cloud/` prefix never resolves to a local key (no llama-swap escape);
  managed cloud routes keep the v1 two-step shape and NEVER widen.

### 4. v2.0-p6 migration (`src/migration.rs`)

Exact v1 keys (no `manifest.model`): `enabled` + `base_url` + `api_key_env`
→ provider `{id: "manifest", kind: manifest, base_url, credential:
env:<api_key_env>, auto: true, models: []}`, `enabled` carried. Snapshot at
`migrations/v2.0-p6/config.yaml` (stable per-phase dir, pre-migration bytes),
`config.yaml.bak`, YAML block removed, monthly ledger created 0600.
`serve --dry-run` prints the preview and writes nothing. `migration restore
v2.0-p6` restores the YAML byte-for-byte and removes `providers.json` / the
ledger when they were absent pre-migration. One-shot: no manifest block = no
writes. `manifest` in `BrainrouterConfig` is now optional so the migrated
file boots.

### 5. LAN second listener (`src/lan.rs`)

`lan: {enabled, bind, token_credential, expose, rate:{burst,per_sec}}`
(default disabled → listener never created). Bind + spawn in `daemon.rs`
after boot; boot logs the plaintext-LAN risk marker. Route table is
listener-level: `POST /v1/chat/completions`, `POST /v1/messages`,
`GET /v1/models` (expose-filtered), `GET /api/peer-info`; everything else 404
(405 for wrong methods on those four). Bearer compared
`constant_time_eq` against the resolved vault value on every request
(rotation immediate). Per-token bucket → 429. Expose double-enforced RAW +
RESOLVED (`resolve_cloud_target` / `resolve_bare_model`), refusing alias and
qualified escapes. `peer-info` shape per spec, read-only. LAN inbound role
headers go through the peer's own role cards (`route_tagged_role`).

### 6. Budget guard (`src/provider_budget.rs`)

`provider_budget.json` 0600, `{month: "YYYY-MM", providers: {id: {spent_usd,
estimated_requests, actual_requests, input_tokens, output_tokens}}}`.
Calendar rollover on key change (restart cannot reset). Cost = price ×
usage tokens when the stream carries a `usage` frame; otherwise the
`(bytes+3)/4` estimator, flagged `estimated`. `warn` = event (`RouteEvent.
budget`) + Providers badge, never blocks. `stop` = refused before dispatch as
a PRIMARY failure (managed `fallback_model` hop fires; review strictness
escalates) and excluded from the auto pool.

### 7. Credential export

`brainrouter credential export --env-line <file> [--force]` (also under
`brainrouter cli credential export`): 0600 atomic, refuse-if-exists,
`store` → `NAME=VALUE`, `env`/`file` → comment references only, masked list
rendering unchanged, CLI emits an audited log event. NAME = upper-snake-case
of the credential id (valid shell identifier).

### 8. Events + dashboard

Additive `RouteEvent.provider`, `RouteEvent.hop {host, endpoint}` (only for
non-loopback provider base_urls), `RouteEvent.budget {provider, mode, state,
month, spent_usd, monthly_budget_usd}`. Sankey ROUTING maps any provider
route to cloud and appends ` · lan <host>` when a hop exists (cards grouping
unchanged). Observability's poll loop gained an additive `providers` fan-out
(registry + breaker + budget + cheap `/models` probe).

### 9. Admission

`cloud_available` (≥1 enabled non-`lan_peer` provider) replaces
`AdmissionInputs.manifest_enabled` at every consumer (review admission,
router catalog, daemon). `AdmittedBackend::Peer` + `BackendDecision::Peer`:
a qualified `cloud/<lan-peer>/...` review card derives
`configured_backend = peer` (no local memory gate, cloud-parity strictness).

## Deviations / gaps (with spec cites)

1. **Ledger file name** — spec "Budget guard" names `budget.json`; L4 already
   owns `budget.json` for swap reservations. Chosen spelling:
   `provider_budget.json` in the same state dir (Ray's suite accepts any
   state-dir ledger file and declares the collision as a spec gap). Stable.
2. **Providers API route spellings** — spec names verbs, not paths. Picked
   the probe-list spellings above; stable.
3. **Breaker JSON shape** — spec pins `state()`, not its wire shape. Payload
   carries `"breaker": "healthy|degraded|open"` on every provider record.
4. **Price spelling** — spec "$/1K in,out"; chosen `{"in": N, "out": N}`
   (deserializer also accepts `input`/`output`).
5. **Export NAME spelling** — unpinned by the spec; upper-snake credential id.
6. **Vault freshness** — provider credentials resolve the vault FILE when it
   exists (Ray's fixtures seed the vault after boot), falling back to the
   daemon's in-memory copy (in-memory test stores). Rotation via the API is
   observed by both paths.
7. **LAN rate limiting excludes `/api/peer-info`** — spec's bucket is
   "per-token" for the listener; the read-only discovery/readiness probe is
   not an inference path, and the acceptance row expects a full `burst: 2`
   after readiness plus rotation probes. `/v1/*` routes are limited.
8. **Export audit surface** — no HTTP audit route is pinned; the CLI logs the
   audit event and the lane 404s `/api/credentials/export` (declared gap in
   the suite header).
9. **`manifest` config block stays parse-compatible** — optional with
   defaults so pre-migration YAML boots and the post-migration file (block
   removed) loads; `manifest.enabled: true` still requires a `base_url`.
10. **Conversation fingerprint without a system message** — Ray's
    distributed-role fixture sends bare user requests; the anchor is now the
    first user message (system hashed when present). An empty user message
    still yields no conv_id (old bucket grouping).
11. **Anthropic non-stream mode** — not implemented: the router always speaks
    streaming (`stream: true`), matching the OpenAI adapter contract.
12. **P6 strips `manifest:` textually, not via a serde round-trip** — a
    round-trip re-emits EVERY key with current spellings (and collides with
    the L7 `llama_swap:`→`swap:` migration by rewriting the section before p7
    runs). `remove_manifest_block` removes only the top-level block, leaving
    all other bytes intact (unit-tested).
13. **`peer-info.ctx` for local keys** — no ctx source is pinned for
    llama-swap keys; emitted as `0` unless the key is a provider model.

## Review round (Bruce L46): GO-with-gate disposition

- **L6-1 (blocking) — managed expose bypass.** `lan.rs` now enforces expose
  against `Router::expose_candidates`, a preview computed with the SAME
  helpers dispatch uses (card resolution, decision slot + clamps, local chain,
  cloud resolution), including the managed substitutions (`local` ⇒
  `fallback_model`, `subs` ⇒ `subs_model`, `auto` ⇒ chain/pool, `cloud` ⇒
  target + fallback) and the post-alias values (`swap_catalog` alias and
  provider id/alias). EVERY candidate must be exposed or the LAN request is
  400. Unit tests in `src/lan.rs` cover raw, managed-substitution, alias, and
  constant-time compare. Wire note: a managed token (`local`/`auto`) therefore
  requires its resolved fallback/chain models to be exposed too.
- **L6-2 — breaker blind to mid-stream death + badge semantics.** The provider
  meter (`BudgetCaptureStream`, now attached to every provider route from
  `route_resolved` after the event id exists) reports `HealthTracker::
  report_failure(provider)` on a stream error, accrues usage/estimator as
  before, and backfills the already-emitted event's `budget` badge with
  POST-accrual spend via `RoutingEvents::update_budget(event_id, ..)` (the
  same correlation TPS capture uses). The badge emitted at headers remains the
  pre-dispatch view for ~one stream duration.
- **L6-3 — Anthropic role forwarding.** `AnthropicProvider::
  chat_completion_with_role` sends `x-brainrouter-role`; the adapter enum
  forwards the effective role for both wire families.
- **L6-4 — p6 one-shot marker + atomic restore.** `migrations/v2.0-p6/.done`
  is written after a successful migration; a later re-added `manifest:` block
  is a no-op (restore-then-re-add cannot clobber the snapshot). Restore writes
  the YAML through the 0600 atomic machinery.
- **Nits.** `ProviderBudget` recovers poisoned locks and writes the ledger
  file OUTSIDE the mutex; provider store locks recover poison too; LAN auth
  and provider credentials both use `CredentialStore::resolve_live`
  (file-first rotation); `peer-info` host/version risk is stated in UPGRADE;
  credential export appends `credentials.audit.jsonl` (ids only); provider
  Test probes the first declared model (manifest `auto`) and refuses to invent
  `"probe"`; `hop.endpoint` is the concrete `/chat/completions` or `/messages`
  URL; `validate_record` rejects `user:pass@` base URLs; LAN `/v1/messages`
  carries the real peer address.

## Verification run (final, after the Bruce L46 round)

- `cargo test --test v2_l6_acceptance` — **25/25 green** (the bar; suite
  untouched).
- `cargo test --lib` — **487/487** on the combined L6+L7 tree (includes the
  new LAN expose, budget rollover/estimator, provider resolution, migration
  stripper, and peer-admission unit tests).
- Prior suites: L1 6, L2 14, L2-fixpass 3, L3 21, L3-plugin-decision 7,
  L4 17/17, L5 11/11, L5-smoke 9/9, `daemon_availability` 4 — all green on
  the combined tree.
  (An earlier run showed L5 9/11 with the `external-occupant` gate firing on
  the owner's live `strix_server`; the host freed the device and the rows are
  green.) An intermediate combined-tree run failed L4 on L7's p7
  `llama_swap:`→`swap:` config rewrite (test precondition, not L6); Builder-L7
  resolved it and L4 re-verified 17/17.
- `cargo check` + `cargo clippy`, default and `--features benchmarks`: no
  errors (only pre-existing warnings elsewhere).
- `node scripts/test-benchmark-ui.cjs` (0 fail), `scripts/check-html-js.sh`
  OK.
- Live smoke: LAN second listener against mock gateways (auth + rotation,
  rate burst, admin 404, expose raw/resolved/managed, peer-info), role-header
  forwarding to a mock peer (both wire families), qualified selector routing,
  and budget stop exclusion (Ray's fixture rows exercise all live).
- Coordination note: v2 L7 landed its `llama_swap`→`swap` rename in the same
  worktree mid-leg; the L6 files (providers/budget/lan/router provider arms/
  migration p6/credentials/cli) were kept disjoint and the battery above was
  re-run on the combined tree.
