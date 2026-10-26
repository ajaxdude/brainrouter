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
12. **`peer-info.ctx` for local keys** — no ctx source is pinned for
    llama-swap keys; emitted as `0` unless the key is a provider model.

## Verification run

- `cargo test --test v2_l6_acceptance` — 25/25 green (the bar).
- Prior suites: `cargo test --lib` + L1, L2, L2-fixpass, L3,
  L3-plugin-decision, L4, L5, L5-smoke acceptance suites.
- `cargo check` / `cargo clippy` both modes; `scripts/check-html-js.sh`.
- Live smoke: LAN second listener against mock gateways (auth/rotation/rate/
  admin-404/expose/peer-info), role-header forwarding to a mock peer,
  qualified selector routing, and budget stop exclusion (fixture-tier rows in
  Ray's suite exercise all three live).
