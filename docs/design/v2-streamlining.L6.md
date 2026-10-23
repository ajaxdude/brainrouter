This file is part of the design: the Map lives in `v2-streamlining.md`; this is leg L6's full spec (drafted at L5 closure; leg-scoped review pending).

# Leg L6 in full — Cloud providers, Manifest absorbed, LAN federation

Goal (R7): cloud is brainrouter's own decision — providers registered directly, keys from the store, the decision model choosing the cloud model. Federation-lite: a role can bind to any OpenAI-compatible endpoint on the LAN, including a peer brainrouter. The distributed-role acceptance flows land here.

## Providers (`providers.json`, state dir, 0600 atomic — fleet.json machinery)

- Record: `{id, kind: openai-compatible | anthropic | manifest, base_url, credential_id (store), models: [{id, alias?, ctx, tools, quant?}], auto: bool (manifest kind: let the gateway pick), enabled}`. CLI `brainrouter provider add|list|remove|test <id>` (mutations via daemon, L3 single-writer pattern); dashboard Providers section on the Credentials page (bind credential → provider, Test = cheap completion probe, live per-provider breaker state).
- Outbound adapters: `provider/openai.rs` generalized from manifest-only to any OpenAI-compat base_url (auth header from credential resolution); NEW Anthropic outbound adapter (wire translation, streaming, error mapping — the inbound Anthropic translator `anthropic.rs` is the reverse-direction reference). Per-provider circuit breakers via `health.rs` (already provider-name keyed; manifest today = one such entry, generalized to N).
- Auto-selection for cloud: the L3 auto pool extends — decision `route=cloud` ⇒ pool = enabled providers' models filtered by ctx/tools, ranked by tier fit (catalog metadata; `quant?` fields); chain = next cloud candidate, then local candidates (preserving v1's managed cloud→local fallback hop for role main/subagent; role review cloud stays strict per L3).
- Migration (v2.0-p6, snapshot + dry-run + restore per contract): `manifest.enabled/manifest_url/model/api_key_env` ⇒ provider `{id: "manifest", kind: manifest, auto: true, credential env:<api_key_env>}` — routing behavior byte-identical through the migration boundary (the shim-parity pattern from L4: same breaker key, same stream-peek actual-model reporting, same fallback). `manifest.*` YAML keys removed post-migration. Retirement (R8): once ≥1 native provider serves the owner's cloud roles, manifest stays as a removable kind; deletion only with owner sign-off.
- The owner's reference deployment, first-class: `deepinfra` provider (openai-compatible, base_url `https://api.deepinfra.com/v1/openai`, credential `deepinfra_token` from the store, model `deepseek-ai/DeepSeek-V4-flash` ctx 163840 tools true) bound to the Reviewer card (L3 migration moves `review.forced_*` there). Raon/fresh-session failure modes closed: reviewer model visible + Test-able on the Credentials page; "route is down" shows as provider breaker state, not a dead session.

## LAN federation (Q3: bearer token — accepted risk: plaintext HTTP on the trusted home LAN, owner's reserved LAN-enable nod carries it)

- Inbound: `lan: {enabled: false, bind, token_credential (store id), expose: [model-keys]}`. Disabled by default (v1 loopback posture unchanged). When enabled: the proxy binds `bind`; ONLY `/v1/chat/completions`, `/v1/messages`, `/v1/models` (filtered to `expose`), and `GET /api/peer-info` answer for non-loopback peers — route-guarded at the middleware level (admin/dashboard APIs never reachable off-loopback, tested). Auth: `Authorization: Bearer <token>`, constant-time compare against the credential's resolved value; 401 otherwise. `peer-info`: `{host_id, version, models: [{key, ctx, tools, state}], serving: [...]}` — the discovery feed for the provider editor (add peer → pick exposed models) and for role cards (pin to `peer/<model>` selectors).
- Outbound: a peer brainrouter is just an openai-compatible provider (`base_url http://host:9099/v1`, credential = the peer's lan token); `peer/<model>` card selectors resolve through the provider. Cross-host review/dev/plan rides L3 role headers.
- Events: additive `hop: {host, endpoint}` on LAN-provider routes; Sankey ROUTING shows the hop (fedora1 node visible in the flow), cards group unchanged.
- Distributed-role acceptance flows (host tier, pilots): (a) plan→fedora1/hipfire-Swift, dev→strixite local, review→deepinfra — one session, three hops; (b) plan→mac/ds4 (L8), dev→cloud, review→strixite. Fixture tier: mock peer servers.

## Removals

`route_cloud` manifest special-casing in the router (providers replace it); `manifest.*` YAML (migrated); `config.manifest` struct (post-migration); manifest-only naming in health breaker (generic keys already). The Manifest binary itself is untouched on fedora hosts (it's their own daemon) — brainrouter just no longer requires it.

## Contracts kept

Selector vocabulary: `cloud`/`cloud/<id>` now resolve through providers (L7 note: rebind completes here — the L3 contract's L6 rebind point); pinned/auto strictness semantics unchanged (L3); credentials store unchanged (L3 UI extends); events additive; v1 configs boot (manifest migration, snapshot, restore `v2.0-p6`); LAN default-off keeps v1 attack surface.

## Build order

providers.json + CLI + Credentials-page binding UI → openai-compat generalization + anthropic outbound + per-provider breakers → cloud pool/chain in the router → manifest migration (parity harness: v1 cloud corpus through migrated config) → LAN inbound (guard + token + peer-info) → provider-from-peer editor flow → hop events + Sankey → docs/UPGRADE (fedora hosts, deepinfra setup).

## Acceptance (Ray)

(1) Provider CRUD + Test probes per kind (mock upstreams); breaker trip/health per provider visible. (2) Cloud auto matrix: tier/ctx/tools filtering, ranking, chain order (cloud→cloud→local for main/subagent; review strict; pinned cloud = L3 semantics). (3) Manifest migration parity: v1 cloud corpus (success, stream-peek actual-model, breaker, cloud→local fallback) byte-identical through v2.0-p6; `manifest.*` keys gone post-migration; restore round-trip. (4) LAN off = v1 posture (loopback only, /v1 refused off-loopback). (5) LAN on (fixture peers): token good/bad/rotated, expose filter (unlisted model 404/403), admin routes unreachable off-loopback, peer-info shape, provider-from-peer add, plan-role stream to a mock peer (header + role card + hop event + Sankey field). (6) DeepInfra reference config renders on the Credentials page (model fixture, no live calls). (7) All prior suites green (L1–L5 acceptance, workspace both modes, node, html-js).
