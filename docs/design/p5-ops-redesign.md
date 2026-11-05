# P5 — Role-card console, native strixite swapping, peer review chain

Status: DRAFT (awaiting Dory gates / approval)
Date: 2026-11-04
Owner: Jerome
Program leg: v2.0-p5 (successor to the merged L1–L7 work; independent of L8 Metal spike)

## 0. Locked direction (interview, 2026-11-04)

| Decision | Outcome |
|---|---|
| Primary selection surface | **Three true role cards** — Main / Review / Subagent, each with its own model picker. The shared `#card-model` text input is deleted. |
| Management surface | **One Ops view** — Engines, Models (local/cloud/peer), Downloads, Serving collapse into a single resident-first page. |
| strixite lifecycle | **Full native cutover** — strixite models enter the swap catalog; brainrouter spawns/kills `strix_server` like llama-swap does llama-server, after soak + boot-refusal gate. |
| Reviewer backend | **Pinned peer + ordered failover chain** — review card names a primary; brainrouter walks a configurable chain (peer → peer → cloud) with health checks. |
| Selector grammar | **Exact pins survive, grammar hidden** — `auto` is the only selector humans see in the UI; exact-model pinning remains as an explicit "pin exactly this model" affordance and over the wire. |

Non-goals: automatic peer *ranking* (ordered chain only), peer autodiscovery, Server-Mode
containers as routable traffic, Quant Lab / Model Compare execution, L8 host classes.

---

## Part A — Role-card console (UI)

### A0. Evidence of today's complexity
Model selection is reachable from 7 surfaces: command strip, Models (Local/Downloads/Serving),
Engines, Serving launch forms, Config (incl. two raw-YAML editors), Credentials, plugins.
Confirmed defect: `card-model` (`src/escalation/templates/main_dashboard.html`, `cardAssignment`)
is ONE input appended to all three roles — different per-card cloud ids are unsavable, and
`loadRoleCards()` backfills it from whichever role parsed last. Grammar prose
(`cloud/<id>`, bare keys) leaks into labels.

### A1. Per-role pickers
- Each card gets `<select id="card-<role>">` with groups:
  - `auto` (default), `local` (backend default), `cloud` (cloud pool)
  - **Pin exactly this model…** optgroup, populated from `/api/routing-models` extended to
    return grouped entries: `{ local: [...], peer: [{provider, id, model, healthy}], cloud: [{provider, id}] }`.
  - A pinned choice is compiled client-side to the existing assignment grammar
    (`cloud/<provider>/<id>`, bare key) — the wire format and `/api/roles/set` API do NOT change.
- `#card-model` input and its backfill logic are deleted, not repurposed.
- Scope selector (global + per-project with inheritance) unchanged; saving remains one POST per role.

### A2. Single Ops view
Replaces nav items Models + Engines and absorbs Serving + Downloads tabs:
1. **Residents (live)** first: model, engine (`llama_cpp`/`ds4`/`strixite`/`hipfire`/`llama-swap(shim)`),
   ctx, resident source, tg t/s, PP progress, active consumers, idle-unload countdown (Part B3).
   Row actions: unload, pin/flush, adopt/takeover (existing L5 endpoints).
2. **Catalog**: swap_catalog entries with provenance, groups, engine; load/evict actions
   (existing `/api/engines/ensure`/`evict`).
3. **Providers & peers**: provider records incl. `lan_peer` with health column (Part C4); this is
   also where review-chain members show their live state.
4. **Downloads** and **Serving launch** forms move below the fold inside `<details>` — unchanged
   APIs, reduced chrome.
5. **Raw config editors stay in Config view** (operator escape hatch), never on the Ops page.

### A3. Command strip after
Scope · three card pickers · Save · privacy/quality chips · Flush · Restart local stack · health.
Anything model-management that is not a card assignment moves out.

### A4. Verification
Extend the puppeteer pattern (`scripts/test-nav-ui.cjs`, `test-gufo-ui.cjs`): per-card pinning
round-trips through `/api/roles` with distinct cloud ids per role; no `#card-model` references
remain (`escalation/mod.rs` HTML-assert tests updated in the same commit).

---

## Part B — Native strixite swapping (llama-swap parity for strix_server)

Today the native loop (queue, readiness, bounded restart, locks, abort, budget/LRU) already
supports `engine: "strixite"` (`src/engine_strixite.rs`, `src/swap_scheduler.rs`), but the live
catalog has zero strixite entries, `swap.mode` is `shim`, and the L7 soak was never recorded.
Strixite switching on this box is operator procedure, not code. This part makes it the same
product behavior llama-swap provides for llama.cpp.

### B1. YAML config block
```yaml
strixite:
  min_mem_available_gib: 32     # secondary floor (mirrors strix-wait-clear.sh)
  gtt_need_gib: 88              # primary gate: GTT free required before spawn
  drain_timeout_s: 900          # wait for a killed predecessor's GTT to return
  idle_unload_s: 900            # 0 disables idle unload
  models:                       # seeds swap_catalog entries (id, weights dir, vision/ngram files, ctx)
    - id: qwen3.8-flash-next
      weights_dir: ~/models/Qwen3.8-Flash-Next
      ctx: 262144
```
Env vars (`BRAINROUTER_STRIXITE_MIN_MEM_GIB`, port/conf/profile vars) remain as overrides,
documented as deprecated for new deployments; migration pass adds the block with defaults.

### B2. Gate hardening (the actual OOM trap)
The real hazard (measured 2026-10/11): a killed `strix_server` holds ~96–110 GiB of GTT for
seconds; a too-early spawn hits `hipErrorOutOfMemory` and can hang holding memory.
- `wait_gate` (swap_scheduler.rs:2368) gains a **GTT-free requirement** for strixite spawns:
  probe via existing `platform.rs` GTT plumbing (`gtt_gib`, the GTT total is the real budget —
  VRAM carve-out is a decoy), require free ≥ `gtt_need_gib` AND MemAvailable ≥
  `min_mem_available_gib`.
- Post-kill **drain wait**: after SIGTERM/SIGKILL of a strixite child, poll GTT-free until the
  requirement is met before allowing the next spawn; bounded by `drain_timeout_s`, on expiry
  the ensure fails with a named error (`SwapError::DrainTimeout`) and the queue is preserved.
- A wedged failed spawn is SIGKILLed and its GTT waited out exactly like a planned kill.

### B3. Idle unload
- Scheduler gains an idle timer per entry (`idle_unload_s`, strixite default 900 s from B1):
  when no active consumer for the window, evict. This is what frees the ~100 GiB for llama_cpp
  workloads — the behavior llama-swap users expect, missing natively today.
- LRU `evict_to_fit` unchanged (demand path); idle unload is the time path. Consumers count via
  `active_consumers.rs` only; grace period unchanged.

### B4. Queue persistence + boot re-attach
- Pending ensure queue survives daemon restart (state file under the swap state dir; on boot,
  queued ensures re-enter `ensure` normally).
- On boot in native mode, a running external `strix_server` on a catalogued port is discovered
  via `engine_observer`/`engine_systemd` and **adopted** as a resident, never preempted.

### B5. Safety invariants
1. **Never preempt a live external engine** — attach/adopt only (L5 rules hold; native spawn
   happens only where nothing is attached).
2. **Self-hosted assistant**: brainrouter's own harness may run on the catalogued strixite
   model; a cutover to a different model is request-driven and goes through cutover re-check +
   Active-Consumer grace. This is the only sanctioned path that migrates the assistant.
3. `--force-gate` remains operator-only; it bypasses gates but never the attach rule (1).

### B6. Cutover sequence
1. Land B1–B4 with unit coverage (gate math, drain wait with fake probe, idle unload, queue
   reload) while shim stays default.
2. Seed strixite catalog entries from B1 on this box; run the **L7 soak checklist** and record
   `## Soak` results in `docs/design/v2-streamlining.L7.soak.md` (currently empty — boot-refusal
   gate, mode-pinned tests, strixite load/unload cycle under a live OMP session).
3. Flip `swap.mode: native` in the live config. Shim remains the factory default until the
   soak record exists in-tree.

### B7. Known accepted limits (documented, not fixed here)
- `GroupsPending` refusal for group members stands.
- strixite build exposes no `/slots` → PP progress stays indeterminate for strixite rows; ETA
  EWMA continues to drive the progress bar.

---

## Part C — Peer review chain

Today: review card `cloud/fedora1/<model>` already routes reviews to the fedora1 hipfire peer
(`review/admission.rs` derives `Peer`); peers are excluded from `cloud_available`/`pool_top` and
failover is one-shot `review.fallback_selector` (cloud cards only). We keep explicitness, add
health-awareness and a chain.

### C1. Chain config
```yaml
review:
  # The review card remains the primary selector source — no `primary:` key
  # is introduced; this block is the fallback POLICY only.
  fallback_chain:               # ordered; tried only when the primary fails/unhealthy
    - cloud/fedora1/swift-1.5-qwen3.8-27b.mq4
    - cloud/deepinfra/deepseek-ai/DeepSeek-V4-Flash
```
- `review.fallback_selector` (v2.0-p4) migrates into a one-element `fallback_chain` with a
  named restore point (migration phase tag **`v2.0-p8`** — the chronological next tag after
  the shipped `v2.0-p7`; the program's "P5" naming is independent of migration tags);
  behavior with a single element is the old behavior.
- Chain applies to **any** primary, peers included (removes today's cloud-cards-only limit).
- Every hop attempt is recorded; the terminal `reviewModel` string already reports failover
  hops (shipped ea449df) — chain hops reuse it.

### C2. Health-aware dispatch
- Pre-dispatch probe for peer members: `/v1/models` with 1.5 s timeout + breaker state
  (`healthy|degraded|broken` already per provider); unhealthy member is skipped and logged.
- Chain walk is bounded: at most one attempt per member, then normal escalation path.
- No load/capacity ranking: chain order is the human's declared preference.

### C3. Peer capability cache
- `POST /api/providers/peer-info` probe results are persisted per provider
  (`host_id`, served models + ctx/tools/tier, last_seen) in the provider record. Ops view (A2.3)
  and Part A pickers read from the cache; liveness still probed at use time.

### C4. Default chain for this deployment
`fedora1 Swift MQ4 (local-first) → Synthetic (cloud auto) → DeepInfra DeepSeek-V4-Flash`.
Tradeoff, stated: the fedora1 card is a Qwen3.8-family model — self-family review bias is
possible; the chain gives a cheap second opinion on failover. Jerome confirms/edits the order
in config; no code assumes it.

---

## Test gates

- Part A: puppeteer UI tests per §A4 + `escalation/mod.rs` HTML assertions updated.
- Part B: gate/drain/idle/queue unit tests; mode-pinned L7 acceptance suites re-run; recorded
  soak document before any factory-default change.
- Part C: integration test with a fake peer OpenAI server (healthy → routed; down → chain walks;
  all dead → escalation, no hang — resolve pattern per existing parked-session pitfall).

## Rollout order

C (smallest, config+probes) → A (UI consolidation, no behavior loss) → B (native cutover,
soak-gated). Each part lands behind its own approved design gate and review cycle.

## Open items for confirmation (defaults chosen; any override is config, not code)

1. C4 chain order — fedora1 primary vs cloud primary.
2. B1 idle_unload_s = 900 for strixite; llama_cpp entries default to 0 (current behavior).
3. `brainrouter.yaml:79` `design_doc_path` points at nonexistent `docs/design/raon1-mini-engines.md`
   — dangling pointer to fix during P5 (or repoint when this doc is approved).
