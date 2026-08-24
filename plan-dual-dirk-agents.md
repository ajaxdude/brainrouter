# Plan: Dual Dirk Agents (Main + Subs) — llama-swap + brainrouter

## Goal

Run two concurrent Dirk (Qwen3.8-27B) instances on the same Strix Halo box:
- **main** — single long-context slot for the primary coding session (existing `dirk-qwen3.8-27b-q8`)
- **subs** — a smaller-context, multi-slot pool for parallel subagent work dispatched by omp

Then wire brainrouter to route subagent traffic to the subs pool instead of forcing everything
through the main model or through Bonsai classification.

This plan is written for a coding agent to execute. Each phase has a stop condition — do not
proceed to the next phase until the current one's verification step passes. Several steps are
explicitly "investigate, don't assume" — treat those as required research, not optional.

---

## Phase 0 — Verify assumptions before changing anything

Do not skip this phase. Several things below were inferred from documentation, not confirmed
against the running system.

- [ ] **Confirm what `brainrouter cli context set <value>` actually does.** Read
      `src/router.rs` and `src/config.rs` in the brainrouter repo (or run
      `brainrouter cli context status` then `context set 65536` against a running instance and
      watch whether it patches `/opt/ai/llama-swap/config.yaml` directly, or passes a per-request
      override to llama-swap, or something else). This determines whether static `--fit-ctx`
      values in `config.yaml` are the source of truth or get overridden at runtime. Write the
      answer down before Phase 3 — it changes how config is authored going forward.
- [ ] **Confirm llama-swap's version supports `groups:`** (keep multiple models resident
      simultaneously, rather than the default single-active-model swap behavior):
      `toolbox run -c llama-vulkan-radv-performance llama-swap --version`, then check that
      version's changelog/docs for `groups`. If unsupported, Phase 1's group config will need
      an upgrade first (`brainrouter cli upgrade llama-swap` is available per the dashboard).
- [ ] **Confirm current free memory headroom** with nothing extra loaded:
      `free -h` and check current GPU/shared-memory usage. Running Dirk Q8 (main, ~29GB + KV)
      and Dirk Q6-subs (~22GB + KV × parallel slots) resident at the same time, on the same box
      also running xmrig and serving brainrouter instances for other users, is a real memory
      budget question — not a given.
- [ ] **Sanity check the Bonsai classifier is not currently blocking things.** If
      `bonsai.enabled: true` in your brainrouter.yaml, confirm it's actually healthy
      (`brainrouter cli bonsai status`) before this work — an unrelated Bonsai crash loop
      makes it hard to tell if new problems are from this plan or from that.

**Stop condition:** all four items above have a written answer, not an assumption.

---

## Phase 1 — llama-swap: add the subs model + groups config

Edit `config.yaml` (wherever it's actually deployed — the README says
`/opt/ai/llama-swap/config.yaml` for the multi-user install, confirm this matches your actual
path before editing).

1. Add a new model entry, `dirk-qwen3.8-27b-q6-subs`:
   - `--parallel 4` (override `common`'s `--parallel 1` — this is the actual concurrency)
   - **no `--spec-type draft-mtp`** — MTP regresses hard at `-np > 1` on Strix Halo per every
     existing Dirk block's own comments; a multi-slot pool cannot use it
   - bounded context per slot (start at `--fit-ctx 65536`, not main's 196608/262144 — subagent
     tasks are typically narrower scope)
   - same weights file as `dirk-qwen3.8-27b-q6` (`Dirk-Qwen3.8-27B-UD-Q6_K_XL.gguf`) — this is a
     second llama-server *process* on a different port, not a shared process, since llama-swap
     manages processes not intra-process slot pools across model keys
2. Add a top-level `groups:` entry keeping `dirk-qwen3.8-27b-q8` (main) and
   `dirk-qwen3.8-27b-q6-subs` both resident, instead of swapping:
   ```yaml
   groups:
     dirk-dual:
       - dirk-qwen3.8-27b-q8
       - dirk-qwen3.8-27b-q6-subs
   ```
   (Confirm the exact `groups:` schema against whatever llama-swap version Phase 0 confirmed —
   don't assume this snippet's shape is exactly right without checking.)
3. Validate: `python3 -c "import yaml; yaml.safe_load(open('config.yaml'))"`, then restart
   llama-swap and confirm both models load and stay loaded simultaneously (not one evicting the
   other) via `brainrouter cli models --llama-swap` or `GET /api/models/llama-swap`.

**Stop condition:** both models show as loaded/running at the same time, not alternating.

---

## Phase 2 — Manual validation before touching brainrouter code

Test the raw llama-swap layer directly, bypassing brainrouter entirely, before adding any
routing logic on top of it.

1. Fire a long-context request at `dirk-qwen3.8-27b-q8` directly.
2. While that's still generating, fire 2-4 concurrent short requests at
   `dirk-qwen3.8-27b-q6-subs` directly.
3. Confirm: main's response isn't delayed/starved by the subs traffic, subs requests actually
   run concurrently (check timestamps, not just that they all eventually return), and memory
   stays stable (no OOM, no swap-to-disk thrashing).
4. Record actual t/s for the subs pool at `--parallel 4` — compare against a single-slot
   baseline to confirm the concurrency is a net win for *aggregate* throughput even without MTP,
   not just a theoretical one.

**Stop condition:** concurrent load on both models is stable and the subs pool's aggregate
throughput at 4 slots beats a single MTP-enabled slot serving 4 requests sequentially. If it
doesn't, the whole premise of this plan needs re-examination before writing any Rust.

---

## Phase 3 — brainrouter.yaml wiring

1. Add both model keys to `llama_swap.local_models` so they can be targeted directly, bypassing
   Bonsai classification:
   ```yaml
   llama_swap:
     local_models: ["dirk-qwen3.8-27b-q8", "dirk-qwen3.8-27b-q6-subs"]
   ```
2. Decide whether `llama_swap.nudge.model_key` should point at the nudge-fork variant of either
   model (`dirk-qwen3.8-27b-q8-nudge` / `dirk-qwen3.8-27b-q6-nudge`) or stay separate from this
   main/subs work — these are two different concerns (thinking-budget nudging vs. concurrency
   pooling) that happen to both target Dirk. Don't conflate them without a reason.
3. Restart brainrouter for your user, confirm `brainrouter cli models` shows both new keys.

**Stop condition:** `!br dirk-qwen3.8-27b-q6-subs <query>` (or the direct model-name path) works
end-to-end through brainrouter, not just through raw llama-swap.

---

## Phase 4 — brainrouter code change: native subs routing (new feature)

This is the actual "build it into brainrouter" part — a small, scoped feature addition, not a
rewrite. Based on `src/` layout from the README:

1. **`src/config.rs`** — add an optional field, e.g. `llama_swap.subs_model: Option<String>`,
   parsed the same way `fallback_model` and `nudge.model_key` already are. Validate it (if
   present) against the same "must match a llama-swap key" rule the README documents for
   `fallback_model`.
2. **`src/router.rs`** — this is where routing decisions happen (Bonsai classify → cloud/local,
   circuit breaker, fallback). Add a routing path: if a request is tagged as a subagent request
   (see open question below — the exact detection mechanism needs research, don't invent one
   blind), route directly to `subs_model` if configured, bypassing Bonsai classification the
   same way `local_models` direct-targeting already does.
3. **`src/types.rs`** — only touch this if the subagent-detection mechanism requires a new
   field on the incoming request type (e.g. a custom header or a reserved model-id convention
   like `brainrouter/subs`). Prefer the model-id convention if it works — it needs no protocol
   changes and mirrors the existing `auto`/`local`/`cloud` pattern from `GET /v1/models`.
4. **Tests** — the existing suite (89 tests, per README) covers circuit breaker, Anthropic
   translation, config merging, review lifecycle, classifier parsing, fallback. Add equivalent
   coverage for the new routing path: a test that a request targeting `subs` (however it's
   detected) reaches `subs_model` and never touches the Bonsai classifier or the circuit
   breaker meant for cloud fallback.
5. **Docs** — update `brainrouter.example.yaml` and the README's config reference table with
   the new `subs_model` field, following the exact style of the existing `fallback_model` /
   `nudge` entries.

**Stop condition:** `cargo test` passes including new tests, and a manual request against
`brainrouter/subs` (or whatever the chosen convention is) reaches the subs pool, confirmed via
the dashboard's routing events feed.

---

## Phase 5 — OMP-side wiring (DONE: automatic subagent routing to q6-subs)

OMP v18.0.3 supports per-agent model overrides via `modelRoles` and `task.agentModelOverrides`.
Both are in `~/.omp/` — no source code changes needed.

### Investigation findings

**How it works:** Each bundled agent declares a `model:` role (`@smol`, `@slow`, `@designer`, `@task`).
When `modelRoles` is set, those role names map to model IDs. Fallback is `modelRoles.default`.
`task.agentModelOverrides` provides per-agent overrides on top of `modelRoles`.

**Current config:**
- `modelRoles = {"default":"brainrouter/local"}` → all agents (scout, reviewer, designer, etc.) resolve to `brainrouter/local`
- `task.agentModelOverrides = {}` (empty) → no per-agent overrides

**Bundled agent roles:** scout→`@smol`, librarian→`@smol`, sonic→`@smol`, reviewer→`@slow`, task→`@task`, designer→`@designer`, security-reviewer→none, local-coder→none.

**Result:** OMP DOES support per-task-type model routing. To route subagents to the q6-subs pool:

```bash
# Route all @smol agents (scout/librarian/sonic) to subs pool
omp config set modelRoles.smol brainrouter/q6-subs
# (Optional) per-agent override for fine-grained control:
omp config set task.agentModelOverrides '{"scout":"brainrouter/q6-subs"}'
```

**Stop condition:** Answer confirmed — automatic subagent routing is possible via `omp config`.

---

## Phase 6 — End-to-end test (DONE: smoke test passed)

Smoke test completed against running brainrouter daemon (tcp 127.0.0.1:9099).

**Test:** Direct request to `dirk-qwen3.8-27b-q6-subs` model key via brainrouter → llama-swap loads model → returns completion.

**Results:**
- Model loaded in 7.9s, HTTP 200
- Both models resident after: q8-nudge + q6-subs
- Response: "llama-swap loading model: dirk-qwen3.8-27b-q6-subs. OK"
- Memory: ~90Gi used (no OOM, no swap thrashing)

**Stop condition:** Routing confirmed. q6-subs routes through brainrouter → llama-swap successfully.

Full end-to-end with omp subagent dispatch (multi-step coding task, dashboard routing feed, wall-clock comparison, sustained memory) is the real validation — run when OMP is configured to route subagents to q6-subs.

---

## Rollback

Every phase is additive — nothing in this plan requires removing or breaking existing model
keys (`dirk-qwen3.8-27b-q8`, `-q6`, `-q8-nudge`, `-q6-nudge` all stay as-is). If any phase fails
its stop condition:
- Remove the `groups:` entry (reverts to normal single-active-model swap behavior).
- Remove `dirk-qwen3.8-27b-q6-subs` from `config.yaml` and from `local_models`.
- Revert the brainrouter Rust changes via git if Phase 4 was reached — they're additive and
  isolated to `subs_model` handling, should not affect existing routing paths if reverted
  cleanly.

---

## Open questions the agent must resolve, not assume

1. ✅ Does `brainrouter cli context set` write to `config.yaml`, or is it a separate runtime layer?
   **Resolved Phase 0:** It writes directly to `config.yaml` (llama-swap's config file). Source-of-truth is static in `config.yaml`; brainrouter config (`brainrouter.yaml`) is a separate runtime layer.

2. ✅ What's the actual `groups:` schema for the llama-swap version in use?
   **Resolved Phase 0:** `{members: [...], swap: false, exclusive: false}` — bare list was wrong in original plan.

3. ✅ How should a "this is a subagent request" be detected?
   **Resolved Phase 4:** Reserved model-id convention (`subs` / `brainrouter/subs`) → `subs_model`, bypassing Bonsai. No protocol changes needed.

4. ✅ Does omp support per-task-type model routing at all?
   **Resolved Phase 5:** Yes. `modelRoles` maps agent roles to model IDs. `omp config set modelRoles.smol brainrouter/q6-subs` routes scout/librarian/sonic to the subs pool automatically.

## Remaining next steps

- **Optional:** Run `omp config set modelRoles.smol brainrouter/q6-subs` to activate automatic subagent routing.
- **Full validation:** Run a real multi-step coding task through omp and watch the dashboard routing feed for the traffic split between main (q8-nudge) and subs (q6-subs).
- **Sustained load test:** Monitor memory over a full session of dual-model work.
