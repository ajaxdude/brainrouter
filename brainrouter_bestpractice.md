# brainrouter best practice — coding on Strix Halo

Practical reference for routing models effectively on this box. Grounded in the live
`brainrouter.yaml` + llama-swap `config.yaml`, the measured throughput (2026-08-21), and
the routing logic in `src/router.rs::route_tagged`. Update it when the config changes.

---

## 1. The 30-second mental model

brainrouter sits in front of two backends:

- **llama-swap** (`:8081`) — local models (this box's 124 GB unified memory).
- **Manifest** (`:3001`) — cloud models (off by default; this machine opts in).

You name a model in the request; brainrouter decides where it goes. The full decision
table (from `route_tagged`):

| You request | What happens | Route tag |
|---|---|---|
| `auto` (or nothing / empty) | Bonsai classifies. **Bonsai is OFF here** → all local → **q8-nudge** (the nudge model) | `auto` → local |
| `subs` / `brainrouter/subs` | Direct to the **subs pool** (`subs_model` = q6-subs, 4 slots) | `local-subs` |
| `brainrouter/<model-key>` | Direct to that exact llama-swap model | `local-specific` |
| a key in `local_models` (`q8-nudge`, `q6-subs`) | Direct to it, bypasses Bonsai | `local-known` |
| `local` / `brainrouter/local` | `fallback_model` (Nail) + lean system-prompt rewrite | `local-direct` |
| `cloud` / `brainrouter/cloud` | Manifest cloud model | `cloud-direct` |

**Daily coding = just use `auto`.** You get q8-nudge (best 27B balance). No prefix needed.

---

## 2. Model ranking for coding

Measured 2026-08-21 (Strix Halo, 124 GB unified). `t/s` = decode speed.

| # | Model | Size / quant | t/s | ctx | Role | Use for |
|---|---|---|---|---|---|---|
| 1 | `dirk-qwen3.8-27b-q8-nudge` | 27B Q8, MTP | **12.1** | 196k | **default main** | 90% of coding: full quality, fast, capped thinking |
| 2 | `dirk-qwen3.8-27b-q8` | 27B Q8, MTP | ~11 | 196k | max 27B | hard debugging / architecture; wants long thinking (no nudge cap) |
| 3 | `deepseek-v4-flash-0731-iq3xxs` | 671B MoE IQ3 | slow (~80 GB) | 524k | **capability ceiling** | hardest problems, big multi-file refactors — evicts the dual-Dirk group |
| 4 | `deepseek-v4-flash-0731-strixhalo-verified` | 671B MoE q6kattn (ROCm) | slow | 262k | fastest 671B | big-model work; fastest of the DeepSeeks on this HW |
| 5 | `qwen3.5-122b-a10b` | 122B MoE Q4 | mid | 131k | mid-big | hard reasoning without the full 671B cost |
| 6 | `dirk-qwen3.8-27b-q6-subs` | 27B Q6, 4 slots | **20 agg** | 65k | **subs pool** | parallel subagent fan-out (via `brainrouter/subs`) |
| 7 | `nail-qwen3.6-35b-a3b-mtp-q6kxl` | 35B MoE Q6 | fast | 196k | **fast fallback** | quick edits, simple tasks, high throughput |

### Long tail — situational / experimental / unverified

| Model | Note |
|---|---|
| `dirk-qwen3.8-27b-q6` | Q6, 262k ctx (native ceiling). Near-Q8 quality, biggest context of the Dirks. |
| `dirk-qwen3.8-27b-q6-nudge` | Q6 + nudge budget. Q6-speed main if you want the nudge at a smaller quant. |
| `deepseek-v4-flash-0731-iq2xxs` | 671B 2-bit. Fastest DeepSeek, lower quality than iq3xxs. |
| `ds4-deepseek-v4-flash-0731-layers37` | 671B via antirez **ds4** engine (hybrid quant). Experimental — A/B vs the llama.cpp DeepSeek. |
| `ds4-deepseek-v4-flash-0731-iq2xxs` | 671B ds4 2-bit. Experimental. |
| `motif-3-iq2s` | 314B MoE, **unverified architecture** (Grouped PolyNorm / GDLA). May not load — test first. |
| `ornith-1.5-35b-a3b-q6kxl` | 35B MoE, **no model card yet** — sampling/spec flags unverified. Test first. |

### Memory reality (important)

- The box is **124 GB unified**. The dual-Dirk group (q8-nudge ~31.5 GB + q6-subs ~25 GB
  ≈ 56 GB) fits with headroom.
- **Any 671B DeepSeek loads ~80 GB and evicts the dual-Dirk group.** You cannot run a
  DeepSeek and the main + subs pool at the same time. Pick a DeepSeek → the 27B group
  unloads → the next `auto` request reloads q8-nudge.
- So a DeepSeek is a **deliberate switch**, not a background option.

---

## 3. The dual-Dirk workflow (main + parallel subs)

The point of the recent setup: run the interactive agent **and** parallel subagents without
one evicting the other.

- **Main agent** (your interactive session) runs `q8-nudge` (via `auto`).
- **Subagents** (parallel coding tasks) route to `brainrouter/subs` → the q6-subs 4-slot pool.
- The llama-swap **group `dirk-dual`** (`swap: false`) keeps q8-nudge + q6-subs co-resident,
  so subagent fan-out does **not** evict the main.

**Measured tradeoff (2026-08-21):**
- q6-subs 4-slot aggregate = **20 t/s** (2.94x a single slot) — real concurrency for parallel work.
- Under **full** 4-slot subs load the main drops 12.1 → 7.7 t/s (**64% of solo**). Not
  starved, but ~1/3 slower while all subs slots are busy. Fine for occasional fan-out; if it
  ever hurts, drop the pool to `--parallel 2`.

**When to use subs:** when you fan out 2–4+ parallel subagent coding tasks
(e.g. "run these independent fixes concurrently"). For a single interactive task, just use `auto`.

---

## 4. Nudge (thinking budget)

q8-nudge carries a per-request reasoning-token budget, chosen by tier:

- **`light`** (10240 tokens) — quick tasks; want it to converge fast.
- **`deep`** (12288 tokens) — hard reasoning; let it think longer.

```
brainrouter cli nudge tier light     # cap thinking tighter
brainrouter cli nudge tier deep      # let it think longer
brainrouter cli nudge status         # current tier
```

Higher budget = better on hard problems, slower. `light` is a good default for agent loops.

---

## 5. Code review (before you commit)

Reviews route to **cloud** (`review.forced_mode: cloud`) to avoid spinning up a local model
mid-inference (would thrash memory).

```
brainrouter cli review request <taskId> <summary> [--details ...] [--cwd ...]
brainrouter cli review list
brainrouter cli review approve <sessionId>
```

`taskId` format: `fix-YYYYMMDD-NNN` / `feature-YYYYMMDD-NNN`.

---

## 6. Day-to-day CLI control (headless, no browser)

```
brainrouter cli status          # health of all backends
brainrouter cli inference       # what's loaded, live slot progress
brainrouter cli models          # auto/local/cloud + llama-swap keys
brainrouter cli events          # recent per-request routing decisions
brainrouter cli routing-mode auto|local|cloud   # global override
brainrouter cli flush-models    # unload loaded models, free memory
brainrouter cli sync-omp        # push llama-swap models into OMP's models.yml
brainrouter cli context         # llama-swap context size
brainrouter cli restart llama-swap|brainrouter|manifest   # restart a service
```

---

## 7. Quick decision guide

- **Just coding, normal task** → `auto` (q8-nudge). Done.
- **Stuck / hard bug / architecture** → `brainrouter/dirk-qwen3.8-27b-q8` (full thinking).
  If still not enough → `brainrouter/deepseek-v4-flash-0731-iq3xxs` (max capability; accept
  the eviction + slowdown).
- **Independent parallel tasks** → route them to `brainrouter/subs` (q6-subs pool).
- **Quick small edit, want it fast** → `brainrouter/nail-qwen3.6-35b-a3b-mtp-q6kxl`.
- **Need a cloud model** → `brainrouter/cloud` (Manifest).
- **Free up memory before a big load** → `brainrouter cli flush-models`.
- **Route OMP subagents to the q6-subs pool** → `omp config set modelRoles.smol brainrouter/q6-subs`
  (scout/librarian/sonic auto-route to subs; main stays on `brainrouter/local`)


## 8. OMP subagent routing (Phase 5)

OMP v18.0.3 natively supports per-agent model overrides. Two config keys control this:


**Bundled agent roles:** scout → `@smol`, librarian → `@smol`, sonic → `@smol`, reviewer → `@slow`, task → `@task`, designer → `@designer`.

**To route subagents to the subs pool (automatic):**
```bash
omp config set modelRoles.smol brainrouter/q6-subs
```
This sends all three `@smol` agents to the q6-subs 4-slot pool. The main conversation stays on `modelRoles.default` → `brainrouter/local` → q8-nudge.

**To route only scout (fine-grained):**
```bash
omp config set task.agentModelOverrides '{"scout":"brainrouter/q6-subs"}'
```

**To revert:**
```bash
omp config set modelRoles.smol brainrouter/local
```
