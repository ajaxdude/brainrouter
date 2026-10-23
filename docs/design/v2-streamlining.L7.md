This file is part of the design: the Map lives in `v2-streamlining.md`; this is leg L7's full spec (drafted at L6 closure; leg-scoped review pending).

# Leg L7 in full — llama-swap absorbed

Goal (R8): the native swap supervisor serves every model llama-swap served; llama-swap is deleted from the product. All machinery exists by then: L4 imported the config (macro-expanded, groups as metadata), L4's native `llama_cpp` adapter runs in parallel and passed the shim-parity floors, L6 made cloud native. L7 = cutover + deletion.

## Cutover (the one-way moment, owner sign-off-gated)

- **Parity checklist (gate, all green before the switch):** L4 acceptance (4) shim-parity suite green; L1–L6 acceptance suites green in native mode; import fidelity (every key serves its card/GGUF with resolved args, same stem, same ctx); owner's live model set swapped via dashboard on the Strix Halo (Bailey card: timings, memory, queue behavior); golden routing corpus (v1 selector matrix, L3) byte-stable with native endpoints; v1 selector tests green with `llama_swap_shim` disabled.
- **Switch = config, not code:** `swap: {mode: native}` (default remains `shim` through the whole leg). `mode: native` ⇒ the router's llama-swap path resolves through the swap registry (every ex-llama-swap key is a native catalog entry; `ensure` owns lifecycle); the shim adapter stops being consulted; `:8081` is no longer dialed. Rollback = flip back to `shim` (both stacks installed and configured throughout; the flip is tested both directions in acceptance).
- Group semantics (llama-swap groups with `swap: false` etc., if present in the imported config): recorded as metadata; native slot-sharing (multi-model single-process endpoints, e.g. context-shared slots) implemented ONLY if the owner's imported config contains live groups — inventory at L4 import time decides; if none, groups are a documented non-goal.

## Deletion (after the owner runs native for a soak period, nod-gated)

- llama-swap endpoints/dialect: `try_llama_swap`, `poll_llama_swap_model`, `/running` shim observer (Residency shim source retires — strixite/hipfire/native residents remain), `llama_swap_register.rs` (GGUF registration: replaced by `swap catalog add`, which writes the native catalog; llama-swap config.yaml is no longer written), GGUF-registration dashboard flows, versions/health/upgrades rows for llama-swap, `install.sh` llama-swap install step + service, PRD/README surfaces. `fallback_model`/`subs_model` keys: remain (chain seeds, L3 semantics), validated against the native catalog.
- The llama-swap BINARY is left installed on fedora hosts (uninstall is an operator action; brainrouter stops touching it). The fedora1 hipfire squat incident's final closure: nothing in brainrouter knows port 8081.

## Contracts kept

Selector vocabulary (bare-ids/`local`/`subs` now route 100% native — same strings, same tests, the L7 endpoint of the Selector-vocabulary contract); events/observability additive; `swap: {mode}` config with dry-run flip + rollback documented; catalog import is L4's, idempotent.

## Build order
native-mode router flag → parity checklist execution (all suites in native mode; owner dashboard-swap pilot card) → soak → deletion pass (endpoints, register, install.sh, docs) → grep-gate (zero `llama_swap`/`8081` in src except migration-history notes + `swap_catalog` import provenance) → PRD/README rewrite of the serving story.

## Acceptance (Ray)

Host tier: (5) owner soak report (Bailey card: swaps, queue, memory on the live box). **Pilot constraint (codified from practice): brainrouter's own harness session runs on the strixite provider — swaps are observable live, and a swap evicting the provider model mid-session is the Map's named risk (L5 self-provider pin covers it); soak begins on strix-halo with the provider model pinned; fedora hosts run native mode from the flip.**
(1) `swap mode` flip: shim→native→shim with the same catalog, v1 selector matrix green in BOTH modes (fixture tier: fake native engines + mock shim). (2) Deletion grep-gate clean; suites green post-deletion; `catalog add` writes native entries the router serves. (3) Fallback keys validate against native catalog. (4) All prior suites green in native mode. Host tier: (5) owner soak report (Bailey card: swaps, queue, memory on the live box).
