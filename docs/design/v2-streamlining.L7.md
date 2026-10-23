This file is part of the design: the Map lives in `v2-streamlining.md`; this is leg L7's full spec. Version L7-v2 (2026-10-23): round-1 fold — flip sequence with quiesce + refusal (B1), config-section migration + honest grep-gate (B2), suite mode-pinning + provider-string end-state (B3), fedora import path key (B4), groups kickoff artifact (B5), consumer list completed (I1), soak exit criteria (I2), R8 scope line (I3), nits (N1–N3).

# Leg L7 in full — llama-swap absorbed

Goal (R8, llama-swap clauses): the native swap supervisor serves every model llama-swap served; llama-swap is deleted from the product. L4 imported the config, L4's native adapter runs in parallel, L6 made cloud native. L7 = the cutover sequence, the soak, the deletion. (R8's manifest-retirement clause is a separate owner sign-off at L8 closure, recorded in that leg's log; mac/distributed-role clauses are L8.)

## The flip (config + restart, with a gate)

- Pre-conditions enforced by `swap: {mode: native}` activation at boot: if the shim reports loaded models (llama-swap `/running` non-empty), boot in native mode REFUSES with the list — the operator flushes first (dashboard "Unload models" / `flush_models`, which in native mode drives the SWAP SUPERVISOR, not llama-swap). No silent double residency: the shim Residency source retires at the flip, and refusal makes the retire safe.
- At native boot: every ex-llama-swap key resolves through the swap registry (imported catalog; `ensure` owns lifecycle); `:8081` is not dialed by any component (consumer list below). Queue is in-memory — the flip is a restart; queued loads do not survive it (L4 contract), clients retry.
- Rollback (native→shim): daemon shutdown stops native children (L4 lifecycle: SIGTERM-grace, flock released at child exit); boot in shim mode; llama-swap reloads on demand. Acceptance pins BOTH directions: flip with shim-resident models = refusal + no :8081 traffic after flush+flip; flip-back with native children = none survive.
- `swap: {mode}` end state: key exists through the soak; at deletion the shim code goes, `mode: shim` becomes a config error (migration strips the key, logged), single-mode engine.

## Config-section migration (v2.0-p7) — the grep-gate's honesty condition

Every kept key today lives under `llama_swap:` (`fallback_model`, `subs_model`, `local_models`, `nudge.model_key` — `src/config.rs`; plus `base_url`, which dies). v2.0-p7 renames the section to `swap:` (keys move verbatim; values validated against the native catalog; snapshot + dry-run + restore per contract; `base_url` removed with a "no longer used" migration note). Grep-gate after deletion: zero `llama_swap`/`llama-swap`/`8081` in src/tests/scripts EXCEPT: migration provenance (`swap_catalog` import, v2.0-p7 migration code, migration-history notes), the provider-string literal (below), and the prompt-rewrite text (fixed by N3).

## Provider-string end-state

Native llama.cpp-family routes report `effective_provider = "llama-swap"` through the flip and the soak (v1 floors keep passing byte-identical: L3 suite asserts the literal on routed events). At deletion, the literal renames to `"local"` in one pass — and Ray's frozen suites (L1 fixtures, L2 shim-stand-in mock, L3 assertions) are retargeted in an AUTHORIZED protocol round owned by Ray (builder never edits; the retarget list is a named L7 acceptance item). Suite mode-pinning meanwhile: L1/L2/L3 fixtures run in shim mode (their committed configs pin `mode: shim`; default stays shim through the whole leg); native mode is covered by L7's own fixture suite (fake-engine adapter).

## Consumer list (every `:8081`/llama-swap touchpoint the flip must retarget)

Router (`try_llama_swap`, `route_local` seed), model listing (`/api/models` source + llama-swap keys), `poll_llama_swap_model` (startup model poll), `poll_llama_swap_running`/`poll_llama_swap_slot` (inference-status enrichment — PP-progress/activity panels) ⇒ retarget to swap-supervisor per-model slot data; `flush_models` endpoint ⇒ drives the supervisor in native mode; `POST /api/restart/llama-swap` + service allowlist + dashboard "Restart Local Stack" ⇒ per-engine restart; GGUF registration (`llama_swap_register.rs`, download flow) ⇒ `swap catalog add`; versions/upgrades row for llama-swap; toolbox runtime health rows (container source — the full four-source Residency list is shim/native/container/external, shim retires, the rest remain); nudge budget prompt text (prompt_rewriter "running on a local LLM via llama-swap" ⇒ engine-honest text, part of the deletion pass); install.sh llama-swap install/service; docs surfaces (PRD/README/UPGRADE).

## Import path (fedora reality)

`llama_swap.config_path` key (pre-migration name; lands under `swap:`) — import source, defaulting to `~/.config/llama-swap/config.yaml`; fedora hosts set it to their live config (`/opt/ai/llama-swap/config.yaml`, the docker-mounted root:aistack 664 file; per-host profile step in UPGRADE). Without it the fedora native catalog is empty and the flip strands those hosts — the fedora pilot step is: set path → import → list keys → parity, then flip.

## Groups (decidable at kickoff)

Kickoff artifact (Bailey card from L4's import metadata): group inventory. No groups (expected: owner's llama-swap config today) ⇒ slot-sharing is a documented non-goal, closed. Groups present ⇒ native multi-model slot-sharing is an owner-scope decision (reserved item) before any group key routes natively — groups never silently degrade to single-model entries.

## Deletion pass (nod-gated; exit criteria for the nod)

Soak: ≥7 days native on strix-halo; a full swap cycle of every catalog key including the owner's daily-driver sessions; zero double-residency/no-fit/`blocked` events beyond explained ones; memory ledger consistent with the budget view; owner reports it green. Then deletion: shim adapter, `poll_llama_swap_model` startup shim path, register-to-llama-swap, `:8081` config (base_url), install.sh step, versions/health/upgrades rows, provider literal → `"local"`, Ray's authorized retarget round, grep-gate clean, PRD/README serving-story rewrite (one page: engines + swap).

## Contracts kept

Selector vocabulary through L7 (this leg IS the L7 rebind endpoint: `local`/`subs`/bare-ids resolve native); pinned/auto strictness per L3; events additive; configs boot (v2.0-p7 section rename with dry-run + snapshot + restore); `swap.mode` flip reversible until deletion.

## Build order

config_path key + fedora-capable import (can land early, pre-deletion) → mode gate + refusal at boot → consumer retargets (listing, flush, inference-status, restart endpoints, dashboard) → suite mode-pinning + native fixture suite → v2.0-p7 section migration → parity checklist in native mode (all suites + owner dashboard-swap pilot, Bailey card) → soak (exit criteria) → deletion pass + Ray retarget round + grep-gate → docs.

## Acceptance (Ray)

Fixture tier: (1) flip gate: shim `/running` non-empty at native boot ⇒ refusal + model list; flush ⇒ boot native; zero `:8081` connections after flip (mock shim socket asserts silence). (2) flip-back: native children (fake engines) die at shutdown (flock released), shim mode serves after. (3) consumer matrix in native mode against mocks: model listing, flush, inference-status slots, restart-per-engine, GGUF download ⇒ catalog add (no llama-swap write). (4) v2.0-p7: section rename fidelity (every kept key + value), old `llama_swap:` section absent post-migration, dry-run inert, restore round-trip, `base_url` note. (5) native-mode provider literal = "llama-swap" during soak; deletion-pass suite = literal "local" + Ray's authorized retarget green. (6) grep-gate (honest exception list). (7) All prior suites green, shim-pinned, both feature modes; node + html-js. Host tier (nod-gated): (8) fedora import-from-config_path parity (keys listed, fidelity); (9) owner soak report (the exit-criteria list, green).
