This file is a record, not a rule.

# v2-streamlining history

## Origins (2026-10-19 → 2026-10-21, Hank phase 1)
- Owner kickoff: streamline brainrouter; keep Sankey + cards; kill harness-side model selection; goals = absorb llama-swap/Manifest, strixite llama-swap-equivalent, plugins for bridges, pluggable decision model (≤2GB, Clef-like contract, LFM2.5 class), credentials UI (Manifest-style), icon fleets (strixite + fedora1 hipfire/OMP) replaced, hardware-aware engine offers, distributed roles (plan fedora1/hipfire · dev fedora2/strixite · review cloud deepinfra; mac/ds4 variant), mac added as test surface, v2 revertible to v1 on github.
- Working notes: `local://brainrouter-streamline-plan.md` (pre-design planning doc, same content lineage; kept outside repo, this file and the design supersede it).
- Evidence gathered with citations (baseline `790fd9b`): LoC inventory; 8-layer selection surface; Feature B live routing (`router.rs` `find_server_mode_identity`); platform hardcodes (`server.rs` toolbox default, dashboard `CURRENT_PLATFORM_ID`); credential friction (Raon-strixw session 01a15f09, aborted setup 01a12e25, `fresh` reviewer-credential block); hipfire facts + `omp-launch.sh` invariants; strixite cutover practice; Clef = Cloudflare/clef 27B decision contract (bartowski GGUF); LFM2.5 sizes.
- v1 snapshot: tag `v1-classic` + branch `v1` pushed to origin, 2026-10-21. Branch `v2-streamlining` created.

## Gate history
- (pending) Batch 1: Step 5/5b + Step 6 on v1.

## Hank handoffs
- (none yet)
