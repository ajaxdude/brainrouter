# v2-streamlining charter

Version: v0 — 2026-10-21. Only the owner changes this file.

- **Final objective**: brainrouter v2 — the model-operations app (engines launch/swap, credentials UI, role cards, pluggable decision model, hardware-aware recommendations, LAN distributed roles), llama-swap and Manifest absorbed, v1 permanently revertible.
- **Milestones**: legs L1–L8 of `v2-streamlining.md` (Purge → Plugins → Roles/Decision/Credentials → Swap supervisor/Probe → strixite+hipfire → Providers/Federation → llama-swap removed → mac + distributed roles). Demo per leg (design's leg list).
- **Deadline**: none set. **Review budget**: 6 h, 3 critic rounds lifetime for the Map (careful pace).
- **Pace**: careful (touches auth, schema/config migrations, LAN security — "Always standard"; charter names no faster pace).
- **Delegated to Marlin**: file naming/placement, checkpoint granularity within legs, question defaults Q1–Q4 after 24 h, acceptance-test detail from R1–R9, spike scheduling.
- **Reserved for owner**: pace changes, merges to `master`, scope increases beyond L1–L8, llama-swap/Manifest production removal (L7/L6 cutovers), LAN exposure enabling, anything touching user data files, `v1` branch deletion.
- **Built = accepted when**: Ray's acceptance tests pass, Step 9 (Bruce + brainrouter MCP review) clean, demo shown.
- **Parallel streams**: ≤3. Dory reviewers per batch per pace table.
- **Machines**: strix-halo local (primary), fedora1 (gfx1100, hipfire), fedora2 (prod/DeepInfra), mac — heavy runs booked through each host's Crush; nothing destructive on fedora1/fedora2 without a reserved nod.
- **Backup remote**: `origin` = https://github.com/ajaxdude/brainrouter. v1 restore: tag `v1-classic`, branch `v1` (pushed 2026-10-21).
- **Dory updates**: this chat (digest format per skill).
- **Cost policy**: standard-tier reviewers at ≥ Hank effort for Steps 6/7; helper tier for checks.
