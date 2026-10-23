This file is part of the design: the Map lives in `v2-streamlining.md`; this is leg L8's full spec (drafted at L6 closure; leg-scoped review pending).

# Leg L8 in full — Mac surface + distributed-role acceptance

Goal (R8/R5): the mac joins the fleet; the owner's distributed-role flows run end-to-end across real machines. Everything upstream (roles, decisions, credentials, swap, engines, providers, LAN federation) exists; L8 is platform + proof.

- **ds4 on macOS** — web evidence (2026-10-23): antirez/ds4 is "a local inference engine for Metal, CUDA and ROCm" with a documented OpenAI-compatible server mode (`model: "deepseek-v4-flash"`); Metal is a first-class backend (plus a maintained mitsuhiko Metal fork). The owner's expectation is well-founded. Spike (Bailey card, on hardware, before the adapter is committed): `ds4 serve` flag surface for the launch spec, model weights for the mac's unified memory (Flash sizes), tool-call support level. Fallback if the card fails: llama.cpp Metal (native adapter, GGUFs) as the mac engine.

## Mac platform surface

- CI: aarch64-apple-darwin build target (`cargo build --target`, release asset; no macOS runner tests beyond build + `cargo test --lib` — daemon integration suites stay linux-tier).
- Platform probe macOS: `sysctl hw.model/machdep.cpu`, Metal present ⇒ `PlatformInfo{id: "macos-<chip>", gpus:[apple-soc], budget: unified-memory policy}`; engine matrix: ds4 (per spike) and llama.cpp Metal offered; strixite/gufo/hipfire NOT offered (reasons).
- macOS-specifics: UDS peer-cred (`LOCAL_PEERCRED`), no systemd (adoption fallback = plain spawn + marker scan, best-effort documented), launchd user agent (`brainrouter.macos.plist` in install templates), GTT/MemAvailable equivalents (`sysctl hw.memsize`, `vm_stat`) for the budget layer.
- L6 federation: the mac runs its own per-user brainrouter (profiles: engine = ds4-or-Metal-llama.cpp, `lan.enabled` with token, expose its models) → peers see `cloud/mac/<model>`.

## Distributed-role acceptance (fleet tier, host-bound, nod-gated — the voyage's final demo)

- (a) strix-halo: plan→fedora1/hipfire-Swift, dev→strixite-local, review→deepinfra — one harness session; Sankey shows plan hop + cloud review hop; in-flight rows on fedora1 visible in its dashboard.
- (b) mac: plan→mac/ds4(-Metal), dev→cloud, review→strixite — proves the mac as a role endpoint.
- (3) Failures: peer down mid-session ⇒ L3 strictness + escalation semantics over LAN; breaker open ⇒ cloud→local fallback; all role flows survive daemon restarts on every host (adoption).
- Bailey cards per host: swap timings, memory ledger (GTT on strix, VRAM on fedora1, unified on mac), role-flow transcripts.

## Contracts kept

Everything upstream (selector vocabulary final: no further grammar changes; events; LAN default-off; credentials).

## Build order

spike card → CI target + probe + matrix + macOS plumbing (peer-cred, launchd, memory budget) → mac engine adapter per spike verdict → fleet pilot (a) on the Linux hosts (nod-gated) → mac onboard + pilot (b) → docs (fleet guide: per-host profiles, icon-retirement complete).

## Acceptance (Ray)

Fixture tier: (1) probe fixture macos sysctl ⇒ matrix offers correct, strixite/gufo/hipfire refused with reasons; (2) budget layer on unified-memory fixture; (3) LAN role-flow suite (L6) extended with a third mock peer (mac) — plan/dev/review across three peers, events + Sankey fields; CI: (4) darwin build target compiles (`--target aarch64-apple-darwin`, `cargo check --all-targets` at minimum lib+bins). Host tier (nod-gated, Bailey cards): (5) fleet flow (a) live; (6) mac onboarded, flow (b) live, swap/queue behavior on mac; (7) failure drill (peer kill mid-session ⇒ escalation/fallback per strictness). All prior suites green.
