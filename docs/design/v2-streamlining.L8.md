This file is part of the design: the Map lives in `v2-streamlining.md`; this is leg L8's full spec. Version L8-v3 (2026-10-24): round-2 fold — selectors respelled to L6's `cloud/<provider>/<id>` grammar (F1), flow (c) drill aligned to the frozen strictness matrix (F2), darwin CI job producer in the build order (A1), mac-as-served hop added (A2), nits (N1, N3); budget-number determinism stated (I5 close). L8-v2 folded round 1 (B1 wiring tables, I1–I5, N1–N4).

# Leg L8 in full — mac surface + distributed-role acceptance

Goal (R8 remaining clauses): the third platform (macOS) joins the fleet; the distributed-role story is accepted end-to-end: plan/dev/review bound to model endpoints on ANY combination of hosts, one session.

## macOS engine surface (the spike first — hardware-bound, owner nod)

- Spike (Bailey card on the mac): ds4-Metal serving (`ds4 serve` flag surface, Flash sizes for unified memory, tool-call support on the Metal path); fallback llama.cpp Metal (the standard OpenAI-compat server; the ds4 evidence is "Metal, CUDA and ROCm" — sizes/flags/tools are the open questions). The spike card's measured model-fit numbers feed the budget rule (below) — fixtures take that recorded number as a parameter; hardware never enters a test at run time.
- Platform probe: macOS branch (`sysctl hw.memsize`, `vm_stat` for headroom, Metal presence via the `iogpu` device-tree sysctl, chip id from `hw.model`); budget cap on Apple silicon = the spike card's measured number (GPU-wired-limit fraction of unified memory with free-memory headroom).
- Engines: llama-server (native adapter, Metal flags) on mac; strixite/gufo/hipfire NOT offered (gfx1151-only); strixite llama-swap on macOS is the spike's secondary question — the icon fleet on mac gets the same brainrouter treatment only if the spike shows the Metal path serves the owner's mac models.
- No systemd (adoption fallback = plain **setsid** + marker scan, per L4's binding contract — setsid, not same-group spawn: a Ctrl-C in a manual run must not signal the engine's process group; launchd sends SIGTERM per-pid for the deployed path). LaunchAgent user service (installed via the app bundle pattern; runs at login, restarts on crash). The platform id family `macos-<chip>` is deliberate (feed-match rule not applicable; the unknown-platform banner is expected on mac).
- Residency: registry + containers (absent engine managers degrade to an empty container source); `lan: {listen}` on the mac daemon (peers see `cloud/mac/<model>`); LOCAL_PEERCRED via tokio's portable peer credentials (pid-side cwd lookup degrades silently without /proc — Linux-only enrichment, documented).
- Dashboard toolbox rows on mac: fall-open gives "unknown platform" + all rows; on macOS those rows are inert-browse (podman Linux images have no serving path) — one clause, no new machinery.

## Distributed-role acceptance (the R7 proof)

Prerequisites (host-tier): L5 pilots complete — the fedora1 hipfire user-service takeover (Q4) resolved per L5's default, live strixite swap on strix-halo; L6 + L7 built; L7 soak in flight. Flow (a) stalls mid-demo if the fedora1 pilot has not run — it is a scheduled prerequisite, not a demo-time surprise.

Role mapping (authoritative for the demo; supersedes L6's flow-name note — dev rides the **subagent** role, recorded as an L8 discovery): "plan" = the Main card, "dev" = the Subagent card, "review" = the Reviewer card. Pinned cards bypass the decision slot, so each verb resolves deterministically. Selectors use L6's grammar: qualified federation is `cloud/<provider-id>/<model-id>` (no `peer/` form exists; the first segment must name an enabled provider).

Flow (a) — harness on strix-halo (its daemon owns the session); three dashboards inspected:
| verb | card | pinned selector | forwarded role header | serving host |
|---|---|---|---|---|
| plan | Main | `cloud/fedora1/hipfire-swift` | main | fedora1 (hipfire, supervisor-managed per L5 pilot) |
| dev | Subagent | `strixite-local` | subagent | strix-halo (strixite, L5) |
| review | Reviewer | `cloud/deepinfra/deepseek-v4-flash` | review | deepinfra (L6) |
| served-proof | Reviewer (re-pin for this hop) | `cloud/mac/ds4-flash` | review | the mac's LAN listener (below) |
Wiring: strix-halo's daemon has enabled providers `fedora1` (`lan_peer`, bearer from fedora1's `lan.token_credential`), `deepinfra` (`deepinfra_token`), and `mac` (`lan_peer`, bearer = the mac's `lan.token_credential`); fedora1's expose list contains `hipfire-swift`, the mac's contains `ds4-flash` (double enforcement: provider expose list AND the request's resolved model); fedora1's and the mac's dashboards show in-flight rows (LAN listener → normal proxy path). The served-proof row is the mac-as-SERVED-endpoint proof — a remote session's hop landing on the mac's engine through its LAN listener (the macOS serving path: LAN bind, expose, launchd persistence), and it doubles as L6's mac flow executed for real.

Flow (b) — harness on the mac (mac daemon owns the session; the new platform's cards are the ones under test); two hosts (mac + strix-halo), not three: Main → `cloud/strix/claude` (strix-halo peer provider, bearer from strix-halo's `lan.token_credential`; the mac holds the credential), Subagent → `llama-metal-local` (bare registry key on the mac), Reviewer → `cloud/deepinfra/deepseek-v4-flash`. Same proof shape, mirrored — the mac runs the full daemon, not an agent.

Flow (c) — failure drill, aligned to the frozen strictness matrix: cloud-shape cards (including qualified `cloud/<peer>/<id>`) keep the managed two-step — target, then the single `fallback_model` hop, one FailedAttempt event, stop on second failure (L3 matrix; L6 byte-identical cloud parity; escalation arm is the review rule only). So: mid-session, kill the fedora1 daemon; the plan-role request pinned `cloud/fedora1/hipfire-swift` records the FailedAttempt and takes its card's fallback hop (explicit, in events — never silent); restart fedora1, the hop recovers; the in-flight tracker shows the stall live. The drill demonstrates event truth + recovery, not strict-no-fallback (that belongs to bare-key and review cards).

Fixture tier (mock peers, deterministic): (3) extends L6's federation fixtures — three mock peers, role headers forwarded, cards resolve AT the peer, pinned-resolved model bypasses peer cards, LAN hop events render the Sankey trace; restarts: all role flows survive daemon restarts on every host (adoption).

## The rest

Manifest retirement (R8): owner sign-off at L8 KICKOFF (recorded in this leg's log; the sign-off gates the manifest kind deletion in the docs/removals pass — kickoff placement lets the deletion ride L8's own docs pass; supersedes L7's "at L8 closure" parenthetical). After manifest deletion, the L7 grep-gate suite re-runs (manifest strings are removed, so the gate must stay clean).

Build order: spike (owner nod; feeds budget rule + mac model list) → probe mac branch + ids → native adapter on mac + setsid/launchd → **darwin CI job** (macos aarch64 runner; scope: build + `cargo test --lib` + `cargo check --all-targets`; integration suites stay Linux-tier — feasibility verified: no compile-time Linux-only APIs; flock/UDS/peer_cred are apple-darwin-available) → LAN peer flows (b) wiring verified → distributed-role acceptance (fixture tier first, host-tier second, flows a/b/c per the tables) → fleet guide doc (the wiring tables generalized per host, naming the fleet-profile file for engine/pin keys and daemon config for `lan.*`) → manifest sign-off recorded + deletion in docs pass + grep-gate re-run.

Acceptance (Ray): (1) probe mac fixtures (ids, Metal presence, budget from spike value); (2) budget layer on unified-memory fixture (cap = spike number as a fixture parameter; over-budget → no-fit/evict path); (3) mac adapter lifecycle (setsid detachment — a daemon SIGTERM does not reach the engine group; adoption via marker scan after restart; launchd plist shape); (4) flows (a)/(b)/(c) at fixture tier (wiring-table selectors resolve in L6 grammar, role headers forward, peer cards apply, pinned bypass at peer, drill = FailedAttempt + managed fallback hop, restart survival) + host tier: the tables executed literally, three dashboards (a) / two (b), cross-host Sankey, failure drill; all prior suites green, both feature modes; node + html-js; `--all-targets` compile on the darwin CI job (producer in build order above).

Docs: UPGRADE (mac install + fleet guide), PRD/README (distributed roles), CHANGELOG.
