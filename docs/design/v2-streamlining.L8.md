This file is part of the design: the Map lives in `v2-streamlining.md`; this is leg L8's full spec. Version L8-v2 (2026-10-24): round-1 fold — per-flow wiring tables + plan/dev role mapping (B1), manifest sign-off at kickoff (I1), prerequisites (I3), setsid parity (I4), budget rule from spike (I5), nit clauses (N1–N4); Map spike parenthetical corrected in the same revision.

# Leg L8 in full — mac surface + distributed-role acceptance

Goal (R8 remaining clauses): the third platform (macOS) joins the fleet; the distributed-role story is accepted end-to-end: plan/dev/review bound to model endpoints on ANY combination of hosts, one session.

## macOS engine surface (the spike first — hardware-bound, owner nod)

- Spike (Bailey card on the mac): ds4-Metal serving (`ds4 serve` flag surface, Flash sizes for unified memory, tool-call support on the Metal path); fallback llama.cpp Metal (the standard OpenAI-compat server; the ds4 evidence is "Metal, CUDA and ROCm" — sizes/flags/tools are the open questions). The spike card's measured model-fit numbers feed the budget rule (below).
- Platform probe: macOS branch (`sysctl hw.memsize`, `vm_util`/`vm_stat` for headroom, Metal presence via the `iogpu` device-tree sysctl, chip id from `hw.model`); budget cap on Apple silicon = the spike card's measured number (GPU-wired-limit fraction of unified memory with free-memory headroom; the fixture builds on the spike value, not an invented fraction).
- Engines: llama-server (native adapter, Metal flags) on mac; strixite/gufo/hipfire NOT offered (gfx1151-only); strixite llama-swap on macOS is the spike's secondary question — the icon fleet on mac gets the same brainrouter treatment only if the spike shows the Metal path serves the owner's mac models.
- No systemd (adoption fallback = plain **setsid** + marker scan, per L4's binding contract — setsid, not same-group spawn: a Ctrl-C in a manual run must not signal the engine's process group; launchd sends SIGTERM per-pid for the deployed path). LaunchAgent user service (installed via the app bundle pattern; runs at login, restarts on crash). The platform id family `macos-<chip>` is deliberate (feed-match rule not applicable; the unknown-platform banner is expected on mac).
- Residency: registry + containers (podman/docker destopped where absent — container source degrades to empty); `lan: {listen}` on the mac daemon (peers see `cloud/mac/<model>`); LOCAL_PEERCRED via tokio's portable peer credentials (pid-side cwd lookup degrades silently without /proc — Linux-only enrichment, documented).
- Dashboard toolbox rows on mac: fall-open gives "unknown platform" + all rows; on macOS those rows are inert-browse (podman Linux images have no serving path) — one clause, no new machinery.

## Distributed-role acceptance (the R7 proof)

Prerequisites (host-tier): L5 pilots complete — the fedora1 hipfire user-service takeover (Q4) resolved per L5's default, live strixite swap on strix-halo; L6 + L7 built; L7 soak in flight. Flow (a) stalls mid-demo if the fedora1 pilot has not run — it is a scheduled prerequisite, not a demo-time surprise.

Role mapping (authoritative for the demo; corrects L6's flow-name note — dev rides the **subagent** role, recorded as an L8 discovery): "plan" = the Main card, "dev" = the Subagent card, "review" = the Reviewer card. Pinned cards bypass the decision slot, so each verb resolves deterministically.

Flow (a) — harness on strix-halo (its daemon owns the session):
| verb | card | pinned selector | forwarded role header | serving host |
|---|---|---|---|---|
| plan | Main | `peer-fedora1/hipfire-swift` | main | fedora1 (hipfire, supervisor-managed per L5 pilot) |
| dev | Subagent | `strixite-local` | subagent | strix-halo (strixite, L5) |
| review | Reviewer | `cloud/deepinfra/deepseek-v4-flash` | review | deepinfra (L6) |
Wiring: strix-halo holds provider credentials for fedora1 (`lan_peer`, bearer from its `lan.token_credential`) and deepinfra (`deepinfra_token`); fedora1's expose list contains `hipfire-swift` (double enforcement: provider expose list AND the request's resolved model); fedora1's dashboard shows the in-flight row (LAN listener → normal proxy path). Expected: one session, three hops, three dashboards, Sankey shows the cross-host trace.

Flow (b) — harness on the mac (mac daemon owns the session; the new platform's cards are the ones under test): Main → `peer-strix/claude` (LAN peer to strix-halo, cloud role on the mac's daemon = peer provider), Subagent → `llama-metal-local` (native mac endpoint), Reviewer → `cloud/mac/ds4-flash` (the spike's engine, native provider on the mac). Tokens: the mac holds the strix bearer; strix-halo exposes the requested model. Same three-dashboard proof, mirrored — the mac runs the full daemon, not an agent.

Flow (c) — failure drill: mid-session, kill the fedora1 daemon (or cut its LAN listener); a plan-role request fails per the pinned-card strictness (no silent local fallback — the pinned peer is an explicit selector); events record the failure; restart fedora1, the hop recovers; the in-flight tracker shows the stall live.

Fixture tier (mock peers, deterministic): (3) extends L6's federation fixtures — three mock peers, role headers forwarded, cards resolve AT the peer, pinned-resolved model bypasses peer cards, LAN hop events render the Sankey trace; restarts: all role flows survive daemon restarts on every host (adoption).

## The rest

Manifest retirement (R8): owner sign-off at L8 KICKOFF (recorded in this leg's log; the sign-off gates the manifest kind deletion in the docs/removals pass — kickoff placement lets the deletion ride L8's own docs pass).

Build order: spike (owner nod; feeds budget rule + mac model list) → probe mac branch + ids → native adapter on mac + setsid/launchd → LAN peer flows (b) wiring verified → distributed-role acceptance (fixture tier first, host-tier second, flows a/b/c per the tables) → fleet guide doc (the wiring tables generalized per host, naming the fleet-profile file for engine/pin keys and daemon config for `lan.*`) → manifest sign-off recorded + deletion in docs pass.

Acceptance (Ray): (1) probe mac fixtures (ids, Metal presence, budget from spike value); (2) budget layer on unified-memory fixture (cap = spike number; over-budget → no-fit/evict path); (3) mac adapter lifecycle (setsid detachment — a daemon SIGTERM does not reach the engine group; adoption via marker scan after restart; launchd plist shape); (4) flows (a)/(b)/(c) at fixture tier (wiring-table selectors resolve, role headers forward, peer cards apply, pinned bypass at peer, restart survival) + host tier: the two tables executed literally, three dashboards, cross-host Sankey, failure drill; all prior suites green, both feature modes; node + html-js; `--all-targets` compile on the darwin CI job (integration suites stay Linux-tier).

Docs: UPGRADE (mac install + fleet guide), PRD/README (distributed roles), CHANGELOG.
