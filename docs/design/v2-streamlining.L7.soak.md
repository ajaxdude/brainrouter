# v2 L7 soak protocol — native supervisor under real conditions

Suite tier (`tests/v2_l7_acceptance.rs`) proves the fixture-visible contract.
These rows need the things a fixture cannot synthesize: real llama-server
children, real signals, real VRAM, real systemd lifetime. Run them on a
staging host (fedora2-class) before a fedora1-class flip; record results by
appending a dated `## Soak <date>` section to this file (this is the artifact
the L7 review row cites — completion is evidenced, not asserted).

Conventions: `brainrouter` = the v2 binary under test; the daemon logs to
`brainrouter serve` stdout (systemd journal when unit-managed). `state` =
`$XDG_STATE_HOME/brainrouter` (default `~/.local/state/brainrouter`).

## S1 — Flip-gate field behavior

1. Shim running with ≥ 1 loaded model; set `swap.mode: native`; start daemon.
   EXPECT: exit non-zero; message names every loaded model; unit reports failed.
2. Dashboard (shim mode) **Engines → Unload models** (or `brainrouter cli
   flush-models`); start daemon again. EXPECT: boots; log line confirms import.
3. Stop llama-swap entirely; restart daemon. EXPECT: boots (unreachable shim =
   nothing resident).
4. Black-hole the shim port (`iptables -A INPUT -p tcp --dport 8081 -j DROP`
   on the loopback-less box, or bind-and-not-accept on the shim port):
   EXPECT: boot REFUSES with "state is unknown" — NOT a silent boot.
   This is the double-residency guard; if this row ever boots, treat it as a
   release blocker.
5. Undo; shim back up. EXPECT: row 2 behavior.

## S2 — Crash-restart is bounded (no restart storms against real VRAM)

1. Native boot; request a model so a real llama-server spawns.
2. `kill -KILL` the llama-server child mid-serve.
   EXPECT: supervisor marks the model failed/crashed within the poll interval;
   a bounded number of restarts (log lines `swap crash-restart`); after the cap
   the model enters `failed` and a NEW request re-drives a single spawn
   (`/api/engines` shows one child per model, ever).
3. `kill -TERM` the child. EXPECT: same accounting, graceful stop path
   (`SIGTERM→grace→KILL` per adapter contract); no orphaned GPU context
   (`rocm-smi`/`nvidia-smi`/`amdgpu` sysfs util back to baseline).

## S3 — `shutdown_all` leaves zero children (rollback contract)

1. Two+ models resident (main + pinned group).
2. `systemctl --user stop brainrouter` (or SIGTERM the daemon).
   EXPECT: log shows per-engine stop; within the grace window NO `llama-server`
   process survives (`pgrep -af llama-server` empty); flock files under
   `state/swap` released; adoption markers for stopped engines removed
   (`ls state/adoption*` empty for this daemon's ids).
3. Start the daemon again. EXPECT: clean import; no adoption of dead pids.

## S4 — Adoption across a daemon restart (children outrun the daemon)

1. Resident model; `kill -TERM` the DAEMON only (child must survive — scope
   contract).
   EXPECT: child still serving (curl its `/health`).
2. Start the daemon. EXPECT: boot log reports adoption (`adopted N`);
   `/api/engines` shows the resident with `source: native` and the SAME pid;
   no second spawn; PP progress feed live.
3. Spec-stale variant: edit the model's launch args in the catalog while the
   child runs, restart daemon. EXPECT: adopted + `spec_stale` badge; child NOT
   killed; `swap evict` + next request converges it.

## S5 — PP progress under multi-residency + aliases

1. Resident main + second resident (pinned). Issue concurrent streams to both.
   EXPECT: `/api/inflight` rows carry pre-alias requested keys AND both rows
   show advancing `pp_progress` (one poller pass per ready resident — the I3
   fix; single-resident feeds regress this row).
2. Request the same model BY ALIAS while resident. EXPECT: its row also shows
   progress (alias fan-out via `served_names`).
3. A llama-server build without `/slots` (ds4-DSpark class): EXPECT: rows
   still appear; progress fields stay absent (graceful no-op, not zeros that
   lie about 0%).

## S6 — Groups refusal UX

1. Catalog holds llama-swap GROUP members (imported). Request a member.
   EXPECT: HTTP 503; message explains slot-sharing + owner-decision pending
   (`blocked:groups-pending`); dashboard card surfaces the reason text; the
   group NAME as a model 404s as "not in the swap catalog".
2. Non-member models unaffected.

## S7 — Long soak (wall-clock)

Run 24 h of the normal harness mix (OMP + one cloud-fallback path) on native:
- `journalctl --user -u brainrouter` contains zero `WARN.*swap` beyond expected
  (crash-restarts only if you induce them), zero flip-gate messages after boot.
- `/api/routing-events`: no event ever carries `effective_provider: llama-swap`
  for local traffic (mode-pin proof over real traffic).
- Memory: `rocm-smi`/journal GTT trend flat across the window; no double
  residency (each catalog key at most one live pid).

## Sign-off

A soak is PASS when S1–S6 pass in one window and S7 runs clean for 24 h.
Record: host, binary commit, dates, per-row notes, and any deviations —
under this file's `## Soak <date>` sections.
