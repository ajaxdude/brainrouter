# Upgrade and host-procedure guide

Operator procedures that are **owner-nod-gated** live here. Every procedure
below is destructive on a host running live work; obtain the nod (charter:
fedora1/fedora2 destructive actions are reserved) before running it.

The supervisor invariants these procedures rely on:

- `swap takeover` / `swap adopt` attach to what is already serving; they never
  spawn a second owner on the same device.
- A cutover signals **only** supervisor-owned marker pids (or a unit's own
  pids after an explicit `--override`). Icon-launched servers stay untouched
  until adopted.
- A model inside the Active-Consumer Set (live routed stream, or the idle
  grace `active_consumer_grace_s`, default 120 s) is never evicted. The block
  reason is `blocked:active-consumer`.
- `MemAvailable ≥ min_mem_gib` is re-checked after the drain and settle; a
  below-floor admission shows `Waiting(gate)` with threshold + current value
  and fails after `BRAINROUTER_GATE_TIMEOUT_MS` (default 10 min) unless
  `--force-gate` is passed.

Environment names used below (all optional overrides; defaults in
`src/fleet_profile.rs`, `src/engine_strixite.rs`, `src/engine_hipfire.rs`):

| Variable | Meaning |
|---|---|
| `BRAINROUTER_FLEET_ENGINES` | force the adapter set (`strixite,hipfire`) |
| `BRAINROUTER_ACTIVE_CONSUMER_GRACE_S` / `_MS` | idle grace (default 120 s) |
| `BRAINROUTER_MEM_AVAILABLE_GIB` | override the memory probe (fixtures/ops) |
| `BRAINROUTER_GATE_TIMEOUT_MS`, `_GATE_POLL_MS` | gate timeout / poll |
| `BRAINROUTER_CUTOVER_SETTLE_MS` | post-pid-gone settle (default 2000 ms) |
| `BRAINROUTER_CUTOVER_RECHECK_MS` | bounded single ACS re-check (default 5000 ms) |
| `BRAINROUTER_STRIXITE_BIN`, `_CONF_DIR`, `_CONF_TEMPLATE(_FILE)`, `_PORT_RANGE`, `_MIN_MEM_GIB`, `_UNIT` | strixite adapter |
| `BRAINROUTER_HIPFIRE_BIN`, `_PORT`, `_MODELS_DIR`, `_MAX_SEQ`, `_MIN_MEM_GIB`, `_UNIT` | hipfire adapter |
| `BRAINROUTER_SYSTEMCTL` | `systemctl` replacement (wrapper script / fixture) |

---

## 1. strix-halo: live strixite pilot (A to B swap, no GUI icons)

Goal: swap two strixite models from the dashboard/CLI while a harness session
is live, and prove the session is protected by the Active-Consumer span.

Prerequisites (per host, once):

1. `strix_server` on `PATH` (or set `BRAINROUTER_STRIXITE_BIN`), strixite build
   root present so the platform probe offers `strixite` (gfx1151 + GTT ≥ 100
   GiB). `brainrouter platform show` must list `strixite` as offered.
2. Catalog the two models once, with the conf-owned flags **absent** from the
   launch args (`--config/--weights/--model/--port` are generated):
   `brainrouter swap add model-a.json` / `model-b.json`.
3. Confirm the port scope in the Engines tab: generated confs allocate from
   `BRAINROUTER_STRIXITE_PORT_RANGE` (default 5940–5999) and avoid every
   Residency endpoint, imported explicit `--port`, and llama-swap's dynamic
   `${PORT}` range.

Pilot checklist:

1. Start a harness session against brainrouter and send a prompt that holds a
   stream for at least a few seconds (long generation or tool loop).
2. In the dashboard Engines tab, confirm the model card shows
   `active consumer (live)` and `/api/engines/consumers` lists it protected.
3. With a second model catalogued, press **Ensure** on model B. Expected:
   the request waits (loading) while model A is **not** stopped; if the
   admission reaches eviction it returns `blocked:active-consumer` and
   `/api/engines/events` records the block.
4. Wait for the session's stream to finish; within the grace window the block
   persists. After `active_consumer_grace_s`, press **Ensure** again.
   Expected cutover log order in `/api/engines/logs?model=model-a`:
   `cutover: SIGTERM …` → `cutover: pid gone …` → `cutover: settle 2000 ms`
   → `cutover: mem gate ok (available X ≥ 100.0 GiB)` → model B ready.
5. Verify exclusivity: `/api/engines` shows model A `stopped`, model B `ready`;
   only one `strix_server` pid exists for the device.
6. Negative check: lower `BRAINROUTER_MEM_AVAILABLE_GIB` below 100 (or run
   with the real probe on a loaded box) and Ensure model B. Expected:
   `state: waiting`, `gate.threshold_gib=100`, current value shown; after the
   gate timeout the model is `failed` with the reason, and a re-run with the
   dashboard **Swap anyway** / `--force-gate` proceeds (audit row in
   `budget.json` kind `override`).
7. Rollback: `brainrouter swap evict model-b`; click the model A icon (or run
   the icon script) as before. The supervisor observes the icon process as an
   **external** resident and attaches instead of spawning. To hand ownership
   back explicitly: `brainrouter swap evict model-a` then start the icon.

Retirement (operator-gated): once the pilot passes, stop launching icons,
`brainrouter swap adopt <model>` each still-running icon server (marker
written, no kill), and keep the icons only as an emergency fallback for one
week.

---

## 2. fedora1: hipfire unit migration (Q4 default)

Goal: the `hipfire-serve.service` system unit stops owning `:8080`; the
supervisor runs hipfire as a native resident under a per-child transient user
scope, with adoption surviving daemon restarts.

Port ownership is a single-owner rule on `:8080`. Do **not** start the icon
while the supervisor owns the port.

Checklist (owner nod required):

1. Record the current unit state:
   `systemctl status hipfire-serve --no-pager` and
   `systemctl show hipfire-serve -p MainPID,ActiveState,ControlGroup`.
2. Catalog the card(s): `ls ~/.hipfire/models/*.mq4`; a missing `.mtp`
   sidecar is a warning only (the engine auto-loads by stem, and the Engines
   tab shows the warning). `brainrouter swap add swift.json` with
   `engine: "hipfire"`, `files: ["~/.hipfire/models/swift.mq4"]`.
3. **Disable the system unit** (privilege required, operator action):
   `sudo systemctl disable --now hipfire-serve.service`.
   The boot probe now reports the unit inactive; the port is free.
4. Start the managed engine: `brainrouter swap ensure swift` (or the Engines
   tab Ensure). Expected: the child starts under
   `systemd-run --user --scope`, adoption marker
   `$XDG_STATE_HOME/brainrouter/adoption/swift.json` is written, and
   `/api/engines` shows source `native`, the hipfire schema version + flags,
   and no `schema_drift` badge.
5. Migrate the harness: point requests at the model id (role card pinned or
   bare key). The first request queues at "loading" and streams when ready.
6. Squat drill (non-destructive): start a mock HTTP server on `:8080` while
   the managed engine is stopped, then Ensure. Expected: `failed` with
   `port is owned by a foreign server serving […] (expected […])`; the
   Engines tab shows the banner, and nothing on the port was signalled.
7. Engine-upgrade drill: replace the hipfire binary/build (version marker
   changes). Restart the daemon; the probe re-derives the flag schema and the
   Engines tab shows the `schema_drift` badge. Re-run `swap ensure`: an
   unknown stored flag is rejected at spec build instead of poisoning the
   spawn.
8. Rollback: `brainrouter swap evict swift`, then
   `sudo systemctl enable --now hipfire-serve.service`. Verify `:8080` is
   owned by the unit again (`ss -ltnp | grep 8080`).

`swap takeover swift --override` is the one-action alternative for step 3+4
when the system unit is still active: it resolves the unit's own pids
(`MainPID` + control-group members), attaches when they already serve the
model, and otherwise SIGTERMs those pids, stops the unit, and spawns through
the normal gate. Without `--override` it refuses with
`blocked:unit-owned`.

---

## 3. fedora2: strixite prod (additive)

Same as §1 with the prod model set. Confirm the feed platform id resolves to
`fedora2` (`brainrouter platform show`) and that `deepinfra` cloud routing is
untouched (L6). Rollback = evict the managed model and start the existing
launch path.

---

## 4. Recovery and rollback invariants

- **Daemon restart** never kills adopted children: scoped children survive and
  are re-adopted from markers at boot. A `spec_stale` badge after an upgrade
  converges with `swap evict` + `swap ensure`; it is never a kill.
- **Stale markers** (dead pid or a cmdline that no longer names the GGUF) are
  swept at boot and recorded in `budget.json` (`kind: stop`).
- **Override audit**: every `--override` / `--force-gate` / `swap adopt` /
  `swap takeover` writes a `budget.json` entry (`kind: override|adopt|takeover`)
  with `ts`, `model`, `reason` and `actor` (API mutations: `admin@<origin>`;
  CLI: `cli:uid=<uid>`). Every refused admission writes `kind: blocked` with
  the `blocked:*` reason. Read them via `GET /api/engines/events` or
  `brainrouter swap status`.
- **Unregistered external occupant**: a spawn on an exclusive device is
  refused with `blocked:external-occupant` while a matching external process
  resolves to no catalog entry. Fix it by cataloguing the model and running
  `swap adopt --endpoint <url>` (or stopping the icon), then re-running the
  ensure; `--override` is the audited one-action bypass.
- **Full L5 revert**: the L4 surfaces are unchanged; stop using the adapters
  (`BRAINROUTER_FLEET_ENGINES=` empty) and run the icon scripts. External
  residents are still observed and attached, never signalled.
