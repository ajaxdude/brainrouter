# Fix `toolbox create --label` — create all toolboxes via a sidecar ownership record

## Status

- Workflow state: `drafting` (v2 — resolves Round-1 FAIL). User chose the **full security-hardened** scope ("do all of them").
- Change classification: **standard** — changes container creation + the ownership-detection semantics that gate destructive ops (delete/update/recreate). Security-relevant (must keep protecting cockpit-created containers) → full rigor.
- Human review: user unavailable, delegated ("work autonomously, make good decisions"). Skipped per HankNDory rule 7; isolated Dory critic + mean review still run.

Revisions:
- v1 — 2026-09-24 — Initial draft.
- v2 — 2026-09-24 — Dory round-1 fixes: **container-ID ownership** (not name+image); **gate every destructive path** (update, delete, legacy `/api/upgrade/toolbox`); **ID-bound removal** (`podman rm <id>`) to close external TOCTOU; **non-destructive adopt** (record live ID, no recreate); explicit **partial-failure state machine**; `AppState`+`daemon.rs` wiring + changed signatures; explicit chmod 0600; correct live AC endpoint; rollback-recovery note.

## Problem

brainrouter's "Toolboxes" panel "+ create" (and update/adopt) build the container with `toolbox create --image <image> --label io.brainrouter.managed=true --label … <name>`. On the strix host (and any **Fedora Toolbx** host) `toolbox create` has **no `--label` flag** — verified live: its only flags are `--authfile/--distro/--image/--release`. So every create/update/adopt fails with `Error: unknown flag: --label`, and **no toolbox can be created via brainrouter for any backend** (llama.cpp/ds4/vllm/halogen/r9v). This blocks provisioning the toolboxes brainrouter is meant to manage. (Server Mode is unaffected — it uses `podman run --label`, which podman supports.)

## Goals and non-goals

Goals:
- Make `create`/`update`/`adopt` succeed on Fedora Toolbx by using the proven-working `toolbox create --image <image> <name>` (no `--label`).
- Preserve the ownership guarantee that made labels exist: brainrouter only ever mutates/deletes containers **it** created or explicitly adopted; cockpit/hand-made containers stay "unmanaged" until adopted.
- Keep `toolbox enter` compatibility (stay on `toolbox create`, not a bare `podman create`).

Non-goals:
- Server Mode's `podman run --label` ownership (works; untouched).
- Retroactively labeling pre-existing containers (podman can't add labels in place — same constraint the design already documents).
- Changing the adopt UX contract (still an explicit, destructive recreate-in-place).

## Current system

Verified (`src/server.rs`):
- Ownership label constants (`:2389-2391`): `io.brainrouter.managed` (+ `catalog_id`, `catalog_revision`). Only `LABEL_MANAGED` is ever **read**; `catalog_id`/`catalog_revision` are write-only/informational.
- `toolbox_container_is_managed(name)` (`:2418`) — `podman inspect --format '{{ index .Config.Labels "io.brainrouter.managed" }}'` == `"true"`. Callers: container listing (`:2634`, shows `managed`), `adopt` (`:2807`, refuse if already managed). `delete`/`update` operate through the managed gate / recreate primitive.
- `recreate_toolbox_container(tb, pull)` (`:2673`) — the shared create/update/adopt primitive: optional `podman pull`; `toolbox rm --force`; then `toolbox create --image <image> --label …×3 <name>` (the failing call, `:2700-2713`).
- `create_toolbox_container` (`:2743`), `update_toolbox_container` (`:2766`, pull+recreate), `adopt_toolbox_container` (`:2787`, recreate-in-place unchanged image), `delete_toolbox_container` (`:2816`). All hold a per-container-name lock (`:160`).
- Server Mode's separate `LABEL_MANAGED` on `podman run` (`src/server_mode.rs:316…`) is unrelated and works.
- Live: `toolbox --version` = Fedora Toolbx; `toolbox create --help` lists only `--authfile/--distro/--image/--release` — **no `--label`**.

## Requirements and acceptance criteria

Ownership is now **container-ID-based**. The sidecar maps `container_name → { container_id, image, catalog_id, catalog_revision, created_at }`. `container_id` is the full engine ID from `podman inspect --format '{{.Id}}'` taken right after the container is created/adopted.

R1. **Create** builds `toolbox create --image <image> <name>` with **no `--label`** and **no pre-emptive force-remove**: if a container of that name already exists → 409 (never silently clobber). On success, inspect the new container's `.Id` and record the sidecar entry. — AC: create succeeds on strix for any backend; a name collision returns 409 without touching the existing container.

R2. **`toolbox_container_is_managed(name)`** returns true iff **either** the legacy podman label `io.brainrouter.managed == "true"` (back-compat for any label-bearing container, incl. Server-Mode) **or** the sidecar has an entry for `name` whose `container_id` **exactly equals** the live `podman inspect --format '{{.Id}}' <name>`. Image is metadata only. — AC: unit matrix — labeled→managed; sidecar id-match→managed; sidecar present but **id-mismatch (external same-name replacement)→unmanaged**; neither→unmanaged.

R3. **Every destructive path is ownership-gated**, using **one inspect snapshot** (`.Id`, `.Config.Labels`, `.Image`) taken under the per-container-name lock, authorized, then an **ID-bound** mutation:
   - `delete_toolbox_container` — refuse (403) if not managed; else `podman rm --force <id>` (the snapshot id), then remove the sidecar entry.
   - `update_toolbox_container` — refuse (403) if not managed; else replace: `podman rm --force <old_id>`, `toolbox create --image <image> <name>`, inspect new `.Id`, rewrite the sidecar entry.
   - legacy `POST /api/upgrade/toolbox[/<name>]` (`upgrade_toolbox`) — bring under the same gate + per-name lock + sidecar update, **or** retire the routes. (Chosen: gate it — same authorize-snapshot + ID-bound recreate + sidecar update.)
   — AC: direct HTTP tests — unmanaged container → 403 on delete/update/upgrade, container untouched; an external same-name replacement between snapshot and removal is **not** deleted (ID-bound `rm` targets the snapshot id, which no longer exists → no-op/err, surfaced).

R4. **Non-destructive adopt:** `adopt_toolbox_container` inspects the existing live container, records its `.Id` in the sidecar, and returns — **no recreate, no image change** (container-ID ownership needs no label to attach). Refuse (409) if already managed. — AC: adopt records the exact live id; the container is byte-for-byte unchanged (same id after adopt).

R5. **Partial-failure state machine** (explicit): create/adopt return 200 **only after** the sidecar entry is durably written; if the sidecar write fails after a create, remove the exact newly-created `.Id` and return 500 (no orphaned unmanaged container). For update, if the post-recreate sidecar write fails, report 500 with the new id noted (do not leave a silent unmanaged container). Delete removes the sidecar entry even if the container was already absent (idempotent cleanup of a stale entry). A corrupt/unreadable sidecar is treated as "no ownership" for reads (fail safe) and surfaced as 500 on writes. — AC: fault-injection tests for each ordering boundary.

R6. **Sidecar persistence:** atomic (uuid temp in the same dir + `sync_all` + rename + best-effort parent fsync) under a dedicated write lock; the temp file is created **0600** before rename (verified in a Unix test — the runtime-state modules do *not* chmod, so this is explicit here). Path from `AppState.managed_toolboxes_path`, derived next to the other state files (honoring a custom `--config`). — AC: perms test; concurrent record/remove serialize.

R7. **No security regression / back-compat:** a cockpit/hand-made container (no label, not in sidecar) is unmanaged → delete/update/upgrade refuse it. A container still carrying a real `io.brainrouter.managed` label is still recognized via the OR. — AC: cockpit container → managed:false + refused; labeled container → managed:true.

R8. **Rollback recovery:** reverting to pre-v2 code makes label-less sidecar-owned containers show unmanaged (they have no podman label). Documented, with a manual recovery (re-adopt, which is now non-destructive) — AC: doc note + adopt re-establishes management post-rollback without data loss.

## Technical plan

New module `src/managed_toolboxes.rs`: persisted `BTreeMap<String, OwnershipEntry>` (`{ container_id, image, catalog_id, catalog_revision, created_at }`) with a global write lock. API `load/record/remove/get`. Atomic write: uuid temp in same dir, **created 0600**, `sync_all`, rename, best-effort parent fsync (adds the explicit chmod the runtime-state modules lack).

`src/server.rs` (thread `AppState.managed_toolboxes_path`):
- `inspect_container(name) -> Option<Snapshot{id, managed_label, image}>` — one `podman inspect` with combined `--format`.
- `toolbox_container_is_managed(name)` = `snapshot.managed_label=="true" OR sidecar.get(name).container_id == snapshot.id` (R2).
- create: `toolbox create --image <img> <name>` (no `--label`, no pre-emptive rm, 409 if exists) → inspect `.Id` → `record` → on record-fail `podman rm --force <new id>`+500 (R1/R5).
- adopt: inspect live → 409 if managed → `record(id)`, no recreate (R4).
- update: authorize snapshot → `podman rm --force <old id>` → create → inspect new id → rewrite sidecar (R3).
- delete: authorize snapshot → `podman rm --force <id>` → `remove(name)` (idempotent) (R3/R5).
- legacy `upgrade_toolbox`: add per-name lock + authorize snapshot + ID-bound recreate + sidecar update (R3).
- drop `--label` args + the write-only `LABEL_CATALOG_ID/REVISION` constants; keep `LABEL_MANAGED` read for the back-compat OR.

`src/daemon.rs`: build `managed_toolboxes_path` (`~/.local/share/brainrouter/managed_toolboxes.json`, honoring custom config dir) into `AppState`.

```
create ─ toolbox create --image <img> <name> (no --label, fail if exists) ─ inspect .Id ─ sidecar.record
adopt  ─ inspect live .Id ─ sidecar.record            (non-destructive)
update ─ authorize(snapshot) ─ podman rm --force <old id> ─ create ─ inspect .Id ─ sidecar.record
delete ─ authorize(snapshot) ─ podman rm --force <id> ─ sidecar.remove
is_managed(name) = snapshot.label=="true" OR sidecar[name].container_id == snapshot.id
```

## Detailed implementation
Files: `src/managed_toolboxes.rs` (new); `src/server.rs` (inspect_container + is_managed refactor, create/adopt/update/delete/upgrade gating + ID-bound removal, drop `--label`, `AppState` field); `src/daemon.rs` (path + AppState construction — omitted in v1); `src/lib.rs` (module decl). Tests: sidecar record/remove/get/atomic/0600 (tempdir); is_managed/authorize matrix + partial-failure ordering via an injected snapshot/failure seam (no podman). Order: sidecar module+tests → inspect+is_managed refactor+matrix → create(no-label+record+rollback) → adopt(non-destructive) → delete(ID-bound+remove) → update+legacy-upgrade gating → daemon wiring. `cargo build --release` on strix after the sidecar module and after server wiring.

## Alternatives considered

- **Bare `podman create` for managed containers** (the design doc's other documented fallback). Benefit: could pass `--label`. Cost: loses Toolbx's default host integration + the `toolbox enter` "Enter" parity feature; a bigger behavioral divergence. Rejected: the sidecar keeps `toolbox create` semantics intact and is smaller.
- **Runtime capability-detect `--label` (probe `toolbox create --help`)**, use labels when supported else sidecar. Benefit: labels on hosts that support them. Cost: two ownership sources to keep consistent + a probe. Rejected as primary, but the R3 **OR** with the podman label already gives forward-compat for a labeled container without a probe.
- **Name/image match only, no sidecar.** Rejected — exactly the "silently treat any name/image match as owned" risk §5 called out (could clobber a cockpit container that matches a catalog name). The sidecar records *explicit* brainrouter ownership.

## Detailed implementation
Files: `src/managed_toolboxes.rs` (new); `src/server.rs` (recreate/is_managed/delete + `AppState` field + wiring); `src/lib.rs` (module decl); `AppState` construction site (set the path). Tests: `managed_toolboxes` unit tests (record/remove/get/atomic) + `is_managed` decision matrix (label/sidecar/image-match) via a small seam. No migration (additive sidecar; absence = empty). Ordered: sidecar module + tests → is_managed OR-logic + test → recreate drop-label + record → delete remove → wiring.

## Testing and evaluation
- Rust unit: sidecar record/remove/get + atomic-write (tempdir); `is_managed` matrix (mock the label + image lookups via an injected resolver so it's filesystem-/podman-free). `cargo test --locked -- --test-threads=1`; clippy.
- Live (strix, via brainrouter API): `POST /api/toolbox-containers {toolbox_id:"strix-halo-halogen-flash"}` → 200 (was 500 unknown-flag); container exists; `/api/toolboxes` shows it `managed:true`; a pre-existing cockpit container still `managed:false`; delete removes it + clears the sidecar. (Toolbox image pull is real but bounded.)

## Security, privacy, reliability, and operations
- Preserves the anti-clobber guarantee (R6): only explicit brainrouter create/adopt writes the sidecar; image-match guards name reuse (R3). Sidecar is local JSON, atomic-written, chmod 0600 (mirrors existing state files). No secrets. Failure to read the sidecar → treat as "not owned" (fail safe toward not touching a container).

## Rollout, migration, and rollback
- Additive; no migration (empty sidecar = nothing owned yet; existing labeled containers still recognized via the OR). Rollback = revert commit; the sidecar file is inert if unused. Deploy via the runbook to strix `master` + GitHub; back up `benchmarks.sqlite3` first.

## Risks and mitigations
- **Name-reuse false positive** (sidecar entry + a different same-named container) → mitigated by the image match (R3).
- **Sidecar/container divergence** (container removed outside brainrouter) → inert stale entry; managed also requires existence (callers check `podman ps`); delete tolerates already-absent.
- **Concurrency** → per-container-name lock (existing) + sidecar write lock.

## Open questions
None blocking. (Whether to also backfill sidecar entries for any container currently carrying a real label — not needed; the OR handles them live.)

## Decision log
- Drop `--label` (Fedora Toolbx has no such flag — verified) and keep `toolbox create` (preserve `toolbox enter` compat) rather than switching to `podman create`.
- Sidecar ownership + image-matched `is_managed` (OR the legacy label) — preserves the cockpit-protection guarantee without labels.

## Referenced files
- `src/server.rs` — labels (`:2389`), `is_managed` (`:2418` + callers `:2634`,`:2807`), `recreate_toolbox_container` (`:2673`, create call `:2700-2713`), create/update/adopt/delete (`:2743`/`:2766`/`:2787`/`:2816`), per-name lock (`:160`), `AppState`.
- `src/server_mode.rs` — separate `podman run --label` ownership (untouched, for contrast).
- existing runtime-state modules (e.g. `src/review/runtime_state.rs`) — atomic-write pattern to mirror for the sidecar.
- live `toolbox create --help` (no `--label`), `/api/toolbox-containers` create/list.

## Dory validation record

- **Critic review — Round 1** (rubber-duck sub-agent, isolated fresh context; doc v1; verified against `server.rs`, `server_mode.rs`, `review/runtime_state.rs`, `daemon.rs`).
  - Verdict: **FAIL** (4 blocking, 5 important).
  - **B1** — the doc wrongly claimed update/delete "operate through the managed gate": in fact **only listing (`:2634`) and adopt (`:2807`) call `is_managed`**. `update_toolbox_container` (`:2766`) and `delete_toolbox_container` (`:2816`) do **no ownership check**, and the legacy `/api/upgrade/toolbox[/<name>]` (`:572-596` → `upgrade_toolbox` `:3864-3917`) pulls+removes+recreates with no ownership check, lock, label, or sidecar. → A safe fix must enumerate + gate **every** destructive path (or retire legacy upgrade), with direct HTTP tests. (Pre-existing gap, widened scope.)
  - **B2** — `name + image` doesn't establish ownership (same-name+same-image cockpit replacement returns managed). Use the engine **container ID**: record it on create/adopt; `managed = legacy-label==true OR sidecar[name].container_id == live container id`. Image/digest kept only as metadata. (Also: code reads `{{.Image}}` at `:3851`, not `.ImageName`.)
  - **B3** — the per-name lock only serializes this daemon; cockpit/another podman client can replace a name between inspect and `toolbox rm <name>`. Split create (no pre-emptive force-remove) from replace; take one inspect snapshot (ID+label+image); authorize it; use an **ID-bound removal**, not name-based.
  - **B4** — undefined partial-failure semantics (create ok/sidecar-write fails; update destroys old then persist fails; delete ok/remove fails; already-absent delete + stale entry). Needs an explicit state machine + fault-injection tests.
  - Important: I1 `AppState` has no state-file field + construction is in `src/daemon.rs:422-453` (omitted from plan); `toolbox_containers_list`/`recreate_toolbox_container` take no state — signatures change. I2 `review/runtime_state.rs` does **not** chmod 0600 (my "mirrors existing" claim false) — must set mode explicitly + test. I3 adopt "unchanged image" is inaccurate (recreate always uses catalog `tb.image`); with container-ID ownership, adopt can record the live container **without** destroying it. I4 live AC named `/api/toolboxes` (legacy, no `managed`) — should be `/api/toolbox-containers`. I5 rollback strands sidecar-only (label-less) containers as unmanaged — document recovery.
  - Self-audit: relied only on the doc + files/AppState-construction it directed inspection to.
  - Disposition: **needs v2** with container-ID ownership + full destructive-path gating + ID-bound removal + partial-failure state machine. Scope decision for the user: minimal "make create succeed" vs. also fixing the pre-existing ungated update/delete/legacy-upgrade paths (a broader security change).

## Human approval
Skipped by explicit user delegation ("work autonomously, make good decisions"); user unavailable. Recorded per HankNDory rule 7. Isolated Dory critic + mean review still performed.
