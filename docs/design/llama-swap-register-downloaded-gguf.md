# Register a downloaded llama_cpp GGUF into llama-swap

## Status

- Workflow state: `approved-for-implementation` — Dory critic round 1 blockers B1–B6 and important findings I1–I3 have all been resolved by verified follow-up investigation and the v2 design below. Implementation may proceed from this document.
- Change classification: **standard** — new public endpoint + mutation of the critical, hand-tuned `llama-swap` `config.yaml` (gates all local inference) + dashboard UI. High blast radius on the config file → full rigor.
- Human approval: skipped by explicit user delegation/autopilot for this implementation session. The coordinator supplied concrete resolutions for every Dory blocker and instructed implementation; recorded per HankNDory rule 7. Safety guardrails remain mandatory: lock + backup + YAML validation + atomic write + reload health poll + restore-on-failure.

Revisions:
- v1 — 2026-09-24 — Initial draft; Dory critic round 1 found 6 blockers and 3 important findings.
- v2 — 2026-09-27 — Incorporated resolved B1–B6/I1–I3 decisions: pattern-scoped GGUF resolver, job-retained `quant_pattern`, structural YAML dedup + safe text insertion, shared config writer lock, backup/atomic/reload/health/restore contract, destructive route gate, toolbox path visibility warning, and expanded tests. Marked approved-for-implementation.

## Problem

Models fetched through brainrouter's Downloads tab land as files in a models dir but are **not** registered anywhere llama-swap can serve them — llama-swap only serves models that have a hand-written entry in its `config.yaml`. So after downloading a llama.cpp GGUF via brainrouter, the user still has to hand-edit `config.yaml` (compute the exact `--model` path, add a `cmd:` block, reload) before it is routable. The user asked: "will models downloaded like that be available in llama-swap? if not can we make it happen." This feature makes a completed **llama_cpp** GGUF download one-click-registerable into llama-swap.

## Goals and non-goals

Goals:
- An explicit, user-triggered action that appends a working baseline llama-swap `config.yaml` entry for a completed **llama_cpp** download and reloads llama-swap.
- Resolve the exact on-disk `.gguf` (incl. sharded and folder-quant layouts) for the `--model` flag, or fail clearly rather than write a wrong path.
- Never corrupt the existing config: backup + YAML-validate + atomic write + dedup + restore-on-failure.

Non-goals:
- ds4/halogen/vllm/r9v: these are served by their own Server-Mode servers (serving-identities), not llama-swap — out of scope here (a separate "route to Server-Mode endpoint" concern). Only `llama_cpp` GGUFs are in scope.
- Automatic registration on download completion (rejected — surprising mutation of a hand-tuned file; explicit action only).
- Inferring per-model tuning (reasoning/spec-draft/sampling/mmproj/context). The generated entry is a **generic baseline** (`${ls} --port ${PORT} ${common} --model <path>`); the UI states it must be tuned.
- Editing/removing existing entries, or reformatting/round-trip-reserializing the whole YAML (would strip the user's comments/macros/block scalars).

## Current system

Verified from repository + live (`127.0.0.1:9099` via brainrouter):
- **llama-swap config** (`/api/llama-swap-config`, `src/server.rs:1307` GET / `:1327` POST): a YAML file at `state.llama_swap_config_path` with a `macros:` block (`ls` = `/home/papa/.local/bin/llama-server-toolbox`, `common` = shared flags, `ctx`, `npredict`…) and a `models:` map. Each entry:
  ```yaml
  "model-id":
    name: "Display Name"
    aliases: [...]           # optional
    macros: { ctx: "..." }   # optional per-model
    env: [...]               # optional
    cmd: |
      ${ls}
      --port ${PORT}
      ${common}
      --model /mnt/models/.../file.gguf
  ```
  The POST handler already: rejects >1 MB, `serde_yaml::from_str::<Value>`-validates, then atomic tmp+rename writes (`src/server.rs:1338-1358`). `serde_yaml` is a dependency.
- **Reload:** `POST /api/restart/llama-swap` (`src/server.rs:521`) restarts the llama-swap service (the existing "reload config" path). llama-swap loads models on demand, so a restart with a valid config + one extra entry does not affect the other models beyond a brief restart; inference is currently idle.
- **llama_cpp download layout** (`src/model_downloads.rs:439-471`, `build_download`): destination = `effective_models_dir(LlamaCpp).join(repo_basename)` where `repo_basename = repo.rsplit('/').next()`. `quant_pattern` is one of: a specific `X.gguf` (downloaded to `destination/…/X.gguf`), a glob `*…*.gguf` (`--include`), or a folder name like `BF16`/`UD-IQ2_M` (`--include <pattern>/*`, i.e. shards under `destination/<pattern>/`). **`expected_files` is empty for llama_cpp** — brainrouter tracks no exact file list, so the `.gguf` must be resolved by scanning the destination.
- **models dir resolution** (`src/model_downloads.rs:263`, `effective_models_dir`): `backends.<id>.models_dir` cockpit override → catalog default → `~/models`.
- **Catalog** (`src/toolbox_catalog`): the llama_cpp model entry carries `repo`, `name`, and the quant options; `toolbox_models` exposes entries per backend.

## Requirements and acceptance criteria

R1. New `POST /api/llama-swap/register-model` with typed JSON body `{ "model_id": "<catalog llama_cpp id>", "quant_pattern": "<the pattern the completed download job used>" }`. Returns `{status, llama_swap_id, model_path, message, reloaded, warning?}`. — AC: a completed llama_cpp download can be registered; the config gains exactly one entry.

R2. Backend restriction: the request implies backend `llama_cpp`; resolve `model_id` only in the llama_cpp catalog. A non-llama_cpp/server-mode model id is rejected with 400 guidance to use Server Mode. — AC: ds4/halogen/vllm/r9v/gufo ids do not register into llama-swap.

R3. GGUF path resolution is deterministic and quant-scoped. `resolve_primary_gguf(destination, quant_pattern)` recursively gathers `*.gguf`; if `quant_pattern` ends in `.gguf`, match the basename with `*` wildcard support; otherwise scope to `destination/<pattern>/`. Reject auxiliary GGUFs containing `mmproj`, `mtp`, `draft`, `vision`, `dspark`, `-spec`, or `speculat` (case-insensitive). If a first shard `*-00001-of-NNNNN.gguf` is in scope, verify all `NNNNN` shards exist and choose shard 1; otherwise choose exactly one primary GGUF; zero/incomplete/ambiguous states return 409 and never guess. Canonicalize the selected path. — AC: unit tests cover exact, glob, folder, auxiliary, complete shard, incomplete shard, none, and ambiguous branches.

R4. llama-swap id + dedup: `llama_swap_id` = catalog `model_id`. `insert_model_entry` parses the full YAML, requires a top-level `models` mapping, and rejects existing structural keys (including quoted keys). — AC: re-registering the same id returns 409 and config is unchanged.

R5. Safe insertion preserves hand-tuned config text: build a 2-space-indented entry, insert immediately after the top-level block-style `models:` line (or convert `models: {}` to block style), re-parse the full YAML, and assert the id is under top-level `models`. Non-empty inline `models: {...}` returns a clear error instead of corrupting. — AC: tests cover block insertion, quoted-key dedup, `models:` inside a block scalar, `models: {}`, non-empty inline, invalid base config, and scalar/path escaping.

R6. Concurrency + rollback: add `AppState.llama_swap_config_lock` and acquire it in both the new register handler and existing `POST /api/llama-swap-config`. Under the register lock, read bytes + hash, build/validate the new text, re-read/hash to detect external changes, write a unique `.bak`, atomic write via a unique temp filename + rename, restart llama-swap, poll `{llama_swap_url}/v1/models` for up to about 30 seconds, and atomically restore from `.bak` on reload/health failure. — AC: no half-written or unvalidated config is left behind; response distinguishes `registered` from `registered_reload_failed_restored`.

R7. Generated entry is a documented baseline, not tuned:
   ```yaml
     "<id>":
       name: "<catalog name> (registered baseline)"
       cmd: |
         ${ls}
         --port ${PORT}
         ${common}
         --model '<resolved absolute path>'
   ```
   Paths are shell-quoted. If the canonical path is not under `/mnt` or `$HOME`, registration is allowed but the response includes a toolbox visibility warning. The success message says the baseline was verified only if `/v1/models` contains the id after reload. — AC: output parses and uses existing macros; UI message includes baseline/reload status.

R8. Dashboard: on completed `llama_cpp` download-job rows only, show "Register in llama-swap". The button posts the row's retained `{model_id, quant_pattern}`. It is absent for incomplete/running jobs and all non-llama_cpp backends. — AC: UI test asserts render gating and request body.

R9. Tests and gates: add Rust unit tests for resolver + inserter + destructive gate, a Node UI test, and run `cargo test --locked -- --test-threads=1`, `cargo clippy --all-targets -- -D warnings`, `bash scripts/check-html-js.sh`, and `node --test scripts/test-llama-swap-register-ui.cjs`. No validation may load a model or manually restart llama-swap.

## Technical plan

New module `src/llama_swap_register.rs` contains pure, testable cores:
- `resolve_primary_gguf(destination: &Path, quant_pattern: &str) -> Result<PathBuf, RegisterError>` implements the B1 resolver contract, including quant scoping, auxiliary rejection, shard completeness, canonicalization, and fail-on-ambiguity.
- `build_entry(id, name, model_path) -> String` emits the R7 baseline block with YAML-escaped id/name and shell-quoted model path.
- `insert_model_entry(config_text, id, entry_block) -> Result<String, RegisterError>` parses structurally, dedups structurally, inserts text after top-level `models:`, re-parses, and verifies placement.
- `RegisterError` maps to 400/409/500 for the handler.

`src/model_downloads.rs` adds `quant_pattern: Option<String>` to `ModelDownloadJob`, populates it from `StartDownloadRequest`, and exposes `effective_models_dir` + `resolve_catalog_entry` as `pub(crate)` so the server can re-validate against the catalog and on-disk layout rather than trusting the browser.

`src/server.rs` adds `POST /api/llama-swap/register-model`, the `/api/llama-swap/` destructive gate, and `AppState.llama_swap_config_lock`. The existing raw config writer also takes the lock. The register handler resolves the llama_cpp catalog entry, computes destination from `effective_models_dir + repo_basename`, resolves the GGUF, inserts the entry, writes backup + unique temp + rename under the lock, restarts llama-swap through the same systemd mechanism, polls `/v1/models`, and restores the backup on reload/health failure.

Dashboard flow:
```
completed llama_cpp download job (with quant_pattern)
  └─ Register in llama-swap
      └─ POST /api/llama-swap/register-model
          └─ resolve GGUF → insert validated entry → backup+atomic write → restart → /v1/models poll → success or restored failure
```

## Alternatives considered

- **Automatic registration on completion.** Rejected: silent mutation of a hand-tuned file; surprising; harder to attribute breakage.
- **Parse → mutate `serde_yaml::Value` → reserialize whole file.** Rejected: strips comments, macros formatting, and `|`/`>-` block scalars the user maintains; high risk of mangling the config. Text-insert-after-`models:` + full-parse-validate preserves formatting while guaranteeing validity.
- **Infer tuned flags from the model.** Rejected: reasoning/spec-draft/sampling/mmproj/context are expert-authored and not derivable; a baseline entry + explicit "tune it" note is honest.
- **Append at EOF.** Rejected: only valid if `models:` is the last top-level section; insert-after-`models:` is layout-independent.
- **ds4/halogen/r9v in llama-swap.** Out of scope: served by their own Server-Mode servers; the equivalent is routing to the serving-identity (separate feature).

## Detailed implementation

Files:
- `src/llama_swap_register.rs` — **new**. Implements `RegisterError`, `resolve_primary_gguf`, shard/auxiliary helpers, `build_entry`, `insert_model_entry`, `path_visibility_warning`, and unit tests.
- `src/lib.rs` — **modify**. Add `pub mod llama_swap_register;`.
- `src/model_downloads.rs` — **modify**. Add `quant_pattern` to `ModelDownloadJob`, populate it for download jobs, keep it `None` for PLE jobs, and make `effective_models_dir` / `resolve_catalog_entry` `pub(crate)`.
- `src/server.rs` — **modify**. Add typed request/response for registration, route `POST /api/llama-swap/register-model`, shared `llama_swap_config_lock`, destructive gate coverage, locked raw config writer, unique temp/backup helpers, restart helper refactor, `/v1/models` health poll, restore-on-failure behavior, and gate unit test.
- `src/daemon.rs` — **modify**. Initialize `llama_swap_config_lock` in `AppState`.
- `src/escalation/templates/main_dashboard.html` — **modify**. Render Register button only for completed llama_cpp download jobs with `quant_pattern`; add `registerInLlamaSwap(modelId, quantPattern)` POST + operator alert.
- `scripts/test-llama-swap-register-ui.cjs` — **new**. Node VM test for button gating and POST body.
- `docs/design/llama-swap-register-downloaded-gguf.md` — **modify**. Record v2 blocker resolutions and implementation log.

Implementation order:
1. Update this design doc to v2 and mark `approved-for-implementation`.
2. Add pure module + unit tests.
3. Persist `quant_pattern` on download jobs and expose the catalog/path helpers.
4. Add server state lock, route gate, handler, transaction helpers, restart health poll, and locked existing config writer.
5. Add dashboard button/action and UI test.
6. Run focused tests, full Rust tests, clippy, HTML/JS check, and Node UI test.

## Testing and evaluation
- Rust: `resolve_primary_gguf` (single/glob/folder/sharded/none/ambiguous via tempdir); `insert_model_entry` (insert-after-models, dedup 409, re-parse valid, invalid-base errors). `cargo test --locked -- --test-threads=1`.
- UI: `node --test` asserts the button appears for completed llama_cpp rows and posts the right body.
- Live (strix, via brainrouter): register against an existing on-disk GGUF dir into a **scratch copy** of the config first, confirm parse+reload, then the real one; verify `/api/models/llama-swap` lists it; `git`-safe backup retained.
- `cargo clippy`, `bash scripts/check-html-js.sh`.

## Security, privacy, reliability, and operations
- No secrets. Writes only the llama-swap config (already user-writable via `/api/llama-swap-config`). Backup + validate + atomic + restore bound the blast radius. Reload briefly restarts llama-swap (idle now). Body size capped; `model_id`/`quant_pattern` validated against the catalog (no arbitrary path injection — the path is derived from the catalog repo + resolved on disk, not user-supplied).

## Rollout, migration, and rollback
- Additive endpoint + UI; no migration. Rollback = revert commit; any written entry is a single removable `models:` key, and `.bak` retains the pre-change config. Deploy via the proven runbook to strix `master` + GitHub; back up `benchmarks.sqlite3` and the live `config.yaml` first.

## Risks and mitigations
- **Wrong `--model` path** → model fails to load. Mitigated by the deterministic R3 resolver that fails (409) on ambiguity rather than guessing, + the baseline entry being isolated (doesn't affect other models).
- **Config corruption** → mitigated by full-parse validate + `.bak` + atomic + restore-on-failure.
- **Reload disrupts in-flight inference** → mitigated by acting when idle; reload failure reported distinctly.

## Open questions
None blocking. (ds4-via-llama-swap-wrapper and Server-Mode routing are explicitly separate features.)

## Decision log
- Explicit action, not automatic (blast radius on a hand-tuned file).
- Text insert-after-`models:` + validate, not reserialize (preserve formatting).
- Fail on ambiguous GGUF, never guess.
- Backup + atomic + restore-on-failure mandatory (unattended deploy on a critical config).

## Referenced files
- `src/server.rs` — `/api/llama-swap-config` GET/POST (`:1307`/`:1327`, atomic write `:1349`), `/api/restart/llama-swap` (`:521`), route table, `AppState.llama_swap_config_path` (`:125`).
- `src/model_downloads.rs` — `build_download` llama_cpp layout (`:439-471`), `effective_models_dir` (`:263`).
- `src/toolbox_catalog` — catalog model lookup (repo/name/backend).
- `src/escalation/templates/main_dashboard.html` — `renderToolboxModelRow`, download-row rendering, `escHtml`.
- live `/api/llama-swap-config` (entry shape), `/api/models/llama-swap` (id list).

## Dory validation record

- **Critic review — Round 1** (rubber-duck sub-agent, isolated fresh context; doc v1; independently verified all cited code incl. `serde_yaml 0.9`, the config POST writer, `restart llama-swap` semantics, llama_cpp download layout, catalog lookup).
  - Verdict: **FAIL** (6 blocking, 3 important).
  - **B1** — GGUF resolver can pick the wrong quant/shard/incomplete model (all quants share one destination dir; scan isn't pattern-scoped; no shard-group completeness; could pick `mmproj`). Fix: pattern-scoped resolution + shard-group validation + reject aux GGUFs + canonicalize + fail-on-ambiguous-after-filtering.
  - **B2** — the UI cannot supply `quant_pattern`: `ModelDownloadJob` doesn't store it, llama_cpp is excluded from local presence, and completed llama_cpp downloads render no "complete" row/action. Fix: add `quant_pattern` to `ModelDownloadJob`, render registration on the completed-job row, send `job_id`; server re-validates. (Structural — needs download-subsystem change.)
  - **B3** — text-insert-after-`models:` + generic parse doesn't prove placement/dedup (a `models:` inside a block scalar; inline `models: {…}`; quoted-key dedup). Fix: parse first, operate on the parsed root `models` mapping, structural-equivalence check.
  - **B4** — not race-safe / incomplete rollback: needs a shared mutation lock across ALL llama-swap config writers (incl. the existing POST), external-change detection, unique temp files, post-restart health poll of `/v1/models`, atomic restore, and a resolved write-vs-reload-failure contract.
  - **B5** — the new `/api/llama-swap/*` endpoint bypasses the localhost/CSRF destructive-request gate (`server.rs:250-315`). Fix: add the path to the gate + 403 tests; use the hard-limited typed-JSON body pattern, not collect-then-check.
  - **B6** — the generated `--model` host path isn't proven valid inside `llama-server-toolbox`'s runtime (likely a container with specific mounts; existing entries use `/mnt/models`); plus no YAML quoting/escaping contract for path/id/name. Fix: verify the wrapper's mount/path contract, canonicalize, safely serialize scalars, quote the path; weaken "working baseline" to "registered baseline; load not auto-verified" unless a smoke test is added.
  - Important: I1 reload disruption not actually mitigated (nothing enforces idle) — decide 409-while-active/force/accept; I2 `effective_models_dir`/`resolve_catalog_entry` are private + `model_downloads.rs` omitted from the file plan, and catalog IDs are only backend-unique (R2 cross-backend lookup ambiguous — include `backend` in the request); I3 a pure string builder can't test backup/atomic/restart-failure/restore — need file-transaction + failure-injection + concurrency + security tests, and a named UI test.
  - Self-audit: relied on nothing beyond the doc + referenced files; treated unverified `hf`/`llama-server-toolbox` behavior as blockers, not assumptions.
  - Disposition: **deferred** (see Status). B2 (UX/download-tracking) and B6 (deployment path-namespace) need investigation + a user decision before a safe v2; the epic proceeds with the toolbox `--label` fix and Feature B first.

- **Readiness resolution — Round 2** (doc v2; 2026-09-27; coordinator-provided verified decisions plus repository re-read during implementation).
  - Verdict: **READY / approved-for-implementation**.
  - B1 resolved by quant-scoped `resolve_primary_gguf`, auxiliary rejection, shard completeness, canonicalization, and 409-on-ambiguity.
  - B2 resolved by storing `quant_pattern` on `ModelDownloadJob` and rendering registration on completed llama_cpp job rows; server still re-validates catalog + disk.
  - B3 resolved by structural YAML parse/dedup before text insertion and re-parse/placement assertion after insertion; inline `models: {}` handled safely.
  - B4 resolved by shared `llama_swap_config_lock`, backup, hash re-check, unique temp+rename, restart, `/v1/models` health poll, and restore-on-failure response.
  - B5 resolved by `/api/llama-swap/` destructive gate and typed, size-capped JSON body.
  - B6 resolved by verified toolbox path contract: `/mnt/models` and `$HOME` are visible; canonical paths outside `/mnt`/`$HOME` are allowed with a warning; baseline label does not claim load tuning.
  - I1 resolved by reloading only after a successful validated write and reporting reload/restored status distinctly.
  - I2 resolved by making `effective_models_dir` / catalog lookup `pub(crate)` and validating the request as llama_cpp-only.
  - I3 resolved by adding Rust resolver/inserter/gate tests and a named UI test.
  - Self-audit: no unresolved implementation question remains outside this v2 document and its referenced files.

## Implementation log

- 2026-09-27 — Implemented v2 design in `src/llama_swap_register.rs`, `src/model_downloads.rs`, `src/server.rs`, `src/daemon.rs`, `src/lib.rs`, dashboard HTML, and `scripts/test-llama-swap-register-ui.cjs`. Validation results are recorded in the final session report.

## Human approval
Skipped by explicit user delegation/autopilot for this implementation session. The coordinator supplied concrete, verified resolutions for every prior Dory blocker and instructed implementation. Recorded per HankNDory rule 7.
