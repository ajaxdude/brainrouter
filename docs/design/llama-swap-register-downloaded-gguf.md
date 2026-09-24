# Register a downloaded llama_cpp GGUF into llama-swap

## Status

- Workflow state: `critic-revisions-required` — Dory critic round 1 returned **FAIL (6 blocking)**. Two are structural/deployment blockers beyond config-safety: **B2** (brainrouter does not track completed llama_cpp downloads or their `quant_pattern` — `expected_files` empty + llama_cpp excluded from presence, so there is no "completed llama_cpp row" to attach the action to) and **B6** (the `${ls}`=`llama-server-toolbox` wrapper likely maps only specific host dirs into llama-server's runtime, so a model in `~/models`/`/home/papa/...` may not be loadable). **Deferred** pending (a) B6 deployment verification and (b) a B2 UX decision (add `quant_pattern` to `ModelDownloadJob` + a completed-jobs registration row, or a different entry point) — both benefit from the user's input. Prioritizing the toolbox `--label` fix and Feature B (routing) first: higher-value to the "llama-swap parity" goal and lower-risk (no critical-config mutation).
- Change classification: **standard** — new public endpoint + mutation of the critical, hand-tuned `llama-swap` `config.yaml` (gates all local inference) + dashboard UI. High blast radius on the config file → full rigor.
- Human review: user unavailable, delegated ("work autonomously, make good decisions"). Human review before implementation skipped by delegation per HankNDory rule 7; isolated Dory critic + mean review still run. **Because this mutates the live llama-swap config, every write is backup-guarded + YAML-validated + atomic + restore-on-failure**, so a wrong entry degrades to "one extra baseline model that fails to load / a clean abort," never corruption of existing models.

Revisions:
- v1 — 2026-09-24 — Initial draft.

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

R1. New `POST /api/llama-swap/register-model` with body `{ "model_id": "<catalog llama_cpp id>", "quant_pattern": "<the pattern the user downloaded>" }`. Returns `{status, llama_swap_id, model_path, message}` on success. — AC: curl with a completed download returns 200 + the appended id; the config gains exactly one entry.

R2. Backend restriction: only `llama_cpp`. A non-llama_cpp `model_id` → 400 with a message pointing at Server Mode. — AC: ds4/halogen id → 400.

R3. GGUF path resolution (deterministic, or fail): given `destination = effective_models_dir(LlamaCpp).join(repo_basename)`, resolve the primary `.gguf`:
   1. gather `*.gguf` under `destination` recursively;
   2. if none → 409 "no .gguf found (download incomplete?)";
   3. if any name matches `*-00001-of-*.gguf` → use it (first shard; llama-server finds the rest);
   4. else if `quant_pattern` ends in `.gguf` and a file whose name == basename(pattern) exists → use it;
   5. else if exactly one `.gguf` → use it;
   6. else → 409 "ambiguous: multiple GGUFs, none a first shard; register manually" (never guess). — AC: unit tests cover each branch with a tempdir.

R4. llama-swap id + dedup: `llama_swap_id` = the catalog `model_id` (already a stable slug). If a top-level key of that name already exists in the config's `models:` map → 409 "already registered". — AC: re-registering the same id → 409, config unchanged.

R5. Safe mutation: read config → verify `models:` key exists → insert the new entry text immediately after the `models:` line (2-space-indented key) → `serde_yaml::from_str::<Value>`-validate the full result → **write a `.bak` copy of the pre-change config** → atomic tmp+rename → if the post-write validate/reload path errors, restore from `.bak`. — AC: a forced-invalid entry never replaces the good config (restore verified in a test of the pure builder).

R6. Reload llama-swap after a successful write (reuse the `/api/restart/llama-swap` mechanism). Report reload failure distinctly from write success. — AC: after register, `/api/models/llama-swap` lists the new id.

R7. Generated entry is a documented baseline:
   ```yaml
     "<id>":
       name: "<catalog name> (registered from download)"
       cmd: |
         ${ls}
         --port ${PORT}
         ${common}
         --model <resolved absolute path>
   ```
   — AC: the entry parses and references the real macros; UI shows a "baseline — tune as needed" note.

R8. Dashboard: on a **completed llama_cpp** download row (Downloads tab), a "Register in llama-swap" button → POST → toast the result (id + baseline note, or the specific error). Absent for other backends / incomplete rows. — AC: button present only for completed llama_cpp entries; asserted by a UI test.

R9. Tests: Rust unit tests for `resolve_primary_gguf` (all R3 branches, tempdir) and the pure `insert_model_entry(config_text, id, entry) -> Result<String>` builder (inserts after `models:`, dedup rejects, output re-parses; invalid base config errors). A UI test asserting the button wiring. — AC: `cargo test --locked` + `node --test` pass.

## Technical plan

New module `src/llama_swap_register.rs` with pure, testable cores:
- `resolve_primary_gguf(destination: &Path, quant_pattern: &str) -> Result<PathBuf, RegisterError>` (R3).
- `build_entry(id, name, model_path) -> String` (R7 baseline block).
- `insert_model_entry(config_text: &str, id: &str, entry_block: &str) -> Result<String, RegisterError>` (R4 dedup + R5 insert-after-`models:` + full-parse validate).

`server.rs` handler `register_llama_swap_model(state, body)`:
1. parse body; resolve catalog entry for `model_id`; enforce llama_cpp (R2).
2. compute `destination`; `resolve_primary_gguf` (R3).
3. read config; `insert_model_entry` (dedup + build + validate) (R4/R5).
4. backup `.bak`; atomic write (mirror `:1349`); on write error restore (R5).
5. trigger llama-swap reload (R6); return result.

Dashboard: `renderToolboxModelRow` (llama_cpp, completed) gains the button → `registerInLlamaSwap(modelId, quantPattern)` → `POST /api/llama-swap/register-model` → toast.

```
Downloads(llama_cpp, complete) ──"Register in llama-swap"──▶ POST /api/llama-swap/register-model
   └─ resolve_primary_gguf → insert_model_entry(dedup+validate) → backup+atomic write → reload → /api/models/llama-swap shows it
```

## Alternatives considered

- **Automatic registration on completion.** Rejected: silent mutation of a hand-tuned file; surprising; harder to attribute breakage.
- **Parse → mutate `serde_yaml::Value` → reserialize whole file.** Rejected: strips comments, macros formatting, and `|`/`>-` block scalars the user maintains; high risk of mangling the config. Text-insert-after-`models:` + full-parse-validate preserves formatting while guaranteeing validity.
- **Infer tuned flags from the model.** Rejected: reasoning/spec-draft/sampling/mmproj/context are expert-authored and not derivable; a baseline entry + explicit "tune it" note is honest.
- **Append at EOF.** Rejected: only valid if `models:` is the last top-level section; insert-after-`models:` is layout-independent.
- **ds4/halogen/r9v in llama-swap.** Out of scope: served by their own Server-Mode servers; the equivalent is routing to the serving-identity (separate feature).

## Detailed implementation
(enumerated files: `src/llama_swap_register.rs` new; `src/server.rs` handler + route + reload reuse; `src/lib.rs` module decl; `src/escalation/templates/main_dashboard.html` button + `registerInLlamaSwap` + a UI test in `scripts/`; no schema/migration.) Ordered: pure cores + tests → handler/route → UI + test → docs.

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

## Human approval
Skipped by explicit user delegation ("work autonomously, make good decisions"); user unavailable. Recorded per HankNDory rule 7. Isolated Dory critic + mean review still performed. Safety design (backup/validate/atomic/restore) chosen specifically because deploy is unattended on a critical config.
