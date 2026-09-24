# Hugging Face CLI preflight + actionable download/serve diagnostics

## Status

- Workflow state: `implementation-complete` (Dory Round 5 READY; implemented + validated + mean-reviewed SOUND; human review skipped by explicit user delegation — see Human approval)
- Change classification: **standard** — adds a field to a public JSON API response (`/api/model-downloads/status`), adds a new public function in `model_downloads.rs`, and changes dashboard behavior. Blast radius is localized (one status endpoint, dashboard JS, docs) but it touches an API contract, so full rigor applies.
- Human review: the requesting user is unavailable and has explicitly delegated ("Work autonomously and make good decisions"). Per HankNDory core rule 7, human review before implementation is **skipped by explicit user delegation** and recorded as such in the Human approval section. One isolated Dory critic round is still run.

Revisions:
- v1 — 2026-09-24 — Initial draft (A: hf preflight + banner; C-light: hf-aware empty-state hint for ds4/halogen).
- v2 — 2026-09-24 — Dory critic round 1 fixes: (B1) redefined the signal as an honest PATH-candidate check (`found_on_path`), not a launchability guarantee, with exact edge-case behavior; (B2) added mandatory pure-function UI tests (`scripts/test-hf-preflight-ui.cjs`) + backend response-shape test, and stopped treating a skipped JS gate as a pass; (B3) single verified install command + single source of truth via structured `message`/`install_command` fields; (I4) preflight state assigned only on successful fetch (preserve last-known); (I5) softened API-compat claims + PRD schema; (nits) split_paths wording, r9v line/behavior, stale-comment updates.
- v3 — 2026-09-24 — Dory critic round 2 fixes: (B1) dropped the "necessary precondition / matches `Command::new` / by construction / downloads will fail" overclaim — the scan is a **conservative diagnostic** that skips empty PATH entries; softened banner/message wording accordingly. (B2) made the Rust plan buildable+deterministic: add `tempfile` dev-dep (already in Cargo.lock); pure `resolve_in(entries, name)` + `build_preflight(found)` cores with **no global `PATH` mutation**; owned `String` values; tests pass under plain `cargo test --locked`. (B3/B4) UI test now exercises `renderHfBanner` against a fake DOM, extracts the **real** `escHtml`, and adds static source assertions that both banner mounts exist, both loaders call `renderHfBanner`, and ds4/halogen use `serverModeEmptyModelHint`; backend test now asserts the **endpoint response value** (via a pure `status_response_value(presence, hf)` builder), not just `HfPreflight` serialization; registered in the README Tests block; runs under `node --test`. (B5) full halogen `<option>` assignment spelled out (single `escHtml`, no double-escape). (I) "single **runtime** source of truth" + README/PRD command-consistency assertion; API-compat claim limited to the two in-repo consumers + strict-schema caveat in Rollout. (nits) marker extraction called an adaptation; require unique sentinels asserted once.
- v4 — 2026-09-24 — Dory critic round 3 fixes: (B1) concrete `status_response_value(presence: Vec<ModelPresence>, hf) -> Value`; handler keeps `json_response` only (route already does `into_unsync`) — no double-wrap. (B2) enumerated `Cargo.lock`; refresh lock before any `--locked` check. (B3) scrubbed the last overclaim phrases (R3 banner, Alternatives, Risks, Decision log) to advisory wording. (B4) added pure `hfFromStatus` + `emptyModelOption` helpers so the UI test executes the real state-propagation (success/preserve-on-null) and both empty-branch outputs, not just source substrings. (B5) R10 moved to a Rust `include_str!` test bound to `HF_INSTALL_COMMAND`. (I) whole non-exec test `#[cfg(unix)]`-gated.
- v5 — 2026-09-24 — Dory readiness round 4 fixes: (B) corrected R10 `include_str!` paths to `../README.md`/`../PRD.md` (repo root is one level above `src/`) and reordered the implementation sequence so the README/PRD command edits precede the compiled-in R10 test. (I) replaced the ineffective empty-`PATH`-entry test with predicate-injected `resolve_in_with` + a spy predicate that proves empty entries are never probed (filesystem-/cwd-free).

## Problem

On the Strix Halo host, when the user tries to download a GGUF for a non-`llama_cpp` backend (ds4/halogen) or start a server for one, "a lot are disabled." The observed cause (verified live): brainrouter downloads models by running the `hf` (Hugging Face) CLI on the host (`Command::new("hf")`), but `hf` was not installed, so every download fails at process-exec time. Because the download-job registry is in-memory, the failure evidence disappears on every service restart (each deploy restarts the service), so the user never sees a persistent reason. Downstream, Server Mode only lists already-downloaded models, so with zero downloads every non-`llama_cpp` model dropdown is empty and the runtimes appear disabled. The user needs brainrouter to tell them, persistently and proactively, that the `hf` CLI is missing and how to install it — instead of silently failing.

## Goals and non-goals

Goals:
- Detect, proactively and persistently, whether the `hf` CLI brainrouter shells out to is resolvable on `PATH`, and expose that in `/api/model-downloads/status`.
- Show a clear, actionable banner in both the Downloads and Server Mode dashboard views when `hf` is missing, including the exact install command.
- Make the Server Mode empty-model hint for ds4/halogen distinguish "the Hugging Face CLI is missing" from "no models downloaded yet."

Non-goals:
- Running `hf download` inside a toolbox container (deferred option B; see Alternatives). Host `hf` works today.
- A full Server Mode inline-download redesign across all four backends (deferred; see Alternatives).
- Changing r9v's platform-gating or its already-differentiated empty states (deliberate existing design at `main_dashboard.html:808`; out of scope).
- Auto-installing `hf` from brainrouter. Installation stays a user/operator action; brainrouter only detects and instructs.
- Changing vllm (it has no host download step — HF-repo-served at start).

## Current system

Verified from repository files:

- `src/model_downloads.rs`
  - `hf_binary()` (~313) returns the constant `"hf"`; the binary is resolved via `PATH`.
  - The download worker (~985) runs `Command::new(hf_binary())` with args from `build_download(...)`, `stdout`/`stderr` piped, `kill_on_drop(true)`, and env from `hf_subprocess_env()`.
  - On spawn error (~991-1000) it already sets the job to `Failed` with `"failed to exec `hf`: {e} (is the `hf` CLI installed and on PATH?)"`. This is per-job only and is lost when the in-memory registry is cleared on restart.
  - r9v "Prepare PLE" uses `Command::new("podman")` (~1113) with its own good "is podman installed and on PATH?" message — not affected.
  - `local_presence_snapshot()` computes per-model on-disk presence (used by the status endpoint).
- `src/server.rs`
  - `model_downloads_status_response()` (~2851) returns `json!({ "models": presence })` from `local_presence_snapshot()`. The response is an ad-hoc `serde_json::json!` object, so adding a sibling key is additive and breaks no struct.
  - Route `GET /api/model-downloads/status` (~1082) calls it.
- `src/escalation/templates/main_dashboard.html`
  - Downloads view container `#toolbox-models-view` (580); intro paragraph then `#toolbox-models-list` (593). `loadToolboxModels()` fetches `/api/model-downloads/status` (~3163) into `toolboxModelPresence`.
  - Server Mode view container `#server-mode-view` (615); `loadServerMode()` (~3372) also fetches `/api/model-downloads/status` (~3382) into `toolboxModelPresence`.
  - ds4 empty hint: `renderServerModeModelOptions()` (~3463) → `'<option value="">No downloaded ds4 models — use Downloads to fetch one first</option>'`.
  - halogen empty hint: `renderServerModeHalogenBundleOptions()` (~3543) → analogous.
  - r9v empty hints: `renderServerModeR9vPackageOptions()` already distinguishes "PLE not prepared" vs "no downloads" — leave as is.
- Live facts (strix, `strix-halo` platform): toolboxes present for ds4(2)/vllm(2)/halogen(1); r9v(0) (r9700-only). Downloaded complete models: 0 for all non-llama_cpp backends. `hf` was absent from `PATH`; after `pip install --user "huggingface_hub[cli]"` it resolves at `/home/papa/.local/bin/hf` (on the service `PATH`), and a real ds4 download then transitions to `downloading` (exec succeeds). This confirms the diagnosis and the fix direction.

## Requirements and acceptance criteria

R1. `/api/model-downloads/status` includes an `hf` object with structured, single-source-of-truth fields: `{ "found_on_path": bool, "binary": string, "message": string|null, "install_command": string|null }`. `message` and `install_command` are non-null only when `found_on_path` is false. — AC: `curl .../status | jq .hf` shows the field; when `hf` is on PATH, `found_on_path=true`, `message=null`, `install_command=null`.

R2. The signal is a **conservative diagnostic**, not a launchability guarantee and not a strict model of the OS command search: it scans the process's own `PATH` (via `std::env::split_paths`, **skipping empty entries** so a stray `./hf` is never reported as "found") for an executable file named `hf_binary()` — no subprocess spawned. It is a best-effort "is `hf` visibly installed on PATH?" check; the authoritative check remains the per-job download spawn (which already reports a `Failed` job on exec error). Wording never claims the binary will run or that the check matches `Command::new` resolution exactly. — AC: unit tests over the pure `resolve_in` core cover each edge case in Detailed implementation §1.

R3. Downloads view shows a warning banner when `hf.found_on_path == false`, containing `message` and the `install_command`, and advising that `hf` must be installed and on the service PATH for model downloads to work. Banner absent when found. — AC: not-found → banner visible in `#toolbox-models-view`; found → element cleared.

R4. Server Mode view shows the same banner under the same condition. — AC: analogous in `#server-mode-view`.

R5. ds4 and halogen empty-model hints say "Install the Hugging Face CLI to download models (see banner above)" when `hf.found_on_path == false`, else keep the existing "No downloaded … — use Downloads to fetch one first". — AC: toggling the hf state changes the `<option>` text (asserted by unit test).

R6. No behavior change for the two in-repo consumers: the `models` field keeps its exact current semantics; the two dashboard consumers (which read only `.models`) are unaffected; the added `hf` key is additive. This is not a byte-for-byte-identical response and is **not** claimed to be universally backward-compatible — a strict/`deny_unknown_fields` external decoder of `/status` would need updating (called out in Rollout). — AC: existing tests pass; `models` array unchanged vs. before for the same on-disk state.

R7. README and PRD document the `hf` (Hugging Face CLI, `huggingface_hub[cli]`) prerequisite for model downloads, and the PRD documents the `/api/model-downloads/status` response schema including the additive `hf` object. — AC: grep finds the prerequisite in both; PRD shows the `hf` schema.

R8. **Mandatory** automated tests for the missing-`hf` behavior, exercising real wiring (not just static string checks). `scripts/test-hf-preflight-ui.cjs` (run via `node --test`, the repo's canonical UI-test runner; missing `node` naturally fails the run):
   - Extracts the **real** `escHtml` plus the marker-delimited pure-helper block (`hfFromStatus`, `hfBannerHtml`, `serverModeEmptyModelHint`, `emptyModelOption`, `renderHfBanner`) from `main_dashboard.html`, asserting each sentinel occurs exactly once, and evaluates them in a `vm` context with a fake `document` + mutable `toolboxHfPreflight`.
   - `hfFromStatus`: success propagates `statusRes.hf`; **failed fetch (`null`) preserves the previous value**; success-without-hf → `null`.
   - `hfBannerHtml`: non-empty + contains the install command + `⚠` when `found_on_path:false`; `''` when found or `null`; untrusted `message`/`install_command` HTML-escaped (assert `&lt;`/`&gt;` appear and raw `<script>`/`<b>` do not) via the real `escHtml`.
   - `serverModeEmptyModelHint`/`emptyModelOption`: hf-missing → `/Hugging Face CLI/`; found → exact legacy strings for both `'ds4 models'` and `'halogen bundles'`, wrapped once in `<option value="">…</option>`.
   - `renderHfBanner(elId)` against fake DOM elements: sets banner HTML when `found_on_path:false` and **clears** it when found.
   - Static source assertions (catch mis-wiring): raw HTML contains both mount `<div id="toolbox-hf-banner"`/`<div id="server-mode-hf-banner"`; both `renderHfBanner('toolbox-hf-banner')`/`renderHfBanner('server-mode-hf-banner')` call sites; both ds4/halogen empty branches call `emptyModelOption('ds4 models'`/`emptyModelOption('halogen bundles'`; and both loaders assign via `hfFromStatus(`.
   - Registered in the README "Tests" block.
   Plus a **backend endpoint-shape** test (Rust) over the pure `status_response_value(presence, hf)` builder used by the handler: asserts top-level `models` is the passed array (semantics unchanged) and top-level `hf` has exactly `{found_on_path, binary, message, install_command}` with correct nullability for found vs not-found. — AC: `cargo test --locked` and `node --test scripts/test-hf-preflight-ui.cjs` both pass.

R9. Source comments that currently describe `/api/model-downloads/status` as local-presence-only are updated to mention the added preflight responsibility (`server.rs` endpoint comment; `main_dashboard.html` state comment). — AC: both comments mention `hf` preflight.

R10. The exact install command lives in one runtime source of truth (`HF_INSTALL_COMMAND` in `model_downloads.rs`, served via the API) and is bound to the docs by a **Rust** test using `include_str!("../README.md")` and `include_str!("../PRD.md")` asserting both `.contains(HF_INSTALL_COMMAND)`. — AC: changing `HF_INSTALL_COMMAND` alone fails `cargo test` until both docs are updated.

## Technical plan

Two layers:

1. **Backend preflight (A).** Add `hf_preflight()` to `model_downloads.rs` returning a serializable struct `HfPreflight { found_on_path, binary, message, install_command }`. It reads `hf_binary()` and checks whether that name resolves to an executable file on the process's own `PATH` via a pure filesystem scan (split `$PATH`, join binary, check is-file + executable bit; skip empty entries; unset `PATH` → not found). This is an honest PATH-candidate signal, not a launchability guarantee. `server.rs` adds `"hf": crate::model_downloads::hf_preflight()` to the status JSON.

2. **Dashboard surfacing (A + C-light).** `loadToolboxModels()` and `loadServerMode()` already fetch the status; capture `status.hf` into a global `toolboxHfPreflight`. Add two banner mount `<div>`s (one per view) populated by a shared `renderHfBanner(elId)` helper. Add a shared `serverModeEmptyModelHint(backendLabel)` helper returning the hf-aware vs default option text, used by the ds4 and halogen render functions.

```
/api/model-downloads/status
  { models:[...], hf:{found_on_path,binary,message,install_command} }
         │
         ├── loadToolboxModels()  → toolboxHfPreflight → renderHfBanner('toolbox-hf-banner')
         └── loadServerMode()     → toolboxHfPreflight → renderHfBanner('server-mode-hf-banner')
                                                        → ds4/halogen empty hint via serverModeEmptyModelHint()
```

## Architecture and flows

Status poll → JSON now carries `hf`. Dashboard render reads it: if unavailable, both banners show the install command; ds4/halogen empty dropdowns point at the banner instead of only "use Downloads." No new endpoints, no new routes, no DB, no persisted state. Preflight is recomputed each status call (cheap PATH scan).

## Alternatives considered

- **B — run `hf download` inside the selected toolbox container (podman).** Benefit: removes the host `hf` dependency; matches "brainrouter as a webui to cockpit." Cost/risk: container UID/GID mapping, bind-mounting the host models dir, passing `HF_TOKEN`/cache dirs, and reconciling the destination path between container and host; meaningful new failure modes. Rejected for now: host `hf` works, and this is a larger design deserving its own doc. Recorded as future work.
- **Full Server Mode inline-download (heavy C).** Benefit: download→serve in one place. Cost/risk: four inconsistent per-backend render paths, duplicated download logic in Server Mode, higher regression surface while the user is away. Deferred; the banner + hf-aware hint resolve the reported confusion at far lower risk.
- **Run `hf --version` as the preflight probe.** Benefit: proves the binary actually runs. Cost: spawns a subprocess on every status poll; slower and noisier. Rejected in favor of a PATH existence+exec-bit scan — an advisory "is `hf` visibly on PATH?" signal, not a claim that it will launch.
- **A new `/api/model-downloads/preflight` endpoint.** Rejected: the dashboard already polls `/status`; a sibling key avoids a second fetch and an extra route.

## Detailed implementation

1. `src/model_downloads.rs` — **modify**
   - Add `#[derive(Debug, Clone, Serialize)] pub struct HfPreflight { pub found_on_path: bool, pub binary: String, pub message: Option<String>, pub install_command: Option<String> }` (`Serialize` already imported at `model_downloads.rs:32`).
   - Add `const HF_INSTALL_COMMAND: &str = "python3 -m pip install --user -U \"huggingface_hub[cli]\"";` — the exact command **verified on the strix service account** (installs to `~/.local/bin`, on the service PATH). Single runtime source of truth.
   - Factor into pure, testable cores (no global-env mutation in any test):
     - `fn is_executable_file(p: &Path) -> bool`: `std::fs::metadata(p)` (**follows symlinks** — executable symlink target resolves; dangling symlink errors → `false`), require `md.is_file()`, and `#[cfg(unix)] { use std::os::unix::fs::PermissionsExt; md.permissions().mode() & 0o111 != 0 }` (documented heuristic: *any* execute bit, not the effective-user bit); `#[cfg(not(unix))]` → `md.is_file()`.
     - `fn resolve_in_with<I, F>(entries: I, name: &str, is_exec: F) -> bool where I: IntoIterator<Item = PathBuf>, F: Fn(&Path) -> bool`: if `name` contains `'/'`, return `is_exec(Path::new(name))`; else for each entry, **skip empty `PathBuf`s (`entry.as_os_str().is_empty()`)**, and return `true` on the first `is_exec(&entry.join(name))`, else `false`. (Predicate injected so the empty-skip invariant is testable without touching the filesystem or cwd.)
     - `fn resolve_in<I: IntoIterator<Item = PathBuf>>(entries: I, name: &str) -> bool { resolve_in_with(entries, name, is_executable_file) }`.
     - `fn resolve_on_path(name: &str) -> bool`: read `std::env::var_os("PATH")`; if unset → `false`; else `resolve_in(std::env::split_paths(&path), name)`.
     - `fn build_preflight(found_on_path: bool) -> HfPreflight`: on `false`, `message = Some("The Hugging Face CLI (`hf`) was not found on the brainrouter service's PATH. Model downloads run `hf`, so they need it installed and on the service PATH to work.".to_owned())` and `install_command = Some(HF_INSTALL_COMMAND.to_owned())`; on `true`, both `None`; `binary = hf_binary().to_owned()`.
     - `pub fn hf_preflight() -> HfPreflight { build_preflight(resolve_on_path(hf_binary())) }`.
   - Tests (`#[cfg(test)]`, deterministic, no `PATH`/cwd mutation — pass under plain `cargo test --locked`):
     - **Empty-skip invariant (race-free, filesystem-free):** call `resolve_in_with(vec![PathBuf::new(), PathBuf::from("/real")], "hf", spy)` where `spy` is a closure over a `RefCell<Vec<PathBuf>>` (interior mutability, since the bound is `Fn(&Path) -> bool`) that records every path it is asked about and returns `false`. Assert the recorded calls are exactly `["/real/hf"]` — i.e. the empty entry was **never** probed (a non-skipping impl would first probe the relative `"hf"` from `PathBuf::new().join("hf")`). This proves R2's skip invariant without cwd/fs.
     - **Exec-bit cases, gated entirely behind `#[cfg(unix)]`** (not just the `chmod`): via `tempfile::TempDir`, `resolve_in([dir], "hf")` with `hf` chmod `0o755` → `true`; `0o644` → `false`; direct `/`-path arg → checks that path.
     - **Cross-platform:** `resolve_in([empty_dir], "hf")` (no `hf` present) → `false`.
     - `build_preflight(true)` → `found_on_path:true`, `message`/`install_command` `None`, `binary=="hf"`; `build_preflight(false)` → non-null `message`+`install_command`, `install_command == HF_INSTALL_COMMAND`.
     - R10: `#[test]` asserting `include_str!("../README.md")` and `include_str!("../PRD.md")` (repo root is one level up from `src/model_downloads.rs`) both `.contains(HF_INSTALL_COMMAND)`. **Prerequisite:** the README/PRD command additions (step 6) must land **before** this test is first run — see the reordered implementation sequence.
   - Rationale: recomputed each poll (persistent), pure cores make it deterministic and dependency-light; honest conservative-diagnostic framing.

2. `Cargo.toml` — **modify**
   - Add `tempfile = "3"` under `[dev-dependencies]`. `tempfile` is already resolved in `Cargo.lock` (transitively, e.g. via `native-tls`), so **no new version is selected**; but adding a *direct* dev-dependency edge changes the recorded `brainrouter` dependency list in `Cargo.lock`.

2b. `Cargo.lock` — **modify (regenerate)**
   - After editing `Cargo.toml`, refresh the lockfile with `cargo build` (or `cargo generate-lockfile`) so the `brainrouter` package records the new `tempfile` edge, then commit `Cargo.lock`. Do this **before** any `--locked` check — otherwise the first `cargo test --locked`/`cargo build --release` on strix rejects the stale lock. (No new version selection is expected; a fresh environment may still perform a metadata/fetch of the already-locked version.)

3. `src/server.rs` — **modify**
   - Add a pure builder next to the handler:
     ```rust
     fn status_response_value(
         presence: Vec<crate::model_downloads::ModelPresence>,
         hf: crate::model_downloads::HfPreflight,
     ) -> serde_json::Value {
         serde_json::json!({ "models": presence, "hf": hf })
     }
     ```
     (`ModelPresence` is the element type of `local_presence_snapshot()`'s `Ok(...)` — confirm the exact public type name/path when implementing; it is the same value currently serialized as `"models"`.)
   - In `model_downloads_status_response()` (~2851), the success arm becomes `json_response(StatusCode::OK, &status_response_value(presence, crate::model_downloads::hf_preflight()))`. **Do not** add `into_unsync` here — the handler returns `Response<Full<Bytes>>` and the route at ~1083 already wraps it with `into_unsync(resp)`. Only the `json!(...)` argument changes vs. today.
   - Update the endpoint's preceding comment (~2848): it now also returns the `hf` PATH-preflight, not just the local-presence sweep.
   - Test: `status_response_value(sample_presence, build_preflight(false))` → parsed `Value` has `["models"]` equal to `sample_presence` (semantics unchanged) and `["hf"]` with exactly `found_on_path/binary/message/install_command` and non-null message/command; with `build_preflight(true)` → `["hf"]["message"]` and `["install_command"]` are `null`. Expose `build_preflight` as `pub(crate)` (or construct an `HfPreflight` literal in the test) so this is deterministic without touching global `PATH`.
   - Interfaces affected: additive `hf` key on `/api/model-downloads/status`.

4. `src/escalation/templates/main_dashboard.html` — **modify**
   - HTML: add `<div id="toolbox-hf-banner" style="margin-bottom:12px"></div>` immediately before `#toolbox-models-list` (~593); add `<div id="server-mode-hf-banner" style="margin-bottom:12px"></div>` immediately after the Server Mode intro paragraph, before the first backend panel (~621).
   - JS state: add `let toolboxHfPreflight = null;`. Update the state comment (~3143) to note `/status` now also carries `hf` preflight.
   - **Assign only on success, via a testable pure helper.** Add `hfFromStatus(statusRes, prev)` to the marker block (below), then in `loadToolboxModels()` (inside the existing `if (statusRes)` block at ~3167) and `loadServerMode()` (~3390) set `toolboxHfPreflight = hfFromStatus(statusRes, toolboxHfPreflight);`. (`hfFromStatus` returns `prev` when `statusRes` is falsy — preserving the last-known warning across a transient `safeFetch`→`null` — and `statusRes.hf || null` on success. Placing the logic in a pure helper lets the UI test assert the exact success-only/preserve-on-failure behavior without a full loader harness.)
   - Marker-delimited helper block (so the test extracts exactly these) — depends only on `escHtml` (real one at ~1339) plus globals `document`/`toolboxHfPreflight`:
     ```
     // === hf-preflight pure helpers (unit-tested by scripts/test-hf-preflight-ui.cjs) ===
     function hfFromStatus(statusRes, prev) {
       // success-only: keep last-known preflight when the status fetch failed (statusRes falsy)
       return statusRes ? (statusRes.hf || null) : prev;
     }
     function hfBannerHtml(hf) {
       if (!hf || hf.found_on_path !== false) return '';
       const msg = escHtml(hf.message || 'Hugging Face CLI (hf) not found on PATH.');
       const cmd = escHtml(hf.install_command || '');
       return '<div style="border:1px solid var(--amber);background:rgba(240,180,60,0.12);'
         + 'color:var(--text);border-radius:8px;padding:10px 12px;font-size:12px">'
         + '⚠ ' + msg + (cmd ? ' <code style="font-family:\'JetBrains Mono\',monospace">' + cmd + '</code>' : '')
         + '</div>';
     }
     function serverModeEmptyModelHint(kindLabel, hf) {
       return (hf && hf.found_on_path === false)
         ? 'Install the Hugging Face CLI to download models (see banner above)'
         : 'No downloaded ' + kindLabel + ' — use Downloads to fetch one first';
     }
     function emptyModelOption(kindLabel, hf) {
       // single source for the ds4/halogen empty-dropdown <option>; escapes once (plain-text hint)
       return '<option value="">' + escHtml(serverModeEmptyModelHint(kindLabel, hf)) + '</option>';
     }
     function renderHfBanner(elId) {
       const el = document.getElementById(elId);
       if (!el) return;
       el.innerHTML = hfBannerHtml(toolboxHfPreflight);
     }
     // === end hf-preflight pure helpers ===
     ```
     `hfBannerHtml` escapes both `message` and `install_command` (single source: `hf.install_command` from the backend, rendered once).
   - Call `renderHfBanner('toolbox-hf-banner')` in the `loadToolboxModels()` render path and `renderHfBanner('server-mode-hf-banner')` in `loadServerMode()`.
   - Wire the shared `emptyModelOption` into the two empty branches (identical shape; single `escHtml`, no double-escape):
     - ds4 `renderServerModeModelOptions` (~3475): `el.innerHTML = emptyModelOption('ds4 models', toolboxHfPreflight);`
     - halogen `renderServerModeHalogenBundleOptions` (~3555): `el.innerHTML = emptyModelOption('halogen bundles', toolboxHfPreflight);`
     - (r9v left as-is per non-goals.)
   - Constraint: `bash scripts/check-html-js.sh` passes (parse-only gate; the behavior gate is R8's tests).

5. `scripts/test-hf-preflight-ui.cjs` — **create**
   - **Adapt** the `scripts/test-benchmark-ui.cjs` pattern (`node:test` + `node:vm` + `fs.readFileSync` of the template). Extract **two name/marker-delimited regions**, asserting each occurs exactly once:
     - the real `escHtml` (regex `/function escHtml\(s\)\s*\{[\s\S]*?\n\}/`), and
     - the block between `// === hf-preflight pure helpers ...` and `// === end hf-preflight pure helpers ===`.
   - Build a minimal fake `document.getElementById(id)` → object with a settable `innerHTML`, seeded with `toolbox-hf-banner`, `server-mode-hf-banner`; a mutable `toolboxHfPreflight` in the vm context. Eval `escHtml` + the block.
   - Executable assertions (real wiring logic, not just strings):
     - `hfFromStatus({hf:{found_on_path:false}}, null)` → the object; `hfFromStatus(null, PREV)` → `PREV` (preserve on failed fetch); `hfFromStatus({}, PREV)` → `null` (success but no hf).
     - `hfBannerHtml`: non-empty + contains command + `⚠` when `found_on_path:false`; `''` when found/`null`; untrusted `message:'<script>x'`/`install_command:'<b>'` → escaped (`&lt;`/`&gt;` present, raw `<script>`/`<b>` absent) via the REAL `escHtml`.
     - `serverModeEmptyModelHint`/`emptyModelOption` for `'ds4 models'` and `'halogen bundles'` under both states (found → exact legacy string inside the `<option>`).
     - `renderHfBanner('toolbox-hf-banner')`: set `toolboxHfPreflight={found_on_path:false,...}` → element `innerHTML` non-empty; set `={found_on_path:true}` → `innerHTML===''` (clears).
   - Static source assertions on the raw HTML (catch mis-wiring): both `<div id="toolbox-hf-banner"` and `<div id="server-mode-hf-banner"` mount markup present; both `renderHfBanner('toolbox-hf-banner')` and `renderHfBanner('server-mode-hf-banner')` call sites present; both ds4/halogen empty branches call `emptyModelOption('ds4 models'`/`emptyModelOption('halogen bundles'`; and `hfFromStatus(` appears in both loader assignments.
   - Invoked with `node --test`; missing `node` fails the run (per R8).

6. `README.md` and `PRD.md` — **modify**
   - README: add a prerequisite line — model downloads require the Hugging Face CLI (`python3 -m pip install --user -U "huggingface_hub[cli]"`), resolvable on the brainrouter service's PATH; brainrouter shows a banner if it is missing. Add `node --test scripts/test-hf-preflight-ui.cjs` to the "Tests" block (~1388-1394).
   - PRD: document the `/api/model-downloads/status` response schema, including the additive `hf` object (`found_on_path`, `binary`, `message`, `install_command`), the same `hf` prerequisite, and the exact install command string.
   - **R10 enforcement (Rust):** add a `#[test]` in `model_downloads.rs` asserting `include_str!("../README.md")` and `include_str!("../PRD.md")` (repo root is one level above `src/`) both `.contains(HF_INSTALL_COMMAND)`. This binds the docs directly to the Rust constant, so changing `HF_INSTALL_COMMAND` alone fails the build until both docs are updated. Because this test compiles the docs in, the README/PRD command additions above must be in place **before** this test is first compiled/run (see order).

Implementation order (docs land before the R10 test compiles): (1) `Cargo.toml` dev-dep + `cargo build` to refresh & commit `Cargo.lock`. (2) `README.md` + `PRD.md` edits: add the exact `python3 -m pip install --user -U "huggingface_hub[cli]"` prerequisite line, the PRD `/status` schema, and the `node --test scripts/test-hf-preflight-ui.cjs` Tests-block entry — do this first so the R10 `include_str!` test can pass. (3) backend: `HF_INSTALL_COMMAND` + preflight cores + unit tests (incl. R10 `include_str!` test) → `cargo test --locked`/`clippy`. (4) `status_response_value` + endpoint-shape test + endpoint comment. (5) dashboard HTML+JS (marker helpers, banner mounts, `hfFromStatus` assignment in both loaders, `renderHfBanner` calls, ds4/halogen `emptyModelOption` wiring, state comment). (6) `scripts/test-hf-preflight-ui.cjs` → run `node --test …` + `check-html-js.sh`. Checkpoint after (3) and (5) with `cargo build --release` on strix (Linux compile gate that caught the P2a `#[non_exhaustive]`/cfg break).

## Testing and evaluation

- Rust unit (deterministic, no `PATH` mutation; plain `cargo test --locked`): `resolve_in` edge cases via `tempfile::TempDir` (Unix-gated exec cases; cross-platform missing/empty → false); `build_preflight(true/false)` field/nullability + `install_command == HF_INSTALL_COMMAND`; R10 `include_str!` README/PRD consistency.
- Rust endpoint-shape: `status_response_value(sample_presence, build_preflight(false/true))` → parsed `Value` has `models` == input and `hf` with exactly the four keys + correct nullability.
- **Mandatory UI test (R8):** `node --test scripts/test-hf-preflight-ui.cjs` — executable tests of `hfFromStatus` (success/preserve-on-null), real `escHtml` + `renderHfBanner` fake-DOM set/clear, `hfBannerHtml`/`emptyModelOption` outputs+escaping, plus static wiring assertions (mount markup, banner calls, ds4/halogen `emptyModelOption` sites, both loaders use `hfFromStatus`). Missing `node` fails the run.
- Lint/build: `cargo clippy --locked --all-targets`; `cargo build --release` on strix (Linux compile gate); `bash scripts/check-html-js.sh` (parse-only).
- Manual on strix (hf now installed): `curl /api/model-downloads/status | jq .hf` → `found_on_path:true`, `message:null`, `install_command:null`; Downloads + Server Mode show **no** banner; ds4/halogen empty hints unchanged (still "use Downloads" since hf present but 0 downloaded). The missing-hf path is proven by the UI + endpoint-shape unit tests (synthetic state), not by uninstalling hf on the live host.

## Security, privacy, reliability, and operations

- No secrets, no new external calls, no new writes. Preflight is a read-only filesystem scan of the process's own `PATH` dirs. `message`/`install_command` are static strings. Failure to read `PATH` → treat as `found_on_path:false` with the message/command (fail toward telling the user to install), which is safe. Rendered `message`/`install_command` are HTML-escaped in the banner.

## Rollout, migration, and rollback

- Additive; no migration. Rollback = revert the commit. The in-memory job registry and all existing behavior are untouched.
- **API compatibility caveat:** `/api/model-downloads/status` gains a top-level `hf` key. The two in-repo consumers read only named keys and are unaffected, but this is not universally backward-compatible — any external client decoding the response with a strict/`deny_unknown_fields` schema must add `hf` (or ignore unknowns). No such external consumer is known in-repo.
- Deploy via the proven runbook: push to strix remote branch → `git checkout master && git merge --ff-only … && cargo build --release && systemctl --user restart brainrouter` → `curl :9099/health` → `git push origin master` + branch from strix. Back up `benchmarks.sqlite3` first.

## Risks and mitigations

- **Portability of the exec-bit check** (Mac worktree vs Linux target). Mitigation: `#[cfg(unix)]` gate; build-gate on strix. This exact class of bug (`#[non_exhaustive]`/cfg) bit P2a and was caught by the Linux build — repeat that gate.
- **False "missing" if the service PATH differs from an interactive shell.** Mitigation: preflight reads the process's own `PATH` (the same env `Command::new` inherits), so it reflects the service's actual view of PATH; it remains advisory, and the per-job spawn is authoritative.
- **Banner noise for vllm-only users** (vllm needs no host hf). Accepted: the banner states downloads need hf; vllm users who never download simply see an accurate note. Could later scope the banner per active backend tab; not now.

## Open questions

None blocking. (Per-tab banner scoping and options B/heavy-C are explicitly deferred, not open.)

## Decision log

- Chose a sibling key on `/status` over a new endpoint (avoid a second fetch/route).
- **Chose an honest, advisory PATH-candidate check (`found_on_path`) over an "available"/launchable claim** (Dory R1): a filesystem scan cannot prove the binary runs for the service user (effective perms, `noexec`, bad shebang), so the field is documented as advisory/best-effort and the per-job spawn error stays authoritative. Rejected a cached `hf --version` probe as more complex/noisy than the reported problem (a *missing* binary) warrants.
- Chose PATH scan over `hf --version` subprocess (cheaper, no per-poll process spawn).
- **Single source of truth for install guidance** (Dory B3): backend returns structured `message` + `install_command`; the renderer composes them once; one verified command constant `HF_INSTALL_COMMAND` (`python3 -m pip install --user -U "huggingface_hub[cli]"`, verified on the strix service account) used by backend and docs.
- **Preflight state assigned only on successful status fetch** (Dory I4): a transient failure preserves the last-known warning instead of clearing it.
- **Mandatory UI behavior test** (Dory B2): `check-html-js.sh` only parses and self-skips without a runtime, so it is insufficient; pure string helpers are unit-tested and a missing JS runtime is a failed run.
- **Conservative-diagnostic framing** (Dory R2-B1): dropped "necessary precondition / matches `Command::new`"; the scan skips empty PATH entries deliberately (never advertise availability from cwd) and is documented as best-effort, with the per-job spawn authoritative.
- **Buildable/deterministic tests** (Dory R2-B2): `tempfile` dev-dep (already in lock); pure `resolve_in`/`build_preflight` cores tested without mutating global `PATH`; owned `String` values.
- **Endpoint-shape tested via a pure `status_response_value` builder** (Dory R2-B4), not just `HfPreflight` serialization; UI test exercises `renderHfBanner` + real `escHtml` + static wiring assertions (Dory R2-B3); full halogen assignment specified (Dory R2-B5).
- **Docs kept in sync by test** (Dory R2-I): README/PRD must contain the exact `HF_INSTALL_COMMAND`; "single **runtime** source of truth."
- Deferred in-container `hf` (B) and full inline-download (heavy C) as separate, riskier designs; host `hf` unblocks the user now.
- Left r9v gating/empty-states untouched — the r9v toolbox is *deliberately* not platform-filtered (`main_dashboard.html:806-813`) and its empty states are already differentiated; the shared banner covers its hf story.
- Human review skipped by explicit user delegation ("Work autonomously and make good decisions"); isolated Dory critic rounds still run.

## Referenced files

- `src/model_downloads.rs` — `hf_binary()` (~312-314), `Serialize` import (~32), download exec (~985-1000), r9v podman (~1113); home of `hf_preflight()`/`resolve_on_path`/`is_executable_file`.
- `src/server.rs` — `model_downloads_status_response()` (~2851-2858) + its comment (~2848), route (~1082-1085).
- `src/escalation/templates/main_dashboard.html` — Downloads view (~580) + list anchor (~593), Server Mode view (~615) + intro (~619-626); state comment (~3143); `escHtml` (~1339); `safeFetch` null-on-failure (~976); `loadToolboxModels` (~3160-3168), `loadServerMode` (~3372-3390); ds4 empty hint `renderServerModeModelOptions` (~3463-3476), halogen `renderServerModeHalogenBundleOptions` (~3543-3556); r9v `renderServerModeR9vPackageOptions` (~3753-3775, deliberately left as-is); r9v deliberate platform-filter bypass note (~806-813).
- `Cargo.toml` / `Cargo.lock` — add `tempfile = "3"` dev-dep and regenerate the lock (already resolved transitively in `Cargo.lock`; adding the direct edge updates the `brainrouter` package entry, so refresh before `--locked`).
- `scripts/test-benchmark-ui.cjs` — model for the new pure-function UI test (node:test + node:vm; extracts a whole `<script>` block — the new test **adapts** this to extract the real `escHtml` + the marker-delimited helper block).
- `scripts/test-hf-preflight-ui.cjs` — **new**; the mandatory R8 UI behavior test.
- `scripts/check-html-js.sh` — JS parse gate (self-skips without a runtime; not the behavior gate).
- `README.md`, `PRD.md` — prerequisite + `/status` schema documentation.

## Dory validation record

- **Critic review — Round 1** (rubber-duck sub-agent, isolated fresh context; doc v1)
  - Inputs: design doc v1 + all referenced files (verified line numbers/behavior independently).
  - Verdict: **FAIL** (3 blocking, 2 important, 2 nits).
  - Blocking: (1) `available` overclaimed launchability vs. a PATH+exec-bit scan; (2) no real test for the missing-hf UI behavior — `check-html-js.sh` only parses and self-skips without a runtime; (3) install command unresolved (`-U` vs verified `--user`) and duplicated across `hint` + renderer.
  - Important: (4) prescribed unconditional `toolboxHfPreflight` assignment clears the warning on a transient fetch failure; (5) "no consumer breaks"/"exactly as before" overstated for a public API + `/status` schema undocumented.
  - Nits: (6) `split_paths` mis-described as `MAIN_SEPARATOR`-aware; r9v renderer line ref wrong; "r9v platform-gating" mischaracterized (it is a deliberate bypass); (7) stale endpoint/state comments.
  - Document changes: v2 addresses all seven — field renamed to `found_on_path` with exact edge-case semantics + honest necessary-not-sufficient framing; structured `message`/`install_command` with one verified command constant; success-only state assignment; mandatory `scripts/test-hf-preflight-ui.cjs` + backend shape test with runtime-required gating; softened compat claims + PRD schema; corrected wording/line refs; added comment-update tasks.
  - Isolation: sub-agent-isolated (fresh context), not a separate OS session — recorded as sub-agent-isolated, not fully certified.
  - Remaining non-blocking: per-active-tab banner scoping (accepted, deferred); effective-user exec permission not checked (documented heuristic; per-job spawn authoritative).
- **Critic review — Round 2** (rubber-duck sub-agent, isolated fresh context; doc v2; independently ran `node --test scripts/test-benchmark-ui.cjs` = 38 pass)
  - Verdict: **FAIL** (5 blocking, 3 important, 2 nits).
  - Blocking: (1) `found_on_path` still overclaimed ("necessary precondition / matches `Command::new`") while skipping empty PATH entries diverges from the real unmodified-PATH spawn; (2) Rust plan not buildable/deterministic — `tempfile` not a dev-dep, `Option<String>` assigned `&str` literals, `hf_preflight()` shape tests read global `PATH`, and "already `--test-threads=1`" contradicted the canonical `cargo test --locked`; (3) UI test asserted only the string helpers, not `renderHfBanner`/mounts/loader calls/actual ds4+halogen branches; (4) "backend response-shape test" tested only `HfPreflight` serialization, not the endpoint's top-level `{models,hf}`; (5) halogen wiring under-specified vs. ds4 (risk of assigning plain text / dropping `escHtml`).
  - Important: real vs. faked `escHtml` mismatch (use the real one); "single source of truth" true only at runtime, docs still duplicate; API-compat wording still too broad.
  - Nits: benchmark test extracts a whole `<script>` (call the marker approach an adaptation); require unique sentinels asserted once.
  - Document changes: v3 addresses all — conservative-diagnostic framing (dropped overclaims); `tempfile` dev-dep + pure `resolve_in`/`build_preflight` cores + owned `String`s + deterministic tests; endpoint-shape via a pure `status_response_value` builder + test; UI test exercises `renderHfBanner` + real `escHtml` + static wiring assertions; full halogen `<option>` assignment; "single **runtime** source of truth" + README/PRD command-consistency test (R10); API-compat scoped to in-repo consumers + strict-schema caveat in Rollout; marker/adaptation + unique-sentinel wording.
  - Isolation: sub-agent-isolated (fresh context), not a separate OS session — recorded as sub-agent-isolated, not fully certified.
  - Remaining non-blocking: none material carried forward beyond the deferred items already recorded.
- **Critic review — Round 3** (rubber-duck sub-agent, isolated fresh context; doc v3; independently confirmed `Path`/`PathBuf`/`Serialize` imports, `tempfile` absent from dev-deps but locked, presence type `Vec<ModelPresence>`)
  - Verdict: **NOT READY** (5 blocking, 1 important).
  - Blocking: (1) prescribed handler used `into_unsync(json_response(...))` — double-wraps vs. the handler's `Response<Full<Bytes>>` return + route-level `into_unsync`; placeholder `<presence type>`; (2) adding a direct `tempfile` dev-dep changes `Cargo.lock`, so the first `cargo test --locked` fails on a stale lock — `Cargo.lock` not enumerated + no refresh step; (3) residual overclaim phrases ("will fail"/"matches `Command::new`"/"by construction"/"necessary-not-sufficient") contradict the conservative framing; (4) UI test still only string-asserted the loader/branch wiring; (5) R10's Node test hard-coded the command instead of binding to `HF_INSTALL_COMMAND`.
  - Important: `#[cfg(unix)]` must gate the whole non-exec test (non-Unix treats any file as executable).
  - Document changes: v4 — concrete `status_response_value(presence: Vec<ModelPresence>, hf)` + handler keeps `json_response` only (route does `into_unsync`); enumerated `Cargo.lock` + refresh-before-`--locked` step; scrubbed all four overclaim phrases to advisory wording; added pure `hfFromStatus` + `emptyModelOption` helpers with executable tests for state-propagation and both empty branches; R10 moved to a Rust `include_str!` test bound to the constant; whole non-exec test `#[cfg(unix)]`-gated.
  - Isolation: sub-agent-isolated (fresh context), not a separate OS session — recorded as sub-agent-isolated, not fully certified.
  - Remaining non-blocking: none material carried forward.
- **Readiness review — Round 4** (rubber-duck sub-agent, isolated fresh context; doc v4; independently verified handler return type `Response<Full<Bytes>>` + route `into_unsync`, `local_presence_snapshot()` → `Vec<ModelPresence>` (`ModelPresence` public), imports, both empty branches, both `if(statusRes)` guards, `escHtml`, `tempfile` in lock, and that the helper block runs in a Node `vm` with no undefined global)
  - Verdict: **NOT READY** (1 blocking, 1 important) — everything else verified sound.
  - Blocking: R10 `include_str!("../../…")` resolves above the repo (correct is `../README.md`/`../PRD.md`), and the sequence ran the R10 test (step 2) before adding the command to the docs (step 6) — failing by construction.
  - Important: the leading-empty-then-real-dir `resolve_in` test returns `true` for both a skipping and a non-skipping impl, so it can't prove R2's empty-skip invariant.
  - Document changes: v5 — fixed both `include_str!` paths to `../`; reordered the sequence so README/PRD command edits precede the R10 test; introduced predicate-injected `resolve_in_with` + a spy-predicate test that asserts the empty entry is **never probed** (recorded calls == `["/real/hf"]`), proving the invariant filesystem-/cwd-free.
  - Isolation: sub-agent-isolated (fresh context), not a separate OS session — recorded as sub-agent-isolated, not fully certified.
  - Remaining non-blocking: none.
- **Readiness review — Round 5**: **READY** (rubber-duck sub-agent, isolated fresh context; doc v5). Independently confirmed: `README.md`/`PRD.md` at repo root so `include_str!("../…")` resolves and the sequence edits docs (step 2) before the R10 test compiles (step 3); the spy-predicate empty-skip test genuinely proves empties are never probed and is race-/fs-free; `status_response_value(Vec<ModelPresence>, hf)` type + handler-keeps-`json_response` correct; the JS helper block references only `escHtml`/`document`/`toolboxHfPreflight`; both loader `if(statusRes)` guards and both ds4/halogen empty branches exist; R1–R10 each map to a concrete change + objective test. No blocking, no important. One nit (spy needs `RefCell` interior mutability under the `Fn` bound) — applied to the doc. Isolation: sub-agent-isolated (fresh context), not a separate OS session — recorded as sub-agent-isolated, not fully certified.

## Human approval

Skipped by explicit user delegation: the user is unavailable and instructed "Work autonomously and make good decisions." Recorded per HankNDory rule 7 (solo/unavailable operator). Five isolated Dory rounds were performed (3 critic + 2 readiness); Round 5 returned **READY** with no blocking/important findings. Additive and revertible (single commit) if the user wants to review before it ships more widely.

## Implementation log

- State: `complete` — implemented per the approved plan, validated, mean-reviewed.
- Files: `Cargo.toml`/`Cargo.lock` (tempfile dev-dep), `src/model_downloads.rs` (`HfPreflight`, `HF_INSTALL_COMMAND`, `is_executable_file`, `resolve_in_with`/`resolve_in`/`resolve_on_path`, `build_preflight`, `hf_preflight` + 6 tests incl. R10 `include_str!`), `src/server.rs` (`status_response_value` builder + handler wiring + endpoint-shape test), `src/escalation/templates/main_dashboard.html` (two banner mounts, `toolboxHfPreflight` state, marker-delimited pure helpers, loader wiring, ds4/halogen `emptyModelOption` branches), `scripts/test-hf-preflight-ui.cjs` (new, 5 tests), `README.md` + `PRD.md`.
- Validation: `cargo test --locked --lib -- --test-threads=1` → 400/400 (the 1 multi-thread failure was the pre-existing `stale_uds_is_removed_before_binding` UDS race, green single-threaded); `cargo clippy --locked --all-targets` → 0 errors (1 pre-existing `unnecessary_map_or` warning in `src/daemon.rs:134`, not this change); `node --test scripts/test-hf-preflight-ui.cjs` → 5/5; `bash scripts/check-html-js.sh` → OK.
- **Mean code review (Step 9): verdict SOUND** — no blocking/high/medium findings; 5 low-severity residual notes, all "no action required" (chief: `hfFromStatus`'s preserve-on-failure arm is guard-shadowed at both call sites but correct and independently unit-tested; direct-path split is `/`-only, irrelevant to the fixed `"hf"` on the Linux target). Reviewer confirmed no double-`into_unsync`, `models` bytes unchanged, escaping correct, empty-hint output byte-identical when hf present.
