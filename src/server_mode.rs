//! PR7/PR8/PR9: ds4, halogen, and vllm Server Mode — headless, detached
//! `podman run` servers, distinct from the Toolboxes-tab dev-shell
//! containers (`src/server.rs`'s `recreate_toolbox_container`/`toolbox
//! create` family). Kept as one file across backends (not split into
//! submodules), mirroring `src/model_downloads.rs`'s own single-file,
//! per-backend-match organization rather than inventing a different
//! convention mid-rollout (§13a).
//!
//! See `docs/design/ai-toolbox-cockpit-integration.md` §11 for why these
//! are two structurally different code paths upstream (`toolbox create`
//! never receives a `runtime_profile`'s `engine_args`; only Server Mode's
//! plain `podman run` does), §12 for ds4's v1 scope-narrowing, §13 for
//! halogen's, and §14 for vllm's.
//!
//! **ds4 v1 scope: standalone ("single-node") launch only.** Multi-node
//! coordinator/worker roles, DSpark, SSD-streaming, and MTP/vision-projector
//! auto-configuration are deliberately not ported — reachable instead via
//! the `custom_args` free-text escape hatch, exactly mirroring upstream's
//! own identically-named `server_runner.py::build_server_cmd(custom_args)`
//! parameter (not a brainrouter invention). See §12 for the full rationale.
//!
//! **halogen v1 scope: the full upstream `build_server_cmd()` surface** —
//! unlike ds4, halogen's own command builder has no multi-node/DSpark-style
//! parameters to narrow away, so there is no `custom_args` escape hatch for
//! it (§13: not a brainrouter invention to omit, upstream simply has
//! nothing there to mirror).
//!
//! **vllm v1 scope: the full upstream `build_server_cmd()` surface, podman
//! engine only.** vllm's own command builder supports both podman and
//! docker (§14 item 4); brainrouter narrows to podman only, same standing
//! decision as ds4/halogen ("no user-facing engine-selection concept").
//! Everything else upstream's builder does is ported: HF-repo-based model
//! identity (catalog or free-form custom repo), the per-model/per-toolbox
//! policy-override merge, the 4 persistent cache-directory mounts (plus an
//! opt-in `reset_caches` safety-gated wipe), and the `extra_args` escape
//! hatch (vllm has one, unlike halogen — mirrors ds4's `custom_args`).
//!
//! Lifecycle is synchronous request/response for every backend (§12 item
//! 5), not an async job registry like `model_downloads.rs` — starting/
//! stopping a container is a sub-second podman operation, and `status`
//! always reads live `podman inspect` state rather than persisting
//! anything.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::model_downloads::{self, ResolvedDs4Model, ResolvedHalogenBundle, ResolvedR9vPackage};
use crate::toolbox_catalog::{self, RuntimeProfile, SupportedServingBackend, ToolboxCatalog, ToolboxDefinition};

/// Fixed container name for the (single-instance, v1) ds4 server-mode
/// container — mirrors upstream's own fixed `ds4-cockpit-server` name.
/// §12's open item: single-instance is the proposed v1 default, matching
/// cockpit's own TUI (starting a second model requires stopping the
/// first), pending explicit sign-off.
pub const DS4_SERVER_CONTAINER_NAME: &str = "brainrouter-ds4-server";

/// Fixed container name for the (single-instance, v1) halogen server-mode
/// container. Deliberately brainrouter's own naming scheme, not upstream's
/// literal `CONTAINER_NAME = "ai-toolbox-cockpit-halogen-server"` — same
/// departure already established for ds4 above (avoids ownership confusion
/// if cockpit itself is also installed and run on the same host).
pub const HALOGEN_SERVER_CONTAINER_NAME: &str = "brainrouter-halogen-server";

/// Fixed container name for the (single-instance, v1) vllm server-mode
/// container. Same brainrouter-own naming departure as ds4/halogen above,
/// not upstream's literal `ai-toolbox-cockpit-vllm-server`.
pub const VLLM_SERVER_CONTAINER_NAME: &str = "brainrouter-vllm-server";

/// Fixed container name for the (single-instance, v1) r9v server-mode
/// container. Same brainrouter-own naming departure as the other three
/// backends, not upstream's literal `ai-toolbox-cockpit-r9v-server`.
pub const R9V_SERVER_CONTAINER_NAME: &str = "brainrouter-r9v-server";

/// Fallback binary name used when a toolbox's `backend_config` doesn't
/// declare a `server_binary` override — every real vendored ds4 toolbox
/// entry today doesn't, so this is also, in practice, the only value ever
/// used (§12 item 3).
const DS4_DEFAULT_SERVER_BINARY: &str = "ds4-server";

/// Reused from the Toolboxes-tab ownership convention (`src/server.rs`),
/// duplicated here rather than imported because that constant is private
/// to `server.rs` — both modules independently attach the same label value.
const LABEL_MANAGED: &str = "io.brainrouter.managed";
const LABEL_SERVER_BACKEND: &str = "io.brainrouter.server_backend";
const LABEL_SERVER_MODEL: &str = "io.brainrouter.server_model";

/// `POST /api/server-mode/ds4/start` request body.
#[derive(Debug, Clone, Deserialize)]
pub struct StartDs4ServerRequest {
    /// Which vendored ds4 toolbox variant to source the image/
    /// `runtime_profile` from (e.g. `strix-halo-ds4-rocm-10-0` vs.
    /// `strix-halo-ds4-therock-nightly`) — mirrors
    /// `create_toolbox_container`'s explicit-`toolbox_id` pattern rather
    /// than re-deriving an "effective default toolbox" server-side (§12
    /// item 7; the dashboard already computes this client-side via
    /// `effectiveDefaultToolbox()`, §4a).
    pub toolbox_id: String,
    /// Which already-downloaded ds4 catalog model to serve (same id space
    /// as `model_downloads::StartDownloadRequest::model_id`).
    pub model_id: String,
    /// Context length. **Required** — `build_server_cmd(ctx: int, ...)` has
    /// no default upstream and always emits `--ctx <n>`; there is no
    /// "let ds4-server pick its own default" case to omit it for (§12 item
    /// 3's correction). The dashboard form suggests upstream's own observed
    /// TUI default (126000, standalone mode) but the API itself always
    /// requires an explicit value.
    pub ctx: u32,
    /// Podman port-mapping **bind address** only. This is *not* passed to
    /// the server's own `--host` flag — upstream's `build_server_cmd`
    /// always hardcodes `"--host", "0.0.0.0"` inside the container
    /// regardless of this value (§12 item 3's correction). `"localhost"`
    /// is translated to `127.0.0.1`; `"0.0.0.0"` omits the bind-IP prefix
    /// entirely (`-p <port>:<port>`, reachable from the LAN).
    pub host: String,
    pub port: u16,
    /// Free-text, shell-word-split and appended verbatim after the core
    /// args — the v1 escape hatch for anything §12 item 2 deliberately
    /// doesn't natively port (multi-node roles, DSpark, SSD-streaming,
    /// MTP/vision), exactly mirroring upstream's own `custom_args`
    /// parameter.
    #[serde(default)]
    pub custom_args: Option<String>,
}

/// `GET /api/server-mode/ds4/status` response.
#[derive(Debug, Clone, Serialize)]
pub struct ServerStatus {
    pub backend: &'static str,
    pub container_name: String,
    /// Whether a container of this name exists at all (running or
    /// exited) — distinct from `running`, so a crashed server (exists,
    /// not running) reads differently in the dashboard than "never
    /// started" (doesn't exist).
    pub exists: bool,
    pub running: bool,
    /// The `model_id` the running/most-recently-started container was
    /// launched with, read back from its own ownership label — `None` if
    /// the container doesn't exist or predates this label being attached.
    pub model_id: Option<String>,
}

/// Errors from this module's operations, mapped to HTTP status by the
/// caller in `src/server.rs` (mirrors `model_downloads::DownloadError`'s
/// shape, not reused directly since the error space here is different —
/// e.g. no `Conflict` on a racing cancel, since there's no async job).
#[derive(Debug)]
pub enum ServerModeError {
    Validation(String),
    NotFound(String),
    Internal(String),
}

impl ServerModeError {
    pub fn status(&self) -> u16 {
        match self {
            Self::Validation(_) => 400,
            Self::NotFound(_) => 404,
            Self::Internal(_) => 500,
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Validation(m) | Self::NotFound(m) | Self::Internal(m) => m,
        }
    }
}

impl From<model_downloads::DownloadError> for ServerModeError {
    fn from(e: model_downloads::DownloadError) -> Self {
        match e {
            model_downloads::DownloadError::Validation(m) => ServerModeError::Validation(m),
            model_downloads::DownloadError::NotFound(m) => ServerModeError::NotFound(m),
            model_downloads::DownloadError::Conflict(m) | model_downloads::DownloadError::Internal(m) => {
                ServerModeError::Internal(m)
            }
        }
    }
}

/// PR9: lets `save_vllm_cache_paths()` (§14) use `?` directly against
/// `cockpit_config::apply_backend_setting_values()`'s own error type,
/// mirroring the `model_downloads::DownloadError` conversion above.
/// `NotAvailable` (cockpit's config directory doesn't exist on this host)
/// maps to `NotFound`/404 — the same status the existing
/// `apply_cockpit_default_toolbox`/`apply_cockpit_active_platform` HTTP
/// handlers already return for it (`src/server.rs`), just reached here via
/// `From` instead of a duplicated match in the handler.
impl From<crate::cockpit_config::ApplyError> for ServerModeError {
    fn from(e: crate::cockpit_config::ApplyError) -> Self {
        match e {
            crate::cockpit_config::ApplyError::NotAvailable => ServerModeError::NotFound(e.to_string()),
            other => ServerModeError::Internal(other.to_string()),
        }
    }
}

/// Mirrors upstream's `server_runner.py::_clean_engine_args()`: drops
/// `--group-add sudo` (both the two-token and `--group-add=sudo` forms) —
/// not needed for a headless server, only for the interactive dev-shell
/// use case.
fn clean_engine_args_for_server(engine_args: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(engine_args.len());
    let mut i = 0;
    while i < engine_args.len() {
        let arg = engine_args[i].as_str();
        if arg == "--group-add" && engine_args.get(i + 1).map(String::as_str) == Some("sudo") {
            i += 2;
            continue;
        }
        if arg == "--group-add=sudo" {
            i += 1;
            continue;
        }
        out.push(engine_args[i].clone());
        i += 1;
    }
    out
}

/// Mirrors upstream's `runtime/toolboxes.py::upgrade_groups_for_podman()`:
/// podman's rootless `--group-add` semantics don't support arbitrary named
/// supplementary groups the way Docker's do, so any of `video`/`render`/
/// `rdma`/`keep-groups` collapses into a single `--group-add keep-groups`.
/// brainrouter is podman-only (`src/server.rs` never shells out to
/// `docker`), so upstream's `engine != "podman"` no-op branch and
/// `adapt_nvidia_runtime_args()`'s Docker-only GB10 rewrite are both
/// deliberately not ported — there is no second engine for them to ever
/// apply to.
fn upgrade_groups_for_podman(engine_args: &[String]) -> Vec<String> {
    let mut group_values: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < engine_args.len() {
        if engine_args[i] == "--group-add" && i + 1 < engine_args.len() {
            group_values.push(engine_args[i + 1].as_str());
            i += 2;
            continue;
        }
        if let Some(v) = engine_args[i].strip_prefix("--group-add=") {
            group_values.push(v);
        }
        i += 1;
    }
    let needs_upgrade = group_values
        .iter()
        .any(|g| matches!(*g, "video" | "render" | "rdma" | "keep-groups"));
    if !needs_upgrade {
        return engine_args.to_vec();
    }

    let mut out = Vec::new();
    let mut added = false;
    let mut i = 0;
    while i < engine_args.len() {
        if engine_args[i] == "--group-add" && i + 1 < engine_args.len() {
            if !added {
                out.push("--group-add".to_string());
                out.push("keep-groups".to_string());
                added = true;
            }
            i += 2;
            continue;
        }
        if engine_args[i].starts_with("--group-add=") {
            if !added {
                out.push("--group-add".to_string());
                out.push("keep-groups".to_string());
                added = true;
            }
            i += 1;
            continue;
        }
        out.push(engine_args[i].clone());
        i += 1;
    }
    out
}

/// Builds the podman argument list for `podman <args>` (no leading
/// `"podman"` binary name — the caller supplies that), mirroring
/// `server_runner.py::build_server_cmd()`'s standalone-mode shape exactly
/// for the parts §12 item 4 says are ported verbatim, with `-d --name
/// brainrouter-ds4-server` replacing `--rm -it --name ds4-cockpit-server`
/// (§12 item 1, detached not foreground) and the multi-node/DSpark/
/// SSD-streaming/MTP/vision parameters entirely absent (§12 item 2).
pub fn build_ds4_server_command(
    toolbox_image: &str,
    runtime_profile: &RuntimeProfile,
    model: &ResolvedDs4Model,
    req: &StartDs4ServerRequest,
) -> Result<Vec<String>, ServerModeError> {
    let engine_args = upgrade_groups_for_podman(&clean_engine_args_for_server(&runtime_profile.engine_args));

    let mut args: Vec<String> = vec![
        "run".to_string(),
        "-d".to_string(),
        "--name".to_string(),
        DS4_SERVER_CONTAINER_NAME.to_string(),
    ];
    args.extend(engine_args);

    // ROCm requires host IPC sharing and ptrace capabilities to avoid HSA
    // memory mapping errors — always on, not conditional on backend,
    // mirroring upstream exactly.
    args.push("--ipc=host".to_string());
    args.extend(["--cap-add".to_string(), "SYS_PTRACE".to_string()]);
    args.extend(["--env".to_string(), "DS4_ROCM_ENABLE_MXFP4_TILE4=1".to_string()]);
    args.extend(["--env".to_string(), "DS4_ROCM_MXFP4_DOWN_RGROUP=4".to_string()]);
    // Podman-specific (brainrouter is podman-only, so this is unconditional
    // rather than gated on an `engine == "podman"` check upstream needs).
    args.extend(["--security-opt".to_string(), "label=disable".to_string()]);
    args.push("--userns=keep-id".to_string());

    args.extend(["--label".to_string(), format!("{LABEL_MANAGED}=true")]);
    args.extend(["--label".to_string(), format!("{LABEL_SERVER_BACKEND}=ds4")]);
    args.extend(["--label".to_string(), format!("{LABEL_SERVER_MODEL}={}", req.model_id)]);

    // Standalone-only port mapping (§12 item 2: the multi-node
    // `--network=host` branch is out of v1 scope). `host` only ever
    // selects the bind address of this mapping, never the in-container
    // `--host` flag below.
    let port_mapping = if req.host.is_empty() || req.host == "0.0.0.0" {
        format!("{}:{}", req.port, req.port)
    } else {
        let bind_ip = if req.host == "localhost" { "127.0.0.1" } else { req.host.as_str() };
        format!("{bind_ip}:{}:{}", req.port, req.port)
    };
    args.extend(["-p".to_string(), port_mapping]);

    args.extend(["-v".to_string(), format!("{}:/models:ro", model.models_dir.display())]);
    args.push(toolbox_image.to_string());

    args.push(DS4_DEFAULT_SERVER_BINARY.to_string());
    args.extend(["-m".to_string(), format!("/models/{}", model.filename)]);
    args.extend(["--ctx".to_string(), req.ctx.to_string()]);
    args.extend(["--host".to_string(), "0.0.0.0".to_string()]);
    args.extend(["--port".to_string(), req.port.to_string()]);

    if let Some(custom) = req.custom_args.as_deref().filter(|s| !s.trim().is_empty()) {
        let extra = shlex::split(custom).ok_or_else(|| {
            ServerModeError::Validation("custom_args is not valid shell-quoted text".to_string())
        })?;
        args.extend(extra);
    }

    Ok(args)
}

/// Loads and type-parses the vendored toolbox catalog once, shared by every
/// backend's toolbox resolver below.
fn load_typed_toolbox_catalog() -> Result<ToolboxCatalog, ServerModeError> {
    let vendored = toolbox_catalog::load_vendored_catalog();
    vendored
        .typed()
        .map(|(catalog, _models)| catalog)
        .map_err(|e| ServerModeError::Internal(format!("failed to parse vendored toolbox catalog: {e}")))
}

/// Resolves `toolbox_id` to a catalog toolbox of `backend`, gated on
/// `features.server` not being declared `Unavailable`. **`Experimental` is
/// accepted, not rejected** — this mirrors cockpit's own TUI filter
/// (`*/server.py::set_platform()`/`platform_toolboxes()`:
/// `item.feature_state("server") != "unavailable"`), which still lists an
/// experimental toolbox in the Server Mode dropdown, just badged
/// `[experimental]`, never blocks it outright. A stricter "must be exactly
/// `Supported`" check would incorrectly reject every real vllm/r9v/halogen
/// server-mode toolbox today — all three are vendored as `server:
/// experimental`; only ds4's happen to be `server: supported` (verified
/// against `assets/cockpit-catalog/toolboxes.json`).
fn resolve_toolbox_for_server(
    catalog: &ToolboxCatalog,
    backend: SupportedServingBackend,
    toolbox_id: &str,
) -> Result<(ToolboxDefinition, RuntimeProfile), ServerModeError> {
    let tb = catalog
        .toolbox_by_id(toolbox_id)
        .filter(|t| t.supported_backend() == Some(backend))
        .ok_or_else(|| {
            ServerModeError::NotFound(format!("no {} catalog toolbox with id `{toolbox_id}`", backend.as_str()))
        })?;
    if tb.features.server == toolbox_catalog::FeatureState::Unavailable {
        return Err(ServerModeError::Validation(format!(
            "toolbox `{toolbox_id}` does not declare Server Mode support"
        )));
    }
    let profile = catalog
        .runtime_profiles
        .get(&tb.runtime_profile)
        .cloned()
        .ok_or_else(|| {
            ServerModeError::Internal(format!(
                "toolbox `{toolbox_id}` references unknown runtime_profile `{}`",
                tb.runtime_profile
            ))
        })?;
    Ok((tb.clone(), profile))
}

fn resolve_ds4_toolbox(toolbox_id: &str) -> Result<(ToolboxDefinition, RuntimeProfile), ServerModeError> {
    let catalog = load_typed_toolbox_catalog()?;
    resolve_toolbox_for_server(&catalog, SupportedServingBackend::Ds4, toolbox_id)
}

/// Same as [`resolve_ds4_toolbox`], plus the owning platform id (needed by
/// [`build_halogen_server_command`]'s upstream-mirrored `platform_id`
/// check — halogen's `runner.py::build_server_cmd()` takes `platform_id` as
/// an explicit caller-supplied parameter with no other way to learn it;
/// brainrouter derives it from the catalog's own `platforms[].toolbox_ids`
/// instead of adding a redundant platform field to the HTTP request body).
fn resolve_halogen_toolbox(
    toolbox_id: &str,
) -> Result<(ToolboxDefinition, RuntimeProfile, String), ServerModeError> {
    let catalog = load_typed_toolbox_catalog()?;
    let (tb, profile) = resolve_toolbox_for_server(&catalog, SupportedServingBackend::Halogen, toolbox_id)?;
    let platform_id = catalog
        .platform_id_for_toolbox(toolbox_id)
        .ok_or_else(|| {
            ServerModeError::Internal(format!("toolbox `{toolbox_id}` is not listed under any catalog platform"))
        })?
        .to_string();
    Ok((tb, profile, platform_id))
}

/// `POST /api/server-mode/ds4/start`: resolves the toolbox + already-
/// downloaded model, force-removes any pre-existing container of the fixed
/// name (idempotent — mirrors upstream's own `podman rm -f` immediately
/// before every start, `runtime/server_process.py::run_foreground_server`),
/// then runs the built command.
pub async fn start_ds4_server(req: &StartDs4ServerRequest) -> Result<(), ServerModeError> {
    let (tb, profile) = resolve_ds4_toolbox(&req.toolbox_id)?;
    let model = model_downloads::resolve_downloaded_ds4_model(&req.model_id)?;
    let args = build_ds4_server_command(&tb.image, &profile, &model, req)?;

    let _ = tokio::process::Command::new("podman")
        .args(["rm", "-f", DS4_SERVER_CONTAINER_NAME])
        .output()
        .await;

    let out = tokio::process::Command::new("podman")
        .args(&args)
        .output()
        .await
        .map_err(|e| ServerModeError::Internal(format!("failed to exec podman: {e}")))?;
    if out.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&out.stderr);
        Err(ServerModeError::Internal(format!("podman run failed: {}", stderr.trim())))
    }
}

/// `POST /api/server-mode/ds4/stop`. Unlike upstream (which only ever
/// stops on its own foreground Ctrl+C, `rm -f`-only), brainrouter's
/// detached container gets an actual graceful `podman stop` first (bounded
/// timeout, SIGTERM then SIGKILL) — a natural corollary of the
/// already-decided detached-lifecycle model (§5), not something upstream
/// needed since it never runs detached. `podman rm -f` afterward is always
/// run too, both to guarantee cleanup and because it's a no-op (not an
/// error) if the container is already gone — idempotent either way.
pub async fn stop_ds4_server() -> Result<(), ServerModeError> {
    let _ = tokio::process::Command::new("podman")
        .args(["stop", "--time", "10", DS4_SERVER_CONTAINER_NAME])
        .output()
        .await;
    let out = tokio::process::Command::new("podman")
        .args(["rm", "-f", DS4_SERVER_CONTAINER_NAME])
        .output()
        .await
        .map_err(|e| ServerModeError::Internal(format!("failed to exec podman: {e}")))?;
    if out.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&out.stderr);
        Err(ServerModeError::Internal(format!("podman rm failed: {}", stderr.trim())))
    }
}

/// Reads one ownership-label value off a fixed-name container, `None` if
/// the container or label is absent. Mirrors `src/server.rs`'s
/// `toolbox_container_is_managed()` inspect-format pattern. Parametrized
/// over `container_name` (PR8+) so every backend's server-status check
/// shares one implementation instead of duplicating it per backend.
async fn server_container_label(container_name: &str, label: &str) -> Option<String> {
    let out = tokio::process::Command::new("podman")
        .args([
            "inspect",
            "--format",
            &format!("{{{{ index .Config.Labels \"{label}\" }}}}"),
            container_name,
        ])
        .output()
        .await
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!value.is_empty() && value != "<no value>").then_some(value)
}

/// Shared `GET /api/server-mode/<backend>/status` implementation: no
/// persisted state at all (§12 item 5) — always reads live `podman
/// inspect` output for the fixed container name.
async fn container_server_status(backend: &'static str, container_name: &str, label_key: &str) -> ServerStatus {
    let out = tokio::process::Command::new("podman")
        .args(["inspect", "--format", "{{.State.Running}}", container_name])
        .output()
        .await;
    match out {
        Ok(o) if o.status.success() => {
            let running = String::from_utf8_lossy(&o.stdout).trim() == "true";
            ServerStatus {
                backend,
                container_name: container_name.to_string(),
                exists: true,
                running,
                model_id: server_container_label(container_name, label_key).await,
            }
        }
        _ => ServerStatus {
            backend,
            container_name: container_name.to_string(),
            exists: false,
            running: false,
            model_id: None,
        },
    }
}

pub async fn ds4_server_status() -> ServerStatus {
    container_server_status("ds4", DS4_SERVER_CONTAINER_NAME, LABEL_SERVER_MODEL).await
}

// ── PR8: halogen Server Mode (§13) ────────────────────────────────────────
//
// Structurally different from ds4 in two ways confirmed against upstream's
// `backends/halogen/{server,runner,model_manager}.py`:
//  1. Model resolution is bundle-based (a checkpoint + precision overlay +
//     flat tokenizer, optionally a vision tower), not a single flat file —
//     see `model_downloads::ResolvedHalogenBundle`.
//  2. The launched container takes **no in-container CLI args at all**;
//     every tunable (model paths, port, context, KV pool, slots, prompt
//     cache) is passed as an `-e HALOGEN_*=...` environment variable, and
//     the image's own entrypoint reads them. There is no
//     `custom_args`-equivalent escape hatch because upstream's own
//     `build_server_cmd()` has no free-text parameter to mirror.

/// `POST /api/server-mode/halogen/start` request body. Field names mirror
/// upstream's own saved-settings keys (`host`/`port`/`context`/`pool`/
/// `slots`/`prompt_cache`/`bundle_id` in `backends.halogen` of cockpit's
/// config.json) rather than brainrouter-invented names, so a human
/// inspecting the shared config file sees the same vocabulary regardless of
/// which tool wrote it (§13a).
#[derive(Debug, Clone, Deserialize)]
pub struct StartHalogenServerRequest {
    /// The single vendored halogen toolbox (`strix-halo-halogen-flash`
    /// today) — still explicit, not hardcoded, in case a second
    /// platform/toolbox is added later (§13's open item).
    pub toolbox_id: String,
    /// Which already-downloaded halogen catalog bundle to serve.
    pub bundle_id: String,
    pub host: String,
    pub port: u16,
    /// Requested context length. Upstream's own bound: 1..=262144 (the
    /// Flash-Next release's native context).
    pub context_size: u32,
    /// KV cache pool size in positions. Must be >= `context_size` —
    /// upstream's own invariant, not a brainrouter addition.
    pub kv_pool_positions: u32,
    /// Concurrent request slots. Upstream's own minimum: 1.
    pub kv_slots: u32,
    /// `"0"` (off), `"1"` (exact-repeat reuse), or `"2"` (fast prefix
    /// reuse, upstream's own default) — modeled as a string, matching
    /// upstream's own `SearchableSelect` value type, not an enum, since
    /// it's stored verbatim as an env var string either way.
    pub prompt_cache: String,
}

/// Validates and normalizes `host` exactly as upstream's `build_server_cmd`
/// does: `"localhost"` → `"127.0.0.1"`, must otherwise parse as a literal
/// IP address (bare hostnames are rejected — a real divergence from ds4,
/// whose `host` is passed through unvalidated), IPv6 addresses get
/// bracket-wrapped for the `-p host:port:port` mapping.
fn validate_halogen_host(host: &str) -> Result<String, ServerModeError> {
    let trimmed = host.trim();
    let normalized = if trimmed.eq_ignore_ascii_case("localhost") { "127.0.0.1" } else { trimmed };
    let addr: std::net::IpAddr = normalized
        .parse()
        .map_err(|_| ServerModeError::Validation("host must be an IP address or \"localhost\"".to_string()))?;
    Ok(match addr {
        std::net::IpAddr::V6(_) => format!("[{addr}]"),
        std::net::IpAddr::V4(_) => addr.to_string(),
    })
}

/// Builds the podman argument list for `podman <args>` (no leading
/// `"podman"` binary name), mirroring `runner.py::build_server_cmd()`'s
/// validation and env-var shape exactly, with `-d --name
/// brainrouter-halogen-server` replacing `--rm -it --name
/// ai-toolbox-cockpit-halogen-server` (same detached-not-foreground
/// departure as ds4, §12 item 1) and brainrouter's own ownership `--label`s
/// inserted after the engine_args, mirroring ds4's builder's own label
/// placement.
///
/// **Deliberately does *not* call [`clean_engine_args_for_server`]** —
/// only [`upgrade_groups_for_podman`] — because upstream's own
/// `runner.py::build_server_cmd()` doesn't call `_clean_engine_args()`
/// either (verified by reading the full function body, not assumed from
/// ds4's shape); halogen's vendored `engine_args` also carries no
/// `--group-add sudo` to begin with, so this is fidelity, not a behavior
/// difference in practice today.
pub fn build_halogen_server_command(
    toolbox_image: &str,
    runtime_profile: &RuntimeProfile,
    platform_id: &str,
    bundle: &ResolvedHalogenBundle,
    req: &StartHalogenServerRequest,
) -> Result<Vec<String>, ServerModeError> {
    if platform_id != "strix-halo" {
        return Err(ServerModeError::Validation(
            "Halogen Flash supports Strix Halo (gfx1151) only".to_string(),
        ));
    }
    if !(1..=262144).contains(&req.context_size) {
        return Err(ServerModeError::Validation(
            "context_size must be between 1 and the native 262144 positions".to_string(),
        ));
    }
    if req.kv_pool_positions < req.context_size {
        return Err(ServerModeError::Validation(
            "kv_pool_positions must be at least the request context_size".to_string(),
        ));
    }
    if req.kv_slots < 1 {
        return Err(ServerModeError::Validation("kv_slots must be at least 1".to_string()));
    }
    if !matches!(req.prompt_cache.as_str(), "0" | "1" | "2") {
        return Err(ServerModeError::Validation(
            "prompt_cache must be \"0\", \"1\", or \"2\"".to_string(),
        ));
    }
    let host = validate_halogen_host(&req.host)?;
    let models_dir_str = bundle.models_dir.display().to_string();
    if models_dir_str.contains(':') {
        return Err(ServerModeError::Validation(
            "models directory cannot contain ':' in a container volume mount".to_string(),
        ));
    }

    let engine_args = upgrade_groups_for_podman(&runtime_profile.engine_args);

    let mut args: Vec<String> = vec![
        "run".to_string(),
        "-d".to_string(),
        "--name".to_string(),
        HALOGEN_SERVER_CONTAINER_NAME.to_string(),
    ];
    args.extend(engine_args);

    args.extend(["--label".to_string(), format!("{LABEL_MANAGED}=true")]);
    args.extend(["--label".to_string(), format!("{LABEL_SERVER_BACKEND}=halogen")]);
    args.extend(["--label".to_string(), format!("{LABEL_SERVER_MODEL}={}", req.bundle_id)]);

    args.extend(["-p".to_string(), format!("{host}:{}:{}", req.port, req.port)]);
    args.extend(["-v".to_string(), format!("{models_dir_str}:/models:ro")]);

    let mut env: Vec<(&str, String)> = vec![
        ("HALOGEN_CHECKPOINT", format!("/models/{}", bundle.checkpoint)),
        ("HALOGEN_CK_OVERLAY", format!("/models/{}", bundle.overlay)),
        ("HALOGEN_TOKENIZER", format!("/models/{}", bundle.tokenizer_dir)),
        ("HALOGEN_API_PORT", req.port.to_string()),
        ("HALOGEN_CTX", req.context_size.to_string()),
        ("HALOGEN_KV_POOL_POSITIONS", req.kv_pool_positions.to_string()),
        ("HALOGEN_KV_SLOTS", req.kv_slots.to_string()),
        ("HALOGEN_PROMPT_CACHE", req.prompt_cache.clone()),
    ];
    if let Some(vision_tower) = &bundle.vision_tower {
        env.push(("HALOGEN_VISION_TOWER", format!("/models/{vision_tower}")));
    }
    for (key, value) in env {
        args.extend(["-e".to_string(), format!("{key}={value}")]);
    }

    args.push(toolbox_image.to_string());
    Ok(args)
}

/// `POST /api/server-mode/halogen/start`: resolves the toolbox + already-
/// downloaded bundle, force-removes any pre-existing container of the
/// fixed name, runs the built command, then best-effort persists the
/// chosen settings into cockpit's shared config.json — mirroring
/// upstream's own `save_backend_settings("halogen", {...})` call in
/// `_start_confirmed()` (a behavior ds4's own `server.py` does *not* have,
/// confirmed by reading it; not something PR7 omitted by mistake). A
/// config.json write failure here does not fail the request: the server
/// container is already running by this point, so a missing cockpit config
/// directory (§4, `ApplyError::NotAvailable`) shouldn't roll back a
/// successful start.
pub async fn start_halogen_server(req: &StartHalogenServerRequest) -> Result<(), ServerModeError> {
    let (tb, profile, platform_id) = resolve_halogen_toolbox(&req.toolbox_id)?;
    let bundle = model_downloads::resolve_downloaded_halogen_bundle(&req.bundle_id)?;
    let args = build_halogen_server_command(&tb.image, &profile, &platform_id, &bundle, req)?;

    let _ = tokio::process::Command::new("podman")
        .args(["rm", "-f", HALOGEN_SERVER_CONTAINER_NAME])
        .output()
        .await;

    let out = tokio::process::Command::new("podman")
        .args(&args)
        .output()
        .await
        .map_err(|e| ServerModeError::Internal(format!("failed to exec podman: {e}")))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(ServerModeError::Internal(format!("podman run failed: {}", stderr.trim())));
    }

    let _ = crate::cockpit_config::apply_backend_setting_values(
        "halogen",
        vec![
            ("host", serde_json::Value::String(req.host.clone())),
            ("port", serde_json::Value::String(req.port.to_string())),
            ("context", serde_json::Value::String(req.context_size.to_string())),
            ("pool", serde_json::Value::String(req.kv_pool_positions.to_string())),
            ("slots", serde_json::Value::String(req.kv_slots.to_string())),
            ("prompt_cache", serde_json::Value::String(req.prompt_cache.clone())),
            ("bundle_id", serde_json::Value::String(req.bundle_id.clone())),
        ],
    );

    Ok(())
}

/// `POST /api/server-mode/halogen/stop` — same graceful-stop-then-rm
/// contract as [`stop_ds4_server`].
pub async fn stop_halogen_server() -> Result<(), ServerModeError> {
    let _ = tokio::process::Command::new("podman")
        .args(["stop", "--time", "10", HALOGEN_SERVER_CONTAINER_NAME])
        .output()
        .await;
    let out = tokio::process::Command::new("podman")
        .args(["rm", "-f", HALOGEN_SERVER_CONTAINER_NAME])
        .output()
        .await
        .map_err(|e| ServerModeError::Internal(format!("failed to exec podman: {e}")))?;
    if out.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&out.stderr);
        Err(ServerModeError::Internal(format!("podman rm failed: {}", stderr.trim())))
    }
}

pub async fn halogen_server_status() -> ServerStatus {
    container_server_status("halogen", HALOGEN_SERVER_CONTAINER_NAME, LABEL_SERVER_MODEL).await
}

// ── PR9: vllm Server Mode (§14) ───────────────────────────────────────────
//
// Structurally different from both ds4 and halogen, confirmed against
// upstream's full `backends/vllm/{server,runner}.py`:
//  1. Model identity is a raw Hugging Face repo string (catalog-sourced or
//     free-form custom), not a downloaded local file/bundle — vLLM itself
//     pulls from HF at container start, so there is no completeness gate.
//  2. A catalog model entry *is* its own launch policy (`valid_tp`, `ctx`,
//     `trust_remote`, `enforce_eager`, `attention_backend`, `env`,
//     `extra_flags`), optionally shallow-merged with a toolbox's
//     `backend_config.policy_overrides` — mirrored here as raw
//     `serde_json::Map` merges (`dict.update()` semantics), exactly as
//     upstream's `apply_toolbox_policy_overrides()` does.
//  3. `attention_backend` distinguishes "key absent" (apply upstream's own
//     `"TRITON_ATTN"` default, operator override wins) from "key present
//     with JSON `null`" (model-specific, no flag emitted at all — an
//     operator override in this case is a validation error here, a
//     deliberate brainrouter-side stricter-than-upstream choice: upstream
//     achieves this only by disabling the TUI selector, which has no
//     headless-API equivalent to silently mirror — see §14a).
//  4. Podman-only, same standing decision as ds4/halogen — upstream also
//     supports docker with a different flag set and an NVIDIA-specific
//     `adapt_nvidia_runtime_args()` rewrite, neither ported (§14 item 4).
//  5. Four persistent, operator-configurable cache directories (HF/vllm/
//     triton/aiter), not a read-only models directory — mounted on every
//     start, and independently persistable via a *separate* endpoint
//     mirroring upstream's own distinct "Save Cache Paths" action.
//  6. `hf_token` never blocks a start: present -> `-e HF_TOKEN=<token>`,
//     absent -> bare `-e HF_TOKEN` (ambient host-process passthrough),
//     exactly mirroring upstream's own fallback.
//  7. Unlike halogen, vllm's upstream `_start_confirmed()` does **not**
//     call `save_backend_settings()` — only its own "Save Cache Paths"
//     button does. `start_vllm_server()` below therefore does not persist
//     run-time settings into config.json, a deliberate fidelity choice,
//     not an omission (§14).

/// A vllm model/toolbox's effective launch policy, resolved from raw JSON
/// into named fields up front. `attention_backend` is the one field that
/// genuinely needs Rust's `Option<Option<T>>` (not collapsible to a single
/// `Option<String>`) to preserve upstream's own 3-way distinction: `None`
/// (key absent from the merged policy map — apply the `"TRITON_ATTN"`
/// default), `Some(None)` (key present, explicit JSON `null` — model-
/// specific, no `--attention-backend` flag at all), `Some(Some(s))` (key
/// present with a string default).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VllmEffectivePolicy {
    valid_tp: Vec<u32>,
    trust_remote: bool,
    ctx: Option<String>,
    enforce_eager: bool,
    attention_backend: Option<Option<String>>,
    env: Vec<(String, String)>,
    extra_flags: Vec<String>,
}

/// Builds the raw policy map for a catalog-sourced vllm model: the subset
/// of `VllmModel`'s fields upstream's `build_server_cmd()` actually reads
/// (`valid_tp`, `trust_remote` from the typed fields; `ctx`/`enforce_eager`/
/// `attention_backend`/`env`/`extra_flags` copied verbatim from `extra` if
/// present — preserving JSON `null` vs. absent exactly, since `extra` is a
/// raw `serde_json::Map`).
fn vllm_model_policy_map(model: &toolbox_catalog::VllmModel) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert(
        "valid_tp".to_string(),
        Value::Array(model.valid_tp.iter().map(|n| Value::from(*n)).collect()),
    );
    m.insert("trust_remote".to_string(), Value::Bool(model.trust_remote));
    for key in ["ctx", "enforce_eager", "attention_backend", "env", "extra_flags"] {
        if let Some(v) = model.extra.get(key) {
            m.insert(key.to_string(), v.clone());
        }
    }
    m
}

/// Upstream's own generic fallback policy for a custom (non-catalog) HF
/// repo, verified against `_prepare_start()`'s literal
/// `{"valid_tp": [1, 2], "attention_backend": "TRITON_ATTN", "extra_flags":
/// [], "env": {}}`.
fn vllm_generic_default_policy_map() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("valid_tp".to_string(), Value::Array(vec![Value::from(1u32), Value::from(2u32)]));
    m.insert("attention_backend".to_string(), Value::String("TRITON_ATTN".to_string()));
    m.insert("extra_flags".to_string(), Value::Array(Vec::new()));
    m.insert("env".to_string(), Value::Object(Map::new()));
    m
}

/// Mirrors `runner.py::apply_toolbox_policy_overrides()`: a shallow merge
/// (`dict.update()` semantics — every overridden key is wholly replaced,
/// never deep-merged) of `backend_config.policy_overrides` on top of the
/// base policy map.
fn apply_vllm_toolbox_policy_overrides(base: Map<String, Value>, backend_config: Option<&Value>) -> Map<String, Value> {
    let mut result = base;
    if let Some(overrides) = backend_config.and_then(|c| c.get("policy_overrides")).and_then(Value::as_object) {
        for (k, v) in overrides {
            result.insert(k.clone(), v.clone());
        }
    }
    result
}

/// Resolves the final typed [`VllmEffectivePolicy`] from a merged raw
/// policy map, applying the same last-resort defaults upstream's
/// `build_server_cmd()` itself falls back to (`valid_tp` -> `[1]` if
/// absent/empty; `enforce_eager` -> `false`; `attention_backend` absent ->
/// `TRITON_ATTN` is applied at command-build time, not here, so this
/// function's own `None` case is preserved for the builder to interpret).
fn resolve_vllm_effective_policy(policy_map: &Map<String, Value>) -> VllmEffectivePolicy {
    let valid_tp: Vec<u32> = policy_map
        .get("valid_tp")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(|v| v.as_u64()).map(|n| n as u32).collect())
        .filter(|v: &Vec<u32>| !v.is_empty())
        .unwrap_or_else(|| vec![1]);
    let trust_remote = policy_map.get("trust_remote").and_then(Value::as_bool).unwrap_or(false);
    let ctx = policy_map.get("ctx").map(|v| match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    });
    let enforce_eager = policy_map.get("enforce_eager").and_then(Value::as_bool).unwrap_or(false);
    let attention_backend = match policy_map.get("attention_backend") {
        None => None,
        Some(Value::Null) => Some(None),
        Some(Value::String(s)) => Some(Some(s.clone())),
        Some(other) => Some(Some(other.to_string())),
    };
    let env = policy_map
        .get("env")
        .and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .map(|(k, v)| (k.clone(), match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                }))
                .collect()
        })
        .unwrap_or_default();
    let extra_flags = policy_map
        .get("extra_flags")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    VllmEffectivePolicy { valid_tp, trust_remote, ctx, enforce_eager, attention_backend, env, extra_flags }
}

/// Persistent host cache directories mounted into every vllm server
/// container (§14 item 5) — read-write, operator-configurable, resolved
/// the same override-then-catalog-default precedence as
/// `model_downloads::effective_models_dir()`, but sourced from upstream's
/// own hardcoded defaults (there is no catalog `storage.default` for these,
/// since they're not "the models directory").
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VllmCacheDirs {
    huggingface: PathBuf,
    vllm: PathBuf,
    triton: PathBuf,
    aiter: PathBuf,
}

/// Resolves the effective cache directories for a vllm launch: a cockpit
/// `backends.vllm.<key>` config.json override first, falling back to
/// upstream's own defaults (`~/.cache/{huggingface,vllm,triton}`,
/// `~/.aiter`) — mirrors `runner.py::default_cache_paths()` exactly.
fn effective_vllm_cache_dirs() -> VllmCacheDirs {
    let cockpit = crate::cockpit_config::load();
    let settings = cockpit.config.as_ref().and_then(|c| c.backends.get("vllm"));
    let get = |key: &str, default: &str| -> PathBuf {
        settings
            .and_then(|s| s.extra.get(key))
            .and_then(Value::as_str)
            .map(model_downloads::expand_tilde)
            .unwrap_or_else(|| model_downloads::expand_tilde(default))
    };
    VllmCacheDirs {
        huggingface: get("hf_cache", "~/.cache/huggingface"),
        vllm: get("vllm_cache", "~/.cache/vllm"),
        triton: get("triton_cache", "~/.cache/triton"),
        aiter: get("aiter_cache", "~/.aiter"),
    }
}

/// `POST /api/server-mode/vllm/start` request body. Exactly one of
/// `model_id` (a vendored vllm catalog entry id) or `custom_repo` (a raw
/// `owner/model` Hugging Face repo, bypassing the catalog entirely) must
/// be given — mirrors upstream's own mutually-exclusive curated-dropdown/
/// free-text-input pair.
#[derive(Debug, Clone, Deserialize)]
pub struct StartVllmServerRequest {
    pub toolbox_id: String,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub custom_repo: Option<String>,
    pub host: String,
    pub port: u16,
    pub tensor_parallel: u32,
    pub max_num_seqs: u32,
    /// `"auto"` (resolves to the effective policy's own `ctx`, or the
    /// literal string `"auto"` if the policy has none — matching upstream
    /// exactly, `--max-model-len` is always emitted) or a positive integer
    /// string.
    pub max_model_len: String,
    pub gpu_memory_utilization: f64,
    #[serde(default)]
    pub attention_backend: Option<String>,
    #[serde(default)]
    pub enforce_eager: Option<bool>,
    pub dtype: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub extra_args: Option<String>,
    /// Wipes the vllm/triton/aiter (never huggingface) compiled-cache
    /// directories immediately before launch — a real data-deletion
    /// action, gated by [`validate_compiled_cache_root`]'s safety check
    /// before any deletion, mirroring upstream's own
    /// `validate_compiled_cache_roots()`/`_clear_compiled_caches()`.
    #[serde(default)]
    pub reset_caches: bool,
}

fn resolve_vllm_toolbox(toolbox_id: &str) -> Result<(ToolboxDefinition, RuntimeProfile), ServerModeError> {
    let catalog = load_typed_toolbox_catalog()?;
    resolve_toolbox_for_server(&catalog, SupportedServingBackend::Vllm, toolbox_id)
}

/// Resolves `req`'s model reference into `(effective HF repo string, base
/// policy map)` — either a vendored vllm catalog entry (`model_id`) or a
/// custom free-form repo (`custom_repo`, generic default policy). Exactly
/// one of the two request fields must be given.
fn resolve_vllm_model_and_base_policy(req: &StartVllmServerRequest) -> Result<(String, Map<String, Value>), ServerModeError> {
    let model_id = req.model_id.as_deref().filter(|s| !s.trim().is_empty());
    let custom_repo = req.custom_repo.as_deref().filter(|s| !s.trim().is_empty());
    match (model_id, custom_repo) {
        (Some(_), Some(_)) => Err(ServerModeError::Validation(
            "specify exactly one of model_id or custom_repo, not both".to_string(),
        )),
        (None, None) => Err(ServerModeError::Validation("either model_id or custom_repo is required".to_string())),
        (None, Some(repo)) => Ok((repo.to_string(), vllm_generic_default_policy_map())),
        (Some(id), None) => {
            let vendored = toolbox_catalog::load_vendored_catalog();
            let (_toolboxes, models) = vendored
                .typed()
                .map_err(|e| ServerModeError::Internal(format!("failed to parse vendored model catalog: {e}")))?;
            let backend_catalog = models
                .backends
                .iter()
                .find(|b| b.backend.as_str() == "vllm")
                .ok_or_else(|| ServerModeError::Internal("catalog has no `vllm` backend section".to_string()))?;
            let entry = backend_catalog
                .entries
                .iter()
                .find(|e| e.id == id)
                .ok_or_else(|| ServerModeError::NotFound(format!("no vllm catalog entry with id `{id}`")))?;
            let model = match &entry.payload {
                Some(toolbox_catalog::ModelPayload::Vllm(m)) => m,
                _ => {
                    return Err(ServerModeError::Internal(format!(
                        "catalog entry `{id}` has no typed vllm payload"
                    )));
                }
            };
            Ok((model.repo.clone(), vllm_model_policy_map(model)))
        }
    }
}

/// Mirrors upstream's `validate_compiled_cache_roots()`: rejects `/`, the
/// home directory, any path with fewer than 3 components, or any path
/// missing `marker` as a (case-insensitive) substring of one of its own
/// components — run before any deletion, not a mere warning.
fn validate_compiled_cache_root(path: &std::path::Path, marker: &str) -> Result<PathBuf, ServerModeError> {
    let resolved = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let home = std::env::var("HOME").ok().map(PathBuf::from);
    let has_marker = resolved
        .components()
        .any(|c| c.as_os_str().to_string_lossy().to_lowercase().contains(marker));
    if resolved == std::path::Path::new("/")
        || home.as_deref() == Some(resolved.as_path())
        || resolved.components().count() < 3
        || !has_marker
    {
        return Err(ServerModeError::Validation(format!(
            "refusing unsafe cache root {}: must contain a `{marker}` path component and not be `/` or the home directory",
            resolved.display()
        )));
    }
    Ok(resolved)
}

/// Wipes the contents (not the directory itself) of the vllm/triton/aiter
/// cache roots — huggingface is deliberately excluded, matching upstream's
/// own `_clear_compiled_caches()` exactly (it never touches the HF cache).
fn reset_vllm_compiled_caches(cache_dirs: &VllmCacheDirs) -> Result<(), ServerModeError> {
    for (path, marker) in [
        (&cache_dirs.vllm, "vllm"),
        (&cache_dirs.triton, "triton"),
        (&cache_dirs.aiter, "aiter"),
    ] {
        let resolved = validate_compiled_cache_root(path, marker)?;
        if !resolved.exists() {
            std::fs::create_dir_all(&resolved)
                .map_err(|e| ServerModeError::Internal(format!("failed to create {}: {e}", resolved.display())))?;
            continue;
        }
        let entries = std::fs::read_dir(&resolved)
            .map_err(|e| ServerModeError::Internal(format!("failed to read {}: {e}", resolved.display())))?;
        for entry in entries {
            let entry = entry.map_err(|e| ServerModeError::Internal(format!("failed to read cache entry: {e}")))?;
            let p = entry.path();
            let is_dir = entry.file_type().map(|t| t.is_dir() && !t.is_symlink()).unwrap_or(false);
            let result = if is_dir { std::fs::remove_dir_all(&p) } else { std::fs::remove_file(&p) };
            result.map_err(|e| ServerModeError::Internal(format!("failed to remove {}: {e}", p.display())))?;
        }
    }
    Ok(())
}

/// Builds the podman argument list for `podman <args>` (no leading
/// `"podman"` binary name), mirroring `runner.py::build_server_cmd()`'s
/// podman branch exactly — see the module-level PR9 note above for the
/// six structural differences from ds4/halogen this reflects.
pub(crate) fn build_vllm_server_command(
    toolbox_image: &str,
    runtime_profile: &RuntimeProfile,
    model_repo: &str,
    policy: &VllmEffectivePolicy,
    cache_dirs: &VllmCacheDirs,
    hf_token: Option<&str>,
    req: &StartVllmServerRequest,
) -> Result<Vec<String>, ServerModeError> {
    if model_repo.trim().is_empty() {
        return Err(ServerModeError::Validation("a model repository is required".to_string()));
    }
    if !policy.valid_tp.contains(&req.tensor_parallel) {
        return Err(ServerModeError::Validation(format!(
            "tensor_parallel {} is not permitted for this model (valid: {:?})",
            req.tensor_parallel, policy.valid_tp
        )));
    }
    if req.port == 0 {
        return Err(ServerModeError::Validation("port must be nonzero".to_string()));
    }
    if req.max_num_seqs == 0 {
        return Err(ServerModeError::Validation("max_num_seqs must be at least 1".to_string()));
    }
    if !(req.gpu_memory_utilization > 0.0 && req.gpu_memory_utilization <= 1.0) {
        return Err(ServerModeError::Validation(
            "gpu_memory_utilization must be greater than 0 and at most 1".to_string(),
        ));
    }
    if req.max_model_len != "auto" && req.max_model_len.parse::<u64>().is_err() {
        return Err(ServerModeError::Validation(
            "max_model_len must be \"auto\" or a positive integer".to_string(),
        ));
    }
    let requested_attention = req.attention_backend.as_deref().filter(|s| !s.is_empty());
    if requested_attention.is_some() && matches!(policy.attention_backend, Some(None)) {
        return Err(ServerModeError::Validation(
            "this model requires a specific attention backend and does not accept an override".to_string(),
        ));
    }

    let engine_args = upgrade_groups_for_podman(&clean_engine_args_for_server(&runtime_profile.engine_args));

    let mut args: Vec<String> = vec![
        "run".to_string(),
        "-d".to_string(),
        "--name".to_string(),
        VLLM_SERVER_CONTAINER_NAME.to_string(),
    ];
    args.extend(engine_args);
    args.push("--ipc=host".to_string());
    args.push("--cap-add=SYS_PTRACE".to_string());
    // Podman-specific (brainrouter is podman-only, §14 item 4) — upstream's
    // `elif engine == "docker"` branch is not ported, same reasoning
    // already established for ds4's builder.
    args.extend(["--security-opt".to_string(), "label=disable".to_string()]);
    args.push("--userns=keep-id".to_string());

    args.extend(["--label".to_string(), format!("{LABEL_MANAGED}=true")]);
    args.extend(["--label".to_string(), format!("{LABEL_SERVER_BACKEND}=vllm")]);
    args.extend(["--label".to_string(), format!("{LABEL_SERVER_MODEL}={model_repo}")]);

    let bind_host = if req.host == "localhost" { "127.0.0.1" } else { req.host.as_str() };
    let mapping = if bind_host == "0.0.0.0" {
        format!("{}:{}", req.port, req.port)
    } else {
        format!("{bind_host}:{}:{}", req.port, req.port)
    };
    args.extend(["-p".to_string(), mapping]);

    for (key, value) in [
        ("HOME", "/workspace".to_string()),
        ("VLLM_CONFIG_ROOT", "/workspace/.cache/vllm/config".to_string()),
        ("TRITON_CACHE_DIR", "/workspace/.cache/triton".to_string()),
        ("TILELANG_CACHE_DIR", "/workspace/.cache/triton/tilelang".to_string()),
        ("VLLM_NO_USAGE_STATS", "1".to_string()),
    ] {
        args.extend(["-e".to_string(), format!("{key}={value}")]);
    }
    let hf_token_env = match hf_token.filter(|t| !t.is_empty()) {
        Some(t) => format!("HF_TOKEN={t}"),
        None => "HF_TOKEN".to_string(),
    };
    args.extend(["-e".to_string(), hf_token_env]);

    for (host_path, container_path) in [
        (&cache_dirs.huggingface, "/workspace/.cache/huggingface"),
        (&cache_dirs.vllm, "/workspace/.cache/vllm"),
        (&cache_dirs.triton, "/workspace/.cache/triton"),
        (&cache_dirs.aiter, "/workspace/.aiter"),
    ] {
        args.extend(["-v".to_string(), format!("{}:{container_path}", host_path.display())]);
    }
    for (key, value) in &policy.env {
        args.extend(["-e".to_string(), format!("{key}={value}")]);
    }

    args.push(toolbox_image.to_string());
    args.extend(["vllm".to_string(), "serve".to_string(), model_repo.to_string()]);

    let resolved_max_model_len = if req.max_model_len == "auto" {
        policy.ctx.clone().unwrap_or_else(|| "auto".to_string())
    } else {
        req.max_model_len.clone()
    };
    args.extend(["--host".to_string(), "0.0.0.0".to_string()]);
    args.extend(["--port".to_string(), req.port.to_string()]);
    args.extend(["--tensor-parallel-size".to_string(), req.tensor_parallel.to_string()]);
    args.extend(["--max-num-seqs".to_string(), req.max_num_seqs.to_string()]);
    args.extend(["--max-model-len".to_string(), resolved_max_model_len]);
    args.extend(["--gpu-memory-utilization".to_string(), req.gpu_memory_utilization.to_string()]);
    args.extend(["--dtype".to_string(), req.dtype.clone()]);

    if policy.trust_remote {
        args.push("--trust-remote-code".to_string());
    }
    let eager = req.enforce_eager.unwrap_or(policy.enforce_eager);
    if eager {
        args.push("--enforce-eager".to_string());
    }
    if let Some(key) = req.api_key.as_deref().filter(|k| !k.is_empty()) {
        args.extend(["--api-key".to_string(), key.to_string()]);
    }
    match &policy.attention_backend {
        None => {
            let backend = requested_attention.unwrap_or("TRITON_ATTN");
            args.extend(["--attention-backend".to_string(), backend.to_string()]);
        }
        Some(Some(default_backend)) => {
            let backend = requested_attention.unwrap_or(default_backend.as_str());
            args.extend(["--attention-backend".to_string(), backend.to_string()]);
        }
        Some(None) => {
            // Model-specific: no flag at all, no override allowed (validated above).
        }
    }
    args.extend(policy.extra_flags.clone());

    if let Some(extra) = req.extra_args.as_deref().filter(|s| !s.trim().is_empty()) {
        let parsed = shlex::split(extra)
            .ok_or_else(|| ServerModeError::Validation("extra_args is not valid shell-quoted text".to_string()))?;
        args.extend(parsed);
    }

    Ok(args)
}

/// The hf_token source for the vllm container's `HF_TOKEN` env var: the
/// daemon process's own `HF_TOKEN` env var first, falling back to
/// cockpit's saved top-level `hf_token` config.json setting — the same
/// precedence `model_downloads::hf_subprocess_env()` already established
/// for the `hf download` subprocess, reused here for consistency rather
/// than inventing a second resolution order.
fn resolve_vllm_hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| crate::cockpit_config::load().config.and_then(|c| c.hf_token))
}

/// `POST /api/server-mode/vllm/start`: resolves the toolbox + model/policy,
/// creates the (possibly overridden) cache directories, optionally wipes
/// the compiled ones first, force-removes any pre-existing container of
/// the fixed name, then runs the built command. **Does not** persist
/// run-time settings into cockpit's config.json (§14 item 7) — unlike
/// halogen, upstream's own vllm `_start_confirmed()` has no
/// `save_backend_settings()` call; only the separate
/// [`save_vllm_cache_paths`] does.
pub async fn start_vllm_server(req: &StartVllmServerRequest) -> Result<(), ServerModeError> {
    let (tb, profile) = resolve_vllm_toolbox(&req.toolbox_id)?;
    let (model_repo, base_policy) = resolve_vllm_model_and_base_policy(req)?;
    let merged_policy = apply_vllm_toolbox_policy_overrides(base_policy, tb.backend_config.as_ref());
    let policy = resolve_vllm_effective_policy(&merged_policy);

    let cache_dirs = effective_vllm_cache_dirs();
    for dir in [&cache_dirs.huggingface, &cache_dirs.vllm, &cache_dirs.triton, &cache_dirs.aiter] {
        std::fs::create_dir_all(dir)
            .map_err(|e| ServerModeError::Internal(format!("failed to create cache directory {}: {e}", dir.display())))?;
    }
    if req.reset_caches {
        reset_vllm_compiled_caches(&cache_dirs)?;
    }

    let hf_token = resolve_vllm_hf_token();
    let args = build_vllm_server_command(&tb.image, &profile, &model_repo, &policy, &cache_dirs, hf_token.as_deref(), req)?;

    let _ = tokio::process::Command::new("podman")
        .args(["rm", "-f", VLLM_SERVER_CONTAINER_NAME])
        .output()
        .await;

    let out = tokio::process::Command::new("podman")
        .args(&args)
        .output()
        .await
        .map_err(|e| ServerModeError::Internal(format!("failed to exec podman: {e}")))?;
    if out.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&out.stderr);
        Err(ServerModeError::Internal(format!("podman run failed: {}", stderr.trim())))
    }
}

/// `POST /api/server-mode/vllm/stop` — same graceful-stop-then-rm contract
/// as [`stop_ds4_server`]/[`stop_halogen_server`].
pub async fn stop_vllm_server() -> Result<(), ServerModeError> {
    let _ = tokio::process::Command::new("podman")
        .args(["stop", "--time", "10", VLLM_SERVER_CONTAINER_NAME])
        .output()
        .await;
    let out = tokio::process::Command::new("podman")
        .args(["rm", "-f", VLLM_SERVER_CONTAINER_NAME])
        .output()
        .await
        .map_err(|e| ServerModeError::Internal(format!("failed to exec podman: {e}")))?;
    if out.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&out.stderr);
        Err(ServerModeError::Internal(format!("podman rm failed: {}", stderr.trim())))
    }
}

pub async fn vllm_server_status() -> ServerStatus {
    container_server_status("vllm", VLLM_SERVER_CONTAINER_NAME, LABEL_SERVER_MODEL).await
}

/// `POST /api/server-mode/vllm/cache-paths` request body — the direct
/// analogue of upstream's own separate "Save Cache Paths" button (§14 item
/// 7): every field is optional, only the ones supplied are persisted/
/// `mkdir -p`'d, mirroring `save_caches_pressed()`'s all-four-at-once shape
/// loosely (brainrouter allows a partial update; upstream's UI always
/// submits all four since they're all always-visible form fields, but nothing
/// about `apply_backend_setting_values()` requires that).
#[derive(Debug, Clone, Deserialize)]
pub struct VllmCachePathsRequest {
    #[serde(default)]
    pub hf_cache: Option<String>,
    #[serde(default)]
    pub vllm_cache: Option<String>,
    #[serde(default)]
    pub triton_cache: Option<String>,
    #[serde(default)]
    pub aiter_cache: Option<String>,
}

/// Creates each supplied cache directory, then persists it into cockpit's
/// shared config.json via the generic [`crate::cockpit_config::apply_backend_setting_values`]
/// helper built during PR8 (§13a) — its first reuse by a second backend, as
/// designed.
pub fn save_vllm_cache_paths(req: &VllmCachePathsRequest) -> Result<(), ServerModeError> {
    let provided: Vec<(&str, &str)> = [
        ("hf_cache", req.hf_cache.as_deref()),
        ("vllm_cache", req.vllm_cache.as_deref()),
        ("triton_cache", req.triton_cache.as_deref()),
        ("aiter_cache", req.aiter_cache.as_deref()),
    ]
    .into_iter()
    .filter_map(|(k, v)| v.map(|v| (k, v)))
    .collect();
    if provided.is_empty() {
        return Err(ServerModeError::Validation("at least one cache path must be provided".to_string()));
    }
    for (_, raw) in &provided {
        let expanded = model_downloads::expand_tilde(raw);
        std::fs::create_dir_all(&expanded)
            .map_err(|e| ServerModeError::Internal(format!("failed to create cache directory {}: {e}", expanded.display())))?;
    }
    let updates: Vec<(&str, Value)> = provided.into_iter().map(|(k, v)| (k, Value::String(v.to_string()))).collect();
    crate::cockpit_config::apply_backend_setting_values("vllm", updates)?;
    Ok(())
}

// ── PR11: r9v Server Mode (§15/§15a) ──────────────────────────────────────
//
// Structurally the most different backend of the four: upstream's own
// `runner.py::DEFAULTS` dict makes *every* tuning field optional (merged
// via `{**DEFAULTS, **(values or {})}`), so unlike ds4/halogen/vllm's
// mostly-required request shapes, `StartR9vServerRequest` mirrors that
// shape — only `toolbox_id`/`package_id` are required, everything else
// falls back to upstream's own literal defaults below. r9v also uses
// `--user 0:0` (root inside the container) rather than the
// `--userns=keep-id` every other backend's builder uses — confirmed by
// reading `runner.py::build_server_cmd()` in full, not assumed by
// analogy — and has its own cache-directory separation safety check
// (`model == cache` / `model` an ancestor of `cache` / `ple == cache`)
// with no equivalent in any other backend.

/// Every one of upstream's `DEFAULTS` dict values (`runner.py`), applied
/// whenever the corresponding [`StartR9vServerRequest`] field is omitted.
const R9V_DEFAULT_HOST: &str = "127.0.0.1";
const R9V_DEFAULT_PORT: u16 = 8004;
const R9V_DEFAULT_DEVICES: &str = "0,1";
const R9V_DEFAULT_CONTEXT: u32 = 131072;
const R9V_DEFAULT_BATCH: u32 = 1024;
const R9V_DEFAULT_SEQUENCES: u32 = 1;
const R9V_DEFAULT_KV_BYTES: u64 = 2_285_670_400;
const R9V_DEFAULT_EXPERT_CACHE_SLOTS: u32 = 16;
const R9V_DEFAULT_OFFLOAD: &str = "112.5";
const R9V_DEFAULT_OFFLOAD_DEVICES: &str = "112.5,112.5";
const R9V_DEFAULT_SERVED_MODEL: &str = "qwen3.8-flash-next";

/// Upstream's own `RESERVED_ARGS` set (`runner.py`) — `extra_args` tokens
/// (shlex-split, `--flag=value` and bare `--flag` both checked via the
/// substring before the first `=`) may not override any of these, since
/// they're already controlled by r9v's own dedicated form fields or the
/// fixed TP2/MTP2/SSD-residency profile.
const R9V_RESERVED_ARGS: &[&str] = &[
    "--model",
    "--tokenizer",
    "--speculative-config",
    "--load-format",
    "--quantization",
    "--tensor-parallel-size",
    "-tp",
    "--pipeline-parallel-size",
    "-pp",
    "--max-model-len",
    "--max-num-seqs",
    "--max-num-batched-tokens",
    "--kv-cache-memory-bytes",
    "--cpu-offload-gb",
    "--cpu-offload-params",
    "--host",
    "--port",
    "--served-model-name",
    "--async-scheduling",
    "--no-async-scheduling",
    "--compilation-config",
    "--api-key",
];

/// `POST /api/server-mode/r9v/start` request body. Every tuning field is
/// optional (see the `R9V_DEFAULT_*` constants above) — a genuine
/// divergence from ds4/halogen/vllm's mostly-required shapes, driven by
/// upstream's own `DEFAULTS` dict, not a brainrouter simplification.
#[derive(Debug, Clone, Deserialize)]
pub struct StartR9vServerRequest {
    /// The single vendored r9v toolbox (`r9700-r9v-rocm-10-0` today).
    pub toolbox_id: String,
    /// Which already-downloaded r9v catalog package to serve.
    pub package_id: String,
    #[serde(default)]
    pub host: Option<String>,
    #[serde(default)]
    pub port: Option<u16>,
    /// Two distinct GPU device indices, e.g. `"0,1"`.
    #[serde(default)]
    pub devices: Option<String>,
    #[serde(default)]
    pub context: Option<u32>,
    #[serde(default)]
    pub batch: Option<u32>,
    #[serde(default)]
    pub sequences: Option<u32>,
    #[serde(default)]
    pub kv_bytes: Option<u64>,
    #[serde(default)]
    pub expert_cache_slots: Option<u32>,
    /// Logical CPU-offload budget in GB, kept as a string (not parsed into
    /// a float and reformatted) because the original text is what's
    /// written verbatim into the `R9V_CPU_OFFLOAD_GB` env var — matching
    /// upstream, which validates-then-preserves rather than round-trips
    /// through `float`.
    #[serde(default)]
    pub offload: Option<String>,
    /// Two comma-separated per-device offload budgets, e.g. `"112.5,112.5"`
    /// — same preserve-the-string rationale as `offload`.
    #[serde(default)]
    pub offload_devices: Option<String>,
    /// The `served-model-name` reported by the OpenAI-compatible API.
    #[serde(default)]
    pub served_model: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub extra_args: Option<String>,
}

/// Same as [`resolve_halogen_toolbox`]/[`resolve_ds4_toolbox`], plus the
/// owning platform id — [`build_r9v_server_command`]'s upstream-mirrored
/// `platform_id != "r9700"` check needs it, same rationale as halogen's.
fn resolve_r9v_toolbox(toolbox_id: &str) -> Result<(ToolboxDefinition, RuntimeProfile, String), ServerModeError> {
    let catalog = load_typed_toolbox_catalog()?;
    let (tb, profile) = resolve_toolbox_for_server(&catalog, SupportedServingBackend::R9v, toolbox_id)?;
    let platform_id = catalog
        .platform_id_for_toolbox(toolbox_id)
        .ok_or_else(|| {
            ServerModeError::Internal(format!("toolbox `{toolbox_id}` is not listed under any catalog platform"))
        })?
        .to_string();
    Ok((tb, profile, platform_id))
}

/// Builds the podman argument list for `podman <args>` (no leading
/// `"podman"` binary name), mirroring `runner.py::build_server_cmd()`'s
/// full validation battery and command shape byte-for-byte, with `-d
/// --name brainrouter-r9v-server` replacing `--rm -it --name
/// <cockpit-name>` (same detached-not-foreground departure as the other
/// three backends) and brainrouter's own ownership `--label`s inserted
/// after `--security-opt label=disable` (mirroring vllm's label
/// placement, since r9v's own `--user 0:0` sits where vllm's
/// `--userns=keep-id` does).
///
/// Applies [`clean_engine_args_for_server`] before [`upgrade_groups_for_podman`]
/// even though upstream's own `build_server_cmd()` does not call an
/// equivalent of the former — a deliberate, minor non-literal-port choice
/// (§15a): the cleaning step is idempotent/safe and consistent with
/// brainrouter's own established headless-server invariant used by every
/// other backend's builder. It is a no-op in practice today regardless,
/// since the vendored `amd-rocm` runtime profile's `engine_args` carries
/// no `--group-add sudo`.
pub fn build_r9v_server_command(
    toolbox_image: &str,
    runtime_profile: &RuntimeProfile,
    platform_id: &str,
    package: &ResolvedR9vPackage,
    req: &StartR9vServerRequest,
) -> Result<Vec<String>, ServerModeError> {
    if platform_id != "r9700" {
        return Err(ServerModeError::Validation(
            "r9v is tested only on the AMD Radeon AI PRO R9700 platform (platform_id must be `r9700`)".to_string(),
        ));
    }
    if !package.ple_ready {
        return Err(ServerModeError::Validation(
            "the per-layer-embedding (PLE) file has not been prepared for this package yet — use \
             the Models tab's \"Prepare PLE\" action first"
                .to_string(),
        ));
    }

    // Mirrors upstream's `re.fullmatch(r"\d+,\d+", devices)` (exactly two
    // all-digit groups, nothing else) plus the follow-up "must parse to
    // two distinct integers" check.
    let devices = req.devices.as_deref().unwrap_or(R9V_DEFAULT_DEVICES).trim().to_string();
    let device_parts: Vec<&str> = devices.split(',').collect();
    let is_digits = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
    let devices_valid = device_parts.len() == 2
        && device_parts.iter().all(|p| is_digits(p))
        && device_parts[0].parse::<u64>().ok() != device_parts[1].parse::<u64>().ok();
    if !devices_valid {
        return Err(ServerModeError::Validation(
            "devices must be two distinct non-negative integers, e.g. \"0,1\"".to_string(),
        ));
    }

    let port = req.port.unwrap_or(R9V_DEFAULT_PORT);
    if port == 0 {
        return Err(ServerModeError::Validation("port must be nonzero".to_string()));
    }
    let context = req.context.unwrap_or(R9V_DEFAULT_CONTEXT);
    if context == 0 || context > 262_144 {
        return Err(ServerModeError::Validation("context must be between 1 and 262144".to_string()));
    }
    let batch = req.batch.unwrap_or(R9V_DEFAULT_BATCH);
    if batch == 0 || batch > 131_072 {
        return Err(ServerModeError::Validation("batch must be between 1 and 131072".to_string()));
    }
    let sequences = req.sequences.unwrap_or(R9V_DEFAULT_SEQUENCES);
    if sequences == 0 || sequences > 16 {
        return Err(ServerModeError::Validation("sequences must be between 1 and 16".to_string()));
    }
    if batch < sequences {
        return Err(ServerModeError::Validation("batch must be at least sequences".to_string()));
    }
    let kv_bytes = req.kv_bytes.unwrap_or(R9V_DEFAULT_KV_BYTES);
    if kv_bytes == 0 || kv_bytes > 32 * 1024 * 1024 * 1024 {
        return Err(ServerModeError::Validation("kv_bytes must be between 1 and 34359738368 (32 GiB)".to_string()));
    }
    let expert_cache_slots = req.expert_cache_slots.unwrap_or(R9V_DEFAULT_EXPERT_CACHE_SLOTS);
    if expert_cache_slots > 16 {
        return Err(ServerModeError::Validation("expert_cache_slots must be between 0 and 16".to_string()));
    }

    let offload = req.offload.as_deref().unwrap_or(R9V_DEFAULT_OFFLOAD).trim().to_string();
    let offload_devices = req.offload_devices.as_deref().unwrap_or(R9V_DEFAULT_OFFLOAD_DEVICES).trim().to_string();
    let offload_device_parts: Vec<&str> = offload_devices.split(',').collect();
    if offload_device_parts.len() != 2 {
        return Err(ServerModeError::Validation(
            "offload_devices must be two comma-separated values, e.g. \"112.5,112.5\"".to_string(),
        ));
    }
    for value in std::iter::once(offload.as_str()).chain(offload_device_parts.iter().copied()) {
        match value.parse::<f64>() {
            Ok(f) if f.is_finite() && f >= 0.0 => {}
            _ => {
                return Err(ServerModeError::Validation(
                    "offload and offload_devices must be non-negative finite numbers".to_string(),
                ));
            }
        }
    }

    let host_raw = req.host.as_deref().unwrap_or(R9V_DEFAULT_HOST).trim().to_string();
    let host_for_parse = if host_raw.eq_ignore_ascii_case("localhost") { "127.0.0.1".to_string() } else { host_raw };
    let addr: std::net::IpAddr = host_for_parse
        .parse()
        .map_err(|_| ServerModeError::Validation("host must be an IP address or \"localhost\"".to_string()))?;
    let binding = match addr {
        std::net::IpAddr::V6(_) => format!("[{addr}]"),
        std::net::IpAddr::V4(_) => addr.to_string(),
    };

    let served_model = req.served_model.as_deref().unwrap_or(R9V_DEFAULT_SERVED_MODEL).trim().to_string();
    if served_model.is_empty() || served_model.chars().any(char::is_whitespace) {
        return Err(ServerModeError::Validation(
            "served_model must be non-empty and contain no whitespace".to_string(),
        ));
    }

    let extra_args: Vec<String> = match req.extra_args.as_deref().filter(|s| !s.trim().is_empty()) {
        Some(s) => shlex::split(s)
            .ok_or_else(|| ServerModeError::Validation("extra_args is not valid shell-quoted text".to_string()))?,
        None => Vec::new(),
    };
    if let Some(bad) = extra_args
        .iter()
        .find(|a| R9V_RESERVED_ARGS.contains(&a.split('=').next().unwrap_or(a.as_str())))
    {
        return Err(ServerModeError::Validation(format!(
            "extra_args cannot override `{bad}` — it is controlled by a dedicated form field or the fixed r9v profile"
        )));
    }

    // Upstream's own cache-directory separation safety check
    // (`runner.py::build_server_cmd`): `model == cache or model in
    // cache.parents or ple == cache` → reject. `starts_with` covers both
    // the equality and ancestor cases in one comparison.
    if package.cache_dir.starts_with(&package.models_dir) || package.ple_dir == package.cache_dir {
        return Err(ServerModeError::Validation(
            "cache_dir must be a separate directory outside of models_dir and distinct from ple_dir".to_string(),
        ));
    }

    let engine_args = upgrade_groups_for_podman(&clean_engine_args_for_server(&runtime_profile.engine_args));

    let mut args: Vec<String> = vec![
        "run".to_string(),
        "-d".to_string(),
        "--name".to_string(),
        R9V_SERVER_CONTAINER_NAME.to_string(),
        "--runtime".to_string(),
        "crun".to_string(),
    ];
    args.extend(engine_args);
    // r9v runs the container as root, unlike ds4/halogen/vllm's
    // `--userns=keep-id` — confirmed against upstream's own
    // `build_server_cmd()`, not an oversight.
    args.extend(["--user".to_string(), "0:0".to_string()]);
    args.push("--ipc=host".to_string());
    args.extend(["--security-opt".to_string(), "label=disable".to_string()]);

    args.extend(["--label".to_string(), format!("{LABEL_MANAGED}=true")]);
    args.extend(["--label".to_string(), format!("{LABEL_SERVER_BACKEND}=r9v")]);
    args.extend(["--label".to_string(), format!("{LABEL_SERVER_MODEL}={}", req.package_id)]);

    args.extend(["-p".to_string(), format!("{binding}:{port}:8000")]);
    args.extend(["-v".to_string(), format!("{}:/models:ro", package.models_dir.display())]);
    args.extend([
        "-v".to_string(),
        format!(
            "{}:/ple/{}:ro",
            package.ple_dir.join(&package.ple_filename).display(),
            package.ple_filename
        ),
    ]);
    args.extend(["-v".to_string(), format!("{}:/cache", package.cache_dir.display())]);

    for (key, value) in [
        ("R9V_VISIBLE_DEVICES", devices.clone()),
        ("R9V_MAX_MODEL_LEN", context.to_string()),
        ("R9V_MAX_NUM_BATCHED_TOKENS", batch.to_string()),
        ("R9V_MAX_NUM_SEQS", sequences.to_string()),
        ("R9V_KV_CACHE_MEMORY_BYTES", kv_bytes.to_string()),
        ("R9V_CPU_OFFLOAD_GB", offload),
        ("R9V_CPU_OFFLOAD_GB_BY_DEVICE", offload_devices),
        ("R9V_SERVED_MODEL_NAME", served_model),
        ("R9V_TENSOR_PARALLEL_SIZE", "2".to_string()),
        ("R9V_MTP_SPEC_TOKENS", "2".to_string()),
        ("R9V_PLE_RESIDENCY_MODE", "ssd".to_string()),
        ("R9V_PLE_WORKER_TIMING", "1".to_string()),
        ("R9V_SERIALIZE_EXPERT_LOAD", "1".to_string()),
        ("R9V_TIERED_EXPERT_CACHE_SLOTS", expert_cache_slots.to_string()),
    ] {
        args.extend(["-e".to_string(), format!("{key}={value}")]);
    }
    args.push(toolbox_image.to_string());
    args.push("r9v-serve".to_string());
    if let Some(key) = req.api_key.as_deref().filter(|k| !k.is_empty()) {
        args.extend(["--api-key".to_string(), key.to_string()]);
    }
    args.extend(extra_args);

    Ok(args)
}

/// `POST /api/server-mode/r9v/start`: resolves the toolbox + already-
/// downloaded (and PLE-prepared) package, force-removes any pre-existing
/// container of the fixed name, then runs the built command. Does not
/// persist run-time settings into cockpit's config.json — same rationale
/// as vllm's `start_vllm_server`, only [`save_r9v_paths`] does.
pub async fn start_r9v_server(req: &StartR9vServerRequest) -> Result<(), ServerModeError> {
    let (tb, profile, platform_id) = resolve_r9v_toolbox(&req.toolbox_id)?;
    let package = model_downloads::resolve_downloaded_r9v_package(&req.package_id)?;
    let args = build_r9v_server_command(&tb.image, &profile, &platform_id, &package, req)?;

    let _ = tokio::process::Command::new("podman")
        .args(["rm", "-f", R9V_SERVER_CONTAINER_NAME])
        .output()
        .await;

    let out = tokio::process::Command::new("podman")
        .args(&args)
        .output()
        .await
        .map_err(|e| ServerModeError::Internal(format!("failed to exec podman: {e}")))?;
    if out.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&out.stderr);
        Err(ServerModeError::Internal(format!("podman run failed: {}", stderr.trim())))
    }
}

/// `POST /api/server-mode/r9v/stop` — same graceful-stop-then-rm contract
/// as the other three backends.
pub async fn stop_r9v_server() -> Result<(), ServerModeError> {
    let _ = tokio::process::Command::new("podman")
        .args(["stop", "--time", "10", R9V_SERVER_CONTAINER_NAME])
        .output()
        .await;
    let out = tokio::process::Command::new("podman")
        .args(["rm", "-f", R9V_SERVER_CONTAINER_NAME])
        .output()
        .await
        .map_err(|e| ServerModeError::Internal(format!("failed to exec podman: {e}")))?;
    if out.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&out.stderr);
        Err(ServerModeError::Internal(format!("podman rm failed: {}", stderr.trim())))
    }
}

pub async fn r9v_server_status() -> ServerStatus {
    container_server_status("r9v", R9V_SERVER_CONTAINER_NAME, LABEL_SERVER_MODEL).await
}

/// `POST /api/server-mode/r9v/paths` request body — the r9v analogue of
/// [`VllmCachePathsRequest`], three keys matching
/// `model_downloads::R9vPaths`/`effective_r9v_paths()` exactly:
/// `models_dir`/`ple_dir`/`cache_dir`.
#[derive(Debug, Clone, Deserialize)]
pub struct R9vPathsRequest {
    #[serde(default)]
    pub models_dir: Option<String>,
    #[serde(default)]
    pub ple_dir: Option<String>,
    #[serde(default)]
    pub cache_dir: Option<String>,
}

/// Creates each supplied path, then persists it into cockpit's shared
/// config.json via [`crate::cockpit_config::apply_backend_setting_values`]
/// — same pattern as [`save_vllm_cache_paths`].
pub fn save_r9v_paths(req: &R9vPathsRequest) -> Result<(), ServerModeError> {
    let provided: Vec<(&str, &str)> = [
        ("models_dir", req.models_dir.as_deref()),
        ("ple_dir", req.ple_dir.as_deref()),
        ("cache_dir", req.cache_dir.as_deref()),
    ]
    .into_iter()
    .filter_map(|(k, v)| v.map(|v| (k, v)))
    .collect();
    if provided.is_empty() {
        return Err(ServerModeError::Validation("at least one path must be provided".to_string()));
    }
    for (_, raw) in &provided {
        let expanded = model_downloads::expand_tilde(raw);
        std::fs::create_dir_all(&expanded)
            .map_err(|e| ServerModeError::Internal(format!("failed to create directory {}: {e}", expanded.display())))?;
    }
    let updates: Vec<(&str, Value)> = provided.into_iter().map(|(k, v)| (k, Value::String(v.to_string()))).collect();
    crate::cockpit_config::apply_backend_setting_values("r9v", updates)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toolbox_catalog::RuntimeProfile;


    fn amd_rocm_profile() -> RuntimeProfile {
        // Exact vendored `amd-rocm` runtime_profile used by every real ds4
        // strix-halo toolbox entry (`assets/cockpit-catalog/toolboxes.json`).
        RuntimeProfile {
            id: "amd-rocm".to_string(),
            engine_args: vec![
                "--device".to_string(),
                "/dev/dri".to_string(),
                "--device".to_string(),
                "/dev/kfd".to_string(),
                "--group-add".to_string(),
                "video".to_string(),
                "--group-add".to_string(),
                "render".to_string(),
                "--security-opt".to_string(),
                "seccomp=unconfined".to_string(),
            ],
        }
    }

    fn model() -> ResolvedDs4Model {
        ResolvedDs4Model {
            models_dir: std::path::PathBuf::from("/data/ds4-models"),
            filename: "DeepSeek-V4-Flash-Q4.gguf".to_string(),
        }
    }

    fn req() -> StartDs4ServerRequest {
        StartDs4ServerRequest {
            toolbox_id: "strix-halo-ds4-rocm-10-0".to_string(),
            model_id: "deepseek-v4-flash-q4".to_string(),
            ctx: 126000,
            host: "localhost".to_string(),
            port: 8000,
            custom_args: None,
        }
    }

    #[test]
    fn upgrade_groups_for_podman_collapses_video_and_render_into_keep_groups() {
        let cleaned = clean_engine_args_for_server(&amd_rocm_profile().engine_args);
        let upgraded = upgrade_groups_for_podman(&cleaned);
        assert_eq!(
            upgraded,
            vec![
                "--device",
                "/dev/dri",
                "--device",
                "/dev/kfd",
                "--group-add",
                "keep-groups",
                "--security-opt",
                "seccomp=unconfined",
            ]
        );
    }

    #[test]
    fn clean_engine_args_for_server_drops_group_add_sudo_both_forms() {
        let two_token = vec!["--group-add".to_string(), "sudo".to_string(), "--device".to_string(), "/dev/dri".to_string()];
        assert_eq!(clean_engine_args_for_server(&two_token), vec!["--device".to_string(), "/dev/dri".to_string()]);

        let equals_form = vec!["--group-add=sudo".to_string(), "--device".to_string(), "/dev/dri".to_string()];
        assert_eq!(clean_engine_args_for_server(&equals_form), vec!["--device".to_string(), "/dev/dri".to_string()]);
    }

    #[test]
    fn build_ds4_server_command_matches_upstream_standalone_shape() {
        let cmd = build_ds4_server_command("docker.io/kyuz0/strix-halo-ds4-toolbox:rocm-10.0", &amd_rocm_profile(), &model(), &req())
            .expect("must build");

        assert_eq!(cmd[0..4], ["run", "-d", "--name", DS4_SERVER_CONTAINER_NAME]);
        // engine_args (post group-upgrade) come right after --name.
        assert_eq!(cmd[4..10], ["--device", "/dev/dri", "--device", "/dev/kfd", "--group-add", "keep-groups"]);
        assert!(cmd.contains(&"--ipc=host".to_string()));
        assert!(cmd.windows(2).any(|w| w == ["--cap-add", "SYS_PTRACE"]));
        assert!(cmd.windows(2).any(|w| w == ["--env", "DS4_ROCM_ENABLE_MXFP4_TILE4=1"]));
        assert!(cmd.windows(2).any(|w| w == ["--env", "DS4_ROCM_MXFP4_DOWN_RGROUP=4"]));
        assert!(cmd.windows(2).any(|w| w == ["--security-opt", "label=disable"]));
        assert!(cmd.contains(&"--userns=keep-id".to_string()));
        assert!(cmd.windows(2).any(|w| w == ["--label", "io.brainrouter.managed=true"]));
        assert!(cmd.windows(2).any(|w| w == ["--label", "io.brainrouter.server_backend=ds4"]));
        assert!(cmd.windows(2).any(|w| w == ["--label", "io.brainrouter.server_model=deepseek-v4-flash-q4"]));
        // host=localhost -> bind to 127.0.0.1, not the in-container --host.
        assert!(cmd.windows(2).any(|w| w == ["-p", "127.0.0.1:8000:8000"]));
        assert!(cmd.windows(2).any(|w| w == ["-v", "/data/ds4-models:/models:ro"]));
        assert!(cmd.contains(&"docker.io/kyuz0/strix-halo-ds4-toolbox:rocm-10.0".to_string()));
        // server_binary + core args, always --host 0.0.0.0 regardless of request.host.
        let tail = &cmd[cmd.len() - 9..];
        assert_eq!(
            tail,
            &[
                "ds4-server",
                "-m",
                "/models/DeepSeek-V4-Flash-Q4.gguf",
                "--ctx",
                "126000",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ]
        );
    }

    #[test]
    fn build_ds4_server_command_binds_all_interfaces_when_host_is_0_0_0_0() {
        let mut r = req();
        r.host = "0.0.0.0".to_string();
        let cmd = build_ds4_server_command("img", &amd_rocm_profile(), &model(), &r).expect("must build");
        assert!(cmd.windows(2).any(|w| w == ["-p", "8000:8000"]));
    }

    #[test]
    fn build_ds4_server_command_appends_shlex_split_custom_args() {
        let mut r = req();
        r.custom_args = Some("--mtp --extra-flag value".to_string());
        let cmd = build_ds4_server_command("img", &amd_rocm_profile(), &model(), &r).expect("must build");
        let tail = &cmd[cmd.len() - 3..];
        assert_eq!(tail, &["--mtp", "--extra-flag", "value"]);
    }

    #[test]
    fn build_ds4_server_command_rejects_unterminated_quote_in_custom_args() {
        let mut r = req();
        r.custom_args = Some("--mtp \"unterminated".to_string());
        let err = build_ds4_server_command("img", &amd_rocm_profile(), &model(), &r).unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn resolve_ds4_toolbox_finds_the_real_vendored_entry_and_its_runtime_profile() {
        let (tb, profile) = resolve_ds4_toolbox("strix-halo-ds4-rocm-10-0").expect("must resolve");
        assert_eq!(tb.id, "strix-halo-ds4-rocm-10-0");
        assert_eq!(profile.id, "amd-rocm");
        assert!(!profile.engine_args.is_empty());
    }

    #[test]
    fn resolve_ds4_toolbox_rejects_unknown_id() {
        let err = resolve_ds4_toolbox("no-such-toolbox").unwrap_err();
        assert_eq!(err.status(), 404);
    }

    #[test]
    fn resolve_ds4_toolbox_rejects_non_ds4_backend() {
        // A real llama_cpp toolbox id must not resolve through the ds4-only lookup.
        let err = resolve_ds4_toolbox("strix-halo-llama-rocm-10-0").unwrap_err();
        assert_eq!(err.status(), 404);
    }

    #[test]
    fn resolve_toolbox_for_server_accepts_experimental_not_just_supported() {
        // Regression test for the bug this PR fixes: cockpit's own TUI
        // filter is `feature_state("server") != "unavailable"` (verified by
        // reading `halogen/server.py::set_platform()` in full) — an
        // `Experimental` toolbox is still selectable, only badged
        // `[experimental]`. The vendored halogen toolbox
        // (`strix-halo-halogen-flash`) is `server: experimental`, so this
        // also doubles as the real-world case that motivated the fix.
        let (tb, _profile) = resolve_halogen_toolbox("strix-halo-halogen-flash")
            .map(|(tb, profile, _platform_id)| (tb, profile))
            .expect("an Experimental-server toolbox must still resolve");
        assert_eq!(tb.features.server, toolbox_catalog::FeatureState::Experimental);
    }

    fn halogen_strix_halo_profile() -> RuntimeProfile {
        // Exact vendored `halogen-strix-halo` runtime_profile
        // (`assets/cockpit-catalog/toolboxes.json`).
        RuntimeProfile {
            id: "halogen-strix-halo".to_string(),
            engine_args: vec![
                "--pull=always".to_string(),
                "--device".to_string(),
                "/dev/kfd".to_string(),
                "--device".to_string(),
                "/dev/dri".to_string(),
                "--group-add".to_string(),
                "video".to_string(),
                "--group-add".to_string(),
                "render".to_string(),
                "--security-opt".to_string(),
                "seccomp=unconfined".to_string(),
                "--ipc=host".to_string(),
                "--ulimit".to_string(),
                "memlock=-1:-1".to_string(),
            ],
        }
    }

    fn halogen_bundle() -> ResolvedHalogenBundle {
        ResolvedHalogenBundle {
            models_dir: std::path::PathBuf::from("/data/halogen-models"),
            checkpoint: "qwen38-flash-next-w4b-quality.safetensors".to_string(),
            overlay: "overlay-quality.safetensors".to_string(),
            tokenizer_dir: "tokenizer".to_string(),
            vision_tower: None,
        }
    }

    fn halogen_req() -> StartHalogenServerRequest {
        StartHalogenServerRequest {
            toolbox_id: "strix-halo-halogen-flash".to_string(),
            bundle_id: "qwen38-flash-next-w4b-quality".to_string(),
            host: "localhost".to_string(),
            port: 8731,
            context_size: 262144,
            kv_pool_positions: 524288,
            kv_slots: 4,
            prompt_cache: "2".to_string(),
        }
    }

    #[test]
    fn resolve_halogen_toolbox_finds_the_real_vendored_entry_and_platform() {
        let (tb, profile, platform_id) =
            resolve_halogen_toolbox("strix-halo-halogen-flash").expect("must resolve");
        assert_eq!(tb.id, "strix-halo-halogen-flash");
        assert_eq!(profile.id, "halogen-strix-halo");
        assert_eq!(platform_id, "strix-halo");
    }

    #[test]
    fn resolve_halogen_toolbox_rejects_non_halogen_backend() {
        let err = resolve_halogen_toolbox("strix-halo-ds4-rocm-10-0").unwrap_err();
        assert_eq!(err.status(), 404);
    }

    #[test]
    fn validate_halogen_host_normalizes_localhost_and_rejects_bare_hostnames() {
        assert_eq!(validate_halogen_host("localhost").unwrap(), "127.0.0.1");
        assert_eq!(validate_halogen_host("LOCALHOST").unwrap(), "127.0.0.1");
        assert_eq!(validate_halogen_host("192.168.1.5").unwrap(), "192.168.1.5");
        assert_eq!(validate_halogen_host("::1").unwrap(), "[::1]");
        // Unlike ds4's `host` (passed through unvalidated), halogen's own
        // `build_server_cmd` requires a literal IP address — a bare
        // hostname is rejected.
        assert!(validate_halogen_host("my-host.example.com").is_err());
    }

    #[test]
    fn build_halogen_server_command_matches_upstream_env_var_shape() {
        let cmd = build_halogen_server_command(
            "ghcr.io/peonist-ai/halogen-flash-server:latest",
            &halogen_strix_halo_profile(),
            "strix-halo",
            &halogen_bundle(),
            &halogen_req(),
        )
        .expect("must build");

        assert_eq!(cmd[0..4], ["run", "-d", "--name", HALOGEN_SERVER_CONTAINER_NAME]);
        // engine_args, post group-upgrade (video+render -> keep-groups),
        // come right after --name — no clean_engine_args_for_server call.
        assert_eq!(
            cmd[4..14],
            [
                "--pull=always",
                "--device",
                "/dev/kfd",
                "--device",
                "/dev/dri",
                "--group-add",
                "keep-groups",
                "--security-opt",
                "seccomp=unconfined",
                "--ipc=host",
            ]
        );
        assert!(cmd.windows(2).any(|w| w == ["--label", "io.brainrouter.managed=true"]));
        assert!(cmd.windows(2).any(|w| w == ["--label", "io.brainrouter.server_backend=halogen"]));
        assert!(cmd.windows(2).any(|w| w == [
            "--label",
            "io.brainrouter.server_model=qwen38-flash-next-w4b-quality"
        ]));
        assert!(cmd.windows(2).any(|w| w == ["-p", "127.0.0.1:8731:8731"]));
        assert!(cmd.windows(2).any(|w| w == ["-v", "/data/halogen-models:/models:ro"]));
        assert!(cmd.windows(2).any(|w| w == [
            "-e",
            "HALOGEN_CHECKPOINT=/models/qwen38-flash-next-w4b-quality.safetensors"
        ]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "HALOGEN_CK_OVERLAY=/models/overlay-quality.safetensors"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "HALOGEN_TOKENIZER=/models/tokenizer"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "HALOGEN_API_PORT=8731"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "HALOGEN_CTX=262144"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "HALOGEN_KV_POOL_POSITIONS=524288"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "HALOGEN_KV_SLOTS=4"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "HALOGEN_PROMPT_CACHE=2"]));
        // No CLI args after the image at all — the entrypoint reads the env vars.
        assert_eq!(cmd.last().unwrap(), "ghcr.io/peonist-ai/halogen-flash-server:latest");
    }

    #[test]
    fn build_halogen_server_command_sets_vision_tower_env_only_when_present() {
        let mut bundle = halogen_bundle();
        bundle.vision_tower = Some("vision-tower.safetensors".to_string());
        let cmd =
            build_halogen_server_command("img", &halogen_strix_halo_profile(), "strix-halo", &bundle, &halogen_req())
                .expect("must build");
        assert!(cmd.windows(2).any(|w| w == ["-e", "HALOGEN_VISION_TOWER=/models/vision-tower.safetensors"]));

        let cmd_no_vision = build_halogen_server_command(
            "img",
            &halogen_strix_halo_profile(),
            "strix-halo",
            &halogen_bundle(),
            &halogen_req(),
        )
        .expect("must build");
        assert!(!cmd_no_vision.iter().any(|a| a.starts_with("HALOGEN_VISION_TOWER")));
    }

    #[test]
    fn build_halogen_server_command_rejects_non_strix_halo_platform() {
        let err = build_halogen_server_command(
            "img",
            &halogen_strix_halo_profile(),
            "r9700",
            &halogen_bundle(),
            &halogen_req(),
        )
        .unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn build_halogen_server_command_rejects_context_size_out_of_bounds() {
        let mut r = halogen_req();
        r.context_size = 300_000;
        let err = build_halogen_server_command("img", &halogen_strix_halo_profile(), "strix-halo", &halogen_bundle(), &r)
            .unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn build_halogen_server_command_rejects_kv_pool_below_context_size() {
        let mut r = halogen_req();
        r.kv_pool_positions = r.context_size - 1;
        let err = build_halogen_server_command("img", &halogen_strix_halo_profile(), "strix-halo", &halogen_bundle(), &r)
            .unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn build_halogen_server_command_rejects_zero_kv_slots() {
        let mut r = halogen_req();
        r.kv_slots = 0;
        let err = build_halogen_server_command("img", &halogen_strix_halo_profile(), "strix-halo", &halogen_bundle(), &r)
            .unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn build_halogen_server_command_rejects_invalid_prompt_cache() {
        let mut r = halogen_req();
        r.prompt_cache = "3".to_string();
        let err = build_halogen_server_command("img", &halogen_strix_halo_profile(), "strix-halo", &halogen_bundle(), &r)
            .unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn build_halogen_server_command_rejects_models_dir_containing_colon() {
        let mut bundle = halogen_bundle();
        bundle.models_dir = std::path::PathBuf::from("/data/weird:dir");
        let err =
            build_halogen_server_command("img", &halogen_strix_halo_profile(), "strix-halo", &bundle, &halogen_req())
                .unwrap_err();
        assert_eq!(err.status(), 400);
    }

    // ── PR9: vllm Server Mode (§14) ───────────────────────────────────────

    fn vllm_req() -> StartVllmServerRequest {
        StartVllmServerRequest {
            toolbox_id: "strix-vllm-latest".to_string(),
            model_id: Some("vllm-meta-llama-meta-llama-3-1-8b-instruct".to_string()),
            custom_repo: None,
            host: "localhost".to_string(),
            port: 8000,
            tensor_parallel: 1,
            max_num_seqs: 64,
            max_model_len: "auto".to_string(),
            gpu_memory_utilization: 0.9,
            attention_backend: None,
            enforce_eager: None,
            dtype: "auto".to_string(),
            api_key: None,
            extra_args: None,
            reset_caches: false,
        }
    }

    fn vllm_cache_dirs() -> VllmCacheDirs {
        VllmCacheDirs {
            huggingface: PathBuf::from("/data/cache/huggingface"),
            vllm: PathBuf::from("/data/cache/vllm"),
            triton: PathBuf::from("/data/cache/triton"),
            aiter: PathBuf::from("/data/cache/aiter"),
        }
    }

    fn plain_policy() -> VllmEffectivePolicy {
        VllmEffectivePolicy {
            valid_tp: vec![1, 2],
            trust_remote: false,
            ctx: None,
            enforce_eager: false,
            attention_backend: None,
            env: Vec::new(),
            extra_flags: Vec::new(),
        }
    }

    #[test]
    fn resolve_vllm_toolbox_finds_the_real_vendored_entry_and_its_runtime_profile() {
        let (tb, profile) = resolve_vllm_toolbox("strix-vllm-latest").expect("must resolve");
        assert_eq!(tb.id, "strix-vllm-latest");
        assert_eq!(profile.id, "amd-rocm-keep-groups");
    }

    #[test]
    fn resolve_vllm_toolbox_rejects_non_vllm_backend() {
        let err = resolve_vllm_toolbox("strix-halo-ds4-rocm-10-0").unwrap_err();
        assert_eq!(err.status(), 404);
    }

    #[test]
    fn resolve_vllm_model_and_base_policy_rejects_both_and_neither() {
        let mut r = vllm_req();
        r.custom_repo = Some("owner/model".to_string());
        let err = resolve_vllm_model_and_base_policy(&r).unwrap_err();
        assert_eq!(err.status(), 400);

        r.model_id = None;
        r.custom_repo = None;
        let err = resolve_vllm_model_and_base_policy(&r).unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn resolve_vllm_model_and_base_policy_resolves_real_catalog_entry_with_explicit_null_attention_backend() {
        let mut r = vllm_req();
        r.model_id = Some("vllm-deepseek-ai-deepseek-v4-flash-0731".to_string());
        let (repo, policy_map) = resolve_vllm_model_and_base_policy(&r).expect("must resolve");
        assert_eq!(repo, "deepseek-ai/DeepSeek-V4-Flash-0731");
        // Real vendored entry has `"attention_backend": null` — present but
        // explicitly null, distinct from absent (§14 item 2/3).
        assert_eq!(policy_map.get("attention_backend"), Some(&Value::Null));
        assert_eq!(policy_map.get("ctx"), Some(&Value::String("262144".to_string())));
        assert_eq!(policy_map.get("enforce_eager"), Some(&Value::Bool(true)));
        assert!(policy_map.get("env").is_some());
    }

    #[test]
    fn resolve_vllm_model_and_base_policy_rejects_unknown_id() {
        let mut r = vllm_req();
        r.model_id = Some("no-such-model".to_string());
        let err = resolve_vllm_model_and_base_policy(&r).unwrap_err();
        assert_eq!(err.status(), 404);
    }

    #[test]
    fn resolve_vllm_model_and_base_policy_uses_generic_default_for_custom_repo() {
        let mut r = vllm_req();
        r.model_id = None;
        r.custom_repo = Some("acme/custom-model".to_string());
        let (repo, policy_map) = resolve_vllm_model_and_base_policy(&r).expect("must resolve");
        assert_eq!(repo, "acme/custom-model");
        assert_eq!(policy_map.get("valid_tp"), Some(&serde_json::json!([1, 2])));
        assert_eq!(policy_map.get("attention_backend"), Some(&Value::String("TRITON_ATTN".to_string())));
    }

    #[test]
    fn apply_vllm_toolbox_policy_overrides_shallow_merges_and_replaces_whole_keys() {
        let mut base = Map::new();
        base.insert("valid_tp".to_string(), serde_json::json!([1, 2]));
        base.insert("attention_backend".to_string(), Value::String("TRITON_ATTN".to_string()));
        base.insert("trust_remote".to_string(), Value::Bool(false));

        let backend_config = serde_json::json!({
            "policy_overrides": {
                "valid_tp": [1],
                "attention_backend": Value::Null,
            }
        });
        let merged = apply_vllm_toolbox_policy_overrides(base, Some(&backend_config));
        assert_eq!(merged.get("valid_tp"), Some(&serde_json::json!([1])));
        assert_eq!(merged.get("attention_backend"), Some(&Value::Null));
        // Untouched key round-trips.
        assert_eq!(merged.get("trust_remote"), Some(&Value::Bool(false)));
    }

    #[test]
    fn resolve_vllm_effective_policy_distinguishes_absent_vs_null_vs_string_attention_backend() {
        let mut m = Map::new();
        assert_eq!(resolve_vllm_effective_policy(&m).attention_backend, None);

        m.insert("attention_backend".to_string(), Value::Null);
        assert_eq!(resolve_vllm_effective_policy(&m).attention_backend, Some(None));

        m.insert("attention_backend".to_string(), Value::String("ROCM_ATTN".to_string()));
        assert_eq!(resolve_vllm_effective_policy(&m).attention_backend, Some(Some("ROCM_ATTN".to_string())));
    }

    #[test]
    fn resolve_vllm_effective_policy_falls_back_to_valid_tp_one_when_absent_or_empty() {
        let m = Map::new();
        assert_eq!(resolve_vllm_effective_policy(&m).valid_tp, vec![1]);

        let mut m2 = Map::new();
        m2.insert("valid_tp".to_string(), serde_json::json!([]));
        assert_eq!(resolve_vllm_effective_policy(&m2).valid_tp, vec![1]);
    }

    #[test]
    fn build_vllm_server_command_matches_upstream_shape() {
        let cmd = build_vllm_server_command(
            "docker.io/kyuz0/vllm-therock-gfx1151:latest",
            &amd_rocm_profile(),
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            &plain_policy(),
            &vllm_cache_dirs(),
            Some("hf_abc123"),
            &vllm_req(),
        )
        .expect("must build");

        assert_eq!(cmd[0..4], ["run", "-d", "--name", VLLM_SERVER_CONTAINER_NAME]);
        assert_eq!(cmd[4..10], ["--device", "/dev/dri", "--device", "/dev/kfd", "--group-add", "keep-groups"]);
        assert!(cmd.windows(2).any(|w| w == ["--ipc=host", "--cap-add=SYS_PTRACE"]));
        assert!(cmd.windows(2).any(|w| w == ["--security-opt", "label=disable"]));
        assert!(cmd.contains(&"--userns=keep-id".to_string()));
        assert!(cmd.windows(2).any(|w| w == [
            "--label",
            "io.brainrouter.server_model=meta-llama/Meta-Llama-3.1-8B-Instruct"
        ]));
        assert!(cmd.windows(2).any(|w| w == ["-p", "127.0.0.1:8000:8000"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "HOME=/workspace"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "VLLM_CONFIG_ROOT=/workspace/.cache/vllm/config"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "TRITON_CACHE_DIR=/workspace/.cache/triton"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "TILELANG_CACHE_DIR=/workspace/.cache/triton/tilelang"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "VLLM_NO_USAGE_STATS=1"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "HF_TOKEN=hf_abc123"]));
        assert!(cmd.windows(2).any(|w| w == ["-v", "/data/cache/huggingface:/workspace/.cache/huggingface"]));
        assert!(cmd.windows(2).any(|w| w == ["-v", "/data/cache/vllm:/workspace/.cache/vllm"]));
        assert!(cmd.windows(2).any(|w| w == ["-v", "/data/cache/triton:/workspace/.cache/triton"]));
        assert!(cmd.windows(2).any(|w| w == ["-v", "/data/cache/aiter:/workspace/.aiter"]));
        assert!(cmd.windows(4).any(|w| w == [
            "docker.io/kyuz0/vllm-therock-gfx1151:latest",
            "vllm",
            "serve",
            "meta-llama/Meta-Llama-3.1-8B-Instruct"
        ]));
        assert!(cmd.windows(2).any(|w| w == ["--tensor-parallel-size", "1"]));
        assert!(cmd.windows(2).any(|w| w == ["--max-num-seqs", "64"]));
        // policy.ctx is None and max_model_len is "auto" -> resolves to the
        // literal string "auto", matching upstream exactly (never omitted).
        assert!(cmd.windows(2).any(|w| w == ["--max-model-len", "auto"]));
        assert!(cmd.windows(2).any(|w| w == ["--gpu-memory-utilization", "0.9"]));
        assert!(cmd.windows(2).any(|w| w == ["--dtype", "auto"]));
        assert!(cmd.windows(2).any(|w| w == ["--attention-backend", "TRITON_ATTN"]));
        assert!(!cmd.contains(&"--trust-remote-code".to_string()));
        assert!(!cmd.contains(&"--enforce-eager".to_string()));
    }

    #[test]
    fn build_vllm_server_command_uses_bare_hf_token_when_absent() {
        let cmd =
            build_vllm_server_command("img", &amd_rocm_profile(), "owner/model", &plain_policy(), &vllm_cache_dirs(), None, &vllm_req())
                .expect("must build");
        assert!(cmd.windows(2).any(|w| w == ["-e", "HF_TOKEN"]));
        assert!(!cmd.iter().any(|a| a.starts_with("HF_TOKEN=")));
    }

    #[test]
    fn build_vllm_server_command_resolves_max_model_len_from_policy_ctx_when_auto() {
        let mut policy = plain_policy();
        policy.ctx = Some("262144".to_string());
        let cmd = build_vllm_server_command("img", &amd_rocm_profile(), "owner/model", &policy, &vllm_cache_dirs(), None, &vllm_req())
            .expect("must build");
        assert!(cmd.windows(2).any(|w| w == ["--max-model-len", "262144"]));
    }

    #[test]
    fn build_vllm_server_command_explicit_max_model_len_overrides_policy_ctx() {
        let mut policy = plain_policy();
        policy.ctx = Some("262144".to_string());
        let mut r = vllm_req();
        r.max_model_len = "16384".to_string();
        let cmd = build_vllm_server_command("img", &amd_rocm_profile(), "owner/model", &policy, &vllm_cache_dirs(), None, &r)
            .expect("must build");
        assert!(cmd.windows(2).any(|w| w == ["--max-model-len", "16384"]));
    }

    #[test]
    fn build_vllm_server_command_rejects_invalid_max_model_len() {
        let mut r = vllm_req();
        r.max_model_len = "not-a-number".to_string();
        let err = build_vllm_server_command("img", &amd_rocm_profile(), "owner/model", &plain_policy(), &vllm_cache_dirs(), None, &r)
            .unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn build_vllm_server_command_emits_trust_remote_code_and_enforce_eager_and_env_and_extra_flags() {
        let policy = VllmEffectivePolicy {
            valid_tp: vec![1],
            trust_remote: true,
            ctx: None,
            enforce_eager: true,
            attention_backend: Some(Some("ROCM_ATTN".to_string())),
            env: vec![("VLLM_ROCM_USE_AITER".to_string(), "1".to_string())],
            extra_flags: vec!["--enable-auto-tool-choice".to_string(), "--tool-call-parser".to_string(), "llama3_json".to_string()],
        };
        let cmd = build_vllm_server_command("img", &amd_rocm_profile(), "owner/model", &policy, &vllm_cache_dirs(), None, &vllm_req())
            .expect("must build");
        assert!(cmd.contains(&"--trust-remote-code".to_string()));
        assert!(cmd.contains(&"--enforce-eager".to_string()));
        assert!(cmd.windows(2).any(|w| w == ["-e", "VLLM_ROCM_USE_AITER=1"]));
        assert!(cmd.windows(2).any(|w| w == ["--attention-backend", "ROCM_ATTN"]));
        assert!(cmd.windows(3).any(|w| w == ["--enable-auto-tool-choice", "--tool-call-parser", "llama3_json"]));
    }

    #[test]
    fn build_vllm_server_command_operator_attention_backend_override_wins_over_policy_default() {
        let mut policy = plain_policy();
        policy.attention_backend = Some(Some("TRITON_ATTN".to_string()));
        let mut r = vllm_req();
        r.attention_backend = Some("ROCM_AITER_UNIFIED_ATTN".to_string());
        let cmd = build_vllm_server_command("img", &amd_rocm_profile(), "owner/model", &policy, &vllm_cache_dirs(), None, &r)
            .expect("must build");
        assert!(cmd.windows(2).any(|w| w == ["--attention-backend", "ROCM_AITER_UNIFIED_ATTN"]));
    }

    #[test]
    fn build_vllm_server_command_omits_attention_backend_flag_when_model_specific() {
        let mut policy = plain_policy();
        policy.attention_backend = Some(None);
        let cmd = build_vllm_server_command("img", &amd_rocm_profile(), "owner/model", &policy, &vllm_cache_dirs(), None, &vllm_req())
            .expect("must build");
        assert!(!cmd.contains(&"--attention-backend".to_string()));
    }

    #[test]
    fn build_vllm_server_command_rejects_attention_backend_override_when_model_specific() {
        let mut policy = plain_policy();
        policy.attention_backend = Some(None);
        let mut r = vllm_req();
        r.attention_backend = Some("TRITON_ATTN".to_string());
        let err = build_vllm_server_command("img", &amd_rocm_profile(), "owner/model", &policy, &vllm_cache_dirs(), None, &r)
            .unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn build_vllm_server_command_rejects_tensor_parallel_not_in_valid_tp() {
        let mut r = vllm_req();
        r.tensor_parallel = 4;
        let err = build_vllm_server_command("img", &amd_rocm_profile(), "owner/model", &plain_policy(), &vllm_cache_dirs(), None, &r)
            .unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn build_vllm_server_command_rejects_zero_port_and_zero_max_num_seqs_and_bad_gpu_util() {
        let mut r = vllm_req();
        r.port = 0;
        assert_eq!(
            build_vllm_server_command("img", &amd_rocm_profile(), "owner/model", &plain_policy(), &vllm_cache_dirs(), None, &r)
                .unwrap_err()
                .status(),
            400
        );

        let mut r = vllm_req();
        r.max_num_seqs = 0;
        assert_eq!(
            build_vllm_server_command("img", &amd_rocm_profile(), "owner/model", &plain_policy(), &vllm_cache_dirs(), None, &r)
                .unwrap_err()
                .status(),
            400
        );

        let mut r = vllm_req();
        r.gpu_memory_utilization = 1.5;
        assert_eq!(
            build_vllm_server_command("img", &amd_rocm_profile(), "owner/model", &plain_policy(), &vllm_cache_dirs(), None, &r)
                .unwrap_err()
                .status(),
            400
        );

        let mut r = vllm_req();
        r.gpu_memory_utilization = 0.0;
        assert_eq!(
            build_vllm_server_command("img", &amd_rocm_profile(), "owner/model", &plain_policy(), &vllm_cache_dirs(), None, &r)
                .unwrap_err()
                .status(),
            400
        );
    }

    #[test]
    fn build_vllm_server_command_appends_shlex_split_extra_args() {
        let mut r = vllm_req();
        r.extra_args = Some("--speculative-config '{}'".to_string());
        let cmd = build_vllm_server_command("img", &amd_rocm_profile(), "owner/model", &plain_policy(), &vllm_cache_dirs(), None, &r)
            .expect("must build");
        let tail = &cmd[cmd.len() - 2..];
        assert_eq!(tail, &["--speculative-config", "{}"]);
    }

    #[test]
    fn build_vllm_server_command_rejects_unterminated_quote_in_extra_args() {
        let mut r = vllm_req();
        r.extra_args = Some("--flag \"unterminated".to_string());
        let err = build_vllm_server_command("img", &amd_rocm_profile(), "owner/model", &plain_policy(), &vllm_cache_dirs(), None, &r)
            .unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn validate_compiled_cache_root_rejects_root_home_and_missing_marker() {
        assert!(validate_compiled_cache_root(std::path::Path::new("/"), "vllm").is_err());
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        assert!(validate_compiled_cache_root(std::path::Path::new(&home), "vllm").is_err());
        assert!(validate_compiled_cache_root(std::path::Path::new("/data/cache/other"), "vllm").is_err());
        assert!(validate_compiled_cache_root(std::path::Path::new("/data/cache/vllm"), "vllm").is_ok());
    }

    #[test]
    fn save_vllm_cache_paths_rejects_when_nothing_provided() {
        let req = VllmCachePathsRequest { hf_cache: None, vllm_cache: None, triton_cache: None, aiter_cache: None };
        let err = save_vllm_cache_paths(&req).unwrap_err();
        assert_eq!(err.status(), 400);
    }

    fn amd_rocm_r9v_profile() -> RuntimeProfile {
        // Exact vendored `amd-rocm` runtime_profile used by the real r9v
        // toolbox entry (`assets/cockpit-catalog/toolboxes.json`) — no
        // `--group-add sudo` present, so `clean_engine_args_for_server` is
        // a no-op here; `upgrade_groups_for_podman` still collapses
        // video/render into keep-groups.
        RuntimeProfile {
            id: "amd-rocm".to_string(),
            engine_args: vec![
                "--device".to_string(),
                "/dev/dri".to_string(),
                "--device".to_string(),
                "/dev/kfd".to_string(),
                "--group-add".to_string(),
                "video".to_string(),
                "--group-add".to_string(),
                "render".to_string(),
                "--security-opt".to_string(),
                "seccomp=unconfined".to_string(),
            ],
        }
    }

    fn r9v_package() -> ResolvedR9vPackage {
        ResolvedR9vPackage {
            models_dir: std::path::PathBuf::from("/data/r9v-models"),
            ple_dir: std::path::PathBuf::from("/data/r9v-ple"),
            cache_dir: std::path::PathBuf::from("/data/r9v-cache"),
            ple_filename: "per_layer_token_embd.iq4_nl.bin".to_string(),
            ple_size_bytes: 28_800_138_240,
            ple_ready: true,
        }
    }

    fn r9v_req() -> StartR9vServerRequest {
        StartR9vServerRequest {
            toolbox_id: "r9700-r9v-rocm-10-0".to_string(),
            package_id: "r9v-qwen38-flash-next-iq4-xs".to_string(),
            host: None,
            port: None,
            devices: None,
            context: None,
            batch: None,
            sequences: None,
            kv_bytes: None,
            expert_cache_slots: None,
            offload: None,
            offload_devices: None,
            served_model: None,
            api_key: None,
            extra_args: None,
        }
    }

    #[test]
    fn resolve_r9v_toolbox_finds_the_real_vendored_entry_and_platform() {
        let (tb, profile, platform_id) = resolve_r9v_toolbox("r9700-r9v-rocm-10-0").expect("must resolve");
        assert_eq!(tb.id, "r9700-r9v-rocm-10-0");
        assert_eq!(profile.id, "amd-rocm");
        assert_eq!(platform_id, "r9700");
    }

    #[test]
    fn resolve_r9v_toolbox_rejects_unknown_id() {
        let err = resolve_r9v_toolbox("no-such-toolbox").unwrap_err();
        assert_eq!(err.status(), 404);
    }

    #[test]
    fn resolve_r9v_toolbox_rejects_non_r9v_backend() {
        let err = resolve_r9v_toolbox("strix-halo-llama-rocm-10-0").unwrap_err();
        assert_eq!(err.status(), 404);
    }

    #[test]
    fn build_r9v_server_command_matches_upstream_default_shape() {
        let cmd = build_r9v_server_command(
            "docker.io/kyuz0/amd-r9700-toolboxes:r9v-rocm-10.0",
            &amd_rocm_r9v_profile(),
            "r9700",
            &r9v_package(),
            &r9v_req(),
        )
        .expect("must build");

        assert_eq!(cmd[0..4], ["run", "-d", "--name", R9V_SERVER_CONTAINER_NAME]);
        assert!(cmd.windows(2).any(|w| w == ["--runtime", "crun"]));
        // engine_args (post group-upgrade) come right after --runtime crun.
        assert!(cmd.windows(6).any(|w| w == ["--device", "/dev/dri", "--device", "/dev/kfd", "--group-add", "keep-groups"]));
        assert!(cmd.windows(2).any(|w| w == ["--user", "0:0"]));
        assert!(!cmd.contains(&"--userns=keep-id".to_string()));
        assert!(cmd.contains(&"--ipc=host".to_string()));
        assert!(cmd.windows(2).any(|w| w == ["--security-opt", "label=disable"]));
        assert!(cmd.windows(2).any(|w| w == ["--label", "io.brainrouter.server_backend=r9v"]));
        assert!(cmd.windows(2).any(|w| w == ["--label", "io.brainrouter.server_model=r9v-qwen38-flash-next-iq4-xs"]));
        // Defaults: host 127.0.0.1, port 8004.
        assert!(cmd.windows(2).any(|w| w == ["-p", "127.0.0.1:8004:8000"]));
        assert!(cmd.windows(2).any(|w| w == ["-v", "/data/r9v-models:/models:ro"]));
        assert!(cmd.windows(2).any(|w| w == [
            "-v",
            "/data/r9v-ple/per_layer_token_embd.iq4_nl.bin:/ple/per_layer_token_embd.iq4_nl.bin:ro"
        ]));
        assert!(cmd.windows(2).any(|w| w == ["-v", "/data/r9v-cache:/cache"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "R9V_VISIBLE_DEVICES=0,1"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "R9V_MAX_MODEL_LEN=131072"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "R9V_MAX_NUM_BATCHED_TOKENS=1024"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "R9V_MAX_NUM_SEQS=1"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "R9V_KV_CACHE_MEMORY_BYTES=2285670400"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "R9V_CPU_OFFLOAD_GB=112.5"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "R9V_CPU_OFFLOAD_GB_BY_DEVICE=112.5,112.5"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "R9V_SERVED_MODEL_NAME=qwen3.8-flash-next"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "R9V_TENSOR_PARALLEL_SIZE=2"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "R9V_MTP_SPEC_TOKENS=2"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "R9V_PLE_RESIDENCY_MODE=ssd"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "R9V_PLE_WORKER_TIMING=1"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "R9V_SERIALIZE_EXPERT_LOAD=1"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "R9V_TIERED_EXPERT_CACHE_SLOTS=16"]));
        assert!(cmd.contains(&"docker.io/kyuz0/amd-r9700-toolboxes:r9v-rocm-10.0".to_string()));
        assert!(cmd.windows(2).any(|w| w == ["docker.io/kyuz0/amd-r9700-toolboxes:r9v-rocm-10.0", "r9v-serve"]));
        assert_eq!(cmd.last().unwrap(), "r9v-serve");
        assert!(!cmd.contains(&"--api-key".to_string()));
    }

    #[test]
    fn build_r9v_server_command_rejects_non_r9700_platform() {
        let err = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "strix-halo", &r9v_package(), &r9v_req()).unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn build_r9v_server_command_rejects_when_ple_not_ready() {
        let mut pkg = r9v_package();
        pkg.ple_ready = false;
        let err = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "r9700", &pkg, &r9v_req()).unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn build_r9v_server_command_rejects_non_distinct_devices() {
        let mut r = r9v_req();
        r.devices = Some("0,0".to_string());
        let err = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "r9700", &r9v_package(), &r).unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn build_r9v_server_command_rejects_malformed_devices() {
        for bad in ["0", "0,1,2", "a,b", "0, 1", ""] {
            let mut r = r9v_req();
            r.devices = Some(bad.to_string());
            let err = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "r9700", &r9v_package(), &r).unwrap_err();
            assert_eq!(err.status(), 400, "expected `{bad}` to be rejected");
        }
    }

    #[test]
    fn build_r9v_server_command_rejects_out_of_range_numerics() {
        type MutateCase = (fn(&mut StartR9vServerRequest), &'static str);
        let cases: Vec<MutateCase> = vec![
            (|r| r.port = Some(0), "port"),
            (|r| r.context = Some(0), "context zero"),
            (|r| r.context = Some(262_145), "context too large"),
            (|r| r.batch = Some(0), "batch zero"),
            (|r| r.batch = Some(131_073), "batch too large"),
            (|r| r.sequences = Some(0), "sequences zero"),
            (|r| r.sequences = Some(17), "sequences too large"),
            (|r| r.kv_bytes = Some(0), "kv_bytes zero"),
            (|r| r.kv_bytes = Some(32 * 1024 * 1024 * 1024 + 1), "kv_bytes too large"),
            (|r| r.expert_cache_slots = Some(17), "expert_cache_slots too large"),
        ];
        for (mutate, label) in cases {
            let mut r = r9v_req();
            mutate(&mut r);
            let err = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "r9700", &r9v_package(), &r).unwrap_err();
            assert_eq!(err.status(), 400, "case `{label}` should have been rejected");
        }
    }

    #[test]
    fn build_r9v_server_command_rejects_batch_below_sequences() {
        let mut r = r9v_req();
        r.batch = Some(1);
        r.sequences = Some(2);
        let err = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "r9700", &r9v_package(), &r).unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn build_r9v_server_command_rejects_non_finite_or_negative_offload() {
        for bad in ["-1", "nan", "inf", "not-a-number"] {
            let mut r = r9v_req();
            r.offload = Some(bad.to_string());
            let err = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "r9700", &r9v_package(), &r).unwrap_err();
            assert_eq!(err.status(), 400, "expected offload `{bad}` to be rejected");
        }
        for bad in ["112.5", "112.5,-1", "112.5,112.5,112.5"] {
            let mut r = r9v_req();
            r.offload_devices = Some(bad.to_string());
            let err = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "r9700", &r9v_package(), &r).unwrap_err();
            assert_eq!(err.status(), 400, "expected offload_devices `{bad}` to be rejected");
        }
    }

    #[test]
    fn build_r9v_server_command_preserves_original_offload_strings_verbatim() {
        let mut r = r9v_req();
        r.offload = Some("50".to_string());
        r.offload_devices = Some("25,25".to_string());
        let cmd = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "r9700", &r9v_package(), &r).expect("must build");
        assert!(cmd.windows(2).any(|w| w == ["-e", "R9V_CPU_OFFLOAD_GB=50"]));
        assert!(cmd.windows(2).any(|w| w == ["-e", "R9V_CPU_OFFLOAD_GB_BY_DEVICE=25,25"]));
    }

    #[test]
    fn build_r9v_server_command_rejects_bad_host() {
        let mut r = r9v_req();
        r.host = Some("not-an-ip".to_string());
        let err = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "r9700", &r9v_package(), &r).unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn build_r9v_server_command_normalizes_localhost_and_brackets_ipv6() {
        let mut r = r9v_req();
        r.host = Some("localhost".to_string());
        let cmd = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "r9700", &r9v_package(), &r).expect("must build");
        assert!(cmd.windows(2).any(|w| w == ["-p", "127.0.0.1:8004:8000"]));

        let mut r = r9v_req();
        r.host = Some("::1".to_string());
        let cmd = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "r9700", &r9v_package(), &r).expect("must build");
        assert!(cmd.windows(2).any(|w| w == ["-p", "[::1]:8004:8000"]));
    }

    #[test]
    fn build_r9v_server_command_rejects_empty_or_whitespace_served_model() {
        for bad in ["", "  ", "has space", "tab\tchar"] {
            let mut r = r9v_req();
            r.served_model = Some(bad.to_string());
            let err = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "r9700", &r9v_package(), &r).unwrap_err();
            assert_eq!(err.status(), 400, "expected served_model `{bad:?}` to be rejected");
        }
    }

    #[test]
    fn build_r9v_server_command_rejects_reserved_extra_args() {
        for bad in ["--host 0.0.0.0", "--port=9000", "--tensor-parallel-size 4", "-tp 4", "--api-key abc"] {
            let mut r = r9v_req();
            r.extra_args = Some(bad.to_string());
            let err = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "r9700", &r9v_package(), &r).unwrap_err();
            assert_eq!(err.status(), 400, "expected extra_args `{bad}` to be rejected");
        }
    }

    #[test]
    fn build_r9v_server_command_appends_shlex_split_extra_args() {
        let mut r = r9v_req();
        r.extra_args = Some("--enable-log-requests --foo bar".to_string());
        let cmd = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "r9700", &r9v_package(), &r).expect("must build");
        let tail = &cmd[cmd.len() - 3..];
        assert_eq!(tail, &["--enable-log-requests", "--foo", "bar"]);
    }

    #[test]
    fn build_r9v_server_command_rejects_unterminated_quote_in_extra_args() {
        let mut r = r9v_req();
        r.extra_args = Some("--foo \"unterminated".to_string());
        let err = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "r9700", &r9v_package(), &r).unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn build_r9v_server_command_includes_api_key_when_given() {
        let mut r = r9v_req();
        r.api_key = Some("secret-key".to_string());
        let cmd = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "r9700", &r9v_package(), &r).expect("must build");
        assert!(cmd.windows(2).any(|w| w == ["--api-key", "secret-key"]));
    }

    #[test]
    fn build_r9v_server_command_rejects_cache_dir_nested_in_models_dir() {
        let mut pkg = r9v_package();
        pkg.cache_dir = pkg.models_dir.join("cache");
        let err = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "r9700", &pkg, &r9v_req()).unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn build_r9v_server_command_rejects_cache_dir_equal_to_models_dir() {
        let mut pkg = r9v_package();
        pkg.cache_dir = pkg.models_dir.clone();
        let err = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "r9700", &pkg, &r9v_req()).unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn build_r9v_server_command_rejects_cache_dir_equal_to_ple_dir() {
        let mut pkg = r9v_package();
        pkg.cache_dir = pkg.ple_dir.clone();
        let err = build_r9v_server_command("img", &amd_rocm_r9v_profile(), "r9700", &pkg, &r9v_req()).unwrap_err();
        assert_eq!(err.status(), 400);
    }

    #[test]
    fn save_r9v_paths_rejects_when_nothing_provided() {
        let req = R9vPathsRequest { models_dir: None, ple_dir: None, cache_dir: None };
        let err = save_r9v_paths(&req).unwrap_err();
        assert_eq!(err.status(), 400);
    }
}

