//! PR7/PR8: ds4 and halogen Server Mode — headless, detached `podman run`
//! servers, distinct from the Toolboxes-tab dev-shell containers
//! (`src/server.rs`'s `recreate_toolbox_container`/`toolbox create`
//! family). Kept as one file across backends (not split into submodules),
//! mirroring `src/model_downloads.rs`'s own single-file, per-backend-match
//! organization rather than inventing a different convention mid-rollout
//! (§13a).
//!
//! See `docs/design/ai-toolbox-cockpit-integration.md` §11 for why these
//! are two structurally different code paths upstream (`toolbox create`
//! never receives a `runtime_profile`'s `engine_args`; only Server Mode's
//! plain `podman run` does), §12 for ds4's v1 scope-narrowing, and §13 for
//! halogen's.
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
//! Lifecycle is synchronous request/response for every backend (§12 item
//! 5), not an async job registry like `model_downloads.rs` — starting/
//! stopping a container is a sub-second podman operation, and `status`
//! always reads live `podman inspect` state rather than persisting
//! anything.

use serde::{Deserialize, Serialize};

use crate::model_downloads::{self, ResolvedDs4Model, ResolvedHalogenBundle};
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
}

