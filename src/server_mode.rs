//! PR7: ds4 Server Mode — headless, detached `podman run` servers, distinct
//! from the Toolboxes-tab dev-shell containers (`src/server.rs`'s
//! `recreate_toolbox_container`/`toolbox create` family).
//!
//! See `docs/design/ai-toolbox-cockpit-integration.md` §11 for why these
//! are two structurally different code paths upstream (`toolbox create`
//! never receives a `runtime_profile`'s `engine_args`; only Server Mode's
//! plain `podman run` does), and §12 for the concrete v1 scope-narrowing
//! this module implements.
//!
//! **v1 scope: ds4 only, single-node ("Standalone") launch.** Multi-node
//! coordinator/worker roles, DSpark, SSD-streaming, and MTP/vision-projector
//! auto-configuration are deliberately not ported — reachable instead via
//! the `custom_args` free-text escape hatch, exactly mirroring upstream's
//! own identically-named `server_runner.py::build_server_cmd(custom_args)`
//! parameter (not a brainrouter invention). See §12 for the full rationale
//! and the two open items flagged there for explicit sign-off (host/port
//! default policy; single- vs. multi-instance naming).
//!
//! Lifecycle is synchronous request/response (§12 item 5), not an async job
//! registry like `model_downloads.rs` — starting/stopping a container is a
//! sub-second podman operation, and `status` always reads live `podman
//! inspect` state rather than persisting anything.

use serde::{Deserialize, Serialize};

use crate::model_downloads::{self, ResolvedDs4Model};
use crate::toolbox_catalog::{self, RuntimeProfile, SupportedServingBackend, ToolboxDefinition};

/// Fixed container name for the (single-instance, v1) ds4 server-mode
/// container — mirrors upstream's own fixed `ds4-cockpit-server` name.
/// §12's open item: single-instance is the proposed v1 default, matching
/// cockpit's own TUI (starting a second model requires stopping the
/// first), pending explicit sign-off.
pub const DS4_SERVER_CONTAINER_NAME: &str = "brainrouter-ds4-server";

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
pub struct StartServerRequest {
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
    req: &StartServerRequest,
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

/// Resolves `toolbox_id` to a ds4 catalog toolbox, narrowed to entries that
/// actually declare `features.server` as supported (matches upstream's own
/// `ToolboxFeatures` gate on which toolboxes even offer a Server Mode
/// panel).
fn resolve_ds4_toolbox(toolbox_id: &str) -> Result<(ToolboxDefinition, RuntimeProfile), ServerModeError> {
    let vendored = toolbox_catalog::load_vendored_catalog();
    let (catalog, _models) = vendored
        .typed()
        .map_err(|e| ServerModeError::Internal(format!("failed to parse vendored toolbox catalog: {e}")))?;
    let tb = catalog
        .toolbox_by_id(toolbox_id)
        .filter(|t| t.supported_backend() == Some(SupportedServingBackend::Ds4))
        .ok_or_else(|| ServerModeError::NotFound(format!("no ds4 catalog toolbox with id `{toolbox_id}`")))?;
    if tb.features.server != toolbox_catalog::FeatureState::Supported {
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

/// `POST /api/server-mode/ds4/start`: resolves the toolbox + already-
/// downloaded model, force-removes any pre-existing container of the fixed
/// name (idempotent — mirrors upstream's own `podman rm -f` immediately
/// before every start, `runtime/server_process.py::run_foreground_server`),
/// then runs the built command.
pub async fn start_ds4_server(req: &StartServerRequest) -> Result<(), ServerModeError> {
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

/// Reads one ownership-label value off the fixed-name container, `None` if
/// the container or label is absent. Mirrors `src/server.rs`'s
/// `toolbox_container_is_managed()` inspect-format pattern.
async fn ds4_server_label(label: &str) -> Option<String> {
    let out = tokio::process::Command::new("podman")
        .args([
            "inspect",
            "--format",
            &format!("{{{{ index .Config.Labels \"{label}\" }}}}"),
            DS4_SERVER_CONTAINER_NAME,
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

/// `GET /api/server-mode/ds4/status`: no persisted state at all (§12 item
/// 5) — always reads live `podman inspect` output for the fixed container
/// name.
pub async fn ds4_server_status() -> ServerStatus {
    let out = tokio::process::Command::new("podman")
        .args(["inspect", "--format", "{{.State.Running}}", DS4_SERVER_CONTAINER_NAME])
        .output()
        .await;
    match out {
        Ok(o) if o.status.success() => {
            let running = String::from_utf8_lossy(&o.stdout).trim() == "true";
            ServerStatus {
                backend: "ds4",
                container_name: DS4_SERVER_CONTAINER_NAME.to_string(),
                exists: true,
                running,
                model_id: ds4_server_label(LABEL_SERVER_MODEL).await,
            }
        }
        _ => ServerStatus {
            backend: "ds4",
            container_name: DS4_SERVER_CONTAINER_NAME.to_string(),
            exists: false,
            running: false,
            model_id: None,
        },
    }
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

    fn req() -> StartServerRequest {
        StartServerRequest {
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
}
