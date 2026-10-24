//! Config migration framework (v2 L2).
//!
//! Cross-leg contract: every leg that migrates state FIRST snapshots the
//! pre-migration state (owner-only), supports `--dry-run` preview, logs what
//! it did, and ships a named restore command. Boot-time migrations run before
//! config validation so v1 YAML always boots.
//!
//! Snapshot layout:
//! `$XDG_STATE_HOME/brainrouter/migrations/<phase-tag>/<UTC timestamp>/`
//! containing a `snapshot/` directory with the pre-migration config YAML and
//! `credentials.json` (whichever exist). `restore` reads the newest snapshot
//! for a phase.

use crate::credentials::{CredentialKind, CredentialRecord, CredentialStore};
use anyhow::{Context, Result};
use std::os::unix::fs::PermissionsExt;
use serde_yaml::Value as YamlValue;
use std::path::{Path, PathBuf};

/// The L2 phase tag: `brainrouter migration restore v2.0-p2`.
pub const PHASE_V2_P2: &str = "v2.0-p2";

/// The L3 phase tag: `brainrouter migration restore v2.0-p3`. Runs BEFORE
/// the L2 phase so its snapshot holds the pristine pre-migration state and
/// `restore v2.0-p3` round-trips the v1 YAML byte-for-byte.
pub const PHASE_V2_P3: &str = "v2.0-p3";

/// Credential id the Discord-token migration creates (stable, conventional
/// id; the plugin resolves it via `secrets.get`).
pub const DISCORD_TOKEN_CREDENTIAL_ID: &str = "discord_token";

/// The plugin id the migrated credential grants.
pub const BRIDGE_PLUGIN_ID: &str = "bridge";

/// The `v2.0-p2` migration: move the v1 plaintext `bridge.discord.token` into
/// the credentials store (granted to plugin `bridge`), snapshot first, then
/// strip the key from the YAML (with a `.bak` sidecar). Idempotent.
///
/// `state_root` is `$XDG_STATE_HOME/brainrouter` (snapshots live there).
/// Returns the human-readable actions taken (worded for logs and CLI output,
/// token value masked). Empty = nothing to do.
pub fn run_v2_p2(
    config_path: &Path,
    store: &CredentialStore,
    state_root: &Path,
) -> Result<Vec<String>> {
    let Some(token) = read_discord_token(config_path)? else {
        return Ok(Vec::new());
    };
    let snapshot_dir = snapshot(config_path, store, state_root, PHASE_V2_P2)?;

    store.put(CredentialRecord {
        id: DISCORD_TOKEN_CREDENTIAL_ID.to_string(),
        description: "Discord bot token (migrated from v1 bridge.discord.token)".to_string(),
        kind: CredentialKind::Store {
            value: crate::credentials::SecretString::new(token),
        },
        grants: vec![BRIDGE_PLUGIN_ID.to_string()],
        created_at: String::new(),
        updated_at: String::new(),
    })?;

    strip_discord_token(config_path)?;

    Ok(vec![format!(
        "migrated v1 bridge.discord.token into the credentials store as credential \
         '{DISCORD_TOKEN_CREDENTIAL_ID}' (kind store, grants: {BRIDGE_PLUGIN_ID}); \
         key removed from {} (backup: {}.bak); pre-migration snapshot: {}",
        config_path.display(),
        config_path.display(),
        snapshot_dir.display()
    )])
}

// ── v2.0-p3: role cards + decision slot (L3) ────────────────────────────────

/// Raw-YAML inspection of the v1 state the L3 migration consumes.
#[derive(Debug, Default)]
pub struct V1State {
    pub has_routing_block: bool,
    pub has_bonsai_block: bool,
    pub has_decision_model: bool,
    pub routing: Option<crate::roles::LegacyRoutingProfile>,
    pub review: Option<crate::roles::ReviewExtras>,
    pub forced_mode: Option<String>,
    pub forced_model: Option<String>,
    pub bonsai_enabled: bool,
    pub bonsai_port: u16,
    pub bonsai_fork_path: Option<String>,
    pub bonsai_model_path: Option<String>,
}

/// Read the v1-relevant keys out of the config YAML (no validation — this
/// runs BEFORE config::load so a v1 file always inspects).
pub fn read_v1_state(config_path: &Path) -> Result<V1State> {
    let text = std::fs::read_to_string(config_path)
        .with_context(|| format!("reading {} for migration", config_path.display()))?;
    let root: YamlValue = serde_yaml::from_str(&text)
        .with_context(|| format!("parsing {} for migration", config_path.display()))?;
    let mut out = V1State::default();
    let Some(map) = root.as_mapping() else {
        return Ok(out);
    };
    let get = |key: &str| map.get(YamlValue::String(key.into()));
    out.has_decision_model = get("decision_model").is_some();
    if let Some(routing) = get("routing") {
        out.has_routing_block = true;
        // The v1 `routing:` block is the serde shape of RoutingProfile; parse
        // it through the relocated ModelChoice deserializer.
        let json: serde_json::Value = serde_json::to_value(routing)
            .context("converting routing block")?;
        out.routing = Some(
            serde_json::from_value(json)
                .context("invalid v1 routing block in migration")?,
        );
    }
    if let Some(review) = get("review").and_then(|r| r.as_mapping()) {
        let g = |key: &str| review.get(YamlValue::String(key.into()));
        let max_iterations = g("max_iterations")
            .and_then(|v| v.as_i64())
            .map(|v| v.clamp(1, u32::MAX as i64) as u32);
        let hankndory = g("hankndory_integration")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let design_doc_path = g("design_doc_path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        out.review = Some(crate::roles::ReviewExtras {
            max_iterations: max_iterations.unwrap_or(5),
            hankndory_integration: hankndory,
            design_doc_path,
        });
        out.forced_mode = g("forced_mode").and_then(|v| v.as_str()).map(|s| s.to_string());
        out.forced_model = g("forced_model").and_then(|v| v.as_str()).map(|s| s.to_string());
    }
    if let Some(bonsai) = get("bonsai").and_then(|b| b.as_mapping()) {
        out.has_bonsai_block = true;
        let g = |key: &str| bonsai.get(YamlValue::String(key.into()));
        out.bonsai_enabled = g("enabled").and_then(|v| v.as_bool()).unwrap_or(false);
        out.bonsai_port = g("server_port")
            .and_then(|v| v.as_i64())
            .map(|v| v.clamp(1, u16::MAX as i64) as u16)
            .unwrap_or(9200);
        out.bonsai_fork_path = g("fork_path").and_then(|v| v.as_str()).map(|s| s.to_string());
        out.bonsai_model_path = g("model_path").and_then(|v| v.as_str()).map(|s| s.to_string());
    }
    Ok(out)
}

impl V1State {
    /// Whether this YAML is pre-L3 (no decision slot yet) — the migration
    /// trigger. An L3-era YAML always carries `decision_model:`.
    pub fn needs_l3_migration(&self) -> bool {
        !self.has_decision_model
    }

    /// The effective v1 profile: YAML `routing:` block, else the v1 defaults
    /// (`config.routing_profile()` @ v1-classic): main = Local{None} (the v1
    /// default — "default-local stays local"), reviewer = review.forced_*
    /// (`ReviewConfig::model_choice`), subagent ONLY from the profile block
    /// (`config.subs_model` stays as the auto-pool compatibility seed).
    pub fn effective_profile(&self) -> Result<crate::roles::LegacyRoutingProfile> {
        if let Some(profile) = &self.routing {
            return Ok(profile.clone());
        }
        // v1 read-parity (`ReviewConfig::model_choice`): mode `auto` ignores a
        // leftover forced_model (the "legacy auto yaml" keeper).
        let forced_model = if self.forced_mode.as_deref().unwrap_or("auto") == "auto" {
            None
        } else {
            self.forced_model.clone()
        };
        let reviewer = crate::roles::ModelChoice::from_legacy(
            self.forced_mode.as_deref().unwrap_or("auto"),
            forced_model,
        )?;
        Ok(crate::roles::LegacyRoutingProfile {
            preset: crate::roles::LegacyPreset::Custom,
            main: crate::roles::ModelChoice::local(),
            reviewer,
            subagent_model: None,
        })
    }
}

/// The `v2.0-p3` migration (L3): v1 routing profile + project pins +
/// `review.forced_*` + `bonsai.*` → role cards (`role_cards.json`) + a
/// `decision_model:` block; the consumed YAML keys are stripped (snapshot +
/// `.bak` first). Idempotent: an L3-era YAML (has `decision_model:`) is a
/// no-op. Runs BEFORE the L2 phase so the snapshot is pristine v1 state.
pub fn run_v2_p3(
    config_path: &Path,
    store: &CredentialStore,
    state_root: &Path,
    config_dir: &Path,
    cards: &crate::roles::RoleCardStore,
) -> Result<Vec<String>> {
    let state = read_v1_state(config_path)?;
    if !state.needs_l3_migration() {
        return Ok(Vec::new());
    }
    let snapshot_dir = snapshot_with_v1_stores(config_path, store, state_root, config_dir, PHASE_V2_P3, cards)?;

    // Profile precedence: persisted routing_state.json (UI overrides) > YAML
    // routing block > v1 defaults.
    let profile = match crate::roles::read_legacy_routing_state(
        &config_dir.join("routing_state.json"),
    ) {
        Ok(profile) => profile,
        Err(e) if e.root_cause().to_string().contains("No such file") => state.effective_profile()?,
        Err(e) => return Err(e),
    };
    let pins_path = config_dir.join("project_pins.json");
    let pins = match crate::roles::read_legacy_pins(&pins_path) {
        Ok(pins) => pins,
        Err(e) if e.root_cause().to_string().contains("No such file") => Default::default(),
        Err(e) => return Err(e),
    };
    let output = crate::roles::migrate_v1(&profile, &pins, state.review.clone());

    let data = crate::roles::StoreDataForMigration {
        assignments: output.cards,
        review_extras: output.review_extras,
    };
    cards.replace_from_migration(data)?;

    strip_v1_keys_and_add_decision_model(config_path, &state)?;

    let mut actions = vec![format!(
        "migrated v1 routing profile + project pins + review.forced_* into {} (cards: {}); \
         keys `routing:`, `review.forced_mode/forced_model` removed from {}",
        cards
            .path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "role cards".into()),
        cards.assignments_snapshot().len(),
        config_path.display()
    )];
    if state.has_bonsai_block {
        actions.push(format!(
            "migrated bonsai.* into the decision slot (kind openai-compatible, prompt_profile bonsai, enabled: {}); launch backend config preserved under decision_model.launch; key `bonsai:` removed (backup: {}.bak)",
            state.bonsai_enabled,
            config_path.display()
        ));
    }
    actions.push(format!(
        "pre-migration snapshot (config, credentials, project pins, routing state): {}",
        snapshot_dir.display()
    ));
    Ok(actions)
}

/// `--dry-run` preview for `v2.0-p3` — describes actions, never writes.
pub fn preview_v2_p3(
    config_path: &Path,
    state_root: &Path,
    config_dir: &Path,
) -> Result<Vec<String>> {
    let mut lines = Vec::new();
    lines.push(format!("dry-run preview: phase {PHASE_V2_P3} (state untouched)"));
    let state = match read_v1_state(config_path) {
        Ok(state) => state,
        Err(e) => {
            lines.push(format!("  cannot inspect {}: {e}", config_path.display()));
            return Ok(lines);
        }
    };
    if !state.needs_l3_migration() {
        lines.push(format!(
            "  {} is an L3-era config (decision_model present) — migration {PHASE_V2_P3} has nothing to do",
            config_path.display()
        ));
        return Ok(lines);
    }
    let profile = state.effective_profile();
    match &profile {
        Ok(profile) => {
            lines.push(format!(
                "  would migrate the v1 profile: main `{}` -> `{}`, reviewer `{}` -> `{}`, subagent {:?}",
                profile.main.backend(),
                profile.main.card_selector(),
                profile.reviewer.backend(),
                profile.reviewer.card_selector(),
                profile.subagent_model
            ));
        }
        Err(e) => lines.push(format!("  v1 profile unreadable: {e}")),
    }
    for file in ["project_pins.json", "routing_state.json"] {
        if config_dir.join(file).exists() {
            lines.push(format!("  would migrate v1 state store: {}", config_dir.join(file).display()));
        }
    }
    lines.push(format!(
        "  would snapshot pre-migration state under {}",
        snapshot_root(state_root, PHASE_V2_P3).display()
    ));
    lines.push(format!(
        "  would write role cards to {}",
        crate::roles::RoleCardStore::default_path().display()
    ));
    if state.has_bonsai_block {
        lines.push(
            "  would migrate bonsai.* into decision_model (kind openai-compatible, prompt_profile bonsai)".to_string(),
        );
    }
    lines.push(format!(
        "  would remove routing/review.forced_*/bonsai keys from {} (backup: {}.bak)",
        config_path.display(),
        config_path.display()
    ));
    Ok(lines)
}

/// Snapshot that also captures the v1 state stores L3 replaces.
fn snapshot_with_v1_stores(
    config_path: &Path,
    store: &CredentialStore,
    state_root: &Path,
    config_dir: &Path,
    phase: &str,
    cards: &crate::roles::RoleCardStore,
) -> Result<PathBuf> {
    let dir = snapshot(config_path, store, state_root, phase)?;
    for (src, name) in [
        (config_dir.join("project_pins.json"), SNAPSHOT_PINS_FILE),
        (config_dir.join("routing_state.json"), SNAPSHOT_ROUTING_STATE_FILE),
    ] {
        if let Ok(text) = std::fs::read(&src) {
            crate::plugins::store::atomic_write_0600(&dir.join(name), &text)?;
        }
    }
    // v2 L3 Step 9 B1: capture the cards store too, when present, so a
    // snapshot restore can roll the cards back (the daemon's own state file
    // is passed in, not re-derived from env).
    if let Some(cards_path) = cards.path() {
        if cards_path.exists() {
            if let Ok(text) = std::fs::read(cards_path) {
                crate::plugins::store::atomic_write_0600(&dir.join(SNAPSHOT_ROLE_CARDS_FILE), &text)?;
            }
        }
    }
    Ok(dir)
}

/// Rewrite the YAML: strip the keys L3 consumed and insert the `decision_model`
/// block (from `bonsai.*`). Leaves a `.bak` copy of the original.
fn strip_v1_keys_and_add_decision_model(config_path: &Path, state: &V1State) -> Result<()> {
    let text = std::fs::read_to_string(config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    std::fs::write(bak_path(config_path), &text).context("writing config .bak backup")?;
    let mut root: YamlValue = serde_yaml::from_str(&text).context("re-parsing config YAML")?;
    let Some(map) = root.as_mapping_mut() else {
        return Ok(());
    };
    map.remove(YamlValue::String("routing".into()));
    if let Some(review) = map
        .get_mut(YamlValue::String("review".into()))
        .and_then(|r| r.as_mapping_mut())
    {
        review.remove(YamlValue::String("forced_mode".into()));
        review.remove(YamlValue::String("forced_model".into()));
    }
    map.remove(YamlValue::String("bonsai".into()));
    // v2 L3 Step 9 B1: the migrated YAML ALWAYS carries `decision_model:` —
    // it is the marker that makes the p3 guard (`!has_decision_model`)
    // one-shot. Gating the insertion on the bonsai block made every
    // bonsai-less v1 config re-migrate on EVERY boot: cards reset to the
    // (then-stale) v1 profile, snapshots and `.bak` rewritten per start.
    // The no-bonsai case writes the behavior-neutral parity anchor
    // (`enabled: false, kind: heuristic` = classifier-off v1 semantics).
    let mut slot = serde_yaml::Mapping::new();
    if state.has_bonsai_block {
        slot.insert(
            YamlValue::String("enabled".into()),
            YamlValue::Bool(state.bonsai_enabled),
        );
        slot.insert(
            YamlValue::String("kind".into()),
            YamlValue::String("openai-compatible".into()),
        );
        slot.insert(
            YamlValue::String("base_url".into()),
            YamlValue::String(format!("http://127.0.0.1:{}", state.bonsai_port)),
        );
        slot.insert(
            YamlValue::String("model".into()),
            YamlValue::String("bonsai".into()),
        );
        slot.insert(
            YamlValue::String("prompt_profile".into()),
            YamlValue::String("bonsai".into()),
        );
        let mut launch = serde_yaml::Mapping::new();
        launch.insert(
            YamlValue::String("server_port".into()),
            YamlValue::Number(serde_yaml::Number::from(state.bonsai_port as i64)),
        );
        if let Some(fork) = &state.bonsai_fork_path {
            launch.insert(
                YamlValue::String("fork_path".into()),
                YamlValue::String(fork.clone()),
            );
        }
        if let Some(model) = &state.bonsai_model_path {
            launch.insert(
                YamlValue::String("model_path".into()),
                YamlValue::String(model.clone()),
            );
        }
        slot.insert(
            YamlValue::String("launch".into()),
            YamlValue::Mapping(launch),
        );
    } else {
        slot.insert(
            YamlValue::String("enabled".into()),
            YamlValue::Bool(false),
        );
        slot.insert(
            YamlValue::String("kind".into()),
            YamlValue::String("heuristic".into()),
        );
    }
    map.insert(
        YamlValue::String("decision_model".into()),
        YamlValue::Mapping(slot),
    );
    let out = serde_yaml::to_string(&root).context("serializing migrated config YAML")?;
    // v2 L3 Step 9 (nit): atomic write — the migrated YAML is the daemon's
    // live config; a torn write here bricks every future boot. Byte-equality
    // for `restore`: the `.bak`/snapshot copies above are the pristine bytes.
    crate::plugins::store::atomic_write_0600(config_path, out.as_bytes())
        .context("writing migrated config YAML")?;
    Ok(())
}

// ── v2.0-p4: swap catalog import (L4) ───────────────────────────────────────

/// The L4 phase tag: `brainrouter migration restore v2.0-p4`. On first boot it
/// imports the live llama-swap `config.yaml` into `swap_catalog.json`
/// (snapshot + dry-run + restore, one stable snapshot directory per phase —
/// the v2.0-p3 shape).
pub const PHASE_V2_P4: &str = "v2.0-p4";
pub const SNAPSHOT_SWAP_CATALOG_FILE: &str = "swap_catalog.json";
pub const SNAPSHOT_BUDGET_FILE: &str = "budget.json";

/// `--dry-run` preview for `v2.0-p4` — describes the import, never writes.
pub fn preview_v2_p4(
    llama_swap_config_path: &Path,
    state_root: &Path,
    catalog_path: &Path,
) -> Result<Vec<String>> {
    let mut lines = Vec::new();
    lines.push(format!("dry-run preview: phase {PHASE_V2_P4} (state untouched)"));
    if !llama_swap_config_path.exists() {
        lines.push(format!(
            "  no llama-swap config at {} — import would be a no-op",
            llama_swap_config_path.display()
        ));
        return Ok(lines);
    }
    match crate::swap_catalog::import_llama_swap_file(llama_swap_config_path) {
        Ok(plan) => lines.extend(plan.summary_lines()),
        Err(e) => lines.push(format!("  llama-swap config unreadable: {e}")),
    }
    if catalog_path.exists() {
        match crate::swap_catalog::SwapCatalog::load(catalog_path.to_path_buf()) {
            Ok(catalog) if catalog.migration_done(PHASE_V2_P4) => lines.push(format!(
                "  {} already carries the {PHASE_V2_P4} marker — nothing to do",
                catalog_path.display()
            )),
            Ok(_) => lines.push(format!(
                "  would merge into {} (CLI-added entries are preserved)",
                catalog_path.display()
            )),
            Err(e) => lines.push(format!("  existing catalog unreadable: {e}")),
        }
    } else {
        lines.push(format!(
            "  would create {}",
            catalog_path.display()
        ));
    }
    lines.push(format!(
        "  would snapshot the pre-migration catalog + budget under {}",
        snapshot_root(state_root, PHASE_V2_P4).display()
    ));
    Ok(lines)
}

/// The `v2.0-p4` migration: import the live llama-swap config into the swap
/// catalog. Idempotent (a catalog already carrying the phase marker is a
/// no-op); snapshots the pre-migration catalog + budget ledger first.
pub fn run_v2_p4(
    llama_swap_config_path: &Path,
    state_root: &Path,
    catalog_path: &Path,
    budget_path: &Path,
) -> Result<Vec<String>> {
    let catalog = crate::swap_catalog::SwapCatalog::load(catalog_path.to_path_buf())?;
    if catalog.migration_done(PHASE_V2_P4) {
        return Ok(Vec::new());
    }
    if !llama_swap_config_path.exists() {
        // No llama-swap here (e.g. a native-only host): nothing to import, and
        // no marker — a later config appearing still migrates on some boot.
        return Ok(Vec::new());
    }
    let plan = crate::swap_catalog::import_llama_swap_file(llama_swap_config_path)?;
    let snapshot_dir = snapshot_root(state_root, PHASE_V2_P4);
    std::fs::create_dir_all(&snapshot_dir)
        .with_context(|| format!("creating {}", snapshot_dir.display()))?;
    let _ = std::fs::set_permissions(&snapshot_dir, std::fs::Permissions::from_mode(0o700));
    for (source, name) in [
        (catalog_path, SNAPSHOT_SWAP_CATALOG_FILE),
        (budget_path, SNAPSHOT_BUDGET_FILE),
    ] {
        if source.exists() {
            let bytes = std::fs::read(source)
                .with_context(|| format!("reading {} for snapshot", source.display()))?;
            crate::plugins::store::atomic_write_0600(&snapshot_dir.join(name), &bytes)?;
        }
    }
    let outcome = catalog.apply_import(&plan)?;
    catalog.mark_migration(PHASE_V2_P4)?;
    Ok(vec![format!(
        "imported llama-swap config {} into {} (added {}, updated {}, unchanged {}, skipped {} CLI-owned);          pre-migration snapshot: {}",
        llama_swap_config_path.display(),
        catalog_path.display(),
        outcome.added,
        outcome.updated,
        outcome.unchanged,
        outcome.skipped.len(),
        snapshot_dir.display()
    )])
}

/// Restore `v2.0-p4`: put the pre-migration catalog + budget ledger back
/// (absent in the snapshot = remove, the pre-migration state).
pub fn restore_v2_p4(
    state_root: &Path,
    catalog_path: &Path,
    budget_path: &Path,
) -> Result<String> {
    let latest = newest_snapshot(PHASE_V2_P4, state_root)?.with_context(|| {
        format!(
            "no {PHASE_V2_P4} migration snapshot found under {}",
            snapshot_root(state_root, PHASE_V2_P4).display()
        )
    })?;
    let mut restored = Vec::new();
    for (name, target, label) in [
        (
            SNAPSHOT_SWAP_CATALOG_FILE,
            catalog_path,
            "swap catalog",
        ),
        (SNAPSHOT_BUDGET_FILE, budget_path, "budget ledger"),
    ] {
        let snapshot = latest.join(name);
        if snapshot.exists() {
            std::fs::copy(&snapshot, target)
                .with_context(|| format!("restoring {label} to {}", target.display()))?;
            restored.push(label);
        } else if target.exists() {
            std::fs::remove_file(target)
                .with_context(|| format!("removing {} (absent before the migration)", target.display()))?;
            restored.push(match label {
                "swap catalog" => "swap catalog (removed: absent before the migration)",
                _ => "budget ledger (removed: absent before the migration)",
            });
        }
    }
    if restored.is_empty() {
        anyhow::bail!("snapshot {} contains no restorable swap state", latest.display());
    }
    Ok(format!(
        "restored {PHASE_V2_P4} from {} ({})",
        latest.display(),
        restored.join(", ")
    ))
}

/// `--dry-run` preview: describe exactly what the `v2.0-p2` migration would
/// do, without touching any state. Token values are never rendered.
pub fn preview_v2_p2(
    config_path: &Path,
    store_path: &Path,
    state_root: &Path,
) -> Result<Vec<String>> {
    let mut lines = Vec::new();
    lines.push(format!("dry-run preview: phase {PHASE_V2_P2} (state untouched)"));
    match read_discord_token(config_path)? {
        Some(_token) => {
            lines.push(format!(
                "  would migrate bridge.discord.token (value masked) from {} into credential \
                 '{DISCORD_TOKEN_CREDENTIAL_ID}' (kind store, grants: {BRIDGE_PLUGIN_ID})",
                config_path.display()
            ));
            lines.push(format!(
                "  would snapshot the pre-migration config + credentials under {}",
                snapshot_root(state_root, PHASE_V2_P2).display()
            ));
            lines.push(format!(
                "  would remove bridge.discord.token from {} (backup: {}.bak)",
                config_path.display(),
                config_path.display()
            ));
        }
        None => lines.push(format!(
            "  no bridge.discord.token in {} — migration {PHASE_V2_P2} has nothing to do",
            config_path.display()
        )),
    }
    if store_path.exists() {
        lines.push(format!("  credentials store present: {}", store_path.display()));
    } else {
        lines.push(format!(
            "  credentials store would be created at {}",
            store_path.display()
        ));
    }
    Ok(lines)
}

/// `brainrouter migration restore <phase-tag>`: restore the newest snapshot
/// for the phase — config YAML and credentials store files — and with them
/// the pre-migration YAML keys (the Discord token) and grants.
pub fn restore(
    phase: &str,
    config_path: &Path,
    store: &CredentialStore,
    state_root: &Path,
    config_dir: &Path,
) -> Result<String> {
    // v2 L4: the p4 snapshot holds swap state only (no config/credentials/
    // cards); restoring it must not touch those unrelated files.
    if phase == PHASE_V2_P4 {
        return restore_v2_p4(
            state_root,
            &state_root.join(crate::swap_catalog::CATALOG_FILE),
            &state_root.join(crate::swap_scheduler::BUDGET_FILE),
        );
    }
    let latest = newest_snapshot(phase, state_root)?.with_context(|| {
        format!(
            "no {phase} migration snapshot found under {}",
            snapshot_root(state_root, phase).display()
        )
    })?;
    let snap_dir = latest.clone();
    let mut restored = Vec::new();

    let snap_config = snap_dir.join(SNAPSHOT_CONFIG_FILE);
    if snap_config.exists() {
        let text = std::fs::read_to_string(&snap_config)
            .with_context(|| format!("reading snapshot {}", snap_config.display()))?;
        std::fs::write(config_path, &text)
            .with_context(|| format!("restoring config to {}", config_path.display()))?;
        restored.push("config YAML (v1 keys incl. bridge.discord.token)");
    }
    let snap_creds = snap_dir.join(SNAPSHOT_CREDENTIALS_FILE);
    if snap_creds.exists() {
        // Restore the pre-migration credential set verbatim (this drops the
        // migrated credential + its grants, i.e. the v1 state).
        let text = std::fs::read_to_string(&snap_creds)
            .with_context(|| format!("reading snapshot {}", snap_creds.display()))?;
        let records = extract_records(&text)?;
        store.replace_all(records)?;
        restored.push("credentials store (pre-migration values + grants)");
    }
    // L3: restore the v1 state stores that role cards replaced, and drop the
    // cards store so the next boot re-derives it from the restored v1 state.
    let snap_pins = snap_dir.join(SNAPSHOT_PINS_FILE);
    if snap_pins.exists() {
        std::fs::copy(&snap_pins, config_dir.join("project_pins.json"))
            .context("restoring v1 project pins")?;
        restored.push("v1 project pins store");
    }
    let snap_profile = snap_dir.join(SNAPSHOT_ROUTING_STATE_FILE);
    if snap_profile.exists() {
        std::fs::copy(&snap_profile, config_dir.join("routing_state.json"))
            .context("restoring v1 routing profile state")?;
        restored.push("v1 routing profile state");
    }
    let cards = crate::roles::RoleCardStore::default_path();
    let snap_cards = snap_dir.join(SNAPSHOT_ROLE_CARDS_FILE);
    if snap_cards.exists() {
        // v2 L3 Step 9 B1: the p3 snapshot captured the cards store —
        // restore it (the re-derivation on next boot must see the
        // pre-migration cards, not none).
        std::fs::copy(&snap_cards, &cards).context("restoring role cards store")?;
        restored.push("role cards store (pre-migration snapshot)");
    } else if cards.exists() {
        std::fs::remove_file(&cards)
            .with_context(|| format!("removing {}", cards.display()))?;
        restored.push("role cards store (removed)");
    }
    if restored.is_empty() {
        anyhow::bail!("snapshot {} contains no restorable files", snap_dir.display());
    }
    Ok(format!(
        "restored {phase} from {} ({})",
        latest.display(),
        restored.join(", ")
    ))
}

fn extract_records(creds_json: &str) -> Result<Vec<CredentialRecord>> {
    let v: serde_json::Value =
        serde_json::from_str(creds_json).context("parsing snapshot credentials.json")?;
    let creds = v.get("credentials").cloned().unwrap_or(serde_json::Value::Null);
    // The vault's `credentials` field serializes as a map (BTreeMap keyed by
    // id); an array is the legacy shape; null/missing is an empty vault. Any
    // other shape is corrupt input: bail rather than restore an empty set
    // (a restore that silently wipes the vault is the worst outcome).
    let items: Vec<serde_json::Value> = match &creds {
        serde_json::Value::Object(map) => map.values().cloned().collect(),
        serde_json::Value::Array(arr) => arr.clone(),
        serde_json::Value::Null => Vec::new(),
        other => {
            let rendered = other.to_string();
            anyhow::bail!(
                "snapshot credentials.json has an unreadable credentials shape ({})",
                &rendered[..rendered.len().min(80)]
            )
        }
    };
    let mut out = Vec::new();
    for c in items {
        out.push(serde_json::from_value(c).context("decoding snapshot credential")?);
    }
    Ok(out)
}

pub fn snapshot_root(state_root: &Path, phase: &str) -> PathBuf {
    state_root.join("migrations").join(phase)
}

/// Snapshot files (directly under the timestamp directory).
pub const SNAPSHOT_CONFIG_FILE: &str = "snapshot-config.yaml";
pub const SNAPSHOT_CREDENTIALS_FILE: &str = "snapshot-credentials.json";
/// L3: the v1 state stores replaced by role cards.
pub const SNAPSHOT_PINS_FILE: &str = "snapshot-project-pins.json";
pub const SNAPSHOT_ROLE_CARDS_FILE: &str = "role_cards.json";
pub const SNAPSHOT_ROUTING_STATE_FILE: &str = "snapshot-routing-state.json";

pub fn newest_snapshot(phase: &str, state_root: &Path) -> Result<Option<PathBuf>> {
    let root = snapshot_root(state_root, phase);
    // v2 L3 Step 9 B1: the snapshot is one directory per phase (no per-boot
    // timestamps). Accept the legacy timestamped layout too (restore of a
    // pre-B1 state dir): if the phase root directly contains a snapshot
    // file, it IS the snapshot.
    // The phase root IS the snapshot for the stable layout. p2/p3 carry a
    // config snapshot; p4 carries swap state only — accept either marker.
    if root.join(SNAPSHOT_CONFIG_FILE).exists()
        || root.join(SNAPSHOT_SWAP_CATALOG_FILE).exists()
        || root.join(SNAPSHOT_BUDGET_FILE).exists()
    {
        return Ok(Some(root));
    }
    let mut dirs: Vec<PathBuf> = match std::fs::read_dir(&root) {
        Ok(rd) => rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).context("listing migration snapshots"),
    };
    // Timestamps are fixed-width `YYYYMMDDTHHMMSSZ`, so lexical order is time order.
    dirs.sort();
    Ok(dirs.pop())
}

/// Copy the current config YAML + credentials.json (whichever exist) into a
/// fresh `snapshot/` directory for the phase. Returns the snapshot dir.
pub fn snapshot(
    config_path: &Path,
    store: &CredentialStore,
    state_root: &Path,
    phase: &str,
) -> Result<PathBuf> {
    // v2 L3 Step 9 B1: ONE snapshot directory per phase (was a per-boot
    // UTC-second timestamp). A one-shot migration has one pre-state; per-boot
    // duplicates only multiplied restore ambiguity.
    let migrations = state_root.join("migrations");
    let dir = snapshot_root(state_root, phase);
    std::fs::create_dir_all(&dir).context("creating migration snapshot directory")?;
    for d in [state_root, &migrations, &snapshot_root(state_root, phase)] {
        std::fs::set_permissions(d, std::fs::Permissions::from_mode(0o700)).ok();
    }

    // Secret-bearing files: atomic 0600 writes (the config snapshot holds
    // the pre-migration plaintext token).
    if config_path.exists() {
        let text = std::fs::read_to_string(config_path)
            .with_context(|| format!("reading {} for snapshot", config_path.display()))?;
        crate::plugins::store::atomic_write_0600(&dir.join(SNAPSHOT_CONFIG_FILE), text.as_bytes())?;
    }
    if store.path().exists() {
        let text = std::fs::read_to_string(store.path())
            .with_context(|| format!("reading {} for snapshot", store.path().display()))?;
        crate::plugins::store::atomic_write_0600(&dir.join(SNAPSHOT_CREDENTIALS_FILE), text.as_bytes())?;
    }
    Ok(dir)
}

/// The plaintext `bridge.discord.token` value from raw YAML, if present.
pub fn read_discord_token(config_path: &Path) -> Result<Option<String>> {
    let text = match std::fs::read_to_string(config_path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).context("reading config for migration"),
    };
    let root: YamlValue = serde_yaml::from_str(&text)
        .with_context(|| format!("parsing {} for migration", config_path.display()))?;
    let token = root
        .get("bridge")
        .and_then(|b| b.get("discord"))
        .and_then(|d| d.get("token"))
        .and_then(|t| t.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    Ok(token)
}

/// Rewrite the config YAML with `bridge.discord.token` removed, leaving a
/// `.bak` copy of the original.
fn strip_discord_token(config_path: &Path) -> Result<()> {
    let text = std::fs::read_to_string(config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    std::fs::write(bak_path(config_path), &text).context("writing config .bak backup")?;
    let mut root: YamlValue = serde_yaml::from_str(&text).context("re-parsing config YAML")?;
    if let Some(map) = root.as_mapping_mut() {
        let bridge_key = YamlValue::String("bridge".into());
        if let Some(bridge) = map.get_mut(&bridge_key) {
            if let Some(b) = bridge.as_mapping_mut() {
                let discord_key = YamlValue::String("discord".into());
                if let Some(discord) = b.get_mut(&discord_key) {
                    if let Some(d) = discord.as_mapping_mut() {
                        d.remove(YamlValue::String("token".into()));
                    }
                }
            }
        }
    }
    let out = serde_yaml::to_string(&root).context("serializing migrated config YAML")?;
    // v2 L3 Step 9 (nit): atomic write — the migrated YAML is the daemon's
    // live config; a torn write here bricks every future boot. Byte-equality
    // for `restore`: the `.bak`/snapshot copies above are the pristine bytes.
    crate::plugins::store::atomic_write_0600(config_path, out.as_bytes())
        .context("writing migrated config YAML")?;
    Ok(())
}

fn bak_path(config_path: &Path) -> PathBuf {
    let mut p = config_path.as_os_str().to_os_string();
    p.push(".bak");
    PathBuf::from(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v1_yaml(token: &str) -> String {
        format!(
            "manifest:\n  base_url: http://127.0.0.1:1/cloud/v1\n  enabled: false\n\
             llama_swap:\n  base_url: http://127.0.0.1:1/v1\n  fallback_model: m\n\
             bonsai:\n  enabled: false\n  fork_path: /nonexistent\n\
             bridge:\n  omp_path: /nonexistent/omp\n  discord:\n    enabled: false\n    token: {token}\n  signal:\n    enabled: false\n"
        )
    }

    /// v2 L3 Step 9 B1: a v1 config WITHOUT a `bonsai:` block must migrate
    /// exactly ONCE: the migrated YAML carries the parity-anchor
    /// `decision_model:` (enabled false, heuristic), the second run is a
    /// no-op, cards survive, and exactly one snapshot directory exists.
    #[test]
    fn p3_without_bonsai_block_is_one_shot() {
        let dir = tempfile::tempdir().unwrap();
        let state = dir.path().join("state");
        let config_dir = dir.path().join("config");
        std::fs::create_dir_all(&config_dir).unwrap();
        let config = dir.path().join("config.yaml");
        std::fs::write(
            &config,
            "manifest:\n  base_url: http://127.0.0.1:1/cloud/v1\n  enabled: false\n\
             llama_swap:\n  base_url: http://127.0.0.1:1/v1\n  fallback_model: m\n\
             routing:\n  main: {backend: local}\n  reviewer: {backend: cloud}\n",
        )
        .unwrap();
        let store = CredentialStore::load(dir.path().join("credentials.json")).unwrap();
        let cards = crate::roles::RoleCardStore::load(state.join("role_cards.json"));

        let actions = run_v2_p3(&config, &store, &state, &config_dir, &cards).unwrap();
        assert!(!actions.is_empty(), "v1 YAML migrates");
        let after_first = std::fs::read_to_string(&config).unwrap();
        assert!(
            after_first.contains("decision_model:")
                && after_first.contains("kind: heuristic")
                && after_first.contains("enabled: false"),
            "no-bonsai v1 gets the parity anchor: {after_first}"
        );
        let cards_first = cards.assignments_snapshot().len();
        assert!(cards_first > 0, "cards seeded");

        // Second boot over the migrated YAML: complete no-op.
        let actions = run_v2_p3(&config, &store, &state, &config_dir, &cards).unwrap();
        assert!(actions.is_empty(), "the migrated YAML is a no-op: {actions:?}");
        assert_eq!(cards.assignments_snapshot().len(), cards_first, "cards survive boot 2");
        assert_eq!(std::fs::read_to_string(&config).unwrap(), after_first, "YAML untouched");

        // Exactly ONE snapshot dir for the phase (no per-boot duplicates).
        let snaps = snapshot_root(&state, PHASE_V2_P3);
        let entries: Vec<_> = std::fs::read_dir(&snaps).unwrap().collect();
        assert_eq!(entries.len(), 1, "one snapshot per phase: {entries:?}");

        // Restore returns the pristine v1 YAML byte-for-byte.
        std::fs::write(&config, "junk: true\n").unwrap();
        let msg = restore(PHASE_V2_P3, &config, &store, &state, &config_dir).unwrap();
        assert!(msg.contains("restored"), "{msg}");
        assert!(std::fs::read_to_string(&config).unwrap().contains("routing:"), "v1 keys back");
    }

    #[test]
    fn migration_moves_the_token_out_of_yaml() {
        let dir = tempfile::tempdir().unwrap();
        let state = dir.path().join("state");
        let config = dir.path().join("config.yaml");
        std::fs::write(&config, v1_yaml("PLAINTEXT-TOK-1234")).unwrap();
        let store = CredentialStore::load(dir.path().join("credentials.json")).unwrap();

        let actions = run_v2_p2(&config, &store, &state).unwrap();
        assert_eq!(actions.len(), 1);
        assert!(!actions[0].contains("PLAINTEXT-TOK-1234"), "actions are log-safe: {}", actions[0]);

        // Value lives in the store, granted to `bridge`.
        assert_eq!(store.resolve("discord_token").unwrap().expose(), "PLAINTEXT-TOK-1234");
        assert!(store.get_masked("discord_token").unwrap()["grants"]
            .to_string()
            .contains("bridge"));

        // YAML no longer has the key; .bak keeps the original.
        let after = std::fs::read_to_string(&config).unwrap();
        assert!(!after.contains("PLAINTEXT-TOK-1234"));
        assert!(after.contains("bridge:"), "remaining bridge keys survive: {after}");
        let bak = std::fs::read_to_string(dir.path().join("config.yaml.bak")).unwrap();
        assert!(bak.contains("PLAINTEXT-TOK-1234"));

        // Snapshot under the snapshot-named path carries the pre-migration YAML.
        let snap = newest_snapshot(PHASE_V2_P2, &state).unwrap().expect("snapshot exists");
        let found = std::fs::read_to_string(snap.join(SNAPSHOT_CONFIG_FILE)).unwrap();
        assert!(found.contains("PLAINTEXT-TOK-1234"));

        // Idempotent: a second run sees no token key.
        assert!(run_v2_p2(&config, &store, &state).unwrap().is_empty());

        // Restore puts the YAML key (and value) back from the snapshot.
        let msg = restore(PHASE_V2_P2, &config, &store, &state, dir.path()).unwrap();
        assert!(msg.contains("restored v2.0-p2"));
        assert!(std::fs::read_to_string(&config).unwrap().contains("PLAINTEXT-TOK-1234"));
        assert!(store.exists("discord_token").unwrap());
    }

    #[test]
    fn restore_roundtrip_preserves_a_nonempty_vault() {
        // Step 9 B1: the snapshot's credentials file is the vault's own
        // serialization (map-shaped `credentials`); a restore with the vault
        // pre-populated must survive every credential and every grant.
        let dir = tempfile::tempdir().unwrap();
        let state = dir.path().join("state");
        let config = dir.path().join("config.yaml");
        std::fs::write(&config, v1_yaml("TOK-B1")).unwrap();
        let store = CredentialStore::load(dir.path().join("credentials.json")).unwrap();
        store
            .put(CredentialRecord {
                id: "api_key".into(),
                description: "pre-existing".into(),
                kind: CredentialKind::Store {
                    value: crate::credentials::SecretString::new("pre-vault-secret"),
                },
                grants: vec!["other".into(), "bridge".into()],
                created_at: String::new(),
                updated_at: String::new(),
            })
            .unwrap();
        store
            .put(CredentialRecord {
                id: "signal_key".into(),
                description: String::new(),
                kind: CredentialKind::Env { var: "SIG_KEY".into() },
                grants: vec!["bridge".into()],
                created_at: String::new(),
                updated_at: String::new(),
            })
            .unwrap();

        run_v2_p2(&config, &store, &state).unwrap();
        assert!(store.exists("discord_token").unwrap());

        // Wipe a marker credential added after the migration, then restore.
        store
            .put(CredentialRecord {
                id: "post_migration".into(),
                description: String::new(),
                kind: CredentialKind::Store {
                    value: crate::credentials::SecretString::new("x"),
                },
                grants: vec![],
                created_at: String::new(),
                updated_at: String::new(),
            })
            .unwrap();
        let msg = restore(PHASE_V2_P2, &config, &store, &state, dir.path()).unwrap();
        assert!(msg.contains("credentials store"), "{msg}");

        // Every pre-migration credential + grant survives.
        let ids: Vec<String> = store
            .list_masked()
            .unwrap()
            .iter()
            .map(|c| c["id"].as_str().unwrap().to_string())
            .collect();
        assert!(ids.contains(&"api_key".to_string()), "{ids:?}");
        assert!(ids.contains(&"signal_key".to_string()), "{ids:?}");
        assert!(!ids.contains(&"discord_token".to_string()), "pre-migration vault had none");
        assert!(!ids.contains(&"post_migration".to_string()), "restore is the snapshot set");
        assert_eq!(store.resolve("api_key").unwrap().expose(), "pre-vault-secret");
        let masked = store.get_masked("api_key").unwrap();
        assert_eq!(masked["grants"], serde_json::json!(["other", "bridge"]));
        assert!(std::fs::read_to_string(&config).unwrap().contains("TOK-B1"));
    }

    #[test]
    fn restore_rejects_unreadable_credential_shape() {
        // Corrupt (scalar) credentials shape must bail, not wipe the vault.
        let dir = tempfile::tempdir().unwrap();
        let state = dir.path().join("state");
        let snap_dir = snapshot_root(&state, PHASE_V2_P2).join("20260101T000000Z");
        std::fs::create_dir_all(&snap_dir).unwrap();
        std::fs::write(
            snap_dir.join(SNAPSHOT_CREDENTIALS_FILE),
            r#"{"credentials": 12345}"#,
        )
        .unwrap();
        let config = dir.path().join("config.yaml");
        std::fs::write(&config, v1_yaml("TOK")).unwrap();
        let store = CredentialStore::load(dir.path().join("credentials.json")).unwrap();
        store
            .put(CredentialRecord {
                id: "keep".into(),
                description: String::new(),
                kind: CredentialKind::Store {
                    value: crate::credentials::SecretString::new("v"),
                },
                grants: vec![],
                created_at: String::new(),
                updated_at: String::new(),
            })
            .unwrap();
        let err = restore(PHASE_V2_P2, &config, &store, &state, dir.path()).unwrap_err();
        assert!(err.to_string().contains("shape"), "{err}");
        assert!(store.exists("keep").unwrap(), "vault untouched after a failed restore");
    }

    #[test]
    fn dry_run_previews_without_writing() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.yaml");
        std::fs::write(&config, v1_yaml("TOK-DRY")).unwrap();
        let store_path = dir.path().join("credentials.json");
        let lines = preview_v2_p2(&config, &store_path, &dir.path().join("state")).unwrap();
        let text = lines.join("\n").to_lowercase();
        assert!(text.contains("discord") && text.contains("migrat") && text.contains("dry"));
        assert!(!text.contains("tok-dry"), "preview masks the value");
        assert!(!store_path.exists());
        assert!(!dir.path().join("state").exists());
        assert!(std::fs::read_to_string(&config).unwrap().contains("TOK-DRY"));
    }
    // ── v2.0-p4 (L4 swap catalog import) ───────────────────────────────────

    fn llama_swap_fixture() -> &'static str {
        r#"
startPort: 5800
macros:
  ls: /usr/bin/llama-server
  common: >-
    --jinja -ngl 999
groups:
  pair:
    swap: false
    members: [m1, m2]
models:
  m1:
    cmd: |
      ${ls} --port ${PORT} ${common} --model /models/m1.gguf
  m2:
    cmd: |
      ${ls} --port ${PORT} --model /models/m2.gguf
"#
    }

    #[test]
    fn p4_imports_once_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let state = dir.path().join("state");
        let llama = dir.path().join("llama-swap.yaml");
        std::fs::write(&llama, llama_swap_fixture()).unwrap();
        let catalog_path = dir.path().join(crate::swap_catalog::CATALOG_FILE);
        let budget_path = dir.path().join(crate::swap_scheduler::BUDGET_FILE);

        // Pre-migration state: one CLI-owned entry.
        let mut cli_entry = crate::swap_catalog::CatalogEntry::default();
        cli_entry.id = "cli-model".into();
        cli_entry.files = vec!["/models/cli.gguf".into()];
        let catalog = crate::swap_catalog::SwapCatalog::load(catalog_path.clone()).unwrap();
        catalog.upsert_entry(cli_entry).unwrap();

        let actions = run_v2_p4(&llama, &state, &catalog_path, &budget_path).unwrap();
        assert_eq!(actions.len(), 1, "{actions:?}");
        let after_import = std::fs::read_to_string(&catalog_path).unwrap();
        let loaded = crate::swap_catalog::SwapCatalog::load(catalog_path.clone()).unwrap();
        assert!(loaded.migration_done(PHASE_V2_P4));
        assert_eq!(loaded.entries().len(), 3, "two imported + one CLI entry");
        assert!(loaded.entry("m1").is_some() && loaded.entry("m2").is_some());
        assert_eq!(loaded.groups().get("pair").unwrap().members.len(), 2);

        // Second boot: one-shot no-op, bytes untouched.
        let actions = run_v2_p4(&llama, &state, &catalog_path, &budget_path).unwrap();
        assert!(actions.is_empty(), "{actions:?}");
        assert_eq!(std::fs::read_to_string(&catalog_path).unwrap(), after_import);

        // Mutate, then restore the pre-migration catalog (CLI entry only).
        std::fs::write(&catalog_path, "{}").unwrap();
        let msg = restore_v2_p4(&state, &catalog_path, &budget_path).unwrap();
        assert!(msg.contains("restored"), "{msg}");
        let restored = crate::swap_catalog::SwapCatalog::load(catalog_path.clone()).unwrap();
        assert_eq!(restored.entries().len(), 1);
        assert!(restored.entry("cli-model").is_some());
    }

    #[test]
    fn p4_dry_run_is_inert_and_stable() {
        let dir = tempfile::tempdir().unwrap();
        let state = dir.path().join("state");
        let llama = dir.path().join("llama-swap.yaml");
        std::fs::write(&llama, llama_swap_fixture()).unwrap();
        let catalog_path = dir.path().join(crate::swap_catalog::CATALOG_FILE);
        let first = preview_v2_p4(&llama, &state, &catalog_path).unwrap();
        let second = preview_v2_p4(&llama, &state, &catalog_path).unwrap();
        assert_eq!(first, second);
        let text = first.join("\n");
        assert!(text.contains("would import 2 model"), "{text}");
        assert!(text.contains("dry-run"));
        assert!(!state.exists(), "dry-run writes no state");
        assert!(!catalog_path.exists(), "dry-run writes no catalog");
    }

    #[test]
    fn p4_without_a_llama_swap_config_is_a_quiet_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let state = dir.path().join("state");
        let catalog_path = dir.path().join(crate::swap_catalog::CATALOG_FILE);
        let budget_path = dir.path().join(crate::swap_scheduler::BUDGET_FILE);
        let actions = run_v2_p4(&dir.path().join("missing.yaml"), &state, &catalog_path, &budget_path).unwrap();
        assert!(actions.is_empty());
        assert!(!catalog_path.exists());
    }


}
