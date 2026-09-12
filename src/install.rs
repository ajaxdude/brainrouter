//! `brainrouter install <harness>` subcommand.
//!
//! Idempotently patches the target harness config to point at brainrouter.
//! For harnesses that have a dedicated CLI (claude), delegates to that CLI.
//! For harnesses with JSON/TOML configs, merges the brainrouter section.
//!
//! Principle: print what would change, then write it. Never silently overwrite.

use anyhow::{bail, Context, Result};
use clap::Args;
use std::path::{Path, PathBuf};

/// Arguments for the `install` subcommand.
#[derive(Args)]
pub struct InstallArgs {
    /// Target harness to configure: omp, vibe, opencode, codex, droid, claude, pi
    pub harness: String,

    /// Apply changes without prompting (default: prompt)
    #[arg(long)]
    pub yes: bool,

    /// Also update ~/.zshrc with ANTHROPIC_BASE_URL (claude harness only)
    #[arg(long)]
    pub shell_rc: bool,

    /// Path to brainrouter binary (default: self)
    #[arg(long)]
    pub bin: Option<PathBuf>,
}

pub fn run(args: InstallArgs) -> Result<()> {
    let bin = args
        .bin
        .unwrap_or_else(|| std::env::current_exe().unwrap_or_else(|_| PathBuf::from("brainrouter")));

    match args.harness.as_str() {
        "omp" => install_omp(&bin, args.yes),
        "vibe" => install_vibe(&bin, args.yes),
        "opencode" => install_opencode(&bin, args.yes),
        "codex" => install_codex(&bin, args.yes),
        "droid" => install_droid(&bin, args.yes),
        "claude" => install_claude(&bin, args.yes, args.shell_rc),
        "pi" => {
            print_pi_instructions();
            Ok(())
        }
        other => bail!("Unknown harness: {}. Supported: omp, vibe, opencode, codex, droid, claude, pi", other),
    }
}

// ─── OMP ──────────────────────────────────────────────────────────────────────

fn install_omp(bin: &Path, yes: bool) -> Result<()> {
    let mcp_path = home_path(".omp/agent/mcp.json");

    let new_entry = serde_json::json!({
        "mcpServers": {
            "brainrouter": {
                "type": "stdio",
                "command": bin.to_str().unwrap_or("brainrouter"),
                "args": ["mcp", "--socket", uds_socket_path().as_str()],
                "timeout": 300000
            }
        }
    });

    let current: serde_json::Value = read_json_or_empty(&mcp_path)?;

    // Check if already installed
    if current.get("mcpServers").and_then(|m| m.get("brainrouter")).is_some() {
        println!("brainrouter already registered in {}. Updating...", mcp_path.display());
    }

    let merged = merge_json(current, new_entry);
    write_json_with_preview(&mcp_path, &merged, yes)?;
    println!("OMP MCP configured. Restart OMP for changes to take effect.");
    println!();
    println!("Also add brainrouter models to ~/.omp/agent/models.yml:");
    println!("  brainrouter:");
    println!("    baseUrl: http://127.0.0.1:9099/v1");
    println!("    api: openai-completions");
    println!("    auth: none");
    println!("    models:");
    println!("      - id: auto");
    println!("        name: Brainrouter (auto)");
    println!("      - id: local");
    println!("        name: Brainrouter (local)");
    println!("      - id: cloud");
    println!("        name: Brainrouter (cloud)");
    Ok(())
}

// ─── Vibe ─────────────────────────────────────────────────────────────────────

fn install_vibe(_bin: &PathBuf, _yes: bool) -> Result<()> {
    println!("Add this snippet to ~/.vibe/config.toml:");
    println!();
    println!("{}", vibe_snippet());
    Ok(())
}

fn vibe_snippet() -> String {
    let socket = uds_socket_path();
    format!(
        r#"[[providers]]
name = "brainrouter"
api_base = "http://127.0.0.1:9099/v1"
api_style = "openai"
backend = "generic"

[[models]]
name = "brainrouter-auto"
provider = "brainrouter"
alias = "auto"

mcp_servers = [
  {{ name = "brainrouter", command = "brainrouter", args = ["mcp", "--socket", "{}"] }},
]"#,
        socket
    )
}

// ─── OpenCode ─────────────────────────────────────────────────────────────────

fn install_opencode(bin: &Path, yes: bool) -> Result<()> {
    let config_path = home_path(".config/opencode/config.json");

    let new_section = serde_json::json!({
        "provider": {
            "brainrouter": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Brainrouter",
                "options": { "baseURL": "http://127.0.0.1:9099/v1" },
                "models": { "auto": { "model": "auto", "name": "Brainrouter (auto)" } }
            }
        },
        "mcp": {
            "brainrouter": {
                "type": "local",
                "command": [bin.to_str().unwrap_or("brainrouter"), "mcp", "--socket", uds_socket_path().as_str()]
            }
        }
    });

    let current = read_json_or_empty(&config_path)?;
    let merged = merge_json(current, new_section);
    write_json_with_preview(&config_path, &merged, yes)?;
    println!("OpenCode configured.");
    Ok(())
}

// ─── Codex ────────────────────────────────────────────────────────────────────

fn install_codex(_bin: &PathBuf, _yes: bool) -> Result<()> {
    println!("Add this snippet to ~/.codex/config.toml:");
    println!();
    println!("{}", codex_snippet());
    Ok(())
}

fn codex_snippet() -> String {
    let socket = uds_socket_path();
    format!(
        r#"model = "auto"
model_provider = "brainrouter"

[model_providers.brainrouter]
name = "Brainrouter"
base_url = "http://127.0.0.1:9099/v1"
env_key = "BRAINROUTER_API_KEY"

[mcp_servers.brainrouter]
command = "brainrouter"
args = ["mcp", "--socket", "{}"]"#,
        socket
    )
}

// ─── Droid ────────────────────────────────────────────────────────────────────

fn install_droid(bin: &Path, yes: bool) -> Result<()> {
    let mcp_path = home_path(".factory/mcp.json");

    let new_entry = serde_json::json!({
        "mcpServers": {
            "brainrouter": {
                "type": "stdio",
                "command": bin.to_str().unwrap_or("brainrouter"),
                "args": ["mcp", "--socket", uds_socket_path().as_str()]
            }
        }
    });

    let current = read_json_or_empty(&mcp_path)?;
    let merged = merge_json(current, new_entry);
    write_json_with_preview(&mcp_path, &merged, yes)?;

    // Also print the custom_models entry (droid settings UI manages that field)
    println!("\nAlso add this to droid's custom_models settings:");
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "model": "brainrouter-auto",
        "base_url": "http://127.0.0.1:9099/v1",
        "api_key": "not-used",
        "provider": "anthropic"
    })).unwrap());
    println!("\nNote: provider=anthropic routes droid through /v1/messages (Anthropic shim), not /responses.");
    Ok(())
}

// ─── Claude Code ──────────────────────────────────────────────────────────────

fn install_claude(bin: &Path, yes: bool, shell_rc: bool) -> Result<()> {
    let mcp_json = serde_json::json!({
        "type": "stdio",
        "command": bin.to_str().unwrap_or("brainrouter"),
        "args": ["mcp", "--socket", uds_socket_path().as_str()]
    });

    println!("Registering brainrouter MCP server with Claude Code...");
    let mcp_json_str = serde_json::to_string(&mcp_json)?;

    if !yes {
        println!("Will run: claude mcp add-json brainrouter '{}' --scope user", mcp_json_str);
        print!("Proceed? [y/N] ");
        use std::io::{self, BufRead};
        let mut line = String::new();
        io::stdin().lock().read_line(&mut line).ok();
        if !line.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }

    let status = std::process::Command::new("claude")
        .args(["mcp", "add-json", "brainrouter", &mcp_json_str, "--scope", "user"])
        .status()
        .context("Failed to run `claude` CLI. Is Claude Code installed and in PATH?")?;

    if !status.success() {
        bail!("`claude mcp add-json` failed with status: {}", status);
    }
    println!("Claude Code MCP registered.");

    if shell_rc {
        let rc = home_path(".zshrc");
        append_if_missing(&rc, "export ANTHROPIC_BASE_URL=http://127.0.0.1:9099")?;
        append_if_missing(&rc, "export ANTHROPIC_AUTH_TOKEN=not-used")?;
        println!("Shell rc updated: {}. Run `source ~/.zshrc` or open a new shell.", rc.display());
    } else {
        println!("\nTo route Claude Code through brainrouter, also set:");
        println!("  export ANTHROPIC_BASE_URL=http://127.0.0.1:9099");
        println!("  export ANTHROPIC_AUTH_TOKEN=not-used");
        println!("(Run with --shell-rc to append these to ~/.zshrc automatically.)");
    }

    Ok(())
}

// ─── Pi ───────────────────────────────────────────────────────────────────────

fn print_pi_instructions() {
    println!("Pi uses extensions, not MCP. See configs/harness/pi.md for instructions.");
    println!("Review API: http://127.0.0.1:9099/review/api/request-async");
    println!("Dashboard:  http://127.0.0.1:9099/review/");
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn home_path(rel: &str) -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    PathBuf::from(home).join(rel)
}

fn uds_socket_path() -> String {
    // Must match the daemon's own default exactly — install writes MCP
    // configs for the socket the daemon actually creates. When
    // XDG_RUNTIME_DIR is unset (e.g. systemd --user-less headless setups)
    // the daemon falls back to /run/brainrouter.sock, not /run/user/$UID.
    brainrouter::config::default_socket_path()
        .to_string_lossy()
        .into_owned()
}

fn read_json_or_empty(path: &Path) -> Result<serde_json::Value> {
    match std::fs::read_to_string(path) {
        Ok(content) => serde_json::from_str(&content)
            .with_context(|| format!("Refusing to overwrite malformed JSON in {}", path.display())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            if std::fs::symlink_metadata(path)
                .is_ok_and(|metadata| metadata.file_type().is_symlink())
            {
                bail!("Refusing to replace dangling config symlink {}", path.display());
            }
            Ok(serde_json::json!({}))
        }
        Err(error) => Err(error).with_context(|| format!("Failed to read {}", path.display())),
    }
}

/// Deep-merge `patch` into `base`. Array fields in `patch` replace those in `base`.
/// Object fields in `patch` are recursively merged into `base`.
fn merge_json(mut base: serde_json::Value, patch: serde_json::Value) -> serde_json::Value {
    if let (Some(base_obj), Some(patch_obj)) = (base.as_object_mut(), patch.as_object()) {
        for (key, val) in patch_obj {
            let existing = base_obj.entry(key.clone()).or_insert(serde_json::Value::Null);
            if existing.is_object() && val.is_object() {
                *existing = merge_json(existing.clone(), val.clone());
            } else {
                *existing = val.clone();
            }
        }
        base
    } else {
        patch
    }
}

fn write_json_with_preview(path: &Path, value: &serde_json::Value, yes: bool) -> Result<()> {
    let mut pretty = serde_json::to_string_pretty(value)?;
    pretty.push('\n');

    println!("Will write to {}:", path.display());
    println!("{}", pretty);

    if !yes {
        print!("Proceed? [y/N] ");
        use std::io::{self, BufRead};
        let mut line = String::new();
        io::stdin().lock().read_line(&mut line).ok();
        if !line.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }

    let write_path = if std::fs::symlink_metadata(path)
        .is_ok_and(|metadata| metadata.file_type().is_symlink())
    {
        std::fs::canonicalize(path)
            .with_context(|| format!("Failed to resolve config symlink {}", path.display()))?
    } else {
        path.to_path_buf()
    };
    if let Some(parent) = write_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }
    let file_name = write_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("config.json");
    let temp_path = write_path.with_file_name(format!(
        ".{file_name}.brainrouter-{}.tmp",
        uuid::Uuid::new_v4().simple()
    ));
    let result = (|| -> Result<()> {
        use std::io::Write;
        let mut options = std::fs::OpenOptions::new();
        options.create_new(true).write(true);
        let mut file = options
            .open(&temp_path)
            .with_context(|| format!("Failed to create {}", temp_path.display()))?;
        if let Ok(metadata) = std::fs::metadata(&write_path) {
            file.set_permissions(metadata.permissions())
                .with_context(|| format!("Failed to preserve permissions for {}", write_path.display()))?;
        }
        file.write_all(pretty.as_bytes())
            .with_context(|| format!("Failed to write {}", temp_path.display()))?;
        file.sync_all()
            .with_context(|| format!("Failed to sync {}", temp_path.display()))?;
        std::fs::rename(&temp_path, &write_path)
            .with_context(|| format!("Failed to replace {}", write_path.display()))?;
        #[cfg(unix)]
        if let Some(parent) = path.parent() {
            std::fs::File::open(parent)
                .and_then(|directory| directory.sync_all())
                .with_context(|| format!("Failed to sync {}", parent.display()))?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result?;
    Ok(())
}

fn append_if_missing(path: &PathBuf, line: &str) -> Result<()> {
    let existing = std::fs::read_to_string(path).unwrap_or_default();
    if existing.contains(line) {
        return Ok(());
    }
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("Failed to open {}", path.display()))?;
    writeln!(file, "{}", line)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_existing_json_is_never_treated_as_empty() {
        let root = std::env::temp_dir().join(format!(
            "brainrouter-install-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("config.json");
        std::fs::write(&path, "{not json").unwrap();

        let error = read_json_or_empty(&path).unwrap_err().to_string();
        assert!(error.contains("Refusing to overwrite malformed JSON"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{not json");

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn json_replacement_leaves_complete_document() {
        let root = std::env::temp_dir().join(format!(
            "brainrouter-install-write-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("config.json");
        std::fs::write(&path, "{\"old\":true}\n").unwrap();
        let value = serde_json::json!({"new": {"enabled": true}});

        write_json_with_preview(&path, &value, true).unwrap();

        let written: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(written, value);
        assert!(std::fs::read_dir(&root)
            .unwrap()
            .all(|entry| !entry.unwrap().file_name().to_string_lossy().contains(".tmp")));

        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn json_replacement_preserves_config_symlink() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!(
            "brainrouter-install-symlink-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("managed.json");
        let link = root.join("config.json");
        std::fs::write(&target, "{\"old\":true}\n").unwrap();
        symlink(&target, &link).unwrap();
        let value = serde_json::json!({"new": true});

        write_json_with_preview(&link, &value, true).unwrap();

        assert!(std::fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());
        let written: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&target).unwrap()).unwrap();
        assert_eq!(written, value);
        let _ = std::fs::remove_dir_all(root);
    }
}
