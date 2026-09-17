//! Standalone CLI used by `scripts/sync-toolbox-catalog.sh` (and available
//! for manual use) to structurally validate a candidate pair of
//! `toolboxes.json`/`models.json` files *before* they're vendored into
//! `assets/cockpit-catalog/`. Kept as a tiny separate binary (rather than a
//! `brainrouter cli` subcommand) so the sync script — a plain bash script
//! with no other brainrouter dependency — can invoke it directly via
//! `cargo run --quiet --bin toolbox_catalog_check -- <toolboxes.json> <models.json>`
//! without needing the full daemon/config machinery.
//!
//! Exit code: 0 if the candidate files have no *structural* errors (warnings
//! — e.g. an unrecognized backend id — are printed but do not fail the
//! run, per `docs/design/ai-toolbox-cockpit-integration.md` §3: brainrouter
//! must keep accepting new upstream backends before it has code for them).
//! Non-zero otherwise.

use std::{fs, path::PathBuf, process::ExitCode};

use brainrouter::toolbox_catalog::schema_validate::validate_catalog;

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let (Some(toolboxes_path), Some(models_path)) = (args.next(), args.next()) else {
        eprintln!(
            "usage: toolbox_catalog_check <toolboxes.json> <models.json>\n\n\
             Structurally validates a candidate pair of ai-toolbox-cockpit catalog\n\
             files, mirroring (a permissive subset of) upstream's catalog/schema.py.\n\
             See src/toolbox_catalog/schema_validate.rs for exactly what is checked."
        );
        return ExitCode::from(2);
    };

    let toolboxes_path = PathBuf::from(toolboxes_path);
    let models_path = PathBuf::from(models_path);

    let toolboxes_raw = match fs::read_to_string(&toolboxes_path) {
        Ok(contents) => contents,
        Err(error) => {
            eprintln!("failed to read {}: {error}", toolboxes_path.display());
            return ExitCode::FAILURE;
        }
    };
    let models_raw = match fs::read_to_string(&models_path) {
        Ok(contents) => contents,
        Err(error) => {
            eprintln!("failed to read {}: {error}", models_path.display());
            return ExitCode::FAILURE;
        }
    };

    let toolboxes_json: serde_json::Value = match serde_json::from_str(&toolboxes_raw) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{}: invalid JSON: {error}", toolboxes_path.display());
            return ExitCode::FAILURE;
        }
    };
    let models_json: serde_json::Value = match serde_json::from_str(&models_raw) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{}: invalid JSON: {error}", models_path.display());
            return ExitCode::FAILURE;
        }
    };

    let report = validate_catalog(&toolboxes_json, &models_json);

    for warning in &report.warnings {
        println!("warning: {warning}");
    }
    for error in &report.errors {
        eprintln!("error: {error}");
    }

    if report.is_ok() {
        println!(
            "OK: {} structural error(s), {} warning(s)",
            report.errors.len(),
            report.warnings.len()
        );
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "FAILED: {} structural error(s), {} warning(s)",
            report.errors.len(),
            report.warnings.len()
        );
        ExitCode::FAILURE
    }
}
