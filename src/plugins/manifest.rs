//! Plugin manifest (`plugin.toml`) parsing and validation.
//!
//! Keys (design `docs/design/v2-streamlining.L2.md`): `id`, `name`,
//! `version`, `binary`, `min_daemon_version`, `grants` (array of RPC method
//! names), and an optional `config_schema` path.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::Path;

/// RPC grants a manifest may request. The RPC v1 surface is closed: plugins
/// get NO system ops (no restart/flush/upgrade methods).
pub const KNOWN_GRANTS: &[&str] = &[
    "chat.completions",
    "models.list",
    "routing.events.tail",
    "inflight.list",
    "config.get",
    "config.set",
    "kv.get",
    "kv.set",
    "secrets.get",
    "notify",
    "set_status",
];

#[derive(Debug, Clone, Deserialize)]
pub struct PluginManifest {
    pub id: String,
    pub name: String,
    pub version: String,
    pub binary: String,
    #[serde(default = "default_min_daemon_version")]
    pub min_daemon_version: String,
    #[serde(default)]
    pub grants: Vec<String>,
    /// Path (relative to the plugin dir) of the config-schema file. Optional;
    /// fixtures assert no schema.
    #[serde(default)]
    pub config_schema: Option<String>,
}

fn default_min_daemon_version() -> String {
    "1.0.0".to_string()
}

impl PluginManifest {
    /// Parse + validate `plugin.toml`. The id is restricted to a filesystem-
    /// safe token (it names the plugin's state dir and socket path).
    pub fn parse(text: &str) -> Result<Self> {
        let manifest: PluginManifest =
            toml::from_str(text).context("parsing plugin.toml")?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn load(dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("plugin.toml"))
            .with_context(|| format!("reading {}/plugin.toml", dir.display()))?;
        Self::parse(&text)
    }

    pub fn validate(&self) -> Result<()> {
        if !self
            .id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
            || self.id.is_empty()
            || self.id == "."
            || self.id == ".."
        {
            bail!(
                "plugin id {:?} is invalid: use letters, digits, '-' and '_'",
                self.id
            );
        }
        if self.name.trim().is_empty() {
            bail!("plugin manifest needs a name");
        }
        if self.binary.contains('/') || self.binary.is_empty() {
            bail!("plugin binary must be a plain file name in the plugin directory");
        }
        for g in &self.grants {
            if !KNOWN_GRANTS.contains(&g.as_str()) {
                bail!("plugin grant {g:?} is not part of the RPC v1 surface");
            }
        }
        parse_version(&self.min_daemon_version)
            .context("min_daemon_version must be a `x.y.z` version")?;
        Ok(())
    }
}

/// Parse `x.y.z` into a comparable tuple.
pub fn parse_version(v: &str) -> Result<(u64, u64, u64)> {
    let mut parts = [0u64; 3];
    for (i, p) in v.split('.').take(3).enumerate() {
        parts[i] = p
            .trim_end_matches(|c: char| !c.is_ascii_digit() && c != '-' && c != '+')
            .split(['-', '+'])
            .next()
            .unwrap_or("0")
            .parse()
            .with_context(|| format!("bad version component {p:?} in {v:?}"))?;
    }
    Ok((parts[0], parts[1], parts[2]))
}

/// True when `daemon >= required`.
pub fn daemon_version_satisfies(daemon: &str, required: &str) -> bool {
    match (parse_version(daemon), parse_version(required)) {
        (Ok(d), Ok(r)) => d >= r,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_roundtrip_and_validation() {
        let m = PluginManifest::parse(
            "id = \"bridge\"\nname = \"Bridge\"\nversion = \"0.1.0\"\nbinary = \"brainrouter-bridge\"\nmin_daemon_version = \"1.0.0\"\ngrants = [\"chat.completions\", \"secrets.get\"]\n",
        )
        .unwrap();
        assert_eq!(m.id, "bridge");
        assert_eq!(m.grants.len(), 2);
        // Unknown grant rejected.
        assert!(PluginManifest::parse(
            "id = \"x\"\nname = \"X\"\nversion = \"1\"\nbinary = \"x\"\ngrants = [\"system.restart\"]\n"
        )
        .is_err());
        // Id path traversal rejected.
        assert!(PluginManifest::parse(
            "id = \"../evil\"\nname = \"E\"\nversion = \"1\"\nbinary = \"x\"\n"
        )
        .is_err());
        // Mixed-case ids (Ray's fixtures) are valid.
        assert!(PluginManifest::parse(
            "id = \"rayA\"\nname = \"A\"\nversion = \"1\"\nbinary = \"x\"\n"
        )
        .is_ok());
        // Missing min_daemon_version defaults.
        let m = PluginManifest::parse(
            "id = \"x\"\nname = \"X\"\nversion = \"1.2.3\"\nbinary = \"x.bin\"\n",
        )
        .unwrap();
        assert_eq!(m.min_daemon_version, "1.0.0");
    }

    #[test]
    fn version_compare() {
        assert!(daemon_version_satisfies("1.1.8", "1.0.0"));
        assert!(daemon_version_satisfies("2.0.0", "2.0.0"));
        assert!(!daemon_version_satisfies("1.1.8", "1.2.0"));
    }
}
