//! systemd user-unit probing + pid-scoped takeover (v2 L5).
//!
//! The cutover must be able to take over an engine currently owned by an
//! operator unit (`strix_server.service`, `hipfire-serve.service`) without
//! ever killing a process it cannot prove belongs to that unit. Everything
//! here is pid-scoped: the unit's `MainPID` and the pids in its
//! `ControlGroup`, never a global process-name pattern.
//!
//! The real `systemctl` is replaceable through `BRAINROUTER_SYSTEMCTL` (a
//! wrapper script in fixtures) and the whole runtime is behind
//! [`UnitRuntime`], so the take-over state machine is unit-testable without a
//! systemd session.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct UnitState {
    pub unit: String,
    pub active: bool,
    pub main_pid: Option<u32>,
    pub control_group: Option<String>,
    /// Human-readable probe failure (never fatal: a missing unit is "not
    /// running", which is the attach-not-spawn path).
    pub error: Option<String>,
}

/// One `systemctl` invocation boundary.
pub trait Systemctl: Send + Sync {
    /// `systemctl --user <args...>` output on success.
    fn run(&self, args: &[String]) -> Result<String>;
}

pub struct SystemctlCli {
    pub program: String,
    pub user: bool,
}

impl Default for SystemctlCli {
    fn default() -> Self {
        Self {
            program: std::env::var("BRAINROUTER_SYSTEMCTL")
                .unwrap_or_else(|_| "systemctl".to_string()),
            user: true,
        }
    }
}

impl SystemctlCli {
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            user: true,
        }
    }

    pub fn args(&self, tail: &[&str]) -> Vec<String> {
        let mut args = Vec::new();
        if self.user {
            args.push("--user".to_string());
        }
        args.extend(tail.iter().map(|s| s.to_string()));
        args
    }
}

impl Systemctl for SystemctlCli {
    fn run(&self, args: &[String]) -> Result<String> {
        let output = std::process::Command::new(&self.program)
            .args(args)
            .stdin(std::process::Stdio::null())
            .output()
            .with_context(|| format!("running `{} {}`", self.program, args.join(" ")))?;
        if !output.status.success() {
            anyhow::bail!(
                "`{} {}` failed: {}",
                self.program,
                args.join(" "),
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}

/// Parse `systemctl show -p ActiveState -p MainPID -p ControlGroup` output.
pub fn parse_show(unit: &str, text: &str) -> UnitState {
    let mut state = UnitState {
        unit: unit.to_string(),
        ..UnitState::default()
    };
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        match key.trim() {
            "ActiveState" if value.trim() == "active" => state.active = true,
            "ActiveState" => state.active = false,
            "SubState" if value.trim() == "running" => state.active = true,
            "SubState" => {}
            "MainPID" => {
                state.main_pid = value.trim().parse::<u32>().ok().filter(|pid| *pid > 0);
            }
            "ControlGroup" => {
                let value = value.trim();
                if !value.is_empty() {
                    state.control_group = Some(value.to_string());
                }
            }
            _ => {}
        }
    }
    state
}

/// Probe one unit. A `systemctl` failure becomes `active: false` + `error`
/// (the daemon must boot on hosts without a user session).
pub fn probe_unit(sys: &dyn Systemctl, unit: &str) -> UnitState {
    let args = vec![
        "--user".to_string(),
        "show".to_string(),
        unit.to_string(),
        "-p".to_string(),
        "ActiveState".to_string(),
        "-p".to_string(),
        "MainPID".to_string(),
        "-p".to_string(),
        "ControlGroup".to_string(),
    ];
    match sys.run(&args) {
        Ok(text) => parse_show(unit, &text),
        Err(error) => UnitState {
            unit: unit.to_string(),
            active: false,
            main_pid: None,
            control_group: None,
            error: Some(error.to_string()),
        },
    }
}

/// The unit's pids: `MainPID` plus the members of its control group (read from
/// the cgroup filesystem when available). Never a name pattern.
pub fn unit_pids(state: &UnitState) -> Vec<u32> {
    let mut pids = Vec::new();
    if let Some(pid) = state.main_pid.filter(|pid| *pid > 0) {
        pids.push(pid);
    }
    if let Some(group) = &state.control_group {
        let path = std::path::Path::new("/sys/fs/cgroup")
            .join(group.trim_start_matches('/'))
            .join("cgroup.procs");
        if let Ok(text) = std::fs::read_to_string(path) {
            for pid in text.lines().filter_map(|line| line.trim().parse::<u32>().ok()) {
                if !pids.contains(&pid) {
                    pids.push(pid);
                }
            }
        }
    }
    pids.retain(|pid| *pid > 0 && unsafe { libc::kill(*pid as i32, 0) == 0 });
    pids
}

/// The injectable surface the scheduler talks to.
pub trait UnitRuntime: Send + Sync {
    fn probe(&self, unit: &str) -> UnitState;
    fn pids(&self, unit: &str) -> Vec<u32>;
    fn stop(&self, unit: &str) -> Result<()>;
}

pub struct SystemdUnitRuntime {
    sys: Arc<dyn Systemctl>,
}

impl SystemdUnitRuntime {
    pub fn new(sys: Arc<dyn Systemctl>) -> Self {
        Self { sys }
    }

    pub fn cli() -> Self {
        Self::new(Arc::new(SystemctlCli::default()))
    }
}

impl UnitRuntime for SystemdUnitRuntime {
    fn probe(&self, unit: &str) -> UnitState {
        probe_unit(self.sys.as_ref(), unit)
    }

    fn pids(&self, unit: &str) -> Vec<u32> {
        let state = probe_unit(self.sys.as_ref(), unit);
        unit_pids(&state)
    }

    fn stop(&self, unit: &str) -> Result<()> {
        self.sys.run(&[
            "--user".to_string(),
            "stop".to_string(),
            unit.to_string(),
        ])?;
        Ok(())
    }
}

/// Fixture runtime: scripted units + resolved pids + recorded stops.
pub struct MockUnitRuntime {
    units: Mutex<BTreeMap<String, MockUnit>>,
    pub stops: Mutex<Vec<String>>,
}

#[derive(Debug, Clone, Default)]
pub struct MockUnit {
    pub active: bool,
    pub main_pid: Option<u32>,
    pub pids: Vec<u32>,
    pub control_group: Option<String>,
}

impl MockUnitRuntime {
    pub fn new() -> Self {
        Self {
            units: Mutex::new(BTreeMap::new()),
            stops: Mutex::new(Vec::new()),
        }
    }

    pub fn set(&self, unit: &str, unit_state: MockUnit) {
        self.units
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(unit.to_string(), unit_state);
    }

    pub fn stopped(&self) -> Vec<String> {
        self.stops.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

impl Default for MockUnitRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl UnitRuntime for MockUnitRuntime {
    fn probe(&self, unit: &str) -> UnitState {
        let units = self.units.lock().unwrap_or_else(|e| e.into_inner());
        match units.get(unit) {
            Some(state) => UnitState {
                unit: unit.to_string(),
                active: state.active,
                main_pid: state.main_pid,
                control_group: state.control_group.clone(),
                error: None,
            },
            None => UnitState {
                unit: unit.to_string(),
                active: false,
                main_pid: None,
                control_group: None,
                error: None,
            },
        }
    }

    fn pids(&self, unit: &str) -> Vec<u32> {
        let units = self.units.lock().unwrap_or_else(|e| e.into_inner());
        units.get(unit).map(|state| state.pids.clone()).unwrap_or_default()
    }

    fn stop(&self, unit: &str) -> Result<()> {
        self.stops
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(unit.to_string());
        let mut units = self.units.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(state) = units.get_mut(unit) {
            state.active = false;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_show_output() {
        let state = parse_show(
            "strix_server.service",
            "ActiveState=active\nMainPID=4242\nControlGroup=/user.slice/user-1000.slice/strix_server.service\n",
        );
        assert!(state.active);
        assert_eq!(state.main_pid, Some(4242));
        assert!(state.control_group.is_some());
    }

    #[test]
    fn failed_probe_is_not_running_not_fatal() {
        struct Failing;
        impl Systemctl for Failing {
            fn run(&self, _args: &[String]) -> Result<String> {
                anyhow::bail!("no user session")
            }
        }
        let state = probe_unit(&Failing, "hipfire-serve.service");
        assert!(!state.active);
        assert!(state.error.is_some());
        assert!(unit_pids(&state).is_empty());
    }

    #[test]
    fn mock_runtime_stops_the_named_unit_only() {
        let runtime = MockUnitRuntime::new();
        runtime.set(
            "hipfire-serve.service",
            MockUnit {
                active: true,
                main_pid: Some(77),
                pids: vec![77, 78],
                control_group: None,
            },
        );
        assert_eq!(runtime.pids("hipfire-serve.service"), vec![77, 78]);
        runtime.stop("hipfire-serve.service").unwrap();
        assert_eq!(runtime.stopped(), vec!["hipfire-serve.service".to_string()]);
        assert!(!runtime.probe("hipfire-serve.service").active);
    }
}
