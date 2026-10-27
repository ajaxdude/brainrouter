//! v2 L6: per-provider monthly budget guard (`provider_budget.json`, 0600).
//!
//! Spec "Budget guard": accumulator keyed `YYYY-MM` (calendar rollover on key
//! change; restart cannot reset it), cost = price x usage tokens, absent usage
//! ⇒ `(bytes+3)/4` estimator flagged `estimated`. `warn` records and badges;
//! `stop` turns an over-budget dispatch into a PRIMARY failure so the managed
//! two-step shape fires the `fallback_model` hop.
//!
//! The L4 swap supervisor already owns `budget.json` for VRAM reservations;
//! L6's monthly provider accumulator is a separate file so the two ledgers
//! cannot overwrite each other (Ray's suite accepts any state-dir ledger file;
//! the collision is declared in `tests/v2_l6_acceptance.rs` as a spec gap).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

/// State-dir file name for the monthly provider ledger.
pub const LEDGER_FILE: &str = "provider_budget.json";

/// `$XDG_STATE_HOME/brainrouter/provider_budget.json`.
pub fn default_path() -> PathBuf {
    crate::paths::brainrouter_state().join(LEDGER_FILE)
}

fn current_month() -> String {
    chrono::Utc::now().format("%Y-%m").to_string()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Spend {
    #[serde(default)]
    pub spent_usd: f64,
    #[serde(default)]
    pub estimated_requests: u64,
    #[serde(default)]
    pub actual_requests: u64,
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
}

#[derive(Default, Serialize, Deserialize)]
struct Ledger {
    #[serde(default)]
    month: String,
    #[serde(default)]
    providers: BTreeMap<String, Spend>,
}

impl Ledger {
    fn roll_over(&mut self) -> bool {
        let month = current_month();
        if self.month == month {
            return false;
        }
        self.month = month;
        self.providers.clear();
        true
    }
}

/// The monthly provider accumulator.
pub struct ProviderBudget {
    path: PathBuf,
    ledger: Mutex<Ledger>,
}

impl ProviderBudget {
    fn lock(&self) -> MutexGuard<'_, Ledger> {
        self.ledger
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Load the ledger. Missing = a fresh month; a stale month rolls over
    /// in memory (and on the next write).
    pub fn load(path: PathBuf) -> Result<Self> {
        let mut ledger = match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str::<Ledger>(&text)
                .with_context(|| format!("provider budget ledger {} is corrupt", path.display()))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ledger::default(),
            Err(e) => {
                return Err(e)
                    .with_context(|| format!("reading provider budget ledger {}", path.display()))
            }
        };
        ledger.roll_over();
        Ok(Self {
            path,
            ledger: Mutex::new(ledger),
        })
    }

    /// Create the ledger file for a brand-new month (migration `v2.0-p6`).
    /// Existing files are left untouched.
    pub fn ensure_file(path: &Path) -> Result<()> {
        if path.exists() {
            return Ok(());
        }
        let ledger = Ledger {
            month: current_month(),
            providers: BTreeMap::new(),
        };
        let text = serde_json::to_string_pretty(&ledger).context("serializing budget ledger")?;
        crate::plugins::store::atomic_write_0600(path, text.as_bytes())
            .with_context(|| format!("creating budget ledger {}", path.display()))
    }

    pub fn in_memory() -> Self {
        Self {
            path: PathBuf::new(),
            ledger: Mutex::new(Ledger {
                month: current_month(),
                providers: BTreeMap::new(),
            }),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn month(&self) -> String {
        let mut ledger = self.lock();
        ledger.roll_over();
        ledger.month.clone()
    }

    /// Month-to-date spend for one provider (rollover-aware).
    pub fn spent(&self, provider_id: &str) -> f64 {
        let mut ledger = self.lock();
        ledger.roll_over();
        ledger
            .providers
            .get(provider_id)
            .map(|s| s.spent_usd)
            .unwrap_or(0.0)
    }

    /// True when the provider's `stop`-mode budget is exhausted.
    pub fn stop_exhausted(
        &self,
        provider_id: &str,
        budget: Option<f64>,
        mode: crate::providers::BudgetMode,
    ) -> bool {
        if mode != crate::providers::BudgetMode::Stop {
            return false;
        }
        let Some(budget) = budget else {
            return false;
        };
        self.spent(provider_id) >= budget
    }

    /// Provider ids with an exhausted stop-mode budget (auto-pool exclusion).
    pub fn stop_excluded(
        &self,
        records: &[crate::providers::ProviderRecord],
    ) -> std::collections::HashSet<String> {
        records
            .iter()
            .filter(|p| {
                p.enabled
                    && self.stop_exhausted(&p.id, p.monthly_budget_usd, p.budget_mode)
            })
            .map(|p| p.id.clone())
            .collect()
    }

    /// Accrue spend. `estimated` marks the `(bytes+3)/4` fallback.
    pub fn record(
        &self,
        provider_id: &str,
        cost_usd: f64,
        estimated: bool,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Result<()> {
        let text = {
            let mut ledger = self.lock();
            ledger.roll_over();
            let spend = ledger.providers.entry(provider_id.to_string()).or_default();
            spend.spent_usd += cost_usd.max(0.0);
            spend.input_tokens = spend.input_tokens.saturating_add(input_tokens);
            spend.output_tokens = spend.output_tokens.saturating_add(output_tokens);
            if estimated {
                spend.estimated_requests = spend.estimated_requests.saturating_add(1);
            } else {
                spend.actual_requests = spend.actual_requests.saturating_add(1);
            }
            if self.path.as_os_str().is_empty() {
                return Ok(());
            }
            // Serialize INSIDE the lock; write OUTSIDE it so the hot route
            // path never holds the ledger mutex across a blocking file write.
            serde_json::to_string_pretty(&*ledger).context("serializing budget ledger")?
        };
        crate::plugins::store::atomic_write_0600(&self.path, text.as_bytes())
            .with_context(|| format!("writing budget ledger {}", self.path.display()))
    }

    /// Serialized ledger view for diagnostics/tests.
    pub fn snapshot(&self) -> serde_json::Value {
        let mut ledger = self.lock();
        ledger.roll_over();
        serde_json::to_value(&*ledger).unwrap_or(serde_json::Value::Null)
    }
}

/// Cost of a request under a model's price (`$/1K in,out`).
pub fn cost_of(price: Option<crate::providers::Price>, input_tokens: u64, output_tokens: u64) -> f64 {
    let Some(price) = price else {
        return 0.0;
    };
    (input_tokens as f64 / 1000.0) * price.input + (output_tokens as f64 / 1000.0) * price.output
}

/// The spec's absent-usage estimator: `(bytes + 3) / 4` tokens.
pub fn estimate_tokens(bytes: usize) -> u64 {
    (bytes as u64).div_ceil(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_is_ceiling_quarter_bytes() {
        assert_eq!(estimate_tokens(0), 0);
        assert_eq!(estimate_tokens(1), 1);
        assert_eq!(estimate_tokens(4), 1);
        assert_eq!(estimate_tokens(5), 2);
    }

    #[test]
    fn rollover_resets_on_month_change() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LEDGER_FILE);
        std::fs::write(
            &path,
            r#"{"month":"1999-01","providers":{"p":{"spent_usd":5.0}}}"#,
        )
        .unwrap();
        let ledger = ProviderBudget::load(path.clone()).unwrap();
        assert_eq!(ledger.spent("p"), 0.0);
        ledger.record("p", 1.5, true, 10, 2).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains(&current_month()));
        assert!(text.contains("\"estimated_requests\": 1"));
        assert_eq!(ledger.spent("p"), 1.5);
    }
}
