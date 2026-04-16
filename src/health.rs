use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const FAILURE_THRESHOLD: u32 = 3;
const COOLDOWN_PERIOD: Duration = Duration::from_secs(60);

#[derive(Clone)]
pub enum HealthState {
    Healthy,
    Degraded(u32, Instant), // failures, last failure time
    Broken(Instant),      // time when broken
}

pub struct HealthTracker {
    states: Mutex<HashMap<String, HealthState>>,
}

impl HealthTracker {
    pub fn new() -> Self {
        Self {
            states: Mutex::new(HashMap::new()),
        }
    }

    pub fn is_healthy(&self, provider_id: &str) -> bool {
        let mut states = self.states.lock().unwrap();
        let state = states.entry(provider_id.to_string()).or_insert(HealthState::Healthy);

        match state {
            HealthState::Healthy => true,
            HealthState::Degraded(_, _) => true,
            HealthState::Broken(broken_at) => {
                if broken_at.elapsed() > COOLDOWN_PERIOD {
                    *state = HealthState::Healthy;
                    true
                } else {
                    false
                }
            }
        }
    }

    pub fn report_failure(&self, provider_id: &str) {
        let mut states = self.states.lock().unwrap();
        let state = states.entry(provider_id.to_string()).or_insert(HealthState::Healthy);

        let now = Instant::now();
        *state = match state {
            HealthState::Healthy => HealthState::Degraded(1, now),
            HealthState::Degraded(failures, _last_failure) => {
                let new_failures = *failures + 1;
                if new_failures >= FAILURE_THRESHOLD {
                    HealthState::Broken(now)
                } else {
                    HealthState::Degraded(new_failures, now)
                }
            }
            HealthState::Broken(_) => return, // Already broken
        };
    }

    pub fn report_success(&self, provider_id: &str) {
        let mut states = self.states.lock().unwrap();
        if let Some(state) = states.get_mut(provider_id) {
            *state = HealthState::Healthy;
        }
    }
}
