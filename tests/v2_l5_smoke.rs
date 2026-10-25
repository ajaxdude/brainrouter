//! L5 integration smoke (builder-authored, not an acceptance suite).
//!
//! Exercises the leg's fixture-tier promises end-to-end through the public
//! library surface with scripted mock engines (L4's `FakeAdapter` real-child
//! argv pattern), an injectable `/proc` source, an injectable unit runtime and
//! an injectable `MemAvailable` probe:
//!
//! * takeover attach-not-spawn from an observed external resident;
//! * `min_mem_gib` refusal (named `blocked:min-mem-gate`) + `--force-gate`;
//! * Active-Consumer protection: live stream and grace span both block
//!   evict-to-fit with a `blocked:active-consumer` ledger event; the
//!   one-action override bypasses it;
//! * exclusive cutover: same-device ensure drains the predecessor before
//!   spawning (never co-resident) and the SIGTERM-point re-check refuses a
//!   newly protected predecessor.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use serde_json::json;

use brainrouter::active_consumers::ActiveConsumers;
use brainrouter::engine_observer::{ExternalObserver, ProcInfo, ProcSource};
use brainrouter::swap_catalog::{CatalogEntry, SwapCatalog};
use brainrouter::swap_scheduler::{
    BudgetLedger, EngineAdapter, EngineHandle, EnsureOptions, FakeAdapter, HealthReport,
    LoadProgress, SpawnRequest, SwapScheduler, SwapSchedulerConfig, TakeoverOptions,
};

// ── Fixtures ────────────────────────────────────────────────────────────────

fn root(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "brainrouter-l5-smoke-{name}-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn fake_entry(id: &str, gguf: &std::path::Path, estimate_gib: f64) -> CatalogEntry {
    serde_json::from_value(json!({
        "id": id,
        "engine": "fake",
        "files": [gguf.display().to_string()],
        "ctx": 4096,
        "vram_estimate_gib": estimate_gib,
        "launch_spec": {
            "program": "fake",
            "args": [],
            "env": {"FAKE_LOAD_MS": "0"},
            "port_policy": "managed",
            "served_model_alias": id,
        },
        "provenance": {"source": "cli"},
        "fingerprints": [],
    }))
    .expect("fake catalog entry")
}

struct Fixture {
    scheduler: Arc<SwapScheduler>,
    fake: Arc<FakeAdapter>,
    catalog: Arc<SwapCatalog>,
    dir: PathBuf,
}

fn fixture(name: &str, models: &[(&str, f64)], cap: f64) -> Fixture {
    let dir = root(name);
    let catalog = Arc::new(SwapCatalog::load(dir.join("swap_catalog.json")).unwrap());
    for (id, gib) in models {
        let gguf = dir.join(format!("{id}.gguf"));
        std::fs::write(&gguf, b"gguf").unwrap();
        catalog.upsert_entry(fake_entry(id, &gguf, *gib)).unwrap();
    }
    let ledger = Arc::new(BudgetLedger::load(dir.join("budget.json")).unwrap());
    let fake = Arc::new(FakeAdapter::new());
    let mut adapters: BTreeMap<String, Arc<dyn EngineAdapter>> = BTreeMap::new();
    adapters.insert("fake".into(), fake.clone() as Arc<dyn EngineAdapter>);
    let config = SwapSchedulerConfig {
        wait_timeout: Duration::from_secs(5),
        queue_cap: 4,
        stop_grace: Duration::from_millis(500),
        max_restarts: 1,
        port_min: 15700,
        port_max: 15799,
        llama_swap_url: "http://127.0.0.1:1".into(),
        settle: Duration::ZERO,
        cutover_recheck: Duration::from_millis(40),
        gate_poll: Duration::from_millis(10),
        gate_timeout: Duration::from_millis(150),
        ..SwapSchedulerConfig::default()
    };
    let scheduler = Arc::new(
        SwapScheduler::new(Arc::clone(&catalog), ledger, config, adapters)
            .with_state_root(dir.clone())
            .with_kill_children_on_drop(true)
            .with_budget_cap(cap),
    );
    Fixture {
        scheduler,
        fake,
        catalog,
        dir,
    }
}

fn blocks(error: &anyhow::Error, needle: &str) -> bool {
    format!("{error:#}").contains(needle)
}

fn ledger_has(scheduler: &SwapScheduler, kind: &str, needle: &str) -> bool {
    scheduler.ledger().entries().iter().any(|entry| {
        entry.kind == kind && entry.reason.contains(needle)
    })
}

// ── (4) external residents: attach-not-spawn + takeover ────────────────────

struct FixtureProcs(Vec<ProcInfo>);

impl ProcSource for FixtureProcs {
    fn scan(&self) -> Vec<ProcInfo> {
        self.0.clone()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn takeover_attaches_to_an_observed_external_without_spawning() {
    let fixture = fixture("external-attach", &[("icon-model", 10.0)], 100.0);
    // The icon-launched server: a real child (so adoption can validate the
    // pid) whose observed argv names strixite on 5991.
    let mut icon = std::process::Command::new("sleep")
        .arg("30")
        .stdin(std::process::Stdio::null())
        .spawn()
        .expect("icon child");
    let icon_pid = icon.id();
    let observer = Arc::new(
        ExternalObserver::new(
            Arc::new(FixtureProcs(vec![ProcInfo {
                pid: icon_pid,
                cmdline: "strix_server --config /tmp/x.conf --weights /models/icon.gguf --port 5991"
                    .into(),
                comm: "strix_server".into(),
            }])),
            fixture.dir.join("adoption"),
            Arc::clone(&fixture.catalog),
        )
        .with_probe(Arc::new(|_| Some(vec!["icon-model".to_string()]))),
    );
    let scheduler = Arc::new(
        SwapScheduler::new(
            Arc::clone(&fixture.catalog),
            Arc::new(BudgetLedger::load(fixture.dir.join("budget.json")).unwrap()),
            SwapSchedulerConfig {
                settle: Duration::ZERO,
                ..SwapSchedulerConfig::default()
            },
            {
                let mut adapters: BTreeMap<String, Arc<dyn EngineAdapter>> = BTreeMap::new();
                adapters.insert("fake".into(), fixture.fake.clone() as Arc<dyn EngineAdapter>);
                adapters
            },
        )
        .with_state_root(fixture.dir.clone())
        .with_kill_children_on_drop(true)
        .with_external_observer(observer),
    );
    scheduler.refresh_external();
    let status = scheduler.status();
    assert_eq!(status.external.len(), 1, "{:?}", status.external);
    assert_eq!(status.external[0].catalog_id.as_deref(), Some("icon-model"));
    assert!(!status.external[0].adopted);

    let outcome = scheduler.ensure("icon-model").await.unwrap();
    assert!(!outcome.spawned(), "attach must not spawn");
    assert_eq!(outcome.endpoint(), "http://127.0.0.1:5991");
    assert_eq!(fixture.fake.spawn_count(), 0);

    let report = scheduler
        .takeover("icon-model", TakeoverOptions::default())
        .await
        .unwrap();
    assert!(report.attached && !report.spawned, "{report:?}");
    assert_eq!(fixture.fake.spawn_count(), 0);

    // Adopting the pid writes a marker (supervisor-owned from now on) and the
    // model keeps its Ready-at-external endpoint.
    let adopted = scheduler.adopt_external("icon-model", None).unwrap();
    assert_eq!(adopted.pid, icon_pid);
    assert!(fixture.dir.join("adoption/icon-model.json").exists());
    assert!(scheduler.status().external[0].adopted);
    let _ = icon.kill();
    let _ = icon.wait();
}

// ── min_mem gate ───────────────────────────────────────────────────────────

/// A fake adapter with an L5 memory floor, delegating the engine behavior.
struct FlooredFake {
    inner: Arc<FakeAdapter>,
    min_mem_gib: f64,
}

impl EngineAdapter for FlooredFake {
    fn name(&self) -> &'static str {
        "fake"
    }
    fn spawn(&self, request: &SpawnRequest) -> Result<EngineHandle> {
        self.inner.spawn(request)
    }
    fn stop(&self, handle: &EngineHandle) -> Result<()> {
        self.inner.stop(handle)
    }
    fn health(&self, endpoint: &str, handle: &EngineHandle) -> Result<HealthReport> {
        self.inner.health(endpoint, handle)
    }
    fn progress(&self, endpoint: &str, handle: &EngineHandle) -> LoadProgress {
        self.inner.progress(endpoint, handle)
    }
    fn min_mem_gib(&self) -> Option<f64> {
        Some(self.min_mem_gib)
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn min_mem_gate_refuses_by_name_and_force_gate_proceeds() {
    let dir = root("gate");
    let catalog = Arc::new(SwapCatalog::load(dir.join("swap_catalog.json")).unwrap());
    let gguf = dir.join("big.gguf");
    std::fs::write(&gguf, b"gguf").unwrap();
    catalog.upsert_entry(fake_entry("big", &gguf, 10.0)).unwrap();
    let fake = Arc::new(FakeAdapter::new());
    let mut adapters: BTreeMap<String, Arc<dyn EngineAdapter>> = BTreeMap::new();
    adapters.insert(
        "fake".into(),
        Arc::new(FlooredFake {
            inner: Arc::clone(&fake),
            min_mem_gib: 100.0,
        }) as Arc<dyn EngineAdapter>,
    );
    let scheduler = Arc::new(
        SwapScheduler::new(
            Arc::clone(&catalog),
            Arc::new(BudgetLedger::load(dir.join("budget.json")).unwrap()),
            SwapSchedulerConfig {
                settle: Duration::ZERO,
                gate_poll: Duration::from_millis(10),
                gate_timeout: Duration::from_millis(120),
                wait_timeout: Duration::from_secs(5),
                port_min: 15850,
                port_max: 15899,
                ..SwapSchedulerConfig::default()
            },
            adapters,
        )
        .with_state_root(dir.clone())
        .with_kill_children_on_drop(true)
        .with_budget_cap(200.0)
        .with_mem_source(Arc::new(|| 8.0)),
    );
    let error = scheduler.ensure("big").await.unwrap_err();
    assert!(blocks(&error, "blocked:min-mem-gate"), "{error:#}");
    assert!(ledger_has(&scheduler, "blocked", "blocked:min-mem-gate"));
    assert!(ledger_has(&scheduler, "gate-fail", "min_mem gate"));
    let status = scheduler.status();
    let model = status.models.iter().find(|m| m.id == "big").unwrap();
    assert_eq!(model.state, "failed", "{model:?}");
    assert_eq!(fake.spawn_count(), 0);

    let outcome = scheduler
        .ensure_with_options(
            "big",
            None,
            EnsureOptions {
                force_gate: true,
                ..EnsureOptions::default()
            },
        )
        .await
        .unwrap();
    assert!(outcome.spawned());
    assert!(ledger_has(&scheduler, "override", "force-gate"));
}

// ── (3) Active-Consumer Set ────────────────────────────────────────────────

fn acs_fixture(name: &str, grace: Duration) -> (Fixture, Arc<ActiveConsumers>) {
    let dir = root(name);
    let catalog = Arc::new(SwapCatalog::load(dir.join("swap_catalog.json")).unwrap());
    for id in ["a", "b"] {
        let gguf = dir.join(format!("{id}.gguf"));
        std::fs::write(&gguf, b"gguf").unwrap();
        catalog.upsert_entry(fake_entry(id, &gguf, 10.0)).unwrap();
    }
    let fake = Arc::new(FakeAdapter::new());
    let mut adapters: BTreeMap<String, Arc<dyn EngineAdapter>> = BTreeMap::new();
    adapters.insert("fake".into(), fake.clone() as Arc<dyn EngineAdapter>);
    let consumers = Arc::new(ActiveConsumers::new(grace));
    let scheduler = Arc::new(
        SwapScheduler::new(
            Arc::clone(&catalog),
            Arc::new(BudgetLedger::load(dir.join("budget.json")).unwrap()),
            SwapSchedulerConfig {
                settle: Duration::ZERO,
                cutover_recheck: Duration::from_millis(20),
                port_min: 15900,
                port_max: 15949,
                ..SwapSchedulerConfig::default()
            },
            adapters,
        )
        .with_state_root(dir.clone())
        .with_kill_children_on_drop(true)
        .with_budget_cap(10.0)
        .with_active_consumers(Arc::clone(&consumers)),
    );
    (
        Fixture {
            scheduler,
            fake,
            catalog,
            dir,
        },
        consumers,
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn in_process_stream_blocks_evict_to_fit_and_override_bypasses_it() {
    let (fixture, consumers) = acs_fixture("acs-live", Duration::from_secs(0));
    fixture.scheduler.ensure("a").await.unwrap();
    consumers.enter("a"); // the router's in-process stream span

    let error = fixture.scheduler.ensure("b").await.unwrap_err();
    assert!(blocks(&error, "blocked:active-consumer"), "{error:#}");
    assert!(
        ledger_has(&fixture.scheduler, "blocked", "blocked:active-consumer"),
        "{:?}",
        fixture.scheduler.ledger().entries()
    );
    let status = fixture.scheduler.status();
    let a = status.models.iter().find(|m| m.id == "a").unwrap();
    assert_eq!(a.state, "ready", "protected model must stay up: {a:?}");
    assert!(a.active_consumer.as_ref().is_some_and(|span| span.protected));

    // One action: `--override` (dashboard "Swap anyway").
    let outcome = fixture
        .scheduler
        .ensure_with_options(
            "b",
            None,
            EnsureOptions {
                override_active_consumers: true,
                ..EnsureOptions::default()
            },
        )
        .await
        .unwrap();
    assert!(outcome.spawned());
    assert!(ledger_has(&fixture.scheduler, "override", "active-consumer override"));
}

#[tokio::test(flavor = "multi_thread")]
async fn grace_window_bridges_the_gap_then_expires() {
    let (fixture, consumers) = acs_fixture("acs-grace", Duration::from_millis(80));
    fixture.scheduler.ensure("a").await.unwrap();
    consumers.enter("a");
    consumers.leave("a"); // stream ended: grace span starts

    let error = fixture.scheduler.ensure("b").await.unwrap_err();
    assert!(blocks(&error, "blocked:active-consumer"), "{error:#}");

    tokio::time::sleep(Duration::from_millis(120)).await;
    let outcome = fixture.scheduler.ensure("b").await.unwrap();
    assert!(outcome.spawned(), "grace expired: eviction proceeds");
}

// ── exclusive cutover + SIGTERM-point re-check ─────────────────────────────

#[derive(Default)]
struct Steps {
    spawns: Vec<String>,
    stops: Vec<String>,
}

struct ExclusiveFake {
    inner: Arc<FakeAdapter>,
    steps: Arc<Mutex<Steps>>,
}

impl EngineAdapter for ExclusiveFake {
    fn name(&self) -> &'static str {
        "fake"
    }
    fn spawn(&self, request: &SpawnRequest) -> Result<EngineHandle> {
        self.steps
            .lock()
            .unwrap()
            .spawns
            .push(request.model_id.clone());
        self.inner.spawn(request)
    }
    fn stop(&self, handle: &EngineHandle) -> Result<()> {
        self.steps.lock().unwrap().stops.push(handle.endpoint.clone());
        self.inner.stop(handle)
    }
    fn health(&self, endpoint: &str, handle: &EngineHandle) -> Result<HealthReport> {
        self.inner.health(endpoint, handle)
    }
    fn exclusive_key(&self) -> Option<String> {
        Some("device:test".into())
    }
    fn min_mem_gib(&self) -> Option<f64> {
        None
    }
    fn allocate_port(&self, _avoid: &BTreeSet<u16>) -> Option<u16> {
        None
    }
}

fn exclusive_fixture() -> (Arc<SwapScheduler>, Arc<Mutex<Steps>>, Arc<ActiveConsumers>) {
    let dir = root("cutover");
    let catalog = Arc::new(SwapCatalog::load(dir.join("swap_catalog.json")).unwrap());
    for id in ["a", "b"] {
        let gguf = dir.join(format!("{id}.gguf"));
        std::fs::write(&gguf, b"gguf").unwrap();
        catalog.upsert_entry(fake_entry(id, &gguf, 10.0)).unwrap();
    }
    let fake = Arc::new(FakeAdapter::new());
    let steps = Arc::new(Mutex::new(Steps::default()));
    let mut adapters: BTreeMap<String, Arc<dyn EngineAdapter>> = BTreeMap::new();
    adapters.insert(
        "fake".into(),
        Arc::new(ExclusiveFake {
            inner: fake,
            steps: Arc::clone(&steps),
        }) as Arc<dyn EngineAdapter>,
    );
    let consumers = Arc::new(ActiveConsumers::new(Duration::from_secs(0)));
    let scheduler = Arc::new(
        SwapScheduler::new(
            Arc::clone(&catalog),
            Arc::new(BudgetLedger::load(dir.join("budget.json")).unwrap()),
            SwapSchedulerConfig {
                settle: Duration::ZERO,
                cutover_recheck: Duration::from_millis(30),
                wait_timeout: Duration::from_secs(5),
                port_min: 15950,
                port_max: 15999,
                ..SwapSchedulerConfig::default()
            },
            adapters,
        )
        .with_state_root(dir.clone())
        .with_kill_children_on_drop(true)
        .with_budget_cap(100.0)
        .with_active_consumers(Arc::clone(&consumers)),
    );
    (scheduler, steps, consumers)
}

#[tokio::test(flavor = "multi_thread")]
async fn exclusive_cutover_evicts_before_spawn_never_co_resident() {
    let (scheduler, steps, _consumers) = exclusive_fixture();
    scheduler.ensure("a").await.unwrap();
    scheduler.ensure("b").await.unwrap();
    let status = scheduler.status();
    let a = status.models.iter().find(|m| m.id == "a").unwrap();
    let b = status.models.iter().find(|m| m.id == "b").unwrap();
    assert_eq!(a.state, "stopped", "{a:?}");
    assert_eq!(b.state, "ready", "{b:?}");
    let steps = steps.lock().unwrap();
    assert_eq!(steps.spawns, vec!["a".to_string(), "b".to_string()]);
    assert_eq!(steps.stops.len(), 1, "same-device predecessor drained first");
}

#[tokio::test(flavor = "multi_thread")]
async fn sigterm_point_recheck_refuses_a_newly_protected_predecessor() {
    let (scheduler, _steps, consumers) = exclusive_fixture();
    scheduler.ensure("a").await.unwrap();
    consumers.enter("a"); // a stream arrived after b's admission

    let error = scheduler.ensure("b").await.unwrap_err();
    assert!(blocks(&error, "blocked:active-consumer"), "{error:#}");
    let status = scheduler.status();
    let a = status.models.iter().find(|m| m.id == "a").unwrap();
    assert_eq!(a.state, "ready", "the re-check must not SIGTERM a new consumer");
    let b = status.models.iter().find(|m| m.id == "b").unwrap();
    assert_eq!(b.state, "stopped", "{b:?}");

    let outcome = scheduler
        .ensure_with_options(
            "b",
            None,
            EnsureOptions {
                override_active_consumers: true,
                ..EnsureOptions::default()
            },
        )
        .await
        .unwrap();
    assert!(outcome.spawned());
}

// ── L5 findings round: occupant refusal, audit actor, adopt by endpoint ─────

struct StrixiteTestAdapter {
    inner: Arc<FakeAdapter>,
}

impl EngineAdapter for StrixiteTestAdapter {
    fn name(&self) -> &'static str {
        "strixite"
    }
    fn spawn(&self, request: &SpawnRequest) -> Result<EngineHandle> {
        self.inner.spawn(request)
    }
    fn stop(&self, handle: &EngineHandle) -> Result<()> {
        self.inner.stop(handle)
    }
    fn health(&self, endpoint: &str, handle: &EngineHandle) -> Result<HealthReport> {
        self.inner.health(endpoint, handle)
    }
    fn exclusive_key(&self) -> Option<String> {
        Some("strixite:test".into())
    }
    fn allocate_port(&self, _avoid: &BTreeSet<u16>) -> Option<u16> {
        None
    }
}

fn strixite_fixture(
    name: &str,
) -> (
    Arc<SwapScheduler>,
    Arc<FakeAdapter>,
    PathBuf,
    std::process::Child,
) {
    let dir = root(name);
    let mut icon = std::process::Command::new("sleep")
        .arg("30")
        .stdin(std::process::Stdio::null())
        .spawn()
        .expect("icon child");
    let icon_pid = icon.id();
    let catalog = Arc::new(SwapCatalog::load(dir.join("swap_catalog.json")).unwrap());
    let gguf = dir.join("target.gguf");
    std::fs::write(&gguf, b"gguf").unwrap();
    let mut entry = fake_entry("target-model", &gguf, 10.0);
    entry.engine = "strixite".into();
    catalog.upsert_entry(entry).unwrap();
    let fake = Arc::new(FakeAdapter::new());
    let mut adapters: BTreeMap<String, Arc<dyn EngineAdapter>> = BTreeMap::new();
    adapters.insert(
        "strixite".into(),
        Arc::new(StrixiteTestAdapter {
            inner: Arc::clone(&fake),
        }) as Arc<dyn EngineAdapter>,
    );
    let observer = Arc::new(
        ExternalObserver::new(
            Arc::new(FixtureProcs(vec![ProcInfo {
                pid: icon_pid,
                cmdline: "strix_server --config /tmp/x.conf --weights /models/other.gguf --port 5991"
                    .into(),
                comm: "strix_server".into(),
            }])),
            dir.join("adoption"),
            Arc::clone(&catalog),
        )
        .with_probe(Arc::new(|_| Some(vec!["someone-else".to_string()]))),
    );
    let scheduler = Arc::new(
        SwapScheduler::new(
            Arc::clone(&catalog),
            Arc::new(BudgetLedger::load(dir.join("budget.json")).unwrap()),
            SwapSchedulerConfig {
                settle: Duration::ZERO,
                cutover_recheck: Duration::from_millis(20),
                wait_timeout: Duration::from_secs(5),
                port_min: 16000,
                port_max: 16049,
                ..SwapSchedulerConfig::default()
            },
            adapters,
        )
        .with_state_root(dir.clone())
        .with_kill_children_on_drop(true)
        .with_budget_cap(200.0)
        .with_external_observer(observer),
    );
    scheduler.refresh_external();
    let _ = &mut icon;
    (scheduler, fake, dir, icon)
}

#[tokio::test(flavor = "multi_thread")]
async fn unregistered_external_occupant_blocks_and_override_forces() {
    let (scheduler, fake, _dir, mut icon) = strixite_fixture("occupant");
    let error = scheduler.ensure("target-model").await.unwrap_err();
    assert!(blocks(&error, "blocked:external-occupant"), "{error:#}");
    assert!(ledger_has(&scheduler, "blocked", "blocked:external-occupant"));
    assert_eq!(fake.spawn_count(), 0, "never spawn beside an unregistered owner");

    let outcome = scheduler
        .ensure_with_options(
            "target-model",
            None,
            EnsureOptions {
                override_active_consumers: true,
                actor: Some("admin@test".into()),
                ..EnsureOptions::default()
            },
        )
        .await
        .unwrap();
    assert!(outcome.spawned());
    assert!(scheduler
        .ledger()
        .entries()
        .iter()
        .any(|e| e.kind == "override"
            && e.reason.contains("external occupant override")
            && e.actor.as_deref() == Some("admin@test")));
    let _ = icon.kill();
    let _ = icon.wait();
}

#[tokio::test(flavor = "multi_thread")]
async fn adopt_by_endpoint_registers_the_observed_pid_with_actor() {
    let (scheduler, _fake, dir, mut icon) = strixite_fixture("adopt-endpoint");
    let target_id = scheduler
        .catalog()
        .resolve_alias("target-model")
        .expect("catalog entry");
    let mut catalog_data = scheduler.catalog().snapshot();
    // Make the observer resolve the external to our catalog entry.
    let entry = catalog_data.entries.get_mut(&target_id).unwrap();
    entry.files = vec!["/models/other.gguf".into()];
    scheduler.catalog().upsert_entry(entry.clone()).unwrap();
    scheduler.refresh_external();
    let observed = scheduler.observed_external();
    assert_eq!(observed.len(), 1, "{observed:?}");
    assert_eq!(observed[0].catalog_id.as_deref(), Some(target_id.as_str()));

    let report = scheduler
        .adopt_endpoint(&observed[0].endpoint, None, Some("cli:uid=1000"))
        .unwrap();
    assert_eq!(report.pid, icon.id());
    assert!(dir.join("adoption/target-model.json").exists());
    assert!(scheduler
        .ledger()
        .entries()
        .iter()
        .any(|e| e.kind == "adopt" && e.actor.as_deref() == Some("cli:uid=1000")));
    let _ = icon.kill();
    let _ = icon.wait();
}

#[tokio::test(flavor = "multi_thread")]
async fn override_ledger_entry_carries_the_actor() {
    let (fixture, consumers) = acs_fixture("actor", Duration::from_secs(0));
    fixture.scheduler.ensure("a").await.unwrap();
    consumers.enter("a");
    fixture
        .scheduler
        .ensure_with_options(
            "b",
            None,
            EnsureOptions {
                override_active_consumers: true,
                actor: Some("admin@127.0.0.1:1".into()),
                ..EnsureOptions::default()
            },
        )
        .await
        .unwrap();
    let entry = fixture
        .scheduler
        .ledger()
        .entries()
        .into_iter()
        .find(|e| e.kind == "override")
        .expect("override entry");
    assert_eq!(entry.actor.as_deref(), Some("admin@127.0.0.1:1"));
}
