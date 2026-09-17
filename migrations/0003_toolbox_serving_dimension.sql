PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;

BEGIN IMMEDIATE;

-- Anchors `serving_runtimes` rows (design doc §2/§7) to the exact vendored
-- ai-toolbox-cockpit catalog snapshot in effect when they were recorded.
-- Catalog ids/entries can be renamed or removed upstream over time, so a
-- historical benchmark row referencing a since-renamed toolbox id would be
-- ambiguous without an anchor to the exact snapshot content that was current
-- then. `id` is a content hash (`toolbox_catalog::vendored_catalog_revision()`),
-- not a database-generated surrogate key, so it's reproducible from the
-- vendored files themselves.
CREATE TABLE toolbox_catalog_snapshot (
  id TEXT PRIMARY KEY,
  models_json TEXT NOT NULL,
  toolboxes_json TEXT NOT NULL,
  source_commit TEXT,
  synced_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
) STRICT;

-- "What toolbox/container actually served this benchmark run" — a
-- container-image identity (which catalog toolbox, which compute API, which
-- exact image), distinct from `runtimes`' source-build identity
-- (repository/commit_sha/compiler), which doesn't naturally fit non-llama.cpp
-- -fork backends (design doc §7 decision: new table, not a widened
-- `runtimes`). `runtimes` is left completely unchanged by this migration.
CREATE TABLE serving_runtimes (
  id TEXT PRIMARY KEY,
  toolbox_backend TEXT NOT NULL CHECK(toolbox_backend IN ('llama_cpp','ds4','halogen','vllm','r9v')),
  toolbox_id TEXT NOT NULL,
  compute_api TEXT NOT NULL CHECK(compute_api IN ('vulkan','rocm','cuda','cpu','metal','sycl','rpc','other')),
  container_image TEXT NOT NULL,
  catalog_snapshot_id TEXT NOT NULL REFERENCES toolbox_catalog_snapshot(id) ON DELETE RESTRICT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  UNIQUE(toolbox_backend, toolbox_id, container_image, catalog_snapshot_id)
) STRICT;

-- Nullable and additive: existing `experiments` rows get NULL here
-- unconditionally (SQLite raises no FK violation for a NULL referencing
-- column), so no backfill is needed and `runtime_id` keeps its NOT NULL
-- contract untouched. Populated only once a benchmark run is actually served
-- by a catalog toolbox (llama_cpp today; ds4/halogen/vllm/r9v starting with
-- their own Rollout PRs 7-11, per the "additive per-backend population"
-- pattern already used for the Sankey diagram in PR4).
ALTER TABLE experiments ADD COLUMN serving_runtime_id TEXT REFERENCES serving_runtimes(id) ON DELETE RESTRICT;

CREATE INDEX idx_serving_runtimes_backend ON serving_runtimes(toolbox_backend);
CREATE INDEX idx_experiments_serving_runtime ON experiments(serving_runtime_id);

DROP VIEW run_summary;
CREATE VIEW run_summary AS
SELECT r.id AS run_id, r.status, r.repetition, e.experiment_hash,
       m.family, m.architecture, a.quant_name, rt.fork_name, rt.backend,
       sr.toolbox_backend, sr.toolbox_id, sr.compute_api,
       e.context_tokens, w.name AS workload,
       pm.prompt_tps, pm.generation_tps, pm.ttft_ms,
       pm.peak_rss_bytes, pm.peak_vram_bytes,
       sm.speculator_type, sm.acceptance_rate
FROM runs r
JOIN experiments e ON e.id=r.experiment_id
JOIN artifacts a ON a.id=e.artifact_id
JOIN models m ON m.id=a.model_id
JOIN runtimes rt ON rt.id=e.runtime_id
JOIN workloads w ON w.id=e.workload_id
LEFT JOIN serving_runtimes sr ON sr.id=e.serving_runtime_id
LEFT JOIN performance_metrics pm ON pm.run_id=r.id
LEFT JOIN speculative_metrics sm ON sm.run_id=r.id;

INSERT INTO schema_migrations(version, name) VALUES (3, 'toolbox_serving_dimension');

COMMIT;
