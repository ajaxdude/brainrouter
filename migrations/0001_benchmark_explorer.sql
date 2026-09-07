PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;

BEGIN IMMEDIATE;

CREATE TABLE schema_migrations (
  version INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
) STRICT;

CREATE TABLE models (
  id TEXT PRIMARY KEY,
  family TEXT NOT NULL,
  architecture TEXT NOT NULL,
  checkpoint TEXT NOT NULL,
  revision TEXT NOT NULL,
  tokenizer_id TEXT NOT NULL,
  tokenizer_revision TEXT,
  parameter_count_total INTEGER CHECK(parameter_count_total IS NULL OR parameter_count_total > 0),
  parameter_count_active INTEGER CHECK(parameter_count_active IS NULL OR parameter_count_active > 0),
  model_kind TEXT NOT NULL CHECK(model_kind IN ('dense','moe','hybrid','unknown')),
  native_context_tokens INTEGER CHECK(native_context_tokens IS NULL OR native_context_tokens > 0),
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  UNIQUE(checkpoint, revision)
) STRICT;

CREATE TABLE artifacts (
  id TEXT PRIMARY KEY,
  model_id TEXT NOT NULL REFERENCES models(id) ON DELETE RESTRICT,
  format TEXT NOT NULL,
  quant_family TEXT NOT NULL,
  quant_name TEXT NOT NULL,
  average_bits_per_weight REAL CHECK(average_bits_per_weight IS NULL OR average_bits_per_weight > 0),
  disk_bytes INTEGER NOT NULL CHECK(disk_bytes >= 0),
  sha256 TEXT NOT NULL CHECK(length(sha256)=64),
  source_uri TEXT,
  imatrix_used INTEGER NOT NULL DEFAULT 0 CHECK(imatrix_used IN (0,1)),
  conversion_tool TEXT,
  conversion_commit TEXT,
  conversion_command TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  UNIQUE(sha256)
) STRICT;

CREATE TABLE runtimes (
  id TEXT PRIMARY KEY,
  repository TEXT NOT NULL,
  fork_name TEXT NOT NULL,
  commit_sha TEXT NOT NULL,
  dirty_tree INTEGER NOT NULL DEFAULT 0 CHECK(dirty_tree IN (0,1)),
  compiler TEXT NOT NULL,
  compiler_version TEXT NOT NULL,
  backend TEXT NOT NULL CHECK(backend IN ('cpu','cuda','rocm','vulkan','metal','sycl','rpc','other')),
  build_flags_json TEXT NOT NULL DEFAULT '[]',
  capabilities_json TEXT NOT NULL DEFAULT '{}',
  executable_sha256 TEXT CHECK(executable_sha256 IS NULL OR length(executable_sha256)=64),
  container_digest TEXT,
  built_at TEXT NOT NULL,
  UNIQUE(repository, commit_sha, backend, build_flags_json, dirty_tree)
) STRICT;

CREATE TABLE hardware_profiles (
  id TEXT PRIMARY KEY,
  hostname_hash TEXT,
  cpu_model TEXT NOT NULL,
  physical_cores INTEGER CHECK(physical_cores IS NULL OR physical_cores > 0),
  logical_cores INTEGER CHECK(logical_cores IS NULL OR logical_cores > 0),
  system_ram_bytes INTEGER NOT NULL CHECK(system_ram_bytes > 0),
  gpu_json TEXT NOT NULL DEFAULT '[]',
  unified_memory INTEGER NOT NULL DEFAULT 0 CHECK(unified_memory IN (0,1)),
  os_name TEXT NOT NULL,
  os_version TEXT NOT NULL,
  kernel TEXT NOT NULL,
  driver_versions_json TEXT NOT NULL DEFAULT '{}',
  power_profile TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  captured_at TEXT NOT NULL
) STRICT;

CREATE TABLE workloads (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  workload_type TEXT NOT NULL CHECK(workload_type IN ('performance','code_quality','perplexity','retrieval','mixed')),
  manifest_sha256 TEXT NOT NULL CHECK(length(manifest_sha256)=64),
  input_tokens INTEGER CHECK(input_tokens IS NULL OR input_tokens >= 0),
  output_tokens INTEGER CHECK(output_tokens IS NULL OR output_tokens >= 0),
  corpus_bytes INTEGER CHECK(corpus_bytes IS NULL OR corpus_bytes >= 0),
  license TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  UNIQUE(name, version, manifest_sha256)
) STRICT;

CREATE TABLE experiment_specs (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  spec_sha256 TEXT NOT NULL UNIQUE CHECK(length(spec_sha256)=64),
  canonical_spec_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
) STRICT;

CREATE TABLE experiments (
  id TEXT PRIMARY KEY,
  experiment_hash TEXT NOT NULL UNIQUE CHECK(length(experiment_hash)=64),
  spec_id TEXT NOT NULL REFERENCES experiment_specs(id) ON DELETE RESTRICT,
  artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE RESTRICT,
  runtime_id TEXT NOT NULL REFERENCES runtimes(id) ON DELETE RESTRICT,
  hardware_id TEXT NOT NULL REFERENCES hardware_profiles(id) ON DELETE RESTRICT,
  workload_id TEXT NOT NULL REFERENCES workloads(id) ON DELETE RESTRICT,
  context_tokens INTEGER NOT NULL CHECK(context_tokens > 0),
  prompt_tokens INTEGER NOT NULL CHECK(prompt_tokens >= 0),
  generation_tokens INTEGER NOT NULL CHECK(generation_tokens >= 0),
  batch_size INTEGER NOT NULL CHECK(batch_size > 0),
  micro_batch_size INTEGER NOT NULL CHECK(micro_batch_size > 0),
  threads INTEGER CHECK(threads IS NULL OR threads > 0),
  gpu_layers INTEGER,
  flash_attention INTEGER NOT NULL DEFAULT 0 CHECK(flash_attention IN (0,1)),
  kv_cache_type_k TEXT,
  kv_cache_type_v TEXT,
  optimization_json TEXT NOT NULL DEFAULT '{}',
  sampling_json TEXT NOT NULL DEFAULT '{}',
  command_template TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
) STRICT;

CREATE TABLE runs (
  id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE RESTRICT,
  repetition INTEGER NOT NULL CHECK(repetition >= 0),
  status TEXT NOT NULL CHECK(status IN ('planned','running','succeeded','failed','oom','timeout','cancelled','skipped')),
  started_at TEXT,
  ended_at TEXT,
  exit_code INTEGER,
  random_seed INTEGER,
  warmup_count INTEGER NOT NULL DEFAULT 0 CHECK(warmup_count >= 0),
  exact_command TEXT NOT NULL,
  cwd TEXT,
  environment_json TEXT NOT NULL DEFAULT '{}',
  stdout_path TEXT,
  stderr_path TEXT,
  failure_reason TEXT,
  raw_result_json TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  UNIQUE(experiment_id, repetition),
  CHECK(ended_at IS NULL OR started_at IS NOT NULL)
) STRICT;

CREATE TABLE performance_metrics (
  run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
  model_load_ms REAL CHECK(model_load_ms IS NULL OR model_load_ms >= 0),
  prompt_processing_ms REAL CHECK(prompt_processing_ms IS NULL OR prompt_processing_ms >= 0),
  prompt_tps REAL CHECK(prompt_tps IS NULL OR prompt_tps >= 0),
  ttft_ms REAL CHECK(ttft_ms IS NULL OR ttft_ms >= 0),
  generation_ms REAL CHECK(generation_ms IS NULL OR generation_ms >= 0),
  generation_tps REAL CHECK(generation_tps IS NULL OR generation_tps >= 0),
  inter_token_p50_ms REAL CHECK(inter_token_p50_ms IS NULL OR inter_token_p50_ms >= 0),
  inter_token_p95_ms REAL CHECK(inter_token_p95_ms IS NULL OR inter_token_p95_ms >= 0),
  inter_token_p99_ms REAL CHECK(inter_token_p99_ms IS NULL OR inter_token_p99_ms >= 0),
  peak_rss_bytes INTEGER CHECK(peak_rss_bytes IS NULL OR peak_rss_bytes >= 0),
  peak_vram_bytes INTEGER CHECK(peak_vram_bytes IS NULL OR peak_vram_bytes >= 0),
  kv_cache_bytes INTEGER CHECK(kv_cache_bytes IS NULL OR kv_cache_bytes >= 0),
  energy_joules REAL CHECK(energy_joules IS NULL OR energy_joules >= 0),
  avg_power_watts REAL CHECK(avg_power_watts IS NULL OR avg_power_watts >= 0)
) STRICT;

CREATE TABLE speculative_metrics (
  run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
  speculator_type TEXT NOT NULL CHECK(speculator_type IN ('none','mtp','ngram','draft_model','eagle','dflash','combined','other')),
  proposals INTEGER CHECK(proposals IS NULL OR proposals >= 0),
  proposed_tokens INTEGER CHECK(proposed_tokens IS NULL OR proposed_tokens >= 0),
  accepted_tokens INTEGER CHECK(accepted_tokens IS NULL OR accepted_tokens >= 0),
  target_evaluations INTEGER CHECK(target_evaluations IS NULL OR target_evaluations >= 0),
  acceptance_rate REAL CHECK(acceptance_rate IS NULL OR (acceptance_rate >= 0 AND acceptance_rate <= 1)),
  speculator_memory_bytes INTEGER CHECK(speculator_memory_bytes IS NULL OR speculator_memory_bytes >= 0),
  overhead_ms REAL CHECK(overhead_ms IS NULL OR overhead_ms >= 0),
  CHECK(accepted_tokens IS NULL OR proposed_tokens IS NULL OR accepted_tokens <= proposed_tokens)
) STRICT;

CREATE TABLE quality_results (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  task_id TEXT NOT NULL,
  metric_name TEXT NOT NULL,
  metric_value REAL,
  passed INTEGER CHECK(passed IS NULL OR passed IN (0,1)),
  compile_succeeded INTEGER CHECK(compile_succeeded IS NULL OR compile_succeeded IN (0,1)),
  tests_passed INTEGER CHECK(tests_passed IS NULL OR tests_passed >= 0),
  tests_total INTEGER CHECK(tests_total IS NULL OR tests_total >= 0),
  generated_tokens INTEGER CHECK(generated_tokens IS NULL OR generated_tokens >= 0),
  duration_ms REAL CHECK(duration_ms IS NULL OR duration_ms >= 0),
  output_path TEXT,
  log_path TEXT,
  details_json TEXT NOT NULL DEFAULT '{}',
  UNIQUE(run_id, task_id, metric_name),
  CHECK(tests_passed IS NULL OR tests_total IS NULL OR tests_passed <= tests_total)
) STRICT;

CREATE TABLE telemetry_samples (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  sampled_at TEXT NOT NULL,
  cpu_percent REAL CHECK(cpu_percent IS NULL OR (cpu_percent >= 0 AND cpu_percent <= 100)),
  rss_bytes INTEGER CHECK(rss_bytes IS NULL OR rss_bytes >= 0),
  gpu_index INTEGER,
  gpu_util_percent REAL CHECK(gpu_util_percent IS NULL OR (gpu_util_percent >= 0 AND gpu_util_percent <= 100)),
  vram_used_bytes INTEGER CHECK(vram_used_bytes IS NULL OR vram_used_bytes >= 0),
  temperature_c REAL,
  power_watts REAL CHECK(power_watts IS NULL OR power_watts >= 0),
  metadata_json TEXT NOT NULL DEFAULT '{}'
) STRICT;

CREATE TABLE exclusions (
  id INTEGER PRIMARY KEY,
  spec_id TEXT NOT NULL REFERENCES experiment_specs(id) ON DELETE CASCADE,
  candidate_json TEXT NOT NULL,
  reason_code TEXT NOT NULL,
  reason TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
) STRICT;

CREATE TABLE entity_fingerprints (
  entity_type TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  sha256 TEXT NOT NULL CHECK(length(sha256)=64),
  PRIMARY KEY(entity_type, entity_id)
) STRICT;

CREATE INDEX idx_artifacts_model ON artifacts(model_id);
CREATE INDEX idx_experiments_dims ON experiments(artifact_id, runtime_id, hardware_id, workload_id, context_tokens);
CREATE INDEX idx_runs_experiment_status ON runs(experiment_id, status);
CREATE INDEX idx_runs_started ON runs(started_at);
CREATE INDEX idx_quality_run_metric ON quality_results(run_id, metric_name);
CREATE INDEX idx_telemetry_run_time ON telemetry_samples(run_id, sampled_at);
CREATE INDEX idx_exclusions_spec ON exclusions(spec_id);

CREATE VIEW run_summary AS
SELECT r.id AS run_id, r.status, r.repetition, e.experiment_hash,
       m.family, m.architecture, a.quant_name, rt.fork_name, rt.backend,
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
LEFT JOIN performance_metrics pm ON pm.run_id=r.id
LEFT JOIN speculative_metrics sm ON sm.run_id=r.id;

INSERT INTO schema_migrations(version, name) VALUES (1, 'benchmark_explorer_initial');

COMMIT;
