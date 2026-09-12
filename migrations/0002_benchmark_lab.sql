PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;

BEGIN IMMEDIATE;

CREATE TABLE benchmark_jobs (
  id TEXT PRIMARY KEY,
  identity_sha256 TEXT NOT NULL CHECK(length(identity_sha256)=64),
  suite TEXT NOT NULL CHECK(suite IN ('riddllr','plumebench')),
  case_id TEXT NOT NULL,
  model TEXT NOT NULL,
  repetition INTEGER NOT NULL CHECK(repetition >= 0),
  status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed','timeout','cancelled','interrupted')),
  progress REAL NOT NULL DEFAULT 0 CHECK(progress >= 0 AND progress <= 1),
  message TEXT NOT NULL DEFAULT '',
  queued_at TEXT NOT NULL,
  started_at TEXT,
  ended_at TEXT,
  run_id TEXT,
  work_dir TEXT,
  request_json TEXT NOT NULL,
  result_json TEXT,
  error TEXT,
  updated_at TEXT NOT NULL,
  CHECK(ended_at IS NULL OR started_at IS NOT NULL)
) STRICT;

CREATE UNIQUE INDEX idx_benchmark_jobs_active_identity
ON benchmark_jobs(identity_sha256)
WHERE status IN ('queued','running');

CREATE INDEX idx_benchmark_jobs_queued ON benchmark_jobs(queued_at DESC);
CREATE INDEX idx_benchmark_jobs_status ON benchmark_jobs(status, updated_at DESC);

INSERT INTO schema_migrations(version, name) VALUES (2, 'native_benchmark_lab');

COMMIT;
