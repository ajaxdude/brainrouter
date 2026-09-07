use super::*;

const EXAMPLE_BUNDLE: &str = include_str!("../../examples/benchmarks/synthetic-bundle.json");
const EXAMPLE_MATRIX: &str = include_str!("../../examples/benchmarks/matrix.yaml");
const EXAMPLE_LLAMA: &str = include_str!("../../examples/benchmarks/llama-bench.json");

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ImportTemplate {
    model: ModelDefinition,
    artifact: ArtifactDefinition,
    runtime: RuntimeDefinition,
    hardware: HardwareProfile,
    workload: WorkloadDefinition,
    experiment: ExperimentConfig,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PrepareImport {
    template: ImportTemplate,
    #[serde(default)]
    experiment: Option<ExperimentConfig>,
    repetition: u64,
    status: RunStatus,
    exact_command: String,
    #[serde(default)]
    started_at: Option<DateTime<Utc>>,
    #[serde(default)]
    ended_at: Option<DateTime<Utc>>,
    #[serde(default)]
    llama_bench: Option<Value>,
}

impl BenchmarkStore {
    pub(super) fn prepare_import(&self, input: PrepareImport) -> BenchmarkResult<IngestBundle> {
        require_sqlite_integer("repetition", input.repetition)?;
        let ImportTemplate {
            model,
            mut artifact,
            runtime,
            hardware,
            workload,
            mut experiment,
        } = input.template;
        artifact.model_id = model.id.clone();
        if let Some(selected) = input.experiment {
            if selected.artifact_id != artifact.id
                || selected.runtime_id != runtime.id
                || selected.hardware_id != hardware.id
                || selected.workload_id != workload.id
            {
                return Err(BenchmarkError::Validation(
                    "selected plan experiment references different registry definitions; load the matching template first".into()
                ));
            }
            experiment = selected;
        } else {
            experiment.artifact_id = artifact.id.clone();
            experiment.runtime_id = runtime.id.clone();
            experiment.hardware_id = hardware.id.clone();
            experiment.workload_id = workload.id.clone();
        }
        // Keep previously imported custom IDs, including failed runs that can be
        // completed. New identities are stable across previews and repetitions.
        experiment.id = "pending".into();
        let hash = experiment.experiment_hash()?;
        let connection = self.connect()?;
        experiment.id = connection
            .query_row(
                "SELECT id FROM experiments WHERE experiment_hash=?1",
                params![hash],
                |row| row.get(0),
            )
            .optional()?
            .unwrap_or_else(|| format!("experiment-{hash}"));
        let run_id = connection
            .query_row(
                "SELECT id FROM runs WHERE experiment_id=?1 AND repetition=?2",
                params![experiment.id, input.repetition],
                |row| row.get(0),
            )
            .optional()?
            .unwrap_or_else(|| format!("run-{hash}-{}", input.repetition));
        let run = RunRecord {
            id: run_id,
            experiment_id: experiment.id.clone(),
            repetition: input.repetition,
            status: input.status,
            started_at: input.started_at,
            ended_at: input.ended_at,
            exit_code: None,
            random_seed: None,
            warmup_count: 0,
            exact_command: input.exact_command,
            cwd: None,
            environment: BTreeMap::new(),
            stdout_path: None,
            stderr_path: None,
            failure_reason: None,
            raw_result: None,
        };
        let mut bundle = IngestBundle {
            model,
            artifact,
            runtime,
            hardware,
            workload,
            experiment,
            run,
            performance_metrics: None,
            speculative_metrics: None,
            quality_results: Vec::new(),
            telemetry_samples: Vec::new(),
        };
        if let Some(output) = input.llama_bench {
            apply_llama_bench_metrics(&mut bundle, &output)?;
        }
        bundle.validate()?;
        Ok(bundle)
    }
}

pub(super) fn example(path: &str) -> BenchmarkResult<(String, &'static str, &'static str)> {
    let json_type = "application/json; charset=utf-8";
    match path.trim_start_matches("/api/benchmarks/examples/") {
        "bundle.json" => Ok((EXAMPLE_BUNDLE.into(), json_type, "synthetic-bundle.json")),
        "llama-bench.json" => Ok((
            EXAMPLE_LLAMA.into(),
            json_type,
            "synthetic-llama-bench.json",
        )),
        "matrix.yaml" => Ok((
            EXAMPLE_MATRIX.into(),
            "application/yaml; charset=utf-8",
            "synthetic-matrix.yaml",
        )),
        "matrix.json" => {
            let matrix: ExperimentMatrix =
                serde_yaml::from_str(EXAMPLE_MATRIX).map_err(|error| {
                    BenchmarkError::Validation(format!("invalid built-in matrix: {error}"))
                })?;
            Ok((to_json(&matrix)?, json_type, "synthetic-matrix.json"))
        }
        "template.json" => {
            let bundle: IngestBundle = serde_json::from_str(EXAMPLE_BUNDLE).map_err(|error| {
                BenchmarkError::Validation(format!("invalid built-in bundle: {error}"))
            })?;
            let template = ImportTemplate {
                model: bundle.model,
                artifact: bundle.artifact,
                runtime: bundle.runtime,
                hardware: bundle.hardware,
                workload: bundle.workload,
                experiment: bundle.experiment,
            };
            Ok((to_json(&template)?, json_type, "synthetic-template.json"))
        }
        _ => Err(BenchmarkError::NotFound("unknown benchmark example".into())),
    }
}
