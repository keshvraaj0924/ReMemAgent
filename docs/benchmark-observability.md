# Benchmark observability

The benchmark runner exposes an optional `ObservationCollector` for low-dependency execution telemetry. The external benchmark CLI can persist this telemetry as a deterministic JSON snapshot with `--observability-output`.

## Episode lifecycle

For every requested episode, the runner records `benchmark.episodes.started`. When the episode completes normally it records `benchmark.episodes.succeeded` and `benchmark.episodes.completed`. If environment construction, policy construction, environment reset/step execution, success evaluation, memory ingestion, or transfer attribution raises, the exception is re-raised and the runner records `benchmark.episodes.failed` instead.

This distinction is intentional: a failed episode must not appear as an uncompleted-but-successful run, and the original exception remains the authoritative failure signal. Environment cleanup still runs for an environment that was successfully constructed.

The success counter is an execution-outcome counter, not the benchmark's task-success metric. `benchmark.episodes.successful` continues to represent the task-level success evaluator result for completed episodes. These counters therefore answer different questions:

- `benchmark.episodes.succeeded`: did the framework finish the episode lifecycle without an exception?
- `benchmark.episodes.successful`: did the configured benchmark success evaluator mark the completed episode successful?
- `benchmark.episodes.failed`: did the framework encounter an exception while executing the episode?

This separation prevents infrastructure failures from being silently mixed with task-level failures in future experiment analysis.

## Persisting CLI telemetry

Measured CLI runs may pass `--observability-output artifacts/benchmark.observability.json`. The destination is validated before any measured environment is constructed, and existing files require `--overwrite`, matching the benchmark report artifact policy.

Single-seed runs use the runner's per-episode telemetry and add `benchmark.runs.completed` after successful measured execution. Repeated runs execute through the existing independent-seed runner path and record aggregate run, completed-episode, task-success, and attributed-transfer counters after all reports have been produced. The observability snapshot is supplementary telemetry; it is never included in benchmark statistics or used to manufacture missing experiment results.

The persisted format is the same deterministic `ObservationSnapshot` JSON used by the core observability module. It contains sorted `counters` and `durations_seconds` mappings and can therefore be archived alongside the measured benchmark report and integrity manifest.

## Failure semantics

Failure telemetry is emitted before the original exception is propagated. The runner does not convert failures into synthetic `BenchmarkEpisodeReport` values and does not fabricate reward or success values for failed episodes.

The collector remains optional and backend-neutral. External telemetry exporters can consume these counters without becoming part of the benchmark execution contract.

## Runtime provenance

Measured benchmark artifacts also capture `working_tree_state` alongside `code_revision`. The state is `clean` when `git status --porcelain` reports no changes, `dirty` when tracked or untracked changes are present, and `unknown` when the checkout cannot be inspected. CI/container environments may explicitly provide `REMEM_GIT_STATE` when Git metadata is unavailable.

A dirty or unknown state is evidence about the execution environment, not a benchmark result. The framework records it rather than assuming a clean checkout, so downstream analysis can distinguish experiments executed from a committed tree from experiments executed with local modifications.
