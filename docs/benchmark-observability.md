# Benchmark observability

The benchmark runner exposes an optional `ObservationCollector` for low-dependency execution telemetry.

## Episode lifecycle

For every requested episode, the runner records `benchmark.episodes.started`. When the episode completes normally it records `benchmark.episodes.succeeded` and `benchmark.episodes.completed`. If environment construction, policy construction, environment reset/step execution, success evaluation, memory ingestion, or transfer attribution raises, the exception is re-raised and the runner records `benchmark.episodes.failed` instead.

This distinction is intentional: a failed episode must not appear as an uncompleted-but-successful run, and the original exception remains the authoritative failure signal. Environment cleanup still runs for an environment that was successfully constructed.

The success counter is an execution-outcome counter, not the benchmark's task-success metric. `benchmark.episodes.successful` continues to represent the task-level success evaluator result for completed episodes. These counters therefore answer different questions:

- `benchmark.episodes.succeeded`: did the framework finish the episode lifecycle without an exception?
- `benchmark.episodes.successful`: did the configured benchmark success evaluator mark the completed episode successful?
- `benchmark.episodes.failed`: did the framework encounter an exception while executing the episode?

This separation prevents infrastructure failures from being silently mixed with task-level failures in future experiment analysis.

## Failure semantics

Failure telemetry is emitted before the original exception is propagated. The runner does not convert failures into synthetic `BenchmarkEpisodeReport` values and does not fabricate reward or success values for failed episodes.

The collector remains optional and backend-neutral. External telemetry exporters can consume these counters without becoming part of the benchmark execution contract.

## Runtime provenance

Measured benchmark artifacts also capture `working_tree_state` alongside `code_revision`. The state is `clean` when `git status --porcelain` reports no changes, `dirty` when tracked or untracked changes are present, and `unknown` when the checkout cannot be inspected. CI/container environments may explicitly provide `REMEM_GIT_STATE` when Git metadata is unavailable.

A dirty or unknown state is evidence about the execution environment, not a benchmark result. The framework records it rather than assuming a clean checkout, so downstream analysis can distinguish experiments executed from a committed tree from experiments executed with local modifications.
