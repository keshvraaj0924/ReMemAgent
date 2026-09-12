# Research observability

ReMemAgent exposes a small standard-library-only observability layer in `remem.observability`.

`ObservationCollector` records scalar counters and aggregate durations. `ObservationEvent` is immutable, and `ObservationSnapshot` validates, normalizes, and freezes its aggregate mappings at construction time so callers can serialize or report a stable point-in-time view without retaining mutable collector state.

```python
from remem.observability import ObservationCollector

collector = ObservationCollector()
collector.increment("retrieval.calls")

with collector.timed("routing.seconds"):
    route_memory()

snapshot = collector.snapshot()
```

The collector is thread-safe and uses `time.monotonic()` for durations. It rejects empty metric names, non-finite counter values, and invalid durations. Snapshot construction applies the same finite, non-negative aggregate validation and trims metric names consistently.

## Durable snapshots

`write_observation_snapshot(path, snapshot)` persists one snapshot as deterministic JSON. The writer creates missing parent directories and replaces the destination atomically after flushing and syncing a temporary file in the same directory. This makes local telemetry suitable for inclusion in experiment artifacts without introducing a telemetry backend dependency.

The serialized representation sorts metric keys and ends with a newline, so equivalent snapshots produce byte-identical files. Persistence is intentionally snapshot-oriented: it does not turn the collector into an event log or distributed tracing system.

Snapshot mappings are detached from caller-owned dictionaries and exposed as read-only mappings. Whitespace-normalized metric-name collisions are rejected instead of silently overwriting one aggregate with another. This keeps a persisted snapshot's metric identity unambiguous.

## Multi-worker aggregation

`merge_observation_snapshots(snapshots)` combines independent snapshots after parallel or distributed execution. Counter and duration values are added by metric name, source snapshots are never mutated, and invalid aggregate values are rejected before they enter the merged artifact.

The merge operation deliberately does not invent event ordering, timestamps, percentiles, or cross-process trace relationships. It is therefore appropriate for additive research metrics such as call counts and total elapsed time, while richer telemetry remains an optional deployment concern.

## Deterministic latency distributions

`remem.observability_distributions` adds fixed-bucket histograms for experiments that need more than a single duration total without introducing a metrics SDK or changing the existing `ObservationSnapshot` schema.

```python
from remem.observability_distributions import ObservationHistogram

latency = ObservationHistogram((0.01, 0.05, 0.1, 0.5, 1.0))
latency.observe(0.034)
latency.observe(0.72)

snapshot = latency.snapshot()
```

`ObservationHistogram` is thread-safe and accepts finite, non-negative observations. Bucket upper bounds are finite, non-negative, and strictly increasing. Snapshot bucket counts are non-cumulative: one count is stored for every configured upper bound plus a final overflow bucket. The snapshot also preserves the exact observation count and sum, so the arithmetic mean is derived without sampling or approximation.

`merge_histogram_snapshots(...)` merges independent workers only when their bucket contracts match exactly. `histogram_snapshot_delta(previous, current)` converts two cumulative snapshots into an interval distribution and fails closed when any bucket count or total regresses. ReMemAgent deliberately does not silently rebucket mismatched histograms because doing so would make research comparisons ambiguous.

These histograms provide deterministic distribution evidence, but they do not claim exact percentiles: a percentile estimated from fixed buckets is bounded by the bucket resolution. Experiments that publish percentiles should record and report the bucket contract alongside the result.

## External exporter boundary

`remem.observability_exporters` provides a dependency-free protocol for handing validated snapshots to caller-owned operational telemetry systems. ReMemAgent does not import Prometheus, OpenTelemetry, Datadog, or other vendor SDKs; an application can implement `ObservationExporter` directly or adapt an existing payload callback through `CallbackObservationExporter`.

```python
from remem.observability_exporters import CallbackObservationExporter

exporter = CallbackObservationExporter(send_to_metrics_backend)
exporter.export(collector.snapshot())
```

The callback receives a detached, versioned JSON-compatible payload produced by `ObservationSnapshot.to_dict()`. Mutating that callback payload cannot mutate the original research snapshot.

`CompositeObservationExporter` supports deterministic ordered fan-out to multiple backends. Export is deliberately fail-fast: if one backend raises, the exception is preserved and later exporters are not invoked. This prevents a partially failed telemetry path from being silently reported as fully successful. Applications that need retry, buffering, asynchronous delivery, or best-effort fan-out should implement those policies outside the deterministic research core.

The convenience function `export_observation_snapshot(snapshot, exporters)` applies the same ordered contract without requiring applications to retain a composite object.

## Benchmark integration

`BenchmarkSuiteRunner` accepts an optional `ObservationCollector`. When supplied, it records suite starts, episode starts/completions, successful episodes, attributed memory transfers, and aggregate episode duration. The instrumentation is deliberately additive: benchmark reports and memory behavior are unchanged when no collector is supplied.

This provides a useful research boundary for measuring execution overhead and memory-transfer activity without coupling the benchmark runner to a metrics vendor or external tracing SDK.

## Runtime provenance

Measured benchmark artifacts capture the checkout state alongside the Git revision. `REMEM_GIT_STATE` can be supplied by controlled CI/container execution and must be exactly `clean`, `dirty`, or `unknown`; invalid values fail closed instead of being persisted as authoritative provenance. When the variable is absent, the runtime probes `git status --porcelain` and records `unknown` if the checkout cannot be inspected.

This distinction matters because a valid commit SHA alone does not prove that the executed source tree matched that revision. Provenance remains descriptive metadata: it does not provide cryptographic authenticity, prove dependency availability outside the recorded environment, or establish scientific reproducibility by itself.

## Current limitation

The core collector still intentionally exposes counters and total durations only; histogram distributions live in a separate module so existing snapshot artifacts keep their schema and byte contract. ReMemAgent does not provide distributed traces. External snapshot export has a vendor-neutral boundary, but backend-specific metric translation, authentication, retries, batching, buffering, sampling, trace correlation, and percentile estimation policy remain deployment-specific and are intentionally not dependencies of the deterministic research core.
