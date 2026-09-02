# Research observability

ReMemAgent exposes a small standard-library-only observability layer in `remem.observability`.

`ObservationCollector` records scalar counters and aggregate durations. `ObservationEvent` is immutable, and `ObservationSnapshot` copies the current aggregates so callers can serialize or report them without retaining mutable collector state.

```python
from remem.observability import ObservationCollector

collector = ObservationCollector()
collector.increment("retrieval.calls")

with collector.timed("routing.seconds"):
    route_memory()

snapshot = collector.snapshot()
```

The collector is thread-safe and uses `time.monotonic()` for durations. It rejects empty metric names, non-finite counter values, and invalid durations.

## Durable snapshots

`write_observation_snapshot(path, snapshot)` persists one snapshot as deterministic JSON. The writer creates missing parent directories and replaces the destination atomically after flushing and syncing a temporary file in the same directory. This makes local telemetry suitable for inclusion in experiment artifacts without introducing a telemetry backend dependency.

The serialized representation sorts metric keys and ends with a newline, so equivalent snapshots produce byte-identical files. Persistence is intentionally snapshot-oriented: it does not turn the collector into an event log or distributed tracing system.

## Benchmark integration

`BenchmarkSuiteRunner` accepts an optional `ObservationCollector`. When supplied, it records suite starts, episode starts/completions, successful episodes, attributed memory transfers, and aggregate episode duration. The instrumentation is deliberately additive: benchmark reports and memory behavior are unchanged when no collector is supplied.

This provides a useful research boundary for measuring execution overhead and memory-transfer activity without coupling the benchmark runner to a metrics vendor or external tracing SDK.

## Current limitation

The collector aggregates counters and total durations only. It does not provide histograms, distributed traces, or external exporters. Those concerns should remain optional integrations rather than becoming dependencies of the deterministic research core. Snapshot persistence is available for local artifacts, but exporting to Prometheus, OpenTelemetry, or another operational backend remains deployment-specific.
