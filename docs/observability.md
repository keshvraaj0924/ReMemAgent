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

## Benchmark integration

`BenchmarkSuiteRunner` accepts an optional `ObservationCollector`. When supplied, it records suite starts, episode starts/completions, successful episodes, attributed memory transfers, and aggregate episode duration. The instrumentation is deliberately additive: benchmark reports and memory behavior are unchanged when no collector is supplied.

This provides a useful research boundary for measuring execution overhead and memory-transfer activity without coupling the benchmark runner to a metrics vendor or external tracing SDK.

## Current limitation

The collector aggregates counters and total durations only. It does not provide histograms, distributed traces, persistence, or an external metrics exporter. Those concerns should remain optional integrations rather than becoming dependencies of the deterministic research core.
