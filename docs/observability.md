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

This layer intentionally does not depend on OpenTelemetry, Prometheus, logging vendors, or a benchmark platform. A future application can export snapshots to its preferred backend without changing the research core.

## Current limitation

The collector aggregates counters and total durations only. It does not provide histograms, distributed traces, persistence, or an external metrics exporter. Those concerns should remain optional integrations rather than becoming dependencies of the deterministic research core.
