# Observation Snapshot Deltas

ReMemAgent's `ObservationCollector.snapshot()` returns cumulative process-lifetime aggregates. Exporting those cumulative values directly to a backend that expects increments can double-count counters and durations when snapshots are pushed repeatedly.

Use `observation_snapshot_delta(previous, current)` when the destination consumes interval increments. The function computes deterministic per-metric differences across counters and duration totals, omits unchanged metrics, and treats metrics first seen in the current snapshot as increments from zero.

The delta boundary is intentionally fail-closed. If a metric decreases or disappears between snapshots, the function raises `ValueError` rather than silently emitting a negative or misleading increment. A regression usually means snapshots came from different collector lifecycles, were supplied in the wrong order, or were externally modified.

```python
from remem.observability import ObservationCollector, observation_snapshot_delta
from remem.observability_exporters import export_observation_snapshot

collector = ObservationCollector()
start = collector.snapshot()

collector.increment("benchmark.episodes")
collector.observe_duration("benchmark.seconds", 0.25)

current = collector.snapshot()
interval = observation_snapshot_delta(start, current)
export_observation_snapshot(interval, exporters)
```

This helper does not persist exporter checkpoints or provide exactly-once delivery. Caller-owned infrastructure remains responsible for retaining the previous successfully exported cumulative snapshot, retry policy, and backend-specific delivery guarantees.
