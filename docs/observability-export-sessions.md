# Observation Export Sessions

`ObservationCollector.snapshot()` is cumulative. `observation_snapshot_delta()` converts two cumulative snapshots into interval increments, but callers still need to advance the previous-snapshot checkpoint only after a successful export.

`ObservationExportSession` centralizes that rule. It stores the last successfully exported cumulative snapshot, computes the next delta, exports that delta, and advances the checkpoint only when the backend returns successfully. If the exporter raises, the previous checkpoint is preserved so a retry does not lose observations.

```python
from remem.observability import ObservationCollector
from remem.observability_exporters import ObservationExportSession

collector = ObservationCollector()
session = ObservationExportSession(exporter)

collector.increment("benchmark.episodes")
session.export(collector.snapshot())
```

Empty deltas are accepted without invoking the backend. Regressing or disappearing cumulative metrics still fail closed through `observation_snapshot_delta()`.

The session is process-local and intentionally vendor-neutral. It does not persist checkpoints across process restarts and does not provide exactly-once delivery. External infrastructure remains responsible for durable checkpoint storage, backend-specific retries, authentication, batching, and transport guarantees.
