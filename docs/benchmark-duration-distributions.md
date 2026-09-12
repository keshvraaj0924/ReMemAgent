# Benchmark duration distributions

ReMemAgent can capture fixed-bucket benchmark timing distributions without changing the stable `ObservationSnapshot` schema.

Use `DistributionObservationCollector` with an explicit histogram registration for `benchmark.episode.duration_seconds`:

```python
from remem.observability_distribution_collector import DistributionObservationCollector
from remem.observability_distributions import ObservationHistogram

collector = DistributionObservationCollector(
    {
        "benchmark.episode.duration_seconds": ObservationHistogram(
            (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
    }
)
```

Pass that collector to `BenchmarkSuiteRunner`. The runner's existing timing scope records one duration per attempted episode execution. The distribution-aware collector forwards the exact measured value to both the aggregate duration total and the configured fixed-bucket histogram.

`collector.snapshot()` therefore remains backward compatible and contains the existing counters and aggregate duration totals. Distribution data is retrieved separately through `collector.duration_distribution_snapshots()`.

This separation is intentional. Existing benchmark artifacts and snapshot schema version 1 are not silently changed, while experiments that need latency distributions can opt in explicitly.

## Semantics

Histogram bucket counts are non-cumulative and include one overflow bucket after the configured upper bounds. The histogram total and the base collector aggregate are derived from the same duration values. Fixed buckets support deterministic merging across workers through `merge_histogram_snapshots`.

The histogram does not claim exact percentiles. A bucketed distribution only establishes the interval containing observations. Exact p50/p95/p99 values require retaining samples or a separately specified quantile sketch with its own reproducibility contract.

Snapshots of the aggregate collector and distribution histograms are individually thread-safe but are not a single cross-object transactional snapshot. For offline benchmark publication, capture them after the benchmark runner has completed and worker activity has stopped.
