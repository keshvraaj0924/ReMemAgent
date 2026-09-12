# Benchmark duration distributions

ReMemAgent can capture fixed-bucket benchmark timing distributions without changing the stable `ObservationSnapshot` or benchmark-report schemas.

## CLI usage

`remem-benchmark` supports opt-in episode-duration distributions through a paired configuration:

```bash
remem-benchmark \
  --benchmark webshop \
  --episodes 50 \
  --max-steps 20 \
  --seed 7 \
  --environment-factory package.module:make_environment \
  --policy-factory package.module:make_policy \
  --success-evaluator package.module:evaluate_success \
  --output artifacts/webshop.json \
  --observability-output artifacts/webshop.observability.json \
  --distribution-output artifacts/webshop.duration-distribution.json \
  --episode-duration-buckets 0.1,0.25,0.5,1.0,2.5,5.0,10.0
```

`--distribution-output` and `--episode-duration-buckets` form one complete contract. Supplying either without the other fails before measured execution. Bucket values are seconds and must be finite, non-negative, and strictly increasing.

The distribution output path is validated before measurement and must be distinct from the report, manifest, and aggregate observability paths. Existing outputs are rejected unless `--overwrite` is explicit.

When duration distributions are enabled, the CLI constructs one `DistributionObservationCollector` and supplies it to `BenchmarkSuiteRunner`. The runner's existing timing scope records one duration per attempted episode execution. The distribution-aware collector forwards that exact measured value to both the aggregate duration total and the configured fixed-bucket histogram. If `--observability-output` is also requested, aggregate and distribution evidence therefore refer to the same timing observations rather than independently timed runs.

After measurement finishes, aggregate and distribution snapshots are frozen and published with the report and optional integrity manifest through the rollback-safe benchmark artifact bundle. The report remains the final bundle publication marker. A failed publication does not intentionally leave newly published observability sidecars behind.

Distribution collection may also be used directly in Python:

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

Pass that collector to `BenchmarkSuiteRunner`. `collector.snapshot()` remains backward compatible and contains the existing counters and aggregate duration totals. Distribution data is retrieved separately through `collector.duration_distribution_snapshots()`.

This separation is intentional. Existing benchmark artifacts and snapshot schema version 1 are not silently changed, while experiments that need latency distributions can opt in explicitly.

## Semantics

Histogram bucket counts are non-cumulative and include one overflow bucket after the configured upper bounds. The histogram total and the base collector aggregate are derived from the same duration values. Fixed buckets support deterministic merging across workers through `merge_histogram_snapshots`.

The histogram does not claim exact percentiles. A bucketed distribution only establishes the interval containing observations. Exact p50/p95/p99 values require retaining samples or a separately specified quantile sketch with its own reproducibility contract.

Snapshots of the aggregate collector and distribution histograms are individually thread-safe but are not a single cross-object transactional snapshot. The benchmark CLI avoids concurrent snapshotting by freezing both only after measured execution has completed.
