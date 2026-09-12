# Distribution observability sidecars

ReMemAgent keeps fixed-bucket latency distributions separate from the stable `ObservationSnapshot` contract. The versioned `DistributionObservationSnapshot` sidecar records named `ObservationHistogramSnapshot` values without changing existing benchmark report or aggregate observability schemas.

Sidecars are serialized as canonical JSON with sorted keys, compact separators, finite numeric values, and a trailing newline. Persistence uses a private temporary file plus `os.replace`, so readers never observe a partially written snapshot. Existing files are rejected unless overwrite is explicitly enabled.

Loading is fail-closed. The reader validates the schema version, metric names, bucket contracts, counts, totals, and the persisted derived `count` and `mean` fields. A file whose derived fields were edited independently from its raw bucket counts or total is rejected rather than normalized silently.

A distribution sidecar is measurement evidence, not a percentile report. Fixed buckets can be merged exactly only when their upper-bound contracts are identical; they do not recover exact p50/p95/p99 values. Experiment configurations should therefore retain the bucket boundaries alongside the sidecar artifact and avoid comparing histograms with different bucket contracts as though they were equivalent.

The current persistence layer is intentionally independent of the benchmark report schema. The next integration step is to stage and publish this sidecar in the same rollback-safe benchmark artifact bundle used for reports, manifests, and aggregate observability snapshots.
