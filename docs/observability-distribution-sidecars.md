# Distribution observability sidecars

ReMemAgent keeps fixed-bucket latency distributions separate from the stable `ObservationSnapshot` contract. The versioned `DistributionObservationSnapshot` sidecar records named `ObservationHistogramSnapshot` values without changing existing benchmark report or aggregate observability schemas.

Sidecars are serialized as canonical JSON with sorted keys, compact separators, finite numeric values, and a trailing newline. Persistence uses a private temporary file plus `os.replace`, so readers never observe a partially written snapshot. Existing files are rejected unless overwrite is explicitly enabled.

Loading is fail-closed. The reader validates the schema version, metric names, bucket contracts, counts, totals, and the persisted derived `count` and `mean` fields. A file whose derived fields were edited independently from its raw bucket counts or total is rejected rather than normalized silently.

A distribution sidecar is measurement evidence, not a percentile report. Fixed buckets can be merged exactly only when their upper-bound contracts are identical; they do not recover exact p50/p95/p99 values. Experiment configurations should therefore retain the bucket boundaries alongside the sidecar artifact and avoid comparing histograms with different bucket contracts as though they were equivalent.

## Transactional benchmark publication

`experiments.benchmark_observability_bundle.persist_benchmark_observability_bundle` stages the benchmark report together with any requested integrity manifest, aggregate observability snapshot, and distribution sidecar before publication. Auxiliary artifacts are published first and the benchmark report is published last, preserving the report as the bundle commit marker.

Destination aliases are rejected before staging begins. If publication fails after one or more auxiliary artifacts have been published—for example because another process claims the report path—the existing bundle publisher rolls those auxiliary paths back. In overwrite mode, pre-existing destination bytes are backed up and restored on publication failure.

This transactional layer does not change benchmark report schema version 1 or the aggregate `ObservationSnapshot` schema. The distribution sidecar remains opt-in and independently versioned. The remaining integration work is CLI configuration of explicit episode-duration bucket boundaries and sidecar output paths; no implicit bucket contract should be introduced because changing bucket boundaries changes the measurement semantics.
