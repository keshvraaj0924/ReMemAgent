# Observability evidence consistency

Benchmark verification treats aggregate observability and fixed-bucket duration distributions as two views of the same measured execution when both sidecars are supplied.

`remem-verify-benchmark` therefore validates more than each sidecar schema independently. Before emitting a combined `bundle_sha256`, it requires:

- the distribution sidecar to contain `benchmark.episode.duration_seconds`;
- the aggregate observability sidecar to contain the same duration aggregate;
- the aggregate observability sidecar to contain `benchmark.episodes.completed`;
- the completed-episode count to equal the histogram observation count; and
- the aggregate duration total to match the histogram total within a fixed absolute tolerance of `1e-12` seconds.

This closes an evidence-integrity gap where two structurally valid sidecars from different benchmark executions could otherwise be cryptographically associated with one report.

The check is intentionally applied only when both sidecars are provided. Aggregate-only and distribution-only verification remain backward compatible with their established contracts.

These checks establish internal consistency between retained telemetry artifacts. They do not prove that the benchmark result is scientifically valid, that the policy improved performance, or that the framework is production ready.
