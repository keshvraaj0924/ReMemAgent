# Paired policy persistence invariants

Paired benchmark artifacts are scientific evidence only when each condition represents one stable policy configuration across the complete seed set and the two conditions are genuinely distinct.

`save_paired_benchmark_result` therefore validates policy identity again at the persistence boundary instead of assuming that reports were produced by `run_paired_external_benchmarks`.

The serializer enforces three independent contracts before writing bytes:

1. Every baseline report must share one configuration apart from its run seed.
2. Every treatment report must share one configuration apart from its run seed.
3. Baseline and treatment must use distinct policy configurations while continuing to share the same benchmark protocol apart from seed and policy identity.

This closes a provenance gap for callers that construct `BenchmarkRunReport` collections directly or load reports from another execution path. Without this validation, a condition could silently switch policy factories between seeds, or two identical policies could be persisted as if they were a meaningful treatment comparison.

The check is intentionally performed before artifact serialization and experiment-identity construction. Invalid report collections therefore cannot receive a paired experiment identity or integrity manifest through the normal persistence path.

These invariants establish artifact consistency only. They do not establish benchmark improvement, statistical significance, transfer advantage, or production readiness. Those claims still require controlled execution against the real benchmark environments and selected model checkpoints.
