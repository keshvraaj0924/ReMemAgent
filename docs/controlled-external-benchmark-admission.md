# Controlled external benchmark admission

`experiments.controlled_external_benchmark` provides the provenance-preserving execution boundary for ordinary (non-paired) external benchmark runs.

The legacy preflight helpers validate runtime and source-checkout state before measurement, but their return types intentionally remain benchmark reports for compatibility. That means they cannot carry the exact admitted runtime/source snapshots into later artifact persistence without recollecting mutable state.

The controlled API closes that handoff gap:

- `run_controlled_external_benchmark(...)` validates one runtime snapshot and one source-checkout snapshot before environment preflight or measurement, then returns both snapshots beside the measured report.
- `run_controlled_repeated_external_benchmarks(...)` validates the shared runtime/source state once, probes every requested seed independently, then returns the exact admitted snapshots beside all measured reports.
- Runtime drift or source-checkout drift fails before benchmark callable resolution, environment construction, or measured execution.
- Repeated runs reuse one admitted machine/source state but retain seed-isolated environment preflight and measurement.

The returned snapshots are intended for artifact identity binding. Callers must persist these exact objects rather than collecting Git or package metadata again after measurement.

## Current boundary

This module establishes a correct admission-to-measurement provenance handoff. The general `remem-benchmark` CLI still uses the legacy result path as of this increment, so ordinary report artifacts do not yet persist the controlled source snapshots automatically. Wiring this controlled result into CLI persistence and identity verification is the next reproducibility milestone.

No benchmark-effectiveness claim follows from this mechanism. ALFWorld/WebShop results remain evidence only after real pinned executions are run and their artifacts pass the repository's validation and integrity checks.
