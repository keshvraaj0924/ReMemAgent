# Benchmark cleanup semantics

The benchmark runner treats environment cleanup as part of the runtime contract.

`BenchmarkSuiteRunner` always attempts `EnvironmentAdapter.close()` after an episode. Cleanup behavior is deliberately fail-closed without obscuring the causal episode error:

- If the episode succeeds and `close()` fails, the cleanup exception is raised.
- If the episode already failed and `close()` also fails, the original episode exception remains the primary exception.
- When observability is enabled, every cleanup failure increments `benchmark.environment.close_failures`.
- A missing or non-callable `close` method remains supported for adapters that do not require explicit cleanup.

This distinction matters for benchmark infrastructure. A cleanup failure is an infrastructure defect, not an unsuccessful task outcome, and must not be silently converted into a benchmark result. Conversely, replacing an actionable episode exception with a cleanup exception makes the root failure harder to diagnose.

The cleanup counter is diagnostic only and does not change episode success metrics. A benchmark report is never manufactured after an execution exception.
