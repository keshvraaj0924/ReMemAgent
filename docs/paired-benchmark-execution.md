# Paired benchmark execution

ReMemAgent provides `experiments.paired_benchmark` for controlled baseline-versus-treatment external benchmark runs.

## Contract

`run_paired_external_benchmarks` accepts two `ExternalBenchmarkSpec` values and one ordered sequence of independent seeds. Before execution it requires both conditions to share:

- benchmark name;
- episode count;
- maximum steps;
- environment factory;
- success evaluator;
- optional transfer-success evaluator;
- minimum trust threshold.

The policy specification is intentionally allowed to differ. This is the expected baseline-versus-memory comparison: both conditions execute the same evaluation protocol while using different policies.

Each condition is executed independently for every requested seed through the existing repeated external benchmark runner. The resulting reports are passed to `compare_benchmark_reports`, which aligns the conditions by explicit seed and computes descriptive treatment-minus-baseline deltas.

## Preflight

`preflight_paired_external_benchmarks` performs the same configuration validation and then runs the existing repeated runtime preflight independently for both conditions. An optional concrete action can be supplied to exercise one normalized environment step before measurement.

Preflight is not benchmark data. A measured run should only begin after both conditions pass the same preflight contract.

## Research interpretation

The paired runner does not perform hypothesis testing or claim statistical significance. It produces seed-aligned descriptive deltas. Any inferential analysis must be added as a separately justified statistical layer.

The paired runner also does not load models, tokenize prompts, or own third-party benchmark dependencies. Policy factories remain caller-owned, preserving the separation between ReMemAgent's deterministic memory heuristics and learned components.

## Testing

`tests/test_paired_benchmark.py` covers:

- shared-seed execution for both conditions;
- rejection of evaluation configuration drift before execution;
- independent preflight of both conditions;
- rejection of benchmark-family mismatches.

The repository quality workflow remains the authoritative execution gate for these tests.
