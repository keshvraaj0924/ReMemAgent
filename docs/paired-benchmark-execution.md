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

## CLI execution

The `remem-paired-benchmark` command exposes the same paired contract for real experiments without embedding model or benchmark SDK dependencies in ReMemAgent. A typical run supplies exactly one policy factory form per condition:

```text
remem-paired-benchmark \
  --benchmark alfworld-text \
  --episodes 100 \
  --max-steps 30 \
  --seeds 11,17,29 \
  --environment-factory your_module:make_environment \
  --success-evaluator your_module:evaluate_success \
  --baseline-policy-factory your_module:make_baseline_policy \
  --treatment-policy-factory your_module:make_memory_policy \
  --output artifacts/alfworld-paired.json \
  --manifest artifacts/alfworld-paired.json.manifest
```

Baseline and treatment may instead use the lower-level `--*-action-policy-factory` form when the caller wants ReMemAgent to own memory-guided policy composition while the caller owns model inference. The CLI enforces exactly one policy factory form for each condition at argument parsing time, requires explicit independent seeds, and always runs the paired runtime preflight before measured execution. Configuration and execution errors are surfaced as concise process errors rather than Python tracebacks. The resulting artifact includes runtime provenance and can be protected by the exact-byte integrity manifest.

The CLI is a production execution surface, not a benchmark result generator: the configured factories and evaluator remain caller-owned, and no model checkpoint is loaded by the framework itself.

## Artifact persistence

`save_paired_benchmark_result` persists both ordered per-seed condition reports and the paired descriptive comparison in one deterministic JSON artifact. The artifact records:

- the common benchmark name and ordered seed set;
- baseline and treatment labels;
- every measured report for both conditions;
- seed-aligned success-rate, reward, and transfer-success deltas;
- the shared configuration fingerprint when configuration metadata is available;
- optional runtime provenance.

Input reports are validated and sorted by seed before serialization, so caller iteration order does not change artifact bytes. The persistence layer does not calculate or claim statistical significance; the stored comparison remains descriptive.

## Preflight

`preflight_paired_external_benchmarks` performs the same configuration validation and then runs the existing repeated runtime preflight independently for both conditions. An optional concrete action can be supplied to exercise one normalized environment step before measurement.

`run_paired_external_benchmarks_with_preflight` composes that validation into a fail-fast measured workflow: both conditions are preflighted for every requested seed before either condition starts measured execution. This prevents a later policy or environment construction failure from leaving an experiment with only one measured condition.

Preflight is not benchmark data. A measured run should only begin after both conditions pass the same preflight contract.

## Research interpretation

The paired runner does not perform hypothesis testing or claim statistical significance. It produces seed-aligned descriptive deltas. Any inferential analysis must be added as a separately justified statistical layer.

The paired runner also does not load models, tokenize prompts, or own third-party benchmark dependencies. Policy factories remain caller-owned, preserving the separation between ReMemAgent's deterministic memory heuristics and learned components.

## Testing

`tests/test_paired_benchmark.py` covers:

- shared-seed execution for both conditions;
- rejection of evaluation configuration drift before execution;
- independent preflight of both conditions;
- preflight completing before measured execution;
- rejection of benchmark-family mismatches.

`tests/test_paired_benchmark_cli.py` covers condition-specific policy factory wiring, strict seed parsing, required policy selection, and rejection of conflicting policy factory forms. `tests/test_benchmark_report.py` additionally covers paired artifact ordering, deterministic bytes, and comparison-seed validation.

The repository quality workflow remains the authoritative execution gate for these tests.
