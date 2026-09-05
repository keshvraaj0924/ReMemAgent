# Paired benchmark condition contract

`compare_benchmark_reports` computes treatment-minus-baseline descriptive deltas per independent seed. Pairing is valid only when the two conditions describe the same experiment apart from the policy condition.

Before computing deltas, the statistics layer now compares each report's canonical benchmark configuration fingerprint. The fingerprint excludes the independent run seed but includes the remaining declared configuration, including episode count, step limit, callable specifications, and trust threshold.

This prevents a common experimental error: comparing two conditions on the same seed number while silently changing the benchmark configuration. Such reports are rejected instead of producing an apparently valid paired delta.

The comparison remains descriptive. It does not pool episode observations, perform a hypothesis test, estimate causal effects, or claim statistical significance.

## Required workflow

1. Define baseline and treatment configurations with the same benchmark and evaluation settings.
2. Run both conditions independently for the same unique seed set.
3. Persist the per-run reports with their configuration metadata.
4. Call `compare_benchmark_reports` to obtain seed-aligned treatment-minus-baseline deltas.
5. If an inferential claim is required, apply `exact_paired_sign_flip_test` to one metric's paired deltas only after checking the experimental assumptions and analysis plan.

## Exact paired sign-flip test

`exact_paired_sign_flip_test` is a separate, dependency-free inferential primitive for a single metric. It enumerates every sign assignment of the non-zero paired seed differences and computes a two-sided exact p-value using the absolute mean delta as the test statistic. Zero differences contribute no sign choice. The implementation is bounded to 20 non-zero pairs so an accidental large production run cannot trigger unbounded exponential work.

The test is deliberately not embedded in `compare_benchmark_reports`: descriptive effect sizes and inferential testing remain separate concerns. The framework also does not perform multiple-comparison correction, power analysis, equivalence testing, or causal identification. Those require an experiment-specific statistical plan.

Reports without configuration metadata remain supported for backward compatibility, but the comparison cannot distinguish configuration drift when both sides omit metadata. New benchmark execution should preserve configuration metadata at the report boundary.
