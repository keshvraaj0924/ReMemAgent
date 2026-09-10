# Paired benchmark condition contract

`compare_benchmark_reports` computes treatment-minus-baseline descriptive deltas per independent seed. Pairing is valid only when the two conditions describe the same experiment apart from the policy condition.

Before measured execution, `experiments.paired_benchmark` validates that baseline and treatment share the benchmark, episode count, step limit, environment factory, success evaluators, transfer evaluator, and trust threshold. It also now requires the two conditions to identify distinct policy configurations. Passing the same `policy_factory` to both sides, or the same `action_policy_factory` to both sides, fails closed before callable resolution, runtime preflight, or episode execution. This prevents an accidentally self-compared experiment from being persisted as if it represented a treatment contrast. A complete `policy_factory` and an `action_policy_factory` remain distinct policy boundaries even when their import strings happen to match, because ReMemAgent composes memory guidance only around the latter.

Before computing deltas, the statistics layer compares each report's canonical benchmark configuration fingerprint. The fingerprint excludes the independent run seed and policy identity but includes the remaining declared configuration, including episode count, step limit, callable specifications, and trust threshold.

This prevents a common experimental error: comparing two conditions on the same seed number while silently changing the benchmark configuration. Such reports are rejected instead of producing an apparently valid paired delta.

The comparison remains descriptive. It does not pool episode observations, perform a hypothesis test, estimate causal effects, or claim statistical significance.

## Required workflow

1. Define baseline and treatment configurations with the same benchmark and evaluation settings but distinct policy configurations.
2. Run both conditions independently for the same unique seed set.
3. Persist the per-run reports with their configuration metadata.
4. Call `compare_benchmark_reports` to obtain seed-aligned treatment-minus-baseline deltas.
5. If an inferential claim is required, apply `exact_paired_sign_flip_test` to one metric's paired deltas only after checking the experimental assumptions and analysis plan.

## Exact paired sign-flip test

`exact_paired_sign_flip_test` is a separate, dependency-free inferential primitive for a single metric. It enumerates every sign assignment of the non-zero paired seed differences and computes a two-sided exact p-value using the absolute mean delta as the test statistic. Zero differences contribute no sign choice. The implementation is bounded to 20 non-zero pairs so an accidental large production run cannot trigger unbounded exponential work.

The test is deliberately not embedded in `compare_benchmark_reports`: descriptive effect sizes and inferential testing remain separate concerns. The framework also does not perform multiple-comparison correction, power analysis, equivalence testing, or causal identification. Those require an experiment-specific statistical plan.

Reports without configuration metadata remain supported for backward compatibility, but the comparison cannot distinguish configuration drift when both sides omit metadata. New benchmark execution should preserve configuration metadata at the report boundary.
