# Paired benchmark condition contract

`compare_benchmark_reports` computes treatment-minus-baseline descriptive deltas per independent seed. Pairing is valid only when the two conditions describe the same evaluation protocol apart from the policy condition.

Before computing deltas, the statistics layer compares a canonical paired-configuration fingerprint. The fingerprint excludes the independent run seed and the policy callable identity, because changing the policy is the experimental condition being compared. It still includes the benchmark name, episode count, step limit, environment callable, success evaluator, transfer evaluator, and trust threshold.

This prevents a common experimental error: comparing two conditions on the same seed number while silently changing the benchmark or evaluation configuration. Such reports are rejected instead of producing an apparently valid paired delta. A baseline policy and a memory-guided policy can legitimately have different callable specifications while remaining paired observations.

The comparison remains descriptive. It does not pool episode observations, perform a hypothesis test, estimate causal effects, or claim statistical significance.

## Required workflow

1. Define baseline and treatment configurations with the same benchmark and evaluation settings.
2. Change only the policy condition when constructing the paired runs.
3. Run both conditions independently for the same unique seed set.
4. Persist the per-run reports with their configuration metadata.
5. Call `compare_benchmark_reports` to obtain seed-aligned treatment-minus-baseline deltas.
6. Apply any separately justified inferential statistical procedure only after checking its assumptions.

Reports without configuration metadata remain supported for backward compatibility, but the comparison cannot distinguish configuration drift when both sides omit metadata. New benchmark execution should preserve configuration metadata at the report boundary.
