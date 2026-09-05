# Paired inference

ReMemAgent keeps inferential statistics separate from benchmark execution. A benchmark run produces measured reports; the statistics layer consumes those reports without changing how episodes are sampled or executed.

## Exact paired sign-flip test

`experiments.benchmark_statistics.exact_paired_sign_flip_test` performs an exact two-sided paired sign-flip test over one metric's seed-level treatment-minus-baseline deltas.

The test:

- treats each independent seed as one paired observation;
- excludes zero deltas from sign enumeration while retaining them in the observed mean and sample size;
- enumerates every `2**n` sign assignment for at most 20 non-zero deltas;
- returns the observed mean delta, exact two-sided p-value, sample size, non-zero count, and number of evaluated permutations;
- does not pool episode-level observations;
- does not claim a causal effect merely because a p-value is small.

The 20-observation bound is deliberate: exact enumeration is exponential, so larger studies need a different, explicitly designed computational method rather than silently switching algorithms.

## Multiple metrics

A paired benchmark typically evaluates several metrics. `holm_bonferroni_adjust` provides a dependency-free Holm-Bonferroni family-wise error correction for a finite mapping of hypothesis names to raw p-values.

The correction:

1. validates that every input is a finite probability in `[0, 1]`;
2. sorts hypotheses by raw p-value, breaking ties deterministically by name;
3. applies the Holm step-down multiplier;
4. enforces monotonicity of adjusted values;
5. caps adjusted values at one;
6. restores the caller's original key order.

The function intentionally does not select an alpha level or label hypotheses as significant. The experiment owner must declare the hypothesis family and decision rule before interpreting adjusted values.

## What this does not establish

These utilities do not establish benchmark superiority, causal impact, transferability, or production readiness. Statistical validity still depends on the experimental design, independent seed construction, metric choice, and assumptions behind the selected test.

For the current paired benchmark design, the repository therefore records both the protocol configuration and the independent seeds before comparison. Missing configuration, mismatched seed sets, duplicate seeds, or configuration drift are rejected rather than inferred.

## Recommended usage

Use descriptive seed-level summaries first. For a pre-registered paired hypothesis family, compute one inferential result per metric from the matched seed deltas and apply Holm-Bonferroni to that declared family. Preserve the raw p-values, adjusted p-values, seed set, code revision, runtime provenance, and benchmark configuration with the final artifact.
