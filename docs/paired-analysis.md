# Paired benchmark analysis

ReMemAgent evaluates baseline and treatment conditions at the **independent-seed** level. Episode observations are not pooled across seeds for inferential claims.

## Analysis contract

`experiments.paired_analysis.analyze_paired_benchmark_reports` composes four existing statistical primitives:

1. Match baseline and treatment reports by their explicit seed.
2. Validate that the benchmark and seed-independent evaluation configuration agree. Policy identity is intentionally excluded from this configuration fingerprint because policy is the experimental condition.
3. Compute one exact two-sided paired sign-flip test for each primary metric:
   - `success_rate`
   - `mean_reward`
   - `transfer_success_rate`
4. Apply Holm-Bonferroni correction across those three metric hypotheses.
5. Compute paired Cohen's `d_z` from the same seed-level deltas.

The analysis returns descriptive effect direction, raw p-values, adjusted p-values, effect sizes, and sample sizes. It does not apply an alpha threshold or label results as significant.

## Evidence boundary

These functions are statistical infrastructure, not benchmark evidence. A valid analysis object requires actual `BenchmarkRunReport` objects with explicit paired seeds and compatible configuration. The repository must still execute the external ALFWorld/WebShop environments and caller-owned model policies to produce scientific evidence.

## Reproducibility

For a defensible comparison, preserve the serialized benchmark configuration and all seed-level reports alongside the analysis output. Do not replace missing seeds with synthetic values, pool episode-level observations to inflate sample size, or treat a single seed as an inferential result.
