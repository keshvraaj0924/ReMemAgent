# Paired condition label validation

Paired external benchmark execution treats condition labels as part of the experiment identity rather than presentation-only metadata.

`run_paired_external_benchmarks` and `run_paired_external_benchmarks_with_preflight` now validate labels before callable resolution, runtime preflight, or measured execution. Labels must be strings, must contain non-whitespace content, and must identify distinct conditions after trimming and case-folding.

This matters for expensive ALFWorld/WebShop runs because a malformed label previously surfaced only when `compare_benchmark_reports` was called after both conditions had already completed. The new fail-fast boundary prevents model/environment compute from being spent on an artifact that cannot be compared or persisted as a valid paired result.

Examples that are rejected before execution include `"memory"` versus `" MEMORY "`, an empty label, a whitespace-only label, or a non-string label.

The statistical comparison layer still validates labels independently. That secondary validation is intentional defense in depth for callers that construct `BenchmarkRunReport` collections directly without using the paired execution orchestration layer.

This validation does not establish that the treatment is scientifically meaningful; distinct policy configuration, shared evaluation protocol, independent seed isolation, real benchmark execution, and downstream statistical interpretation remain separate requirements.
