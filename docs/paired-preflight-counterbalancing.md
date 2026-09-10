# Paired preflight counterbalancing

Paired ALFWorld/WebShop-style runs use runtime preflight to catch environment and policy failures before measured episodes begin. Because preflight can construct real environments, load model-backed policies, warm caches, and touch shared services, its order can influence the state immediately preceding measurement even though preflight observations are not benchmark data.

`preflight_paired_external_benchmarks` therefore uses the same deterministic seed-position counterbalancing as measured paired execution. For even seed positions, baseline is probed before treatment. For odd seed positions, treatment is probed before baseline. Each runtime validator receives exactly one seed at a time, and both conditions still probe the identical ordered seed set.

All static paired-specification, repeated-seed, and callable validation still completes before the first runtime probe. A malformed treatment callable therefore cannot trigger baseline environment/model construction.

This change reduces systematic condition-first warm-up and temporal-order bias. It does not eliminate nondeterminism from external APIs, benchmark packages, model servers, hardware, caches, or caller-owned factories, and it does not convert preflight output into measured evidence.

Regression coverage is in `tests/test_paired_preflight_counterbalancing.py`. The test asserts the exact baseline/treatment probe sequence across multiple seeds while preserving the common seed set and probe action.
