# Paired execution-order provenance

Paired ALFWorld/WebShop experiments counterbalance condition order across the canonical seed sequence so one condition is not always measured after the other.

`run_paired_external_benchmarks` now returns an explicit `execution_order` trace in `PairedBenchmarkResult`. Each `PairedSeedExecution` records the seed and the first and second condition roles (`baseline` or `treatment`) used for that measured pair.

This makes temporal ordering a first-class part of the in-memory experiment result instead of requiring downstream code to reconstruct it from the current runner implementation. The order remains deterministic: canonical even seed positions execute baseline then treatment, while canonical odd positions execute treatment then baseline.

The trace is execution provenance only. It does not imply that warm-up, throttling, cache state, or other temporal effects have been eliminated, and it does not establish benchmark improvement or statistical significance.

Persisted paired benchmark artifacts do not yet serialize this trace. The next persistence increment should add a validated execution-order field rather than infer or fabricate one for externally assembled report collections.
