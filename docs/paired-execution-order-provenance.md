# Paired execution-order provenance

Paired ALFWorld/WebShop experiments counterbalance condition order across the canonical seed sequence so one condition is not always measured after the other.

`run_paired_external_benchmarks` returns an explicit `execution_order` trace in `PairedBenchmarkResult`. Each `PairedSeedExecution` records the seed and the first and second condition roles (`baseline` or `treatment`) used for that measured pair.

This makes temporal ordering a first-class part of the experiment result instead of requiring downstream code to reconstruct it from runner implementation details. The order remains deterministic: canonical even seed positions execute baseline then treatment, while canonical odd positions execute treatment then baseline.

## Artifact persistence

The paired CLI persists the measured result through `save_paired_execution_result`. Before writing an artifact, the persistence boundary verifies that the trace contains exactly one entry per paired seed, that trace seeds exactly match the statistical comparison seeds, and that every entry follows the deterministic counterbalanced condition plan. Invalid or externally fabricated traces fail closed before the lower-level paired report serializer runs.

The full validated trace is serialized under `execution_order`. A canonical SHA-256 digest of that trace is also stored in runtime provenance as `paired_execution_order_sha256` before experiment identity is constructed. This binds temporal execution order into the paired experiment identity without discarding the human-readable trace. Callers cannot override the reserved provenance key.

Artifact replacement remains atomic: serialization is completed on a private temporary file and the final path is replaced only after the paired report and execution provenance have both been written successfully.

## Verification

`remem-verify-benchmark` now performs semantic verification of persisted paired execution provenance after byte-integrity and generic configuration checks. For artifacts carrying `execution_order`, verification requires an exact seed-by-seed trace schema, exact agreement with the artifact seed list, the expected deterministic AB/BA counterbalancing plan, and a matching `paired_execution_order_sha256` value in runtime provenance.

This closes a gap where a byte-valid artifact with a newly generated sidecar manifest could otherwise carry a malformed or stale temporal trace. Legacy artifacts that predate `execution_order` remain verifiable under their existing byte/configuration contracts and are not retroactively assigned temporal provenance they never recorded.

The trace is execution provenance only. It does not imply that warm-up, throttling, cache state, or other temporal effects have been eliminated, and it does not establish benchmark improvement or statistical significance. Real ALFWorld/WebShop effectiveness claims still require controlled measured runs across independent seeds.
