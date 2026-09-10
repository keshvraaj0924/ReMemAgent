# Reproducibility manifests

ReMemAgent provides a deterministic manifest contract for benchmark configuration artifacts and a separate artifact-serialization boundary.

## Manifest shape

`benchmark_configuration_manifest(configuration)` returns three top-level fields:

- `schema_version`: the reproducibility identity schema version;
- `configuration_digest`: the SHA-256 identity of the canonical versioned payload;
- `payload`: the versioned benchmark configuration envelope.

The digest is deterministic for the same validated configuration. Changes to benchmark parameters, seed, or structured runtime provenance change the identity.

## Benchmark artifacts

`benchmark_run_artifact(report)` converts a `BenchmarkRunReport` into a JSON-compatible mapping and embeds the configuration manifest under `configuration_manifest`. Artifact creation fails when the report has no configuration provenance, preventing unidentifiable results from being serialized as reproducible research artifacts.

`validate_benchmark_run_artifact(artifact, report)` verifies the embedded manifest against the report's configuration before the artifact is accepted. This keeps serialization separate from benchmark execution and makes provenance validation reusable for file writers, CI checks, and downstream analysis tools.

## Validation

`validate_benchmark_configuration_manifest(manifest, configuration)` is fail-closed. It rejects:

- unknown top-level fields;
- schema-version mismatches;
- payload mismatches;
- digest mismatches.

This is intentionally stricter than checking only the digest. Persisted benchmark artifacts should not silently acquire additional fields whose semantics are unknown to the current reader.

## Scope

A matching manifest proves that the supplied configuration has the recorded configuration identity. It does **not** prove that an external environment, model checkpoint, dependency lock, hardware runtime, or stochastic execution is scientifically equivalent. Those factors require their own provenance and repeated execution evidence.
