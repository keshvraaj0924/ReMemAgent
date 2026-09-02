# Benchmark artifact integrity

ReMemAgent can produce measured benchmark JSON artifacts, but serialization alone does not prove that an artifact was left unchanged after a run. `experiments.benchmark_manifest` provides a small, dependency-free integrity boundary for persisted reports.

## Contract

`build_benchmark_artifact_manifest(path)`:

1. reads the exact bytes on disk;
2. parses the UTF-8 JSON document;
3. requires the current `BENCHMARK_REPORT_SCHEMA_VERSION`;
4. records the exact byte count and SHA-256 digest.

`verify_benchmark_artifact(path, manifest)` repeats that process and fails closed if either the bytes or declared schema differs.

The digest covers the complete serialized report, including its final newline. A manifest is therefore a statement about the exact file that was measured or archived, not about an equivalent parsed JSON object.

## Intended use

A benchmark launch pipeline can persist the report, generate its manifest, archive both together, and verify the pair before downstream analysis. Verification should happen before results are copied into a paper table or imported into another analysis system.

This module deliberately does not claim that the report is scientifically valid. Structural validation remains the responsibility of `validate_benchmark_run_report`, and statistical interpretation remains a separate research layer.

## Limitations

The manifest is not a cryptographic signature and does not establish who produced the artifact. It detects accidental or post-run byte changes when the trusted manifest is preserved separately. Authenticity requires a separate signing or trusted-storage mechanism.
