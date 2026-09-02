# Benchmark artifact integrity

ReMemAgent can produce measured benchmark JSON artifacts, but serialization alone does not prove that an artifact was left unchanged after a run. `experiments.benchmark_manifest` provides a small, dependency-free integrity boundary for persisted reports.

## Contract

`build_benchmark_artifact_manifest(path)`:

1. reads the exact bytes on disk;
2. parses the UTF-8 JSON document;
3. requires the current `BENCHMARK_REPORT_SCHEMA_VERSION`;
4. records the exact byte count and SHA-256 digest.

`save_benchmark_artifact_manifest(report_path, manifest_path)` validates and persists that manifest as a deterministic JSON sidecar. The default sidecar path is `<report>.manifest.json`; callers can supply an explicit path when an experiment has a fixed artifact layout.

`load_benchmark_artifact_manifest(path)` validates the manifest schema and required integrity fields before returning the typed manifest.

`verify_benchmark_artifact(path, manifest)` repeats the report hashing process and fails closed if either the bytes or declared schema differs.

The digest covers the complete serialized report, including its final newline. A manifest is therefore a statement about the exact file that was measured or archived, not about an equivalent parsed JSON object.

## CLI usage

The external benchmark CLI accepts `--manifest` on measured runs. For example, a launch can request a report and its sidecar together:

```text
python -m experiments.benchmark_cli ... --output artifacts/alfworld.json --manifest artifacts/alfworld.json.manifest.json
```

The manifest option is rejected during callable or runtime preflight because those modes do not produce benchmark artifacts.

A benchmark archive should preserve the report and manifest together and verify the pair before downstream analysis or publication.

## Intended use

A benchmark launch pipeline can persist the report, generate its manifest, archive both together, and verify the pair before results are copied into a paper table or imported into another analysis system. This makes the integrity mechanism operational rather than merely available as a library helper.

This module deliberately does not claim that the report is scientifically valid. Structural validation remains the responsibility of `validate_benchmark_run_report`, and statistical interpretation remains a separate research layer.

## Limitations

The manifest is not a cryptographic signature and does not establish who produced the artifact. It detects accidental or post-run byte changes when the trusted manifest is preserved separately. Authenticity requires a separate signing or trusted-storage mechanism.
