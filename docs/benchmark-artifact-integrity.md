# Benchmark artifact integrity

ReMemAgent can produce measured benchmark JSON artifacts, but serialization alone does not prove that an artifact was left unchanged after a run. `experiments.benchmark_manifest` provides a small, dependency-free integrity boundary for persisted reports.

## Contract

`BenchmarkArtifactManifest` is the typed integrity boundary. Construction rejects unsupported report schema versions, negative/non-integer byte counts, and non-canonical SHA-256 digests. This prevents malformed provenance metadata from entering downstream verification code.

`build_benchmark_artifact_manifest(path)`:

1. reads the exact bytes on disk;
2. parses the UTF-8 JSON document;
3. requires the current `BENCHMARK_REPORT_SCHEMA_VERSION`;
4. records the exact byte count and SHA-256 digest.

`save_benchmark_artifact_manifest(report_path, manifest_path)` validates and persists that manifest as a deterministic JSON sidecar. The default sidecar path is `<report>.manifest.json`; callers can supply an explicit path when an experiment has a fixed artifact layout.

`load_benchmark_artifact_manifest(path)` validates the manifest schema and required integrity fields before returning the typed manifest.

`verify_benchmark_artifact(path, manifest)` repeats the report hashing process and fails closed if either the bytes or declared schema differs. Schema and byte-count mismatches are rejected before the digest comparison, and the digest comparison uses a constant-time comparison primitive.

`verify_benchmark_artifact_manifest(report_path, manifest_path)` is the complete load-and-verify operation for archive consumers. It validates the sidecar, verifies the report against it, and returns the verified metadata for downstream logging.

The digest covers the complete serialized report, including its final newline. A manifest is therefore a statement about the exact file that was measured or archived, not about an equivalent parsed JSON object.

## CLI usage

The external benchmark CLI accepts `--manifest` on measured runs. For example, a launch can request a report and its sidecar together:

```text
python -m experiments.benchmark_cli ... --output artifacts/alfworld.json --manifest artifacts/alfworld.json.manifest.json
```

Before measured execution begins, the CLI validates both destinations. Existing report or manifest files require explicit `--overwrite`. The CLI also rejects a manifest path that resolves to the same file as the report path, preventing a successful report from being replaced by its own integrity sidecar. These checks happen before environment construction or measured execution so a failed artifact precondition cannot consume benchmark episodes first.

The manifest option is rejected during callable or runtime preflight because those modes do not produce benchmark artifacts.

The verification helper is intended for archive and analysis tooling:

```text
from pathlib import Path
from experiments.benchmark_manifest import verify_benchmark_artifact_manifest

verify_benchmark_artifact_manifest(
    Path("artifacts/alfworld.json"),
    Path("artifacts/alfworld.json.manifest.json"),
)
```

A benchmark archive should preserve the report and manifest together and verify the pair before downstream analysis or publication.

## Reproducibility boundary

The manifest authenticates the serialized benchmark report bytes and report schema. It does not authenticate external datasets, model checkpoints, dependency environments, or source revisions; those remain separate experiment provenance concerns. A verified artifact therefore means the report matches its declared manifest, not that the experiment itself is independently reproducible.

## Intended use

A benchmark launch pipeline can persist the report, generate its manifest, archive both together, and verify the pair before results are copied into a paper table or imported into another analysis system. This makes the integrity mechanism operational rather than merely available as a library helper.

This module deliberately does not claim that the report is scientifically valid. Structural validation remains the responsibility of `validate_benchmark_run_report`, and statistical interpretation remains a separate research layer.

## Limitations

The manifest is not a cryptographic signature and does not establish who produced the artifact. It detects accidental or post-run byte changes when the trusted manifest is preserved separately. Authenticity requires a separate signing or trusted-storage mechanism.
