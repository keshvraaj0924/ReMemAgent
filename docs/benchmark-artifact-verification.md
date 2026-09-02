# Benchmark artifact verification

ReMemAgent can persist an exact-byte SHA-256 manifest beside a benchmark report. The manifest is useful when a measured artifact is copied between machines, archived for later analysis, or handed to a training/evaluation pipeline.

## Verify an artifact offline

After a measured run has produced `artifacts/benchmark.json` and its sidecar manifest, use either the module form or the installed console command:

```text
python -m experiments.verify_benchmark_artifact artifacts/benchmark.json
remem-verify-benchmark artifacts/benchmark.json
```

The verifier derives the conventional sidecar path:

```text
artifacts/benchmark.json.manifest.json
```

A manifest stored elsewhere can be selected explicitly:

```text
python -m experiments.verify_benchmark_artifact \
  artifacts/benchmark.json \
  --manifest archived/benchmark.manifest.json
```

The command exits successfully only when the report is valid for the current benchmark-report schema and its byte count and SHA-256 digest exactly match the manifest. A changed report, malformed manifest, unsupported manifest schema, or unsupported report schema fails with an error rather than producing a warning.

## Evidence boundary

The verifier establishes **artifact integrity**, not scientific validity. A matching digest does not prove that the experiment used an appropriate model, benchmark configuration, randomization protocol, or statistical analysis. Those remain separate research controls.

Likewise, the manifest is not an authenticity or signing mechanism: anyone who can modify both the report and manifest can generate a new matching pair. Use an authenticated archival or signing system when provenance against an untrusted writer is required.

## Recommended publication workflow

1. Execute the benchmark with its declared configuration and seeds.
2. Persist the deterministic benchmark report.
3. Generate the exact-byte manifest with the benchmark CLI's `--manifest` option.
4. Verify the report from the saved files before analysis or archival.
5. Preserve the report, manifest, runtime provenance, and code revision together.
6. Run statistical analysis only on the verified measured artifacts.

This workflow keeps integrity checking deterministic and dependency-light while avoiding any claim that a checksum substitutes for reproducible scientific methodology.
