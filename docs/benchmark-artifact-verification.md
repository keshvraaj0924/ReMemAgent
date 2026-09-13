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

Configuration provenance is checked after exact-byte verification. Persisted run-level configuration manifests must match their serialized configurations. For paired artifacts that carry an `experiment_identity`, verification additionally recomputes the top-level paired configuration fingerprint and experiment identity from the persisted baseline configuration, treatment configuration, independent seeds, and runtime provenance. When paired temporal `execution_order` provenance is present, its deterministic counterbalancing contract and SHA-256 digest are also verified. Legacy artifacts that predate a particular identity field remain readable, but the verifier does not invent an identity guarantee that the artifact never recorded.

Evidence-bound paired artifacts additionally require the retained readiness JSON:

```text
remem-verify-benchmark \
  artifacts/webshop-paired.json \
  --manifest artifacts/webshop-paired.json.manifest.json \
  --preflight-evidence artifacts/webshop-readiness.json
```

The verifier authenticates the readiness object and requires its canonical `evidence_sha256` to match the digest embedded in the measured artifact's runtime provenance. A missing, malformed, tampered, or unrelated readiness object fails closed.

## Bind a duration-distribution sidecar

When a benchmark run also persisted the optional fixed-bucket duration sidecar, verification can validate and bind those exact bytes into the same machine-readable attestation:

```text
remem-verify-benchmark \
  artifacts/webshop.json \
  --manifest artifacts/webshop.json.manifest.json \
  --distribution-sidecar artifacts/webshop.distributions.json \
  --json
```

The sidecar is first parsed through the versioned `DistributionObservationSnapshot` contract. Unsupported schemas, malformed histogram fields, and inconsistent derived values such as a persisted count or mean that disagrees with the bucket counts/total fail verification. Only after structural validation succeeds does the verifier attest the sidecar's exact byte count and SHA-256 digest.

When a distribution sidecar is supplied, the verifier also emits `bundle_sha256`. This is a deterministic, domain-separated SHA-256 over the already-verified report digest and the validated distribution-sidecar digest. It gives CI and archival systems one stable identifier for that exact report/distribution pair without changing the benchmark-report or report-manifest schemas.

`bundle_sha256` is an association checksum, not an authenticity mechanism. It proves which exact report and distribution bytes were presented together to the verifier; it does not prove that an untrusted writer originally produced those files together. Use authenticated storage or signatures when provenance against a malicious writer matters.

## Machine-readable verification attestation

CI and experiment orchestration can request a stable JSON attestation instead of parsing the human-readable success line:

```text
remem-verify-benchmark \
  artifacts/webshop-paired.json \
  --manifest artifacts/webshop-paired.json.manifest.json \
  --preflight-evidence artifacts/webshop-readiness.json \
  --json
```

Successful output is one canonical JSON object containing:

- the verified report schema version;
- exact byte count and exact report SHA-256;
- `benchmark_name` when the report records one;
- the verified top-level `configuration_fingerprint` when present;
- the verified top-level `experiment_identity` when present;
- the authenticated readiness-evidence SHA-256 when the artifact is evidence-bound;
- the distribution schema version, exact byte count, and exact SHA-256 when `--distribution-sidecar` is supplied; and
- `bundle_sha256` when a distribution sidecar is supplied.

Legacy artifacts that do not record one of these optional identity fields emit `null` rather than having an identity inferred for them. If an identity field is present but is not a non-empty string, verification fails instead of silently coercing it into an attestation. Distribution fields and `bundle_sha256` likewise remain `null` when no distribution sidecar is supplied, preserving backward-compatible verification behavior.

The identity fields make the attestation directly useful for CI correlation and archival indexing: downstream automation can associate a verified byte-level artifact with its validated benchmark/configuration/experiment identity without reparsing the report. For legacy artifacts without readiness binding, `preflight_evidence_sha256` is `null`.

This output is derived only after all artifact, identity, controlled-admission, paired-provenance, readiness-binding, and requested distribution-sidecar checks succeed. It does not recollect mutable runtime or Git state. The attestation is intended for deterministic downstream automation; it is not a digital signature and should not be treated as proof against an untrusted artifact writer.

## Evidence boundary

The verifier establishes **artifact integrity and internal provenance consistency**, not scientific validity. A matching digest and identity do not prove that the experiment used an appropriate model, benchmark configuration, randomization protocol, or statistical analysis. Those remain separate research controls.

Likewise, the manifest and optional bundle digest are not authenticity or signing mechanisms: anyone who can modify all associated artifacts can generate a new internally consistent set. Use an authenticated archival or signing system when provenance against an untrusted writer is required.

## Recommended publication workflow

1. Execute the benchmark with its declared configuration and seeds.
2. Persist the deterministic benchmark report.
3. Generate the exact-byte manifest with the benchmark CLI's `--manifest` option.
4. When latency distributions are enabled, retain the deterministic distribution sidecar beside the report.
5. Verify the report from the saved files, supplying the retained readiness evidence and distribution sidecar when applicable.
6. Preserve the report, manifest, runtime provenance, readiness evidence when applicable, distribution sidecar when applicable, and code revision together.
7. Optionally capture the `--json` verification attestation for downstream CI or archival indexing.
8. Run statistical analysis only on the verified measured artifacts.

This workflow keeps integrity checking deterministic and dependency-light while avoiding any claim that a checksum or internally consistent experiment identity substitutes for reproducible scientific methodology.
