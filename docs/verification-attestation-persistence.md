# Persisted Benchmark Verification Attestations

`remem-verify-benchmark` can retain its machine-readable verification result as a canonical JSON artifact instead of relying on terminal capture.

```bash
remem-verify-benchmark \
  artifacts/webshop.json \
  --manifest artifacts/webshop.json.manifest.json \
  --observability-sidecar artifacts/webshop.observability.json \
  --distribution-sidecar artifacts/webshop.distributions.json \
  --attestation-output artifacts/webshop.verification.json \
  --json
```

The persisted file contains the same canonical JSON bytes emitted by `--json`, including the verified report digest, optional experiment identity fields, sidecar byte hashes, and the applicable bundle digest.

## Publication semantics

Attestations are written to a temporary file in the destination directory, flushed and `fsync`'d, and then published with a no-overwrite hard-link operation. An existing destination causes verification to fail rather than replacing retained evidence. Temporary files are removed on both success and failure.

This behavior is deliberate: a verification attestation is evidence about one exact set of retained bytes. Silently replacing it during a later run would make experiment audit trails ambiguous.

## Evidence boundary

A persisted attestation records what the verifier established at that moment. It is not a digital signature, timestamp authority, benchmark score, or proof of model effectiveness. `bundle_sha256` is a deterministic association digest for the validated report and optional sidecars; it does not certify the scientific quality of the experiment.

For evidence-bound measured runs, retain the report, manifest, readiness evidence, aggregate observability sidecar, distribution sidecar when enabled, and verification attestation together. Re-run verification from the retained inputs when independently checking the experiment bundle.
