# Paired readiness evidence

Controlled paired preflight now has a machine-readable evidence representation for CI and research orchestration.

`build_controlled_paired_preflight_evidence(...)` accepts the exact `ControlledPairedPreflightResult` returned by controlled readiness admission plus the declared source-checkout requirement contract. It does not recollect Git or runtime state after preflight.

The evidence records the admitted ReMemAgent runtime snapshot, the declared external source-checkout contract, the admitted external source snapshots, deterministic SHA-256 fingerprints for the source contract and observations, and an overall canonical evidence fingerprint.

`verify_controlled_paired_preflight_evidence(...)` rejects schema drift and content changes that no longer match the recorded overall fingerprint. `preflight_evidence_json(...)` emits deterministic compact JSON suitable for CI logs, comparison across hosts, or persistence.

## CLI persistence

`remem-paired-benchmark --preflight-only` can persist this exact controlled admission result with `--preflight-evidence PATH`. The option is fail-closed: it requires `--preflight-only` and a complete `--source-checkout` / `--require-source-revision` contract. The normal measured benchmark output and manifest are not created by this path.

Example with intentionally unspecified real revisions and environment-specific callables:

```bash
remem-paired-benchmark \
  --benchmark webshop \
  --episodes 10 \
  --max-steps 50 \
  --seeds 11,17,23 \
  --environment-factory <module:factory> \
  --success-evaluator <module:evaluator> \
  --baseline-policy-factory <module:baseline_factory> \
  --treatment-policy-factory <module:memory_factory> \
  --source-checkout webshop=/path/to/WebShop \
  --require-source-revision webshop=<exact-git-revision> \
  --preflight-only \
  --preflight-evidence artifacts/webshop-readiness.json
```

The persisted file is the canonical verified readiness payload generated from the snapshots that actually passed controlled preflight. `--overwrite` is required to replace an existing readiness file, matching benchmark artifact overwrite behavior.

This object is readiness evidence, not a benchmark result. It contains no measured episode outcomes and must not be used to claim ALFWorld/WebShop effectiveness. Measured claims still require a completed paired benchmark artifact and its integrity verification.
