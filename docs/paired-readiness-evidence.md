# Paired readiness evidence

Controlled paired preflight has a machine-readable evidence representation for CI and research orchestration.

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

## Binding readiness to later measurement

A readiness file can be verified structurally without proving that a later process still has the same runtime and source state. For measured orchestration, use `run_evidence_bound_paired_external_benchmarks(...)` or the measured CLI evidence-binding option rather than treating standalone verification as sufficient.

The evidence-bound runner performs fresh controlled admission, compares the previously persisted evidence with the exact runtime/source snapshots admitted in the current process, and starts measured episodes only when the canonical evidence fingerprint matches. A malformed, tampered, or stale readiness artifact therefore fails before measurement.

Internally, `run_admitted_paired_external_benchmarks(...)` provides a narrow measurement boundary for an already-admitted `ControlledPairedPreflightResult`. It deliberately does not recollect runtime state, recollect external Git state, or rerun environment readiness probes. This separation prevents a second mutable observation from being substituted between evidence validation and measurement.

For measured CLI execution, pass the persisted artifact with `--require-preflight-evidence PATH` while declaring the same controlled source-checkout contract used for the run:

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
  --require-preflight-evidence artifacts/webshop-readiness.json \
  --output artifacts/webshop-paired.json \
  --manifest artifacts/webshop-paired.manifest.json
```

`--require-preflight-evidence` is valid only for measured execution and requires explicit source checkout paths plus exact required source revisions. The CLI fresh-admits runtime, source checkouts, and paired environment readiness; validates the persisted evidence against those exact admitted snapshots; and only then starts measured episodes. If the evidence is malformed, tampered, or stale, the run fails before measured execution and before the paired report or manifest is persisted.

For an evidence-bound measured CLI run, the paired artifact also records the verified readiness artifact's canonical SHA-256 as `runtime_provenance.preflight_evidence_sha256`. Runtime provenance already participates in paired experiment identity construction, so the measured artifact identity is cryptographically bound to the exact readiness evidence that authorized measurement. This digest is an audit link, not a substitute for retaining and independently verifying the readiness JSON itself.

This object is readiness evidence, not a benchmark result. It contains no measured episode outcomes and must not be used to claim ALFWorld/WebShop effectiveness. Measured claims still require a completed paired benchmark artifact and its integrity verification.
