# Paired readiness evidence

Controlled paired preflight now has a machine-readable evidence representation for CI and research orchestration.

`build_controlled_paired_preflight_evidence(...)` accepts the exact `ControlledPairedPreflightResult` returned by controlled readiness admission plus the declared source-checkout requirement contract. It does not recollect Git or runtime state after preflight.

The evidence records the admitted ReMemAgent runtime snapshot, the declared external source-checkout contract, the admitted external source snapshots, deterministic SHA-256 fingerprints for the source contract and observations, and an overall canonical evidence fingerprint.

`verify_controlled_paired_preflight_evidence(...)` rejects schema drift and content changes that no longer match the recorded overall fingerprint. `preflight_evidence_json(...)` emits deterministic compact JSON suitable for CI logs, comparison across hosts, or explicit persistence by an orchestration layer.

This object is readiness evidence, not a benchmark result. It contains no measured episode outcomes and must not be used to claim ALFWorld/WebShop effectiveness. Measured claims still require a completed paired benchmark artifact and its integrity verification.
