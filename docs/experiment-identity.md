# Experiment identity

`experiments.experiment_identity` provides a deterministic identity for a benchmark protocol rather than for its serialized output bytes.

## Contract

`build_experiment_identity()` binds three inputs:

- the complete `BenchmarkRunConfiguration` protocol fields;
- the independent seed set, normalized into sorted unique integer order;
- JSON-compatible runtime provenance such as code revision and dependency versions.

The canonical payload is serialized with sorted keys and no insignificant whitespace, then hashed with SHA-256. Equivalent seed or mapping order therefore produces the same identity, while changing a protocol field, seed set, or runtime provenance changes the identity.

This identity is deliberately distinct from the benchmark artifact manifest. The manifest answers **"are these exact report bytes unchanged?"**; the experiment identity answers **"which protocol and runtime does this artifact represent?"**.

## Persisted benchmark artifacts

The benchmark report writers now persist `experiment_identity` whenever a report has an explicit configuration and runtime provenance. A single seeded report uses its run seed as the one-element seed set. A repeated report artifact removes the per-run seed from the shared protocol configuration and binds the complete ordered seed set instead. Consequently, changing the repeated seed set or runtime provenance produces a different identity, while merely changing the order in which the same seed reports are supplied does not.

The persisted identity is an audit key, not a substitute for the underlying configuration, per-seed reports, provenance, or exact-byte integrity manifest.

## Evidence boundary

An experiment identity is not a scientific-validity claim. It does not prove that a benchmark was executed correctly, that a model checkpoint is appropriate, or that two environments are semantically equivalent. It is a compact reproducibility key that makes those inputs explicit and comparable.

The identity also does not replace the stored configuration, per-seed reports, runtime provenance, or exact-byte manifest. Those records remain the authoritative evidence needed to reconstruct and audit an experiment.

## Intended integration

Measured benchmark launchers can compute an identity after validating the final configuration and runtime provenance, then store it alongside the report. Analysis tooling can use the identity as a fast compatibility check before combining artifacts, while still loading and validating the full underlying metadata.

The current module is intentionally dependency-free and does not alter benchmark execution or claim that external ALFWorld/WebShop runs have been performed.
