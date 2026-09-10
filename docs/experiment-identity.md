# Experiment identity

`experiments.experiment_identity` provides deterministic identities for benchmark protocols rather than for their serialized output bytes.

## Single-condition contract

`build_experiment_identity()` binds three inputs:

- the complete `BenchmarkRunConfiguration` protocol fields;
- the independent seed set, normalized into sorted unique integer order;
- JSON-compatible runtime provenance such as code revision and dependency versions.

The configuration preserves `policy_factory` and `action_policy_factory` as separate provenance fields. A caller-owned complete policy therefore cannot be confused with a learned action policy that ReMemAgent wraps with memory guidance, even when both callables happen to use the same import path.

The canonical payload is serialized with sorted keys and no insignificant whitespace, then hashed with SHA-256. Equivalent seed or mapping order therefore produces the same identity, while changing a protocol field, seed set, or runtime provenance changes the identity.

`verify_experiment_identity()` provides the corresponding semantic verification boundary. It validates the persisted identity format, recomputes the expected identity from the supplied configuration, seed set, and provenance, and fails closed when they disagree. The comparison uses a constant-time digest comparison so callers do not need to duplicate identity-checking logic.

Verification should be performed against the complete persisted metadata rather than an identity copied from an artifact. A changed configuration, seed set, or runtime provenance must invalidate the check.

## Paired-condition contract

`build_paired_experiment_identity()` binds both condition configurations, the canonical independent seed set, and the shared runtime provenance. Baseline and treatment configurations remain separately named in the canonical payload so changing only the treatment policy changes the paired identity.

Paired report persistence uses a paired configuration fingerprint for the same reason: it hashes the complete seed-independent baseline and treatment configurations instead of deriving the fingerprint from the baseline alone. The protocol-compatibility check still intentionally removes policy identity when deciding whether two conditions are comparable; this lets baseline and treatment use different policies while continuing to reject drift in benchmark, environment, evaluator, episode count, step limit, trust threshold, or other shared protocol fields.

This separation is important: **comparability** asks whether the two conditions differ only where an experiment permits them to differ, while **identity** must record exactly which two conditions were executed.

## Persisted benchmark artifacts

The benchmark report writers persist `experiment_identity` whenever a report has an explicit configuration and runtime provenance. A single seeded report uses its run seed as the one-element seed set. A repeated report artifact removes the per-run seed from the shared protocol configuration and binds the complete ordered seed set instead. Consequently, changing the repeated seed set or runtime provenance produces a different identity, while merely changing the order in which the same seed reports are supplied does not.

Paired artifacts bind both seed-independent condition configurations. Their `configuration_fingerprint` and `experiment_identity` therefore change if either the baseline or treatment policy identity changes.

The persisted identity is an audit key, not a substitute for the underlying configuration, per-seed reports, provenance, or exact-byte integrity manifest.

This identity is deliberately distinct from the benchmark artifact manifest. The manifest answers **"are these exact report bytes unchanged?"**; the experiment identity answers **"which protocol and runtime does this artifact represent?"**.

## Evidence boundary

An experiment identity is not a scientific-validity claim. It does not prove that a benchmark was executed correctly, that a model checkpoint is appropriate, or that two environments are semantically equivalent. It is a compact reproducibility key that makes those inputs explicit and comparable.

The identity also does not replace the stored configuration, per-seed reports, runtime provenance, or exact-byte manifest. Those records remain the authoritative evidence needed to reconstruct and audit an experiment.

## Intended integration

Measured benchmark launchers can compute an identity after validating the final configuration and runtime provenance, then store it alongside the report. Analysis tooling can use identity verification as a fast compatibility check before combining artifacts, while still loading and validating the full underlying metadata.

The current module is intentionally dependency-free and does not alter benchmark execution or claim that external ALFWorld/WebShop runs have been performed.
