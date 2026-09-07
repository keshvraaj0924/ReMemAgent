# Reproducibility fingerprint validation

`fingerprint_experiment_inputs` is the identity boundary for synthetic experiment inputs and configuration.

Before hashing configuration, ReMemAgent recursively validates the JSON-compatible subset used by the fingerprint:

- mapping keys must be strings;
- floats must be finite;
- nested mappings and lists are validated recursively;
- unsupported Python values are rejected instead of relying on `json.dumps` failure behavior.

Mapping key order is intentionally ignored because canonical JSON serialization sorts keys. List order remains significant because experiment case order can affect execution semantics.

This validation is about deterministic artifact identity. It does not prove that a configuration is scientifically valid, that a dependency is trustworthy, or that two environments are operationally equivalent. Runtime and dependency provenance are recorded separately by the runtime-provenance layer.

The corresponding unit tests cover nested mapping normalization and rejection of non-string keys, non-finite floats, and unsupported nested values.
