# Experiment manifest fingerprints

`remem.reproducibility.ExperimentManifest` provides a dependency-free way to fingerprint experiment configuration.

## Contract

A manifest accepts JSON-compatible mappings, recursively validates finite numeric values, detaches nested mutable containers, and serializes them with sorted keys and stable separators. Its `sha256` property hashes that canonical UTF-8 JSON representation.

The digest is useful for storing beside benchmark artifacts so a later analysis can distinguish identical configuration from configuration drift without depending on Python object representations.

## Scope

The manifest fingerprints **configuration**, not source code, installed packages, model weights, environment state, or random-number-generator state. Those remain separate reproducibility dimensions and must be recorded by the experiment runner when applicable.

This utility intentionally does not fabricate or infer missing provenance. Unsupported runtime objects are rejected instead of being stringified into an ambiguous fingerprint.
