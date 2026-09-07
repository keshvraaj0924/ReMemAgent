# Runtime provenance

`experiments.runtime_provenance` records the execution environment that produced a measured benchmark artifact.

## Contract

`RuntimeProvenance` contains:

- `schema_version` — version of the provenance record contract;
- `code_revision` — the Git revision supplied by `REMEM_GIT_COMMIT`, or discovered with `git rev-parse HEAD`;
- `working_tree_state` — `clean`, `dirty`, or `unknown`;
- `python_version` and `platform` — runtime identity;
- `package_version` — installed ReMemAgent distribution version, or `unknown`;
- `dependency_versions` — installed Python distribution versions in deterministic order;
- `dependency_fingerprint` — SHA-256 of the normalized dependency-version mapping.

The schema version is persisted with every record so readers can reject or migrate future incompatible provenance formats instead of silently interpreting a changed structure.

## Explicit CI/container inputs

Controlled execution environments can set:

```text
REMEM_GIT_COMMIT=<full source revision>
REMEM_GIT_STATE=clean|dirty|unknown
```

Explicit values take precedence over Git probing. Invalid working-tree states are rejected. Missing Git metadata is represented as `unknown`; the implementation never fabricates a revision or cleanliness state.

## Boundary

Runtime provenance is evidence about the execution environment, not proof of scientific reproducibility. The dependency fingerprint does not authenticate package contents, datasets, checkpoints, hardware, external benchmark state, or model weights. Those inputs remain part of the experiment's external provenance contract.

Benchmark CLI measured runs already attach this record to their persisted report. Artifact SHA-256 manifests provide a separate integrity layer for detecting later byte changes to the serialized report.

## Verification

The unit tests cover explicit revision/state precedence, unknown Git metadata, deterministic dependency fingerprinting, clean/dirty state detection, invalid explicit states, and schema-version serialization.
