# Benchmark artifact publication

The general benchmark CLI treats its report, optional integrity manifest, and optional observability snapshot as one staged experiment bundle.

Every requested artifact is first written to a private temporary file in its destination directory. The integrity manifest is computed from the exact staged report bytes, and the observability snapshot is also fully serialized before any final destination is claimed. Only after every requested artifact is prepared does publication begin.

## Publication order and commit marker

Optional companion artifacts are published first and the benchmark report is published **last**. The report therefore acts as the bundle commit marker: under normal publication failure handling, a newly visible report implies that its requested companion artifacts were already published successfully.

By default, final publication is no-overwrite and race-safe. The CLI still performs early existence checks so obvious conflicts fail before measured execution, but correctness does not depend on those checks. Publication uses atomic hard links. If another process creates a requested destination after the precheck, ReMemAgent raises `FileExistsError`, preserves the competing artifact, removes any companion files it published earlier in the same attempt, and cleans its private staging files.

When `--overwrite` is explicitly supplied, existing destination bytes are copied to private rollback files before replacement begins. Each staged artifact is then published with `os.replace()`. If a later publication fails, destinations already replaced by the bundle are restored from their rollback copies; destinations that did not exist before the attempt are removed.

The lower-level manifest and observability persistence APIs keep their standalone overwrite contracts. Bundle rollback is owned by the benchmark CLI publication layer rather than by those serializers.

## Failure model

This is a rollback-capable multi-file publication protocol, not a filesystem transaction. It protects against ordinary Python exceptions and destination races during one live CLI process, but it cannot guarantee rollback after abrupt process termination, host failure, or storage failure between individual filesystem operations. ReMemAgent does not claim crash-atomic multi-file commits without a journaled artifact store or a single-container artifact format.

This publication contract applies to both single and repeated benchmark CLI runs. Lower-level serializers remain responsible for deterministic JSON generation, validation, reproducibility identity, integrity metadata, and runtime provenance; the CLI owns staging, publication ordering, overwrite policy, and rollback.
