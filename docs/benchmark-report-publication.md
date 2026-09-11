# Benchmark artifact publication

The general benchmark CLI publishes its report, optional integrity manifest, and optional observability snapshot through private temporary files in each destination directory.

By default, each final publication boundary is **no-overwrite and race-safe**. The CLI still performs early existence checks so obvious conflicts fail before measured execution, but correctness does not depend on those checks. After each artifact is fully written and synced, publication uses an atomic hard-link operation. If another process creates a requested destination after the precheck but before publication, ReMemAgent raises `FileExistsError`, preserves the competing artifact, and removes its private temporary file.

When `--overwrite` is explicitly supplied, publication uses `os.replace()` so the completed temporary artifact atomically replaces the destination. The lower-level manifest and observability persistence APIs retain overwrite-by-default behavior for backward compatibility, while the benchmark CLI explicitly forwards its `--overwrite` policy to them.

This closes the check-then-replace races that previously remained for the optional manifest and observability snapshot after report publication had already been hardened.

## Transaction boundary

The report, manifest, and observability snapshot are independently atomic files; they are **not** committed as one transactional multi-file bundle. A failure after report publication can therefore leave a valid report without one or both optional companion artifacts. Callers that require all-or-nothing bundle semantics should treat this as a current limitation rather than inferring completeness from the presence of the report alone.

The publication contract applies to both single and repeated benchmark CLI runs. Lower-level serializers remain responsible for deterministic JSON generation, validation, reproducibility identity, integrity metadata, and runtime provenance; the CLI owns whether a completed artifact may claim its requested destination path.
