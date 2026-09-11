# Benchmark report publication

The general benchmark CLI publishes single-run and repeated-run JSON reports through a private temporary file in the destination directory.

By default, final publication is **no-overwrite and race-safe**. The CLI still performs an early existence check so obvious conflicts fail before measured execution, but correctness does not depend on that check. After the benchmark completes, publication uses an atomic hard-link operation. If another process created the destination while the benchmark was running, publication fails with `FileExistsError`, preserves the competing artifact, and removes the private temporary file.

When `--overwrite` is explicitly supplied, publication uses `os.replace()` so the completed temporary report atomically replaces the destination.

This closes the check-then-replace race that existed when `--output` was validated before measurement but the final report writer unconditionally replaced the path afterward. It does not make the optional manifest or observability snapshot a transactional multi-file bundle; those artifacts are still written after the report and should be treated as separate persistence boundaries.

The publication contract applies to the CLI execution path for both single and repeated benchmark runs. Lower-level report serialization remains responsible for deterministic JSON generation, validation, reproducibility identity, and runtime provenance; the CLI owns whether the completed report may claim the requested destination path.
