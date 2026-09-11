# Paired artifact publication

Paired benchmark reports are written to a private temporary file in the destination directory and are published only after serialization, execution-order provenance injection, and file synchronization complete.

## No-overwrite contract

`save_paired_execution_result(..., overwrite=False)` fails closed when the destination already exists. The final publication step also enforces this rule atomically, so a report created by another process after an earlier CLI existence check is not silently replaced.

On supported local filesystems the writer publishes a new artifact by creating a hard link from the fully written temporary file to the final destination. Link creation fails atomically if the destination appeared concurrently. The private temporary name is then removed, leaving the final path attached to the already-synchronized bytes.

This closes the time-of-check/time-of-use window that exists when a long benchmark performs only a pre-run `Path.exists()` check and later uses unconditional replacement.

## Explicit overwrite

`overwrite=True` is an explicit request to replace an existing destination. In that mode publication uses `os.replace`, preserving the previous atomic-replacement behavior. The paired benchmark CLI forwards `--overwrite` to the persistence layer, so the command-line guard and the final filesystem operation enforce the same policy.

## Scope

This contract protects benchmark evidence from accidental concurrent replacement. It does not provide authenticity, distributed locking, remote-object-store semantics, or protection against an actor that intentionally modifies files after publication. Exact-byte integrity manifests remain the separate mechanism for detecting later artifact mutation.
