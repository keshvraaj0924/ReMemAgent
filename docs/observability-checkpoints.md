# Durable observability export checkpoints

`ObservationExportSession` can persist its last successfully checkpointed cumulative snapshot through the `ObservationCheckpointStore` protocol. This allows an exporter process to restart without replaying every metric accumulated since process start.

`FileObservationCheckpointStore` is the built-in dependency-free implementation. It stores the same versioned `ObservationSnapshot` JSON schema used elsewhere in the framework and uses the existing atomic snapshot writer.

## Delivery semantics

The session exports the interval delta first, then persists the new cumulative checkpoint, and only then advances its in-memory checkpoint. This ordering is intentional: an exporter failure leaves the checkpoint unchanged, and a checkpoint-write failure also leaves the in-memory checkpoint unchanged.

The resulting contract is **at-least-once delivery**. If the external backend accepts a delta but checkpoint persistence fails, retrying can export that interval again. This is preferable to silently dropping observations. Backends that require exactly-once accounting should use an idempotency or transaction mechanism outside ReMemAgent.

## Restart behavior

Constructing a session with `checkpoint_store=...` loads the durable checkpoint immediately. A missing file represents an empty checkpoint. Invalid JSON, unsupported snapshot schema versions, or invalid metric aggregates fail closed during session construction.

`initial_snapshot` and `checkpoint_store` are mutually exclusive so there is one authoritative checkpoint source.

## Current limitations

The file store is intended for a single process owning one checkpoint file. ReMemAgent does not currently provide distributed locking, multi-writer coordination, remote checkpoint databases, or exactly-once coupling between an external telemetry backend and checkpoint persistence.
