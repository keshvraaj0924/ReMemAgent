"""Dependency-free export boundary for observation snapshots.

Vendor SDKs remain caller-owned. ReMemAgent exports validated snapshots through a
small protocol so OpenTelemetry, Prometheus push gateways, logging systems, or
custom research infrastructure can be integrated without adding those SDKs to
the core package.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol

from remem.observability import (
    ObservationSnapshot,
    observation_snapshot_delta,
    write_observation_snapshot,
)


class ObservationExporter(Protocol):
    """Protocol implemented by external observation backends."""

    def export(self, snapshot: ObservationSnapshot) -> None:
        """Export one validated observation snapshot."""


class ObservationCheckpointStore(Protocol):
    """Persistence boundary for the last successfully exported cumulative snapshot."""

    def load(self) -> ObservationSnapshot:
        """Load the latest durable checkpoint."""

    def save(self, snapshot: ObservationSnapshot) -> None:
        """Persist the latest cumulative checkpoint."""


ObservationPayloadCallback = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class CallbackObservationExporter:
    """Adapt a caller-owned payload callback to the observation exporter protocol."""

    callback: ObservationPayloadCallback

    def export(self, snapshot: ObservationSnapshot) -> None:
        """Export a detached JSON-compatible snapshot payload through the callback."""

        if not callable(self.callback):
            raise TypeError("callback must be callable")
        self.callback(snapshot.to_dict())


@dataclass(frozen=True, slots=True)
class CompositeObservationExporter:
    """Fan out one snapshot to multiple exporters in deterministic order."""

    exporters: tuple[ObservationExporter, ...]

    def __init__(self, exporters: Sequence[ObservationExporter]) -> None:
        selected_exporters = tuple(exporters)
        if not selected_exporters:
            raise ValueError("exporters must contain at least one exporter")
        if any(not callable(getattr(exporter, "export", None)) for exporter in selected_exporters):
            raise TypeError("each exporter must provide a callable export method")
        object.__setattr__(self, "exporters", selected_exporters)

    def export(self, snapshot: ObservationSnapshot) -> None:
        """Export sequentially and preserve the first backend failure."""

        for exporter in self.exporters:
            exporter.export(snapshot)


@dataclass(frozen=True, slots=True)
class FileObservationCheckpointStore:
    """Atomically persist observation export checkpoints as versioned JSON."""

    path: Path

    def __init__(self, path: str | Path) -> None:
        object.__setattr__(self, "path", Path(path))

    def load(self) -> ObservationSnapshot:
        """Load and validate the checkpoint, or return an empty snapshot if absent."""

        if not self.path.exists():
            return ObservationSnapshot(counters={}, durations_seconds={})
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("observation checkpoint payload must be a mapping")
        return ObservationSnapshot.from_dict(payload)

    def save(self, snapshot: ObservationSnapshot) -> None:
        """Atomically replace the checkpoint with ``snapshot``."""

        write_observation_snapshot(self.path, snapshot)


class ObservationExportSession:
    """Export cumulative snapshots as retry-safe deltas.

    The session advances its in-memory checkpoint only after the configured exporter
    accepts the delta. When a checkpoint store is configured, the durable checkpoint
    is written after backend export and before in-memory advancement. This provides
    at-least-once delivery across process restarts: a checkpoint-write failure may
    cause a repeated delta, but cannot silently skip an uncheckpointed interval.
    """

    def __init__(
        self,
        exporter: ObservationExporter,
        *,
        initial_snapshot: ObservationSnapshot | None = None,
        checkpoint_store: ObservationCheckpointStore | None = None,
    ) -> None:
        if not callable(getattr(exporter, "export", None)):
            raise TypeError("exporter must provide a callable export method")
        if checkpoint_store is not None:
            if not callable(getattr(checkpoint_store, "load", None)) or not callable(
                getattr(checkpoint_store, "save", None)
            ):
                raise TypeError("checkpoint_store must provide callable load and save methods")
            if initial_snapshot is not None:
                raise ValueError("initial_snapshot and checkpoint_store are mutually exclusive")
            checkpoint = checkpoint_store.load()
        else:
            checkpoint = initial_snapshot or ObservationSnapshot(
                counters={}, durations_seconds={}
            )
        self._exporter = exporter
        self._checkpoint_store = checkpoint_store
        self._checkpoint = checkpoint
        self._lock = Lock()

    @property
    def checkpoint(self) -> ObservationSnapshot:
        """Return the last cumulative snapshot checkpointed successfully."""

        with self._lock:
            return ObservationSnapshot(
                counters=dict(self._checkpoint.counters),
                durations_seconds=dict(self._checkpoint.durations_seconds),
            )

    def export(self, current: ObservationSnapshot) -> ObservationSnapshot:
        """Export the delta from the last successful checkpoint to ``current``.

        Returns the exported delta. Empty deltas are valid and do not invoke the
        backend, avoiding unnecessary writes while still accepting an unchanged
        cumulative checkpoint.
        """

        with self._lock:
            delta = observation_snapshot_delta(self._checkpoint, current)
            if delta.counters or delta.durations_seconds:
                self._exporter.export(delta)
                if self._checkpoint_store is not None:
                    self._checkpoint_store.save(current)
            self._checkpoint = current
            return delta


def export_observation_snapshot(
    snapshot: ObservationSnapshot,
    exporters: Sequence[ObservationExporter],
) -> None:
    """Export one snapshot through an explicit ordered exporter sequence."""

    CompositeObservationExporter(exporters).export(snapshot)


__all__ = [
    "CallbackObservationExporter",
    "CompositeObservationExporter",
    "FileObservationCheckpointStore",
    "ObservationCheckpointStore",
    "ObservationExporter",
    "ObservationExportSession",
    "ObservationPayloadCallback",
    "export_observation_snapshot",
]
