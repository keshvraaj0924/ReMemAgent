"""Low-dependency observability primitives for research execution traces."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from time import monotonic
from typing import Self


@dataclass(frozen=True, slots=True)
class ObservationEvent:
    """Immutable event emitted by a memory or integration boundary."""

    name: str
    value: float = 1.0
    attributes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ObservationSnapshot:
    """Point-in-time aggregate counters and latency totals."""

    counters: Mapping[str, float]
    durations_seconds: Mapping[str, float]

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "counters": dict(sorted(self.counters.items())),
            "durations_seconds": dict(sorted(self.durations_seconds.items())),
        }


class ObservationCollector:
    """Thread-safe in-process collector with deterministic snapshots."""

    def __init__(self) -> None:
        self._counters: dict[str, float] = {}
        self._durations_seconds: dict[str, float] = {}
        self._lock = Lock()

    def record(self, event: ObservationEvent) -> None:
        """Record one scalar observation event."""

        if not event.name:
            raise ValueError("event name must not be empty")
        if not isfinite(event.value):
            raise ValueError("event value must be finite")
        with self._lock:
            self._counters[event.name] = self._counters.get(event.name, 0.0) + event.value

    def increment(self, name: str, value: float = 1.0) -> None:
        """Increment a named counter."""

        self.record(ObservationEvent(name=name, value=value))

    def observe_duration(self, name: str, duration_seconds: float) -> None:
        """Add one measured duration to a named aggregate."""

        if not name:
            raise ValueError("duration name must not be empty")
        if not isfinite(duration_seconds) or duration_seconds < 0.0:
            raise ValueError("duration must be finite and non-negative")
        with self._lock:
            self._durations_seconds[name] = (
                self._durations_seconds.get(name, 0.0) + duration_seconds
            )

    def snapshot(self) -> ObservationSnapshot:
        """Return an isolated snapshot suitable for serialization or reporting."""

        with self._lock:
            return ObservationSnapshot(
                counters=dict(self._counters),
                durations_seconds=dict(self._durations_seconds),
            )

    def timed(self, name: str) -> ObservationTimer:
        """Create a monotonic timer that records its duration on exit."""

        return ObservationTimer(self, name)


class ObservationTimer:
    """Context manager for recording elapsed monotonic time."""

    def __init__(self, collector: ObservationCollector, name: str) -> None:
        if not name:
            raise ValueError("duration name must not be empty")
        self._collector = collector
        self._name = name
        self._started_at: float | None = None

    def __enter__(self) -> Self:
        self._started_at = monotonic()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._started_at is None:
            return
        self._collector.observe_duration(self._name, monotonic() - self._started_at)


def merge_observation_snapshots(
    snapshots: Sequence[ObservationSnapshot],
) -> ObservationSnapshot:
    """Combine independent snapshots without mutating any source mapping.

    This is intended for aggregating per-worker or per-process telemetry after
    execution. Counters and duration totals are additive; no event-level
    ordering or timestamp information is reconstructed.
    """

    counters: dict[str, float] = {}
    durations_seconds: dict[str, float] = {}
    for snapshot in snapshots:
        for name, value in snapshot.counters.items():
            _validate_snapshot_value(name, value, "counter")
            counters[name] = counters.get(name, 0.0) + value
        for name, value in snapshot.durations_seconds.items():
            _validate_snapshot_value(name, value, "duration")
            durations_seconds[name] = durations_seconds.get(name, 0.0) + value
    return ObservationSnapshot(counters=counters, durations_seconds=durations_seconds)


def _validate_snapshot_value(name: str, value: float, value_type: str) -> None:
    """Validate a persisted aggregate value before including it in a merge."""

    if not name:
        raise ValueError(f"{value_type} name must not be empty")
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{value_type} value must be finite and non-negative")


def write_observation_snapshot(path: str | Path, snapshot: ObservationSnapshot) -> None:
    """Atomically persist one deterministic observation snapshot as JSON.

    The destination is replaced only after the complete JSON document has been
    flushed to a temporary file in the same directory. This avoids leaving a
    partially written telemetry artifact when a process fails during a write.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        snapshot.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"

    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            os.replace(temporary_path, destination)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


__all__ = [
    "ObservationCollector",
    "ObservationEvent",
    "ObservationSnapshot",
    "ObservationTimer",
    "merge_observation_snapshots",
    "write_observation_snapshot",
]
