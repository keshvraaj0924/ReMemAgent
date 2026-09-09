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


OBSERVATION_SNAPSHOT_SCHEMA_VERSION = 1


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

    def to_dict(self) -> dict[str, object]:
        """Return a versioned, deterministic JSON-compatible representation."""

        return {
            "schema_version": OBSERVATION_SNAPSHOT_SCHEMA_VERSION,
            "counters": dict(sorted(self.counters.items())),
            "durations_seconds": dict(sorted(self.durations_seconds.items())),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Validate and reconstruct a snapshot loaded from persisted JSON."""

        if not isinstance(payload, Mapping):
            raise TypeError("observation snapshot must be a mapping")
        schema_version = payload.get("schema_version")
        if schema_version != OBSERVATION_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported observation snapshot schema version: {schema_version!r}"
            )
        counters = _parse_aggregate_mapping(payload.get("counters"), "counters")
        durations = _parse_aggregate_mapping(
            payload.get("durations_seconds"), "durations_seconds"
        )
        return cls(counters=counters, durations_seconds=durations)


class ObservationCollector:
    """Thread-safe in-process collector with deterministic snapshots."""

    def __init__(self) -> None:
        self._counters: dict[str, float] = {}
        self._durations_seconds: dict[str, float] = {}
        self._lock = Lock()

    def record(self, event: ObservationEvent) -> None:
        """Record one scalar observation event."""

        name = _normalize_metric_name(event.name, "event name")
        if not isfinite(event.value):
            raise ValueError("event value must be finite")
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + event.value

    def increment(self, name: str, value: float = 1.0) -> None:
        """Increment a named counter."""

        self.record(ObservationEvent(name=name, value=value))

    def record_outcome(self, name: str, succeeded: bool) -> None:
        """Record a mutually exclusive success or failure outcome."""

        normalized_name = _normalize_metric_name(name, "outcome name")
        suffix = "succeeded" if succeeded else "failed"
        self.increment(f"{normalized_name}.{suffix}")

    def observe_duration(self, name: str, duration_seconds: float) -> None:
        """Add one measured duration to a named aggregate."""

        normalized_name = _normalize_metric_name(name, "duration name")
        if not isfinite(duration_seconds) or duration_seconds < 0.0:
            raise ValueError("duration must be finite and non-negative")
        with self._lock:
            self._durations_seconds[normalized_name] = (
                self._durations_seconds.get(normalized_name, 0.0) + duration_seconds
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

    def observed(self, name: str) -> ObservationOperation:
        """Create an operation scope that records duration and outcome."""

        return ObservationOperation(self, name)


class ObservationTimer:
    """Context manager for recording elapsed monotonic time."""

    def __init__(self, collector: ObservationCollector, name: str) -> None:
        self._collector = collector
        self._name = _normalize_metric_name(name, "duration name")
        self._started_at: float | None = None

    def __enter__(self) -> Self:
        self._started_at = monotonic()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._started_at is None:
            return
        self._collector.observe_duration(self._name, monotonic() - self._started_at)


class ObservationOperation:
    """Context manager for timing and outcome accounting of one operation."""

    def __init__(self, collector: ObservationCollector, name: str) -> None:
        self._collector = collector
        self._name = _normalize_metric_name(name, "operation name")
        self._started_at: float | None = None

    def __enter__(self) -> Self:
        self._started_at = monotonic()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        if self._started_at is None:
            return
        self._collector.observe_duration(self._name, monotonic() - self._started_at)
        self._collector.record_outcome(self._name, succeeded=exc_type is None)


def merge_observation_snapshots(
    snapshots: Sequence[ObservationSnapshot],
) -> ObservationSnapshot:
    """Combine independent snapshots without mutating any source mapping."""

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


def _parse_aggregate_mapping(value: object, field_name: str) -> dict[str, float]:
    """Parse and validate one persisted aggregate mapping."""

    if not isinstance(value, Mapping):
        raise TypeError(f"observation snapshot {field_name} must be a mapping")
    parsed: dict[str, float] = {}
    for name, aggregate in value.items():
        if not isinstance(name, str):
            raise TypeError(f"observation snapshot {field_name} names must be strings")
        if not isinstance(aggregate, (int, float)) or isinstance(aggregate, bool):
            raise TypeError(f"observation snapshot {field_name} values must be numbers")
        _validate_snapshot_value(name, float(aggregate), field_name)
        parsed[name] = float(aggregate)
    return dict(sorted(parsed.items()))


def _validate_snapshot_value(name: str, value: float, value_type: str) -> None:
    """Validate a persisted aggregate value before including it in a merge."""

    _normalize_metric_name(name, f"{value_type} name")
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{value_type} value must be finite and non-negative")


def _normalize_metric_name(name: str, field_name: str) -> str:
    """Normalize and validate a metric name at every collection boundary."""

    if not isinstance(name, str):
        raise TypeError(f"{field_name} must be a string")
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError(f"{field_name} must not be empty")
    return normalized_name


def write_observation_snapshot(path: str | Path, snapshot: ObservationSnapshot) -> None:
    """Atomically persist one deterministic observation snapshot as JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            snapshot.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )

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
    "OBSERVATION_SNAPSHOT_SCHEMA_VERSION",
    "ObservationCollector",
    "ObservationEvent",
    "ObservationOperation",
    "ObservationSnapshot",
    "merge_observation_snapshots",
    "write_observation_snapshot",
]
