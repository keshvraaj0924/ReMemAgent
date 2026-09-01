"""Low-dependency observability primitives for research execution traces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
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
        """Return a deterministic JSON-compatible representation.

        Keys are sorted at the serialization boundary so experiment artifacts
        do not depend on the order in which concurrent observations arrived.
        """

        return {
            "counters": dict(sorted(self.counters.items())),
            "durations_seconds": dict(sorted(self.durations_seconds.items())),
        }


class ObservationCollector:
    """Thread-safe in-process collector with deterministic snapshots.

    The collector deliberately stores only scalar counters and duration totals.
    It has no dependency on OpenTelemetry, Prometheus, or a logging backend, so
    research components can be instrumented without coupling the core to an
    operational stack.
    """

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
