"""Dependency-free observability primitives for experiment and agent execution."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import isfinite
from time import perf_counter
from typing import Mapping


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    """Immutable snapshot of counters and timing aggregates."""

    counters: Mapping[str, int]
    timing_seconds: Mapping[str, float]
    timing_counts: Mapping[str, int]


@dataclass(slots=True)
class MetricsRecorder:
    """Collect lightweight in-process counters and duration aggregates."""

    _counters: Counter[str] = field(default_factory=Counter, init=False, repr=False)
    _timing_seconds: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _timing_counts: Counter[str] = field(default_factory=Counter, init=False, repr=False)

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a named counter by a positive integer amount."""

        metric_name = _validate_metric_name(name)
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise TypeError("counter amount must be an integer")
        if amount <= 0:
            raise ValueError("counter amount must be positive")
        self._counters[metric_name] += amount

    def observe_duration(self, name: str, duration_seconds: float) -> None:
        """Record one finite, non-negative duration observation."""

        metric_name = _validate_metric_name(name)
        duration = float(duration_seconds)
        if not isfinite(duration) or duration < 0.0:
            raise ValueError("duration_seconds must be finite and non-negative")
        self._timing_seconds[metric_name] = self._timing_seconds.get(metric_name, 0.0) + duration
        self._timing_counts[metric_name] += 1

    def timer(self, name: str) -> MetricTimer:
        """Return a context manager that records elapsed wall-clock duration."""

        return MetricTimer(recorder=self, name=_validate_metric_name(name))

    def snapshot(self) -> MetricSnapshot:
        """Return an immutable copy suitable for logging or report persistence."""

        return MetricSnapshot(
            counters=dict(sorted(self._counters.items())),
            timing_seconds=dict(sorted(self._timing_seconds.items())),
            timing_counts=dict(sorted(self._timing_counts.items())),
        )


@dataclass(slots=True)
class MetricTimer:
    """Context manager that records one elapsed duration observation."""

    recorder: MetricsRecorder
    name: str
    _started_at: float | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> MetricTimer:
        if self._started_at is not None:
            raise RuntimeError("metric timer cannot be entered more than once")
        self._started_at = perf_counter()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._started_at is None:
            raise RuntimeError("metric timer was not started")
        self.recorder.observe_duration(self.name, perf_counter() - self._started_at)


def _validate_metric_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("metric name must be a string")
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("metric name must be non-empty")
    return normalized_name


__all__ = ["MetricSnapshot", "MetricTimer", "MetricsRecorder"]
