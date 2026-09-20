"""Dependency-free observability primitives for experiment and agent execution."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from time import perf_counter
from types import MappingProxyType
from typing import Self, TypeVar

MetricValue = TypeVar("MetricValue", int, float)
_REQUIRED_SNAPSHOT_FIELDS = frozenset({"counters", "timing_seconds", "timing_counts"})


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    """Immutable snapshot of counters and timing aggregates."""

    counters: Mapping[str, int]
    timing_seconds: Mapping[str, float]
    timing_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, dict[str, int | float]]:
        """Return a deterministic JSON-serializable representation."""

        return {
            "counters": dict(sorted(self.counters.items())),
            "timing_seconds": dict(sorted(self.timing_seconds.items())),
            "timing_counts": dict(sorted(self.timing_counts.items())),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Restore and validate a snapshot from persisted JSON-like data.

        Validation is intentionally strict so malformed or partially written
        observability artifacts cannot silently contaminate aggregated metrics.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("metric snapshot payload must be a mapping")
        payload_fields = set(payload)
        if payload_fields != _REQUIRED_SNAPSHOT_FIELDS:
            missing_fields = sorted(_REQUIRED_SNAPSHOT_FIELDS - payload_fields)
            unknown_fields = sorted(payload_fields - _REQUIRED_SNAPSHOT_FIELDS)
            raise ValueError(
                "metric snapshot fields do not match schema: "
                f"missing={missing_fields}, unknown={unknown_fields}"
            )

        counters = _require_mapping(payload["counters"], "counters")
        timing_seconds = _require_mapping(payload["timing_seconds"], "timing_seconds")
        timing_counts = _require_mapping(payload["timing_counts"], "timing_counts")

        candidate = cls(
            counters=counters,  # type: ignore[arg-type]
            timing_seconds=timing_seconds,  # type: ignore[arg-type]
            timing_counts=timing_counts,  # type: ignore[arg-type]
        )
        recorder = MetricsRecorder()
        recorder.merge(candidate)
        return recorder.snapshot()


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

    def merge(self, snapshot: MetricSnapshot) -> None:
        """Merge a validated snapshot into this recorder.

        This supports deterministic aggregation of metrics collected by separate
        workers without coupling the core research framework to a telemetry SDK.
        """

        if not isinstance(snapshot, MetricSnapshot):
            raise TypeError("snapshot must be a MetricSnapshot")

        for name, amount in snapshot.counters.items():
            self.increment(name, amount)

        timing_names = set(snapshot.timing_seconds) | set(snapshot.timing_counts)
        if set(snapshot.timing_seconds) != set(snapshot.timing_counts):
            raise ValueError("timing_seconds and timing_counts must contain the same metric names")

        for name in sorted(timing_names):
            metric_name = _validate_metric_name(name)
            duration = snapshot.timing_seconds[name]
            count = snapshot.timing_counts[name]
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError("timing count must be an integer")
            if count <= 0:
                raise ValueError("timing count must be positive")
            if isinstance(duration, bool) or not isinstance(duration, (int, float)):
                raise TypeError("timing duration must be numeric")
            total_duration = float(duration)
            if not isfinite(total_duration) or total_duration < 0.0:
                raise ValueError("timing duration must be finite and non-negative")
            self._timing_seconds[metric_name] = (
                self._timing_seconds.get(metric_name, 0.0) + total_duration
            )
            self._timing_counts[metric_name] += count

    def timer(self, name: str) -> "MetricTimer":
        """Return a context manager that records elapsed wall-clock duration."""

        return MetricTimer(recorder=self, name=_validate_metric_name(name))

    def snapshot(self) -> MetricSnapshot:
        """Return a deeply immutable copy suitable for persistence or logging."""

        return MetricSnapshot(
            counters=_immutable_mapping(self._counters),
            timing_seconds=_immutable_mapping(self._timing_seconds),
            timing_counts=_immutable_mapping(self._timing_counts),
        )


@dataclass(slots=True)
class MetricTimer:
    """Context manager that records one elapsed duration observation."""

    recorder: MetricsRecorder
    name: str
    _started_at: float | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> Self:
        if self._started_at is not None:
            raise RuntimeError("metric timer cannot be entered more than once")
        self._started_at = perf_counter()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._started_at is None:
            raise RuntimeError("metric timer was not started")
        self.recorder.observe_duration(self.name, perf_counter() - self._started_at)


def _immutable_mapping(values: Mapping[str, MetricValue]) -> Mapping[str, MetricValue]:
    """Return a deterministic read-only copy of metric values."""

    return MappingProxyType(dict(sorted(values.items())))


def _require_mapping(value: object, field_name: str) -> Mapping[object, object]:
    """Require a mapping-valued snapshot field before semantic validation."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _validate_metric_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("metric name must be a string")
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("metric name must be non-empty")
    return normalized_name


__all__ = ["MetricSnapshot", "MetricTimer", "MetricsRecorder"]
