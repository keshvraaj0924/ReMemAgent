"""Deterministic histogram primitives for research observability."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from threading import Lock
from types import MappingProxyType
from typing import Self


@dataclass(frozen=True, slots=True)
class ObservationHistogramSnapshot:
    """Immutable histogram snapshot using fixed finite upper bounds.

    ``bucket_counts`` contains one bucket per configured upper bound plus a final
    overflow bucket. Buckets are non-cumulative to make merging exact and to
    avoid losing information during serialization or aggregation.
    """

    upper_bounds: tuple[float, ...]
    bucket_counts: tuple[int, ...]
    total: float

    def __post_init__(self) -> None:
        bounds = _validate_upper_bounds(self.upper_bounds)
        counts = _validate_bucket_counts(self.bucket_counts, len(bounds) + 1)
        total = _validate_total(self.total, sum(counts))
        object.__setattr__(self, "upper_bounds", bounds)
        object.__setattr__(self, "bucket_counts", counts)
        object.__setattr__(self, "total", total)

    @property
    def count(self) -> int:
        """Return the total number of observations represented by the snapshot."""

        return sum(self.bucket_counts)

    @property
    def mean(self) -> float | None:
        """Return the arithmetic mean, or ``None`` when the histogram is empty."""

        if self.count == 0:
            return None
        return self.total / self.count

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "upper_bounds": list(self.upper_bounds),
            "bucket_counts": list(self.bucket_counts),
            "count": self.count,
            "total": self.total,
            "mean": self.mean,
        }


class ObservationHistogram:
    """Thread-safe fixed-bucket histogram for non-negative scalar observations."""

    def __init__(self, upper_bounds: Sequence[float]) -> None:
        self._upper_bounds = _validate_upper_bounds(tuple(upper_bounds))
        self._bucket_counts = [0] * (len(self._upper_bounds) + 1)
        self._total = 0.0
        self._lock = Lock()

    def observe(self, value: float) -> None:
        """Record one finite non-negative value."""

        numeric_value = _validate_observation(value)
        bucket_index = _bucket_index(self._upper_bounds, numeric_value)
        with self._lock:
            self._bucket_counts[bucket_index] += 1
            self._total += numeric_value

    def snapshot(self) -> ObservationHistogramSnapshot:
        """Return an isolated immutable snapshot."""

        with self._lock:
            return ObservationHistogramSnapshot(
                upper_bounds=self._upper_bounds,
                bucket_counts=tuple(self._bucket_counts),
                total=self._total,
            )


def merge_histogram_snapshots(
    snapshots: Sequence[ObservationHistogramSnapshot],
) -> ObservationHistogramSnapshot:
    """Merge compatible fixed-bucket histograms exactly.

    Empty input is rejected because no bucket contract can be inferred. All
    snapshots must use identical upper bounds; silently rebucketing research
    measurements would make comparisons ambiguous.
    """

    if not snapshots:
        raise ValueError("at least one histogram snapshot is required")

    expected_bounds = snapshots[0].upper_bounds
    merged_counts = [0] * (len(expected_bounds) + 1)
    total = 0.0
    for snapshot in snapshots:
        if snapshot.upper_bounds != expected_bounds:
            raise ValueError("histogram upper bounds must match exactly")
        for index, count in enumerate(snapshot.bucket_counts):
            merged_counts[index] += count
        total += snapshot.total

    return ObservationHistogramSnapshot(
        upper_bounds=expected_bounds,
        bucket_counts=tuple(merged_counts),
        total=total,
    )


def histogram_snapshot_delta(
    previous: ObservationHistogramSnapshot,
    current: ObservationHistogramSnapshot,
) -> ObservationHistogramSnapshot:
    """Return interval observations between cumulative histogram snapshots."""

    if previous.upper_bounds != current.upper_bounds:
        raise ValueError("histogram upper bounds must match exactly")

    delta_counts: list[int] = []
    for previous_count, current_count in zip(
        previous.bucket_counts,
        current.bucket_counts,
        strict=True,
    ):
        if current_count < previous_count:
            raise ValueError("histogram bucket count regressed")
        delta_counts.append(current_count - previous_count)

    total = current.total - previous.total
    if total < 0.0:
        raise ValueError("histogram total regressed")

    return ObservationHistogramSnapshot(
        upper_bounds=current.upper_bounds,
        bucket_counts=tuple(delta_counts),
        total=total,
    )


def _validate_upper_bounds(upper_bounds: Sequence[float]) -> tuple[float, ...]:
    """Validate and normalize a fixed histogram bucket contract."""

    normalized: list[float] = []
    previous: float | None = None
    for raw_bound in upper_bounds:
        if isinstance(raw_bound, bool) or not isinstance(raw_bound, (int, float)):
            raise TypeError("histogram upper bounds must be numbers")
        bound = float(raw_bound)
        if not isfinite(bound) or bound < 0.0:
            raise ValueError("histogram upper bounds must be finite and non-negative")
        if previous is not None and bound <= previous:
            raise ValueError("histogram upper bounds must be strictly increasing")
        normalized.append(bound)
        previous = bound
    return tuple(normalized)


def _validate_bucket_counts(bucket_counts: Sequence[int], expected_length: int) -> tuple[int, ...]:
    """Validate exact per-bucket counts for one snapshot."""

    if len(bucket_counts) != expected_length:
        raise ValueError("histogram bucket count length does not match upper bounds")
    normalized: list[int] = []
    for count in bucket_counts:
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("histogram bucket counts must be integers")
        if count < 0:
            raise ValueError("histogram bucket counts must be non-negative")
        normalized.append(count)
    return tuple(normalized)


def _validate_total(total: float, count: int) -> float:
    """Validate aggregate total consistency without inventing observations."""

    if isinstance(total, bool) or not isinstance(total, (int, float)):
        raise TypeError("histogram total must be a number")
    normalized = float(total)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError("histogram total must be finite and non-negative")
    if count == 0 and normalized != 0.0:
        raise ValueError("empty histogram must have a zero total")
    return normalized


def _validate_observation(value: float) -> float:
    """Validate one observed scalar."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("histogram observation must be a number")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError("histogram observation must be finite and non-negative")
    return normalized


def _bucket_index(upper_bounds: tuple[float, ...], value: float) -> int:
    """Return the first matching upper-bound bucket, or the overflow bucket."""

    for index, upper_bound in enumerate(upper_bounds):
        if value <= upper_bound:
            return index
    return len(upper_bounds)


__all__ = [
    "ObservationHistogram",
    "ObservationHistogramSnapshot",
    "histogram_snapshot_delta",
    "merge_histogram_snapshots",
]
