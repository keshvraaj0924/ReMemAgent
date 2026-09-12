"""Distribution-aware extension of the core observability collector.

The base :class:`~remem.observability.ObservationCollector` intentionally keeps
its stable snapshot contract limited to counters and aggregate duration totals.
This module layers fixed-bucket duration distributions on top without changing
that schema or forcing distribution storage on callers that do not need it.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from remem.observability import ObservationCollector
from remem.observability_distributions import (
    ObservationHistogram,
    ObservationHistogramSnapshot,
)


class DistributionObservationCollector(ObservationCollector):
    """Collect aggregate durations and selected fixed-bucket distributions.

    Histogram registration is explicit by metric name. Every configured duration
    observation is first accepted by the base collector and then recorded in the
    corresponding histogram, so aggregate and distribution measurements share
    the exact same measured value.
    """

    def __init__(
        self,
        duration_histograms: Mapping[str, ObservationHistogram] | None = None,
    ) -> None:
        super().__init__()
        self._duration_histograms = MappingProxyType(
            _normalize_duration_histograms(duration_histograms or {})
        )

    def observe_duration(self, name: str, duration_seconds: float) -> None:
        """Record one duration in aggregate and, when configured, its histogram."""

        normalized_name = _normalize_metric_name(name)
        super().observe_duration(normalized_name, duration_seconds)
        histogram = self._duration_histograms.get(normalized_name)
        if histogram is not None:
            histogram.observe(duration_seconds)

    def duration_distribution_snapshots(
        self,
    ) -> Mapping[str, ObservationHistogramSnapshot]:
        """Return deterministic immutable snapshots of configured histograms."""

        return MappingProxyType(
            {
                name: self._duration_histograms[name].snapshot()
                for name in sorted(self._duration_histograms)
            }
        )


def _normalize_duration_histograms(
    duration_histograms: Mapping[str, ObservationHistogram],
) -> dict[str, ObservationHistogram]:
    """Validate histogram registrations and reject normalized-name collisions."""

    normalized: dict[str, ObservationHistogram] = {}
    for name, histogram in duration_histograms.items():
        normalized_name = _normalize_metric_name(name)
        if normalized_name in normalized:
            raise ValueError(
                "duration histogram names normalize to the same metric: "
                f"{name!r}"
            )
        if not isinstance(histogram, ObservationHistogram):
            raise TypeError("duration histogram values must be ObservationHistogram instances")
        normalized[normalized_name] = histogram
    return normalized


def _normalize_metric_name(name: str) -> str:
    """Normalize one metric name consistently with the base collector contract."""

    if not isinstance(name, str):
        raise TypeError("duration histogram name must be a string")
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("duration histogram name must not be empty")
    return normalized_name


__all__ = ["DistributionObservationCollector"]
