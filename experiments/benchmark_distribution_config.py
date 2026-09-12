"""Configuration helpers for opt-in benchmark duration distributions.

This module keeps CLI-facing parsing and collector construction separate from
benchmark execution. It defines the fixed-bucket contract used by
``remem-benchmark`` without changing the stable benchmark report schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from remem.observability_distribution_artifacts import DistributionObservationSnapshot
from remem.observability_distribution_collector import DistributionObservationCollector
from remem.observability_distributions import ObservationHistogram

BENCHMARK_EPISODE_DURATION_METRIC = "benchmark.episode.duration_seconds"


@dataclass(frozen=True, slots=True)
class BenchmarkDistributionConfig:
    """Validated configuration for one benchmark duration-distribution sidecar."""

    output_path: Path
    episode_duration_upper_bounds: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.output_path, Path):
            raise TypeError("distribution output_path must be a pathlib.Path")
        # ObservationHistogram owns the canonical fixed-bucket validation contract.
        validated_bounds = (
            ObservationHistogram(self.episode_duration_upper_bounds).snapshot().upper_bounds
        )
        if not validated_bounds:
            raise ValueError("episode duration buckets must contain at least one upper bound")
        object.__setattr__(self, "episode_duration_upper_bounds", validated_bounds)

    def create_collector(self) -> DistributionObservationCollector:
        """Create a fresh collector using this immutable bucket contract."""

        return DistributionObservationCollector(
            {
                BENCHMARK_EPISODE_DURATION_METRIC: ObservationHistogram(
                    self.episode_duration_upper_bounds
                )
            }
        )

    def snapshot(
        self,
        collector: DistributionObservationCollector,
    ) -> DistributionObservationSnapshot:
        """Freeze one collector into the persisted distribution-sidecar schema."""

        if not isinstance(collector, DistributionObservationCollector):
            raise TypeError("collector must be a DistributionObservationCollector")
        histograms = collector.duration_distribution_snapshots()
        expected_bounds = self.episode_duration_upper_bounds
        snapshot = histograms.get(BENCHMARK_EPISODE_DURATION_METRIC)
        if snapshot is None:
            raise ValueError("collector is missing the benchmark episode duration histogram")
        if snapshot.upper_bounds != expected_bounds:
            raise ValueError("collector episode duration bucket contract does not match configuration")
        return DistributionObservationSnapshot(duration_histograms=histograms)


def build_benchmark_distribution_config(
    *,
    output_path: Path | None,
    episode_duration_buckets: str | None,
) -> BenchmarkDistributionConfig | None:
    """Build an opt-in distribution config from paired CLI-style values.

    The output path and bucket specification are intentionally a complete pair.
    Persisting an unconfigured sidecar or collecting a histogram without a
    destination would create ambiguous experiment evidence, so both cases fail
    before benchmark execution.
    """

    if output_path is None and episode_duration_buckets is None:
        return None
    if output_path is None:
        raise ValueError("--episode-duration-buckets requires --distribution-output")
    if episode_duration_buckets is None:
        raise ValueError("--distribution-output requires --episode-duration-buckets")
    return BenchmarkDistributionConfig(
        output_path=output_path,
        episode_duration_upper_bounds=parse_episode_duration_buckets(episode_duration_buckets),
    )


def parse_episode_duration_buckets(value: str) -> tuple[float, ...]:
    """Parse comma-separated finite non-negative strictly increasing seconds."""

    if not isinstance(value, str):
        raise TypeError("--episode-duration-buckets must be a string")
    parts = tuple(part.strip() for part in value.split(","))
    if not parts or any(not part for part in parts):
        raise ValueError("--episode-duration-buckets must contain comma-separated numbers")
    try:
        bounds = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--episode-duration-buckets must contain comma-separated numbers") from exc
    # Constructing a histogram applies the canonical finite/non-negative/order validation.
    validated_bounds = ObservationHistogram(bounds).snapshot().upper_bounds
    if not validated_bounds:
        raise ValueError("--episode-duration-buckets must contain at least one upper bound")
    return validated_bounds


__all__ = [
    "BENCHMARK_EPISODE_DURATION_METRIC",
    "BenchmarkDistributionConfig",
    "build_benchmark_distribution_config",
    "parse_episode_duration_buckets",
]
