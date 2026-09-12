from __future__ import annotations

from pathlib import Path

import pytest

from experiments.benchmark_distribution_config import (
    BENCHMARK_EPISODE_DURATION_METRIC,
    BenchmarkDistributionConfig,
)
from experiments.benchmark_observability_session import BenchmarkObservabilitySession
from remem.observability import ObservationCollector
from remem.observability_distribution_collector import DistributionObservationCollector


def _distribution_config(tmp_path: Path) -> BenchmarkDistributionConfig:
    return BenchmarkDistributionConfig(
        output_path=tmp_path / "durations.json",
        episode_duration_upper_bounds=(0.5, 1.0),
    )


def test_session_without_outputs_does_not_allocate_collector() -> None:
    session = BenchmarkObservabilitySession()

    assert session.create_collector() is None
    snapshots = session.freeze(None)
    assert snapshots.observation_snapshot is None
    assert snapshots.distribution_snapshot is None


def test_session_with_aggregate_output_uses_base_collector(tmp_path: Path) -> None:
    session = BenchmarkObservabilitySession(
        observability_output_path=tmp_path / "observability.json"
    )

    collector = session.create_collector()
    assert type(collector) is ObservationCollector
    assert collector is not None
    collector.observe_duration(BENCHMARK_EPISODE_DURATION_METRIC, 0.25)

    snapshots = session.freeze(collector)
    assert snapshots.observation_snapshot is not None
    assert snapshots.observation_snapshot.durations_seconds[
        BENCHMARK_EPISODE_DURATION_METRIC
    ] == pytest.approx(0.25)
    assert snapshots.distribution_snapshot is None


def test_session_with_distribution_uses_distribution_collector(tmp_path: Path) -> None:
    session = BenchmarkObservabilitySession(distribution_config=_distribution_config(tmp_path))

    collector = session.create_collector()
    assert isinstance(collector, DistributionObservationCollector)
    collector.observe_duration(BENCHMARK_EPISODE_DURATION_METRIC, 0.25)

    snapshots = session.freeze(collector)
    assert snapshots.observation_snapshot is None
    assert snapshots.distribution_snapshot is not None
    histogram = snapshots.distribution_snapshot.duration_histograms[
        BENCHMARK_EPISODE_DURATION_METRIC
    ]
    assert histogram.bucket_counts == (1, 0, 0)
    assert histogram.total == pytest.approx(0.25)


def test_session_shares_one_measurement_between_aggregate_and_distribution(tmp_path: Path) -> None:
    session = BenchmarkObservabilitySession(
        observability_output_path=tmp_path / "observability.json",
        distribution_config=_distribution_config(tmp_path),
    )

    collector = session.create_collector()
    assert isinstance(collector, DistributionObservationCollector)
    collector.observe_duration(BENCHMARK_EPISODE_DURATION_METRIC, 0.75)

    snapshots = session.freeze(collector)
    assert snapshots.observation_snapshot is not None
    assert snapshots.distribution_snapshot is not None
    aggregate_total = snapshots.observation_snapshot.durations_seconds[
        BENCHMARK_EPISODE_DURATION_METRIC
    ]
    histogram = snapshots.distribution_snapshot.duration_histograms[
        BENCHMARK_EPISODE_DURATION_METRIC
    ]
    assert aggregate_total == pytest.approx(histogram.total)
    assert histogram.bucket_counts == (0, 1, 0)


def test_session_rejects_missing_collector_for_configured_output(tmp_path: Path) -> None:
    session = BenchmarkObservabilitySession(
        observability_output_path=tmp_path / "observability.json"
    )

    with pytest.raises(ValueError, match="requires a collector"):
        session.freeze(None)


def test_session_rejects_base_collector_for_distribution_config(tmp_path: Path) -> None:
    session = BenchmarkObservabilitySession(distribution_config=_distribution_config(tmp_path))

    with pytest.raises(TypeError, match="DistributionObservationCollector"):
        session.freeze(ObservationCollector())
