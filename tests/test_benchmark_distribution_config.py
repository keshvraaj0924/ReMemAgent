from __future__ import annotations

from pathlib import Path

import pytest

from experiments.benchmark_distribution_config import (
    BENCHMARK_EPISODE_DURATION_METRIC,
    BenchmarkDistributionConfig,
    build_benchmark_distribution_config,
    parse_episode_duration_buckets,
)
from remem.observability_distribution_collector import DistributionObservationCollector


def test_parse_episode_duration_buckets_normalizes_numeric_values() -> None:
    assert parse_episode_duration_buckets("0.1, 1, 2.5") == (0.1, 1.0, 2.5)


@pytest.mark.parametrize(
    "value",
    ["", "0.1,", ",0.1", "0.1,,1", "not-a-number"],
)
def test_parse_episode_duration_buckets_rejects_malformed_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_episode_duration_buckets(value)


@pytest.mark.parametrize(
    "value",
    ["-0.1,1", "nan,1", "inf,2", "1,1", "2,1"],
)
def test_parse_episode_duration_buckets_reuses_histogram_validation(value: str) -> None:
    with pytest.raises(ValueError):
        parse_episode_duration_buckets(value)


def test_build_distribution_config_requires_output_and_buckets_together(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires --distribution-output"):
        build_benchmark_distribution_config(
            output_path=None,
            episode_duration_buckets="0.5,1.0",
        )

    with pytest.raises(ValueError, match="requires --episode-duration-buckets"):
        build_benchmark_distribution_config(
            output_path=tmp_path / "distribution.json",
            episode_duration_buckets=None,
        )


def test_build_distribution_config_is_optional_when_both_values_are_omitted() -> None:
    assert (
        build_benchmark_distribution_config(
            output_path=None,
            episode_duration_buckets=None,
        )
        is None
    )


def test_distribution_config_collects_and_freezes_episode_durations(tmp_path: Path) -> None:
    config = BenchmarkDistributionConfig(
        output_path=tmp_path / "distribution.json",
        episode_duration_upper_bounds=(0.5, 1.0),
    )
    collector = config.create_collector()

    collector.observe_duration(BENCHMARK_EPISODE_DURATION_METRIC, 0.25)
    collector.observe_duration(BENCHMARK_EPISODE_DURATION_METRIC, 0.75)
    collector.observe_duration(BENCHMARK_EPISODE_DURATION_METRIC, 1.5)

    aggregate_snapshot = collector.snapshot()
    distribution_snapshot = config.snapshot(collector)
    histogram = distribution_snapshot.duration_histograms[BENCHMARK_EPISODE_DURATION_METRIC]

    assert aggregate_snapshot.durations_seconds[BENCHMARK_EPISODE_DURATION_METRIC] == pytest.approx(
        2.5
    )
    assert histogram.upper_bounds == (0.5, 1.0)
    assert histogram.bucket_counts == (1, 1, 1)
    assert histogram.count == 3
    assert histogram.total == pytest.approx(2.5)


def test_distribution_config_rejects_collector_with_different_bucket_contract(
    tmp_path: Path,
) -> None:
    config = BenchmarkDistributionConfig(
        output_path=tmp_path / "distribution.json",
        episode_duration_upper_bounds=(0.5, 1.0),
    )
    collector = BenchmarkDistributionConfig(
        output_path=tmp_path / "other.json",
        episode_duration_upper_bounds=(0.25, 1.0),
    ).create_collector()

    with pytest.raises(ValueError, match="bucket contract does not match"):
        config.snapshot(collector)


def test_distribution_config_requires_distribution_collector(tmp_path: Path) -> None:
    config = BenchmarkDistributionConfig(
        output_path=tmp_path / "distribution.json",
        episode_duration_upper_bounds=(0.5,),
    )

    with pytest.raises(TypeError, match="DistributionObservationCollector"):
        config.snapshot(object())  # type: ignore[arg-type]


def test_distribution_config_creates_fresh_collectors(tmp_path: Path) -> None:
    config = BenchmarkDistributionConfig(
        output_path=tmp_path / "distribution.json",
        episode_duration_upper_bounds=(0.5, 1.0),
    )

    first = config.create_collector()
    second = config.create_collector()

    assert isinstance(first, DistributionObservationCollector)
    assert isinstance(second, DistributionObservationCollector)
    assert first is not second
    first.observe_duration(BENCHMARK_EPISODE_DURATION_METRIC, 0.25)
    assert second.duration_distribution_snapshots()[BENCHMARK_EPISODE_DURATION_METRIC].count == 0
