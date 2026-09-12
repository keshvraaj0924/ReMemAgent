from types import MappingProxyType

import pytest

from remem.benchmark import BenchmarkSuiteRunner
from remem.observability_distribution_collector import DistributionObservationCollector
from remem.observability_distributions import ObservationHistogram
from tests.test_benchmark import FakeEnvironment


def test_configured_duration_records_same_value_in_aggregate_and_histogram() -> None:
    histogram = ObservationHistogram((0.1, 1.0))
    collector = DistributionObservationCollector({"operation.duration": histogram})

    collector.observe_duration(" operation.duration ", 0.25)

    aggregate = collector.snapshot()
    distributions = collector.duration_distribution_snapshots()
    snapshot = distributions["operation.duration"]
    assert aggregate.durations_seconds["operation.duration"] == 0.25
    assert snapshot.count == 1
    assert snapshot.total == 0.25
    assert snapshot.bucket_counts == (0, 1, 0)


def test_unconfigured_duration_only_records_aggregate() -> None:
    collector = DistributionObservationCollector()

    collector.observe_duration("operation.duration", 0.25)

    assert collector.snapshot().durations_seconds["operation.duration"] == 0.25
    assert collector.duration_distribution_snapshots() == MappingProxyType({})


def test_histogram_registration_rejects_normalized_name_collisions() -> None:
    first = ObservationHistogram((1.0,))
    second = ObservationHistogram((1.0,))

    with pytest.raises(ValueError, match="normalize to the same metric"):
        DistributionObservationCollector(
            {
                "benchmark.duration": first,
                " benchmark.duration ": second,
            }
        )


def test_histogram_registration_rejects_non_histogram_values() -> None:
    with pytest.raises(TypeError, match="ObservationHistogram"):
        DistributionObservationCollector({"benchmark.duration": object()})  # type: ignore[arg-type]


def test_benchmark_episode_timing_populates_duration_distribution() -> None:
    histogram = ObservationHistogram((60.0,))
    collector = DistributionObservationCollector(
        {"benchmark.episode.duration_seconds": histogram}
    )

    BenchmarkSuiteRunner(observation_collector=collector).run(
        benchmark_name="distribution-observability-smoke",
        episode_count=2,
        max_steps=1,
        environment_factory=lambda index: FakeEnvironment(index),
        policy_factory=lambda index, store: lambda state: "act",
        success_evaluator=lambda episode: True,
    )

    aggregate = collector.snapshot()
    distribution = collector.duration_distribution_snapshots()[
        "benchmark.episode.duration_seconds"
    ]
    assert distribution.count == 2
    assert distribution.total == aggregate.durations_seconds[
        "benchmark.episode.duration_seconds"
    ]
    assert sum(distribution.bucket_counts) == 2
