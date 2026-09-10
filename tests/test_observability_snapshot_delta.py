"""Regression tests for cumulative observation snapshot delta semantics."""

from __future__ import annotations

import pytest

from remem.observability import ObservationSnapshot, observation_snapshot_delta


def test_observation_snapshot_delta_returns_interval_increments() -> None:
    previous = ObservationSnapshot(
        counters={"benchmark.runs": 2.0, "benchmark.failures": 1.0},
        durations_seconds={"benchmark.seconds": 1.25},
    )
    current = ObservationSnapshot(
        counters={
            "benchmark.runs": 5.0,
            "benchmark.failures": 1.0,
            "memory.retrievals": 4.0,
        },
        durations_seconds={"benchmark.seconds": 2.0, "memory.seconds": 0.5},
    )

    delta = observation_snapshot_delta(previous, current)

    assert delta.counters == {"benchmark.runs": 3.0, "memory.retrievals": 4.0}
    assert delta.durations_seconds == {"benchmark.seconds": 0.75, "memory.seconds": 0.5}


def test_observation_snapshot_delta_does_not_mutate_source_snapshots() -> None:
    previous = ObservationSnapshot(counters={"runs": 1.0}, durations_seconds={})
    current = ObservationSnapshot(counters={"runs": 3.0}, durations_seconds={})

    delta = observation_snapshot_delta(previous, current)

    assert delta.counters == {"runs": 2.0}
    assert previous.counters == {"runs": 1.0}
    assert current.counters == {"runs": 3.0}


def test_observation_snapshot_delta_rejects_counter_regression() -> None:
    previous = ObservationSnapshot(counters={"runs": 3.0}, durations_seconds={})
    current = ObservationSnapshot(counters={"runs": 2.0}, durations_seconds={})

    with pytest.raises(ValueError, match="counter aggregate regressed.*'runs'"):
        observation_snapshot_delta(previous, current)


def test_observation_snapshot_delta_rejects_disappearing_metric() -> None:
    previous = ObservationSnapshot(counters={"runs": 1.0}, durations_seconds={})
    current = ObservationSnapshot(counters={}, durations_seconds={})

    with pytest.raises(ValueError, match="counter aggregate regressed.*'runs'"):
        observation_snapshot_delta(previous, current)


def test_observation_snapshot_delta_rejects_duration_regression() -> None:
    previous = ObservationSnapshot(counters={}, durations_seconds={"latency": 0.4})
    current = ObservationSnapshot(counters={}, durations_seconds={"latency": 0.3})

    with pytest.raises(ValueError, match="duration aggregate regressed.*'latency'"):
        observation_snapshot_delta(previous, current)


def test_observation_snapshot_delta_of_equal_snapshots_is_empty() -> None:
    snapshot = ObservationSnapshot(
        counters={"runs": 2.0},
        durations_seconds={"latency": 0.75},
    )

    delta = observation_snapshot_delta(snapshot, snapshot)

    assert delta.counters == {}
    assert delta.durations_seconds == {}
