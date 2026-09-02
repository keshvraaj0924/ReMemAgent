"""Tests for deterministic observability primitives."""

from __future__ import annotations

import pytest

from remem.observability import (
    ObservationCollector,
    ObservationEvent,
    ObservationSnapshot,
    merge_observation_snapshots,
)


def test_snapshot_isolated_from_future_mutation() -> None:
    collector = ObservationCollector()
    collector.increment("retrieval.calls")
    collector.increment("retrieval.calls", 2.0)

    snapshot = collector.snapshot()
    collector.increment("retrieval.calls")

    assert snapshot.counters == {"retrieval.calls": 3.0}
    assert collector.snapshot().counters == {"retrieval.calls": 4.0}


def test_duration_context_records_elapsed_time() -> None:
    collector = ObservationCollector()

    with collector.timed("routing.seconds"):
        pass

    snapshot = collector.snapshot()
    assert snapshot.durations_seconds["routing.seconds"] >= 0.0


def test_observed_scope_records_success_and_duration() -> None:
    collector = ObservationCollector()

    with collector.observed("retrieval"):
        pass

    snapshot = collector.snapshot()
    assert snapshot.counters == {"retrieval.succeeded": 1.0}
    assert snapshot.durations_seconds["retrieval"] >= 0.0


def test_observed_scope_records_failure_and_preserves_exception() -> None:
    collector = ObservationCollector()

    with pytest.raises(RuntimeError, match="boom"):
        with collector.observed("policy"):
            raise RuntimeError("boom")

    snapshot = collector.snapshot()
    assert snapshot.counters == {"policy.failed": 1.0}
    assert snapshot.durations_seconds["policy"] >= 0.0


def test_observed_scope_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="operation name"):
        ObservationCollector().observed("   ")


def test_invalid_events_are_rejected() -> None:
    collector = ObservationCollector()

    with pytest.raises(ValueError, match="event name"):
        collector.record(ObservationEvent(name=""))

    with pytest.raises(ValueError, match="finite"):
        collector.increment("invalid", float("inf"))

    with pytest.raises(ValueError, match="non-negative"):
        collector.observe_duration("invalid", -1.0)


def test_record_outcome_tracks_mutually_exclusive_results() -> None:
    collector = ObservationCollector()

    collector.record_outcome("benchmark.episode", True)
    collector.record_outcome("benchmark.episode", False)

    assert collector.snapshot().counters == {
        "benchmark.episode.failed": 1.0,
        "benchmark.episode.succeeded": 1.0,
    }


def test_record_outcome_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="outcome name"):
        ObservationCollector().record_outcome("   ", True)


def test_merge_observation_snapshots_adds_workers_without_mutating_inputs() -> None:
    first = ObservationSnapshot(
        counters={"retrieval.calls": 2.0},
        durations_seconds={"routing.seconds": 1.5},
    )
    second = ObservationSnapshot(
        counters={"retrieval.calls": 3.0, "memory.writes": 1.0},
        durations_seconds={"routing.seconds": 0.5},
    )

    merged = merge_observation_snapshots([first, second])

    assert merged.counters == {"memory.writes": 1.0, "retrieval.calls": 5.0}
    assert merged.durations_seconds == {"routing.seconds": 2.0}
    assert first.counters == {"retrieval.calls": 2.0}
    assert second.counters["retrieval.calls"] == 3.0


def test_merge_observation_snapshots_rejects_invalid_values() -> None:
    invalid = ObservationSnapshot(
        counters={"bad": -1.0},
        durations_seconds={},
    )

    with pytest.raises(ValueError, match="counter value"):
        merge_observation_snapshots([invalid])
