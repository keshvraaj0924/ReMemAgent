"""Tests for deterministic observability primitives."""

from __future__ import annotations

import pytest

from remem.observability import ObservationCollector, ObservationEvent


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


def test_invalid_events_are_rejected() -> None:
    collector = ObservationCollector()

    with pytest.raises(ValueError, match="event name"):
        collector.record(ObservationEvent(name=""))

    with pytest.raises(ValueError, match="finite"):
        collector.increment("invalid", float("inf"))

    with pytest.raises(ValueError, match="non-negative"):
        collector.observe_duration("invalid", -1.0)
