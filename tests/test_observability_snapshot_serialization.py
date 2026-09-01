"""Tests for deterministic observability snapshot serialization."""

from __future__ import annotations

from remem.observability import ObservationCollector


def test_snapshot_to_dict_sorts_metric_names() -> None:
    collector = ObservationCollector()
    collector.increment("zeta")
    collector.increment("alpha", 2.0)
    collector.observe_duration("zeta.seconds", 0.5)
    collector.observe_duration("alpha.seconds", 0.25)

    assert collector.snapshot().to_dict() == {
        "counters": {"alpha": 2.0, "zeta": 1.0},
        "durations_seconds": {"alpha.seconds": 0.25, "zeta.seconds": 0.5},
    }
