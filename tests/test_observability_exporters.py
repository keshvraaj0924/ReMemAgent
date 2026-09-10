"""Tests for the dependency-free observability exporter boundary."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from remem.observability import ObservationSnapshot
from remem.observability_exporters import (
    CallbackObservationExporter,
    CompositeObservationExporter,
    export_observation_snapshot,
)


class RecordingExporter:
    """Minimal test exporter that records received snapshots."""

    def __init__(self, events: list[str], name: str) -> None:
        self._events = events
        self._name = name

    def export(self, snapshot: ObservationSnapshot) -> None:
        assert snapshot.counters == {"benchmark.runs": 1.0}
        self._events.append(self._name)


class FailingExporter:
    """Exporter used to verify fail-fast backend semantics."""

    def export(self, snapshot: ObservationSnapshot) -> None:
        raise RuntimeError("backend unavailable")


def _snapshot() -> ObservationSnapshot:
    return ObservationSnapshot(
        counters={"benchmark.runs": 1.0},
        durations_seconds={"benchmark.seconds": 0.25},
    )


def test_callback_exporter_receives_detached_versioned_payload() -> None:
    received: list[Mapping[str, object]] = []
    exporter = CallbackObservationExporter(received.append)
    snapshot = _snapshot()

    exporter.export(snapshot)

    assert received == [snapshot.to_dict()]
    counters = received[0]["counters"]
    assert isinstance(counters, dict)
    counters["benchmark.runs"] = 99.0
    assert snapshot.counters == {"benchmark.runs": 1.0}


def test_composite_exporter_preserves_configured_order() -> None:
    events: list[str] = []
    exporter = CompositeObservationExporter(
        [RecordingExporter(events, "first"), RecordingExporter(events, "second")]
    )

    exporter.export(_snapshot())

    assert events == ["first", "second"]


def test_composite_exporter_fails_fast_and_preserves_backend_error() -> None:
    events: list[str] = []
    exporter = CompositeObservationExporter(
        [
            RecordingExporter(events, "before"),
            FailingExporter(),
            RecordingExporter(events, "after"),
        ]
    )

    with pytest.raises(RuntimeError, match="backend unavailable"):
        exporter.export(_snapshot())

    assert events == ["before"]


def test_composite_exporter_rejects_empty_exporter_set() -> None:
    with pytest.raises(ValueError, match="at least one exporter"):
        CompositeObservationExporter([])


def test_composite_exporter_rejects_objects_without_export_method() -> None:
    with pytest.raises(TypeError, match="callable export method"):
        CompositeObservationExporter([object()])  # type: ignore[list-item]


def test_export_observation_snapshot_uses_same_ordered_contract() -> None:
    events: list[str] = []

    export_observation_snapshot(
        _snapshot(),
        [RecordingExporter(events, "first"), RecordingExporter(events, "second")],
    )

    assert events == ["first", "second"]
