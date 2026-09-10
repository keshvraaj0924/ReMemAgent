"""Tests for the dependency-free observability exporter boundary."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from remem.observability import ObservationSnapshot
from remem.observability_exporters import (
    CallbackObservationExporter,
    CompositeObservationExporter,
    ObservationExportSession,
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


class SnapshotRecordingExporter:
    """Record exported snapshots for export-session assertions."""

    def __init__(self) -> None:
        self.snapshots: list[ObservationSnapshot] = []

    def export(self, snapshot: ObservationSnapshot) -> None:
        self.snapshots.append(snapshot)


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


def test_export_session_emits_only_interval_delta() -> None:
    exporter = SnapshotRecordingExporter()
    initial = ObservationSnapshot(
        counters={"benchmark.runs": 2.0},
        durations_seconds={"benchmark.seconds": 1.0},
    )
    session = ObservationExportSession(exporter, initial_snapshot=initial)
    current = ObservationSnapshot(
        counters={"benchmark.runs": 5.0},
        durations_seconds={"benchmark.seconds": 1.75},
    )

    delta = session.export(current)

    assert delta.counters == {"benchmark.runs": 3.0}
    assert delta.durations_seconds == {"benchmark.seconds": 0.75}
    assert exporter.snapshots == [delta]
    assert session.checkpoint == current


def test_export_session_does_not_advance_checkpoint_after_backend_failure() -> None:
    initial = ObservationSnapshot(counters={"runs": 1.0}, durations_seconds={})
    current = ObservationSnapshot(counters={"runs": 3.0}, durations_seconds={})
    session = ObservationExportSession(FailingExporter(), initial_snapshot=initial)

    with pytest.raises(RuntimeError, match="backend unavailable"):
        session.export(current)

    assert session.checkpoint == initial


def test_export_session_skips_backend_for_empty_delta() -> None:
    exporter = SnapshotRecordingExporter()
    snapshot = ObservationSnapshot(counters={"runs": 2.0}, durations_seconds={})
    session = ObservationExportSession(exporter, initial_snapshot=snapshot)

    delta = session.export(snapshot)

    assert delta.counters == {}
    assert delta.durations_seconds == {}
    assert exporter.snapshots == []
    assert session.checkpoint == snapshot


def test_export_session_rejects_regressing_cumulative_snapshot() -> None:
    exporter = SnapshotRecordingExporter()
    initial = ObservationSnapshot(counters={"runs": 3.0}, durations_seconds={})
    session = ObservationExportSession(exporter, initial_snapshot=initial)
    regressed = ObservationSnapshot(counters={"runs": 2.0}, durations_seconds={})

    with pytest.raises(ValueError, match="counter aggregate regressed"):
        session.export(regressed)

    assert exporter.snapshots == []
    assert session.checkpoint == initial
