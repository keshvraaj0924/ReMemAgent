"""Tests for durable observation export checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest

from remem.observability import ObservationSnapshot
from remem.observability_exporters import (
    FileObservationCheckpointStore,
    ObservationExportSession,
)


class SnapshotRecordingExporter:
    """Record exported snapshots for persistence assertions."""

    def __init__(self) -> None:
        self.snapshots: list[ObservationSnapshot] = []

    def export(self, snapshot: ObservationSnapshot) -> None:
        self.snapshots.append(snapshot)


class FailingCheckpointStore:
    """Checkpoint store that fails after a successful backend export."""

    def __init__(self, initial: ObservationSnapshot) -> None:
        self.initial = initial

    def load(self) -> ObservationSnapshot:
        return self.initial

    def save(self, snapshot: ObservationSnapshot) -> None:
        raise OSError("checkpoint unavailable")


def test_file_checkpoint_store_returns_empty_snapshot_when_absent(tmp_path: Path) -> None:
    store = FileObservationCheckpointStore(tmp_path / "checkpoint.json")

    snapshot = store.load()

    assert snapshot.counters == {}
    assert snapshot.durations_seconds == {}


def test_file_checkpoint_store_round_trips_validated_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "checkpoint.json"
    store = FileObservationCheckpointStore(path)
    snapshot = ObservationSnapshot(
        counters={"benchmark.runs": 3.0},
        durations_seconds={"benchmark.seconds": 1.25},
    )

    store.save(snapshot)

    assert store.load() == snapshot
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_export_session_resumes_from_durable_checkpoint(tmp_path: Path) -> None:
    store = FileObservationCheckpointStore(tmp_path / "checkpoint.json")
    first_exporter = SnapshotRecordingExporter()
    first_session = ObservationExportSession(first_exporter, checkpoint_store=store)
    first_current = ObservationSnapshot(counters={"runs": 2.0}, durations_seconds={})

    first_delta = first_session.export(first_current)

    second_exporter = SnapshotRecordingExporter()
    resumed_session = ObservationExportSession(second_exporter, checkpoint_store=store)
    second_current = ObservationSnapshot(counters={"runs": 5.0}, durations_seconds={})
    second_delta = resumed_session.export(second_current)

    assert first_delta.counters == {"runs": 2.0}
    assert second_delta.counters == {"runs": 3.0}
    assert second_exporter.snapshots == [second_delta]
    assert resumed_session.checkpoint == second_current
    assert store.load() == second_current


def test_checkpoint_failure_preserves_in_memory_checkpoint_for_retry() -> None:
    initial = ObservationSnapshot(counters={"runs": 1.0}, durations_seconds={})
    current = ObservationSnapshot(counters={"runs": 3.0}, durations_seconds={})
    exporter = SnapshotRecordingExporter()
    session = ObservationExportSession(
        exporter,
        checkpoint_store=FailingCheckpointStore(initial),
    )

    with pytest.raises(OSError, match="checkpoint unavailable"):
        session.export(current)

    assert len(exporter.snapshots) == 1
    assert exporter.snapshots[0].counters == {"runs": 2.0}
    assert session.checkpoint == initial


def test_export_session_rejects_ambiguous_initial_checkpoint_sources(tmp_path: Path) -> None:
    initial = ObservationSnapshot(counters={}, durations_seconds={})

    with pytest.raises(ValueError, match="mutually exclusive"):
        ObservationExportSession(
            SnapshotRecordingExporter(),
            initial_snapshot=initial,
            checkpoint_store=FileObservationCheckpointStore(tmp_path / "checkpoint.json"),
        )


def test_file_checkpoint_store_rejects_malformed_payload(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text('{"schema_version":999,"counters":{},"durations_seconds":{}}', encoding="utf-8")
    store = FileObservationCheckpointStore(path)

    with pytest.raises(ValueError, match="unsupported observation snapshot schema version"):
        store.load()
