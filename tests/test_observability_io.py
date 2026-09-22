"""Tests for durable observability artifacts."""

from pathlib import Path

import pytest

from remem.observability import MetricSnapshot, MetricsRecorder
from remem.observability_io import (
    load_metric_snapshot,
    merge_metric_snapshots,
    save_metric_snapshot,
)


def test_metric_snapshot_persistence_round_trip(tmp_path: Path) -> None:
    recorder = MetricsRecorder()
    recorder.increment("episodes", 3)
    recorder.observe_duration("episode", 1.25)
    destination = tmp_path / "metrics.json"

    assert save_metric_snapshot(recorder.snapshot(), destination) == destination
    assert load_metric_snapshot(destination).to_dict() == recorder.snapshot().to_dict()
    assert destination.read_text(encoding="utf-8").endswith("\n")


def test_save_rejects_malformed_snapshot_before_replacing_file(tmp_path: Path) -> None:
    destination = tmp_path / "metrics.json"
    destination.write_text("existing\n", encoding="utf-8")
    malformed = MetricSnapshot(counters={}, timing_seconds={"episode": 1.0}, timing_counts={})

    with pytest.raises(ValueError, match="same metric names"):
        save_metric_snapshot(malformed, destination)
    assert destination.read_text(encoding="utf-8") == "existing\n"


def test_save_requires_existing_parent(tmp_path: Path) -> None:
    snapshot = MetricsRecorder().snapshot()
    with pytest.raises(FileNotFoundError, match="parent directory"):
        save_metric_snapshot(snapshot, tmp_path / "missing" / "metrics.json")


def test_load_rejects_invalid_json_and_root(tmp_path: Path) -> None:
    destination = tmp_path / "metrics.json"
    destination.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="valid JSON"):
        load_metric_snapshot(destination)

    destination.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="root must be an object"):
        load_metric_snapshot(destination)


def test_save_rejects_wrong_snapshot_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="MetricSnapshot"):
        save_metric_snapshot(object(), tmp_path / "metrics.json")  # type: ignore[arg-type]


def test_merge_metric_snapshots_aggregates_worker_artifacts(tmp_path: Path) -> None:
    first = MetricsRecorder()
    first.increment("episodes", 2)
    first.observe_duration("episode", 1.25)
    second = MetricsRecorder()
    second.increment("episodes", 3)
    second.increment("failures")
    second.observe_duration("episode", 0.75)

    first_path = save_metric_snapshot(first.snapshot(), tmp_path / "worker-1.json")
    second_path = save_metric_snapshot(second.snapshot(), tmp_path / "worker-2.json")

    merged = merge_metric_snapshots([second_path, first_path])

    assert merged.to_dict() == {
        "counters": {"episodes": 5, "failures": 1},
        "timing_seconds": {"episode": 2.0},
        "timing_counts": {"episode": 2},
    }


def test_merge_metric_snapshots_rejects_duplicate_paths(tmp_path: Path) -> None:
    path = save_metric_snapshot(MetricsRecorder().snapshot(), tmp_path / "worker.json")

    with pytest.raises(ValueError, match="duplicate metric snapshot path"):
        merge_metric_snapshots([path, path])


def test_merge_metric_snapshots_requires_path_collection(tmp_path: Path) -> None:
    path = save_metric_snapshot(MetricsRecorder().snapshot(), tmp_path / "worker.json")

    with pytest.raises(TypeError, match="iterable of snapshot paths"):
        merge_metric_snapshots(path)
    with pytest.raises(ValueError, match="at least one"):
        merge_metric_snapshots([])
    with pytest.raises(TypeError, match="each snapshot path"):
        merge_metric_snapshots([object()])  # type: ignore[list-item]


def test_merge_metric_snapshots_rejects_malformed_worker(tmp_path: Path) -> None:
    valid_path = save_metric_snapshot(MetricsRecorder().snapshot(), tmp_path / "valid.json")
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fields do not match schema"):
        merge_metric_snapshots([valid_path, malformed_path])
