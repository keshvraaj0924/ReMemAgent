"""Tests for durable observability artifacts."""

from pathlib import Path

import pytest

from remem.observability import MetricSnapshot, MetricsRecorder
from remem.observability_io import load_metric_snapshot, save_metric_snapshot


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
    malformed = MetricSnapshot(
        counters={}, timing_seconds={"episode": 1.0}, timing_counts={}
    )

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
