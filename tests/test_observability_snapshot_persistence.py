from pathlib import Path

import pytest

from remem.observability import (
    ObservationCollector,
    ObservationSnapshot,
    write_observation_snapshot,
)


def test_write_observation_snapshot_is_deterministic(tmp_path: Path) -> None:
    snapshot = ObservationSnapshot(
        counters={"z.metric": 2.0, "a.metric": 1.0},
        durations_seconds={"route": 0.25},
    )
    first_path = tmp_path / "nested" / "first.json"
    second_path = tmp_path / "nested" / "second.json"

    write_observation_snapshot(first_path, snapshot)
    write_observation_snapshot(second_path, snapshot)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.read_text(encoding="utf-8") == (
        '{"counters":{"a.metric":1.0,"z.metric":2.0},'
        '"durations_seconds":{"route":0.25}}\n'
    )


def test_write_observation_snapshot_replaces_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "observations.json"
    path.write_text("stale", encoding="utf-8")

    collector = ObservationCollector()
    collector.increment("episodes")
    write_observation_snapshot(path, collector.snapshot())

    assert path.read_text(encoding="utf-8") == (
        '{"counters":{"episodes":1.0},"durations_seconds":{}}\n'
    )


def test_write_observation_snapshot_cleans_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "observations.json"
    write_observation_snapshot(path, ObservationSnapshot({}, {}))

    assert list(tmp_path.iterdir()) == [path]


def test_write_observation_snapshot_accepts_path_like_values(tmp_path: Path) -> None:
    path = tmp_path / "observations.json"
    write_observation_snapshot(str(path), ObservationSnapshot({}, {}))

    assert path.exists()


def test_write_observation_snapshot_rejects_unwritable_destination(tmp_path: Path) -> None:
    directory = tmp_path / "not-a-file"
    directory.mkdir()

    with pytest.raises(OSError):
        write_observation_snapshot(directory, ObservationSnapshot({}, {}))
