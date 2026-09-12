from __future__ import annotations

import json
from pathlib import Path

import pytest

from remem.observability_distribution_artifacts import (
    DISTRIBUTION_OBSERVATION_SCHEMA_VERSION,
    DistributionObservationSnapshot,
    read_distribution_observation_snapshot,
    write_distribution_observation_snapshot,
)
from remem.observability_distributions import ObservationHistogramSnapshot


def _histogram() -> ObservationHistogramSnapshot:
    return ObservationHistogramSnapshot(
        upper_bounds=(0.1, 0.5, 1.0),
        bucket_counts=(1, 2, 0, 1),
        total=1.35,
    )


def test_distribution_snapshot_serializes_metrics_in_sorted_order() -> None:
    snapshot = DistributionObservationSnapshot(
        duration_histograms={
            "z.metric": _histogram(),
            "a.metric": _histogram(),
        }
    )

    assert list(snapshot.to_dict()["duration_histograms"]) == ["a.metric", "z.metric"]
    assert snapshot.to_dict()["schema_version"] == DISTRIBUTION_OBSERVATION_SCHEMA_VERSION


def test_distribution_snapshot_rejects_normalized_name_collisions() -> None:
    with pytest.raises(ValueError, match="normalize to the same value"):
        DistributionObservationSnapshot(
            duration_histograms={
                "benchmark.duration": _histogram(),
                " benchmark.duration ": _histogram(),
            }
        )


def test_distribution_snapshot_round_trips_through_canonical_json(tmp_path: Path) -> None:
    snapshot = DistributionObservationSnapshot(
        duration_histograms={"benchmark.episode.duration_seconds": _histogram()}
    )
    path = tmp_path / "duration-distributions.json"

    write_distribution_observation_snapshot(path, snapshot)

    assert read_distribution_observation_snapshot(path) == snapshot
    persisted = path.read_text(encoding="utf-8")
    assert persisted.endswith("\n")
    assert (
        persisted
        == json.dumps(
            snapshot.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def test_distribution_snapshot_writer_rejects_existing_file_without_overwrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duration-distributions.json"
    path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        write_distribution_observation_snapshot(
            path,
            DistributionObservationSnapshot(duration_histograms={}),
        )

    assert path.read_text(encoding="utf-8") == "existing"


def test_distribution_snapshot_writer_replaces_existing_file_with_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "duration-distributions.json"
    path.write_text("existing", encoding="utf-8")
    snapshot = DistributionObservationSnapshot(
        duration_histograms={"benchmark.episode.duration_seconds": _histogram()}
    )

    write_distribution_observation_snapshot(path, snapshot, overwrite=True)

    assert read_distribution_observation_snapshot(path) == snapshot


def test_distribution_snapshot_reader_rejects_unsupported_schema(tmp_path: Path) -> None:
    path = tmp_path / "duration-distributions.json"
    path.write_text(
        json.dumps({"schema_version": 999, "duration_histograms": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported distribution observation schema version"):
        read_distribution_observation_snapshot(path)


def test_distribution_snapshot_reader_rejects_tampered_derived_count(tmp_path: Path) -> None:
    path = tmp_path / "duration-distributions.json"
    payload = DistributionObservationSnapshot(
        duration_histograms={"benchmark.episode.duration_seconds": _histogram()}
    ).to_dict()
    histogram = payload["duration_histograms"]["benchmark.episode.duration_seconds"]
    histogram["count"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="count does not match"):
        read_distribution_observation_snapshot(path)


def test_distribution_snapshot_reader_rejects_tampered_derived_mean(tmp_path: Path) -> None:
    path = tmp_path / "duration-distributions.json"
    payload = DistributionObservationSnapshot(
        duration_histograms={"benchmark.episode.duration_seconds": _histogram()}
    ).to_dict()
    histogram = payload["duration_histograms"]["benchmark.episode.duration_seconds"]
    histogram["mean"] = 42.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="mean does not match"):
        read_distribution_observation_snapshot(path)
