from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.benchmark_manifest import verify_benchmark_artifact_manifest
from experiments.benchmark_observability_bundle import persist_benchmark_observability_bundle
from experiments.benchmark_report import BENCHMARK_REPORT_SCHEMA_VERSION
from remem.observability import ObservationSnapshot
from remem.observability_distribution_artifacts import (
    DistributionObservationSnapshot,
    read_distribution_observation_snapshot,
)
from remem.observability_distributions import ObservationHistogramSnapshot


def _write_valid_report(path: Path) -> None:
    path.write_text(
        json.dumps({"schema_version": BENCHMARK_REPORT_SCHEMA_VERSION}) + "\n",
        encoding="utf-8",
    )


def _distribution_snapshot() -> DistributionObservationSnapshot:
    return DistributionObservationSnapshot(
        duration_histograms={
            "benchmark.episode.duration_seconds": ObservationHistogramSnapshot(
                upper_bounds=(0.5, 1.0),
                bucket_counts=(1, 1, 1),
                total=2.1,
            )
        }
    )


def test_persist_bundle_publishes_report_manifest_and_observability_sidecars(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    manifest_path = tmp_path / "report.manifest.json"
    observability_path = tmp_path / "observability.json"
    distribution_path = tmp_path / "observability.distributions.json"
    observation_snapshot = ObservationSnapshot(
        counters={"benchmark.runs.completed": 1.0},
        durations_seconds={"benchmark.episode.duration_seconds": 2.1},
    )
    distribution_snapshot = _distribution_snapshot()

    persisted_path = persist_benchmark_observability_bundle(
        report_path,
        overwrite=False,
        report_writer=_write_valid_report,
        manifest_path=manifest_path,
        observability_path=observability_path,
        observation_snapshot=observation_snapshot,
        distribution_path=distribution_path,
        distribution_snapshot=distribution_snapshot,
    )

    assert persisted_path == report_path
    assert report_path.exists()
    assert manifest_path.exists()
    assert observability_path.exists()
    assert distribution_path.exists()
    verify_benchmark_artifact_manifest(report_path, manifest_path)
    assert read_distribution_observation_snapshot(distribution_path) == distribution_snapshot


def test_persist_bundle_rolls_back_sidecars_if_report_is_claimed_concurrently(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    observability_path = tmp_path / "observability.json"
    distribution_path = tmp_path / "observability.distributions.json"

    def writer(path: Path) -> None:
        _write_valid_report(path)
        report_path.write_text("concurrent report\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        persist_benchmark_observability_bundle(
            report_path,
            overwrite=False,
            report_writer=writer,
            observability_path=observability_path,
            observation_snapshot=ObservationSnapshot(counters={}, durations_seconds={}),
            distribution_path=distribution_path,
            distribution_snapshot=_distribution_snapshot(),
        )

    assert report_path.read_text(encoding="utf-8") == "concurrent report\n"
    assert not observability_path.exists()
    assert not distribution_path.exists()
    assert not tuple(tmp_path.glob(".*.bundle.tmp"))


def test_persist_bundle_rejects_incomplete_distribution_pair_before_writing(tmp_path) -> None:
    report_path = tmp_path / "report.json"

    with pytest.raises(ValueError, match="distribution observability_path"):
        persist_benchmark_observability_bundle(
            report_path,
            overwrite=False,
            report_writer=_write_valid_report,
            distribution_path=tmp_path / "distribution.json",
        )

    assert not report_path.exists()


def test_persist_bundle_rejects_destination_aliases_before_writing(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    distribution_snapshot = _distribution_snapshot()

    with pytest.raises(ValueError, match="destinations must be distinct"):
        persist_benchmark_observability_bundle(
            report_path,
            overwrite=False,
            report_writer=_write_valid_report,
            distribution_path=report_path,
            distribution_snapshot=distribution_snapshot,
        )

    assert not report_path.exists()
