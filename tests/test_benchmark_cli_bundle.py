from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.benchmark_cli import _persist_benchmark_bundle
from experiments.benchmark_report import BENCHMARK_REPORT_SCHEMA_VERSION
from experiments.benchmark_manifest import verify_benchmark_artifact_manifest
from remem.observability import ObservationSnapshot


def _write_valid_report(path: Path) -> None:
    path.write_text(
        json.dumps({"schema_version": BENCHMARK_REPORT_SCHEMA_VERSION}) + "\n",
        encoding="utf-8",
    )


def test_persist_bundle_publishes_mutually_consistent_artifacts(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    manifest_path = tmp_path / "report.manifest.json"
    observability_path = tmp_path / "observability.json"
    snapshot = ObservationSnapshot(
        counters={"benchmark.runs.completed": 1.0},
        durations_seconds={},
    )

    persisted_path = _persist_benchmark_bundle(
        report_path,
        overwrite=False,
        writer=_write_valid_report,
        manifest_path=manifest_path,
        observability_path=observability_path,
        observation_snapshot=snapshot,
    )

    assert persisted_path == report_path
    assert report_path.exists()
    assert manifest_path.exists()
    assert observability_path.exists()
    verify_benchmark_artifact_manifest(report_path, manifest_path)


def test_persist_bundle_rolls_back_auxiliary_files_if_report_is_claimed_concurrently(
    tmp_path,
) -> None:
    report_path = tmp_path / "report.json"
    manifest_path = tmp_path / "report.manifest.json"
    observability_path = tmp_path / "observability.json"
    snapshot = ObservationSnapshot(counters={}, durations_seconds={})

    def writer(path: Path) -> None:
        _write_valid_report(path)
        report_path.write_text("concurrent report\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        _persist_benchmark_bundle(
            report_path,
            overwrite=False,
            writer=writer,
            manifest_path=manifest_path,
            observability_path=observability_path,
            observation_snapshot=snapshot,
        )

    assert report_path.read_text(encoding="utf-8") == "concurrent report\n"
    assert not manifest_path.exists()
    assert not observability_path.exists()
    assert not tuple(tmp_path.glob(".*.bundle.tmp"))


def test_persist_bundle_rejects_incomplete_observability_pair(tmp_path) -> None:
    report_path = tmp_path / "report.json"

    with pytest.raises(ValueError, match="must either both be provided or both omitted"):
        _persist_benchmark_bundle(
            report_path,
            overwrite=False,
            writer=_write_valid_report,
            observability_path=tmp_path / "observability.json",
        )

    assert not report_path.exists()
