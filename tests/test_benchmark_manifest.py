import json
from pathlib import Path

import pytest

from experiments.benchmark_manifest import (
    BenchmarkArtifactManifest,
    build_benchmark_artifact_manifest,
    verify_benchmark_artifact,
)


def _write_valid_report(path: Path) -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "benchmark_name": "test"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_build_benchmark_artifact_manifest_hashes_serialized_report(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    _write_valid_report(report_path)

    manifest = build_benchmark_artifact_manifest(report_path)

    assert manifest.schema_version == 1
    assert manifest.byte_count == report_path.stat().st_size
    assert len(manifest.sha256) == 64


def test_verify_benchmark_artifact_accepts_unchanged_report(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    _write_valid_report(report_path)
    manifest = build_benchmark_artifact_manifest(report_path)

    verify_benchmark_artifact(report_path, manifest)


def test_verify_benchmark_artifact_rejects_mutated_report(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    _write_valid_report(report_path)
    manifest = build_benchmark_artifact_manifest(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["seed"] = 18
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="integrity verification failed"):
        verify_benchmark_artifact(report_path, manifest)


def test_build_benchmark_artifact_manifest_rejects_invalid_schema(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported benchmark report schema version"):
        build_benchmark_artifact_manifest(report_path)


def test_verify_benchmark_artifact_rejects_wrong_manifest(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    _write_valid_report(report_path)
    manifest = BenchmarkArtifactManifest(schema_version=1, byte_count=0, sha256="0" * 64)

    with pytest.raises(ValueError, match="integrity verification failed"):
        verify_benchmark_artifact(report_path, manifest)
