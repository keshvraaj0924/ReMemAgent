"""Regression tests for the benchmark artifact verification CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.benchmark_manifest import save_benchmark_artifact_manifest
from experiments.verify_benchmark_artifact import verify_report_artifact


def _write_report(path: Path) -> None:
    """Write the minimum JSON document accepted by the artifact manifest layer."""

    path.write_text('{"schema_version":1,"episodes":[]}', encoding="utf-8")


def test_verify_report_artifact_uses_default_sidecar(tmp_path: Path) -> None:
    """The verifier should resolve the conventional sidecar path."""

    report_path = tmp_path / "benchmark.json"
    _write_report(report_path)
    manifest_path = save_benchmark_artifact_manifest(report_path)

    verify_report_artifact(report_path)

    assert manifest_path == tmp_path / "benchmark.json.manifest.json"


def test_verify_report_artifact_rejects_modified_report(tmp_path: Path) -> None:
    """The CLI helper should fail closed after report bytes change."""

    report_path = tmp_path / "benchmark.json"
    _write_report(report_path)
    manifest_path = save_benchmark_artifact_manifest(report_path)
    report_path.write_text(report_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="integrity verification failed"):
        verify_report_artifact(report_path, manifest_path)


def test_verify_report_artifact_rejects_missing_manifest(tmp_path: Path) -> None:
    """A missing sidecar must not be interpreted as an unverified success."""

    report_path = tmp_path / "benchmark.json"
    _write_report(report_path)

    with pytest.raises(ValueError, match="invalid benchmark artifact manifest"):
        verify_report_artifact(report_path)
