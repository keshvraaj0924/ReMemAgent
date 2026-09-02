"""Regression coverage for the standalone benchmark artifact verifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.benchmark_manifest import save_benchmark_artifact_manifest
from experiments.verify_benchmark_artifact import main


def _write_report(path: Path) -> None:
    """Write the smallest schema-valid benchmark artifact for CLI tests."""

    path.write_text(
        json.dumps({"schema_version": 1, "benchmark_name": "smoke"}),
        encoding="utf-8",
    )


def test_main_verifies_default_manifest_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The verifier derives the conventional sidecar path when omitted."""

    report_path = tmp_path / "benchmark.json"
    _write_report(report_path)
    save_benchmark_artifact_manifest(report_path)
    monkeypatch.setattr("sys.argv", ["verify_benchmark_artifact", str(report_path)])

    assert main() == 0
    assert "integrity verified" in capsys.readouterr().out


def test_main_accepts_explicit_manifest_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The verifier accepts a separately located manifest."""

    report_path = tmp_path / "benchmark.json"
    manifest_path = tmp_path / "checks" / "manifest.json"
    _write_report(report_path)
    save_benchmark_artifact_manifest(report_path, manifest_path)
    monkeypatch.setattr(
        "sys.argv",
        ["verify_benchmark_artifact", str(report_path), "--manifest", str(manifest_path)],
    )

    assert main() == 0


def test_main_rejects_mutated_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A modified report must fail closed rather than merely warn."""

    report_path = tmp_path / "benchmark.json"
    _write_report(report_path)
    save_benchmark_artifact_manifest(report_path)
    report_path.write_text(
        json.dumps({"schema_version": 1, "benchmark_name": "mutated"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["verify_benchmark_artifact", str(report_path)])

    with pytest.raises(ValueError, match="integrity verification failed"):
        main()
