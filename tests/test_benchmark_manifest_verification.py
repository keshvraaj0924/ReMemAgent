"""Regression tests for benchmark artifact verification helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.benchmark_manifest import (
    save_benchmark_artifact_manifest,
    verify_benchmark_artifact_manifest,
)
from experiments.benchmark_report import BENCHMARK_REPORT_SCHEMA_VERSION


def _write_report(path: Path, *, reward: float = 1.0) -> None:
    """Write the smallest valid report document required by the manifest layer."""

    path.write_text(
        json.dumps(
            {
                "schema_version": BENCHMARK_REPORT_SCHEMA_VERSION,
                "benchmark_name": "verification-test",
                "episode_count": 1,
                "mean_reward": reward,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_verify_benchmark_artifact_manifest_returns_verified_metadata(tmp_path: Path) -> None:
    """A manifest created from a report must verify the unchanged report."""

    report_path = tmp_path / "report.json"
    manifest_path = tmp_path / "report.manifest.json"
    _write_report(report_path)
    save_benchmark_artifact_manifest(report_path, manifest_path)

    manifest = verify_benchmark_artifact_manifest(report_path, manifest_path)

    assert manifest.byte_count == report_path.stat().st_size
    assert len(manifest.sha256) == 64


def test_verify_benchmark_artifact_manifest_rejects_modified_report(tmp_path: Path) -> None:
    """Verification must fail when report bytes change after manifest creation."""

    report_path = tmp_path / "report.json"
    manifest_path = tmp_path / "report.manifest.json"
    _write_report(report_path)
    save_benchmark_artifact_manifest(report_path, manifest_path)
    _write_report(report_path, reward=2.0)

    with pytest.raises(ValueError, match="integrity verification failed"):
        verify_benchmark_artifact_manifest(report_path, manifest_path)
