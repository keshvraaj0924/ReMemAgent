from __future__ import annotations

import json

import pytest

from experiments.benchmark_report import save_benchmark_report
from tests.test_benchmark_report import _build_report


def test_save_benchmark_report_accepts_runtime_provenance_schema(tmp_path) -> None:
    provenance = {
        "schema_version": 1,
        "code_revision": "abc123",
        "working_tree_state": "clean",
        "python_version": "3.12.0",
        "platform": "linux",
        "package_version": "0.1.0",
        "dependency_fingerprint": "a" * 64,
        "dependency_versions": {
            "Zeta": "2.0",
            "alpha": "1.0",
        },
    }

    output_path = save_benchmark_report(
        _build_report(),
        tmp_path / "report.json",
        runtime_provenance=provenance,
    )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["runtime_provenance"]["schema_version"] == 1
    assert persisted["runtime_provenance"]["dependency_versions"] == {
        "alpha": "1.0",
        "Zeta": "2.0",
    }


def test_save_benchmark_report_rejects_non_string_dependency_version(tmp_path) -> None:
    provenance = {
        "schema_version": 1,
        "dependency_versions": {"numpy": 2},
    }

    with pytest.raises(TypeError, match="dependency versions must be strings"):
        save_benchmark_report(
            _build_report(),
            tmp_path / "report.json",
            runtime_provenance=provenance,
        )
