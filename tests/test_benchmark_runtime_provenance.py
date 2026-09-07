from __future__ import annotations

import json

import pytest

from experiments.benchmark_report import save_benchmark_report
from experiments.runtime_provenance import (
    CLEAN_STATE,
    RUNTIME_PROVENANCE_SCHEMA_VERSION,
    RuntimeProvenance,
)
from tests.test_benchmark_report import _build_report


VALID_DEPENDENCY_FINGERPRINT = "a" * 64


def _provenance() -> RuntimeProvenance:
    return RuntimeProvenance(
        schema_version=RUNTIME_PROVENANCE_SCHEMA_VERSION,
        code_revision="abc123",
        working_tree_state=CLEAN_STATE,
        python_version="3.12.0",
        platform="test-platform",
        package_version="0.1.0",
        dependency_fingerprint=VALID_DEPENDENCY_FINGERPRINT,
        dependency_versions={"zeta": "2.0", "alpha": "1.0"},
    )


def test_save_benchmark_report_preserves_structured_runtime_provenance(tmp_path) -> None:
    output_path = save_benchmark_report(
        _build_report(seed=7),
        tmp_path / "report.json",
        runtime_provenance=_provenance().to_dict(),
    )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    provenance = persisted["runtime_provenance"]
    assert provenance["schema_version"] == RUNTIME_PROVENANCE_SCHEMA_VERSION
    assert provenance["dependency_versions"] == {"alpha": "1.0", "zeta": "2.0"}


def test_save_benchmark_report_rejects_invalid_structured_provenance(tmp_path) -> None:
    with pytest.raises(TypeError, match="values must be strings"):
        save_benchmark_report(
            _build_report(seed=7),
            tmp_path / "report.json",
            runtime_provenance={"schema_version": 1, "python_version": 3.12},
        )
