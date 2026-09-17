from __future__ import annotations

import json

import pytest

from experiments.benchmark_report import save_benchmark_report
from experiments.runtime_provenance import CLEAN_STATE, RuntimeProvenance, dependency_fingerprint
from tests.test_benchmark_report import _build_report


def _provenance() -> RuntimeProvenance:
    return RuntimeProvenance.create(
        code_revision="abc123",
        working_tree_state=CLEAN_STATE,
        python_version="3.12.0",
        platform="test-platform",
        package_version="0.1.0",
        dependency_versions={"zeta": "2.0", "alpha": "1.0"},
    )


def test_save_benchmark_report_preserves_structured_runtime_provenance(tmp_path) -> None:
    output_path = save_benchmark_report(
        _build_report(seed=7),
        tmp_path / "report.json",
        runtime_provenance=_provenance(),
    )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    provenance = persisted["runtime_provenance"]
    assert provenance["schema_version"] == 1
    assert provenance["dependency_versions"] == {"alpha": "1.0", "zeta": "2.0"}


def test_save_benchmark_report_rejects_incomplete_structured_provenance(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing runtime provenance fields"):
        save_benchmark_report(
            _build_report(seed=7),
            tmp_path / "report.json",
            runtime_provenance={"schema_version": 1, "python_version": "3.12"},
        )


def test_dependency_fingerprint_is_order_independent() -> None:
    first = dependency_fingerprint({"alpha": "1.0", "zeta": "2.0"})
    second = dependency_fingerprint({"zeta": "2.0", "alpha": "1.0"})

    assert first == second
    assert len(first) == 64


def test_runtime_provenance_rejects_tampered_dependency_versions() -> None:
    provenance = _provenance()

    with pytest.raises(ValueError, match="does not match dependency_versions"):
        RuntimeProvenance(
            schema_version=provenance.schema_version,
            code_revision=provenance.code_revision,
            working_tree_state=provenance.working_tree_state,
            python_version=provenance.python_version,
            platform=provenance.platform,
            package_version=provenance.package_version,
            dependency_fingerprint=provenance.dependency_fingerprint,
            dependency_versions={"alpha": "1.1", "zeta": "2.0"},
        )


def test_runtime_provenance_from_dict_round_trips_verified_payload() -> None:
    provenance = _provenance()

    restored = RuntimeProvenance.from_dict(provenance.to_dict())

    assert restored.to_dict() == provenance.to_dict()


def test_runtime_provenance_fingerprint_is_stable_across_dependency_order() -> None:
    first = _provenance()
    second = RuntimeProvenance.create(
        code_revision="abc123",
        working_tree_state=CLEAN_STATE,
        python_version="3.12.0",
        platform="test-platform",
        package_version="0.1.0",
        dependency_versions={"alpha": "1.0", "zeta": "2.0"},
    )

    assert first.fingerprint() == second.fingerprint()
    assert len(first.fingerprint()) == 64


def test_runtime_provenance_fingerprint_changes_with_code_revision() -> None:
    provenance = _provenance()
    changed = RuntimeProvenance.create(
        code_revision="def456",
        working_tree_state=CLEAN_STATE,
        python_version=provenance.python_version,
        platform=provenance.platform,
        package_version=provenance.package_version,
        dependency_versions=provenance.dependency_versions,
    )

    assert changed.fingerprint() != provenance.fingerprint()


def test_save_benchmark_report_rejects_tampered_provenance_mapping(tmp_path) -> None:
    provenance = _provenance().to_dict()
    provenance["dependency_versions"] = {"alpha": "9.9", "zeta": "2.0"}

    with pytest.raises(ValueError, match="does not match dependency_versions"):
        save_benchmark_report(
            _build_report(seed=7),
            tmp_path / "report.json",
            runtime_provenance=provenance,
        )


def test_runtime_provenance_rejects_unknown_fields() -> None:
    payload = _provenance().to_dict()
    payload["unexpected"] = "value"

    with pytest.raises(ValueError, match="unknown runtime provenance fields"):
        RuntimeProvenance.from_dict(payload)
