from __future__ import annotations

from pathlib import Path

from experiments.runtime_provenance import UNKNOWN_VALUE, collect_runtime_provenance


def test_collect_runtime_provenance_prefers_explicit_commit() -> None:
    provenance = collect_runtime_provenance(
        repository_path=Path("/missing"),
        environment={"REMEM_GIT_COMMIT": "abc123"},
    )

    assert provenance.code_revision == "abc123"
    assert provenance.python_version
    assert provenance.platform
    assert provenance.dependency_fingerprint
    assert provenance.dependency_versions
    assert provenance.to_dict()["code_revision"] == "abc123"


def test_collect_runtime_provenance_does_not_fabricate_missing_revision() -> None:
    provenance = collect_runtime_provenance(repository_path=Path("/missing"))

    assert provenance.code_revision == UNKNOWN_VALUE


def test_dependency_fingerprint_is_stable_for_mapping_order() -> None:
    from experiments import runtime_provenance

    first = runtime_provenance._dependency_fingerprint({"zeta": "2", "alpha": "1"})
    second = runtime_provenance._dependency_fingerprint({"alpha": "1", "zeta": "2"})

    assert first == second
