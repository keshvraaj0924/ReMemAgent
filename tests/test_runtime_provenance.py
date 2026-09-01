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
    assert provenance.to_dict()["code_revision"] == "abc123"


def test_collect_runtime_provenance_does_not_fabricate_missing_revision() -> None:
    provenance = collect_runtime_provenance(repository_path=Path("/missing"))

    assert provenance.code_revision == UNKNOWN_VALUE
