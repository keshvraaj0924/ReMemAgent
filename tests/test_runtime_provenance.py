from __future__ import annotations

from pathlib import Path

import pytest

from experiments.runtime_provenance import (
    CLEAN_STATE,
    DIRTY_STATE,
    RUNTIME_PROVENANCE_SCHEMA_VERSION,
    UNKNOWN_VALUE,
    collect_runtime_provenance,
)


def test_collect_runtime_provenance_prefers_explicit_commit_and_state() -> None:
    provenance = collect_runtime_provenance(
        repository_path=Path("/missing"),
        environment={"REMEM_GIT_COMMIT": "abc123", "REMEM_GIT_STATE": DIRTY_STATE},
    )

    assert provenance.schema_version == RUNTIME_PROVENANCE_SCHEMA_VERSION
    assert provenance.code_revision == "abc123"
    assert provenance.working_tree_state == DIRTY_STATE
    assert provenance.python_version
    assert provenance.platform
    assert provenance.dependency_fingerprint
    assert provenance.dependency_versions
    assert provenance.to_dict()["schema_version"] == RUNTIME_PROVENANCE_SCHEMA_VERSION
    assert provenance.to_dict()["code_revision"] == "abc123"
    assert provenance.to_dict()["working_tree_state"] == DIRTY_STATE


def test_collect_runtime_provenance_does_not_fabricate_missing_revision_or_state() -> None:
    provenance = collect_runtime_provenance(repository_path=Path("/missing"))

    assert provenance.code_revision == UNKNOWN_VALUE
    assert provenance.working_tree_state == UNKNOWN_VALUE


def test_dependency_fingerprint_is_stable_for_mapping_order() -> None:
    from experiments import runtime_provenance

    first = runtime_provenance._dependency_fingerprint({"zeta": "2", "alpha": "1"})
    second = runtime_provenance._dependency_fingerprint({"alpha": "1", "zeta": "2"})

    assert first == second


def test_working_tree_state_distinguishes_clean_and_dirty_output(monkeypatch) -> None:
    from experiments import runtime_provenance

    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    outputs = iter(("", " M experiment.py\n"))
    monkeypatch.setattr(
        runtime_provenance.subprocess,
        "run",
        lambda *args, **kwargs: Result(next(outputs)),
    )

    assert runtime_provenance._git_working_tree_state(Path(".")) == CLEAN_STATE
    assert runtime_provenance._git_working_tree_state(Path(".")) == DIRTY_STATE


@pytest.mark.parametrize("invalid_state", ["", "modified", "CLEAN", "false"])
def test_collect_runtime_provenance_rejects_invalid_explicit_state(invalid_state: str) -> None:
    with pytest.raises(ValueError, match="REMEM_GIT_STATE"):
        collect_runtime_provenance(
            repository_path=Path("/missing"),
            environment={"REMEM_GIT_STATE": invalid_state},
        )


def test_collect_runtime_provenance_accepts_unknown_explicit_state() -> None:
    provenance = collect_runtime_provenance(
        repository_path=Path("/missing"),
        environment={"REMEM_GIT_STATE": UNKNOWN_VALUE},
    )

    assert provenance.working_tree_state == UNKNOWN_VALUE
