from __future__ import annotations

import pytest

from experiments.runtime_provenance import (
    CLEAN_STATE,
    DIRTY_STATE,
    RUNTIME_PROVENANCE_SCHEMA_VERSION,
    UNKNOWN_VALUE,
    RuntimeProvenance,
)
from experiments.runtime_requirements import RuntimeRequirements, validate_runtime_requirements

VALID_DEPENDENCY_FINGERPRINT = "a" * 64


def _provenance(
    *,
    code_revision: str = "abc123",
    working_tree_state: str = CLEAN_STATE,
    dependencies: dict[str, str] | None = None,
) -> RuntimeProvenance:
    return RuntimeProvenance(
        schema_version=RUNTIME_PROVENANCE_SCHEMA_VERSION,
        code_revision=code_revision,
        working_tree_state=working_tree_state,
        python_version="3.12.0",
        platform="test-platform",
        package_version="0.1.0",
        dependency_fingerprint=VALID_DEPENDENCY_FINGERPRINT,
        dependency_versions=dependencies or {"ALFWorld": "0.4.2", "transformers": "5.0.0"},
    )


def test_validate_runtime_requirements_accepts_exact_controlled_runtime() -> None:
    requirements = RuntimeRequirements(
        expected_code_revision="abc123",
        require_clean_working_tree=True,
        dependency_versions={"alfworld": "0.4.2", "Transformers": "5.0.0"},
    )

    validate_runtime_requirements(_provenance(), requirements)


def test_validate_runtime_requirements_rejects_wrong_revision() -> None:
    requirements = RuntimeRequirements(expected_code_revision="required-sha")

    with pytest.raises(ValueError, match="code revision"):
        validate_runtime_requirements(_provenance(code_revision="other-sha"), requirements)


@pytest.mark.parametrize("state", [DIRTY_STATE, UNKNOWN_VALUE])
def test_validate_runtime_requirements_rejects_non_clean_checkout(state: str) -> None:
    requirements = RuntimeRequirements(require_clean_working_tree=True)

    with pytest.raises(ValueError, match="working tree must be clean"):
        validate_runtime_requirements(_provenance(working_tree_state=state), requirements)


def test_validate_runtime_requirements_rejects_missing_dependency() -> None:
    requirements = RuntimeRequirements(dependency_versions={"webshop": "1.0.0"})

    with pytest.raises(ValueError, match="not installed"):
        validate_runtime_requirements(_provenance(), requirements)


def test_validate_runtime_requirements_rejects_dependency_version_drift() -> None:
    requirements = RuntimeRequirements(dependency_versions={"transformers": "4.57.0"})

    with pytest.raises(ValueError, match="version mismatch"):
        validate_runtime_requirements(_provenance(), requirements)


def test_runtime_requirements_detach_dependency_mapping() -> None:
    dependencies = {"ALFWorld": "0.4.2"}
    requirements = RuntimeRequirements(dependency_versions=dependencies)

    dependencies["ALFWorld"] = "changed"

    assert requirements.dependency_versions == {"ALFWorld": "0.4.2"}


def test_runtime_requirements_dependency_mapping_is_immutable() -> None:
    requirements = RuntimeRequirements(dependency_versions={"ALFWorld": "0.4.2"})

    with pytest.raises(TypeError):
        requirements.dependency_versions["ALFWorld"] = "changed"  # type: ignore[index]

    assert requirements.dependency_versions == {"ALFWorld": "0.4.2"}


def test_runtime_requirements_round_trip_canonical_contract() -> None:
    requirements = RuntimeRequirements(
        expected_code_revision="abc123",
        require_clean_working_tree=True,
        dependency_versions={"Transformers": "5.0.0", "ALFWorld": "0.4.2"},
    )

    payload = requirements.to_dict()
    restored = RuntimeRequirements.from_dict(payload)

    assert restored == requirements
    assert restored.sha256 == requirements.sha256
    assert len(requirements.sha256) == 64


def test_runtime_requirements_fingerprint_changes_with_admission_contract() -> None:
    baseline = RuntimeRequirements(dependency_versions={"ALFWorld": "0.4.2"})
    changed = RuntimeRequirements(dependency_versions={"ALFWorld": "0.4.3"})

    assert baseline.sha256 != changed.sha256


def test_runtime_requirements_rejects_invalid_persisted_schema() -> None:
    requirements = RuntimeRequirements()
    payload = requirements.to_dict()
    payload["schema_version"] = 999

    with pytest.raises(ValueError, match="unsupported runtime requirements schema"):
        RuntimeRequirements.from_dict(payload)


def test_runtime_requirements_reject_case_insensitive_duplicate_dependencies() -> None:
    with pytest.raises(ValueError, match="unique"):
        RuntimeRequirements(dependency_versions={"ALFWorld": "0.4.2", " alfworld ": "0.4.2"})


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"expected_code_revision": ""}, ValueError),
        ({"require_clean_working_tree": 1}, TypeError),
        ({"dependency_versions": []}, TypeError),
        ({"dependency_versions": {"": "1.0"}}, ValueError),
        ({"dependency_versions": {"package": ""}}, ValueError),
        ({"dependency_versions": {"package": "   "}}, ValueError),
    ],
)
def test_runtime_requirements_reject_malformed_configuration(
    kwargs: dict[str, object],
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        RuntimeRequirements(**kwargs)


def test_validate_runtime_requirements_rejects_wrong_input_types() -> None:
    provenance = _provenance()
    requirements = RuntimeRequirements()

    with pytest.raises(TypeError, match="provenance"):
        validate_runtime_requirements(object(), requirements)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="requirements"):
        validate_runtime_requirements(provenance, object())  # type: ignore[arg-type]
