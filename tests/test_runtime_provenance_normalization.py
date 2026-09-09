from __future__ import annotations

import hashlib

import pytest

from experiments.runtime_provenance import (
    CLEAN_STATE,
    RUNTIME_PROVENANCE_SCHEMA_VERSION,
    RuntimeProvenance,
)


def _build_provenance(dependency_versions: dict[str, str]) -> RuntimeProvenance:
    """Build a valid provenance object for normalization tests."""

    return RuntimeProvenance(
        schema_version=RUNTIME_PROVENANCE_SCHEMA_VERSION,
        code_revision="abc123",
        working_tree_state=CLEAN_STATE,
        python_version="3.12.0",
        platform="test-platform",
        package_version="0.1.0",
        dependency_fingerprint=hashlib.sha256(b"dependencies").hexdigest(),
        dependency_versions=dependency_versions,
    )


def test_runtime_provenance_normalizes_dependency_name_whitespace() -> None:
    provenance = _build_provenance({"  pytest  ": "8.4.0"})

    assert provenance.dependency_versions == {"pytest": "8.4.0"}


def test_runtime_provenance_rejects_case_insensitive_dependency_collisions() -> None:
    with pytest.raises(ValueError, match="unique after whitespace and case normalization"):
        _build_provenance({"pytest": "8.4.0", " PYTEST ": "8.4.1"})


def test_runtime_provenance_rejects_whitespace_only_dependency_name() -> None:
    with pytest.raises(ValueError, match="dependency name must not be whitespace-only"):
        _build_provenance({"   ": "8.4.0"})
