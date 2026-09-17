"""Tests for structured runtime provenance comparison."""

import pytest

from experiments.provenance_comparison import compare_runtime_provenance
from experiments.runtime_provenance import CLEAN_STATE, DIRTY_STATE, RuntimeProvenance


def _provenance(*, revision: str = "abc123", tree_state: str = CLEAN_STATE) -> RuntimeProvenance:
    return RuntimeProvenance.create(
        code_revision=revision,
        working_tree_state=tree_state,
        python_version="3.12.0",
        platform="test-platform",
        package_version="0.1.0",
        dependency_versions={"alpha": "1.0", "zeta": "2.0"},
    )


def test_equivalent_provenance_has_no_differences() -> None:
    comparison = compare_runtime_provenance(_provenance(), _provenance())

    assert comparison.is_equivalent
    assert comparison.differences == ()
    comparison.assert_equivalent()


def test_comparison_reports_differences_in_deterministic_field_order() -> None:
    comparison = compare_runtime_provenance(
        _provenance(),
        _provenance(revision="def456", tree_state=DIRTY_STATE),
    )

    assert not comparison.is_equivalent
    assert tuple(item.field_name for item in comparison.differences) == (
        "code_revision",
        "working_tree_state",
    )
    assert comparison.differences[0].expected == "abc123"
    assert comparison.differences[0].actual == "def456"


def test_assert_equivalent_identifies_incompatible_fields() -> None:
    comparison = compare_runtime_provenance(_provenance(), _provenance(revision="def456"))

    with pytest.raises(ValueError, match="code_revision"):
        comparison.assert_equivalent()


def test_comparison_rejects_unvalidated_inputs() -> None:
    with pytest.raises(TypeError, match="expected must be RuntimeProvenance"):
        compare_runtime_provenance({}, _provenance())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="actual must be RuntimeProvenance"):
        compare_runtime_provenance(_provenance(), {})  # type: ignore[arg-type]
