"""Comparison helpers for reproducible benchmark runtime provenance."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.runtime_provenance import RuntimeProvenance


@dataclass(frozen=True, slots=True)
class ProvenanceDifference:
    """One reproducibility-relevant difference between two runtime artifacts."""

    field_name: str
    expected: object
    actual: object


@dataclass(frozen=True, slots=True)
class ProvenanceComparison:
    """Structured comparison of two validated runtime provenance artifacts."""

    differences: tuple[ProvenanceDifference, ...]

    @property
    def is_equivalent(self) -> bool:
        """Return whether the two runtimes are reproducibly equivalent."""
        return not self.differences

    def assert_equivalent(self) -> None:
        """Raise with actionable field names when runtime provenance differs."""
        if self.differences:
            field_names = ", ".join(difference.field_name for difference in self.differences)
            raise ValueError(f"runtime provenance differs in fields: {field_names}")


def compare_runtime_provenance(
    expected: RuntimeProvenance,
    actual: RuntimeProvenance,
) -> ProvenanceComparison:
    """Compare canonical validated provenance without relying on object identity."""
    if not isinstance(expected, RuntimeProvenance):
        raise TypeError("expected must be RuntimeProvenance")
    if not isinstance(actual, RuntimeProvenance):
        raise TypeError("actual must be RuntimeProvenance")

    expected_payload = expected.to_dict()
    actual_payload = actual.to_dict()
    differences = tuple(
        ProvenanceDifference(
            field_name=field_name,
            expected=expected_payload[field_name],
            actual=actual_payload[field_name],
        )
        for field_name in sorted(expected_payload)
        if expected_payload[field_name] != actual_payload[field_name]
    )
    return ProvenanceComparison(differences=differences)


__all__ = [
    "ProvenanceComparison",
    "ProvenanceDifference",
    "compare_runtime_provenance",
]
