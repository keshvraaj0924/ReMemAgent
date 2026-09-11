"""Fail-closed runtime requirements for controlled benchmark measurement.

Runtime provenance records what executed an experiment. This module adds the
complementary pre-measurement gate: callers can require a clean checkout, an
exact code revision, and exact versions for benchmark/model runtime packages
before expensive external environments are constructed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from experiments.runtime_provenance import CLEAN_STATE, RuntimeProvenance


@dataclass(frozen=True, slots=True)
class RuntimeRequirements:
    """Exact runtime constraints required before a measured experiment starts."""

    expected_code_revision: str | None = None
    require_clean_working_tree: bool = False
    dependency_versions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate, detach, and freeze requirement metadata at construction time."""

        if self.expected_code_revision is not None:
            _require_non_empty_string("expected_code_revision", self.expected_code_revision)
        if not isinstance(self.require_clean_working_tree, bool):
            raise TypeError("require_clean_working_tree must be a boolean")
        if not isinstance(self.dependency_versions, Mapping):
            raise TypeError("dependency_versions must be a mapping")

        normalized_requirements: dict[str, str] = {}
        display_names: dict[str, str] = {}
        for package_name, package_version in self.dependency_versions.items():
            validated_name = _require_non_empty_string("dependency name", package_name).strip()
            validated_version = _require_non_empty_string(
                f"dependency version for {validated_name!r}", package_version
            ).strip()
            if not validated_name:
                raise ValueError("dependency name must not be whitespace-only")
            if not validated_version:
                raise ValueError(
                    f"dependency version for {validated_name!r} must not be whitespace-only"
                )

            normalized_name = validated_name.lower()
            if normalized_name in normalized_requirements:
                existing_name = display_names[normalized_name]
                raise ValueError(
                    "dependency requirements must be unique after whitespace and case "
                    f"normalization: {existing_name!r} and {validated_name!r}"
                )
            normalized_requirements[normalized_name] = validated_version
            display_names[normalized_name] = validated_name

        detached_requirements = {
            display_names[name]: normalized_requirements[name]
            for name in sorted(normalized_requirements)
        }
        object.__setattr__(
            self,
            "dependency_versions",
            MappingProxyType(detached_requirements),
        )


def validate_runtime_requirements(
    provenance: RuntimeProvenance,
    requirements: RuntimeRequirements,
) -> None:
    """Raise when collected runtime provenance violates measurement requirements.

    Dependency names are matched case-insensitively after surrounding whitespace
    normalization, while versions use exact string equality. Exact version
    comparison is intentional: a controlled benchmark should not silently accept
    a semantically different environment because a broad version specifier still
    happens to match.
    """

    if not isinstance(provenance, RuntimeProvenance):
        raise TypeError("provenance must be a RuntimeProvenance instance")
    if not isinstance(requirements, RuntimeRequirements):
        raise TypeError("requirements must be a RuntimeRequirements instance")

    expected_revision = requirements.expected_code_revision
    if expected_revision is not None and provenance.code_revision != expected_revision:
        raise ValueError(
            "runtime code revision does not match the required revision: "
            f"expected {expected_revision!r}, got {provenance.code_revision!r}"
        )

    if requirements.require_clean_working_tree and provenance.working_tree_state != CLEAN_STATE:
        raise ValueError(
            "runtime working tree must be clean for measurement: "
            f"got {provenance.working_tree_state!r}"
        )

    installed_versions = {
        package_name.strip().lower(): package_version
        for package_name, package_version in provenance.dependency_versions.items()
    }
    for package_name, required_version in requirements.dependency_versions.items():
        normalized_name = package_name.strip().lower()
        installed_version = installed_versions.get(normalized_name)
        if installed_version is None:
            raise ValueError(
                f"required runtime dependency is not installed: {package_name}=={required_version}"
            )
        if installed_version != required_version:
            raise ValueError(
                "runtime dependency version mismatch: "
                f"{package_name} requires {required_version!r}, got {installed_version!r}"
            )


def _require_non_empty_string(field_name: str, value: object) -> str:
    """Return a validated non-empty string used by a requirement field."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


__all__ = ["RuntimeRequirements", "validate_runtime_requirements"]
