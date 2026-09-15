"""Validated runtime provenance for reproducible benchmark artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

RUNTIME_PROVENANCE_SCHEMA_VERSION = 1
CLEAN_STATE = "clean"
DIRTY_STATE = "dirty"
_VALID_WORKING_TREE_STATES = frozenset({CLEAN_STATE, DIRTY_STATE})


@dataclass(frozen=True, slots=True)
class RuntimeProvenance:
    """Immutable description of the runtime used to produce an experiment."""

    schema_version: int
    code_revision: str
    working_tree_state: str
    python_version: str
    platform: str
    package_version: str
    dependency_fingerprint: str
    dependency_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_PROVENANCE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {RUNTIME_PROVENANCE_SCHEMA_VERSION}")
        for field_name in (
            "code_revision",
            "python_version",
            "platform",
            "package_version",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.working_tree_state not in _VALID_WORKING_TREE_STATES:
            raise ValueError("working_tree_state must be 'clean' or 'dirty'")
        if len(self.dependency_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.dependency_fingerprint.lower()
        ):
            raise ValueError("dependency_fingerprint must be a 64-character hex digest")

        dependencies: dict[str, str] = {}
        for name, version in self.dependency_versions.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("dependency names must be non-empty strings")
            if not isinstance(version, str) or not version.strip():
                raise ValueError("dependency versions must be non-empty strings")
            dependencies[name] = version
        object.__setattr__(
            self,
            "dependency_versions",
            MappingProxyType(dict(sorted(dependencies.items()))),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "code_revision": self.code_revision,
            "working_tree_state": self.working_tree_state,
            "python_version": self.python_version,
            "platform": self.platform,
            "package_version": self.package_version,
            "dependency_fingerprint": self.dependency_fingerprint,
            "dependency_versions": dict(self.dependency_versions),
        }


__all__ = [
    "CLEAN_STATE",
    "DIRTY_STATE",
    "RUNTIME_PROVENANCE_SCHEMA_VERSION",
    "RuntimeProvenance",
]
