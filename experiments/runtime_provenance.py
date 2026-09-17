"""Validated runtime provenance for reproducible benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

RUNTIME_PROVENANCE_SCHEMA_VERSION = 1
CLEAN_STATE = "clean"
DIRTY_STATE = "dirty"
_VALID_WORKING_TREE_STATES = frozenset({CLEAN_STATE, DIRTY_STATE})
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "code_revision",
        "working_tree_state",
        "python_version",
        "platform",
        "package_version",
        "dependency_fingerprint",
        "dependency_versions",
    }
)


def _require_non_empty_string(value: object, field_name: str) -> str:
    """Validate and return a non-empty string field."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def dependency_fingerprint(dependency_versions: Mapping[str, str]) -> str:
    """Return a deterministic SHA-256 fingerprint for dependency versions."""
    normalized_dependencies: dict[str, str] = {}
    for name, version in dependency_versions.items():
        normalized_name = _require_non_empty_string(name, "dependency names")
        normalized_version = _require_non_empty_string(version, "dependency versions")
        normalized_dependencies[normalized_name] = normalized_version

    payload = json.dumps(
        dict(sorted(normalized_dependencies.items())),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


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
        if not isinstance(self.schema_version, int) or isinstance(self.schema_version, bool):
            raise TypeError("schema_version must be an integer")
        if self.schema_version != RUNTIME_PROVENANCE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {RUNTIME_PROVENANCE_SCHEMA_VERSION}")
        for field_name in (
            "code_revision",
            "working_tree_state",
            "python_version",
            "platform",
            "package_version",
        ):
            _require_non_empty_string(getattr(self, field_name), field_name)
        if self.working_tree_state not in _VALID_WORKING_TREE_STATES:
            raise ValueError("working_tree_state must be 'clean' or 'dirty'")
        _require_non_empty_string(self.dependency_fingerprint, "dependency_fingerprint")
        if len(self.dependency_fingerprint) != 64:
            raise ValueError("dependency_fingerprint must be a 64-character hex digest")
        if any(
            character not in "0123456789abcdef" for character in self.dependency_fingerprint.lower()
        ):
            raise ValueError("dependency_fingerprint must be a 64-character hex digest")
        if not isinstance(self.dependency_versions, Mapping):
            raise TypeError("dependency_versions must be a mapping")

        dependencies: dict[str, str] = {}
        for name, version in self.dependency_versions.items():
            normalized_name = _require_non_empty_string(name, "dependency names")
            normalized_version = _require_non_empty_string(version, "dependency versions")
            dependencies[normalized_name] = normalized_version
        object.__setattr__(
            self, "dependency_versions", MappingProxyType(dict(sorted(dependencies.items())))
        )

        expected_fingerprint = dependency_fingerprint(dependencies)
        if self.dependency_fingerprint.lower() != expected_fingerprint:
            raise ValueError("dependency_fingerprint does not match dependency_versions")

    @classmethod
    def create(
        cls,
        *,
        code_revision: str,
        working_tree_state: str,
        python_version: str,
        platform: str,
        package_version: str,
        dependency_versions: Mapping[str, str],
    ) -> RuntimeProvenance:
        """Create provenance with a fingerprint derived from dependency versions."""
        return cls(
            schema_version=RUNTIME_PROVENANCE_SCHEMA_VERSION,
            code_revision=code_revision,
            working_tree_state=working_tree_state,
            python_version=python_version,
            platform=platform,
            package_version=package_version,
            dependency_fingerprint=dependency_fingerprint(dependency_versions),
            dependency_versions=dependency_versions,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeProvenance:
        """Deserialize and fully verify an untrusted provenance mapping."""
        if not isinstance(payload, Mapping):
            raise TypeError("runtime provenance must be a mapping")
        payload_fields = set(payload)
        unknown_fields = payload_fields - _PROVENANCE_FIELDS
        missing_fields = _PROVENANCE_FIELDS - payload_fields
        if unknown_fields:
            raise ValueError(f"unknown runtime provenance fields: {sorted(unknown_fields)}")
        if missing_fields:
            raise ValueError(f"missing runtime provenance fields: {sorted(missing_fields)}")

        schema_version = payload["schema_version"]
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise TypeError("schema_version must be an integer")
        dependency_versions = payload["dependency_versions"]
        if not isinstance(dependency_versions, Mapping):
            raise TypeError("dependency_versions must be a mapping")

        string_fields = {
            field_name: payload[field_name]
            for field_name in _PROVENANCE_FIELDS - {"schema_version", "dependency_versions"}
        }
        for field_name, value in string_fields.items():
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")

        return cls(
            schema_version=schema_version,
            code_revision=string_fields["code_revision"],
            working_tree_state=string_fields["working_tree_state"],
            python_version=string_fields["python_version"],
            platform=string_fields["platform"],
            package_version=string_fields["package_version"],
            dependency_fingerprint=string_fields["dependency_fingerprint"],
            dependency_versions=dependency_versions,
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

    def fingerprint(self) -> str:
        """Return a stable SHA-256 identity for the complete provenance artifact."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "CLEAN_STATE",
    "DIRTY_STATE",
    "RUNTIME_PROVENANCE_SCHEMA_VERSION",
    "RuntimeProvenance",
    "dependency_fingerprint",
]
