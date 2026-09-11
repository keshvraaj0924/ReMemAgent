"""Git checkout provenance for source-installed benchmark dependencies.

Package metadata is insufficient for research dependencies that are installed
from source or executed directly from a checkout. This module records the exact
Git revision and working-tree state for named external repositories, validates
those observations against explicit requirements before measurement, and
provides canonical fingerprints for persisted reproducibility evidence.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from experiments.runtime_provenance import (
    CLEAN_STATE,
    DIRTY_STATE,
    UNKNOWN_VALUE,
    VALID_WORKING_TREE_STATES,
)

SOURCE_CHECKOUT_REQUIREMENTS_SCHEMA_VERSION = 1
SOURCE_CHECKOUT_PROVENANCE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SourceCheckoutRequirement:
    """Required Git state for one source-installed benchmark dependency."""

    expected_revision: str
    require_clean_working_tree: bool = True

    def __post_init__(self) -> None:
        """Reject ambiguous source requirements before any checkout is inspected."""

        _require_non_empty_string("expected_revision", self.expected_revision)
        if not isinstance(self.require_clean_working_tree, bool):
            raise TypeError("require_clean_working_tree must be a boolean")

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-compatible requirement representation."""

        return {
            "expected_revision": self.expected_revision,
            "require_clean_working_tree": self.require_clean_working_tree,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SourceCheckoutRequirement:
        """Reconstruct one requirement from its exact persisted schema."""

        if not isinstance(payload, Mapping):
            raise TypeError("source checkout requirement must be a mapping")
        expected_fields = {"expected_revision", "require_clean_working_tree"}
        if set(payload) != expected_fields:
            raise ValueError("source checkout requirement must use the exact persisted schema")
        return cls(
            expected_revision=payload["expected_revision"],
            require_clean_working_tree=payload["require_clean_working_tree"],
        )


@dataclass(frozen=True, slots=True)
class SourceCheckoutProvenance:
    """Observed Git state for one named external source checkout."""

    revision: str
    working_tree_state: str

    def __post_init__(self) -> None:
        """Validate observed source metadata at the collection boundary."""

        _require_non_empty_string("revision", self.revision)
        if self.working_tree_state not in VALID_WORKING_TREE_STATES:
            raise ValueError(
                f"working_tree_state must be one of: {', '.join(sorted(VALID_WORKING_TREE_STATES))}"
            )

    def to_dict(self) -> dict[str, str]:
        """Return a detached JSON-compatible representation."""

        return {
            "revision": self.revision,
            "working_tree_state": self.working_tree_state,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SourceCheckoutProvenance:
        """Reconstruct one observed checkout from its exact persisted schema."""

        if not isinstance(payload, Mapping):
            raise TypeError("source checkout provenance must be a mapping")
        expected_fields = {"revision", "working_tree_state"}
        if set(payload) != expected_fields:
            raise ValueError("source checkout provenance must use the exact persisted schema")
        return cls(
            revision=payload["revision"],
            working_tree_state=payload["working_tree_state"],
        )


def collect_source_checkout_provenance(
    repositories: Mapping[str, Path],
) -> Mapping[str, SourceCheckoutProvenance]:
    """Collect deterministic provenance for named source repositories.

    Repository names are normalized for uniqueness but preserved for persisted
    display. Git failures are represented explicitly as ``unknown`` rather than
    guessed. Controlled validation can therefore fail closed when an expected
    revision cannot be established.
    """

    if not isinstance(repositories, Mapping):
        raise TypeError("repositories must be a mapping")

    normalized_names: dict[str, str] = {}
    collected: dict[str, SourceCheckoutProvenance] = {}
    for repository_name, repository_path in repositories.items():
        validated_name = _validated_repository_name(repository_name)
        normalized_key = validated_name.lower()
        existing_name = normalized_names.get(normalized_key)
        if existing_name is not None:
            raise ValueError(
                "repository names must be unique after whitespace and case normalization: "
                f"{existing_name!r} and {validated_name!r}"
            )
        if not isinstance(repository_path, Path):
            raise TypeError(f"repository path for {validated_name!r} must be a pathlib.Path")

        normalized_names[normalized_key] = validated_name
        collected[validated_name] = SourceCheckoutProvenance(
            revision=_git_revision(repository_path),
            working_tree_state=_git_working_tree_state(repository_path),
        )

    ordered = dict(sorted(collected.items(), key=lambda item: item[0].lower()))
    return MappingProxyType(ordered)


def validate_source_checkout_requirements(
    provenance: Mapping[str, SourceCheckoutProvenance],
    requirements: Mapping[str, SourceCheckoutRequirement],
) -> None:
    """Fail closed when observed source checkouts violate declared requirements."""

    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")
    if not isinstance(requirements, Mapping):
        raise TypeError("requirements must be a mapping")

    observed = _normalize_provenance(provenance)
    normalized_requirements = _normalize_requirements(requirements)

    for normalized_name, (display_name, requirement) in normalized_requirements.items():
        source_state = observed.get(normalized_name)
        if source_state is None:
            raise ValueError(f"required source checkout was not collected: {display_name}")
        if source_state.revision != requirement.expected_revision:
            raise ValueError(
                "source checkout revision mismatch: "
                f"{display_name} requires {requirement.expected_revision!r}, "
                f"got {source_state.revision!r}"
            )
        if (
            requirement.require_clean_working_tree
            and source_state.working_tree_state != CLEAN_STATE
        ):
            raise ValueError(
                "source checkout working tree must be clean: "
                f"{display_name} is {source_state.working_tree_state!r}"
            )


def source_checkout_requirements_to_dict(
    requirements: Mapping[str, SourceCheckoutRequirement],
) -> dict[str, object]:
    """Return the canonical schema-versioned source admission contract."""

    normalized = _normalize_requirements(requirements)
    repositories = {
        normalized_name: requirement.to_dict()
        for normalized_name, (_, requirement) in sorted(normalized.items())
    }
    return {
        "schema_version": SOURCE_CHECKOUT_REQUIREMENTS_SCHEMA_VERSION,
        "repositories": repositories,
    }


def source_checkout_requirements_from_dict(
    payload: Mapping[str, Any],
) -> Mapping[str, SourceCheckoutRequirement]:
    """Reconstruct an immutable source admission contract from persisted JSON."""

    repositories = _validated_persisted_repository_mapping(
        payload,
        expected_schema_version=SOURCE_CHECKOUT_REQUIREMENTS_SCHEMA_VERSION,
        field_name="source checkout requirements",
    )
    restored: dict[str, SourceCheckoutRequirement] = {}
    for repository_name, requirement_payload in repositories.items():
        normalized_name = _validated_repository_name(repository_name).lower()
        if normalized_name in restored:
            raise ValueError("persisted source checkout requirement names must be unique")
        restored[normalized_name] = SourceCheckoutRequirement.from_dict(requirement_payload)
    return MappingProxyType(dict(sorted(restored.items())))


def source_checkout_provenance_to_dict(
    provenance: Mapping[str, SourceCheckoutProvenance],
) -> dict[str, object]:
    """Return a canonical schema-versioned observed source snapshot."""

    normalized = _normalize_provenance(provenance)
    repositories = {
        normalized_name: source_state.to_dict()
        for normalized_name, source_state in sorted(normalized.items())
    }
    return {
        "schema_version": SOURCE_CHECKOUT_PROVENANCE_SCHEMA_VERSION,
        "repositories": repositories,
    }


def source_checkout_provenance_from_dict(
    payload: Mapping[str, Any],
) -> Mapping[str, SourceCheckoutProvenance]:
    """Reconstruct an immutable observed source snapshot from persisted JSON."""

    repositories = _validated_persisted_repository_mapping(
        payload,
        expected_schema_version=SOURCE_CHECKOUT_PROVENANCE_SCHEMA_VERSION,
        field_name="source checkout provenance",
    )
    restored: dict[str, SourceCheckoutProvenance] = {}
    for repository_name, provenance_payload in repositories.items():
        normalized_name = _validated_repository_name(repository_name).lower()
        if normalized_name in restored:
            raise ValueError("persisted source checkout provenance names must be unique")
        restored[normalized_name] = SourceCheckoutProvenance.from_dict(provenance_payload)
    return MappingProxyType(dict(sorted(restored.items())))


def source_checkout_requirements_sha256(
    requirements: Mapping[str, SourceCheckoutRequirement],
) -> str:
    """Return a deterministic SHA-256 digest for a source admission contract."""

    return _canonical_sha256(source_checkout_requirements_to_dict(requirements))


def source_checkout_provenance_sha256(
    provenance: Mapping[str, SourceCheckoutProvenance],
) -> str:
    """Return a deterministic SHA-256 digest for an observed source snapshot."""

    return _canonical_sha256(source_checkout_provenance_to_dict(provenance))


def _normalize_provenance(
    provenance: Mapping[str, SourceCheckoutProvenance],
) -> dict[str, SourceCheckoutProvenance]:
    """Validate and normalize observed checkout names for comparison."""

    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")
    normalized: dict[str, SourceCheckoutProvenance] = {}
    for repository_name, source_state in provenance.items():
        validated_name = _validated_repository_name(repository_name)
        normalized_name = validated_name.lower()
        if normalized_name in normalized:
            raise ValueError(
                "provenance repository names must be unique after whitespace and case normalization"
            )
        if not isinstance(source_state, SourceCheckoutProvenance):
            raise TypeError(
                f"provenance for {validated_name!r} must be a SourceCheckoutProvenance instance"
            )
        normalized[normalized_name] = source_state
    return normalized


def _normalize_requirements(
    requirements: Mapping[str, SourceCheckoutRequirement],
) -> dict[str, tuple[str, SourceCheckoutRequirement]]:
    """Validate and normalize source requirement names for comparison."""

    if not isinstance(requirements, Mapping):
        raise TypeError("requirements must be a mapping")
    normalized: dict[str, tuple[str, SourceCheckoutRequirement]] = {}
    for repository_name, requirement in requirements.items():
        validated_name = _validated_repository_name(repository_name)
        normalized_name = validated_name.lower()
        if normalized_name in normalized:
            raise ValueError(
                "requirement repository names must be unique after whitespace and case normalization"
            )
        if not isinstance(requirement, SourceCheckoutRequirement):
            raise TypeError(
                f"requirement for {validated_name!r} must be a SourceCheckoutRequirement instance"
            )
        normalized[normalized_name] = (validated_name, requirement)
    return normalized


def _validated_persisted_repository_mapping(
    payload: Mapping[str, Any],
    *,
    expected_schema_version: int,
    field_name: str,
) -> Mapping[str, Mapping[str, Any]]:
    """Validate the shared schema envelope used by persisted source metadata."""

    if not isinstance(payload, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    if set(payload) != {"schema_version", "repositories"}:
        raise ValueError(f"{field_name} must use the exact persisted schema")
    if payload["schema_version"] != expected_schema_version:
        raise ValueError(f"unsupported {field_name} schema version")
    repositories = payload["repositories"]
    if not isinstance(repositories, Mapping):
        raise TypeError(f"{field_name} repositories must be a mapping")
    if not repositories:
        raise ValueError(f"{field_name} repositories must not be empty")
    for repository_name, repository_payload in repositories.items():
        _validated_repository_name(repository_name)
        if not isinstance(repository_payload, Mapping):
            raise TypeError(f"{field_name} repository entries must be mappings")
    return repositories


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible source metadata."""

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _git_revision(repository_path: Path) -> str:
    """Return the current Git revision or an explicit unknown value on failure."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_VALUE
    revision = result.stdout.strip()
    return revision or UNKNOWN_VALUE


def _git_working_tree_state(repository_path: Path) -> str:
    """Return clean, dirty, or unknown for one source checkout."""

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_VALUE
    return DIRTY_STATE if result.stdout else CLEAN_STATE


def _validated_repository_name(value: object) -> str:
    """Return one stripped, non-empty source repository display name."""

    name = _require_non_empty_string("repository name", value).strip()
    if not name:
        raise ValueError("repository name must not be whitespace-only")
    return name


def _require_non_empty_string(field_name: str, value: object) -> str:
    """Return a validated non-empty string."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


__all__ = [
    "SOURCE_CHECKOUT_PROVENANCE_SCHEMA_VERSION",
    "SOURCE_CHECKOUT_REQUIREMENTS_SCHEMA_VERSION",
    "SourceCheckoutProvenance",
    "SourceCheckoutRequirement",
    "collect_source_checkout_provenance",
    "source_checkout_provenance_from_dict",
    "source_checkout_provenance_sha256",
    "source_checkout_provenance_to_dict",
    "source_checkout_requirements_from_dict",
    "source_checkout_requirements_sha256",
    "source_checkout_requirements_to_dict",
    "validate_source_checkout_requirements",
]
