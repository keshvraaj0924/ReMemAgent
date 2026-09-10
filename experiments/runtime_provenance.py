from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, distributions, version
from pathlib import Path
from typing import Mapping

PACKAGE_NAME = "rememagent"
RUNTIME_PROVENANCE_SCHEMA_VERSION = 1
UNKNOWN_VALUE = "unknown"
CLEAN_STATE = "clean"
DIRTY_STATE = "dirty"
VALID_WORKING_TREE_STATES = frozenset({CLEAN_STATE, DIRTY_STATE, UNKNOWN_VALUE})
SHA256_HEX_LENGTH = 64


@dataclass(frozen=True, slots=True)
class RuntimeProvenance:
    """Immutable, validated environment metadata for a measured experiment."""

    schema_version: int
    code_revision: str
    working_tree_state: str
    python_version: str
    platform: str
    package_version: str
    dependency_fingerprint: str
    dependency_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validate and detach provenance metadata at the domain boundary."""

        if not isinstance(self.schema_version, int) or isinstance(self.schema_version, bool):
            raise TypeError("schema_version must be an integer")
        if self.schema_version != RUNTIME_PROVENANCE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported runtime provenance schema version: {self.schema_version}"
            )
        _require_non_empty_string("code_revision", self.code_revision)
        if self.working_tree_state not in VALID_WORKING_TREE_STATES:
            raise ValueError(
                f"working_tree_state must be one of: {', '.join(sorted(VALID_WORKING_TREE_STATES))}"
            )
        for field_name, value in (
            ("python_version", self.python_version),
            ("platform", self.platform),
            ("package_version", self.package_version),
        ):
            _require_non_empty_string(field_name, value)
        _validate_sha256("dependency_fingerprint", self.dependency_fingerprint)
        if not isinstance(self.dependency_versions, Mapping):
            raise TypeError("dependency_versions must be a mapping")

        detached_versions: dict[str, str] = {}
        normalized_names: dict[str, str] = {}
        for name, dependency_version in self.dependency_versions.items():
            _require_non_empty_string("dependency name", name)
            normalized_name = name.strip()
            if not normalized_name:
                raise ValueError("dependency name must not be whitespace-only")
            normalized_key = normalized_name.lower()
            existing_name = normalized_names.get(normalized_key)
            if existing_name is not None:
                raise ValueError(
                    "dependency names must be unique after whitespace and case normalization"
                )
            _require_non_empty_string(
                f"dependency version for {normalized_name!r}", dependency_version
            )
            normalized_names[normalized_key] = normalized_name
            detached_versions[normalized_name] = dependency_version
        normalized_versions = dict(
            sorted(detached_versions.items(), key=lambda item: item[0].lower())
        )
        object.__setattr__(self, "dependency_versions", normalized_versions)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation preserving dependency metadata."""

        payload = asdict(self)
        payload["dependency_versions"] = dict(self.dependency_versions)
        return payload


def collect_runtime_provenance(
    *,
    repository_path: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> RuntimeProvenance:
    """Collect code and runtime metadata without requiring benchmark packages."""

    environment_values = environment or {}
    repository = repository_path or Path.cwd()
    code_revision = environment_values.get("REMEM_GIT_COMMIT") or _git_revision(repository)
    working_tree_state = _resolve_working_tree_state(environment_values, repository)
    dependency_versions = _dependency_versions()
    return RuntimeProvenance(
        schema_version=RUNTIME_PROVENANCE_SCHEMA_VERSION,
        code_revision=code_revision,
        working_tree_state=working_tree_state,
        python_version=platform.python_version(),
        platform=platform.platform(),
        package_version=_package_version(),
        dependency_fingerprint=_dependency_fingerprint(dependency_versions),
        dependency_versions=dependency_versions,
    )


def _resolve_working_tree_state(environment: Mapping[str, str], repository_path: Path) -> str:
    """Resolve an explicit or checkout-derived working-tree state."""

    explicit_state = environment.get("REMEM_GIT_STATE")
    if explicit_state is not None:
        if explicit_state not in VALID_WORKING_TREE_STATES:
            raise ValueError(
                f"REMEM_GIT_STATE must be one of: {', '.join(sorted(VALID_WORKING_TREE_STATES))}"
            )
        return explicit_state
    return _git_working_tree_state(repository_path)


def _git_revision(repository_path: Path) -> str:
    """Resolve the current Git revision, returning an explicit unknown value on failure."""

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
    """Return clean, dirty, or unknown for the checkout working tree."""

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


def _package_version() -> str:
    """Return the installed package version or an explicit unknown value."""

    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return UNKNOWN_VALUE


def _dependency_versions() -> dict[str, str]:
    """Return installed distribution versions in deterministic name order."""

    dependency_versions: dict[str, str] = {}
    for distribution in distributions():
        package_name = distribution.metadata["Name"]
        if package_name:
            dependency_versions[package_name] = distribution.version
    return dict(sorted(dependency_versions.items(), key=lambda item: item[0].lower()))


def _dependency_fingerprint(dependency_versions: Mapping[str, str]) -> str:
    """Hash normalized dependency metadata for compact reproducibility checks."""

    payload = json.dumps(
        sorted(dependency_versions.items(), key=lambda item: item[0].lower()),
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_non_empty_string(field_name: str, value: object) -> str:
    """Return a validated string suitable for stable provenance fields."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _validate_sha256(field_name: str, value: object) -> None:
    """Validate a canonical lowercase-or-uppercase SHA-256 hexadecimal digest."""

    normalized_value = _require_non_empty_string(field_name, value)
    is_hex = all(character in "0123456789abcdefABCDEF" for character in normalized_value)
    if len(normalized_value) != SHA256_HEX_LENGTH or not is_hex:
        raise ValueError(f"{field_name} must be a 64-character hexadecimal SHA-256 digest")


__all__ = [
    "CLEAN_STATE",
    "DIRTY_STATE",
    "RuntimeProvenance",
    "RUNTIME_PROVENANCE_SCHEMA_VERSION",
    "UNKNOWN_VALUE",
    "VALID_WORKING_TREE_STATES",
    "collect_runtime_provenance",
]
