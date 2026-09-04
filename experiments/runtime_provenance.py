"""Runtime metadata used to make measured experiments auditable."""

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
UNKNOWN_VALUE = "unknown"
CLEAN_STATE = "clean"
DIRTY_STATE = "dirty"
VALID_WORKING_TREE_STATES = frozenset({CLEAN_STATE, DIRTY_STATE, UNKNOWN_VALUE})


@dataclass(frozen=True, slots=True)
class RuntimeProvenance:
    """Environment metadata captured alongside a measured experiment."""

    code_revision: str
    working_tree_state: str
    python_version: str
    platform: str
    package_version: str
    dependency_fingerprint: str
    dependency_versions: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return asdict(self)


def collect_runtime_provenance(
    *,
    repository_path: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> RuntimeProvenance:
    """Collect code and runtime metadata without requiring benchmark packages.

    ``REMEM_GIT_COMMIT`` can be supplied by CI or a container build. When it is
    absent, a repository checkout is inspected with ``git rev-parse``. Failure
    to resolve a revision is represented explicitly as ``"unknown"`` rather
    than inventing a revision.

    ``REMEM_GIT_STATE`` may be supplied by a controlled execution environment.
    Otherwise the checkout is inspected with ``git status --porcelain``. An
    unknown state is retained explicitly when the repository cannot be probed.
    Invalid explicit states are rejected rather than being persisted as if they
    were authoritative provenance.
    """

    environment_values = environment or {}
    repository = repository_path or Path.cwd()
    code_revision = environment_values.get("REMEM_GIT_COMMIT") or _git_revision(repository)
    working_tree_state = _resolve_working_tree_state(environment_values, repository)
    dependency_versions = _dependency_versions()
    return RuntimeProvenance(
        code_revision=code_revision,
        working_tree_state=working_tree_state,
        python_version=platform.python_version(),
        platform=platform.platform(),
        package_version=_package_version(),
        dependency_fingerprint=_dependency_fingerprint(dependency_versions),
        dependency_versions=dependency_versions,
    )


def _resolve_working_tree_state(
    environment: Mapping[str, str],
    repository_path: Path,
) -> str:
    """Resolve an explicit or checkout-derived working-tree state."""

    explicit_state = environment.get("REMEM_GIT_STATE")
    if explicit_state is not None:
        if explicit_state not in VALID_WORKING_TREE_STATES:
            raise ValueError(
                "REMEM_GIT_STATE must be one of: "
                f"{', '.join(sorted(VALID_WORKING_TREE_STATES))}"
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

    dependency_versions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in distributions()
        if distribution.metadata.get("Name")
    }
    return dict(sorted(dependency_versions.items(), key=lambda item: item[0].lower()))


def _dependency_fingerprint(dependency_versions: Mapping[str, str]) -> str:
    """Hash normalized dependency metadata for compact reproducibility checks."""

    payload = json.dumps(
        sorted(dependency_versions.items(), key=lambda item: item[0].lower()),
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
