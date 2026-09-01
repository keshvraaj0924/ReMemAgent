"""Runtime metadata used to make measured experiments auditable."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Mapping

PACKAGE_NAME = "rememagent"
UNKNOWN_VALUE = "unknown"


@dataclass(frozen=True, slots=True)
class RuntimeProvenance:
    """Environment metadata captured alongside a measured experiment."""

    code_revision: str
    python_version: str
    platform: str
    package_version: str

    def to_dict(self) -> dict[str, str]:
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
    """

    environment_values = environment or {}
    code_revision = environment_values.get("REMEM_GIT_COMMIT") or _git_revision(
        repository_path or Path.cwd()
    )
    return RuntimeProvenance(
        code_revision=code_revision,
        python_version=platform.python_version(),
        platform=platform.platform(),
        package_version=_package_version(),
    )


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


def _package_version() -> str:
    """Return the installed package version or an explicit unknown value."""

    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return UNKNOWN_VALUE
