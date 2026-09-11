"""Transactional publication helpers for benchmark artifact bundles.

The filesystem cannot provide a portable multi-path atomic rename. This module
therefore stages every artifact before publication, publishes the designated
commit marker last, and rolls back already-published paths when a later
publication fails. Explicit overwrite mode additionally snapshots pre-existing
files so rollback can restore their exact bytes.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PreparedArtifact:
    """One fully written private artifact awaiting final publication."""

    temporary_path: Path
    destination_path: Path

    def __post_init__(self) -> None:
        if self.temporary_path.resolve() == self.destination_path.resolve():
            raise ValueError("temporary and destination artifact paths must differ")


def create_private_artifact_path(destination_path: Path) -> Path:
    """Create an empty private path beside its eventual destination."""

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination_path.parent,
        prefix=f".{destination_path.name}.",
        suffix=".bundle.tmp",
    )
    os.close(file_descriptor)
    return Path(temporary_name)


def publish_artifact_bundle(
    artifacts: tuple[PreparedArtifact, ...],
    *,
    overwrite: bool,
) -> None:
    """Publish prepared artifacts as one rollback-capable bundle.

    Artifacts are published in tuple order. Callers should therefore place the
    primary report last so its presence acts as the bundle commit marker.
    Every temporary file must already contain its complete durable payload.
    """

    if not artifacts:
        raise ValueError("artifact bundle must contain at least one artifact")
    _validate_artifacts(artifacts)
    backups: dict[Path, Path | None] = {}
    published_destinations: list[Path] = []
    try:
        if overwrite:
            backups = _snapshot_existing_destinations(artifacts)
        for artifact in artifacts:
            _publish_prepared_artifact(artifact, overwrite=overwrite)
            published_destinations.append(artifact.destination_path)
    except BaseException:
        _rollback_publication(
            published_destinations,
            backups=backups,
            overwrite=overwrite,
        )
        raise
    finally:
        _cleanup_paths(tuple(artifact.temporary_path for artifact in artifacts))
        _cleanup_paths(tuple(path for path in backups.values() if path is not None))


def _validate_artifacts(artifacts: tuple[PreparedArtifact, ...]) -> None:
    """Validate private inputs before mutating any destination."""

    destinations: set[Path] = set()
    for artifact in artifacts:
        resolved_destination = artifact.destination_path.resolve()
        if resolved_destination in destinations:
            raise ValueError(f"duplicate artifact destination: {artifact.destination_path}")
        destinations.add(resolved_destination)
        if not artifact.temporary_path.is_file():
            raise FileNotFoundError(f"prepared artifact does not exist: {artifact.temporary_path}")


def _publish_prepared_artifact(artifact: PreparedArtifact, *, overwrite: bool) -> None:
    """Publish one prepared file while honoring the requested overwrite policy."""

    destination = artifact.destination_path
    if overwrite:
        os.replace(artifact.temporary_path, destination)
        return
    try:
        os.link(artifact.temporary_path, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            f"benchmark artifact already exists: {destination}; pass --overwrite to replace it"
        ) from exc
    artifact.temporary_path.unlink()


def _snapshot_existing_destinations(
    artifacts: tuple[PreparedArtifact, ...],
) -> dict[Path, Path | None]:
    """Copy pre-existing destinations to private rollback files."""

    backups: dict[Path, Path | None] = {}
    try:
        for artifact in artifacts:
            destination = artifact.destination_path
            if not destination.exists():
                backups[destination] = None
                continue
            backup_path = create_private_artifact_path(destination)
            try:
                shutil.copyfile(destination, backup_path)
            except BaseException:
                backup_path.unlink(missing_ok=True)
                raise
            backups[destination] = backup_path
    except BaseException:
        _cleanup_paths(tuple(path for path in backups.values() if path is not None))
        raise
    return backups


def _rollback_publication(
    published_destinations: list[Path],
    *,
    backups: dict[Path, Path | None],
    overwrite: bool,
) -> None:
    """Best-effort rollback without masking the original publication failure."""

    for destination in reversed(published_destinations):
        try:
            backup_path = backups.get(destination) if overwrite else None
            if backup_path is None:
                destination.unlink(missing_ok=True)
            else:
                os.replace(backup_path, destination)
        except OSError:
            # The triggering publication exception remains the causal signal.
            # A crash-safe journal is intentionally outside this helper's scope.
            continue


def _cleanup_paths(paths: tuple[Path, ...]) -> None:
    """Remove private staging files without disturbing completed publication."""

    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


__all__ = [
    "PreparedArtifact",
    "create_private_artifact_path",
    "publish_artifact_bundle",
]
