"""Rollback-safe publication for benchmark observability artifact bundles.

This module extends the existing benchmark bundle semantics with an optional
fixed-bucket distribution sidecar while keeping the stable benchmark report and
aggregate observability schemas unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from experiments.artifact_bundle import (
    PreparedArtifact,
    create_private_artifact_path,
    publish_artifact_bundle,
)
from experiments.benchmark_manifest import save_benchmark_artifact_manifest
from remem.observability import ObservationSnapshot, write_observation_snapshot
from remem.observability_distribution_artifacts import (
    DistributionObservationSnapshot,
    write_distribution_observation_snapshot,
)

ManifestWriter = Callable[[Path, Path], object]


def persist_benchmark_observability_bundle(
    output_path: Path,
    *,
    overwrite: bool,
    report_writer: Callable[[Path], object],
    manifest_path: Path | None = None,
    observability_path: Path | None = None,
    observation_snapshot: ObservationSnapshot | None = None,
    distribution_path: Path | None = None,
    distribution_snapshot: DistributionObservationSnapshot | None = None,
    manifest_writer: Callable[..., object] = save_benchmark_artifact_manifest,
) -> Path:
    """Stage and publish a report and its optional observability sidecars.

    The report is always published last and therefore remains the bundle commit
    marker. Any publication failure rolls back already-published auxiliary
    artifacts through :func:`publish_artifact_bundle`.

    ``manifest_writer`` remains injectable so CLI callers and tests can preserve
    the established manifest-writing seam while sharing this transactional path.
    """

    _validate_optional_pair(
        artifact_path=observability_path,
        snapshot=observation_snapshot,
        artifact_name="observability",
    )
    _validate_optional_pair(
        artifact_path=distribution_path,
        snapshot=distribution_snapshot,
        artifact_name="distribution observability",
    )
    _validate_distinct_destinations(
        output_path,
        manifest_path,
        observability_path,
        distribution_path,
    )

    prepared_artifacts: list[PreparedArtifact] = []
    report_artifact = _prepare_artifact(output_path, prepared_artifacts)
    try:
        report_writer(report_artifact.temporary_path)
        ordered_artifacts: list[PreparedArtifact] = []

        if observability_path is not None and observation_snapshot is not None:
            observability_artifact = _prepare_artifact(
                observability_path,
                prepared_artifacts,
            )
            write_observation_snapshot(
                observability_artifact.temporary_path,
                observation_snapshot,
                overwrite=True,
            )
            ordered_artifacts.append(observability_artifact)

        if distribution_path is not None and distribution_snapshot is not None:
            distribution_artifact = _prepare_artifact(
                distribution_path,
                prepared_artifacts,
            )
            write_distribution_observation_snapshot(
                distribution_artifact.temporary_path,
                distribution_snapshot,
                overwrite=True,
            )
            ordered_artifacts.append(distribution_artifact)

        if manifest_path is not None:
            manifest_artifact = _prepare_artifact(manifest_path, prepared_artifacts)
            manifest_writer(
                report_artifact.temporary_path,
                manifest_artifact.temporary_path,
                overwrite=True,
            )
            ordered_artifacts.append(manifest_artifact)

        ordered_artifacts.append(report_artifact)
        publish_artifact_bundle(tuple(ordered_artifacts), overwrite=overwrite)
    except BaseException:
        for artifact in prepared_artifacts:
            artifact.temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def _prepare_artifact(
    destination_path: Path,
    prepared_artifacts: list[PreparedArtifact],
) -> PreparedArtifact:
    """Create and track one private artifact path for exception cleanup."""

    artifact = PreparedArtifact(
        create_private_artifact_path(destination_path),
        destination_path,
    )
    prepared_artifacts.append(artifact)
    return artifact


def _validate_optional_pair(
    *,
    artifact_path: Path | None,
    snapshot: object | None,
    artifact_name: str,
) -> None:
    """Require optional artifact destinations and snapshots as complete pairs."""

    if (artifact_path is None) != (snapshot is None):
        raise ValueError(
            f"{artifact_name}_path and {artifact_name}_snapshot must either both be provided "
            "or both omitted"
        )


def _validate_distinct_destinations(*paths: Path | None) -> None:
    """Reject path aliases before any private artifacts are created."""

    resolved_paths: set[Path] = set()
    for path in paths:
        if path is None:
            continue
        resolved_path = path.resolve()
        if resolved_path in resolved_paths:
            raise ValueError("benchmark bundle artifact destinations must be distinct")
        resolved_paths.add(resolved_path)


__all__ = ["persist_benchmark_observability_bundle"]
