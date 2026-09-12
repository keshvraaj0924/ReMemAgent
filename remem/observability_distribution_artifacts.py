"""Versioned persistence for deterministic observability distributions.

Distribution sidecars intentionally remain separate from the stable
:class:`remem.observability.ObservationSnapshot` schema. This keeps existing
benchmark artifacts backward compatible while allowing fixed-bucket latency
measurements to be archived, verified, and merged independently.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Self

from remem.observability_distributions import ObservationHistogramSnapshot

DISTRIBUTION_OBSERVATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class DistributionObservationSnapshot:
    """Immutable named collection of fixed-bucket histogram snapshots."""

    duration_histograms: Mapping[str, ObservationHistogramSnapshot]

    def __post_init__(self) -> None:
        """Validate metric names and freeze snapshots in deterministic order."""

        normalized: dict[str, ObservationHistogramSnapshot] = {}
        for raw_name, snapshot in self.duration_histograms.items():
            name = _normalize_metric_name(raw_name)
            if name in normalized:
                raise ValueError(f"distribution metric names normalize to the same value: {raw_name!r}")
            if not isinstance(snapshot, ObservationHistogramSnapshot):
                raise TypeError(
                    "duration histogram values must be ObservationHistogramSnapshot instances"
                )
            normalized[name] = snapshot
        object.__setattr__(
            self,
            "duration_histograms",
            MappingProxyType({name: normalized[name] for name in sorted(normalized)}),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a versioned deterministic JSON-compatible representation."""

        return {
            "schema_version": DISTRIBUTION_OBSERVATION_SCHEMA_VERSION,
            "duration_histograms": {
                name: snapshot.to_dict() for name, snapshot in self.duration_histograms.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Validate and reconstruct a persisted distribution sidecar."""

        if not isinstance(payload, Mapping):
            raise TypeError("distribution observation snapshot must be a mapping")
        schema_version = payload.get("schema_version")
        if schema_version != DISTRIBUTION_OBSERVATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported distribution observation schema version: {schema_version!r}"
            )
        raw_histograms = payload.get("duration_histograms")
        if not isinstance(raw_histograms, Mapping):
            raise TypeError("duration_histograms must be a mapping")

        histograms: dict[str, ObservationHistogramSnapshot] = {}
        for raw_name, raw_snapshot in raw_histograms.items():
            if not isinstance(raw_name, str):
                raise TypeError("distribution metric names must be strings")
            if not isinstance(raw_snapshot, Mapping):
                raise TypeError("histogram snapshot payloads must be mappings")
            histograms[raw_name] = _histogram_snapshot_from_dict(raw_snapshot)
        return cls(duration_histograms=histograms)


def write_distribution_observation_snapshot(
    path: str | Path,
    snapshot: DistributionObservationSnapshot,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically persist a distribution sidecar using canonical JSON formatting."""

    if not isinstance(snapshot, DistributionObservationSnapshot):
        raise TypeError("snapshot must be a DistributionObservationSnapshot")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"distribution observation snapshot already exists: {destination}")

    payload = json.dumps(
        snapshot.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"distribution observation snapshot already exists: {destination}")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def read_distribution_observation_snapshot(path: str | Path) -> DistributionObservationSnapshot:
    """Load and validate one persisted distribution sidecar."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("distribution observation snapshot payload must be a mapping")
    return DistributionObservationSnapshot.from_dict(payload)


def _histogram_snapshot_from_dict(payload: Mapping[str, object]) -> ObservationHistogramSnapshot:
    """Reconstruct one histogram while validating persisted derived fields."""

    upper_bounds = payload.get("upper_bounds")
    bucket_counts = payload.get("bucket_counts")
    total = payload.get("total")
    if not isinstance(upper_bounds, list):
        raise TypeError("histogram upper_bounds must be a list")
    if not isinstance(bucket_counts, list):
        raise TypeError("histogram bucket_counts must be a list")
    snapshot = ObservationHistogramSnapshot(
        upper_bounds=tuple(upper_bounds),
        bucket_counts=tuple(bucket_counts),
        total=total,
    )
    if payload.get("count") != snapshot.count:
        raise ValueError("persisted histogram count does not match bucket counts")
    if payload.get("mean") != snapshot.mean:
        raise ValueError("persisted histogram mean does not match total and count")
    return snapshot


def _normalize_metric_name(name: str) -> str:
    """Normalize one distribution metric name."""

    if not isinstance(name, str):
        raise TypeError("distribution metric name must be a string")
    normalized = name.strip()
    if not normalized:
        raise ValueError("distribution metric name must not be empty")
    return normalized


__all__ = [
    "DISTRIBUTION_OBSERVATION_SCHEMA_VERSION",
    "DistributionObservationSnapshot",
    "read_distribution_observation_snapshot",
    "write_distribution_observation_snapshot",
]
