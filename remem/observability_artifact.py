"""Durable, integrity-checked persistence for observability snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from remem.observability import MetricSnapshot

_SCHEMA_VERSION = 1
_REQUIRED_FIELDS = frozenset({"schema_version", "snapshot", "snapshot_sha256"})


@dataclass(frozen=True, slots=True)
class ObservabilityArtifact:
    """Versioned metric snapshot with a content-integrity digest."""

    snapshot: MetricSnapshot
    snapshot_sha256: str
    schema_version: int = _SCHEMA_VERSION

    @classmethod
    def create(cls, snapshot: MetricSnapshot) -> Self:
        """Create and verify an artifact from a validated metric snapshot."""
        if not isinstance(snapshot, MetricSnapshot):
            raise TypeError("snapshot must be a MetricSnapshot")
        validated_snapshot = MetricSnapshot.from_dict(snapshot.to_dict())
        artifact = cls(
            snapshot=validated_snapshot,
            snapshot_sha256=_snapshot_digest(validated_snapshot),
        )
        artifact.verify()
        return artifact

    def verify(self) -> None:
        """Fail closed when schema, snapshot, or digest integrity is invalid."""
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("schema_version must be an integer")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported observability schema version: {self.schema_version}")
        if not isinstance(self.snapshot, MetricSnapshot):
            raise TypeError("snapshot must be a MetricSnapshot")
        validated_snapshot = MetricSnapshot.from_dict(self.snapshot.to_dict())
        if not isinstance(self.snapshot_sha256, str):
            raise TypeError("snapshot_sha256 must be a string")
        expected_digest = _snapshot_digest(validated_snapshot)
        if self.snapshot_sha256 != expected_digest:
            raise ValueError("observability snapshot digest mismatch")

    def to_json(self) -> str:
        """Serialize the verified artifact deterministically."""
        self.verify()
        payload = {
            "schema_version": self.schema_version,
            "snapshot": self.snapshot.to_dict(),
            "snapshot_sha256": self.snapshot_sha256,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> Self:
        """Deserialize a strict, integrity-checked observability artifact."""
        if not isinstance(payload, str):
            raise TypeError("observability artifact payload must be a string")
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("observability artifact payload must contain valid JSON") from error
        if not isinstance(raw, dict):
            raise TypeError("observability artifact JSON root must be an object")
        _validate_fields(raw)
        snapshot_payload = raw["snapshot"]
        if not isinstance(snapshot_payload, Mapping):
            raise TypeError("snapshot must be a mapping")
        artifact = cls(
            schema_version=raw["schema_version"],
            snapshot=MetricSnapshot.from_dict(snapshot_payload),
            snapshot_sha256=raw["snapshot_sha256"],
        )
        artifact.verify()
        return artifact

    def save(self, path: str | Path) -> Path:
        """Atomically persist the artifact and durably commit its directory entry."""
        self.verify()
        destination = Path(path)
        if not destination.parent.is_dir():
            raise FileNotFoundError(
                f"observability artifact parent directory does not exist: {destination.parent}"
            )

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_file.write(self.to_json())
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_path = Path(temporary_file.name)
            os.replace(temporary_path, destination)
            _sync_directory(destination.parent)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        return destination

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load and verify a persisted observability artifact."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def _snapshot_digest(snapshot: MetricSnapshot) -> str:
    canonical = json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_fields(payload: Mapping[str, Any]) -> None:
    actual_fields = set(payload)
    if actual_fields != _REQUIRED_FIELDS:
        missing_fields = sorted(_REQUIRED_FIELDS - actual_fields)
        unknown_fields = sorted(actual_fields - _REQUIRED_FIELDS)
        raise ValueError(
            "observability artifact fields do not match schema: "
            f"missing={missing_fields}, unknown={unknown_fields}"
        )


def _sync_directory(directory: Path) -> None:
    """Durably flush a replaced directory entry where POSIX supports it."""
    if os.name != "posix":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["ObservabilityArtifact"]
