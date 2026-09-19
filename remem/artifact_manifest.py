"""Run-level manifests for content-addressed experiment artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Self

from remem.artifact_integrity import ArtifactIntegrityRecord

_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Immutable collection of artifact identities produced by one experiment run."""

    schema_version: int
    run_id: str
    artifacts: tuple[ArtifactIntegrityRecord, ...]

    @classmethod
    def capture(
        cls,
        *,
        run_id: str,
        paths: Iterable[str | Path],
        root: str | Path,
    ) -> Self:
        """Capture a deterministic manifest for a run's output files."""
        records = tuple(
            ArtifactIntegrityRecord.capture(run_id=run_id, path=path, root=root)
            for path in paths
        )
        manifest = cls(
            schema_version=_MANIFEST_SCHEMA_VERSION,
            run_id=run_id,
            artifacts=tuple(sorted(records, key=lambda record: record.relative_path)),
        )
        manifest._validate_structure()
        return manifest

    def verify(self, *, root: str | Path) -> None:
        """Verify manifest structure and every referenced artifact's exact bytes."""
        self._validate_structure()
        for record in self.artifacts:
            record.verify(root=root)

    def to_json(self) -> str:
        """Serialize the manifest deterministically for durable provenance storage."""
        self._validate_structure()
        payload = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "artifacts": [json.loads(record.to_json()) for record in self.artifacts],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> Self:
        """Deserialize a manifest while rejecting malformed or ambiguous records."""
        if not isinstance(payload, str):
            raise TypeError("artifact manifest payload must be a string")
        try:
            raw: Any = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("artifact manifest payload must contain valid JSON") from error
        if not isinstance(raw, dict):
            raise TypeError("artifact manifest JSON root must be an object")

        expected_fields = {"schema_version", "run_id", "artifacts"}
        missing_fields = expected_fields - set(raw)
        unknown_fields = set(raw) - expected_fields
        if missing_fields:
            raise ValueError(f"artifact manifest is missing fields: {sorted(missing_fields)}")
        if unknown_fields:
            raise ValueError(
                f"artifact manifest contains unknown fields: {sorted(unknown_fields)}"
            )
        if not isinstance(raw["artifacts"], list):
            raise TypeError("artifacts must be a list")

        records = tuple(
            ArtifactIntegrityRecord.from_json(
                json.dumps(record, sort_keys=True, separators=(",", ":"))
            )
            for record in raw["artifacts"]
        )
        manifest = cls(
            schema_version=raw["schema_version"],
            run_id=raw["run_id"],
            artifacts=records,
        )
        manifest._validate_structure()
        return manifest

    def _validate_structure(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("schema_version must be an integer")
        if self.schema_version != _MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema version: {self.schema_version}")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if not isinstance(self.artifacts, tuple):
            raise TypeError("artifacts must be a tuple")

        relative_paths: set[str] = set()
        previous_path: str | None = None
        for record in self.artifacts:
            if not isinstance(record, ArtifactIntegrityRecord):
                raise TypeError("artifacts must contain ArtifactIntegrityRecord values")
            if record.run_id != self.run_id:
                raise ValueError("artifact run_id must match manifest run_id")
            record._validate_structure()
            if record.relative_path in relative_paths:
                raise ValueError(f"duplicate artifact path: {record.relative_path}")
            if previous_path is not None and record.relative_path < previous_path:
                raise ValueError("artifacts must be sorted by relative_path")
            relative_paths.add(record.relative_path)
            previous_path = record.relative_path


__all__ = ["ArtifactManifest"]
