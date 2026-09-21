"""Content-addressed integrity records for experiment output artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ArtifactIntegrityRecord:
    """Immutable identity for one experiment artifact's exact bytes."""

    schema_version: int
    run_id: str
    relative_path: str
    size_bytes: int
    sha256: str

    @classmethod
    def capture(cls, *, run_id: str, path: str | Path, root: str | Path) -> Self:
        """Hash an artifact and bind it to a run using a portable relative path.

        Relative artifact paths are interpreted relative to ``root`` rather than the
        process working directory. This keeps experiment manifests independent of the
        directory from which a runner happens to invoke the framework.
        """
        cls._validate_run_id(run_id)
        root_path = Path(root).resolve()
        supplied_path = Path(path)
        artifact_path = (
            supplied_path.resolve()
            if supplied_path.is_absolute()
            else (root_path / supplied_path).resolve()
        )
        if not artifact_path.is_file():
            raise FileNotFoundError(f"artifact does not exist: {artifact_path}")
        try:
            relative_path = artifact_path.relative_to(root_path)
        except ValueError as error:
            raise ValueError("artifact must be contained by root") from error

        digest = hashlib.sha256()
        size_bytes = 0
        with artifact_path.open("rb") as artifact_file:
            for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
                digest.update(chunk)
                size_bytes += len(chunk)

        return cls(
            schema_version=_ARTIFACT_SCHEMA_VERSION,
            run_id=run_id,
            relative_path=relative_path.as_posix(),
            size_bytes=size_bytes,
            sha256=digest.hexdigest(),
        )

    def verify(self, *, root: str | Path) -> None:
        """Verify record structure and reject missing, moved, or modified artifacts."""
        self._validate_structure()
        current = type(self).capture(
            run_id=self.run_id,
            path=Path(root) / self.relative_path,
            root=root,
        )
        if current.size_bytes != self.size_bytes or current.sha256 != self.sha256:
            raise ValueError(f"artifact integrity mismatch: {self.relative_path}")

    def to_json(self) -> str:
        """Serialize the record deterministically for durable experiment metadata."""
        self._validate_structure()
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> Self:
        """Deserialize a record while rejecting missing and unknown fields."""
        if not isinstance(payload, str):
            raise TypeError("artifact integrity payload must be a string")
        try:
            raw: Any = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("artifact integrity payload must contain valid JSON") from error
        if not isinstance(raw, dict):
            raise TypeError("artifact integrity JSON root must be an object")
        expected_fields = {"schema_version", "run_id", "relative_path", "size_bytes", "sha256"}
        missing_fields = expected_fields - set(raw)
        unknown_fields = set(raw) - expected_fields
        if missing_fields:
            raise ValueError(f"artifact integrity is missing fields: {sorted(missing_fields)}")
        if unknown_fields:
            raise ValueError(
                f"artifact integrity contains unknown fields: {sorted(unknown_fields)}"
            )
        record = cls(**raw)
        record._validate_structure()
        return record

    def _validate_structure(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("schema_version must be an integer")
        if self.schema_version != _ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported artifact schema version: {self.schema_version}")
        self._validate_run_id(self.run_id)
        if not isinstance(self.relative_path, str) or not self.relative_path:
            raise ValueError("relative_path must be a non-empty string")
        relative_path = Path(self.relative_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("relative_path must stay within root")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("size_bytes must be an integer")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if not isinstance(self.sha256, str) or len(self.sha256) != 64:
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        try:
            bytes.fromhex(self.sha256)
        except ValueError as error:
            raise ValueError("sha256 must be hexadecimal") from error

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")


__all__ = ["ArtifactIntegrityRecord"]
