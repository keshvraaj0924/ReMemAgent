"""Machine-readable evidence requirements for reportable experiment results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Self

from remem.artifact_manifest import ArtifactManifest

_EVIDENCE_CONTRACT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EvidenceContract:
    """Immutable set of artifacts required before a result may be reported."""

    schema_version: int
    name: str
    required_paths: tuple[str, ...]

    @classmethod
    def create(cls, *, name: str, required_paths: Iterable[str | Path]) -> Self:
        """Create a canonical contract from portable run-relative artifact paths."""
        normalized_paths = tuple(
            sorted({_normalize_required_path(path) for path in required_paths})
        )
        contract = cls(
            schema_version=_EVIDENCE_CONTRACT_SCHEMA_VERSION,
            name=name,
            required_paths=normalized_paths,
        )
        contract._validate_structure()
        return contract

    def verify(self, *, manifest: ArtifactManifest, root: str | Path) -> None:
        """Fail unless every required artifact is captured and byte-identical."""
        self._validate_structure()
        if not isinstance(manifest, ArtifactManifest):
            raise TypeError("manifest must be an ArtifactManifest")
        manifest.verify_required(paths=self.required_paths, root=root)

    def to_json(self) -> str:
        """Serialize the contract deterministically for experiment provenance."""
        self._validate_structure()
        payload = {
            "schema_version": self.schema_version,
            "name": self.name,
            "required_paths": list(self.required_paths),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> Self:
        """Deserialize a contract while rejecting malformed or ambiguous input."""
        if not isinstance(payload, str):
            raise TypeError("evidence contract payload must be a string")
        try:
            raw: Any = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("evidence contract payload must contain valid JSON") from error
        if not isinstance(raw, dict):
            raise TypeError("evidence contract JSON root must be an object")

        expected_fields = {"schema_version", "name", "required_paths"}
        missing_fields = expected_fields - set(raw)
        unknown_fields = set(raw) - expected_fields
        if missing_fields:
            raise ValueError(f"evidence contract is missing fields: {sorted(missing_fields)}")
        if unknown_fields:
            raise ValueError(f"evidence contract contains unknown fields: {sorted(unknown_fields)}")
        if not isinstance(raw["required_paths"], list):
            raise TypeError("required_paths must be a list")

        contract = cls(
            schema_version=raw["schema_version"],
            name=raw["name"],
            required_paths=tuple(raw["required_paths"]),
        )
        contract._validate_structure()
        return contract

    def _validate_structure(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("schema_version must be an integer")
        if self.schema_version != _EVIDENCE_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported evidence contract schema version: {self.schema_version}")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("evidence contract name must be a non-empty string")
        if not isinstance(self.required_paths, tuple):
            raise TypeError("required_paths must be a tuple")
        if not self.required_paths:
            raise ValueError("evidence contract must require at least one artifact")

        normalized_paths = tuple(_normalize_required_path(path) for path in self.required_paths)
        if normalized_paths != tuple(sorted(set(normalized_paths))):
            raise ValueError("required_paths must be unique and sorted")


def _normalize_required_path(path: str | Path) -> str:
    """Return a canonical run-relative POSIX path for a contract requirement."""
    if not isinstance(path, (str, Path)):
        raise TypeError("evidence paths must be strings or Path values")
    raw_path = str(path)
    if not raw_path or "\\" in raw_path:
        raise ValueError("evidence paths must use non-empty POSIX-style paths")
    normalized_path = PurePosixPath(raw_path)
    if normalized_path.is_absolute() or any(
        part in {"", ".", ".."} for part in normalized_path.parts
    ):
        raise ValueError("evidence paths must be normalized run-relative paths")
    if normalized_path.as_posix() != raw_path:
        raise ValueError("evidence paths must be normalized run-relative paths")
    return raw_path


__all__ = ["EvidenceContract"]
