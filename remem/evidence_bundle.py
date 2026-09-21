"""Atomic persistence and verification for experiment evidence metadata."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from remem.artifact_manifest import ArtifactManifest
from remem.evidence_contract import EvidenceContract

_EVIDENCE_BUNDLE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Bind an evidence contract to the exact manifest for one experiment run."""

    schema_version: int
    contract: EvidenceContract
    manifest: ArtifactManifest

    @classmethod
    def create(cls, *, contract: EvidenceContract, manifest: ArtifactManifest) -> Self:
        """Create a validated bundle without touching experiment artifacts."""
        bundle = cls(
            schema_version=_EVIDENCE_BUNDLE_SCHEMA_VERSION,
            contract=contract,
            manifest=manifest,
        )
        bundle._validate_structure()
        manifest.require_paths(contract.required_paths)
        return bundle

    def verify(self, *, root: str | Path) -> None:
        """Fail unless the bundle is valid and all mandatory evidence is unchanged."""
        self._validate_structure()
        self.contract.verify(manifest=self.manifest, root=root)

    def to_json(self) -> str:
        """Serialize the complete evidence bundle deterministically."""
        self._validate_structure()
        payload = {
            "schema_version": self.schema_version,
            "contract": json.loads(self.contract.to_json()),
            "manifest": json.loads(self.manifest.to_json()),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> Self:
        """Deserialize a bundle while rejecting malformed or incomplete state."""
        if not isinstance(payload, str):
            raise TypeError("evidence bundle payload must be a string")
        try:
            raw: Any = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("evidence bundle payload must contain valid JSON") from error
        if not isinstance(raw, dict):
            raise TypeError("evidence bundle JSON root must be an object")

        expected_fields = {"schema_version", "contract", "manifest"}
        missing_fields = expected_fields - set(raw)
        unknown_fields = set(raw) - expected_fields
        if missing_fields:
            raise ValueError(f"evidence bundle is missing fields: {sorted(missing_fields)}")
        if unknown_fields:
            raise ValueError(f"evidence bundle contains unknown fields: {sorted(unknown_fields)}")

        bundle = cls(
            schema_version=raw["schema_version"],
            contract=EvidenceContract.from_json(_nested_json(raw["contract"], "contract")),
            manifest=ArtifactManifest.from_json(_nested_json(raw["manifest"], "manifest")),
        )
        bundle._validate_structure()
        bundle.manifest.require_paths(bundle.contract.required_paths)
        return bundle

    def save(self, path: str | Path) -> Path:
        """Atomically persist the bundle so interrupted writes cannot publish partial metadata."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        serialized = self.to_json() + "\n"
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, destination)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        return destination

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a previously persisted evidence bundle."""
        source = Path(path)
        return cls.from_json(source.read_text(encoding="utf-8"))

    def _validate_structure(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("schema_version must be an integer")
        if self.schema_version != _EVIDENCE_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported evidence bundle schema version: {self.schema_version}")
        if not isinstance(self.contract, EvidenceContract):
            raise TypeError("contract must be an EvidenceContract")
        if not isinstance(self.manifest, ArtifactManifest):
            raise TypeError("manifest must be an ArtifactManifest")
        self.contract.to_json()
        self.manifest.to_json()


def _nested_json(value: Any, field_name: str) -> str:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


__all__ = ["EvidenceBundle"]
