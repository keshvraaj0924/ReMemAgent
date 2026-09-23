"""Bind integrity-checked observability to the experiment that produced it."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from remem.experiment_provenance import ExperimentProvenance
from remem.observability_artifact import ObservabilityArtifact

_SCHEMA_VERSION = 1
_REQUIRED_FIELDS = frozenset({"schema_version", "run_id", "provenance", "observability", "bundle_sha256"})


@dataclass(frozen=True, slots=True)
class ProvenancedObservability:
    """Verified telemetry bound to one reproducible experiment run."""

    run_id: str
    provenance: ExperimentProvenance
    observability: ObservabilityArtifact
    bundle_sha256: str
    schema_version: int = _SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        provenance: ExperimentProvenance,
        observability: ObservabilityArtifact,
    ) -> Self:
        """Create a content-bound telemetry bundle from verified components."""
        if not isinstance(provenance, ExperimentProvenance):
            raise TypeError("provenance must be an ExperimentProvenance")
        if not isinstance(observability, ObservabilityArtifact):
            raise TypeError("observability must be an ObservabilityArtifact")
        provenance.verify()
        observability.verify()
        bundle = cls(
            run_id=provenance.run_id,
            provenance=provenance,
            observability=observability,
            bundle_sha256=_bundle_digest(provenance, observability),
        )
        bundle.verify()
        return bundle

    def verify(self) -> None:
        """Reject detached provenance, telemetry tampering, and schema drift."""
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("schema_version must be an integer")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported provenanced observability version: {self.schema_version}")
        self.provenance.verify()
        self.observability.verify()
        if self.run_id != self.provenance.run_id:
            raise ValueError("run_id does not match experiment provenance")
        if not isinstance(self.bundle_sha256, str):
            raise TypeError("bundle_sha256 must be a string")
        if self.bundle_sha256 != _bundle_digest(self.provenance, self.observability):
            raise ValueError("provenanced observability digest mismatch")

    def to_json(self) -> str:
        """Serialize the verified bundle deterministically."""
        self.verify()
        payload = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "provenance": json.loads(self.provenance.to_json()),
            "observability": json.loads(self.observability.to_json()),
            "bundle_sha256": self.bundle_sha256,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> Self:
        """Restore a bundle while strictly validating its complete schema."""
        if not isinstance(payload, str):
            raise TypeError("provenanced observability payload must be a string")
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("provenanced observability payload must contain valid JSON") from error
        if not isinstance(raw, dict):
            raise TypeError("provenanced observability JSON root must be an object")
        _validate_fields(raw)
        if not isinstance(raw["provenance"], Mapping):
            raise TypeError("provenance must be an object")
        if not isinstance(raw["observability"], Mapping):
            raise TypeError("observability must be an object")
        bundle = cls(
            schema_version=raw["schema_version"],
            run_id=raw["run_id"],
            provenance=ExperimentProvenance.from_json(_canonical_json(raw["provenance"])),
            observability=ObservabilityArtifact.from_json(_canonical_json(raw["observability"])),
            bundle_sha256=raw["bundle_sha256"],
        )
        bundle.verify()
        return bundle


def _bundle_digest(provenance: ExperimentProvenance, observability: ObservabilityArtifact) -> str:
    payload = {
        "provenance": json.loads(provenance.to_json()),
        "observability": json.loads(observability.to_json()),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _validate_fields(payload: Mapping[str, Any]) -> None:
    actual_fields = set(payload)
    if actual_fields != _REQUIRED_FIELDS:
        missing_fields = sorted(_REQUIRED_FIELDS - actual_fields)
        unknown_fields = sorted(actual_fields - _REQUIRED_FIELDS)
        raise ValueError(
            "provenanced observability fields do not match schema: "
            f"missing={missing_fields}, unknown={unknown_fields}"
        )


__all__ = ["ProvenancedObservability"]
