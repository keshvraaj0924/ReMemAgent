"""Content-bound provenance envelope for persisted verl reward evidence."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from remem.experiment_provenance import ExperimentProvenance
from remem.training.verl_evidence import (
    serialize_verl_reward_evidence,
    verify_verl_reward_evidence,
)
from remem.training.verl_records import VerlBatchRewardRecord

VERL_PROVENANCE_SCHEMA_VERSION = 1
_TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "run_id", "runtime_fingerprint", "provenance", "reward_evidence"}
)


@dataclass(frozen=True, slots=True)
class ProvenancedVerlRewardEvidence:
    """Verified reward evidence bound to the experiment that produced it."""

    provenance: ExperimentProvenance
    reward_record: VerlBatchRewardRecord

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, ExperimentProvenance):
            raise TypeError("provenance must be an ExperimentProvenance")
        if not isinstance(self.reward_record, VerlBatchRewardRecord):
            raise TypeError("reward_record must be a VerlBatchRewardRecord")
        self.provenance.verify()

    @property
    def run_id(self) -> str:
        """Return the content-derived experiment run identity."""
        return self.provenance.run_id

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible evidence with redundant bindings."""
        self.provenance.verify()
        return {
            "schema_version": VERL_PROVENANCE_SCHEMA_VERSION,
            "run_id": self.provenance.run_id,
            "runtime_fingerprint": self.provenance.runtime_fingerprint,
            "provenance": json.loads(self.provenance.to_json()),
            "reward_evidence": serialize_verl_reward_evidence(self.reward_record),
        }

    def to_json(self) -> str:
        """Serialize the verified envelope using stable key ordering."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        """Reconstruct an envelope and reject schema or provenance drift."""
        if not isinstance(payload, dict):
            raise TypeError("provenanced reward evidence must be a dictionary")
        if frozenset(payload) != _TOP_LEVEL_FIELDS:
            raise ValueError("provenanced reward evidence must contain exactly supported fields")
        if payload["schema_version"] != VERL_PROVENANCE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported provenanced reward evidence schema_version: "
                f"{payload['schema_version']!r}"
            )
        if not isinstance(payload["provenance"], dict):
            raise TypeError("provenance must be an object")
        if not isinstance(payload["reward_evidence"], dict):
            raise TypeError("reward_evidence must be an object")

        provenance = ExperimentProvenance.from_json(
            json.dumps(payload["provenance"], sort_keys=True, separators=(",", ":"))
        )
        if payload["run_id"] != provenance.run_id:
            raise ValueError("run_id does not match embedded experiment provenance")
        if payload["runtime_fingerprint"] != provenance.runtime_fingerprint:
            raise ValueError(
                "runtime_fingerprint does not match embedded experiment provenance"
            )
        reward_record = verify_verl_reward_evidence(payload["reward_evidence"])
        return cls(provenance=provenance, reward_record=reward_record)

    @classmethod
    def from_json(cls, payload: str) -> Self:
        """Deserialize and verify an untrusted JSON envelope."""
        if not isinstance(payload, str):
            raise TypeError("provenanced reward evidence payload must be a string")
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("provenanced reward evidence must contain valid JSON") from error
        if not isinstance(raw, dict):
            raise TypeError("provenanced reward evidence JSON root must be an object")
        return cls.from_dict(raw)

    def save(self, path: str | Path) -> None:
        """Atomically persist verified evidence without leaving partial artifacts."""
        destination = Path(path)
        if not destination.parent.is_dir():
            raise FileNotFoundError(f"evidence parent does not exist: {destination.parent}")

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
                temporary_path = Path(temporary_file.name)
                temporary_file.write(self.to_json())
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, destination)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load and fail-closed verify persisted reward/provenance evidence."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


__all__ = ["ProvenancedVerlRewardEvidence", "VERL_PROVENANCE_SCHEMA_VERSION"]
