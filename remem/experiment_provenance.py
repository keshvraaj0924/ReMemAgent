"""Experiment provenance records binding deterministic seeds to runtime metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Self

from remem.reproducibility import ReproducibilityManifest
from remem.run_identity import ExperimentRunIdentity
from remem.runtime_metadata import RuntimeMetadata

_PROVENANCE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ExperimentProvenance:
    """Verified provenance for replaying and auditing one experiment run."""

    schema_version: int
    run_id: str
    runtime_fingerprint: str
    manifest: ReproducibilityManifest
    runtime: RuntimeMetadata

    @classmethod
    def capture(cls, manifest: ReproducibilityManifest) -> Self:
        """Capture provenance for a verified reproducibility manifest."""
        if not isinstance(manifest, ReproducibilityManifest):
            raise TypeError("manifest must be a ReproducibilityManifest")
        manifest.verify()
        runtime = RuntimeMetadata.capture()
        return cls(
            schema_version=_PROVENANCE_SCHEMA_VERSION,
            run_id=ExperimentRunIdentity.from_manifest(manifest).value,
            runtime_fingerprint=runtime.fingerprint(),
            manifest=manifest,
            runtime=runtime,
        )

    def verify(self) -> None:
        """Verify schema and all content-addressed provenance relationships."""
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("schema_version must be an integer")
        if self.schema_version != _PROVENANCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported experiment provenance version: {self.schema_version}")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be a non-empty string")
        if not isinstance(self.runtime_fingerprint, str) or not self.runtime_fingerprint:
            raise ValueError("runtime_fingerprint must be a non-empty string")

        self.manifest.verify()
        self.runtime.verify()
        expected_run_id = ExperimentRunIdentity.from_manifest(self.manifest).value
        if self.run_id != expected_run_id:
            raise ValueError("run_id does not match reproducibility manifest")
        expected_runtime_fingerprint = self.runtime.fingerprint()
        if self.runtime_fingerprint != expected_runtime_fingerprint:
            raise ValueError("runtime_fingerprint does not match runtime metadata")

    def to_json(self) -> str:
        """Serialize verified provenance with deterministic key ordering."""
        self.verify()
        payload = asdict(self)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> Self:
        """Deserialize provenance and reject malformed or tampered artifacts."""
        if not isinstance(payload, str):
            raise TypeError("experiment provenance payload must be a string")
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("experiment provenance payload must contain valid JSON") from error
        if not isinstance(raw, dict):
            raise TypeError("experiment provenance JSON root must be an object")

        expected_fields = {
            "schema_version",
            "run_id",
            "runtime_fingerprint",
            "manifest",
            "runtime",
        }
        actual_fields = set(raw)
        missing_fields = expected_fields - actual_fields
        unknown_fields = actual_fields - expected_fields
        if missing_fields:
            raise ValueError(f"experiment provenance is missing fields: {sorted(missing_fields)}")
        if unknown_fields:
            raise ValueError(f"experiment provenance contains unknown fields: {sorted(unknown_fields)}")
        if not isinstance(raw["manifest"], dict):
            raise TypeError("manifest must be an object")
        if not isinstance(raw["runtime"], dict):
            raise TypeError("runtime must be an object")

        provenance = cls(
            schema_version=raw["schema_version"],
            run_id=raw["run_id"],
            runtime_fingerprint=raw["runtime_fingerprint"],
            manifest=ReproducibilityManifest.from_json(
                json.dumps(raw["manifest"], sort_keys=True, separators=(",", ":"))
            ),
            runtime=RuntimeMetadata.from_json(
                json.dumps(raw["runtime"], sort_keys=True, separators=(",", ":"))
            ),
        )
        provenance.verify()
        return provenance

    def assert_replay_compatible(self) -> None:
        """Validate provenance integrity and require the recorded runtime for replay."""
        self.verify()
        self.runtime.assert_current_runtime()


__all__ = ["ExperimentProvenance"]
