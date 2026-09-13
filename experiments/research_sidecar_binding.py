"""Verify retained observability sidecars against a benchmark verification attestation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from experiments.research_evidence_record import (
    EvidenceArtifact,
    ResearchEvidenceRecord,
    verify_research_evidence_record,
)
from remem.observability import OBSERVATION_SNAPSHOT_SCHEMA_VERSION, ObservationSnapshot
from remem.observability_distribution_artifacts import (
    DISTRIBUTION_OBSERVATION_SCHEMA_VERSION,
    read_distribution_observation_snapshot,
)

OBSERVABILITY_SIDECAR_ROLE = "observability_sidecar"
DISTRIBUTION_SIDECAR_ROLE = "distribution_sidecar"
VERIFICATION_ATTESTATION_ROLE = "verification"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ResearchSidecarBinding:
    """Exact retained sidecar identities proven by one verification attestation."""

    observability_sha256: str
    distribution_sha256: str
    bundle_sha256: str


def verify_research_sidecar_binding(record_path: str | Path) -> ResearchSidecarBinding:
    """Verify exact observability sidecars and their retained attestation binding.

    The research-evidence record is verified first so all indexed artifacts must
    still match their recorded byte counts and SHA-256 digests. The retained
    sidecars are then parsed through their versioned schemas and compared with
    the exact sidecar identities recorded by ``remem-verify-benchmark``.
    """

    resolved_record_path = Path(record_path).resolve()
    record = verify_research_evidence_record(resolved_record_path)
    artifacts = _artifacts_by_role(record)
    _require_roles(artifacts)
    artifact_paths = _artifact_paths_by_role(resolved_record_path, record)

    observability_artifact = artifacts[OBSERVABILITY_SIDECAR_ROLE]
    distribution_artifact = artifacts[DISTRIBUTION_SIDECAR_ROLE]
    _validate_observability_sidecar(artifact_paths[OBSERVABILITY_SIDECAR_ROLE])
    _validate_distribution_sidecar(artifact_paths[DISTRIBUTION_SIDECAR_ROLE])

    attestation = _load_json_object(
        artifact_paths[VERIFICATION_ATTESTATION_ROLE],
        "verification attestation",
    )
    _verify_attested_artifact(
        attestation,
        prefix="observability",
        artifact=observability_artifact,
        expected_schema_version=OBSERVATION_SNAPSHOT_SCHEMA_VERSION,
    )
    _verify_attested_artifact(
        attestation,
        prefix="distribution",
        artifact=distribution_artifact,
        expected_schema_version=DISTRIBUTION_OBSERVATION_SCHEMA_VERSION,
    )
    bundle_sha256 = _required_sha256(attestation, "bundle_sha256")
    return ResearchSidecarBinding(
        observability_sha256=observability_artifact.sha256,
        distribution_sha256=distribution_artifact.sha256,
        bundle_sha256=bundle_sha256,
    )


def _artifacts_by_role(record: ResearchEvidenceRecord) -> dict[str, EvidenceArtifact]:
    """Index validated evidence artifacts by semantic role."""

    return {artifact.role: artifact for artifact in record.artifacts}


def _artifact_paths_by_role(
    record_path: Path,
    record: ResearchEvidenceRecord,
) -> dict[str, Path]:
    """Resolve indexed artifact paths beneath the verified record directory."""

    root = record_path.parent.resolve()
    return {artifact.role: root / artifact.path for artifact in record.artifacts}


def _require_roles(artifacts: Mapping[str, EvidenceArtifact]) -> None:
    """Reject evidence sets missing any canonical sidecar-binding role."""

    required_roles = {
        OBSERVABILITY_SIDECAR_ROLE,
        DISTRIBUTION_SIDECAR_ROLE,
        VERIFICATION_ATTESTATION_ROLE,
    }
    missing_roles = sorted(required_roles.difference(artifacts))
    if missing_roles:
        raise ValueError(
            "sidecar-binding verification requires evidence roles: " + ", ".join(missing_roles)
        )


def _load_json_object(path: Path, description: str) -> Mapping[str, object]:
    """Load one retained JSON object with a concise fail-closed contract."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} must contain valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{description} root must be a JSON object")
    return payload


def _validate_observability_sidecar(path: Path) -> None:
    """Validate the aggregate sidecar using the public snapshot schema."""

    payload = _load_json_object(path, "observability sidecar")
    ObservationSnapshot.from_dict(payload)


def _validate_distribution_sidecar(path: Path) -> None:
    """Validate the distribution sidecar using its public persistence contract."""

    read_distribution_observation_snapshot(path)


def _verify_attested_artifact(
    attestation: Mapping[str, object],
    *,
    prefix: str,
    artifact: EvidenceArtifact,
    expected_schema_version: int,
) -> None:
    """Match one exact retained sidecar to its benchmark-verification metadata."""

    attested_schema_version = _required_non_negative_integer(
        attestation,
        f"{prefix}_schema_version",
    )
    if attested_schema_version != expected_schema_version:
        raise ValueError(f"verification attestation {prefix} schema-version mismatch")

    attested_byte_count = _required_non_negative_integer(
        attestation,
        f"{prefix}_byte_count",
    )
    if attested_byte_count != artifact.byte_count:
        raise ValueError(f"verification attestation {prefix} byte-count mismatch")

    attested_sha256 = _required_sha256(attestation, f"{prefix}_sha256")
    if attested_sha256 != artifact.sha256:
        raise ValueError(f"verification attestation {prefix} SHA-256 mismatch")


def _required_non_negative_integer(payload: Mapping[str, object], field_name: str) -> int:
    """Return one required non-negative integer from a retained attestation."""

    value = payload.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"verification attestation missing valid {field_name}")
    return value


def _required_sha256(payload: Mapping[str, object], field_name: str) -> str:
    """Return one required lowercase SHA-256 digest from a retained attestation."""

    value = payload.get(field_name)
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"verification attestation missing valid {field_name}")
    return value


__all__ = [
    "DISTRIBUTION_SIDECAR_ROLE",
    "OBSERVABILITY_SIDECAR_ROLE",
    "ResearchSidecarBinding",
    "VERIFICATION_ATTESTATION_ROLE",
    "verify_research_sidecar_binding",
]
