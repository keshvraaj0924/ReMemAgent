"""Machine-readable evidence for controlled paired benchmark readiness preflight.

Readiness evidence is intentionally separate from measured benchmark artifacts. It
captures the exact runtime and external source snapshots admitted before any
measured episode starts, together with deterministic fingerprints that orchestration
systems can compare across hosts or retain in CI logs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from experiments.paired_source_preflight import ControlledPairedPreflightResult
from experiments.source_checkouts import (
    SourceCheckoutRequirement,
    source_checkout_provenance_sha256,
    source_checkout_provenance_to_dict,
    source_checkout_requirements_sha256,
    source_checkout_requirements_to_dict,
)

PREFLIGHT_EVIDENCE_SCHEMA_VERSION = 1
PREFLIGHT_EVIDENCE_PROVENANCE_KEY = "preflight_evidence_sha256"


def build_controlled_paired_preflight_evidence(
    result: ControlledPairedPreflightResult,
    source_checkout_requirements: Mapping[str, SourceCheckoutRequirement],
) -> dict[str, object]:
    """Build canonical JSON-compatible readiness evidence from admitted snapshots.

    The evidence contains only state already admitted by controlled preflight. It
    does not recollect runtime or Git state, so callers cannot accidentally bind a
    later mutable snapshot to the readiness decision.
    """

    if not isinstance(result, ControlledPairedPreflightResult):
        raise TypeError("result must be a ControlledPairedPreflightResult")
    if not isinstance(source_checkout_requirements, Mapping):
        raise TypeError("source_checkout_requirements must be a mapping")

    source_requirements = source_checkout_requirements_to_dict(source_checkout_requirements)
    source_provenance = source_checkout_provenance_to_dict(result.source_checkout_provenance)
    payload: dict[str, object] = {
        "schema_version": PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
        "runtime_provenance": result.runtime_provenance.to_dict(),
        "source_checkout_requirements": source_requirements,
        "source_checkout_requirements_sha256": source_checkout_requirements_sha256(
            source_checkout_requirements
        ),
        "source_checkout_provenance": source_provenance,
        "source_checkout_provenance_sha256": source_checkout_provenance_sha256(
            result.source_checkout_provenance
        ),
    }
    payload["evidence_sha256"] = _canonical_sha256(payload)
    return payload


def verify_controlled_paired_preflight_evidence(payload: Mapping[str, Any]) -> None:
    """Fail closed when serialized readiness evidence is malformed or tampered."""

    if not isinstance(payload, Mapping):
        raise TypeError("preflight evidence must be a mapping")
    expected_fields = {
        "schema_version",
        "runtime_provenance",
        "source_checkout_requirements",
        "source_checkout_requirements_sha256",
        "source_checkout_provenance",
        "source_checkout_provenance_sha256",
        "evidence_sha256",
    }
    if set(payload) != expected_fields:
        raise ValueError("preflight evidence must use the exact persisted schema")
    if payload["schema_version"] != PREFLIGHT_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported preflight evidence schema version")

    expected_digest = payload["evidence_sha256"]
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ValueError("preflight evidence SHA-256 must be a 64-character hexadecimal string")
    try:
        int(expected_digest, 16)
    except ValueError as exc:
        raise ValueError("preflight evidence SHA-256 must be hexadecimal") from exc

    unsigned_payload = dict(payload)
    unsigned_payload.pop("evidence_sha256")
    actual_digest = _canonical_sha256(unsigned_payload)
    if actual_digest != expected_digest:
        raise ValueError("preflight evidence fingerprint mismatch")


def validate_preflight_evidence_matches_admission(
    payload: Mapping[str, Any],
    result: ControlledPairedPreflightResult,
    source_checkout_requirements: Mapping[str, SourceCheckoutRequirement],
) -> None:
    """Require persisted evidence to match the exact state admitted for measurement.

    The persisted evidence is verified first, then compared with a canonical payload
    built only from the already-admitted snapshots. No runtime or source state is
    recollected. A mismatch therefore blocks measurement when state changed between
    an earlier readiness run and the current admission.
    """

    verify_controlled_paired_preflight_evidence(payload)
    admitted_payload = build_controlled_paired_preflight_evidence(
        result,
        source_checkout_requirements,
    )
    if payload["evidence_sha256"] != admitted_payload["evidence_sha256"]:
        raise ValueError("preflight evidence does not match the current admitted state")


def preflight_evidence_json(payload: Mapping[str, Any]) -> str:
    """Serialize verified readiness evidence deterministically for logs or files."""

    verify_controlled_paired_preflight_evidence(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible evidence."""

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


__all__ = [
    "PREFLIGHT_EVIDENCE_PROVENANCE_KEY",
    "PREFLIGHT_EVIDENCE_SCHEMA_VERSION",
    "build_controlled_paired_preflight_evidence",
    "preflight_evidence_json",
    "validate_preflight_evidence_matches_admission",
    "verify_controlled_paired_preflight_evidence",
]
