from __future__ import annotations

import json

import pytest

from experiments.paired_source_preflight import ControlledPairedPreflightResult
from experiments.preflight_evidence import (
    PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
    build_controlled_paired_preflight_evidence,
    preflight_evidence_json,
    verify_controlled_paired_preflight_evidence,
)
from experiments.runtime_provenance import RuntimeProvenance
from experiments.source_checkouts import SourceCheckoutProvenance, SourceCheckoutRequirement


def _runtime_provenance() -> RuntimeProvenance:
    return RuntimeProvenance(
        schema_version=1,
        code_revision="a" * 40,
        working_tree_state="clean",
        python_version="3.12.0",
        platform="test-platform",
        package_version="0.1.0",
        dependency_fingerprint="b" * 64,
        dependency_versions={"pytest": "8.0.0"},
    )


def _preflight_result() -> ControlledPairedPreflightResult:
    return ControlledPairedPreflightResult(
        runtime_provenance=_runtime_provenance(),
        source_checkout_provenance={
            "ALFWorld": SourceCheckoutProvenance(
                revision="c" * 40,
                working_tree_state="clean",
            )
        },
    )


def _requirements() -> dict[str, SourceCheckoutRequirement]:
    return {
        "ALFWorld": SourceCheckoutRequirement(
            expected_revision="c" * 40,
            require_clean_working_tree=True,
        )
    }


def test_build_preflight_evidence_is_deterministic_and_complete() -> None:
    first = build_controlled_paired_preflight_evidence(_preflight_result(), _requirements())
    second = build_controlled_paired_preflight_evidence(_preflight_result(), _requirements())

    assert first == second
    assert first["schema_version"] == PREFLIGHT_EVIDENCE_SCHEMA_VERSION
    assert first["runtime_provenance"]["code_revision"] == "a" * 40
    assert len(first["source_checkout_requirements_sha256"]) == 64
    assert len(first["source_checkout_provenance_sha256"]) == 64
    assert len(first["evidence_sha256"]) == 64
    verify_controlled_paired_preflight_evidence(first)


def test_preflight_evidence_json_is_canonical() -> None:
    evidence = build_controlled_paired_preflight_evidence(_preflight_result(), _requirements())

    serialized = preflight_evidence_json(evidence)

    assert json.loads(serialized) == evidence
    assert " " not in serialized
    assert serialized == preflight_evidence_json(dict(reversed(list(evidence.items()))))


def test_preflight_evidence_verification_rejects_tampering() -> None:
    evidence = build_controlled_paired_preflight_evidence(_preflight_result(), _requirements())
    tampered = dict(evidence)
    tampered_runtime = dict(tampered["runtime_provenance"])
    tampered_runtime["code_revision"] = "d" * 40
    tampered["runtime_provenance"] = tampered_runtime

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_controlled_paired_preflight_evidence(tampered)


def test_preflight_evidence_rejects_schema_drift() -> None:
    evidence = build_controlled_paired_preflight_evidence(_preflight_result(), _requirements())
    evidence["unexpected"] = True

    with pytest.raises(ValueError, match="exact persisted schema"):
        verify_controlled_paired_preflight_evidence(evidence)
