from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.paired_source_preflight import ControlledPairedPreflightResult
from experiments.preflight_evidence import build_controlled_paired_preflight_evidence
from experiments.runtime_provenance import RuntimeProvenance
from experiments.source_checkouts import SourceCheckoutProvenance, SourceCheckoutRequirement
from experiments.verify_preflight_evidence import (
    load_preflight_evidence,
    verify_preflight_evidence_file,
)


def _evidence() -> dict[str, object]:
    runtime = RuntimeProvenance(
        schema_version=1,
        code_revision="a" * 40,
        working_tree_state="clean",
        python_version="3.12.0",
        platform="test-platform",
        package_version="0.1.0",
        dependency_fingerprint="b" * 64,
        dependency_versions={"pytest": "8.0.0"},
    )
    result = ControlledPairedPreflightResult(
        runtime_provenance=runtime,
        source_checkout_provenance={
            "ALFWorld": SourceCheckoutProvenance(
                revision="c" * 40,
                working_tree_state="clean",
            )
        },
    )
    requirements = {
        "ALFWorld": SourceCheckoutRequirement(
            expected_revision="c" * 40,
            require_clean_working_tree=True,
        )
    }
    return build_controlled_paired_preflight_evidence(result, requirements)


def test_verify_preflight_evidence_file_accepts_valid_persisted_evidence(tmp_path: Path) -> None:
    evidence_path = tmp_path / "readiness.json"
    evidence_path.write_text(json.dumps(_evidence()), encoding="utf-8")

    verify_preflight_evidence_file(evidence_path)


def test_verify_preflight_evidence_file_rejects_tampering(tmp_path: Path) -> None:
    evidence = _evidence()
    runtime = dict(evidence["runtime_provenance"])
    runtime["code_revision"] = "d" * 40
    evidence["runtime_provenance"] = runtime
    evidence_path = tmp_path / "readiness.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_preflight_evidence_file(evidence_path)


def test_load_preflight_evidence_rejects_non_object_root(tmp_path: Path) -> None:
    evidence_path = tmp_path / "readiness.json"
    evidence_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="root must be a JSON object"):
        load_preflight_evidence(evidence_path)
