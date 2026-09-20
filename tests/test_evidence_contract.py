"""Tests for machine-readable experiment evidence contracts."""

import pytest

from remem.artifact_manifest import ArtifactManifest
from remem.evidence_contract import EvidenceContract


def test_contract_verifies_required_artifacts_and_ignores_optional_changes(tmp_path) -> None:
    config = tmp_path / "config.json"
    metrics = tmp_path / "metrics.json"
    debug = tmp_path / "debug.log"
    config.write_text('{"seed":7}', encoding="utf-8")
    metrics.write_text('{"success_rate":0.5}', encoding="utf-8")
    debug.write_text("initial", encoding="utf-8")
    manifest = ArtifactManifest.capture(
        run_id="run-contract-001",
        paths=[config, metrics, debug],
        root=tmp_path,
    )
    contract = EvidenceContract.create(
        name="synthetic-ablation",
        required_paths=["metrics.json", "config.json", "metrics.json"],
    )

    assert contract.required_paths == ("config.json", "metrics.json")
    debug.write_text("changed optional output", encoding="utf-8")
    contract.verify(manifest=manifest, root=tmp_path)

    metrics.write_text('{"success_rate":0.9}', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact integrity mismatch"):
        contract.verify(manifest=manifest, root=tmp_path)


def test_contract_fails_closed_when_required_artifact_was_not_captured(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"seed":7}', encoding="utf-8")
    manifest = ArtifactManifest.capture(
        run_id="run-contract-002",
        paths=[config],
        root=tmp_path,
    )
    contract = EvidenceContract.create(
        name="external-environment",
        required_paths=["config.json", "trajectories.jsonl"],
    )

    with pytest.raises(ValueError, match="missing required paths.*trajectories.jsonl"):
        contract.verify(manifest=manifest, root=tmp_path)


def test_contract_json_round_trip_is_deterministic() -> None:
    contract = EvidenceContract.create(
        name="grpo-evaluation",
        required_paths=["provenance.json", "metrics.json", "config.json"],
    )

    payload = contract.to_json()

    assert EvidenceContract.from_json(payload) == contract
    assert EvidenceContract.from_json(payload).to_json() == payload


@pytest.mark.parametrize(
    "invalid_path",
    ["", "/absolute.json", "../outside.json", "raw/../metrics.json", "raw\\metrics.json"],
)
def test_contract_rejects_ambiguous_paths(invalid_path: str) -> None:
    with pytest.raises(ValueError, match="required artifact paths"):
        EvidenceContract.create(name="invalid", required_paths=[invalid_path])


def test_contract_rejects_empty_requirement_set() -> None:
    with pytest.raises(ValueError, match="at least one artifact"):
        EvidenceContract.create(name="empty", required_paths=[])


def test_contract_rejects_non_manifest_verification_target(tmp_path) -> None:
    contract = EvidenceContract.create(name="typed", required_paths=["metrics.json"])

    with pytest.raises(TypeError, match="ArtifactManifest"):
        contract.verify(manifest=object(), root=tmp_path)  # type: ignore[arg-type]


def test_contract_deserialization_rejects_unsorted_or_duplicate_paths() -> None:
    unsorted_payload = (
        '{"name":"bad","required_paths":["metrics.json","config.json"],'
        '"schema_version":1}'
    )
    duplicate_payload = (
        '{"name":"bad","required_paths":["metrics.json","metrics.json"],'
        '"schema_version":1}'
    )

    with pytest.raises(ValueError, match="unique and sorted"):
        EvidenceContract.from_json(unsorted_payload)
    with pytest.raises(ValueError, match="unique and sorted"):
        EvidenceContract.from_json(duplicate_payload)
