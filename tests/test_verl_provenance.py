from __future__ import annotations

import json

import pytest

from remem.experiment_provenance import ExperimentProvenance
from remem.reproducibility import ReproducibilityConfig
from remem.training.grpo_metrics import GrpoBatchMetrics
from remem.training.verl_provenance import ProvenancedVerlRewardEvidence
from remem.training.verl_records import VerlBatchRewardRecord, VerlRewardRecord


def _provenance() -> ExperimentProvenance:
    manifest = ReproducibilityConfig(base_seed=73).create_manifest(
        [("training", 0), ("rollout", 0)]
    )
    return ExperimentProvenance.capture(manifest)


def _reward_record() -> VerlBatchRewardRecord:
    return VerlBatchRewardRecord(
        rewards=(
            VerlRewardRecord(0, 1.0, 0.2, -0.1, 1.1),
            VerlRewardRecord(1, 0.0, -0.4, -0.1, -0.5),
        ),
        metrics=GrpoBatchMetrics(
            sample_count=2,
            mean_total_reward=0.3,
            mean_task_component=0.5,
            mean_transfer_component=-0.1,
            mean_memory_cost_component=-0.1,
            positive_transfer_rate=0.5,
            negative_transfer_rate=0.5,
        ),
    )


def _evidence() -> ProvenancedVerlRewardEvidence:
    return ProvenancedVerlRewardEvidence(provenance=_provenance(), reward_record=_reward_record())


def test_provenanced_reward_evidence_round_trip(tmp_path) -> None:
    evidence = _evidence()
    path = tmp_path / "verl-run-evidence.json"

    evidence.save(path)
    restored = ProvenancedVerlRewardEvidence.load(path)

    assert restored == evidence
    assert restored.run_id == evidence.provenance.run_id
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]


def test_rejects_detached_run_identity() -> None:
    payload = _evidence().to_dict()
    payload["run_id"] = "v1-" + "0" * 64

    with pytest.raises(ValueError, match="run_id does not match embedded"):
        ProvenancedVerlRewardEvidence.from_dict(payload)


def test_rejects_detached_runtime_fingerprint() -> None:
    payload = _evidence().to_dict()
    payload["runtime_fingerprint"] = "0" * 64

    with pytest.raises(ValueError, match="runtime_fingerprint does not match embedded"):
        ProvenancedVerlRewardEvidence.from_dict(payload)


def test_rejects_tampered_embedded_provenance() -> None:
    payload = _evidence().to_dict()
    payload["provenance"]["manifest"]["assignments"][0]["seed"] += 1

    with pytest.raises(ValueError, match="seed mismatch"):
        ProvenancedVerlRewardEvidence.from_dict(payload)


def test_rejects_tampered_reward_evidence() -> None:
    payload = _evidence().to_dict()
    payload["reward_evidence"]["metrics"]["mean_total_reward"] = 99.0

    with pytest.raises(ValueError, match="mean_total_reward"):
        ProvenancedVerlRewardEvidence.from_dict(payload)


def test_rejects_unknown_and_unsupported_schema_fields() -> None:
    payload = _evidence().to_dict()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="exactly supported fields"):
        ProvenancedVerlRewardEvidence.from_dict(payload)

    payload = _evidence().to_dict()
    payload["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        ProvenancedVerlRewardEvidence.from_dict(payload)


def test_json_validation_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        ProvenancedVerlRewardEvidence.from_json("not-json")
    with pytest.raises(TypeError, match="JSON root"):
        ProvenancedVerlRewardEvidence.from_json("[]")


def test_save_requires_existing_parent(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="evidence parent does not exist"):
        _evidence().save(tmp_path / "missing" / "evidence.json")


def test_serialization_is_deterministic() -> None:
    evidence = _evidence()
    assert (
        evidence.to_json() == ProvenancedVerlRewardEvidence.from_json(evidence.to_json()).to_json()
    )
    assert json.loads(evidence.to_json())["run_id"] == evidence.run_id
