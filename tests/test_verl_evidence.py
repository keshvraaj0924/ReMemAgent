from __future__ import annotations

import json

import pytest

from remem.training.grpo_metrics import GrpoBatchMetrics
from remem.training.verl_evidence import (
    load_verl_reward_evidence,
    serialize_verl_reward_evidence,
    verify_verl_reward_evidence,
    write_verl_reward_evidence,
)
from remem.training.verl_records import VerlBatchRewardRecord, VerlRewardRecord


def _record() -> VerlBatchRewardRecord:
    rewards = (
        VerlRewardRecord(0, 1.0, 0.2, -0.1, 1.1),
        VerlRewardRecord(1, 0.0, -0.4, -0.1, -0.5),
    )
    return VerlBatchRewardRecord(
        rewards=rewards,
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


def test_reward_evidence_round_trip(tmp_path) -> None:
    record = _record()
    path = tmp_path / "reward-evidence.json"

    write_verl_reward_evidence(path, record)

    assert load_verl_reward_evidence(path) == record
    assert json.loads(path.read_text(encoding="utf-8")) == serialize_verl_reward_evidence(record)


def test_verifier_rejects_tampered_aggregate() -> None:
    payload = serialize_verl_reward_evidence(_record())
    payload["metrics"]["mean_total_reward"] = 99.0

    with pytest.raises(ValueError, match="mean_total_reward"):
        verify_verl_reward_evidence(payload)


def test_verifier_rejects_non_contiguous_order() -> None:
    payload = serialize_verl_reward_evidence(_record())
    payload["rewards"][0]["sample_index"] = 1

    with pytest.raises(ValueError, match="contiguous batch order"):
        verify_verl_reward_evidence(payload)


def test_verifier_rejects_unknown_fields() -> None:
    payload = serialize_verl_reward_evidence(_record())
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="top-level fields"):
        verify_verl_reward_evidence(payload)


def test_verifier_rejects_unknown_reward_fields() -> None:
    payload = serialize_verl_reward_evidence(_record())
    payload["rewards"][0]["unexpected"] = True

    with pytest.raises(ValueError, match="each reward"):
        verify_verl_reward_evidence(payload)


def test_verifier_rejects_unsupported_schema_version() -> None:
    payload = serialize_verl_reward_evidence(_record())
    payload["schema_version"] = 2

    with pytest.raises(ValueError, match="schema_version"):
        verify_verl_reward_evidence(payload)


def test_verifier_rejects_empty_rewards() -> None:
    payload = serialize_verl_reward_evidence(_record())
    payload["rewards"] = []

    with pytest.raises(ValueError, match="non-empty"):
        verify_verl_reward_evidence(payload)


def test_serializer_requires_typed_record() -> None:
    with pytest.raises(TypeError, match="VerlBatchRewardRecord"):
        serialize_verl_reward_evidence({})  # type: ignore[arg-type]
