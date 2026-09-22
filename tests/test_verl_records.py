"""Tests for serializable verl trainer reward records."""

import pytest

from remem.training.grpo import GrpoRewardBreakdown
from remem.training.grpo_metrics import GrpoBatchMetrics
from remem.training.verl_adapter import VerlRewardAdapter
from remem.training.verl_records import (
    VerlBatchRewardRecord,
    VerlRewardRecord,
    build_verl_batch_reward_record,
)


def test_build_batch_record_preserves_order_and_reward_attribution() -> None:
    adapter = VerlRewardAdapter()
    samples = [
        {"task_reward": 1.0, "memory_used": True, "counterfactual_delta": 0.4},
        {"task_reward": 0.5, "memory_used": True, "counterfactual_delta": -0.2},
        {"task_reward": 0.8, "memory_used": False},
    ]

    record = build_verl_batch_reward_record(adapter, samples)
    payload = record.to_dict()

    assert [item.sample_index for item in record.rewards] == [0, 1, 2]
    assert [item.total_reward for item in record.rewards] == pytest.approx([1.19, 0.29, 0.8])
    assert record.metrics.positive_transfer_rate == pytest.approx(1 / 3)
    assert record.metrics.negative_transfer_rate == pytest.approx(1 / 3)
    assert payload["rewards"][0]["transfer_component"] == pytest.approx(0.2)
    assert payload["metrics"]["sample_count"] == 3


def test_reward_record_rejects_invalid_breakdown_type() -> None:
    with pytest.raises(TypeError, match="GrpoRewardBreakdown"):
        VerlRewardRecord.from_breakdown(0, object())  # type: ignore[arg-type]


def test_batch_record_rejects_non_contiguous_order() -> None:
    breakdown = GrpoRewardBreakdown(1.0, 0.0, 0.0, 1.0)
    reward = VerlRewardRecord.from_breakdown(1, breakdown)
    metrics = GrpoBatchMetrics(1, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="contiguous batch order"):
        VerlBatchRewardRecord((reward,), metrics)


def test_batch_record_rejects_metric_count_mismatch() -> None:
    breakdown = GrpoRewardBreakdown(1.0, 0.0, 0.0, 1.0)
    reward = VerlRewardRecord.from_breakdown(0, breakdown)
    metrics = GrpoBatchMetrics(2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="sample_count"):
        VerlBatchRewardRecord((reward,), metrics)


def test_build_batch_record_preserves_adapter_validation() -> None:
    adapter = VerlRewardAdapter()

    with pytest.raises(ValueError, match="missing required reward field"):
        build_verl_batch_reward_record(adapter, [{"task_reward": 1.0}])

    with pytest.raises(ValueError, match="at least one"):
        build_verl_batch_reward_record(adapter, [])


def test_build_batch_record_rejects_wrong_adapter_type() -> None:
    with pytest.raises(TypeError, match="VerlRewardAdapter"):
        build_verl_batch_reward_record(object(), [])  # type: ignore[arg-type]
