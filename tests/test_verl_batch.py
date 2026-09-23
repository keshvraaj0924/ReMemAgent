"""Tests for the executable verl batch reward boundary."""

from dataclasses import replace

import pytest

from remem.training.verl_adapter import VerlRewardAdapter
from remem.training.verl_batch import VerlBatchRewardResult, compute_verl_batch_rewards


def test_compute_verl_batch_rewards_keeps_rewards_and_metrics_in_lockstep() -> None:
    samples = [
        {"task_reward": 1.0, "memory_used": True, "counterfactual_delta": 0.4},
        {"task_reward": 0.5, "memory_used": True, "counterfactual_delta": -0.2},
        {"task_reward": 0.8, "memory_used": False},
    ]

    result = compute_verl_batch_rewards(samples)

    assert result.rewards == pytest.approx((1.19, 0.29, 0.8))
    assert tuple(item.total_reward for item in result.record.rewards) == pytest.approx(
        result.rewards
    )
    assert result.record.metrics.sample_count == 3
    assert result.record.metrics.positive_transfer_rate == pytest.approx(1 / 3)
    assert result.record.metrics.negative_transfer_rate == pytest.approx(1 / 3)


def test_compute_verl_batch_rewards_uses_supplied_adapter_configuration() -> None:
    adapter = VerlRewardAdapter()
    sample = {"task_reward": 0.75, "memory_used": False}

    result = compute_verl_batch_rewards([sample], adapter=adapter)

    assert result.rewards == pytest.approx((0.75,))


def test_compute_verl_batch_rewards_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="at least one trainer sample"):
        compute_verl_batch_rewards([])


def test_compute_verl_batch_rewards_rejects_invalid_adapter() -> None:
    with pytest.raises(TypeError, match="VerlRewardAdapter"):
        compute_verl_batch_rewards([], adapter=object())  # type: ignore[arg-type]


def test_compute_verl_batch_rewards_preserves_adapter_validation() -> None:
    with pytest.raises(ValueError, match="missing required reward field"):
        compute_verl_batch_rewards([{"task_reward": 1.0}])


def test_batch_result_rejects_scalar_record_divergence() -> None:
    result = compute_verl_batch_rewards(
        [{"task_reward": 1.0, "memory_used": False}]
    )

    with pytest.raises(ValueError, match="scalar rewards must match"):
        replace(result, rewards=(0.0,))


def test_batch_result_rejects_empty_scalar_rewards() -> None:
    result = compute_verl_batch_rewards(
        [{"task_reward": 1.0, "memory_used": False}]
    )

    with pytest.raises(ValueError, match="at least one scalar reward"):
        VerlBatchRewardResult(rewards=(), record=result.record)
