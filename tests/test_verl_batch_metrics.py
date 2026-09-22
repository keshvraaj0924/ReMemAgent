"""Tests for verl adapter batch-level reward attribution metrics."""

import pytest

from remem.training.verl_adapter import VerlRewardAdapter


def test_summarize_batch_uses_same_reward_attribution_path() -> None:
    adapter = VerlRewardAdapter()
    samples = [
        {"task_reward": 1.0, "memory_used": True, "counterfactual_delta": 0.2},
        {"task_reward": 0.5, "memory_used": True, "counterfactual_delta": -0.3},
        {"task_reward": 0.8, "memory_used": False, "counterfactual_delta": 0.9},
    ]

    breakdowns = adapter.compute_batch_breakdowns(samples)
    metrics = adapter.summarize_batch(samples)

    assert metrics.sample_count == len(samples)
    assert metrics.mean_total_reward == pytest.approx(
        sum(item.total_reward for item in breakdowns) / len(breakdowns)
    )
    assert metrics.mean_task_component == pytest.approx(
        sum(item.task_component for item in breakdowns) / len(breakdowns)
    )
    assert metrics.mean_transfer_component == pytest.approx(
        sum(item.transfer_component for item in breakdowns) / len(breakdowns)
    )
    assert metrics.mean_memory_cost_component == pytest.approx(
        sum(item.memory_cost_component for item in breakdowns) / len(breakdowns)
    )
    assert metrics.positive_transfer_rate == pytest.approx(1 / 3)
    assert metrics.negative_transfer_rate == pytest.approx(1 / 3)


def test_summarize_batch_rejects_empty_trainer_batch() -> None:
    adapter = VerlRewardAdapter()

    with pytest.raises(ValueError, match="at least one"):
        adapter.summarize_batch([])


def test_summarize_batch_preserves_sample_validation() -> None:
    adapter = VerlRewardAdapter()

    with pytest.raises(ValueError, match="missing required reward field"):
        adapter.summarize_batch([{"task_reward": 1.0}])
