"""Tests for aggregate GRPO reward attribution metrics."""

import pytest

from remem.training.grpo import GrpoRewardBreakdown
from remem.training.grpo_metrics import summarize_grpo_breakdowns


def _breakdown(task: float, transfer: float, cost: float) -> GrpoRewardBreakdown:
    return GrpoRewardBreakdown(
        task_component=task,
        transfer_component=transfer,
        memory_cost_component=cost,
        total_reward=task + transfer + cost,
    )


def test_summary_aggregates_reward_components_and_transfer_rates() -> None:
    metrics = summarize_grpo_breakdowns(
        [
            _breakdown(1.0, 0.2, -0.01),
            _breakdown(0.5, -0.3, -0.01),
            _breakdown(0.8, 0.0, 0.0),
        ]
    )

    assert metrics.sample_count == 3
    assert metrics.mean_total_reward == pytest.approx((1.19 + 0.19 + 0.8) / 3)
    assert metrics.mean_task_component == pytest.approx(2.3 / 3)
    assert metrics.mean_transfer_component == pytest.approx(-0.1 / 3)
    assert metrics.mean_memory_cost_component == pytest.approx(-0.02 / 3)
    assert metrics.positive_transfer_rate == pytest.approx(1 / 3)
    assert metrics.negative_transfer_rate == pytest.approx(1 / 3)


def test_summary_treats_zero_transfer_as_neutral() -> None:
    metrics = summarize_grpo_breakdowns([_breakdown(0.5, 0.0, 0.0)])

    assert metrics.positive_transfer_rate == 0.0
    assert metrics.negative_transfer_rate == 0.0


def test_summary_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_grpo_breakdowns([])


def test_summary_rejects_untyped_breakdown() -> None:
    with pytest.raises(TypeError, match="GrpoRewardBreakdown"):
        summarize_grpo_breakdowns([object()])  # type: ignore[list-item]
