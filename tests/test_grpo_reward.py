"""Tests for framework-neutral GRPO reward composition."""

import pytest

from remem.training.grpo import (
    GrpoRewardConfig,
    GrpoTrajectory,
    compute_grpo_reward,
    compute_grpo_reward_breakdown,
)


def test_reward_preserves_task_reward_without_memory_use() -> None:
    trajectory = GrpoTrajectory(task_reward=0.75, memory_used=False, counterfactual_delta=-0.8)

    assert compute_grpo_reward(trajectory) == pytest.approx(0.75)


def test_reward_credits_positive_transfer_and_charges_memory_cost() -> None:
    trajectory = GrpoTrajectory(task_reward=1.0, memory_used=True, counterfactual_delta=0.4)

    assert compute_grpo_reward(trajectory) == pytest.approx(1.19)


def test_reward_penalizes_negative_transfer() -> None:
    trajectory = GrpoTrajectory(task_reward=1.0, memory_used=True, counterfactual_delta=-0.4)

    assert compute_grpo_reward(trajectory) == pytest.approx(0.59)


def test_reward_weights_are_configurable() -> None:
    config = GrpoRewardConfig(
        task_reward_weight=2.0,
        positive_transfer_weight=1.0,
        negative_transfer_weight=3.0,
        memory_use_cost=0.2,
    )
    trajectory = GrpoTrajectory(task_reward=0.5, memory_used=True, counterfactual_delta=-0.1)

    assert compute_grpo_reward(trajectory, config) == pytest.approx(0.5)


def test_reward_breakdown_exposes_positive_transfer_attribution() -> None:
    trajectory = GrpoTrajectory(task_reward=1.0, memory_used=True, counterfactual_delta=0.4)

    breakdown = compute_grpo_reward_breakdown(trajectory)

    assert breakdown.task_component == pytest.approx(1.0)
    assert breakdown.transfer_component == pytest.approx(0.2)
    assert breakdown.memory_cost_component == pytest.approx(-0.01)
    assert breakdown.total_reward == pytest.approx(1.19)
    assert compute_grpo_reward(trajectory) == pytest.approx(breakdown.total_reward)


def test_reward_breakdown_exposes_negative_transfer_attribution() -> None:
    trajectory = GrpoTrajectory(task_reward=1.0, memory_used=True, counterfactual_delta=-0.4)

    breakdown = compute_grpo_reward_breakdown(trajectory)

    assert breakdown.transfer_component == pytest.approx(-0.4)
    assert breakdown.memory_cost_component == pytest.approx(-0.01)
    assert breakdown.total_reward == pytest.approx(0.59)


def test_reward_breakdown_ignores_transfer_when_memory_was_not_used() -> None:
    trajectory = GrpoTrajectory(task_reward=0.75, memory_used=False, counterfactual_delta=0.9)

    breakdown = compute_grpo_reward_breakdown(trajectory)

    assert breakdown.task_component == pytest.approx(0.75)
    assert breakdown.transfer_component == pytest.approx(0.0)
    assert breakdown.memory_cost_component == pytest.approx(0.0)
    assert breakdown.total_reward == pytest.approx(0.75)


def test_reward_breakdown_rejects_wrong_input_types() -> None:
    with pytest.raises(TypeError, match="trajectory must"):
        compute_grpo_reward_breakdown(object())  # type: ignore[arg-type]

    trajectory = GrpoTrajectory(task_reward=1.0, memory_used=False)
    with pytest.raises(TypeError, match="config must"):
        compute_grpo_reward_breakdown(trajectory, object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "config",
    [
        GrpoRewardConfig,
    ],
)
def test_config_type_is_constructible(config: type[GrpoRewardConfig]) -> None:
    assert config() == GrpoRewardConfig()


def test_negative_reward_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        GrpoRewardConfig(memory_use_cost=-0.1)


def test_non_finite_trajectory_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        GrpoTrajectory(task_reward=float("nan"), memory_used=False)
