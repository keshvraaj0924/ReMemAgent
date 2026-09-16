"""Tests for framework-neutral GRPO reward composition."""

import pytest

from remem.training.grpo import GrpoRewardConfig, GrpoTrajectory, compute_grpo_reward


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
