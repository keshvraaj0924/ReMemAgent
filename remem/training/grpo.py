"""Framework-neutral reward contract for GRPO-style memory-policy training.

This module intentionally does not import verl. It defines the measured trajectory
and deterministic reward composition that a verl-agent adapter can consume later.
Keeping this boundary dependency-free makes reward semantics independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class GrpoRewardConfig:
    """Weights used to convert measured memory behavior into a scalar reward."""

    task_reward_weight: float = 1.0
    positive_transfer_weight: float = 0.5
    negative_transfer_weight: float = 1.0
    memory_use_cost: float = 0.01

    def __post_init__(self) -> None:
        values = (
            self.task_reward_weight,
            self.positive_transfer_weight,
            self.negative_transfer_weight,
            self.memory_use_cost,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("GRPO reward weights must be finite")
        if any(value < 0.0 for value in values):
            raise ValueError("GRPO reward weights must be non-negative")


@dataclass(frozen=True, slots=True)
class GrpoTrajectory:
    """Measured outcome required to train a memory-routing policy."""

    task_reward: float
    memory_used: bool
    counterfactual_delta: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.task_reward):
            raise ValueError("task_reward must be finite")
        if not isfinite(self.counterfactual_delta):
            raise ValueError("counterfactual_delta must be finite")


def compute_grpo_reward(
    trajectory: GrpoTrajectory,
    config: GrpoRewardConfig | None = None,
) -> float:
    """Return task reward adjusted for measured memory transfer and usage cost.

    Positive counterfactual delta rewards memory that improved expected utility;
    negative delta is penalized more strongly by default to discourage harmful
    transfer. Memory cost applies only when the memory route was actually used.
    """

    reward_config = config or GrpoRewardConfig()
    reward = reward_config.task_reward_weight * trajectory.task_reward

    if trajectory.memory_used:
        reward -= reward_config.memory_use_cost
        if trajectory.counterfactual_delta >= 0.0:
            reward += reward_config.positive_transfer_weight * trajectory.counterfactual_delta
        else:
            reward += reward_config.negative_transfer_weight * trajectory.counterfactual_delta

    if not isfinite(reward):
        raise ValueError("computed GRPO reward must be finite")
    return reward


__all__ = ["GrpoRewardConfig", "GrpoTrajectory", "compute_grpo_reward"]
