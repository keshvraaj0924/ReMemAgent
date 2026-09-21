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


@dataclass(frozen=True, slots=True)
class GrpoRewardBreakdown:
    """Auditable components contributing to one scalar GRPO reward."""

    task_component: float
    transfer_component: float
    memory_cost_component: float
    total_reward: float

    def __post_init__(self) -> None:
        values = (
            self.task_component,
            self.transfer_component,
            self.memory_cost_component,
            self.total_reward,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("GRPO reward components must be finite")


def compute_grpo_reward_breakdown(
    trajectory: GrpoTrajectory,
    config: GrpoRewardConfig | None = None,
) -> GrpoRewardBreakdown:
    """Return independently inspectable reward components for one trajectory.

    Counterfactual transfer and memory-use cost are applied only when the memory route
    was actually selected. Keeping these terms separate lets training logs distinguish
    task success from positive transfer, negative transfer, and memory-efficiency cost.
    """

    if not isinstance(trajectory, GrpoTrajectory):
        raise TypeError("trajectory must be a GrpoTrajectory")
    if config is not None and not isinstance(config, GrpoRewardConfig):
        raise TypeError("config must be a GrpoRewardConfig")

    reward_config = config or GrpoRewardConfig()
    task_component = reward_config.task_reward_weight * trajectory.task_reward
    transfer_component = 0.0
    memory_cost_component = 0.0

    if trajectory.memory_used:
        memory_cost_component = -reward_config.memory_use_cost
        transfer_weight = (
            reward_config.positive_transfer_weight
            if trajectory.counterfactual_delta >= 0.0
            else reward_config.negative_transfer_weight
        )
        transfer_component = transfer_weight * trajectory.counterfactual_delta

    total_reward = task_component + transfer_component + memory_cost_component
    return GrpoRewardBreakdown(
        task_component=task_component,
        transfer_component=transfer_component,
        memory_cost_component=memory_cost_component,
        total_reward=total_reward,
    )


def compute_grpo_reward(
    trajectory: GrpoTrajectory,
    config: GrpoRewardConfig | None = None,
) -> float:
    """Return task reward adjusted for measured memory transfer and usage cost."""

    return compute_grpo_reward_breakdown(trajectory, config).total_reward


__all__ = [
    "GrpoRewardBreakdown",
    "GrpoRewardConfig",
    "GrpoTrajectory",
    "compute_grpo_reward",
    "compute_grpo_reward_breakdown",
]
