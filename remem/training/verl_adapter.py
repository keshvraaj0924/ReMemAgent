"""Dependency-light adapter between verl-style samples and ReMemAgent rewards.

The adapter intentionally accepts plain mappings instead of importing verl. This keeps
ReMemAgent's reward semantics independently testable while providing a stable callable
boundary for trainer integrations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from remem.training.grpo import GrpoRewardConfig, GrpoTrajectory, compute_grpo_reward


@dataclass(frozen=True, slots=True)
class VerlRewardAdapter:
    """Convert trainer sample metadata into the framework-neutral GRPO reward."""

    config: GrpoRewardConfig = GrpoRewardConfig()

    def __call__(self, sample: Mapping[str, Any]) -> float:
        """Compute reward from a trainer sample containing measured trajectory fields."""
        return compute_grpo_reward(self.to_trajectory(sample), self.config)

    def compute_batch(self, samples: Sequence[Mapping[str, Any]]) -> list[float]:
        """Compute rewards for an ordered trainer batch without mutating its samples."""
        return [self(sample) for sample in samples]

    @staticmethod
    def to_trajectory(sample: Mapping[str, Any]) -> GrpoTrajectory:
        """Validate and convert a trainer sample into a typed trajectory."""
        try:
            task_reward = sample["task_reward"]
            memory_used = sample["memory_used"]
        except KeyError as exc:
            raise ValueError(f"missing required reward field: {exc.args[0]}") from exc

        if isinstance(task_reward, bool) or not isinstance(task_reward, (int, float)):
            raise TypeError("task_reward must be a real number")
        if not isinstance(memory_used, bool):
            raise TypeError("memory_used must be a boolean")

        counterfactual_delta = sample.get("counterfactual_delta", 0.0)
        if isinstance(counterfactual_delta, bool) or not isinstance(
            counterfactual_delta, (int, float)
        ):
            raise TypeError("counterfactual_delta must be a real number")

        return GrpoTrajectory(
            task_reward=float(task_reward),
            memory_used=memory_used,
            counterfactual_delta=float(counterfactual_delta),
        )


__all__ = ["VerlRewardAdapter"]
