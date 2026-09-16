"""Dependency-light adapter between verl-style samples and ReMemAgent rewards.

The adapter intentionally accepts plain mappings instead of importing verl. This keeps
ReMemAgent's reward semantics independently testable while providing a stable callable
boundary for trainer integrations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from remem.training.grpo import GrpoRewardConfig, GrpoTrajectory, compute_grpo_reward


@dataclass(frozen=True, slots=True)
class VerlRewardFields:
    """Trainer metadata keys used to construct a measured GRPO trajectory."""

    task_reward: str = "task_reward"
    memory_used: str = "memory_used"
    counterfactual_delta: str = "counterfactual_delta"

    def __post_init__(self) -> None:
        values = (self.task_reward, self.memory_used, self.counterfactual_delta)
        if any(not value.strip() for value in values):
            raise ValueError("verl reward field names must be non-empty")
        if len(set(values)) != len(values):
            raise ValueError("verl reward field names must be unique")


@dataclass(frozen=True, slots=True)
class VerlRewardAdapter:
    """Convert trainer sample metadata into the framework-neutral GRPO reward."""

    config: GrpoRewardConfig = field(default_factory=GrpoRewardConfig)
    fields: VerlRewardFields = field(default_factory=VerlRewardFields)

    def __call__(
        self,
        sample: Mapping[str, Any],
        solution_str: str | None = None,
        ground_truth: str | None = None,
        extra_info: Mapping[str, Any] | None = None,
    ) -> float:
        """Compute reward from either direct metadata or a verl reward-manager call.

        ``solution_str`` and ``ground_truth`` are accepted for compatibility with
        reward-manager call sites but are intentionally not used to infer memory
        transfer. Transfer must come from measured trajectory metadata.
        """
        del solution_str, ground_truth
        reward_sample = self._merge_extra_info(sample, extra_info)
        return compute_grpo_reward(self.to_trajectory(reward_sample), self.config)

    def from_extra_info(
        self,
        solution_str: str,
        ground_truth: str,
        extra_info: Mapping[str, Any],
    ) -> float:
        """Compute reward for verl call sites that carry measurements in ``extra_info``.

        Generated and reference text are deliberately ignored. The method exists so a
        trainer can pass the common ``(solution_str, ground_truth, extra_info)`` shape
        without manufacturing an otherwise-empty sample mapping.
        """
        return self(
            {},
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

    def compute_batch(self, samples: Sequence[Mapping[str, Any]]) -> list[float]:
        """Compute rewards for an ordered trainer batch without mutating its samples."""
        return [self(sample) for sample in samples]

    @staticmethod
    def _merge_extra_info(
        sample: Mapping[str, Any], extra_info: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """Merge trainer metadata while rejecting ambiguous duplicate measurements."""
        merged = dict(sample)
        if extra_info is None:
            return merged

        for key, value in extra_info.items():
            if key in merged and merged[key] != value:
                raise ValueError(f"conflicting reward field: {key}")
            merged[key] = value
        return merged

    def to_trajectory(self, sample: Mapping[str, Any]) -> GrpoTrajectory:
        """Validate and convert a trainer sample into a typed trajectory."""
        try:
            task_reward = sample[self.fields.task_reward]
            memory_used = sample[self.fields.memory_used]
        except KeyError as exc:
            raise ValueError(f"missing required reward field: {exc.args[0]}") from exc

        if isinstance(task_reward, bool) or not isinstance(task_reward, (int, float)):
            raise TypeError(f"{self.fields.task_reward} must be a real number")
        if not isinstance(memory_used, bool):
            raise TypeError(f"{self.fields.memory_used} must be a boolean")

        counterfactual_delta = sample.get(self.fields.counterfactual_delta, 0.0)
        if isinstance(counterfactual_delta, bool) or not isinstance(
            counterfactual_delta, (int, float)
        ):
            raise TypeError(f"{self.fields.counterfactual_delta} must be a real number")

        return GrpoTrajectory(
            task_reward=float(task_reward),
            memory_used=memory_used,
            counterfactual_delta=float(counterfactual_delta),
        )


__all__ = ["VerlRewardAdapter", "VerlRewardFields"]
