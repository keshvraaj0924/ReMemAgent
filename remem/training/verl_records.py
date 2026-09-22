"""Serializable trainer records for auditable verl reward integration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

from remem.training.grpo import GrpoRewardBreakdown
from remem.training.grpo_metrics import GrpoBatchMetrics
from remem.training.verl_adapter import VerlRewardAdapter


@dataclass(frozen=True, slots=True)
class VerlRewardRecord:
    """Trainer-safe reward attribution for one ordered sample."""

    sample_index: int
    task_component: float
    transfer_component: float
    memory_cost_component: float
    total_reward: float

    def __post_init__(self) -> None:
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        values = (
            self.task_component,
            self.transfer_component,
            self.memory_cost_component,
            self.total_reward,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("reward record components must be finite")

    @classmethod
    def from_breakdown(cls, sample_index: int, breakdown: GrpoRewardBreakdown) -> VerlRewardRecord:
        """Create a record from the framework-neutral reward attribution."""
        if not isinstance(breakdown, GrpoRewardBreakdown):
            raise TypeError("breakdown must be a GrpoRewardBreakdown")
        return cls(
            sample_index=sample_index,
            task_component=breakdown.task_component,
            transfer_component=breakdown.transfer_component,
            memory_cost_component=breakdown.memory_cost_component,
            total_reward=breakdown.total_reward,
        )

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerlBatchRewardRecord:
    """Ordered reward records plus aggregate metrics for one trainer batch."""

    rewards: tuple[VerlRewardRecord, ...]
    metrics: GrpoBatchMetrics

    def __post_init__(self) -> None:
        if not self.rewards:
            raise ValueError("at least one reward record is required")
        expected_indices = tuple(range(len(self.rewards)))
        actual_indices = tuple(record.sample_index for record in self.rewards)
        if actual_indices != expected_indices:
            raise ValueError("reward records must preserve contiguous batch order")
        if self.metrics.sample_count != len(self.rewards):
            raise ValueError("metrics sample_count must match reward records")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable trainer logging payload."""
        return {
            "rewards": [record.to_dict() for record in self.rewards],
            "metrics": asdict(self.metrics),
        }


def build_verl_batch_reward_record(
    adapter: VerlRewardAdapter,
    samples: Sequence[Mapping[str, Any]],
) -> VerlBatchRewardRecord:
    """Build one auditable batch record from the adapter's canonical reward path."""
    if not isinstance(adapter, VerlRewardAdapter):
        raise TypeError("adapter must be a VerlRewardAdapter")

    breakdowns = adapter.compute_batch_breakdowns(samples)
    metrics = adapter.summarize_batch(samples)
    records = tuple(
        VerlRewardRecord.from_breakdown(index, breakdown)
        for index, breakdown in enumerate(breakdowns)
    )
    return VerlBatchRewardRecord(rewards=records, metrics=metrics)


__all__ = [
    "VerlBatchRewardRecord",
    "VerlRewardRecord",
    "build_verl_batch_reward_record",
]
