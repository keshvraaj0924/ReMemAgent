"""Executable, dependency-light batch boundary for verl reward integration.

This module keeps trainer transport concerns separate from ReMemAgent's deterministic
reward semantics.  A trainer supplies an ordered batch of plain mappings; ReMemAgent
returns scalar rewards and an auditable record derived from the exact same breakdowns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from remem.training.grpo_metrics import summarize_grpo_breakdowns
from remem.training.verl_adapter import VerlRewardAdapter
from remem.training.verl_records import VerlBatchRewardRecord, VerlRewardRecord


@dataclass(frozen=True, slots=True)
class VerlBatchRewardResult:
    """Scalar trainer rewards paired with their auditable attribution record."""

    rewards: tuple[float, ...]
    record: VerlBatchRewardRecord

    def __post_init__(self) -> None:
        if not self.rewards:
            raise ValueError("at least one scalar reward is required")
        recorded_rewards = tuple(item.total_reward for item in self.record.rewards)
        if self.rewards != recorded_rewards:
            raise ValueError("scalar rewards must match the auditable reward record")


def compute_verl_batch_rewards(
    samples: Sequence[Mapping[str, Any]],
    *,
    adapter: VerlRewardAdapter | None = None,
) -> VerlBatchRewardResult:
    """Compute one ordered trainer batch through a single canonical reward pass.

    The breakdowns are computed exactly once and are then used for both scalar rewards
    and aggregate metrics.  This prevents trainer-facing rewards and observability from
    silently diverging when the reward implementation evolves.
    """
    reward_adapter = adapter or VerlRewardAdapter()
    if not isinstance(reward_adapter, VerlRewardAdapter):
        raise TypeError("adapter must be a VerlRewardAdapter")
    if not samples:
        raise ValueError("at least one trainer sample is required")

    breakdowns = reward_adapter.compute_batch_breakdowns(samples)
    records = tuple(
        VerlRewardRecord.from_breakdown(index, breakdown)
        for index, breakdown in enumerate(breakdowns)
    )
    record = VerlBatchRewardRecord(
        rewards=records,
        metrics=summarize_grpo_breakdowns(breakdowns),
    )
    rewards = tuple(item.total_reward for item in records)
    return VerlBatchRewardResult(rewards=rewards, record=record)


__all__ = ["VerlBatchRewardResult", "compute_verl_batch_rewards"]
