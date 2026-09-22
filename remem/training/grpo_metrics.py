"""Aggregate auditable GRPO reward attribution for trainer observability."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from remem.training.grpo import GrpoRewardBreakdown


@dataclass(frozen=True, slots=True)
class GrpoBatchMetrics:
    """Aggregate reward components for one non-empty trainer batch."""

    sample_count: int
    mean_total_reward: float
    mean_task_component: float
    mean_transfer_component: float
    mean_memory_cost_component: float
    positive_transfer_rate: float
    negative_transfer_rate: float

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        numeric_values = (
            self.mean_total_reward,
            self.mean_task_component,
            self.mean_transfer_component,
            self.mean_memory_cost_component,
            self.positive_transfer_rate,
            self.negative_transfer_rate,
        )
        if not all(isfinite(value) for value in numeric_values):
            raise ValueError("GRPO batch metrics must be finite")
        for rate in (self.positive_transfer_rate, self.negative_transfer_rate):
            if not 0.0 <= rate <= 1.0:
                raise ValueError("GRPO transfer rates must be between zero and one")


def summarize_grpo_breakdowns(
    breakdowns: Sequence[GrpoRewardBreakdown],
) -> GrpoBatchMetrics:
    """Summarize typed reward attribution without inferring unmeasured outcomes.

    Positive and negative transfer rates use the full batch as denominator. A zero
    transfer component is deliberately neutral rather than being classified as either
    positive or negative transfer.
    """
    if not breakdowns:
        raise ValueError("at least one GRPO reward breakdown is required")
    if any(not isinstance(item, GrpoRewardBreakdown) for item in breakdowns):
        raise TypeError("breakdowns must contain only GrpoRewardBreakdown values")

    sample_count = len(breakdowns)
    denominator = float(sample_count)
    return GrpoBatchMetrics(
        sample_count=sample_count,
        mean_total_reward=sum(item.total_reward for item in breakdowns) / denominator,
        mean_task_component=sum(item.task_component for item in breakdowns) / denominator,
        mean_transfer_component=sum(item.transfer_component for item in breakdowns) / denominator,
        mean_memory_cost_component=(
            sum(item.memory_cost_component for item in breakdowns) / denominator
        ),
        positive_transfer_rate=(
            sum(item.transfer_component > 0.0 for item in breakdowns) / denominator
        ),
        negative_transfer_rate=(
            sum(item.transfer_component < 0.0 for item in breakdowns) / denominator
        ),
    )


__all__ = ["GrpoBatchMetrics", "summarize_grpo_breakdowns"]
