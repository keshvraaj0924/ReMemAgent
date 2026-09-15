"""Derived metrics for comparing matched memory-routing ablations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from experiments.ablations import AblationResult, AblationStrategy


@dataclass(frozen=True, slots=True)
class StrategyComparison:
    """One strategy's measured outcome relative to self reasoning."""

    strategy: AblationStrategy
    mean_utility: float
    utility_delta: float
    selected_memory_rate: float
    negative_transfer_rate: float
    routing_regret: float

    def __post_init__(self) -> None:
        for value_name in ("mean_utility", "utility_delta", "routing_regret"):
            if not isfinite(getattr(self, value_name)):
                raise ValueError(f"{value_name} must be finite")
        for rate_name in ("selected_memory_rate", "negative_transfer_rate"):
            rate = getattr(self, rate_name)
            if not isfinite(rate) or not 0.0 <= rate <= 1.0:
                raise ValueError(f"{rate_name} must be finite and within [0, 1]")


def compare_strategies(results: Sequence[AblationResult]) -> tuple[StrategyComparison, ...]:
    """Compare matched ablation results against self-reasoning utility.

    The function derives metrics only from supplied measured ablation results.
    It requires exactly one self-reasoning baseline and rejects inconsistent
    case counts so deltas are not reported across unmatched evaluations.
    """

    selected_results = tuple(results)
    baselines = tuple(
        result
        for result in selected_results
        if result.strategy is AblationStrategy.SELF_REASONING_ALWAYS
    )
    if len(baselines) != 1:
        raise ValueError("results must contain exactly one self-reasoning baseline")

    baseline = baselines[0]
    if any(result.total_cases != baseline.total_cases for result in selected_results):
        raise ValueError("strategy results must contain the same number of cases")

    return tuple(
        StrategyComparison(
            strategy=result.strategy,
            mean_utility=result.mean_utility,
            utility_delta=result.mean_utility - baseline.mean_utility,
            selected_memory_rate=(
                result.selected_memory / result.total_cases if result.total_cases else 0.0
            ),
            negative_transfer_rate=result.negative_transfer_rate,
            routing_regret=result.routing_regret,
        )
        for result in selected_results
    )


__all__ = ["StrategyComparison", "compare_strategies"]
