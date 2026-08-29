"""Derived metrics for comparing matched ablation strategies."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.ablations import AblationResult, AblationStrategy


@dataclass(frozen=True, slots=True)
class StrategyComparison:
    """A strategy's performance relative to the self-reasoning baseline."""

    strategy: AblationStrategy
    mean_utility: float
    utility_delta: float
    negative_transfer_rate: float
    selected_memory_rate: float
    routing_regret: float


def compare_strategies(results: list[AblationResult]) -> list[StrategyComparison]:
    """Normalize strategy metrics against exactly one self-reasoning baseline."""

    baselines = [
        result for result in results if result.strategy is AblationStrategy.SELF_REASONING_ALWAYS
    ]
    if len(baselines) != 1:
        raise ValueError("results must contain exactly one self-reasoning baseline")

    baseline = baselines[0]
    return [
        StrategyComparison(
            strategy=result.strategy,
            mean_utility=result.mean_utility,
            utility_delta=result.mean_utility - baseline.mean_utility,
            negative_transfer_rate=result.negative_transfer_rate,
            selected_memory_rate=(
                result.selected_memory / result.total_cases if result.total_cases else 0.0
            ),
            routing_regret=result.routing_regret,
        )
        for result in results
    ]
