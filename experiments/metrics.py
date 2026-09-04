"""Derived metrics for comparing matched ablation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

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

    _validate_results(results)
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


def _validate_results(results: list[AblationResult]) -> None:
    """Reject malformed aggregate metrics before deriving research statistics."""

    if not results:
        raise ValueError("results must not be empty")

    seen_strategies: set[AblationStrategy] = set()
    for result in results:
        if not isinstance(result, AblationResult):
            raise TypeError("results must contain only AblationResult values")
        if result.strategy in seen_strategies:
            raise ValueError("results must contain each strategy at most once")
        seen_strategies.add(result.strategy)
        if isinstance(result.total_cases, bool) or result.total_cases < 0:
            raise ValueError("total_cases must be a non-negative integer")
        if isinstance(result.selected_memory, bool) or not 0 <= result.selected_memory <= result.total_cases:
            raise ValueError("selected_memory must be between zero and total_cases")
        if isinstance(result.negative_transfer_cases, bool) or not 0 <= result.negative_transfer_cases <= result.total_cases:
            raise ValueError("negative_transfer_cases must be between zero and total_cases")
        if isinstance(result.selected_negative_transfer_cases, bool) or not 0 <= result.selected_negative_transfer_cases <= result.selected_memory:
            raise ValueError(
                "selected_negative_transfer_cases must be between zero and selected_memory"
            )
        for field_name in ("mean_utility", "routing_regret"):
            if not isfinite(getattr(result, field_name)):
                raise ValueError(f"{field_name} must be finite")
        if result.routing_regret < 0.0:
            raise ValueError("routing_regret must be non-negative")
