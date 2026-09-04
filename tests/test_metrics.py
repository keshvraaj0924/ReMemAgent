import math

import pytest

from experiments.ablations import AblationResult, AblationStrategy, run_ablations
from experiments.metrics import compare_strategies
from experiments.synthetic_negative_transfer import BenchmarkCase
from remem.routing.counterfactual import CounterfactualRouter


def test_compare_strategies_uses_self_reasoning_as_baseline() -> None:
    results = run_ablations(
        [BenchmarkCase("helpful", 0.9, 0.6), BenchmarkCase("harmful", 0.3, 0.8)],
        CounterfactualRouter(minimum_delta=0.05),
    )

    comparisons = compare_strategies(results)
    by_strategy = {item.strategy: item for item in comparisons}

    assert by_strategy[AblationStrategy.SELF_REASONING_ALWAYS].utility_delta == pytest.approx(0.0)
    assert by_strategy[AblationStrategy.MEMORY_ALWAYS].utility_delta == pytest.approx(-0.1)
    assert by_strategy[AblationStrategy.COUNTERFACTUAL].utility_delta == pytest.approx(0.15)
    assert by_strategy[AblationStrategy.COUNTERFACTUAL].selected_memory_rate == pytest.approx(0.5)


def test_compare_strategies_rejects_missing_baseline() -> None:
    results = run_ablations([BenchmarkCase("case", 0.8, 0.7)])
    without_baseline = [
        result
        for result in results
        if result.strategy is not AblationStrategy.SELF_REASONING_ALWAYS
    ]

    with pytest.raises(ValueError, match="results must contain exactly one self-reasoning baseline"):
        compare_strategies(without_baseline)


def test_compare_strategies_rejects_duplicate_strategies() -> None:
    result = AblationResult(
        strategy=AblationStrategy.MEMORY_ALWAYS,
        total_cases=1,
        selected_memory=1,
        mean_utility=0.5,
        negative_transfer_cases=0,
        selected_negative_transfer_cases=0,
        routing_regret=0.0,
    )

    with pytest.raises(ValueError, match="each strategy at most once"):
        compare_strategies([result, result])


def test_compare_strategies_rejects_inconsistent_counts() -> None:
    result = AblationResult(
        strategy=AblationStrategy.SELF_REASONING_ALWAYS,
        total_cases=2,
        selected_memory=3,
        mean_utility=0.5,
        negative_transfer_cases=0,
        selected_negative_transfer_cases=0,
        routing_regret=0.0,
    )

    with pytest.raises(ValueError, match="selected_memory must be between"):
        compare_strategies([result])


def test_compare_strategies_rejects_non_finite_metrics() -> None:
    result = AblationResult(
        strategy=AblationStrategy.SELF_REASONING_ALWAYS,
        total_cases=1,
        selected_memory=0,
        mean_utility=math.nan,
        negative_transfer_cases=0,
        selected_negative_transfer_cases=0,
        routing_regret=0.0,
    )

    with pytest.raises(ValueError, match="mean_utility must be finite"):
        compare_strategies([result])
