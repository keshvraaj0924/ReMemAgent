import math

import pytest

from experiments.synthetic_negative_transfer import BenchmarkCase, run_benchmark
from remem.routing.counterfactual import CounterfactualRouter


def test_router_rejects_memory_when_counterfactual_value_is_lower() -> None:
    cases = [
        BenchmarkCase("positive", 0.90, 0.70),
        BenchmarkCase("negative", 0.40, 0.80),
    ]

    result = run_benchmark(cases, CounterfactualRouter(minimum_delta=0.05))

    assert result.total_cases == 2
    assert result.memory_selected == 1
    assert result.self_reasoning_selected == 1
    assert result.negative_transfer_rate == 0.5
    assert result.selected_negative_transfer_cases == 0
    assert result.avoided_negative_transfer_cases == 1
    assert result.negative_transfer_avoidance_rate == 1.0
    assert result.memory_induced_negative_transfer_rate == 0.0
    assert result.routing_regret == 0.0
    assert result.mean_routing_regret == 0.0


def test_benchmark_reports_regret_when_router_selects_harmful_memory() -> None:
    result = run_benchmark(
        [BenchmarkCase("negative", 0.40, 0.80)],
        CounterfactualRouter(minimum_delta=-0.5),
    )

    assert result.memory_selected == 1
    assert result.selected_negative_transfer_cases == 1
    assert result.avoided_negative_transfer_cases == 0
    assert result.memory_induced_negative_transfer_rate == 1.0
    assert result.negative_transfer_avoidance_rate == 0.0
    assert result.routing_regret == pytest.approx(0.40)
    assert result.mean_routing_regret == pytest.approx(0.40)


def test_benchmark_reports_regret_when_router_rejects_beneficial_memory() -> None:
    result = run_benchmark(
        [BenchmarkCase("beneficial", 0.90, 0.60)],
        CounterfactualRouter(minimum_delta=0.5),
    )

    assert result.memory_selected == 0
    assert result.self_reasoning_selected == 1
    assert result.negative_transfer_cases == 0
    assert result.routing_regret == pytest.approx(0.30)
    assert result.mean_routing_regret == pytest.approx(0.30)


def test_mean_routing_regret_is_normalized_across_cases() -> None:
    result = run_benchmark(
        [
            BenchmarkCase("missed-memory", 0.90, 0.60),
            BenchmarkCase("correct-self", 0.40, 0.80),
        ],
        CounterfactualRouter(minimum_delta=0.5),
    )

    assert result.routing_regret == pytest.approx(0.30)
    assert result.mean_routing_regret == pytest.approx(0.15)


def test_empty_benchmark_has_zero_mean_routing_regret() -> None:
    result = run_benchmark([], CounterfactualRouter())

    assert result.total_cases == 0
    assert result.routing_regret == 0.0
    assert result.mean_routing_regret == 0.0


def test_benchmark_rejects_duplicate_case_ids() -> None:
    cases = [BenchmarkCase("duplicate", 0.8, 0.7), BenchmarkCase("duplicate", 0.7, 0.8)]

    with pytest.raises(ValueError, match="benchmark case_id values must be unique"):
        run_benchmark(cases, CounterfactualRouter())


@pytest.mark.parametrize("invalid_utility", [math.nan, math.inf, -math.inf])
def test_benchmark_case_rejects_non_finite_memory_utility(invalid_utility: float) -> None:
    with pytest.raises(ValueError, match="utility_with_memory must be finite"):
        BenchmarkCase("invalid-memory", invalid_utility, 0.5)


@pytest.mark.parametrize("invalid_utility", [math.nan, math.inf, -math.inf])
def test_benchmark_case_rejects_non_finite_self_reasoning_utility(
    invalid_utility: float,
) -> None:
    with pytest.raises(ValueError, match="utility_without_memory must be finite"):
        BenchmarkCase("invalid-self-reasoning", 0.5, invalid_utility)
