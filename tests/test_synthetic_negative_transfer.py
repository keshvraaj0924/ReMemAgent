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
    assert result.routing_regret == 0.40


def test_benchmark_rejects_duplicate_case_ids() -> None:
    cases = [BenchmarkCase("duplicate", 0.8, 0.7), BenchmarkCase("duplicate", 0.7, 0.8)]

    try:
        run_benchmark(cases, CounterfactualRouter())
    except ValueError as error:
        assert str(error) == "benchmark case_id values must be unique"
    else:
        raise AssertionError("Expected duplicate case identifiers to be rejected")
