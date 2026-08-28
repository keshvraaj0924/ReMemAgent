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
