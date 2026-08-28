from experiments.ablations import AblationStrategy, run_ablations
from experiments.synthetic_negative_transfer import BenchmarkCase
from remem.routing.counterfactual import CounterfactualRouter


def test_ablations_compare_fixed_and_counterfactual_strategies() -> None:
    cases = [
        BenchmarkCase("helpful", 0.9, 0.6),
        BenchmarkCase("harmful", 0.3, 0.8),
    ]

    results = run_ablations(cases, CounterfactualRouter(minimum_delta=0.05))
    by_strategy = {result.strategy: result for result in results}

    assert by_strategy[AblationStrategy.MEMORY_ALWAYS].selected_memory == 2
    assert by_strategy[AblationStrategy.SELF_REASONING_ALWAYS].selected_memory == 0
    assert by_strategy[AblationStrategy.COUNTERFACTUAL].selected_memory == 1
    assert by_strategy[AblationStrategy.COUNTERFACTUAL].mean_utility == 0.85
    assert by_strategy[AblationStrategy.COUNTERFACTUAL].routing_regret == 0.0


def test_ablation_reports_memory_induced_negative_transfer() -> None:
    results = run_ablations(
        [BenchmarkCase("harmful", 0.2, 0.9)],
        CounterfactualRouter(minimum_delta=-1.0),
    )

    result = next(item for item in results if item.strategy is AblationStrategy.COUNTERFACTUAL)
    assert result.selected_negative_transfer_cases == 1
    assert result.negative_transfer_rate == 1.0
    assert result.routing_regret == 0.7


def test_ablation_rejects_duplicate_case_ids() -> None:
    cases = [BenchmarkCase("same", 0.8, 0.7), BenchmarkCase("same", 0.7, 0.8)]

    try:
        run_ablations(cases)
    except ValueError as error:
        assert str(error) == "benchmark case_id values must be unique"
    else:
        raise AssertionError("Expected duplicate case identifiers to be rejected")
