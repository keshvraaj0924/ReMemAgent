from experiments.memory_transfer import summarize_memory_transfers
from experiments.synthetic_negative_transfer import BenchmarkCase, run_benchmark
from remem.routing.counterfactual import CounterfactualRouter


def test_memory_identity_is_preserved_in_case_results() -> None:
    result = run_benchmark(
        [BenchmarkCase("negative", 0.40, 0.80, memory_id="memory_a")],
        CounterfactualRouter(minimum_delta=-0.5),
    )

    assert result.case_results[0].memory_id == "memory_a"


def test_memory_transfer_summary_groups_selected_cases() -> None:
    result = run_benchmark(
        [
            BenchmarkCase("helpful", 0.90, 0.70, memory_id="memory_a"),
            BenchmarkCase("harmful", 0.40, 0.80, memory_id="memory_a"),
            BenchmarkCase("unattributed", 0.40, 0.80),
        ],
        CounterfactualRouter(minimum_delta=-0.5),
    )

    summaries = summarize_memory_transfers(result.case_results)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.memory_id == "memory_a"
    assert summary.transfer_attempts == 2
    assert summary.negative_transfer_cases == 1
    assert summary.negative_transfer_rate == 0.5
    assert summary.routing_regret == 0.40


def test_memory_transfer_summary_excludes_self_reasoning_cases() -> None:
    result = run_benchmark(
        [BenchmarkCase("avoided", 0.40, 0.80, memory_id="memory_a")],
        CounterfactualRouter(minimum_delta=0.05),
    )

    assert summarize_memory_transfers(result.case_results) == ()


def test_benchmark_rejects_blank_memory_id() -> None:
    try:
        BenchmarkCase("case", 0.8, 0.7, memory_id="   ")
    except ValueError as error:
        assert str(error) == "memory_id must not be empty when provided"
    else:
        raise AssertionError("Expected blank memory identifiers to be rejected")
