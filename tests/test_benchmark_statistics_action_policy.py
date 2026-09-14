from __future__ import annotations

from experiments.benchmark_statistics import compare_benchmark_reports
from remem.benchmark import BenchmarkRunConfiguration, BenchmarkRunReport


def _action_policy_report(seed: int, action_policy_factory: str) -> BenchmarkRunReport:
    configuration = BenchmarkRunConfiguration(
        benchmark_name="synthetic-action-policy",
        episode_count=0,
        max_steps=1,
        seed=seed,
        environment_factory="tests.fixtures:environment_factory",
        action_policy_factory=action_policy_factory,
    )
    return BenchmarkRunReport(
        benchmark_name="synthetic-action-policy",
        episodes=(),
        final_memory_count=0,
        seed=seed,
        configuration=configuration,
    )


def test_compare_benchmark_reports_allows_action_policy_change() -> None:
    """Action-policy identity is a treatment variable, not evaluation drift."""

    baseline = (
        _action_policy_report(1, "tests.fixtures:baseline_action_policy"),
        _action_policy_report(2, "tests.fixtures:baseline_action_policy"),
    )
    treatment = (
        _action_policy_report(1, "tests.fixtures:treatment_action_policy"),
        _action_policy_report(2, "tests.fixtures:treatment_action_policy"),
    )

    comparison = compare_benchmark_reports(baseline, treatment)

    assert comparison.seeds == (1, 2)
    assert comparison.success_rate_delta.mean == 0.0
    assert comparison.mean_reward_delta.mean == 0.0
