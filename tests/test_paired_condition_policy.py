from __future__ import annotations

from dataclasses import replace

from experiments.benchmark_statistics import compare_benchmark_reports
from remem.benchmark import BenchmarkEpisodeReport, BenchmarkRunConfiguration, BenchmarkRunReport
from remem.execution import EpisodeResult


def _report(seed: int, reward: float, policy_factory: str) -> BenchmarkRunReport:
    """Build a minimal measured report for paired-condition validation."""

    episode = EpisodeResult(
        initial_observation="start",
        steps=(),
        total_reward=reward,
        terminated=True,
        truncated=False,
    )
    configuration = BenchmarkRunConfiguration(
        benchmark_name="paired-test",
        episode_count=1,
        max_steps=5,
        seed=seed,
        environment_factory="tests.fixtures:environment_factory",
        policy_factory=policy_factory,
        success_evaluator="tests.fixtures:success_evaluator",
    )
    return BenchmarkRunReport(
        benchmark_name="paired-test",
        episodes=(
            BenchmarkEpisodeReport(
                episode_id=f"paired-test:{seed}",
                episode=episode,
                episode_success=reward > 0.0,
                retained_memory_count=0,
            ),
        ),
        final_memory_count=0,
        seed=seed,
        configuration=configuration,
    )


def test_compare_allows_policy_identity_to_change_between_conditions() -> None:
    """Paired experiments may change policy while holding evaluation fixed."""

    baseline = (_report(1, 0.0, "policies:baseline"), _report(2, 0.0, "policies:baseline"))
    treatment = (_report(1, 1.0, "policies:memory"), _report(2, 0.0, "policies:memory"))

    comparison = compare_benchmark_reports(baseline, treatment)

    assert comparison.seeds == (1, 2)
    assert comparison.success_rate_delta.mean == 0.5


def test_compare_still_rejects_evaluation_configuration_drift() -> None:
    """Changing max steps remains invalid for a paired experimental comparison."""

    baseline_report = _report(1, 0.0, "policies:baseline")
    treatment_report = _report(1, 1.0, "policies:memory")
    treatment_configuration = replace(treatment_report.configuration, max_steps=10)
    treatment_report = replace(treatment_report, configuration=treatment_configuration)

    try:
        compare_benchmark_reports((baseline_report,), (treatment_report,))
    except ValueError as exc:
        assert "configuration apart from the seed and policy" in str(exc)
    else:
        raise AssertionError("evaluation configuration drift must be rejected")
