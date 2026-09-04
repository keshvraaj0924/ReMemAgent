from __future__ import annotations

from experiments.benchmark_statistics import (
    compare_benchmark_reports,
    summarize_benchmark_reports,
)
from remem.benchmark import BenchmarkEpisodeReport, BenchmarkRunConfiguration, BenchmarkRunReport
from remem.execution import EpisodeResult


def _report(
    seed: int,
    reward: float,
    success: bool,
    *,
    max_steps: int | None = None,
    policy_factory: str = "tests.fixtures:policy_factory",
) -> BenchmarkRunReport:
    episode = EpisodeResult(
        initial_observation="start",
        steps=(),
        total_reward=reward,
        terminated=True,
        truncated=False,
    )
    configuration = None
    if max_steps is not None:
        configuration = BenchmarkRunConfiguration(
            benchmark_name="synthetic-test",
            episode_count=1,
            max_steps=max_steps,
            seed=seed,
            environment_factory="tests.fixtures:environment_factory",
            policy_factory=policy_factory,
            success_evaluator="tests.fixtures:success_evaluator",
        )
    return BenchmarkRunReport(
        benchmark_name="synthetic-test",
        episodes=(
            BenchmarkEpisodeReport(
                episode_id=f"synthetic-test:{seed}",
                episode=episode,
                episode_success=success,
                retained_memory_count=0,
            ),
        ),
        final_memory_count=0,
        seed=seed,
        configuration=configuration,
    )


def _unseeded_report(reward: float, success: bool) -> BenchmarkRunReport:
    report = _report(0, reward, success)
    return BenchmarkRunReport(
        benchmark_name=report.benchmark_name,
        episodes=report.episodes,
        final_memory_count=report.final_memory_count,
        seed=None,
    )


def test_summarize_benchmark_reports_uses_seed_level_observations() -> None:
    summary = summarize_benchmark_reports(
        (_report(1, 1.0, True), _report(2, 0.0, False), _report(3, 1.0, True))
    )

    assert summary.benchmark_name == "synthetic-test"
    assert summary.seeds == (1, 2, 3)
    assert summary.success_rate.mean == 2 / 3
    assert summary.success_rate.sample_stddev > 0.0
    assert summary.mean_reward.mean == 2 / 3


def test_single_seed_summary_has_zero_uncertainty() -> None:
    summary = summarize_benchmark_reports((_report(7, 0.5, True),))

    assert summary.success_rate.sample_stddev == 0.0
    assert summary.success_rate.standard_error == 0.0
    assert summary.success_rate.confidence_interval_95 == (1.0, 1.0)


def test_statistics_reject_duplicate_seeds() -> None:
    reports = (_report(1, 1.0, True), _report(1, 0.0, False))

    try:
        summarize_benchmark_reports(reports)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate seeds must be rejected")


def test_statistics_reject_mixed_benchmarks() -> None:
    first = _report(1, 1.0, True)
    second = _report(2, 0.0, False)
    second = BenchmarkRunReport(
        benchmark_name="other-benchmark",
        episodes=second.episodes,
        final_memory_count=second.final_memory_count,
        seed=second.seed,
    )

    try:
        summarize_benchmark_reports((first, second))
    except ValueError as exc:
        assert "one benchmark name" in str(exc)
    else:
        raise AssertionError("mixed benchmarks must be rejected")


def test_statistics_validate_report_structure_before_aggregation() -> None:
    invalid_report = _report(1, float("nan"), True)

    try:
        summarize_benchmark_reports((invalid_report,))
    except ValueError as exc:
        assert "total_reward must be finite" in str(exc)
    else:
        raise AssertionError("invalid benchmark reports must be rejected")


def test_compare_benchmark_reports_pairs_metrics_by_seed() -> None:
    baseline = (_report(1, 0.0, False), _report(2, 0.5, True))
    treatment = (_report(1, 1.0, True), _report(2, 0.5, True))

    comparison = compare_benchmark_reports(
        baseline,
        treatment,
        baseline_label="no-memory",
        treatment_label="memory",
    )

    assert comparison.baseline_label == "no-memory"
    assert comparison.treatment_label == "memory"
    assert comparison.seeds == (1, 2)
    assert comparison.success_rate_delta.mean == 0.5
    assert comparison.mean_reward_delta.mean == 0.5
    assert comparison.transfer_success_rate_delta.mean == 0.0


def test_compare_benchmark_reports_canonicalizes_seed_order() -> None:
    baseline = (_report(2, 0.5, True), _report(1, 0.0, False))
    treatment = (_report(1, 1.0, True), _report(2, 0.5, True))

    comparison = compare_benchmark_reports(baseline, treatment)

    assert comparison.seeds == (1, 2)
    assert comparison.success_rate_delta.mean == 0.5


def test_compare_benchmark_reports_rejects_unseeded_runs() -> None:
    baseline = (_unseeded_report(0.0, False),)
    treatment = (_unseeded_report(1.0, True),)

    try:
        compare_benchmark_reports(baseline, treatment)
    except ValueError as exc:
        assert "explicit seeds" in str(exc)
    else:
        raise AssertionError("paired comparisons must require explicit seeds")


def test_compare_benchmark_reports_rejects_mismatched_seed_sets() -> None:
    baseline = (_report(1, 0.0, False), _report(2, 0.5, True))
    treatment = (_report(1, 1.0, True), _report(3, 0.5, True))

    try:
        compare_benchmark_reports(baseline, treatment)
    except ValueError as exc:
        assert "same seed set" in str(exc)
    else:
        raise AssertionError("mismatched seed sets must be rejected")


def test_compare_benchmark_reports_rejects_configuration_drift() -> None:
    baseline = (_report(1, 0.0, False, max_steps=5),)
    treatment = (_report(1, 1.0, True, max_steps=10),)

    try:
        compare_benchmark_reports(baseline, treatment)
    except ValueError as exc:
        assert "configuration apart from the seed and policy" in str(exc)
    else:
        raise AssertionError("paired conditions with different configurations must be rejected")


def test_compare_benchmark_reports_rejects_missing_configuration() -> None:
    baseline = (_report(1, 0.0, False),)
    treatment = (_report(1, 1.0, True),)

    try:
        compare_benchmark_reports(baseline, treatment)
    except ValueError as exc:
        assert "explicit configuration" in str(exc)
    else:
        raise AssertionError("paired comparisons must require explicit protocol configuration")


def test_compare_benchmark_reports_allows_policy_change() -> None:
    baseline = (
        _report(1, 0.0, False, max_steps=5, policy_factory="tests.fixtures:baseline_policy"),
        _report(2, 0.5, True, max_steps=5, policy_factory="tests.fixtures:baseline_policy"),
    )
    treatment = (
        _report(1, 1.0, True, max_steps=5, policy_factory="tests.fixtures:treatment_policy"),
        _report(2, 0.5, True, max_steps=5, policy_factory="tests.fixtures:treatment_policy"),
    )

    comparison = compare_benchmark_reports(baseline, treatment)

    assert comparison.seeds == (1, 2)
    assert comparison.success_rate_delta.mean == 0.5


def test_compare_benchmark_reports_accepts_same_configuration_across_conditions() -> None:
    baseline = (_report(1, 0.0, False, max_steps=5), _report(2, 0.5, True, max_steps=5))
    treatment = (_report(1, 1.0, True, max_steps=5), _report(2, 0.5, True, max_steps=5))

    comparison = compare_benchmark_reports(baseline, treatment)

    assert comparison.seeds == (1, 2)
    assert comparison.success_rate_delta.mean == 0.5


def test_compare_benchmark_reports_rejects_empty_labels() -> None:
    reports = (_report(1, 0.0, False),)

    try:
        compare_benchmark_reports(reports, reports, baseline_label=" ")
    except ValueError as exc:
        assert "baseline_label" in str(exc)
    else:
        raise AssertionError("empty condition labels must be rejected")


def test_compare_benchmark_reports_rejects_non_string_labels() -> None:
    reports = (_report(1, 0.0, False),)

    try:
        compare_benchmark_reports(reports, reports, baseline_label=123)  # type: ignore[arg-type]
    except TypeError as exc:
        assert "baseline_label" in str(exc)
    else:
        raise AssertionError("non-string condition labels must be rejected")
