from __future__ import annotations

from experiments.benchmark_statistics import summarize_benchmark_reports
from remem.benchmark import BenchmarkEpisodeReport, BenchmarkRunReport
from remem.environments.base import StepResult
from remem.execution import EpisodeResult


def _report(seed: int, reward: float, success: bool) -> BenchmarkRunReport:
    episode = EpisodeResult(
        initial_observation="start",
        steps=(),
        total_reward=reward,
        terminated=True,
        truncated=False,
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
