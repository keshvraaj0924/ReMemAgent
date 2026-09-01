from __future__ import annotations

import pytest

from experiments.benchmark_statistics import summarize_benchmark_reports
from remem.benchmark import BenchmarkEpisodeReport, BenchmarkRunReport
from remem.environments.base import StepResult
from remem.execution import EpisodeResult, EpisodeStep


def _build_report(seed: int, reward: float, success: bool) -> BenchmarkRunReport:
    episode = EpisodeResult(
        initial_observation="start",
        steps=(
            EpisodeStep(
                step_index=0,
                observation="start",
                action="finish",
                result=StepResult(
                    observation="done",
                    reward=reward,
                    terminated=True,
                    truncated=False,
                ),
            ),
        ),
        total_reward=reward,
        terminated=True,
        truncated=False,
    )
    return BenchmarkRunReport(
        benchmark_name="alfworld-test",
        episodes=(
            BenchmarkEpisodeReport(
                episode_id=f"alfworld-test:{seed}",
                episode=episode,
                episode_success=success,
                retained_memory_count=1,
            ),
        ),
        final_memory_count=1,
        seed=seed,
    )


def test_summarize_benchmark_reports_uses_seed_level_observations() -> None:
    reports = (
        _build_report(1, 1.0, True),
        _build_report(2, 3.0, False),
        _build_report(3, 5.0, True),
    )

    statistics = summarize_benchmark_reports(reports)

    assert statistics.benchmark_name == "alfworld-test"
    assert statistics.seeds == (1, 2, 3)
    assert statistics.success_rate.mean == pytest.approx(2.0 / 3.0)
    assert statistics.mean_reward.mean == pytest.approx(3.0)
    assert statistics.mean_reward.sample_stddev == pytest.approx(2.0)
    assert statistics.mean_reward.standard_error == pytest.approx(2.0 / 3.0**0.5)
    assert statistics.transfer_success_rate.mean == pytest.approx(0.0)


def test_single_seed_summary_has_zero_uncertainty() -> None:
    statistics = summarize_benchmark_reports((_build_report(7, 4.0, True),))

    assert statistics.mean_reward.sample_stddev == 0.0
    assert statistics.mean_reward.standard_error == 0.0
    assert statistics.mean_reward.confidence_interval_95 == (4.0, 4.0)


def test_summarize_benchmark_reports_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_benchmark_reports(())


def test_summarize_benchmark_reports_rejects_duplicate_seeds() -> None:
    reports = (_build_report(1, 1.0, True), _build_report(1, 2.0, True))

    with pytest.raises(ValueError, match="seeds must be unique"):
        summarize_benchmark_reports(reports)


def test_summarize_benchmark_reports_rejects_mixed_benchmarks() -> None:
    first = _build_report(1, 1.0, True)
    second = _build_report(2, 2.0, True)
    second = BenchmarkRunReport(
        benchmark_name="webshop-test",
        episodes=second.episodes,
        final_memory_count=second.final_memory_count,
        seed=second.seed,
    )

    with pytest.raises(ValueError, match="one benchmark name"):
        summarize_benchmark_reports((first, second))
