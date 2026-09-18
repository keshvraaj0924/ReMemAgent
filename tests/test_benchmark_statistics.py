from __future__ import annotations

import pytest

from experiments.benchmark_statistics import (
    MetricSummary,
    _critical_value_95,
    _summarize,
    summarize_benchmark_reports,
)
from remem.benchmark import BenchmarkEpisodeReport, BenchmarkRunReport
from remem.environments.base import StepResult
from remem.execution import EpisodeResult, EpisodeStep


def _build_report(
    seed: int,
    reward: float,
    success: bool,
    *,
    episode_count: int = 1,
) -> BenchmarkRunReport:
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
        episodes=tuple(
            BenchmarkEpisodeReport(
                episode_id=f"alfworld-test:{seed}:{episode_index}",
                episode=episode,
                episode_success=success,
                retained_memory_count=1,
            )
            for episode_index in range(episode_count)
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
    assert statistics.success_rate.sample_size == 3
    assert statistics.mean_reward.sample_size == 3
    assert statistics.transfer_success_rate.sample_size == 3
    assert statistics.success_rate.mean == pytest.approx(2.0 / 3.0)
    assert statistics.mean_reward.mean == pytest.approx(3.0)
    assert statistics.mean_reward.sample_stddev == pytest.approx(2.0)
    assert statistics.mean_reward.standard_error == pytest.approx(2.0 / 3.0**0.5)
    expected_margin = 4.303 * statistics.mean_reward.standard_error
    assert statistics.mean_reward.confidence_interval_95 == pytest.approx(
        (3.0 - expected_margin, 3.0 + expected_margin)
    )
    assert statistics.transfer_success_rate.mean == pytest.approx(0.0)


def test_single_seed_summary_has_zero_uncertainty() -> None:
    statistics = summarize_benchmark_reports((_build_report(7, 4.0, True),))

    assert statistics.mean_reward.sample_size == 1
    assert statistics.mean_reward.sample_stddev == 0.0
    assert statistics.mean_reward.standard_error == 0.0
    assert statistics.mean_reward.confidence_interval_95 == (4.0, 4.0)


def test_critical_value_uses_asymptotic_normal_value_after_table() -> None:
    assert _critical_value_95(31) == pytest.approx(2.042)
    assert _critical_value_95(32) == pytest.approx(1.96)


def test_critical_value_requires_two_observations() -> None:
    with pytest.raises(ValueError, match="at least two"):
        _critical_value_95(1)


def test_summarize_benchmark_reports_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_benchmark_reports(())


def test_metric_summary_rejects_empty_observations() -> None:
    with pytest.raises(ValueError, match="observations must not be empty"):
        _summarize((), metric_name="mean_reward")


def test_summarize_benchmark_reports_rejects_unseeded_report() -> None:
    seeded_report = _build_report(1, 1.0, True)
    unseeded_report = BenchmarkRunReport(
        benchmark_name=seeded_report.benchmark_name,
        episodes=seeded_report.episodes,
        final_memory_count=seeded_report.final_memory_count,
        seed=None,
    )

    with pytest.raises(ValueError, match="seeds must be explicit"):
        summarize_benchmark_reports((unseeded_report,))


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


def test_summarize_benchmark_reports_rejects_different_episode_counts() -> None:
    complete_report = _build_report(1, 1.0, True, episode_count=2)
    partial_report = _build_report(2, 1.0, True, episode_count=1)

    with pytest.raises(ValueError, match="same number of episodes"):
        summarize_benchmark_reports((complete_report, partial_report))


@pytest.mark.parametrize("invalid_reward", [float("nan"), float("inf"), float("-inf")])
def test_metric_summary_rejects_non_finite_observations(invalid_reward: float) -> None:
    with pytest.raises(ValueError, match="mean_reward observations must be finite"):
        _summarize((1.0, invalid_reward), metric_name="mean_reward")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sample_size": 0}, "sample_size must be at least one"),
        ({"mean": float("nan")}, "metric summary values must be finite"),
        ({"sample_stddev": -0.1}, "sample_stddev must be non-negative"),
        ({"standard_error": -0.1}, "standard_error must be non-negative"),
        (
            {"confidence_interval_95": (float("-inf"), 1.0)},
            "confidence interval bounds must be finite",
        ),
        (
            {"confidence_interval_95": (2.0, 1.0)},
            "lower bound must not exceed upper bound",
        ),
        (
            {"confidence_interval_95": (0.0, 0.5)},
            "confidence interval must contain the metric mean",
        ),
    ],
)
def test_metric_summary_rejects_malformed_state(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "sample_size": 2,
        "mean": 1.0,
        "sample_stddev": 0.5,
        "standard_error": 0.25,
        "confidence_interval_95": (0.5, 1.5),
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        MetricSummary(**values)  # type: ignore[arg-type]
