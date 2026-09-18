from __future__ import annotations

import pytest

from experiments.paired_statistics import compare_paired_benchmark_reports
from remem.benchmark import BenchmarkEpisodeReport, BenchmarkRunReport
from remem.environments.base import StepResult
from remem.execution import EpisodeResult, EpisodeStep


def _build_report(
    seed: int | None,
    reward: float,
    success: bool,
    *,
    benchmark_name: str = "synthetic-negative-transfer",
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
        benchmark_name=benchmark_name,
        episodes=tuple(
            BenchmarkEpisodeReport(
                episode_id=f"episode:{seed}:{episode_index}",
                episode=episode,
                episode_success=success,
                retained_memory_count=0,
            )
            for episode_index in range(episode_count)
        ),
        final_memory_count=0,
        seed=seed,
    )


def test_compare_paired_reports_uses_candidate_minus_baseline_deltas() -> None:
    baseline = (
        _build_report(1, 1.0, False),
        _build_report(2, 2.0, True),
        _build_report(3, 3.0, False),
    )
    candidate = (
        _build_report(3, 5.0, True),
        _build_report(1, 2.0, True),
        _build_report(2, 1.0, True),
    )

    comparison = compare_paired_benchmark_reports(baseline, candidate)

    assert comparison.benchmark_name == "synthetic-negative-transfer"
    assert comparison.seeds == (1, 2, 3)
    assert comparison.mean_reward_delta.sample_size == 3
    assert comparison.mean_reward_delta.mean == pytest.approx(2.0 / 3.0)
    assert comparison.success_rate_delta.mean == pytest.approx(2.0 / 3.0)
    assert comparison.transfer_success_rate_delta.mean == 0.0
    assert comparison.to_dict()["seeds"] == (1, 2, 3)


def test_compare_paired_reports_rejects_different_seed_sets() -> None:
    baseline = (_build_report(1, 1.0, True),)
    candidate = (_build_report(2, 1.0, True),)

    with pytest.raises(ValueError, match="same seed set"):
        compare_paired_benchmark_reports(baseline, candidate)


def test_compare_paired_reports_rejects_mixed_benchmarks() -> None:
    baseline = (_build_report(1, 1.0, True),)
    candidate = (_build_report(1, 1.0, True, benchmark_name="other"),)

    with pytest.raises(ValueError, match="same benchmark"):
        compare_paired_benchmark_reports(baseline, candidate)


def test_compare_paired_reports_rejects_mismatched_episode_counts() -> None:
    baseline = (_build_report(1, 1.0, True, episode_count=2),)
    candidate = (_build_report(1, 1.0, True),)

    with pytest.raises(ValueError, match="same number of episodes"):
        compare_paired_benchmark_reports(baseline, candidate)


@pytest.mark.parametrize("label", ["baseline", "candidate"])
def test_compare_paired_reports_rejects_empty_policy_collection(label: str) -> None:
    report = (_build_report(1, 1.0, True),)
    baseline = () if label == "baseline" else report
    candidate = () if label == "candidate" else report

    with pytest.raises(ValueError, match=f"{label} reports must contain at least one"):
        compare_paired_benchmark_reports(baseline, candidate)


def test_compare_paired_reports_rejects_unseeded_report() -> None:
    baseline = (_build_report(None, 1.0, True),)
    candidate = (_build_report(1, 1.0, True),)

    with pytest.raises(ValueError, match="baseline report seeds must be explicit"):
        compare_paired_benchmark_reports(baseline, candidate)


def test_compare_paired_reports_rejects_duplicate_seed() -> None:
    baseline = (_build_report(1, 1.0, True), _build_report(1, 2.0, False))
    candidate = (_build_report(1, 1.0, True),)

    with pytest.raises(ValueError, match="baseline report seeds must be unique"):
        compare_paired_benchmark_reports(baseline, candidate)


def test_compare_paired_reports_rejects_mixed_benchmark_within_policy() -> None:
    baseline = (
        _build_report(1, 1.0, True),
        _build_report(2, 1.0, True, benchmark_name="other"),
    )
    candidate = (_build_report(1, 1.0, True), _build_report(2, 1.0, True))

    with pytest.raises(ValueError, match="baseline reports must use one benchmark name"):
        compare_paired_benchmark_reports(baseline, candidate)
