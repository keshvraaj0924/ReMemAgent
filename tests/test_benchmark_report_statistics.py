from __future__ import annotations

import json

from experiments.benchmark_report import save_repeated_benchmark_reports
from experiments.benchmark_statistics import summarize_benchmark_reports
from remem.benchmark import BenchmarkEpisodeReport, BenchmarkRunConfiguration, BenchmarkRunReport
from remem.execution import EpisodeResult


def _report(seed: int, reward: float, success: bool) -> BenchmarkRunReport:
    return BenchmarkRunReport(
        benchmark_name="synthetic-eval",
        episodes=(
            BenchmarkEpisodeReport(
                episode_id=f"episode-{seed}",
                episode=EpisodeResult(
                    initial_observation="start",
                    steps=(),
                    total_reward=reward,
                    terminated=True,
                    truncated=False,
                ),
                episode_success=success,
                retained_memory_count=0,
            ),
        ),
        final_memory_count=0,
        seed=seed,
        configuration=BenchmarkRunConfiguration(
            benchmark_name="synthetic-eval",
            episode_count=1,
            max_steps=1,
            seed=seed,
            environment_factory="tests.test_benchmark_report_statistics:make_environment",
            policy_factory="tests.test_benchmark_report_statistics:make_policy",
            success_evaluator="tests.test_benchmark_report_statistics:evaluate_success",
            minimum_trust=0.0,
        ),
    )


def make_environment(seed: int) -> object:
    """Provide an importable environment factory for configuration metadata tests."""

    del seed
    return object()


def make_policy(seed: int, store: object) -> object:
    """Provide an importable policy factory for configuration metadata tests."""

    del seed, store
    return lambda state: "look"


def evaluate_success(episode: EpisodeResult) -> bool:
    """Provide an importable evaluator for configuration metadata tests."""

    return episode.total_reward > 0


def test_repeated_report_serializer_persists_statistics(tmp_path) -> None:
    reports = (_report(1, 1.0, True), _report(2, 0.0, False))
    statistics = summarize_benchmark_reports(reports).to_dict()

    output_path = save_repeated_benchmark_reports(
        reports,
        tmp_path / "nested" / "repeated.json",
        statistics=statistics,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["seeds"] == [1, 2]
    assert payload["statistics"]["success_rate"]["mean"] == 0.5
    assert payload["statistics"]["mean_reward"]["mean"] == 0.5


def test_statistics_reject_empty_reports() -> None:
    try:
        summarize_benchmark_reports(())
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("empty reports must be rejected")
