from __future__ import annotations

import json

import pytest

from experiments.benchmark_report import save_benchmark_report
from remem.benchmark import BenchmarkEpisodeReport, BenchmarkRunReport
from remem.environments.base import StepResult
from remem.execution import EpisodeResult, EpisodeStep


def _build_report(seed: int = 1) -> BenchmarkRunReport:
    result = StepResult(
        observation="done",
        reward=1.0,
        terminated=True,
        truncated=False,
    )
    episode = EpisodeResult(
        initial_observation="start",
        steps=(EpisodeStep(0, "start", "finish", result),),
        total_reward=1.0,
        terminated=True,
    )
    return BenchmarkRunReport(
        benchmark_name="alfworld-test",
        episodes=(BenchmarkEpisodeReport("episode-0", episode, True, 1),),
        final_memory_count=1,
        seed=seed,
    )


def test_save_benchmark_report_persists_measured_metrics(tmp_path) -> None:
    output_path = save_benchmark_report(_build_report(7), tmp_path / "report.json")

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["benchmark_name"] == "alfworld-test"
    assert persisted["seed"] == 7
    assert persisted["success_rate"] == pytest.approx(1.0)
    assert persisted["mean_reward"] == pytest.approx(1.0)
    assert persisted["transfer_success_rate"] == pytest.approx(0.0)


def test_benchmark_report_rejects_duplicate_episode_ids() -> None:
    report = _build_report()
    episode = report.episodes[0]

    with pytest.raises(ValueError, match="episode ids must be unique"):
        BenchmarkRunReport(
            benchmark_name=report.benchmark_name,
            episodes=(episode, episode),
            final_memory_count=1,
            seed=1,
        )
