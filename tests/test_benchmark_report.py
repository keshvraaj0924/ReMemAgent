from __future__ import annotations

import json

from experiments.benchmark_report import benchmark_report_to_dict, save_benchmark_report
from remem.benchmark import BenchmarkEpisodeReport, BenchmarkRunReport
from remem.environments.base import StepResult
from remem.execution import EpisodeResult, EpisodeStep


def _build_report() -> BenchmarkRunReport:
    episode = EpisodeResult(
        initial_observation="start",
        steps=(
            EpisodeStep(
                step_index=0,
                observation="start",
                action="look",
                result=StepResult(
                    observation="next",
                    reward=1.0,
                    terminated=True,
                    truncated=False,
                    info={"opaque": object()},
                ),
            ),
        ),
        total_reward=1.0,
        terminated=True,
        truncated=False,
    )
    return BenchmarkRunReport(
        benchmark_name="alfworld-test",
        episodes=(
            BenchmarkEpisodeReport(
                episode_id="alfworld-test:0",
                episode=episode,
                episode_success=True,
                retained_memory_count=1,
            ),
        ),
        final_memory_count=1,
    )


def test_benchmark_report_to_dict_preserves_measured_core_fields() -> None:
    payload = benchmark_report_to_dict(_build_report())

    assert payload["benchmark_name"] == "alfworld-test"
    assert payload["episodes"][0]["episode"]["total_reward"] == 1.0
    assert payload["episodes"][0]["episode"]["steps"][0]["action"] == "look"
    assert "info" not in payload["episodes"][0]["episode"]["steps"][0]["result"]


def test_save_benchmark_report_writes_json(tmp_path) -> None:
    output_path = save_benchmark_report(_build_report(), tmp_path / "nested" / "report.json")

    assert output_path.exists()
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["final_memory_count"] == 1
    assert persisted["episodes"][0]["transfer_outcomes"] == []
