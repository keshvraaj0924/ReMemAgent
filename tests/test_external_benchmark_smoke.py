from __future__ import annotations

import json
from pathlib import Path

from experiments.benchmark_report import save_benchmark_report
from experiments.external_benchmark import ExternalBenchmarkSpec, run_external_benchmark
from experiments.smoke_benchmark import SMOKE_ACTION, SMOKE_OBSERVATION


SMOKE_ENVIRONMENT_FACTORY = "experiments.smoke_benchmark:build_environment"
SMOKE_ACTION_POLICY_FACTORY = "experiments.smoke_benchmark:build_action_policy"
SMOKE_SUCCESS_EVALUATOR = "experiments.smoke_benchmark:is_success"


def _build_smoke_spec() -> ExternalBenchmarkSpec:
    """Build the canonical deterministic smoke experiment specification."""

    return ExternalBenchmarkSpec(
        benchmark_name="alfworld-smoke",
        episode_count=2,
        max_steps=2,
        environment_factory=SMOKE_ENVIRONMENT_FACTORY,
        policy_factory=None,
        action_policy_factory=SMOKE_ACTION_POLICY_FACTORY,
        success_evaluator=SMOKE_SUCCESS_EVALUATOR,
        seed=0,
    )


def test_external_benchmark_executes_memory_guided_smoke_path() -> None:
    """Exercise adaptation, ingestion, retrieval, and guided policy execution."""

    report = run_external_benchmark(_build_smoke_spec())

    assert len(report.episodes) == 2
    assert report.success_count == 2
    assert report.success_rate == 1.0
    assert report.mean_reward == 1.0
    assert report.final_memory_count == 1

    second_episode = report.episodes[1].episode
    assert second_episode.initial_observation == SMOKE_OBSERVATION
    assert second_episode.steps[0].action == SMOKE_ACTION


def test_external_benchmark_smoke_report_preserves_provenance(tmp_path: Path) -> None:
    """Verify real smoke factory specifications survive into JSON output."""

    report = run_external_benchmark(_build_smoke_spec())
    output_path = save_benchmark_report(report, tmp_path / "smoke.json")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    configuration = payload["configuration"]
    assert configuration["benchmark_name"] == "alfworld-smoke"
    assert configuration["seed"] == 0
    assert configuration["environment_factory"] == SMOKE_ENVIRONMENT_FACTORY
    assert configuration["policy_factory"] == SMOKE_ACTION_POLICY_FACTORY
    assert configuration["success_evaluator"] == SMOKE_SUCCESS_EVALUATOR
