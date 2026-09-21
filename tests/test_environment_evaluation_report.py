from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.environment_evaluation_report import (
    build_environment_evaluation_report,
    save_environment_evaluation_report,
)
from remem.environment_evaluation import EnvironmentEvaluation, SeededEpisodeResult
from remem.environments.base import StepResult
from remem.execution import EpisodeResult, EpisodeStep


def _evaluation() -> EnvironmentEvaluation:
    transition = StepResult(
        observation="goal reached",
        reward=1.0,
        terminated=True,
        info={"task": "pick_and_place"},
    )
    result = EpisodeResult(
        initial_observation="room",
        steps=(
            EpisodeStep(
                step_index=0,
                observation="room",
                action="take apple",
                result=transition,
            ),
        ),
        total_reward=1.0,
        terminated=True,
    )
    return EnvironmentEvaluation(episodes=(SeededEpisodeResult(seed=7, result=result),))


def test_report_preserves_configuration_aggregates_and_raw_transitions() -> None:
    report = build_environment_evaluation_report(
        _evaluation(), benchmark_name="ALFWorld", max_steps=50
    )

    assert report["benchmark_name"] == "ALFWorld"
    assert report["configuration"] == {"max_steps": 50, "seeds": [7]}
    assert report["aggregates"] == {
        "episode_count": 1,
        "mean_reward": 1.0,
        "mean_steps": 1.0,
        "termination_rate": 1.0,
        "truncation_rate": 0.0,
    }
    episode = report["episodes"][0]
    assert episode["seed"] == 7
    assert episode["result"]["steps"][0]["action"] == "take apple"
    assert episode["result"]["steps"][0]["result"]["info"] == {
        "task": "pick_and_place"
    }


def test_report_persistence_is_deterministic_and_atomic(tmp_path: Path) -> None:
    destination = tmp_path / "evaluation.json"
    evaluation = _evaluation()

    save_environment_evaluation_report(
        destination, evaluation, benchmark_name="WebShop", max_steps=20
    )
    first_bytes = destination.read_bytes()
    save_environment_evaluation_report(
        destination, evaluation, benchmark_name="WebShop", max_steps=20
    )

    assert destination.read_bytes() == first_bytes
    assert json.loads(first_bytes)["benchmark_name"] == "WebShop"
    assert not tuple(tmp_path.glob(".evaluation.json.*.tmp"))


def test_report_rejects_invalid_metadata() -> None:
    evaluation = _evaluation()

    with pytest.raises(ValueError, match="benchmark_name"):
        build_environment_evaluation_report(evaluation, benchmark_name=" ", max_steps=1)
    with pytest.raises(ValueError, match="max_steps"):
        build_environment_evaluation_report(evaluation, benchmark_name="ALFWorld", max_steps=0)


def test_persistence_fails_closed_for_non_json_environment_metadata(tmp_path: Path) -> None:
    transition = StepResult(
        observation="next",
        reward=0.0,
        terminated=True,
        info={"opaque": object()},
    )
    result = EpisodeResult(
        initial_observation="start",
        steps=(EpisodeStep(0, "start", "act", transition),),
        total_reward=0.0,
        terminated=True,
    )
    evaluation = EnvironmentEvaluation(
        episodes=(SeededEpisodeResult(seed=1, result=result),)
    )
    destination = tmp_path / "invalid.json"

    with pytest.raises(TypeError):
        save_environment_evaluation_report(
            destination, evaluation, benchmark_name="ALFWorld", max_steps=1
        )

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".invalid.json.*.tmp"))
