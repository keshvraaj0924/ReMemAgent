from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.environment_evaluation_report import (
    ENVIRONMENT_EVALUATION_REPORT_SCHEMA_VERSION,
    build_environment_evaluation_report,
    save_environment_evaluation_report,
)
from experiments.runtime_provenance import RuntimeProvenance
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


def _provenance() -> RuntimeProvenance:
    return RuntimeProvenance.create(
        code_revision="abc123",
        working_tree_state="clean",
        python_version="3.12.1",
        platform="linux-x86_64",
        package_version="0.1.0",
        dependency_versions={"gymnasium": "1.0.0", "remem-agent": "0.1.0"},
    )


def test_report_preserves_policy_configuration_provenance_and_raw_transitions() -> None:
    provenance = _provenance()
    report = build_environment_evaluation_report(
        _evaluation(),
        benchmark_name="ALFWorld",
        policy_name="counterfactual-memory-router",
        policy_configuration={"minimum_delta": 0.15, "retrieval_limit": 5},
        max_steps=50,
        provenance=provenance,
    )

    assert report["schema_version"] == ENVIRONMENT_EVALUATION_REPORT_SCHEMA_VERSION
    assert report["benchmark_name"] == "ALFWorld"
    assert report["policy"] == {
        "name": "counterfactual-memory-router",
        "configuration": {"minimum_delta": 0.15, "retrieval_limit": 5},
    }
    assert report["configuration"] == {"max_steps": 50, "seeds": [7]}
    assert report["provenance"] == provenance.to_dict()
    assert report["provenance_fingerprint"] == provenance.fingerprint()
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
    assert episode["result"]["steps"][0]["result"]["info"] == {"task": "pick_and_place"}


def test_report_persistence_is_deterministic_and_atomic(tmp_path: Path) -> None:
    destination = tmp_path / "evaluation.json"
    evaluation = _evaluation()
    provenance = _provenance()
    policy_configuration = {"temperature": 0.0, "memory_enabled": True}

    save_environment_evaluation_report(
        destination,
        evaluation,
        benchmark_name="WebShop",
        policy_name="deterministic-baseline",
        policy_configuration=policy_configuration,
        max_steps=20,
        provenance=provenance,
    )
    first_bytes = destination.read_bytes()
    save_environment_evaluation_report(
        destination,
        evaluation,
        benchmark_name="WebShop",
        policy_name="deterministic-baseline",
        policy_configuration=policy_configuration,
        max_steps=20,
        provenance=provenance,
    )

    assert destination.read_bytes() == first_bytes
    persisted = json.loads(first_bytes)
    assert persisted["benchmark_name"] == "WebShop"
    assert persisted["policy"]["name"] == "deterministic-baseline"
    assert persisted["provenance_fingerprint"] == provenance.fingerprint()
    assert not tuple(tmp_path.glob(".evaluation.json.*.tmp"))


def test_report_rejects_invalid_metadata_policy_and_provenance() -> None:
    evaluation = _evaluation()
    provenance = _provenance()
    valid_arguments = {
        "benchmark_name": "ALFWorld",
        "policy_name": "baseline",
        "policy_configuration": {},
        "max_steps": 1,
        "provenance": provenance,
    }

    with pytest.raises(ValueError, match="benchmark_name"):
        build_environment_evaluation_report(
            evaluation, **{**valid_arguments, "benchmark_name": " "}
        )
    with pytest.raises(ValueError, match="policy_name"):
        build_environment_evaluation_report(evaluation, **{**valid_arguments, "policy_name": " "})
    with pytest.raises(TypeError, match="policy_configuration"):
        build_environment_evaluation_report(
            evaluation, **{**valid_arguments, "policy_configuration": None}
        )
    with pytest.raises(ValueError, match="max_steps"):
        build_environment_evaluation_report(evaluation, **{**valid_arguments, "max_steps": 0})
    with pytest.raises(TypeError, match="provenance"):
        build_environment_evaluation_report(evaluation, **{**valid_arguments, "provenance": None})


def test_persistence_fails_closed_for_non_json_policy_configuration(tmp_path: Path) -> None:
    destination = tmp_path / "invalid-policy.json"

    with pytest.raises(TypeError):
        save_environment_evaluation_report(
            destination,
            _evaluation(),
            benchmark_name="ALFWorld",
            policy_name="baseline",
            policy_configuration={"opaque": object()},
            max_steps=1,
            provenance=_provenance(),
        )

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".invalid-policy.json.*.tmp"))


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
    evaluation = EnvironmentEvaluation(episodes=(SeededEpisodeResult(seed=1, result=result),))
    destination = tmp_path / "invalid.json"

    with pytest.raises(TypeError):
        save_environment_evaluation_report(
            destination,
            evaluation,
            benchmark_name="ALFWorld",
            policy_name="baseline",
            policy_configuration={},
            max_steps=1,
            provenance=_provenance(),
        )

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".invalid.json.*.tmp"))
