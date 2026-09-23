from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.environment_evaluation_evidence import load_verified_environment_evidence
from experiments.environment_evaluation_report import build_environment_evaluation_report
from experiments.runtime_provenance import RuntimeProvenance
from remem.environment_evaluation import EnvironmentEvaluation, SeededEpisodeResult
from remem.environments.base import StepResult
from remem.execution import EpisodeResult, EpisodeStep


def _report() -> dict[str, object]:
    transition = StepResult("done", 1.0, True, info={"task": "pick_and_place"})
    result = EpisodeResult(
        initial_observation="room",
        steps=(EpisodeStep(0, "room", "take apple", transition),),
        total_reward=1.0,
        terminated=True,
    )
    evaluation = EnvironmentEvaluation((SeededEpisodeResult(seed=7, result=result),))
    provenance = RuntimeProvenance.create(
        code_revision="abc123",
        working_tree_state="clean",
        python_version="3.12.1",
        platform="linux-x86_64",
        package_version="0.1.0",
        dependency_versions={"remem-agent": "0.1.0"},
    )
    return build_environment_evaluation_report(
        evaluation,
        benchmark_name="ALFWorld",
        policy_name="baseline",
        policy_configuration={"temperature": 0.0},
        max_steps=10,
        provenance=provenance,
    )


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loader_reconstructs_verified_environment_evidence(tmp_path: Path) -> None:
    evidence = load_verified_environment_evidence(_write(tmp_path, _report()))

    assert evidence.benchmark_name == "ALFWorld"
    assert evidence.policy_name == "baseline"
    assert evidence.policy_configuration == {"temperature": 0.0}
    assert evidence.max_steps == 10
    assert evidence.evaluation.episode_count == 1
    assert evidence.evaluation.episodes[0].result.steps[0].action == "take apple"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report["aggregates"].__setitem__("mean_reward", 0.0), "mean_reward"),
        (lambda report: report["configuration"].__setitem__("seeds", [8]), "seeds"),
        (
            lambda report: report["episodes"][0]["result"].__setitem__("total_reward", 0.0),
            "total_reward",
        ),
        (
            lambda report: report.__setitem__("provenance_fingerprint", "0" * 64),
            "provenance_fingerprint",
        ),
    ],
)
def test_loader_rejects_mutated_derived_evidence(tmp_path: Path, mutation, message: str) -> None:
    report = copy.deepcopy(_report())
    mutation(report)

    with pytest.raises(ValueError, match=message):
        load_verified_environment_evidence(_write(tmp_path, report))


def test_loader_rejects_episode_longer_than_configured_limit(tmp_path: Path) -> None:
    report = _report()
    report["configuration"]["max_steps"] = 0

    with pytest.raises(ValueError, match="max_steps"):
        load_verified_environment_evidence(_write(tmp_path, report))


def test_loader_rejects_unknown_report_fields(tmp_path: Path) -> None:
    report = _report()
    report["unexpected"] = True

    with pytest.raises(ValueError, match="fields mismatch"):
        load_verified_environment_evidence(_write(tmp_path, report))


def test_loader_rejects_non_object_json(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="JSON object"):
        load_verified_environment_evidence(_write(tmp_path, []))
