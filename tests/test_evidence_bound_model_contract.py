"""Regression coverage for model/checkpoint binding in evidence-bound measurement."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import experiments.evidence_bound_paired_cli as evidence_cli
from experiments.research_experiment_plan import build_research_experiment_plan

REMEM_REVISION = "1" * 40
SOURCE_REVISION = "2" * 40


def _plan(*, model_identity: str | None = None, parameters=None):
    return build_research_experiment_plan(
        experiment_name="webshop-memory-study",
        remem_revision=REMEM_REVISION,
        benchmark_name="webshop",
        episode_count=10,
        max_steps=30,
        seeds=(11, 17),
        environment_factory="study.environments:build_webshop",
        success_evaluator="study.metrics:is_success",
        source_revisions={"webshop": SOURCE_REVISION},
        dependency_versions={"torch": "2.6.0"},
        baseline_policy_factory="study.policies:build_baseline",
        treatment_action_policy_factory="study.policies:build_remem_action",
        model_identity=model_identity,
        parameters=parameters,
    )


def test_extract_experiment_parameters_parses_json_scalars_and_removes_options() -> None:
    parameters, delegated = evidence_cli._extract_experiment_parameters(
        [
            "remem-paired-benchmark",
            "--benchmark",
            "webshop",
            "--experiment-parameter",
            "temperature=0.2",
            "--experiment-parameter=max_tokens=128",
            "--experiment-parameter",
            "do_sample=false",
            "--experiment-parameter",
            'stop_token="</s>"',
        ]
    )

    assert parameters == {
        "temperature": 0.2,
        "max_tokens": 128,
        "do_sample": False,
        "stop_token": "</s>",
    }
    assert delegated == ["remem-paired-benchmark", "--benchmark", "webshop"]


def test_extract_experiment_parameters_rejects_duplicates_and_structured_values() -> None:
    with pytest.raises(SystemExit, match="duplicate"):
        evidence_cli._extract_experiment_parameters(
            [
                "remem-paired-benchmark",
                "--experiment-parameter=temperature=0.1",
                "--experiment-parameter=temperature=0.2",
            ]
        )

    with pytest.raises(SystemExit, match="JSON scalar"):
        evidence_cli._extract_experiment_parameters(
            ["remem-paired-benchmark", '--experiment-parameter=sampling={"top_p":0.9}']
        )


def test_validate_experiment_model_contract_requires_exact_frozen_values() -> None:
    plan = _plan(
        model_identity="Qwen/Qwen2.5-7B-Instruct@0123456789abcdef",
        parameters={"temperature": 0.0, "max_tokens": 128, "do_sample": False},
    )

    evidence_cli._validate_experiment_model_contract(
        plan,
        "Qwen/Qwen2.5-7B-Instruct@0123456789abcdef",
        {"temperature": 0.0, "max_tokens": 128, "do_sample": False},
    )

    with pytest.raises(ValueError, match="model_identity"):
        evidence_cli._validate_experiment_model_contract(
            plan,
            "Qwen/Qwen2.5-7B-Instruct@different",
            {"temperature": 0.0, "max_tokens": 128, "do_sample": False},
        )

    with pytest.raises(ValueError, match="parameters"):
        evidence_cli._validate_experiment_model_contract(
            plan,
            "Qwen/Qwen2.5-7B-Instruct@0123456789abcdef",
            {"temperature": 0.2, "max_tokens": 128, "do_sample": False},
        )


def test_main_rejects_model_configuration_without_frozen_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evidence_cli.paired_cli,
        "main",
        lambda: pytest.fail("unbound model configuration must not delegate"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-paired-benchmark",
            "--model-identity",
            "Qwen/Qwen2.5-7B-Instruct@revision",
            "--experiment-parameter=temperature=0.0",
        ],
    )

    with pytest.raises(SystemExit, match="require --require-experiment-plan"):
        evidence_cli.main()


def test_plan_bound_saver_persists_model_identity_and_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness_payload: dict[str, object] = {"evidence_sha256": "a" * 64}
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(json.dumps(readiness_payload), encoding="utf-8")
    plan = _plan(model_identity="model@revision", parameters={"temperature": 0.0})
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        evidence_cli,
        "verify_controlled_paired_preflight_evidence",
        lambda payload: None,
    )
    monkeypatch.setattr(evidence_cli, "_load_experiment_plan", lambda path: plan)
    monkeypatch.setattr(evidence_cli, "_parse_delegated_arguments", lambda argv: object())
    monkeypatch.setattr(evidence_cli, "_validate_experiment_plan_contract", lambda plan, args: None)

    def delegated_main() -> int:
        evidence_cli.paired_cli.save_paired_execution_result(
            "result",
            tmp_path / "paired.json",
            runtime_provenance={},
        )
        return 0

    def fake_saver(*args, **kwargs):
        observed.update(kwargs["runtime_provenance"])
        return tmp_path / "paired.json"

    monkeypatch.setattr(evidence_cli.paired_cli, "main", delegated_main)
    monkeypatch.setattr(evidence_cli.paired_cli, "save_paired_execution_result", fake_saver)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-paired-benchmark",
            "--require-preflight-evidence",
            str(readiness_path),
            "--require-experiment-plan",
            str(tmp_path / "plan.json"),
            "--model-identity",
            "model@revision",
            "--experiment-parameter=temperature=0.0",
            "--source-checkout=webshop=/tmp/webshop",
            f"--require-source-revision=webshop={SOURCE_REVISION}",
        ],
    )

    assert evidence_cli.main() == 0
    assert observed[evidence_cli.MODEL_IDENTITY_PROVENANCE_KEY] == "model@revision"
    assert observed[evidence_cli.EXPERIMENT_PARAMETERS_PROVENANCE_KEY] == {"temperature": 0.0}
    assert observed[evidence_cli.EXPERIMENT_PLAN_PROVENANCE_KEY] == plan.sha256
