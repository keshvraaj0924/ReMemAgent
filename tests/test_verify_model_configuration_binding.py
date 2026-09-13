"""Regression tests for retained model-configuration verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.evidence_bound_paired_cli import (
    EXPERIMENT_PARAMETERS_PROVENANCE_KEY,
    EXPERIMENT_PLAN_PROVENANCE_KEY,
    MODEL_IDENTITY_PROVENANCE_KEY,
)
from experiments.research_experiment_plan import (
    build_research_experiment_plan,
    write_research_experiment_plan,
)
from experiments.verify_model_configuration_binding import (
    main,
    verify_model_configuration_binding,
)

REVISION = "a" * 40
MODEL_IDENTITY = "Qwen/Qwen2.5-7B-Instruct@pinned"


def _write_plan(path: Path) -> str:
    plan = build_research_experiment_plan(
        experiment_name="webshop-controlled",
        remem_revision=REVISION,
        benchmark_name="webshop",
        episode_count=2,
        max_steps=20,
        seeds=(11, 29),
        environment_factory="tests.factories:environment",
        success_evaluator="tests.factories:success",
        source_revisions={"webshop": "b" * 40},
        dependency_versions={"python": "3.12.0"},
        parameters={"temperature": 0.0, "max_tokens": 128, "do_sample": False},
        baseline_action_policy_factory="tests.factories:baseline",
        treatment_action_policy_factory="tests.factories:treatment",
        model_identity=MODEL_IDENTITY,
    )
    write_research_experiment_plan(path, plan)
    return plan.sha256


def _write_report(
    path: Path,
    plan_sha256: str,
    *,
    model_identity: object = MODEL_IDENTITY,
    parameters: object | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runtime_provenance": {
                    "code_revision": REVISION,
                    EXPERIMENT_PLAN_PROVENANCE_KEY: plan_sha256,
                    MODEL_IDENTITY_PROVENANCE_KEY: model_identity,
                    EXPERIMENT_PARAMETERS_PROVENANCE_KEY: (
                        parameters
                        if parameters is not None
                        else {"temperature": 0.0, "max_tokens": 128, "do_sample": False}
                    ),
                },
            }
        ),
        encoding="utf-8",
    )


def test_verify_model_configuration_binding_accepts_exact_runtime_declaration(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    report_path = tmp_path / "report.json"
    plan_sha256 = _write_plan(plan_path)
    _write_report(report_path, plan_sha256)

    result = verify_model_configuration_binding(report_path, plan_path)

    assert result.experiment_plan_sha256 == plan_sha256
    assert result.model_identity == MODEL_IDENTITY
    assert dict(result.parameters) == {
        "do_sample": False,
        "max_tokens": 128,
        "temperature": 0.0,
    }
    assert result.report_byte_count == len(report_path.read_bytes())


def test_verify_model_configuration_binding_rejects_model_identity_drift(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    report_path = tmp_path / "report.json"
    plan_sha256 = _write_plan(plan_path)
    _write_report(report_path, plan_sha256, model_identity="other-model")

    with pytest.raises(ValueError, match="model identity does not match"):
        verify_model_configuration_binding(report_path, plan_path)


def test_verify_model_configuration_binding_rejects_parameter_drift(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    report_path = tmp_path / "report.json"
    plan_sha256 = _write_plan(plan_path)
    _write_report(
        report_path,
        plan_sha256,
        parameters={"temperature": 0.2, "max_tokens": 128, "do_sample": False},
    )

    with pytest.raises(ValueError, match="experiment parameters do not match"):
        verify_model_configuration_binding(report_path, plan_path)


def test_verify_model_configuration_binding_distinguishes_integer_and_float_parameters(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    report_path = tmp_path / "report.json"
    plan_sha256 = _write_plan(plan_path)
    _write_report(
        report_path,
        plan_sha256,
        parameters={"temperature": 0.0, "max_tokens": 128.0, "do_sample": False},
    )

    with pytest.raises(ValueError, match="experiment parameters do not match"):
        verify_model_configuration_binding(report_path, plan_path)


def test_verify_model_configuration_binding_rejects_missing_runtime_model_identity(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    report_path = tmp_path / "report.json"
    plan_sha256 = _write_plan(plan_path)
    report_path.write_text(
        json.dumps(
            {
                "runtime_provenance": {
                    "code_revision": REVISION,
                    EXPERIMENT_PLAN_PROVENANCE_KEY: plan_sha256,
                    EXPERIMENT_PARAMETERS_PROVENANCE_KEY: {
                        "temperature": 0.0,
                        "max_tokens": 128,
                        "do_sample": False,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing admitted model identity provenance"):
        verify_model_configuration_binding(report_path, plan_path)


def test_main_emits_deterministic_json_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = tmp_path / "plan.json"
    report_path = tmp_path / "report.json"
    plan_sha256 = _write_plan(plan_path)
    _write_report(report_path, plan_sha256)
    monkeypatch.setattr(
        "sys.argv",
        ["remem-verify-model-binding", str(report_path), str(plan_path), "--json"],
    )

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["model_configuration_binding_verified"] is True
    assert payload["experiment_plan_sha256"] == plan_sha256
    assert payload["model_identity"] == MODEL_IDENTITY
