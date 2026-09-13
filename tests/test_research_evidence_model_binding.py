"""Regression tests for model configuration binding in research evidence verification."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from experiments.evidence_bound_paired_cli import (
    EXPERIMENT_PARAMETERS_PROVENANCE_KEY,
    EXPERIMENT_PLAN_PROVENANCE_KEY,
    MODEL_IDENTITY_PROVENANCE_KEY,
)
from experiments.research_evidence_cli import main
from experiments.research_evidence_record import (
    build_research_evidence_record,
    write_research_evidence_record,
)
from experiments.research_experiment_plan import (
    build_research_experiment_plan,
    write_research_experiment_plan,
)

REVISION = "a" * 40
MODEL_IDENTITY = "Qwen/Qwen2.5-7B-Instruct@pinned"


def _write_model_bound_record(root: Path, *, model_identity: str = MODEL_IDENTITY) -> Path:
    plan_path = root / "experiment-plan.json"
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
    write_research_experiment_plan(plan_path, plan)

    report_path = root / "paired-report.json"
    report_path.write_text(
        json.dumps(
            {
                "runtime_provenance": {
                    "code_revision": REVISION,
                    EXPERIMENT_PLAN_PROVENANCE_KEY: plan.sha256,
                    MODEL_IDENTITY_PROVENANCE_KEY: model_identity,
                    EXPERIMENT_PARAMETERS_PROVENANCE_KEY: {
                        "temperature": 0.0,
                        "max_tokens": 128,
                        "do_sample": False,
                    },
                }
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    record_path = root / "research-evidence.json"
    record = build_research_evidence_record(
        experiment_name=plan.experiment_name,
        evidence_level="E3",
        remem_revision=REVISION,
        artifacts={
            "experiment_plan": plan_path,
            "paired_report": report_path,
        },
        record_directory=root,
    )
    write_research_evidence_record(record_path, record)
    return record_path


def test_verify_model_binding_proves_retained_plan_and_report_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record_path = _write_model_bound_record(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-research-evidence",
            "verify",
            str(record_path),
            "--require-model-binding",
            "--json",
        ],
    )

    assert main() == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["model_configuration_binding_verified"] is True
    assert payload["model_identity"] == MODEL_IDENTITY
    assert payload["experiment_parameters"] == {
        "do_sample": False,
        "max_tokens": 128,
        "temperature": 0.0,
    }


def test_verify_model_binding_rejects_retained_runtime_model_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record_path = _write_model_bound_record(tmp_path, model_identity="different-model")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-research-evidence",
            "verify",
            str(record_path),
            "--require-model-binding",
        ],
    )

    assert main() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "model identity does not match research experiment plan" in captured.err


def test_verify_model_binding_requires_canonical_evidence_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "paired-report.json"
    report_path.write_text("{}\n", encoding="utf-8")
    record_path = tmp_path / "research-evidence.json"
    record = build_research_evidence_record(
        experiment_name="webshop-controlled",
        evidence_level="E3",
        remem_revision=REVISION,
        artifacts={"paired_report": report_path},
        record_directory=tmp_path,
    )
    write_research_evidence_record(record_path, record)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-research-evidence",
            "verify",
            str(record_path),
            "--require-model-binding",
        ],
    )

    assert main() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "model-binding verification requires evidence roles" in captured.err
    assert "experiment_plan" in captured.err
