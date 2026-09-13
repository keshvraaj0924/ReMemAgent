"""Tests for frozen research-plan binding during benchmark verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.benchmark_manifest import save_benchmark_artifact_manifest
from experiments.evidence_bound_paired_cli import EXPERIMENT_PLAN_PROVENANCE_KEY
from experiments.research_experiment_plan import (
    RESEARCH_EXPERIMENT_PLAN_SCHEMA_VERSION,
    ResearchExperimentPlan,
    build_research_experiment_plan,
    write_research_experiment_plan,
)
from experiments.verify_benchmark_artifact import main, verify_report_artifact

REMEM_REVISION = "1" * 40
WEBSHOP_REVISION = "2" * 40
OTHER_REVISION = "3" * 40


def _build_plan(*, experiment_name: str = "webshop-memory-study") -> ResearchExperimentPlan:
    """Build one minimal valid frozen paired-experiment declaration."""

    return build_research_experiment_plan(
        experiment_name=experiment_name,
        remem_revision=REMEM_REVISION,
        benchmark_name="webshop",
        episode_count=4,
        max_steps=20,
        seeds=(11, 17),
        environment_factory="study.environments:build_webshop",
        success_evaluator="study.metrics:is_success",
        source_revisions={"webshop": WEBSHOP_REVISION},
        dependency_versions={"torch": "2.6.0"},
        baseline_policy_factory="study.policies:baseline",
        treatment_policy_factory="study.policies:treatment",
    )


def _write_plan(path: Path, plan: ResearchExperimentPlan) -> None:
    """Persist one canonical frozen plan for verification tests."""

    write_research_experiment_plan(path, plan)


def _write_report(
    path: Path,
    *,
    plan_sha256: str | None,
    code_revision: str = REMEM_REVISION,
) -> Path:
    """Persist a minimal report and return its integrity manifest path."""

    runtime_provenance: dict[str, object] = {"code_revision": code_revision}
    if plan_sha256 is not None:
        runtime_provenance[EXPERIMENT_PLAN_PROVENANCE_KEY] = plan_sha256
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episodes": [],
                "runtime_provenance": runtime_provenance,
            }
        ),
        encoding="utf-8",
    )
    return save_benchmark_artifact_manifest(path)


def test_verify_report_artifact_binds_frozen_experiment_plan(tmp_path: Path) -> None:
    """A plan-bound report should attest both canonical and exact retained-plan identities."""

    plan = _build_plan()
    plan_path = tmp_path / "experiment-plan.json"
    _write_plan(plan_path, plan)
    report_path = tmp_path / "paired.json"
    manifest_path = _write_report(report_path, plan_sha256=plan.sha256)

    result = verify_report_artifact(
        report_path,
        manifest_path,
        experiment_plan_path=plan_path,
    )

    plan_bytes = plan_path.read_bytes()
    assert result.experiment_plan_schema_version == RESEARCH_EXPERIMENT_PLAN_SCHEMA_VERSION
    assert result.experiment_plan_sha256 == plan.sha256
    assert result.experiment_plan_byte_count == len(plan_bytes)
    assert result.experiment_plan_file_sha256 == hashlib.sha256(plan_bytes).hexdigest()
    assert result.experiment_plan_name == plan.experiment_name
    assert result.experiment_plan_remem_revision == REMEM_REVISION


def test_verify_report_artifact_requires_retained_plan_for_plan_bound_report(
    tmp_path: Path,
) -> None:
    """A report that claims a plan digest must fail closed without the original plan."""

    plan = _build_plan()
    report_path = tmp_path / "paired.json"
    manifest_path = _write_report(report_path, plan_sha256=plan.sha256)

    with pytest.raises(ValueError, match="--experiment-plan"):
        verify_report_artifact(report_path, manifest_path)


def test_verify_report_artifact_rejects_plan_for_unbound_report(tmp_path: Path) -> None:
    """Supplying an unrelated plan must not imply a binding absent from the report."""

    plan = _build_plan()
    plan_path = tmp_path / "experiment-plan.json"
    _write_plan(plan_path, plan)
    report_path = tmp_path / "paired.json"
    manifest_path = _write_report(report_path, plan_sha256=None)

    with pytest.raises(ValueError, match="not bound to a research experiment plan"):
        verify_report_artifact(
            report_path,
            manifest_path,
            experiment_plan_path=plan_path,
        )


def test_verify_report_artifact_rejects_mismatched_plan_digest(tmp_path: Path) -> None:
    """A semantically different retained plan must not satisfy the persisted digest."""

    expected_plan = _build_plan(experiment_name="expected-study")
    supplied_plan = _build_plan(experiment_name="different-study")
    plan_path = tmp_path / "experiment-plan.json"
    _write_plan(plan_path, supplied_plan)
    report_path = tmp_path / "paired.json"
    manifest_path = _write_report(report_path, plan_sha256=expected_plan.sha256)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_report_artifact(
            report_path,
            manifest_path,
            experiment_plan_path=plan_path,
        )


def test_verify_report_artifact_rejects_plan_revision_drift(tmp_path: Path) -> None:
    """Report runtime revision and the frozen plan revision must describe the same code."""

    plan = _build_plan()
    plan_path = tmp_path / "experiment-plan.json"
    _write_plan(plan_path, plan)
    report_path = tmp_path / "paired.json"
    manifest_path = _write_report(
        report_path,
        plan_sha256=plan.sha256,
        code_revision=OTHER_REVISION,
    )

    with pytest.raises(ValueError, match="code revision does not match"):
        verify_report_artifact(
            report_path,
            manifest_path,
            experiment_plan_path=plan_path,
        )


def test_cli_json_attests_frozen_experiment_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public verifier CLI should expose retained-plan identity in canonical JSON."""

    plan = _build_plan()
    plan_path = tmp_path / "experiment-plan.json"
    _write_plan(plan_path, plan)
    report_path = tmp_path / "paired.json"
    _write_report(report_path, plan_sha256=plan.sha256)
    monkeypatch.setattr(
        "sys.argv",
        [
            "remem-verify-benchmark",
            str(report_path),
            "--experiment-plan",
            str(plan_path),
            "--json",
        ],
    )

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    attestation = json.loads(captured.out)
    assert attestation["experiment_plan_sha256"] == plan.sha256
    assert attestation["experiment_plan_name"] == plan.experiment_name
    assert attestation["experiment_plan_remem_revision"] == REMEM_REVISION
