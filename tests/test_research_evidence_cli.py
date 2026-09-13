from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from experiments.research_evidence_cli import main
from experiments.research_evidence_record import (
    build_research_evidence_record,
    write_research_evidence_record,
)
from experiments.research_experiment_plan import (
    build_research_experiment_plan,
    write_research_experiment_plan,
)

REVISION = "1" * 40
OTHER_REVISION = "2" * 40
SOURCE_REVISION = "3" * 40


def _write_record(root: Path) -> Path:
    report = root / "report.json"
    report.write_text('{"success":true}\n', encoding="utf-8")
    record_path = root / "research-evidence.json"
    record = build_research_evidence_record(
        experiment_name="webshop-seed-study",
        evidence_level="E3",
        remem_revision=REVISION,
        artifacts={"paired_report": report},
        record_directory=root,
        notes=("predeclared seeds",),
    )
    write_research_evidence_record(record_path, record)
    return record_path


def _write_plan_bound_record(
    root: Path,
    *,
    attested_plan_sha256: str | None = None,
) -> tuple[Path, str]:
    plan_path = root / "experiment-plan.json"
    plan = build_research_experiment_plan(
        experiment_name="webshop-seed-study",
        remem_revision=REVISION,
        benchmark_name="webshop",
        episode_count=10,
        max_steps=50,
        seeds=(11, 17),
        environment_factory="example.environments:build_webshop",
        success_evaluator="example.metrics:is_success",
        source_revisions={"webshop": SOURCE_REVISION},
        dependency_versions={"torch": "2.8.0"},
        baseline_policy_factory="example.policies:build_baseline",
        treatment_policy_factory="example.policies:build_remem",
    )
    write_research_experiment_plan(plan_path, plan)
    plan_file_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    attestation_path = root / "verification.json"
    attestation_path.write_text(
        json.dumps(
            {
                "experiment_plan_sha256": attested_plan_sha256 or plan.sha256,
                "experiment_plan_file_sha256": plan_file_sha256,
                "experiment_plan_name": plan.experiment_name,
                "experiment_plan_remem_revision": plan.remem_revision,
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
            "verification": attestation_path,
        },
        record_directory=root,
    )
    write_research_evidence_record(record_path, record)
    return record_path, plan.sha256


def test_verify_json_reports_exact_record_identity(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    record_path = _write_record(tmp_path)
    raw_bytes = record_path.read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-research-evidence",
            "verify",
            str(record_path),
            "--expected-revision",
            REVISION,
            "--json",
        ],
    )

    assert main() == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == {
        "artifact_count": 1,
        "evidence_level": "E3",
        "experiment_name": "webshop-seed-study",
        "record_byte_count": len(raw_bytes),
        "record_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "remem_revision": REVISION,
        "schema_version": 1,
        "verified": True,
    }


def test_verify_fails_closed_on_revision_mismatch(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    record_path = _write_record(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-research-evidence",
            "verify",
            str(record_path),
            "--expected-revision",
            OTHER_REVISION,
            "--json",
        ],
    )

    assert main() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "research evidence revision mismatch" in captured.err
    assert REVISION in captured.err
    assert OTHER_REVISION in captured.err


def test_human_verify_output_includes_bound_revision(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    record_path = _write_record(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["remem-research-evidence", "verify", str(record_path)],
    )

    assert main() == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert f"revision {REVISION}" in captured.out


def test_verify_plan_binding_proves_retained_plan_attestation_identity(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    record_path, plan_sha256 = _write_plan_bound_record(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-research-evidence",
            "verify",
            str(record_path),
            "--expected-revision",
            REVISION,
            "--require-plan-binding",
            "--json",
        ],
    )

    assert main() == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["experiment_plan_sha256"] == plan_sha256
    assert payload["plan_binding_verified"] is True
    assert payload["remem_revision"] == REVISION


def test_verify_plan_binding_rejects_semantically_unrelated_attestation(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    record_path, _ = _write_plan_bound_record(
        tmp_path,
        attested_plan_sha256="f" * 64,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-research-evidence",
            "verify",
            str(record_path),
            "--require-plan-binding",
        ],
    )

    assert main() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "experiment-plan SHA-256 mismatch" in captured.err


def test_verify_plan_binding_requires_canonical_evidence_roles(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    record_path = _write_record(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-research-evidence",
            "verify",
            str(record_path),
            "--require-plan-binding",
        ],
    )

    assert main() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "plan-binding verification requires evidence roles" in captured.err
    assert "experiment_plan" in captured.err
    assert "verification" in captured.err
