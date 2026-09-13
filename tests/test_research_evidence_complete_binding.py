from __future__ import annotations

import json
import sys
from pathlib import Path

import experiments.research_evidence_cli as research_evidence_cli
from experiments.research_evidence_record import (
    build_research_evidence_record,
    write_research_evidence_record,
)
from experiments.research_sidecar_binding import ResearchSidecarBinding

REVISION = "1" * 40
PLAN_SHA256 = "2" * 64
REPORT_SHA256 = "3" * 64
MANIFEST_SHA256 = "4" * 64
OBSERVABILITY_SHA256 = "5" * 64
DISTRIBUTION_SHA256 = "6" * 64
BUNDLE_SHA256 = "7" * 64


def _write_minimal_record(root: Path) -> Path:
    artifact_path = root / "retained-note.txt"
    artifact_path.write_text("retained evidence\n", encoding="utf-8")
    record_path = root / "research-evidence.json"
    record = build_research_evidence_record(
        experiment_name="webshop-seed-study",
        evidence_level="E2",
        remem_revision=REVISION,
        artifacts={"retained_note": artifact_path},
        record_directory=root,
    )
    write_research_evidence_record(record_path, record)
    return record_path


def test_complete_binding_runs_every_semantic_verifier_and_reports_identity(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    record_path = _write_minimal_record(tmp_path)
    calls: list[str] = []

    def verify_plan_binding(record_path: Path, record: object) -> str:
        calls.append("plan")
        return PLAN_SHA256

    def verify_report_binding(record_path: Path, record: object) -> str:
        calls.append("report")
        return REPORT_SHA256

    def verify_manifest_binding(record_path: Path, record: object) -> str:
        calls.append("manifest")
        return MANIFEST_SHA256

    def verify_sidecar_binding(record_path: Path) -> ResearchSidecarBinding:
        calls.append("sidecar")
        return ResearchSidecarBinding(
            observability_sha256=OBSERVABILITY_SHA256,
            distribution_sha256=DISTRIBUTION_SHA256,
            bundle_sha256=BUNDLE_SHA256,
        )

    monkeypatch.setattr(research_evidence_cli, "_verify_plan_binding", verify_plan_binding)
    monkeypatch.setattr(research_evidence_cli, "_verify_report_binding", verify_report_binding)
    monkeypatch.setattr(research_evidence_cli, "_verify_manifest_binding", verify_manifest_binding)
    monkeypatch.setattr(
        research_evidence_cli,
        "verify_research_sidecar_binding",
        verify_sidecar_binding,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-research-evidence",
            "verify",
            str(record_path),
            "--require-complete-binding",
            "--json",
        ],
    )

    assert research_evidence_cli.main() == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert calls == ["plan", "report", "manifest", "sidecar"]
    assert payload["complete_binding_verified"] is True
    assert payload["plan_binding_verified"] is True
    assert payload["report_binding_verified"] is True
    assert payload["manifest_binding_verified"] is True
    assert payload["sidecar_binding_verified"] is True
    assert payload["experiment_plan_sha256"] == PLAN_SHA256
    assert payload["paired_report_sha256"] == REPORT_SHA256
    assert payload["report_manifest_sha256"] == MANIFEST_SHA256
    assert payload["observability_sidecar_sha256"] == OBSERVABILITY_SHA256
    assert payload["distribution_sidecar_sha256"] == DISTRIBUTION_SHA256
    assert payload["sidecar_bundle_sha256"] == BUNDLE_SHA256


def test_complete_binding_fails_closed_when_chain_is_incomplete(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    record_path = _write_minimal_record(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-research-evidence",
            "verify",
            str(record_path),
            "--require-complete-binding",
        ],
    )

    assert research_evidence_cli.main() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "plan-binding verification requires evidence roles" in captured.err
    assert "experiment_plan" in captured.err
    assert "verification" in captured.err
