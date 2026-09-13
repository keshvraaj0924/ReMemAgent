from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from experiments.benchmark_manifest import save_benchmark_artifact_manifest
from experiments.research_evidence_cli import main
from experiments.research_evidence_record import (
    build_research_evidence_record,
    write_research_evidence_record,
)

REVISION = "1" * 40


def _write_manifest_bound_record(
    root: Path,
    *,
    use_unrelated_manifest: bool = False,
) -> tuple[Path, str]:
    report_path = root / "paired-report.json"
    report_path.write_text('{"schema_version":1}\n', encoding="utf-8")

    manifest_report_path = report_path
    if use_unrelated_manifest:
        manifest_report_path = root / "unrelated-report.json"
        manifest_report_path.write_text(
            '{"schema_version":1,"unrelated":true}\n',
            encoding="utf-8",
        )

    manifest_path = root / "paired-report.manifest.json"
    save_benchmark_artifact_manifest(manifest_report_path, manifest_path)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    record_path = root / "research-evidence.json"
    record = build_research_evidence_record(
        experiment_name="webshop-seed-study",
        evidence_level="E3",
        remem_revision=REVISION,
        artifacts={
            "paired_report": report_path,
            "report_manifest": manifest_path,
        },
        record_directory=root,
    )
    write_research_evidence_record(record_path, record)
    return record_path, manifest_sha256


def test_verify_manifest_binding_proves_manifest_matches_retained_report(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    record_path, manifest_sha256 = _write_manifest_bound_record(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-research-evidence",
            "verify",
            str(record_path),
            "--require-manifest-binding",
            "--json",
        ],
    )

    assert main() == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["report_manifest_sha256"] == manifest_sha256
    assert payload["manifest_binding_verified"] is True


def test_verify_manifest_binding_rejects_manifest_for_different_report(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    record_path, _ = _write_manifest_bound_record(
        tmp_path,
        use_unrelated_manifest=True,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-research-evidence",
            "verify",
            str(record_path),
            "--require-manifest-binding",
        ],
    )

    assert main() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "benchmark artifact integrity verification failed" in captured.err


def test_verify_manifest_binding_requires_canonical_evidence_roles(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    report_path = tmp_path / "paired-report.json"
    report_path.write_text('{"schema_version":1}\n', encoding="utf-8")
    record_path = tmp_path / "research-evidence.json"
    record = build_research_evidence_record(
        experiment_name="webshop-seed-study",
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
            "--require-manifest-binding",
        ],
    )

    assert main() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "manifest-binding verification requires evidence roles" in captured.err
    assert "report_manifest" in captured.err
