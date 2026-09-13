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

REVISION = "1" * 40
OTHER_REVISION = "2" * 40


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
