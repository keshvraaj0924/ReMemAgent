from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.research_evidence_record import (
    RESEARCH_EVIDENCE_RECORD_SCHEMA_VERSION,
    ResearchEvidenceRecord,
    build_research_evidence_record,
    canonical_research_evidence_record_json,
    load_research_evidence_record,
    verify_research_evidence_record,
    write_research_evidence_record,
)

REVISION = "1" * 40


def _write_artifacts(root: Path) -> dict[str, Path]:
    report = root / "report.json"
    attestation = root / "verification.json"
    report.write_text('{"result":1}\n', encoding="utf-8")
    attestation.write_text('{"verified":true}\n', encoding="utf-8")
    return {"benchmark_report": report, "verification_attestation": attestation}


def test_build_record_is_deterministic_and_role_sorted(tmp_path: Path) -> None:
    artifacts = _write_artifacts(tmp_path)

    record = build_research_evidence_record(
        experiment_name="webshop-seed-study",
        evidence_level="E3",
        remem_revision=REVISION,
        artifacts={
            "verification_attestation": artifacts["verification_attestation"],
            "benchmark_report": artifacts["benchmark_report"],
        },
        record_directory=tmp_path,
        notes=("predeclared seeds",),
    )

    assert record.schema_version == RESEARCH_EVIDENCE_RECORD_SCHEMA_VERSION
    assert [artifact.role for artifact in record.artifacts] == [
        "benchmark_report",
        "verification_attestation",
    ]
    payload = canonical_research_evidence_record_json(record)
    assert payload.endswith("\n")
    assert payload == canonical_research_evidence_record_json(record)


def test_round_trip_and_verify_exact_bytes(tmp_path: Path) -> None:
    record_path = tmp_path / "research-evidence.json"
    record = build_research_evidence_record(
        experiment_name="alfworld-study",
        evidence_level="E2",
        remem_revision=REVISION,
        artifacts=_write_artifacts(tmp_path),
        record_directory=tmp_path,
    )

    write_research_evidence_record(record_path, record)

    assert load_research_evidence_record(record_path) == record
    assert verify_research_evidence_record(record_path) == record


def test_verify_rejects_tampered_artifact(tmp_path: Path) -> None:
    record_path = tmp_path / "research-evidence.json"
    artifacts = _write_artifacts(tmp_path)
    record = build_research_evidence_record(
        experiment_name="tamper-check",
        evidence_level="E2",
        remem_revision=REVISION,
        artifacts=artifacts,
        record_directory=tmp_path,
    )
    write_research_evidence_record(record_path, record)
    artifacts["benchmark_report"].write_text('{"result":2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_research_evidence_record(record_path)


def test_writer_refuses_to_overwrite_and_cleans_temporary_files(tmp_path: Path) -> None:
    record_path = tmp_path / "research-evidence.json"
    record = build_research_evidence_record(
        experiment_name="no-overwrite",
        evidence_level="E1",
        remem_revision=REVISION,
        artifacts=_write_artifacts(tmp_path),
        record_directory=tmp_path,
    )
    write_research_evidence_record(record_path, record)
    original = record_path.read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        write_research_evidence_record(record_path, record)

    assert record_path.read_bytes() == original
    assert list(tmp_path.glob(f".{record_path.name}.*.tmp")) == []


def test_build_rejects_artifact_outside_record_directory(tmp_path: Path) -> None:
    record_root = tmp_path / "record"
    record_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="beneath record directory"):
        build_research_evidence_record(
            experiment_name="unsafe-path",
            evidence_level="E1",
            remem_revision=REVISION,
            artifacts={"readiness": outside},
            record_directory=record_root,
        )


def test_build_rejects_duplicate_resolved_artifact_paths(tmp_path: Path) -> None:
    artifact = tmp_path / "same.json"
    artifact.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="same path"):
        build_research_evidence_record(
            experiment_name="duplicate-path",
            evidence_level="E2",
            remem_revision=REVISION,
            artifacts={"report": artifact, "manifest": artifact},
            record_directory=tmp_path,
        )


def test_from_dict_rejects_path_traversal() -> None:
    payload = {
        "schema_version": RESEARCH_EVIDENCE_RECORD_SCHEMA_VERSION,
        "experiment_name": "unsafe",
        "evidence_level": "E2",
        "remem_revision": REVISION,
        "artifacts": [
            {
                "role": "report",
                "path": "../report.json",
                "byte_count": 2,
                "sha256": "0" * 64,
            }
        ],
        "notes": [],
    }

    with pytest.raises(ValueError, match="safe relative path"):
        ResearchEvidenceRecord.from_dict(payload)


def test_load_rejects_unknown_schema_fields(tmp_path: Path) -> None:
    record_path = tmp_path / "record.json"
    record_path.write_text(
        json.dumps(
            {
                "schema_version": RESEARCH_EVIDENCE_RECORD_SCHEMA_VERSION,
                "experiment_name": "unknown-field",
                "evidence_level": "E1",
                "remem_revision": REVISION,
                "artifacts": [
                    {
                        "role": "readiness",
                        "path": "readiness.json",
                        "byte_count": 2,
                        "sha256": "0" * 64,
                    }
                ],
                "notes": [],
                "unexpected": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected or missing fields"):
        load_research_evidence_record(record_path)
