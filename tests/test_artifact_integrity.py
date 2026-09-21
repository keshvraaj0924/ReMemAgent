"""Tests for content-addressed experiment artifact records."""

from __future__ import annotations

import json

import pytest

from remem.artifact_integrity import ArtifactIntegrityRecord


def test_capture_and_verify_artifact(tmp_path) -> None:
    artifact = tmp_path / "raw" / "metrics.jsonl"
    artifact.parent.mkdir()
    artifact.write_text('{"reward":0.5}\n', encoding="utf-8")

    record = ArtifactIntegrityRecord.capture(run_id="run-123", path=artifact, root=tmp_path)

    assert record.relative_path == "raw/metrics.jsonl"
    assert record.size_bytes == len(artifact.read_bytes())
    assert len(record.sha256) == 64
    record.verify(root=tmp_path)


def test_capture_resolves_relative_path_against_root(tmp_path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    artifact = run_root / "result.json"
    artifact.write_text("{}", encoding="utf-8")
    other_directory = tmp_path / "caller"
    other_directory.mkdir()
    monkeypatch.chdir(other_directory)

    record = ArtifactIntegrityRecord.capture(
        run_id="run-123",
        path="result.json",
        root=run_root,
    )

    assert record.relative_path == "result.json"
    record.verify(root=run_root)


def test_verify_rejects_modified_artifact(tmp_path) -> None:
    artifact = tmp_path / "result.json"
    artifact.write_text("original", encoding="utf-8")
    record = ArtifactIntegrityRecord.capture(run_id="run-123", path=artifact, root=tmp_path)
    artifact.write_text("modified", encoding="utf-8")

    with pytest.raises(ValueError, match="integrity mismatch"):
        record.verify(root=tmp_path)


def test_capture_rejects_artifact_outside_root(tmp_path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    artifact = tmp_path / "outside.json"
    artifact.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="contained by root"):
        ArtifactIntegrityRecord.capture(run_id="run-123", path=artifact, root=root)


def test_json_round_trip_preserves_record(tmp_path) -> None:
    artifact = tmp_path / "config.json"
    artifact.write_text('{"seed":7}', encoding="utf-8")
    record = ArtifactIntegrityRecord.capture(run_id="run-123", path=artifact, root=tmp_path)

    restored = ArtifactIntegrityRecord.from_json(record.to_json())

    assert restored == record


def test_from_json_rejects_path_traversal(tmp_path) -> None:
    artifact = tmp_path / "result.json"
    artifact.write_text("{}", encoding="utf-8")
    record = ArtifactIntegrityRecord.capture(run_id="run-123", path=artifact, root=tmp_path)
    payload = json.loads(record.to_json())
    payload["relative_path"] = "../result.json"

    with pytest.raises(ValueError, match="stay within root"):
        ArtifactIntegrityRecord.from_json(json.dumps(payload))


def test_from_json_rejects_unknown_fields(tmp_path) -> None:
    artifact = tmp_path / "result.json"
    artifact.write_text("{}", encoding="utf-8")
    record = ArtifactIntegrityRecord.capture(run_id="run-123", path=artifact, root=tmp_path)
    payload = json.loads(record.to_json())
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="unknown fields"):
        ArtifactIntegrityRecord.from_json(json.dumps(payload))
