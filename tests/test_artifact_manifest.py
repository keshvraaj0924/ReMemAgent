import json

import pytest

from remem.artifact_integrity import ArtifactIntegrityRecord
from remem.artifact_manifest import ArtifactManifest


def test_manifest_capture_is_sorted_and_verifiable(tmp_path) -> None:
    second = tmp_path / "second.json"
    first = tmp_path / "first.json"
    second.write_text('{"value":2}', encoding="utf-8")
    first.write_text('{"value":1}', encoding="utf-8")

    manifest = ArtifactManifest.capture(
        run_id="run-001",
        paths=[second, first],
        root=tmp_path,
    )

    assert [record.relative_path for record in manifest.artifacts] == [
        "first.json",
        "second.json",
    ]
    manifest.verify(root=tmp_path)


def test_manifest_round_trip_is_deterministic(tmp_path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text('{"accuracy":0.5}', encoding="utf-8")
    manifest = ArtifactManifest.capture(
        run_id="run-002",
        paths=[artifact],
        root=tmp_path,
    )

    restored = ArtifactManifest.from_json(manifest.to_json())

    assert restored == manifest
    assert restored.to_json() == manifest.to_json()


def test_manifest_rejects_duplicate_artifact_paths(tmp_path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate artifact path"):
        ArtifactManifest.capture(
            run_id="run-003",
            paths=[artifact, artifact],
            root=tmp_path,
        )


def test_manifest_rejects_mixed_run_ids(tmp_path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}", encoding="utf-8")
    record = ArtifactIntegrityRecord.capture(
        run_id="other-run",
        path=artifact,
        root=tmp_path,
    )
    payload = {
        "schema_version": 1,
        "run_id": "run-004",
        "artifacts": [json.loads(record.to_json())],
    }

    with pytest.raises(ValueError, match="artifact run_id must match manifest run_id"):
        ArtifactManifest.from_json(json.dumps(payload))


def test_manifest_detects_post_run_artifact_mutation(tmp_path) -> None:
    artifact = tmp_path / "raw.jsonl"
    artifact.write_text('{"reward":1}\n', encoding="utf-8")
    manifest = ArtifactManifest.capture(
        run_id="run-005",
        paths=[artifact],
        root=tmp_path,
    )
    artifact.write_text('{"reward":0}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="artifact integrity mismatch"):
        manifest.verify(root=tmp_path)


def test_manifest_rejects_unsorted_serialized_records(tmp_path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    first_record = ArtifactIntegrityRecord.capture(
        run_id="run-006",
        path=first,
        root=tmp_path,
    )
    second_record = ArtifactIntegrityRecord.capture(
        run_id="run-006",
        path=second,
        root=tmp_path,
    )
    payload = {
        "schema_version": 1,
        "run_id": "run-006",
        "artifacts": [
            json.loads(second_record.to_json()),
            json.loads(first_record.to_json()),
        ],
    }

    with pytest.raises(ValueError, match="artifacts must be sorted"):
        ArtifactManifest.from_json(json.dumps(payload))
