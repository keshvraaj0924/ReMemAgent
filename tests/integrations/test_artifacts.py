"""Tests for deterministic training-artifact integrity manifests."""

from pathlib import Path

import pytest

from remem.integrations.artifacts import (
    TRAINING_ARTIFACT_SCHEMA_VERSION,
    build_training_artifact_manifest,
    verify_training_artifact,
    write_training_artifact_manifest,
)


def test_build_manifest_counts_rows_and_hashes_exact_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "batch.jsonl"
    artifact.write_bytes(b'{"a":1}\n{"a":2}\n')

    manifest = build_training_artifact_manifest(artifact, artifact_type="grpo")

    assert manifest.schema_version == TRAINING_ARTIFACT_SCHEMA_VERSION
    assert manifest.artifact_type == "grpo"
    assert manifest.row_count == 2
    assert len(manifest.sha256) == 64
    assert verify_training_artifact(artifact, manifest)


def test_manifest_detects_artifact_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "batch.jsonl"
    artifact.write_bytes(b'{"reward":1.0}\n')
    manifest = build_training_artifact_manifest(artifact, artifact_type="verl")

    artifact.write_bytes(b'{"reward":2.0}\n')

    assert not verify_training_artifact(artifact, manifest)


def test_manifest_rejects_malformed_or_blank_jsonl(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_bytes(b"not-json\n")
    blank = tmp_path / "blank.jsonl"
    blank.write_bytes(b'{"a":1}\n\n')

    with pytest.raises(ValueError):
        build_training_artifact_manifest(malformed, artifact_type="grpo")
    with pytest.raises(ValueError):
        build_training_artifact_manifest(blank, artifact_type="grpo")


def test_manifest_writer_is_deterministic(tmp_path: Path) -> None:
    artifact = tmp_path / "batch.jsonl"
    artifact.write_bytes(b'{"a":1}\n')
    manifest = build_training_artifact_manifest(artifact, artifact_type="grpo")
    first = tmp_path / "one" / "manifest.json"
    second = tmp_path / "two" / "manifest.json"

    write_training_artifact_manifest(manifest, first)
    write_training_artifact_manifest(manifest, second)

    assert first.read_bytes() == second.read_bytes()
