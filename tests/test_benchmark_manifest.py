import json
from pathlib import Path

import pytest

from experiments.benchmark_manifest import (
    BenchmarkArtifactManifest,
    build_benchmark_artifact_manifest,
    load_benchmark_artifact_manifest,
    save_benchmark_artifact_manifest,
    verify_benchmark_artifact,
)


def _write_valid_report(path: Path) -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "benchmark_name": "test"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_build_benchmark_artifact_manifest_hashes_serialized_report(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    _write_valid_report(report_path)

    manifest = build_benchmark_artifact_manifest(report_path)

    assert manifest.schema_version == 1
    assert manifest.byte_count == report_path.stat().st_size
    assert len(manifest.sha256) == 64


def test_build_benchmark_artifact_manifest_rejects_boolean_schema(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"schema_version": True}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported benchmark report schema version"):
        build_benchmark_artifact_manifest(report_path)


def test_verify_benchmark_artifact_accepts_unchanged_report(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    _write_valid_report(report_path)
    manifest = build_benchmark_artifact_manifest(report_path)

    verify_benchmark_artifact(report_path, manifest)


def test_verify_benchmark_artifact_rejects_mutated_report(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    _write_valid_report(report_path)
    manifest = build_benchmark_artifact_manifest(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["seed"] = 18
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="integrity verification failed"):
        verify_benchmark_artifact(report_path, manifest)


def test_verify_benchmark_artifact_rejects_wrong_digest_with_matching_shape(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    _write_valid_report(report_path)
    manifest = build_benchmark_artifact_manifest(report_path)
    wrong_manifest = BenchmarkArtifactManifest(
        schema_version=manifest.schema_version,
        byte_count=manifest.byte_count,
        sha256=("0" if manifest.sha256[0] != "0" else "1") + manifest.sha256[1:],
    )

    with pytest.raises(ValueError, match="integrity verification failed"):
        verify_benchmark_artifact(report_path, wrong_manifest)


def test_build_benchmark_artifact_manifest_rejects_invalid_schema(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported benchmark report schema version"):
        build_benchmark_artifact_manifest(report_path)


def test_verify_benchmark_artifact_rejects_wrong_manifest(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    _write_valid_report(report_path)
    manifest = BenchmarkArtifactManifest(schema_version=1, byte_count=0, sha256="0" * 64)

    verify_error = pytest.raises(ValueError, match="integrity verification failed")
    with verify_error:
        verify_benchmark_artifact(report_path, manifest)


def test_save_and_load_benchmark_artifact_manifest_round_trip(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    manifest_path = tmp_path / "integrity" / "report.manifest.json"
    _write_valid_report(report_path)

    saved_path = save_benchmark_artifact_manifest(report_path, manifest_path)
    loaded_manifest = load_benchmark_artifact_manifest(saved_path)

    assert saved_path == manifest_path
    verify_benchmark_artifact(report_path, loaded_manifest)


def test_load_benchmark_artifact_manifest_rejects_unsupported_schema(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"manifest_schema_version": 999}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported benchmark artifact manifest schema version"):
        load_benchmark_artifact_manifest(manifest_path)


def test_load_benchmark_artifact_manifest_rejects_boolean_manifest_schema(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    manifest_path = tmp_path / "manifest.json"
    _write_valid_report(report_path)
    valid_manifest = build_benchmark_artifact_manifest(report_path).to_dict()
    manifest_path.write_text(
        json.dumps({"manifest_schema_version": True, **valid_manifest}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported benchmark artifact manifest schema version"):
        load_benchmark_artifact_manifest(manifest_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("byte_count", True),
        ("byte_count", -1),
        ("sha256", "g" * 64),
        ("sha256", "0" * 63),
    ],
)
def test_load_benchmark_artifact_manifest_rejects_invalid_integrity_fields(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    report_path = tmp_path / "report.json"
    manifest_path = tmp_path / "manifest.json"
    _write_valid_report(report_path)
    valid_manifest = build_benchmark_artifact_manifest(report_path).to_dict()
    valid_manifest[field] = value
    manifest_path.write_text(
        json.dumps(
            {"manifest_schema_version": 1, **valid_manifest},
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid integrity fields"):
        load_benchmark_artifact_manifest(manifest_path)
