"""Integrity manifests for persisted benchmark report artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.benchmark_report import BENCHMARK_REPORT_SCHEMA_VERSION

MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BenchmarkArtifactManifest:
    """Integrity metadata for one serialized benchmark report."""

    schema_version: int
    byte_count: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible manifest representation."""

        return {
            "schema_version": self.schema_version,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
        }


def build_benchmark_artifact_manifest(report_path: Path) -> BenchmarkArtifactManifest:
    """Hash a persisted benchmark report and validate its declared schema."""

    payload = report_path.read_bytes()
    document = _load_json_document(report_path, payload)
    schema_version = document.get("schema_version")
    if schema_version != BENCHMARK_REPORT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported benchmark report schema version: "
            f"{schema_version!r}"
        )
    return BenchmarkArtifactManifest(
        schema_version=schema_version,
        byte_count=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def save_benchmark_artifact_manifest(
    report_path: Path,
    manifest_path: Path | None = None,
) -> Path:
    """Persist an exact-byte integrity manifest beside a benchmark report."""

    selected_manifest_path = manifest_path or report_path.with_suffix(
        report_path.suffix + ".manifest.json"
    )
    manifest = build_benchmark_artifact_manifest(report_path)
    payload = json.dumps(
        {"manifest_schema_version": MANIFEST_SCHEMA_VERSION, **manifest.to_dict()},
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    selected_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{selected_manifest_path.name}.",
        dir=selected_manifest_path.parent,
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, selected_manifest_path)
        _sync_directory(selected_manifest_path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return selected_manifest_path


def load_benchmark_artifact_manifest(manifest_path: Path) -> BenchmarkArtifactManifest:
    """Load and validate a persisted benchmark artifact manifest."""

    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid benchmark artifact manifest: {manifest_path}") from exc
    if not isinstance(document, dict):
        raise ValueError("benchmark artifact manifest root must be an object")
    if document.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported benchmark artifact manifest schema version: "
            f"{document.get('manifest_schema_version')!r}"
        )
    try:
        schema_version = int(document["schema_version"])
        byte_count = int(document["byte_count"])
        sha256 = document["sha256"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("benchmark artifact manifest is missing required fields") from exc
    if byte_count < 0 or not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError("benchmark artifact manifest contains invalid integrity fields")
    return BenchmarkArtifactManifest(
        schema_version=schema_version,
        byte_count=byte_count,
        sha256=sha256,
    )


def verify_benchmark_artifact(
    report_path: Path,
    manifest: BenchmarkArtifactManifest,
) -> None:
    """Fail closed when a benchmark report differs from its integrity manifest."""

    current = build_benchmark_artifact_manifest(report_path)
    if current != manifest:
        raise ValueError(
            "benchmark artifact integrity verification failed: "
            f"expected {manifest.sha256}, found {current.sha256}"
        )


def _load_json_document(report_path: Path, payload: bytes) -> dict[str, Any]:
    """Parse a benchmark artifact and require a JSON object at its root."""

    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid benchmark report JSON: {report_path}") from exc
    if not isinstance(document, dict):
        raise ValueError("benchmark report JSON root must be an object")
    return document


def _sync_directory(directory: Path) -> None:
    """Best-effort sync of directory metadata after atomic replacement."""

    try:
        file_descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


__all__ = [
    "BenchmarkArtifactManifest",
    "MANIFEST_SCHEMA_VERSION",
    "build_benchmark_artifact_manifest",
    "load_benchmark_artifact_manifest",
    "save_benchmark_artifact_manifest",
    "verify_benchmark_artifact",
]
