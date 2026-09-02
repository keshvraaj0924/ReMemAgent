"""Integrity manifests for persisted benchmark report artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.benchmark_report import BENCHMARK_REPORT_SCHEMA_VERSION


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


__all__ = [
    "BenchmarkArtifactManifest",
    "build_benchmark_artifact_manifest",
    "verify_benchmark_artifact",
]
