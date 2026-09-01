"""Integrity manifests for offline GRPO and verl training artifacts.

The manifest records the exact JSONL byte digest and row count produced by a
ReMemAgent dataset writer. It is intentionally framework-neutral: downstream
trainers can verify the artifact before ingestion without importing a trainer
or tokenizer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


TRAINING_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TrainingArtifactManifest:
    """Integrity metadata for one deterministic JSONL training artifact."""

    schema_version: int
    artifact_type: str
    row_count: int
    sha256: str

    def __post_init__(self) -> None:
        """Validate manifest fields before persistence."""

        if self.schema_version != TRAINING_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported training artifact schema version")
        if not self.artifact_type.strip():
            raise ValueError("artifact_type must not be empty")
        if self.row_count < 1:
            raise ValueError("row_count must be positive")
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible manifest."""

        return {
            "artifact_type": self.artifact_type,
            "row_count": self.row_count,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
        }


def build_training_artifact_manifest(
    jsonl_path: str | Path,
    *,
    artifact_type: str,
) -> TrainingArtifactManifest:
    """Compute integrity metadata for an existing JSONL artifact.

    The file is parsed while hashing so malformed JSONL is rejected before a
    manifest can certify the artifact. Blank lines are rejected because they
    would make row counting ambiguous across downstream readers.
    """

    path = Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    digest = hashlib.sha256()
    row_count = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            if not raw_line.endswith(b"\n"):
                raise ValueError("JSONL artifact must end every row with a newline")
            line = raw_line[:-1]
            if not line:
                raise ValueError("JSONL artifact must not contain blank lines")
            json.loads(line.decode("utf-8"))
            digest.update(raw_line)
            row_count += 1

    if row_count == 0:
        raise ValueError("JSONL artifact must contain at least one row")

    return TrainingArtifactManifest(
        schema_version=TRAINING_ARTIFACT_SCHEMA_VERSION,
        artifact_type=artifact_type,
        row_count=row_count,
        sha256=digest.hexdigest(),
    )


def write_training_artifact_manifest(
    manifest: TrainingArtifactManifest,
    output_path: str | Path,
) -> None:
    """Persist a training-artifact manifest with deterministic JSON encoding."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":"))
    path.write_text(payload + "\n", encoding="utf-8")


def verify_training_artifact(
    jsonl_path: str | Path,
    manifest: TrainingArtifactManifest,
) -> bool:
    """Return whether a JSONL artifact still matches its certified manifest."""

    try:
        observed = build_training_artifact_manifest(
            jsonl_path,
            artifact_type=manifest.artifact_type,
        )
    except (FileNotFoundError, TypeError, UnicodeDecodeError, ValueError):
        return False
    return observed == manifest
