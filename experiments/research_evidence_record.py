"""Versioned research-evidence records for retained experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

RESEARCH_EVIDENCE_RECORD_SCHEMA_VERSION = 1
EVIDENCE_LEVELS = frozenset({"E0", "E1", "E2", "E3"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    """Exact-byte identity for one artifact retained with an experiment."""

    role: str
    path: str
    byte_count: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "role": self.role,
            "path": self.path,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ResearchEvidenceRecord:
    """Versioned index that binds a research claim level to retained artifact bytes."""

    schema_version: int
    experiment_name: str
    evidence_level: str
    remem_revision: str
    artifacts: tuple[EvidenceArtifact, ...]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "evidence_level": self.evidence_level,
            "remem_revision": self.remem_revision,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResearchEvidenceRecord:
        """Parse and strictly validate one persisted record payload."""

        expected_keys = {
            "schema_version",
            "experiment_name",
            "evidence_level",
            "remem_revision",
            "artifacts",
            "notes",
        }
        if set(payload) != expected_keys:
            raise ValueError("research evidence record has unexpected or missing fields")
        schema_version = payload["schema_version"]
        if schema_version != RESEARCH_EVIDENCE_RECORD_SCHEMA_VERSION:
            raise ValueError(f"unsupported research evidence record schema: {schema_version!r}")
        experiment_name = _validated_non_empty_string(payload["experiment_name"], "experiment_name")
        evidence_level = _validated_evidence_level(payload["evidence_level"])
        remem_revision = _validated_revision(payload["remem_revision"])
        artifacts_payload = payload["artifacts"]
        if not isinstance(artifacts_payload, list) or not artifacts_payload:
            raise ValueError("research evidence record artifacts must be a non-empty list")
        artifacts = tuple(_parse_artifact(item) for item in artifacts_payload)
        _validate_unique_artifacts(artifacts)
        notes_payload = payload["notes"]
        if not isinstance(notes_payload, list):
            raise ValueError("research evidence record notes must be a list")
        notes = tuple(_validated_non_empty_string(note, "note") for note in notes_payload)
        return cls(
            schema_version=RESEARCH_EVIDENCE_RECORD_SCHEMA_VERSION,
            experiment_name=experiment_name,
            evidence_level=evidence_level,
            remem_revision=remem_revision,
            artifacts=artifacts,
            notes=notes,
        )


def build_research_evidence_record(
    *,
    experiment_name: str,
    evidence_level: str,
    remem_revision: str,
    artifacts: Mapping[str, str | Path],
    record_directory: str | Path,
    notes: Sequence[str] = (),
) -> ResearchEvidenceRecord:
    """Build an exact-byte record for artifacts beneath the record directory."""

    validated_name = _validated_non_empty_string(experiment_name, "experiment_name")
    validated_level = _validated_evidence_level(evidence_level)
    validated_revision = _validated_revision(remem_revision)
    if not artifacts:
        raise ValueError("at least one retained evidence artifact is required")
    root = Path(record_directory).resolve()
    evidence_artifacts: list[EvidenceArtifact] = []
    resolved_paths: set[Path] = set()
    for role, raw_path in sorted(artifacts.items()):
        validated_role = _validated_non_empty_string(role, "artifact role")
        path = Path(raw_path).resolve()
        try:
            relative_path = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"artifact path must be beneath record directory: {path}") from exc
        if path in resolved_paths:
            raise ValueError(f"multiple artifact roles reference the same path: {path}")
        if not path.is_file():
            raise ValueError(f"retained evidence artifact is not a file: {path}")
        resolved_paths.add(path)
        raw_bytes = path.read_bytes()
        evidence_artifacts.append(
            EvidenceArtifact(
                role=validated_role,
                path=relative_path.as_posix(),
                byte_count=len(raw_bytes),
                sha256=hashlib.sha256(raw_bytes).hexdigest(),
            )
        )
    validated_notes = tuple(_validated_non_empty_string(note, "note") for note in notes)
    return ResearchEvidenceRecord(
        schema_version=RESEARCH_EVIDENCE_RECORD_SCHEMA_VERSION,
        experiment_name=validated_name,
        evidence_level=validated_level,
        remem_revision=validated_revision,
        artifacts=tuple(evidence_artifacts),
        notes=validated_notes,
    )


def canonical_research_evidence_record_json(record: ResearchEvidenceRecord) -> str:
    """Serialize one record canonically for hashing, persistence, and review."""

    return (
        json.dumps(
            record.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def write_research_evidence_record(path: str | Path, record: ResearchEvidenceRecord) -> None:
    """Atomically persist one canonical record without overwriting existing evidence."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_research_evidence_record_json(record)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            try:
                os.link(temporary_path, destination)
            except FileExistsError as exc:
                raise FileExistsError(f"research evidence record already exists: {destination}") from exc
            temporary_path.unlink()
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def load_research_evidence_record(path: str | Path) -> ResearchEvidenceRecord:
    """Load and strictly validate one persisted research evidence record."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("research evidence record must contain valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("research evidence record root must be a JSON object")
    return ResearchEvidenceRecord.from_dict(payload)


def verify_research_evidence_record(path: str | Path) -> ResearchEvidenceRecord:
    """Verify that every retained artifact still matches the recorded exact bytes."""

    record_path = Path(path)
    record = load_research_evidence_record(record_path)
    root = record_path.parent.resolve()
    for artifact in record.artifacts:
        artifact_path = (root / artifact.path).resolve()
        try:
            artifact_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"artifact escapes research evidence record directory: {artifact.path}") from exc
        if not artifact_path.is_file():
            raise ValueError(f"retained evidence artifact is missing: {artifact.path}")
        raw_bytes = artifact_path.read_bytes()
        if len(raw_bytes) != artifact.byte_count:
            raise ValueError(f"artifact byte count mismatch for role {artifact.role!r}")
        actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if actual_sha256 != artifact.sha256:
            raise ValueError(f"artifact SHA-256 mismatch for role {artifact.role!r}")
    return record


def _parse_artifact(payload: object) -> EvidenceArtifact:
    """Parse one exact-byte artifact identity from a persisted record."""

    if not isinstance(payload, Mapping):
        raise ValueError("research evidence artifact must be a JSON object")
    expected_keys = {"role", "path", "byte_count", "sha256"}
    if set(payload) != expected_keys:
        raise ValueError("research evidence artifact has unexpected or missing fields")
    role = _validated_non_empty_string(payload["role"], "artifact role")
    path = _validated_relative_path(payload["path"])
    byte_count = payload["byte_count"]
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
        raise ValueError("artifact byte_count must be a non-negative integer")
    sha256 = payload["sha256"]
    if not isinstance(sha256, str) or _SHA256_PATTERN.fullmatch(sha256) is None:
        raise ValueError("artifact sha256 must be a lowercase 64-character hexadecimal digest")
    return EvidenceArtifact(role=role, path=path, byte_count=byte_count, sha256=sha256)


def _validate_unique_artifacts(artifacts: Sequence[EvidenceArtifact]) -> None:
    """Reject ambiguous records with duplicate roles or paths."""

    roles = [artifact.role for artifact in artifacts]
    paths = [artifact.path for artifact in artifacts]
    if len(roles) != len(set(roles)):
        raise ValueError("research evidence artifact roles must be unique")
    if len(paths) != len(set(paths)):
        raise ValueError("research evidence artifact paths must be unique")


def _validated_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validated_evidence_level(value: object) -> str:
    level = _validated_non_empty_string(value, "evidence_level")
    if level not in EVIDENCE_LEVELS:
        raise ValueError(f"evidence_level must be one of {sorted(EVIDENCE_LEVELS)}")
    return level


def _validated_revision(value: object) -> str:
    revision = _validated_non_empty_string(value, "remem_revision")
    if _COMMIT_SHA_PATTERN.fullmatch(revision) is None:
        raise ValueError("remem_revision must be a lowercase 40-character Git commit SHA")
    return revision


def _validated_relative_path(value: object) -> str:
    raw_path = _validated_non_empty_string(value, "artifact path")
    path = Path(raw_path)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ValueError("artifact path must be a safe relative path")
    return path.as_posix()
