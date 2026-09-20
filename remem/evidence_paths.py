"""Canonical path validation shared by experiment evidence components."""

from __future__ import annotations

from pathlib import Path, PurePosixPath


def normalize_evidence_path(path: str | Path) -> str:
    """Return a canonical run-relative POSIX path or reject ambiguous input."""
    if not isinstance(path, (str, Path)):
        raise TypeError("evidence paths must be strings or Path values")
    raw_path = str(path)
    if not raw_path or "\\" in raw_path:
        raise ValueError("evidence paths must use non-empty POSIX-style paths")
    normalized_path = PurePosixPath(raw_path)
    if normalized_path.is_absolute() or any(
        part in {"", ".", ".."} for part in normalized_path.parts
    ):
        raise ValueError("evidence paths must be normalized run-relative paths")
    if normalized_path.as_posix() != raw_path:
        raise ValueError("evidence paths must be normalized run-relative paths")
    return raw_path


__all__ = ["normalize_evidence_path"]
