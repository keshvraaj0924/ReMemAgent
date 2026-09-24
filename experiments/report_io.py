"""Shared deterministic persistence helpers for experiment reports."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def atomic_write_json(destination: str | Path, payload: Mapping[str, Any]) -> Path:
    """Persist a JSON mapping deterministically, atomically, and durably.

    The completed temporary file is flushed and fsynced before replacement so
    readers never observe a partially written report. On POSIX systems the
    parent directory is also fsynced after replacement so the rename itself is
    durable across a crash.
    """

    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        _fsync_directory(output_path.parent)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def _fsync_directory(directory: Path) -> None:
    """Flush directory metadata on POSIX after an atomic replacement."""

    if os.name != "posix":
        return

    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["atomic_write_json"]
