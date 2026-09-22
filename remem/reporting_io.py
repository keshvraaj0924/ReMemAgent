"""Persistence helpers for validated experiment summaries."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from remem.reporting import ExperimentSummary


def save_experiment_summary(summary: ExperimentSummary, path: str | Path) -> Path:
    """Atomically persist a validated experiment summary as deterministic JSON."""
    if not isinstance(summary, ExperimentSummary):
        raise TypeError("summary must be an ExperimentSummary")

    validated = ExperimentSummary.from_dict(summary.to_dict())
    destination = Path(path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"summary parent directory does not exist: {destination.parent}"
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            json.dump(
                validated.to_dict(),
                temporary_file,
                sort_keys=True,
                separators=(",", ":"),
            )
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def load_experiment_summary(path: str | Path) -> ExperimentSummary:
    """Load a persisted experiment summary and enforce its reporting schema."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("experiment summary must contain valid JSON") from error
    if not isinstance(payload, dict):
        raise TypeError("experiment summary JSON root must be an object")
    return ExperimentSummary.from_dict(payload)


__all__ = ["load_experiment_summary", "save_experiment_summary"]
