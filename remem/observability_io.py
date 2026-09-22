"""Persistence helpers for validated observability snapshots."""

from __future__ import annotations

from collections.abc import Iterable
import json
import os
from pathlib import Path
import tempfile

from remem.observability import MetricSnapshot, MetricsRecorder


def save_metric_snapshot(snapshot: MetricSnapshot, path: str | Path) -> Path:
    """Atomically persist a metric snapshot as deterministic UTF-8 JSON."""
    if not isinstance(snapshot, MetricSnapshot):
        raise TypeError("snapshot must be a MetricSnapshot")
    # Round-trip through the strict loader before writing so directly constructed,
    # malformed snapshots cannot become trusted experiment artifacts.
    validated = MetricSnapshot.from_dict(snapshot.to_dict())
    destination = Path(path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"snapshot parent directory does not exist: {destination.parent}")

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
            json.dump(validated.to_dict(), temporary_file, sort_keys=True, separators=(",", ":"))
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


def load_metric_snapshot(path: str | Path) -> MetricSnapshot:
    """Load a persisted snapshot and enforce the observability schema."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("metric snapshot must contain valid JSON") from error
    if not isinstance(payload, dict):
        raise TypeError("metric snapshot JSON root must be an object")
    return MetricSnapshot.from_dict(payload)


def merge_metric_snapshots(paths: Iterable[str | Path]) -> MetricSnapshot:
    """Load and deterministically aggregate persisted worker metric snapshots.

    Duplicate paths are rejected rather than silently double-counted. Every input
    passes through :func:`load_metric_snapshot`, so malformed worker artifacts fail
    the aggregation before a combined snapshot can be reported as evidence.
    """
    if isinstance(paths, (str, Path)):
        raise TypeError("paths must be an iterable of snapshot paths, not a single path")

    normalized_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for raw_path in paths:
        if not isinstance(raw_path, (str, Path)):
            raise TypeError("each snapshot path must be a string or Path")
        path = Path(raw_path).resolve(strict=False)
        if path in seen_paths:
            raise ValueError(f"duplicate metric snapshot path: {raw_path}")
        seen_paths.add(path)
        normalized_paths.append(path)

    if not normalized_paths:
        raise ValueError("at least one metric snapshot path is required")

    recorder = MetricsRecorder()
    for path in sorted(normalized_paths, key=lambda item: str(item)):
        recorder.merge(load_metric_snapshot(path))
    return recorder.snapshot()


__all__ = ["load_metric_snapshot", "merge_metric_snapshots", "save_metric_snapshot"]
