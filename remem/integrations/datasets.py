"""Deterministic offline dataset writers for training integrations.

The writer keeps filesystem I/O outside the trajectory and training contracts:
callers construct validated batches first, then explicitly persist those rows
as JSON Lines for a framework-specific ingestion step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from remem.integrations.grpo import GrpoBatch
from remem.integrations.verl import VerlTrainingBatch


def write_grpo_jsonl(batch: GrpoBatch, output_path: str | Path) -> None:
    """Write an ordered GRPO batch as one JSON object per line.

    Parent directories are created when necessary. The output is written with
    UTF-8 encoding and deterministic JSON key ordering so identical batches
    produce byte-identical artifacts.
    """

    rows = batch.to_dicts()
    _write_jsonl(rows, output_path)


def write_verl_jsonl(batch: VerlTrainingBatch, output_path: str | Path) -> None:
    """Write an ordered verl training batch as deterministic JSON Lines."""

    rows = batch.to_dicts()
    _write_jsonl(rows, output_path)


def _write_jsonl(rows: tuple[Mapping[str, object], ...], output_path: str | Path) -> None:
    """Persist validated rows without changing their order or values."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
