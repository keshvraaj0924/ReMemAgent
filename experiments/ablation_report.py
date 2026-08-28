"""Stable serialization for reproducible ablation experiment results."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.ablations import AblationResult


def results_to_records(results: list[AblationResult]) -> list[dict[str, Any]]:
    """Convert ablation results into JSON-compatible, stable records."""

    return [
        {
            **asdict(result),
            "strategy": result.strategy.value,
            "negative_transfer_rate": result.negative_transfer_rate,
        }
        for result in results
    ]


def write_results_json(results: list[AblationResult], output_path: Path) -> None:
    """Write ablation results as deterministic, human-readable JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"results": results_to_records(results)}
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
