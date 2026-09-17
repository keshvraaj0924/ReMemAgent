"""Deterministic JSON persistence for measured benchmark reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.runtime_provenance import RuntimeProvenance
from remem.benchmark import BenchmarkRunReport


def save_benchmark_report(
    report: BenchmarkRunReport,
    output_path: str | Path,
    *,
    runtime_provenance: RuntimeProvenance | Mapping[str, Any] | None = None,
) -> Path:
    """Persist a benchmark report without inventing or modifying measurements."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    payload.update(
        {
            "success_rate": report.success_rate,
            "mean_reward": report.mean_reward,
            "transfer_success_rate": report.transfer_success_rate,
        }
    )
    if runtime_provenance is not None:
        payload["runtime_provenance"] = _verified_runtime_provenance(runtime_provenance)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _verified_runtime_provenance(
    provenance: RuntimeProvenance | Mapping[str, Any],
) -> dict[str, object]:
    """Return deterministic provenance only after full integrity validation."""
    if isinstance(provenance, RuntimeProvenance):
        return provenance.to_dict()
    return RuntimeProvenance.from_dict(provenance).to_dict()


__all__ = ["save_benchmark_report"]
