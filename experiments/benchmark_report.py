"""Deterministic JSON persistence for measured benchmark reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.report_io import atomic_write_json
from experiments.runtime_provenance import RuntimeProvenance
from remem.benchmark import BenchmarkRunReport


def save_benchmark_report(
    report: BenchmarkRunReport,
    output_path: str | Path,
    *,
    runtime_provenance: RuntimeProvenance | Mapping[str, Any] | None = None,
) -> Path:
    """Persist a benchmark report atomically without inventing measurements."""

    destination = Path(output_path)
    payload = asdict(report)
    payload.update(
        {
            "success_rate": report.success_rate,
            "mean_reward": report.mean_reward,
            "transfer_success_rate": report.transfer_success_rate,
        }
    )
    if runtime_provenance is not None:
        verified_provenance = _verified_runtime_provenance(runtime_provenance)
        payload["runtime_provenance"] = verified_provenance.to_dict()
        payload["runtime_provenance_fingerprint"] = verified_provenance.fingerprint()
    return atomic_write_json(destination, payload)


def _verified_runtime_provenance(
    provenance: RuntimeProvenance | Mapping[str, Any],
) -> RuntimeProvenance:
    """Return runtime provenance only after full integrity validation."""
    if isinstance(provenance, RuntimeProvenance):
        return provenance
    return RuntimeProvenance.from_dict(provenance)


__all__ = ["save_benchmark_report"]
