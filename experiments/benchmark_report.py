"""Deterministic JSON persistence for measured benchmark reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from remem.benchmark import BenchmarkRunReport


def save_benchmark_report(
    report: BenchmarkRunReport,
    output_path: str | Path,
    *,
    runtime_provenance: Mapping[str, Any] | None = None,
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
        payload["runtime_provenance"] = _normalize_runtime_provenance(runtime_provenance)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _normalize_runtime_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and deterministically order structured runtime provenance."""

    normalized: dict[str, Any] = {}
    for key, value in provenance.items():
        if not isinstance(key, str):
            raise TypeError("runtime provenance keys must be strings")
        if key == "dependency_versions":
            if not isinstance(value, Mapping):
                raise TypeError("dependency_versions must be a mapping")
            dependencies: dict[str, str] = {}
            for dependency, version in value.items():
                if not isinstance(dependency, str):
                    raise TypeError("dependency names must be strings")
                if not isinstance(version, str):
                    raise TypeError("dependency versions must be strings")
                dependencies[dependency] = version
            normalized[key] = dict(sorted(dependencies.items()))
            continue
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            raise TypeError("runtime provenance values must be strings or integers")
        normalized[key] = value
    return dict(sorted(normalized.items()))


__all__ = ["save_benchmark_report"]
