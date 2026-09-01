"""Serialize measured benchmark reports without coupling to benchmark SDKs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from remem.benchmark import BenchmarkRunReport


def benchmark_report_to_dict(report: BenchmarkRunReport) -> dict[str, Any]:
    """Convert a benchmark report into a JSON-safe core trajectory representation."""

    payload = asdict(report)
    for episode in payload["episodes"]:
        for step in episode["episode"]["steps"]:
            step["result"].pop("info", None)
    return payload


def save_benchmark_report(report: BenchmarkRunReport, output_path: Path) -> Path:
    """Persist a measured benchmark report as deterministic, indented JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            benchmark_report_to_dict(report),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def save_repeated_benchmark_reports(
    reports: tuple[BenchmarkRunReport, ...] | list[BenchmarkRunReport],
    output_path: Path,
) -> Path:
    """Persist independent seed reports while retaining per-run provenance."""

    selected_reports = tuple(reports)
    if not selected_reports:
        raise ValueError("reports must contain at least one benchmark report")

    seeds = tuple(report.seed for report in selected_reports)
    if len(seeds) != len(set(seeds)):
        raise ValueError("benchmark report seeds must be unique")

    benchmark_names = {report.benchmark_name for report in selected_reports}
    if len(benchmark_names) != 1:
        raise ValueError("repeated benchmark reports must use one benchmark name")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_name": selected_reports[0].benchmark_name,
        "seeds": list(seeds),
        "reports": [benchmark_report_to_dict(report) for report in selected_reports],
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path
