"""Serialization and validation helpers for benchmark research artifacts.

The artifact boundary keeps benchmark execution separate from persistence. A
serialized run carries the exact configuration manifest used to identify the
run, while validation re-checks that identity before an artifact is accepted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from remem.benchmark import BenchmarkRunReport
from remem.reproducibility_manifest import (
    benchmark_configuration_manifest,
    validate_benchmark_configuration_manifest,
)


def benchmark_run_artifact(report: BenchmarkRunReport) -> dict[str, Any]:
    """Build a JSON-compatible benchmark artifact with its configuration manifest."""

    if report.configuration is None:
        raise ValueError("benchmark report must contain configuration provenance")

    artifact = asdict(report)
    artifact["configuration_manifest"] = benchmark_configuration_manifest(report.configuration)
    return artifact


def validate_benchmark_run_artifact(
    artifact: Mapping[str, Any],
    report: BenchmarkRunReport,
) -> None:
    """Validate the configuration identity embedded in a benchmark artifact."""

    if report.configuration is None:
        raise ValueError("benchmark report must contain configuration provenance")
    manifest = artifact.get("configuration_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("benchmark artifact is missing configuration_manifest")
    validate_benchmark_configuration_manifest(manifest, report.configuration)


__all__ = ["benchmark_run_artifact", "validate_benchmark_run_artifact"]
