"""Serialization and validation helpers for benchmark research artifacts.

The artifact boundary keeps benchmark execution separate from persistence. A
serialized run carries the exact configuration manifest used to identify the
run, while validation re-checks that identity before an artifact is accepted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from remem.benchmark import BenchmarkRunConfiguration, BenchmarkRunReport
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


def validate_serialized_benchmark_run_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate configuration identity using only a persisted run payload.

    This boundary is intentionally narrower than full report reconstruction. It
    proves that the serialized configuration and its deterministic manifest
    agree, allowing persisted artifacts to be checked without executing a
    benchmark or trusting a separately supplied in-memory report.
    """

    configuration_payload = artifact.get("configuration")
    if configuration_payload is None and "configuration_manifest" not in artifact:
        return
    if not isinstance(configuration_payload, Mapping):
        raise ValueError("benchmark artifact configuration must be a mapping")

    try:
        configuration = BenchmarkRunConfiguration(**dict(configuration_payload))
    except (TypeError, ValueError) as exc:
        raise ValueError("benchmark artifact contains invalid configuration provenance") from exc

    manifest = artifact.get("configuration_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("benchmark artifact is missing configuration_manifest")
    validate_benchmark_configuration_manifest(manifest, configuration)


def validate_persisted_benchmark_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate configuration manifests throughout a persisted benchmark artifact.

    Single-run artifacts store a run directly. Repeated and paired artifacts
    contain run payloads in deterministic collections. Legacy artifacts that do
    not contain configuration provenance remain byte-verifiable but cannot gain
    configuration-identity guarantees from this validator.
    """

    repeated_reports = artifact.get("reports")
    if repeated_reports is not None:
        _validate_report_sequence(repeated_reports, "reports")
        return

    baseline = artifact.get("baseline")
    treatment = artifact.get("treatment")
    if baseline is not None or treatment is not None:
        _validate_condition_reports(baseline, "baseline")
        _validate_condition_reports(treatment, "treatment")
        return

    validate_serialized_benchmark_run_artifact(artifact)


def _validate_condition_reports(condition: object, condition_name: str) -> None:
    """Validate one persisted paired-condition report collection."""

    if not isinstance(condition, Mapping):
        raise ValueError(f"benchmark artifact {condition_name} must be a mapping")
    _validate_report_sequence(condition.get("reports"), f"{condition_name}.reports")


def _validate_report_sequence(reports: object, field_name: str) -> None:
    """Validate each run payload in a persisted report collection."""

    if not isinstance(reports, Sequence) or isinstance(reports, (str, bytes, bytearray)):
        raise ValueError(f"benchmark artifact {field_name} must be a sequence")
    if not reports:
        raise ValueError(f"benchmark artifact {field_name} must not be empty")
    for index, report in enumerate(reports):
        if not isinstance(report, Mapping):
            raise ValueError(f"benchmark artifact {field_name}[{index}] must be a mapping")
        validate_serialized_benchmark_run_artifact(report)


__all__ = [
    "benchmark_run_artifact",
    "validate_benchmark_run_artifact",
    "validate_persisted_benchmark_artifact",
    "validate_serialized_benchmark_run_artifact",
]
