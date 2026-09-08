from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from experiments.experiment_identity import build_experiment_identity
from remem.benchmark import BenchmarkRunConfiguration, BenchmarkRunReport
from remem.benchmark_validation import validate_benchmark_run_report

if TYPE_CHECKING:
    from experiments.benchmark_statistics import BenchmarkConditionComparison

BENCHMARK_REPORT_SCHEMA_VERSION = 1


def benchmark_report_to_dict(report: BenchmarkRunReport) -> dict[str, Any]:
    """Convert a structurally valid benchmark report into JSON-safe data."""

    validate_benchmark_run_report(report)
    payload = asdict(report)
    payload["schema_version"] = BENCHMARK_REPORT_SCHEMA_VERSION
    for episode in payload["episodes"]:
        for step in episode["episode"]["steps"]:
            step["result"].pop("info", None)
    configuration = report.configuration
    if configuration is not None:
        payload["configuration_fingerprint"] = benchmark_configuration_fingerprint(configuration)
    return payload


def benchmark_configuration_fingerprint(configuration: BenchmarkRunConfiguration) -> str:
    """Return a deterministic fingerprint for configuration independent of seed."""

    canonical_configuration = asdict(replace(configuration, seed=None))
    canonical_payload = json.dumps(
        canonical_configuration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical_payload).hexdigest()


def save_benchmark_report(
    report: BenchmarkRunReport,
    output_path: Path,
    *,
    runtime_provenance: Mapping[str, object] | None = None,
) -> Path:
    """Persist a structurally valid benchmark report with optional provenance."""

    payload = benchmark_report_to_dict(report)
    if runtime_provenance is not None:
        normalized_provenance = _normalize_runtime_provenance(runtime_provenance)
        payload["runtime_provenance"] = normalized_provenance
        if report.configuration is not None and report.seed is not None:
            payload["experiment_identity"] = build_experiment_identity(
                report.configuration,
                (report.seed,),
                normalized_provenance,
            )
    _write_json(payload, output_path)
    return output_path


def save_repeated_benchmark_reports(
    reports: tuple[BenchmarkRunReport, ...] | list[BenchmarkRunReport],
    output_path: Path,
    *,
    runtime_provenance: Mapping[str, object] | None = None,
    statistics: Mapping[str, Any] | None = None,
) -> Path:
    """Persist independent seed reports and optional descriptive statistics."""

    selected_reports = tuple(reports)
    if not selected_reports:
        raise ValueError("reports must contain at least one benchmark report")

    for report in selected_reports:
        validate_benchmark_run_report(report)

    seeds = tuple(report.seed for report in selected_reports)
    if any(seed is None for seed in seeds):
        raise ValueError("repeated benchmark reports require an explicit seed for every run")
    if len(seeds) != len(set(seeds)):
        raise ValueError("benchmark report seeds must be unique")

    benchmark_names = {report.benchmark_name for report in selected_reports}
    if len(benchmark_names) != 1:
        raise ValueError("repeated benchmark reports must use one benchmark name")

    _validate_repeated_configuration(selected_reports)
    ordered_reports = tuple(sorted(selected_reports, key=_seed_sort_key))
    if statistics is not None:
        _validate_repeated_statistics(ordered_reports, statistics)

    payload: dict[str, Any] = {
        "schema_version": BENCHMARK_REPORT_SCHEMA_VERSION,
        "benchmark_name": ordered_reports[0].benchmark_name,
        "seeds": [report.seed for report in ordered_reports],
        "reports": [benchmark_report_to_dict(report) for report in ordered_reports],
    }
    reference_configuration = ordered_reports[0].configuration
    if reference_configuration is not None:
        payload["configuration_fingerprint"] = benchmark_configuration_fingerprint(
            reference_configuration
        )
    if runtime_provenance is not None:
        normalized_provenance = _normalize_runtime_provenance(runtime_provenance)
        payload["runtime_provenance"] = normalized_provenance
        if reference_configuration is not None:
            identity_configuration = replace(reference_configuration, seed=None)
            payload["experiment_identity"] = build_experiment_identity(
                identity_configuration,
                tuple(report.seed for report in ordered_reports if report.seed is not None),
                normalized_provenance,
            )
    if statistics is not None:
        payload["statistics"] = dict(statistics)
    _write_json(payload, output_path)
    return output_path


def save_paired_benchmark_result(
    baseline_reports: tuple[BenchmarkRunReport, ...] | list[BenchmarkRunReport],
    treatment_reports: tuple[BenchmarkRunReport, ...] | list[BenchmarkRunReport],
    comparison: BenchmarkConditionComparison,
    output_path: Path,
    *,
    runtime_provenance: Mapping[str, object] | None = None,
) -> Path:
    """Persist paired condition reports and their descriptive comparison."""

    baseline = _validate_paired_report_collection(baseline_reports, "baseline")
    treatment = _validate_paired_report_collection(treatment_reports, "treatment")
    if baseline[0].benchmark_name != treatment[0].benchmark_name:
        raise ValueError("paired benchmark reports must use one benchmark name")
    _validate_paired_configuration(baseline, treatment)

    baseline_seeds = tuple(report.seed for report in baseline)
    treatment_seeds = tuple(report.seed for report in treatment)
    if baseline_seeds != tuple(comparison.seeds) or treatment_seeds != tuple(comparison.seeds):
        raise ValueError("comparison seeds must match both paired report seed sets")
    _validate_paired_comparison(baseline, treatment, comparison)

    payload: dict[str, Any] = {
        "schema_version": BENCHMARK_REPORT_SCHEMA_VERSION,
        "benchmark_name": baseline[0].benchmark_name,
        "seeds": list(comparison.seeds),
        "baseline": {
            "label": comparison.baseline_label,
            "reports": [benchmark_report_to_dict(report) for report in baseline],
        },
        "treatment": {
            "label": comparison.treatment_label,
            "reports": [benchmark_report_to_dict(report) for report in treatment],
        },
        "comparison": comparison.to_dict(),
    }
    reference_configuration = baseline[0].configuration
    if reference_configuration is not None:
        payload["configuration_fingerprint"] = benchmark_configuration_fingerprint(
            reference_configuration
        )
    if runtime_provenance is not None:
        payload["runtime_provenance"] = _normalize_runtime_provenance(runtime_provenance)
    _write_json(payload, output_path)
    return output_path


def _validate_paired_report_collection(
    reports: tuple[BenchmarkRunReport, ...] | list[BenchmarkRunReport],
    condition_label: str,
) -> tuple[BenchmarkRunReport, ...]:
    """Validate and deterministically order one condition's seed reports."""

    selected_reports = tuple(reports)
    if not selected_reports:
        raise ValueError(f"{condition_label} reports must contain at least one benchmark report")
    for report in selected_reports:
        validate_benchmark_run_report(report)
    seeds = tuple(report.seed for report in selected_reports)
    if any(seed is None for seed in seeds):
        raise ValueError(f"{condition_label} reports require explicit seeds")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{condition_label} report seeds must be unique")
    benchmark_names = {report.benchmark_name for report in selected_reports}
    if len(benchmark_names) != 1:
        raise ValueError(f"{condition_label} reports must use one benchmark name")
    return tuple(sorted(selected_reports, key=_seed_sort_key))


def _validate_paired_configuration(
    baseline: tuple[BenchmarkRunReport, ...],
    treatment: tuple[BenchmarkRunReport, ...],
) -> None:
    """Ensure paired conditions share protocol configuration apart from policy."""

    if any(report.configuration is None for report in baseline + treatment):
        raise ValueError("paired benchmark reports must include explicit configuration")

    baseline_fingerprints = {_paired_configuration_fingerprint(report) for report in baseline}
    treatment_fingerprints = {_paired_configuration_fingerprint(report) for report in treatment}
    if baseline_fingerprints != treatment_fingerprints:
        raise ValueError(
            "baseline and treatment reports must share configuration apart from the seed and policy"
        )


def _paired_configuration_fingerprint(report: BenchmarkRunReport) -> str:
    """Fingerprint paired protocol metadata while excluding seed and policy identity."""

    configuration = report.configuration
    if configuration is None:
        raise ValueError("paired benchmark reports must include explicit configuration")
    return benchmark_configuration_fingerprint(
        replace(configuration, seed=None, policy_factory=None)
    )


def _validate_paired_comparison(
    baseline: tuple[BenchmarkRunReport, ...],
    treatment: tuple[BenchmarkRunReport, ...],
    comparison: BenchmarkConditionComparison,
) -> None:
    """Reject a paired comparison that does not describe the supplied reports."""

    from experiments.benchmark_statistics import compare_benchmark_reports

    expected = compare_benchmark_reports(
        baseline,
        treatment,
        baseline_label=comparison.baseline_label,
        treatment_label=comparison.treatment_label,
    )
    if comparison.to_dict() != expected.to_dict():
        raise ValueError("comparison must exactly match the supplied paired benchmark reports")


def _validate_repeated_statistics(
    reports: tuple[BenchmarkRunReport, ...],
    statistics: Mapping[str, Any],
) -> None:
    """Reject aggregate statistics that do not describe the supplied reports."""

    if not isinstance(statistics, Mapping):
        raise TypeError("statistics must be a mapping")
    from experiments.benchmark_statistics import summarize_benchmark_reports

    expected = summarize_benchmark_reports(reports).to_dict()
    actual = dict(statistics)
    if actual != expected:
        raise ValueError("statistics must exactly match the supplied benchmark reports")


def _normalize_runtime_provenance(
    runtime_provenance: Mapping[str, object],
) -> dict[str, object]:
    """Validate and detach the structured runtime provenance schema."""

    if not isinstance(runtime_provenance, Mapping):
        raise TypeError("runtime_provenance must be a mapping")

    normalized: dict[str, object] = {}
    for key, value in runtime_provenance.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("runtime_provenance keys must be non-empty strings")
        if key == "schema_version":
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError("runtime_provenance.schema_version must be an integer")
            if value <= 0:
                raise ValueError("runtime_provenance.schema_version must be positive")
            normalized[key] = value
            continue
        if not isinstance(value, str):
            raise TypeError("runtime_provenance values must be strings")
        normalized[key] = value
    return normalized


def _seed_sort_key(report: BenchmarkRunReport) -> int:
    """Return a total-order key for explicitly seeded reports."""

    if report.seed is None:
        raise ValueError("benchmark report seed must be explicit")
    return report.seed


def _write_json(payload: Mapping[str, Any], output_path: Path) -> None:
    """Atomically write deterministic JSON to the requested path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
