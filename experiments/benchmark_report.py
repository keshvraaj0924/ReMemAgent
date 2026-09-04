from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

from remem.benchmark import BenchmarkRunConfiguration, BenchmarkRunReport
from remem.benchmark_validation import validate_benchmark_run_report

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
    """Return a deterministic fingerprint for configuration independent of seed.

    The seed identifies an independent stochastic run and therefore is excluded
    from this fingerprint. All other declared configuration fields participate
    in the digest, making the value useful for grouping comparable runs.
    """

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
    runtime_provenance: Mapping[str, str] | None = None,
) -> Path:
    """Persist a structurally valid benchmark report with optional provenance."""

    payload = benchmark_report_to_dict(report)
    if runtime_provenance is not None:
        payload["runtime_provenance"] = dict(runtime_provenance)
    _write_json(payload, output_path)
    return output_path


def save_repeated_benchmark_reports(
    reports: tuple[BenchmarkRunReport, ...] | list[BenchmarkRunReport],
    output_path: Path,
    *,
    runtime_provenance: Mapping[str, str] | None = None,
    statistics: Mapping[str, Any] | None = None,
) -> Path:
    """Persist independent seed reports and optional descriptive statistics.

    Repeated reports must describe the same experimental configuration apart
    from their independent seed. Reports are serialized in ascending seed order
    so artifact bytes do not depend on caller iteration order.
    """

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
    ordered_reports = tuple(sorted(selected_reports, key=lambda report: report.seed))

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
        payload["runtime_provenance"] = dict(runtime_provenance)
    if statistics is not None:
        payload["statistics"] = dict(statistics)
    _write_json(payload, output_path)
    return output_path


def _validate_repeated_configuration(reports: tuple[BenchmarkRunReport, ...]) -> None:
    """Ensure repeated reports share all configuration except their seed."""

    configurations = tuple(report.configuration for report in reports)
    if all(configuration is None for configuration in configurations):
        return
    if any(configuration is None for configuration in configurations):
        raise ValueError("repeated benchmark reports must use consistent configuration metadata")

    reference = _without_seed(configurations[0])
    if any(_without_seed(configuration) != reference for configuration in configurations[1:]):
        raise ValueError(
            "repeated benchmark reports must share configuration apart from the seed"
        )


def _without_seed(
    configuration: BenchmarkRunConfiguration,
) -> BenchmarkRunConfiguration:
    """Return configuration with the independent-run seed removed."""

    return replace(configuration, seed=None)


def _write_json(payload: Mapping[str, Any], output_path: Path) -> None:
    """Atomically write deterministic JSON, replacing the destination only on success."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        _fsync_directory(output_path.parent)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(directory: Path) -> None:
    """Flush directory metadata when supported by the current platform."""

    try:
        directory_descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
