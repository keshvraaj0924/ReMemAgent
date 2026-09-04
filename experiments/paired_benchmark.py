"""Reproducible paired execution for baseline and memory-guided policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from experiments.benchmark_statistics import (
    BenchmarkConditionComparison,
    compare_benchmark_reports,
)
from experiments.external_benchmark import (
    ExternalBenchmarkSpec,
    run_repeated_external_benchmarks,
)
from experiments.external_preflight import validate_repeated_external_benchmark_runtime
from remem.benchmark import BenchmarkRunReport


@dataclass(frozen=True, slots=True)
class PairedBenchmarkResult:
    """Measured reports and paired descriptive comparison for two conditions."""

    baseline_reports: tuple[BenchmarkRunReport, ...]
    treatment_reports: tuple[BenchmarkRunReport, ...]
    comparison: BenchmarkConditionComparison


def run_paired_external_benchmarks(
    baseline_spec: ExternalBenchmarkSpec,
    treatment_spec: ExternalBenchmarkSpec,
    seeds: Sequence[int],
    *,
    baseline_label: str = "baseline",
    treatment_label: str = "treatment",
) -> PairedBenchmarkResult:
    """Run two policy conditions on the same independent seed set.

    Only the policy specification may differ between conditions. Environment,
    evaluation, episode-count, step-limit, and trust configuration are held
    fixed so the resulting reports form a valid paired experimental design.
    """

    _validate_paired_specs(baseline_spec, treatment_spec)
    baseline_reports = run_repeated_external_benchmarks(baseline_spec, seeds)
    treatment_reports = run_repeated_external_benchmarks(treatment_spec, seeds)
    comparison = compare_benchmark_reports(
        baseline_reports,
        treatment_reports,
        baseline_label=baseline_label,
        treatment_label=treatment_label,
    )
    return PairedBenchmarkResult(
        baseline_reports=baseline_reports,
        treatment_reports=treatment_reports,
        comparison=comparison,
    )


def preflight_paired_external_benchmarks(
    baseline_spec: ExternalBenchmarkSpec,
    treatment_spec: ExternalBenchmarkSpec,
    seeds: Sequence[int],
    *,
    probe_action: str | None = None,
) -> None:
    """Preflight both policy conditions independently before measured execution."""

    _validate_paired_specs(baseline_spec, treatment_spec)
    validate_repeated_external_benchmark_runtime(
        baseline_spec,
        seeds,
        probe_action=probe_action,
    )
    validate_repeated_external_benchmark_runtime(
        treatment_spec,
        seeds,
        probe_action=probe_action,
    )


def _validate_paired_specs(
    baseline_spec: ExternalBenchmarkSpec,
    treatment_spec: ExternalBenchmarkSpec,
) -> None:
    """Ensure paired conditions differ only in policy implementation."""

    fields = (
        "benchmark_name",
        "episode_count",
        "max_steps",
        "environment_factory",
        "success_evaluator",
        "transfer_success_evaluator",
        "minimum_trust",
    )
    mismatches = [
        field_name
        for field_name in fields
        if getattr(baseline_spec, field_name) != getattr(treatment_spec, field_name)
    ]
    if mismatches:
        joined_fields = ", ".join(mismatches)
        raise ValueError(
            "paired benchmark specifications must share evaluation configuration: "
            f"{joined_fields}"
        )


__all__ = [
    "PairedBenchmarkResult",
    "preflight_paired_external_benchmarks",
    "run_paired_external_benchmarks",
]
