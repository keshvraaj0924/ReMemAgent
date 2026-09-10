"""Reproducible paired execution for baseline and memory-guided policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from experiments.benchmark_statistics import (
    BenchmarkConditionComparison,
    compare_benchmark_reports,
)
from experiments.external_benchmark import (
    ExternalBenchmarkSpec,
    run_repeated_external_benchmarks,
    validate_external_benchmark,
    validate_seed_sequence,
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

    All callable and pairing contracts are validated before either condition
    starts. This prevents a malformed treatment configuration from consuming
    baseline episodes and leaving an incomplete paired experiment behind.
    """

    _validate_paired_specs(baseline_spec, treatment_spec)
    selected_seeds = validate_seed_sequence(seeds)
    validate_external_benchmark(baseline_spec)
    validate_external_benchmark(treatment_spec)
    baseline_reports = run_repeated_external_benchmarks(baseline_spec, selected_seeds)
    treatment_reports = run_repeated_external_benchmarks(treatment_spec, selected_seeds)
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


def run_paired_external_benchmarks_with_preflight(
    baseline_spec: ExternalBenchmarkSpec,
    treatment_spec: ExternalBenchmarkSpec,
    seeds: Sequence[int],
    *,
    baseline_label: str = "baseline",
    treatment_label: str = "treatment",
    probe_action: str | None = None,
) -> PairedBenchmarkResult:
    """Preflight both conditions before running a paired benchmark experiment."""

    preflight_paired_external_benchmarks(
        baseline_spec,
        treatment_spec,
        seeds,
        probe_action=probe_action,
    )
    return run_paired_external_benchmarks(
        baseline_spec,
        treatment_spec,
        seeds,
        baseline_label=baseline_label,
        treatment_label=treatment_label,
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
    """Ensure paired conditions share evaluation settings but differ in policy."""

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
            f"paired benchmark specifications must share evaluation configuration: {joined_fields}"
        )

    if _policy_identity(baseline_spec) == _policy_identity(treatment_spec):
        raise ValueError("paired benchmark specifications must use distinct policy configurations")


def _policy_identity(spec: ExternalBenchmarkSpec) -> tuple[str, str]:
    """Return the configured policy boundary and callable identity for one condition."""

    if spec.policy_factory is not None:
        return ("policy_factory", spec.policy_factory)
    if spec.action_policy_factory is not None:
        return ("action_policy_factory", spec.action_policy_factory)
    raise ValueError("one of policy_factory or action_policy_factory is required")


__all__ = [
    "PairedBenchmarkResult",
    "preflight_paired_external_benchmarks",
    "run_paired_external_benchmarks",
    "run_paired_external_benchmarks_with_preflight",
]
