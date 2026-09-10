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
    validate_repeated_benchmark_request,
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

    All callable, pairing, repeated-seed, and condition-label contracts are
    validated before either condition starts. Measured execution is then
    counterbalanced by seed: baseline runs first for even seed positions and
    treatment runs first for odd positions. This prevents one condition from
    always being measured later than the other while preserving identical seed
    ownership for both conditions.
    """

    _validate_paired_specs(baseline_spec, treatment_spec)
    _validate_condition_labels(baseline_label, treatment_label)
    selected_seeds = validate_seed_sequence(seeds)
    _validate_paired_repeated_requests(baseline_spec, treatment_spec, selected_seeds)
    validate_external_benchmark(baseline_spec)
    validate_external_benchmark(treatment_spec)
    baseline_reports, treatment_reports = _run_counterbalanced_pairs(
        baseline_spec,
        treatment_spec,
        selected_seeds,
    )
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

    _validate_condition_labels(baseline_label, treatment_label)
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
    """Validate both conditions completely before probing either policy.

    Paired preflight can construct one real environment per seed and condition.
    Validate repeated-seed ownership and resolve every configured callable first
    so a broken treatment specification cannot waste baseline environment/model
    setup before failing.
    """

    _validate_paired_specs(baseline_spec, treatment_spec)
    selected_seeds = validate_seed_sequence(seeds)
    _validate_paired_repeated_requests(baseline_spec, treatment_spec, selected_seeds)
    validate_external_benchmark(baseline_spec)
    validate_external_benchmark(treatment_spec)
    validate_repeated_external_benchmark_runtime(
        baseline_spec,
        selected_seeds,
        probe_action=probe_action,
    )
    validate_repeated_external_benchmark_runtime(
        treatment_spec,
        selected_seeds,
        probe_action=probe_action,
    )


def _run_counterbalanced_pairs(
    baseline_spec: ExternalBenchmarkSpec,
    treatment_spec: ExternalBenchmarkSpec,
    seeds: tuple[int, ...],
) -> tuple[tuple[BenchmarkRunReport, ...], tuple[BenchmarkRunReport, ...]]:
    """Execute matched seeds with deterministic alternating condition order.

    The output remains ordered by the caller-provided seed sequence regardless
    of which condition executes first for an individual seed. Each one-seed run
    therefore contributes exactly one report to each paired collection while
    reducing systematic warm-up, throttling, or temporal-drift bias.
    """

    baseline_reports: list[BenchmarkRunReport] = []
    treatment_reports: list[BenchmarkRunReport] = []

    for seed_index, seed in enumerate(seeds):
        if seed_index % 2 == 0:
            baseline_report = run_repeated_external_benchmarks(baseline_spec, (seed,))[0]
            treatment_report = run_repeated_external_benchmarks(treatment_spec, (seed,))[0]
        else:
            treatment_report = run_repeated_external_benchmarks(treatment_spec, (seed,))[0]
            baseline_report = run_repeated_external_benchmarks(baseline_spec, (seed,))[0]
        baseline_reports.append(baseline_report)
        treatment_reports.append(treatment_report)

    return tuple(baseline_reports), tuple(treatment_reports)


def _validate_paired_repeated_requests(
    baseline_spec: ExternalBenchmarkSpec,
    treatment_spec: ExternalBenchmarkSpec,
    seeds: tuple[int, ...],
) -> None:
    """Validate repeated-run seed ownership for both conditions before execution."""

    validate_repeated_benchmark_request(baseline_spec, seeds)
    validate_repeated_benchmark_request(treatment_spec, seeds)


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


def _validate_condition_labels(baseline_label: str, treatment_label: str) -> None:
    """Reject unusable or ambiguous condition labels before expensive execution."""

    for field_name, value in (
        ("baseline_label", baseline_label),
        ("treatment_label", treatment_label),
    ):
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")
        if not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")

    if baseline_label.strip().casefold() == treatment_label.strip().casefold():
        raise ValueError("baseline_label and treatment_label must identify distinct conditions")


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
