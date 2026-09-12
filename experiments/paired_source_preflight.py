"""Controlled paired benchmark orchestration with source-checkout provenance.

This module closes the provenance gap between source-checkout admission and paired
measurement without changing the legacy paired API. Controlled callers validate
runtime and external source state exactly once before any benchmark callable or
environment probe is allowed to run, then carry those admitted snapshots beside
the measured paired result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.paired_benchmark import (
    PairedBenchmarkResult,
    preflight_paired_external_benchmarks,
    run_paired_external_benchmarks,
)
from experiments.runtime_provenance import RuntimeProvenance, collect_runtime_provenance
from experiments.runtime_requirements import RuntimeRequirements, validate_runtime_requirements
from experiments.source_checkouts import (
    SourceCheckoutProvenance,
    SourceCheckoutRequirement,
    collect_source_checkout_provenance,
    validate_source_checkout_requirements,
)


@dataclass(frozen=True, slots=True)
class ControlledPairedPreflightResult:
    """Exact runtime and source snapshots admitted before paired measurement."""

    runtime_provenance: RuntimeProvenance
    source_checkout_provenance: Mapping[str, SourceCheckoutProvenance]


@dataclass(frozen=True, slots=True)
class ControlledPairedBenchmarkResult:
    """Measured paired result plus the exact pre-measurement admission snapshots."""

    paired_result: PairedBenchmarkResult
    runtime_provenance: RuntimeProvenance
    source_checkout_provenance: Mapping[str, SourceCheckoutProvenance]


def preflight_controlled_paired_external_benchmarks(
    baseline_spec: ExternalBenchmarkSpec,
    treatment_spec: ExternalBenchmarkSpec,
    seeds: Sequence[int],
    *,
    runtime_requirements: RuntimeRequirements,
    source_checkout_paths: Mapping[str, Path],
    source_checkout_requirements: Mapping[str, SourceCheckoutRequirement],
    probe_action: str | None = None,
) -> ControlledPairedPreflightResult:
    """Validate controlled state and both benchmark conditions without measurement.

    This is the fail-closed readiness boundary for expensive external experiments.
    Runtime and source snapshots are collected before benchmark callables are
    resolved. Only after both contracts pass do the paired environment probes run.
    The returned snapshots are therefore the exact state admitted by preflight and
    can be reused by a subsequent measured execution in the same process.
    """

    runtime_provenance = _validate_runtime_contract(runtime_requirements)
    source_checkout_provenance = _validate_source_checkout_contract(
        source_checkout_paths,
        source_checkout_requirements,
    )
    preflight_paired_external_benchmarks(
        baseline_spec,
        treatment_spec,
        seeds,
        probe_action=probe_action,
        runtime_requirements=None,
    )
    return ControlledPairedPreflightResult(
        runtime_provenance=runtime_provenance,
        source_checkout_provenance=source_checkout_provenance,
    )


def run_controlled_paired_external_benchmarks(
    baseline_spec: ExternalBenchmarkSpec,
    treatment_spec: ExternalBenchmarkSpec,
    seeds: Sequence[int],
    *,
    runtime_requirements: RuntimeRequirements,
    source_checkout_paths: Mapping[str, Path],
    source_checkout_requirements: Mapping[str, SourceCheckoutRequirement],
    baseline_label: str = "baseline",
    treatment_label: str = "treatment",
    probe_action: str | None = None,
) -> ControlledPairedBenchmarkResult:
    """Validate controlled state once, preflight both conditions, then measure.

    Runtime and source-checkout requirements are admitted before the paired
    callable/environment preflight. The exact snapshots that passed admission are
    returned with the measurement so persistence can bind evidence to the state
    that was actually validated rather than recollecting mutable Git state later.
    """

    preflight_result = preflight_controlled_paired_external_benchmarks(
        baseline_spec,
        treatment_spec,
        seeds,
        runtime_requirements=runtime_requirements,
        source_checkout_paths=source_checkout_paths,
        source_checkout_requirements=source_checkout_requirements,
        probe_action=probe_action,
    )
    paired_result = run_paired_external_benchmarks(
        baseline_spec,
        treatment_spec,
        seeds,
        baseline_label=baseline_label,
        treatment_label=treatment_label,
    )
    paired_result = replace(
        paired_result,
        runtime_provenance=preflight_result.runtime_provenance,
    )
    return ControlledPairedBenchmarkResult(
        paired_result=paired_result,
        runtime_provenance=preflight_result.runtime_provenance,
        source_checkout_provenance=preflight_result.source_checkout_provenance,
    )


def _validate_runtime_contract(requirements: RuntimeRequirements) -> RuntimeProvenance:
    """Collect and validate one runtime snapshot before paired side effects."""

    if not isinstance(requirements, RuntimeRequirements):
        raise TypeError("runtime_requirements must be a RuntimeRequirements instance")
    provenance = collect_runtime_provenance()
    validate_runtime_requirements(provenance, requirements)
    return provenance


def _validate_source_checkout_contract(
    repositories: Mapping[str, Path],
    requirements: Mapping[str, SourceCheckoutRequirement],
) -> Mapping[str, SourceCheckoutProvenance]:
    """Collect and validate external source repositories before paired probing."""

    if not isinstance(repositories, Mapping):
        raise TypeError("source_checkout_paths must be a mapping")
    if not isinstance(requirements, Mapping):
        raise TypeError("source_checkout_requirements must be a mapping")
    if not requirements:
        raise ValueError("source_checkout_requirements must not be empty")

    provenance = collect_source_checkout_provenance(repositories)
    validate_source_checkout_requirements(provenance, requirements)
    return provenance


__all__ = [
    "ControlledPairedBenchmarkResult",
    "ControlledPairedPreflightResult",
    "preflight_controlled_paired_external_benchmarks",
    "run_controlled_paired_external_benchmarks",
]
