"""Controlled external benchmark execution with admission provenance.

This module preserves the exact runtime and source-checkout snapshots that pass
controlled admission before an external benchmark is probed or measured. It is
kept separate from the legacy preflight API so existing callers retain their
current return types while reproducible callers can persist the admitted state
without recollecting mutable repository metadata after measurement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from experiments.external_benchmark import (
    ExternalBenchmarkSpec,
    run_external_benchmark,
    run_repeated_external_benchmarks,
    validate_external_benchmark_runtime,
    validate_repeated_benchmark_request,
)
from experiments.external_preflight import validate_repeated_external_benchmark_runtime
from experiments.runtime_provenance import RuntimeProvenance, collect_runtime_provenance
from experiments.runtime_requirements import RuntimeRequirements, validate_runtime_requirements
from experiments.source_checkouts import (
    SourceCheckoutProvenance,
    SourceCheckoutRequirement,
    collect_source_checkout_provenance,
    validate_source_checkout_requirements,
)
from remem.benchmark import BenchmarkRunReport, BenchmarkSuiteRunner


@dataclass(frozen=True, slots=True)
class ControlledExternalBenchmarkResult:
    """One measured report plus the exact snapshots admitted before measurement."""

    report: BenchmarkRunReport
    runtime_provenance: RuntimeProvenance
    source_checkout_provenance: Mapping[str, SourceCheckoutProvenance]


@dataclass(frozen=True, slots=True)
class ControlledRepeatedExternalBenchmarkResult:
    """Repeated measured reports plus one shared pre-measurement admission state."""

    reports: tuple[BenchmarkRunReport, ...]
    runtime_provenance: RuntimeProvenance
    source_checkout_provenance: Mapping[str, SourceCheckoutProvenance]


def run_controlled_external_benchmark(
    spec: ExternalBenchmarkSpec,
    *,
    runtime_requirements: RuntimeRequirements,
    source_checkout_paths: Mapping[str, Path],
    source_checkout_requirements: Mapping[str, SourceCheckoutRequirement],
    probe_action: str | None = None,
    runner: BenchmarkSuiteRunner | None = None,
) -> ControlledExternalBenchmarkResult:
    """Admit runtime/source state, preflight the environment, then measure once.

    Admission occurs before callable resolution and environment construction. The
    returned snapshots are the exact objects that passed validation, allowing
    persistence to bind evidence to pre-measurement state without a second Git or
    package metadata collection after the benchmark has run.
    """

    runtime_provenance = _validate_runtime_contract(runtime_requirements)
    source_checkout_provenance = _validate_source_checkout_contract(
        source_checkout_paths,
        source_checkout_requirements,
    )
    validate_external_benchmark_runtime(spec, probe_action=probe_action)
    report = run_external_benchmark(spec, runner=runner)
    return ControlledExternalBenchmarkResult(
        report=report,
        runtime_provenance=runtime_provenance,
        source_checkout_provenance=source_checkout_provenance,
    )


def run_controlled_repeated_external_benchmarks(
    spec: ExternalBenchmarkSpec,
    seeds: Sequence[int],
    *,
    runtime_requirements: RuntimeRequirements,
    source_checkout_paths: Mapping[str, Path],
    source_checkout_requirements: Mapping[str, SourceCheckoutRequirement],
    probe_action: str | None = None,
    runner: BenchmarkSuiteRunner | None = None,
) -> ControlledRepeatedExternalBenchmarkResult:
    """Admit shared state once, preflight every seed, then measure every seed.

    Runtime and source admission are intentionally performed once because all
    repeated runs belong to one experiment execution. Environment probes remain
    seed-isolated and happen only after admission succeeds.
    """

    selected_seeds = validate_repeated_benchmark_request(spec, seeds)
    runtime_provenance = _validate_runtime_contract(runtime_requirements)
    source_checkout_provenance = _validate_source_checkout_contract(
        source_checkout_paths,
        source_checkout_requirements,
    )
    validate_repeated_external_benchmark_runtime(
        spec,
        selected_seeds,
        probe_action=probe_action,
    )
    reports = run_repeated_external_benchmarks(spec, selected_seeds, runner=runner)
    return ControlledRepeatedExternalBenchmarkResult(
        reports=reports,
        runtime_provenance=runtime_provenance,
        source_checkout_provenance=source_checkout_provenance,
    )


def _validate_runtime_contract(requirements: RuntimeRequirements) -> RuntimeProvenance:
    """Collect and validate one runtime snapshot before external side effects."""

    if not isinstance(requirements, RuntimeRequirements):
        raise TypeError("runtime_requirements must be a RuntimeRequirements instance")
    provenance = collect_runtime_provenance()
    validate_runtime_requirements(provenance, requirements)
    return provenance


def _validate_source_checkout_contract(
    repositories: Mapping[str, Path],
    requirements: Mapping[str, SourceCheckoutRequirement],
) -> Mapping[str, SourceCheckoutProvenance]:
    """Collect and validate one immutable snapshot of external source checkouts."""

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
    "ControlledExternalBenchmarkResult",
    "ControlledRepeatedExternalBenchmarkResult",
    "run_controlled_external_benchmark",
    "run_controlled_repeated_external_benchmarks",
]
