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
from types import MappingProxyType

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

_EMPTY_SOURCE_PROVENANCE: Mapping[str, SourceCheckoutProvenance] = MappingProxyType({})


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
    runtime_requirements: RuntimeRequirements | None = None,
    source_checkout_paths: Mapping[str, Path] | None = None,
    source_checkout_requirements: Mapping[str, SourceCheckoutRequirement] | None = None,
    probe_action: str | None = None,
    runner: BenchmarkSuiteRunner | None = None,
) -> ControlledExternalBenchmarkResult:
    """Admit runtime/source state, preflight the environment, then measure once.

    Runtime provenance is always collected before benchmark side effects so the
    exact admitted snapshot can be identity-bound during persistence. Runtime
    requirements and source-checkout requirements are optional independent
    controls; source paths and requirements must be supplied together.
    """

    runtime_provenance = _validate_runtime_contract(runtime_requirements)
    source_checkout_provenance = _validate_optional_source_checkout_contract(
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
    runtime_requirements: RuntimeRequirements | None = None,
    source_checkout_paths: Mapping[str, Path] | None = None,
    source_checkout_requirements: Mapping[str, SourceCheckoutRequirement] | None = None,
    probe_action: str | None = None,
    runner: BenchmarkSuiteRunner | None = None,
) -> ControlledRepeatedExternalBenchmarkResult:
    """Admit shared state once, preflight every seed, then measure every seed.

    Runtime and optional source admission are intentionally performed once because
    all repeated runs belong to one experiment execution. Environment probes
    remain seed-isolated and happen only after admission succeeds.
    """

    selected_seeds = validate_repeated_benchmark_request(spec, seeds)
    runtime_provenance = _validate_runtime_contract(runtime_requirements)
    source_checkout_provenance = _validate_optional_source_checkout_contract(
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


def _validate_runtime_contract(
    requirements: RuntimeRequirements | None,
) -> RuntimeProvenance:
    """Collect one runtime snapshot and optionally validate its declared contract."""

    if requirements is not None and not isinstance(requirements, RuntimeRequirements):
        raise TypeError("runtime_requirements must be a RuntimeRequirements instance or None")
    provenance = collect_runtime_provenance()
    if requirements is not None:
        validate_runtime_requirements(provenance, requirements)
    return provenance


def _validate_optional_source_checkout_contract(
    repositories: Mapping[str, Path] | None,
    requirements: Mapping[str, SourceCheckoutRequirement] | None,
) -> Mapping[str, SourceCheckoutProvenance]:
    """Collect and validate source state when a complete source contract is supplied."""

    if repositories is None and requirements is None:
        return _EMPTY_SOURCE_PROVENANCE
    if repositories is None or requirements is None:
        raise ValueError(
            "source_checkout_paths and source_checkout_requirements must be provided together"
        )
    if not isinstance(repositories, Mapping):
        raise TypeError("source_checkout_paths must be a mapping")
    if not isinstance(requirements, Mapping):
        raise TypeError("source_checkout_requirements must be a mapping")
    if not repositories:
        raise ValueError("source_checkout_paths must not be empty")
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
