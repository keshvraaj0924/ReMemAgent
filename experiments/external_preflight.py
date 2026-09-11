"""Reusable preflight orchestration for external benchmark runs.

The measured benchmark runner owns execution. This module validates the same
configured environment and policy boundary before measurement begins. Controlled
runs can additionally require exact repository, runtime, and source-checkout
state before any third-party environment is constructed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from remem.benchmark import BenchmarkRunReport, BenchmarkSuiteRunner
from remem.environments import EnvironmentContractReport

from experiments.external_benchmark import (
    ExternalBenchmarkSpec,
    run_external_benchmark,
    run_repeated_external_benchmarks,
    validate_external_benchmark_runtime,
    validate_repeated_benchmark_request,
)
from experiments.runtime_provenance import collect_runtime_provenance
from experiments.runtime_requirements import RuntimeRequirements, validate_runtime_requirements
from experiments.source_checkouts import (
    SourceCheckoutProvenance,
    SourceCheckoutRequirement,
    collect_source_checkout_provenance,
    validate_source_checkout_requirements,
)


def validate_controlled_external_benchmark_runtime(
    spec: ExternalBenchmarkSpec,
    *,
    probe_action: str | None = None,
    runtime_requirements: RuntimeRequirements | None = None,
    source_checkout_paths: Mapping[str, Path] | None = None,
    source_checkout_requirements: Mapping[str, SourceCheckoutRequirement] | None = None,
) -> EnvironmentContractReport:
    """Validate controlled state before probing one external benchmark.

    Runtime and source-checkout gates run before callable resolution or environment
    construction. A revision, working-tree, dependency, or external checkout
    mismatch therefore cannot trigger third-party benchmark side effects.
    """

    _validate_controlled_runtime(runtime_requirements)
    _validate_controlled_source_checkouts(source_checkout_paths, source_checkout_requirements)
    return validate_external_benchmark_runtime(spec, probe_action=probe_action)


def run_external_benchmark_with_preflight(
    spec: ExternalBenchmarkSpec,
    *,
    probe_action: str | None = None,
    runner: BenchmarkSuiteRunner | None = None,
    runtime_requirements: RuntimeRequirements | None = None,
    source_checkout_paths: Mapping[str, Path] | None = None,
    source_checkout_requirements: Mapping[str, SourceCheckoutRequirement] | None = None,
) -> BenchmarkRunReport:
    """Preflight one external benchmark and only then launch measurement."""

    validate_controlled_external_benchmark_runtime(
        spec,
        probe_action=probe_action,
        runtime_requirements=runtime_requirements,
        source_checkout_paths=source_checkout_paths,
        source_checkout_requirements=source_checkout_requirements,
    )
    return run_external_benchmark(spec, runner=runner)


def validate_repeated_external_benchmark_runtime(
    spec: ExternalBenchmarkSpec,
    seeds: Sequence[int],
    *,
    probe_action: str | None = None,
    runtime_requirements: RuntimeRequirements | None = None,
    source_checkout_paths: Mapping[str, Path] | None = None,
    source_checkout_requirements: Mapping[str, SourceCheckoutRequirement] | None = None,
) -> tuple[EnvironmentContractReport, ...]:
    """Validate every independent seed through the real external boundary.

    Each seed is probed independently using the same runtime-preflight path as
    measured execution. The function does not create benchmark reports and does
    not share probe environments or memory stores across seeds. A single-run
    ``spec.seed`` is rejected so preflight and measured execution share one
    unambiguous repeated-seed contract.

    Controlled runtime and source-checkout requirements are validated once before
    the first external environment is constructed. A failed admission gate
    therefore produces no benchmark probe side effects.
    """

    selected_seeds = validate_repeated_benchmark_request(spec, seeds)
    _validate_controlled_runtime(runtime_requirements)
    _validate_controlled_source_checkouts(source_checkout_paths, source_checkout_requirements)
    return tuple(
        validate_external_benchmark_runtime(
            replace(spec, seed=seed),
            probe_action=probe_action,
        )
        for seed in selected_seeds
    )


def run_repeated_external_benchmarks_with_preflight(
    spec: ExternalBenchmarkSpec,
    seeds: Sequence[int],
    *,
    probe_action: str | None = None,
    runner: BenchmarkSuiteRunner | None = None,
    runtime_requirements: RuntimeRequirements | None = None,
    source_checkout_paths: Mapping[str, Path] | None = None,
    source_checkout_requirements: Mapping[str, SourceCheckoutRequirement] | None = None,
) -> tuple[BenchmarkRunReport, ...]:
    """Preflight every seed and only then launch measured benchmark runs.

    The preflight is deliberately a separate phase. A failed runtime, source
    checkout, environment, or policy probe prevents any measured run from
    starting, while successful probes are never included in the returned
    benchmark evidence. A supplied runner is reused for all measured seeds so
    suite-level observability remains attached to the same execution lifecycle as
    non-preflight repeated runs.
    """

    selected_seeds = validate_repeated_benchmark_request(spec, seeds)
    validate_repeated_external_benchmark_runtime(
        spec,
        selected_seeds,
        probe_action=probe_action,
        runtime_requirements=runtime_requirements,
        source_checkout_paths=source_checkout_paths,
        source_checkout_requirements=source_checkout_requirements,
    )
    return run_repeated_external_benchmarks(spec, selected_seeds, runner=runner)


def _validate_controlled_runtime(runtime_requirements: RuntimeRequirements | None) -> None:
    """Fail closed on declared runtime drift before constructing environments."""

    if runtime_requirements is None:
        return
    if not isinstance(runtime_requirements, RuntimeRequirements):
        raise TypeError("runtime_requirements must be a RuntimeRequirements instance or None")
    provenance = collect_runtime_provenance()
    validate_runtime_requirements(provenance, runtime_requirements)


def _validate_controlled_source_checkouts(
    source_checkout_paths: Mapping[str, Path] | None,
    source_checkout_requirements: Mapping[str, SourceCheckoutRequirement] | None,
) -> Mapping[str, SourceCheckoutProvenance] | None:
    """Validate declared source-installed benchmark repositories before probing.

    Paths and requirements form one admission contract and must be supplied
    together. Returning the exact collected mapping keeps the boundary reusable
    for future artifact-provenance binding without recollecting Git state.
    """

    if source_checkout_paths is None and source_checkout_requirements is None:
        return None
    if source_checkout_paths is None or source_checkout_requirements is None:
        raise ValueError(
            "source_checkout_paths and source_checkout_requirements must be provided together"
        )
    if not isinstance(source_checkout_paths, Mapping):
        raise TypeError("source_checkout_paths must be a mapping or None")
    if not isinstance(source_checkout_requirements, Mapping):
        raise TypeError("source_checkout_requirements must be a mapping or None")
    if not source_checkout_requirements:
        raise ValueError("source_checkout_requirements must not be empty when source paths are set")

    provenance = collect_source_checkout_provenance(source_checkout_paths)
    validate_source_checkout_requirements(provenance, source_checkout_requirements)
    return provenance


__all__ = [
    "run_external_benchmark_with_preflight",
    "run_repeated_external_benchmarks_with_preflight",
    "validate_controlled_external_benchmark_runtime",
    "validate_repeated_external_benchmark_runtime",
]
