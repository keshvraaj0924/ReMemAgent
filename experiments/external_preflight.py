"""Reusable preflight orchestration for repeated external benchmark runs.

The measured benchmark runner owns execution. This module validates the
same configured environment and policy boundary for every independent seed
before a multi-seed experiment is launched. Controlled runs can additionally
require an exact repository/runtime state before any third-party environment is
constructed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from remem.benchmark import BenchmarkRunReport, BenchmarkSuiteRunner
from remem.environments import EnvironmentContractReport

from experiments.external_benchmark import (
    ExternalBenchmarkSpec,
    run_repeated_external_benchmarks,
    validate_external_benchmark_runtime,
    validate_repeated_benchmark_request,
)
from experiments.runtime_provenance import collect_runtime_provenance
from experiments.runtime_requirements import RuntimeRequirements, validate_runtime_requirements


def validate_repeated_external_benchmark_runtime(
    spec: ExternalBenchmarkSpec,
    seeds: Sequence[int],
    *,
    probe_action: str | None = None,
    runtime_requirements: RuntimeRequirements | None = None,
) -> tuple[EnvironmentContractReport, ...]:
    """Validate every independent seed through the real external boundary.

    Each seed is probed independently using the same runtime-preflight path as
    measured execution. The function does not create benchmark reports and does
    not share probe environments or memory stores across seeds. A single-run
    ``spec.seed`` is rejected so preflight and measured execution share one
    unambiguous repeated-seed contract.

    When ``runtime_requirements`` is supplied, code revision, working-tree state,
    and pinned dependency versions are validated before the first external
    environment is constructed. A failed runtime gate therefore produces no
    benchmark probe side effects.
    """

    selected_seeds = validate_repeated_benchmark_request(spec, seeds)
    _validate_controlled_runtime(runtime_requirements)
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
) -> tuple[BenchmarkRunReport, ...]:
    """Preflight every seed and only then launch measured benchmark runs.

    The preflight is deliberately a separate phase. A failed runtime,
    environment, or policy probe prevents any measured run from starting, while
    successful probes are never included in the returned benchmark evidence. A
    supplied runner is reused for all measured seeds so suite-level
    observability remains attached to the same execution lifecycle as
    non-preflight repeated runs.
    """

    selected_seeds = validate_repeated_benchmark_request(spec, seeds)
    validate_repeated_external_benchmark_runtime(
        spec,
        selected_seeds,
        probe_action=probe_action,
        runtime_requirements=runtime_requirements,
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


__all__ = [
    "run_repeated_external_benchmarks_with_preflight",
    "validate_repeated_external_benchmark_runtime",
]
