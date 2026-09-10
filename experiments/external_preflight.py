"""Reusable preflight orchestration for repeated external benchmark runs.

The measured benchmark runner owns execution. This module validates the
same configured environment and policy boundary for every independent seed
before a multi-seed experiment is launched.
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


def validate_repeated_external_benchmark_runtime(
    spec: ExternalBenchmarkSpec,
    seeds: Sequence[int],
    *,
    probe_action: str | None = None,
) -> tuple[EnvironmentContractReport, ...]:
    """Validate every independent seed through the real external boundary.

    Each seed is probed independently using the same runtime-preflight path as
    measured execution. The function does not create benchmark reports and does
    not share probe environments or memory stores across seeds. A single-run
    ``spec.seed`` is rejected so preflight and measured execution share one
    unambiguous repeated-seed contract.
    """

    selected_seeds = validate_repeated_benchmark_request(spec, seeds)
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
) -> tuple[BenchmarkRunReport, ...]:
    """Preflight every seed and only then launch measured benchmark runs.

    The preflight is deliberately a separate phase. A failed environment or
    policy probe prevents any measured run from starting, while successful probes
    are never included in the returned benchmark evidence. A supplied runner is
    reused for all measured seeds so suite-level observability remains attached
    to the same execution lifecycle as non-preflight repeated runs.
    """

    selected_seeds = validate_repeated_benchmark_request(spec, seeds)
    validate_repeated_external_benchmark_runtime(
        spec,
        selected_seeds,
        probe_action=probe_action,
    )
    return run_repeated_external_benchmarks(spec, selected_seeds, runner=runner)


__all__ = [
    "run_repeated_external_benchmarks_with_preflight",
    "validate_repeated_external_benchmark_runtime",
]
