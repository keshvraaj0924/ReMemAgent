"""Reusable preflight orchestration for repeated external benchmark runs.

The measured benchmark runner owns execution. This module validates the
same configured environment and policy boundary for every independent seed
before a multi-seed experiment is launched.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from remem.benchmark import BenchmarkRunReport
from remem.environments import EnvironmentContractReport

from experiments.external_benchmark import (
    ExternalBenchmarkSpec,
    run_repeated_external_benchmarks,
    validate_external_benchmark_runtime,
    validate_seed_sequence,
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
    not share probe environments or memory stores across seeds.
    """

    selected_seeds = validate_seed_sequence(seeds)
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
) -> tuple[BenchmarkRunReport, ...]:
    """Preflight every seed and only then launch measured benchmark runs.

    The preflight is deliberately a separate phase. A failed environment or
    policy probe prevents any measured run from starting, while successful probes
    are never included in the returned benchmark evidence.
    """

    selected_seeds = validate_seed_sequence(seeds)
    validate_repeated_external_benchmark_runtime(
        spec,
        selected_seeds,
        probe_action=probe_action,
    )
    return run_repeated_external_benchmarks(spec, selected_seeds)


__all__ = [
    "run_repeated_external_benchmarks_with_preflight",
    "validate_repeated_external_benchmark_runtime",
]
