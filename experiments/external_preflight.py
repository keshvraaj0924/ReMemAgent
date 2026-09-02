"""Reusable preflight orchestration for repeated external benchmark runs.

The measured benchmark runner owns execution. This module only validates the
same configured environment and policy boundary for every independent seed
before a multi-seed experiment is launched.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from remem.environments import EnvironmentContractReport

from experiments.external_benchmark import (
    ExternalBenchmarkSpec,
    validate_external_benchmark_runtime,
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

    selected_seeds = tuple(seeds)
    if not selected_seeds:
        raise ValueError("seeds must contain at least one seed")
    if len(selected_seeds) != len(set(selected_seeds)):
        raise ValueError("seeds must be unique")

    return tuple(
        validate_external_benchmark_runtime(
            replace(spec, seed=seed),
            probe_action=probe_action,
        )
        for seed in selected_seeds
    )


__all__ = ["validate_repeated_external_benchmark_runtime"]
