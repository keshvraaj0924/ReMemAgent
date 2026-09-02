"""Regression tests for preflight environment ownership and cleanup."""

from __future__ import annotations

from experiments.external_benchmark import validate_external_benchmark_runtime
from tests.test_external_benchmark import CLOSED_SEEDS, _build_spec


def test_runtime_preflight_closes_probe_environment_exactly_once() -> None:
    """The environment-contract validator is the sole owner of probe cleanup."""

    CLOSED_SEEDS.clear()

    validate_external_benchmark_runtime(_build_spec(seed=29))

    assert CLOSED_SEEDS == [29]


def test_runtime_preflight_runs_policy_after_environment_cleanup() -> None:
    """Policy validation uses the reset observation without requiring a live environment."""

    CLOSED_SEEDS.clear()

    report = validate_external_benchmark_runtime(_build_spec(seed=31))

    assert report.initial_observation == "seed-31"
    assert CLOSED_SEEDS == [31]
