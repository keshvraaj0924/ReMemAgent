import math

import pytest

from experiments.external_benchmark import ExternalBenchmarkSpec


def build_spec(minimum_trust: float) -> ExternalBenchmarkSpec:
    """Build a minimally valid external benchmark configuration."""

    return ExternalBenchmarkSpec(
        benchmark_name="alfworld",
        episode_count=1,
        max_steps=1,
        environment_factory="example:environment_factory",
        policy_factory="example:policy_factory",
        success_evaluator="example:success_evaluator",
        minimum_trust=minimum_trust,
    )


def test_external_benchmark_spec_accepts_finite_trust_threshold() -> None:
    spec = build_spec(0.5)

    assert spec.minimum_trust == 0.5


@pytest.mark.parametrize("minimum_trust", [math.nan, math.inf, -math.inf])
def test_external_benchmark_spec_rejects_non_finite_trust_threshold(
    minimum_trust: float,
) -> None:
    with pytest.raises(ValueError, match="minimum_trust must be finite"):
        build_spec(minimum_trust)
