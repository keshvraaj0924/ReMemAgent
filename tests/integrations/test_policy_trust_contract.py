"""Regression tests for strict policy trust-threshold validation."""

from __future__ import annotations

import pytest

from remem.integrations.policies import build_memory_guided_policy_factory


@pytest.mark.parametrize("minimum_trust", [True, False])
def test_policy_factory_rejects_boolean_minimum_trust(minimum_trust: bool) -> None:
    """Boolean values must not silently behave like numeric trust thresholds."""

    with pytest.raises(TypeError, match="minimum_trust"):
        build_memory_guided_policy_factory(
            lambda _seed: lambda _observation: "look",
            minimum_trust=minimum_trust,
        )


@pytest.mark.parametrize("minimum_trust", [float("nan"), float("inf"), -float("inf")])
def test_policy_factory_rejects_non_finite_minimum_trust(minimum_trust: float) -> None:
    """Non-finite thresholds must fail before a policy can be constructed."""

    with pytest.raises(ValueError, match="finite"):
        build_memory_guided_policy_factory(
            lambda _seed: lambda _observation: "look",
            minimum_trust=minimum_trust,
        )


@pytest.mark.parametrize("minimum_trust", [-0.01, 1.01])
def test_policy_factory_rejects_out_of_range_minimum_trust(minimum_trust: float) -> None:
    """Trust thresholds outside the closed unit interval are invalid."""

    with pytest.raises(ValueError, match="between 0 and 1"):
        build_memory_guided_policy_factory(
            lambda _seed: lambda _observation: "look",
            minimum_trust=minimum_trust,
        )
