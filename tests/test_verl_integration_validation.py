"""Regression tests for the verl token-level training boundary."""

from __future__ import annotations

from math import inf, nan

import pytest

from remem.integrations.verl import VerlTrajectory, VerlTrainingBatch


def _trajectory() -> VerlTrajectory:
    """Build a minimal valid trajectory for batch validation tests."""

    return VerlTrajectory(
        prompt_ids=(1,),
        response_ids=(2,),
        response_mask=(1,),
        reward=0.0,
        metadata={},
    )


@pytest.mark.parametrize("reward", [nan, inf, -inf])
def test_verl_trajectory_rejects_non_finite_reward(reward: float) -> None:
    """Training trajectories must not carry NaN or infinite rewards."""

    with pytest.raises(ValueError, match="reward must be finite"):
        VerlTrajectory(
            prompt_ids=(1,),
            response_ids=(2,),
            response_mask=(1,),
            reward=reward,
            metadata={},
        )


@pytest.mark.parametrize("advantage", [nan, inf, -inf])
def test_verl_training_batch_rejects_non_finite_advantage(advantage: float) -> None:
    """Direct batch construction must preserve the finite-advantage invariant."""

    with pytest.raises(ValueError, match="advantages must be finite"):
        VerlTrainingBatch(trajectories=(_trajectory(),), advantages=(advantage,))
