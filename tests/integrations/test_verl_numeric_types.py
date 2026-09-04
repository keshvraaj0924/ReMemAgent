"""Reject boolean values at the verl numeric training boundary."""

from __future__ import annotations

import pytest

from remem.integrations.verl import VerlTrainingBatch, VerlTrajectory


def _trajectory() -> VerlTrajectory:
    """Build a minimal valid trajectory."""

    return VerlTrajectory(
        prompt_ids=(1,),
        response_ids=(2,),
        response_mask=(1,),
        reward=1.0,
        metadata={},
    )


def test_verl_trajectory_rejects_boolean_reward() -> None:
    """Boolean rewards must not pass as real-valued rewards."""

    with pytest.raises(TypeError, match="reward must be a real number"):
        VerlTrajectory(
            prompt_ids=(1,),
            response_ids=(2,),
            response_mask=(1,),
            reward=True,
            metadata={},
        )


def test_verl_training_batch_rejects_boolean_advantage() -> None:
    """Boolean advantages must not pass as real-valued advantages."""

    with pytest.raises(TypeError, match="advantages must be real numbers"):
        VerlTrainingBatch(trajectories=(_trajectory(),), advantages=(True,))
