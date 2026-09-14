"""Validate and normalize numeric values at the verl training boundary."""

from __future__ import annotations

from fractions import Fraction
from typing import cast

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


def test_verl_trajectory_detaches_mutable_token_sequences() -> None:
    """Direct construction must not retain caller-owned mutable token containers."""

    prompt_ids = [1, 2]
    response_ids = [3, 4]

    trajectory = VerlTrajectory(
        prompt_ids=cast(tuple[int, ...], prompt_ids),
        response_ids=cast(tuple[int, ...], response_ids),
        response_mask=(1, 1),
        reward=1.0,
        metadata={},
    )

    prompt_ids.append(99)
    response_ids[0] = 88

    assert trajectory.prompt_ids == (1, 2)
    assert trajectory.response_ids == (3, 4)
    assert type(trajectory.prompt_ids) is tuple
    assert type(trajectory.response_ids) is tuple


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


def test_verl_trajectory_rejects_non_real_reward() -> None:
    """Non-real rewards must fail through the explicit numeric contract."""

    with pytest.raises(TypeError, match="reward must be a real number"):
        VerlTrajectory(
            prompt_ids=(1,),
            response_ids=(2,),
            response_mask=(1,),
            reward="1.0",
            metadata={},
        )


def test_verl_trajectory_normalizes_real_scalars_to_float() -> None:
    """Compatible real scalar implementations should serialize as plain floats."""

    trajectory = VerlTrajectory(
        prompt_ids=(1,),
        response_ids=(2,),
        response_mask=(1,),
        reward=Fraction(3, 2),
        metadata={},
        response_logprobs=(Fraction(-1, 4),),
    )

    assert trajectory.reward == 1.5
    assert type(trajectory.reward) is float
    assert trajectory.response_logprobs == (-0.25,)
    assert type(trajectory.response_logprobs[0]) is float


def test_verl_trajectory_rejects_non_real_logprob() -> None:
    """Rollout log probabilities must obey the same real-valued contract."""

    with pytest.raises(TypeError, match="response_logprobs must be real numbers"):
        VerlTrajectory(
            prompt_ids=(1,),
            response_ids=(2,),
            response_mask=(1,),
            reward=1.0,
            metadata={},
            response_logprobs=("-0.25",),
        )


def test_verl_training_batch_rejects_boolean_advantage() -> None:
    """Boolean advantages must not pass as real-valued advantages."""

    with pytest.raises(TypeError, match="advantages must be real numbers"):
        VerlTrainingBatch(trajectories=(_trajectory(),), advantages=(True,))


def test_verl_training_batch_rejects_non_real_advantage() -> None:
    """Non-real advantages must fail through the explicit numeric contract."""

    with pytest.raises(TypeError, match="advantages must be real numbers"):
        VerlTrainingBatch(trajectories=(_trajectory(),), advantages=("1.0",))


def test_verl_training_batch_normalizes_real_advantage_to_float() -> None:
    """Compatible real advantage scalars should be normalized for framework handoff."""

    batch = VerlTrainingBatch(
        trajectories=(_trajectory(),),
        advantages=(Fraction(1, 3),),
    )

    assert batch.advantages == (pytest.approx(1 / 3),)
    assert type(batch.advantages[0]) is float
