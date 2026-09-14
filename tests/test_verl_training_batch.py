"""Tests for the dependency-free verl training batch boundary."""

from typing import cast

import pytest

from remem.integrations.verl import (
    VerlTrainingBatch,
    VerlTrajectory,
    build_verl_training_batch,
)


def _trajectory(reward: float) -> VerlTrajectory:
    return VerlTrajectory(
        prompt_ids=(1,),
        response_ids=(2, 3),
        response_mask=(1, 1),
        reward=reward,
        metadata={"memory_ids": []},
    )


def test_verl_training_batch_detaches_mutable_trajectory_sequence() -> None:
    first = _trajectory(1.0)
    second = _trajectory(0.0)
    trajectories = [first, second]

    batch = VerlTrainingBatch(
        trajectories=cast(tuple[VerlTrajectory, ...], trajectories),
        advantages=(0.5, -0.5),
    )

    trajectories.clear()

    assert batch.trajectories == (first, second)
    assert type(batch.trajectories) is tuple


def test_verl_training_batch_rejects_invalid_trajectory_values() -> None:
    with pytest.raises(TypeError, match="only VerlTrajectory"):
        VerlTrainingBatch(
            trajectories=cast(tuple[VerlTrajectory, ...], (object(),)),
            advantages=(0.0,),
        )


def test_build_verl_training_batch_preserves_order_and_alignment() -> None:
    first = _trajectory(1.0)
    second = _trajectory(0.0)

    batch = build_verl_training_batch((first, second), (0.75, -0.75))

    assert batch.trajectories == (first, second)
    assert batch.advantages == (0.75, -0.75)
    assert batch.to_dicts()[0]["advantage"] == 0.75
    assert batch.to_dicts()[1]["advantage"] == -0.75


def test_build_verl_training_batch_rejects_misaligned_advantages() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        build_verl_training_batch((_trajectory(1.0),), ())


def test_build_verl_training_batch_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one trajectory"):
        build_verl_training_batch((), ())


def test_build_verl_training_batch_does_not_pad_variable_length_trajectories() -> None:
    short = _trajectory(1.0)
    long = VerlTrajectory(
        prompt_ids=(1, 2),
        response_ids=(3, 4, 5),
        response_mask=(1, 1, 1),
        reward=0.5,
        metadata={"memory_ids": ["memory-a"]},
    )

    batch = build_verl_training_batch((short, long), (1.0, -1.0))

    assert batch.trajectories[0].response_ids == (2, 3)
    assert batch.trajectories[1].response_ids == (3, 4, 5)
