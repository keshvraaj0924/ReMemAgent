"""Tests for the verl-compatible batched reward entrypoint."""

import pytest

from remem.training.verl_reward import compute_score_batched


def test_compute_score_batched_preserves_order_and_scalar_semantics() -> None:
    rewards = compute_score_batched(
        data_sources=["remem/synthetic", "remem/synthetic"],
        solution_strs=["first", "second"],
        ground_truths=["reference", "reference"],
        extra_infos=[
            {"task_reward": 1.0, "memory_used": True, "counterfactual_delta": 0.4},
            {"task_reward": 0.5, "memory_used": False, "counterfactual_delta": -0.8},
        ],
    )

    assert rewards == pytest.approx([1.19, 0.5])


def test_compute_score_batched_supports_reward_configuration() -> None:
    rewards = compute_score_batched(
        data_sources=["remem/synthetic"],
        solution_strs=["generated"],
        ground_truths=["reference"],
        extra_infos=[{"score": 1.0, "used": True, "delta": 0.5}],
        task_weight=2.0,
        positive_transfer_weight=1.0,
        memory_use_cost=0.25,
        task_reward_field="score",
        memory_used_field="used",
        counterfactual_delta_field="delta",
    )

    assert rewards == pytest.approx([2.25])


def test_compute_score_batched_rejects_misaligned_batches() -> None:
    with pytest.raises(ValueError, match="batch lengths must match"):
        compute_score_batched(
            data_sources=["first", "second"],
            solution_strs=["only-one"],
            ground_truths=["first", "second"],
            extra_infos=[
                {"task_reward": 1.0, "memory_used": False},
                {"task_reward": 1.0, "memory_used": False},
            ],
        )


def test_compute_score_batched_accepts_empty_batch() -> None:
    assert compute_score_batched([], [], [], []) == []
