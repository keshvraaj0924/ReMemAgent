"""Tests for the verl-compatible custom reward entrypoint."""

import pytest

from remem.training.verl_reward import compute_score


def test_compute_score_accepts_verl_callback_shape() -> None:
    reward = compute_score(
        data_source="remem/synthetic",
        solution_str="generated answer",
        ground_truth="reference answer",
        extra_info={
            "task_reward": 1.0,
            "memory_used": True,
            "counterfactual_delta": 0.4,
        },
    )

    assert reward == pytest.approx(1.19)


def test_compute_score_requires_measured_extra_info() -> None:
    with pytest.raises(ValueError, match="extra_info is required"):
        compute_score(
            data_source="remem/synthetic",
            solution_str="generated answer",
            ground_truth="reference answer",
        )


def test_compute_score_does_not_infer_missing_measurements_from_text() -> None:
    with pytest.raises(ValueError, match="task_reward"):
        compute_score(
            data_source="remem/synthetic",
            solution_str="correct generated answer",
            ground_truth="correct generated answer",
            extra_info={"memory_used": False},
        )
