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


def test_compute_score_accepts_configurable_reward_weights() -> None:
    reward = compute_score(
        data_source="remem/synthetic",
        solution_str="generated answer",
        ground_truth="reference answer",
        extra_info={
            "task_reward": 1.0,
            "memory_used": True,
            "counterfactual_delta": 0.4,
        },
        task_weight=2.0,
        positive_transfer_weight=1.0,
        memory_use_cost=0.2,
    )

    assert reward == pytest.approx(2.2)


def test_compute_score_accepts_configurable_metadata_fields() -> None:
    reward = compute_score(
        data_source="remem/synthetic",
        solution_str="generated answer",
        ground_truth="reference answer",
        extra_info={"score": 0.8, "used_memory": True, "memory_delta": -0.3},
        task_reward_field="score",
        memory_used_field="used_memory",
        counterfactual_delta_field="memory_delta",
    )

    assert reward == pytest.approx(0.49)


def test_compute_score_rejects_unknown_reward_options() -> None:
    with pytest.raises(ValueError, match="unsupported ReMemAgent reward option"):
        compute_score(
            data_source="remem/synthetic",
            solution_str="generated answer",
            ground_truth="reference answer",
            extra_info={"task_reward": 1.0, "memory_used": False},
            unsupported_option=1.0,
        )


def test_compute_score_rejects_invalid_reward_option_types() -> None:
    with pytest.raises(TypeError, match="task_weight must be a real number"):
        compute_score(
            data_source="remem/synthetic",
            solution_str="generated answer",
            ground_truth="reference answer",
            extra_info={"task_reward": 1.0, "memory_used": False},
            task_weight=True,
        )
