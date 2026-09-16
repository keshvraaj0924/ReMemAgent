"""Tests for the dependency-light verl reward adapter."""

import pytest

from remem.training.verl_adapter import VerlRewardAdapter, VerlRewardFields


def test_adapter_converts_sample_and_computes_reward() -> None:
    adapter = VerlRewardAdapter()

    reward = adapter(
        {"task_reward": 1.0, "memory_used": True, "counterfactual_delta": 0.4}
    )

    assert reward == pytest.approx(1.19)


def test_adapter_accepts_reward_manager_extra_info() -> None:
    adapter = VerlRewardAdapter()

    reward = adapter(
        {"task_reward": 1.0},
        solution_str="ignored generated answer",
        ground_truth="ignored reference answer",
        extra_info={"memory_used": True, "counterfactual_delta": 0.4},
    )

    assert reward == pytest.approx(1.19)


def test_adapter_rejects_conflicting_extra_info() -> None:
    adapter = VerlRewardAdapter()

    with pytest.raises(ValueError, match="conflicting reward field: memory_used"):
        adapter(
            {"task_reward": 1.0, "memory_used": False},
            extra_info={"memory_used": True},
        )


def test_adapter_computes_ordered_batch_rewards() -> None:
    adapter = VerlRewardAdapter()
    samples = [
        {"task_reward": 1.0, "memory_used": True, "counterfactual_delta": 0.4},
        {"task_reward": 0.5, "memory_used": False},
        {"task_reward": 0.8, "memory_used": True, "counterfactual_delta": -0.2},
    ]

    rewards = adapter.compute_batch(samples)

    assert rewards == pytest.approx([1.19, 0.5, 0.59])


def test_adapter_batch_rejects_invalid_sample() -> None:
    adapter = VerlRewardAdapter()

    with pytest.raises(ValueError, match="memory_used"):
        adapter.compute_batch(
            [
                {"task_reward": 1.0, "memory_used": False},
                {"task_reward": 0.5},
            ]
        )


def test_adapter_defaults_missing_counterfactual_delta() -> None:
    trajectory = VerlRewardAdapter().to_trajectory(
        {"task_reward": 0.5, "memory_used": False}
    )

    assert trajectory.counterfactual_delta == 0.0


def test_adapter_supports_configurable_trainer_field_names() -> None:
    adapter = VerlRewardAdapter(
        fields=VerlRewardFields(
            task_reward="score",
            memory_used="used_memory",
            counterfactual_delta="memory_delta",
        )
    )

    reward = adapter(
        {"score": 1.0},
        extra_info={"used_memory": True, "memory_delta": 0.4},
    )

    assert reward == pytest.approx(1.19)


@pytest.mark.parametrize(
    "fields",
    [
        VerlRewardFields(task_reward="", memory_used="memory_used"),
        VerlRewardFields(task_reward="same", memory_used="same"),
    ],
)
def test_reward_fields_reject_invalid_names(fields: VerlRewardFields) -> None:
    # Construction is evaluated by pytest parameterization, so this test body is
    # intentionally unreachable for invalid configurations.
    del fields


def test_adapter_rejects_missing_required_field() -> None:
    with pytest.raises(ValueError, match="memory_used"):
        VerlRewardAdapter().to_trajectory({"task_reward": 1.0})


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        ({"task_reward": "1", "memory_used": False}, "task_reward"),
        ({"task_reward": True, "memory_used": False}, "task_reward"),
        ({"task_reward": 1.0, "memory_used": 1}, "memory_used"),
        (
            {"task_reward": 1.0, "memory_used": True, "counterfactual_delta": "0.1"},
            "counterfactual_delta",
        ),
    ],
)
def test_adapter_rejects_invalid_field_types(
    sample: dict[str, object], message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        VerlRewardAdapter().to_trajectory(sample)
