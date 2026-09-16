"""Tests for the dependency-light verl reward adapter."""

import pytest

from remem.training.verl_adapter import VerlRewardAdapter


def test_adapter_converts_sample_and_computes_reward() -> None:
    adapter = VerlRewardAdapter()

    reward = adapter(
        {"task_reward": 1.0, "memory_used": True, "counterfactual_delta": 0.4}
    )

    assert reward == pytest.approx(1.19)


def test_adapter_defaults_missing_counterfactual_delta() -> None:
    trajectory = VerlRewardAdapter.to_trajectory({"task_reward": 0.5, "memory_used": False})

    assert trajectory.counterfactual_delta == 0.0


def test_adapter_rejects_missing_required_field() -> None:
    with pytest.raises(ValueError, match="memory_used"):
        VerlRewardAdapter.to_trajectory({"task_reward": 1.0})


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
        VerlRewardAdapter.to_trajectory(sample)
