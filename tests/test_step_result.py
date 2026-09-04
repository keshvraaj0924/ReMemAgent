"""Tests for the normalized environment step result contract."""

from __future__ import annotations

import math

import pytest

from remem.environments import StepResult


def test_step_result_accepts_finite_numeric_reward() -> None:
    result = StepResult(
        observation="state",
        reward=1,
        terminated=False,
        truncated=False,
    )

    assert result.reward == 1
    assert not result.done


@pytest.mark.parametrize("reward", [math.nan, math.inf, -math.inf])
def test_step_result_rejects_non_finite_reward(reward: float) -> None:
    with pytest.raises(ValueError, match="reward must be finite"):
        StepResult(
            observation="state",
            reward=reward,
            terminated=False,
            truncated=False,
        )


def test_step_result_rejects_boolean_reward() -> None:
    with pytest.raises(TypeError, match="reward must be a finite number"):
        StepResult(
            observation="state",
            reward=True,  # type: ignore[arg-type]
            terminated=False,
            truncated=False,
        )


@pytest.mark.parametrize("field", ["terminated", "truncated"])
def test_step_result_requires_boolean_terminal_flags(field: str) -> None:
    values = {
        "observation": "state",
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
    }
    values[field] = 1

    with pytest.raises(TypeError, match=f"{field} must be a boolean"):
        StepResult(**values)  # type: ignore[arg-type]


def test_step_result_requires_string_observation() -> None:
    with pytest.raises(TypeError, match="observation must be a string"):
        StepResult(
            observation=None,  # type: ignore[arg-type]
            reward=0.0,
            terminated=False,
            truncated=False,
        )


def test_step_result_requires_dictionary_info() -> None:
    with pytest.raises(TypeError, match="info must be a dictionary"):
        StepResult(
            observation="state",
            reward=0.0,
            terminated=False,
            truncated=False,
            info=[],  # type: ignore[arg-type]
        )


def test_step_result_done_reflects_terminal_or_truncated_state() -> None:
    terminated = StepResult("state", 0.0, True, False)
    truncated = StepResult("state", 0.0, False, True)

    assert terminated.done
    assert truncated.done
