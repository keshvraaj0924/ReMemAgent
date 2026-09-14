"""Regression tests for normalized environment transition ownership."""

from fractions import Fraction

import pytest

from remem.environments.base import StepResult


def test_step_result_detaches_nested_info_from_environment_owned_metadata() -> None:
    environment_info = {"metrics": {"success": True}, "tags": ["alfworld"]}

    result = StepResult("observation", 1.0, True, False, environment_info)

    environment_info["metrics"]["success"] = False
    environment_info["tags"].append("mutated")

    assert result.info == {"metrics": {"success": True}, "tags": ["alfworld"]}


def test_step_result_keeps_nested_info_independent_between_instances() -> None:
    source_info = {"metrics": {"score": 0.5}}

    first = StepResult("first", 0.0, False, False, source_info)
    second = StepResult("second", 0.0, False, False, source_info)

    first.info["metrics"]["score"] = 1.0

    assert second.info["metrics"]["score"] == 0.5
    assert source_info["metrics"]["score"] == 0.5


def test_step_result_normalizes_real_numeric_rewards_to_float() -> None:
    result = StepResult("observation", Fraction(3, 4), False, False)

    assert result.reward == 0.75
    assert isinstance(result.reward, float)


@pytest.mark.parametrize("reward", [True, "1.0", 1 + 0j, float("inf"), float("nan")])
def test_step_result_rejects_invalid_rewards(reward: object) -> None:
    expected_error = ValueError if isinstance(reward, float) else TypeError

    with pytest.raises(expected_error):
        StepResult("observation", reward, False, False)  # type: ignore[arg-type]


def test_step_result_rejects_non_string_metadata_keys() -> None:
    with pytest.raises(TypeError, match="info keys must be strings"):
        StepResult("observation", 1.0, False, False, {1: "invalid"})  # type: ignore[dict-item]
