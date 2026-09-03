"""Type-contract regression tests for external verl token records."""

from __future__ import annotations

import pytest

from remem.integrations.verl import VerlTrajectory
from remem.integrations.verl_contract import validate_agent_loop_output


@pytest.mark.parametrize("field_name", ["prompt_ids", "response_ids"])
def test_validate_agent_loop_output_rejects_boolean_token_ids(field_name: str) -> None:
    """Boolean values must not pass the integer token-ID boundary."""

    output = {
        "prompt_ids": [1],
        "response_ids": [2],
        "response_mask": [1],
    }
    output[field_name] = [True]

    with pytest.raises(TypeError, match="must contain only integer token IDs"):
        validate_agent_loop_output(output)


def test_validate_agent_loop_output_rejects_boolean_response_mask() -> None:
    """Boolean masks must not be silently accepted as integer mask values."""

    with pytest.raises(TypeError, match="response_mask must contain only integer values"):
        validate_agent_loop_output(
            {
                "prompt_ids": [1],
                "response_ids": [2],
                "response_mask": [True],
            }
        )


def test_validate_agent_loop_output_accepts_integer_mask_and_token_ids() -> None:
    """Valid integer token fields remain unchanged by strict type validation."""

    result = validate_agent_loop_output(
        {
            "prompt_ids": [1, 2],
            "response_ids": [3, 4],
            "response_mask": [1, 0],
        }
    )

    assert result.prompt_ids == (1, 2)
    assert result.response_ids == (3, 4)
    assert result.response_mask == (1, 0)


def test_verl_trajectory_rejects_boolean_response_mask() -> None:
    """Direct construction cannot bypass strict mask typing."""

    with pytest.raises(TypeError, match="response_mask must contain only integer values"):
        VerlTrajectory(
            prompt_ids=(1,),
            response_ids=(2,),
            response_mask=(True,),
            reward=1.0,
            metadata={},
        )


def test_verl_trajectory_rejects_mismatched_response_mask_length() -> None:
    """A response mask must align one-to-one with response tokens."""

    with pytest.raises(ValueError, match="same length as response_ids"):
        VerlTrajectory(
            prompt_ids=(1,),
            response_ids=(2, 3),
            response_mask=(1,),
            reward=1.0,
            metadata={},
        )


def test_verl_trajectory_rejects_mask_values_other_than_zero_or_one() -> None:
    """Masks remain binary at the framework-facing trajectory boundary."""

    with pytest.raises(ValueError, match="either 0 or 1"):
        VerlTrajectory(
            prompt_ids=(1,),
            response_ids=(2,),
            response_mask=(2,),
            reward=1.0,
            metadata={},
        )
