"""Tests for the external verl agent-loop output validation boundary."""

import pytest

from remem.integrations.verl_contract import validate_agent_loop_output


def test_validate_agent_loop_output_preserves_exact_token_fields() -> None:
    output = validate_agent_loop_output(
        {
            "prompt_ids": [11, 12],
            "response_ids": [21, 22, 23],
            "response_mask": [1, 0, 1],
        }
    )

    assert output.prompt_ids == (11, 12)
    assert output.response_ids == (21, 22, 23)
    assert output.response_mask == (1, 0, 1)
    assert output.to_dict() == {
        "prompt_ids": [11, 12],
        "response_ids": [21, 22, 23],
        "response_mask": [1, 0, 1],
    }


def test_validate_agent_loop_output_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="missing fields"):
        validate_agent_loop_output(
            {
                "prompt_ids": [1],
                "response_ids": [2],
            }
        )


def test_validate_agent_loop_output_rejects_mask_length_mismatch() -> None:
    with pytest.raises(ValueError, match="one entry per response token"):
        validate_agent_loop_output(
            {
                "prompt_ids": [1],
                "response_ids": [2, 3],
                "response_mask": [1],
            }
        )


def test_validate_agent_loop_output_rejects_invalid_masks_and_token_ids() -> None:
    with pytest.raises(ValueError, match="either 0 or 1"):
        validate_agent_loop_output(
            {
                "prompt_ids": [1],
                "response_ids": [2],
                "response_mask": [2],
            }
        )

    with pytest.raises(ValueError, match="non-negative"):
        validate_agent_loop_output(
            {
                "prompt_ids": [-1],
                "response_ids": [2],
                "response_mask": [1],
            }
        )


def test_validate_agent_loop_output_rejects_non_integer_token_ids() -> None:
    with pytest.raises(TypeError, match="integer token IDs"):
        validate_agent_loop_output(
            {
                "prompt_ids": ["token"],  # type: ignore[list-item]
                "response_ids": [2],
                "response_mask": [1],
            }
        )
