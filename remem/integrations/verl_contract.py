"""Validation boundary for externally generated verl agent-loop outputs.

The validator mirrors the token-level fields required by verl without importing
verl. External ``AgentLoopBase`` implementations can pass their exact output
through this boundary before converting it into training records. No
re-encoding is performed, which preserves token-level trajectory fidelity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValidatedAgentLoopOutput:
    """Validated token fields emitted by an external agent loop."""

    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]

    def to_dict(self) -> dict[str, list[int]]:
        """Return the framework-facing token fields as fresh lists."""

        return {
            "prompt_ids": list(self.prompt_ids),
            "response_ids": list(self.response_ids),
            "response_mask": list(self.response_mask),
        }


def validate_agent_loop_output(
    output: Mapping[str, Sequence[int]],
) -> ValidatedAgentLoopOutput:
    """Validate an external verl-style agent-loop output without re-encoding it."""

    required_fields = ("prompt_ids", "response_ids", "response_mask")
    missing_fields = [field for field in required_fields if field not in output]
    if missing_fields:
        raise ValueError(f"agent-loop output is missing fields: {', '.join(missing_fields)}")

    prompt_ids = _validate_token_ids(output["prompt_ids"], "prompt_ids")
    response_ids = _validate_token_ids(output["response_ids"], "response_ids")
    response_mask = _validate_response_mask(output["response_mask"], len(response_ids))
    if not response_ids:
        raise ValueError("agent-loop output must contain at least one response token")

    return ValidatedAgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
    )


def _validate_token_ids(token_ids: Sequence[int], field_name: str) -> tuple[int, ...]:
    """Validate and normalize token IDs into immutable integer tuples."""

    normalized = tuple(token_ids)
    if any(not isinstance(token_id, int) for token_id in normalized):
        raise TypeError(f"{field_name} must contain only integer token IDs")
    if any(token_id < 0 for token_id in normalized):
        raise ValueError(f"{field_name} must contain non-negative token IDs")
    return normalized


def _validate_response_mask(
    response_mask: Sequence[int],
    response_length: int,
) -> tuple[int, ...]:
    """Validate a binary response mask with one entry per response token."""

    normalized = tuple(response_mask)
    if len(normalized) != response_length:
        raise ValueError("response_mask must have one entry per response token")
    if any(mask not in (0, 1) for mask in normalized):
        raise ValueError("response_mask values must be either 0 or 1")
    return normalized
