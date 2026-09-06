"""Validation boundary for externally generated verl agent-loop outputs.

The validator mirrors the token-level fields required by verl without importing
verl. External ``AgentLoopBase`` implementations can pass their exact output
through this boundary before converting it into training records. No
re-encoding is performed, which preserves token-level trajectory fidelity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class ValidatedAgentLoopOutput:
    """Validated token fields and optional verl metadata emitted by an agent loop."""

    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    extra_fields: Mapping[str, Any]
    response_logprobs: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return token fields and copied optional fields as a fresh mapping."""

        output: dict[str, object] = {
            "prompt_ids": list(self.prompt_ids),
            "response_ids": list(self.response_ids),
            "response_mask": list(self.response_mask),
            "extra_fields": dict(self.extra_fields),
        }
        if self.response_logprobs is not None:
            output["response_logprobs"] = list(self.response_logprobs)
        return output


def validate_agent_loop_output(
    output: Mapping[str, object],
) -> ValidatedAgentLoopOutput:
    """Validate an external verl-style agent-loop output without re-encoding it.

    The current verl ``AgentLoopOutput`` also exposes optional response
    log-probabilities and ``extra_fields`` for dynamic trajectory metadata.
    Both are preserved rather than discarded so memory provenance and rollout
    likelihood data can cross the adapter boundary without coupling ReMemAgent
    to verl's Pydantic model.
    """

    required_fields = ("prompt_ids", "response_ids", "response_mask")
    missing_fields = [field for field in required_fields if field not in output]
    if missing_fields:
        raise ValueError(f"agent-loop output is missing fields: {', '.join(missing_fields)}")

    prompt_ids = _validate_token_ids(output["prompt_ids"], "prompt_ids")
    response_ids = _validate_token_ids(output["response_ids"], "response_ids")
    response_mask = _validate_response_mask(output["response_mask"], len(response_ids))
    if not response_ids:
        raise ValueError("agent-loop output must contain at least one response token")

    response_logprobs = _validate_response_logprobs(
        output.get("response_logprobs"),
        len(response_ids),
    )
    extra_fields = output.get("extra_fields", {})
    if not isinstance(extra_fields, Mapping):
        raise TypeError("agent-loop output extra_fields must be a mapping")
    if any(not isinstance(key, str) for key in extra_fields):
        raise TypeError("agent-loop output extra_fields keys must be strings")

    return ValidatedAgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        extra_fields=dict(extra_fields),
        response_logprobs=response_logprobs,
    )


def _validate_token_ids(token_ids: object, field_name: str) -> tuple[int, ...]:
    """Validate and normalize token IDs into immutable integer tuples."""

    if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
        raise TypeError(f"{field_name} must contain only integer token IDs")
    normalized = tuple(token_ids)
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in normalized):
        raise TypeError(f"{field_name} must contain only integer token IDs")
    if any(token_id < 0 for token_id in normalized):
        raise ValueError(f"{field_name} must contain non-negative token IDs")
    return normalized


def _validate_response_mask(
    response_mask: object,
    response_length: int,
) -> tuple[int, ...]:
    """Validate a binary response mask with one entry per response token."""

    if isinstance(response_mask, (str, bytes)) or not isinstance(response_mask, Sequence):
        raise TypeError("response_mask must contain only integer values")
    normalized = tuple(response_mask)
    if len(normalized) != response_length:
        raise ValueError("response_mask must have one entry per response token")
    if any(isinstance(mask, bool) or not isinstance(mask, int) for mask in normalized):
        raise TypeError("response_mask must contain only integer values")
    if any(mask not in (0, 1) for mask in normalized):
        raise ValueError("response_mask values must be either 0 or 1")
    return normalized


def _validate_response_logprobs(
    response_logprobs: object,
    response_length: int,
) -> tuple[float, ...] | None:
    """Validate optional finite log probabilities aligned to response tokens."""

    if response_logprobs is None:
        return None
    if isinstance(response_logprobs, (str, bytes)) or not isinstance(response_logprobs, Sequence):
        raise TypeError("response_logprobs must contain real numbers")
    normalized = tuple(response_logprobs)
    if len(normalized) != response_length:
        raise ValueError("response_logprobs must have one entry per response token")
    for logprob in normalized:
        if isinstance(logprob, bool) or not isinstance(logprob, (int, float)):
            raise TypeError("response_logprobs must contain real numbers")
        if not isfinite(float(logprob)):
            raise ValueError("response_logprobs must be finite")
    return tuple(float(logprob) for logprob in normalized)
