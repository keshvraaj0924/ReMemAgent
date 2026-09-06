"""Dependency-free trajectory encoding for verl agent-loop integration.

The module mirrors the stable token-level contract documented by verl's
``AgentLoopOutput`` without importing verl itself. This keeps the core package
lightweight while giving an external adapter an explicit, testable boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
import numbers

from remem.execution import EpisodeResult
from remem.integrations.grpo import GrpoBatch, GrpoSample

TokenEncoder = Callable[[str], Sequence[int]]


@dataclass(frozen=True, slots=True)
class VerlTrajectory:
    """Token-level trajectory plus research metadata for one completed episode."""

    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    reward: float
    metadata: Mapping[str, object]
    response_logprobs: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Validate and detach the token-level contract for direct construction."""

        _validate_token_ids(self.prompt_ids, "prompt")
        _validate_token_ids(self.response_ids, "response")
        if not self.response_ids:
            raise ValueError("response_ids must contain at least one token")
        _validate_real_number(self.reward, "reward")
        if not isfinite(self.reward):
            raise ValueError("reward must be finite")

        normalized_mask = tuple(self.response_mask)
        if len(normalized_mask) != len(self.response_ids):
            raise ValueError("response_mask must have the same length as response_ids")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in normalized_mask):
            raise TypeError("response_mask must contain only integer values")
        if any(value not in (0, 1) for value in normalized_mask):
            raise ValueError("response_mask values must be either 0 or 1")
        if not any(normalized_mask):
            raise ValueError("response_mask must contain at least one active response token")
        object.__setattr__(self, "response_mask", normalized_mask)

        if self.response_logprobs is not None:
            normalized_logprobs = tuple(self.response_logprobs)
            if len(normalized_logprobs) != len(self.response_ids):
                raise ValueError(
                    "response_logprobs must have the same length as response_ids"
                )
            for logprob in normalized_logprobs:
                _validate_real_number(logprob, "response_logprobs")
                if not isfinite(logprob):
                    raise ValueError("response_logprobs must be finite")
            object.__setattr__(self, "response_logprobs", normalized_logprobs)

        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_agent_loop_output(self) -> dict[str, list[int] | list[float]]:
        """Return fields required by verl plus optional rollout log probabilities."""

        output: dict[str, list[int] | list[float]] = {
            "prompt_ids": list(self.prompt_ids),
            "response_ids": list(self.response_ids),
            "response_mask": list(self.response_mask),
        }
        if self.response_logprobs is not None:
            output["response_logprobs"] = list(self.response_logprobs)
        return output

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible offline-training representation."""

        return {
            **self.to_agent_loop_output(),
            "reward": self.reward,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class VerlTrainingBatch:
    """Ordered token trajectories paired with their GRPO advantages."""

    trajectories: tuple[VerlTrajectory, ...]
    advantages: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate one finite advantage exists for every trajectory."""

        if not self.trajectories:
            raise ValueError("verl training batches must contain at least one trajectory")
        if len(self.trajectories) != len(self.advantages):
            raise ValueError("trajectories and advantages must have equal lengths")
        for advantage in self.advantages:
            _validate_real_number(advantage, "advantages")
            if not isfinite(advantage):
                raise ValueError("advantages must be finite")

    def to_dicts(self) -> tuple[dict[str, object], ...]:
        """Return ordered rows for framework-specific collation."""

        return tuple(
            {**trajectory.to_dict(), "advantage": advantage}
            for trajectory, advantage in zip(self.trajectories, self.advantages, strict=True)
        )


def encode_episode_for_verl(
    episode: EpisodeResult,
    *,
    encode_prompt: TokenEncoder,
    encode_completion: TokenEncoder,
    memory_ids: Sequence[str] = (),
) -> VerlTrajectory:
    """Encode an episode into verl's token-level agent-loop contract.

    Tokenization stays outside ReMemAgent so the integration works with any
    tokenizer and does not introduce a model dependency. The response mask is
    all ones because the current normalized episode stores only agent actions;
    tool-response token masking belongs in a future multi-turn adapter.
    """

    prompt_ids = _validate_token_ids(encode_prompt(episode.initial_observation), "prompt")
    completion = _format_completion(episode)
    response_ids = _validate_token_ids(encode_completion(completion), "response")
    if not response_ids:
        raise ValueError("encode_completion must return at least one token")

    normalized_memory_ids = tuple(memory_ids)
    if any(not memory_id.strip() for memory_id in normalized_memory_ids):
        raise ValueError("memory_ids must contain non-empty identifiers")

    return VerlTrajectory(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=(1,) * len(response_ids),
        reward=episode.total_reward,
        metadata={
            "memory_ids": list(normalized_memory_ids),
            "step_count": len(episode.steps),
            "terminated": episode.terminated,
            "truncated": episode.truncated,
        },
    )


def encode_grpo_batch_for_verl(
    batch: GrpoBatch,
    *,
    encode_prompt: TokenEncoder,
    encode_completion: TokenEncoder,
) -> VerlTrainingBatch:
    """Encode an existing GRPO batch while preserving order and advantages.

    This is the explicit handoff between ReMemAgent's framework-neutral GRPO
    representation and verl's token-level representation. Tokenization is
    injected by the caller; memory provenance is copied from each GRPO sample
    into trajectory metadata, and the already-computed advantage remains paired
    with the same sample.
    """

    trajectories = tuple(
        _encode_grpo_sample_for_verl(
            sample,
            encode_prompt=encode_prompt,
            encode_completion=encode_completion,
        )
        for sample in batch.samples
    )
    return VerlTrainingBatch(trajectories=trajectories, advantages=batch.advantages)


def _encode_grpo_sample_for_verl(
    sample: GrpoSample,
    *,
    encode_prompt: TokenEncoder,
    encode_completion: TokenEncoder,
) -> VerlTrajectory:
    """Encode one framework-neutral GRPO sample into token-level fields."""

    if not sample.prompt.strip():
        raise ValueError("GRPO sample prompt must not be empty")
    if not sample.completion.strip():
        raise ValueError("GRPO sample completion must not be empty")
    if not sample.group_id.strip():
        raise ValueError("GRPO sample group_id must not be empty")

    prompt_ids = _validate_token_ids(encode_prompt(sample.prompt), "prompt")
    response_ids = _validate_token_ids(encode_completion(sample.completion), "response")
    if not response_ids:
        raise ValueError("encode_completion must return at least one token")
    normalized_memory_ids = tuple(sample.memory_ids)
    if any(not memory_id.strip() for memory_id in normalized_memory_ids):
        raise ValueError("memory_ids must contain non-empty identifiers")

    return VerlTrajectory(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=(1,) * len(response_ids),
        reward=sample.reward,
        metadata={
            "group_id": sample.group_id,
            "memory_ids": list(normalized_memory_ids),
        },
    )


def build_verl_training_batch(
    trajectories: Sequence[VerlTrajectory],
    advantages: Sequence[float],
) -> VerlTrainingBatch:
    """Create a validated token-level batch without assuming trainer collation.

    Padding, truncation, tensor conversion, device placement, and distributed
    sharding remain framework concerns. ReMemAgent only guarantees stable
    ordering and one-to-one alignment between encoded trajectories and their
    precomputed GRPO advantages.
    """

    return VerlTrainingBatch(
        trajectories=tuple(trajectories),
        advantages=tuple(advantages),
    )


def _format_completion(episode: EpisodeResult) -> str:
    """Serialize the executed action trajectory deterministically."""

    return "\n".join(step.action for step in episode.steps).strip()


def _validate_token_ids(token_ids: Sequence[int], field_name: str) -> tuple[int, ...]:
    """Validate and normalize non-negative integer tokenizer IDs."""

    normalized = tuple(token_ids)
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in normalized):
        raise TypeError(f"{field_name} tokenizer must return integer token IDs")
    if any(token_id < 0 for token_id in normalized):
        raise ValueError(f"{field_name} tokenizer must return non-negative token IDs")
    return normalized


def _validate_real_number(value: object, field_name: str) -> None:
    """Reject booleans and non-real values at numeric training boundaries."""

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        if field_name == "reward":
            raise TypeError("reward must be a real number")
        raise TypeError(f"{field_name} must be real numbers")
