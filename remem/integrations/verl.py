"""Dependency-free trajectory encoding for verl agent-loop integration.

The module mirrors the stable token-level contract documented by verl's
``AgentLoopOutput`` without importing verl itself. This keeps the core package
lightweight while giving an external adapter an explicit, testable boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from remem.execution import EpisodeResult
from remem.integrations.grpo import GrpoBatch

TokenEncoder = Callable[[str], Sequence[int]]


@dataclass(frozen=True, slots=True)
class VerlTrajectory:
    """Token-level trajectory plus research metadata for one completed episode."""

    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    reward: float
    metadata: Mapping[str, object]

    def to_agent_loop_output(self) -> dict[str, list[int]]:
        """Return the fields required by verl's ``AgentLoopOutput`` contract."""

        return {
            "prompt_ids": list(self.prompt_ids),
            "response_ids": list(self.response_ids),
            "response_mask": list(self.response_mask),
        }

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
        """Validate one advantage exists for every encoded trajectory."""

        if not self.trajectories:
            raise ValueError("verl training batches must contain at least one trajectory")
        if len(self.trajectories) != len(self.advantages):
            raise ValueError("trajectories and advantages must have equal lengths")

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


def build_verl_training_batch(
    batch: GrpoBatch,
    *,
    encode_prompt: TokenEncoder,
    encode_completion: TokenEncoder,
) -> VerlTrainingBatch:
    """Encode a validated GRPO batch for a framework-specific verl adapter.

    Sample order is preserved exactly, so each GRPO advantage remains attached
    to the trajectory from which it was computed. Memory identifiers already
    carried by each sample become trajectory metadata. The function does not
    pad, truncate, or move tensors because those operations depend on the
    external trainer's collation and device strategy.
    """

    trajectories = tuple(
        encode_episode_for_verl(
            _episode_from_sample(sample),
            encode_prompt=encode_prompt,
            encode_completion=encode_completion,
            memory_ids=sample.memory_ids,
        )
        for sample in batch.samples
    )
    return VerlTrainingBatch(trajectories=trajectories, advantages=batch.advantages)


def _episode_from_sample(sample: object) -> EpisodeResult:
    """Reject unsupported conversion rather than inventing an episode history."""

    raise TypeError(
        "build_verl_training_batch requires EpisodeResult-backed GRPO samples; "
        "use build_verl_training_samples for the current text-only boundary"
    )


def _format_completion(episode: EpisodeResult) -> str:
    """Serialize the executed action trajectory deterministically."""

    return "\n".join(step.action for step in episode.steps).strip()


def _validate_token_ids(token_ids: Sequence[int], field_name: str) -> tuple[int, ...]:
    """Validate and normalize a tokenizer result into immutable integer IDs."""

    normalized = tuple(token_ids)
    if any(not isinstance(token_id, int) for token_id in normalized):
        raise TypeError(f"{field_name} tokenizer must return integer token IDs")
    return normalized
