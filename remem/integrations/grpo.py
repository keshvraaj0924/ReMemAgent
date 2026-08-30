"""Framework-neutral trajectory records for GRPO-style training loops.

The integration deliberately has no dependency on GRPO or verl. It converts a
completed ReMemAgent episode into the small prompt/completion/reward contract
that training frameworks can map to their own dataset schemas. Memory
identifiers are retained as metadata so experiments can analyze whether
training examples were memory-guided.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from remem.execution import EpisodeResult
from remem.memory.policy import MemoryGuidanceDecision

PromptBuilder = Callable[[EpisodeResult], str]
GroupIdBuilder = Callable[[int, EpisodeResult], str]


@dataclass(frozen=True, slots=True)
class GrpoSample:
    """One completed episode represented as a GRPO-compatible training sample."""

    prompt: str
    completion: str
    reward: float
    group_id: str
    memory_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation for dataset writers."""

        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "reward": self.reward,
            "group_id": self.group_id,
            "memory_ids": list(self.memory_ids),
        }


def build_grpo_samples(
    episodes: Sequence[EpisodeResult],
    *,
    prompt_builder: PromptBuilder | None = None,
    decision_histories: Sequence[Sequence[MemoryGuidanceDecision]] | None = None,
    group_id_builder: GroupIdBuilder | None = None,
) -> tuple[GrpoSample, ...]:
    """Convert completed episodes into deterministic GRPO training samples.

    ``decision_histories`` is optional because memory-free baselines are valid
    training inputs. When supplied, it must contain exactly one history per
    episode and each history must align with that episode's executed steps.
    Group identifiers default to ``episode-{index}``; callers can override this
    when multiple completions should share a GRPO group.
    """

    if decision_histories is not None and len(decision_histories) != len(episodes):
        raise ValueError("decision_histories must contain one history per episode")

    build_prompt = prompt_builder or _default_prompt_builder
    build_group_id = group_id_builder or _default_group_id_builder
    samples: list[GrpoSample] = []

    for index, episode in enumerate(episodes):
        prompt = build_prompt(episode).strip()
        if not prompt:
            raise ValueError("prompt_builder must return a non-empty prompt")

        completion = _format_completion(episode)
        if not completion:
            raise ValueError("episode must contain at least one action")

        group_id = build_group_id(index, episode).strip()
        if not group_id:
            raise ValueError("group_id_builder must return a non-empty group id")

        memory_ids: tuple[str, ...] = ()
        if decision_histories is not None:
            history = decision_histories[index]
            if len(history) != len(episode.steps):
                raise ValueError("decision history must contain one entry per episode step")
            memory_ids = tuple(
                decision.memory_id for decision in history if decision.memory_id is not None
            )

        samples.append(
            GrpoSample(
                prompt=prompt,
                completion=completion,
                reward=episode.total_reward,
                group_id=group_id,
                memory_ids=memory_ids,
            )
        )

    return tuple(samples)


def _default_prompt_builder(episode: EpisodeResult) -> str:
    """Use the initial environment observation as the training prompt."""

    return episode.initial_observation


def _default_group_id_builder(index: int, _: EpisodeResult) -> str:
    """Assign each episode to its own group unless an experiment overrides it."""

    return f"episode-{index}"


def _format_completion(episode: EpisodeResult) -> str:
    """Serialize the action trajectory without introducing model-specific syntax."""

    return "\n".join(step.action for step in episode.steps).strip()
