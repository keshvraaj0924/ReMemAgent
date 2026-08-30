"""Framework-neutral trajectory records for GRPO-style training loops.

The integration deliberately has no dependency on GRPO or verl. It converts a
completed ReMemAgent episode into the small prompt/completion/reward/group
contract that training frameworks can map to their own dataset schemas. Memory
identifiers are retained as metadata so experiments can analyze whether
training examples were memory-guided.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import sqrt

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


def compute_group_relative_advantages(
    samples: Sequence[GrpoSample],
    *,
    epsilon: float = 1e-8,
) -> tuple[float, ...]:
    """Compute deterministic group-normalized rewards for GRPO-style training.

    For each group, rewards are centered by the group mean and normalized by
    the population standard deviation. A zero-variance group receives zero
    advantages rather than producing NaNs. The returned tuple preserves the
    input sample order so callers can attach each value to the corresponding
    training row without reordering the dataset.
    """

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    rewards_by_group: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        rewards_by_group[sample.group_id].append(sample.reward)

    means: dict[str, float] = {}
    standard_deviations: dict[str, float] = {}
    for group_id, rewards in rewards_by_group.items():
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        means[group_id] = mean
        standard_deviations[group_id] = sqrt(variance)

    return tuple(
        (sample.reward - means[sample.group_id])
        / max(standard_deviations[sample.group_id], epsilon)
        if standard_deviations[sample.group_id] > 0.0
        else 0.0
        for sample in samples
    )


def _default_prompt_builder(episode: EpisodeResult) -> str:
    """Use the initial environment observation as the training prompt."""

    return episode.initial_observation


def _default_group_id_builder(index: int, _: EpisodeResult) -> str:
    """Assign each episode to its own group unless an experiment overrides it."""

    return f"episode-{index}"


def _format_completion(episode: EpisodeResult) -> str:
    """Serialize the action trajectory without introducing model-specific syntax."""

    return "\n".join(step.action for step in episode.steps).strip()
