"""Tests for framework-neutral GRPO trajectory conversion."""

import pytest

from remem.execution import EpisodeResult, EpisodeStep
from remem.environments.base import StepResult
from remem.integrations.grpo import build_grpo_samples
from remem.memory.policy import MemoryGuidanceDecision


def _episode() -> EpisodeResult:
    steps = (
        EpisodeStep(0, "start", "look", StepResult("middle", 0.0, False, False, {})),
        EpisodeStep(1, "middle", "open door", StepResult("goal", 1.0, True, False, {})),
    )
    return EpisodeResult("start", steps, 1.0, True, False)


def test_build_grpo_sample_preserves_actions_reward_and_memory_metadata() -> None:
    episode = _episode()
    decisions = (
        MemoryGuidanceDecision("memory-a", "look around", 0.8, 0.9),
        MemoryGuidanceDecision(None, "", 0.0, 0.0),
    )

    samples = build_grpo_samples([episode], decision_histories=[decisions])

    assert samples[0].prompt == "start"
    assert samples[0].completion == "look\nopen door"
    assert samples[0].reward == 1.0
    assert samples[0].group_id == "episode-0"
    assert samples[0].memory_ids == ("memory-a",)
    assert samples[0].to_dict()["memory_ids"] == ["memory-a"]


def test_build_grpo_samples_supports_memory_free_baselines() -> None:
    sample = build_grpo_samples([_episode()])[0]

    assert sample.memory_ids == ()


def test_build_grpo_samples_rejects_misaligned_histories() -> None:
    with pytest.raises(ValueError, match="one history per episode"):
        build_grpo_samples([_episode()], decision_histories=[])

    decisions = [MemoryGuidanceDecision(None, "", 0.0, 0.0)]
    with pytest.raises(ValueError, match="one entry per episode step"):
        build_grpo_samples([_episode()], decision_histories=[decisions])


def test_build_grpo_samples_allows_shared_group_ids() -> None:
    episodes = [_episode(), _episode()]

    samples = build_grpo_samples(
        episodes,
        group_id_builder=lambda index, _: "task-7",
    )

    assert [sample.group_id for sample in samples] == ["task-7", "task-7"]
