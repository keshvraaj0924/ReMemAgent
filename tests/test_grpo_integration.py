"""Tests for framework-neutral GRPO trajectory conversion."""

import pytest

from remem.execution import EpisodeResult, EpisodeStep
from remem.environments.base import StepResult
from remem.integrations.grpo import (
    GrpoBatch,
    GrpoSample,
    build_grpo_batch,
    build_grpo_samples,
    compute_group_relative_advantages,
)
from remem.memory.policy import MemoryGuidanceDecision


def _episode(reward: float = 1.0) -> EpisodeResult:
    steps = (
        EpisodeStep(0, "start", "look", StepResult("middle", 0.0, False, False, {})),
        EpisodeStep(1, "middle", "open door", StepResult("goal", reward, True, False, {})),
    )
    return EpisodeResult("start", steps, reward, True, False)


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


def test_compute_group_relative_advantages_normalizes_rewards() -> None:
    samples = (
        GrpoSample("prompt", "a", 1.0, "task-1", ()),
        GrpoSample("prompt", "b", 2.0, "task-1", ()),
        GrpoSample("prompt", "c", 3.0, "task-1", ()),
    )

    advantages = compute_group_relative_advantages(samples)

    assert advantages[0] == pytest.approx(-1.2247448714)
    assert advantages[1] == pytest.approx(0.0)
    assert advantages[2] == pytest.approx(1.2247448714)


def test_compute_group_relative_advantages_keeps_groups_independent() -> None:
    samples = (
        GrpoSample("prompt", "a", 0.0, "task-1", ()),
        GrpoSample("prompt", "b", 2.0, "task-1", ()),
        GrpoSample("prompt", "c", 10.0, "task-2", ()),
        GrpoSample("prompt", "d", 14.0, "task-2", ()),
    )

    advantages = compute_group_relative_advantages(samples)

    assert advantages == pytest.approx((-1.0, 1.0, -1.0, 1.0))


def test_compute_group_relative_advantages_zeroes_constant_groups() -> None:
    samples = (
        GrpoSample("prompt", "a", 4.0, "task-1", ()),
        GrpoSample("prompt", "b", 4.0, "task-1", ()),
    )

    assert compute_group_relative_advantages(samples) == (0.0, 0.0)


def test_compute_group_relative_advantages_rejects_non_positive_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon must be positive"):
        compute_group_relative_advantages((GrpoSample("p", "a", 1.0, "g", ()),), epsilon=0.0)


def test_build_grpo_batch_keeps_sample_advantage_alignment() -> None:
    samples = build_grpo_samples(
        [_episode(1.0), _episode(3.0)],
        group_id_builder=lambda _index, _: "shared",
    )

    batch = build_grpo_batch(samples)

    assert isinstance(batch, GrpoBatch)
    assert batch.samples == samples
    assert batch.advantages == pytest.approx((-1.0, 1.0))
    assert batch.to_dicts()[0]["advantage"] == pytest.approx(-1.0)


def test_grpo_batch_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="zero samples"):
        build_grpo_batch([])


def test_grpo_batch_rejects_mismatched_lengths() -> None:
    samples = build_grpo_samples([_episode()])

    with pytest.raises(ValueError, match="equal lengths"):
        GrpoBatch(samples=samples, advantages=())
