"""Tests for the dependency-free verl agent-loop trajectory boundary."""

import pytest

from remem.environments.base import StepResult
from remem.execution import EpisodeResult, EpisodeStep
from remem.integrations.verl import encode_episode_for_verl


def _episode() -> EpisodeResult:
    steps = (
        EpisodeStep(0, "start", "look", StepResult("middle", 0.0, False, False, {})),
        EpisodeStep(1, "middle", "open door", StepResult("goal", 1.0, True, False, {})),
    )
    return EpisodeResult("start", steps, 1.0, True, False)


def test_encode_episode_matches_verl_agent_loop_contract() -> None:
    trajectory = encode_episode_for_verl(
        _episode(),
        encode_prompt=lambda text: [len(text)],
        encode_completion=lambda text: list(range(1, len(text.splitlines()) + 1)),
        memory_ids=("memory-a",),
    )

    assert trajectory.prompt_ids == (5,)
    assert trajectory.response_ids == (1, 2)
    assert trajectory.response_mask == (1, 1)
    assert trajectory.reward == 1.0
    assert trajectory.metadata["memory_ids"] == ["memory-a"]
    assert trajectory.to_agent_loop_output() == {
        "prompt_ids": [5],
        "response_ids": [1, 2],
        "response_mask": [1, 1],
    }


def test_encode_episode_serializes_offline_training_metadata() -> None:
    trajectory = encode_episode_for_verl(
        _episode(),
        encode_prompt=lambda _: [1, 2],
        encode_completion=lambda _: [3, 4, 5],
    )

    assert trajectory.to_dict() == {
        "prompt_ids": [1, 2],
        "response_ids": [3, 4, 5],
        "response_mask": [1, 1, 1],
        "reward": 1.0,
        "metadata": {
            "memory_ids": [],
            "step_count": 2,
            "terminated": True,
            "truncated": False,
        },
    }


def test_encode_episode_rejects_empty_completion_tokens() -> None:
    with pytest.raises(ValueError, match="at least one token"):
        encode_episode_for_verl(
            _episode(),
            encode_prompt=lambda _: [1],
            encode_completion=lambda _: [],
        )


def test_encode_episode_rejects_non_integer_token_ids() -> None:
    with pytest.raises(TypeError, match="integer token IDs"):
        encode_episode_for_verl(
            _episode(),
            encode_prompt=lambda _: ["not-an-id"],  # type: ignore[list-item]
            encode_completion=lambda _: [1],
        )


def test_encode_episode_rejects_blank_memory_identifiers() -> None:
    with pytest.raises(ValueError, match="non-empty identifiers"):
        encode_episode_for_verl(
            _episode(),
            encode_prompt=lambda _: [1],
            encode_completion=lambda _: [2],
            memory_ids=("",),
        )
