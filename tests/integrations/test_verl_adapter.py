"""Regression tests for the framework-facing verl adapter."""

from __future__ import annotations

import pytest

from remem.integrations import adapt_agent_loop_output


def test_adapt_agent_loop_output_preserves_tokens_and_metadata() -> None:
    """External token IDs and research provenance survive adaptation unchanged."""

    metadata = {"memory_ids": ["mem-1"], "episode_id": "episode-7"}
    trajectory = adapt_agent_loop_output(
        {
            "prompt_ids": (10, 11),
            "response_ids": (20, 21),
            "response_mask": (1, 0),
        },
        reward=0.75,
        metadata=metadata,
    )

    assert trajectory.prompt_ids == (10, 11)
    assert trajectory.response_ids == (20, 21)
    assert trajectory.response_mask == (1, 0)
    assert trajectory.reward == 0.75
    assert trajectory.metadata == metadata


def test_adapt_agent_loop_output_copies_metadata() -> None:
    """Mutating caller-owned metadata does not mutate the trajectory record."""

    metadata = {"memory_ids": ["mem-1"]}
    trajectory = adapt_agent_loop_output(
        {"prompt_ids": (), "response_ids": (1,), "response_mask": (1,)},
        reward=1.0,
        metadata=metadata,
    )

    metadata["episode_id"] = "episode-8"

    assert "episode_id" not in trajectory.metadata


def test_adapt_agent_loop_output_rejects_non_finite_reward() -> None:
    """NaN and infinity cannot enter an offline training record."""

    with pytest.raises(ValueError, match="reward must be finite"):
        adapt_agent_loop_output(
            {"prompt_ids": (), "response_ids": (1,), "response_mask": (1,)},
            reward=float("nan"),
        )
