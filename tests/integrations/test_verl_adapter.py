"""Regression tests for the framework-facing verl adapter."""

from __future__ import annotations

import asyncio

import pytest

from remem.integrations import (
    AgentLoopRequest,
    adapt_agent_loop_output,
    run_agent_loop,
    run_agent_loop_batch,
)


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


def test_run_agent_loop_forwards_sampling_params_and_dataset_fields() -> None:
    """The async bridge matches the external AgentLoopBase invocation boundary."""

    async def agent_loop(
        sampling_params: dict[str, object],
        **kwargs: object,
    ) -> dict[str, list[int]]:
        assert sampling_params == {"temperature": 0.2}
        assert kwargs == {"raw_prompt": [{"role": "user", "content": "hello"}]}
        return {
            "prompt_ids": [4, 5],
            "response_ids": [8, 9],
            "response_mask": [1, 0],
        }

    trajectory = asyncio.run(
        run_agent_loop(
            agent_loop,
            sampling_params={"temperature": 0.2},
            reward=0.5,
            metadata={"episode_id": "episode-9"},
            raw_prompt=[{"role": "user", "content": "hello"}],
        )
    )

    assert trajectory.prompt_ids == (4, 5)
    assert trajectory.response_ids == (8, 9)
    assert trajectory.response_mask == (1, 0)
    assert trajectory.reward == 0.5


def test_run_agent_loop_rejects_invalid_external_output() -> None:
    """Invalid token contracts are rejected after the external loop completes."""

    async def agent_loop(
        sampling_params: dict[str, object],
        **kwargs: object,
    ) -> dict[str, list[int]]:
        return {
            "prompt_ids": [4, -1],
            "response_ids": [8],
            "response_mask": [1],
        }

    with pytest.raises(ValueError, match="prompt_ids must contain non-negative integer token IDs"):
        asyncio.run(run_agent_loop(agent_loop, sampling_params={}, reward=0.0))


def test_run_agent_loop_batch_preserves_request_order() -> None:
    """Concurrent completion order cannot change the deterministic batch order."""

    async def agent_loop(
        sampling_params: dict[str, object],
        **kwargs: object,
    ) -> dict[str, list[int]]:
        await asyncio.sleep(float(kwargs["delay"]))
        token_id = int(sampling_params["token_id"])
        return {
            "prompt_ids": [token_id],
            "response_ids": [token_id + 10],
            "response_mask": [1],
        }

    requests = (
        AgentLoopRequest(sampling_params={"token_id": 1}, reward=0.1, kwargs={"delay": 0.02}),
        AgentLoopRequest(sampling_params={"token_id": 2}, reward=0.2, kwargs={"delay": 0.0}),
    )

    trajectories = asyncio.run(run_agent_loop_batch(agent_loop, requests, max_concurrency=2))

    assert [trajectory.prompt_ids for trajectory in trajectories] == [(1,), (2,)]
    assert [trajectory.reward for trajectory in trajectories] == [0.1, 0.2]


def test_run_agent_loop_batch_rejects_non_positive_concurrency() -> None:
    """An invalid concurrency limit fails before external execution begins."""

    async def agent_loop(
        sampling_params: dict[str, object],
        **kwargs: object,
    ) -> dict[str, list[int]]:
        return {"prompt_ids": [], "response_ids": [1], "response_mask": [1]}

    with pytest.raises(ValueError, match="max_concurrency must be positive"):
        asyncio.run(run_agent_loop_batch(agent_loop, (), max_concurrency=0))
