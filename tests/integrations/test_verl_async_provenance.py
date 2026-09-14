"""Regression coverage for async verl provenance ownership."""

from __future__ import annotations

import asyncio

import pytest

from remem.integrations import run_agent_loop


def test_run_agent_loop_snapshots_metadata_before_external_await() -> None:
    """Concurrent caller mutation must not rewrite dispatch-time provenance."""

    memory_ids = ["memory-1"]
    metadata: dict[str, object] = {
        "episode_id": "episode-1",
        "memory_ids": memory_ids,
    }

    async def agent_loop(
        sampling_params: dict[str, object],
        **kwargs: object,
    ) -> dict[str, list[int]]:
        assert sampling_params == {"temperature": 0.2}
        assert kwargs == {"raw_prompt": [{"role": "user", "content": "hello"}]}

        metadata["episode_id"] = "episode-mutated"
        memory_ids.append("memory-2")
        await asyncio.sleep(0)

        return {
            "prompt_ids": [1],
            "response_ids": [2],
            "response_mask": [1],
        }

    trajectory = asyncio.run(
        run_agent_loop(
            agent_loop,
            sampling_params={"temperature": 0.2},
            reward=1.0,
            metadata=metadata,
            raw_prompt=[{"role": "user", "content": "hello"}],
        )
    )

    assert trajectory.metadata == {
        "episode_id": "episode-1",
        "memory_ids": ["memory-1"],
    }
    assert metadata == {
        "episode_id": "episode-mutated",
        "memory_ids": ["memory-1", "memory-2"],
    }


def test_run_agent_loop_rejects_invalid_reward_before_external_execution() -> None:
    """Invalid research rewards must fail before an external rollout starts."""

    execution_count = 0

    async def agent_loop(
        sampling_params: dict[str, object],
        **kwargs: object,
    ) -> dict[str, list[int]]:
        nonlocal execution_count
        execution_count += 1
        return {
            "prompt_ids": [1],
            "response_ids": [2],
            "response_mask": [1],
        }

    with pytest.raises(ValueError, match="reward must be finite"):
        asyncio.run(
            run_agent_loop(
                agent_loop,
                sampling_params={"temperature": 0.2},
                reward=float("nan"),
            )
        )

    assert execution_count == 0
