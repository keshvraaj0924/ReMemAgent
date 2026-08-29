"""Tests for the composed episode execution service."""

import pytest

from remem.environments.base import StepResult
from remem.memory import MemoryGuidedPolicy, MemoryStore
from remem.services import EpisodeExecutionService


class FakeEnvironment:
    """Deterministic environment for service-level tests."""

    def __init__(self, initial: str = "start", goal: str = "goal") -> None:
        self.initial = initial
        self.goal = goal
        self.actions: list[str] = []

    def reset(self, **_kwargs: object) -> str:
        """Return the configured initial observation."""

        return self.initial

    def step(self, action: str) -> StepResult:
        """Return a terminal transition after recording the action."""

        self.actions.append(action)
        return StepResult(self.goal, 1.0, True, False, {})

    def close(self) -> None:
        """Close the test environment."""


def test_service_executes_policy_and_ingests_trajectory() -> None:
    environment = FakeEnvironment()
    store = MemoryStore()

    result = EpisodeExecutionService().execute_and_ingest(
        environment,
        lambda observation: f"act:{observation}",
        store,
        episode_id="episode-1",
        max_steps=3,
        success_evaluator=lambda episode: episode.total_reward > 0,
    )

    assert result.episode_id == "episode-1"
    assert result.episode_success is True
    assert result.episode.total_reward == 1.0
    assert len(result.ingestion.retained_memories) == 1
    assert len(store.all()) == 1
    assert environment.actions == ["act:start"]


def test_service_preserves_failed_episode_attribution() -> None:
    store = MemoryStore()

    result = EpisodeExecutionService().execute_and_ingest(
        FakeEnvironment(),
        lambda _: "stop",
        store,
        episode_id="failed",
        max_steps=2,
        success_evaluator=lambda _: False,
    )

    memory = result.ingestion.retained_memories[0]
    assert result.episode_success is False
    assert memory.successes == 0
    assert memory.failures == 1


def test_service_rejects_blank_episode_id_before_execution() -> None:
    with pytest.raises(ValueError, match="episode_id"):
        EpisodeExecutionService().execute_and_ingest(
            FakeEnvironment(),
            lambda _: "stop",
            MemoryStore(),
            episode_id="   ",
            max_steps=1,
            success_evaluator=lambda _: True,
        )


def test_stored_episode_becomes_guidance_for_later_execution() -> None:
    store = MemoryStore()
    service = EpisodeExecutionService()

    service.execute_and_ingest(
        FakeEnvironment(initial="The cabinet is closed.", goal="The cabinet opened."),
        lambda _: "open cabinet",
        store,
        episode_id="episode-source",
        max_steps=1,
        success_evaluator=lambda _: True,
    )

    guidance_seen: list[str] = []

    def action_policy(_state: str, guidance: str) -> str:
        guidance_seen.append(guidance)
        return "open cabinet"

    guided_policy = MemoryGuidedPolicy(store, action_policy)
    result = service.execute_and_ingest(
        FakeEnvironment(initial="The cabinet is closed.", goal="The cabinet opened."),
        guided_policy,
        store,
        episode_id="episode-guided",
        max_steps=1,
        success_evaluator=lambda _: True,
    )

    assert result.episode_success is True
    assert guidance_seen
    assert "Relevant action: open cabinet" in guidance_seen[0]
