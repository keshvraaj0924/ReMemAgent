"""Application services that compose benchmark execution and memory workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from remem.environments.base import EnvironmentAdapter
from remem.execution import EpisodeResult, EpisodeRunner, Policy
from remem.memory.episode import EpisodeMemoryAttribution
from remem.memory.ingestion import EpisodeMemoryIngestor, MemoryIngestionResult
from remem.memory.store import MemoryStore

SuccessEvaluator = Callable[[EpisodeResult], bool]


@dataclass(frozen=True, slots=True)
class EpisodeExecutionResult:
    """Outcome of executing one policy and ingesting its trajectory."""

    episode_id: str
    episode: EpisodeResult
    ingestion: MemoryIngestionResult
    episode_success: bool


class EpisodeExecutionService:
    """Compose execution and memory ingestion without benchmark-specific policy."""

    def __init__(
        self,
        *,
        runner: EpisodeRunner | None = None,
        ingestor: EpisodeMemoryIngestor | None = None,
    ) -> None:
        """Create a service from replaceable execution and ingestion components."""

        self.runner = runner or EpisodeRunner()
        self.ingestor = ingestor or EpisodeMemoryIngestor()

    def execute_and_ingest(
        self,
        environment: EnvironmentAdapter,
        policy: Policy,
        store: MemoryStore,
        *,
        episode_id: str,
        max_steps: int,
        success_evaluator: SuccessEvaluator,
        reset_kwargs: dict[str, Any] | None = None,
    ) -> EpisodeExecutionResult:
        """Execute one episode, attribute its outcome, and ingest its memories."""

        normalized_episode_id = episode_id.strip()
        if not normalized_episode_id:
            raise ValueError("episode_id must not be empty")

        episode = self.runner.run(
            environment,
            policy,
            max_steps=max_steps,
            reset_kwargs=reset_kwargs,
        )
        episode_success = bool(success_evaluator(episode))
        ingestion = self.ingestor.ingest(
            store,
            episode_id=normalized_episode_id,
            episode=episode,
            attribution=EpisodeMemoryAttribution(episode_success=episode_success),
        )
        return EpisodeExecutionResult(
            episode_id=normalized_episode_id,
            episode=episode,
            ingestion=ingestion,
            episode_success=episode_success,
        )
