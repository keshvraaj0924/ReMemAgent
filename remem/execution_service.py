"""End-to-end execution and memory ingestion orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from remem.environments.base import EnvironmentAdapter
from remem.memory.episode import EpisodeMemoryAttribution
from remem.memory.ingestion import EpisodeMemoryIngestor, MemoryIngestionResult
from remem.memory.store import MemoryStore

from .execution import EpisodeResult, EpisodeRunner, Policy

EpisodeSuccessEvaluator = Callable[[EpisodeResult], bool]


@dataclass(frozen=True, slots=True)
class EpisodeExecutionResult:
    """Immutable result of execution followed by memory ingestion."""

    episode: EpisodeResult
    ingestion: MemoryIngestionResult


class EpisodeExecutionService:
    """Run an episode and persist its attributed trajectory memories."""

    def __init__(
        self,
        *,
        runner: EpisodeRunner | None = None,
        ingestor: EpisodeMemoryIngestor | None = None,
        success_evaluator: EpisodeSuccessEvaluator,
    ) -> None:
        """Create an orchestration service from replaceable components."""

        self.runner = runner or EpisodeRunner()
        self.ingestor = ingestor or EpisodeMemoryIngestor()
        self.success_evaluator = success_evaluator

    def execute_and_record(
        self,
        environment: EnvironmentAdapter,
        policy: Policy,
        store: MemoryStore,
        *,
        episode_id: str,
        max_steps: int,
        reset_kwargs: dict[str, Any] | None = None,
    ) -> EpisodeExecutionResult:
        """Execute one episode, evaluate its outcome, and ingest its memories."""

        normalized_episode_id = episode_id.strip()
        if not normalized_episode_id:
            raise ValueError("episode_id must not be empty")

        episode = self.runner.run(
            environment,
            policy,
            max_steps=max_steps,
            reset_kwargs=reset_kwargs,
        )
        attribution = EpisodeMemoryAttribution(
            episode_success=self.success_evaluator(episode),
        )
        ingestion = self.ingestor.ingest(
            store,
            episode_id=normalized_episode_id,
            episode=episode,
            attribution=attribution,
        )
        return EpisodeExecutionResult(episode=episode, ingestion=ingestion)
