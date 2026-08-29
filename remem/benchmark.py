"""Benchmark-suite orchestration over normalized environment adapters.

This module deliberately does not import ALFWorld, WebShop, or any model SDK.
Experiments provide concrete environment and policy factories, while the suite
runner provides deterministic lifecycle, memory persistence, and aggregate
reporting shared by benchmark implementations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from remem.environments.base import EnvironmentAdapter
from remem.execution import EpisodeResult, Policy
from remem.memory.store import MemoryStore
from remem.services import EpisodeExecutionResult, EpisodeExecutionService, SuccessEvaluator

EnvironmentFactory = Callable[[int], EnvironmentAdapter]
PolicyFactory = Callable[[int, MemoryStore], Policy]


@dataclass(frozen=True, slots=True)
class BenchmarkEpisodeReport:
    """Immutable report for one benchmark episode."""

    episode_id: str
    episode: EpisodeResult
    episode_success: bool
    retained_memory_count: int


@dataclass(frozen=True, slots=True)
class BenchmarkRunReport:
    """Aggregate report for a benchmark suite run."""

    benchmark_name: str
    episodes: tuple[BenchmarkEpisodeReport, ...]
    final_memory_count: int

    @property
    def success_count(self) -> int:
        """Return the number of successful episodes."""

        return sum(episode.episode_success for episode in self.episodes)

    @property
    def success_rate(self) -> float:
        """Return the observed episode success rate, or zero when empty."""

        if not self.episodes:
            return 0.0
        return self.success_count / len(self.episodes)

    @property
    def mean_reward(self) -> float:
        """Return the arithmetic mean episode reward, or zero when empty."""

        if not self.episodes:
            return 0.0
        return sum(episode.episode.total_reward for episode in self.episodes) / len(self.episodes)


class BenchmarkSuiteRunner:
    """Run multiple episodes while sharing one memory store across episodes."""

    def __init__(self, execution_service: EpisodeExecutionService | None = None) -> None:
        """Create a suite runner with an injectable execution service."""

        self.execution_service = execution_service or EpisodeExecutionService()

    def run(
        self,
        *,
        benchmark_name: str,
        episode_count: int,
        max_steps: int,
        environment_factory: EnvironmentFactory,
        policy_factory: PolicyFactory,
        success_evaluator: SuccessEvaluator,
        store: MemoryStore | None = None,
        reset_kwargs: dict[str, Any] | None = None,
    ) -> BenchmarkRunReport:
        """Execute a benchmark suite and persist trajectory memories.

        The same ``MemoryStore`` is shared across episodes so a policy factory
        can deliberately expose accumulated experience to later episodes.
        Environment instances are created per episode and closed when they
        provide a callable ``close`` method.
        """

        normalized_name = benchmark_name.strip()
        if not normalized_name:
            raise ValueError("benchmark_name must not be empty")
        if episode_count < 0:
            raise ValueError("episode_count must be non-negative")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        memory_store = store or MemoryStore()
        reports: list[BenchmarkEpisodeReport] = []

        for episode_index in range(episode_count):
            environment = environment_factory(episode_index)
            try:
                policy = policy_factory(episode_index, memory_store)
                execution_result = self.execution_service.execute_and_ingest(
                    environment,
                    policy,
                    memory_store,
                    episode_id=f"{normalized_name}:{episode_index}",
                    max_steps=max_steps,
                    success_evaluator=success_evaluator,
                    reset_kwargs=reset_kwargs,
                )
            finally:
                _close_environment(environment)

            reports.append(_build_episode_report(execution_result))

        return BenchmarkRunReport(
            benchmark_name=normalized_name,
            episodes=tuple(reports),
            final_memory_count=len(memory_store.all()),
        )


def _build_episode_report(result: EpisodeExecutionResult) -> BenchmarkEpisodeReport:
    """Convert an execution result into the stable benchmark report contract."""

    return BenchmarkEpisodeReport(
        episode_id=result.episode_id,
        episode=result.episode,
        episode_success=result.episode_success,
        retained_memory_count=len(result.ingestion.retained_memories),
    )


def _close_environment(environment: EnvironmentAdapter) -> None:
    """Close an adapter when its concrete implementation supports cleanup."""

    close = getattr(environment, "close", None)
    if callable(close):
        close()
