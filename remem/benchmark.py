"""Benchmark-suite orchestration over normalized environment adapters.

This module deliberately does not import ALFWorld, WebShop, or any model SDK.
Experiments provide concrete environment and policy factories, while the suite
runner provides deterministic lifecycle, memory persistence, transfer tracing,
aggregate reporting, and optional low-dependency observability.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from remem.environments.base import EnvironmentAdapter
from remem.execution import EpisodeResult, Policy
from remem.memory.attribution import (
    MemoryTransferOutcome,
    MemoryTransferRecorder,
    TransferSuccessEvaluator,
)
from remem.memory.policy import MemoryGuidedPolicy
from remem.memory.store import MemoryStore
from remem.observability import ObservationCollector
from remem.services import EpisodeExecutionResult, EpisodeExecutionService, SuccessEvaluator

EnvironmentFactory = Callable[[int], EnvironmentAdapter]
PolicyFactory = Callable[[int, MemoryStore], Policy]


@dataclass(frozen=True, slots=True)
class BenchmarkRunConfiguration:
    """Reproducibility metadata describing one benchmark suite invocation."""

    benchmark_name: str
    episode_count: int
    max_steps: int
    seed: int | None
    environment_factory: str | None = None
    policy_factory: str | None = None
    success_evaluator: str | None = None
    transfer_success_evaluator: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkEpisodeReport:
    """Immutable report for one benchmark episode."""

    episode_id: str
    episode: EpisodeResult
    episode_success: bool
    retained_memory_count: int
    transfer_outcomes: tuple[MemoryTransferOutcome, ...] = ()

    @property
    def transfer_count(self) -> int:
        """Return the number of memory selections attributed in this episode."""

        return len(self.transfer_outcomes)

    @property
    def transfer_success_count(self) -> int:
        """Return the number of successful attributed memory transfers."""

        return sum(outcome.success for outcome in self.transfer_outcomes)


@dataclass(frozen=True, slots=True)
class BenchmarkRunReport:
    """Aggregate report for a benchmark suite run."""

    benchmark_name: str
    episodes: tuple[BenchmarkEpisodeReport, ...]
    final_memory_count: int
    seed: int | None = None
    configuration: BenchmarkRunConfiguration | None = None

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

    @property
    def transfer_count(self) -> int:
        """Return the total number of attributed memory transfers."""

        return sum(episode.transfer_count for episode in self.episodes)

    @property
    def transfer_success_rate(self) -> float:
        """Return observed transfer success, or zero when no transfers occurred."""

        if not self.transfer_count:
            return 0.0
        return (
            sum(episode.transfer_success_count for episode in self.episodes) / self.transfer_count
        )


class BenchmarkSuiteRunner:
    """Run multiple episodes while sharing one memory store across episodes."""

    def __init__(
        self,
        execution_service: EpisodeExecutionService | None = None,
        transfer_recorder: MemoryTransferRecorder | None = None,
        observation_collector: ObservationCollector | None = None,
    ) -> None:
        """Create a suite runner with injectable execution and observability services."""

        self.execution_service = execution_service or EpisodeExecutionService()
        self.transfer_recorder = transfer_recorder or MemoryTransferRecorder()
        self.observation_collector = observation_collector

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
        transfer_success_evaluator: TransferSuccessEvaluator | None = None,
        seed: int | None = None,
        configuration: BenchmarkRunConfiguration | None = None,
    ) -> BenchmarkRunReport:
        """Execute a benchmark suite, persist memories, and trace guided transfers.

        Policies created as :class:`MemoryGuidedPolicy` expose one guidance
        decision per executed step. Those decisions are attributed after the
        episode using the supplied transfer evaluator. Non-memory-guided
        policies remain fully supported and produce no transfer outcomes.
        When an :class:`ObservationCollector` is configured, the runner records
        suite and episode counters plus aggregate episode duration without
        changing benchmark behavior.

        If ``seed`` is supplied, factories receive ``seed + episode_index`` as
        their episode seed. This gives externally owned benchmark integrations a
        deterministic seed contract without introducing a global random-state
        dependency. When omitted, factories retain the historical episode-index
        argument.
        """

        normalized_name = benchmark_name.strip()
        if not normalized_name:
            raise ValueError("benchmark_name must not be empty")
        if episode_count < 0:
            raise ValueError("episode_count must be non-negative")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        memory_store = store if store is not None else MemoryStore()
        reports: list[BenchmarkEpisodeReport] = []
        if self.observation_collector is not None:
            self.observation_collector.increment("benchmark.runs")

        for episode_index in range(episode_count):
            factory_seed = episode_index if seed is None else seed + episode_index
            environment = environment_factory(factory_seed)
            if self.observation_collector is not None:
                self.observation_collector.increment("benchmark.episodes.started")
            try:
                if self.observation_collector is None:
                    execution_result = self._execute_episode(
                        environment,
                        memory_store,
                        factory_seed,
                        episode_index,
                        normalized_name,
                        max_steps,
                        policy_factory,
                        success_evaluator,
                        reset_kwargs,
                        transfer_success_evaluator,
                    )
                else:
                    with self.observation_collector.timed("benchmark.episode.duration_seconds"):
                        execution_result = self._execute_episode(
                            environment,
                            memory_store,
                            factory_seed,
                            episode_index,
                            normalized_name,
                            max_steps,
                            policy_factory,
                            success_evaluator,
                            reset_kwargs,
                            transfer_success_evaluator,
                        )
            finally:
                _close_environment(environment)

            transfer_outcomes = execution_result[1]
            if self.observation_collector is not None:
                self.observation_collector.increment("benchmark.episodes.completed")
                self.observation_collector.increment(
                    "benchmark.transfers.attributed",
                    float(len(transfer_outcomes)),
                )
                self.observation_collector.increment(
                    "benchmark.episodes.successful",
                    float(execution_result[0].episode_success),
                )
            reports.append(_build_episode_report(execution_result[0], transfer_outcomes))

        return BenchmarkRunReport(
            benchmark_name=normalized_name,
            episodes=tuple(reports),
            final_memory_count=len(memory_store.all()),
            seed=seed,
            configuration=configuration
            or BenchmarkRunConfiguration(
                benchmark_name=normalized_name,
                episode_count=episode_count,
                max_steps=max_steps,
                seed=seed,
            ),
        )

    def _execute_episode(
        self,
        environment: EnvironmentAdapter,
        memory_store: MemoryStore,
        factory_seed: int,
        episode_index: int,
        benchmark_name: str,
        max_steps: int,
        policy_factory: PolicyFactory,
        success_evaluator: SuccessEvaluator,
        reset_kwargs: dict[str, Any] | None,
        transfer_success_evaluator: TransferSuccessEvaluator | None,
    ) -> tuple[EpisodeExecutionResult, tuple[MemoryTransferOutcome, ...]]:
        """Execute one episode and attribute memory transfers."""

        policy = policy_factory(factory_seed, memory_store)
        execution_result = self.execution_service.execute_and_ingest(
            environment,
            policy,
            memory_store,
            episode_id=f"{benchmark_name}:{episode_index}",
            max_steps=max_steps,
            success_evaluator=success_evaluator,
            reset_kwargs=reset_kwargs,
        )
        transfer_outcomes = _record_transfer_outcomes(
            self.transfer_recorder,
            memory_store,
            policy,
            execution_result,
            success_evaluator=transfer_success_evaluator,
        )
        return execution_result, transfer_outcomes


def _record_transfer_outcomes(
    recorder: MemoryTransferRecorder,
    store: MemoryStore,
    policy: Policy,
    result: EpisodeExecutionResult,
    *,
    success_evaluator: TransferSuccessEvaluator | None,
) -> tuple[MemoryTransferOutcome, ...]:
    """Attribute traced decisions only for policies that provide memory guidance."""

    if not isinstance(policy, MemoryGuidedPolicy):
        return ()
    return recorder.record_episode(
        store,
        policy.decision_history,
        result.episode,
        success_evaluator=success_evaluator,
    )


def _build_episode_report(
    result: EpisodeExecutionResult,
    transfer_outcomes: tuple[MemoryTransferOutcome, ...] = (),
) -> BenchmarkEpisodeReport:
    """Convert an execution result into the stable benchmark report contract."""

    return BenchmarkEpisodeReport(
        episode_id=result.episode_id,
        episode=result.episode,
        episode_success=result.episode_success,
        retained_memory_count=len(result.ingestion.retained_memories),
        transfer_outcomes=transfer_outcomes,
    )


def _close_environment(environment: EnvironmentAdapter) -> None:
    """Close an adapter when its concrete implementation supports cleanup."""

    close = getattr(environment, "close", None)
    if callable(close):
        close()
