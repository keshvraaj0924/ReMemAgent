"""Benchmark-suite orchestration over normalized environment adapters.

This module deliberately does not import ALFWorld, WebShop, or any model SDK.
Experiments provide concrete environment and policy factories, while the suite
runner provides deterministic lifecycle, memory persistence, transfer tracing,
aggregate reporting, and optional low-dependency observability.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import sys
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
    action_policy_factory: str | None = None
    success_evaluator: str | None = None
    transfer_success_evaluator: str | None = None
    minimum_trust: float = 0.0

    def __post_init__(self) -> None:
        """Reject malformed provenance metadata at construction time."""

        if not isinstance(self.benchmark_name, str) or not self.benchmark_name.strip():
            raise ValueError("benchmark_name must be a non-empty string")
        if not _is_strict_integer(self.episode_count) or self.episode_count < 0:
            raise ValueError("episode_count must be a non-negative integer")
        if not _is_strict_integer(self.max_steps) or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if self.seed is not None and not _is_strict_integer(self.seed):
            raise ValueError("seed must be an integer when provided")
        for field_name, value in (
            ("environment_factory", self.environment_factory),
            ("policy_factory", self.policy_factory),
            ("action_policy_factory", self.action_policy_factory),
            ("success_evaluator", self.success_evaluator),
            ("transfer_success_evaluator", self.transfer_success_evaluator),
        ):
            _validate_optional_string(field_name, value)
        if self.policy_factory is not None and self.action_policy_factory is not None:
            raise ValueError("policy_factory and action_policy_factory are mutually exclusive")
        if isinstance(self.minimum_trust, bool) or not isinstance(self.minimum_trust, (int, float)):
            raise TypeError("minimum_trust must be a number between 0 and 1")
        if not math.isfinite(float(self.minimum_trust)):
            raise ValueError("minimum_trust must be finite")
        if not 0.0 <= self.minimum_trust <= 1.0:
            raise ValueError("minimum_trust must be between 0 and 1")


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
        """Return the number of successful attributed memory selections."""

        return sum(1 for outcome in self.transfer_outcomes if outcome.successful)

    @property
    def transfer_success_rate(self) -> float:
        """Return transfer-success rate for this episode."""

        if not self.transfer_outcomes:
            return 0.0
        return self.transfer_success_count / len(self.transfer_outcomes)


@dataclass(frozen=True, slots=True)
class BenchmarkRunReport:
    """Aggregate report from a benchmark-suite invocation."""

    benchmark_name: str
    seed: int | None
    episodes: tuple[BenchmarkEpisodeReport, ...]
    retained_memory_count: int
    configuration: BenchmarkRunConfiguration | None = None

    @property
    def episode_count(self) -> int:
        """Return number of completed episodes."""

        return len(self.episodes)

    @property
    def success_count(self) -> int:
        """Return number of successful episodes."""

        return sum(1 for episode in self.episodes if episode.episode_success)

    @property
    def success_rate(self) -> float:
        """Return episode success rate."""

        if not self.episodes:
            return 0.0
        return self.success_count / len(self.episodes)

    @property
    def transfer_count(self) -> int:
        """Return total number of attributed memory selections."""

        return sum(episode.transfer_count for episode in self.episodes)

    @property
    def transfer_success_count(self) -> int:
        """Return total number of successful attributed memory selections."""

        return sum(episode.transfer_success_count for episode in self.episodes)

    @property
    def transfer_success_rate(self) -> float:
        """Return transfer-success rate across all attributed memory selections."""

        if self.transfer_count == 0:
            return 0.0
        return self.transfer_success_count / self.transfer_count


class BenchmarkSuiteRunner:
    """Run benchmark episodes with deterministic seed and lifecycle management."""

    def __init__(
        self,
        *,
        store: MemoryStore | None = None,
        observation_collector: ObservationCollector | None = None,
    ) -> None:
        self._store = store or MemoryStore()
        self._observation_collector = observation_collector

    def run(
        self,
        *,
        benchmark_name: str,
        episode_count: int,
        max_steps: int,
        environment_factory: EnvironmentFactory,
        policy_factory: PolicyFactory,
        success_evaluator: SuccessEvaluator,
        transfer_success_evaluator: TransferSuccessEvaluator | None = None,
        seed: int | None = None,
        configuration: BenchmarkRunConfiguration | None = None,
    ) -> BenchmarkRunReport:
        """Execute a benchmark suite and return an immutable aggregate report."""

        _validate_benchmark_arguments(benchmark_name, episode_count, max_steps, seed)
        reports: list[BenchmarkEpisodeReport] = []
        for episode_index in range(episode_count):
            episode_seed = None if seed is None else seed + episode_index
            environment = environment_factory(0 if episode_seed is None else episode_seed)
            transfer_recorder = MemoryTransferRecorder()
            try:
                policy = policy_factory(0 if episode_seed is None else episode_seed, self._store)
                if isinstance(policy, MemoryGuidedPolicy):
                    policy.set_transfer_recorder(transfer_recorder)
                execution_service = EpisodeExecutionService(
                    store=self._store,
                    observation_collector=self._observation_collector,
                )
                execution_result = execution_service.execute(
                    environment=environment,
                    policy=policy,
                    max_steps=max_steps,
                    success_evaluator=success_evaluator,
                    episode_id=_episode_id(benchmark_name, episode_index, episode_seed),
                )
                transfer_outcomes = _attribute_transfers(
                    transfer_recorder=transfer_recorder,
                    episode_result=execution_result,
                    evaluator=transfer_success_evaluator,
                )
                reports.append(
                    BenchmarkEpisodeReport(
                        episode_id=execution_result.episode_id,
                        episode=execution_result.episode,
                        episode_success=execution_result.successful,
                        retained_memory_count=len(self._store),
                        transfer_outcomes=transfer_outcomes,
                    )
                )
            finally:
                _close_environment(environment)
        return BenchmarkRunReport(
            benchmark_name=benchmark_name,
            seed=seed,
            episodes=tuple(reports),
            retained_memory_count=len(self._store),
            configuration=configuration,
        )


def _attribute_transfers(
    *,
    transfer_recorder: MemoryTransferRecorder,
    episode_result: EpisodeExecutionResult,
    evaluator: TransferSuccessEvaluator | None,
) -> tuple[MemoryTransferOutcome, ...]:
    """Convert memory-use selections into transfer outcomes after an episode."""

    if evaluator is None:
        return ()
    outcomes: list[MemoryTransferOutcome] = []
    for selection in transfer_recorder.selections:
        outcomes.append(
            MemoryTransferOutcome(
                memory_id=selection.memory_id,
                source_task=selection.source_task,
                target_task=selection.target_task,
                successful=evaluator(selection, episode_result.episode),
            )
        )
    return tuple(outcomes)


def _episode_id(benchmark_name: str, episode_index: int, episode_seed: int | None) -> str:
    """Return a deterministic episode identifier."""

    if episode_seed is None:
        return f"{benchmark_name}:episode-{episode_index}"
    return f"{benchmark_name}:seed-{episode_seed}:episode-{episode_index}"


def _close_environment(environment: object) -> None:
    """Close an environment when its adapter exposes lifecycle cleanup."""

    close = getattr(environment, "close", None)
    if close is None:
        return
    if not callable(close):
        raise TypeError("environment close attribute must be callable")
    close()


def _validate_benchmark_arguments(
    benchmark_name: str,
    episode_count: int,
    max_steps: int,
    seed: int | None,
) -> None:
    """Validate runner inputs before constructing environments."""

    if not isinstance(benchmark_name, str) or not benchmark_name.strip():
        raise ValueError("benchmark_name must be a non-empty string")
    if not _is_strict_integer(episode_count) or episode_count < 0:
        raise ValueError("episode_count must be a non-negative integer")
    if not _is_strict_integer(max_steps) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if seed is not None and not _is_strict_integer(seed):
        raise ValueError("seed must be an integer when provided")


def _is_strict_integer(value: object) -> bool:
    """Return whether ``value`` is an integer excluding booleans."""

    return isinstance(value, int) and not isinstance(value, bool)


def _validate_optional_string(field_name: str, value: str | None) -> None:
    """Reject blank or non-string optional provenance values."""

    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string when provided")


__all__ = [
    "BenchmarkEpisodeReport",
    "BenchmarkRunConfiguration",
    "BenchmarkRunReport",
    "BenchmarkSuiteRunner",
    "EnvironmentFactory",
    "PolicyFactory",
]
