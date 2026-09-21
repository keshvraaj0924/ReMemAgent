"""Deterministic multi-seed evaluation for normalized text environments."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import isfinite

from remem.environment_runner import ActionPolicy, run_environment_episode
from remem.environments.base import EnvironmentAdapter
from remem.execution import EpisodeResult

EnvironmentFactory = Callable[[int], EnvironmentAdapter]
PolicyFactory = Callable[[int], ActionPolicy]


@dataclass(frozen=True, slots=True)
class SeededEpisodeResult:
    """One episode result tied to the seed that generated its environment."""

    seed: int
    result: EpisodeResult


@dataclass(frozen=True, slots=True)
class EnvironmentEvaluation:
    """Immutable aggregate of a deterministic multi-seed evaluation."""

    episodes: tuple[SeededEpisodeResult, ...]

    def __post_init__(self) -> None:
        if not self.episodes:
            raise ValueError("evaluation must contain at least one episode")
        seeds = tuple(episode.seed for episode in self.episodes)
        if len(seeds) != len(set(seeds)):
            raise ValueError("evaluation seeds must be unique")

    @property
    def episode_count(self) -> int:
        """Return the number of evaluated episodes."""

        return len(self.episodes)

    @property
    def mean_reward(self) -> float:
        """Return arithmetic mean reward across episodes."""

        return sum(episode.result.total_reward for episode in self.episodes) / self.episode_count

    @property
    def termination_rate(self) -> float:
        """Return the fraction of episodes terminated by the environment."""

        terminated = sum(episode.result.terminated for episode in self.episodes)
        return terminated / self.episode_count

    @property
    def truncation_rate(self) -> float:
        """Return the fraction of episodes truncated by environment or runner."""

        truncated = sum(episode.result.truncated for episode in self.episodes)
        return truncated / self.episode_count

    @property
    def mean_steps(self) -> float:
        """Return arithmetic mean transition count across episodes."""

        return sum(episode.result.step_count for episode in self.episodes) / self.episode_count


def evaluate_environment_seeds(
    environment_factory: EnvironmentFactory,
    policy_factory: PolicyFactory,
    *,
    seeds: Iterable[int],
    max_steps: int,
) -> EnvironmentEvaluation:
    """Evaluate one independently constructed environment and policy per seed.

    Seeds are materialized before execution so invalid duplicate/empty suites fail
    without partially running an experiment. Input order is preserved to keep the
    resulting evidence aligned with caller-owned benchmark task ordering.
    """

    seed_values = tuple(seeds)
    if not seed_values:
        raise ValueError("seeds must not be empty")
    if len(seed_values) != len(set(seed_values)):
        raise ValueError("seeds must be unique")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    episodes = tuple(
        SeededEpisodeResult(
            seed=seed,
            result=run_environment_episode(
                environment_factory(seed),
                policy_factory(seed),
                max_steps=max_steps,
            ),
        )
        for seed in seed_values
    )
    evaluation = EnvironmentEvaluation(episodes=episodes)
    if not all(
        isfinite(value)
        for value in (
            evaluation.mean_reward,
            evaluation.termination_rate,
            evaluation.truncation_rate,
            evaluation.mean_steps,
        )
    ):
        raise ValueError("evaluation aggregates must be finite")
    return evaluation


__all__ = [
    "EnvironmentEvaluation",
    "EnvironmentFactory",
    "PolicyFactory",
    "SeededEpisodeResult",
    "evaluate_environment_seeds",
]
