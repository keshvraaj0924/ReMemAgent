"""Typed benchmark result model used by experiment reporting and statistics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from remem.execution import EpisodeResult


@dataclass(frozen=True, slots=True)
class BenchmarkEpisodeReport:
    """Measured outcome and memory state for one benchmark episode."""

    episode_id: str
    episode: EpisodeResult
    episode_success: bool
    retained_memory_count: int
    transfer_success: bool = False

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode_id must be non-empty")
        if self.retained_memory_count < 0:
            raise ValueError("retained_memory_count must be non-negative")


@dataclass(frozen=True, slots=True)
class BenchmarkRunReport:
    """Measured results for one deterministic benchmark seed."""

    benchmark_name: str
    episodes: tuple[BenchmarkEpisodeReport, ...]
    final_memory_count: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.benchmark_name.strip():
            raise ValueError("benchmark_name must be non-empty")
        if not self.episodes:
            raise ValueError("episodes must contain at least one episode")
        if self.final_memory_count < 0:
            raise ValueError("final_memory_count must be non-negative")
        episode_ids = tuple(item.episode_id for item in self.episodes)
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("episode ids must be unique within a benchmark run")

    @property
    def success_rate(self) -> float:
        """Return the measured fraction of successful episodes."""

        return sum(item.episode_success for item in self.episodes) / len(self.episodes)

    @property
    def mean_reward(self) -> float:
        """Return mean measured episode reward."""

        value = sum(item.episode.total_reward for item in self.episodes) / len(self.episodes)
        if not isfinite(value):
            raise ValueError("mean benchmark reward must be finite")
        return value

    @property
    def transfer_success_rate(self) -> float:
        """Return fraction of episodes marked as successful memory transfer."""

        return sum(item.transfer_success for item in self.episodes) / len(self.episodes)


__all__ = ["BenchmarkEpisodeReport", "BenchmarkRunReport"]
