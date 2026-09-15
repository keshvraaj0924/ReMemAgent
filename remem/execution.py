"""Immutable execution records shared by benchmark and training integrations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from remem.environments.base import StepResult


@dataclass(frozen=True, slots=True)
class EpisodeStep:
    """One observed action transition within an episode."""

    step_index: int
    observation: str
    action: str
    result: StepResult

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if not self.action.strip():
            raise ValueError("action must be non-empty")


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Complete immutable outcome of one environment episode."""

    initial_observation: str
    steps: tuple[EpisodeStep, ...]
    total_reward: float
    terminated: bool
    truncated: bool = False

    def __post_init__(self) -> None:
        if not isfinite(self.total_reward):
            raise ValueError("total_reward must be finite")
        expected_indices = tuple(range(len(self.steps)))
        actual_indices = tuple(step.step_index for step in self.steps)
        if actual_indices != expected_indices:
            raise ValueError("episode step indices must be contiguous from zero")
        if self.terminated and self.truncated:
            raise ValueError("episode cannot be both terminated and truncated")

    @property
    def step_count(self) -> int:
        """Return the number of environment transitions in the episode."""

        return len(self.steps)


__all__ = ["EpisodeResult", "EpisodeStep"]
