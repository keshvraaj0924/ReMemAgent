"""Benchmark-independent environment protocol used by research runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class StepResult:
    """Normalized result of one environment transition."""

    observation: str
    reward: float
    terminated: bool
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """Return whether the episode can no longer accept actions."""

        return self.terminated or self.truncated


class EnvironmentAdapter(ABC):
    """Minimal benchmark-independent interface consumed by ReMemAgent."""

    @abstractmethod
    def reset(self) -> str:
        """Reset the underlying environment and return its text observation."""

    @abstractmethod
    def step(self, action: str) -> StepResult:
        """Apply an action and normalize the resulting transition."""
