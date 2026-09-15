"""Benchmark-independent environment protocol used by research runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self


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

    def close(self) -> None:
        """Release resources owned by the underlying environment, if any."""

    def __enter__(self) -> Self:
        """Return this adapter for deterministic resource management."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release environment resources when leaving a context manager."""

        self.close()
