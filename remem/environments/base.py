"""Common contracts for external benchmark environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class StepResult:
    """Normalized result of one environment action."""

    observation: str
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """Return whether the episode has ended for any reason."""

        return self.terminated or self.truncated


class EnvironmentAdapter(Protocol):
    """Minimal interface consumed by an agent runner."""

    def reset(self, **kwargs: Any) -> str:
        """Start an episode and return the initial observation."""
        ...

    def step(self, action: str) -> StepResult:
        """Apply an action and return its normalized outcome."""
        ...

    def close(self) -> None:
        """Release resources owned by the wrapped environment."""
        ...
