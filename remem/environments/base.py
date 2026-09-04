"""Common contracts for external benchmark environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class StepResult:
    """Normalized result of one environment action.

    The normalized contract rejects non-finite rewards and malformed terminal
    flags at the adapter boundary so invalid environment data cannot silently
    propagate into benchmark reports or training artifacts.
    """

    observation: str
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate values that every environment adapter must guarantee."""

        if not isinstance(self.observation, str):
            raise TypeError("observation must be a string")
        if isinstance(self.reward, bool) or not isinstance(self.reward, (int, float)):
            raise TypeError("reward must be a finite number")
        if not isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        if not isinstance(self.terminated, bool):
            raise TypeError("terminated must be a boolean")
        if not isinstance(self.truncated, bool):
            raise TypeError("truncated must be a boolean")
        if not isinstance(self.info, dict):
            raise TypeError("info must be a dictionary")

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
        """Apply an action and return a normalized outcome."""
        ...

    def close(self) -> None:
        """Release resources owned by the wrapped environment."""
        ...
