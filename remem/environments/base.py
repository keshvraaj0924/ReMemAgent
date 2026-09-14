"""Common contracts for external benchmark environments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class StepResult:
    """Normalized result of one environment action.

    The normalized contract rejects non-finite rewards and malformed terminal
    flags at the adapter boundary so invalid environment data cannot silently
    propagate into benchmark reports or training artifacts. Compatible
    real-valued scalar implementations are normalized to ``float`` so the field
    remains stable for downstream serialization and metrics code.
    """

    observation: str
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate values and detach nested mutable metadata."""

        if not isinstance(self.observation, str):
            raise TypeError("observation must be a string")
        if isinstance(self.reward, bool) or not isinstance(self.reward, Real):
            raise TypeError("reward must be a finite number")
        normalized_reward = float(self.reward)
        if not isfinite(normalized_reward):
            raise ValueError("reward must be finite")
        if not isinstance(self.terminated, bool):
            raise TypeError("terminated must be a boolean")
        if not isinstance(self.truncated, bool):
            raise TypeError("truncated must be a boolean")
        if not isinstance(self.info, dict):
            raise TypeError("info must be a dictionary")
        if any(not isinstance(key, str) for key in self.info):
            raise TypeError("info keys must be strings")

        object.__setattr__(self, "reward", normalized_reward)
        object.__setattr__(self, "info", deepcopy(self.info))

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
