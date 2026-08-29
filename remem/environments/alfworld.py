"""Adapter for ALFWorld environments without importing ALFWorld itself."""

from __future__ import annotations

from typing import Any

from remem.environments._compat import normalize_reset, normalize_step, require_callable
from remem.environments.base import StepResult


class AlfWorldAdapter:
    """Normalize an ALFWorld-compatible environment for ReMemAgent runners.

    The adapter accepts an already-created environment instance. This keeps the
    research core independent from ALFWorld installation and configuration while
    allowing the real benchmark environment to be supplied by an experiment.
    """

    def __init__(self, environment: Any) -> None:
        require_callable(environment, "reset")
        require_callable(environment, "step")
        self._environment = environment

    def reset(self, **kwargs: Any) -> str:
        """Reset ALFWorld and return its textual observation."""

        return normalize_reset(self._environment.reset(**kwargs))

    def step(self, action: str) -> StepResult:
        """Execute one textual ALFWorld action and normalize its result."""

        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a non-empty string")
        observation, reward, terminated, truncated, info = normalize_step(
            self._environment.step(action)
        )
        return StepResult(observation, reward, terminated, truncated, info)

    def close(self) -> None:
        """Close ALFWorld when the wrapped environment supports it."""

        close = getattr(self._environment, "close", None)
        if callable(close):
            close()
