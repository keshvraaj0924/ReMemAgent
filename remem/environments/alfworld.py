"""Adapter for ALFWorld-compatible text environments."""

from __future__ import annotations

from typing import Any

from remem.environments._normalization import normalize_reset_result, normalize_step_result
from remem.environments.base import EnvironmentAdapter, StepResult


class AlfWorldAdapter(EnvironmentAdapter):
    """Normalize an ALFWorld/Gym-style environment for research runners."""

    def __init__(self, environment: Any) -> None:
        self._environment = environment

    def reset(self) -> str:
        """Reset ALFWorld and return its textual observation."""

        return normalize_reset_result(self._environment.reset())

    def step(self, action: str) -> StepResult:
        """Execute one text action and normalize the transition."""

        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a non-empty string")
        return normalize_step_result(self._environment.step(action))

    def close(self) -> None:
        """Release resources when the wrapped ALFWorld environment supports it."""

        close = getattr(self._environment, "close", None)
        if callable(close):
            close()
