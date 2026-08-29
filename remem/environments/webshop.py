"""Adapter for WebShop-compatible environments without hard dependencies."""

from __future__ import annotations

from typing import Any

from remem.environments._compat import normalize_reset, normalize_step, require_callable
from remem.environments.base import StepResult


class WebShopAdapter:
    """Normalize a WebShop-compatible environment for ReMemAgent runners.

    WebShop versions expose Gym-like ``reset`` and ``step`` methods. The adapter
    deliberately accepts the concrete instance so benchmark setup remains an
    experiment concern rather than a dependency of the memory engine.
    """

    def __init__(self, environment: Any) -> None:
        require_callable(environment, "reset")
        require_callable(environment, "step")
        self._environment = environment

    def reset(self, **kwargs: Any) -> str:
        """Reset WebShop and return its textual observation."""

        return normalize_reset(self._environment.reset(**kwargs))

    def step(self, action: str) -> StepResult:
        """Execute one textual WebShop action and normalize its result."""

        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a non-empty string")
        observation, reward, terminated, truncated, info = normalize_step(
            self._environment.step(action)
        )
        return StepResult(observation, reward, terminated, truncated, info)

    def close(self) -> None:
        """Close WebShop when the wrapped environment supports it."""

        close = getattr(self._environment, "close", None)
        if callable(close):
            close()
