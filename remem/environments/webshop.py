"""Adapter for WebShop-compatible environments without hard dependencies."""

from __future__ import annotations

from math import isfinite
from typing import Any

from remem.environments._compat import normalize_reset, require_callable
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
        """Execute one textual WebShop action and normalize its result.

        The adapter intentionally rejects ambiguous reward and terminal values
        instead of silently coercing them. This keeps malformed upstream
        benchmark output from entering measured trajectories.
        """

        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a non-empty string")
        values = tuple(self._environment.step(action))
        if len(values) == 5:
            observation, reward, terminated, truncated, info = values
        elif len(values) == 4:
            observation, reward, done, info = values
            terminated, truncated = done, False
        else:
            raise ValueError("WebShop step() must return four or five values")

        return StepResult(
            observation=str(observation),
            reward=_normalize_reward(reward),
            terminated=_normalize_terminal_flag(terminated, "terminated"),
            truncated=_normalize_terminal_flag(truncated, "truncated"),
            info=dict(info) if isinstance(info, dict) else {},
        )

    def close(self) -> None:
        """Close WebShop when the wrapped environment supports it."""

        close = getattr(self._environment, "close", None)
        if callable(close):
            close()


def _normalize_reward(value: Any) -> float:
    """Normalize a WebShop reward while rejecting ambiguous values."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("WebShop reward must be a finite numeric value")
    reward = float(value)
    if not isfinite(reward):
        raise ValueError("WebShop reward must be finite")
    return reward


def _normalize_terminal_flag(value: Any, field_name: str) -> bool:
    """Normalize a WebShop terminal flag without truthiness coercion."""

    if not isinstance(value, bool):
        raise TypeError(f"WebShop {field_name} flag must be a boolean")
    return value
