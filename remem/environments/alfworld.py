"""Adapter for ALFWorld environments without importing ALFWorld itself."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from remem.environments._compat import require_callable, unwrap_singleton
from remem.environments.base import StepResult


class AlfWorldAdapter:
    """Normalize an ALFWorld environment configured with a single batch item.

    The upstream text environment commonly exposes a batch-oriented interface:
    ``reset()`` returns a one-item observation batch and ``step()`` expects a
    one-item action batch. This adapter removes that batch dimension so the
    research runner can consume the same scalar contract as WebShop and local
    test environments.
    """

    def __init__(self, environment: Any) -> None:
        require_callable(environment, "reset")
        require_callable(environment, "step")
        self._environment = environment

    def reset(self, **kwargs: Any) -> str:
        """Reset ALFWorld and return the first textual observation."""

        result = self._environment.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            observation, _info = result
            return str(unwrap_singleton(observation))
        return str(unwrap_singleton(result))

    def step(self, action: str) -> StepResult:
        """Execute one textual ALFWorld action and normalize its result."""

        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a non-empty string")

        values = tuple(self._environment.step([action]))
        if len(values) == 5:
            observation, reward, terminated, truncated, info = values
        elif len(values) == 4:
            observation, reward, done, info = values
            terminated, truncated = done, False
        else:
            raise ValueError("ALFWorld step() must return four or five values")

        return StepResult(
            observation=str(unwrap_singleton(observation)),
            reward=_normalize_reward(reward),
            terminated=_normalize_terminal_flag(terminated, "terminated"),
            truncated=_normalize_terminal_flag(truncated, "truncated"),
            info=_unwrap_info(info),
        )

    def close(self) -> None:
        """Close ALFWorld when the wrapped environment supports it."""

        close = getattr(self._environment, "close", None)
        if callable(close):
            close()


def _normalize_reward(value: Any) -> float:
    """Normalize an ALFWorld reward while rejecting ambiguous values."""

    unwrapped = unwrap_singleton(value)
    if isinstance(unwrapped, bool) or not isinstance(unwrapped, (int, float)):
        raise TypeError("ALFWorld reward must be a finite numeric value")
    reward = float(unwrapped)
    if not isfinite(reward):
        raise ValueError("ALFWorld reward must be finite")
    return reward


def _normalize_terminal_flag(value: Any, field_name: str) -> bool:
    """Normalize a scalar ALFWorld terminal flag without truthiness coercion."""

    unwrapped = unwrap_singleton(value)
    if not isinstance(unwrapped, bool):
        raise TypeError(f"ALFWorld {field_name} flag must be a boolean")
    return unwrapped


def _unwrap_info(info: Any) -> dict[str, Any]:
    """Remove the first batch dimension from ALFWorld info values."""

    if not isinstance(info, Mapping):
        return {}
    return {key: unwrap_singleton(value) for key, value in info.items()}
