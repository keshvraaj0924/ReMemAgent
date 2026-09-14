"""Adapter for ALFWorld environments without importing ALFWorld itself."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from remem.environments._compat import (
    normalize_text_observation,
    require_callable,
    unwrap_singleton,
)
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
        """Reset ALFWorld and return the first textual observation.

        Gymnasium-style reset metadata is validated even though the normalized
        runner contract currently returns only the textual observation. This
        prevents malformed upstream reset payloads from being silently ignored.
        """

        result = self._environment.reset(**kwargs)
        if isinstance(result, tuple):
            if len(result) != 2:
                raise ValueError("ALFWorld reset() tuple must contain observation and info")
            observation, info = result
            _normalize_info(info)
        else:
            observation = result
        return normalize_text_observation(
            unwrap_singleton(observation),
            benchmark_name="ALFWorld",
        )

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
            observation=normalize_text_observation(
                unwrap_singleton(observation),
                benchmark_name="ALFWorld",
            ),
            reward=_normalize_reward(reward),
            terminated=_normalize_terminal_flag(terminated, "terminated"),
            truncated=_normalize_terminal_flag(truncated, "truncated"),
            info=_normalize_info(info),
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


def _normalize_info(info: Any) -> dict[str, Any]:
    """Normalize ALFWorld metadata without hiding malformed benchmark payloads."""

    if not isinstance(info, Mapping):
        raise TypeError("ALFWorld info must be a mapping")

    normalized_info: dict[str, Any] = {}
    for key, value in info.items():
        if not isinstance(key, str):
            raise TypeError("ALFWorld info keys must be strings")
        normalized_info[key] = unwrap_singleton(value)
    return normalized_info
