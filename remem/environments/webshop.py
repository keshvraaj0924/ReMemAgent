"""Adapter for WebShop-compatible environments without hard dependencies."""

from __future__ import annotations

from typing import Any

from remem.environments._compat import (
    normalize_boolean_flag,
    normalize_finite_reward,
    normalize_string_keyed_info,
    normalize_text_observation,
    require_callable,
)
from remem.environments.base import StepResult


class WebShopAdapter:
    """Normalize a WebShop-compatible environment for ReMemAgent runners."""

    def __init__(self, environment: Any) -> None:
        require_callable(environment, "reset")
        require_callable(environment, "step")
        self._environment = environment

    def reset(self, **kwargs: Any) -> str:
        """Reset WebShop and return its textual observation."""

        result = self._environment.reset(**kwargs)
        if isinstance(result, tuple):
            if len(result) != 2:
                raise ValueError("WebShop reset() tuple must contain observation and info")
            observation, info = result
            normalize_string_keyed_info(info, benchmark_name="WebShop")
        else:
            observation = result
        return normalize_text_observation(observation, benchmark_name="WebShop")

    def step(self, action: str) -> StepResult:
        """Execute one textual WebShop action and normalize its result."""

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
            observation=normalize_text_observation(observation, benchmark_name="WebShop"),
            reward=normalize_finite_reward(reward, benchmark_name="WebShop"),
            terminated=normalize_boolean_flag(
                terminated,
                "terminated",
                benchmark_name="WebShop",
            ),
            truncated=normalize_boolean_flag(
                truncated,
                "truncated",
                benchmark_name="WebShop",
            ),
            info=normalize_string_keyed_info(info, benchmark_name="WebShop"),
        )

    def close(self) -> None:
        """Close WebShop when the wrapped environment supports it."""

        close = getattr(self._environment, "close", None)
        if callable(close):
            close()
