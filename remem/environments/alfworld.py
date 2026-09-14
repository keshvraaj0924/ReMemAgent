"""Adapter for ALFWorld environments without importing ALFWorld itself."""

from __future__ import annotations

from typing import Any

from remem.environments._compat import (
    normalize_boolean_flag,
    normalize_finite_reward,
    normalize_string_keyed_info,
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
            normalize_string_keyed_info(
                info,
                benchmark_name="ALFWorld",
                unwrap_values=True,
            )
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
            reward=normalize_finite_reward(
                reward,
                benchmark_name="ALFWorld",
                unwrap_singleton_value=True,
            ),
            terminated=normalize_boolean_flag(
                terminated,
                "terminated",
                benchmark_name="ALFWorld",
                unwrap_singleton_value=True,
            ),
            truncated=normalize_boolean_flag(
                truncated,
                "truncated",
                benchmark_name="ALFWorld",
                unwrap_singleton_value=True,
            ),
            info=normalize_string_keyed_info(
                info,
                benchmark_name="ALFWorld",
                unwrap_values=True,
            ),
        )

    def close(self) -> None:
        """Close ALFWorld when the wrapped environment supports it."""

        close = getattr(self._environment, "close", None)
        if callable(close):
            close()
