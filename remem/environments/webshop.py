"""Adapter for the official WebShop text environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from remem.environments._normalization import normalize_reset_result, normalize_step_result
from remem.environments.base import EnvironmentAdapter, StepResult


class WebShopAdapter(EnvironmentAdapter):
    """Normalize the official WebShop/Gym environment for research runners.

    The upstream ``WebAgentTextEnv`` follows the legacy Gym transition shape but
    returns ``None`` for reset and transition metadata. ReMemAgent's shared
    normalizers intentionally require mappings at external boundaries, so this
    adapter translates only that documented WebShop sentinel to an empty mapping
    before delegating to the strict shared validation layer.
    """

    def __init__(self, environment: Any) -> None:
        self._environment = environment

    def reset(self) -> str:
        """Reset WebShop and return its textual observation."""

        return normalize_reset_result(_normalize_webshop_reset(self._environment.reset()))

    def step(self, action: str) -> StepResult:
        """Execute one WebShop text action and normalize the transition."""

        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a non-empty string")
        return normalize_step_result(_normalize_webshop_step(self._environment.step(action)))

    def close(self) -> None:
        """Release resources when the wrapped WebShop environment supports it."""

        close = getattr(self._environment, "close", None)
        if callable(close):
            close()


def _normalize_webshop_reset(result: Any) -> Any:
    """Translate the official WebShop ``(observation, None)`` reset sentinel."""

    if isinstance(result, tuple) and len(result) == 2 and result[1] is None:
        return result[0], {}
    return result


def _normalize_webshop_step(result: Any) -> Any:
    """Translate the official WebShop ``info=None`` transition sentinel."""

    if (
        isinstance(result, Sequence)
        and not isinstance(result, (str, bytes))
        and len(result) == 4
        and result[3] is None
    ):
        observation, reward, done, _ = result
        return observation, reward, done, {}
    return result
