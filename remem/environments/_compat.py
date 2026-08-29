"""Compatibility helpers for Gym-like and legacy benchmark APIs."""

from __future__ import annotations

from typing import Any, Iterable


def normalize_reset(result: Any) -> str:
    """Normalize reset output from legacy and Gymnasium-style environments."""

    if isinstance(result, tuple) and len(result) == 2:
        observation, _info = result
    else:
        observation = result
    return str(observation)


def normalize_step(result: Iterable[Any]) -> tuple[str, float, bool, bool, dict[str, Any]]:
    """Normalize four- or five-field step results into one stable representation."""

    values = tuple(result)
    if len(values) == 5:
        observation, reward, terminated, truncated, info = values
        return str(observation), float(reward), bool(terminated), bool(truncated), dict(info)
    if len(values) == 4:
        observation, reward, done, info = values
        return str(observation), float(reward), bool(done), False, dict(info)
    raise ValueError("environment step() must return four or five values")


def require_callable(environment: Any, method_name: str) -> None:
    """Validate that a wrapped environment exposes a required callable method."""

    method = getattr(environment, method_name, None)
    if not callable(method):
        raise TypeError(f"environment must provide callable {method_name}()")
