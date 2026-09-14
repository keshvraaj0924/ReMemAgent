"""Compatibility helpers for Gym-like and legacy benchmark APIs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite
from typing import Any

from remem.environments.base import StepResult


def normalize_text_observation(value: Any, *, benchmark_name: str = "environment") -> str:
    """Validate one textual benchmark observation without lossy coercion.

    External benchmarks occasionally surface malformed values when their data
    or wrappers are misconfigured. Converting arbitrary objects with ``str``
    would turn values such as ``None`` into apparently valid trajectory text,
    contaminating benchmark evidence instead of failing at the adapter boundary.
    """

    if not isinstance(value, str):
        raise TypeError(f"{benchmark_name} observation must be a string")
    return value


def normalize_finite_reward(
    value: Any,
    *,
    benchmark_name: str = "environment",
    unwrap_singleton_value: bool = False,
) -> float:
    """Validate a finite numeric reward without accepting boolean coercion."""

    normalized_value = unwrap_singleton(value) if unwrap_singleton_value else value
    if isinstance(normalized_value, bool) or not isinstance(normalized_value, (int, float)):
        raise TypeError(f"{benchmark_name} reward must be a finite numeric value")
    reward = float(normalized_value)
    if not isfinite(reward):
        raise ValueError(f"{benchmark_name} reward must be finite")
    return reward


def normalize_boolean_flag(
    value: Any,
    field_name: str,
    *,
    benchmark_name: str = "environment",
    unwrap_singleton_value: bool = False,
) -> bool:
    """Validate one terminal flag without truthiness coercion."""

    normalized_value = unwrap_singleton(value) if unwrap_singleton_value else value
    if not isinstance(normalized_value, bool):
        raise TypeError(f"{benchmark_name} {field_name} flag must be a boolean")
    return normalized_value


def normalize_string_keyed_info(
    info: Any,
    *,
    benchmark_name: str = "environment",
    unwrap_values: bool = False,
) -> dict[str, Any]:
    """Validate benchmark metadata and preserve only its declared mapping shape."""

    if not isinstance(info, Mapping):
        raise TypeError(f"{benchmark_name} info must be a mapping")

    normalized_info: dict[str, Any] = {}
    for key, value in info.items():
        if not isinstance(key, str):
            raise TypeError(f"{benchmark_name} info keys must be strings")
        normalized_info[key] = unwrap_singleton(value) if unwrap_values else value
    return normalized_info


def normalize_reset(result: Any) -> str:
    """Normalize reset output from legacy and Gymnasium-style environments."""

    if isinstance(result, tuple) and len(result) == 2:
        observation, _info = result
    else:
        observation = result
    return normalize_text_observation(observation)


def normalize_step(
    result: Iterable[Any] | StepResult,
) -> tuple[str, float, bool, bool, dict[str, Any]]:
    """Normalize native or four-/five-field step results into one stable representation."""

    if isinstance(result, StepResult):
        return (
            result.observation,
            result.reward,
            result.terminated,
            result.truncated,
            dict(result.info),
        )

    values = tuple(result)
    if len(values) == 5:
        observation, reward, terminated, truncated, info = values
        return (
            normalize_text_observation(observation),
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info),
        )
    if len(values) == 4:
        observation, reward, done, info = values
        return normalize_text_observation(observation), float(reward), bool(done), False, dict(info)
    raise ValueError("environment step() must return four or five values")


def unwrap_singleton(value: Any) -> Any:
    """Remove one singleton batch dimension from sequence-like benchmark values.

    Some external environments expose observations and scalar metadata as NumPy
    arrays rather than Python ``Sequence`` instances. This helper intentionally
    avoids importing NumPy: any object with a safe ``len``/index operation is
    treated as batch-like when it contains exactly one item. Strings and bytes
    remain scalar values.
    """

    if isinstance(value, (str, bytes, Mapping)):
        return value
    try:
        length = len(value)
    except TypeError:
        return value
    if length != 1:
        return value
    try:
        return value[0]
    except (IndexError, KeyError, TypeError):
        return value


def require_callable(environment: Any, method_name: str) -> None:
    """Validate that a wrapped environment exposes a required callable method."""

    method = getattr(environment, method_name, None)
    if not callable(method):
        raise TypeError(f"environment must provide callable {method_name}()")
