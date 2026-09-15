"""Shared normalization helpers for external text environments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from remem.environments.base import StepResult


def normalize_reset_result(result: Any) -> str:
    """Normalize Gym-style reset outputs to a textual observation."""

    observation = result[0] if isinstance(result, tuple) and len(result) == 2 else result
    return _as_text(observation)


def normalize_step_result(result: Any) -> StepResult:
    """Normalize legacy Gym and Gymnasium step tuples."""

    if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
        raise TypeError("environment step() must return a 4- or 5-item sequence")

    if len(result) == 4:
        observation, reward, terminated, info = result
        truncated = False
    elif len(result) == 5:
        observation, reward, terminated, truncated, info = result
    else:
        raise ValueError("environment step() must return exactly 4 or 5 items")

    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise TypeError("environment reward must be numeric")
    if not isinstance(terminated, bool) or not isinstance(truncated, bool):
        raise TypeError("environment termination flags must be bool")
    if not isinstance(info, Mapping):
        raise TypeError("environment info must be a mapping")

    return StepResult(
        observation=_as_text(observation),
        reward=float(reward),
        terminated=terminated,
        truncated=truncated,
        info=dict(info),
    )


def _as_text(observation: Any) -> str:
    """Require benchmark observations to have an unambiguous text representation."""

    if isinstance(observation, str):
        return observation
    if isinstance(observation, bytes):
        return observation.decode("utf-8")
    raise TypeError("environment observation must be str or UTF-8 bytes")
