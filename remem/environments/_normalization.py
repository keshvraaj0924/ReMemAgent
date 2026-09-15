"""Shared normalization helpers for external text environments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from numbers import Real
from typing import Any

from remem.environments.base import StepResult

LEGACY_TIME_LIMIT_TRUNCATED_KEY = "TimeLimit.truncated"


def normalize_reset_result(result: Any) -> str:
    """Normalize Gym-style reset outputs to a textual observation.

    Gymnasium reset results are ``(observation, info)`` pairs. Validate the
    metadata shape instead of silently accepting malformed two-item tuples at
    the external benchmark boundary.
    """

    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError("environment reset() tuple must contain observation and info")
        observation, info = result
        if not isinstance(info, Mapping):
            raise TypeError("environment reset info must be a mapping")
        return _as_text(observation)
    return _as_text(result)


def normalize_step_result(result: Any) -> StepResult:
    """Normalize legacy Gym and Gymnasium step tuples.

    Rewards cross an untrusted benchmark boundary, so they are normalized to a
    finite Python ``float`` before they can reach experiment metrics or training
    code. NaN and infinity are rejected rather than allowed to silently poison
    aggregate results. Environment metadata is deeply detached so later mutation
    by the benchmark cannot rewrite an already-recorded transition.

    Legacy Gym exposes a single ``done`` flag. When ``TimeLimit.truncated`` is
    present in ``info``, preserve the distinction between a true terminal state
    and a time-limit truncation instead of treating every ``done`` as terminal.
    """

    if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
        raise TypeError("environment step() must return a 4- or 5-item sequence")

    if len(result) == 4:
        observation, reward, done, info = result
        if not isinstance(info, Mapping):
            raise TypeError("environment info must be a mapping")
        terminated, truncated = _normalize_legacy_done(done, info)
    elif len(result) == 5:
        observation, reward, terminated, truncated, info = result
        if not isinstance(terminated, bool) or not isinstance(truncated, bool):
            raise TypeError("environment termination flags must be bool")
        if not isinstance(info, Mapping):
            raise TypeError("environment info must be a mapping")
    else:
        raise ValueError("environment step() must return exactly 4 or 5 items")

    normalized_reward = _normalize_reward(reward)

    return StepResult(
        observation=_as_text(observation),
        reward=normalized_reward,
        terminated=terminated,
        truncated=truncated,
        info=deepcopy(dict(info)),
    )


def _normalize_legacy_done(done: Any, info: Mapping[str, Any]) -> tuple[bool, bool]:
    """Split legacy Gym ``done`` into termination and truncation flags."""

    if not isinstance(done, bool):
        raise TypeError("environment termination flags must be bool")

    time_limit_truncated = info.get(LEGACY_TIME_LIMIT_TRUNCATED_KEY, False)
    if not isinstance(time_limit_truncated, bool):
        raise TypeError("TimeLimit.truncated must be bool when present")

    truncated = done and time_limit_truncated
    terminated = done and not truncated
    return terminated, truncated


def _normalize_reward(reward: object) -> float:
    """Return a finite real-valued reward as a Python float.

    Gym-compatible environments may return scalar implementations beyond the
    built-in ``int`` and ``float`` types. Accept the standard ``Real`` numeric
    contract while explicitly excluding booleans, which are ``int`` subclasses
    but are not meaningful benchmark rewards.
    """

    if isinstance(reward, bool) or not isinstance(reward, Real):
        raise TypeError("environment reward must be a real number")

    normalized_reward = float(reward)
    if not math.isfinite(normalized_reward):
        raise ValueError("environment reward must be finite")
    return normalized_reward


def _as_text(observation: Any) -> str:
    """Require benchmark observations to have an unambiguous text representation."""

    if isinstance(observation, str):
        return observation
    if isinstance(observation, bytes):
        return observation.decode("utf-8")
    raise TypeError("environment observation must be str or UTF-8 bytes")
