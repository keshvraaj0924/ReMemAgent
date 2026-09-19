"""Adapter for ALFWorld-compatible text environments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from remem.environments._normalization import (
    normalize_reset_result,
    normalize_step_result,
)
from remem.environments.base import EnvironmentAdapter, StepResult


class AlfWorldAdapter(EnvironmentAdapter):
    """Normalize an ALFWorld environment for research runners.

    Official ALFWorld TextWorld environments are batch environments even when
    initialized with ``batch_size=1``: observations, rewards, and done flags are
    length-one sequences, metadata is a mapping of batched columns, and actions
    must be dispatched as a length-one list. The adapter also accepts ordinary
    Gym-style scalar environments to keep the boundary easy to test.

    ReMemAgent deliberately supports only one ALFWorld episode per adapter.
    Multi-episode batching belongs in the experiment runner, where independent
    seeds and episode provenance can remain explicit.
    """

    def __init__(self, environment: Any) -> None:
        self._environment = environment
        self._uses_alfworld_batch_contract = False

    def reset(self) -> str:
        """Reset ALFWorld and return its textual observation."""

        result = self._environment.reset()
        if _is_single_item_batch_reset(result):
            self._uses_alfworld_batch_contract = True
            observation, _ = result
            return normalize_reset_result((_single_item(observation, "observation"), {}))

        self._uses_alfworld_batch_contract = False
        return normalize_reset_result(result)

    def step(self, action: str) -> StepResult:
        """Execute one text action and normalize the transition."""

        if not isinstance(action, str):
            raise TypeError("action must be a string")
        if not action.strip():
            raise ValueError("action must be a non-empty string")

        dispatched_action: str | list[str]
        dispatched_action = (
            [action] if self._uses_alfworld_batch_contract else action
        )
        result = self._environment.step(dispatched_action)
        if self._uses_alfworld_batch_contract:
            result = _unbatch_step_result(result)
        return normalize_step_result(result)

    def close(self) -> None:
        """Release resources when the wrapped ALFWorld environment supports it."""

        close = getattr(self._environment, "close", None)
        if callable(close):
            close()


def _is_single_item_batch_reset(result: Any) -> bool:
    """Return whether reset output matches ALFWorld's batch-size-one contract."""

    if not isinstance(result, tuple) or len(result) != 2:
        return False
    observations, info = result
    return _is_non_text_sequence(observations) and isinstance(info, Mapping)


def _unbatch_step_result(result: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Convert ALFWorld's batch-size-one step result to the legacy Gym contract."""

    if (
        not isinstance(result, Sequence)
        or isinstance(result, (str, bytes))
        or len(result) != 4
    ):
        raise ValueError("batched ALFWorld step() must return exactly 4 items")

    observations, rewards, dones, infos = result
    if not isinstance(infos, Mapping):
        raise TypeError("batched ALFWorld info must be a mapping")

    return (
        _single_item(observations, "observation"),
        _single_item(rewards, "reward"),
        _single_item(dones, "done"),
        _unbatch_info(infos),
    )


def _unbatch_info(info: Mapping[Any, Any]) -> dict[str, Any]:
    """Unbatch ALFWorld metadata while rejecting ambiguous column shapes."""

    normalized: dict[str, Any] = {}
    for key, value in info.items():
        if not isinstance(key, str):
            raise TypeError("batched ALFWorld info keys must be strings")
        if _is_non_text_sequence(value):
            normalized[key] = _single_item(value, f"info[{key!r}]")
        else:
            normalized[key] = value
    return normalized


def _single_item(value: Any, field_name: str) -> Any:
    """Extract exactly one item from an ALFWorld batch field."""

    if not _is_non_text_sequence(value):
        raise TypeError(f"batched ALFWorld {field_name} must be a sequence")
    if len(value) != 1:
        raise ValueError(f"batched ALFWorld {field_name} must contain exactly one item")
    return value[0]


def _is_non_text_sequence(value: Any) -> bool:
    """Return whether a value is a sequence but not a textual scalar."""

    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
