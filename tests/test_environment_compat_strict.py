"""Regression tests for strict external-environment compatibility helpers."""

from __future__ import annotations

import pytest

from remem.environments._compat import (
    normalize_boolean_flag,
    normalize_finite_reward,
    normalize_string_keyed_info,
)


class SingletonBatch:
    """Minimal array-like singleton batch used to exercise optional unwrapping."""

    def __init__(self, value: object) -> None:
        self._value = value

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> object:
        if index != 0:
            raise IndexError(index)
        return self._value


def test_normalize_finite_reward_rejects_boolean_and_non_finite_values() -> None:
    assert normalize_finite_reward(1, benchmark_name="Test") == 1.0

    with pytest.raises(TypeError, match="finite numeric value"):
        normalize_finite_reward(True, benchmark_name="Test")
    with pytest.raises(ValueError, match="must be finite"):
        normalize_finite_reward(float("inf"), benchmark_name="Test")


def test_normalize_finite_reward_can_unwrap_singleton_batches() -> None:
    assert (
        normalize_finite_reward(
            SingletonBatch(0.75),
            benchmark_name="Test",
            unwrap_singleton_value=True,
        )
        == 0.75
    )


def test_normalize_boolean_flag_rejects_truthy_non_booleans() -> None:
    assert normalize_boolean_flag(False, "terminated", benchmark_name="Test") is False

    with pytest.raises(TypeError, match="terminated flag must be a boolean"):
        normalize_boolean_flag(1, "terminated", benchmark_name="Test")


def test_normalize_string_keyed_info_rejects_malformed_metadata() -> None:
    assert normalize_string_keyed_info({"score": 1}, benchmark_name="Test") == {"score": 1}

    with pytest.raises(TypeError, match="info must be a mapping"):
        normalize_string_keyed_info(None, benchmark_name="Test")
    with pytest.raises(TypeError, match="info keys must be strings"):
        normalize_string_keyed_info({1: "bad"}, benchmark_name="Test")


def test_normalize_string_keyed_info_can_unwrap_values() -> None:
    assert normalize_string_keyed_info(
        {"score": SingletonBatch(3)},
        benchmark_name="Test",
        unwrap_values=True,
    ) == {"score": 3}
