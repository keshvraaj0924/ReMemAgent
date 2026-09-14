"""Regression tests for strict external-environment compatibility helpers."""

from __future__ import annotations

from fractions import Fraction

import pytest

from remem.environments._compat import (
    normalize_boolean_flag,
    normalize_finite_reward,
    normalize_reset,
    normalize_step,
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


def test_normalize_finite_reward_accepts_real_numeric_scalars() -> None:
    assert normalize_finite_reward(Fraction(3, 4), benchmark_name="Test") == 0.75


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


def test_normalize_reset_validates_gymnasium_metadata() -> None:
    assert normalize_reset(("ready", {"episode": 7})) == "ready"

    with pytest.raises(ValueError, match="tuple must contain observation and info"):
        normalize_reset(("ready",))
    with pytest.raises(TypeError, match="info must be a mapping"):
        normalize_reset(("ready", None))
    with pytest.raises(TypeError, match="info keys must be strings"):
        normalize_reset(("ready", {1: "bad"}))


def test_normalize_step_strictly_normalizes_five_field_results() -> None:
    assert normalize_step(("next", Fraction(3, 4), False, True, {"score": 1})) == (
        "next",
        0.75,
        False,
        True,
        {"score": 1},
    )


def test_normalize_step_strictly_normalizes_legacy_results() -> None:
    assert normalize_step(("next", 1, True, {"score": 1})) == (
        "next",
        1.0,
        True,
        False,
        {"score": 1},
    )


@pytest.mark.parametrize(
    ("step_result", "error_type", "message"),
    [
        (("next", "1.0", False, False, {}), TypeError, "finite numeric value"),
        (("next", 1.0, 1, False, {}), TypeError, "terminated flag must be a boolean"),
        (("next", 1.0, False, 0, {}), TypeError, "truncated flag must be a boolean"),
        (("next", 1.0, False, False, None), TypeError, "info must be a mapping"),
        (("next", 1.0, False, False, {1: "bad"}), TypeError, "info keys must be strings"),
        (("next", 1.0, 1, {}), TypeError, "terminated flag must be a boolean"),
    ],
)
def test_normalize_step_rejects_coercible_malformed_values(
    step_result: tuple[object, ...],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        normalize_step(step_result)
