"""Regression tests for NumPy RNG isolation at external benchmark boundaries."""

from __future__ import annotations

import random
import sys
import types
from typing import Any

import pytest

from remem.integrations import official_benchmarks


class FakeNumpyRandom:
    """Minimal NumPy RNG surface needed by the scoped seed helper."""

    def __init__(self) -> None:
        self._random = random.Random()

    def seed(self, seed: int) -> None:
        self._random.seed(seed)

    def random(self) -> float:
        return self._random.random()

    def get_state(self) -> object:
        return self._random.getstate()

    def set_state(self, state: object) -> None:
        self._random.setstate(state)


class FakeNumpyModule(types.ModuleType):
    """Minimal module object exposing a NumPy-compatible random namespace."""

    def __init__(self) -> None:
        super().__init__("numpy")
        self.random = FakeNumpyRandom()


class RandomConsumingEnvironment:
    """Environment double that consumes both legacy module-level RNG streams."""

    def __init__(self, fake_numpy: FakeNumpyModule) -> None:
        self._fake_numpy = fake_numpy

    def reset(self) -> tuple[float, float]:
        return random.random(), self._fake_numpy.random.random()

    def step(self, action: str) -> tuple[str, float, float]:
        return action, random.random(), self._fake_numpy.random.random()


def test_scoped_random_seed_restores_numpy_and_python_rng(monkeypatch: Any) -> None:
    fake_numpy = FakeNumpyModule()
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)

    random.seed(123)
    fake_numpy.random.seed(456)
    expected_python_next = random.random()
    expected_numpy_next = fake_numpy.random.random()
    random.seed(123)
    fake_numpy.random.seed(456)

    with official_benchmarks._scoped_random_seed(17):
        assert random.random() == random.Random(17).random()
        seeded_numpy = fake_numpy.random.random()
        expected_seeded_numpy = random.Random(17).random()
        assert seeded_numpy == expected_seeded_numpy

    assert random.random() == expected_python_next
    assert fake_numpy.random.random() == expected_numpy_next


def test_scoped_random_seed_does_not_require_numpy(monkeypatch: Any) -> None:
    monkeypatch.setitem(sys.modules, "numpy", None)

    random.seed(321)
    expected_next = random.random()
    random.seed(321)

    with official_benchmarks._scoped_random_seed(17):
        assert random.random() == random.Random(17).random()

    assert random.random() == expected_next


@pytest.mark.parametrize(
    "wrapper_type",
    [
        official_benchmarks._SeededAlfWorldEnvironment,
        official_benchmarks._SeededWebShopEnvironment,
    ],
)
def test_seeded_environment_preserves_rng_continuity_across_steps(
    monkeypatch: Any,
    wrapper_type: type[Any],
) -> None:
    fake_numpy = FakeNumpyModule()
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    environment = RandomConsumingEnvironment(fake_numpy)
    wrapped_environment = wrapper_type(environment, 17)

    expected_python = random.Random(17)
    expected_numpy = random.Random(17)

    assert wrapped_environment.reset() == (
        expected_python.random(),
        expected_numpy.random(),
    )
    assert wrapped_environment.step("first") == (
        "first",
        expected_python.random(),
        expected_numpy.random(),
    )
    assert wrapped_environment.step("second") == (
        "second",
        expected_python.random(),
        expected_numpy.random(),
    )


def test_seeded_environment_restores_caller_rng_after_each_operation(monkeypatch: Any) -> None:
    fake_numpy = FakeNumpyModule()
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    environment = RandomConsumingEnvironment(fake_numpy)
    wrapped_environment = official_benchmarks._SeededWebShopEnvironment(environment, 17)

    random.seed(123)
    fake_numpy.random.seed(456)
    expected_python = random.Random(123)
    expected_numpy = random.Random(456)

    wrapped_environment.reset()
    assert random.random() == expected_python.random()
    assert fake_numpy.random.random() == expected_numpy.random()

    wrapped_environment.step("next")
    assert random.random() == expected_python.random()
    assert fake_numpy.random.random() == expected_numpy.random()


def test_seeded_environment_reset_restarts_episode_rng_stream(monkeypatch: Any) -> None:
    fake_numpy = FakeNumpyModule()
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    environment = RandomConsumingEnvironment(fake_numpy)
    wrapped_environment = official_benchmarks._SeededAlfWorldEnvironment(environment, 17)

    first_reset = wrapped_environment.reset()
    wrapped_environment.step("advance")

    assert wrapped_environment.reset() == first_reset
