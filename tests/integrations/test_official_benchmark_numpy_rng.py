"""Regression tests for NumPy RNG isolation at external benchmark boundaries."""

from __future__ import annotations

import random
import sys
import types
from typing import Any

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
