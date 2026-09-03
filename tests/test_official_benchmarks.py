"""Tests for the concrete upstream benchmark factory boundaries."""

from __future__ import annotations

import random
import sys
import types

import pytest

from remem.integrations import official_benchmarks


class FakeWebShopEnvironment:
    """Minimal WebShop-shaped environment with observable reset behavior."""

    def __init__(self) -> None:
        self.reset_values: list[int] = []
        self.closed = False

    def reset(self) -> str:
        self.reset_values.append(random.randrange(1_000_000))
        return "state"

    def close(self) -> None:
        self.closed = True


def test_seeded_webshop_reset_is_reproducible_and_restores_rng() -> None:
    first = FakeWebShopEnvironment()
    second = FakeWebShopEnvironment()
    first_adapter = official_benchmarks._SeededWebShopEnvironment(first, 17)
    second_adapter = official_benchmarks._SeededWebShopEnvironment(second, 17)

    random.seed(1234)
    expected_next = random.randrange(1_000_000)
    random.seed(1234)
    first_adapter.reset()
    observed_next = random.randrange(1_000_000)

    assert observed_next == expected_next

    first_adapter.reset()
    second_adapter.reset()
    assert first.reset_values == [first.reset_values[0], first.reset_values[0]]
    assert second.reset_values == [second.reset_values[0]]
    assert first.reset_values[0] == second.reset_values[0]


def test_webshop_factory_restores_rng_after_gym_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_gym = types.ModuleType("gym")
    environment = FakeWebShopEnvironment()
    construction_values: list[int] = []

    def make(_environment_id: str, **_kwargs: object) -> FakeWebShopEnvironment:
        construction_values.append(random.randrange(1_000_000))
        return environment

    fake_gym.make = make  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gym", fake_gym)

    factory = official_benchmarks.build_webshop_text_environment_factory(num_products=10)
    random.seed(99)
    expected_next = random.randrange(1_000_000)
    random.seed(99)
    adapter = factory(42)
    observed_next = random.randrange(1_000_000)

    assert construction_values
    assert observed_next == expected_next
    assert adapter.reset() == "state"
    assert environment.reset_values


def test_seeded_webshop_reset_rejects_explicit_seed() -> None:
    adapter = official_benchmarks._SeededWebShopEnvironment(FakeWebShopEnvironment(), 3)

    with pytest.raises(TypeError, match="owns the reset seed"):
        adapter.reset(seed=4)


def test_alfworld_factory_isolates_rng_during_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    environment_module = types.ModuleType("alfworld.agents.environment")
    agents_module = types.ModuleType("alfworld.agents")
    alfworld_module = types.ModuleType("alfworld")
    construction_values: list[int] = []

    class FakeAlfWorldEnvironment:
        def init_env(self, *, batch_size: int) -> "FakeAlfWorldEnvironment":
            assert batch_size == 1
            construction_values.append(random.randrange(1_000_000))
            return self

        def reset(self) -> str:
            return "state"

    def get_environment(_environment_type: str) -> type[FakeAlfWorldEnvironment]:
        return FakeAlfWorldEnvironment

    environment_module.get_environment = get_environment  # type: ignore[attr-defined]
    agents_module.environment = environment_module  # type: ignore[attr-defined]
    alfworld_module.agents = agents_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "alfworld", alfworld_module)
    monkeypatch.setitem(sys.modules, "alfworld.agents", agents_module)
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", environment_module)

    config = {"env": {"type": "AlfredTWEnv"}}
    factory = official_benchmarks.build_alfworld_text_environment_factory(config)
    random.seed(101)
    expected_next = random.randrange(1_000_000)
    random.seed(101)
    adapter = factory(42)
    observed_next = random.randrange(1_000_000)

    assert construction_values
    assert observed_next == expected_next
    assert adapter.reset() == "state"
