import random
import sys
import types
from typing import Any

import pytest

from remem.integrations.official_benchmarks import (
    build_alfworld_text_environment_factory,
    build_webshop_text_environment_factory,
)


class FakeAlfWorldEnvironment:
    def __init__(self, config: dict[str, Any], train_eval: str) -> None:
        self.config = config
        self.train_eval = train_eval
        self.batch_size: int | None = None
        self.reset_random_values: list[float] = []
        self.reset_calls = 0

    def init_env(self, *, batch_size: int) -> "FakeAlfWorldEnvironment":
        self.batch_size = batch_size
        return self

    def reset(self) -> str:
        self.reset_calls += 1
        value = random.random()
        self.reset_random_values.append(value)
        return "observation"


class FakeWebShopEnvironment:
    def __init__(self, *, reset_accepts_seed: bool = True, reset_error: BaseException | None = None) -> None:
        self.reset_accepts_seed = reset_accepts_seed
        self.reset_error = reset_error
        self.reset_seeds: list[int | None] = []
        self.close_calls = 0

    def reset(self, **kwargs: Any) -> None:
        if not self.reset_accepts_seed and "seed" in kwargs:
            raise AssertionError("seed should not be passed to this legacy reset")
        self.reset_seeds.append(kwargs.get("seed"))
        if self.reset_error is not None:
            raise self.reset_error

    def close(self) -> None:
        self.close_calls += 1


def test_alfworld_factory_uses_upstream_environment_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[FakeAlfWorldEnvironment] = []

    def get_environment(env_type: str):
        assert env_type == "AlfredTWEnv"

        def constructor(config: dict[str, Any], train_eval: str) -> FakeAlfWorldEnvironment:
            environment = FakeAlfWorldEnvironment(config, train_eval)
            created.append(environment)
            return environment

        return constructor

    environment_module = types.ModuleType("alfworld.agents.environment")
    environment_module.get_environment = get_environment  # type: ignore[attr-defined]
    agents_module = types.ModuleType("alfworld.agents")
    alfworld_module = types.ModuleType("alfworld")
    monkeypatch.setitem(sys.modules, "alfworld", alfworld_module)
    monkeypatch.setitem(sys.modules, "alfworld.agents", agents_module)
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", environment_module)

    config = {"env": {"type": "AlfredTWEnv"}}
    factory = build_alfworld_text_environment_factory(config, train_eval="eval")
    environment = factory(17)

    assert environment is not created[0]
    assert environment.config == config
    assert environment.train_eval == "eval"
    assert environment.batch_size == 1


def test_alfworld_factory_seeds_reset_without_leaking_global_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[FakeAlfWorldEnvironment] = []

    def get_environment(_env_type: str):
        def constructor(config: dict[str, Any], train_eval: str) -> FakeAlfWorldEnvironment:
            environment = FakeAlfWorldEnvironment(config, train_eval)
            created.append(environment)
            return environment

        return constructor

    environment_module = types.ModuleType("alfworld.agents.environment")
    environment_module.get_environment = get_environment  # type: ignore[attr-defined]
    agents_module = types.ModuleType("alfworld.agents")
    alfworld_module = types.ModuleType("alfworld")
    monkeypatch.setitem(sys.modules, "alfworld", alfworld_module)
    monkeypatch.setitem(sys.modules, "alfworld.agents", agents_module)
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", environment_module)

    random.seed(12345)
    expected_next_value = random.Random(12345).random()
    random.seed(12345)

    factory = build_alfworld_text_environment_factory({"env": {"type": "AlfredTWEnv"}})
    environment = factory(17)
    assert environment.reset() == "observation"
    assert environment.reset() == "observation"

    assert created[0].reset_random_values[0] == created[0].reset_random_values[1]
    assert random.random() == expected_next_value


def test_alfworld_factory_rejects_non_singleton_batch() -> None:
    with pytest.raises(ValueError, match="batch_size=1"):
        build_alfworld_text_environment_factory({}, batch_size=2)


def test_webshop_factory_creates_and_seeds_gym_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = FakeWebShopEnvironment()
    gym_module = types.ModuleType("gym")
    gym_module.make = lambda environment_id, **kwargs: (  # type: ignore[attr-defined]
        _assert_webshop_make(environment_id, kwargs, environment)
    )
    monkeypatch.setitem(sys.modules, "gym", gym_module)

    factory = build_webshop_text_environment_factory(num_products=1000)
    result = factory(23)

    assert result is environment
    assert environment.reset_seeds == [23]
    assert environment.close_calls == 0


def test_webshop_factory_supports_legacy_reset_without_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = FakeWebShopEnvironment(reset_accepts_seed=False)
    gym_module = types.ModuleType("gym")
    gym_module.make = lambda _environment_id, **_kwargs: environment  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gym", gym_module)

    factory = build_webshop_text_environment_factory()
    result = factory(23)

    assert result is environment
    assert environment.reset_seeds == [None]


def test_webshop_factory_closes_environment_when_reset_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = RuntimeError("reset failed")
    environment = FakeWebShopEnvironment(reset_error=failure)
    gym_module = types.ModuleType("gym")
    gym_module.make = lambda _environment_id, **_kwargs: environment  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gym", gym_module)

    factory = build_webshop_text_environment_factory()

    with pytest.raises(RuntimeError, match="reset failed") as raised:
        factory(23)

    assert raised.value is failure
    assert environment.close_calls == 1


def test_webshop_factory_closes_environment_when_reset_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    class MissingResetEnvironment:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    environment = MissingResetEnvironment()
    gym_module = types.ModuleType("gym")
    gym_module.make = lambda _environment_id, **_kwargs: environment  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gym", gym_module)

    factory = build_webshop_text_environment_factory()

    with pytest.raises(TypeError, match="reset"):
        factory(23)

    assert environment.close_calls == 1


def test_webshop_factory_rejects_non_positive_product_count() -> None:
    with pytest.raises(ValueError, match="num_products"):
        build_webshop_text_environment_factory(num_products=0)


def _assert_webshop_make(
    environment_id: str,
    kwargs: dict[str, Any],
    environment: FakeWebShopEnvironment,
) -> FakeWebShopEnvironment:
    assert environment_id == "WebAgentTextEnv-v0"
    assert kwargs == {"observation_mode": "text", "num_products": 1000}
    return environment
