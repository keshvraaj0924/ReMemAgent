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

    def init_env(self, *, batch_size: int) -> "FakeAlfWorldEnvironment":
        self.batch_size = batch_size
        return self


class FakeWebShopEnvironment:
    def __init__(self) -> None:
        self.reset_seeds: list[int | None] = []

    def reset(self, *, seed: int | None = None) -> None:
        self.reset_seeds.append(seed)


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

    assert environment is created[0]
    assert environment.config == config
    assert environment.train_eval == "eval"
    assert environment.batch_size == 1


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
