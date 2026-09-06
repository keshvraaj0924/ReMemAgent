from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from remem.integrations.official_benchmarks import (
    build_alfworld_text_environment_factory,
    build_webshop_text_environment_factory,
)


class FakeAlfWorldEnvironment:
    """Minimal upstream-shaped ALFWorld environment double."""

    def __init__(self, config: dict[str, Any], train_eval: str) -> None:
        self.config = config
        self.train_eval = train_eval
        self.initialized_batch_size: int | None = None

    def init_env(self, batch_size: int = 1) -> FakeAlfWorldEnvironment:
        self.initialized_batch_size = batch_size
        return self


class FakeWebShopEnvironment:
    """Minimal Gym-shaped WebShop environment double."""

    def __init__(self, observation_mode: str, **kwargs: Any) -> None:
        self.observation_mode = observation_mode
        self.kwargs = kwargs


def test_alfworld_factory_isolates_config_and_constructor_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[FakeAlfWorldEnvironment] = []

    environment_module = types.ModuleType("alfworld.agents.environment")

    def get_environment(env_type: str):
        assert env_type == "AlfredTWEnv"

        def constructor(config: dict[str, Any], train_eval: str) -> FakeAlfWorldEnvironment:
            environment = FakeAlfWorldEnvironment(config, train_eval)
            created.append(environment)
            return environment

        return constructor

    environment_module.get_environment = get_environment  # type: ignore[attr-defined]
    agents_module = types.ModuleType("alfworld.agents")
    alfworld_module = types.ModuleType("alfworld")
    monkeypatch.setitem(sys.modules, "alfworld", alfworld_module)
    monkeypatch.setitem(sys.modules, "alfworld.agents", agents_module)
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", environment_module)

    source_config = {"env": {"type": "AlfredTWEnv", "nested": {"enabled": True}}}
    factory = build_alfworld_text_environment_factory(source_config)

    first = factory(1)
    second = factory(2)

    assert first.config["env"]["constructor_seed_marker"] == 0
    assert second.config["env"]["constructor_seed_marker"] == 1
    assert source_config == {"env": {"type": "AlfredTWEnv", "nested": {"enabled": True}}}
    assert created[0].config is not created[1].config
    assert created[0].config["env"] is not created[1].config["env"]


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
    monkeypatch.setitem(sys.modules, "alfworld", types.ModuleType("alfworld"))
    monkeypatch.setitem(sys.modules, "alfworld.agents", types.ModuleType("alfworld.agents"))
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", environment_module)

    factory = build_alfworld_text_environment_factory(
        {"env": {"type": "AlfredTWEnv"}},
        train_eval="eval_out_of_distribution",
        batch_size=1,
    )

    environment = factory(7)

    assert environment.train_eval == "eval_out_of_distribution"
    assert environment.initialized_batch_size == 1
    assert created == [environment]


def test_webshop_factory_constructs_gym_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[FakeWebShopEnvironment] = []

    gym_module = types.ModuleType("gym")

    def make(environment_id: str, *, observation_mode: str, **kwargs: Any) -> FakeWebShopEnvironment:
        assert environment_id == "WebAgentTextEnv-v0"
        environment = FakeWebShopEnvironment(observation_mode, **kwargs)
        created.append(environment)
        return environment

    gym_module.make = make  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gym", gym_module)

    factory = build_webshop_text_environment_factory(
        observation_mode="text",
        environment_id="WebAgentTextEnv-v0",
        num_products=10,
    )

    environment = factory(3)

    assert environment.observation_mode == "text"
    assert environment.kwargs["num_products"] == 10
    assert created == [environment]
