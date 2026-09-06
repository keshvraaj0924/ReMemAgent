from __future__ import annotations

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

    def reset(self) -> None:
        self.reset_calls += 1
        self.reset_random_values.append(random.random())

    def close(self) -> None:
        pass


def test_alfworld_factory_uses_expected_upstream_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[FakeAlfWorldEnvironment] = []

    class FakeAlfWorldModule:
        def __init__(self) -> None:
            self.configs: list[dict[str, Any]] = []

        def get_environment(self, config: dict[str, Any], train_eval: str) -> FakeAlfWorldEnvironment:
            self.configs.append(config)
            environment = FakeAlfWorldEnvironment(config, train_eval)
            created.append(environment)
            return environment

    module = FakeAlfWorldModule()
    monkeypatch.setitem(sys.modules, "alfworld", module)

    config = {"env": "AlfredThorEnv", "task": "test"}
    factory = build_alfworld_text_environment_factory(config=config, train_eval="eval")
    environment = factory(23)

    assert len(created) == 1
    assert environment.batch_size == 1
    assert environment.config == config
    assert environment.train_eval == "eval"
    assert environment.reset_calls == 0


def test_alfworld_factory_seeds_environment_and_restores_global_random_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = FakeAlfWorldEnvironment({}, "eval")

    class FakeAlfWorldModule:
        @staticmethod
        def get_environment(config: dict[str, Any], train_eval: str) -> FakeAlfWorldEnvironment:
            return environment

    monkeypatch.setitem(sys.modules, "alfworld", FakeAlfWorldModule())

    random.seed(991)
    expected = random.getstate()
    expected_value = random.random()
    random.setstate(expected)

    factory = build_alfworld_text_environment_factory(config={}, train_eval="eval")
    result = factory(17)
    result.reset()

    assert result.reset_random_values == [expected_value]
    assert random.getstate() == expected


def test_alfworld_factory_closes_environment_when_reset_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = RuntimeError("reset failed")
    environment = FakeAlfWorldEnvironment({}, "eval")
    environment.reset_error = failure

    class FakeAlfWorldModule:
        @staticmethod
        def get_environment(config: dict[str, Any], train_eval: str) -> FakeAlfWorldEnvironment:
            return environment

    monkeypatch.setitem(sys.modules, "alfworld", FakeAlfWorldModule())

    factory = build_alfworld_text_environment_factory(config={}, train_eval="eval")
    result = factory(23)

    with pytest.raises(RuntimeError, match="reset failed") as raised:
        result.reset()

    assert raised.value is failure
    assert environment.close_calls == 0
    environment.close()
    assert environment.close_calls == 1


def test_webshop_factory_closes_environment_when_reset_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    class MissingResetEnvironment:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    environment = MissingResetEnvironment()
    gym_module = types.ModuleType("gym")
    gym_module.make = lambda _environment_id, **_kwargs: environment
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


class FakeWebShopEnvironment:
    def __init__(self, *, reset_error: BaseException | None = None) -> None:
        self.reset_error = reset_error
        self.reset_seeds: list[int | None] = []
        self.close_calls = 0

    def reset(self, *, seed: int | None = None) -> None:
        self.reset_seeds.append(seed)
        if self.reset_error is not None:
            raise self.reset_error

    def close(self) -> None:
        self.close_calls += 1


class LegacyWebShopEnvironment:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.close_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.close_calls += 1
