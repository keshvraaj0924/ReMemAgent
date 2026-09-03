import random
import sys
import types
from typing import Any

import pytest

from remem.integrations.benchmarks import BenchmarkEnvironmentFactory
from remem.integrations.official_benchmarks import (
    build_alfworld_text_environment_factory,
    build_webshop_text_environment_factory,
)


def test_webshop_factory_scopes_construction_and_reset_randomness(monkeypatch: Any) -> None:
    construction_values: list[float] = []

    class FakeEnvironment:
        def reset(self) -> tuple[str, float]:
            return "reset", random.random()

    def make(environment_id: str, **kwargs: object) -> FakeEnvironment:
        assert environment_id == "WebAgentTextEnv-v0"
        assert kwargs == {"observation_mode": "text"}
        construction_values.append(random.random())
        return FakeEnvironment()

    fake_gym = types.ModuleType("gym")
    fake_gym.make = make  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gym", fake_gym)

    random.seed(1234)
    expected_state = random.getstate()
    factory = build_webshop_text_environment_factory()
    environment = factory(17)

    assert random.getstate() == expected_state
    assert construction_values == [random.Random(17).random()]
    first_reset = environment.reset()
    second_reset = environment.reset()
    assert first_reset == second_reset
    assert random.getstate() == expected_state


def test_alfworld_factory_scopes_construction_randomness(monkeypatch: Any) -> None:
    construction_values: list[float] = []

    class FakeInitializedEnvironment:
        def reset(self) -> tuple[str, float]:
            return "reset", random.random()

    class FakeEnvironment:
        def __init__(self, config: object, train_eval: str) -> None:
            assert config == {"env": {"type": "AlfredTWEnv"}}
            assert train_eval == "eval"
            construction_values.append(random.random())

        def init_env(self, batch_size: int) -> FakeInitializedEnvironment:
            assert batch_size == 1
            construction_values.append(random.random())
            return FakeInitializedEnvironment()

    fake_environment_module = types.ModuleType("alfworld.agents.environment")
    fake_environment_module.get_environment = lambda name: FakeEnvironment  # type: ignore[attr-defined]
    fake_agents_module = types.ModuleType("alfworld.agents")
    fake_alfworld_module = types.ModuleType("alfworld")
    fake_alfworld_module.agents = fake_agents_module  # type: ignore[attr-defined]
    fake_agents_module.environment = fake_environment_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "alfworld", fake_alfworld_module)
    monkeypatch.setitem(sys.modules, "alfworld.agents", fake_agents_module)
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", fake_environment_module)

    random.seed(4321)
    expected_state = random.getstate()
    factory = build_alfworld_text_environment_factory({"env": {"type": "AlfredTWEnv"}})
    environment = factory(23)

    expected_rng = random.Random(23)
    assert construction_values == [expected_rng.random(), expected_rng.random()]
    assert random.getstate() == expected_state
    assert environment.reset() == environment.reset()
    assert random.getstate() == expected_state


def test_alfworld_factory_rejects_invalid_text_configuration() -> None:
    with pytest.raises(ValueError, match="train_eval must be a non-empty string"):
        build_alfworld_text_environment_factory({}, train_eval="   ")
    with pytest.raises(ValueError, match="train_eval must be a non-empty string"):
        build_alfworld_text_environment_factory({}, train_eval=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="env_type must be a non-empty string"):
        build_alfworld_text_environment_factory({}, env_type="   ")
    with pytest.raises(ValueError, match="env_type must be a non-empty string"):
        build_alfworld_text_environment_factory({}, env_type=1)  # type: ignore[arg-type]


def test_alfworld_factory_rejects_non_mapping_environment_config() -> None:
    with pytest.raises(TypeError, match="config\['env'\] must be a mapping"):
        build_alfworld_text_environment_factory({"env": "invalid"})


def test_webshop_factory_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="observation_mode must be a non-empty string"):
        build_webshop_text_environment_factory(observation_mode="   ")
    with pytest.raises(ValueError, match="observation_mode must be a non-empty string"):
        build_webshop_text_environment_factory(observation_mode=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="environment_id must be a non-empty string"):
        build_webshop_text_environment_factory(environment_id="   ")
    with pytest.raises(ValueError, match="environment_id must be a non-empty string"):
        build_webshop_text_environment_factory(environment_id=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="num_products must be an integer"):
        build_webshop_text_environment_factory(num_products=True)
    with pytest.raises(ValueError, match="num_products must be positive"):
        build_webshop_text_environment_factory(num_products=0)


def test_benchmark_environment_factory_rejects_empty_benchmark_name() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        BenchmarkEnvironmentFactory("   ", lambda seed: object())


def test_benchmark_environment_factory_rejects_non_string_benchmark_name() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        BenchmarkEnvironmentFactory(123, lambda seed: object())  # type: ignore[arg-type]


def test_benchmark_environment_factory_rejects_unsupported_benchmark_name() -> None:
    with pytest.raises(ValueError, match="unsupported benchmark"):
        BenchmarkEnvironmentFactory("unknown-v1", lambda seed: object())
