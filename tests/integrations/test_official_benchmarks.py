"""Tests for concrete upstream benchmark factory boundaries."""

from __future__ import annotations

import random
import sys
import types
from typing import Any

import pytest

from remem.integrations import official_benchmarks
from remem.integrations.benchmarks import BenchmarkEnvironmentFactory
from remem.integrations.official_benchmarks import (
    build_alfworld_text_environment_factory,
    build_webshop_text_environment_factory,
)


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


def test_seeded_webshop_reset_rejects_explicit_seed() -> None:
    adapter = official_benchmarks._SeededWebShopEnvironment(FakeWebShopEnvironment(), 3)

    with pytest.raises(TypeError, match="owns the reset seed"):
        adapter.reset(seed=4)


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


def test_alfworld_factory_closes_environment_when_initialization_fails(monkeypatch: Any) -> None:
    class FailingEnvironment:
        closed = False

        def __init__(self, _config: object, train_eval: str) -> None:
            assert train_eval == "eval"

        def init_env(self, batch_size: int) -> Any:
            assert batch_size == 1
            raise RuntimeError("initialization failed")

        def close(self) -> None:
            self.closed = True

    instances: list[FailingEnvironment] = []

    def environment_class(config: object, train_eval: str) -> FailingEnvironment:
        instance = FailingEnvironment(config, train_eval)
        instances.append(instance)
        return instance

    fake_environment_module = types.ModuleType("alfworld.agents.environment")
    fake_environment_module.get_environment = lambda _name: environment_class  # type: ignore[attr-defined]
    fake_agents_module = types.ModuleType("alfworld.agents")
    fake_alfworld_module = types.ModuleType("alfworld")
    fake_alfworld_module.agents = fake_agents_module  # type: ignore[attr-defined]
    fake_agents_module.environment = fake_environment_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "alfworld", fake_alfworld_module)
    monkeypatch.setitem(sys.modules, "alfworld.agents", fake_agents_module)
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", fake_environment_module)

    factory = build_alfworld_text_environment_factory({"env": {"type": "AlfredTWEnv"}})

    with pytest.raises(RuntimeError, match="initialization failed"):
        factory(23)

    assert len(instances) == 1
    assert instances[0].closed is True


def test_alfworld_factory_freezes_mutable_configuration(monkeypatch: Any) -> None:
    observed_configs: list[dict[str, Any]] = []

    class FakeInitializedEnvironment:
        def reset(self) -> str:
            return "state"

    class FakeEnvironment:
        def __init__(self, config: dict[str, Any], train_eval: str) -> None:
            observed_configs.append(config)
            assert train_eval == "eval"

        def init_env(self, batch_size: int) -> FakeInitializedEnvironment:
            assert batch_size == 1
            return FakeInitializedEnvironment()

    fake_environment_module = types.ModuleType("alfworld.agents.environment")
    fake_environment_module.get_environment = lambda _name: FakeEnvironment  # type: ignore[attr-defined]
    fake_agents_module = types.ModuleType("alfworld.agents")
    fake_alfworld_module = types.ModuleType("alfworld")
    fake_alfworld_module.agents = fake_agents_module  # type: ignore[attr-defined]
    fake_agents_module.environment = fake_environment_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "alfworld", fake_alfworld_module)
    monkeypatch.setitem(sys.modules, "alfworld.agents", fake_agents_module)
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", fake_environment_module)

    config = {"env": {"type": "AlfredTWEnv", "nested": {"split": "eval"}}}
    factory = build_alfworld_text_environment_factory(config, train_eval=" eval ")
    config["env"]["type"] = "MutatedEnv"
    config["env"]["nested"]["split"] = "train"

    factory(7)

    assert observed_configs == [
        {"env": {"type": "AlfredTWEnv", "nested": {"split": "eval"}}}
    ]


def test_alfworld_factory_normalizes_train_eval_before_external_construction(monkeypatch: Any) -> None:
    observed_train_eval: list[str] = []

    class FakeInitializedEnvironment:
        def reset(self) -> str:
            return "state"

    class FakeEnvironment:
        def __init__(self, _config: object, train_eval: str) -> None:
            observed_train_eval.append(train_eval)

        def init_env(self, batch_size: int) -> FakeInitializedEnvironment:
            assert batch_size == 1
            return FakeInitializedEnvironment()

    fake_environment_module = types.ModuleType("alfworld.agents.environment")
    fake_environment_module.get_environment = lambda _name: FakeEnvironment  # type: ignore[attr-defined]
    fake_agents_module = types.ModuleType("alfworld.agents")
    fake_alfworld_module = types.ModuleType("alfworld")
    fake_alfworld_module.agents = fake_agents_module  # type: ignore[attr-defined]
    fake_agents_module.environment = fake_environment_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "alfworld", fake_alfworld_module)
    monkeypatch.setitem(sys.modules, "alfworld.agents", fake_agents_module)
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", fake_environment_module)

    factory = build_alfworld_text_environment_factory({}, train_eval=" eval ")
    factory(5)

    assert observed_train_eval == ["eval"]


def test_webshop_factory_normalizes_external_identifiers(monkeypatch: Any) -> None:
    fake_gym = types.ModuleType("gym")
    observed: list[tuple[str, dict[str, object]]] = []

    def make(environment_id: str, **kwargs: object) -> FakeWebShopEnvironment:
        observed.append((environment_id, kwargs))
        return FakeWebShopEnvironment()

    fake_gym.make = make  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gym", fake_gym)

    factory = build_webshop_text_environment_factory(
        observation_mode=" text ",
        environment_id=" WebAgentTextEnv-v0 ",
    )
    factory(11)

    assert observed == [("WebAgentTextEnv-v0", {"observation_mode": "text"})]


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
    with pytest.raises(TypeError, match=r"config\['env'\] must be a mapping"):
        build_alfworld_text_environment_factory({"env": "invalid"})


def test_alfworld_factory_rejects_invalid_configured_environment_type() -> None:
    with pytest.raises(ValueError, match=r"config\['env'\]\['type'\] must be a non-empty string"):
        build_alfworld_text_environment_factory({"env": {"type": 123}})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match=r"config\['env'\]\['type'\] must be a non-empty string"):
        build_alfworld_text_environment_factory({"env": {"type": "   "}})


def test_alfworld_factory_rejects_invalid_configuration_before_import(monkeypatch: Any) -> None:
    def fail_import(_name: str) -> types.ModuleType:
        raise AssertionError("ALFWorld import should not be reached")

    monkeypatch.setattr("builtins.__import__", fail_import)
    with pytest.raises(ValueError, match=r"config\['env'\]\['type'\] must be a non-empty string"):
        build_alfworld_text_environment_factory({"env": {"type": False}})  # type: ignore[dict-item]


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
