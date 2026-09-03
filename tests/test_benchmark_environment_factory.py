"""Tests for benchmark environment factory binding."""

from __future__ import annotations

import sys
import types

import pytest

from remem.environments import AlfWorldAdapter, WebShopAdapter
from remem.integrations.benchmarks import (
    BenchmarkEnvironmentFactory,
    load_benchmark_environment_factory,
    resolve_environment_factory,
)


class FakeEnvironment:
    """Minimal raw environment used to verify adapter selection."""

    def __init__(self) -> None:
        self.actions: list[object] = []

    def reset(self) -> str:
        return "state"

    def step(self, action: object) -> tuple[str, float, bool, dict[str, object]]:
        self.actions.append(action)
        return "next", 1.0, True, {}


def test_factory_wraps_alfworld_environment() -> None:
    factory = BenchmarkEnvironmentFactory("ALFWorld", lambda _seed: FakeEnvironment())

    adapter = factory(17)

    assert isinstance(adapter, AlfWorldAdapter)
    assert adapter.reset() == "state"


def test_factory_wraps_webshop_environment() -> None:
    factory = BenchmarkEnvironmentFactory("WebShop", lambda _seed: FakeEnvironment())

    adapter = factory(17)

    assert isinstance(adapter, WebShopAdapter)
    assert adapter.reset() == "state"


def test_factory_passes_episode_seed_to_raw_factory() -> None:
    observed_seeds: list[int] = []

    def raw_factory(seed: int) -> FakeEnvironment:
        observed_seeds.append(seed)
        return FakeEnvironment()

    factory = BenchmarkEnvironmentFactory("alfworld", raw_factory)
    factory(23)

    assert observed_seeds == [23]


def test_factory_rejects_non_integer_seed_before_environment_creation() -> None:
    created = False

    def raw_factory(_seed: int) -> FakeEnvironment:
        nonlocal created
        created = True
        return FakeEnvironment()

    factory = BenchmarkEnvironmentFactory("alfworld", raw_factory)

    with pytest.raises(TypeError, match="seed must be an integer"):
        factory(1.5)  # type: ignore[arg-type]

    assert not created


def test_factory_rejects_boolean_seed_before_environment_creation() -> None:
    created = False

    def raw_factory(_seed: int) -> FakeEnvironment:
        nonlocal created
        created = True
        return FakeEnvironment()

    factory = BenchmarkEnvironmentFactory("webshop", raw_factory)

    with pytest.raises(TypeError, match="seed must be an integer"):
        factory(True)  # type: ignore[arg-type]

    assert not created


def test_factory_rejects_non_callable_raw_factory() -> None:
    with pytest.raises(TypeError, match="raw_factory must be callable"):
        BenchmarkEnvironmentFactory("alfworld", object())  # type: ignore[arg-type]


def test_factory_adds_benchmark_context_when_environment_creation_fails() -> None:
    def raw_factory(seed: int) -> FakeEnvironment:
        raise OSError(f"seed {seed} unavailable")

    factory = BenchmarkEnvironmentFactory("webshop", raw_factory)

    with pytest.raises(
        RuntimeError,
        match="failed to create webshop environment for seed 31",
    ) as error:
        factory(31)

    assert isinstance(error.value.__cause__, OSError)


def test_factory_rejects_unsupported_benchmark() -> None:
    with pytest.raises(ValueError, match="unsupported benchmark"):
        BenchmarkEnvironmentFactory("unknown", lambda _seed: FakeEnvironment())


def test_resolve_environment_factory_supports_nested_attributes() -> None:
    module = types.ModuleType("test_benchmark_factory_module")
    namespace = types.SimpleNamespace(factory=lambda _seed: FakeEnvironment())
    module.namespace = namespace
    sys.modules[module.__name__] = module
    try:
        resolved = resolve_environment_factory("test_benchmark_factory_module:namespace.factory")
        assert isinstance(resolved(1), FakeEnvironment)
    finally:
        del sys.modules[module.__name__]


def test_resolve_environment_factory_rejects_non_callable() -> None:
    module = types.ModuleType("test_non_callable_factory_module")
    module.value = object()
    sys.modules[module.__name__] = module
    try:
        with pytest.raises(TypeError, match="not callable"):
            resolve_environment_factory("test_non_callable_factory_module:value")
    finally:
        del sys.modules[module.__name__]


def test_load_benchmark_environment_factory_resolves_and_wraps() -> None:
    module = types.ModuleType("test_loaded_factory_module")
    module.make_environment = lambda _seed: FakeEnvironment()
    sys.modules[module.__name__] = module
    try:
        factory = load_benchmark_environment_factory(
            "webshop", "test_loaded_factory_module:make_environment"
        )
        assert isinstance(factory(1), WebShopAdapter)
    finally:
        del sys.modules[module.__name__]
