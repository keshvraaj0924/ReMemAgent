from typing import Any
import sys
import types

import pytest

from remem.integrations.official_benchmarks import (
    build_alfworld_text_environment_factory,
    build_webshop_text_environment_factory,
)


@pytest.fixture
def alfworld_factory(monkeypatch: pytest.MonkeyPatch) -> Any:
    environment_module = types.ModuleType("alfworld.agents.environment")
    environment_module.get_environment = lambda _env_type: _constructor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "alfworld", types.ModuleType("alfworld"))
    monkeypatch.setitem(sys.modules, "alfworld.agents", types.ModuleType("alfworld.agents"))
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", environment_module)
    return build_alfworld_text_environment_factory({})


@pytest.fixture
def webshop_factory(monkeypatch: pytest.MonkeyPatch) -> Any:
    gym_module = types.ModuleType("gym")
    gym_module.make = lambda *_args, **_kwargs: _WebShopEnvironment()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gym", gym_module)
    return build_webshop_text_environment_factory()


def test_alfworld_factory_rejects_boolean_seed_at_call_boundary(alfworld_factory: Any) -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        alfworld_factory(True)


def test_webshop_factory_rejects_non_integer_seed(webshop_factory: Any) -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        webshop_factory(1.5)


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_alfworld_factory_rejects_seed_outside_legacy_numpy_domain(
    alfworld_factory: Any,
    seed: int,
) -> None:
    with pytest.raises(ValueError, match="seed must be between 0 and 4294967295 inclusive"):
        alfworld_factory(seed)


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_webshop_factory_rejects_seed_outside_legacy_numpy_domain(
    webshop_factory: Any,
    seed: int,
) -> None:
    with pytest.raises(ValueError, match="seed must be between 0 and 4294967295 inclusive"):
        webshop_factory(seed)


@pytest.mark.parametrize("seed", [0, (2**32) - 1])
def test_alfworld_factory_accepts_legacy_numpy_seed_boundaries(
    alfworld_factory: Any,
    seed: int,
) -> None:
    environment = alfworld_factory(seed)

    assert environment.reset() == "observation"


@pytest.mark.parametrize("seed", [0, (2**32) - 1])
def test_webshop_factory_accepts_legacy_numpy_seed_boundaries(
    webshop_factory: Any,
    seed: int,
) -> None:
    environment = webshop_factory(seed)

    assert environment.reset() == "observation"


def _constructor(config: dict[str, Any], train_eval: str) -> Any:
    del config, train_eval
    return _Environment()


class _Environment:
    def init_env(self, *, batch_size: int) -> "_Environment":
        del batch_size
        return self

    def reset(self) -> str:
        return "observation"


class _WebShopEnvironment:
    def reset(self) -> str:
        return "observation"
